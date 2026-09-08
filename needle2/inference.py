"""Independent single-request greedy inference and explicit timing boundaries."""
import time
from pathlib import Path
import numpy as np
from .archive import Archive
from .prompt import render_prompt,parse_response
from .tokenizer import RefTokenizer,parse_tokenizer_blob

def retrieve_tools(archive, tokenizer, prompt, tools, top_k=4, threads=1):
    """Rank candidate tools against prompt using NativeProbeEncoder and return top-k."""
    if "contrastive_head.probes" not in archive.tensors:
        raise ValueError("retrieval requested but archive contains no exported contrastive_head")
    from .heads import NativeProbeEncoder
    encoder = NativeProbeEncoder(archive, threads=threads)
    max_len = archive.metadata.get("max_seq_len", 8192)
    query_ids = ([2] + tokenizer.encode(prompt))[:max_len]
    query_emb = encoder.encode(query_ids)

    scored_tools = []
    for tool in tools:
        if "_embedding" in tool:
            tool_emb = np.asarray(tool["_embedding"], dtype=np.float32)
        else:
            desc = tool.get("description", "")
            text = f"{tool.get('name', '')}: {desc}".strip(": ") if desc else tool.get("name", "")
            t_ids = ([2] + tokenizer.encode(text))[:max_len]
            tool_emb = encoder.encode(t_ids)
        sim = float(np.dot(query_emb, tool_emb))
        scored_tools.append((sim, tool))

    scored_tools.sort(key=lambda item: item[0], reverse=True)
    k = min(top_k, len(tools))
    selected = [tool for _, tool in scored_tools[:k]]
    return selected, scored_tools


def generate(model_path,prompt,*,tools=None,system=None,backend='native',max_new_tokens=96,threads=1,quant_activations=False,prefill_backend='native',constrain=True,matmul='fp32',kv_cache='fp32',retrieval=False,top_k_tools=None):
    if max_new_tokens < 0 or threads < 1:
        raise ValueError('max_new_tokens must be nonnegative; threads must be positive')
    model_path=Path(model_path)
    archive_path=model_path if model_path.is_file() else model_path/'source.cact'
    if backend=='native' and model_path.is_dir():
        raise ValueError('export the PyTorch checkpoint to .cact before using the native backend')
    archive=Archive.load(archive_path)
    tokenizer=RefTokenizer(parse_tokenizer_blob(archive.tensors['tokenizer'].blob))
    retrieval_performed = False
    retrieved_tools_info = None
    if tools is not None:
        from .prompt import normalize_tools
        tools = normalize_tools(tools)
        if (retrieval or top_k_tools is not None) and len(tools) > 0:
            k = top_k_tools if top_k_tools is not None else min(4, len(tools))
            selected_tools, scored = retrieve_tools(archive, tokenizer, prompt, tools, top_k=k, threads=threads)
            tools = selected_tools
            retrieval_performed = True
            retrieved_tools_info = [{"name": t.get("name"), "score": round(s, 4)} for s, t in scored]
    text=render_prompt(prompt,tools,system) if tools is not None else prompt
    ids=[2]+tokenizer.encode(text)
    prefix_len=0
    if tools is not None:
        # The marker is a user-defined whole token; pin through </tools>.
        prefix_len=ids.index(tokenizer.p2id['</tools>'])+1
    if len(ids)>=archive.metadata['max_seq_len']:
        raise ValueError('prompt leaves no room within max_seq_len')
    cap=min(max_new_tokens,archive.metadata['max_seq_len']-len(ids))
    grammar=None
    if tools is not None and constrain:
        from .grammar import ToolGrammar
        grammar=ToolGrammar(tools,tokenizer)
    if backend=='native':
        from .native import NativeEngine
        if prefill_backend=='torch':
            import torch
            torch.set_num_threads(threads)
        engine=NativeEngine(archive,threads=threads,activation_bits=8 if quant_activations else 0,matmul=matmul,kv_cache=kv_cache)
        engine.reset(prefix_len=prefix_len)
        def consume(tokens):
            if len(tokens)>1:
                return engine.prefill(tokens,last_only=True,backend=prefill_backend)
            last=None
            for token in tokens: last=engine.step(int(token))
            return last
    elif backend=='torch':
        if matmul!='fp32':raise ValueError('SDOT matmul requires the native backend')
        import torch
        from .convert import load_torch_model
        torch.set_num_threads(threads)
        model=load_torch_model(model_path,quant_activations=quant_activations).eval()
        cache=None
        first=True
        def consume(tokens):
            nonlocal cache,first
            with torch.inference_mode():
                sink=torch.arange(len(tokens))[None,:]<prefix_len if first else None
                logits,cache=model(torch.tensor([tokens],dtype=torch.long),cache,use_cache=True,sink_mask=sink)
            first=False
            return logits[0,-1].cpu().numpy()
    else: raise ValueError('unknown backend')
    start=time.perf_counter(); logits=consume(ids); prefill_s=time.perf_counter()-start
    output=[]; decode_s=0.; decode_steps=0
    # First token is selected from prefill; subsequent token forwards are timed
    # separately. The last selected token does not require an extra forward.
    decode_started=time.perf_counter()
    candidates = None
    for i in range(cap):
        if candidates is not None:
            token = grammar.select_candidate(candidates, logits) if grammar is not None else int(np.argmax(logits))
        else:
            token = grammar.select(logits) if grammar is not None else int(np.argmax(logits))
        if grammar is not None:grammar.accept(token)
        output.append(token)
        if token in (1,5) or (grammar is not None and grammar.finished) or i==cap-1: break
        candidates = grammar.candidate_tokens() if (grammar is not None and backend == 'native') else None
        start=time.perf_counter()
        if candidates is not None and len(candidates) == 1:
            engine.step(token, compute_logits=False)
            logits = np.array([0.0], dtype=np.float32)
        elif candidates is not None:
            logits = engine.step_candidates(token, candidates)
        else:
            logits = consume([token])
        decode_s+=time.perf_counter()-start; decode_steps+=1
    decode_wall=time.perf_counter()-decode_started
    decoded=tokenizer.decode(output)
    result=parse_response(decoded) if tools is not None else dict(text=decoded)
    arithmetic=('approximate_sdot_rotated_a8_kv_' + kv_cache if matmul=='sdot' else 'public_reference_a8_kv_' + kv_cache if quant_activations else 'fp32_reference_kv_' + kv_cache if kv_cache=='int8' else 'fp32_reference')
    result.update(backend=backend,arithmetic=arithmetic,matmul=matmul,kv_cache=kv_cache,activation_fake_quant=quant_activations,
                  token_ids=output,prompt_tokens=len(ids),generated_tokens=len(output),
                  prefill_seconds=prefill_s,decode_seconds=decode_wall,decode_forward_seconds=decode_s,
                  prefill_tokens_per_second=len(ids)/prefill_s,
                  decode_forward_steps=decode_steps,
                  decode_tokens_per_second=decode_steps/decode_wall if decode_steps else None,
                  model_sha256=archive.sha256,
                  grammar_constrained=grammar is not None,
                  retrieval_enabled=retrieval_performed,
                  retrieved_tools=[t['name'] for t in tools] if (retrieval_performed and tools is not None) else None,
                  prefix_tokens=prefix_len,prefill_backend=prefill_backend if backend=='native' else 'torch')
    return result
