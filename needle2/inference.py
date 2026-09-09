"""Reusable inference sessions and a compatible one-shot convenience entry point."""
import time
import json
import threading
from pathlib import Path
import numpy as np
from .archive import Archive
from .prompt import render_prompt,parse_response
from .tokenizer import RefTokenizer,parse_tokenizer_blob

def retrieve_tools(archive, tokenizer, prompt, tools, top_k=4, threads=1):
    """Rank candidate tools against prompt using NativeProbeEncoder and return top-k."""
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    from .prompt import normalize_tools
    tools = normalize_tools(tools)
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
            if tool_emb.shape != query_emb.shape or not np.isfinite(tool_emb).all():
                raise ValueError("cached tool embedding must be finite and match the query dimension")
            norm = float(np.linalg.norm(tool_emb))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("cached tool embedding must have a finite nonzero norm")
            tool_emb = tool_emb / norm
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


def _validate_request(max_new_tokens, top_k_tools):
    if isinstance(max_new_tokens, (bool, np.bool_)) or not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if top_k_tools is not None and (isinstance(top_k_tools, (bool, np.bool_)) or
            not isinstance(top_k_tools, (int, np.integer)) or top_k_tools < 1):
        raise ValueError('top_k_tools must be a positive integer')


class InferenceSession:
    """Reuse model, tokenizer, last tool grammar and native prefix snapshot.

    Requests are independent, not multi-turn conversation continuations. Calls
    on one instance are serialized; use separate sessions for parallel streams.
    Tools/system may change; prefix reuse requires identical prefix token IDs.
    """

    def __init__(self, model_path, *, backend='native', threads=1,
                 quant_activations=False, prefill_backend='native', matmul='fp32', kv_cache='fp32'):
        started = time.perf_counter()
        if threads < 1:
            raise ValueError('threads must be positive')
        if backend not in ('native', 'torch'):
            raise ValueError('unknown backend')
        if prefill_backend not in ('native', 'torch'):
            raise ValueError('prefill backend must be native or torch')
        if kv_cache not in ('fp32', 'int8') or (backend == 'torch' and kv_cache != 'fp32'):
            raise ValueError('INT8 KV cache requires the native backend')
        if backend == 'torch' and matmul != 'fp32':
            raise ValueError('SDOT matmul requires the native backend')
        model_path = Path(model_path)
        if backend == 'native' and model_path.is_dir():
            raise ValueError('export the PyTorch checkpoint to .cact before using the native backend')
        archive_path = model_path if model_path.is_file() else model_path/'source.cact'
        self.archive = Archive.load(archive_path)
        self.model_sha256 = self.archive.sha256
        if backend == 'native':
            from .frontend import NativeTokenizer
            self.tokenizer = NativeTokenizer(self.archive.tensors['tokenizer'].blob)
        else:
            self.tokenizer = RefTokenizer(parse_tokenizer_blob(self.archive.tensors['tokenizer'].blob))
        self.backend, self.threads = backend, threads
        self.quant_activations, self.prefill_backend = quant_activations, prefill_backend
        self.matmul, self.kv_cache = matmul, kv_cache
        self._lock = threading.Lock()
        self._grammar_key = self._grammar = self._dfa = self._prefix_ids = None
        self._prefix_text = self._prefix_token_ids = None
        if backend == 'torch' or prefill_backend == 'torch':
            import torch
            torch.set_num_threads(threads)
        if backend == 'native':
            from .native import NativeEngine
            self.engine = NativeEngine(self.archive, threads=threads,
                activation_bits=8 if quant_activations else 0, matmul=matmul, kv_cache=kv_cache)
        else:
            from .convert import load_torch_model
            self.model = load_torch_model(model_path, quant_activations=quant_activations).eval()
        self.setup_seconds = time.perf_counter()-started

    def generate(self, prompt, *, tools=None, system=None, max_new_tokens=96,
                 constrain=True, retrieval=False, top_k_tools=None):
        _validate_request(max_new_tokens, top_k_tools)
        with self._lock:
            return self._generate(prompt, tools=tools, system=system, max_new_tokens=max_new_tokens,
                                  constrain=constrain, retrieval=retrieval, top_k_tools=top_k_tools)

    def _generate(self, prompt, *, tools, system, max_new_tokens, constrain, retrieval, top_k_tools):
        request_started = time.perf_counter()
        archive, tokenizer = self.archive, self.tokenizer
        backend, threads = self.backend, self.threads
        quant_activations, prefill_backend = self.quant_activations, self.prefill_backend
        matmul, kv_cache = self.matmul, self.kv_cache
        retrieval_performed = False
        if tools is not None:
            from .prompt import normalize_tools
            tools = normalize_tools(tools)
            if (retrieval or top_k_tools is not None) and len(tools) > 0:
                k = top_k_tools if top_k_tools is not None else min(4, len(tools))
                selected_tools, _ = retrieve_tools(archive, tokenizer, prompt, tools, top_k=k, threads=threads)
                tools = selected_tools
                retrieval_performed = True
            tools = [{key: value for key, value in tool.items() if key != "_embedding"} for tool in tools]
        text=render_prompt(prompt,tools,system) if tools is not None else prompt
        # A user-defined marker flushes the BPE segment. Reusing everything
        # through </tools> is exact if no other marker can contain it.
        marker = '</tools>'
        can_split = tools is not None and marker in tokenizer.markers and not any(
            m != marker and marker in m for m in tokenizer.markers)
        if can_split:
            split = text.index(marker) + len(marker)
            prefix_text = text[:split]
            if prefix_text != self._prefix_text:
                self._prefix_token_ids = [2] + tokenizer.encode(prefix_text)
                self._prefix_text = prefix_text
            ids = self._prefix_token_ids + tokenizer.encode(text[split:], add_dummy_prefix=False)
        else:
            ids=[2]+tokenizer.encode(text)
        prefix_len=0
        if tools is not None:
            # The marker is a user-defined whole token; pin through </tools>.
            prefix_len=ids.index(tokenizer.p2id['</tools>'])+1
        if len(ids)>=archive.metadata['max_seq_len']:
            raise ValueError('prompt leaves no room within max_seq_len')
        cap=min(max_new_tokens,archive.metadata['max_seq_len']-len(ids))
        grammar=None
        grammar_dfa=None
        if tools is not None and constrain:
            key = json.dumps(tools, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            if key != self._grammar_key:
                if backend == 'native':
                    from .frontend import CompiledGrammar
                    new_grammar = CompiledGrammar(tokenizer, key)
                    new_dfa = new_grammar
                else:
                    from .grammar import ToolGrammar
                    new_grammar = ToolGrammar(tools, tokenizer)
                    new_dfa = None
                self._grammar, self._dfa = new_grammar, new_dfa
                self._grammar_key = key
            grammar, grammar_dfa = self._grammar, self._dfa
            if backend != 'native':
                grammar.reset()
        prepare_s = time.perf_counter() - request_started
        restore_s = 0.0
        prefix_hit = False
        prefill_tokens = len(ids)
        if backend=='native':
            engine = self.engine
            def consume(tokens):
                if len(tokens)>1:
                    return engine.prefill(tokens,last_only=True,backend=prefill_backend)
                return engine.step(int(tokens[0]))
            wanted_prefix = tuple(ids[:prefix_len])
            prefix_hit = (prefill_backend == 'native' and bool(wanted_prefix)
                          and wanted_prefix == self._prefix_ids)
            start = time.perf_counter()
            if prefix_hit:
                engine.reset_to_prefix()
            else:
                engine.reset(prefix_len=prefix_len)
                self._prefix_ids = None
            restore_s = time.perf_counter() - start
            start = time.perf_counter()
            if prefix_len and prefill_backend == 'native':
                if not prefix_hit:
                    consume(ids[:prefix_len])
                    engine.cache_prefix()
                    self._prefix_ids = wanted_prefix
                else:
                    prefill_tokens -= prefix_len
                logits = consume(ids[prefix_len:])
            else:
                logits = consume(ids)
            prefill_s = time.perf_counter() - start
        else:
            import torch
            cache=None
            first=True
            def consume(tokens):
                nonlocal cache,first
                with torch.inference_mode():
                    sink=torch.arange(len(tokens))[None,:]<prefix_len if first else None
                    logits,cache=self.model(torch.tensor([tokens],dtype=torch.long),cache,use_cache=True,sink_mask=sink)
                first=False
                return logits[0,-1].cpu().numpy()
            start=time.perf_counter(); logits=consume(ids); prefill_s=time.perf_counter()-start
        output=[]; decode_s=0.; decode_steps=0
        # First token is selected from prefill; subsequent token forwards are timed
        # separately. The last selected token does not require an extra forward.
        decode_started=time.perf_counter()
        native_decode_s = None
        if cap == 0:
            decode_wall = 0.0
        elif backend=='native' and (grammar_dfa is not None or grammar is None):
            native_start = time.perf_counter()
            output=engine.decode_from_logits(logits,max_new_tokens=cap,grammar=grammar_dfa)
            native_decode_s = time.perf_counter() - native_start
            decode_wall=time.perf_counter()-decode_started
            decode_steps=max(0,len(output)-1)
            decode_s=decode_wall
        else:
            for i in range(cap):
                token = grammar.select(logits) if grammar is not None else int(np.argmax(logits))
                if grammar is not None:
                    grammar.accept(token)
                output.append(token)
                if token in (1, 5) or (grammar is not None and grammar.finished) or i == cap-1:
                    break
                start = time.perf_counter()
                logits = consume([token])
                decode_s += time.perf_counter()-start
                decode_steps += 1
            decode_wall = time.perf_counter()-decode_started
        parse_started = time.perf_counter()
        decoded=tokenizer.decode(output)
        result=parse_response(decoded) if tools is not None else dict(text=decoded)
        arithmetic=('approximate_sdot_rotated_a8_kv_' + kv_cache if matmul=='sdot' else 'public_reference_a8_kv_' + kv_cache if quant_activations else 'fp32_reference_kv_' + kv_cache if kv_cache=='int8' else 'fp32_reference')
        result.update(backend=backend,arithmetic=arithmetic,matmul=matmul,kv_cache=kv_cache,activation_fake_quant=quant_activations,
                      token_ids=output,prompt_tokens=len(ids),generated_tokens=len(output),
                      prefill_seconds=prefill_s,decode_seconds=decode_wall,decode_forward_seconds=decode_s,
                      prefill_tokens_per_second=prefill_tokens/prefill_s,
                      decode_forward_steps=decode_steps,
                      decode_tokens_per_second=decode_steps/decode_wall if decode_steps else None,
                      model_sha256=self.model_sha256,
                      grammar_constrained=grammar is not None,
                      grammar_backend='native_dfa' if grammar_dfa is not None else 'python_regex' if grammar is not None else 'none',
                      retrieval_enabled=retrieval_performed,
                      retrieved_tools=[t['name'] for t in tools] if (retrieval_performed and tools is not None) else None,
                      prefix_tokens=prefix_len,prefill_backend=prefill_backend if backend=='native' else 'torch')
        result.update(tokenizer_backend='cpp' if backend == 'native' else 'python',
                      grammar_compiler='cpp' if grammar_dfa is not None else 'python' if grammar is not None else None,
                      prepare_seconds=prepare_s, prefix_restore_seconds=restore_s,
                      prefix_cache_hit=prefix_hit, prefill_tokens=prefill_tokens,
                      native_decode_call_seconds=native_decode_s,
                      parse_seconds=time.perf_counter()-parse_started)
        result['request_seconds'] = time.perf_counter()-request_started
        return result


def generate(model_path, prompt, *, tools=None, system=None, backend='native', max_new_tokens=96,
             threads=1, quant_activations=False, prefill_backend='native', constrain=True,
             matmul='fp32', kv_cache='fp32', retrieval=False, top_k_tools=None):
    """One-shot API; use InferenceSession to amortize initialization across requests."""
    started = time.perf_counter()
    _validate_request(max_new_tokens, top_k_tools)
    session = InferenceSession(model_path, backend=backend, threads=threads,
        quant_activations=quant_activations, prefill_backend=prefill_backend,
        matmul=matmul, kv_cache=kv_cache)
    result = session.generate(prompt, tools=tools, system=system, max_new_tokens=max_new_tokens,
                              constrain=constrain, retrieval=retrieval, top_k_tools=top_k_tools)
    result.update(setup_seconds=session.setup_seconds, total_seconds=time.perf_counter()-started)
    return result
