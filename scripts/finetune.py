#!/usr/bin/env python3
"""Small supervised PyTorch/CQ-QAT fine-tuning entry point for Needle 2.

JSONL rows: tools, query, answers (list of name/arguments), optional reasoning.
Training quality must be evaluated on held-out domain examples after export.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('data');p.add_argument('output')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--qat',action='store_true');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if a.steps<1 or a.lr<=0:p.error('positive steps and learning rate required')
    import torch
    from needle2.convert import load_torch_model,save_torch_weights
    from needle2.qat import enable_qat,disable_qat
    from needle2.tokenizer import RefTokenizer
    from needle2.prompt import render_prompt
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    source=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
    if out.exists():raise ValueError('output must be a new directory to preserve the source checkpoint')
    tokenizer=RefTokenizer.from_cact(source/'source.cact')
    rows=[json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    if not rows:raise ValueError('empty training data')
    model=load_torch_model(source,quant_activations=a.qat).to(a.device).train()
    samples=[]
    for row in rows:
        prompt=[2]+tokenizer.encode(render_prompt(row['query'],row['tools'],row.get('system')))
        reasoning=(row.get('reasoning') or '').strip()
        target=('<think>\n'+reasoning+'\n</think>\n' if reasoning else '')+'<tool_call>'+json.dumps(row['answers'],separators=(',',':'),ensure_ascii=False)+'</tool_call><|im_end|>'
        ids=prompt+tokenizer.encode(target)+[1]
        if len(ids)>model.config.max_seq_len:raise ValueError('training example exceeds max_seq_len; shorten it explicitly')
        labels=torch.tensor(ids[1:],device=a.device);labels[:len(prompt)-1]=-100
        samples.append((torch.tensor([ids[:-1]],device=a.device),labels))
    if a.qat:enable_qat(model,source/'source.cact')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    history=[]
    for step in range(a.steps):
        tokens,labels=samples[step%len(samples)]
        optimizer.zero_grad(set_to_none=True)
        logits=model(tokens)[0]
        loss=torch.nn.functional.cross_entropy(logits,labels)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        history.append(float(loss.detach()));print(f'step={step+1} loss={history[-1]:.6f}',flush=True)
    if a.qat:disable_qat(model)
    shutil.copytree(source,out)
    save_torch_weights(model,out)
    (out/'training.json').write_text(json.dumps(dict(steps=a.steps,lr=a.lr,seed=a.seed,qat=a.qat,loss=history),indent=2)+'\n')

if __name__=='__main__':main()
