"""Command line for the independent Needle 2 implementation."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(prog='needle2')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('build-native',help='build or load the native shared library and report CPU capabilities')
    inspect=sub.add_parser('inspect',help='inspect CACT precision, geometry and hash')
    inspect.add_argument('model')
    convert=sub.add_parser('to-torch',help='official .cact or FP16 .pkl -> canonical PyTorch safetensors')
    convert.add_argument('source'); convert.add_argument('output'); convert.add_argument('--template')
    quant=sub.add_parser('quantize',help='PyTorch checkpoint -> official mixed CQ2/CQ4 CACT')
    quant.add_argument('checkpoint'); quant.add_argument('output')
    quant.add_argument('--force-requantize',action='store_true',help='recompute even unchanged CQ weights (can introduce additional error)')
    run=sub.add_parser('run',help='independent greedy inference from the embedded tokenizer')
    run.add_argument('model'); run.add_argument('--prompt',required=True); run.add_argument('--tools')
    run.add_argument('--system'); run.add_argument('--backend',choices=['native','torch'],default='native')
    run.add_argument('--max-new-tokens',type=int,default=96); run.add_argument('--threads',type=int,default=1)
    run.add_argument('--prefill-backend',choices=['native','torch'],default='native',help='torch uses ~175 MB dense weights for faster prompt processing')
    run.add_argument('--no-grammar',action='store_true',help='disable schema-constrained decoding')
    run.add_argument('--matmul',choices=['fp32','sdot'],default='fp32',help='sdot is an approximate ARM dot-product mode; adds rotated A8/codebook quantization')
    run.add_argument('--kv-cache',choices=['fp32','int8'],default='fp32',help='KV cache precision')
    run.add_argument('--activation-quant',action='store_true',help='public JAX A8 fake quantization; does not emulate the production integer kernel')
    run.add_argument('--retrieval',action='store_true',help='use NativeProbeEncoder to retrieve top-k tools')
    run.add_argument('--top-k-tools',type=int,default=None,help='maximum number of retrieved tools in prompt')
    args=p.parse_args()
    if args.command=='build-native':
        from .native import build_native, features, sdot_available
        try:
            result=dict(library=str(build_native().resolve()),features=features(),sdot_available=sdot_available())
        except (OSError, RuntimeError, AttributeError) as exc:
            p.error(str(exc))
    elif args.command=='inspect':
        from .archive import Archive
        a=Archive.load(args.model)
        result=dict(geometry=a.metadata,storage=a.stats())
    elif args.command=='to-torch':
        from .convert import import_model
        result=import_model(args.source,args.output,args.template)
    elif args.command=='quantize':
        from .convert import export_model
        result=export_model(args.checkpoint,args.output,force_requantize=args.force_requantize)
    else:
        from .inference import generate
        tools=json.loads(Path(args.tools).read_text()) if args.tools else None
        result=generate(args.model,args.prompt,tools=tools,system=args.system,backend=args.backend,
                        max_new_tokens=args.max_new_tokens,threads=args.threads,quant_activations=args.activation_quant,
                        prefill_backend=args.prefill_backend,constrain=not args.no_grammar,matmul=args.matmul,
                        kv_cache=args.kv_cache,retrieval=args.retrieval,top_k_tools=args.top_k_tools)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
