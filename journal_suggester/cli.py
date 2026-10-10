"""Serve the selected Kev model or recommend journals from a JSON manuscript."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Math journal recommendations with fine-tuned Kev')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('serve','suggest'):
        p=sub.add_parser(command)
        p.add_argument('--kev-run',required=True,help='Directory containing the verified adapter package')
        p.add_argument('--split',help='Optional directory containing reference.jsonl for similar papers')
        p.add_argument('--vectors',help='Optional Qwen reference vectors, paired with --split')
        if command=='serve': p.add_argument('--port',type=int,default=8765)
        else: p.add_argument('--paper',type=Path,required=True,help='JSON containing title and abstract')
    args=parser.parse_args()
    if args.command=='serve':
        from .app import serve
        serve(args.split,args.port,args.vectors,args.kev_run)
    else:
        from .web_service import SearchService
        service=SearchService(args.split,args.vectors,args.kev_run)
        print(json.dumps(service.suggest(json.loads(args.paper.read_text())),indent=2))


if __name__=='__main__': main()
