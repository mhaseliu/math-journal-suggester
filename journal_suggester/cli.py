import argparse
import json

from .io import read_json


def main():
    parser = argparse.ArgumentParser(description="Mathematics journal recommender experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="CPU-only demo with fictional supporting papers")
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("catalog", help="Extract and verify the 95-journal catalog")
    p.add_argument("--offline", action="store_true")
    p = sub.add_parser("collect", help="Collect a capped real-paper pilot")
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--output", default="data/processed/pilot")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--crossref-only", action="store_true")
    p = sub.add_parser("split", help="Freeze disjoint duplicate-safe splits")
    p.add_argument("--input", default="data/processed/pilot/papers.jsonl")
    p.add_argument("--output", default="artifacts/pilot/splits")
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--publication-counts", help="JSON mapping journals to audited eligible counts")
    p = sub.add_parser("prepare", help="Build candidates, evidence and Kev requests")
    p.add_argument("--split", default="artifacts/pilot/splits")
    p.add_argument("--output", default="artifacts/pilot/lexical")
    p.add_argument("--vectors", help="Use cached Qwen vectors; omit for lexical smoke baseline")
    p.add_argument("--partitions", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    p.add_argument("--candidate-order", choices=["retrieval", "shuffle"], default="retrieval")
    p = sub.add_parser("embed", help="Cache frozen Qwen embeddings on GB10 only")
    p.add_argument("--split", default="artifacts/pilot/splits")
    p.add_argument("--output", default="artifacts/pilot/vectors")
    p.add_argument("--partitions", nargs="+", choices=["reference", "train", "validation", "test"], default=["reference", "train", "validation", "test"])
    p = sub.add_parser("serve", help="Preview the journal website on localhost")
    p.add_argument("--split", default="artifacts/experiment/splits")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--vectors")
    p.add_argument("--kev-run")
    p.add_argument("--ssh-backend", help="Run semantic retrieval on this GB10 SSH host; connects on first search")
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import serve_demo
        serve_demo(args.port)
    elif args.command == "catalog":
        from .catalog import extract, verify
        extract()
        verify(offline=args.offline)
    elif args.command == "collect":
        from .collect import collect
        c = read_json(args.config)
        collect(args.output, c["journal_ids"], c["per_journal"], c["max_records"], args.offline, not args.crossref_only)
    elif args.command == "split":
        from .splits import split
        c = read_json(args.config)
        print(json.dumps(split(args.input, args.output, c["train_queries"], c["validation_queries"], c["test_queries"],
                               c["seed"], read_json(args.publication_counts) if args.publication_counts else None,
                               c.get("minimum_per_journal", {}).get("validation")), indent=2))
    elif args.command == "prepare":
        from .pipeline import prepare
        print(json.dumps(prepare(args.split, args.output, args.vectors, partitions=args.partitions, candidate_order=args.candidate_order), indent=2))
    elif args.command == "embed":
        from .gpu import embed_split
        embed_split(args.split, args.output, read_json("configs/models.json"), args.partitions)
    elif args.command == "serve":
        from .app import serve
        serve(args.split, args.port, args.vectors, args.kev_run, args.ssh_backend)


if __name__ == "__main__":
    main()
