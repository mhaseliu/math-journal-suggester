"""Length-bounded JSON over filesystem Unix sockets; no pickle or shell commands."""
import argparse
import json
import os
import socket
import socketserver
from pathlib import Path

MAX_REQUEST = 150000
MAX_RESPONSE = 1024 * 1024


def read_message(stream, limit):
    line = stream.readline(limit + 1)
    if len(line) > limit or not line.endswith(b"\n"):
        raise ValueError("Invalid message length")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value


def rpc(path, body, timeout=90):
    raw = json.dumps(body, ensure_ascii=False).encode() + b"\n"
    if len(raw) > MAX_REQUEST:
        raise ValueError("Request too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(str(path))
        conn.sendall(raw)
        with conn.makefile("rb") as stream:
            return read_message(stream, MAX_RESPONSE)


def make_rpc_server(path, callback):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(25)
            try:
                value = callback(read_message(self.rfile, MAX_REQUEST))
                raw = json.dumps(value, ensure_ascii=False).encode() + b"\n"
                if len(raw) > MAX_RESPONSE:
                    raise ValueError("Response too large")
            except Exception:
                raw = b'{"error":"Request failed"}\n'
            try:
                self.wfile.write(raw)
            except OSError:
                pass
    # One worker at a time, bounded listen backlog; no unbounded thread creation.
    class Server(socketserver.UnixStreamServer):
        request_queue_size = 2
    path = Path(path)
    if path.exists():
        path.unlink()
    server = Server(str(path), Handler)
    os.chmod(path, 0o660)
    return server


def import_metadata(importer, body):
    from .paper_import import ArxivScopeError, ArxivUnavailableError
    try:
        return importer.fetch(body.get("url"))
    except ArxivScopeError:
        # A fixed code crosses the service boundary; arbitrary exception text does not.
        return {"error": "arxiv_out_of_scope"}
    except ArxivUnavailableError:
        return {"error": "arxiv_unavailable"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["model", "metadata"])
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--split", default=os.environ.get("JOURNAL_SPLIT_DIR", "artifacts/experiment/splits"))
    parser.add_argument("--vectors", default=os.environ.get("JOURNAL_VECTORS_DIR", "artifacts/experiment/vectors"))
    parser.add_argument("--kev-checkpoint")
    parser.add_argument("--verification-fixture", type=Path)
    args = parser.parse_args()
    if args.kind == "model":
        from .web_service import DEFAULT_SPLIT, ROOT, SearchService
        service = SearchService(args.split, args.vectors,
                                kev_run=args.kev_checkpoint)
        if args.kev_checkpoint:
            if not args.verification_fixture:
                raise ValueError("Kev startup requires the verified parity fixtures")
            comparison = service.ranker.verify_reference(json.loads(args.verification_fixture.read_text()))
            if not all(c['max_probability_delta'] <= .02 and c['top1_matches'] and c['top3_set_matches'] for c in comparison):
                raise RuntimeError("GB10 checkpoint parity failed; refusing to serve")
            print(json.dumps({"checkpoint_parity_pass": True, "checks": comparison}), flush=True)
        # Warm before opening the socket; tunnel users never trigger cold startup.
        service.suggest({"title": "Spectral gaps of finite graphs", "abstract":
                         "We study spectral gaps, random walks and mixing times on finite regular graphs."})
        callback = service.suggest
        print(json.dumps({"ready": True, "ranking": service.info()["ranking"]}), flush=True)
    else:
        from .paper_import import ArxivImporter
        importer = ArxivImporter()
        callback = lambda body: import_metadata(importer, body)
    with make_rpc_server(args.socket or f"/run/journal-{args.kind}/api.sock", callback) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
