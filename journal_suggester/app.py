"""Private loopback web preview; imports never run neural models locally."""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .paper_import import ArxivImporter
from .web_service import SearchService

STATIC = Path(__file__).with_name("web")
ASSETS = {"/": ("index.html", "text/html"), "/methodology": ("methodology.html", "text/html"),
          "/app.css": ("app.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
ROOT = STATIC.parents[1]
# Only aggregate experiment records are served; never expose the artifact/data directories.
from .resources import ROOT as RESOURCE_ROOT
RECORDS = {
    "/methodology/records/test.json": RESOURCE_ROOT / "results/test.json",
    "/methodology/records/training.json": RESOURCE_ROOT / "results/training.json",
    "/methodology/records/tuning.json": RESOURCE_ROOT / "results/tuning.json",
    "/methodology/records/config.json": RESOURCE_ROOT / "results/config.json",
}


def make_server(service, port=8765, importer=None):
    importer = importer or ArxivImporter()
    info = service.info()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(25)

        def log_message(self, *args):
            pass  # Do not retain paper text, filenames or request URLs.

        def send(self, code, value, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self.send(403, {"error": "This preview is available on localhost only."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in expected}:
                self.send(403, {"error": "Requests must come from this preview."})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            if path in ASSETS:
                name, content_type = ASSETS[path]
                self.send(200, (STATIC / name).read_bytes(), content_type)
            elif path in RECORDS and RECORDS[path].is_file():
                self.send(200, RECORDS[path].read_bytes())
            elif path == "/api/info":
                self.send(200, info)
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path not in {"/api/suggest", "/api/import/arxiv"}:
                return self.send(404, {"error": "Not found"})
            try:
                limit = 2000 if self.path == "/api/import/arxiv" else 150000
                length = int(self.headers.get("Content-Length", 0))
                if self.headers.get("Transfer-Encoding") or not 0 < length <= limit:
                    return self.send(413, {"error": "Request is empty or too large."})
                if self.headers.get_content_type() != "application/json":
                    return self.send(415, {"error": "Unsupported request format."})
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("Request was interrupted. Please try again.")
                body = json.loads(data)
                if not isinstance(body, dict):
                    raise ValueError("Provide a JSON object.")
                result = importer.fetch(body.get("url")) if self.path.endswith("/arxiv") else service.suggest(body)
                self.send(200, result)
            except (ValueError, UnicodeError) as exc:
                self.send(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.send(503, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                logging.exception("Website request failed")
                self.send(500, {"error": "The search could not be completed. Please try again."})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(split_dir=None, port=8765, vectors=None, kev_run=None):
    service = SearchService(split_dir, vectors, kev_run)
    server = make_server(service, port)
    print(f"{service.info()['mode']}: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
