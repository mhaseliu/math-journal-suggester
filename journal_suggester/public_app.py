"""Bounded public WSGI gateway. Model/metadata workers use local Unix sockets."""
import argparse
import ipaddress
import json
import logging
import re
import threading
import time
from collections import deque
from http import HTTPStatus
from pathlib import Path

from .app import ASSETS, STATIC
from .paper_import import ARXIV_SCOPE_MESSAGE, ARXIV_UNAVAILABLE_MESSAGE, normalize_arxiv
from .web_service import validate_query
from .web_rpc import rpc


class RateLimit:
    """Memory-only sliding windows; both global and per-visitor work budgets."""
    def __init__(self, clock=time.monotonic):
        self.clock, self.lock = clock, threading.Lock()
        self.global_events, self.clients = deque(), {}

    def allow(self, client):
        with self.lock:
            now = self.clock()
            while self.global_events and self.global_events[0] <= now - 3600:
                self.global_events.popleft()
            # Bound the IP table even when visitors rotate addresses.
            for key in list(self.clients):
                events = self.clients[key]
                while events and events[0] <= now - 3600:
                    events.popleft()
                if not events:
                    del self.clients[key]
            events = self.clients.get(client, deque())
            if (len(self.global_events) >= 300 or sum(t > now - 60 for t in self.global_events) >= 12
                    or len(events) >= 40 or sum(t > now - 60 for t in events) >= 6):
                return False
            self.global_events.append(now)
            events.append(now)
            self.clients[client] = events
            return True


class PublicApplication:
    def __init__(self, hostname_file, model_socket, metadata_socket, records_dir, call=rpc,
                 proxy="cloudflare"):
        if proxy not in {"cloudflare", "tailscale"}:
            raise ValueError("Unknown public proxy")
        self.proxy = proxy
        self.hostname_file, self.records_dir = Path(hostname_file), Path(records_dir)
        self.model_socket, self.metadata_socket, self.call = model_socket, metadata_socket, call
        self.limit = RateLimit()
        self.model_lock, self.metadata_lock = threading.Lock(), threading.Lock()

    def __call__(self, environ, start_response):
        def send(code, value, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            headers = [("Content-Type", content_type + "; charset=utf-8"), ("Content-Length", str(len(body))),
                       ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                       ("Referrer-Policy", "no-referrer"), ("X-Frame-Options", "DENY"),
                       ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
                       ("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                        "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                        "frame-ancestors 'none'; form-action 'self'")]
            if code in (429, 503):
                headers.append(("Retry-After", "60" if code == 429 else "5"))
            start_response(f"{code} {HTTPStatus(code).phrase}", headers)
            return [b"" if environ.get("REQUEST_METHOD") == "HEAD" else body]

        try:
            host = self.hostname_file.read_text().strip()
        except FileNotFoundError:
            return send(503, {"error": "The site is starting. Please try again shortly."})
        if not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}", host):
            return send(503, {"error": "The site is not configured."})
        if environ.get("HTTP_HOST") != host:
            return send(403, {"error": "Invalid site address."})
        method, path = environ.get("REQUEST_METHOD"), environ.get("PATH_INFO")
        if method in ("GET", "HEAD"):
            if path in ASSETS:
                filename, content_type = ASSETS[path]
                return send(200, (STATIC / filename).read_bytes(), content_type)
            prefix = "/methodology/records/"
            # Exact filenames only, never user-controlled filesystem paths.
            if path in {prefix + name for name in ("tuning-b300.json", "training-8000-config.json",
                                                  "training-8000-preflight.json")}:
                record = self.records_dir / path.removeprefix(prefix)
                if record.is_file():
                    return send(200, record.read_bytes())
            return send(404, {"error": "Not found"})
        if method != "POST":
            return send(405, {"error": "Method not allowed"})
        if path not in {"/api/suggest", "/api/import/arxiv"}:
            return send(404, {"error": "Not found"})
        # JSON + exact Origin reject cross-site form/JS submissions. No public CORS.
        if (environ.get("HTTP_ORIGIN") != "https://" + host
                or environ.get("HTTP_SEC_FETCH_SITE", "same-origin") != "same-origin"):
            return send(403, {"error": "Submit through this website."})
        if environ.get("CONTENT_TYPE", "").split(";", 1)[0].strip().lower() != "application/json":
            return send(415, {"error": "Use a title and abstract or an arXiv link."})
        try:
            length = int(environ.get("CONTENT_LENGTH", "0"))
        except ValueError:
            length = 0
        maximum = 2000 if path.endswith("/arxiv") else 150000
        if environ.get("HTTP_TRANSFER_ENCODING") or not 0 < length <= maximum:
            return send(413, {"error": "Request is empty or too large."})
        try:
            # This header is trusted only because the HTTP Unix socket is accessible
            # solely to the tunnel account. Never listen on a public TCP port.
            # Tailscale 1.102.4 overwrites X-Forwarded-For from its connection
            # context. Never use CF's visitor-supplied header on a Funnel route.
            field = "HTTP_X_FORWARDED_FOR" if self.proxy == "tailscale" else "HTTP_CF_CONNECTING_IP"
            address = ipaddress.ip_address(environ.get(field, ""))
            if address.version == 6 and address.ipv4_mapped:
                address = address.ipv4_mapped
            client = str(ipaddress.ip_network(f"{address}/64", strict=False)) if address.version == 6 else str(address)
        except ValueError:
            return send(403, {"error": "Use the public website address."})
        if not self.limit.allow(client):
            return send(429, {"error": "Search limit reached. Please try again later."})
        lock = self.metadata_lock if path.endswith("/arxiv") else self.model_lock
        if not lock.acquire(blocking=False):
            return send(503, {"error": "Another request is running. Please try again in a moment."})
        try:
            raw = environ["wsgi.input"].read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request.")
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Provide a title and abstract or an arXiv link.")
            if path.endswith("/arxiv"):
                result = self.call(self.metadata_socket, {"url": normalize_arxiv(body.get("url"))}, timeout=25)
                if result.get("error") == "arxiv_out_of_scope":
                    return send(400, {"error": ARXIV_SCOPE_MESSAGE})
                if result.get("error") == "arxiv_unavailable":
                    return send(503, {"error": ARXIV_UNAVAILABLE_MESSAGE})
            else:
                result = self.call(self.model_socket, validate_query(body), timeout=90)
            if result.get("error"):
                return send(503, {"error": "Could not complete this request. Please try again or paste your abstract."})
            return send(200, result)
        except (ValueError, UnicodeError):
            return send(400, {"error": "Check the arXiv link, title and abstract, then try again."})
        except (OSError, RuntimeError):
            return send(503, {"error": "Search is temporarily unavailable. Please try again shortly."})
        except Exception:
            # Never log manuscript content or exception messages containing input.
            logging.error("Public request failed")
            return send(500, {"error": "Could not complete this request."})
        finally:
            lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname-file", default="/etc/journal-suggester/hostname")
    parser.add_argument("--http-socket", default="/run/journal-web/http.sock")
    parser.add_argument("--model-socket", default="/run/journal-model/api.sock")
    parser.add_argument("--metadata-socket", default="/run/journal-metadata/api.sock")
    parser.add_argument("--records", default="/opt/journal-suggester/app/public-records")
    parser.add_argument("--proxy", choices=["cloudflare", "tailscale"], default="cloudflare")
    args = parser.parse_args()
    from waitress import serve
    app = PublicApplication(args.hostname_file, args.model_socket,
                            args.metadata_socket, args.records,
                            proxy=args.proxy)
    proxy_options = waitress_proxy_options(args.proxy)
    serve(app, unix_socket=args.http_socket, unix_socket_perms="660", threads=4,
          connection_limit=32, backlog=32, channel_timeout=25, cleanup_interval=5,
          max_request_body_size=150000, max_request_header_size=16384,
          expose_tracebacks=False, ident="", url_scheme="https", **proxy_options)


def waitress_proxy_options(proxy):
    if proxy == "tailscale":
        # Funnel's Unix-socket proxy sends Host: localhost. Recover the original
        # host only from headers it overwrites; the private socket has one proxy.
        return {"trusted_proxy": "localhost", "trusted_proxy_count": 1,
                "trusted_proxy_headers": {"x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}}
    return {}


if __name__ == "__main__":
    main()
