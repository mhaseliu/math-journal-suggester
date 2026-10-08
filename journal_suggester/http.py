"""Public metadata HTTP client with a raw-response cache and bounded retries."""
import hashlib
import json
import os
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .io import write_json, write_text


class CachedHTTP:
    def __init__(self, cache="data/raw/http", offline=False, timeout=40, attempts=5):
        self.cache = Path(cache)
        self.offline = offline
        self.timeout, self.attempts = timeout, attempts
        self.last = {}
        self.rate_lock = threading.Lock()
        self.arxiv_lock = threading.Lock()
        self.arxiv_retry_at = 0

    def get(self, url, params=None):
        host = urllib.parse.urlsplit(url).hostname
        if host in ("export.arxiv.org", "arxiv.org"):
            # arXiv asks for one connection and a pause between completed requests.
            with self.arxiv_lock:
                try:
                    return self._get(url, params)
                finally:
                    self.last[host] = time.monotonic()
        return self._get(url, params)

    def _get(self, url, params=None):
        params = dict(params or {})
        host = urllib.parse.urlsplit(url).hostname
        if host == "api.openalex.org" and os.getenv("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        public_params = {k: v for k, v in params.items() if k != "api_key"}
        public_url = url + (("?" + urllib.parse.urlencode(public_params)) if public_params else "")
        actual_url = url + (("?" + urllib.parse.urlencode(params)) if params else "")
        key = hashlib.sha256(public_url.encode()).hexdigest()
        path = self.cache / (key + ".body")
        if path.exists():
            return path.read_text(encoding="utf-8")
        if host in ("export.arxiv.org", "arxiv.org") and time.monotonic() < self.arxiv_retry_at:
            raise RuntimeError("arXiv fallback cooling down after HTTP 429")
        if self.offline:
            raise RuntimeError(f"Not cached: {public_url}")
        delay = 3.1 if host in ("export.arxiv.org", "arxiv.org") else 0.3
        headers = {"User-Agent": "journal-suggester/0.1 (local research experiment)"}
        if os.getenv("CROSSREF_MAILTO"):
            headers["User-Agent"] += f" mailto:{os.environ['CROSSREF_MAILTO']}"
        for attempt in range(self.attempts):
            with self.rate_lock:
                scheduled = max(time.monotonic(), self.last.get(host, 0) + delay)
                self.last[host] = scheduled
            time.sleep(max(0, scheduled - time.monotonic()))
            try:
                with urllib.request.urlopen(urllib.request.Request(actual_url, headers=headers), timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                    write_text(path, body)
                    write_json(self.cache / (key + ".json"), {"url": public_url, "fetched_at": time.time(),
                               "status": response.status, "content_type": response.headers.get("Content-Type"),
                               "sha256": hashlib.sha256(body.encode()).hexdigest()})
                    return body
            except urllib.error.HTTPError as e:
                if e.code == 429 and host in ("export.arxiv.org", "arxiv.org"):
                    try:
                        cooldown = float(e.headers.get("Retry-After", 300))
                    except ValueError:
                        cooldown = 300
                    self.arxiv_retry_at = time.monotonic() + max(300, cooldown)
                    print("arXiv requested a cooldown; continuing with other verified abstract sources.", flush=True)
                    raise RuntimeError(f"HTTP 429: {public_url}") from None
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.attempts - 1:
                    # Do not expose API keys in error messages or logs.
                    raise RuntimeError(f"HTTP {e.code}: {public_url}") from None
                try:
                    retry = float(e.headers.get("Retry-After", 0))
                except ValueError:
                    retry = 0
                if e.code == 429:
                    retry = max(60, retry)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.attempts - 1:
                    raise RuntimeError(f"Network unavailable: {public_url}") from None
                retry = 0
            time.sleep(min(60, max(retry, 2 ** attempt)))

    def json(self, url, params=None):
        return json.loads(self.get(url, params))
