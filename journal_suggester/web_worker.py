"""Optional GB10 worker; the website sends one JSON query per line over SSH."""
import contextlib
import json
import logging
import sys

from .web_service import DEFAULT_SPLIT, ROOT, SearchService


def main():
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        service = SearchService(DEFAULT_SPLIT, __import__("os").environ.get("JOURNAL_VECTORS_DIR", "artifacts/experiment/vectors"))
    output.write(json.dumps({"ready": service.info()}) + "\n")
    output.flush()
    for line in sys.stdin:
        try:
            if len(line) > 150000:
                raise ValueError("Query is too large")
            with contextlib.redirect_stdout(sys.stderr):
                result = service.suggest(json.loads(line))
        except Exception:
            logging.exception("GB10 search failed")
            result = {"error": "GB10 could not process the search. Check the remote environment and available memory."}
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
