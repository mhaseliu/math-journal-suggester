"""Publisher-archive fallback for NYJM, whose Crossref journal endpoint is absent."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import re
import urllib.parse

from .http import CachedHTTP
from .collect import UNSUITABLE_TITLE
from .io import journals, write_json, write_jsonl
from .records import clean_text, deduplicate


def parse_nyjm(body, year):
    pattern = rf'/j/{year}/{year - 1994}-\d+\.html'
    links = set(re.findall(r'<a\b[^>]*href="(' + pattern + r')"', body, re.I))
    rows, rejected, matched = [], [], set()
    for block in re.split(r'<hr\b[^>]*>', body, flags=re.I):
        urls = set(re.findall(r'<a\b[^>]*href="(' + pattern + r')"', block, re.I))
        if not urls:
            continue
        titles = re.findall(r'<td\b[^>]*class="[^"]*\bstyle60\b[^"]*"[^>]*>(.*?)</td>', block, re.I | re.S)
        if len(urls) != 1 or len(titles) != 1:
            raise ValueError(f"NYJM {year}: cannot uniquely pair archive title and article URL")
        path = urls.pop()
        matched.add(path)
        title = clean_text(titles[0])
        url = urllib.parse.urljoin("https://nyjm.albany.edu", path)
        if not title:
            raise ValueError(f"NYJM {year}: empty title")
        row = {"paper_id": url, "title": title, "abstract": "", "journal_id": "new-york-journal-of-mathematics",
               "year": year, "doi": "", "arxiv_id": "", "url": url}
        if UNSUITABLE_TITLE.match(title):
            rejected.append(row)
        else:
            rows.append(row)
    if not links or matched != links:
        raise ValueError(f"NYJM {year}: incomplete archive parse")
    return rows, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/publication-counts")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    http = CachedHTTP(cache="data/raw/publisher-counts-http", offline=args.offline)
    jid = "new-york-journal-of-mathematics"
    rows, rejected, sources = [], [], []
    for year in range(2016, 2026):
        url = f"https://nyjm.albany.edu/j/{year}/Vol{year - 1994}.htm"
        eligible, bad = parse_nyjm(http.get(url), year)
        rows.extend(eligible)
        rejected.extend(bad)
        sources.append({"url": url, "year": year, "eligible": len(eligible), "rejected": len(bad), "complete": True})
        print(f"NYJM {year}: {len(eligible)} eligible, {len(bad)} rejected", flush=True)
    unique, conflicts = deduplicate(rows)
    if conflicts:
        raise ValueError("Unexpected conflicting publisher rows")
    directory = args.output + "/publishers/" + jid
    write_jsonl(directory + "/eligible.jsonl", unique)
    write_jsonl(directory + "/rejected.jsonl", rejected)
    write_json(directory + "/audit.json", {
        "journal_name": "New York Journal of Mathematics", "issns": "1076-9803", "status": "complete",
        "scan_complete": True, "source_kind": "publisher_archive", "sources": sources, "errors": [],
        "unique_record_keys": len(rows) + len(rejected), "eligible_dois_before_title_dedup": None,
        "rejected": {"unsuitable_title": len(rejected)}, "finished_at": datetime.now(timezone.utc).isoformat(),
        "year_source": "Publisher's annual journal volume", "doi_coverage": "Not required; stable publisher article URLs used",
        "by_year": dict(Counter(p["year"] for p in unique)),
    })
    from .counts import aggregate
    aggregate(args.output, journals())


if __name__ == "__main__":
    main()
