"""Publisher enumeration and bibliographic-year reconciliation for the two series."""
import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import html
import io
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.error
import urllib.request
from urllib.parse import urljoin

from .http import CachedHTTP
from .collect import UNSUITABLE_TITLE
from .io import digest, journals, read_json, write_json, write_jsonl, write_text
from .records import clean_text

CACHE = "data/raw/series-reconciliation-http"
ROOT = Path("data/processed/series-reconciliation")
SMF = "https://smf.emath.fr"
AMS_INDEX = "https://www.ams.org/memo/1143/"
MATCHER = "https://zbmath.org/citationmatching/match"


def normalized(text):
    text = re.sub(r"\\(?:mathbb|mathcal|mathrm|mathbf|mathfrak|operatorname|text|mathsf|mathscr)\b", "", text)
    text = unicodedata.normalize("NFKD", clean_text(text)).casefold()
    return "".join(c for c in text if c.isalnum())


def check_memoir_matches(title, authors, number, matches):
    checked = []
    for match in matches:
        source = match.get("source", "")
        similarity = SequenceMatcher(None, normalized(title), normalized(match.get("title", ""))).ratio()
        families = {normalized(a.split(",")[0]) for a in match.get("authors", "").split(";")}
        author_match = bool(authors) and normalized(authors[0]) in families
        series_match = re.search(r"(?:Mem\.\s*Am\.\s*Math\.\s*Soc\.|Memoirs of the American Mathematical Society)", source)
        number_match = bool(series_match and re.search(r"(?<!\d)" + str(number) + r"(?!\d)", source))
        accepted = similarity >= .9 and author_match and number_match and bool(re.fullmatch(r"\d{4}", match.get("year", "")))
        checked.append({**match, "title_similarity": similarity, "author_match": author_match,
                        "number_match": number_match, "accepted": accepted})
    return checked


def ams_index(body):
    match = re.search(r"productDetailObj\s*=\s*(\{)", body)
    if not match:
        raise ValueError("AMS product metadata absent")
    product, _ = json.JSONDecoder().raw_decode(body[match.start(1):])
    entries = product["SeriesVolumes"]
    numbers = {}
    for entry in entries:
        code = re.fullmatch(r"MEMO/(\d+)/(\d+)", entry["ProductCode"])
        if not code:
            raise ValueError("AMS series code mismatch")
        number = int(code[2])
        if number in numbers:
            raise ValueError("Duplicate AMS memoir number")
        numbers[number] = {"number": number, "volume": int(code[1]),
                           "copyright_year": int(entry["CopyrightYear"]),
                           "publisher_issue_number": entry["IssueNumber"],
                           "publisher_number_conflict": number != entry["IssueNumber"],
                           "url": "https://pubs.ams.org/view?ProductCode=" + entry["ProductCode"]}
    return numbers


def smf_cards(body, year):
    rows = []
    for block in re.findall(r'<a\b[^>]*class="publication [\s\S]*?</a>', body):
        def field(name):
            m = re.search(r'class="publication__' + name + r'"[^>]*>(.*?)</(?:h3|div)>', block, re.S)
            if not m:
                raise ValueError("Missing SMF card field: " + name)
            return clean_text(m[1])
        info = field("more_info")
        match = re.fullmatch(r"(Livre|Article|Fascicule) - Tome ([\d-]+)(?: - pp (.*?))? - (\d{4})", info)
        if not match or int(match[4]) != year or "Astérisque" not in field("collection"):
            raise ValueError("Unexpected SMF card: " + info)
        rows.append({"title": field("title"), "authors": field("text"), "year": year,
                     "kind": match[1], "volume": match[2], "pages": match[3] or "",
                     "url": urljoin(SMF, html.unescape(re.search(r'href="([^"]+)"', block)[1]))})
    return rows


def collect_smf(http):
    body = http.get(SMF + "/les-publications")
    years = dict((int(year), value) for value, year in re.findall(r'<option value="(\d+)">(20\d{2})</option>', body))
    rows, sources, seen = [], [], set()
    for year in range(2016, 2026):
        for page in range(100):
            params = {"field_publication_collection[]": "144", "field_year_of_release[]": years[year], "page": page}
            cards = smf_cards(http.get(SMF + "/les-publications", params), year)
            sources.append({"year": year, "page": page, "params": params, "entries": len(cards)})
            if not cards:
                break
            for card in cards:
                if card["url"] in seen:
                    raise ValueError("SMF pagination repeated a publication")
                seen.add(card["url"])
                rows.append(card)
            write_json(ROOT / "smf-catalog.json", {"entries": rows, "sources": sources, "complete": False})
        else:
            raise ValueError("SMF pagination cap reached")
        print(f"Astérisque {year}: {sum(r['year'] == year for r in rows)} catalog entries", flush=True)
    write_json(ROOT / "smf-catalog.json", {"entries": rows, "sources": sources, "complete": True})


def collect_smf_books(http):
    catalog = read_json(ROOT / "smf-catalog.json")
    if not catalog["complete"]:
        raise ValueError("Incomplete SMF catalog")
    books = [r for r in catalog["entries"] if r["kind"] == "Livre"]
    for i, book in enumerate(books):
        path = ROOT / "smf-books" / (book["volume"] + ".json")
        if path.exists():
            continue
        body = http.get(book["url"])
        dois = re.findall(r'<strong>DOI\s*:</strong>\s*([^<\s]+)', body)
        toc = []
        for match in re.finditer(r'class="product__data-contents--c item-list"[^>]*>\s*<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.S):
            title = re.search(r"<strong>(.*?)</strong>", match[2], re.S)
            if not title:
                raise ValueError("Untitled SMF contents entry")
            toc.append({"url": urljoin(SMF, html.unescape(match[1])), "title": clean_text(title[1]),
                        "text": clean_text(match[2])})
        write_json(path, {**book, "dois": sorted(set(dois)), "contents": toc,
                          "sample_urls": sorted(set(urljoin(SMF, u) for u in re.findall(r'href="([^"]+__sample\.pdf)"', body)))})
        if (i + 1) % 10 == 0:
            print(f"Astérisque volumes checked: {i + 1}/{len(books)}", flush=True)


def asterisque_rows(catalog, books, exclusions, ignored_dois=None):
    ignored_dois = ignored_dois or {}
    groups = defaultdict(list)
    for row in catalog:
        groups[row["volume"]].append(row)
    rows, rejected, recovered = [], [], []
    for volume, entries in sorted(groups.items()):
        book = books[volume]
        if volume in exclusions:
            rejected.append({"volume": volume, "reason": "reprint", "review": exclusions[volume]})
            continue
        if len({r["year"] for r in entries}) != 1:
            raise ValueError("Conflicting publication years within SMF volume")
        articles = {r["url"]: r for r in entries if r["kind"] == "Article"}
        contents = {r["url"]: r for r in book["contents"]}
        if articles.keys() - contents.keys():
            raise ValueError("Catalog articles absent from publisher table of contents")
        if contents:
            rejected.append({"volume": volume, "reason": "enclosing_collection", "url": book["url"]})
            selected = []
            for url, child in contents.items():
                if url in articles:
                    selected.append(articles[url])
                else:
                    child = {**child, "year": book["year"], "volume": volume, "kind": "Article",
                             "year_source": book["url"]}
                    recovered.append(child)
                    selected.append(child)
        else:
            if len(entries) != 1 or entries[0]["kind"] != "Livre":
                raise ValueError("Ambiguous standalone SMF volume")
            selected = [book]
        for work in selected:
            if UNSUITABLE_TITLE.match(work["title"]):
                rejected.append({"volume": volume, "reason": "unsuitable_title", "title": work["title"], "url": work["url"]})
                continue
            dois = work.get("dois", [])
            if volume in ignored_dois:
                invalid = ignored_dois[volume]["doi"]
                if invalid not in dois:
                    raise ValueError("Reviewed SMF DOI no longer matches publisher metadata")
                dois = [doi for doi in dois if doi != invalid]
            doi = dois[0] if len(dois) == 1 else ""
            rows.append({"paper_id": "doi:" + doi if doi else work["url"], "title": work["title"],
                         "abstract": "", "journal_id": "asterisque", "year": work["year"],
                         "doi": doi, "arxiv_id": "", "url": work["url"], "volume": volume})
    return rows, rejected, recovered


def export_asterisque(output):
    catalog = read_json(ROOT / "smf-catalog.json")
    if not catalog["complete"]:
        raise ValueError("Incomplete SMF catalog")
    books = {p.stem: read_json(p) for p in (ROOT / "smf-books").glob("*.json")}
    overrides = read_json("data/series_overrides.json")
    rows, rejected, recovered = asterisque_rows(catalog["entries"], books, overrides["asterisque_excluded_volumes"],
                                               overrides.get("asterisque_ignored_dois"))
    directory = Path(output) / "publishers/asterisque"
    write_jsonl(directory / "eligible.jsonl", rows)
    write_json(directory / "reconciliation.json", {"excluded": rejected, "recovered_from_contents": recovered,
               "ignored_dois": overrides.get("asterisque_ignored_dois", {})})
    write_json(directory / "audit.json", {
        "journal_name": "Astérisque", "status": "complete", "scan_complete": True,
        "source_kind": "publisher_catalog_and_contents", "sources": catalog["sources"], "errors": [],
        "catalog_entries": len(catalog["entries"]), "volumes_checked": len(books),
        "eligible_works": len(rows), "rejected": dict(Counter(r["reason"] for r in rejected)),
        "recovered_from_contents": len(recovered), "finished_at": time.time(),
        "year_source": "Publisher publication year, excluding reviewed reprints",
        "method": "Standalone monographs or individual contributions, never their enclosing collections"})
    from .counts import aggregate
    aggregate(output, journals())
    print("Astérisque eligible works:", len(rows), dict(sorted(Counter(r["year"] for r in rows).items())))


def memoirs_rows(records, overrides, window):
    """Apply reviewed evidence; unresolved window membership never becomes a zero."""
    indexed = {r["number"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate memoir number")
    expected = set(range(window["first_number"], window["last_number"] + 1))
    if expected - indexed.keys():
        raise ValueError("Missing memoir numbers within reviewed window")
    for number, year in window["boundary_checks"].items():
        if indexed[int(number)].get("year") != year:
            raise ValueError("Memoirs boundary evidence changed")
    rows, ledger, unresolved = [], [], []
    for number, record in sorted(indexed.items()):
        year, source, url = None, "", ""
        if record["status"] == "matched":
            accepted = [m for m in record["matches"] if m["accepted"]]
            if len(accepted) != 1 or int(accepted[0]["year"]) != record["year"]:
                raise ValueError("Inconsistent accepted citation")
            year, source = record["year"], "zbmath_bibliographic_citation"
            url = "https://zbmath.org/?q=an:" + accepted[0]["zbl_id"]
        override = overrides.get(str(number))
        if override:
            if override["doi"] != record.get("doi") or not override.get("evidence"):
                raise ValueError("Reviewed memoir DOI/evidence mismatch")
            year, source, url = override["year"], override["source_kind"], override["evidence"][0]
        within = window["first_volume"] <= record["volume"] <= window["last_volume"]
        if within != (number in expected):
            raise ValueError("Memoir volume and number windows disagree")
        if override and override.get("exclude"):
            status = "excluded_after_review"
            year = None
            if override["source_kind"] == "user_manual_review":
                url = ""
        elif UNSUITABLE_TITLE.match(record.get("title", "")):
            status = "excluded_unsuitable_title"
        elif within and year is None:
            status = "unresolved"
            unresolved.append({"number": number, "doi": record.get("doi"), "review": override})
        elif within:
            if not 2016 <= year <= 2025:
                raise ValueError("Memoir year contradicts reviewed window")
            status = "confirmed"
            rows.append({"paper_id": "doi:" + record["doi"], "doi": record["doi"],
                         "title": record["title"], "abstract": "", "arxiv_id": "",
                         "journal_id": "memoirs-of-the-american-mathematical-society", "year": year,
                         "url": "https://doi.org/" + record["doi"], "number": number,
                         "volume": record["volume"], "year_source": source, "year_evidence": url})
        else:
            if year is not None and 2016 <= year <= 2025:
                raise ValueError("Published memoir lies outside reviewed number window")
            status = "outside_volume_window"
        ledger.append({"number": number, "volume": record["volume"], "year": year,
                       "status": status, "doi": record.get("doi", ""),
                       "year_source": source, "source_url": url})
    return rows, ledger, unresolved


def export_memoirs(output):
    overrides = read_json("data/series_overrides.json")
    records = [read_json(p) for p in sorted((ROOT / "memoirs").glob("*.json"))]
    inventory = read_json(ROOT / "ams-inventory.json")
    if {r["number"] for r in records} != set(inventory["candidate_numbers"]):
        raise ValueError("Incomplete Memoirs matching inventory")
    rows, ledger, unresolved = memoirs_rows(records, overrides["memoirs"], overrides["memoirs_window"])
    excluded = [{"number": r["number"], "review": overrides["memoirs"][str(r["number"])]}
                for r in ledger if r["status"] == "excluded_after_review"]
    title_excluded = [r["number"] for r in ledger if r["status"] == "excluded_unsuitable_title"]
    directory = Path(output) / "publishers/memoirs-of-the-american-mathematical-society"
    write_jsonl(directory / "eligible.jsonl", rows)
    write_json(directory / "reconciliation.json", {"ledger": ledger, "unresolved": unresolved, "excluded": excluded,
                                                  "title_excluded_numbers": title_excluded})
    write_json(directory / "audit.json", {
        "journal_name": "Memoirs of the American Mathematical Society",
        "status": "needs_series_review" if unresolved else "complete", "scan_complete": True,
        "source_kind": "publisher_inventory_and_reviewed_bibliographic_years",
        "sources": [AMS_INDEX, MATCHER, "data/series_overrides.json", "reports/memoirs-year-evidence.csv"],
        "errors": [], "inventory_candidates": len(records), "confirmed_works": len(rows),
        "unresolved": unresolved, "excluded": excluded, "title_excluded_numbers": title_excluded, "window": overrides["memoirs_window"],
        "method": "One work per separately numbered memoir; printed volume year where verified, otherwise matched final bibliographic year. Never arXiv submission or copyright year.",
        "year_corrections": {n: r for n, r in overrides["memoirs"].items() if r["source_kind"] == "printed_volume_date"},
        "finished_at": time.time()})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(ledger[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(ledger)
    write_text("reports/memoirs-year-evidence.csv", buffer.getvalue())
    from .counts import aggregate
    aggregate(output, journals())
    print(f"Memoirs: {len(rows)} confirmed; {len(unresolved)} unresolved; {len(excluded)} excluded after review; {len(title_excluded)} excluded by title")


def bulk_matches(http, queries):
    """Documented read-only POST search; stop on rate limits instead of hammering."""
    payload = {"queries": [{**q, "a": [q["a"]] if isinstance(q.get("a"), str) else q.get("a", [])} for q in queries]}
    path = http.cache / ("bulk-" + digest(payload) + ".json")
    if path.exists():
        return read_json(path)["response"]["results"]
    if http.offline:
        raise RuntimeError("Bulk query not cached")
    request = urllib.request.Request(MATCHER, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "User-Agent": "journal-suggester/0.1 (local research experiment)"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200] if exc.code == 422 else ""
        raise RuntimeError(f"Bulk matcher HTTP {exc.code}; Retry-After={exc.headers.get('Retry-After', 'unspecified')}; {detail}") from None
    if len(result["results"]) != len(queries):
        raise ValueError("Bulk response length differs from query length")
    write_json(path, {"url": MATCHER, "method": "POST", "request": payload,
                      "response": result, "fetched_at": time.time()})
    return result["results"]


def refine_memoirs(http):
    pending = []
    for path in sorted((ROOT / "memoirs").glob("*.json")):
        row = read_json(path)
        if row["status"] != "matched" and row.get("title") and row.get("authors"):
            pending.append((path, row, {"t": row["title"], "a": row["authors"][:1],
                                       "j": "Mem. Am. Math. Soc.", "v": str(row["number"]), "n": 10}))
    for start in range(0, len(pending), 50):
        batch = pending[start:start + 50]
        results = bulk_matches(http, [q for _, _, q in batch])
        accepted_count = 0
        for (path, row, query), matches in zip(batch, results):
            checked = check_memoir_matches(row["title"], row["authors"], row["number"], matches)
            write_json(ROOT / "memoirs-refinements" / path.name, {"number": row["number"], "query": query, "matches": checked})
            accepted = [m for m in checked if m["accepted"]]
            if len(accepted) == 1:
                row.update(matches=checked, status="matched", year=int(accepted[0]["year"]),
                           year_source="zbMATH bibliographic citation", refined_by_number=True)
                write_json(path, row)
                accepted_count += 1
        print(f"Refined Memoirs batch: {accepted_count}/{len(batch)} resolved", flush=True)
        if not http.offline and start + 50 < len(pending):
            time.sleep(10)


def collect_memoirs_crossref(http):
    """Retain all years to detect copyright/online dates crossing the window."""
    path = Path("data/processed/series-review/memoirs-all.json")
    items, cursors, cursor = [], set(), "*"
    while cursor not in cursors:
        cursors.add(cursor)
        result = http.json("https://api.crossref.org/journals/0065-9266/works",
                           {"rows": 1000, "cursor": cursor})["message"]
        items.extend(result["items"])
        if len(result["items"]) < 1000:
            if len(items) != result["total-results"]:
                raise ValueError("Crossref inventory total changed or pagination was incomplete")
            write_json(path, {"items": items, "total-results": result["total-results"]})
            return
        cursor = result.get("next-cursor")
        if not cursor:
            break
    raise ValueError("Crossref cursor ended prematurely")


def match_memoirs(http, shard=0, shards=1, bulk=False):
    index = ams_index(http.get(AMS_INDEX))
    crossref = read_json("data/processed/series-review/memoirs-all.json")
    if len(crossref["items"]) != crossref["total-results"]:
        raise ValueError("Incomplete Crossref enumeration")
    numbered = {int(m[1]): row for row in crossref["items"]
                if (m := re.fullmatch(r"10.1090/memo/(\d+)", row["DOI"], re.I))}
    # Deliberately broad 2014+ publisher inventory. Copyright years are NOT final years.
    candidates = {n: r for n, r in index.items() if r["volume"] >= 230 or r["copyright_year"] >= 2014}
    candidates.update({n: {"number": n, "volume": int(r.get("volume") or 0), "copyright_year": None,
                           "url": "https://doi.org/" + r["DOI"]}
                       for n, r in numbered.items() if int(r.get("volume") or 0) >= 230 and n not in candidates})
    write_json(ROOT / "ams-inventory.json", {"source": AMS_INDEX, "entries": list(index.values()),
                                              "candidate_numbers": sorted(candidates)})
    states, pending = Counter(), []
    for n, entry in sorted(candidates.items()):
        if n % shards != shard:
            continue
        path = ROOT / "memoirs" / f"{n}.json"
        if path.exists():
            previous = read_json(path)
            if not bulk:
                states[previous["status"]] += 1
                continue
            checked = check_memoir_matches(previous.get("title", ""), previous.get("authors", []), n, previous.get("matches", []))
            accepted = [m for m in checked if m["accepted"]]
            if len(accepted) == 1:
                previous.update(matches=checked, status="matched", year=int(accepted[0]["year"]),
                                year_source="zbMATH bibliographic citation")
                write_json(path, previous)
                states["matched"] += 1
                continue
        record = numbered.get(n)
        if record is None:
            write_json(path, {**entry, "status": "missing_crossref_title"})
            states["missing_crossref_title"] += 1
            continue
        title = clean_text(" ".join(record.get("title", [])))
        authors = [a.get("family", "") for a in record.get("author", []) if a.get("family")]
        params = {"t": title, "j": "Mem. Am. Math. Soc.", "n": 3}
        if authors:
            params["a"] = authors[0]
        result = {**entry, "doi": record["DOI"], "title": title, "authors": authors,
                  "crossref_dates": {k: record[k] for k in ("published-print", "published-online", "published") if k in record}}
        if not title or not authors:
            write_json(path, {**result, "status": "missing_crossref_title_or_authors"})
            states["missing_crossref_title_or_authors"] += 1
            continue
        if bulk:
            params["n"] = 5
            pending.append((path, result, params))
            continue
        try:
            matches = http.json(MATCHER, params)["results"]
            checked = check_memoir_matches(title, authors, n, matches)
            accepted = [m for m in checked if m["accepted"]]
            result.update(matches=checked, status="matched" if len(accepted) == 1 else "needs_review")
            if len(accepted) == 1:
                result.update(year=int(accepted[0]["year"]), year_source="zbMATH bibliographic citation")
        except (RuntimeError, ValueError, KeyError) as exc:
            result.update(status="source_error", error=str(exc))
        write_json(path, result)
        states[result["status"]] += 1
        if sum(states.values()) % 25 == 0:
            print(f"Memoirs {sum(states.values())}/{len(candidates)}: {dict(states)}", flush=True)
    for offset in range(0, len(pending), 50):
        batch = pending[offset:offset + 50]
        returned = bulk_matches(http, [params for _, _, params in batch])
        for (path, result, _), matches in zip(batch, returned):
            checked = check_memoir_matches(result["title"], result["authors"], result["number"], matches)
            accepted = [m for m in checked if m["accepted"]]
            result.update(matches=checked, status="matched" if len(accepted) == 1 else "needs_review")
            if len(accepted) == 1:
                result.update(year=int(accepted[0]["year"]), year_source="zbMATH bibliographic citation")
            write_json(path, result)
            states[result["status"]] += 1
        print(f"Memoirs bulk {min(offset + 50, len(pending))}/{len(pending)}: {dict(states)}", flush=True)
        if not http.offline and offset + 50 < len(pending):
            time.sleep(10)
    print("Memoirs matching complete:", dict(states), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smf", "smf-books", "memoirs-crossref", "memoirs", "refine-memoirs", "export-asterisque", "export"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument("--output", default="data/processed/publication-counts-20261006-filtered")
    parser.add_argument("--reuse-count-scans", help="Copy compatible scans into a new output directory")
    args = parser.parse_args()
    http = CachedHTTP(cache=CACHE, offline=args.offline, timeout=15, attempts=2)
    if not 0 <= args.shard < args.shards:
        parser.error("Require 0 <= shard < shards")
    if args.reuse_count_scans:
        if args.stage != "export":
            parser.error("Scan reuse is only supported for export")
        from .counts import reuse_scans
        reuse_scans(args.reuse_count_scans, args.output)
    if args.stage == "export":
        export_asterisque(args.output)
        export_memoirs(args.output)
    elif args.stage == "export-asterisque":
        export_asterisque(args.output)
    elif args.stage == "refine-memoirs":
        refine_memoirs(http)
    elif args.stage == "memoirs":
        match_memoirs(http, args.shard, args.shards, args.bulk)
    elif args.stage == "memoirs-crossref":
        collect_memoirs_crossref(http)
    else:
        (collect_smf if args.stage == "smf" else collect_smf_books)(http)


if __name__ == "__main__":
    main()
