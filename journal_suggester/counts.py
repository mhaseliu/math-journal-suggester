"""Resumable metadata-only publication counts, independent of abstract availability."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .collect import UNSUITABLE_TITLE, crossref_record
from .http import CachedHTTP
from .io import digest, journals, read_json, read_jsonl, write_json, write_jsonl
from .records import GENERIC_TITLES, deduplicate, doi_key


DATE_FILTERS = ("pub-date", "print-pub-date", "online-pub-date")
FIELDS = "DOI,title,ISSN,type,published-print,published-online,published,issued"
SERIES = {"asterisque", "memoirs-of-the-american-mathematical-society"}
METHOD = {
    "version": 4, "years": [2016, 2025], "date_filter_union": DATE_FILTERS,
    "unsuitable_title_pattern": UNSUITABLE_TITLE.pattern,
    "year_precedence": ["published-print", "published-online", "published", "issued"],
    "deduplication": "DOI and exact normalized title; cross-journal conflicts quarantined",
    "title_only_identity_exceptions": sorted(GENERIC_TITLES),
    "abstract_requirement": False, "fields": FIELDS, "rows": 1000,
}


def scan_journal(journal, http):
    records, sources, errors = {}, [], []
    for issn in journal["issns"].split(";"):
        for date_filter in DATE_FILTERS:
            cursor, cursors, retrieved = "*", set(), 0
            source = {"issn": issn, "date_filter": date_filter, "complete": False, "pages": 0}
            sources.append(source)
            # These are separate requests (union), not three date constraints ANDed together.
            while cursor not in cursors:
                cursors.add(cursor)
                try:
                    msg = http.json(f"https://api.crossref.org/journals/{issn}/works", {
                        "filter": f"from-{date_filter}:2016-01-01,until-{date_filter}:2025-12-31",
                        "rows": METHOD["rows"], "select": FIELDS, "cursor": cursor})["message"]
                except (RuntimeError, ValueError, KeyError) as exc:
                    errors.append({"issn": issn, "date_filter": date_filter, "error": str(exc)})
                    break
                source.setdefault("reported_total", msg["total-results"])
                items = msg["items"]
                source["pages"] += 1
                retrieved += len(items)
                for item in items:
                    key = doi_key(item.get("DOI")) or "missing-doi:" + digest(item)
                    # Store identical occurrences once; retain metadata variants for validation.
                    records.setdefault(key, {})[digest(item)] = item
                if len(items) < METHOD["rows"]:
                    source["complete"] = retrieved >= source["reported_total"]
                    if not source["complete"]:
                        errors.append({"issn": issn, "date_filter": date_filter, "error": "Fewer records than reported total"})
                    break
                cursor = msg.get("next-cursor")
                if not cursor or cursor in cursors:
                    errors.append({"issn": issn, "date_filter": date_filter, "error": "Pagination ended prematurely"})
                    break
            source["retrieved"] = retrieved
            print(f'{journal["journal_id"]}: {issn} {date_filter}, {retrieved} records, complete={source["complete"]}', flush=True)
            # An absent journal endpoint cannot serve the other date filters either.
            if errors and errors[-1].get("issn") == issn and "HTTP 404:" in errors[-1]["error"]:
                break
    eligible, rejected = [], Counter()
    for variants in records.values():
        parsed = [crossref_record(item, journal) for item in variants.values()]
        good = [paper for paper, _ in parsed if paper]
        signatures = {(p["title"], p["year"], p["journal_id"]) for p in good}
        if len(signatures) > 1 or (good and len(good) != len(parsed)):
            rejected["conflicting_metadata_versions"] += 1
        elif good:
            eligible.append(good[0])
        else:
            rejected[parsed[0][1]] += 1
    complete = bool(sources) and all(s["complete"] for s in sources)
    status = "needs_series_review" if journal["journal_id"] in SERIES else "complete" if complete else "incomplete"
    return eligible, {
        "journal_name": journal["journal_name"], "issns": journal["issns"], "status": status,
        "scan_complete": complete, "sources": sources, "errors": errors,
        "unique_record_keys": len(records), "eligible_dois_before_title_dedup": len(eligible),
        "rejected": dict(rejected), "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def aggregate(output, catalog):
    output = Path(output)
    audits, rows = {}, []
    for journal in catalog:
        jid = journal["journal_id"]
        path = output / "journals" / jid / "audit.json"
        publisher = output / "publishers" / jid / "audit.json"
        if publisher.exists() and (read_json(publisher)["status"] == "complete" or jid in SERIES):
            path = publisher
        if path.exists():
            audits[jid] = read_json(path)
            rows.extend(read_jsonl(path.with_name("eligible.jsonl")))
        else:
            audits[jid] = {"journal_name": journal["journal_name"], "status": "pending"}
    unique, conflicts = deduplicate(rows)
    by_journal, per_year = Counter(), defaultdict(Counter)
    for row in unique:
        by_journal[row["journal_id"]] += 1
        per_year[row["journal_id"]][str(row["year"])] += 1
    affected = Counter()
    for conflict in conflicts:
        for jid in {r["journal_id"] for r in conflict["records"]}:
            affected[jid] += 1
    counts = {}
    for jid, audit in audits.items():
        audit["observed_eligible_groups"] = by_journal[jid]
        audit["conflicting_journal_groups"] = affected[jid]
        audit["by_year"] = {str(y): per_year[jid][str(y)] for y in range(2016, 2026)}
        # Incomplete and unsupported coverage is null, never an invented zero.
        counts[jid] = by_journal[jid] if audit["status"] == "complete" else None
    result = {"method": METHOD, "generated_at": datetime.now(timezone.utc).isoformat(),
              "catalog_size": len(catalog), "status_counts": dict(Counter(a["status"] for a in audits.values())),
              "note": "Indexed eligible works under project rules; not a complete publisher census. Counts may change while the audit is running.",
              "journals": audits}
    write_json(output / "counts.json", counts)
    write_json(output / "audit.json", result)
    write_jsonl(output / "eligible.jsonl", unique)
    write_jsonl(output / "quarantine.jsonl", conflicts)
    return result


def reuse_scans(source, output):
    """Reuse metadata scans only when the change is confined to aggregation."""
    source, output = Path(source), Path(output)
    previous = read_json(source / "manifest.json")
    ignored = {"version", "deduplication", "title_only_identity_exceptions"}
    scan_method = lambda m: {k: v for k, v in m.items() if k not in ignored}
    if digest(scan_method(previous["method"])) != digest(scan_method(METHOD)):
        raise ValueError("Scan eligibility changed; cannot reuse scans")
    if previous["fingerprint"] != digest({"method": previous["method"], "catalog": journals()}):
        raise ValueError("Catalog changed; cannot reuse scans")
    if output.exists():
        raise ValueError("Use a new output directory for scan reuse")
    output.mkdir(parents=True)
    for name in ("journals", "publishers"):
        if (source / name).exists():
            shutil.copytree(source / name, output / name)
    write_json(output / "manifest.json", {
        "method": METHOD, "fingerprint": digest({"method": METHOD, "catalog": journals()}),
        "scans_reused_from": str(source), "previous_fingerprint": previous["fingerprint"]})


def refine_titles(source, output):
    """Apply a stricter title exclusion to saved eligible rows without repeating HTTP scans."""
    source, output = Path(source), Path(output)
    previous = read_json(source / "manifest.json")
    allowed_changes = {"version", "unsuitable_title_pattern"}
    stable = lambda m: {k: v for k, v in m.items() if k not in allowed_changes}
    if digest(stable(previous["method"])) != digest(stable(METHOD)):
        raise ValueError("Only stricter title exclusions can be refined from these saved rows")
    if previous["fingerprint"] != digest({"method": previous["method"], "catalog": journals()}):
        raise ValueError("Catalog changed; cannot refine saved rows")
    if output.exists():
        raise ValueError("Use a new count refinement directory")
    output.mkdir(parents=True)
    removed = []
    for kind in ("journals", "publishers"):
        if not (source / kind).exists():
            continue
        shutil.copytree(source / kind, output / kind)
        for path in (output / kind).glob("*/eligible.jsonl"):
            rows = read_jsonl(path)
            rejected = [r for r in rows if UNSUITABLE_TITLE.match(r["title"])]
            if not rejected:
                continue
            kept = [r for r in rows if not UNSUITABLE_TITLE.match(r["title"])]
            audit = read_json(path.with_name("audit.json"))
            audit.setdefault("rejected", {})["title_rule_refinement"] = len(rejected)
            if "eligible_dois_before_title_dedup" in audit:
                audit["eligible_dois_before_title_dedup"] = len(kept)
            if "eligible_works" in audit:
                audit["eligible_works"] = len(kept)
            removed.extend({**r, "ledger_kind": kind, "reason": "title_rule_refinement"} for r in rejected)
            write_jsonl(path, kept)
            write_json(path.with_name("audit.json"), audit)
    write_json(output / "manifest.json", {
        "method": METHOD, "fingerprint": digest({"method": METHOD, "catalog": journals()}),
        "title_refinement_from": str(source), "previous_fingerprint": previous["fingerprint"]})
    write_jsonl(output / "title-exclusions.jsonl", removed)
    return aggregate(output, journals())


def audit_counts(output="data/processed/publication-counts", offline=False, only=None, retry_incomplete=False):
    output = Path(output)
    catalog = journals()
    fingerprint = digest({"method": METHOD, "catalog": catalog})
    manifest = output / "manifest.json"
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Count method/catalog changed; use a new output directory")
    if not manifest.exists():
        write_json(manifest, {"method": METHOD, "fingerprint": fingerprint})
    if only and set(only) - {j["journal_id"] for j in catalog}:
        raise ValueError("Unknown journal IDs")
    http = CachedHTTP(cache="data/raw/counts-http", offline=offline)
    aggregate(output, catalog)
    for journal in catalog:
        jid = journal["journal_id"]
        if only and jid not in only:
            continue
        publisher = output / "publishers" / jid / "audit.json"
        if publisher.exists() and read_json(publisher)["status"] == "complete":
            continue
        directory = output / "journals" / jid
        done = directory / "audit.json"
        if done.exists() and (not retry_incomplete or read_json(done)["status"] != "incomplete"):
            continue
        eligible, audit = scan_journal(journal, http)
        write_jsonl(directory / "eligible.jsonl", eligible)
        write_json(done, audit)
        report = aggregate(output, catalog)
        print(f'Completed {jid}: {audit["status"]}, {report["status_counts"]}', flush=True)
    return aggregate(output, catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/publication-counts")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--journals", nargs="+")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--refine-titles-from", help="Apply stricter title exclusions to a previous completed audit")
    args = parser.parse_args()
    if args.refine_titles_from:
        refine_titles(args.refine_titles_from, args.output)
    else:
        audit_counts(args.output, args.offline, args.journals, args.retry_incomplete)


if __name__ == "__main__":
    main()
