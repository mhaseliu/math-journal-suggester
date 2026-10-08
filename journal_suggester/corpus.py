"""Seeded abstract collection from the audited publication ledger; resumable by journal."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from .collect import UNSUITABLE_TITLE, inverted_abstract
from .http import CachedHTTP
from .io import digest, journals, read_json, read_jsonl, write_json, write_jsonl
from .records import arxiv_key, clean_text, doi_key, identities, title_key
from .splits import floor_quotas


def same_title(a, b):
    return title_key(clean_text(a)) == title_key(clean_text(b))


def crossref_match(paper, work):
    return doi_key(work.get("DOI")) == paper["doi"] and same_title(" ".join(work.get("title", [])), paper["title"])


def openalex_match(paper, work):
    return doi_key(work.get("doi")) == paper["doi"] and same_title(work.get("title") or "", paper["title"])


def ordered_candidates(papers, seed, jid, excluded):
    rows = [dict(p) for p in papers if not set(identities(p)) & excluded and not UNSUITABLE_TITLE.match(p["title"])]
    rows.sort(key=lambda p: p["paper_id"])
    random.Random(f"collection:{seed}:{jid}").shuffle(rows)
    return rows


def crossref_abstracts(journal, http):
    """Bulk deposits, later joined to the reviewed ledger; dates/labels come from that ledger."""
    records = {}
    # Most paired ISSNs expose the same Crossref source. Union both for complete coverage.
    for issn in journal["issns"].split(";"):
        if not issn:
            continue
        cursor, seen = "*", set()
        while cursor not in seen:
            seen.add(cursor)
            page = http.json(f"https://api.crossref.org/journals/{issn}/works", {
                "filter": "from-pub-date:2015-01-01,until-pub-date:2026-12-31,has-abstract:true",
                "select": "DOI,title,abstract,license", "rows": 1000, "cursor": cursor})["message"]
            for row in page["items"]:
                records[doi_key(row.get("DOI"))] = row
            if len(page["items"]) < 1000:
                break
            cursor = page.get("next-cursor")
            if not cursor:
                break
    return records


def publisher_abstract(paper, body):
    if paper["journal_id"] == "new-york-journal-of-mathematics":
        title = re.search(r'<p[^>]*class=["\']style12["\'][^>]*>(.*?)</p>', body, re.S | re.I)
        abstract = re.search(r'\bAbstract\s*<blockquote[^>]*>(.*?)</blockquote>', body, re.S | re.I)
        if title and abstract and same_title(title[1], paper["title"]):
            return clean_text(abstract[1])
    elif paper["journal_id"] == "asterisque":
        # The audited work URL and displayed DOI must both match, never a neighbouring TOC item.
        displayed = re.findall(r'<strong>DOI\s*:</strong>\s*([^<]+)', body, re.I)
        if paper["doi"] and paper["doi"] not in [doi_key(v) for v in displayed]:
            return ""
        abstract = re.search(r'<div class="product__data-resume data-english[^"\n]*">(.*?)</div>', body, re.S)
        if not abstract:
            abstract = re.search(r'<article class="article-body edito">(.*?)</article>', body, re.S)
        if abstract:
            return clean_text(abstract[1])
    return ""


def enrich_batch(papers, deposits, http, publisher_http):
    provenance = {p["paper_id"]: {"paper_id": p["paper_id"], "errors": []} for p in papers}
    ambiguous = []
    for p in papers:
        row = deposits.get(p["doi"])
        if row and crossref_match(p, row):
            p["abstract"] = clean_text(row.get("abstract", ""))
            provenance[p["paper_id"]].update(abstract_source="crossref", licenses=row.get("license", []))
        elif row:
            ambiguous.append({"paper_id": p["paper_id"], "source": "crossref", "reason": "identity_mismatch"})
    # Bulk DOI requests keep the same conservative identity check as the pilot.
    pending = [p for p in papers if len(p["abstract"].split()) < 15 and p["doi"]]
    if pending:
        works = http.json("https://api.openalex.org/works", {
            "filter": "doi:" + "|".join(p["doi"] for p in pending), "per-page": 100,
            "select": "id,doi,title,abstract_inverted_index,locations,primary_location"})["results"]
        by_doi = defaultdict(list)
        for work in works:
            by_doi[doi_key(work.get("doi"))].append(work)
        for p in pending:
            matches = [w for w in by_doi[p["doi"]] if openalex_match(p, w)]
            if len(matches) != 1:
                if by_doi[p["doi"]]:
                    ambiguous.append({"paper_id": p["paper_id"], "source": "openalex", "reason": "identity_mismatch_or_ambiguity"})
                continue
            work = matches[0]
            prov = provenance[p["paper_id"]]
            prov.update(openalex_id=work["id"], openalex_license=(work.get("primary_location") or {}).get("license"))
            for location in work.get("locations", []):
                landing = location.get("landing_page_url") or ""
                if re.match(r"https?://arxiv.org/abs/", landing):
                    p["arxiv_id"] = arxiv_key(landing)
            try:
                abstract = inverted_abstract(work.get("abstract_inverted_index"))
                if len(abstract.split()) >= 15:
                    p["abstract"] = abstract
                    prov["abstract_source"] = "openalex"
            except ValueError as e:
                prov["errors"].append(str(e))
    # Series and NYJM retain publisher-audited dates, even when abstracts come from elsewhere.
    for p in papers:
        if len(p["abstract"].split()) >= 15 or p["journal_id"] not in {"asterisque", "new-york-journal-of-mathematics"}:
            continue
        try:
            abstract = publisher_abstract(p, publisher_http.get(p["url"]))
            if len(abstract.split()) >= 15:
                p["abstract"] = abstract
                provenance[p["paper_id"]].update(abstract_source="publisher", source_url=p["url"])
        except RuntimeError as e:
            provenance[p["paper_id"]]["errors"].append(str(e))
    pending = [p for p in papers if len(p["abstract"].split()) < 15 and p["doi"]]
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    for start in range(0, len(pending), 60):
        batch = pending[start:start + 60]
        query = " OR ".join(f'doi:"{p["doi"]}"' for p in batch)
        try:
            root = ET.fromstring(http.get("https://export.arxiv.org/api/query", {"search_query": query, "max_results": 200}))
            entries = root.findall("a:entry", ns)
            for p in batch:
                matches = [e for e in entries if doi_key(e.findtext("x:doi", "", ns)) == p["doi"]
                           and same_title(e.findtext("a:title", "", ns), p["title"])]
                if len(matches) == 1:
                    e = matches[0]
                    p["abstract"] = clean_text(e.findtext("a:summary", "", ns))
                    p["arxiv_id"] = arxiv_key(e.findtext("a:id", "", ns))
                    provenance[p["paper_id"]]["abstract_source"] = "arxiv"
                elif len(matches) > 1:
                    ambiguous.append({"paper_id": p["paper_id"], "source": "arxiv", "reason": "multiple_matches"})
        except (RuntimeError, ET.ParseError) as e:
            for p in batch:
                provenance[p["paper_id"]]["errors"].append(str(e))
    return list(provenance.values()), ambiguous


def collect(config_path, output, offline=False):
    config, output = read_json(config_path), Path(output)
    ledger = read_jsonl(config["ledger"])
    counts = read_json(config["publication_counts"])
    previous = read_jsonl(config["exclude_previous_pilot"]) if config.get("exclude_previous_pilot") else []
    excluded = {key for p in previous for key in identities(p)}
    specification = {"config": config, "ledger_hash": digest(ledger), "counts": counts, "excluded": sorted(excluded), "method": 1}
    plan_path = output / "plan.json"
    if plan_path.exists() and digest(read_json(plan_path)) != digest(specification):
        raise ValueError("Collection specification changed; use a fresh directory")
    write_json(plan_path, specification)
    groups = defaultdict(list)
    for p in ledger:
        groups[p["journal_id"]].append(p)
    val = floor_quotas(config["validation_queries"], counts, config["minimum_per_journal"]["validation"])
    test = floor_quotas(config["test_queries"], counts, config["minimum_per_journal"]["test"])
    refs = floor_quotas(config["reference_target"], counts, config["minimum_per_journal"]["reference"])
    targets = {k: val[k] + test[k] + refs[k] + config["collection_spares_per_journal"] for k in counts}
    http = CachedHTTP("data/raw/evaluation-http", offline=offline, timeout=35, attempts=3)
    publisher_http = CachedHTTP("data/raw/series-reconciliation-http", offline=offline, timeout=25, attempts=2)
    def collect_journal(journal):
        jid = journal["journal_id"]
        saved = output / "journals" / (jid + ".json")
        if saved.exists():
            result = read_json(saved)
        else:
            candidates = ordered_candidates(groups[jid], config["seed"], jid, excluded)
            print(f"{jid}: target {targets[jid]}; eligible candidates {len(candidates)}", flush=True)
            try:
                deposits = crossref_abstracts(journal, http)
                errors = []
            except RuntimeError as e:
                deposits, errors = {}, [str(e)]
            accepted, provs, bad, scanned = [], [], [], 0
            for start in range(0, len(candidates), 100):
                batch = candidates[start:start + 100]
                prov, amb = enrich_batch(batch, deposits, http, publisher_http)
                bad.extend(amb)
                for paper, p in zip(batch, prov):
                    scanned += 1
                    provs.append(p)
                    # Remove aliases of the previous pilot discovered during enrichment too.
                    if len(paper["abstract"].split()) >= 15 and not set(identities(paper)) & excluded:
                        accepted.append(paper)
                    if len(accepted) >= targets[jid]:
                        break
                print(f"{jid}: {len(accepted)}/{targets[jid]} usable; scanned {scanned}", flush=True)
                if len(accepted) >= targets[jid]:
                    break
            result = {"papers": accepted, "provenance": provs, "quarantine": bad,
                      "coverage": {"target": targets[jid], "usable": len(accepted), "eligible": len(candidates),
                                   "scanned": scanned, "shortfall": max(0, targets[jid] - len(accepted)),
                                   "crossref_errors": errors, "years": dict(Counter(p["year"] for p in accepted))}}
            write_json(saved, result)
        return jid, result

    completed = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(collect_journal, j) for j in sorted(journals(), key=lambda j: j["journal_id"])]
        for future in as_completed(futures):
            jid, result = future.result()
            completed[jid] = result
            papers, provenance, quarantine, coverage = [], [], [], {}
            for k, result in sorted(completed.items()):
                papers.extend(result["papers"])
                provenance.extend(result["provenance"])
                quarantine.extend(result["quarantine"])
                coverage[k] = result["coverage"]
            write_jsonl(output / "papers.jsonl", papers)
            write_jsonl(output / "provenance.jsonl", provenance)
            write_jsonl(output / "quarantine.jsonl", quarantine)
            write_json(output / "coverage.json", coverage)
    print(json.dumps({"papers": len(papers), "journals": len(coverage), "shortfalls": {k: v["shortfall"] for k, v in coverage.items() if v["shortfall"]}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.json")
    parser.add_argument("--output", default="data/processed/corpus")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    collect(args.config, args.output, args.offline)
