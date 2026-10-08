"""Crossref ISSN collection, DOI-checked OpenAlex enrichment, arXiv fallback."""
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .http import CachedHTTP
from .io import digest, journals, write_json, write_jsonl
from .records import arxiv_key, clean_text, doi_key, title_key


UNSUITABLE_TITLE = re.compile(
    r"^(?:(?:publisher|author)\s+)?(?:correction|corrigendum|corrigenda|erratum|errata|retraction|retracted|editorial|guest editorial|foreword|preface|"
    r"front[\s-]*matter|back[\s-]*matter|contents|table of contents|index to|author index|subject index|"
    r"list of referees|special issue|masthead|information for authors|issue information)\b|"
    r"^.*\b(?:front[\s-]*matter|back[\s-]*matter|table of contents|corrigendum|erratum|obituary)\b|"
    r"^.*\b(?:corrigenda|errata)\s*(?:to\b|$)", re.I)


def publication_year(item):
    # Issue/print year is used consistently when available, then online, then issued.
    for field in ("published-print", "published-online", "published", "issued"):
        parts = item.get(field, {}).get("date-parts", [])
        if parts and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0], field
    return None, None


def inverted_abstract(index):
    if not index:
        return ""
    positions = {}
    for word, slots in index.items():
        for pos in slots:
            if not isinstance(pos, int) or not 0 <= pos < 100000 or pos in positions:
                raise ValueError("Invalid abstract inverted index")
            positions[pos] = word
    if sorted(positions) != list(range(len(positions))):
        raise ValueError("Non-contiguous abstract inverted index")
    return clean_text(" ".join(positions[i] for i in range(len(positions))))


def crossref_record(item, journal):
    if item.get("type") != "journal-article":
        return None, "not_individual_journal_article"
    if not set(item.get("ISSN", [])) & set(journal["issns"].split(";")):
        return None, "issn_mismatch"
    title = clean_text(" ".join(item.get("title", [])))
    if not title or UNSUITABLE_TITLE.match(title):
        return None, "unsuitable_title"
    # Series with mixed volume/work metadata need explicit auditing, not automatic guessing.
    if journal["journal_id"] in {"asterisque", "memoirs-of-the-american-mathematical-society"}:
        return None, "series_requires_work_level_review"
    year, year_source = publication_year(item)
    if year is None or not 2016 <= year <= 2025:
        return None, "outside_publication_window"
    doi = doi_key(item.get("DOI"))
    if not doi:
        return None, "missing_crossref_doi"
    return {"paper_id": "doi:" + doi, "title": title, "abstract": clean_text(item.get("abstract", "")),
            "journal_id": journal["journal_id"], "year": year, "doi": doi, "arxiv_id": "",
            "url": "https://doi.org/" + doi}, year_source


def enrich(paper, http, arxiv=True):
    provenance, ambiguous = {}, []
    if paper["abstract"]:
        return {"abstract_source": "crossref"}, ambiguous
    try:
        work = http.json("https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + paper["doi"], safe=""))
        if doi_key(work.get("doi")) != paper["doi"] or title_key(clean_text(work.get("title", ""))) != title_key(paper["title"]):
            ambiguous.append({"paper_id": paper["paper_id"], "source": "openalex", "reason": "identity_mismatch", "match_id": work.get("id")})
        else:
            paper["abstract"] = inverted_abstract(work.get("abstract_inverted_index"))
            provenance.update(openalex_id=work.get("id"), openalex_license=work.get("primary_location", {}).get("license"))
            for location in work.get("locations", []):
                landing = location.get("landing_page_url") or ""
                if re.match(r"https?://arxiv.org/abs/", landing):
                    paper["arxiv_id"] = arxiv_key(landing)
            if paper["abstract"]:
                provenance["abstract_source"] = "openalex"
    except (RuntimeError, ValueError) as e:
        provenance["openalex_error"] = str(e)
    if not paper["abstract"] and arxiv:
        params = ({"id_list": paper["arxiv_id"]} if paper["arxiv_id"] else
                  {"search_query": "doi:" + paper["doi"], "max_results": 5})
        try:
            root = ET.fromstring(http.get("https://export.arxiv.org/api/query", params))
            ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
            matches = []
            for entry in root.findall("a:entry", ns):
                title = clean_text(entry.findtext("a:title", "", ns))
                doi = doi_key(entry.findtext("x:doi", "", ns))
                if title_key(title) == title_key(paper["title"]) and (doi == paper["doi"] or (not doi and paper["arxiv_id"])):
                    matches.append(entry)
                else:
                    ambiguous.append({"paper_id": paper["paper_id"], "source": "arxiv", "reason": "identity_mismatch", "title": title})
            if len(matches) == 1:
                entry = matches[0]
                paper["abstract"] = clean_text(entry.findtext("a:summary", "", ns))
                paper["arxiv_id"] = arxiv_key(entry.findtext("a:id", "", ns))
                provenance["abstract_source"] = "arxiv"
            elif len(matches) > 1:
                ambiguous.append({"paper_id": paper["paper_id"], "source": "arxiv", "reason": "multiple_matches"})
        except (RuntimeError, ET.ParseError) as e:
            provenance["arxiv_error"] = str(e)
    return provenance, ambiguous


def collect(output, journal_ids, per_journal=20, max_records=200, offline=False, fallback=True):
    output = Path(output)
    catalog = journals()
    unknown = set(journal_ids) - {j["journal_id"] for j in catalog}
    if unknown:
        raise ValueError(f"Unknown journals: {sorted(unknown)}")
    http = CachedHTTP(offline=offline)
    all_papers, provenance, quarantine, coverage = [], [], [], {}
    for journal in catalog:
        jid = journal["journal_id"]
        if jid not in journal_ids:
            continue
        if not journal["issns"]:
            raise ValueError(f"Unverified ISSN: {jid}; run catalog first")
        seen, papers, records, reasons, totals = set(), [], 0, Counter(), {}
        # Query all ISSNs, with DOI deduplication. Stop early only for an explicitly capped pilot.
        for issn in journal["issns"].split(";"):
            cursor, cursors = "*", set()
            while cursor not in cursors:
                cursors.add(cursor)
                try:
                    result = http.json(f"https://api.crossref.org/journals/{issn}/works", {
                        "filter": "from-pub-date:2016-01-01,until-pub-date:2025-12-31,type:journal-article",
                        "rows": 100, "cursor": cursor})["message"]
                except RuntimeError as e:
                    reasons["crossref_endpoint_unavailable"] += 1
                    quarantine.append({"journal_id": jid, "issn": issn, "reason": "crossref_endpoint_unavailable", "error": str(e)})
                    print(f"{jid}: {e}", flush=True)
                    break
                totals[issn] = result["total-results"]
                for item in result["items"]:
                    doi = doi_key(item.get("DOI"))
                    if doi in seen:
                        continue
                    seen.add(doi)
                    records += 1
                    if records % 10 == 0:
                        print(f"{jid}: scanning record {records}; {len(papers)} usable so far", flush=True)
                    paper, reason = crossref_record(item, journal)
                    if not paper:
                        reasons[reason] += 1
                    else:
                        prov = {"paper_id": paper["paper_id"], "publication_year_source": reason,
                                "crossref_issn": issn, "container_title": item.get("container-title"),
                                "licenses": item.get("license", []), "source": "crossref"}
                        if fallback:
                            extra, ambiguous = enrich(paper, http)
                            prov.update(extra)
                            quarantine.extend(ambiguous)
                        elif paper["abstract"]:
                            prov["abstract_source"] = "crossref"
                        provenance.append(prov)
                        if len(paper["abstract"].split()) < 15:
                            reasons["missing_or_short_abstract"] += 1
                        else:
                            papers.append(paper)
                    if (per_journal and len(papers) >= per_journal) or (max_records and records >= max_records):
                        break
                if (per_journal and len(papers) >= per_journal) or (max_records and records >= max_records) or not result["items"]:
                    break
                cursor = result.get("next-cursor")
                if not cursor:
                    break
            if (per_journal and len(papers) >= per_journal) or (max_records and records >= max_records):
                break
        all_papers.extend(papers)
        coverage[jid] = {"indexed_by_issn": totals, "scanned_unique_dois": records, "usable": len(papers),
                         "eligible_scanned": records - sum(n for reason, n in reasons.items() if reason not in {"missing_or_short_abstract", "crossref_endpoint_unavailable"}),
                         "rejected": dict(reasons), "capped_pilot": bool(per_journal or max_records),
                         "count_note": "ISSN endpoint counts overlap; not a deduplicated publication census"}
        write_jsonl(output / "papers.jsonl", all_papers)
        write_jsonl(output / "provenance.jsonl", provenance)
        write_jsonl(output / "quarantine.jsonl", quarantine)
        write_json(output / "coverage.json", coverage)
        print(f"{jid}: {len(papers)} usable / {records} scanned; {dict(reasons)}", flush=True)
    return coverage
