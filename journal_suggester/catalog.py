"""Extract all 95 stable IDs and verify exact journal identities with Crossref."""
import csv
import io
import re
import unicodedata
from pathlib import Path

from .http import CachedHTTP
from .io import journals, read_json, write_json, write_text


def name_key(name):
    name = unicodedata.normalize("NFKD", name).casefold().replace("&", "and")
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^the\s+", "", name))


def valid_issn(value):
    if not re.fullmatch(r"\d{4}-\d{3}[\dX]", value):
        return False
    digits = value.replace("-", "")
    return sum((10 if c == "X" else int(c)) * (8 - i) for i, c in enumerate(digits)) % 11 == 0


def extract(handoff=None, output="data/journals.csv"):
    from .resources import ROOT
    rows = journals(ROOT / "data/journals.csv")
    if not Path(output).exists():
        write_text(output, (ROOT / "data/journals.csv").read_text())
    return rows


def verify(path="data/journals.csv", only=None, offline=False):
    rows = journals(path)
    http = CachedHTTP(offline=offline)
    report_path = Path("data/catalog_provenance.json")
    report = __import__("json").loads(report_path.read_text()) if report_path.exists() else {}
    reviewed = read_json("configs/catalog_reviewed.json")
    for row in rows:
        jid = row["journal_id"]
        if only and jid not in only:
            continue
        if row["issns"] and jid in report and report[jid].get("verified") and jid not in reviewed:
            continue
        query = row["journal_name"].replace("&", "and")
        try:
            override = reviewed.get(jid)
            if override and "seed" in override:
                result = {"items": [http.json("https://api.crossref.org/journals/" + override["seed"])["message"]]}
                expected = override["title"]
            elif override:
                result = {"items": [{"title": row["journal_name"], "ISSN": override["issns"]}]}
                expected = row["journal_name"]
            else:
                result = http.json("https://api.crossref.org/journals", {"query": query, "rows": 100})["message"]
                expected = row["journal_name"]
            matches = [x for x in result["items"] if name_key(x["title"]) == name_key(expected)]
            if len(matches) > 1 and not set.intersection(*(set(x.get("ISSN", [])) for x in matches)):
                raise RuntimeError("Ambiguous same-title journals; supply a reviewed ISSN")
            ids = sorted({i for x in matches for i in x.get("ISSN", []) if valid_issn(i)})
            row["issns"] = ";".join(ids)
            report[jid] = {"verified": bool(ids), "source": "https://api.crossref.org/journals",
                           "query": query, "matches": [{"title": x["title"], "ISSN": x.get("ISSN", [])} for x in matches],
                           "other_titles": [x["title"] for x in result["items"]] if not ids else []}
            if override:
                report[jid]["reviewed"] = override
                report[jid]["source"] = "https://api.crossref.org/journals/" + override["seed"] if "seed" in override else override["source"]
            print(f"{jid}: {row['issns'] or 'UNRESOLVED'}", flush=True)
        except RuntimeError as e:
            report[jid] = {"verified": False, "error": str(e)}
            print(f"{jid}: {e}", flush=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["journal_id", "journal_name", "issns"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        write_text(path, buf.getvalue())
        write_json(report_path, report)
    return report
