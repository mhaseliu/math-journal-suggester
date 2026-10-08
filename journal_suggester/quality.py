"""Conservative corpus hygiene; retain mathematical inequalities and citations."""
import html
import re
from copy import deepcopy

from .records import PlainText, identities

# These individually reviewed texts cannot be repaired from their stored abstracts.
QUARANTINE = {
    "doi:10.2140/agt.2025.25.4037": "PDF body was indexed as abstract",
    "doi:10.1088/1751-8121/ab91d7": "special-issue introduction without a research abstract",
    "doi:10.1112/plms.12338": "repository takedown notice was indexed as abstract",
    "doi:10.1090/proc/13028": "repository takedown notice was indexed as abstract",
}
KEEP_FROM = {
    "doi:10.1088/1751-8121/ab7f66": "In the first part of these lecture notes",
    "doi:10.2140/pjm.2016.284.1": "We give uniform, explicit, and simple face-pairing descriptions",
}
CUT_BEFORE = {
    "doi:10.1353/ajm.2023.0008": "The authors were partially supported",
    "doi:10.1088/1751-8121/ad05f0": "Contribution to the special issue",
}


def plain_math_text(value):
    # Only complete, recognised markup is HTML. A mathematical '<p<1' is text.
    value = html.unescape(value)
    tag = r"</?(?:(?:jats|mml|math):)?(?:p|br|title|abstract|inline-formula|disp-formula|tex-math|math|mrow|mi|mn|mo|msub|msup|msubsup|mfrac|msqrt|mtext|semantics|annotation|i|b|em|strong|sup|sub|span)(?:\s+[^<>]*?)?\s*/?>"
    protected = re.sub(r"<(?!(?:" + tag[1:] + r"))", "&lt;", value, flags=re.I)
    parser = PlainText()
    parser.feed(protected)
    return " ".join(html.unescape("".join(parser.parts)).split())


def clean_paper(paper):
    original = deepcopy(paper)
    pid, abstract, title = paper["paper_id"], paper["abstract"], paper["title"]
    reason = QUARANTINE.get(pid)
    if "\ufffd" in title or re.search(r"\b(?:for|of|in)\s*<\s*$", title):
        reason = "corrupted or incomplete source title"
    if re.search(r"\bPaper Award\b", title, re.I):
        reason = "award notice"
    if abstract.startswith("This special issue"):
        reason = "special-issue introduction"
    if paper["journal_id"] == "bulletin-of-the-australian-mathematical-society" and re.search(r"\bthis (?:thesis|dissertation)\b", abstract, re.I):
        reason = "PhD-thesis summary, not a research article"
    if "urn:x-wiley:" in abstract:
        reason = "unresolved equation-image identifiers in abstract"
    if "\ufffd" in abstract and pid != "doi:10.4310/mrl.2018.v25.n3.a8":
        reason = "unrecoverable replacement characters in abstract"
    if reason:
        return None, {"paper_id": pid, "reason": reason}
    if pid in KEEP_FROM:
        assert KEEP_FROM[pid] in abstract
        abstract = abstract[abstract.index(KEEP_FROM[pid]):]
    if pid in CUT_BEFORE and CUT_BEFORE[pid] in abstract:
        abstract = abstract[:abstract.index(CUT_BEFORE[pid])]
    abstract = re.split(r"©|\(C\)\s*\d{4}|\bCopyright\b", abstract, maxsplit=1, flags=re.I)[0]
    patterns = [
        r"\bMSC(?:19\d\d|20\d\d)?\s*:",
        r"(?:\b20\d\d\s+)?\bMathematics subject classification\s*:",
        r"\bKeywords\s*:",
        r"(?:\b\d{2}[A-Z]\d{2}(?:\s*[,;]\s*\d{2}[A-Z]\d{2})*\s*)?\b1\.\s*Introduction\s+\d{1,5}\b",
        r"(?<=\.)\s*1\.\s*introduction\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, abstract, re.I)
        if match:
            abstract = abstract[:match.start()]
    result = deepcopy(paper)
    result["title"], result["abstract"] = plain_math_text(title), plain_math_text(abstract)
    # Remove a dangling MSC list before a table of contents; it is metadata.
    result["abstract"] = re.sub(r"\s*\b\d{2}[A-Z]\d{2}(?:\s*[,;]\s*\d{2}[A-Z]\d{2})*\s*$", "", result["abstract"]).strip()
    if len(result["abstract"].split()) < 15 or not result["title"]:
        return None, {"paper_id": pid, "reason": "insufficient text after conservative cleanup"}
    result["duplicate_keys"] = [list(k) for k in sorted({tuple(k) for k in identities(paper)} | set(identities(result)))]
    changed = [key for key in ("title", "abstract") if original[key] != result[key]]
    return result, {"paper_id": pid, "changed_fields": changed} if changed else None
