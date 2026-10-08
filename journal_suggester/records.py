"""Conservative normalization and duplicate grouping before any split."""
import html
import re
import unicodedata
from collections import defaultdict
from html.parser import HTMLParser

from .io import digest


# These titles identify a section type, not a particular scholarly work.
GENERIC_TITLES = {"introduction"}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag.split(":")[-1] in {"p", "title", "br"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.split(":")[-1] in {"p", "title"}:
            self.parts.append(" ")


def clean_text(value):
    value = re.sub(r"<(?:\w+:)?title[^>]*>\s*Abstract\s*</(?:\w+:)?title>", "", value or "", flags=re.I)
    parser = PlainText()
    parser.feed(value)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"^\s*Abstract\s*[:.]\s*", "", text, flags=re.I)
    # Remove explicit publication notices, never bare journal/mathematics words.
    text = re.sub(r"(?:^|(?<=[.!?])\s+)(?:This (?:paper|article) (?:has been |was |is )?(?:accepted|published)|Published (?:in|online)|To appear in|Accepted for publication in)[^\n.!?]*(?:[.!?](?=\s|$)|\n|$)", " ", text, flags=re.I)
    text = re.sub(r"(?:©|Copyright\s*\(?[cC]?\)?).*$", "", text, flags=re.I)
    return " ".join(text.split())


def title_key(title):
    # Keep mathematical symbols: x+y and x-y must not collapse to the same title.
    return " ".join(unicodedata.normalize("NFKC", html.unescape(title)).casefold().split()).rstrip(".")


def doi_key(doi):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", (doi or "").strip(), flags=re.I).lower()


def arxiv_key(value):
    value = re.sub(r"^https?://(?:www\.)?arxiv.org/(?:abs|pdf)/", "", value or "")
    return re.sub(r"v\d+$", "", value.removesuffix(".pdf"))


def identities(paper):
    title = title_key(paper["title"])
    out = [("title", title)] if title and title not in GENERIC_TITLES else []
    if paper.get("doi"):
        out.append(("doi", doi_key(paper["doi"])))
    if paper.get("arxiv_id"):
        out.append(("arxiv", arxiv_key(paper["arxiv_id"])))
    # Preserve identities already established by the publication audit or an earlier merge.
    out.extend(tuple(key) for key in paper.get("duplicate_keys", []))
    if not out:
        out.append(("paper", paper["paper_id"]))
    return out


def deduplicate(papers):
    parent = list(range(len(papers)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, paper in enumerate(papers):
        for key in identities(paper):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, paper in enumerate(papers):
        groups[root(i)].append(paper)
    good, quarantine = [], []
    for group in groups.values():
        if len({p["journal_id"] for p in group}) > 1:
            quarantine.append({"reason": "conflicting_journal_labels", "records": group})
            continue
        keys = sorted({key for p in group for key in identities(p)})
        paper = dict(max(group, key=lambda p: (len(p.get("abstract", "")), p["paper_id"])))
        paper["group_id"] = digest(keys)[:24]
        paper["duplicate_keys"] = keys
        paper["versions"] = sorted({v for p in group for v in [p["paper_id"], *p.get("versions", [])]})
        good.append(paper)
    return sorted(good, key=lambda p: p["paper_id"]), quarantine
