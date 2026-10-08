"""Import bounded arXiv metadata using an ID and fixed official HTTPS endpoints."""
import http.client
from html.parser import HTMLParser
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MAX_METADATA_BYTES = 1024 * 1024
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/"
FETCH_TIMEOUT = 5
ARXIV_ID = r"[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}(?:v[1-9][0-9]*)?"
ARXIV_CATEGORY_SCHEMES = {"http://arxiv.org/schemas/atom", "https://arxiv.org/schemas/atom"}
# Mathematics aliases in https://arxiv.org/category_taxonomy.
MATH_ALIASES = {"math-ph", "cs.IT", "cs.NA", "stat.TH"}
ARXIV_SCOPE_MESSAGE = "Please use a paper classified under mathematics on arXiv."
ARXIV_UNAVAILABLE_MESSAGE = "arXiv is unavailable right now. Please try again shortly or paste the title and abstract."


class ArxivScopeError(ValueError):
    def __init__(self):
        super().__init__(ARXIV_SCOPE_MESSAGE)


class ArxivUnavailableError(ValueError):
    def __init__(self):
        super().__init__(ARXIV_UNAVAILABLE_MESSAGE)


def normalize_arxiv(value):
    if not isinstance(value, str) or len(value) > 300:
        raise ValueError("Enter an arXiv link or identifier.")
    # Reject characters URL parsers may silently normalize away.
    if not value.isascii() or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Enter a plain arXiv link or identifier.")
    value = value.strip()
    # Match the entire input against one URL shape, or a bare numeric ID.
    match = re.fullmatch(r"(?:https://arxiv\.org/abs/)?(" + ARXIV_ID + r")", value)
    if not match:
        raise ValueError("Use a link like https://arxiv.org/abs/2610.08776 or a numeric arXiv ID.")
    return match.group(1)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("arXiv redirected the request. Please paste the title and abstract instead.")


class ArxivImporter:
    def __init__(self):
        self.lock = threading.Lock()
        self.last = 0
        self.cache = {}

    def fetch(self, value):
        ident = normalize_arxiv(value)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another arXiv import is running. Please try again in a moment.")
        try:
            if ident in self.cache:
                return dict(self.cache[ident])
            time.sleep(max(0, 3.1 - (time.monotonic() - self.last)))
            # Never request the supplied URL; only the validated ID is used.
            url = ARXIV_API + "?" + urllib.parse.urlencode({"id_list": ident})
            try:
                try:
                    raw = fetch_metadata(url)
                except urllib.error.HTTPError as error:
                    # Do not work around access denials, rate limits or redirects.
                    if not 500 <= error.code < 600:
                        raise ArxivUnavailableError() from None
                    raw = None
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                    raw = None
                if raw is None:
                    # No links from the page are followed; only citation metadata
                    # and its structured subject cell are read as plain text.
                    result = parse_arxiv_html(fetch_metadata(ARXIV_ABS + ident), ident)
                else:
                    result = parse_arxiv(raw, ident)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ET.ParseError):
                raise ArxivUnavailableError() from None
            finally:
                self.last = time.monotonic()
            if len(self.cache) >= 100:
                self.cache.pop(next(iter(self.cache)))
            self.cache[ident] = result
            return dict(result)
        finally:
            self.lock.release()


def fetch_metadata(url):
    request = urllib.request.Request(url, headers={"User-Agent": "journal-suggester/0.1 (single-paper metadata import)"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect).open(request, timeout=FETCH_TIMEOUT) as response:
        raw = response.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("arXiv returned an unexpectedly large response.")
    return raw


def check_identity(ident, requested):
    if not re.fullmatch(ARXIV_ID, ident):
        raise ValueError("arXiv returned an invalid paper identifier.")
    base = lambda s: re.sub(r"v\d+$", "", s)
    if base(ident) != base(requested) or (re.search(r"v\d+$", requested) and ident != requested):
        raise ValueError("arXiv returned a different paper version. Please check the link.")


def parse_arxiv(raw, requested):
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("arXiv returned an unexpectedly large response.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("arXiv returned invalid metadata.") from None
    # Metadata needs no DTD/entities. Decode first to prevent alternate encodings
    # hiding declarations; reject NULs (including UTF-16/32 byte layouts).
    if "\x00" in text or "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("arXiv returned unsupported XML declarations.")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(text).find("a:entry", ns)
    if entry is None:
        raise ValueError("No paper was found for that arXiv identifier.")
    def field(name):
        return " ".join((entry.findtext(name, "", ns)).split())
    # The API still emits HTTP identifier URIs. Parse those as metadata only;
    # accepting them here must not broaden the user-facing URL allowlist.
    entry_id = re.fullmatch(r"https?://arxiv\.org/abs/(" + ARXIV_ID + r")", field("a:id"))
    if not entry_id:
        raise ValueError("arXiv returned an invalid paper identifier.")
    ident = entry_id.group(1)
    check_identity(ident, requested)
    # Inspect both primary and cross-listed subjects in the existing response.
    # Other classification schemes and category-like text in the abstract do not qualify.
    categories = {element.get("term", "") for element in entry.findall("a:category", ns)
                  if element.get("scheme") in ARXIV_CATEGORY_SCHEMES}
    # Current API responses omit scheme on the namespaced primary category.
    categories.update(element.get("term", "") for element in entry.findall("x:primary_category", ns)
                      if element.get("scheme", ns["x"]) in ARXIV_CATEGORY_SCHEMES)
    return paper_metadata(ident, field("a:title"), field("a:summary"), field("x:doi"), categories)


class AbstractPageParser(HTMLParser):
    """Extract specific metadata; never interpret page scripts or follow links."""
    fields = {"citation_title", "citation_abstract", "citation_arxiv_id", "citation_doi", "og:url"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata = {}
        self.subject_text = []
        self.in_subjects = False
        self.subject_cells = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name") or attrs.get("property")
            if key in self.fields:
                if key in self.metadata:
                    raise ValueError("arXiv returned ambiguous metadata.")
                self.metadata[key] = attrs.get("content", "")
        if tag == "td" and "subjects" in attrs.get("class", "").split():
            self.subject_cells += 1
            self.in_subjects = True

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_subjects = False

    def handle_data(self, data):
        if self.in_subjects:
            self.subject_text.append(data)


def parse_arxiv_html(raw, requested):
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("arXiv returned an unexpectedly large response.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("arXiv returned invalid metadata.") from None
    if "\x00" in text:
        raise ValueError("arXiv returned invalid metadata.")
    page = AbstractPageParser()
    page.feed(text)
    page.close()
    # og:url identifies the displayed version, while citation_arxiv_id can be
    # unversioned. Both must describe the requested paper. Neither URL is fetched.
    version = re.fullmatch(re.escape(ARXIV_ABS) + "(" + ARXIV_ID + ")", page.metadata.get("og:url", ""))
    if not version or not re.search(r"v\d+$", version.group(1)):
        raise ValueError("arXiv returned an invalid paper version.")
    ident = version.group(1)
    check_identity(ident, requested)
    citation_id = page.metadata.get("citation_arxiv_id", "")
    check_identity(citation_id, re.sub(r"v\d+$", "", requested))
    check_identity(ident, citation_id)
    if page.subject_cells != 1 or page.in_subjects:
        raise ValueError("arXiv returned no unambiguous subject categories.")
    categories = set(re.findall(r"\(([a-z][a-z0-9-]*(?:\.[A-Za-z-]+)?)\)", "".join(page.subject_text)))
    field = lambda name: " ".join(page.metadata.get(name, "").split())
    return paper_metadata(ident, field("citation_title"), field("citation_abstract"), field("citation_doi"), categories)


def paper_metadata(ident, title, abstract, doi, categories):
    if not categories or not any(categories):
        raise ValueError("arXiv returned no subject categories. Please try again later.")
    if not any(re.fullmatch(r"math\.[A-Z]{2}", category) or category in MATH_ALIASES for category in categories):
        raise ArxivScopeError()
    if not title or not abstract:
        raise ValueError("This arXiv record has no title or abstract.")
    if len(title) > 2000 or len(abstract) > 30000 or len(doi) > 300:
        raise ValueError("arXiv returned unusually long metadata. Paste the title and abstract instead.")
    return {"title": title, "abstract": abstract, "arxiv_id": ident, "doi": doi,
            "source": "arXiv", "source_url": "https://arxiv.org/abs/" + ident,
            "notice": "Imported from arXiv. Review the title and abstract before searching."}
