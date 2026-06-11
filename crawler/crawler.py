"""
crawler.py — Crawls the TAMU ECE website and extracts clean text per page.
Respects crawl delay, stays within the /electrical/ subdomain, and skips
binary files. Returns a list of PageDoc dicts.
"""

import os
import time
import hashlib
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL = os.getenv("TARGET_URL", "https://engineering.tamu.edu/electrical/index.html")
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY_SECONDS", "1"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))

ALLOWED_DOMAINS = {
    "engineering.tamu.edu",
    "calendar.tamu.edu",
}

SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".doc", ".docx"}

# On engineering.tamu.edu the crawl is normally confined to /electrical/.
# These extra path prefixes are ECE-relevant academics pages that live
# elsewhere on the college site and should also be crawled.
ALLOWED_ENG_PREFIXES = (
    "/electrical",
    "/ce",                 # Computer Engineering department
    "/academics/eh",       # Engineering Honors (ECEN track)
    "/academics/global",   # Global Programs / study abroad
)

# Pages outside /electrical/ but worth including for the chatbot
EXTRA_SEEDS = [
    "https://calendar.tamu.edu/ecen/#!view/all",
    "https://engineering.tamu.edu/ce/index.html",
    "https://engineering.tamu.edu/academics/eh/departments/ecen-track/index.html",
    "https://engineering.tamu.edu/academics/global/opportunities-abroad/departments.html",
]

HEADERS = {
    "User-Agent": "TAMU-ECE-Chatbot-Crawler/1.0 (educational; contact aarohi0402@gmail.com)"
}

# The People directory (profiles/index.html) is rendered client-side by
# cfprofiles.js from this college-wide JSON feed; the raw HTML contains no
# names, so BFS crawling alone misses everyone not linked from other pages
# (notably all staff). We pull the feed directly instead.
PROFILE_DATA_URL = "https://engineering.tamu.edu/profile-data.json"
DEPT_TAG = "electrical"


@dataclass
class PageDoc:
    url: str
    title: str
    section: str          # about | academics | research | people | news | events | admissions | other
    text: str
    content_hash: str = field(default="")

    def __post_init__(self):
        self.content_hash = hashlib.md5(self.text.encode()).hexdigest()


def _classify_section(url: str) -> str:
    u = url.lower()
    if "/profiles/" in u or "/people" in u:
        return "people"
    if "/research/" in u or "/highlights/" in u:
        return "research"
    if "/academics/" in u or "/degrees/" in u or "/advising/" in u or "/ce/" in u or u.rstrip("/").endswith("/ce"):
        return "academics"
    if "/admissions" in u or "/scholarships" in u:
        return "admissions"
    if "/about/" in u or "/facts" in u:
        return "about"
    if "news." in u or "/news/" in u:
        return "news"
    if "calendar." in u or "/events" in u:
        return "events"
    return "other"


def _is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    # On the main domain, stay within /electrical/ plus the extra ECE-relevant
    # academics subtrees (computer engineering, engineering honors, global programs)
    if parsed.netloc == "engineering.tamu.edu" and not path.startswith(ALLOWED_ENG_PREFIXES):
        return False
    return True


def _extract_text(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    """Returns (title, clean_text)."""
    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove nav, footer, scripts, styles
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # Prefer main content area
    main = soup.find("main") or soup.find(id="maincontent") or soup.find("article") or soup.body
    if main is None:
        return title, ""

    text = main.get_text(separator="\n", strip=True)

    # Normalize unicode (NFKC) so smart quotes/dashes render consistently.
    # Correct decoding now happens at fetch time (see crawl()), so the old
    # mojibake byte-replacement hacks are no longer needed.
    text = unicodedata.normalize("NFKC", text)

    # Collapse excessive blank lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return title, "\n".join(lines)


def fetch_directory(session: requests.Session) -> tuple[list[PageDoc], list[str]]:
    """Fetch the college-wide profile feed and build directory docs for ECE.

    Returns (docs, profile_urls). Docs are: one card per person (contact info)
    plus one complete roster doc per role (Staff, Faculty, Leadership, ...) so
    'list all staff' queries can retrieve a complete enumeration. profile_urls
    are individual profile pages to feed into the BFS queue (staff pages are
    not linked from anywhere crawlable).
    """
    try:
        resp = session.get(PROFILE_DATA_URL, timeout=30)
        resp.raise_for_status()
        people = resp.json()
    except Exception as e:
        log.warning(f"Profile feed fetch failed ({PROFILE_DATA_URL}): {e}")
        return [], []

    dept = [p for p in people if DEPT_TAG in (p.get("tag") or [])]
    docs: list[PageDoc] = []
    profile_urls: list[str] = []
    rosters: dict[str, list[str]] = {}

    for p in dept:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        roles = [t for t in (p.get("tag") or []) if t != DEPT_TAG]
        role = ", ".join(roles) or "Member"
        titles = "; ".join(p.get("titles") or [])
        link = (p.get("link") or "").strip()
        url = urljoin("https://engineering.tamu.edu/", link) if link else ""
        if url and _is_allowed(url):
            profile_urls.append(url)

        lines = [f"{name} — {role}, Department of Electrical and Computer Engineering"]
        if titles:
            lines.append(f"Title: {titles}")
        if p.get("email"):
            lines.append(f"Email: {p['email']}")
        if p.get("phone"):
            lines.append(f"Phone: {p['phone']}")
        if p.get("office"):
            lines.append(f"Office: {p['office']}")
        if url:
            lines.append(f"Profile: {url}")
        docs.append(PageDoc(
            url=(url or PROFILE_DATA_URL) + "#directory",
            title=f"{name} ({role}) — ECE Directory",
            section="people",
            text="\n".join(lines),
        ))
        for r in roles:
            rosters.setdefault(r, []).append(f"- {name}" + (f" ({titles})" if titles else ""))

    for role, entries in rosters.items():
        docs.append(PageDoc(
            url=f"https://engineering.tamu.edu/electrical/profiles/index.html#{role.replace(' ', '-')}",
            title=f"TAMU ECE {role} — complete directory roster",
            section="people",
            text=(f"Complete list of {role} in the Department of Electrical and "
                  f"Computer Engineering at Texas A&M ({len(entries)} people):\n"
                  + "\n".join(entries)),
        ))

    log.info(f"Directory feed: {len(dept)} ECE people → {len(docs)} docs, "
             f"{len(profile_urls)} profile pages queued")
    return docs, profile_urls


def crawl() -> list[PageDoc]:
    visited: set[str] = set()
    queue: list[str] = [BASE_URL] + EXTRA_SEEDS
    docs: list[PageDoc] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    # Directory feed first: collects every ECE person (incl. staff) and seeds
    # their individual profile pages into the BFS queue.
    dir_docs, profile_urls = fetch_directory(session)
    docs.extend(dir_docs)
    queue.extend(profile_urls)

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        # Normalize: strip fragment
        url = url.split("#")[0].rstrip("/")
        if url in visited:
            continue
        if not _is_allowed(url):
            continue

        visited.add(url)
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning(f"HTTP {resp.status_code} — {url}")
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            log.warning(f"Failed to fetch {url}: {e}")
            continue

        # Decode from raw bytes; honor the charset from headers/meta to avoid mojibake
        resp.encoding = resp.apparent_encoding or resp.encoding
        soup = BeautifulSoup(resp.content, "lxml")

        # Discover links BEFORE _extract_text — that function calls decompose() on
        # nav/header/footer in-place, which would silently drop any links living in
        # site navigation (e.g. the "Patents and Startups" sidebar link on every
        # research page) and prevent them from ever being queued.
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0].rstrip("/")
            if href not in visited and _is_allowed(href):
                queue.append(href)

        title, text = _extract_text(soup, url)

        if len(text) > 100:  # skip near-empty pages
            docs.append(PageDoc(
                url=url,
                title=title,
                section=_classify_section(url),
                text=text,
            ))
            log.info(f"[{len(docs):>3}] {_classify_section(url):10} | {title[:60]}")

        time.sleep(CRAWL_DELAY)

    log.info(f"Crawl complete. {len(docs)} pages collected from {len(visited)} visited URLs.")
    return docs


if __name__ == "__main__":
    pages = crawl()
    print(f"\nTotal pages: {len(pages)}")
    for p in pages[:5]:
        print(f"  {p.section:12} | {p.url}")
