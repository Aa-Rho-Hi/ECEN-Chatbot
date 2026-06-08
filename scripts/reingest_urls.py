"""
reingest_urls.py — Crawl and re-ingest a specific list of URLs into Qdrant.

Usage:
    python3 scripts/reingest_urls.py

Edit the URLS list below to target any set of pages.
"""

import hashlib
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crawler"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from chunker import chunk_docs
from crawler import PageDoc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ecen_docs")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

HEADERS = {"User-Agent": "TAMU-ECE-Bot/1.0 (research crawler)"}
CRAWL_DELAY = 1.0

# ── URLs to crawl ─────────────────────────────────────────────────────────────
# Edit this list for each page group you want to re-index
URLS = [
    # Scholarships subpages
    "https://engineering.tamu.edu/electrical/admissions-and-aid/scholarships-aid/undergraduate.html",
    "https://engineering.tamu.edu/electrical/admissions-and-aid/scholarships-aid/graduate/index.html",

    # Advising subpages
    "https://engineering.tamu.edu/electrical/advising/undergraduate.html",
    "https://engineering.tamu.edu/electrical/advising/graduate.html",
    "https://engineering.tamu.edu/electrical/advising/student-ambassadors.html",

    # Extra academics pages found in nav
    "https://engineering.tamu.edu/electrical/academics/fast-track.html",
    "https://engineering.tamu.edu/electrical/academics/reu.html",
    "https://engineering.tamu.edu/electrical/academics/student-orgs.html",

    # Research section
    "https://engineering.tamu.edu/electrical/research/index.html",
    "https://engineering.tamu.edu/electrical/research/research-areas.html",
    "https://engineering.tamu.edu/electrical/research/research-centers.html",
    "https://engineering.tamu.edu/electrical/research/patents-and-startups.html",
    "https://engineering.tamu.edu/electrical/highlights/index.html",

    # Research area subpages
    "https://engineering.tamu.edu/electrical/research/analog-mixed-signals.html",
    "https://engineering.tamu.edu/electrical/research/artificial-intelligence-and-machine-learning.html",
    "https://engineering.tamu.edu/electrical/research/biomedical-imaging-sensing-genomic-signal-processing.html",
    "https://engineering.tamu.edu/electrical/research/chip-manufacturing.html",
    "https://engineering.tamu.edu/electrical/research/computer-engineering-systems.html",
    "https://engineering.tamu.edu/electrical/research/communications-and-networks.html",
    "https://engineering.tamu.edu/electrical/research/device-science-and-nanotechnology.html",
    "https://engineering.tamu.edu/electrical/research/electromagnetics-microwaves.html",
    "https://engineering.tamu.edu/electrical/research/energy-and-power.html",
    "https://engineering.tamu.edu/electrical/research/information-science-and-systems.html",
    "https://engineering.tamu.edu/electrical/research/security.html",
    # Academics main
    "https://engineering.tamu.edu/electrical/academics/index.html",
    "https://engineering.tamu.edu/electrical/prospective-students.html",
    "https://engineering.tamu.edu/electrical/current-students.html",
    "https://engineering.tamu.edu/electrical/advising/index.html",

    # Undergraduate degrees
    "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/index.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/bs-elen.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/bs-ce.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/undergraduate/minor.html",

    # Graduate degrees
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/index.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/ms-elen.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/ms-mesc.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/ms-ce.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/phd-elen.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/phd-ce.html",

    # Certificates
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/mixed-signal-integrated-circuit-design-online-certificate1.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/circuit-design-certificate.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/online-electromag-cert.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/semiconductor-certificate.html",

    # Online degrees
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/index.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/online-ms-ce.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/online-ms-elen.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/online-phd-ce.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/graduate/distance-learning/online-phd-elen.html",

    # Other options
    "https://engineering.tamu.edu/electrical/academics/professional-development-short-courses.html",
    "https://engineering.tamu.edu/academics/eh/departments/ecen-track/index.html",
]

# ── Manually crafted pages ────────────────────────────────────────────────────
# Use this for pages where the crawler misses structured content (news, dynamic sections).
# Each entry is a PageDoc-like dict: url, title, section, text (pre-cleaned).
MANUAL_PAGES = [

    # ── About ─────────────────────────────────────────────────────────────────
    {
        "url": "https://engineering.tamu.edu/electrical/about/index.html",
        "title": "About Us | TAMU Electrical and Computer Engineering",
        "section": "about",
        "text": """About Us — TAMU Department of Electrical and Computer Engineering

Almost any technology that distinguishes the 20th and 21st centuries from previous history has the imprint of electrical and computer engineering — electric power, radio, television, radar, satellite communication, global positioning system, medical diagnostic and procedure systems, sophisticated domestic appliances, cell phones, computers and sophisticated sensors and control systems used in underwater, space exploration and national security.

Electrical engineering has advanced national and global prosperity through research, development and application of electrical and information technologies and sciences for the benefit of humanity, and has helped create the global village. By choosing electrical or computer engineering, our graduates embark on an exciting and productive career with endless opportunities and help in shaping a better future for mankind.

Mission:
As a major department with an enrollment of about 1,500 bachelor's, 270 doctoral and 280 master's students pursuing degrees in electrical and computer engineering, our mission is fourfold:
- To create new knowledge and challenge young minds by participation in the process of discovery and invention.
- To educate electrical and computer engineers with a solid background of fundamentals, stretching their imaginations.
- To prepare graduates for an exciting future.
- To serve the society through research, education and outreach activities.

Location: Wisenbaker Engineering Building (WEB), Texas A&M University, College Station, TX 77843-3128
Phone: 979-845-7441

Related pages: Accreditations, Facts and Figures, Maps and Directions, Contact Us""",
    },

    {
        "url": "https://engineering.tamu.edu/electrical/about/facts.html",
        "title": "Facts and Figures | TAMU Electrical and Computer Engineering",
        "section": "about",
        "text": """Facts and Figures — TAMU ECE Department

Rankings:
- #13 among public institution graduate programs (US News)
- Consistently ranked among the nation's leading engineering programs

Enrollment:
- 1,531 undergraduate students enrolled
- 657 graduate students enrolled (approximately 270 doctoral, 280 master's)
- About 1,500 bachelor's students total

Faculty:
- Large faculty with expertise spanning all major areas of electrical and computer engineering
- Faculty lead research in AI/ML, communications, power systems, semiconductors, cybersecurity, and more

Degrees Offered:
Undergraduate: Bachelor of Science in Electrical Engineering, Bachelor of Science in Computer Engineering, Minor in Electrical Engineering
Graduate: MS in Electrical Engineering, MS in Computer Engineering, MS in Microelectronics and Semiconductors, PhD in Electrical Engineering, PhD in Computer Engineering
Online: Online MS and PhD programs available
Certificates: Analog/Mixed-Signal IC Design, Digital IC Design, Electromagnetics, Semiconductor Manufacturing""",
    },

    {
        "url": "https://engineering.tamu.edu/electrical/contact.html",
        "title": "Contact | TAMU Electrical and Computer Engineering",
        "section": "about",
        "text": """Contact — TAMU Department of Electrical and Computer Engineering

Main Office:
Department of Electrical and Computer Engineering
Texas A&M University
301 Wisenbaker Engineering Building
College Station, TX 77843-3128
Phone: 979-845-7441

College of Engineering:
Texas A&M University College of Engineering
3127 TAMU, College Station, TX 77843-3127
Email: easa@tamu.edu
Phone: 979-845-7200

Building: Wisenbaker Engineering Building (WEB)

Online Resources:
- Staff Directory: services.tamu.edu/directory-search
- EngNet Intranet: tamucs.sharepoint.com/teams/EngNet

Social Media:
- Facebook: facebook.com/tamuece
- YouTube: TAMU ECE channel
- LinkedIn: Department of Electrical and Computer Engineering, Texas A&M University
- Instagram: @tamuecen""",
    },

    # ── Academics ─────────────────────────────────────────────────────────────
    {
        "url": "https://engineering.tamu.edu/electrical/academics/index.html",
        "title": "Academics | TAMU Electrical and Computer Engineering",
        "section": "academics",
        "text": """Academics — TAMU Department of Electrical and Computer Engineering

At Texas A&M, students in electrical and computer engineering step into fields that power the future, from telecommunications and biomedical systems to energy, electronics, and computer architecture. These disciplines are distinct yet deeply connected, offering limitless opportunities to innovate and make an impact.

Our undergraduate experience extends beyond the classroom, with pathways to contribute to cutting-edge research, explore the world through study abroad, and even pursue a master's degree while completing a bachelor's.

By the Numbers:
- #13 among public institution graduate programs
- 1,531 enrolled undergraduate students
- 657 enrolled graduate students

Student Resources:
- Prospective Students: Explore undergraduate and graduate programs
- Current Students: Access resources for active ECE students
- Advising: Connect with an advisor for degree plan, courses, or program requirements

Undergraduate Degree Programs:
- Bachelor of Science in Electrical Engineering (BS ELEN)
- Bachelor of Science in Computer Engineering (BS CE)
- Minor in Electrical Engineering

Graduate Degree Programs:
- Master of Science in Electrical Engineering (MS ELEN)
- Master of Science in Microelectronics and Semiconductors (MS MESC)
- Master of Science in Computer Engineering (MS CE)
- Doctor of Philosophy in Electrical Engineering (PhD ELEN)
- Doctor of Philosophy in Computer Engineering (PhD CE)
- Master of Science in Data Science (jointly offered with CS, Math, Statistics, and TAMIDS)

Graduate Certificates:
- Analog and Mixed-Signal Integrated Circuit Design Online Certificate
- Digital Integrated Circuit Design Certificate
- Electromagnetic Fields and Microwave Circuit Design Online Certificate
- Semiconductor Manufacturing Certificate

Online Degree Programs:
- Online Master of Science in Computer Engineering
- Online Master of Science in Electrical Engineering
- Online Doctor of Philosophy in Computer Engineering
- Online Doctor of Philosophy in Electrical Engineering

Options for Study:
- Distance Education: Earn graduate degree online
- Study Abroad: Programs offered in several countries
- Engineering Honors: Community of high-achieving students with honors faculty
- Professional Development Short Courses: For working engineers seeking to deepen knowledge without pursuing a full degree""",
    },

    # ── Admissions ────────────────────────────────────────────────────────────
    {
        "url": "https://engineering.tamu.edu/electrical/admissions-and-aid/index.html",
        "title": "Admissions and Aid | TAMU Electrical and Computer Engineering",
        "section": "admissions",
        "text": """Admissions and Aid — TAMU Department of Electrical and Computer Engineering

Resources for prospective students:

Undergraduate Admissions:
Information on how to apply to undergraduate programs in the College of Engineering at Texas A&M University.
Visit: engineering.tamu.edu/electrical/admissions-and-aid/undergraduate-admissions

Graduate Admissions:
Information about the application process for becoming a graduate student in the Department of Electrical and Computer Engineering.
Visit: engineering.tamu.edu/electrical/admissions-and-aid/graduate-admissions

Scholarships and Financial Aid:
Information about opportunities for financial assistance available to undergraduate and graduate students.
Visit: engineering.tamu.edu/electrical/admissions-and-aid/scholarships-aid""",
    },

    # ── Research ──────────────────────────────────────────────────────────────
    {
        "url": "https://engineering.tamu.edu/electrical/research/index.html",
        "title": "Research | TAMU Electrical and Computer Engineering",
        "section": "research",
        "text": """Research — TAMU ECE

Where Knowledge Becomes Impact

The Department of Electrical and Computer Engineering at Texas A&M University is dedicated to advancing discovery and innovation that serve the greater good. Our faculty lead research and teaching in electrical and information technologies that drive progress, strengthen communities, and improve lives. From theory to real-world application, we stand at the intersection of knowledge and impact, developing solutions for Texas, the nation, and the world.

Research spans a wide range of areas including mobile communications, electric vehicles, advanced healthcare, global-scale information systems, power grids, GPS, electronic chips, artificial intelligence and machine learning, robotics, precision healthcare, and quantum computing.

Research Areas:
- Analog and Mixed Signals
- Artificial Intelligence and Machine Learning
- Biomedical Imaging, Sensing and Genomic Signal Processing
- Chip Manufacturing
- Communications and Networks
- Computer Engineering and Systems
- Device Science and Nanotechnology
- Electromagnetics and Microwaves
- Energy and Power
- Information Science and Learning Systems
- Security

Recent Research News:

Chen earns NSF CAREER Award for work on AI solutions (May 2026)
Dr. Xin Chen in Texas A&M's electrical and computer engineering department earned the prestigious NSF CAREER Award and funding for his work on AI-based control for power grids.

Smarter sensors save time and energy (May 2026)
A recent publication from Texas A&M Engineering researchers shows that in-sensor intelligence could increase the speed of data analysis and lead to a future where seeing becomes thinking.

Professor launches AI-powered teaching assistant (March 2026)
Dr. Krishna Narayanan and fellow researchers at Texas A&M have developed Encando, an AI platform designed to empower professors and students in college classrooms.

Research Centers:
The department has multiple research centers where faculty collaborate with industry partners and other institutions to drive innovation and progress.""",
    },

    # ── Degrees index ─────────────────────────────────────────────────────────
    {
        "url": "https://engineering.tamu.edu/electrical/academics/degrees/index.html",
        "title": "Degree Programs | TAMU Electrical and Computer Engineering",
        "section": "academics",
        "text": """Degree Programs — TAMU Department of Electrical and Computer Engineering

Undergraduate Programs:
At Texas A&M, undergraduate degrees in Electrical Engineering and Computer Engineering are at the core of advancing technology and powering the future. Students can explore fields ranging from telecommunications and biomedical systems to electronics, power systems, and computer architecture. Students can study abroad, contribute to world-changing research, and pursue an accelerated master's degree.

Electrical Engineering — Degrees Offered:
- BS - Bachelor of Science in Electrical Engineering
- m - Minor in Electrical Engineering
- MS - Master of Science in Electrical Engineering
- MS - Master of Science in Microelectronics and Semiconductors
- DL MEng - Online Master of Engineering in Electrical Engineering
- PhD - Doctor of Philosophy in Electrical Engineering

Computer Engineering — Degrees Offered:
- BS - Bachelor of Science in Computer Engineering
- MS - Master of Science in Computer Engineering
- PhD - Doctor of Philosophy in Computer Engineering

Graduate Programs:
Graduate programs are nationally recognized for excellence, combining world-class research with exceptional value. Students engage in high-impact scholarship that drives discoveries in energy, health, communications, and beyond.

Courses and Curriculum:
Current students can find course information on the Learning Management System (eCampus). For degree plan information, contact the advising office.
- Undergraduate Catalog course listings: catalog.tamu.edu/undergraduate
- Graduate Catalog course listings: catalog.tamu.edu/graduate""",
    },
]


def classify_section(url: str) -> str:
    if "/profiles/" in url or "/people" in url:
        return "people"
    if "/research/" in url:
        return "research"
    if "/academics/" in url or "/degrees/" in url or "/advising/" in url:
        return "academics"
    if "/admissions/" in url or "/graduate/" in url:
        return "admissions"
    if "/news" in url or "news.engineering" in url:
        return "news"
    if "/calendar" in url:
        return "events"
    if "/about/" in url or "/contact" in url:
        return "about"
    return "other"


def fetch_page(url: str) -> PageDoc | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    main = soup.find("main") or soup.find(id="maincontent") or soup.find("body")
    if not main:
        log.warning(f"No main content found for {url}")
        return None

    text = main.get_text(separator="\n", strip=True)

    # Fix encoding artifacts from smart quotes and special chars
    replacements = {
        "â\x99": "'",
        "â\x93": "–",
        "â\x94": "—",
        "â\x9c": "“",
        "â\x9d": "”",
        "â\xa2": "•",
        "â\x80\x99": "'",
        "â\x80\x93": "–",
        "â\x80\x94": "—",
        "â€™": "'",
        "â€"": "–",
        "â€"": "—",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€¢": "•",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)

    if len(text) < 100:
        log.warning(f"Too little text ({len(text)} chars) for {url}")
        return None

    log.info(f"Fetched {url} — {len(text)} chars")
    return PageDoc(
        url=url,
        title=title,
        section=classify_section(url),
        text=text,
    )


def get_embedder():
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"
    log.info(f"Loading embedder on {device}...")
    return SentenceTransformer(EMBED_MODEL, device=device)


def chunk_id_to_qdrant_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2**63)


def main():
    client = QdrantClient(url=QDRANT_URL)
    embedder = get_embedder()

    all_chunks = []

    # Ingest manually crafted pages first (they override crawled versions via upsert)
    manual_urls = {m["url"] for m in MANUAL_PAGES}
    for m in MANUAL_PAGES:
        page = PageDoc(url=m["url"], title=m["title"], section=m["section"], text=m["text"])
        chunks = chunk_docs([page])
        log.info(f"Manual: {m['url']} → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    for url in URLS:
        if url in manual_urls:
            log.info(f"Skipping crawl (manual override): {url}")
            continue
        log.info(f"Crawling: {url}")
        page = fetch_page(url)
        if page:
            chunks = chunk_docs([page])
            log.info(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)
        time.sleep(CRAWL_DELAY)

    if not all_chunks:
        log.error("No chunks produced — check URLs and network.")
        return

    log.info(f"Embedding {len(all_chunks)} chunks...")
    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    points = [
        PointStruct(
            id=chunk_id_to_qdrant_id(c.chunk_id),
            vector=vec.tolist(),
            payload={
                "chunk_id": c.chunk_id,
                "url": c.url,
                "title": c.title,
                "section": c.section,
                "text": c.text,
                "content_hash": c.content_hash,
                "indexed_at": now,
            },
        )
        for c, vec in zip(all_chunks, vectors)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    log.info(f"✓ Upserted {len(points)} points into '{COLLECTION}'")

    # Show what was indexed
    print("\n── Indexed chunks ──────────────────────────────────")
    for c in all_chunks:
        print(f"  {c.chunk_id} | {len(c.text)} chars")


if __name__ == "__main__":
    main()
