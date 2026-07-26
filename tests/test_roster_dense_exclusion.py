"""
Tests for excluding directory rosters from the dense retrieval arm.

Background: asking prod "What research areas does TAMU ECE specialize in?"
returned these as its top sources, against a healthy 1,859-chunk database:

    profiles/index.html#Qatar-Faculty     — complete directory roster
    profiles/index.html#Leadership        — complete directory roster
    profiles/index.html#Emeritus-Faculty  — complete directory roster

Not one research page. crawler.py builds one roster chunk per role, each listing
every person in that role, so they are long and dense with proper nouns —
attractor documents that score plausibly against almost any query.

The danger in fixing this is over-correcting. graph.json holds only the 71
teaching faculty (no leadership, emeritus, Qatar or staff nodes), so these
chunks are the ONLY source for those questions. They are therefore excluded from
the DENSE arm alone and remain reachable through the keyword and fuzzy arms,
which is sound because those questions name the role out loud. The tests below
are mostly guarding that boundary.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_db_pool_resilience import FakeConn, FakePool, retriever  # noqa: E402,F401


REAL_ROSTER_URLS = [
    "https://engineering.tamu.edu/electrical/profiles/index.html#Qatar-Faculty",
    "https://engineering.tamu.edu/electrical/profiles/index.html#Leadership",
    "https://engineering.tamu.edu/electrical/profiles/index.html#Emeritus-Faculty",
    "https://engineering.tamu.edu/electrical/profiles/index.html#Staff",
    "https://engineering.tamu.edu/electrical/profiles/index.html#Faculty",
]

MUST_STAY_RETRIEVABLE = [
    "https://engineering.tamu.edu/electrical/profiles/narayanan-krishna.html",
    "https://engineering.tamu.edu/electrical/profiles/index.html",
    "https://engineering.tamu.edu/electrical/research/analog-mixed-signals.html",
    "https://engineering.tamu.edu/electrical/academics/degrees/index.html",
]


def _like_to_regex(pattern: str) -> re.Pattern:
    """Translate a SQL LIKE pattern to a regex so the exclusion can be checked
    against real URLs without a database."""
    out = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$")


# ── The pattern selects exactly the right rows ───────────────────────────────

@pytest.mark.parametrize("url", REAL_ROSTER_URLS)
def test_pattern_matches_every_roster_url(url):
    assert _like_to_regex(retriever.ROSTER_URL_PATTERN).match(url), \
        f"roster not excluded from dense arm: {url}"


@pytest.mark.parametrize("url", MUST_STAY_RETRIEVABLE)
def test_pattern_does_not_match_ordinary_pages(url):
    """Individual profiles and the un-anchored directory index must be
    unaffected — only the per-role roster aggregates are attractors."""
    assert not _like_to_regex(retriever.ROSTER_URL_PATTERN).match(url), \
        f"wrongly excluded from dense arm: {url}"


# ── SQL-level assertions ─────────────────────────────────────────────────────

class RecordingCursor:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sink.append((" ".join(sql.split()), params))

    def fetchall(self):
        return []

    def fetchone(self):
        return (0,)


@pytest.fixture
def captured(monkeypatch):
    """Run a search arm against a fake connection and capture the SQL issued."""
    sink = []
    conn = FakeConn(alive=True)
    conn.cursor = lambda: RecordingCursor(sink)
    monkeypatch.setattr(retriever, "_get_pool", lambda: FakePool([conn]))
    retriever._breaker_record_success()
    return sink


def test_dense_search_excludes_rosters(captured):
    retriever._dense_search([0.0] * 384)
    sql, params = captured[-1]
    assert "url NOT LIKE" in sql, "dense arm must filter roster docs"
    assert retriever.ROSTER_URL_PATTERN in params


def test_dense_search_excludes_rosters_with_section_filter(captured):
    """The section-filtered branch is a separate SQL string — easy to update one
    and forget the other."""
    retriever._dense_search([0.0] * 384, section_filter="people")
    sql, params = captured[-1]
    assert "url NOT LIKE" in sql
    assert "section = " in sql
    assert retriever.ROSTER_URL_PATTERN in params


def test_keyword_arm_still_reaches_rosters(captured):
    """"Who are the emeritus faculty" must still work — the graph has no
    emeritus nodes, so these chunks are the only source."""
    retriever._keyword_search("who are the emeritus faculty")
    assert captured, "keyword arm issued no query"
    assert "url NOT LIKE" not in captured[-1][0], \
        "keyword arm must NOT exclude rosters; they'd become unreachable"


def test_fuzzy_arm_still_reaches_rosters(captured):
    retriever._fuzzy_search("emeritus faculty")
    if not captured:
        pytest.skip("fuzzy arm short-circuited for this query")
    assert "url NOT LIKE" not in captured[-1][0], \
        "fuzzy arm must NOT exclude rosters"


# ── The arms are unioned, which is what makes this safe ──────────────────────

def test_retrieve_unions_the_arms_rather_than_chaining_them():
    """This is the load-bearing assumption. If the lexical arms only re-scored
    dense candidates, excluding rosters from dense would remove them entirely
    and silently break every leadership/emeritus/Qatar/staff question."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "backend", "retriever.py")).read()
    body = src[src.index("def retrieve("):]
    body = body[:body.index("\ndef ", 10)] if "\ndef " in body[10:] else body
    assert "candidates = list(dense_results)" in body
    assert "for arm in (keyword_results" in body


def test_graph_cannot_answer_non_faculty_roles():
    """Documents *why* the exclusion is dense-only. If the graph ever gains
    leadership/emeritus/staff nodes, a blanket exclusion becomes an option and
    this test should start failing to prompt that reconsideration."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "backend"))
    import graph_retriever as gr
    nodes = gr._load_graph()["nodes"]
    assert set(nodes) == {"faculty", "research_areas", "degree_programs",
                          "research_centers"}, (
        "graph node types changed — re-evaluate whether roster chunks still "
        "need to stay retrievable via the lexical arms")
