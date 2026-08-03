"""
Tests for research-area-list routing (backend/graph_retriever.py).

Background: "What research areas does TAMU ECE specialize in?" returned
"I don't have specific details about the research areas..." against a perfectly
healthy database with 1,859 chunks. Cause: no graph route existed for the
question, so it fell through to open-ended retrieval — where the `research`
section is ~40 of ~1,860 chunks against ~836 news chunks. The top-k filled with
news and the model correctly reported that its context didn't contain the
answer. Meanwhile graph.json held all eleven areas with descriptions and
faculty, and research_area_names() was imported into main.py but never called.

The risk in fixing it is over-triggering: a route that hijacks "who works on
power systems" would replace a specific, useful answer with a generic list.
Most of these tests are about that boundary.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import graph_retriever as gr  # noqa: E402


# ── Should route to the area list ────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "What research areas does TAMU ECE specialize in?",   # the reported failure
    "What research areas does the ECE department have?",
    "List all research areas in TAMU ECE.",
    "Which research areas are available?",
    "What are the research areas?",
    "what research topics does ece cover",
    "What are the department's research strengths?",
    "what are ece's specializations",
])
def test_area_list_questions_route_to_the_graph(q):
    assert gr.is_area_list_query(q) is True, f"should have routed: {q!r}"


# ── Must NOT hijack other intents ────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    # People-in-an-area questions — build_area_roster answers these far better.
    "Who are the professors in the Energy and Power research area?",
    "Which faculty work in the Security research area?",
    "Whom should I contact about research in power systems?",
    "Who researches artificial intelligence in TAMU ECE?",
    "Can you suggest a faculty mentor for communications research?",
    # A specific named area — retrieval has that page's real detail.
    "Tell me about the Security research area.",
    "What is the Energy and Power research area about?",
    "Tell me about research in Analog and Mixed Signals.",
    # Unrelated intents that happen to contain overlapping words.
    "What degree programs does the ECE department offer?",
    "What courses cover machine learning research?",
    "What is the application deadline for research assistantships?",
    "What scholarships are available for research students?",
])
def test_other_intents_are_not_hijacked(q):
    assert gr.is_area_list_query(q) is False, f"should NOT have routed: {q!r}"


def test_named_area_check_is_case_insensitive():
    assert gr.is_area_list_query("tell me about the security research area") is False


def test_empty_and_none_are_safe():
    assert gr.is_area_list_query("") is False
    assert gr.is_area_list_query(None) is False


# ── Roster content ───────────────────────────────────────────────────────────

def test_roster_lists_every_area_from_the_graph():
    roster = gr.build_area_list_roster()
    assert roster is not None
    for name in gr.research_area_names():
        assert name in roster, f"roster omitted {name!r}"


def test_roster_states_it_is_complete():
    """The degree roster taught us the model will hedge unless told not to —
    hedging is exactly the behaviour being fixed here."""
    roster = gr.build_area_list_roster()
    assert "COMPLETE" in roster
    assert "do NOT add a disclaimer" in roster or "Do NOT" in roster


def test_roster_covers_all_eleven_areas():
    names = gr.research_area_names()
    assert len(names) == 11, f"graph.json has {len(names)} areas, expected 11"
    roster = gr.build_area_list_roster()
    assert roster.count("•") == 11


def test_roster_includes_descriptions():
    roster = gr.build_area_list_roster()
    # Descriptions are what let the model say something useful per area rather
    # than emitting a bare list of headings.
    assert "—" in roster


def test_roster_is_none_when_graph_has_no_areas(monkeypatch):
    monkeypatch.setattr(gr, "_load_graph",
                        lambda: {"nodes": {"research_areas": {}}, "edges": {}})
    assert gr.build_area_list_roster() is None


# ── main.py wiring ───────────────────────────────────────────────────────────

def _main_source():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "backend", "main.py")
    with open(path) as f:
        return f.read()


def test_main_actually_calls_the_new_route():
    """research_area_names was imported and never used — the whole bug. Make
    sure the new helpers don't end up as dead imports the same way."""
    src = _main_source()
    assert "is_area_list_query(req.question)" in src
    assert "build_area_list_roster()" in src


def test_synthetic_roster_url_is_not_cited_as_a_source():
    """Graph-built chunks must not appear to users as a crawled web page."""
    src = _main_source()
    start = src.index("_SYNTHETIC_URLS")
    assert "research-area-list" in src[start:start + 400]


def test_roster_url_is_treated_as_a_curated_source_set():
    src = _main_source()
    start = src.index("_ROSTER_URLS")
    assert "research-area-list" in src[start:start + 300]
