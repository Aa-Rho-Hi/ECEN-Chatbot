"""
Tests for the DeepEval run-to-run comparison (scripts/deepeval_eval.py).

Background: _write_report overwrote eval_reports/deepeval_report.json on every
run, so evaluating a branch destroyed the record of what the previous build
scored — leaving "did answer quality shift?" unanswerable. Runs are now archived
to eval_reports/history/ and can be diffed with --compare.

deepeval_eval.py imports the `deepeval` package at module scope, which is heavy
and not installed in CI, so the comparison helpers are loaded in isolation.
"""
import importlib.util
import json
import os
import sys
import types

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(REPO, "scripts", "deepeval_eval.py")


@pytest.fixture(scope="module")
def mod():
    """Load deepeval_eval.py with its heavy/networked imports stubbed out."""
    for name in ("deepeval", "deepeval.metrics", "deepeval.test_case",
                 "deepeval.models"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    deepeval = sys.modules["deepeval"]
    deepeval.evaluate = lambda *a, **kw: None

    metrics = sys.modules["deepeval.metrics"]
    for cls in ("AnswerRelevancyMetric", "FaithfulnessMetric", "GEval",
                "BaseMetric", "HallucinationMetric"):
        setattr(metrics, cls, type(cls, (), {"__init__": lambda self, *a, **kw: None}))

    test_case = sys.modules["deepeval.test_case"]
    test_case.LLMTestCase = type("LLMTestCase", (), {"__init__": lambda self, *a, **kw: None})
    test_case.LLMTestCaseParams = types.SimpleNamespace(
        INPUT="input", ACTUAL_OUTPUT="actual_output",
        EXPECTED_OUTPUT="expected_output", CONTEXT="context",
        RETRIEVAL_CONTEXT="retrieval_context")

    spec = importlib.util.spec_from_file_location("deepeval_eval_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"deepeval_eval.py could not be imported with stubs: {e}")
    return module


def _case(cid, passed, metrics=None, question="q?"):
    return {
        "id": cid,
        "priority": "P0",
        "tags": [],
        "question": question,
        "passed": passed,
        "deterministic_failures": [] if passed else ["missing keyword"],
        "metrics": metrics or [],
        "answer_preview": "…",
    }


def _run_file(tmp_path, name, results, when="2026-07-26 10:00:00",
              base_url="http://localhost:8000"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "meta": {"when": when, "base_url": base_url,
                 "judge_model": "gpt-4o-mini", "threshold": 0.7, "no_llm": False},
        "results": results,
    }))
    return str(path)


# ── Exit codes: this is what gates a merge ───────────────────────────────────

def test_identical_runs_report_no_regression(tmp_path, mod, capsys):
    results = [_case("A.1", True), _case("A.2", True)]
    a = _run_file(tmp_path, "a.json", results)
    b = _run_file(tmp_path, "b.json", results)
    assert mod.compare_runs(a, b) == 0
    assert "REGRESSED (0)" in capsys.readouterr().out


def test_regression_is_detected_and_exits_nonzero(tmp_path, mod, capsys):
    base = _run_file(tmp_path, "base.json", [_case("A.1", True), _case("A.2", True)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True), _case("A.2", False)])
    assert mod.compare_runs(base, cand) == 1
    out = capsys.readouterr().out
    assert "REGRESSED (1)" in out
    assert "A.2" in out


def test_fixes_do_not_count_as_regressions(tmp_path, mod, capsys):
    base = _run_file(tmp_path, "base.json", [_case("A.1", False)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True)])
    assert mod.compare_runs(base, cand) == 0
    out = capsys.readouterr().out
    assert "FIXED (1)" in out


# ── Mismatched case sets ─────────────────────────────────────────────────────

def test_cases_missing_from_candidate_are_flagged_not_silent(tmp_path, mod, capsys):
    """A --fast run diffed against a full one must say so, rather than looking
    like a clean result because the missing cases were ignored."""
    base = _run_file(tmp_path, "base.json",
                     [_case("A.1", True), _case("A.2", True), _case("A.3", True)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True)])
    assert mod.compare_runs(base, cand) == 0
    out = capsys.readouterr().out
    assert "only in baseline (2)" in out
    assert "A.2" in out and "A.3" in out
    assert "cases compared: 1" in out


def test_new_cases_are_flagged(tmp_path, mod, capsys):
    base = _run_file(tmp_path, "base.json", [_case("A.1", True)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True), _case("A.9", True)])
    mod.compare_runs(base, cand)
    assert "only in candidate (1)" in capsys.readouterr().out


# ── Score drift ──────────────────────────────────────────────────────────────

def _metric(name, score, passed=True):
    return {"metric": name, "score": score, "threshold": 0.55,
            "passed": passed, "reason": "…"}


def test_large_score_drop_is_surfaced_even_when_still_passing(tmp_path, mod, capsys):
    """Early warning: a metric sliding toward the threshold should be visible
    before it actually crosses it."""
    base = _run_file(tmp_path, "base.json",
                     [_case("A.1", True, [_metric("Faithfulness", 0.95)])])
    cand = _run_file(tmp_path, "cand.json",
                     [_case("A.1", True, [_metric("Faithfulness", 0.70)])])
    assert mod.compare_runs(base, cand) == 0
    out = capsys.readouterr().out
    assert "SCORE DRIFT" in out
    assert "0.95 → 0.70" in out


def test_small_jitter_is_ignored(tmp_path, mod, capsys):
    """GEval scores jitter run-to-run; sub-epsilon movement is noise, and
    reporting it would train you to ignore the section."""
    base = _run_file(tmp_path, "base.json",
                     [_case("A.1", True, [_metric("Relevancy", 0.90)])])
    cand = _run_file(tmp_path, "cand.json",
                     [_case("A.1", True, [_metric("Relevancy", 0.85)])])
    mod.compare_runs(base, cand)
    assert "SCORE DRIFT" not in capsys.readouterr().out


def test_metric_absent_from_baseline_does_not_crash(tmp_path, mod):
    base = _run_file(tmp_path, "base.json", [_case("A.1", True, [])])
    cand = _run_file(tmp_path, "cand.json",
                     [_case("A.1", True, [_metric("Faithfulness", 0.8)])])
    assert mod.compare_runs(base, cand) == 0


# ── Archiving ────────────────────────────────────────────────────────────────

# ── Outage runs must not count as evidence ───────────────────────────────────

def _invalid_run_file(tmp_path, name, results, **meta_extra):
    path = tmp_path / name
    meta = {"when": "2026-07-26 00:55:58", "base_url": "http://localhost:8000",
            "judge_model": "gpt-4o-mini", "threshold": 0.7, "no_llm": False,
            "invalid": True, "aborted": False,
            "backend_errors": len(results), "cases_attempted": len(results)}
    meta.update(meta_extra)
    path.write_text(json.dumps({"meta": meta, "results": results}))
    return str(path)


def test_comparison_refuses_an_invalid_candidate(tmp_path, mod, capsys):
    """The real scenario: the database was down, so every case 'failed'.
    Diffing that against a healthy baseline would report a catastrophic quality
    drop that is entirely an artifact of the outage."""
    base = _run_file(tmp_path, "base.json", [_case("A.1", True), _case("A.2", True)])
    cand = _invalid_run_file(tmp_path, "cand.json",
                             [_case("A.1", False), _case("A.2", False)])
    assert mod.compare_runs(base, cand) == 2, "refusal uses a distinct exit code"
    out = capsys.readouterr().out
    assert "COMPARISON REFUSED" in out
    assert "REGRESSED" not in out, "must not report regressions from an outage"


def test_comparison_refuses_an_invalid_baseline(tmp_path, mod, capsys):
    base = _invalid_run_file(tmp_path, "base.json", [_case("A.1", False)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True)])
    assert mod.compare_runs(base, cand) == 2
    assert "baseline" in capsys.readouterr().out


def test_valid_runs_still_compare_normally(tmp_path, mod, capsys):
    """The guard must not fire on healthy runs — reports written before this
    field existed have no 'invalid' key at all."""
    base = _run_file(tmp_path, "base.json", [_case("A.1", True)])
    cand = _run_file(tmp_path, "cand.json", [_case("A.1", True)])
    assert mod.compare_runs(base, cand) == 0
    assert "COMPARISON REFUSED" not in capsys.readouterr().out


def test_report_carries_a_visible_invalid_banner(tmp_path, mod, monkeypatch):
    """The markdown is what gets read weeks later — it must say so itself."""
    monkeypatch.setattr(mod, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HISTORY_DIR", str(tmp_path / "history"))
    meta = {"when": "2026-07-26 00:55:58", "base_url": "http://localhost:8000",
            "judge_model": "m", "threshold": 0.7, "no_llm": False,
            "invalid": True, "aborted": True,
            "backend_errors": 5, "cases_attempted": 5}
    md_path = mod._write_report([_case("A.1", False)], meta)
    text = open(md_path).read()
    assert "THIS RUN IS NOT VALID" in text
    assert "backend outage" in text
    assert "aborted early" in text


def test_valid_report_has_no_banner(tmp_path, mod, monkeypatch):
    monkeypatch.setattr(mod, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HISTORY_DIR", str(tmp_path / "history"))
    meta = {"when": "2026-07-26 10:00:00", "base_url": "http://localhost:8000",
            "judge_model": "m", "threshold": 0.7, "no_llm": False, "invalid": False}
    md_path = mod._write_report([_case("A.1", True)], meta)
    assert "NOT VALID" not in open(md_path).read()


def test_archive_slug_is_sortable_and_names_the_target(mod):
    local = mod._archive_slug({"when": "2026-07-26 10:00:00",
                               "base_url": "http://localhost:8000"})
    prod = mod._archive_slug({"when": "2026-07-06 12:50:11",
                              "base_url": "https://ecen-chatbot-x.run.app"})
    assert local == "20260726-100000-local"
    assert prod == "20260706-125011-prod"
    # Lexical sort == chronological sort, so `ls history/` reads as a timeline.
    assert sorted([local, prod]) == [prod, local]


def test_write_report_archives_a_copy(tmp_path, mod, monkeypatch):
    monkeypatch.setattr(mod, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HISTORY_DIR", str(tmp_path / "history"))
    meta = {"when": "2026-07-26 11:22:33", "base_url": "http://localhost:8000",
            "judge_model": "gpt-4o-mini", "threshold": 0.7, "no_llm": False}
    mod._write_report([_case("A.1", True)], meta)

    archived = tmp_path / "history" / "20260726-112233-local.json"
    assert archived.exists(), "each run must leave an immutable copy behind"
    assert (tmp_path / "deepeval_report.json").exists()
    assert json.loads(archived.read_text())["results"][0]["id"] == "A.1"


def test_second_run_does_not_clobber_the_first_archive(tmp_path, mod, monkeypatch):
    monkeypatch.setattr(mod, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HISTORY_DIR", str(tmp_path / "history"))
    base_meta = {"base_url": "http://localhost:8000", "judge_model": "m",
                 "threshold": 0.7, "no_llm": False}

    mod._write_report([_case("A.1", True)], {**base_meta, "when": "2026-07-26 11:00:00"})
    mod._write_report([_case("A.1", False)], {**base_meta, "when": "2026-07-26 12:00:00"})

    archives = sorted(os.listdir(tmp_path / "history"))
    assert len(archives) == 2, "the earlier run must survive the later one"
