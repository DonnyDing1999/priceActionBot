"""OOS exam v2 — per-exam manifest namespacing (PA_EXAM_ID) + filtered experience source.

Freeze-critical, so it is tested mechanically rather than trusted: a NEW frozen lineage
must be able to take its own one-shot exam, the FIRST exam's unversioned manifest must
still block the legacy path (and must never be written to), and the manifest must sha the
experience file the run ACTUALLY reads (PA_EXPERIENCE_FILE override, else cases.jsonl).

No LLM, no network, no backtest: scripts.backtest_llm is stubbed with a sentinel-raising
main(), so the runner is exercised up to — and no further than — the hand-off.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.oos_exam as oe  # noqa: E402
from pab.experience import CASES, load_all_cases  # noqa: E402
from pab.orchestration import journal_dir  # noqa: E402


class _Handoff(Exception):
    """Stub backtest fired = main() passed every refusal and stamped its manifest."""


def _prep(monkeypatch, jdir: Path, env: dict) -> None:
    stub = types.ModuleType("scripts.backtest_llm")

    def _boom() -> None:
        raise _Handoff

    stub.main = _boom
    monkeypatch.setitem(sys.modules, "scripts.backtest_llm", stub)
    monkeypatch.setattr(oe, "journal_dir", lambda *a, **k: jdir)
    monkeypatch.setattr(oe, "_dirty_tracked", lambda: [])   # the dev tree is dirty by nature
    monkeypatch.setenv("PA_INSTRUMENT", "mes")
    monkeypatch.setenv("PA_WINDOW", "am")
    monkeypatch.setenv("PA_BARS", "mes_5m.parquet")
    for k in ("PA_SESSIONS_FILE", "PA_EXAM_ID", "PA_EXPERIENCE_FILE", "PA_EXAM_CRITERIA"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _run(monkeypatch, jdir: Path, **env) -> dict:
    """main() up to the hand-off; returns the manifest it stamped before that."""
    _prep(monkeypatch, jdir, env)
    with pytest.raises(_Handoff):
        oe.main()
    return json.loads(oe._manifest_path(jdir).read_text("utf-8"))


def _refused(monkeypatch, jdir: Path, **env) -> int:
    _prep(monkeypatch, jdir, env)
    with pytest.raises(SystemExit) as e:
        oe.main()
    return e.value.code


# ---------- (a) manifest path resolution ----------

def test_manifest_path_per_exam_id(monkeypatch, tmp_path):
    monkeypatch.delenv("PA_EXAM_ID", raising=False)
    assert oe._manifest_path(tmp_path) == tmp_path / "oos_manifest.json"   # legacy, unchanged
    monkeypatch.setenv("PA_EXAM_ID", "v5")
    assert oe._manifest_path(tmp_path) == tmp_path / "oos_manifest_v5.json"
    monkeypatch.setenv("PA_EXAM_ID", "v6")
    assert oe._manifest_path(tmp_path) == tmp_path / "oos_manifest_v6.json"
    # the first exam's manifest lives at the unversioned path under the real journal dir
    # (data/ is untracked, so only the resolution is asserted here, not its existence)
    monkeypatch.delenv("PA_EXAM_ID", raising=False)
    jd = journal_dir("mes", "am")
    assert oe._manifest_path(jd) == jd / "oos_manifest.json"


# ---------- (b) refusal is per exam id, and the spent legacy path stays spent ----------

def test_refusal_per_exam_id(monkeypatch, tmp_path):
    legacy = tmp_path / "oos_manifest.json"
    legacy.write_text('{"instrument": "mes", "completed": true}', "utf-8")
    before = legacy.read_bytes()

    assert _refused(monkeypatch, tmp_path) == 3                    # unversioned = spent
    assert _run(monkeypatch, tmp_path, PA_EXAM_ID="v5t")["exam_id"] == "v5t"   # v5 unblocked
    assert (tmp_path / "oos_manifest_v5t.json").exists()
    assert legacy.read_bytes() == before                           # first exam untouched

    assert _refused(monkeypatch, tmp_path, PA_EXAM_ID="v5t") == 3  # v5 is now one-shot too
    assert _refused(monkeypatch, tmp_path) == 3                    # ...and legacy still is
    _run(monkeypatch, tmp_path, PA_EXAM_ID="v6t")                  # a third lineage still runs
    assert legacy.read_bytes() == before

    # a mistyped experience path would run the exam on an EMPTY library -> refuse first
    assert _refused(monkeypatch, tmp_path, PA_EXAM_ID="v7t",
                    PA_EXPERIENCE_FILE=str(tmp_path / "nope.jsonl")) == 1
    assert not (tmp_path / "oos_manifest_v7t.json").exists()


# ---------- (c) the manifest sha's the experience file the run actually reads ----------

def test_manifest_stamps_experience_source_and_criteria(monkeypatch, tmp_path):
    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text(json.dumps({"session": "2024-01-02", "regime": "trend"}) + "\n",
                        "utf-8")
    jdir = tmp_path / "journal"
    crit = "PF>=1.2 over 128 sessions, else the lineage is dead"

    m = _run(monkeypatch, jdir, PA_EXAM_ID="v5t",
             PA_EXPERIENCE_FILE=str(filtered), PA_EXAM_CRITERIA=crit)
    assert m["exam_id"] == "v5t" and m["criteria"] == crit
    assert m["experience_file"].endswith("filtered.jsonl")
    assert m["experience_sha"] == hashlib.sha256(filtered.read_bytes()).hexdigest()
    assert m["experience_sha"] != hashlib.sha256(CASES.read_bytes()).hexdigest()

    # no overrides -> the live library, repo-relative, and criteria/exam_id null (not absent)
    m2 = _run(monkeypatch, jdir, PA_EXAM_ID="v6t")
    assert m2["experience_file"] == "experience/cases.jsonl"
    assert m2["experience_sha"] == hashlib.sha256(CASES.read_bytes()).hexdigest()
    assert m2["criteria"] is None
    m3 = _run(monkeypatch, tmp_path / "j3")
    assert m3["exam_id"] is None and m3["criteria"] is None


# ---------- (d) load_all_cases source override ----------

def test_load_all_cases_source_override(tmp_path):
    rows = [{"session": "2024-01-02", "regime": "trend", "note": "a"},
            {"session": "2024-01-08", "regime": "range", "note": "b"}]
    src = tmp_path / "filtered.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + "\n",
                   "utf-8")
    assert load_all_cases(src) == rows                    # file order kept, blank line skipped
    assert load_all_cases(tmp_path / "missing.jsonl") == []          # missing -> [], as before
    # default path byte-identical to the pre-change behaviour
    assert load_all_cases() == load_all_cases(None) == load_all_cases(CASES)
    assert len(load_all_cases()) > len(rows)              # the live library, not the override
