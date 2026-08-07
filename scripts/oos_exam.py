"""One-shot out-of-sample exam runner — makes the frozen-strategy discipline mechanical.

The discipline (paid for in blood): an OOS test is honest only if you RUN IT ONCE on a
strategy you have already frozen, and REPORT WHATEVER COMES OUT. No peeking at the
result and tweaking, no re-running until the number looks good — the first time a knob
moves in response to OOS output, that data is burned. This runner enforces the
mechanical parts and leaves the honesty to you:
  * refuses to run on a dirty tree (uncommitted tracked changes = not a frozen strategy)
  * refuses to re-run for an instrument/window that already has a manifest (one shot)
  * stamps a manifest (HEAD sha, config, experience sha) BEFORE the run, so the exact
    frozen artifact under test is on the record, then marks it completed afterwards.

A LATER frozen lineage takes its own one-shot exam under PA_EXAM_ID: the manifest is
namespaced per exam id, so a new hypothesis is unblocked without unblocking (or touching)
an earlier exam's manifest — the unversioned path stays permanently spent.

Env: PA_BARS (required — the held-out OOS 5m parquet), PA_SLICE (optional), plus the
usual PA_PROVIDER / PA_MODEL / PA_INSTRUMENT / PA_WINDOW / PA_DAYFILTER, and
PA_EXAM_ID (exam namespace) / PA_EXPERIENCE_FILE (experience set the run reads) /
PA_EXAM_CRITERIA (pre-registered pass/fail text, stamped at freeze time).
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import AgentConfig  # noqa: E402  (loads .env on import)
from pab.orchestration import journal_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "experience" / "cases.jsonl"


def _dirty_tracked() -> list[str]:
    """Porcelain lines for tracked-file changes (untracked '??' entries don't count)."""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in out if ln and not ln.startswith("??")]


def _manifest_path(jdir: Path) -> Path:
    """One manifest per exam id; unset PA_EXAM_ID = the legacy unversioned path (which the
    first exam's manifest keeps blocked forever)."""
    exam = os.getenv("PA_EXAM_ID")
    return jdir / (f"oos_manifest_{exam}.json" if exam else "oos_manifest.json")


def _experience_path() -> Path:
    """The experience file the run will ACTUALLY read — the sha must cover that file, not
    the live library, or the manifest records a set the exam never saw."""
    exp = os.getenv("PA_EXPERIENCE_FILE")
    return Path(exp) if exp else EXPERIENCE


def _rel(p: Path) -> str:
    """Repo-relative when inside ROOT, else absolute."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    dirty = _dirty_tracked()
    if dirty:
        print("refusing: working tree has uncommitted tracked changes — a frozen "
              "strategy has none. Offending paths:")
        for ln in dirty:
            print("  " + ln)
        sys.exit(1)
    if not os.getenv("PA_BARS"):
        print("refusing: set PA_BARS to the held-out OOS 5m parquet (e.g. mnq_5m.parquet)")
        sys.exit(1)
    exp = _experience_path()
    if os.getenv("PA_EXPERIENCE_FILE") and not exp.exists():   # a typo here = empty library
        print(f"refusing: PA_EXPERIENCE_FILE={exp} does not exist — the exam would burn "
              "its one shot on an EMPTY experience library.")
        sys.exit(1)

    cfg = AgentConfig()
    exam_id = os.getenv("PA_EXAM_ID")
    manifest_path = _manifest_path(journal_dir(cfg.instrument, cfg.window))
    if manifest_path.exists():
        tag = f"{cfg.instrument}/{cfg.window}" + (f" exam '{exam_id}'" if exam_id else "")
        print(f"refusing: exam already taken for {tag} — one shot "
              f"only ({manifest_path}). Peeking-and-tweaking is exactly what this prevents.")
        sys.exit(3)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    manifest = {
        "instrument": cfg.instrument, "window": cfg.window,
        "exam_id": exam_id,
        "git_head": head,
        "provider": cfg.provider, "model": cfg.resolved_model(),
        "temperature": cfg.temperature,
        "dayfilter": os.getenv("PA_DAYFILTER", "overlap"),
        "bars": os.getenv("PA_BARS"), "slice": os.getenv("PA_SLICE"),
        "sessions_file": os.getenv("PA_SESSIONS_FILE"),
        "sessions_file_sha": (hashlib.sha256(
            Path(os.getenv("PA_SESSIONS_FILE")).read_bytes()).hexdigest()
            if os.getenv("PA_SESSIONS_FILE") else None),
        "experience_file": _rel(exp),
        "experience_sha": (hashlib.sha256(exp.read_bytes()).hexdigest()
                           if exp.exists() else None),
        "criteria": os.getenv("PA_EXAM_CRITERIA"),   # pre-registered verdict rule, frozen here
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"OOS exam frozen -> {manifest_path}\n{json.dumps(manifest, indent=2)}\n")

    from scripts.backtest_llm import main as run_backtest   # in-process, one shot
    run_backtest()

    manifest["completed"] = True
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nOOS exam complete — manifest marked done: {manifest_path}")


if __name__ == "__main__":
    main()
