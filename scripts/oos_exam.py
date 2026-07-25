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

Env: PA_BARS (required — the held-out OOS 5m parquet), PA_SLICE (optional), plus the
usual PA_PROVIDER / PA_MODEL / PA_INSTRUMENT / PA_WINDOW / PA_DAYFILTER.
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

    cfg = AgentConfig()
    manifest_path = journal_dir(cfg.instrument, cfg.window) / "oos_manifest.json"
    if manifest_path.exists():
        print(f"refusing: exam already taken for {cfg.instrument}/{cfg.window} — one shot "
              f"only ({manifest_path}). Peeking-and-tweaking is exactly what this prevents.")
        sys.exit(3)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    manifest = {
        "instrument": cfg.instrument, "window": cfg.window,
        "git_head": head,
        "provider": cfg.provider, "model": cfg.resolved_model(),
        "temperature": cfg.temperature,
        "dayfilter": os.getenv("PA_DAYFILTER", "overlap"),
        "bars": os.getenv("PA_BARS"), "slice": os.getenv("PA_SLICE"),
        "sessions_file": os.getenv("PA_SESSIONS_FILE"),
        "sessions_file_sha": (hashlib.sha256(
            Path(os.getenv("PA_SESSIONS_FILE")).read_bytes()).hexdigest()
            if os.getenv("PA_SESSIONS_FILE") else None),
        "experience_sha": (hashlib.sha256(EXPERIENCE.read_bytes()).hexdigest()
                           if EXPERIENCE.exists() else None),
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
