"""Run the daily-review agent over all journaled sessions -> lessons + experience cases.

PA_JOURNAL_DIR overrides which journal dir is reviewed (default data/journal/mes/am).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import AgentConfig, key_status  # noqa: E402
from pab.experience import read_cases  # noqa: E402
from pab.review import JDIR as _DEFAULT_JDIR, review_session  # noqa: E402

JDIR = Path(os.getenv("PA_JOURNAL_DIR", str(_DEFAULT_JDIR)))
INSTRUMENT = os.getenv("PA_INSTRUMENT", "mes")   # tags the promoted experience cases
WINDOW = os.getenv("PA_WINDOW", "am")


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No key for provider '{st['provider']}'. Put {st['hint']} in .env.")
        return

    all_sessions = sorted({p.name[:-5] for p in JDIR.glob("*.json")
                           if ".review." not in p.name})
    sessions = [s for s in all_sessions if not (JDIR / f"{s}.review.json").exists()]
    print(f"reviewing {len(sessions)} sessions "
          f"({len(all_sessions) - len(sessions)} already reviewed, skipped) "
          f"with {st['provider']}/{st['model']}\n")

    for s in sessions:
        try:
            r = review_session(s, jdir=JDIR, instrument=INSTRUMENT, window=WINDOW)
        except Exception as e:  # noqa: BLE001
            print(f"=== {s} === ERROR: {type(e).__name__}: {str(e)[:140]}\n")
            continue
        print(f"=== {s} ===")
        print("  summary:", r.get("day_summary"))
        for pe in r.get("process_errors", []):
            print("  [process error]", pe)
        for ls in r.get("lessons", []):
            print("  [lesson]", ls)
        print(f"  experience cases written: {r.get('_experience_cases_written', 0)}")
        print()

    all_cases = read_cases(k=999)
    print(f"experience library now holds {len(all_cases)} cases (experience/cases.jsonl)")


if __name__ == "__main__":
    main()
