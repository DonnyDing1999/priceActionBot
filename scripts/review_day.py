"""Run the daily-review agent over all journaled sessions -> lessons + experience cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import AgentConfig, key_status  # noqa: E402
from pab.experience import read_cases  # noqa: E402
from pab.review import JDIR, review_session  # noqa: E402


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No key for provider '{st['provider']}'. Put {st['hint']} in .env.")
        return

    sessions = sorted({p.name[:-5] for p in JDIR.glob("*.json") if ".review." not in p.name})
    print(f"reviewing {len(sessions)} sessions with {st['provider']}/{st['model']}\n")

    for s in sessions:
        try:
            r = review_session(s)
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
