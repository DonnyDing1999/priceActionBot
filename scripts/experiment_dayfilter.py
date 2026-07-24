"""Free replay experiment: would a prospective day-grade filter have helped v3?

Method: grade every session A/C from prior-day texture (pab.dayfilter, no
lookahead), then remove C-day trades from the v3 journal record (skipping a whole
day has no path side-effects — sessions are independent). Report clean-segment
impact for each variant, feature distributions, and the hindsight-oracle bound
for context. Honest-reporting rules: show ALL variants, not the best one.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.bars import load_bars  # noqa: E402
from pab.dayfilter import day_features, grade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / "data" / "journal" / "mes" / "am"
SPLIT = "2026-02-18"


def stats(trades):
    if not trades:
        return "no trades"
    pnl = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnl if p > 0]
    gw, gl = sum(wins), -sum(p for p in pnl if p <= 0)
    rs = [t["r"] for t in trades]
    t_stat = (statistics.mean(rs) / (statistics.stdev(rs) / len(rs) ** 0.5)
              if len(rs) > 2 else 0.0)
    return (f"{len(trades)}tr WR {len(wins)/len(trades):.0%} ${sum(pnl):+.0f} "
            f"PF {gw/max(0.01, gl):.2f} t={t_stat:+.2f}")


def main() -> None:
    cont = load_bars(ROOT / "data" / "raw" / "mes_5m.parquet")

    sessions: dict[str, list] = {}
    for p in sorted(JDIR.glob("2026-*.json")):
        if ".review." in p.name:
            continue
        j = json.loads(p.read_text("utf-8"))
        if not j["meta"].get("run_id", "").startswith("bt_"):
            continue                     # only the v3 eval run
        sessions[j["meta"]["session"]] = j["trades"]

    feats = {s: day_features(cont, s) for s in sessions}
    clean = [s for s in sessions if s > SPLIT]

    print(f"v3 sessions: {len(sessions)} ({len(clean)} clean) | "
          f"feature quartiles over all sessions:")
    for k in ("overlap", "efficiency", "small_body", "range_vs_adr"):
        vals = sorted(f[k] for f in feats.values() if f)
        q = [vals[int(len(vals) * x)] for x in (0.25, 0.5, 0.75)]
        print(f"  {k:<13} q25/50/75 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")

    base_clean = [t for s in clean for t in sessions[s]]
    print(f"\nbaseline clean segment: {stats(base_clean)}")

    for variant in ("overlap", "efficiency", "combo"):
        grades = {s: grade(feats[s], variant) for s in sessions}
        c_days = [s for s in clean if grades[s] == "C"]
        kept = [t for s in clean if grades[s] == "A" for t in sessions[s]]
        dropped = [t for s in c_days for t in sessions[s]]
        # detector calibration: were the skipped days actually bad?
        c_pnl = sum(t["pnl_usd"] for t in dropped)
        c_traded = [s for s in c_days if sessions[s]]
        c_neg = sum(1 for s in c_traded if sum(t["pnl_usd"] for t in sessions[s]) < 0)
        print(f"\n[{variant}] C-days in clean: {len(c_days)}/{len(clean)} "
              f"(dropped {len(dropped)} trades worth ${c_pnl:+.0f}; "
              f"{c_neg}/{len(c_traded)} traded C-days were negative)")
        print(f"  filtered clean segment: {stats(kept)}")
        by_m: dict[str, int] = {}
        for s in c_days:
            by_m[s[:7]] = by_m.get(s[:7], 0) + 1
        print(f"  C-days by month: {dict(sorted(by_m.items()))}")

    # hindsight oracle (NOT a strategy — context for the max possible)
    oracle = [t for s in clean if sum(x["pnl_usd"] for x in sessions[s]) > 0
              for t in sessions[s]]
    print(f"\n[oracle: drop every losing day — hindsight bound] {stats(oracle)}")


if __name__ == "__main__":
    main()
