"""Calibration scorer for the Decision v2 `next_bar` probability block (WP4/WP5).

Each trade-window decision may carry next_bar = {p_up, p_down, p_neutral} (integer
percents summing 99..101). This scores how well-calibrated those forecasts are with a
multiclass Brier score, against the realized direction of the FOLLOWING bar and against
the trivial base-rate forecast (predict the empirical class frequencies every time).

Realized direction is read from the next recorded decision's bar_type (trend_bull -> up,
trend_bear -> down, doji -> neutral) — no bar reload. The current journals pre-date the
next_bar field, so on real data this reports "0 scored"; the math is exercised on
synthetic journals in tests/test_wp5.py.

Brier = mean over samples of sum_k (p_k - y_k)^2  (0 best, 2 worst; y one-hot).
Skill  = 1 - brier/base_brier  (>0 means better than always predicting the base rate).

CLI: --cohort insample|oos|all (default all); --journal-dir PATH.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.veto_replay import _load_json, cohort_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / "data" / "journal" / "mes" / "am"
CLASSES = ("up", "down", "neutral")


def direction_of(bar_type: str | None) -> str | None:
    return {"trend_bull": "up", "trend_bear": "down", "doji": "neutral"}.get(bar_type or "")


def next_bar_probs(decision: dict) -> list[float] | None:
    """Return [p_up, p_down, p_neutral] in 0..1 for a valid block, else None (same
    validity rule as decision_schema._clean_next_bar: three integer percents, sum 99..101)."""
    nb = decision.get("next_bar")
    if not isinstance(nb, dict):
        return None
    vals = [nb.get("p_up"), nb.get("p_down"), nb.get("p_neutral")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and float(v).is_integer() for v in vals):
        return None
    if not (99 <= sum(int(v) for v in vals) <= 101):
        return None
    s = sum(int(v) for v in vals)
    return [int(v) / s for v in vals]      # renormalize to exactly 1.0


def _onehot(direction: str) -> list[float]:
    return [1.0 if c == direction else 0.0 for c in CLASSES]


def brier(p: list[float], y: list[float]) -> float:
    return sum((pk - yk) ** 2 for pk, yk in zip(p, y))


@dataclass
class Sample:
    session: str
    bar: int
    p: list[float]
    realized: str


def score_journal(journal: dict) -> list[Sample]:
    """Pair each decision carrying a valid next_bar with the realized direction of the
    following bar (from the next recorded decision's bar_type)."""
    decs = journal.get("decisions", [])
    by_bar = {}
    for d in decs:
        b = d.get("bar")
        if isinstance(b, int):
            by_bar.setdefault(b, d)          # first decision recorded at each bar
    session = journal.get("meta", {}).get("session", "?")
    out: list[Sample] = []
    for d in decs:
        p = next_bar_probs(d.get("decision", {}))
        b = d.get("bar")
        if p is None or not isinstance(b, int):
            continue
        nxt = by_bar.get(b + 1)
        if not nxt:
            continue
        ctx = nxt.get("context") or {}
        realized = direction_of(ctx.get("bar_type"))
        if realized is None:
            continue
        out.append(Sample(session=session, bar=b, p=p, realized=realized))
    return out


@dataclass
class CalReport:
    name: str
    samples: list[Sample] = field(default_factory=list)

    def metrics(self) -> dict:
        n = len(self.samples)
        if n == 0:
            return {"n": 0}
        counts = {c: sum(1 for s in self.samples if s.realized == c) for c in CLASSES}
        base = [counts[c] / n for c in CLASSES]        # empirical class frequencies
        model_brier = sum(brier(s.p, _onehot(s.realized)) for s in self.samples) / n
        base_brier = sum(brier(base, _onehot(s.realized)) for s in self.samples) / n
        skill = (1 - model_brier / base_brier) if base_brier else None
        return {"n": n, "brier": round(model_brier, 4),
                "base_rate_brier": round(base_brier, 4),
                "skill": (round(skill, 4) if skill is not None else None),
                "base_rates": {c: round(base[c_i], 3) for c_i, c in enumerate(CLASSES)}}


def analyze_cohort(name: str, files: list[Path]) -> CalReport:
    rep = CalReport(name=name)
    for f in files:
        j = _load_json(f)
        if j:
            rep.samples.extend(score_journal(j))
    return rep


def build_markdown(reports: list[CalReport]) -> str:
    out = ["# next_bar calibration (Brier)\n"]
    out.append("| cohort | n scored | Brier | base-rate Brier | skill | base rates (u/d/n) |")
    out.append("|---|---|---|---|---|---|")
    for r in reports:
        m = r.metrics()
        if m["n"] == 0:
            out.append(f"| {r.name} | 0 | — | — | — | (no next_bar blocks in journals) |")
            continue
        br = m["base_rates"]
        out.append(f"| {r.name} | {m['n']} | {m['brier']} | {m['base_rate_brier']} | "
                   f"{m['skill']} | {br['up']}/{br['down']}/{br['neutral']} |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["insample", "oos", "all"], default="all")
    ap.add_argument("--journal-dir", default=str(JDIR))
    args = ap.parse_args(argv)
    jdir = Path(args.journal_dir)
    names = (["insample", "oos"] if args.cohort == "all" else [args.cohort])
    label = {"insample": "insample_v41", "oos": "oos_2024_25"}
    reports = [analyze_cohort(label[n], cohort_files(jdir, n)) for n in names]
    print(build_markdown(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
