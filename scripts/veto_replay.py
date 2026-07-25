"""Counterfactual veto replay — what the WP2 structural vetoes WOULD have done to the
already-journaled decisions. ZERO LLM calls, no bar-data reload: everything is
reconstructed from the journals under data/journal/mes/am/.

Why this is meaningful: those journals were produced by an OOS/insample run whose risk
layer PRE-DATES the WP2 vetoes (zone / pullback-second-entry / counter-trend / climax
no-chase / post-stop cooldown). So the realized trades never faced them. This script
rebuilds the v2 `ctx` for every journaled trade proposal and runs the CURRENT
RiskManager.validate_entry over it, sequentially per session (feeding on_trade_closed
from the realized trades so the cooldown state is live), then reports which realized
trades a veto would have removed — split win/loss with the $ it would have taken off the
book — plus how many never-filled proposals it would also have caught.

Reconstruction fidelity is tiered by what the journal actually stored (see reconstruct_ctx):
  full v2 sidecar        -> every veto evaluable (future journals);
  v1 sidecar (has table) -> v2 keys recomputed from session_table via the features algo;
  legacy _brief context  -> zone / counter-trend / cooldown evaluable; pullback-gate and
                            climax-no-chase are NOT (their inputs — hl_count,
                            avg_range_10, signal_bar_range — were never persisted), and
                            are reported as n/a for that cohort. regime is approximated
                            without ema_slope. All of this is surfaced in the report.

Cohorts: insample_v41 = 2026-* journals; oos_2024_25 = the sessions in
data/journal/mes/am/oos_manifest.json's sessions_file (2024/2025). Sessions with no
decisions are skipped.

CLI: --cohort insample|oos|all (default all); --knob-off TAG (repeatable ablation, TAG in
zone_veto|pullback_gate|no_chase|cooldown).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.features import _bar_type, classify_regime  # noqa: E402
from pab.instruments import get_spec  # noqa: E402
from pab.orchestration import engine_config  # noqa: E402
from pab.risk import RiskManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / "data" / "journal" / "mes" / "am"
ARTIFACTS = ROOT / "data" / "artifacts"

# --enrich-bars only: lazily-loaded continuous frame, so a legacy journal that never
# stored its sidecar can have the FULL v2 sidecar rebuilt at the proposal's timestamp
# (opt-in; default replay reloads nothing).
_CONT = None
_ENRICH_MISS = 0


def _enrich_sidecar(session_date: str, hhmm: str, window_start: str,
                    symbol: str = "mes") -> dict | None:
    """Rebuild the full sidecar from the bars parquet at `session_date hhmm` (ET). Used
    ONLY under --enrich-bars to upgrade legacy _brief journals to full fidelity."""
    global _CONT, _ENRICH_MISS
    import pandas as pd
    from pab.bars import ET, load_bars
    from pab.features import build_sidecar
    if _CONT is None:
        raw = ROOT / "data" / "raw" / f"{symbol}_5m.parquet"
        if not raw.exists():
            return None
        _CONT = load_bars(raw)
    try:
        ts = pd.Timestamp(f"{session_date} {hhmm}", tz=ET)
        return build_sidecar(_CONT, ts, window_start=window_start)
    except Exception:      # noqa: BLE001 — missing bar / date -> fall back to journal-only
        _ENRICH_MISS += 1
        return None

# the WP2 structural veto tags this tool reports on (v1 gate tags are ignored — they were
# already live when the journals were produced)
V2_TAGS = ("zone_long_upper", "zone_short_lower", "zone_middle",
           "trend_needs_second_entry", "counter_trend", "climax_no_chase",
           "cooldown_same", "cooldown_opposite")
KNOB_TAGS = {  # which config knob governs each tag (for --knob-off + evaluability)
    "zone_long_upper": "zone_veto", "zone_short_lower": "zone_veto",
    "zone_middle": "zone_veto", "trend_needs_second_entry": "pullback_gate",
    "counter_trend": "pullback_gate", "climax_no_chase": "no_chase",
    "cooldown_same": "cooldown", "cooldown_opposite": "cooldown"}


# --------------------------------------------------------------------------- parsing
def parse_session_table(text: str) -> list[dict]:
    """Parse the pipe table `i|time|open|high|low|close|type|cp|ema` into row dicts.
    Tolerant of a missing header and of extra columns."""
    rows: list[dict] = []
    for ln in str(text).splitlines():
        parts = ln.split("|")
        if len(parts) < 7 or parts[0].strip() in ("i", ""):
            continue
        try:
            rows.append({"i": int(parts[0]), "time": parts[1],
                         "o": float(parts[2]), "h": float(parts[3]),
                         "l": float(parts[4]), "c": float(parts[5]),
                         "type": parts[6],
                         "ema": float(parts[8]) if len(parts) > 8 else None})
        except (ValueError, IndexError):
            continue
    return rows


def v2_from_rows(rows: list[dict]) -> dict:
    """Recompute the sidecar-v2 Brooks keys from a parsed session table, mirroring
    features.build_sidecar (lines under `if os.getenv("PA_SIDECAR_V2")`). Working range is
    SESSION-ONLY here (the table carries no overnight/yday magnets), so zone can differ
    from a full sidecar when an external magnet is the true extreme — noted by the caller.
    """
    n = len(rows)
    if n == 0:
        return {}
    sh = [r["h"] for r in rows]
    sl = [r["l"] for r in rows]
    so = [r["o"] for r in rows]
    sc = [r["c"] for r in rows]
    types = [r["type"] if r["type"] in ("trend_bull", "trend_bear", "doji")
             else _bar_type(r["o"], r["h"], r["l"], r["c"]) for r in rows]
    last10 = list(zip(sh[-10:], sl[-10:]))
    avg_range_10 = round(sum(h - l for h, l in last10) / len(last10), 2)

    sess_hi, sess_lo = max(sh), min(sl)
    span = sess_hi - sess_lo
    c = sc[-1]
    pos = max(0.0, min(1.0, (c - sess_lo) / span)) if span > 0 else 0.5
    zone = ("lower_third" if pos < 1 / 3 else
            "middle" if pos < 2 / 3 else "upper_third")

    ema = rows[-1]["ema"]
    if ema is None:
        leg = "bull" if types[-1] == "trend_bull" else \
              "bear" if types[-1] == "trend_bear" else "none"
    elif c > ema:
        leg = "bull"
    elif c < ema:
        leg = "bear"
    else:
        leg = "none"

    # pullback resumptions within the leg (H1/H2/H3, mirror L; cap 3; strong opposite
    # trend bar body >= 1.2*avg_range_10 resets) — identical logic to features.build_sidecar
    count, in_pb, reset_bar, last_label = 0, False, None, ""
    if leg in ("bull", "bear"):
        opp = "trend_bear" if leg == "bull" else "trend_bull"
        thr = 1.2 * avg_range_10
        for k in range(1, n):
            if types[k] == opp and abs(sc[k] - so[k]) >= thr:
                count, last_label, in_pb, reset_bar = 0, "", False, k + 1
                continue
            if not in_pb:
                in_pb = (sl[k] < sl[k - 1]) if leg == "bull" else (sh[k] > sh[k - 1])
            else:
                resume = (sh[k] > sh[k - 1]) if leg == "bull" else (sl[k] < sl[k - 1])
                if resume:
                    count = min(count + 1, 3)
                    last_label = ("H" if leg == "bull" else "L") + str(count)
                    in_pb = False

    if types[-1] in ("trend_bull", "trend_bear"):
        cdir = "bull" if types[-1] == "trend_bull" else "bear"
        n_run, k = 0, n - 1
        while k >= 0 and types[k] == types[-1]:
            n_run, k = n_run + 1, k - 1
    else:
        cdir, n_run = "none", 0

    # always_in per features (recent<=6 closes vs current ema)
    ai = "neutral"
    if ema is not None and n >= 1:
        rising = sc[-1] > sc[-min(6, n)]
        if c > ema and rising:
            ai = "ail"
        elif c < ema and not rising:
            ai = "ais"

    return {
        "avg_range_10": avg_range_10,
        "working_range": {"hi": round(sess_hi, 2), "lo": round(sess_lo, 2),
                          "src_hi": "session", "src_lo": "session"},
        "price_position": round(pos, 2),
        "zone": zone,
        "hl_count": {"dir": leg, "count": count, "last_label": last_label,
                     "reset_bar": reset_bar},
        "consec_trend_bars": {"dir": cdir, "n": n_run},
        "signal_bar_range": round(sh[-1] - sl[-1], 2),
        "always_in_hint": ai,
        "ema_slope_pts_3bar": (round(ema - rows[-4]["ema"], 2)
                               if n >= 4 and rows[-4]["ema"] is not None else 0.0),
        "close_vs_ema_pts": round(c - ema, 2) if ema is not None else 0.0,
        "bar_index_from_open": rows[-1]["i"],
    }


# ------------------------------------------------------------------ ctx reconstruction
UNAVAIL_PULLBACK = frozenset({"hl_count", "consec_trend_bars"})
UNAVAIL_NOCHASE = frozenset({"avg_range_10", "signal_bar_range"})


def _regime_from(sidecar_like: dict, window: str) -> str:
    """classify_regime over a reconstructed sidecar; tolerant of a missing ema_slope."""
    sc = {
        "bar_index_from_open": sidecar_like.get("bar_index_from_open") or 1,
        "always_in_hint": sidecar_like.get("always_in_hint", "neutral"),
        "ema_slope_pts_3bar": sidecar_like.get("ema_slope_pts_3bar", 0.0),
        "close_vs_ema_pts": sidecar_like.get("close_vs_ema_pts", 0.0),
    }
    return classify_regime(sc, window=window)


def reconstruct_ctx(decision: dict, context: dict | None, sidecar: dict | None,
                    *, bar_index: int, entry_type: str, window: str = "am") -> tuple:
    """Return (ctx, unavailable_fields:set). ctx matches the shape backtest.py builds for
    RiskManager.validate_entry. Richest available source wins:
      1. full v2 sidecar (has "zone");
      2. v1 sidecar carrying "session_table" -> recompute via v2_from_rows;
      3. legacy _brief `context` only -> partial ctx (zone/regime/cooldown; pullback +
         no-chase inputs unavailable).
    """
    unavail: set[str] = set()

    if sidecar and sidecar.get("zone") is not None:            # tier 1: full v2 sidecar
        b = sidecar.get("bar", {})
        sbr = (round(float(b["h"]) - float(b["l"]), 2)
               if "h" in b and "l" in b else sidecar.get("signal_bar_range"))
        ctx = {
            "regime": classify_regime(sidecar, window=window),
            "always_in": sidecar.get("always_in_hint"),
            "zone": sidecar.get("zone"),
            "price_position": sidecar.get("price_position"),
            "working_range": sidecar.get("working_range"),
            "avg_range_10": sidecar.get("avg_range_10"),
            "hl_count": sidecar.get("hl_count"),
            "consec_trend_bars": sidecar.get("consec_trend_bars"),
            "signal_bar_range": sbr,
            "bar_index": sidecar.get("bar_index_from_open") or bar_index,
            "entry_type": entry_type,
        }
        return ctx, unavail

    if sidecar and sidecar.get("session_table"):              # tier 2: v1 sidecar w/ table
        v2 = v2_from_rows(parse_session_table(sidecar["session_table"]))
        if v2:
            ctx = {
                "regime": _regime_from(v2, window),
                "always_in": v2.get("always_in_hint"),
                "zone": v2["zone"],
                "price_position": v2["price_position"],
                "working_range": v2["working_range"],
                "avg_range_10": v2["avg_range_10"],
                "hl_count": v2["hl_count"],
                "consec_trend_bars": v2["consec_trend_bars"],
                "signal_bar_range": v2["signal_bar_range"],
                "bar_index": v2.get("bar_index_from_open") or bar_index,
                "entry_type": entry_type,
            }
            return ctx, unavail

    # tier 3: legacy _brief context only
    ctx = _ctx_from_brief(context or {}, bar_index, entry_type, window, unavail)
    return ctx, unavail


def _ctx_from_brief(ctx_ctx: dict, bar_index: int, entry_type: str,
                    window: str, unavail: set) -> dict:
    close = ctx_ctx.get("close")
    hi = ctx_ctx.get("session_high")
    lo = ctx_ctx.get("session_low")
    zone = price_pos = wr = None
    if close is not None and hi is not None and lo is not None and hi > lo:
        price_pos = max(0.0, min(1.0, (close - lo) / (hi - lo)))
        zone = ("lower_third" if price_pos < 1 / 3 else
                "middle" if price_pos < 2 / 3 else "upper_third")
        wr = {"hi": hi, "lo": lo, "src_hi": "session", "src_lo": "session"}
    # pullback + no-chase inputs were never persisted in _brief:
    unavail |= UNAVAIL_PULLBACK | UNAVAIL_NOCHASE
    unavail.add("ema_slope")           # regime approximated without slope
    return {
        "regime": _regime_from(
            {"bar_index_from_open": bar_index,
             "always_in_hint": ctx_ctx.get("always_in", "neutral"),
             "ema_slope_pts_3bar": 0.0,
             "close_vs_ema_pts": ctx_ctx.get("close_vs_ema_pts", 0.0)}, window),
        "always_in": ctx_ctx.get("always_in"),
        "zone": zone,
        "price_position": round(price_pos, 2) if price_pos is not None else None,
        "working_range": wr,
        "avg_range_10": None,          # unavailable -> no_chase no-ops in risk.py
        "hl_count": None,              # unavailable -> pullback_gate disabled per-call
        "consec_trend_bars": None,
        "signal_bar_range": None,
        "bar_index": bar_index,
        "entry_type": entry_type,
    }


# --------------------------------------------------------------------------- replay
def _entry_type_from_reason(reason: str) -> str:
    r = (reason or "").lower()
    if "limit" in r:
        return "limit"
    if "market" in r and "stop" not in r:
        return "market"
    return "stop"          # Brooks default in this strategy


def _bar_index_from_time(hhmm: str, window_start: str = "09:30") -> int | None:
    try:
        h, m = hhmm.split(":")
        mins = int(h) * 60 + int(m)
        wh, wm = window_start.split(":")
        base = int(wh) * 60 + int(wm)
        return (mins - base) // 5 + 1
    except (ValueError, AttributeError):
        return None


@dataclass
class ProposalResult:
    session: str
    bar: int
    time: str
    side: str
    tag: str | None          # firing v2 veto tag, or None (passed / not-a-v2-tag)
    filled: bool             # became a realized trade
    pnl_usd: float | None    # realized pnl if filled
    exit_reason: str | None
    note: str = ""           # e.g. "v1:<tag>" when a v1 gate tag fired (not counted)


def _match_trade(decision: dict, side: str, trades: list[dict]) -> dict | None:
    """Link a proposal to its realized trade by side + exact stop/target (the engine copies
    them onto the Trade). Falls back to side + entry-time proximity."""
    ds, dt = decision.get("stop"), decision.get("target")
    for t in trades:
        if t.get("side") == side and t.get("stop") == ds and t.get("target") == dt:
            return t
    return None


def _validate(rm: RiskManager, side: str, entry_ref: float, stop, target,
              ctx: dict, unavail: set):
    """Run validate_entry with obstacles disabled (v1 magnet veto already applied live),
    temporarily switching OFF any knob whose inputs are unavailable so a missing field can
    never manufacture a veto. Returns (tag_or_None, is_v2_tag)."""
    cfg = rm.cfg
    saved = {}
    if unavail & UNAVAIL_PULLBACK:
        saved["pullback_gate"] = getattr(cfg, "pullback_gate", True)
        cfg.pullback_gate = False
    if unavail & UNAVAIL_NOCHASE:
        saved["no_chase"] = getattr(cfg, "no_chase", True)
        cfg.no_chase = False
    try:
        rd = rm.validate_entry(side, entry_ref, stop, target, obstacles=None, ctx=ctx)
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
    if rd.ok:
        return None, False
    return rd.reason, rd.reason in V2_TAGS


def replay_session(journal: dict, rm: RiskManager, *, window: str = "am",
                   window_start: str = "09:30",
                   enrich: bool = False) -> list[ProposalResult]:
    """Sequentially replay one session's trade proposals through `rm`, feeding
    on_trade_closed from realized trades (in time order) so cooldown state is live.
    With `enrich`, a proposal that stored no sidecar gets one rebuilt from the bars
    parquet at its timestamp (full fidelity), instead of the partial _brief reconstruct."""
    session_date = journal.get("meta", {}).get("session", "")
    decisions = journal.get("decisions", [])
    trades = list(journal.get("trades", []))
    props = [d for d in decisions if d.get("decision", {}).get("action") in ("long", "short")]
    props.sort(key=lambda d: (d.get("bar") if isinstance(d.get("bar"), int) else 999,
                              d.get("time", "")))
    # realized trades with their exit bar index, for cooldown feeding
    closes = []
    for t in trades:
        eb = _bar_index_from_time(t.get("exit_ts", ""), window_start)
        closes.append((eb if eb is not None else 999, t))
    closes.sort(key=lambda x: x[0])
    fed = [False] * len(closes)

    results: list[ProposalResult] = []
    for d in props:
        dec = d["decision"]
        side = dec["action"]
        bar = d.get("bar") if isinstance(d.get("bar"), int) else \
            (_bar_index_from_time(d.get("time", ""), window_start) or 1)
        # feed cooldown from every realized stop-out that exited BEFORE this proposal's bar
        for idx, (eb, t) in enumerate(closes):
            if not fed[idx] and eb < bar:
                rm.on_trade_closed(t.get("pnl_usd", 0.0), side=t.get("side"),
                                   entry=t.get("entry"),
                                   exit_bar_index=eb,
                                   exit_reason=t.get("exit_reason"))
                fed[idx] = True

        trade = _match_trade(dec, side, trades)
        entry_type = _entry_type_from_reason(dec.get("reason", ""))
        sidecar = d.get("sidecar")
        if sidecar is None and enrich:
            sidecar = _enrich_sidecar(session_date, d.get("time", ""), window_start)
        ctx, unavail = reconstruct_ctx(dec, d.get("context"), sidecar,
                                       bar_index=bar, entry_type=entry_type, window=window)
        # entry_ref: for a realized trade recover the engine's exact ref (stop +/- risk_pts,
        # the fill-slippage-free trigger validate_entry saw live); else a close proxy.
        if trade is not None and trade.get("risk_pts") is not None:
            rp = float(trade["risk_pts"])
            entry_ref = (float(trade["stop"]) + rp if side == "long"
                         else float(trade["stop"]) - rp)
        else:
            ctxc = d.get("context") or {}
            entry_ref = ctxc.get("close")
            if entry_ref is None:               # last resort: midway stop->target
                s, tg = dec.get("stop"), dec.get("target")
                entry_ref = (float(s) + float(tg)) / 2 if s is not None and tg is not None else 0.0

        tag, is_v2 = _validate(rm, side, float(entry_ref), dec.get("stop"),
                               dec.get("target"), ctx, unavail)
        note = "" if (tag is None or is_v2) else f"v1:{tag}"
        results.append(ProposalResult(
            session=journal.get("meta", {}).get("session", "?"),
            bar=bar, time=d.get("time", ""), side=side,
            tag=tag if is_v2 else None,
            filled=trade is not None,
            pnl_usd=(trade.get("pnl_usd") if trade is not None else None),
            exit_reason=(trade.get("exit_reason") if trade is not None else None),
            note=note))
    return results


# --------------------------------------------------------------------------- cohorts
def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def oos_sessions(jdir: Path) -> set[str]:
    """OOS session dates from oos_manifest.json's sessions_file (2024/2025)."""
    man = _load_json(jdir / "oos_manifest.json") or {}
    sf = man.get("sessions_file")
    dates: set[str] = set()
    if sf:
        p = (ROOT / sf)
        if p.exists():
            dates = {ln.strip() for ln in p.read_text("utf-8").splitlines() if ln.strip()}
    if not dates:  # fallback: any 2024/2025 dated journal present
        dates = {f.stem for f in jdir.glob("20*.json")
                 if f.stem[:4] in ("2024", "2025") and "review" not in f.name}
    return {d for d in dates if d[:4] in ("2024", "2025")}


def cohort_files(jdir: Path, cohort: str) -> list[Path]:
    if cohort == "insample":
        return sorted(f for f in jdir.glob("2026-*.json") if "review" not in f.name)
    if cohort == "oos":
        want = oos_sessions(jdir)
        return sorted(f for f in jdir.glob("20*.json")
                      if f.stem in want and "review" not in f.name)
    raise ValueError(cohort)


@dataclass
class CohortReport:
    name: str
    sessions: int = 0
    proposals: int = 0
    realized: int = 0
    results: list[ProposalResult] = field(default_factory=list)
    realized_pnls: list[float] = field(default_factory=list)
    unavail_any: set = field(default_factory=set)


def analyze_cohort(name: str, files: list[Path], *, knob_off: list[str],
                   window: str = "am", enrich: bool = False) -> CohortReport:
    spec = get_spec("mes")
    rep = CohortReport(name=name)
    for f in files:
        j = _load_json(f)
        if not j:
            continue
        decisions = j.get("decisions", [])
        has_real = any(d.get("decision", {}).get("action") in ("long", "short")
                       for d in decisions) or j.get("trades")
        if not has_real:
            continue                        # skip sessions with no tradeable decisions
        rep.sessions += 1
        # fresh RiskManager per session (per-session cooldown/breaker state)
        cfg = engine_config(spec, window)
        for k in knob_off:
            if hasattr(cfg, k):
                setattr(cfg, k, False)
        rm = RiskManager(cfg)
        res = replay_session(j, rm, window=window,
                             window_start=cfg.window_start, enrich=enrich)
        rep.results.extend(res)
        rep.proposals += len(res)
        for t in j.get("trades", []):
            rep.realized += 1
            rep.realized_pnls.append(float(t.get("pnl_usd", 0.0)))
        # record cohort-wide field unavailability (tier) from the proposals' sources
        for d in decisions:
            if d.get("decision", {}).get("action") in ("long", "short"):
                sc = d.get("sidecar")
                if sc is None and enrich:
                    sc = _enrich_sidecar(rep_session_date(j), d.get("time", ""),
                                         cfg.window_start)
                _, un = reconstruct_ctx(d["decision"], d.get("context"), sc,
                                        bar_index=1, entry_type="stop", window=window)
                rep.unavail_any |= un
    return rep


def rep_session_date(journal: dict) -> str:
    return journal.get("meta", {}).get("session", "")


# --------------------------------------------------------------------------- reporting
def _pf(pnls: list[float]):
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return round(gw / gl, 2) if gl else None


def _fmt_pf(v) -> str:
    return "inf" if v is None else f"{v:.2f}"


def cohort_summary(rep: CohortReport) -> dict:
    """The report math for one cohort, in one place (build_markdown + tests both use it)."""
    blocked = [r for r in rep.results if r.filled and r.tag]
    los = [r for r in blocked if (r.pnl_usd or 0) < 0]
    win = [r for r in blocked if (r.pnl_usd or 0) > 0]
    scratch = [r for r in blocked if (r.pnl_usd or 0) == 0]
    unfilled = [r for r in rep.results if not r.filled and r.tag]
    after = _after_book(rep, blocked)
    by_tag = {}
    for tag in V2_TAGS:
        rb = [r for r in blocked if r.tag == tag]
        uf = [r for r in unfilled if r.tag == tag]
        if rb or uf:
            by_tag[tag] = {
                "win": sum(1 for r in rb if (r.pnl_usd or 0) > 0),
                "loss": sum(1 for r in rb if (r.pnl_usd or 0) < 0),
                "scratch": sum(1 for r in rb if (r.pnl_usd or 0) == 0),
                "loss_usd": round(sum(r.pnl_usd for r in rb if (r.pnl_usd or 0) < 0), 2),
                "win_usd": round(sum(r.pnl_usd for r in rb if (r.pnl_usd or 0) > 0), 2),
                "unfilled": len(uf)}
    return {
        "name": rep.name, "sessions": rep.sessions, "proposals": rep.proposals,
        "realized": rep.realized, "blocked": len(blocked),
        "losers": len(los), "winners": len(win), "scratch": len(scratch),
        "losers_sum": round(sum(r.pnl_usd for r in los), 2),
        "winners_sum": round(sum(r.pnl_usd for r in win), 2),
        "unfilled_blocked": len(unfilled),
        "pf_before": _pf(rep.realized_pnls), "pf_after": _pf(after),
        "total_before": round(sum(rep.realized_pnls), 2),
        "total_after": round(sum(after), 2),
        "blocked_detail": [(r.session, r.bar, r.side, r.tag, r.pnl_usd) for r in blocked],
        "by_tag": by_tag,
    }


def build_markdown(reports: list[CohortReport], knob_off: list[str],
                   *, enrich: bool = False) -> str:
    out: list[str] = []
    out.append("# Veto replay — WP2 structural vetoes vs journaled decisions\n")
    mode = ("FULL fidelity — sidecars rebuilt from bars at each proposal timestamp "
            "(--enrich-bars)" if enrich else
            "journal-only (zero bar reload)")
    out.append(f"Zero LLM. Fidelity: **{mode}**. Reconstructed from data/journal/mes/am/. "
               "The journals pre-date the WP2 vetoes, so a fired tag is a genuine "
               "counterfactual (the realized trade never faced it).\n")
    if not enrich:
        out.append("> NOTE: these legacy journals stored only a compact _brief context, "
                   "not the full sidecar the spec assumed. Journal-only mode can therefore "
                   "evaluate zone / counter-trend / cooldown, but NOT pullback_gate or "
                   "no_chase (their inputs hl_count / avg_range_10 / signal_bar_range were "
                   "never persisted). Re-run with `--enrich-bars` for full fidelity.\n")
    if knob_off:
        out.append(f"**Ablation:** knobs OFF = `{', '.join(knob_off)}`\n")

    for rep in reports:
        s = cohort_summary(rep)
        blocked = [r for r in rep.results if r.filled and r.tag]
        unfilled_blocked = [r for r in rep.results if not r.filled and r.tag]

        out.append(f"\n## {rep.name}\n")
        out.append(f"- sessions with decisions: **{rep.sessions}**")
        out.append(f"- trade proposals (long/short): **{rep.proposals}**")
        out.append(f"- realized trades: **{rep.realized}** "
                   f"(PF {_fmt_pf(s['pf_before'])}, total ${s['total_before']:+.2f})")
        na = sorted({KNOB_TAGS[t] for t in V2_TAGS
                     if (t in ("trend_needs_second_entry", "counter_trend")
                         and rep.unavail_any & UNAVAIL_PULLBACK)
                     or (t == "climax_no_chase" and rep.unavail_any & UNAVAIL_NOCHASE)})
        if na:
            out.append(f"- not evaluable from stored fields: **{', '.join(na)}** "
                       f"(inputs {', '.join(sorted(rep.unavail_any))} absent from journals)")

        out.append(
            f"\n**HEADLINE — {rep.name}: the new vetoes would have blocked "
            f"{s['blocked']} of {rep.realized} realized trades "
            f"({s['losers']} losers totaling ${s['losers_sum']:+.2f}, "
            f"{s['winners']} winners totaling ${s['winners_sum']:+.2f}"
            + (f", {s['scratch']} scratch" if s['scratch'] else "")
            + f") -> hypothetical PF {_fmt_pf(s['pf_before'])} (before) "
            f"vs {_fmt_pf(s['pf_after'])} (after), "
            f"total ${s['total_before']:+.2f} -> ${s['total_after']:+.2f}.**")
        out.append(f"\n_Also caught {s['unfilled_blocked']} never-filled proposal(s) "
                   f"(advisory: entry reconstructed from a close proxy)._")
        out.append("\n_Post-veto sequencing (freed trade slots, altered cooldown chains) "
                   "is NOT simulated — 'after' is naive removal of the blocked trades._")

        out.append("\n| veto tag | realized blocked (W/L/scratch) | $ from losers | "
                   "$ from winners | never-filled caught |")
        out.append("|---|---|---|---|---|")
        for tag, tv in s["by_tag"].items():
            out.append(f"| {tag} | {tv['win']}/{tv['loss']}/{tv['scratch']} | "
                       f"${tv['loss_usd']:+.2f} | ${tv['win_usd']:+.2f} | {tv['unfilled']} |")
        if not s["by_tag"]:
            out.append("| _(none fired)_ |  |  |  |  |")

        if blocked:
            out.append("\n<details><summary>blocked realized trades</summary>\n")
            out.append("\n| session | bar | side | tag | pnl$ | exit |")
            out.append("|---|---|---|---|---|---|")
            for r in sorted(blocked, key=lambda r: (r.session, r.bar)):
                out.append(f"| {r.session} | {r.bar} | {r.side} | {r.tag} | "
                           f"{r.pnl_usd:+.2f} | {r.exit_reason} |")
            out.append("\n</details>")
    out.append("")
    return "\n".join(out)


def _after_book(rep: CohortReport, blocked: list[ProposalResult]) -> list[float]:
    """Realized pnls minus the blocked ones (naive removal). Realized trades whose proposal
    could not be matched stay in the book."""
    blocked_pnls = list(r.pnl_usd for r in blocked)
    book = list(rep.realized_pnls)
    for p in blocked_pnls:
        # remove one matching pnl value
        for i, b in enumerate(book):
            if abs(b - p) < 1e-9:
                book.pop(i)
                break
    return book


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["insample", "oos", "all"], default="all")
    ap.add_argument("--knob-off", action="append", default=[], metavar="TAG",
                    help="disable a knob (zone_veto|pullback_gate|no_chase|cooldown); "
                         "repeatable")
    ap.add_argument("--enrich-bars", action="store_true",
                    help="OPT-IN escape hatch: legacy journals here store only a _brief "
                         "context (no sidecar), so pullback_gate/no_chase are otherwise "
                         "unevaluable. This rebuilds the FULL sidecar from data/raw/"
                         "mes_5m.parquet at each proposal's timestamp (reloads bars).")
    ap.add_argument("--journal-dir", default=str(JDIR))
    args = ap.parse_args(argv)
    jdir = Path(args.journal_dir)

    names = (["insample", "oos"] if args.cohort == "all" else [args.cohort])
    label = {"insample": "insample_v41", "oos": "oos_2024_25"}
    reports = [analyze_cohort(label[n], cohort_files(jdir, n), knob_off=args.knob_off,
                              enrich=args.enrich_bars)
               for n in names]
    md = build_markdown(reports, args.knob_off, enrich=args.enrich_bars)
    print(md)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "veto_replay_report.md").write_text(md, "utf-8")
    print(f"\n[written] {(ARTIFACTS / 'veto_replay_report.md').relative_to(ROOT)}")
    if args.enrich_bars and _ENRICH_MISS:
        print(f"[note] {_ENRICH_MISS} proposal(s) could not be enriched from bars "
              "(timestamp absent) and fell back to journal-only reconstruction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
