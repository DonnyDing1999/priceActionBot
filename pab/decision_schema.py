"""WP4 — Decision JSON v2 semantic validation (pure, no I/O, no LLM).

`validate_decision(d, sidecar)` runs AFTER the model replies and AFTER the raw reply
is cached, so tightening the validator never invalidates the on-disk decision cache
(which stores what the model actually said). It DOWNGRADES a trade to no_trade when a
semantic check fails — it never re-asks the model. A no_trade always passes untouched.

Returns (decision2, terminal_reason | None): terminal_reason is the short validator tag
when a downgrade happened, else None. A downgrade keeps the model's proposal under
"original_decision" and carries "terminal_reason" so journals / the terminal taxonomy
(WP5) see exactly why the trade was dropped. Top-level "_usage" is preserved (the run
loop reads it after decide() for cache-hit accounting); it is stripped from the nested
snapshot to keep journals small.

Defensive to v1 sidecars and pre-v2 cached replies: every key is read with .get(); the
zone echo check only runs when the sidecar carries "zone" (sidecar v2), and an old reply
that predates v2 (no echo / no next_bar) is fine — absent fields are not mismatches.
"""
from __future__ import annotations

import re
from typing import Any

# bar citation, tolerant forms: "bar 5", "bar#5", "bar #5", "bar 5-7", "bars 3-5".
# \b keeps it from matching inside a word (e.g. "rebar 5"). A number must follow.
_BAR_CITE = re.compile(r"\bbars?\s*#?\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)

_MIN_REASON_CHARS = 30
_TRADE = ("long", "short")

# canonical program token -> accepted echo spellings (compared after _norm)
_ALWAYS_IN = {
    "ail": {"ail", "alwaysinlong", "long", "bull", "bullish", "up"},
    "ais": {"ais", "alwaysinshort", "short", "bear", "bearish", "down"},
    "neutral": {"neutral", "none", "flat", "twosided", "balanced", "mixed"},
}
_ZONE = {
    "lower_third": {"lowerthird", "lower", "bottomthird", "bottom"},
    "middle": {"middle", "mid", "middlethird", "center", "centre"},
    "upper_third": {"upperthird", "upper", "topthird", "top"},
}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""


def _echo_matches(program: Any, claimed: Any, table: dict[str, set]) -> bool:
    """True when the model's echoed token names the same program value (tolerant of
    spelling). A blank/unknown claim counts as a MISMATCH so inattention is caught."""
    prog = _norm(program)
    cl = _norm(claimed)
    if not cl:
        return False
    accepted = table.get(str(program), set()) | {prog}
    return cl in accepted


def _clean_next_bar(d: dict) -> None:
    """Drop a malformed next_bar block IN PLACE (never kills the trade). Valid = three
    integer-valued probabilities p_up/p_down/p_neutral summing to 99..101."""
    nb = d.get("next_bar")
    if nb is None:
        return
    ok = False
    if isinstance(nb, dict):
        vals = [nb.get("p_up"), nb.get("p_down"), nb.get("p_neutral")]
        if all(_is_num(v) and float(v).is_integer() for v in vals):
            ok = 99 <= sum(int(v) for v in vals) <= 101
    if not ok:
        d.pop("next_bar", None)


def _downgrade(d: dict, tag: str) -> tuple[dict, str]:
    out = {
        "action": "no_trade",
        "setup": "",
        "entry_type": "market",
        "stop": None,
        "target": None,
        "reason": f"validator: {tag} (original: {d.get('action')}/{d.get('setup', '')})",
        "terminal_reason": tag,
        # small, JSON-serializable snapshot of what the model proposed (minus meta)
        "original_decision": {k: v for k, v in d.items()
                              if k not in ("_usage", "original_decision")},
    }
    if "_usage" in d:  # keep usage at top level for the run loop's cache accounting
        out["_usage"] = d["_usage"]
    return out, tag


def _has_valid_citation(reason: str, sidecar: dict) -> bool:
    """True when `reason` cites >=1 "bar N" with 1 <= N <= current bar index. When the
    sidecar carries no bar index (partial sidecars), any in-form citation counts."""
    bi = sidecar.get("bar_index_from_open")
    hi = int(bi) if _is_num(bi) else None
    for m in _BAR_CITE.finditer(reason or ""):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        lo_n, hi_n = min(a, b), max(a, b)
        if lo_n >= 1 and (hi is None or hi_n <= hi):
            return True
    return False


def validate_decision(d: dict, sidecar: dict) -> tuple[dict, str | None]:
    """Enforce the Decision v2 contract. See module docstring for the return shape."""
    if not isinstance(d, dict):
        return d, None
    if d.get("action") not in _TRADE:
        return d, None  # no_trade / error / unknown: pass untouched

    # --- a live trade proposal from here on -----------------------------------
    # 1. structural: stop/target must be real numbers
    if not (_is_num(d.get("stop")) and _is_num(d.get("target"))):
        return _downgrade(d, "non_numeric")

    # 2. structural: a limit order needs a resting price
    if d.get("entry_type") == "limit" and not _is_num(d.get("entry_px")):
        return _downgrade(d, "limit_no_px")

    # 3. boilerplate guard: reason too thin
    reason = str(d.get("reason", "") or "")
    if len(reason.strip()) < _MIN_REASON_CHARS:
        return _downgrade(d, "reason_too_thin")

    # 4. bar citation required on a trade
    if not _has_valid_citation(reason, sidecar):
        return _downgrade(d, "reason_uncited")

    # 5. echo vs program state — only when the model actually echoed something. An echo
    #    that contradicts the program WITHOUT a matching override is inattention.
    echo = d.get("echo")
    override = d.get("override")
    if isinstance(echo, dict):
        ov_field = _norm(override.get("field")) if isinstance(override, dict) else ""

        def _covers(target: str) -> bool:  # override filed against this echoed field?
            t = _norm(target)
            return bool(ov_field) and (ov_field == t or t in ov_field)

        prog_ai = sidecar.get("always_in_hint")
        if prog_ai and "always_in" in echo and not _covers("always_in") \
                and not _echo_matches(prog_ai, echo.get("always_in"), _ALWAYS_IN):
            return _downgrade(d, "echo_mismatch")

        prog_zone = sidecar.get("zone")  # sidecar v2 only; v1 has no zone -> skip
        if prog_zone and "zone" in echo and not _covers("zone") \
                and not _echo_matches(prog_zone, echo.get("zone"), _ZONE):
            return _downgrade(d, "echo_mismatch")
        # `regime` is the classifier's output, not carried verbatim in the sidecar, so
        # it is not echo-checked here (a legitimate bar-by-bar re-read must not be punished).

    # 6. next_bar sanity: drop a malformed block, keep the trade
    _clean_next_bar(d)
    return d, None
