"""Perception -> decision agent: numeric sidecar (+ optional chart) + Brooks KB -> decision.

Default perception is NUMERIC (no image): the model reads the `session_bars` sequence in the
sidecar bar-by-bar, the way it used to read the rendered chart. Vision was dropped as the
default because per-bar render + image-reasoning is too slow to keep up live; set PA_VISION=1
to re-enable the image path (same prompt + a rendered chart).

Provider-pluggable. Configure via .env / env vars:
    PA_PROVIDER = anthropic | gemini | zhipu   (default anthropic)
    PA_MODEL    = <model id>                    (default claude-opus-4-8 / gemini-2.5-flash / glm-4.5-flash)
    PA_EFFORT   = low|medium|high|xhigh|max     (anthropic only)
    PA_VISION   = 1 to send a rendered chart image too (default 0 = numeric only)
    PA_TEMPERATURE = sampling temperature        (default 0.1 — near-deterministic so
                     backtests are reproducible; applied to gemini/zhipu. Anthropic runs
                     adaptive thinking, which fixes its own sampling — not overridden.)
    PA_CACHE    = 0 to disable the on-disk decision cache (default 1 — identical
                  (model, prompts, sidecar) calls are served from data/cache/decisions/,
                  so engine/risk re-runs don't re-pay the LLM)
Keys (repo-root .env; Claude Code never handles the plaintext):
    ANTHROPIC_API_KEY   (console.anthropic.com)
    GEMINI_API_KEY      (free: aistudio.google.com)
    ZHIPU_API_KEY       (free vision GLM: bigmodel.cn)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge"

try:  # load repo-root .env (user drops the API key there)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

try:  # corp-managed Macs intercept TLS with a keychain-only CA; use the OS trust
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# numeric path is text-only -> zhipu default is a capable TEXT model (glm-4.5-flash),
# not the slower vision model. Override per run with PA_MODEL.
# claude_cli = the local `claude -p` headless CLI (user's Claude Code subscription,
# official print mode — no API key). Local strong-model test channel.
_DEFAULT_MODEL = {"anthropic": "claude-opus-4-8", "gemini": "gemini-2.5-flash",
                  "zhipu": "glm-4.5-flash", "claude_cli": "claude-opus-4-8"}


# --- decision shape (Pydantic for Gemini response_schema; dict below for Anthropic) ---
class Echo(BaseModel):            # Decision v2: restate the program's own read
    regime: Optional[str] = None
    always_in: Optional[str] = None
    zone: Optional[str] = None


class Override(BaseModel):       # Decision v2: an accountable disagreement with the echo
    field: Optional[str] = None
    claim: Optional[str] = None
    reason: Optional[str] = None


class NextBar(BaseModel):        # Decision v2: integer probabilities for the next bar
    p_up: Optional[int] = None
    p_down: Optional[int] = None
    p_neutral: Optional[int] = None


class Decision(BaseModel):
    action: Literal["long", "short", "no_trade"]
    setup: str
    entry_type: Literal["stop", "limit", "market"]
    stop: Optional[float] = None
    target: Optional[float] = None
    confidence: int
    reason: str
    # Decision v2 fields (optional so a v1/legacy reply still validates): the model
    # populates these only when the OUTPUT v2 prompt is active (PA_DECISION_V2 on).
    entry_px: Optional[float] = None
    echo: Optional[Echo] = None
    override: Optional[Override] = None
    next_bar: Optional[NextBar] = None


DECISION_SCHEMA = {  # Anthropic output_config.format (needs additionalProperties:false)
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["long", "short", "no_trade"]},
        "setup": {"type": "string"},
        "entry_type": {"type": "string", "enum": ["stop", "limit", "market"]},
        "stop": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "target": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["action", "setup", "entry_type", "stop", "target", "confidence", "reason"],
    "additionalProperties": False,
}


def _nullable(*types: str) -> dict:
    return {"anyOf": [{"type": t} for t in types] + [{"type": "null"}]}


def _decision_schema(v2: bool) -> dict:
    """Anthropic output_config schema. v1 (default OFF) is byte-identical to
    DECISION_SCHEMA so a legacy request is unchanged; v2 adds the extra fields (strict
    style: every property required + nullable, nested objects additionalProperties:false)
    so the model can actually emit echo/override/next_bar/entry_px under the v2 prompt.
    The schema is NOT part of the decision-cache key, so this never invalidates caches."""
    if not v2:
        return DECISION_SCHEMA
    import copy
    s = copy.deepcopy(DECISION_SCHEMA)
    s["properties"].update({
        "entry_px": _nullable("number"),
        "echo": {"anyOf": [{
            "type": "object",
            "properties": {"regime": _nullable("string"), "always_in": _nullable("string"),
                           "zone": _nullable("string")},
            "required": ["regime", "always_in", "zone"], "additionalProperties": False},
            {"type": "null"}]},
        "override": {"anyOf": [{
            "type": "object",
            "properties": {"field": _nullable("string"), "claim": _nullable("string"),
                           "reason": _nullable("string")},
            "required": ["field", "claim", "reason"], "additionalProperties": False},
            {"type": "null"}]},
        "next_bar": {"anyOf": [{
            "type": "object",
            "properties": {"p_up": _nullable("integer"), "p_down": _nullable("integer"),
                           "p_neutral": _nullable("integer")},
            "required": ["p_up", "p_down", "p_neutral"], "additionalProperties": False},
            {"type": "null"}]},
    })
    s["required"] = s["required"] + ["entry_px", "echo", "override", "next_bar"]
    return s


@dataclass
class AgentConfig:
    provider: str = field(default_factory=lambda: os.getenv("PA_PROVIDER", "anthropic"))
    model: Optional[str] = field(default_factory=lambda: os.getenv("PA_MODEL") or None)
    effort: str = field(default_factory=lambda: os.getenv("PA_EFFORT", "medium"))
    vision: bool = field(
        default_factory=lambda: os.getenv("PA_VISION", "0").lower() in ("1", "true", "yes"))
    instrument: str = field(
        default_factory=lambda: os.getenv("PA_INSTRUMENT", "mes"))
    window: str = field(
        default_factory=lambda: os.getenv("PA_WINDOW", "am"))   # am | pm session window
    temperature: float = field(
        default_factory=lambda: float(os.getenv("PA_TEMPERATURE", "0.1")))
    cache: bool = field(
        default_factory=lambda: os.getenv("PA_CACHE", "1").lower() in ("1", "true", "yes"))
    max_tokens: int = 8000

    def resolved_model(self) -> str:
        return self.model or _DEFAULT_MODEL.get(self.provider, _DEFAULT_MODEL["anthropic"])


def key_status(cfg: Optional[AgentConfig] = None) -> dict:
    cfg = cfg or AgentConfig()
    if cfg.provider == "zhipu":
        ok = bool(os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY"))
        hint = "ZHIPU_API_KEY  (free vision GLM; key at https://bigmodel.cn or https://z.ai)"
    elif cfg.provider == "gemini":
        ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        hint = "GEMINI_API_KEY  (free key: https://aistudio.google.com)"
    elif cfg.provider == "claude_cli":
        import shutil
        ok = shutil.which("claude") is not None
        hint = "the `claude` CLI on PATH, logged in (official headless -p mode)"
    else:
        ok = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        hint = "ANTHROPIC_API_KEY  (https://console.anthropic.com)"
    return {"ok": ok, "provider": cfg.provider, "model": cfg.resolved_model(), "hint": hint}


# role paragraph now lives on the InstrumentSpec (pab.instruments) — MES text is
# byte-identical to the historical prompt so existing decision-cache keys survive

_PERCEPTION_NUMERIC = """You are given a numeric sidecar for the CURRENT bar (no image). \
`session_table` is the FULL developing session from the 09:30 open to now, one row per bar: \
i|time|open|high|low|close|type|cp|ema  (cp = close position in the bar's range, 0=low \
1=high; ema = the 20-EMA at that bar). READ THIS TABLE BAR-BY-BAR the way Al Brooks reads \
a chart: count the swings (H1/H2/L1/L2), identify spikes/channels/trading-ranges, track \
the Always-In direction, spot pullbacks to the EMA, and locate the CURRENT (last row) bar \
in that structure. Pre-computed helpers: `swings` lists the confirmed pivot points with \
structure labels (HH/HL = uptrend skeleton, LH/LL = downtrend skeleton); \
`last_bars_pattern` flags ii/inside/outside combos on the latest bars; `chop` quantifies \
the session's texture. Read `chop` TOGETHER with always_in: when always_in is NEUTRAL and \
overlap is high / efficiency low / bodies small, it is a two-sided trading range — default \
to no_trade and never stop-enter at its extremes. But when always_in has been clearly one \
side for several bars, texture alone is NOT a veto: trend days often grind with overlapping \
bars, and the with-trend pullback entry is still your trade. \
`magnets` lists the key magnet LEVELS — yesterday's high/low/close and the overnight \
high/low — each with its signed distance from the current close (positive = above). The \
numbers ARE your chart."""

_PERCEPTION_VISION = """You are given, for the CURRENT bar: (1) a rendered 5-minute \
candlestick chart with a 20-EMA, the prior-day close line, an open marker, and bar numbers \
counted from the 09:30 open; (2) a numeric sidecar with exact prices. Use the IMAGE for \
structure/setup ("vision for the setup") and the SIDECAR for precise levels ("numbers for \
the price"). Read the chart bar-by-bar like Al Brooks."""

SYSTEM_RULES = """Decide ONE of: long, short, or no_trade for THIS bar, plus how to enter \
(`entry_type`). Prefer "stop" — the Brooks entry: a stop order 1 tick beyond THIS bar's \
extreme (above its high for long, below its low for short), working for the next bar only; \
if the next bar never triggers it, the order is canceled and no trade happens (a failed \
signal filters itself out). Use "market" only when immediate entry at the next bar's open \
is clearly right (e.g. strong always-in momentum already under way).

BE SELECTIVE — this is the single biggest edge you control. A good session has 0-2 trades; \
trading no bar all morning is often the CORRECT day. Only propose a trade when a knowledge-base \
setup, the signal bar, and the context all agree. Checklist before any entry:
- Signal bar quality: a trend bar closing near its extreme IN your direction. Never off a doji.
- FOLLOW-THROUGH case (critical for stop entries): a stop entry buys above the signal bar / \
sells below it, so the move must CONTINUE to pay you. Ask: is this a with-trend move with room, \
or a breakout into resistance? Do NOT stop-enter when price is mid-range, inside a trading \
range, or heading straight into a magnet closer than your first target — those breakouts fail \
and you buy the top / sell the bottom of the leg. CHECK the `magnets` list (yday high/low/ \
close, overnight high/low, plus the session high/low and EMA): if one sits between your \
entry and your target, either target the near side of it or pass.
- RANGE FADES (double top/bottom, TR fade): enter NEAR the extreme you are fading — short \
close to the high on the lower second top, buy close to the low — with the stop ONE TICK \
beyond the FARTHER of the two extremes (above the higher of two tops, below the lower of \
two bottoms), NEVER between them (a stop wedged between the tops gets tagged on any retest \
of the true high). And never express a range fade as a stop entry through the OPPOSITE \
side of the range: breaking the range low to "confirm" a double top is a breakout bet on \
the ~80%-fail side.
- DEFAULT to second entries (H2/L2). A first entry (H1/L1) is acceptable ONLY in a strong \
spike or tight always-in trend with consecutive trend bars.
- TREND-DAY PARTICIPATION: when always_in has been one-sided for several bars, your job \
flips from finding objections to finding THE entry — a pullback holding at/near the rising \
(falling) EMA followed by a with-trend signal bar IS the H2/L2. On such a day aim to take \
exactly ONE good with-trend entry rather than zero; skipping every bar of a trend day is a \
process error, not caution.

HARD RULES (a separate code risk-layer also enforces these; a proposal that violates them is \
discarded, wasting the call):
- Trade only setups in your knowledge base. When in doubt, no_trade — most bars are no_trade.
- Per-trade risk is capped at 15 MES points ($75). If a valid stop would be wider than 15 \
  points, you MUST return no_trade (do not propose a wider stop).
- Never trade counter to the Always-In direction unless a full MTR has completed (closing \
  trendline break AND a failed test of the prior extreme).
- Never chase: no entry on a climax/oversized bar; wait for a pullback or a second entry.
- Geometry MUST hold around the TRIGGER price, not the close. For entry_type "stop" the \
  trigger is 1 tick beyond THIS bar's extreme — e.g. long stop entry off a bar with high \
  7600.00 triggers at 7600.25, so stop < 7600.25 < target, and risk = 7600.25 - stop. \
  For "market" treat entry as roughly the current bar's close.
- No new signals after bar 19 — entries belong to the opening structure, which is your edge \
  (the code gate enforces this cutoff). An OPEN position is different: it keeps working to \
  its stop or target through the afternoon and is only force-closed at the 16:00 session \
  close, so you are never squeezed out by the clock at 11:30.
- SELF-CHECK before returning any trade (a failed check wastes the whole call): compute \
  your trigger price (this bar's extreme +/- 1 tick for a stop entry, ~the close for \
  market), then verify ALL of: (a) long -> stop < trigger < target, short -> \
  target < trigger < stop; (b) risk = |trigger - stop| is between 2 and 15 points; \
  (c) |target - trigger| >= risk. If any check fails, FIX the numbers or return no_trade.

Output your decision in the required JSON shape. For no_trade, set stop and target to null. \
For a trade, set stop and target as exact PRICES derived from the sidecar levels, keep risk \
<= 15 points, and make target >= 1R (ideally 1-2R) and realistic for the DAY's likely range \
(it has until the session close to get hit, not just the morning). \
In `reason`, cite the specific bar evidence (bar numbers, bar types, EMA relationship, \
prior-day/gap context) and say WHY the move should follow through.

KNOWLEDGE BASE (Al Brooks method — your own distilled cards):
"""

# Decision v2 output contract (PA_DECISION_V2, default on). Spliced into the system
# prompt just BEFORE the KNOWLEDGE BASE trailer by _system(); the SYSTEM_RULES prose
# above is left byte-identical so the legacy (v2-off) prompt — and its decision-cache
# keys — are preserved for the v4.1 replay guard.
_OUTPUT_V2 = """OUTPUT v2 — these additional fields are REQUIRED on every decision; they \
make your read auditable, and a program disagreement you fail to flag gets your trade \
discarded:
- echo {regime, always_in, zone}: restate the PROGRAM's own read, copied verbatim from \
the inputs — `regime` from the DIAGNOSIS HINT below (open|trend|range; if no hint is \
attached, use your own read), `always_in` from the sidecar `always_in_hint` \
(ail|ais|neutral), and `zone` from the sidecar `zone` (lower_third|middle|upper_third; \
omit the zone key when the sidecar has none). Copy the tokens EXACTLY — this proves you \
read the program state.
- override {field, claim, reason} or null: file this ONLY when your bar-by-bar read \
disagrees with a value you just echoed. `field` = the disputed field name (e.g. "zone"), \
`claim` = your value, `reason` = the concrete bar evidence for disagreeing. An echo that \
contradicts the program WITHOUT a matching override is treated as inattention and the \
trade is dropped — so either echo the program value, or echo it AND override it.
- entry_px (number; REQUIRED when entry_type is "limit", null otherwise): the resting \
limit price. A LIMIT resting at a structural pullback price — a touch of the 20-EMA, a \
prior swing point, or the edge of the range — is the PREFERRED entry for trend pullbacks \
and range fades: you buy the pullback low / sell the pullback high instead of chasing. \
Choose a "stop" entry ONLY when you can state an explicit follow-through case (a breakout \
whose momentum must continue to pay). A "limit" with no entry_px is discarded.
- next_bar {p_up, p_down, p_neutral}: your honest INTEGER probabilities for the next \
bar's direction, summing to ~100 (99-101). This is a forecast only; it does not by itself \
gate the trade."""


# ---- two-stage diagnose -> route: a deterministic regime classifier (pab.features.
# classify_regime, zero LLM cost) picks WHICH setup cards enter the prompt. Core
# principles are always included; the full setup + principles INDEXES are always named in
# the note so the model knows what exists even when a card is not attached. Routing is
# provider-aware (PA_ROUTE, see _route_mode): 'auto' (default) gives strong providers the
# full KB + a naming-only note and the free-tier providers the compact regime subset; =1
# forces the subset for all; =0 restores the byte-identical full KB with no note.
CORE_PRINCIPLES = [
    "market-cycle-model", "always-in", "traders-equation", "signal-bar-quality",
    "second-entry-higher-probability", "bar-by-bar-reading", "bar-types-and-signal-bars",
    # universal doctrine — structural stop placement + respect-the-veto — always core:
    "stop-placement-and-management", "no-trade-environments",
]
REGIME_PRINCIPLES = {
    "open": ["high-low-of-day-forms-early", "opening-breakout-50pct-reversal"],
    "trend": ["measured-move"],
    "range": [],
}
REGIME_SETUPS = {
    "open": ["opening-breakout-mode", "opening-18bar-range-breakout",
             "opening-trend-resumption", "failed-breakout-yesterday-2nd-entry",
             "spike-and-channel", "breakout-failure-and-test"],
    "trend": ["channel-pullback-h1h2-l1l2", "spike-and-channel", "micro-channel",
              "opening-trend-resumption", "twenty-gap-bar", "wedge", "final-flag",
              "major-trend-reversal"],
    "range": ["trading-range-fade", "tight-range-fade", "triangle",
              "breakout-failure-and-test", "double-top-bottom",
              "magnet-and-trapped-traders", "failed-breakout-yesterday-2nd-entry"],
}

_CARDS_CACHE: dict[str, dict[str, str]] = {}
_SYSTEM_CACHE: dict[tuple, str] = {}


def _cards(sub: str) -> dict[str, str]:
    if sub not in _CARDS_CACHE:
        d = KB / sub
        _CARDS_CACHE[sub] = {f.stem: f.read_text(encoding="utf-8")
                             for f in sorted(d.glob("*.md"))} if d.exists() else {}
    return _CARDS_CACHE[sub]


def _route_mode() -> str:
    """PA_ROUTE routing mode (default 'auto' when the env var is unset):
      auto   — strong providers (anthropic, claude_cli) get the FULL KB + a full-KB
               diagnosis note; zhipu/gemini get the compact regime SUBSET + note.
      subset — PA_ROUTE=1: force the regime subset for ALL providers (the pre-auto default).
      full   — PA_ROUTE=0: force the full KB with NO note (byte-identical legacy path; the
               v4.1 replay guard depends on these exact bytes and their decision-cache keys).
    Legacy truthy/falsey spellings still map: 1/true/yes/on -> subset, 0/false/no/off -> full.
    """
    v = os.getenv("PA_ROUTE", "auto").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "subset"
    if v in ("0", "false", "no", "off"):
        return "full"
    return "auto"


_FULL_KB_PROVIDERS = ("anthropic", "claude_cli")   # strong models: full KB + diagnosis note


def _kb_style(provider: Optional[str]) -> str:
    """KB routing style for `provider` under the current PA_ROUTE mode. This is the ONLY
    provider-dependent input to the system prompt, so it doubles as the cache-key discriminator:
      'subset'    — regime subset of setup cards + the subset DIAGNOSIS HINT note.
      'full_note' — full KB + a full-KB DIAGNOSIS HINT (auto mode, strong providers).
      'full'      — full KB, no note (the byte-identical legacy PA_ROUTE=0 path).
    provider=None (unspecified: tools/tests) is treated as a non-strong provider -> subset."""
    mode = _route_mode()
    if mode == "full":
        return "full"
    if mode == "subset":
        return "subset"
    return "full_note" if provider in _FULL_KB_PROVIDERS else "subset"


def _decision_v2_enabled() -> bool:
    # Decision contract v2 (OUTPUT v2 prompt + validate_decision). Default ON; set
    # PA_DECISION_V2=0 to fall back to the legacy prompt + raw passthrough (byte-identical
    # system text and decision-cache keys — the v4.1 replay guard turns it off).
    return os.getenv("PA_DECISION_V2", "1").lower() in ("1", "true", "yes")


def _kb_block(regime: Optional[str], style: str = "full") -> str:
    prin, setups = _cards("principles"), _cards("setups")
    # Name-level breadcrumb for cards that may not be physically attached: the full, sorted
    # setup AND principles indexes. Every note that lists the setup index also lists the
    # principles index, so unselected principles still leave a name in the prompt.
    idx_s, idx_p = ", ".join(sorted(setups)), ", ".join(sorted(prin))
    route_ok = regime is not None and regime in REGIME_SETUPS
    if not route_ok or style == "full":
        # Full KB, NO note: smoke/review, PA_ROUTE=0, and the no-regime fallback (partial
        # sidecars). Byte-identical to the historical full-KB prompt -> legacy caches survive.
        names_p, names_s = list(prin), list(setups)
        note = ""
    elif style == "full_note":
        # auto mode, strong providers: every card is attached, so the note only NAMES the
        # classifier's read (no "subset relevant to that regime" sentence) + both indexes.
        names_p, names_s = list(prin), list(setups)
        note = (f"\nDIAGNOSIS HINT: a code pre-classifier reads this bar's regime as "
                f"'{regime}' (open|trend|range); all cards are attached (full knowledge "
                f"base). Full setup index: {idx_s}. Full principles index: {idx_p}. Trust "
                f"your own bar-by-bar read if it differs, but only propose setups from your "
                f"knowledge base.\n")
    else:  # "subset": only the regime-relevant setup cards + the subset note
        names_p = [n for n in CORE_PRINCIPLES if n in prin] \
            + [n for n in REGIME_PRINCIPLES[regime] if n in prin]
        names_s = [n for n in REGIME_SETUPS[regime] if n in setups]
        note = (f"\nDIAGNOSIS HINT: a code pre-classifier reads this bar's regime as "
                f"'{regime}' (open|trend|range). The setup cards below are the subset "
                f"relevant to that regime; core principles are always included. Full "
                f"setup index (exists in your KB even when not attached): " + idx_s
                + f". Full principles index: {idx_p}. Trust your own bar-by-bar read if "
                "it differs, but only propose setups from your knowledge base.\n")
        # Finer subdivision (e.g. trend -> spike / channel / micro-channel sub-modes) is
        # deliberately NOT done here: the sidecar that would drive it is not visible in this
        # string-only routing layer. That granularity rides on the DIAGNOSIS HINT (the
        # classifier's regime read) plus the full setup/principles indexes just above, which
        # name every card the model may legitimately reach for.
    parts = [note] if note else []
    parts += [f"### principles/{n}\n{prin[n]}" for n in names_p]
    parts += [f"### setups/{n}\n{setups[n]}" for n in names_s]
    return "\n\n".join(parts)


_PM_ADDENDUM = """TODAY YOU ARE TRADING THE AFTERNOON WINDOW: entries only 13:30-15:30 ET \
(bar 1 = 13:30; the same bar-19 entry cutoff applies, = 15:00). The MORNING session already \
happened — its high/low/close arrive as `magnets` (am_high / am_low / am_close) and act as \
the day's dominant magnets: the afternoon usually either breaks out of the morning range \
with follow-through or trades back inside it. Exits still force-flat at 15:55, so runway is \
SHORT — prefer nearer targets (~1R) and be extra reluctant to enter after bar 15."""


def _system(vision: bool = False, regime: Optional[str] = None,
            instrument: str = "mes", window: str = "am",
            provider: Optional[str] = None) -> str:
    """Assemble the system prompt. In auto mode (PA_ROUTE unset) `provider` picks the KB
    routing style — anthropic/claude_cli get the full KB + a naming-only note, zhipu/gemini
    get the regime subset + note; provider=None (tools/tests) routes as a non-strong
    provider (subset). PA_ROUTE=1/0 ignore provider (subset-for-all / full-no-note). decide()
    and scripts/backtest_cached both pass cfg.provider, so their decision-cache keys agree."""
    from pab.instruments import get_spec
    v2 = _decision_v2_enabled()
    style = _kb_style(provider)
    if regime is None or regime not in REGIME_SETUPS:
        style = "full"          # no regime to route on / name -> full KB, no note (as today)
    key_regime = regime if style != "full" else None
    # The KB block is the ONLY provider-dependent part of the prompt, and only in auto mode;
    # so the provider joins the cache key there — as `style`, which collapses providers that
    # route alike onto one slot (shared where outputs match, distinct where they differ). In
    # subset/full modes the key stays the historical 5-tuple, leaving PA_ROUTE=0 (and =1)
    # keys byte-for-byte identical to before this change.
    key = ((vision, key_regime, instrument, window, v2, style)
           if _route_mode() == "auto" else
           (vision, key_regime, instrument, window, v2))
    if key not in _SYSTEM_CACHE:
        perception = _PERCEPTION_VISION if vision else _PERCEPTION_NUMERIC
        rules = SYSTEM_RULES
        if v2:  # splice OUTPUT v2 in just before the KB trailer; prose stays untouched
            m = "KNOWLEDGE BASE (Al Brooks method"
            i = rules.find(m)
            rules = (rules[:i] + _OUTPUT_V2 + "\n\n" + rules[i:]) if i != -1 \
                else (rules + "\n\n" + _OUTPUT_V2)
        parts = [get_spec(instrument).role_text]
        if window == "pm":
            parts.append(_PM_ADDENDUM)
        parts += [perception, rules]
        _SYSTEM_CACHE[key] = "\n\n".join(parts) + _kb_block(key_regime, style)
    return _SYSTEM_CACHE[key]


# coarse routing regime -> the review agent's finer regime enum (experience retrieval)
_REGIME_FAMILY = {
    "trend": ("spike", "micro_channel", "tight_channel", "normal_channel",
              "broad_channel"),
    "range": ("trading_range", "extreme_tr"),
    "open": None,   # opening bars: any recent lesson is welcome
}


def _user_text(sidecar: dict, regime: Optional[str] = None,
               cases: Optional[list] = None) -> str:
    from pab.experience import (as_prompt_block, read_cases,  # closed loop: lessons
                                select_cases)
    fam = _REGIME_FAMILY.get(regime)
    before = sidecar.get("session_date")
    # `cases` given = a run's frozen snapshot (reproducible); None = read live from disk.
    # select_cases(load_all_cases(), ...) == read_cases(...), so the text is byte-identical.
    cs = (select_cases(cases, regime=fam, k=6, before=before) if cases is not None
          else read_cases(regime=fam, k=6, before=before))
    exp = as_prompt_block(cs)
    prefix = (exp + "\n\n") if exp else ""
    return (prefix + "Numeric sidecar (exact levels, no-lookahead):\n```json\n"
            + json.dumps(sidecar, ensure_ascii=False) + "\n```\n\n"
            "Read the developing session bar-by-bar and decide for this bar now.")


def _decide_anthropic(system: str, image_bytes: Optional[bytes], user: str,
                      cfg: AgentConfig) -> dict:
    # NOTE: adaptive thinking manages its own sampling — cfg.temperature not applied here.
    import anthropic
    client = anthropic.Anthropic()
    content: list = []
    if image_bytes is not None:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    content.append({"type": "text", "text": user})
    resp = client.messages.create(
        model=cfg.resolved_model(),
        max_tokens=cfg.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": cfg.effort,
                       "format": {"type": "json_schema",
                                  "schema": _decision_schema(_decision_v2_enabled())}},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    u = resp.usage
    out["_usage"] = {"provider": "anthropic", "model": cfg.resolved_model(),
                     "input": u.input_tokens,
                     "cache_read": getattr(u, "cache_read_input_tokens", 0),
                     "cache_write": getattr(u, "cache_creation_input_tokens", 0),
                     "output": u.output_tokens}
    return out


def _decide_gemini(system: str, image_bytes: Optional[bytes], user: str,
                   cfg: AgentConfig) -> dict:
    import time
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    contents: list = []
    if image_bytes is not None:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    contents.append(user)
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=Decision,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens,
    )
    resp, last = None, None
    for attempt in range(3):  # simple backoff for free-tier 429s
        try:
            resp = client.models.generate_content(
                model=cfg.resolved_model(), contents=contents, config=config)
            break
        except Exception as e:  # noqa: BLE001
            last = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if resp is None:
        raise last  # type: ignore[misc]

    out = json.loads(resp.text)
    um = getattr(resp, "usage_metadata", None)
    out["_usage"] = {"provider": "gemini", "model": cfg.resolved_model(),
                     "input": getattr(um, "prompt_token_count", 0) if um else 0,
                     "cache_read": (getattr(um, "cached_content_token_count", 0) or 0) if um else 0,
                     "cache_write": 0,
                     "output": getattr(um, "candidates_token_count", 0) if um else 0}
    return out


_JSON_KEYS_HINT = (
    "Return ONLY a JSON object with EXACTLY these keys (no extra text, no markdown fences):\n"
    '{"action": "long|short|no_trade", "setup": "<name or none>", '
    '"entry_type": "stop|limit|market", "stop": <price or null>, '
    '"target": <price or null>, "confidence": <integer 0-100>, "reason": "<brief>"}'
)

_JSON_KEYS_HINT_V2 = (
    "Return ONLY a JSON object with EXACTLY these keys (no extra text, no markdown fences):\n"
    '{"action": "long|short|no_trade", "setup": "<name or none>", '
    '"entry_type": "stop|limit|market", "stop": <price or null>, '
    '"target": <price or null>, '
    '"entry_px": <price when entry_type is "limit", else null>, '
    '"confidence": <integer 0-100>, '
    '"reason": "<cite bar N evidence + why it follows through>", '
    '"echo": {"regime": "open|trend|range", "always_in": "ail|ais|neutral", '
    '"zone": "lower_third|middle|upper_third"}, '
    '"override": {"field": "<name>", "claim": "<value>", "reason": "<why>"} '
    'or null, '
    '"next_bar": {"p_up": <int>, "p_down": <int>, "p_neutral": <int>}}'
)


def _json_keys_hint() -> str:
    return _JSON_KEYS_HINT_V2 if _decision_v2_enabled() else _JSON_KEYS_HINT


def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def _reask_toward_no_trade(first: Optional[dict], latest: dict) -> dict:
    """Retry-direction guard (Decision v2): a JSON re-ask may only move `action` TOWARD
    no_trade. If an earlier attempt already PARSED a decision (`first`), a later re-ask
    that flips no_trade->trade or flips the trade side is clamped to no_trade
    (terminal_reason 'retry_flip'). first=None (the earlier attempt never parsed at all)
    => `latest` is returned as-is: a trade after a fully failed parse is allowed.

    The current provider loops break on the FIRST successful parse, so in production this
    only ever sees first=None (a no-op). It formalizes the invariant — and clamps — if a
    loop is ever changed to re-ask after a parse."""
    if not isinstance(first, dict):
        return latest
    fa, la = first.get("action"), latest.get("action")
    flip = la in ("long", "short") and (
        fa == "no_trade" or (fa in ("long", "short") and fa != la))
    if not flip:
        return latest
    out = {"action": "no_trade", "setup": "", "entry_type": "market",
           "stop": None, "target": None,
           "reason": f"retry_flip: re-ask changed {fa} -> {la}",
           "terminal_reason": "retry_flip",
           "original_decision": {k: v for k, v in latest.items()
                                 if k not in ("_usage", "original_decision")}}
    if "_usage" in latest:
        out["_usage"] = latest["_usage"]
    return out


_zhipu_gate = threading.Lock()
_zhipu_next_ok = 0.0  # monotonic time when the next call may LAUNCH (slot reservation)


def _zhipu_throttle(min_interval: float) -> None:
    """Thread-safe launch throttle: reserve the next launch slot under a lock, sleep
    outside it. Concurrent sessions space their request STARTS >= min_interval apart
    while their (long) response waits overlap."""
    import time
    global _zhipu_next_ok
    with _zhipu_gate:
        now = time.monotonic()
        wait = _zhipu_next_ok - now
        _zhipu_next_ok = max(now, _zhipu_next_ok) + min_interval
    if wait > 0:
        time.sleep(wait)


def _decide_zhipu(system: str, image_bytes: Optional[bytes], user: str,
                  cfg: AgentConfig) -> dict:
    import time
    from openai import OpenAI  # Zhipu's v4 API is OpenAI-compatible
    key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    base = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    min_interval = float(os.getenv("ZHIPU_MIN_INTERVAL", "3"))
    client = OpenAI(api_key=key, base_url=base, max_retries=0,
                    timeout=float(os.getenv("ZHIPU_TIMEOUT", "180")))
    user = user + "\n\n" + _json_keys_hint()
    content: list = []
    if image_bytes is not None:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": user})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    resp, out, last, first_parsed = None, None, None, None
    for delay in (0, 10, 20, 40, 80):  # long tail: free tier has whole overloaded MINUTES
        _zhipu_throttle(min_interval)
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=cfg.resolved_model(), max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=messages, response_format={"type": "json_object"})
        except Exception as e:  # noqa: BLE001
            last = e
            etype = type(e).__name__
            transient = (
                any(t in str(e) for t in ("429", "1305", "访问量", "rate", "overload", "Too Many"))
                or etype in ("APITimeoutError", "APIConnectionError", "APIStatusError",
                             "InternalServerError", "APIError"))
            if transient:  # rate-limit OR flaky-network -> back off and retry
                continue
            raise
        try:  # malformed JSON (top free-tier failure) is transient too — retry, don't drop the bar
            parsed = _loads_lenient(resp.choices[0].message.content)
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            last = e
            continue
        # retry-direction guard: a re-ask may only move action toward no_trade
        out = parsed if first_parsed is None else _reask_toward_no_trade(first_parsed, parsed)
        first_parsed = first_parsed or parsed
        break
    if out is None:
        raise last
    u = getattr(resp, "usage", None)
    out["_usage"] = {"provider": "zhipu", "model": cfg.resolved_model(),
                     "input": getattr(u, "prompt_tokens", 0) if u else 0,
                     "cache_read": 0, "cache_write": 0,
                     "output": getattr(u, "completion_tokens", 0) if u else 0}
    return out


def _decide_claude_cli(system: str, image_bytes: Optional[bytes], user: str,
                       cfg: AgentConfig) -> dict:
    """Local `claude -p` headless call (user's Claude Code login — official print
    mode, no API key). Numeric path only (no image). Notes: temperature is not
    controllable through the CLI, and the prompt rides on top of Claude Code's own
    system prompt (--append-system-prompt) — a strong-model TEST channel, not the
    airtight primary eval transport."""
    import subprocess
    prompt = user + "\n\n" + _json_keys_hint()
    cmd = ["claude", "--model", cfg.resolved_model(), "-p",
           "--output-format", "json", "--max-turns", "1",
           "--append-system-prompt", system]
    # clean env: a parent Claude Code session leaks ANTHROPIC_BASE_URL / CLAUDE_*
    # vars that break the child CLI's own auth resolution
    clean_env = {k: os.environ[k] for k in ("HOME", "PATH", "USER", "TERM", "SHELL")
                 if k in os.environ}
    last: Exception | None = None
    first_parsed: Optional[dict] = None
    for _ in range(2):                        # one retry on malformed output
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=300, env=clean_env)
        if r.returncode != 0:
            last = RuntimeError(f"claude -p exit {r.returncode}: {r.stderr[:160]}")
            continue
        try:
            wrap = json.loads(r.stdout)
            parsed = _loads_lenient(wrap.get("result", ""))
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        # retry-direction guard: a re-ask may only move action toward no_trade
        out = parsed if first_parsed is None else _reask_toward_no_trade(first_parsed, parsed)
        first_parsed = first_parsed or parsed
        out["_usage"] = {"provider": "claude_cli", "model": cfg.resolved_model(),
                         "input": (wrap.get("usage") or {}).get("input_tokens", 0),
                         "cache_read": (wrap.get("usage") or {}).get(
                             "cache_read_input_tokens", 0),
                         "cache_write": 0,
                         "output": (wrap.get("usage") or {}).get("output_tokens", 0),
                         "cost_usd": wrap.get("total_cost_usd")}
        return out
    raise last if last else RuntimeError("claude -p failed")


CACHE_DIR = ROOT / "data" / "cache" / "decisions"


def _cache_key(cfg: AgentConfig, system: str, user: str,
               image_bytes: Optional[bytes]) -> str:
    ident = {"provider": cfg.provider, "model": cfg.resolved_model(),
             "effort": cfg.effort, "temperature": cfg.temperature,
             "vision": cfg.vision, "system": system, "user": user,
             "image": hashlib.sha256(image_bytes).hexdigest() if image_bytes else None}
    return hashlib.sha256(
        json.dumps(ident, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def decide(sidecar: dict, image_path: str | Path | None = None, *,
           cfg: Optional[AgentConfig] = None, cases: Optional[list] = None) -> dict:
    """Decision call. Numeric by default; sends the chart image only when cfg.vision
    is set AND an image_path is given. Identical (model, prompts, sidecar) calls are
    served from the on-disk cache unless cfg.cache is off. Returns the decision dict
    (+ `_usage`; cache hits carry `_usage.cached: true`).

    NOTE: the user text embeds the experience block, so new reviewed cases correctly
    invalidate the cache (the prompt really did change).

    The cache stores the RAW model output (what the model said). Decision v2 validation
    (validate_decision, PA_DECISION_V2 on) runs on the POST-cache dict EVERY call, so
    tightening the validator never invalidates the on-disk cache; a semantic failure is
    downgraded to no_trade in the returned dict (carrying `terminal_reason`)."""
    cfg = cfg or AgentConfig()
    image_bytes = (Path(image_path).read_bytes()
                   if cfg.vision and image_path is not None else None)
    try:  # stage 1 of diagnose->route: free deterministic regime classifier
        from pab.features import classify_regime
        regime = classify_regime(sidecar, cfg.window)
    except Exception:  # noqa: BLE001 — partial sidecars (tools/tests) -> full KB
        regime = None
    system = _system(cfg.vision, regime, cfg.instrument, cfg.window, cfg.provider)
    user = _user_text(sidecar, regime, cases=cases)

    out = None
    cache_file = None
    if cfg.cache:
        cache_file = CACHE_DIR / f"{_cache_key(cfg, system, user, image_bytes)}.json"
        if cache_file.exists():
            try:
                out = json.loads(cache_file.read_text("utf-8"))
                out["_usage"] = {**out.get("_usage", {}), "cached": True}
            except Exception:  # noqa: BLE001 — corrupt entry -> recompute + overwrite
                out = None

    if out is None:  # cache miss / disabled / corrupt -> ask the provider, cache the RAW
        if cfg.provider == "gemini":
            out = _decide_gemini(system, image_bytes, user, cfg)
        elif cfg.provider == "zhipu":
            out = _decide_zhipu(system, image_bytes, user, cfg)
        elif cfg.provider == "claude_cli":
            out = _decide_claude_cli(system, image_bytes, user, cfg)
        else:
            out = _decide_anthropic(system, image_bytes, user, cfg)
        if cache_file is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
            tmp.replace(cache_file)  # atomic-ish: no torn reads under concurrency

    if _decision_v2_enabled():  # post-cache semantic validation (never touches the cache)
        from pab.decision_schema import validate_decision
        out, _terminal = validate_decision(out, sidecar)
    return out
