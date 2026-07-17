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
_DEFAULT_MODEL = {"anthropic": "claude-opus-4-8", "gemini": "gemini-2.5-flash",
                  "zhipu": "glm-4.5-flash"}


# --- decision shape (Pydantic for Gemini response_schema; dict below for Anthropic) ---
class Decision(BaseModel):
    action: Literal["long", "short", "no_trade"]
    setup: str
    entry_type: Literal["stop", "limit", "market"]
    stop: Optional[float] = None
    target: Optional[float] = None
    confidence: int
    reason: str


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


@dataclass
class AgentConfig:
    provider: str = field(default_factory=lambda: os.getenv("PA_PROVIDER", "anthropic"))
    model: Optional[str] = field(default_factory=lambda: os.getenv("PA_MODEL") or None)
    effort: str = field(default_factory=lambda: os.getenv("PA_EFFORT", "medium"))
    vision: bool = field(
        default_factory=lambda: os.getenv("PA_VISION", "0").lower() in ("1", "true", "yes"))
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
    else:
        ok = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        hint = "ANTHROPIC_API_KEY  (https://console.anthropic.com)"
    return {"ok": ok, "provider": cfg.provider, "model": cfg.resolved_model(), "hint": hint}


_ROLE = """You are an Al Brooks price-action trader trading 1 micro contract of MES \
(Micro E-mini S&P 500) on the 5-minute chart. You ENTER only during the first two hours of \
the US regular session (09:30-11:30 ET); an open position is then managed to its stop or \
target and force-closed only at the 16:00 session close (never held overnight)."""

_PERCEPTION_NUMERIC = """You are given a numeric sidecar for the CURRENT bar (no image). \
`session_bars` is the FULL developing session from the 09:30 open to now — each entry has its \
index from the open (i), time (t), OHLC (o/h/l/c), bar type, close position in its range \
(cp, 0=low 1=high), and the 20-EMA value at that bar (e). READ THIS SEQUENCE BAR-BY-BAR the \
way Al Brooks reads a chart: count the swings (H1/H2/L1/L2), identify spikes/channels/ \
trading-ranges, track the Always-In direction, spot pullbacks to the EMA, and locate the \
CURRENT (last) bar in that structure. The other sidecar fields give exact levels for stops/ \
targets ("numbers for the price"). The numbers ARE your chart."""

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
range, or heading straight into a magnet (EMA, prior close, session high/low) closer than \
your first target — those breakouts fail and you buy the top / sell the bottom of the leg.
- DEFAULT to second entries (H2/L2). A first entry (H1/L1) is acceptable ONLY in a strong \
spike or tight always-in trend with consecutive trend bars.

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

Output your decision in the required JSON shape. For no_trade, set stop and target to null. \
For a trade, set stop and target as exact PRICES derived from the sidecar levels, keep risk \
<= 15 points, and make target >= 1R (ideally 1-2R) and realistic for the DAY's likely range \
(it has until the session close to get hit, not just the morning). \
In `reason`, cite the specific bar evidence (bar numbers, bar types, EMA relationship, \
prior-day/gap context) and say WHY the move should follow through.

KNOWLEDGE BASE (Al Brooks method — your own distilled cards):
"""

_SYSTEM_CACHE: dict[bool, str] = {}


def _load_cards() -> str:
    parts = []
    for sub in ("principles", "setups"):
        d = KB / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.md")):
            parts.append(f"### {sub}/{f.stem}\n{f.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def _system(vision: bool = False) -> str:
    if vision not in _SYSTEM_CACHE:
        perception = _PERCEPTION_VISION if vision else _PERCEPTION_NUMERIC
        _SYSTEM_CACHE[vision] = ("\n\n".join([_ROLE, perception, SYSTEM_RULES])
                                 + _load_cards())
    return _SYSTEM_CACHE[vision]


def _user_text(sidecar: dict) -> str:
    from pab.experience import as_prompt_block, read_cases  # closed loop: reviewed lessons
    exp = as_prompt_block(read_cases(k=6))
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
                       "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
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


def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


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
    user = user + "\n\n" + _JSON_KEYS_HINT
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
    resp, out, last = None, None, None
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
            out = _loads_lenient(resp.choices[0].message.content)
            break
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            last = e
            continue
    if out is None:
        raise last
    u = getattr(resp, "usage", None)
    out["_usage"] = {"provider": "zhipu", "model": cfg.resolved_model(),
                     "input": getattr(u, "prompt_tokens", 0) if u else 0,
                     "cache_read": 0, "cache_write": 0,
                     "output": getattr(u, "completion_tokens", 0) if u else 0}
    return out


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
           cfg: Optional[AgentConfig] = None) -> dict:
    """Decision call. Numeric by default; sends the chart image only when cfg.vision
    is set AND an image_path is given. Identical (model, prompts, sidecar) calls are
    served from the on-disk cache unless cfg.cache is off. Returns the decision dict
    (+ `_usage`; cache hits carry `_usage.cached: true`).

    NOTE: the user text embeds the experience block, so new reviewed cases correctly
    invalidate the cache (the prompt really did change)."""
    cfg = cfg or AgentConfig()
    image_bytes = (Path(image_path).read_bytes()
                   if cfg.vision and image_path is not None else None)
    system = _system(cfg.vision)
    user = _user_text(sidecar)

    cache_file = None
    if cfg.cache:
        cache_file = CACHE_DIR / f"{_cache_key(cfg, system, user, image_bytes)}.json"
        if cache_file.exists():
            try:
                out = json.loads(cache_file.read_text("utf-8"))
                out["_usage"] = {**out.get("_usage", {}), "cached": True}
                return out
            except Exception:  # noqa: BLE001 — corrupt entry -> recompute + overwrite
                pass

    if cfg.provider == "gemini":
        out = _decide_gemini(system, image_bytes, user, cfg)
    elif cfg.provider == "zhipu":
        out = _decide_zhipu(system, image_bytes, user, cfg)
    else:
        out = _decide_anthropic(system, image_bytes, user, cfg)

    if cache_file is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
        tmp.replace(cache_file)  # atomic-ish: no torn reads under concurrency
    return out
