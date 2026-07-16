"""Daily-review / reflection agent (system agent #4 of the swarm).

Reads a session journal (decisions + trades), asks an LLM to reflect with strict
process-vs-outcome discipline, and promotes durable lessons into the experience library.
Text-only LLM call via the same provider config (PA_PROVIDER / PA_MODEL) — no image needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pab.agent import AgentConfig, _loads_lenient
from pab.experience import add_cases

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / "data" / "journal"

REVIEW_SYSTEM = """You are a trading coach reviewing ONE day of an Al Brooks price-action \
bot (MES, opening window 09:30-11:30 ET). You get the day's decision journal (each flat bar: \
what it decided + why + context) and the trades that resulted (entry/exit/pnl/R/exit_reason).

Extract DURABLE lessons with STRICT process-vs-outcome discipline:
- A trade that FOLLOWED the rules but lost is OUTCOME NOISE — do NOT punish it or invent a \
rule from it. Losing trades are normal; a good process still loses ~40-60% of the time.
- A trade that BROKE a rule is a PROCESS ERROR — flag it (rules: only KB setups; per-trade \
risk <=15 pts; R:R>=1; trade WITH always_in unless a completed MTR; prefer 2nd entries; \
don't chase a climax/oversized bar; no new trade after ~bar 20; avoid the middle of a range).
- Look for PATTERNS across the day, not one-off noise. This is a tiny sample — most lessons \
are tentative. If the day is just noise, say so and return few/no lessons.

Output ONLY JSON (no prose, no fences):
{"day_summary": "<1-2 sentences: kind of day + how the bot did>",
 "process_errors": ["<rule broken + which bar/trade>", ...],
 "outcome_notes": ["<rule-following wins/losses that are just noise>", ...],
 "lessons": ["<tentative, generalizable — ONLY if a real pattern>", ...],
 "experience_cases": [{"regime": "spike|micro_channel|tight_channel|normal_channel|broad_channel|trading_range|extreme_tr|unknown", "outcome": "success|failure", "setup": "<name>", "note": "<what to remember for similar setups next time>"}]}"""


def _build_user(journal: dict) -> str:
    m, trades, decs = journal["meta"], journal["trades"], journal["decisions"]
    out = [f"Session {m['session']} — {m['n_trades']} trades, day P&L ${m['day_pnl_usd']}.",
           "\nTRADES:"]
    for t in trades:
        out.append(f"- {t['side']} entry {t['entry']} exit {t['exit']} stop {t['stop']} "
                   f"target {t['target']} -> {t['pnl_usd']:+.2f}$ ({t['r']:+.2f}R) "
                   f"{t['exit_reason']} | {t['reason']}")
    out.append("\nALL DECISIONS (incl no_trade):")
    for d in decs:
        dec, ctx = d["decision"], (d.get("context") or {})
        out.append(f"- bar{d['bar']} {d['time']}: {dec.get('action')} "
                   f"{dec.get('setup', '')} | type={ctx.get('bar_type')} "
                   f"ai={ctx.get('always_in')} c={ctx.get('close')} "
                   f"vsEMA={ctx.get('close_vs_ema_pts')} | {str(dec.get('reason', ''))[:90]}")
    return "\n".join(out)


def _chat_text(system: str, user: str, cfg: AgentConfig) -> str:
    if cfg.provider == "zhipu":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY"),
                        base_url=os.getenv("ZHIPU_BASE_URL",
                                           "https://open.bigmodel.cn/api/paas/v4"),
                        max_retries=4)
        r = client.chat.completions.create(
            model=cfg.resolved_model(), max_tokens=cfg.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"})
        return r.choices[0].message.content
    if cfg.provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        r = client.models.generate_content(
            model=cfg.resolved_model(), contents=[user],
            config=types.GenerateContentConfig(system_instruction=system,
                                               response_mime_type="application/json",
                                               max_output_tokens=cfg.max_tokens))
        return r.text
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=cfg.resolved_model(), max_tokens=cfg.max_tokens,
        thinking={"type": "adaptive"}, output_config={"effort": cfg.effort},
        system=system + "\nReturn ONLY the JSON object.",
        messages=[{"role": "user", "content": user}])
    return next(b.text for b in r.content if b.type == "text")


def review_session(session: str, *, cfg: Optional[AgentConfig] = None,
                   write: bool = True) -> dict:
    cfg = cfg or AgentConfig()
    journal = json.loads((JDIR / f"{session}.json").read_text("utf-8"))
    review = _loads_lenient(_chat_text(REVIEW_SYSTEM, _build_user(journal), cfg))
    if write:
        (JDIR / f"{session}.review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), "utf-8")
        n = add_cases(review.get("experience_cases", []), session)
        review["_experience_cases_written"] = n
    return review
