"""Bar-series helpers (no plotting/LLM deps) shared by renderer, features, backtest."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ET = "America/New_York"

# CME outright contract symbol = <root><month-code><year-digit>, e.g. MESH6, MNQU6.
# Month codes: F G H J K M N Q U V X Z (Jan..Dec).
_CONTRACT_RE = re.compile(r"^([A-Z]+?)[FGHJKMNQUVXZ]\d$")


def contract_root(symbol: str) -> str | None:
    """Instrument root of a CME contract symbol, stripping the month+year suffix:
    'MESH6' -> 'MES', 'MNQU6' -> 'MNQ'. Returns None for anything that isn't a plain
    outright (e.g. calendar spreads like 'MESH6-MESM6') so callers can skip it."""
    m = _CONTRACT_RE.match(symbol)
    return m.group(1) if m else None


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load a parquet of OHLCV bars, sorted, lower-cased columns, tz-aware ET index."""
    df = pd.read_parquet(path).sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def complete_sessions(cont: pd.DataFrame, *, first: str = "09:30",
                      last: str = "11:25") -> list[str]:
    """Dates (as 'YYYY-MM-DD') where both the first and last opening-window bars exist."""
    hhmm = cont.index.strftime("%H:%M")
    a = {d for d, t in zip(cont.index.date, hhmm) if t == first}
    b = {d for d, t in zip(cont.index.date, hhmm) if t == last}
    return [str(d) for d in sorted(a & b)]


def add_ema(cont: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Sorted copy with a causal exponential `ema{period}` (EMA at t uses only bars <= t)."""
    cont = cont.sort_index().copy()
    cont[f"ema{period}"] = cont["close"].ewm(span=period, adjust=False).mean()
    return cont


def prior_rth_close(cont: pd.DataFrame, open_ts: pd.Timestamp,
                    session_date: str) -> float | None:
    """Close of the last bar (<=16:00 ET) on the most recent PRIOR trading date.
    Uses only bars strictly before `open_ts` (no lookahead)."""
    before = cont[cont.index < open_ts]
    if before.empty:
        return None
    sess_d = pd.Timestamp(session_date).date()
    prior_dates = sorted({d for d in before.index.date if d < sess_d})
    if not prior_dates:
        return None
    day = before[before.index.date == prior_dates[-1]]
    rth = day[day.index.strftime("%H:%M") <= "16:00"]
    src = rth if len(rth) else day
    return float(src["close"].iloc[-1])
