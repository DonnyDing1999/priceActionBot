"""Deterministic chart renderer: OHLCV bars -> annotated 5m candlestick + 20EMA PNG.

The SAME renderer is used in backtest and live (no train/serve skew). The image is
the perception input for the vision model; pair it with the numeric sidecar
(pab.features) because vision models misread precise prices.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; save-only
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402

from pab.bars import ET, add_ema, prior_rth_close  # noqa: E402


def render_session(cont: pd.DataFrame, session_date: str, *,
                   pre_open_bars: int = 12, window_end: str = "11:25",
                   ema_period: int = 20, up_to: "pd.Timestamp | str | None" = None,
                   out_path: str | Path | None = None,
                   title: str | None = None) -> dict:
    """Render one session's opening window (+ pre-open context) to a PNG.

    cont : continuous 5m OHLCV, tz-aware ET DatetimeIndex, cols open/high/low/close/volume.
    window_end : last bar's start time (default 11:25 -> the 24 bars of 09:30-11:30).
    up_to : if set, render only bars at/<= this timestamp (no-lookahead for live/backtest).
    """
    cont = add_ema(cont, ema_period)
    ema_col = f"ema{ema_period}"

    open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    win_end_ts = pd.Timestamp(f"{session_date} {window_end}", tz=ET)
    start_ts = open_ts - pd.Timedelta(minutes=5 * pre_open_bars)

    sl = cont[(cont.index >= start_ts) & (cont.index <= win_end_ts)]
    if up_to is not None:
        sl = sl[sl.index <= pd.Timestamp(up_to)]
    if sl.empty:
        raise ValueError(f"no bars for session {session_date}")
    prior_close = prior_rth_close(cont, open_ts, session_date)

    # mplfinance wants capitalized columns + a (naive) DatetimeIndex
    plot = sl.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"}).copy()
    plot.index = plot.index.tz_localize(None)
    open_naive = open_ts.tz_localize(None)

    apds = [mpf.make_addplot(sl[ema_col].to_numpy(), color="#2962ff", width=1.1)]
    hlines = (dict(hlines=[prior_close], colors=["#9e9e9e"], linestyle="--",
                   linewidths=1.0) if prior_close is not None else None)
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":",
                               gridcolor="#e9e9e9")

    fig, axes = mpf.plot(
        plot[["Open", "High", "Low", "Close", "Volume"]],
        type="candle", style=style, addplot=apds, volume=True,
        figsize=(14, 8), returnfig=True, ylabel="ES price",
        vlines=dict(vlines=[open_naive], colors=["#111111"], linewidths=1.2, alpha=0.6),
        hlines=hlines,
        title=title or f"ES 5m — opening window {session_date} (RTH 09:30-11:30 ET)",
    )
    ax = axes[0]

    # bar numbers from the open (bar 1 = 09:30 bar); positional x-axis
    positions = {ts: i for i, ts in enumerate(sl.index)}
    session_bars = sl[sl.index >= open_ts]
    for n, ts in enumerate(session_bars.index, start=1):
        ax.annotate(str(n), (positions[ts], float(session_bars.loc[ts, "low"])),
                    xytext=(0, -11), textcoords="offset points", ha="center",
                    va="top", fontsize=6.5, color="#555555")
    if prior_close is not None:
        ax.annotate(f"prev close {prior_close:.2f}", (0, prior_close),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                    color="#8a8a8a")
    ax.annotate("EMA20", (len(sl) - 1, float(sl[ema_col].iloc[-1])),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color="#2962ff")

    out_path = Path(out_path) if out_path else Path(f"session_{session_date}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"out_path": str(out_path), "n_session_bars": int(len(session_bars)),
            "prior_close": prior_close}
