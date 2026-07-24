"""Live paper-trading loop — SPY via Alpaca (free IEX data + paper execution).

Poll-based by design: a 5-minute-bar bot gains nothing from a websocket except
failure modes. Every 5m boundary (+ a few seconds) we fetch the latest 1m bars
over REST, rebuild the closed 5m frame (START-stamped, ET-aligned — exactly the
backtest's bar semantics), and if a NEW window bar closed while we are flat, run
the same sidecar -> gate -> decide -> RiskManager path as the backtest. Entries
go in as Alpaca BRACKET stop orders (entry stop 1 tick beyond the signal bar +
server-side take-profit / stop-loss legs), canceled if untriggered after one bar
(the Brooks one-bar rule). 15:55 ET: cancel everything + flatten (never hold
overnight). Broker positions are the single source of truth — on divergence we
HALT and journal, we do not auto-correct.

Keys (user-configured in .env, never through Claude):
    ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY   (paper keys from alpaca.markets)
Knobs: PA_INSTRUMENT=spy  LIVE_QTY (default spec.qty)  PA_KILL_FILE (touch to stop)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pab.agent import AgentConfig, decide
from pab.bars import ET
from pab.experience import load_all_cases
from pab.features import build_sidecar, classify_regime  # noqa: F401 (regime via decide)
from pab.instruments import get_spec
from pab.journal import Journal
from pab.llm_strategy import LLMStrategy, obvious_no_trade
from pab.orchestration import WINDOWS, day_gate, journal_dir
from pab.risk import RiskManager

ROOT = Path(__file__).resolve().parents[1]
KILL_FILE = Path(os.getenv("PA_KILL_FILE", ROOT / "KILL"))

WINDOW = os.getenv("PA_WINDOW", "am")           # am | pm session window
WINDOW_FIRST = WINDOWS[WINDOW]["start"]         # entry-decision bars
WINDOW_LAST = WINDOWS[WINDOW]["end"]
FLAT_BY = WINDOWS[WINDOW]["flat_by"]            # cancel + flatten, never overnight


class LiveConfig:
    """Duck-typed cfg for RiskManager (same fields the backtest Config carries).
    All money knobs are env-overridable; point value scales with the actual qty
    (spec.point_value_usd is defined at spec.qty units)."""

    def __init__(self, spec, qty: Optional[int] = None):
        unit_pv = spec.point_value_usd / spec.qty      # $/pt per single share|contract
        self.qty = qty if qty is not None else spec.qty
        self.tick = spec.tick
        self.point_value = unit_pv * self.qty
        self.per_trade_risk_usd = float(os.getenv("PA_RISK_USD",
                                                  str(spec.per_trade_risk_usd)))
        self.min_risk_pts = spec.min_risk_pts
        self.min_rr = float(os.getenv("PA_MIN_RR", "1.0"))
        self.max_trades = int(os.getenv("PA_MAX_TRADES", "3"))
        self.max_consec_loss = int(os.getenv("PA_MAX_CONSEC_LOSS", "2"))
        self.daily_loss_cap = float(os.getenv("PA_DAILY_LOSS_CAP", "500"))


def _clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    key = os.getenv("ALPACA_API_KEY_ID")
    sec = os.getenv("ALPACA_API_SECRET_KEY")
    if not (key and sec):
        raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing from .env")
    return (StockHistoricalDataClient(key, sec),
            TradingClient(key, sec, paper=True))


def fetch_1m(data_client, symbol: str, start_et: pd.Timestamp) -> pd.DataFrame:
    """Closed 1m bars since start_et (IEX feed, free), ET tz, start-stamped."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start_et.tz_convert("UTC").to_pydatetime(),
                           feed=DataFeed.IEX)
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame()
    bars = bars.droplevel(0) if isinstance(bars.index, pd.MultiIndex) else bars
    bars.index = pd.DatetimeIndex(bars.index).tz_convert(ET)
    out = bars[["open", "high", "low", "close", "volume"]].sort_index()
    # drop the current, still-forming minute — closed bars only (no lookahead)
    now = pd.Timestamp.now(tz=ET).floor("min")
    return out[out.index < now]


def to_5m(m1: pd.DataFrame) -> pd.DataFrame:
    """1m -> START-stamped closed 5m bars (identical semantics to the loader)."""
    if m1.empty:
        return m1
    bars = (m1.resample("5min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"}).dropna(subset=["open"]))
    # a 5m bar is closed only when its window has fully elapsed
    now = pd.Timestamp.now(tz=ET)
    return bars[bars.index + pd.Timedelta(minutes=5) <= now]


def _round(px: float, tick: float) -> float:
    return round(round(px / tick) * tick, 2)


class LiveTrader:
    def __init__(self):
        self.spec = get_spec(os.getenv("PA_INSTRUMENT", "spy"))
        self.qty = int(os.getenv("LIVE_QTY", str(self.spec.qty)))
        self.cfg = AgentConfig()
        self.data, self.trading = _clients()
        self.rm = RiskManager(LiveConfig(self.spec, self.qty))
        self.session = str(pd.Timestamp.now(tz=ET).date())
        self.journal = Journal(run_id="live", provider=self.cfg.provider,
                               model=self.cfg.resolved_model())
        self.jdir = journal_dir(self.spec.name, WINDOW, live=True)
        self.frozen_cases = load_all_cases()   # freeze experience for this session
        self.pending: dict | None = None   # {"order_id", "bar_ts"} awaiting fill/cancel
        self.in_position = False
        self.entry_px: float | None = None
        self.halted_reason = ""
        self.decided: set[str] = set()     # bar timestamps already decided (idempotent)

    # ---------- broker truth ----------
    def position_qty(self) -> int:
        for p in self.trading.get_all_positions():
            if p.symbol == self.spec.symbol:
                return int(float(p.qty))
        return 0

    def reconcile(self) -> None:
        """Broker is truth. If broker and our state disagree, HALT (no auto-fix)."""
        broker_long = self.position_qty() > 0
        if self.pending is not None and broker_long:
            # entry stop filled -> we are in a position; bracket legs manage exits
            self.in_position, self.entry_px = True, self._filled_px(self.pending["order_id"])
            self.pending = None
            self._log("entry filled", {"entry": self.entry_px})
        elif self.in_position and not broker_long:
            pnl = self._closed_pnl()
            self.rm.on_trade_closed(pnl)
            self._log("position closed", {"pnl_usd": pnl})
            self.in_position, self.entry_px = False, None
        elif broker_long and not self.in_position and self.pending is None:
            self.halted_reason = "unknown broker position"   # divergence -> halt
            self._log("HALT: broker position with no local state", {})

    def _filled_px(self, order_id: str) -> float | None:
        try:
            o = self.trading.get_order_by_id(order_id)
            return float(o.filled_avg_price) if o.filled_avg_price else None
        except Exception:  # noqa: BLE001
            return None

    def _closed_pnl(self) -> float:
        """Approximate realized pnl of the just-closed round trip from order fills."""
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            orders = self.trading.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, symbols=[self.spec.symbol], limit=10))
            exits = [o for o in orders if o.filled_avg_price and o.side.value == "sell"]
            if self.entry_px and exits:
                exit_px = float(exits[0].filled_avg_price)
                return round((exit_px - self.entry_px) * self.qty, 2)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    # ---------- orders ----------
    def submit_bracket(self, side: str, trigger: float, stop: float, target: float,
                       bar_ts: pd.Timestamp) -> None:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (StopLossRequest, StopOrderRequest,
                                             TakeProfitRequest)
        t = self.spec.tick
        req = StopOrderRequest(
            symbol=self.spec.symbol, qty=self.qty,
            side=OrderSide.BUY if side == "long" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=_round(trigger, t),
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=_round(target, t)),
            stop_loss=StopLossRequest(stop_price=_round(stop, t)),
            client_order_id=f"pab-{self.session}-{bar_ts.strftime('%H%M')}",
        )
        o = self.trading.submit_order(req)
        self.pending = {"order_id": str(o.id), "bar_ts": bar_ts}
        self._log("bracket submitted", {"side": side, "trigger": trigger,
                                        "stop": stop, "target": target})

    def cancel_pending(self) -> None:
        if self.pending is None:
            return
        try:
            self.trading.cancel_order_by_id(self.pending["order_id"])
            self._log("entry canceled (one-bar rule)", {})
        except Exception:  # noqa: BLE001 — may have just filled; reconcile decides
            pass
        self.pending = None

    def flatten_all(self) -> None:
        try:
            self.trading.cancel_orders()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.trading.close_all_positions(cancel_orders=True)
            self._log("eod flatten", {})
        except Exception:  # noqa: BLE001
            pass
        self.pending, self.in_position = None, False

    # ---------- decision ----------
    def decide_bar(self, cont5: pd.DataFrame, bar_ts: pd.Timestamp) -> None:
        key = bar_ts.isoformat()
        if key in self.decided:
            return
        self.decided.add(key)
        sidecar = build_sidecar(cont5, bar_ts, symbol=self.spec.symbol, spec=self.spec)
        bar_i = sidecar["bar_index_from_open"]

        tag = obvious_no_trade(sidecar)
        if tag:
            self._record(bar_i, bar_ts, {"action": "no_trade", "reason": f"gated: {tag}"},
                         sidecar)
            return
        try:
            d = decide(sidecar, cfg=self.cfg, cases=self.frozen_cases)
        except Exception as e:  # noqa: BLE001
            self._record(bar_i, bar_ts, {"action": "error",
                                         "reason": f"{type(e).__name__}: {str(e)[:120]}"},
                         sidecar)
            return
        self._record(bar_i, bar_ts, d, sidecar)

        sig = LLMStrategy._to_signal(d, sidecar)
        if sig is None:
            return
        bar = sidecar["bar"]
        trigger = (bar["h"] + self.spec.tick if sig.side == "long"
                   else bar["l"] - self.spec.tick)
        if sig.entry_type != "stop":     # market entries: trigger ~ current close
            trigger = bar["c"]
        # first-obstacle R:R veto: identical obstacle set to the backtest (magnets +
        # session extremes; a session extreme the signal bar itself printed is not a
        # forward obstacle, so it never counts against its own reward).
        obstacles = [m["px"] for m in sidecar.get("magnets", [])]
        own = sidecar.get("bar", {})
        own_extreme = {"session_high": own.get("h"), "session_low": own.get("l")}
        for k in ("session_high", "session_low", "ema20"):
            v = sidecar.get(k)
            if v is not None and v != own_extreme.get(k):
                obstacles.append(float(v))
        rd = self.rm.validate_entry(sig.side, trigger, sig.stop, sig.target,
                                    obstacles=obstacles)
        if not rd.ok:
            self._log("veto", {"reason": rd.reason})
            return
        self.submit_bracket(sig.side, trigger, sig.stop, sig.target, bar_ts)

    # ---------- logging ----------
    def _record(self, bar_i, bar_ts, decision: dict, sidecar: dict) -> None:
        self.journal.record_decision(self.session, bar_i, bar_ts.strftime("%H:%M"),
                                     decision, sidecar=sidecar)
        self.journal.save(dir=self.jdir)

    def _log(self, msg: str, data: dict) -> None:
        print(f"[{pd.Timestamp.now(tz=ET).strftime('%H:%M:%S')}] {msg} "
              f"{json.dumps(data, ensure_ascii=False)}", flush=True)

    # ---------- main loop ----------
    def run(self) -> None:
        self._log("live loop start", {"symbol": self.spec.symbol, "qty": self.qty,
                                      "provider": self.cfg.provider,
                                      "model": self.cfg.resolved_model(),
                                      "paper": True})
        warmup_start = pd.Timestamp.now(tz=ET).normalize() - timedelta(days=5)
        m1 = fetch_1m(self.data, self.spec.symbol, warmup_start)
        self._log("warmup", {"m1_bars": len(m1)})

        # day-level chop gate from yesterday's texture (PA_DAYFILTER=off to disable)
        dayfilter = os.getenv("PA_DAYFILTER", "overlap")
        gate = (day_gate(to_5m(m1), self.session, mode=dayfilter, window=WINDOW,
                         cfg=self.cfg) if len(m1) else None)
        day_grade = "C" if gate is not None else "A"
        self._log("day grade", {"grade": day_grade, "variant": dayfilter})

        while True:
            if KILL_FILE.exists():
                self._log("KILL file present -> flatten + exit", {})
                self.flatten_all()
                return
            now = pd.Timestamp.now(tz=ET)
            hhmm = now.strftime("%H:%M")

            if hhmm >= FLAT_BY:                       # end of day
                if self.in_position or self.pending:
                    self.flatten_all()
                self._log("day done", {"halted": self.rm.session_halted()})
                return                                 # systemd restarts tomorrow

            # sleep to next 5m boundary + 7s (bars need a moment to land on IEX)
            nxt = (now.floor("5min") + pd.Timedelta(minutes=5, seconds=7))
            time.sleep(max(1.0, (nxt - now).total_seconds()))

            try:
                new = fetch_1m(self.data, self.spec.symbol,
                               m1.index[-1] + pd.Timedelta(minutes=1) if len(m1)
                               else warmup_start)
                if len(new):
                    m1 = pd.concat([m1, new]).loc[
                        lambda d: ~d.index.duplicated(keep="last")].sort_index()
            except Exception as e:  # noqa: BLE001 — feed blip: retry next cycle
                self._log("data fetch error", {"err": str(e)[:120]})
                continue

            cont5 = to_5m(m1)
            if cont5.empty:
                continue
            self.reconcile()
            if self.halted_reason or self.rm.session_halted()[0]:
                continue                               # watch only; eod still flattens

            last = cont5.index[-1]
            in_window = WINDOW_FIRST <= last.strftime("%H:%M") <= WINDOW_LAST
            if self.pending is not None and last > self.pending["bar_ts"]:
                self.cancel_pending()                  # one-bar working rule
            if (day_grade == "A" and in_window and str(last.date()) == self.session
                    and not self.in_position and self.pending is None):
                self.decide_bar(cont5, last)


def main() -> None:
    LiveTrader().run()


if __name__ == "__main__":
    main()
