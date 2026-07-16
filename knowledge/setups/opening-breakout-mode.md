---
name: Opening Breakout Mode
family: breakout
detected_patterns: [breakout_mode]
entry_setup_type: breakout_mode
context: [open, around_bar_10]
cycle_position_hint: [trading_range, unknown]
status: reviewed
exemplar: ../exemplars/opening-breakout-mode__slide5.jpg
source:
  - video: "Brooks Price Action: Trading Opening Breakouts & Reversals"
    channel: "Brooks Trading Course"
    url: "https://www.youtube.com/watch?v=yGDYNoY3b68"
    ts: ["06:53", "07:26", "11:41", "12:17"]
---

## 定义
开盘约第 10 根附近,市场先创新高后反转下跌、又创新低后反转上涨(**任一顺序均可**)。当两个方向的反转都出现,市场进入 **breakout mode**:交易者预期从这个开盘区间向某一方向"甩出"(swing)。

## 前置条件 (preconditions)
- 开盘阶段,约第 10 根 K 线附近。
- 已出现"创新高 → 反转下" **和** "创新低 → 反转上" 两次反转。
- **区间要够大(≈10 根以上)**。若反转发生在 4–6 根的小区间,不算可靠 breakout mode —— 小紧区间是 limit-order 市场(在 bar 上方卖、下方买去 fade),不是突破市场。

## 入场 (entry)
- 突破式:区间高点上方 1 tick 买停(多)/ 区间低点下方 1 tick 卖停(空)。
- 提前式(front-run):也可在区间内反转处提前进,常要试 2–3 次才抓到一个 swing(正常)。

## 止损 (stop)
- 突破做多:止损放区间低点下方 1 tick;做空对称。(受每笔 $75 上限约束,超了就跳过)

## 管理 / 目标 (management)
- 目标 = 基于开盘区间高度的 measured move。
- 若转为区间日:区间内低买高卖 scalp。

## 失效 / 反面 (invalidation / failure_mode)
- 区间太小(<10 根)按此做突破 → 低胜率;改 fade 或等更大形态。
- 突破后立刻失败回到区间 → `breakout_failure`,常给出反向 second entry。

## 短引用
- *"after the tenth bar ... a reversal down from a new high ... and below the low of the day and reverse up ... that is a breakout mode situation"* [06:53]
- *"I want the trading range to last ten bars or more"* [11:41]
