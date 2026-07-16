---
name: Trend Resumption after Failed Reversal
family: trend
detected_patterns: [failed_signal, always_in]
entry_setup_type: none
context: [open, after_failed_reversal]
status: reviewed
exemplar: ../exemplars/opening-trend-resumption__slide31.jpg
source:
  - video: "Brooks Price Action: Trading Opening Breakouts & Reversals"
    channel: "Brooks Trading Course"
    url: "https://www.youtube.com/watch?v=yGDYNoY3b68"
    ts: ["40:02", "41:08", "42:11", "44:44"]
---

## 定义
一个反转 setup(常是对昨日高 / 低的 second-entry 反转)**失败**后,原趋势恢复。你必须**平掉反手仓、转而顺势做**。"失败的反转"本身就是趋势信号。

## 前置条件 (preconditions)
- 先有一个反转尝试(如破昨日高点后的 second-entry 空)。
- 反转失败:价不跌反而 gap 回上 / 顺势强棒推进 → always-in 回到原方向。

## 入场 / 止损 (entry / stop)
- 先平掉失败的反手仓。
- 顺势进:强趋势棒收盘买 / 顺势 pullback 买;止损放失败反转的摆动点外 1 tick。(受每笔 $75 上限约束)

## 管理 (management)
- 当 swing 做,目标顺势 measured move;常表现为 small-pullback trend day。

## 失效 / 反面 (invalidation / failure_mode)
- 任何形态都会失败:趋势恢复本身也可能再失败 → 回到区间 / 反转。

## 短引用
- *"instead of selling off the bull trend resumed and therefore you have to get out of shorts and you have to get long"* [41:08]
- *"any pattern fails ... if it does fail you'll probably end up getting a trend"* [44:44]

## 关联
- 通常紧接 [[failed-breakout-yesterday-2nd-entry]] 的失败分支。
