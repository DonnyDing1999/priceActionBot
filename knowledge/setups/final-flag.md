---
name: Final Flag (+ Failed Final Flag Reversal)
family: reversal
detected_patterns: [final_flag]
entry_setup_type: failed_breakout_reversal
context: [trend_end]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
成熟趋势末端的水平 / 近水平整理(约 10–20+ 根),像"最后一面旗"。**顺趋势突破 FF 的可靠性异常低**;其失败常标志趋势耗尽。

## 玩法一:别追顺势突破
- FF 的顺趋势突破没有 follow-through → 大概率失败。不追。

## 玩法二:Failed FF Reversal(主打)
- 顺势突破(1–2 根、无跟随)回落进旗 → 反向 signal bar 反手做。
- 入场:反向,失败突破极值外 1 tick 止损。(每笔 ≤ $75)
- 目标:反向的 [[measured-move]] / 前结构位。

## 备注
- 常与 [[major-trend-reversal]] 的末段重叠;因锚定具体的末端结构失败,概率通常高于泛泛 MTR 尝试。
- **勿与 spike flag 混淆**:spike flag 允许顺势突破,FF 禁止追顺势(见 [[spike-and-channel]])。
