---
name: Double Top / Double Bottom (+ Micro)
family: reversal
detected_patterns: [double_top_bottom]
entry_setup_type: failed_breakout_reversal
context: [trend_end, range_boundary, reversal]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
两次测试相近极值,第二次未能明显创新极值后反转。可作 [[major-trend-reversal]] 的"回测失败"要素。**Micro** 版 = 相邻两根共享相近高 / 低(容差 ~<2 tick)。

## 信号 / 入场 / 止损
- Double top:确认第二顶失败的 bear signal bar 下方 1 tick 空,止损第二顶 / signal bar 上方 1 tick。Double bottom 镜像。(每笔 ≤ $75)

## 管理 / 目标
- T1 = 两顶 / 底之间的中点(neckline);T2 = 形态高度的 [[measured-move]]。

## 失效 / 反面
- 收盘明显越过两顶 / 底 → 作废,转延续。
- **有效形态需两次测试之间有清晰回调(neckline)**,不是单根长影线。大 double top/bottom 常由多个 micro 组成,更可靠。
