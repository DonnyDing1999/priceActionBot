---
name: Trader's Equation + 60/40
type: principle
tags: [expectancy, risk, management]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 交易者方程
入场前的期望值检验:**胜率 × 回报 > 败率 × 风险** 才有正期望。
- R:R 用**近端(T1)目标**算,不用延伸目标。
- 近端目标最低 R:R = 1:1。
- 与止损 / 目标选择一起在**入场前**评估。

## 止损 / 目标(结构化,非固定 tick)
- 止损:默认 signal bar 极值外 1 tick;入场棒 ≥ 信号棒强时可收紧到入场棒极值;宽噪通道用最近 swing 极值。
- 目标两层:T1 = 最近结构;T2 = MM / 远端结构(见 [[measured-move]])。几何序须成立:多 → 止损 < 入场 < T1 < T2。
- 风险距离体检:止损 > ~2×ATR 或 > 通道宽 50% → 这笔不划算(呼应我们**每笔 ≤ $75** 的硬上限:超了直接跳过,不放宽止损)。

## 60/40 规律
- ~60% 交易是小赢 / 小亏,基本抵消;~20–30% 大赢(趋势单);~10–20% 大亏(反转止损)。
- edge 来自"小额 ≈ 0 + 大赢 ≫ 大亏",不是胜率本身。second entry 主要靠减少小亏来改善分布。
