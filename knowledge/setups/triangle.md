---
name: Triangle (Ascending / Descending / Symmetrical / Expanding)
family: breakout
detected_patterns: [ascending_triangle, descending_triangle, symmetrical_triangle, expanding_triangle]
entry_setup_type: triangle_breakout
context: [range, broad_channel]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
- **Ascending**:上沿平、下沿抬升(偏延续 / 上破有小 edge)。
- **Descending**:下沿平、上沿走低(下破有小 edge)。
- **Symmetrical**:两边收敛,突破前方向不可知 —— 不预判方向。
- **Expanding**:高更高、低更低(发散)—— 最混乱,**不做**,等收敛。

## 信号 / 入场 / 止损
- 需 **3–5 根趋势棒确认突破**,且**在回测 / 失败回测确认后**入,不在突破棒本身入。
- 止损:突破 / 测试 signal bar 极值外 1 tick。(每笔 ≤ $75)

## 管理 / 目标
- 三角最大高度的 [[measured-move]],从突破点投射。

## 失效 / 反面
- 突破无 follow-through → 回进三角(failed breakout,见 [[breakout-failure-and-test]])。
- 别把 ascending triangle 当看跌 —— 它偏延续,常与 rising wedge 混淆(见 [[wedge]])。
