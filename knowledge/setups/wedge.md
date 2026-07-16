---
name: Wedge (Three Pushes)
family: channel
detected_patterns: [wedge]
entry_setup_type: wedge
context: [trend_end, pullback, reversal]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
三次同向推动、力度递减,被两条收敛的同向线框住(趋势线 + 通道线)。两种角色:
- **Wedge pullback**(逆主趋势斜、约 10–20 根):延续结构 → 顺主趋势做突破。
- **Wedge reversal**(顺主趋势斜、≥ 约 20 根):终结结构 → 反转做,但要完整确认。

## 信号 / 入场 / 止损
- 触发 = wedge 边线的**突破 + 确认**(别在第 3 推上裸做)。
- 入场:突破方向,突破棒外 1 tick 停单;弱突破等回测 / second entry(见 [[second-entry-higher-probability]])。
- 止损:wedge 尖端一侧极值外 1 tick。(每笔 ≤ $75)

## 管理 / 目标
- T1 = wedge 起点(它推离的那个极值);T2 = 起点 + wedge 高度的 [[measured-move]]。

## 失效 / 反面
- 突破后又回进 wedge → 失败;可能原是早期反转而非单纯回调。
- **楔形斜向 ≠ 交易方向**:方向由主趋势 + 突破确认决定;约 1/3 的 rising/falling wedge 顺自身斜向突破(延续)。

## 变体
- Micro Wedge(3–4 根,每推 ≈1 根)、Nested Wedge(第 3 推本身是小 wedge)—— 需配合其它信号,勿单独入场。
