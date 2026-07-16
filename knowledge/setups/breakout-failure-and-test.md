---
name: Breakout Failure / Test / Failed-Failure
family: breakout
detected_patterns: [breakout_failure, breakout_test, breakout_pullback]
entry_setup_type: breakout_pullback
context: [breakout, range_boundary]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义(一族三态)
- **Failed breakout**:突破棒无 follow-through,次根回落进原结构 → fade 回原结构。
- **Breakout test**:突破后回测突破位;测试成功(不深收回)→ 顺突破方向入。
- **Failed failure**(= breakout pullback / second entry):失败突破本身又失败、价格恢复越过原突破位 → 顺原突破方向做(H2/L2 式),被套者提供燃料,**较可靠的延续**。

## 通用信号 / 入场 / 止损
- 需确认(反转棒 / 测试保持住),别单根裸做。
- 止损:相关 signal bar 极值外 1 tick。(每笔 ≤ $75)

## 关键数
- **~80% 的区间突破失败**;普通突破(1–2 根带影线)不追;唯 3–5 根连续趋势棒 + follow-through 才推翻默认怀疑。
- spike 级突破后常先 breakout test 再延续 —— 优先在回测处入,不直接追突破棒。
