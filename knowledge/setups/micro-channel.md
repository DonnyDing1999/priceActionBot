---
name: Micro Channel
family: channel
detected_patterns: [always_in]
entry_setup_type: EMA_pullback
context: [trend, spike]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
2–10 根极紧通道:几乎每根 K 的高低都顺向刷新前一根,基本无回调(tight channel 的极端形态)。可容忍 1–2 根很小的 inside / doji 而不破形。

## 关键规则
- **逆 micro channel 的第一次突破 >80% 失败** → 那次失败本身是顺势延续入场点。
- 入场:micro channel 结束后的第一次回调,或逆势突破失败处。
- 止损:失败突破的极值外 1 tick。(每笔 ≤ $75)

## 管理 / 目标
- 结构 / [[measured-move]] 目标;若极紧(3+ 根零回调),当 spike 处理 —— 预期"两腿回调"(到中点、再到起点)后再入。

## 失效 / 反面
- 逆势突破若强 follow-through 成功 → 可能是反转信号,别硬顶。
