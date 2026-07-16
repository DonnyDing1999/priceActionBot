---
name: Spike and Channel (Bull / Bear)
family: trend
detected_patterns: [spike_candidate, spike_active, spike_ending, climax_warning]
entry_setup_type: breakout_pullback
context: [breakout, trend_start]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
趋势的初始猛推(Leg 1):数根大趋势棒、极小回调 / 重叠。之后约 **60% 转通道、30% 转区间、10% 反转**。

## 分级 (grading)
- 1 根 = spike candidate(等确认);**2 根 = 最低可路由**(重叠 <30%);3–5 根 = 标准;6+ 根 = climax 警戒。
- Climax 触发(任一):影线 >50% 实体、实体 <30% 近均实体、或反向收盘棒结束这一串。

## 入场(只做 SPS / spike-flag,不追 climax 棒)
- **SPS**:spike 中 2–4 根回调出现顺向强 signal bar → 破其极值入。
- **Spike flag**:spike 后第一个小 flag 的顺向突破。
- 止损:回调 signal bar 极值外 1 tick(可向 spike 起点放宽,但不超过 spike 高度 50%)。(每笔 ≤ $75)

## 管理 / 目标
- spike 高度的 equal-legs [[measured-move]](100%,再 138.2% / 161.8%);若转通道,用通道高度目标。

## 失效 / 反面
- 单根大棒只是 candidate,需第 2 根确认;climax 触发后转"反转风险背景",**不是**新的顺势入场点。
