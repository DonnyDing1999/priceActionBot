---
name: Always In (AIL / AIS) 方向过滤
type: principle
tags: [trend, bias, filter]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

**Always In** = 从近期 bar 结构读出的持续方向偏置:多数收盘在 EMA 上方、回调浅、反向反转尝试失败、低点抬高 → **AIL**(Always In Long);镜像为 **AIS**。

## 用法
- 它是**背景过滤器**,决定哪一侧的 setup(H1/H2 vs L1/L2 等)被偏好。
- 与 always-in 同向的不完美信号仍可做;逆 always-in 的信号需**明显更强的确认**。
- 逆 always-in 的反转需**双重确认**:收盘破趋势线 **且** 回测前极值失败(即 [[major-trend-reversal]] 的核心),才当真正反转,而非噪声。

## 对我们的意义
- 感知层要持续输出 always-in 方向(`ail` / `ais`)作为决策 gate;宽通道里结构上偏 always-in 一侧,而非去 fade 对侧。
