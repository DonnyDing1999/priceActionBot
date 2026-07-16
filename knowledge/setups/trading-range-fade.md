---
name: Trading Range Boundary Fade (H2/L2)
family: range
detected_patterns: [trading_range, tr_boundary]
entry_setup_type: tr_boundary
context: [range]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
明确区间(上下边界各被测 ≥2 次、EMA 走平)里,在边界**第二次触碰(H2 / L2)** 反手 fade 回区间。

## 前置条件
- 边界质量够(swing 尺寸均匀、各 ≥2 次测试)。**区间中部无 edge,回避**。
- 第一次触边常是突破尝试 / 陷阱;第二次确认边界。

## 信号 / 入场 / 止损
- 上沿:第二次测试的 bear signal bar 下方 1 tick 卖;下沿:bull signal bar 上方 1 tick 买。
- 止损:边界 / 第二测试极值外 1 tick。(每笔 ≤ $75)

## 管理 / 目标
- T1 = 对侧边界 / 中点;T2 = 区间高度的 [[measured-move]]。

## 失效 / 反面
- **~80% 的区间突破会失败** → 顺势的边界 fade 有 edge;但若第二次带 follow-through 突破 → 转突破 / 趋势,改战术。
- 极紧 / 铁丝网区间是另一回事(limit-order 市场),见 [[tight-range-fade]]。
