---
name: K 线类型与信号棒 (Bar Types & Signal Bars)
type: principle
tags: [bar, signal, perception]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## bar_type 词典
- **trend bar**:大实体短影;bull 开近低收近高,bear 镜像。
- **doji**:小 / 无实体,两边平衡(尺度相对)。
- **inside (IB)** / **outside (OB)**:被包 / 包住前一根。`ii`(两连内包)/ `iii`(三连)= 收缩,常先于突破尝试;`ioi`(内-外-内)= 突破形态。
- **2BR**(two-bar reversal):顺势棒后紧跟强反向棒(实体 ≥ 前棒 ~50%);engulfing / 平顶平底(= micro double top/bottom)是变体。可靠 ~50–55%,看上下文。

## 信号棒角色链
- **signal bar**:产生信号;理想 = 收近极值、大实体、短反向影、不超近均 ~1.5×(过大 = 犹豫 / 过伸)。**只有后一根真破其极值才在事后被确认**。
- **entry bar**:触发停单那根;强(大实体收近极值)= 确认,弱(小实体 / doji)= 风险升。
- **confirmation bar**:入场棒之后;强 = 创新极值强收,弱 = 没跟上(预警可能失败)。

## 信号棒可靠性排序(高 → 低)
1) 与交易同向的 trend bar → 2) 标准反转棒 → 3) 小棒 / inside 棒 + 强上下文 → 4) 反向 doji / 反向 trend bar。
(入场 / 止损的 1-tick 逻辑见 [[signal-bar-quality]])
