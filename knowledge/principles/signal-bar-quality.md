---
name: 信号棒质量 (Signal Bar Quality)
type: principle
tags: [open, signal_bar, entry]
status: reviewed
source:
  - video: "Brooks Price Action: Trading Opening Breakouts & Reversals"
    channel: "Brooks Trading Course"
    url: "https://www.youtube.com/watch?v=yGDYNoY3b68"
    ts: ["16:28", "34:47"]
---

好的 signal bar:
- 做空:**bear bar 收在(接近)最低**;做多:**bull bar 收在(接近)最高**。
- 入场:signal bar 外 1 tick 停单;止损:signal bar 另一侧 1 tick。
- 紧区间里都是 doji / 小实体大影线 → limit-order 市场,信号棒弱(见 [[tight-range-fade]])。
- 大形态(≥10 根)给的信号棒比小形态(4–6 根)可靠。

**对我们的意义**:感知层要能判 `bar_type`(trend_bull / trend_bear / doji + 收盘位置);入场 / 止损的 1-tick 逻辑直接进执行层。
