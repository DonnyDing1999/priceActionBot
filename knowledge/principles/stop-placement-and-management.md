---
name: 结构化止损放置 (Structural Stop Placement)
type: principle
tags: [risk, stop, structure, management]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

**初始止损放在结构失效点,而不是固定美元距离**。止损回答一个问题:"价格到哪里,这笔的前提就错了?"——那个点外 1 tick 才是止损,不是"我只想亏 X 美元"倒推出来的位置。

## 结构止损放哪(按 context)
- **顺势入场**:默认 signal bar 反向极值外 1 tick;signal bar 偏小时,放**回调 swing 点**外,别压在噪声里(见 [[signal-bar-quality]])。
- **区间 fade**:放**被 fade 的两极中较远的那个**外侧,**绝不放两极之间**——放中间等于给自己留了个必被扫的假止损。
- **spike 背景**:放 spike **中点**外侧——回调深过 spike 一半就否定了 spike 前提,不必把止损放得比这更宽(见 [[spike-and-channel]])。
- **通道**:放通道线 / 最近主 swing 外;**回调吃掉大半通道宽 = 通道读错了**,该退出而非扛。
- **区间边界 fade**:放被 fade 的边界外(约区间宽度 **10%**),不是随手找个整数位压上去(见 [[trading-range-fade]])。

## 硬上限:超了就跳过
- 若结构止损距离 > 每笔风险上限(**15 MES 点 / $75**),这笔**不做**,而不是换个更紧的假止损硬上。
- 两条歪路都堵死:**放宽**止损吃掉 R:R;**缩进**噪声必被扫。结构止损放不进上限,就是市场在说"这笔不划算",pass 掉去找下一个。

## 对我们的意义
- 引擎只发**一个 bracket**(单止损 + 单目标,盘中不移止损),所以止损质量必须**入场即对**——没有"先进去再调"的机会。
- 代码风控层强制 $75 上限与 R:R 下限([[traders-equation]]);本卡管的是止损**放在哪**,让 proposal 诚实地通过这两道检查,而不是靠捏一个刚好能过检的距离蒙混。
