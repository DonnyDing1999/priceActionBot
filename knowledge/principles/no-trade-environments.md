---
name: No-Trade 环境 (不做也是一种仓位)
type: principle
tags: [discipline, no-trade, environment, filter]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

**不交易本身就是一种仓位**。Brooks 的立场:交易者是**概率管理者,不是动作寻求者**——没有 edge 时,`no_trade` 是正确的"下单",不是漏掉的机会。某些环境里,任一方向的入场期望值都不为正,唯一正确的打法是等下一个 setup。

## 五类 no-trade 环境
- **铁丝网 (barbwire)**:3+ 根重叠小实体 doji 骑在 EMA 上——**limit-order 市场,绝不用停单入**(停单在这里被双向来回扫)。见 [[tight-range-fade]]。
- **发展成熟区间的中部**:buy-low / sell-high 只在**区间的三分位**成立;**中间那三分之一两个方向都没 edge**,回避。见 [[trading-range-fade]]。
- **climax 后的力竭**:出现超大棒(range ≥ ~2× 均幅)后,**第一个回调两个方向都不可交易**——等**第二个信号**再说(见 [[spike-and-channel]] / [[second-entry-higher-probability]])。
- **逆着紧通道**:紧 / 微通道里**不做逆势入场**,不管看着多超买 / 超卖——逆势突破 >80% 失败(见 [[market-cycle-model]])。
- **窗口末端**:9:30–11:30 快收尾时晚入场**没有跑的空间**,R:R 撑不起来。

## 程序信号 → veto 映射
决策 agent 在 sidecar 里读到的量,对应哪条否决:
- `zone == "middle"` 且 range regime → 中部无 edge → veto `zone_middle`。
- `chop.overlap` 高 + efficiency 低 → 铁丝网 / 极乱 → 不路由停单。
- climax 棒 range 对 `avg_range_10` ≥ ~2× → veto `climax_no_chase`。
- `bar_index` > 19(窗口末端)→ 代码 gate `too_late_in_window`(LLM 不会被调用)。
- 刚被扫损后:3 棒内在同一结构(±3 tick)反手 → veto `cooldown_opposite`;2 棒内同向重进 → veto `cooldown_same`。

## 对我们的意义
- 风控层用这些 tag 否决时,**本卡就是否决背后的 doctrine**——不是 bug,是纪律。
- **别跟 veto 较劲**:别为了下单去放宽结构、编个更紧的止损。被否了就**找下一个 setup 或直接 pass**。少做没 edge 的小亏单,分布才会改善([[traders-equation]] 的 60/40)。
