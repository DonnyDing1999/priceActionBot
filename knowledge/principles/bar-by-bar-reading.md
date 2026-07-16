---
name: 逐棒阅读纪律 (Bar-by-Bar Reading)
type: principle
tags: [process, discipline, perception]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

每根**收盘**后重新评估市场状态,把每根当新证据。

## 每根收盘的顺序
分类型(trend / doji)→ 定角色(结构 / 信号 / 入场 / 确认棒)→ 定上下文(趋势 / 通道 / 区间 / 突破后 / 反转尝试)→ 看后 1–2 根 follow-through → 更新计划。

## 纪律要点
- **不在当前棒收盘前行动** —— 抢跑失败的代价常是省下那一跳的数倍。
- signal bar 只能事后确认(要后一根真破其极值)。
- 趋势棒无跟随 = 预警;2–3 根同向跟随 = 确认强度。过大的"耗尽棒"可能是 move 终点而非延续。
- 启发式顺序:先上下文再形态;先信号棒质量再入场计划;把第一反转信号当可交易前先找 second entry;任何信号要 follow-through 才做;拿不准就等下一根收盘。

## 对我们的意义
- 这正是回测的 event-driven 逐 bar 切片逻辑(只喂已收盘棒、次根 open 成交),与 no-lookahead 一致。
