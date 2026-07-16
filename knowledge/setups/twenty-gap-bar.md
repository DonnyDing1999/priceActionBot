---
name: 20 Gap Bars (20GB) — First EMA Touch
family: trend
detected_patterns: [20gb, gap_bar]
entry_setup_type: EMA_pullback
context: [strong_trend]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
约 20 根连续 K 未触及 20EMA —— 极强、过度延伸的趋势。**不是**反转信号,只是提高均值回归回调的概率。

## 入场
- **第一次触碰 EMA** = 高概率顺势延续 scalp,至少预期回测趋势前极值。需合格 signal bar + 紧止损。
- 止损:signal bar 极值外 1 tick(刻意小风险)。(每笔 ≤ $75)

## 关键规则
- **两击规则**:第一次触碰后入场若失败,可试第二次;两次都失败就停,回到顶层重新诊断。
- 20GB 本身**绝不**构成逆势理由 —— 没有确认的反转前,尊重延续。

## 相关
- `gap_bar`(均线缺口棒,整根在 EMA 一侧)是更小尺度的同类;别与 `opening_gap`(开盘跳空)混淆。
