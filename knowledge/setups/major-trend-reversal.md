---
name: Major Trend Reversal (MTR)
family: reversal
detected_patterns: [mtr, reversal_attempt]
entry_setup_type: MTR
context: [trend_end, reversal]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
针对成熟趋势的反转,意在建立新的 HH/HL(或 LL/LH)序列。**四要素**:
1. 已确立的前趋势;
2. **收盘突破**趋势线 / 通道线;
3. 趋势未能恢复(无强 follow-through / 无新极值);
4. **回测前极值失败**(常是 double top / bottom)。

## 信号 / 入场 / 止损
- 在"回测前极值失败"后,做**第二次反转尝试(H2/L2 式)**(见 [[second-entry-higher-probability]])。
- 入场:反转 signal bar 极值破位入;止损:反转 signal bar / 被测极值外 1 tick。(每笔 ≤ $75)

## 管理 / 目标
- T1 = 反转起点;T2 = equal-legs [[measured-move]];T3 = 顺新趋势延续。

## 概率 / 失效
- 即使四要素齐全,**第一次反转尝试仅约 35–40% 成功**,第二次约 40%;趋势内单棒 / 弱反转失败率 ~80%。
- 价格重新创出被测极值之外 → MTR 作废,转回延续。
- 别把 Minor Trend Reversal(MRV,小回调)误标成 MTR。
