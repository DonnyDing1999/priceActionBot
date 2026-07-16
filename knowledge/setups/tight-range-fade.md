---
name: Tight Trading Range -> Fade (Limit-Order Market)
family: range
detected_patterns: [barbwire, overlap, middle_range, trading_range]
entry_setup_type: tr_boundary
context: [open, tight_range]
status: reviewed
exemplar: ../exemplars/tight-range-fade__slide10.jpg
source:
  - video: "Brooks Price Action: Trading Opening Breakouts & Reversals"
    channel: "Brooks Trading Course"
    url: "https://www.youtube.com/watch?v=yGDYNoY3b68"
    ts: ["14:23", "14:55", "15:56", "29:02"]
---

## 定义
紧凑交易区间(doji、小实体大影线、重叠、无明确趋势;常 7+ 根)里,这是 **limit-order 市场**:在 bar 上方限价卖、下方限价买(fade),做 scalp。**不要用停单追突破**(区间高上方买停 / 低下方卖停)—— 在紧区间里这是低概率。

## 关键规则
- 追突破前先等**强突破棒 + follow-through**;没有跟随就别追。
- 高胜率 = 小回报(trader's equation):紧区间 fade 属于 scalp。

## 入场 (fade)
- 近区间上沿:bar 上方限价卖;近下沿:bar 下方限价买。scalp 出。

## 失效 / 反面(regime 切换)
- 出现强突破 + follow-through → 从"区间"切到"突破 / 趋势",改用突破战术(见 [[opening-18bar-range-breakout]])。

## 短引用
- *"buying with the stop at the top selling with the stop at the bottom is usually not the best way to enter"* [15:56]
- *"it's a limit order market ... selling with limit orders above ... buying ... below"* [14:55]

## 对我们的意义
这是"何时**不**做突破"的关键守则。风控上:紧区间 fade scalp 回报小、常不值当。**v1 建议先把紧区间标为 no-trade**,只做突破 / 反转 setup,等成熟再加 fade。
