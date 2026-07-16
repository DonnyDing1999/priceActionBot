---
name: Channel Pullback — H1/H2 (bull) · L1/L2 (bear)
family: channel
detected_patterns: [h1, h2, l1, l2]
entry_setup_type: H2
context: [trend, channel, pullback]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义
趋势通道里的回调计数入场。上升通道(HH + HL):**H1** = 第一段回调后突破前一根 K 高点;**H2** = 第二段回调后再突破。下降通道(LL + LH)镜像:**L1 / L2** = 反弹后跌破前一根 K 低点。

## 前置条件
- always-in 方向明确(见 [[always-in]]),处于通道 / 趋势的**回调**阶段(不是初始 spike)。
- 回调常触及 20EMA;出现一段(H1/L1)或两段(H2/L2)腿。

## 信号 / 入场 / 止损
- 信号棒:回调末与趋势同向的 signal bar(见 [[signal-bar-quality]])。
- 做多:前一根 K 高点上方 1 tick 买停;做空:前一根 K 低点下方 1 tick 卖停。
- 止损:信号棒另一侧 1 tick。(每笔 ≤ $75,超了跳过)

## 管理 / 目标
- T1 = 通道中线 / 近端结构;T2 = 通道线或通道高度的 [[measured-move]]。

## 失效 / 反面
- 回调突破前一个 HL / LH → 序列破,通道失效,转区间重估。
- **H2 / L2 通常比 H1 / L1 可靠**(见 [[second-entry-higher-probability]]);第三次(High3 / Low3)常退化成 [[wedge]]。

## 备注
- Brooks:上升通道在更大窗口本质是 "bear flag"(反之亦然)—— 背景判断,不构成反向交易信号。
