# PA 知识库 · 分类骨架 (taxonomy)

> 我们自己的分类,用 Brooks 标准术语。程序特征用英文枚举,解释用中文。
> 概念清单参考了公开资料(术语属 Brooks 通用词汇),内容从一手视频蒸馏。

## 六大 setup 家族 (setup family)

| family | 中文 | 典型 setup |
|---|---|---|
| `channel` | 通道 | H1/H2、L1/L2、EMA 回调、wedge flag |
| `trend` | 趋势 | spike、always-in、trend resumption |
| `range` | 震荡 | 区间边界反手、barbwire、中部回避 |
| `reversal` | 反转 | MTR、reversal attempt、double top/bottom、failed-breakout reversal |
| `breakout` | 突破 | breakout mode、18-bar range breakout、breakout test/pullback |
| `magnet_trap` | 磁力 / 陷阱 | magnet、trapped traders、failed signal |

## detected_patterns 枚举(结构候选,不含方向)

```
h1 l1 h2 l2 wedge reversal_attempt mtr final_flag
breakout_mode breakout_failure breakout_test breakout_pullback
double_top_bottom barbwire overlap middle_range
always_in ail ais 20gb gap_bar opening_gap
spike_candidate spike_active spike_ending
climax_warning climax_triggered shrinking_stairs
failed_signal magnet trapped_traders
ascending_triangle descending_triangle symmetrical_triangle expanding_triangle
```

## bar_type 枚举
`trend_bull trend_bear doji inside outside_bull outside_bear flat other`

## entry_setup_type 枚举
`H1 L1 H2 L2 MTR wedge tr_boundary breakout_mode range18_breakout breakout_pullback EMA_pullback triangle_breakout failed_breakout_reversal none`

## 市场周期 cycle_position(Brooks spike-and-channel / trading range)
`spike micro_channel tight_channel normal_channel broad_channel trending_tr trading_range extreme_tr unknown`

## v1 作用域
- 标的 **MES**,**5 分钟图**,RTH 头两小时(9:30–11:30 ET,≈前 24 根)。
- 图上叠加 **5min 20-EMA + 60min 20-EMA**(Brooks 本人两条都看,见 principles)。
- 知识库现覆盖 Brooks 全套方法(见 `setups/` + `principles/`);但 **v1 实盘只做开盘 setup**,其余作背景 / 诊断与后续扩展。
