---
name: 市场周期模型 (Market-Cycle Model, 8 states)
type: principle
tags: [diagnosis, cycle, structure]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

从强到弱的 8 态谱,每态有典型 bar 行为和标准打法。**这是感知层"诊断"的主干**(先定周期,再选 setup)。

| 状态 cycle_position | 特征 | 打法 |
|---|---|---|
| `spike` | 数根同向大棒,几乎无回调 | 只顺势;不做逆势 |
| `micro_channel` | 2–10 根极紧;首次逆势突破 >80% 失败 | 顺势;逆突破失败 = 延续入场 |
| `tight_channel` | 陡,回调 <~30% | 顺势;回 EMA 是入场 |
| `normal_channel` | 回调 ~30–50% | 只顺势;等突破回测 |
| `broad_channel` / stairs | 缓,回调 ~50%+,几乎每次突破被回测 | 只顺主向;在 breakout-test 回调入 |
| `trending_tr` | 退化的宽通道(<3 腿或序列破) | 同宽通道,信心更低 |
| `trading_range` | 双边、边界各测 ≥2 次、EMA 平 | 只在边界顺势做;避中部 |
| `extreme_tr` | 最乱、EMA 缠绕 | **不做** |

## 分类规则 (channel vs range)
1. ≥2 腿 HH+HL(或 LL+LH)→ 通道候选;否则区间。
2. 通道候选:≥3 腿 + 稳定平行线 → 按回调深度分 tight / normal / broad;否则 `trending_tr`。
3. 区间候选:边界清晰、各测 ≥2 次、EMA 平 → `trading_range`;彻底无向 → `extreme_tr`。

## 状态切换
- 只在硬证据下确认切换(spike 级突破 + follow-through、成功 breakout-test、确认收盘越界);含糊时降低交易频率。
- 多窗口:同一序列在不同回看窗口给"结构背景 + 执行读数";冲突时**近端结构定方向**,大窗口方向留作支撑 / 阻力风险注记。
