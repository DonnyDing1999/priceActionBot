# knowledge/ — Brooks 价格行为知识库

交易 agent 的知识库(不是 Claude 的对话记忆)。从一手视频/资料蒸馏成结构化卡片,供感知 / 决策 agent 与 RAG 使用。

## 约定
- **原创蒸馏**:只存我们自己整理的要点 + 短引用 + 出处;不存整段逐字稿,不复制他人代码 / prose(保持 AGPL-free)。
- **语言**:结构字段 / 枚举用英文(便于程序),解释用中文;Brooks 术语保留英文。
- **provenance**:每张卡 `source` 标注视频标题 / 频道 / URL / 时间戳。
- **review gate**:新卡 `status: draft`;你(懂 Brooks)审过改成 `reviewed` 才进决策。

## 结构
- `taxonomy.md` — 分类骨架:setup 家族、`detected_patterns` / `bar_type` / `entry_setup_type` 枚举、市场周期。
- `setups/` — 每个 setup 一张卡(前置 / 信号 / 入场 / 止损 / 管理 / 失效 / 反面 / 出处)。
- `principles/` — 心法与统计原则,做 RAG。
- `exemplars/` — 每个 setup 的标注参考图(首批来自视频帧;长期用自渲染图替代,见 `exemplars/README.md`)。

## 已蒸馏来源
- `yGDYNoY3b68` — *Brooks Price Action: Trading Opening Breakouts & Reversals*(Brooks Trading Course,70 min)。主题:开盘的突破与反转,正中 v1 作用域。(带 exemplar 参考图)
- **Brooks 全套方法(标准)** —— 通过 PA_Agent `prompt_engineering` 课程图谱**过一遍作覆盖 checklist**,卡片用我们自己的措辞写(AGPL-free,暂无 exemplar 图)。覆盖通道 / 尖峰 / 楔形 / H1H2-L1L2 / AlwaysIn / 20GB / 区间 / 铁丝网 / 三角 / MTR / 双顶底 / 最终旗形 / 突破失败 / 磁力位,及市场周期模型 / 测量移动 / 交易者方程 / K线信号 / 逐棒阅读。
