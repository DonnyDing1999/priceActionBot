# priceActionBot

用 LLM 做 Al Brooks 价格行为(price action)交易的实验系统。标的是 MES(微型标普期货),只做美股开盘后两小时(09:30 到 11:30 ET)的 5 分钟图。整条链路包括:数值化感知、多模型决策 agent、独立风控否决层、严格无前视的事件驱动回测、复盘反思闭环,以及一条 SPY 纸上实盘线。

先说结论:截至 2026 年 7 月,这个系统还没有跑出可证明的 edge。仓库里真正值钱的是方法论和基建,策略本身仍在迭代。

## 现状(2026-07)

- v3 策略在 128 个交易日的干净段(剔除经验泄漏区后的 97 天、137 笔)上 PF 0.85,avg_R -0.06,和机械规则基线(PF 0.80)基本打平。按预注册标准(PF >= 1.2 且 t >= 2)判定:无 edge。
- 同期买入持有 1 手 MES 收益 +8.47%(+$2,954),途中最大浮亏 $3,402;策略全周期 -$184,最大回撤 $1,270。牛市半年里躺平的绝对收益更好,这一点如实记录;策略的相对优势只在回撤和隔夜零暴露上。
- v4 已合入(震荡纹理感知、日级停手开关、经验检索日期过滤、更省 token 的载荷),尚未做全量评测。
- 风控层从未失守:最差单笔 -$79,最差单日 -$152,都在设计边界内。

## 系统结构

```
Databento 1m 数据 ──> load_databento.py ──> mes_1m / mes_5m parquet
                                                  |
                        backtest.py 事件驱动引擎(逐 bar,无前视)
                                                  |
     features.build_sidecar  数值感知(session 表格、chop 纹理、摆动结构、磁铁位)
                                                  |
     dayfilter  日级 A/C 评分 ──C──> 当天停手(零调用)
                                                  |
     agent.decide  LLM 决策(anthropic / gemini / zhipu / claude_cli 四通道,
                   regime 路由知识卡片,经验库按日期过滤后注入)
                                                  |
     risk.RiskManager  独立否决:几何、单笔上限、R:R 下限、熔断
                                                  |
     成交模拟(stop 挂单、1m 内部判序、滑点手续费、eod 强平)
                                                  |
     journal ──> review agent 复盘 ──> experience 经验库 ──> 回流决策 prompt
```

感知层不用图像。曾经试过渲染 K 线图喂视觉模型,实盘延迟跟不上,后来改成把整段 session 的 OHLC、EMA、结构特征编成紧凑文本表格,模型直接读数字。

## 目录

```
pab/            核心包
  bars.py         bar 数据加载与 EMA
  features.py     数值 sidecar(决策输入)与 regime 分类
  dayfilter.py    日级 chop 评分(前瞻特征,无前视)
  agent.py        决策 agent,四个 LLM 通道,决策缓存
  risk.py         风控否决层
  backtest.py     事件驱动回测引擎
  llm_strategy.py 决策 agent 到回测引擎的适配
  journal.py      决策与成交全量落盘
  review.py       复盘 agent
  experience.py   经验库读写
  metrics.py      资金利用率与绩效指标
  live.py         SPY 纸上实盘循环(Alpaca)
  instruments.py  标的参数表(MES / SPY)
scripts/        入口脚本(数据加载、各类回测、报告、复盘、实验)
knowledge/      27 张 Brooks 知识卡片(setups + principles),按 regime 路由进 prompt
tests/          引擎与风控测试(28 个)
deploy/         systemd 服务单元
experience/     经验库 cases.jsonl(git 跟踪)
data/           行情、journal、决策缓存(gitignore,可再生)
```

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # 填入模型 key,见文件内注释
```

数据:从 Databento 下载 GLBX.MDP3 的 MES.FUT ohlcv-1m(JSON+zstd),zip 放仓库根目录后:

```bash
.venv/bin/python scripts/load_databento.py
```

会生成 `data/raw/mes_1m.parquet` 和 `mes_5m.parquet`。换月按前一日成交量选主力,无日内前视。

## 常用命令

```bash
# 机械规则基线(无 LLM,免费,几秒钟)
.venv/bin/python scripts/backtest_rules.py

# LLM 回测(逐 bar 决策;N_SESSIONS 控制天数,PA_WORKERS 并发)
N_SESSIONS=5 PA_PROVIDER=zhipu .venv/bin/python scripts/backtest_llm.py

# 用缓存重放(零 API 调用,改引擎参数后秒级重评;支持反向实验、延迟仿真)
.venv/bin/python scripts/backtest_cached.py

# 绩效报告(资金利用率、Sharpe、回撤,自动按污染边界分段)
PA_CAPITAL=5000 .venv/bin/python scripts/report.py

# 复盘 agent(增量,跳过已有 review 的 session)
PA_PROVIDER=claude_cli .venv/bin/python scripts/review_day.py

# SPY 纸上实盘(需要 Alpaca paper key,见 deploy/pab-live.service)
.venv/bin/python scripts/live_spy.py
```

## 主要配置(环境变量)

| 变量 | 默认 | 说明 |
|---|---|---|
| PA_PROVIDER | anthropic | anthropic / gemini / zhipu / claude_cli |
| PA_MODEL | 随通道 | 模型覆盖 |
| PA_TEMPERATURE | 0.1 | 低温度保证回测可复现 |
| PA_CACHE | 1 | 磁盘决策缓存,同样输入不重复调用 |
| PA_ROUTE | 1 | 按 regime 只带相关知识卡片 |
| PA_DAYFILTER | overlap | 日级停手开关,off 关闭 |
| PA_GATE | 1 | bar 级零成本预筛 |
| PA_RISK_USD | 75 | 单笔风险上限 |
| PA_DAILY_LOSS_CAP | 500 | 单日亏损熔断 |
| LIVE_QTY | 50 | 实盘每笔股数,点值自动换算 |

claude_cli 通道走本机已登录的 Claude Code(`claude -p` 官方 headless 模式),适合小规模强模型测试和复盘;大规模回测走 zhipu 免费通道。

## 评测纪律

这个仓库最重要的部分。教训都是花过学费的:

1. 无前视是铁律。决策只能看当前 bar 及之前的数据;信号 bar 收盘后下一根 bar 才能成交;1m 数据只用于已提交订单的成交判序。曾经用"整段 session 交给子 agent"的方式跑出过 PF 5.29,后来证明全是 lookahead 污染,同条件的逐 bar 隔离版本只有 PF 0.46。
2. 温度固定 0.1。默认温度下同一天两次跑会给出不同决策,实验不可复现。
3. 经验库按日期过滤。复盘产出的教训只能被之后的交易日引用。这个漏洞曾让 1 到 2 月的成绩虚高(PF 1.49,干净段只有 0.85)。
4. 判定标准预注册,跑完不许挪门柱:干净段 PF >= 1.2 且 t >= 2,再过 out-of-sample 数据段,三关全过才算有 edge。
5. 成本入账:每笔 $1.5 手续费加 1 tick 不利滑点。摩擦大约吃掉毛利的 9%。

## 风控

1 手 MES;单笔风险不超过 $75(约 15 点,止损过宽直接放弃,不许放宽);止损最小 2 点(更窄的单纯给成本打工);R:R 不低于 1;连亏 2 笔当日停机;单日最多 3 笔;单日亏损 $500 熔断;持仓可以跑过入场窗口,但 15:55 必须平掉,从不过夜。这些由代码强制执行,LLM 的提案只是提案。

## 免责

研究与工程实验项目,不构成投资建议。所有实盘相关代码目前只连接纸上账户。期货交易的亏损责任不以保证金为限,真钱之前请确认自己理解这一点。
