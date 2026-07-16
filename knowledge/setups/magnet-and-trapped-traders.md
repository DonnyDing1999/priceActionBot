---
name: Magnet Levels & Trapped Traders
family: magnet_trap
detected_patterns: [magnet, failed_signal, trapped_traders]
entry_setup_type: none
context: [any]
status: reviewed
source:
  - method: "Al Brooks standard price-action method"
    scoped_via: "PA_Agent prompt_engineering curriculum (reference only; our own wording, AGPL-free)"
---

## 定义(多为目标 / 背景概念,非独立入场)
- **Magnet**:价格倾向被吸回去测试 / 停顿 / 反转的位。常见磁力位:失败 signal bar 的极值、失败交易的入场价、突破点、挂单堆积区。
- **Trapped traders**:突破 / 信号失败后被套在错方向、被迫止损,为反向 move 提供燃料。

## 用法
- 磁力位用于**设目标 / 止损参照**:失败信号的极值常成为反向(盈利)方向的目标。
- 被套越彻底,朝磁力位的 move 越强 —— 作为其它 setup(如 [[breakout-failure-and-test]])的强度佐证。

## 注意
- 价格逼近磁力位时,延续概率趋向 50/50。
- **Tick trap(5t / 9t / 17t / 41t)**:价格常在差 1 跳到整数目标处触发止损后反转 —— 止损 / 目标别正好压在整数位。
