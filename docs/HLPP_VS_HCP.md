# HLPP vs HCP

## 为什么必须单独讲这件事

`HCP` 很容易被误解成：

- `HLPP` 的细粒度版本
- `phase3` 的低层扩展
- 或者“把 layer-wise 做得更碎一点”

这些理解都不准确。

`HLPP` 和 `HCP` 解决的是两个不同层级的问题。它们可以共存，但不是同一条主线上的粗细变体。

## 一句话区别

- `HLPP`：`layer-wise / high-boundary`
- `HCP`：`intra-layer / low-boundary`

## 对比

| 维度 | HLPP | HCP |
|------|------|-----|
| 关注问题 | 不同 domain 如何分担不同层 | 多个 domain 如何共同完成同一个 attention 层 |
| 切分对象 | `layer range` | `seq chunk / block` |
| 通信内容 | `hidden_states` 等高边界张量 | `K/V blocks + online-softmax state` |
| 通信频率 | 在 layer boundary 处发生 | 在 attention 计算过程中持续发生 |
| 协议层级 | 高边界 | 低边界 |
| 当前主风险 | placement / orchestration / deployment | correctness / transport / heterogeneous cooperation |

## HLPP 的自然出发点

`HLPP` 的逻辑是：

- 每个 domain 内部仍然是一个黑盒执行器
- coordinator 决定哪些层在哪个 domain 上运行
- 每次跨域只需要传层边界结果

这条路线最适合回答：

- 不同算力设备怎样按层分工
- coordinator 怎样在高边界组织完整模型执行

## HCP 的自然出发点

`HCP` 的逻辑完全不同：

- 一个 attention 层本身已经大到不能只靠单个 domain 处理
- 不同 domain 需要共同完成同一个 attention
- 问题不再是“下一段 layer 交给谁”，而是“当前 attention 的 K/V 和 softmax 状态怎样在不同 domain 之间流动”

所以 HCP 的重点天然变成：

- `intra-layer` 协作
- 低边界协议
- online softmax 正确性
- `send/recv` 形态的持续 P2P

## 为什么要学习 Ring Attention

一旦问题变成“多个 domain 共同完成同一个 attention 层”，就不能再直接沿用 layer-wise 语言，也不能默认 collective 是天然答案。

学习 `Ring Attention` 的原因不是因为它是一个热门名字，而是因为它提供了一个非常贴近 HCP 问题本质的出发点：

- 每个 domain 保留本地 `Q chunk`
- `K/V block` 沿 ring 传递
- online softmax 逐 block 聚合
- 各 domain 不必持有完整 score matrix

对异构环境来说，这个方向尤其重要，因为它比“所有参与者同步到达、同步聚合”的 collective 假设更符合现实约束。

## 为什么 HCP 必须坚持 P2P

HCP 的前提不是“collective 永远不好”，而是：

- 在异构场景下，参与者本来就不对称
- 如果要求所有参与者按对称假设同步配合，最慢设备会主导整体节奏
- 对超长 context 场景来说，这种拖累会更明显

因此 HCP 当前坚持：

- 跨异构 domain 只允许 `P2P`
- 不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主通信假设

## HCP 不该继承 HLPP 的哪些语义

为了保持边界清楚，HCP 不应继续内嵌：

- layer plan 语义
- `BoundaryBatch` 式高边界编排叙事
- “coordinator 只做层边界串联”的 HLPP 视角

HCP 应该收敛成自己的一套最小问题定义：

- 如何表示 ring 中流动的 message
- 如何维护 online softmax state
- 如何把 attention 内协作限制在低边界协议内
- 如何让不同类型设备在不均分前提下依然能参与计算
