# HCP 设计目标

这个仓库只研究一件事：

- **低边界异构跨域 CP / Ring Attention**

也就是多个异构 domain 在同一个 attention layer 内，按 sequence chunk 不均分协作，通过 ring 形态持续传递 K/V block 和 online-softmax state。

## 核心约束

1. **跨异构域只允许 P2P**
   - 不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主通信假设。
2. **允许不均分**
   - 不同 domain 可以有不同 `seq_chunk_len` 和 `block_size`。
3. **域内实现黑盒**
   - 本仓只定义跨域低边界协议，不定义域内 TP、kernel、runtime 细节。
4. **先证明可行性，再讨论性能**
   - 当前阶段优先建立数值闭环与协议闭环。

## 基本数据流

每个 domain 维护：

- 本地 `Q` chunk
- 本域当前收到的 `K/V` block
- online softmax 的运行态：
  - `running_max`
  - `running_sum`
  - `output`

ring 中持续发生的事情是：

1. 本域接收一个 `K/V` block
2. 用该 block 对本地 `Q` chunk 做一次局部 attention 贡献计算
3. 更新 online softmax state
4. 把 block 发给下一个 domain
5. 直到完整遍历 ring

## online softmax 目标

目标不是让每个 domain 保存全量 score matrix，而是通过逐 block 聚合保持数学等价性。

标准更新形式：

```text
m_new = max(m_old, m_local)
l_new = exp(m_old - m_new) * l_old + exp(m_local - m_new) * l_local
o_new = (
  exp(m_old - m_new) * l_old * o_old +
  exp(m_local - m_new) * local_pv
) / l_new
```

其中：

- `m_*` 是 running max
- `l_*` 是 running sum
- `local_pv` 是当前 block 的 `softmax(score_block) @ V_block` 贡献

## 与 HLPP 的边界

HLPP 负责：

- 按 layer range 切分
- 在层边界传 `hidden_states`
- coordinator 只做高边界编排

HCP 负责：

- 在单个 attention layer 内按 sequence chunk 切分
- 持续传 `K/V` block 与 softmax state
- 面向低边界跨域 P2P 协议

因此 HCP 不应再内嵌 HLPP 的高边界 batch、layer plan、domain orchestration 语义。

## 当前骨架包含什么

- 独立 `Status`
- 独立 `TensorDType` / `BoundaryTensor`
- `RingAttnConfig` / `RingAttnRuntime`
- 一个 `NoOp` runtime
- 一个本地 smoke
- 一个 Python 数值占位原型

## 下一阶段应该做什么

1. 把 Python `ringattn_kernel_stub.py` 扩成明确的 online-softmax correctness 报告
2. 为 `RingAttnMessage` 增加真实序列化与最小 P2P 传输实现
3. 引入真正的异构 runtime stub，例如 CUDA / MLX 两侧的 send/recv 协议占位
4. 再决定是否把这个目录迁移到新远程仓库
