# HCP Ring Attention Correctness Report

## 摘要

本报告记录 HCP Ring Attention 的数学正确性验证结果。验证采用 **reference-first** 方法：将逐 block online softmax 的分布式计算结果与标准 full attention reference 在全精度下进行对照，证明两者在数值上保持等价。

当前验证矩阵覆盖 7 个 case，涵盖不均分 chunk、不均分 block size、单 domain、大 sequence（1024）、unit block（block_size=1）等边界条件。全部 case 在 `Strict` tolerance tier 下通过。

---

## 验证目标

证明以下命题：

> 对于任意 domain 划分（`seq_chunk_len` 可不均分）和任意 block 切分（`block_size` 可不同），每个 target domain 按 ring 顺序接收所有 source domain 的 K/V block 并执行 online softmax 更新后，其本地输出 chunk 与 full attention reference 在约定误差范围内一致。

这保证了 HCP 的分布式 attention 数学上与标准 attention 等价，不因 domain 边界或 block 切分引入系统性偏差。

---

## 验证方法论

### Reference Implementation

使用标准 `softmax(QK^T / sqrt(d))V` 公式计算 full attention，作为黄金参考。

### Block-wise Implementation

每个 target domain：
1. 保留本地 Q chunk。
2. 从自身 domain 开始，按 ring 顺序遍历所有 source domain。
3. 每个 source domain 按自己的 `block_size` 切出 K/V block。
4. 对每个 block 计算局部 scores 和未归一化的 `P @ V`。
5. 使用 online softmax 公式迭代更新 `running_max`、`running_sum`、`output`：
   ```
   m_new = max(m_old, max_j(score_ij))
   l_new = l_old * exp(m_old - m_new) + sum_j(exp(score_ij - m_new))
   o_new = o_old * exp(m_old - m_new) + sum_j(exp(score_ij - m_new) * v_j)
   ```
6. 遍历完成后，output = o_final / l_final。

### Tolerance Policy

数值验证采用**分级 tolerance**，而非单一阈值。原因：
- float32 机器 epsilon ≈ 1.2e-7，任何要求 < 1e-7 的比较在 float32 下都不可能稳定达到。
- 异构设备（CPU / MPS / CUDA）的矩阵乘法累加顺序、FMA 策略、kernel 实现不同，会引入 ~1e-6 ~ 1e-5 量级的系统性差异。
- 多层模型中误差会累积，端到端比较需要更宽的容忍度。

| Tier | 场景 | max_abs_err | mean_abs_err | max_rel_err | 来源 |
|------|------|-------------|--------------|-------------|------|
| **Strict** | 同设备算法等价性验证 | 1e-5 | 1e-6 | 1e-5 | PyTorch `assert_close` float32 默认值附近 |
| **Relaxed** | 异构设备交叉验证（MPS vs CUDA） | 1e-4 | 1e-5 | 1e-4 | 机器精度 ~1000 倍余量，容纳 FMA + kernel 差异 |
| **EndToEnd** | 多层模型端到端（误差累积） | 1e-3 | 1e-4 | 1e-3 | 典型 attention 输出范围 ~[-10,10]，对应第 4~5 位有效数字 |

**判定公式**（与 PyTorch `torch.isclose` 一致）：
```
|a - b| <= atol + rtol * |b|
```

---

## 测试矩阵

当前 7 个 case 覆盖以下维度：

| Case | Domains | Global Seq | Heads | Head Dim | 关键特征 |
|------|---------|------------|-------|----------|----------|
| 2domain_uneven_chunks | 2 | 128 | 4 | 16 | 不均分 chunk（80+48） |
| 3domain_uneven_blocks | 3 | 160 | 3 | 24 | 不均分 block size（32,10,14） |
| 4domain_small_tail_blocks | 4 | 192 | 2 | 32 | tail block < block_size（7,16,11,13） |
| 3domain_large_seq | 3 | 1024 | 8 | 64 | 大 sequence，不均分 chunk（512+256+256） |
| 1domain_single_block | 1 | 64 | 4 | 16 | 单 domain，block_size = seq_len |
| 2domain_unit_blocks | 2 | 32 | 2 | 8 | block_size = 1，最小边界 |
| 1domain_medium | 1 | 128 | 4 | 16 | 单 domain，中等规模 |

---

## 实测结果

运行环境：Mac M1 Pro，CPU 路径（Rust correctness model 使用 float64 计算 reference 与 model 输出）。

**全部 case 使用 `Strict` tolerance tier 通过。**

| Case | max_abs_err | mean_abs_err | max_rel_err | 与 Strict tol 余量 |
|------|-------------|--------------|-------------|-------------------|
| 2domain_uneven_chunks | 5.00e-16 | 5.72e-17 | 4.27e-13 | ~1e+5 倍 |
| 3domain_uneven_blocks | 8.88e-16 | 5.39e-17 | 9.82e-12 | ~1e+3 倍 |
| 4domain_small_tail_blocks | 9.99e-16 | 5.60e-17 | 7.14e-13 | ~1e+4 倍 |
| 3domain_large_seq | 9.16e-16 | 5.24e-17 | 1.29e-09 | ~1e+1 倍 |
| 1domain_single_block | 8.88e-16 | 5.64e-17 | 9.36e-13 | ~1e+4 倍 |
| 2domain_unit_blocks | 3.89e-16 | 6.14e-17 | 2.24e-14 | ~1e+5 倍 |
| 1domain_medium | 9.99e-16 | 5.28e-17 | 4.19e-12 | ~1e+3 倍 |

### 结果解读

1. **max_abs_err 在 1e-15 量级**：远低于 float64 机器精度（~2e-16）的合理范围，说明 algorithm-level 实现没有系统性偏差。
2. **max_rel_err 在 1e-14 ~ 1e-9 量级**：远低于 Strict tier 的 `1e-5` 阈值，余量充足。
3. **3domain_large_seq 的 max_rel_err 最大（1.29e-9）**：随 sequence 长度增加，累加次数增多，相对误差自然放大，但仍远小于阈值。

---

## Stress Test

`--stress-test` 模式对 seq_len <= 256 的 case 运行 5 个随机 seed（42~46），验证随机性不会导致异常误差。

当前 stress test 结果：全部通过，误差分布与单 seed 一致。

---

## 结论

1. **数学等价性已验证**：在当前 7 个 case 覆盖的边界条件下，HCP Ring Attention 的 online softmax 逐 block 聚合与 full attention reference 在 `Strict` tolerance 下完全一致。
2. **分级 tolerance 已建立**：`Strict` / `Relaxed` / `EndToEnd` 三级标准分别对应同设备算法验证、异构交叉验证和端到端模型验证，阈值基于 float32 机器精度和业界框架默认值推导。
3. **异常值安全**：所有实测 max_rel_err 与 threshold 之间保持至少 10 倍以上的余量，不会出现在阈值边缘抖动的风险。

---

## 运行方式

### 标准验证（单 seed）

```bash
cd rust
export LIBTORCH=/Users/stark_sim/libtorch
export DYLD_LIBRARY_PATH="${LIBTORCH}/lib:${DYLD_LIBRARY_PATH:-}"
cargo run --bin hcp-ringattn-rust -- --tolerance-tier strict
```

### Stress 验证（5 seeds）

```bash
cargo run --bin hcp-ringattn-rust -- --stress-test --tolerance-tier strict
```

### 切换 tolerance tier

```bash
# 异构设备交叉验证（宽松标准）
cargo run --bin hcp-ringattn-rust -- --tolerance-tier relaxed

# 端到端模型验证（最宽松标准）
cargo run --bin hcp-ringattn-rust -- --tolerance-tier end-to-end
```

Report 输出路径默认：`reports/rust_ringattn_correctness.json`

---

## 附录：Tolerance 推导依据

### float32 机器精度

- float32 有效二进制位：24 位（含隐含前导 1）
- 机器 epsilon：ε = 2^-23 ≈ 1.192e-7
- 约 6~7 位十进制有效数字
- 任何要求误差 < 1e-7 的数值比较在 float32 下都不可能稳定达到

### 异构设备差异来源（按影响排序）

| 来源 | 误差量级 | 说明 |
|------|----------|------|
| 累加顺序不同 | ~1e-6 | GPU warp-reduce 与 CPU 串行累加顺序不同，浮点加法不满足结合律 |
| FMA (fused multiply-add) | ~1e-6 | GPU 单指令 `a*b+c` 只舍入一次，CPU 可能分两步舍入两次 |
| Softmax/Exp 实现 | ~1e-6 | 不同硬件的 exp 近似多项式和 online softmax max 更新时机不同 |
| Kernel 实现差异 | ~1e-5 | MPS Metal kernel vs CUDA cuDNN 数值策略不同 |

### 业界参考

- PyTorch `torch.testing.assert_close` float32 默认：`rtol=1.3e-6, atol=1e-5`
- NumPy `numpy.allclose` float32 默认：`rtol=1e-5, atol=1e-8`
- ICON-A 气候模型 GPU 迁移（float64）：`rtol=1e-12`（机器精度 1e-15 的约 1000 倍余量）

HCP 的 `Relaxed` tier（`rtol=1e-4`）约等于 float32 机器精度的 1000 倍余量，与上述学术/工业实践一致。
