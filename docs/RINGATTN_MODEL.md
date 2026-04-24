# Ring Attention Correctness Model

## 目标

本模型用于在不依赖真实 CUDA / MLX / NPU kernel 的情况下，先固定 HCP Ring Attention 的数学闭环。

它回答的问题是：

- 不同 domain 可以拥有不均分的 `seq_chunk_len`。
- 不同 source domain 可以使用不同的 `block_size`。
- 每个 target domain 只持有本地 `Q chunk`。
- `K/V block` 按 ring 顺序被访问并折叠进 online softmax state。
- 最终结果必须与 full attention reference 对齐。

## 模型边界

当前模型是 correctness model，不是 transport model。

范围内：

- 生成全局 `Q/K/V`。
- 按 domain 配置切分本地 `Q chunk`。
- 按 ring 顺序访问每个 source domain 的 `K/V block`。
- 使用 online softmax 更新 `running_max`、`running_sum`、`output`。
- 与标准 full attention reference 比较 `max_abs_err` 和 `mean_abs_err`。

范围外：

- 真实 P2P send / recv。
- message serialization。
- CUDA / MLX / NPU kernel。
- 性能评估。

## 数据流

每个 target domain 独立执行：

1. 取本地 `Q chunk`。
2. 从自身 domain 开始，按 ring 顺序访问所有 source domain。
3. 每个 source domain 根据自己的 `block_size` 切出多个 `K/V block`。
4. 对每个 block 计算局部 scores 和未归一化 `P @ V`。
5. 使用 online softmax 公式更新状态。
6. 得到该 target domain 的输出 chunk。
7. 所有 target 输出按 sequence offset 拼回全局输出。

## 验证入口

```bash
python3 python/ringattn_kernel_stub.py \
  --report-path reports/ringattn_correctness.json
```

本地 smoke 也会在 `python3` 和 `numpy` 可用时运行 correctness model：

```bash
SKIP_PYTHON_SMOKE=1 bash scripts/run_local_ringattn_smoke.sh
```

## 当前默认 case

- `2domain_uneven_chunks`
- `3domain_uneven_blocks`
- `4domain_small_tail_blocks`

这些 case 覆盖不均分 chunk、不均分 block size、以及 tail block 小于 block size 的情况。

## 通过标准

默认 tolerance：

- `max_abs_err <= 1e-10`
- `mean_abs_err <= 1e-12`

当前模型使用 float64 运行 correctness 对照，以优先验证算法等价性。后续如果要模拟实际 runtime dtype，应增加 float32 / mixed precision tolerance policy，而不是放宽当前数学闭环标准。
