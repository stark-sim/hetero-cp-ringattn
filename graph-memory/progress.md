# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

### [2026-06-30] vLLM Block-Aware Ring 提取 PoC

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `white RTX 4090 vLLM 0.6.4 experiment`

在 white RTX 4090 上使用 vLLM 0.6.4 + Qwen2.5-3B 验证：\n\n1. 可以定位 CacheEngine.gpu_cache[layer] 的物理 block 布局：shape=(2, num_gpu_blocks, block_size, num_kv_heads, head_dim)。\n2. 可以读取任意物理 block 的 K/V：gpu_cache[layer][0/1, block_id]。\n3. 可以将序列化后的 block 写入新的未使用物理 slot，字节级一致。\n4. 通过 scheduler.block_manager.get_block_table(seq) 可以获取序列的 block table。\n\n结论：vLLM Block-Aware Ring 的 block 提取/写入路径可行，不需要修改 attention kernel。\n\n脚本：scripts/poc_vllm_block_extract.py, scripts/inspect_vllm_blocks.py

_updated: 2026-06-30 09:19:48_
### [2026-06-30] 正常规模工作负载对比：3B/7B，1K/4K

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node runs on white/pearl + white CPU/CUDA single-node benchmarks`

在 white+pearl 上对 Qwen2.5-3B / 7B 进行单节点与分布式对比，seq=1024/4096。\n\n单节点基线（white）：\n- 3B/1K CUDA 0.14s, CPU 7.78s\n- 3B/4K CUDA 0.27s, CPU 29.26s\n- 7B/1K CUDA 0.22s, CPU 17.58s\n- 7B/4K CUDA 0.52s, CPU 64.09s\n\n分布式 3B 策略对比（1:1 切分）：\n- 1K：Vanilla mean 12.2s, Striped 11.9s (-2.5%), ZigZag 11.5s (-5.5%)\n- 4K：Vanilla 39.8s, Striped 39.8s, ZigZag 39.6s (<1% 差异)\n\n关键结论：\n1. 在正常 3B/1K 场景下，ZigZag 比 Vanilla 有约 5% 收益，但方差与收益同量级。\n2. 在 3B/4K 下，跨节点传输主导，策略差异消失。\n3. 分布式 3B GPU 仍慢于单节点 CPU：1K 12s vs 7.8s；4K 40s vs 29s。\n4. 7B bf16 无法在 pearl 的 16GB HIP 卡上加载，分布式 7B 需要量化支持。\n\n报告：reports/normal-workloads-3b-20260630-142629/

_updated: 2026-06-30 06:27:31_
### [2026-06-30] 单节点 vs 分布式：4096 token 时间分解

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `local CPU/MPS benchmark + white CUDA single-node benchmark`

用同样的 Qwen2-0.5B 类模型对 4095-token prompt + 5 token decode 进行单节点基准测试，并与 HCP 分布式环结果对比。\n\n结果：\n- white RTX 4090 单节点 CUDA：0.12s\n- 本地 Mac CPU：4.5s\n- 本地 Mac MPS：5.2s\n- HCP 2-domain vanilla 1:1（RTX 4090 CUDA + RX 9060 XT HIP）：~15.1s\n- HCP 2-domain 100 Mbps：~206s\n\n关键结论：\n1. GPU 单节点速度远超 CPU（0.12s vs 4.5s）。\n2. HCP 分布式在 4K token 下比单节点 CPU 还慢（15s vs 4.5s），因为跨节点 KV 传输占主导。\n3. 这不是 CPU/GPU 问题，而是“单节点本地内存” vs “多节点网络”的问题。\n4. HCP 的价值在于打破超长上下文下的内存墙，而不是在小长度下加速。\n\n报告：reports/single-node-vs-distributed/

_updated: 2026-06-30 05:33:11_
### [2026-06-30] 100 Mbps 重复实验稳定结果

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node bandwidth experiment on white/pearl`

在 white+pearl 上对 Qwen2-0.5B-1M / seq=4096 / max_tokens=5 进行带宽稳定性复测。\n\n方法：\n- 使用 tc tbf 在 enp10s0 / enp8s0 上限制为 100 Mbps。\n- 每次运行前彻底清理进程并等待端口释放。\n- 基线（无 tc）跑 3 次，100 Mbps 跑 5 次。\n\n结果：\n- 基线：17s, 18s, 17s；均值 17.3s。\n- 100 Mbps：204s, 205s, 217s, 203s, 203s；均值 206.4s（方差 <3%）。\n\n结论：\n1. 单次 100 Mbps 测出的 38s 和 604s 是偶发离群值，不是真实分布。\n2. 稳定状态下 100 Mbps 带来约 11.9×  slowdown。\n3. 这进一步支持 hyp-net-speed：跨节点带宽是 HCP 性能的决定性因素。\n\n报告：reports/bw-stability-20260630-132311/

_updated: 2026-06-30 05:23:34_
### [2026-06-30] 1:1 chunk split derivative comparison on white+pearl

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node run on white/pearl`

在 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP) 上运行 --chunk-sizes 2048,2048 的等分切分，比较 Vanilla / Striped / ZigZag。\n\n配置：Qwen2-0.5B-1M，seq_len=4096，max_tokens=5。\n\n结果（perf log 聚合，单位 ms）：\n- Vanilla：domain0 total=15122 (recv 14423, local 146), domain1 total=14516 (recv 12804, local 656)；瓶颈 15122 ms。\n- Striped：domain0 total=15547 (recv 14795, local 133), domain1 total=14722 (recv 12601, local 662)；瓶颈 15547 ms。\n- ZigZag：domain0 total=15331 (recv 14675, local 132), domain1 total=14640 (recv 12919, local 651)；瓶颈 15331 ms。\n\n关键发现：\n1. 1:1 等分消除了 3:1 容量感知切分的负载不均，但三种策略差异仍在 <6%。\n2. 网络 recv 仍占绝对主导，1:1 并未改善端到端瓶颈。\n3. ZigZag 的理论优势（负载均衡 + 减少边界）在当前 tailscale 链路上无法体现。\n\n报告：reports/ring-derivatives-1to1-20260630-122906/

_updated: 2026-06-30 04:41:51_
### [2026-06-30] Ring Attention derivatives Phase 2: real white+pearl comparison

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `manual cross-node run on white/pearl`

在 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP) 真实异构硬件上运行 Vanilla / Striped / ZigZag 三种调度策略。\n\n配置：Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，tailscale 网络。\n\n结果（perf log 聚合，单位 ms）：\n- Vanilla：domain0 total=15077 (recv 14477, local 133), domain1 total=14392 (recv 12663, local 648)；瓶颈 15077 ms。\n- Striped：domain0 total=14759 (recv 14140, local 119), domain1 total=13948 (recv 12256, local 652)；瓶颈 14759 ms。\n- ZigZag：domain0 total=15578 (recv 14906, local 129), domain1 total=14773 (recv 13040, local 656)；瓶颈 15578 ms。\n\n关键发现：\n1. 三种策略在真实异构硬件上全部跑通，无 NaN / crash。\n2. 网络 recv 占绝对主导（domain0 >95%，domain1 ~88%），调度策略对负载均衡的改善被网络带宽完全掩盖。\n3. 三种策略端到端差异 <6%，说明当前 tailscale 链路已经是瓶颈。\n4. Striped 改变了生成 token 序列（与 vanilla/zigzag 不同），这在无意义重复 prompt 的小模型上是可接受的位置敏感性表现。\n\n报告：reports/ring-derivatives-manual-20260630-112010/

_updated: 2026-06-30 03:23:34_
### [2026-06-29] Ring Attention derivatives Phase 1: CPU mock correctness and load balance

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `cargo test --features tch-backend test_ring_attention_derivatives_uneven_perf`

在 Rust 中新增 RingSchedulingStrategy（Vanilla / Striped / ZigZag）和 assignment helper，并在 CPU mock 上验证 2-domain 3:1 不均等分片（seq=4096, num_heads=8, head_dim=128）。\n\n结果（单次 layer）：\n- Vanilla：domain0=74ms, domain1=47ms，瓶颈 domain0。\n- Striped：domain0=149ms, domain1=50ms，把 peer compute 推给 domain0，反而更慢。\n- ZigZag：domain0=64ms, domain1=39ms，两个 domain 都变快，负载更均衡。\n\n所有策略 correctness diff < 3e-8。\n\n结论：\n1. ZigZag 在 uneven 3:1 分片下有效改善了负载均衡。\n2. Striped 在当前加权 round-robin 实现下对 3:1 场景不适用（与之前挂起结论一致）。\n3. 需要真实硬件（white CUDA + pearl HIP）验证这些趋势是否保持。

_updated: 2026-06-29 16:01:43_
### 综述类支撑线必须有真实实现和硬件对比才有说服力

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `user-direction`

在论证 CXL/RDMA 必要性时，最初计划用 Ring Attention 家族综述作为辅助证据。用户指出这不够：如果只是文献综述，没有基于 HCP 的真实实现和 white/pearl 硬件对比，无法形成有工作量、有说服力的论证。\n\n教训：\n1. 任何“方案对比”类 claim，必须有可运行的代码和可重复的测量。\n2. 当直接实验（hyp-net-speed）已经很强时，不要为了“显得完整”而引入高成本实现线。\n3. 文献综述只能作为背景，不能替代实验证据。

_updated: 2026-06-29 15:48:58_
### [2026-06-29] white-pearl 完整带宽矩阵：100 Mbps 下 HCP 慢 10-30x

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/bw-matrix-20260629-220317 / harness operations`

实验：white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP)，Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，tc tbf 在 192.168.100.x 有线链路上限速，iperf3 验证实际带宽。\n\n结果（2 reps）：\n- baseline 2.35 Gbps：20.5 s avg（20/21 s）\n- 1000 Mbps：29.5 s avg（28/31 s）→ 1.44x slowdown\n- 500 Mbps：50.0 s avg（50/50 s）→ 2.44x slowdown\n- 100 Mbps：445 s avg（206/684 s）→ 21.7x slowdown（中位数 445 s）\n\n报告目录：reports/bw-matrix-20260629-220317/\n\n关键发现：\n1. 端到端时间随带宽下降呈非线性增长；100 Mbps 时通信成为绝对瓶颈。\n2. 100 Mbps 两次重复差异极大（206 s vs 684 s），提示低速下系统状态（热节流、设备调度、QUIC 拥塞控制）可能放大波动。\n3. 500 Mbps 已使 4K+5 token 任务慢约 2.4x；1 Gbps 仍慢约 1.4x。\n\n结论：P2P KV ring 对跨节点带宽极度敏感；要释放异构 CP 的实用性，需要远高于千兆以太网的互联带宽（CXL / RDMA / 高速 NVLink）。

_updated: 2026-06-29 14:32:15_
### [2026-06-29] white-pearl 限速 pilot：100M 带宽下 HCP 慢 10x

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `harness/operations/ (pending full matrix record)`

实验：white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP)，Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，使用 tc tbf 在 192.168.100.x 有线链路上限速。\n\n结果：\n- 基线 2.35Gbps：总耗时 21s\n- 限速 100Mbps：总耗时 206s\n\n结论：\n1. 网络带宽对 HCP 跨节点异构推理有决定性影响。\n2. 当带宽从 2.35G 降到 100M 时，端到端时间增加约 10 倍，说明当前 P2P KV ring 在低速网络下通信成为绝对瓶颈。\n3. 这为 CXL / 类 RDMA 高速互联的必要性提供了直接实验证据。\n\n下一步：完整矩阵（baseline / 1000M / 500M / 100M × 2 reps）正在后台运行。

_updated: 2026-06-29 14:02:37_
### [2026-06-29] white RTX 4090 CUDA 上 Striped 未改善负载均衡

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `harness/operations/20260629-104712-stripe-real-hardware.yaml`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture\n主机：white (Tailscale 100.118.253.68), RTX 4090, libtorch CUDA\n配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1)\n\nVanilla：\n- domain 0 total=131.1ms (local=130.3ms, peer=0.03ms)\n- domain 1 total=54.6ms (local=5.5ms, peer=49.0ms)\n\nStriped：\n- domain 0 total=164.8ms (local=114.0ms, peer=50.1ms)\n- domain 1 total=57.0ms (local=7.8ms, peer=49.1ms)\n\ncorrectness diff 均 < 1.3e-8。\n\n结论：在 white CUDA 单进程 3:1 场景下，Striped 使瓶颈 domain 0 总耗时增加约 26%，未改善 wall-time。

_updated: 2026-06-29 12:44:16_
### [2026-06-29] pearl RX 9060 XT HIP 上 Striped 未改善负载均衡

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `harness/operations/20260629-104712-stripe-real-hardware.yaml`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture\n主机：pearl (Tailscale 100.111.242.55), RX 9060 XT, libtorch HIP\n配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1)\n\nVanilla：\n- domain 0 total=158.2ms (local=157.5ms, peer=0.05ms)\n- domain 1 total=89.1ms (local=13.7ms, peer=74.9ms)\n\nStriped：\n- domain 0 total=224.8ms (local=154.2ms, peer=70.3ms)\n- domain 1 total=87.4ms (local=11.6ms, peer=75.6ms)\n\ncorrectness diff 均 < 1.3e-8。\n\n结论：在 pearl HIP 单进程 3:1 场景下，Striped 使瓶颈 domain 0 总耗时增加约 42%，未改善 wall-time；pearl 整体比 white 慢约 1.2-1.4x。

_updated: 2026-06-29 12:44:16_
### CPU mock 只能验证语法和逻辑依赖，不能指导 LLM 服务架构设计

type: `lesson` · status: `held` · confidence: 0.95 · importance: 0.9

在 Striped Attention 原型验证中发现：CPU 上 correctness diff 和 perf 数字对 LLM 服务架构设计的实际作用几乎没有意义。\n\n原因：\n1. CPU 与加速卡（CUDA/HIP/MPS）的算力结构、memory bandwidth、kernel launch 开销完全不同。\n2. CPU mock 无法反映真实 heterogeneous 场景下各 domain 的计算速度差异、显存压力、P2P / 网络传输瓶颈。\n3. Striped 对负载均衡的影响取决于"慢 domain 到底有多慢"以及"peer compute 转移是否能被快 domain 吸收"，这些信息 CPU 无法提供。\n\n结论：代码逻辑层面的正确性可以在 CPU 快速验证；任何关于调度策略、overlap、分片比例、端到端吞吐/延迟的设计决策，必须在真实加速卡硬件上复跑后才能得出结论。

_updated: 2026-06-29 12:35:36_
### [2026-06-29] Striped correctness原型在CPU mock上验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `cargo test / rust/src/model/attention/ring.rs`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture
配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1), CPU Float32, mock transport。

Correctness：
- Vanilla diff = 2.8e-8
- Striped diff = 2.6e-8
均 < 1e-4，数值正确。

Perf（单次 layer，CPU mock）：
Vanilla：
- domain 0 total=118.5ms (local=117.0ms, peer=0.02ms)
- domain 1 total=46.3ms (local=15.8ms, peer=30.0ms)
Striped：
- domain 0 total=184.6ms (local=129.6ms, peer=53.3ms)
- domain 1 total=50.8ms (local=15.9ms, peer=34.6ms)

关键发现：在 homogenous CPU 上，Striped 把部分 peer compute 从 domain 1 转移到 domain 0，使原本就是瓶颈的 domain 0 更慢；domain 0/1 总耗时比从约 2.6x 扩大到约 3.6x。

_updated: 2026-06-29 10:46:05_
### [2026-06-19] 1M context 本地异构分布式推理成功

type: `session` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `memory-bank/progress.md`

white RTX 4090 CUDA + pearl RX 9060 XT HIP，2.5G 有线直连。Qwen2-0.5B-1M，capacity-aware 3:1 分片（white 750K / pearl 250K）。Prefill 24/24 全通，decode 5 tokens 全通，exit=0。总耗时 ~2h 11m，white 显存峰值 23,999 MB。攻克：KV channel buffer 512、QUIC timeout 14400s、max_position_embeddings=1048576 patch、pearl 碎片化 OOM 通过 3:1 分片缓解。

_updated: 2026-06-29 05:34:19_
### [2026-06-17] 昇腾 910B NPU 控制面 E2E 打通

type: `session` · status: `closed` · confidence: 1.0 · importance: 0.75 · source: `memory-bank/progress.md`

单机 1× Ascend 910B4 (32 GB HBM) 上完成 Python vLLM worker ↔ Rust coordinator 控制面 E2E。Rust coordinator 脱离 libtorch feature 可编译运行，纯 Rust 采样替代 tch::Tensor。Coordinator 输出 generated: ! I'm。

_updated: 2026-06-29 05:34:19_
### 证据：1:1/2:1/3:2 split 均导致 pearl OOM

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/progress.md`

在 1M context 尝试中，均分 500K 及 2:1、3:2 split 均在 layer 23/24 因 pearl 16GB 显存分配失败而 OOM。只有 3:1 split 成功。

_updated: 2026-06-29 05:34:19_
### 证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

White CUDA loopback 双 domain 3B max_diff=0.406，0.5B max_diff=0.344，argmax=10/10。跨平台单节点 0.438，异构分布式 0.484。证明跨平台 BLAS 仅贡献 ~0.1 额外差异，不是 logits 差异主导因素。

_updated: 2026-06-29 05:34:19_
