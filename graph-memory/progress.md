# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

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
