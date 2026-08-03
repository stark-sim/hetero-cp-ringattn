# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

### Continuation/extend 主流 cache 元数据一手资料审计

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `primary-source-audit-2026-08-03`

[2026-08-03 primary-source audit]
1. Hugging Face Transformers：masking_utils.py 在有 past_key_values 时分别取得 q_offset=get_query_offset(layer_idx) 与 (kv_length, kv_offset)=get_mask_sizes(q_length, layer_idx)。这证明通用 cache attention 不能用 query length/position 推导完整 KV 范围。来源：https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py
2. vLLM：PagedAttention.write_to_paged_cache 用 slot_mapping 只写本轮新 K/V；scheduler 以 request.num_computed_tokens 和 num_tokens_with_spec 计算 num_new_tokens，并显式说明统一覆盖 chunked prefill/prefix caching/decode。物理 cache 地址与本轮 query 工作量是两套元数据。来源：https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/attention/ops/paged_attn.py 与 https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py
3. TensorRT-LLM：KVCacheManager.create_kv_cache 创建 request-owned cache，并通过 radix-tree reuse match 复用历史。来源：https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py
4. SGLang：当前 KNOWN_FAILURES 明确记录 non_monotonic_extend 会破坏假设 out_cache_loc/contiguous K slots 单调的 FA3/FA4、FlashInfer MLA 与 dual-chunk backend。这直接说明 HCP 不能把 layer-assigned decode 后的物理 append 顺序误当成连续逻辑 token 区间。来源：https://github.com/sgl-project/sglang/blob/main/test/registered/attention/unittests/KNOWN_FAILURES.md
5. FlashInfer：prefill/append kernel API 分别接收 qo_indptr 与 kv_indptr/paged_kv_indptr，并用 paged_kv_indices 描述 KV 地址，query/output cardinality 与完整 KV cardinality 是独立输入。来源：https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/prefill.py
审计边界：这些框架主要证明接口原则；它们的 paged allocator、CUDA kernel、radix cache、collective/同构执行环境不能直接替代 HCP 的 capacity-weighted ReservedPositionedKvShard 与 neighbor-only P2P ring。Context7 的 TLS 通过 SSL_CERT_FILE 指向 certifi 后恢复；其 md 选项与当前 API 漂移，改用 txt。三个 Context7 聚合路径返回 404 后，最终证据改用 GitHub tree/raw 当前可访问路径，未把失效链接写成依据。

_updated: 2026-08-03 09:14:32_
### 单 token 本地 prefill shard 的因果回归已验证

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@830a406`

实现提交 830a406 将 LlamaModel attention_mask gate 从 local seq_len > 1 改为 is_prefill，并新增两 worker、两层、1:3 local positions [0]/[1,2,3] 的独立 TCP regression。TDD RED：恢复旧 gate 后 focused test 稳定失败，worker0 first-position max diff=27.771865844726563；GREEN：同 focused test 1 passed、0 failed。完整验证：cargo test --features tch-backend => 110 passed、0 failed、3 ignored，doc tests 0 failed；cargo clippy --features tch-backend --all-targets -- -A warnings、rustfmt --check 两文件、git diff --check 均 exit 0。实现已推送 origin/main。

_updated: 2026-08-03 05:56:23_
### 24 层 Node 4b 越过 initial 后定位 continuation RED

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@830a406`

在同一 causal 语义修复下运行 24 层、两 worker、1:3 的 Node 4b acceptance 草稿：initial prefill first max diff=8.940696716308594e-8，last max diff=1.1920928955078125e-7，last mean diff=4.038214740376134e-8，tokens=65/65；随后 decode 通过，测试首次失败点移动到 continuation logits max diff=0.08049288392066956。结论：local single-token causal 泄漏已与 continuation 问题解耦；0.08049 是 Node 4b 下一轮诊断对象，不属于本 causal checkpoint。

_updated: 2026-08-03 05:56:23_
### Prefill causality 是全局 phase，不是本地长度属性

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.98 · source: `hetero-cp-ringattn@830a406`

分布式 prefill 的因果语义属于全局 forward phase，不能由单 worker 的 local seq_len 推断。capacity-weighted sharding 可以合法产生 local seq_len=1；该 worker 的 Q position 仍必须屏蔽 peer 上更大的 future KV positions。decode 恰好也是 seq_len=1，因此 local length 无法区分 prefill 与 decode；模型已有 is_prefill 状态才是正确 gate。

_updated: 2026-08-03 05:56:23_
### model/backend rustfmt 基线已验证并推送

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `hetero-cp-ringattn@c637298`

实现提交 c637298 仅对 rust/src/model/model.rs 与 rust/src/worker_sdk/tch_backend.rs 建立 rustfmt 基线，未包含 Graph Memory 或其他用户修改。验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 cargo test --manifest-path rust/Cargo.toml --features tch-backend => 109 passed、0 failed、3 ignored，doc tests 0 failed；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -- -A warnings => exit 0；rustfmt --edition 2021 --check 两个文件与 git diff --check => exit 0。提交已推送 origin/main。该 checkpoint 只规范文本，不声称业务行为变化。

_updated: 2026-08-03 05:20:46_
### 同步 TCP submit progress 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@15ce539`

实现提交 15ce539。受控对照：Node 4b 默认 overlap 路径在 buffered framing 修复后仍于 recv_frame 30 秒超时；设置 HCP_DISABLE_OVERLAP=1 让 receive 前 flush 后，测试在 0.07 秒内越过 initial 24-layer ring 并到达数值断言，确认根因是 TCP submit 只进入本地 buffer。RED regression：仅调用 client.submit_send、不调用外层 flush，server 在 100ms 读超时；GREEN：TCP 的 KV/RingPacket/SelfDrivingPacket submit 均在返回前调用既有 flush_send，peer 成功 receive。
Node 4b 默认路径复测：24 层每层双向 KV exchange 在 0.03 秒内完成，稳定到达 initial logits 数值断言，不再网络超时。完整验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 cargo test --manifest-path rust/Cargo.toml --features tch-backend -- --skip reserved_prefill_continues_after_layer_assigned_decode_without_rebuilding_history => 109 passed, 0 failed, 3 ignored, 1 filtered out；同环境 cargo clippy --all-targets -A warnings、rustfmt --check、git diff --check 均 exit 0。
边界：同步 TCP 不再承诺 compute/send overlap；QUIC 后台 send task 与协议不变。本证据不证明 TCP 性能、continuation 数值正确性或真实模型。

_updated: 2026-08-03 03:50:51_
### Split-phase submit 必须真正启动 I/O 才能声称 overlap

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.9 · source: `incident-2026-08-03@15ce539`

症状：双方 submit 后都进入 receive，最后才 flush，形成确定性循环等待。根因：接口命名为 submit 并不等于实现已经启动发送；TCP 只把 bytes 留在调用线程的本地 Vec，没有 writer task。已验证解决：同步 transport 在 submit 内完成 write；异步 transport 由后台 task 取得 buffer ownership。最早预防条件：设计 submit/compute/recv/flush pipeline 时，逐 transport 证明 submit 返回前已经发生可独立前进的 I/O；仅缓存 serialization bytes 不构成 overlap。该条件由 TCP regression 与 Graph Memory lesson 保存，不创建新 skill。

_updated: 2026-08-03 03:50:51_
### TCP poll 半帧恢复与完整回归通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@23e4d0a`

实现提交 23e4d0a。RED：仅发送 length-prefixed KV frame 的前 2 个 header bytes，旧 poll_recv 在 macOS 稳定返回 recv_frame read meta_len failed: Resource temporarily unavailable (os error 35)。GREEN：TcpKvTransport 使用持久 recv_buffer 保存任意半帧，nonblocking poll 只在完整 meta+payload 到齐后解码；补齐余下 bytes 后 layer/range/Int64 position_ids 精确恢复。同步补齐 ring packet poll，并让 self-driving packet poll 共用同一 framing 状态。
验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 cargo test --manifest-path rust/Cargo.toml --features tch-backend tcp_poll_recv_preserves_partial_frame_until_complete -- --nocapture => 1 passed, 0 failed；完整回归以 --skip reserved_prefill_continues_after_layer_assigned_decode_without_rebuilding_history 跳过 Node 4b 故意 RED，结果 108 passed, 0 failed, 3 ignored, 1 filtered out；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -- -A warnings、rustfmt --check、git diff --check 均 exit 0。
边界：未增加最大 frame 配额、生产级背压或协议版本；这是同步 TCP test transport 的 framing correctness，不是性能证据。

_updated: 2026-08-03 03:23:01_
### 非阻塞 TCP frame parser 必须保存部分读取状态

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.9 · source: `incident-2026-08-03@23e4d0a`

症状：poll 无数据或只到达半个 header 时返回平台相关 WouldBlock 错误，且下一次读取可能失去 frame 边界。影响：上层 ring prefill 在 attention 语义执行前失败，错误地掩盖 continuation RED。最小复现：先写 length prefix 的 2/4 bytes，调用一次 poll，再补齐 frame。根因：TCP 是字节流，nonblocking read_exact 可以先消费任意前缀再返回 WouldBlock；字符串匹配只能隐藏错误，不能恢复已消费 bytes。已验证解决：持久 receive buffer + 完整 frame 后解码。最早预防条件：任何对 length-prefixed TCP stream 的非阻塞 poll 都必须有跨调用 decoder state；如果上层需要消息级 try_recv，应先由 reader task/codec 完成 framing。该教训由机械回归测试长期防止，不创建重复 skill。

_updated: 2026-08-03 03:23:01_
### Positioned wire 完成后收敛 continuation 剩余风险

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@0382c24`

旧风险同时包含 wire 丢失与 segment-forward 两个 blocker。commit 0382c24 已消除 wire 部分，因此保留旧节点历史并以 risk-continuation-segment-forward-20260803 替代为当前风险；新风险只描述 request cache 复用以及 Q/KV 位置向量必须分离的问题。

_updated: 2026-08-03 02:47:51_
### TCP/QUIC positioned KV wire 合同验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@0382c24`

实现 commit 0382c24 让 KvBlock optional position_ids 通过 TCP/QUIC KvTransport trait 无损传输：JSON metadata 描述 shape/bytes/dtype，payload 保持原始 Int64，顺序为 K、V、positions；旧 frame 缺失字段时继续解析为 None。范围只涉及两个 transport codec 与测试，没有修改 attention、KV 归属、schedule、backend、runtime 或 coordinator。

TDD：
1. TCP positioned roundtrip 在实现前因接收端 position_ids=None 按预期失败，随后转绿。
2. QUIC positioned roundtrip 在实现前因接收端 position_ids=None 按预期失败，随后转绿。

验证（所有 cargo 命令均使用 LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 前缀）：
1. cargo test --manifest-path rust/Cargo.toml --features tch-backend kv_transport -- --nocapture -> 7 passed, 0 failed；覆盖 TCP/QUIC trait object、[0,9,16777217] Int64 精确往返、旧 frame 缺字段兼容和既有 self-driving packet。
2. cargo test --manifest-path rust/Cargo.toml --features tch-backend test_ring_attention_derivatives_uneven_perf -- --nocapture -> 1 passed；vanilla/striped/zigzag positioned correctness diff 均约 2.6e-8 至 2.8e-8。
3. cargo test --manifest-path rust/Cargo.toml --features tch-backend -> 107 passed, 0 failed, 3 ignored；doc tests 0 failed, 3 ignored。
4. cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -> exit 0，仅既有 warnings。
5. rustfmt --edition 2021 --check 两个 transport 文件、git diff --check、SQLite integrity_check/foreign_key_check -> exit 0 / ok / 无 foreign-key violation。

边界：这是本机 CPU codec/correctness 证据，不是跨主机 wire smoke、MPS/CUDA 完整模型或 continuation segment-forward 证据；risk-continuation-prefill-positioned-wire-20260803 仍保持 open，等待 Node 4b。

_updated: 2026-08-02 19:42:44_
### [2026-08-03] SelfDrivingPacket 统一 transport 合同通过真实跨主机 QUIC

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@58dccc5`

Task 3c 在提交 58dccc5 上完成验证。
1. 本地 transport 回归：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 cargo test --manifest-path rust/Cargo.toml --features tch-backend transport -- --nocapture；结果 6 passed, 0 failed。
2. 本地全量 Rust：同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend；结果 103 passed, 0 failed, 3 ignored。
3. Clippy：同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --lib --bin self_driving_quic_smoke；exit 0，仅有既有 warnings。
4. 本机双进程：server 0.0.0.0:29641 + client 127.0.0.1:29641；两端均报告 layer=7, position=16777217。
5. 真实跨主机：white(stark@inventory-white, commit 58dccc5) 独立 server 监听 0.0.0.0:29642；Mac 独立 client 连接 inventory white 100.118.253.68:29642。Mac 输出 self-driving QUIC client roundtrip ok: peer=100.118.253.68:29642, layer=7, position=16777217；white 输出 server roundtrip ok: peer=100.121.35.138:57639, layer=7, position=16777217。
结论：SelfDrivingPacket 已通过统一 KvTransport trait 在 TCP/QUIC 中保持六个 tensor 的 dtype/值和 route metadata，Int64 position 未经过 f32 丢精度；真实 Mac-white 独立进程 QUIC roundtrip 成立。本证据只证明 transport 合同，不声明真实推理服务完成。

_updated: 2026-08-02 18:34:07_
### [2026-08-02] 现行远程入口已停止使用旧 GPU LAN 地址

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `hetero-cp-ringattn@b2f2753`

实现提交 b2f2753。验证：
1. rg -n old-address AGENTS.md scripts docs/PROTOCOL_SMOKE.md docs/DEPLOYMENT_GUIDE.md 无输出，证明现行规则、脚本和操作文档不再引用旧 endpoint；历史 kimi export、memory-bank/legacy 与 harness inventory 副本未改写。
2. bash -n scripts/run_rust_remote_cp_node.sh scripts/run_rust_remote_cp_3node_smoke.sh scripts/run_rust_remote_p2p_client.sh，exit 0。
3. 未设置 GPU_HOST/CONNECT_ADDR 启动对应脚本均在网络操作前 exit 1，并提示从 ~/.agents/inventory.yaml 解析。
4. ssh -o BatchMode=yes -o ConnectTimeout=8 inventory-white-host 执行严格 set -eu 检查，结果 inventory-endpoint-ok；远端 hostname=white，RTX 4090、~/hetero-cp-ringattn 与 ~/models/Qwen2-0.5B 均存在。
5. git diff --check 通过。
边界：只验证 endpoint 来源和脚本 fail-fast，不声明推理服务或 self-driving runtime 已跨节点完成。

_updated: 2026-08-02 11:39:46_
### 远程 endpoint 漂移应在 inventory 边界被阻断

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.85 · source: `hetero-cp-ringattn@b2f2753`

症状：按项目 AGENTS.md 的固定 GPU 地址连接时 SSH 超时。影响：会把配置漂移误判为节点不可达，并阻断真实异构验证。最小对照：旧规则地址超时，而同一 SSH 用户连接 inventory 中 white.ssh.host 成功，且 CUDA GPU、仓库和模型均存在。根因：项目规则和脚本复制了 infrastructure inventory 中会变化的 endpoint。解决：现行规则只声明 inventory authority；连接脚本要求显式 endpoint 并 fail-fast；文档不再提供旧默认。预防条件：任何远程操作先查 inventory，脚本不得把历史实验地址提升为当前默认。该 lesson 是项目基础设施事实，不创建或修改通用 skill。

_updated: 2026-08-02 11:39:46_
### 真实 Qwen 两 worker decode packet 完成 48 次邻接 TCP hop

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@39ba09a`

实现提交 39ba09a 将 Task 3a 的 ignored真实 Qwen2-0.5B、24层、两-token decode oracle 改为持久loopback TCP数据路径。两个 backend各持一个 TcpKvTransport endpoint；每个 LayerStepOutcome::Forward 都执行 LayerPacket->SelfDrivingPacket、send、recv、SelfDrivingPacket->LayerPacket，hop只在实际send成功后计数，并校验反序列化 layer_idx/current_domain。两 token实际完成48=2×24×(2-1)次TCP hop，累计714940 frame bytes，每帧非空。
核心不变量保持：prefill positions仍为worker0=[0]、worker1=[1,2,3]；schedule counts=[12,36]，最终KV totals=[36,108]=1:3；每层唯一assignee append、position union完整、reserved capacity精确写满。decode 0 max_diff=0.531250、mean_diff=0.083501、tokens=6667/6667；decode 1 max_diff=0.289062、mean_diff=0.051739、tokens=220/220，与Task 3a完全一致。
TDD：RED把oracle改为尚不存在的TCP helper，以E0425失败；GREEN复用现有TcpKvTransport后通过。中间E0616来自测试读取私有LayerPacket.current_domain；按现有wire API改为转换后读取公开SelfDrivingPacket.current_domain，未扩大核心可见性。该一次性编译期误用无重复模式，不单独提升lesson。
验证：cargo test --features tch-backend --lib real_qwen_two_worker_reserved_prefill_decodes_two_self_driving_tokens_over_tcp -- --ignored --nocapture 为1 passed、0 failed；cargo test --features tch-backend --quiet 为101 passed、0 failed、3 ignored且doc tests通过；cargo clippy --features tch-backend --all-targets -- -A warnings exit 0；git diff --check exit 0；tch_backend.rs全文件仍有既有rustfmt债，但本节点新增区间未产生rustfmt hunk。
边界：initial prefill仍使用LinkedMock P2P，只有decode packet经过真实TCP；测试是单进程顺序编排和单请求CPU BF16，不证明独立worker阻塞循环、coordinator、QUIC、跨节点异构硬件、多请求、性能或生产服务。

_updated: 2026-08-02 09:00:54_
### 真实 Qwen 两 worker 完成两 token self-driving decode correctness 闭环

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@59618b9`

实现提交 59618b9：新增 ignored in-process oracle，从两个独立 TchWorkerBackend 的真实 Qwen2-0.5B、24 层、CPU BF16 reserved prefill 状态继续执行两个 self-driving decode token。prefill positions 为 worker0=[0]、worker1=[1,2,3]；冻结 schedule tickets=[1,3]，48 个 layer-token append events 精确计数 [12,36]。每 token 24 个网络等价 hop，即每层 N-1=1；每层仅唯一 assignee append，本地 shard position union 完整且无重复；最终跨层 KV totals=[36,108]=1:3，所有 shard 精确写满 reservation。
数值结果：decode 0 max_diff=0.531250、mean_diff=0.083501、tokens=6667/6667；decode 1 max_diff=0.289062、mean_diff=0.051739、tokens=220/220。两轮 argmax 均与 contiguous reference 精确一致。
验证：聚焦 ignored test real_qwen_two_worker_reserved_prefill_decodes_two_self_driving_tokens 为 1 passed、0 failed；完整 cargo test --features tch-backend 为 101 passed、0 failed、3 ignored；cargo clippy --features tch-backend --all-targets -- -A warnings exit 0；rustfmt --check self_driving.rs、git diff --check 均 exit 0。tch_backend.rs 全文件仍有既有 rustfmt 债，但本节点新增测试区间未产生新的 rustfmt hunk。
边界：只证明本机 libtorch CPU 的真实 BF16、两 worker、in-process correctness；不证明 coordinator、TCP/QUIC、跨节点异构硬件、多请求、性能或生产服务。

_updated: 2026-08-02 06:12:35_
### BF16 prefill 的 max-diff 门槛不能无证据推广到多层 decode continuation

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.92 · source: `hetero-cp-ringattn@59618b9`

症状：把真实两-worker prefill oracle 的 max logits diff<0.5 经验门槛直接沿用到 24 层 decode continuation 后，decode 0 稳定得到 0.531250 而失败；重复运行结果一致。临时只检查 finite 后，第二 token、argmax、position union、唯一 assignee append、reservation 和 capacity-weighted totals 全部通过。
根因边界：prefill 的一次分块 online-softmax 数值包络不能直接代表 24 层跨 backend continuation 的累计 BF16 重排误差；0.531250 本身不证明 ownership 或 layer 数据流错误。
修订：真实 decode oracle 以 argmax exact 和 mean diff<0.1 为主要数值合同，并保留 max diff<0.75 作为当前固定 CPU BF16 oracle 的失控保护。0.75 不是跨硬件数学保证，也不取代旧 prefill oracle 的 max<0.5 门槛。后续跨设备应先收集各硬件分布，再分别校准 guard，不把单一 max 阈值写成算法正确性定理。

_updated: 2026-08-02 06:12:35_
### [2026-08-02] Graph Memory SQLite 写入 transport 规则与回归已验证

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `local-graph-memory-skill-2026-08-02`

数据库诊断：PRAGMA integrity_check=ok，foreign_key_check 无违规，nodes 与 FTS 外部内容映射计数一致且无 missing/orphan/content mismatch；数据库副本上的 FTS5 integrity-check 与 quick_check 均通过。
根因裁定：SQLite 页、事务和索引未损坏；曾经损坏的是交互 PTY 在 SQLite 接收前传输的 Unicode 多行 evidence 正文，当前第 3、4 条验证记录已修正且 UTF-8 原始字节正常。
修复：graph-memory skill 新增 SQLite Write Transport，强制参数绑定或 non-PTY stdin、禁止交互 PTY 与字面反斜杠换行、失败后先查稳定 ID 再重试、写后校验 exact content/hash/哨兵与换行数；tests/run.sh 新增合同断言和中文多行精确 roundtrip。
验证：bash /Users/stark_sim/.agents/skills/graph-memory/tests/run.sh => PASS；python3.11 quick_validate.py /Users/stark_sim/.agents/skills/graph-memory => Skill is valid；Unicode roundtrip exact equality=1/newlines=4；skill SHA256=2902ba421cda2a2bdecb84ae0daf46a1bd801c850e5bb5a27f8a44b70169aa3e，test SHA256=3327f114fb2b82498a79ff105b8e70a2e0154e0c692c65bbe2d68512c84133a1。shellcheck unavailable，未声明 shellcheck 通过。

_updated: 2026-08-01 21:33:21_
### Graph Memory 长文本正确性必须在 SQLite 上下游两侧验证

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `ev-graph-memory-sqlite-write-transport-hardening-20260802`

SQLite integrity_check、事务提交和节点存在只证明数据库结构及收到的字节可持久化，不证明上游输入在 PTY 中没有被串改。多行、长文本或非 ASCII mutation 必须用参数绑定或 non-PTY stdin；禁止交互 PTY 和把字面反斜杠 n 当换行。失败后先查稳定 ID 与事务结果，防止盲目重放；成功后必须回读 exact content/hash，或至少验证多个确定性哨兵与换行数，再导出和提交。此前仅保留项目 lesson、不修改通用 skill 的边界已被本次用户裁定与 skill 缺口审计取代。

_updated: 2026-08-01 21:33:21_
### [2026-08-02] 真实 Qwen 两 worker 1:3 reserved prefill 通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@c465244`

提交 c465244 新增 ignored integration oracle：本地 Qwen2-0.5B、24 层、CPU BF16，两个独立 TchWorkerBackend 每层仅持自己的 ReservedPositioned KV，worker0 positions=[0]、worker1 positions=[1,2,3]，每层 union 恰好覆盖 4-token prompt；worker1 每层经 predecessor P2P 接收一个 3584-byte KV block。末位置 reference/distributed argmax 均为 token 198，max logits diff=0.25，mean diff=0.042441，落在既有同构 BF16 online-softmax 重排包络内。真实 ignored test 1 passed；完整 cargo test --features tch-backend 101 passed、0 failed、2 ignored；cargo clippy --features tch-backend --all-targets exit 0，仅既有 warnings；git diff --check 通过。该证据证明 backend-level 多 worker initial prefill 和 local ownership，不证明 coordinator 已启用 reservation 或 self-driving decode runtime。

_updated: 2026-08-01 21:22:10_
### BF16 分布式 prefill oracle 必须沿用已验证数值包络

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.9 · source: `hetero-cp-ringattn@c465244`

症状：真实两 worker prefill 首跑因临时设定 mean logits diff<0.01 失败。影响：正确的 worker-local/P2P 实现被误判。复现：实际 max=0.25、mean=0.042441，但 reference/distributed argmax 同为198。根因：把 FP32 风格 mean 阈值套到 BF16 online-softmax 分块重排，忽略项目已有同构 BF16 max约0.3-0.4证据。修正：先输出 max/mean/argmax，确认 token exact 且误差落入既有包络；oracle 使用 argmax exact为主，max<0.5、mean<0.1作为失控门。预防条件：新增 BF16分布式测试前先查询项目已验证 tolerance，不凭局部直觉发明更严阈值。

_updated: 2026-08-01 21:22:10_
### [2026-08-02] Prefill 支持显式逐层 local KV reservation

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@b4db994`

实现提交 b4db994：WorkerCommand::Prefill 增加 optional layer_kv_capacities；WorkerBackend 增加默认回退的 prefill_request_with_reservation；WorkerRuntime 透传；TchWorkerBackend 在修改 request/model 状态前校验 layer count 和 capacity>=local prompt len，并按实际 model dtype 创建 ReservedPositioned caches；coordinator 当前显式发送 None，旧路径不变。TDD RED：focused test 因缺少 prefill_request_with_reservation 以 E0599 失败。GREEN：24 层不均 capacities 和 layer 7 under-capacity 原子拒绝 1 passed；protocol bincode roundtrip 1 passed；Tch backend suite 2 passed；完整 cargo test --features tch-backend 101 passed、0 failed、1 ignored；cargo clippy --features tch-backend --all-targets exit 0，仅既有 warnings；git diff --check 通过。证据不包含 coordinator capacity 计算、多 worker P2P prefill 或真实 OOM 恢复。

_updated: 2026-08-01 21:10:25_
### [2026-08-02] 真实 Qwen initial prefill 原地写入 reserved positioned KV

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@d86ac47`

实现提交 d86ac47：KvCache 增加默认 no-op prepare_positions；KvCacheImpl 增加 ReservedPositioned adapter；LlamaModel 每层 forward 前提供绝对 position_ids；ReservedPositionedKvShard 原地 append 并返回 committed active view，keep=false 的 legacy decode 路径明确拒绝。验证：focused synthetic prefill 1 passed；self-driving 20 passed、1 ignored；完整 cargo test --features tch-backend 100 passed、0 failed、1 ignored；手动 ignored 真实模型测试 real_qwen_prefill_matches_reserved_positioned_cache 1 passed，覆盖本地 Qwen2-0.5B、24 层、CPU BF16、reserved/contiguous logits 与最后 token argmax、每层 K/V 内容、positions、committed length 和预留容量；cargo clippy --features tch-backend --all-targets exit 0，仅既有 warnings；git diff --check 通过。证据只证明单 worker 真实 prefill cache 格式闭环，不证明多 worker P2P prefill、完整服务或硬件性能。

_updated: 2026-08-01 20:58:52_
### [2026-08-02] 本地真实 Qwen BF16 单节点基线可运行

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.85 · source: `local-command-2026-08-02`

在 mac-local-shell + libtorch CPU 运行 cargo run --features tch-backend --bin hcp-ringattn-rust -- --infer-model-dir models/Qwen2-0.5B --infer-prompt Hi --infer-max-tokens 1 --infer-temperature 0 --infer-top-p 1 --infer-num-domains 1。实际加载 24 层 Qwen2-0.5B BF16 safetensors，完成 prefill 与 1 token generation，输出逗号，exit 0。该证据只确认本地真实权重/BF16 kernel 基线可用，不证明 reserved cache 或分布式服务。

_updated: 2026-08-01 20:42:55_
### [2026-08-02] Reserved positioned KV 支持显式运行 dtype

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.98 · source: `hetero-cp-ringattn@b6902ba`

实现提交 b6902ba 为 ReservedPositionedKvShard 增加 new_with_kind(config, capacity, device, kind)，现有 new 保持 Float 兼容并委托新构造器。BF16 storage 可原地 append BF16 K/V，active view 保持 BF16；shape/device/dtype mismatch、position、capacity 和 committed prefix 语义不变。TDD RED：focused test 以 E0599 缺少 new_with_kind 失败。GREEN 验证：reserved_positioned_kv_accepts_explicit_runtime_dtype 1 passed；model::self_driving::tests:: 19 passed；完整 cargo test --features tch-backend 99 passed、0 failed；cargo clippy --features tch-backend --all-targets exit 0，仅既有 warnings；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check 通过。证据为本地 libtorch CPU correctness，不是硬件性能。

_updated: 2026-08-01 20:38:25_
### Graph Memory 大段 Unicode SQL 不经交互 PTY 写入

type: `lesson` · status: `superseded` · confidence: 1.0 · importance: 0.8 · source: `incident-2026-08-02-graph-memory-sql-transport`

症状：通过交互式 sqlite3 PTY 写入包含多行中文与长命令的 evidence 时，终端回显出现重复/截断，查询确认第 3、4 条验证记录被串坏。影响：节点状态和边已提交，但证据正文不可信。根因边界：SQL 事务本身正确，损坏发生在大段 Unicode 文本经交互 PTY 传输；此前把含字面反斜杠换行的 SQL 作为 shell 参数也被 SQLite 拒绝且零写入。已验证解决：改用非交互单行 sqlite3 命令，通过 char(10) 在 SQLite 内构造换行；写后用 instr 检查预期哨兵存在、损坏片段不存在，再运行外键与悬空 evidence 检查。预防条件：Graph Memory 的结构化长文本写入避免 PTY；优先非交互 SQL/项目脚本，并把正文读回校验后再导出和提交。证据仅支持项目级操作 lesson，不修改通用 skill。

_updated: 2026-08-01 21:33:21_
### [2026-08-02] ReservedPositionedKvShard 提升为 experimental core API

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.98 · source: `hetero-cp-ringattn@91df8c4`

实现提交 91df8c4 将 ReservedPositionedKvShard 与 process_layer_packet_with_reserved_history 从 cfg(test) 抽取到正常 tch-backend 编译路径，visibility 保持 crate 内部；新增 core API 测试证明 committed prefix、global positions、active K/V view 和 adapter 符号可用。保留 legacy process_layer_packet 的 Tensor::cat，不修改 KvCache trait、runtime、allocator、placement 或多请求路径。

验证：
1. LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend reserved_positioned_kv_core_api_preserves_committed_prefix -- --nocapture -> 1 passed。
2. 同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend model::self_driving::tests:: -- --nocapture -> 18 passed。
3. 同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend -> 98 passed, 0 failed。
4. 同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -> exit 0，只有既有 warnings。
5. rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check -- rust/src/model/self_driving.rs -> exit 0。

边界：cargo fmt --manifest-path rust/Cargo.toml --all -- --check 仍因仓库大量既有未格式化文件失败，未做无关格式化。当前证据是本地 libtorch CPU correctness/compile 证据，不是 MPS/CUDA/HIP 性能或跨节点硬件证据。

_updated: 2026-08-01 20:18:21_
### Rust 核心框架恢复审计定位 test-only positioned KV 边界

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `analysis-2026-08-02`

只读检查 git diff、rust/src/model/self_driving.rs、rust/src/model/cache.rs、transport symbols、Tensor::cat callers 和核心 Graph 节点。证据：最小核心 task 已在 checkpoint 13 关闭；24-layer exact slab 与 two-token reserved TCP 已分别验证；ReservedPositionedKvShard 和 process_layer_packet_with_reserved_history 仅定义在 self_driving.rs 的 cfg(test) 模块；公开 process_layer_packet 接收 (Tensor,Tensor) 并在 assignee 通过 Tensor::cat 增长；现有 KvCache trait 不携带 global positions，直接改造会扩大到通用 cache/runtime。工作区的 placement.rs 是用户未提交的 production placement/ledger 草稿，明确只读并排除。结论：不再补重复规模测试；下一最小框架能力是把已验证的 reserved positioned shard 和 adapter 提升为 experimental core API，同时保留 legacy 路径。

_updated: 2026-08-01 20:01:44_
### HCP 方法草稿中文版完成并与英文稿结构对齐

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@d2562b3`

提交 d2562b3 新增 docs/paper/HCP_METHOD_DRAFT_ZH.md。验证：公式块与英文稿逐字 diff exit 0；中文版 h2=11、h3=22、fences=6、math_open=27、math_close=27；证据标签 method=5、proved=6、prototype=2、open=2、boundary=1；必需章节、禁用论断、尾随空白和 staged diff check 均 exit 0。未新增摘要、结果、最终结论、旧 1M 结果或后端工程适配。

_updated: 2026-08-01 19:58:15_
### HCP 方法与数据流论文草稿完成并通过边界审计

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@4179a50`

提交 4179a50 新增 docs/paper/HCP_METHOD_DRAFT.md。草稿覆盖问题定义、系统模型、capacity-weighted ownership、完整 prefill/decode/continuation 数据流、online-softmax、N-1 hops、positioned mixed-history、正确性与复杂度、限制和 Evaluation Design；用 Method claim、Proved invariant、Prototype evidence、Open empirical question 区分论断层级。验证：git diff --cached --check exit 0；必需章节检查 exit 0；Markdown fences=6、math_open=27、math_close=27；旧 1M/旧后端工程适配/硬件结果禁用词审计 exit 0；SQLite integrity_check=ok 且 foreign_key_check 无输出。暂存检查曾发现新文件 EOF 多余空白行；根因是普通 git diff --check 不包含未跟踪文件，删除空行并重新暂存后 cached check 通过。该一次性事件不升级为新规则，后续新文件提交继续以 staged diff check 作为提交门。

_updated: 2026-08-01 19:46:50_
### 两-token localhost TCP ring 已使用 exact reserved KV slab

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@50465da`

实现 commit 50465da。公开 process_layer_packet 保留 legacy Tensor::cat；仅提取私有 continue_layer_packet 作为 KV 已提交后的共同 attention/finish 原语，并在 cfg(test) 中新增 reserved history adapter。现有 N=3、L=2、两-token localhost TCP worker 按冻结 assignees [[2,1],[1,0]] 为每个 layer×domain 精确预留初始 history + 该层 growth event 数，初始 K/V 一次性 copy 入 positioned slab，decode growth 通过 narrow().copy_() 原地 append，attention 读取 committed view。RED：在原 cat worker 上新增 event pointer-stability 断言，定向测试在 assignee growth 处失败，K/V storage 地址发生变化。GREEN：每个 TCP event 的 K/V data_ptr 前后相同，committed_len 不超过 reservation；第二 token 后所有 layer×domain slab 精确写满。原有 8=2 tokens×2 layers×(3-1) 次 send、四条 successor routes、唯一 assignee 增长、finisher-to-starter continuation、hidden/logits 最大误差小于 4e-4 和 greedy token 对齐均保持。两-token distributed worker 区间无 Tensor::cat；reference dense history 仍用 cat。验证：聚焦测试 1 passed/0 failed；self-driving 17 passed/0 failed；LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend 为 97 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy exit 0，仅仓库既有 warning；rustfmt --edition 2021 --check 与 git diff --check 均 exit 0。边界：只证明 fixed N=3/L=2/two-token localhost CPU 的 reserved TCP 变体；capacity-weighted 1:3:2 仍由 24 层实验承担，不证明公共 cache path、24 层 TCP、开放式生成、runtime、QUIC、远端硬件或 GPU 物理显存。

_updated: 2026-08-01 19:03:42_
### 旧实现专用的大规模实验 artifact 与结果节点已移除

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@59e9944`

提交 59e9944 删除旧实现专用的计划、报告、展示页面、图表和 Graph Memory 重建脚本，并从 blueprint、scaling 文档与 legacy memory 中移除结果引用。验证：精确旧标识 rg 无输出；目标 Graph nodes/edges 计数均为 0；SQLite integrity_check=ok、foreign_key_check 无输出、孤立 evidence 计数 0；graph-memory/migrate.py AST parse 通过；Graph 导出前后五个视图哈希一致；git diff --check 通过。未跟踪的相关二进制资产和生成工作区已移入系统废纸篓。

_updated: 2026-08-01 18:46:47_
### 冻结 schedule 的完整 horizon reservation 合同验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@d1124a8`

实现 commit d1124a8，仅修改 self_driving.rs 的测试合同，FrozenKvAssigneeSchedule 算法与 API 未变。RED：将旧前缀误差小于等于 1 检查推广到 tickets=[1,1,1]、7 units 的全部 phase，定向测试在 phase=1、prefix=5、domain=0 以 scaled error=8 > total_units=7 失败，复现既有数学反例。GREEN：删除单一 [1,3,2] 样例中的 scaled prefix-error 断言，改为对 N=1/2/3/4、零容量节点、多组 tickets/horizon 和全部 phase 枚举每个 prefix；每域 consumed 始终小于等于完整 horizon counts reservation，遍历结束 consumed 精确等于 reservation，越过 horizon 返回 None。同 request 确定性、不同 request phase、capacity-weighted 完整 counts 和零容量排除仍保留。验证：聚焦 frozen_kv_assignee_schedule 3 passed/0 failed；self-driving 17 passed/0 failed；LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend 为 97 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy exit 0，仅仓库既有 warning；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 和 git diff --check 均 exit 0。结论边界：counts 是已知完整 horizon 的 event reservation 上界；不证明 byte admission、开放式生成、运行期扩容、多请求吞吐平滑、GPU 物理显存或 production allocator。

_updated: 2026-08-01 18:14:03_
### Rust 24 层 positioned KV 精确预分配 slab 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@f6da957`

实现 commit f6da957。固定实验 N=3、L=24、tickets=[1,3,2]，同一请求执行 prefill_1(6) -> decode_1(1) -> continuation prefill_2(6) -> decode_2(1)。分布式侧每个 layer×domain 按冻结四阶段计划一次性精确预留 K/V tensor，prefill 和 decode 均通过 write cursor + narrow().copy_() 原地 append，attention 只读取 committed prefix；所有 storage data_ptr 跨四阶段保持不变，每个 slab 最终 committed_len 等于 reservation，最终 domain KV slot 总数仍为 [56,168,112]=1:3:2。独立 slab 测试确认分段 append 内容正确，overflow 在写入前拒绝，cursor、positions、K/V 内容和 storage pointer 均不变。24 层四阶段 hidden/logits 继续与独立 dense GQA + persistent ContiguousKvCache 参考在 1e-3 内对齐，continuation prefill 仍只投影 6×24=144 个新位置。验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend，96 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend exit 0，仅仓库既有 warning；rustfmt --edition 2021 --check rust/src/model/self_driving.rs、git diff --check 均 exit 0；reserved slab/prefill/decode 实现区间无 Tensor::cat。边界：test-only、in-process CPU correctness；未改生产 cache trait，不证明 allocator、admission、runtime、网络、多请求、GPU 物理显存或开放式生成 reservation。

_updated: 2026-08-01 17:00:05_
### 修订 HCP 硬件叙事：从 CXL-class 定点转向通用高速 P2P fabric

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-02`

用户补充 HCP 的价值不应局限于 CXL。修订后，HCP 的算法合同是 transport-agnostic neighbor P2P ring；CXL 只是可能承载该合同的一类互联，与 RDMA/RoCE、InfiniBand、UALink、PCIe peer access 和未来通用高速互联并列。最高层系统故事从特定硬件押注改为：HCP 让异构设备在同一请求和同一逻辑 context 内进行 capacity-weighted 细粒度协作，网络性能提升决定该能力何时从可行走向经济。同步收紧证据：不同品牌与代际设备不能合作不是绝对事实；主流方案是否主要局限于同构并行组需要相关工作审计。低成本也仍需 TCO 和性能数据证明。

_updated: 2026-08-01 16:51:10_
### 修订 CXL 系统论断：区分已证实的带宽瓶颈与待验证的硬件充分性

type: `revision` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `evidence-boundary-review-2026-08-01`

旧假设 hyp-net-speed 将网络敏感性与 CXL/类 RDMA 的潜在收益写在同一结论中。现根据论文系统定位拆分：belief-net-speed-bottleneck-20260629 继续承载已由带宽矩阵支持的事实性结论；hypothesis-hcp-cxl-class-economic-enabler-20260801 单独承载未来互联可带来系统经济性的待验证假设。CXL 不等同普通网卡，论文采用 CXL-class 或 memory-semantic high-bandwidth low-latency P2P fabric 的硬件类别表述，并明确尚无真实 CXL-class 实机证据。

_updated: 2026-08-01 14:42:10_
### 修订论文核心：HCP 是覆盖完整推理生命周期的异构上下文并行

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-01`

用户纠正旧候选把 phase-adaptive ring 误当成最高层贡献。经 blueprint、SCALING_ARGUMENT 与现有 prefill/decode 证据核对，HCP 的中心对象始终是逻辑 attention context，其主要物理载体为逐层 KVCache。Prefill 的 position/context shard 与 decode 的 layer×position shard 都是在异构集群上分割同一个 context；环上传输 KV 或 Q/O/LSE 只是不同阶段实现该 CP 的数据流选择。核心算法不包含 vLLM、context-passing connector 或其他工程适配。

_updated: 2026-08-01 13:31:11_
### 自驱动 decode ring 模块化核心闭环审查通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@96ec868`

只读核对 rust/src/model/self_driving.rs 与 rust/src/model/transport/tcp.rs：schedule ordinal 为 token_offset×num_layers+layer_idx；reserved slab 通过 narrow().copy_() 原地 append 并读取 committed prefix；reserved TCP adapter 与 legacy 路径共用 continue_layer_packet；TCP worker 每层只在 starter 本地创建 packet，其余从 predecessor 接收，finisher hidden 原地启动下一层或下一 token；wire frame 序列化 layer_idx、route 与 residual/normalized/position/Q/O/LSE，不包含历史 KV。新鲜验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend model::self_driving::tests:: -- --nocapture。结果：17 passed、0 failed、80 filtered out；仅既有 warnings。本机 libtorch CPU correctness，不是性能或加速器物理显存证据。未发现层数与 TCP/reservation 间新的不可分解耦合，因此 24 层 TCP 是重复规模验证，不是当前必要证据。

_updated: 2026-08-01 11:33:57_
### 恢复会话后写 Graph Memory 前先查询目标 ID

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.8 · source: `incident-2026-08-01`

症状：交接摘要称审查节点尚未创建，但当前 graph.db 已含同名完成态节点；直接事务插入在重复 edge 唯一键处失败并回滚。根因：摘要落后于源数据库。已验证解决：先查询 nodes/edges，使用新的模块化复审 task ID，避免覆盖既有历史。预防条件：恢复会话时即使已有摘要，也必须把 graph.db 当源数据，在插入稳定 ID 前查询节点和关系是否存在。此规则属于 Graph Memory 项目事实，不修改通用 skill。

_updated: 2026-08-01 11:33:57_
### Rust 24 层 positioned KV 四阶段复用验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@5e95af1`

实现 commit 5e95af1。N=3、L=24、capacity tickets=[1,3,2] 的同一请求完成 prefill_1(positions 0..5) -> decode_1(position 6) -> continuation prefill_2(positions 7..12) -> decode_2(position 13)。分布式侧使用每个 layer×domain 的显式 position shard 和现有 online-softmax；独立参考侧使用标准 dense GQA + 持久 ContiguousKvCache。四阶段 hidden/logits 与参考误差均低于 1e-3，两次 decode 采样一致；每层最终 position union 严格为 0..13，无遗漏无重复；prefill_2 只投影 6×24=144 个新位置；两轮 decode append 各为 [4,12,8]、合计 [8,24,16]；最终跨 24 层的 domain KV slot 数为 [56,168,112]=1:3:2。验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend --quiet，95 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets --quiet exit 0，仅既有 warning；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check exit 0。边界：in-process CPU correctness；未处理 Tensor::cat、预分配、schedule 平滑、网络/runtime 或多请求。

_updated: 2026-08-01 07:38:02_
### 自驱动 decode ring 核心闭环代码审查与聚焦回归

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@5681216`

代码核对：self_driving.rs 的 LayerPacket 显式携带 residual、normalized、position_ids、Q、O/LSE 和 route；process_layer_packet 在唯一 assignee 追加 current K/V，所有 domain 合并 local partial，最后 domain 唯一执行 W_o+residual+post norm+MLP；两 token TCP 测试中末层 finisher 原地 logits+argmax+embedding 并成为下一 token starter。SelfDrivingPacket wire header只有 layer_idx/assignee/route，没有 request_id/token_offset；worker loop 依赖单请求固定循环顺序。KV append 仍使用 Tensor::cat，故不包含远端历史 KV，但未证明物理显存峰值 hard bound。新鲜验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend model::self_driving::tests:: -- --nocapture；结果 14 passed、0 failed、80 filtered out。仅为本地 libtorch CPU correctness，不是 MPS/CUDA/HIP 或性能证据。

_updated: 2026-07-31 19:13:51_
### request phase 旋转会打破普遍的一单位前缀偏差界

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `algorithm-exhaustive-check-2026-08-01`

按 FrozenKvAssigneeSchedule 当前算法穷举得到最小反例：capacity tickets=[1,1,1]，total_kv_units=7 时 counts=[3,2,2]、smooth sequence=[0,1,2,0,1,2,0]；phase=1 的前 5 项为 [1,2,0,1,2]，domain 0 实际 count=1，目标=5*3/7，绝对偏差=8/7>1。完整 7 units 遍历仍回到 [3,2,2]。因此反例只否定任意旋转前缀小于等于 1 的普遍声明，不否定完整 horizon capacity-weighted counts。

_updated: 2026-07-31 19:13:51_
### Rust 两 token TCP 自驱动 continuation 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@b237266`

实现提交 b237266 仅修改 rust/src/model/self_driving.rs 的 cfg(test) 实验。既有 N=3、L=2 localhost TCP 测试扩为两个连续 decode forward：token 0 routes=1->2->0、0->1->2，末层 finisher domain 2 唯一计算 logits 与 greedy argmax，并用本地 embedding 权重产生下一 hidden；token 1 无边界消息，直接由 domain 2 启动，routes=2->0->1、1->2->0，末层 finisher/sampler 轮转到 domain 0。FrozenKvAssigneeSchedule 使用 tickets=[1,3,2]、request_id=2、4 append units，counts=[1,2,1]，token×layer assignee=[[2,1],[1,0]]；每个事件断言只有对应 domain KV +1。position_ids 为 history_len 与 history_len+1。参考侧为每层显式投影并追加 token 0 current K/V，再计算 token 1；两个 token 的 hidden/logits 最大差均低于 4e-4，greedy sampled token 完全一致。发送数精确为 2 tokens*2 layers*(3-1)=8，证明 finisher 原地 sampling/embedding 没有增加 token-boundary hop。新鲜验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend two_token_tcp_ring_continues_from_finisher_with_scheduled_assignees -- --nocapture => 1 passed, 0 failed；同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend => 94 passed, 0 failed, 3 doctests ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -- -A warnings => exit 0；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check => exit 0。边界：仅固定两层、两个 token、greedy、localhost CPU；不声明 EOS、随机采样、多请求、用户 token 回传、QUIC、远端硬件或性能。

_updated: 2026-07-31 17:55:51_
### Rust 两层 TCP ring 已消费冻结 KV assignee schedule

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@271de7f`

实现提交 271de7f 仅修改 rust/src/model/self_driving.rs 中既有两层 localhost TCP 实验：删除手写 assignees=[2,1]，改由 FrozenKvAssigneeSchedule::new([1,3,2], request_id=1, total_kv_units=2) 生成。largest-remainder 在两层 horizon 得到 counts=[0,1,1]；phase=1 后 token 0 的 layer 0/1 assignee 为 [2,1]。这两个值进入 LayerPacket::start，并由既有逐 worker kv_before/kv_after 断言确认每层仅 schedule 指定 domain 增长一个 KV token；所有其他节点保持原长度。既有路径断言同时继续验证 layer routes=1->2->0 与 0->1->2、总 sends=2*(N-1)=4、layer 0 finisher 原地启动 layer 1、唯一末层 final logits、hidden/logits 对齐未切分参考。新鲜验证：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend two_layer_tcp_ring_uses_scheduled_assignees_and_produces_final_logits -- --nocapture => 1 passed, 0 failed；同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend => 94 passed, 0 failed, 3 doctests ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -- -A warnings => exit 0；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check => exit 0。证据边界：仅为 N=3、两层、单 token、localhost CPU 实验；capacity tickets 仍是输入，不证明 byte admission、物理 reservation、多 token/request、远端硬件或性能。

_updated: 2026-07-31 16:50:14_
### Rust 冻结 capacity-weighted KV assignee schedule 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@cfe25d9`

实现提交 cfe25d9 仅修改 rust/src/model/self_driving.rs，新增纯 FrozenKvAssigneeSchedule 与两个单元测试。schedule 以 append_ordinal=token_offset*num_layers+layer_idx 展平 KV append event；largest-remainder 将 capacity tickets 量化为 total_kv_units 内的精确整数份额，smooth weighted sequence 平滑交织 assignee，request_id 仅旋转 phase。验证覆盖：[1,3,2] 在 24 units 上精确计数 [4,12,8]；同 request 稳定；不同 request phase 不同但份额不变；任意前缀比例误差不超过一个 KV unit；零容量节点不获分配；N=1/2/4；空 worker、全零 capacity、零 units 与越界索引。新鲜验证命令与结果：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend frozen_kv_assignee_schedule -- --nocapture => 2 passed, 0 failed；同环境 cargo test --manifest-path rust/Cargo.toml --features tch-backend => 94 passed, 0 failed, 3 doctests ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --all-targets -- -A warnings => exit 0；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check -- rust/src/model/self_driving.rs => exit 0。证据边界：只证明纯 schedule 的确定性、容量份额和 API 边界；尚未接入 TCP runner，不证明物理 KV reservation、动态迁移、吞吐均衡、并发 runtime 或远端硬件性能。

_updated: 2026-07-31 15:27:30_
### 未提交 production placement 草稿的范围审计

type: `evidence` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `working-tree-audit-2026-07-31`

只读审计确认 rust/src/distributed/placement.rs 为 1079 行未跟踪文件，并依赖 rust/src/capacity.rs 的新 largest-remainder helper、distributed/mod.rs 导出及 Cargo/attention 配套工作树修改。草稿数据模型含 WorkerKvProfile、RequestDemand、RequestPlacementPlan、KvReservationLedger；算法含 byte hard bounds、active persistent+max workspace ledger、optional attention rate、prompt/decode calendar、逐层 K/V granularity 计费、integer repair 和 stable placement hash。完整 Rust 回归能编译并运行其中测试，但当前 synthetic localhost 实验没有真实 profile/admission 输入，因此测试通过不能证明生产显存合同。该证据只支持范围判断，不否定草稿未来价值。

_updated: 2026-07-31 09:52:06_
### Rust 两层 TCP ring 末层 finisher 本地唯一 final logits 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@6ef5a18`

实现提交 6ef5a18 只扩展既有 N=3、两层、单 token localhost TCP 实验测试，没有修改生产 runtime、packet 或 transport 合同。每个 worker 保留本地完整 LlamaModel；layer 0 route=1->2->0，domain 0 原地续 layer 1 route=0->1->2。仅 domain 2 的末层 Finished 分支调用既有 project_final_logits，在本地执行 final RMSNorm+独立 LM head；其他 worker 返回 None，机器断言 logits producer 数量恰为 1 且 domain 与 final hidden producer 相同。logits 与未切分两层参考 max diff<4e-4；原发送计数断言仍为 4=2*(N-1)，因此 final head 不增加网络 hop，logits 不进入 ring。TDD：有效 RED 在 logits_outputs.len() 得到 left=0/right=1；最小 GREEN 后定向测试通过。完整 LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --features tch-backend 结果 92 passed、0 failed、3 doctests ignored；cargo clippy --features tch-backend --lib --tests exit 0 且 self_driving.rs 无新增诊断；rustfmt --check 与 git diff --check 通过。证据只覆盖 localhost CPU 的 fixed-two-layer final-logits correctness，不覆盖 tied head 的 TCP 路径、sampling、下一 token、任意 L 网络循环、QUIC、远端硬件或性能。

_updated: 2026-07-31 09:44:09_
### Rust N=3 两层 localhost TCP finisher-to-starter handoff 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@c5751f1`

实现提交 c5751f1 只新增固定 N=3、两层、单 token 的 localhost TCP 实验测试，没有修改生产 runtime 或 transport 合同。三个独立 worker 线程各持本地两层权重与本地 KV shard，并只复用 predecessor/successor socket；initial hidden 仅存在于 domain 1。layer 0 实际 route=1->2->0，domain 0 finisher 将输出 hidden 保存在本线程 next_layer_hidden 并直接 start layer 1；layer 1 route=0->1->2，没有 coordinator 或共享 activation 回传。两层总发送 4=2*(N-1)，每层三个 local partial exact-once、唯一 finisher；assignee 分别为 domain 2/domain 1，只有对应层对应 shard 的 K/V 各增长一个 token；最终 domain 2 hidden 与未切分两层参考 max diff<4e-4。验证环境为 inventory 的 mac-local-shell + libtorch CPU。定向测试通过；完整 LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --features tch-backend 结果 92 passed、0 failed、3 doctests ignored；cargo clippy --features tch-backend --lib --tests exit 0 且 self_driving.rs 无诊断；rustfmt --check 与 git diff --check 通过。证据不覆盖任意 L 网络循环、final logits/sampling、跨 token continuation、QUIC、远端硬件或性能。

_updated: 2026-07-31 08:04:37_
### Rust localhost TCP ring 的任意 N 与 wrap-around 路由验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@2150d7a`

实现提交 2150d7a 只参数化既有 localhost TCP 单层试验，没有新增生产路由机制。测试 helper 直接记录每个 packet 的 visited_index/current_domain，并对 N=2/3/4 使用 starter=N-1、assignee=(starter+1)%N。实际处理顺序分别为 N=2:1->0、N=3:2->0->1、N=4:3->0->1->2，均真实经过尾节点->0 的 wrap-around TCP 边。每例同时断言 send=N-1、每 worker 一次 local partial、finisher=(starter+N-1)%N 且唯一、只有 assignee K/V 各增长一个 token、attention diff<1e-4、layer hidden diff<2e-4。验证：cargo test --features tch-backend 最终 91 passed, 0 failed，3 doctests ignored；新增定向测试通过；定向 clippy 对 self_driving.rs 无 warning；git diff --check 通过。结论只覆盖 mac-local-shell + libtorch CPU 的单层 localhost TCP correctness，不覆盖多层网络循环、sampling、QUIC、远端硬件或性能。

_updated: 2026-07-31 06:40:06_
### 未固定随机权重的 CPU batch 数值阈值可能偶发越界，不能在无关节点顺手调阈值

type: `lesson` · status: `held` · confidence: 0.95 · importance: 0.65 · source: `hetero-cp-ringattn@2150d7a`

症状：首次完整 91 项回归中，既有 test_batch_forward_correctness 的 decode sample 1 mean diff=1.095e-4，略高于 BATCH_TOL=1e-4；新增 arbitrary-N TCP 测试本身通过。定位：本轮 diff 仅在 self_driving.rs；失败用例位于 model.rs，fixture 使用未固定 seed 的 Tensor::randn，且注释已说明 CPU BLAS batched/single 路径存在非确定数值差异。复验：该用例在五个独立 cargo test 进程中 5/5 通过，随后完整 91/91 通过。处理：本节点不调阈值、不改模型；只保留 incident。预防：遇到这种跨模块数值边界失败，先查随机 fixture并独立重复，只有稳定可复现且与改动有数据流联系时才修改测试或实现。

_updated: 2026-07-31 06:40:06_
### Rust N=3 localhost 单层 self-driving TCP ring 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@71c8698`

实现提交 71c8698：新增独立 SelfDrivingPacket（residual、normalized hidden、position_ids、Q、attention output/LSE 与路由字段），LayerPacket 只在已产生首个 local partial 后转换为 wire packet；TcpKvTransport 以私有 TcpFrame 分派和固有 send/recv 承载实验 packet，不修改 KvTransport trait、旧 RingPacket 或 QUIC。N=3 loopback 建立完整定向 ring，每个 worker 各持 incoming predecessor、outgoing successor、完整同层权重和唯一 local KV shard；单请求实际走 0->1->2 两个 TCP hop。机器断言：三个 worker 各处理一次 local partial，发送次数=N-1=2，finisher=1，只有 assignee domain 1 的 K/V 各增长一个 token；attention diff<1e-4、layer hidden diff<2e-4；历史 shard 长度 2 与 47 的首跳实际 TCP frame 字节数相同；Int64 position_id=16777217 无损 roundtrip。验证：设置本地 LIBTORCH 动态库路径后运行 cargo test --features tch-backend => 90 passed, 0 failed, 3 doctests ignored；定向 cargo clippy 对 self_driving/transport touched files 无 warning；git diff --check 通过。证据只声明 mac-local-shell + libtorch CPU correctness 与 localhost 字节流，不声明远端、硬件性能、任意 L 网络循环、sampling、QUIC 或 runtime。

_updated: 2026-07-31 04:47:00_
### 构造长 context 测试时 position_id 必须先满足 fixture 的 RoPE 上限

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.55 · source: `hetero-cp-ringattn@71c8698`

症状：wire-size 测试用 history_len=47 且 position_id=history_len*3=141，在进入 TCP 前由 RotaryEmbedding index_select 报 index out of range。影响：协议测试被无关模型 fixture 约束阻断。根因：test_config.max_position_embeddings=128，而 synthetic position 超界；同文件既有可工作对照为 47*2=94。最小修复：只把测试 position_id 改为 history_len*2，不改模型、RoPE 或协议；原失败测试随后通过，完整 90/90 回归通过。预防条件：构造 synthetic 长历史时同时检查 position_id < max_position_embeddings；该教训目前只归属项目测试夹具，不升级为通用 skill 或生产规则。

_updated: 2026-07-31 04:47:00_
### Rust 任意 L 单 token 全模型 self-driving ring 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@c2a0483`

实现 commit c2a0483。新增 run_model_ring，以 Vec 逐层复用已验证的 run_single_layer_ring：上一层 finisher 成为下一层 starter，末层共享同一个 final norm/head helper 并唯一产生 logits。N=3、starter=1 时，L=3 角色回到 producer domain 1、总 hops=6；L=4 producer 轮转到 domain 0、总 hops=8；两例均满足 producer=(starter-L) mod N、每层 Q projection=1、local partial=3、current K/V projection/commit=1、layer finish=1，且只有指定 assignee 的 local shard 增长。完整 logits 与逐层单节点参考误差低于 1e-3。TDD RED 命令因 run_model_ring 不存在以 E0425 失败，补最小循环后定向测试转绿。最终验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend，87 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --lib --tests exit 0，warnings 均来自既有文件，self_driving.rs 无诊断；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check 均通过。执行环境为 inventory 中 available 的 mac-local-shell 与本地 libtorch CPU；本节点不声明硬件性能。

_updated: 2026-07-30 16:10:16_
### Rust 第五个实验切片：任意 L 的单 token 全模型 ring

type: `task` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@c2a0483`

把固定两层 self-driving runner 推广为任意非零层数 L。输入冻结的逐层 assignee 和逐层 local KV shards；每层让上一层 finisher 成为下一层 starter，末层唯一执行 final norm/head。验证 N=3 下 L=3 与非整倍数 L：logits 与标准参考一致、总 hops=L*(N-1)、producer=(starter-L) mod N、每层 Q/KV commit/partial/finish exact-once。仅使用 mac-local-shell + local libtorch CPU correctness；不加入 sampling、跨 token 状态、serde、QUIC、runtime、动态 planner 或生产治理。

[2026-07-30 完成] 任意非零 L 的单 token 全模型 runner 已闭合；L=3/L=4 在 N=3 下验证 logits、角色递推、L*(N-1) hops 与逐层 exact-once。未加入 sampling、跨 token 状态、serde、P2P 或 runtime。

_updated: 2026-07-30 16:10:16_
### Rust 末层 finisher 唯一 final logits 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@e2c6cd6`

实现 commit e2c6cd6。在固定两层 self-driving runner 上，末层 finisher 的 hidden 就地执行 final RMSNorm 与 LM head；独立 lm_head 和 tied embedding fallback 两条路径均与标准参考 logits 误差低于 4e-4。N=3 主例 producer domain=2、logits projection=1、总 hops=4=2*(N-1)，final head 不增加 ring hop；N=1 tied 例 producer domain=0、hops=0。验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib HCP_ENABLE_TORCH=1 CARGO_NET_OFFLINE=true cargo test --manifest-path rust/Cargo.toml --features tch-backend，85 passed/0 failed，doc tests 0 failed/3 ignored；同环境 cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --lib --tests exit 0，warnings 均来自既有文件，self_driving.rs 无诊断；rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check 均通过。执行环境为 inventory 中 available 的 mac-local-shell 与本地 libtorch CPU；本节点不声明硬件性能。

_updated: 2026-07-30 14:16:34_
### Rust 第四个实验切片：末层 finisher 唯一产生 logits

type: `task` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@e2c6cd6`

在已验证固定两层 self-driving runner 上，末层 finisher 用自己的 hidden 执行 final RMSNorm + 独立 lm_head 或 tied embedding fallback，唯一生成单 token logits。验证 producer domain=末层 finisher、次数=1、无额外 ring hop、logits 与标准参考一致。仅用 mac-local-shell + local libtorch CPU correctness；不加入 sampling、token handoff、serde、网络或 runtime。

[2026-07-30 完成] 末层 finisher 已唯一执行 final norm + 独立或 tied LM head；logits 与参考一致且不增加 hop。未加入 sampling、token handoff、serde、网络或 runtime。

_updated: 2026-07-30 14:16:34_
### Rust 两层 self-driving packet handoff 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@dc1aeb5`

实现 commit dc1aeb5。固定两个真实 DecoderLayer，layer 0 starter=1 经 N=3 ring 在 domain 0 finisher 完成，domain 0 随即用该 hidden 初始化 layer 1 packet；layer 1 在 domain 2 finish，角色递推 1->0->2。每层 hops=2，总 hops=4=2*(N-1)；每层 Q projection=1、local partial=3、current K/V projection/commit=1、layer finish=1；layer 0/1 分别只有 assignee 2/1 的 local KV 长度增加 1。最终 hidden 与两层单节点参考误差低于 4e-4。验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend，84 passed/0 failed；cargo clippy --features tch-backend --all-targets exit 0，只有既有文件 warnings，self_driving.rs 无诊断。执行环境为 inventory 中 available 的 mac-local-shell、arm64、cargo 1.93.0、本地 libtorch CPU；本节点不声明硬件性能。

_updated: 2026-07-30 11:21:43_
### Rust 第三个实验切片：两层 packet handoff

type: `task` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@dc1aeb5`

固定两个真实 DecoderLayer。layer 0 finisher 用自己的输出 hidden 原地创建 layer 1 LayerPacket，并成为下一层 starter；验证两层输出、角色递推、每层 N-1 hops 与 exact-once。仅用 mac-local-shell + local libtorch CPU synthetic correctness；不包含 logits、sampling、通用全模型 driver、serde、QUIC、runtime 或硬件性能结论。
[2026-07-30 完成] 两层 runner 已验证 finisher-to-starter continuation；未加入 logits、网络或通用全模型 driver。

_updated: 2026-07-30 11:21:43_
### Rust 自驱动 LayerPacket 数据边界验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@76be3b6`

实现 commit 76be3b6。LayerPacket 显式携带 residual hidden、normalized hidden、position_ids、Q、online-softmax O/LSE 与最小路由状态；process_layer_packet 的接口只接收 local layer weights、packet 和单个 local KV shard。N=2 直接逐 domain step 与单节点参考一致；原 N=1/2/3/4 runner 回归保持 N-1 hops、Q/KV projection/commit/partial/finisher exact-once。历史 shard 长度 2 与 47 时，首跳后 packet tensor payload 元素数严格相等，证明 payload 不随历史 context 增长；没有远端历史 KV 进入 packet。验证命令：LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend，83 passed/0 failed；cargo clippy --features tch-backend --all-targets exit 0，只有既有文件 warnings，self_driving.rs 无诊断。测试过程中一次 RoPE 越界来自 fixture position=194 超过 max=128，改为 position=94 后原测试通过；按 learning-from-incidents 判定为一次性夹具问题，不升级规则或 skill。

_updated: 2026-07-30 10:23:35_
### Rust 第二个实验切片：显式化自足 LayerPacket 数据边界

type: `task` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@76be3b6`

只修改单层 in-process 实验：引入携带 residual h、normalized h、Q、O/LSE 与最小路由状态的 LayerPacket；每个逻辑 domain step 仅依赖 packet、当前 layer 和自己的 KV shard。验证 N=1/2/3/4 数学、N-1 hops、exact-once 角色不变，以及 tensor payload 元素数不随历史 context 长度变化。不包含 serde、QUIC、worker runtime、多层、logits、重试或生产级 planner。
[2026-07-30 完成] LayerPacket 与 process_layer_packet 已落地；runner 不再通过共享作用域向 assignee/finisher 提供 hidden。

_updated: 2026-07-30 10:25:44_
### 单层 in-process runner 仍通过共享作用域隐藏 packet 必需状态

type: `risk` · status: `resolved` · confidence: 0.98 · importance: 1.0 · source: `hetero-cp-ringattn@76be3b6`

已验证的单层数学成立，但 run_single_layer_ring 中 residual 与 normalized hidden 位于整个函数共享作用域：逻辑 assignee 能直接读取 normalized hidden，逻辑 finisher 能直接读取 residual。真实 P2P worker 不共享地址空间，因此后续 packet 必须显式携带这两类 O(d) 状态，或接受重复 norm / 额外 K/V 传输。该风险不反驳单层 attention 与 layer continuation 的代数证据，也不产生 context-sized 远端 KV；它限制的是“wire-ready 数据面已被证明”这一更强结论。
[2026-07-30 解除] run_single_layer_ring 已完全改用 LayerPacket step；assignee/finisher 所需 normalized hidden 与 residual 均来自 packet。

_updated: 2026-07-30 10:25:44_
### Rust 单层真实 tensor 自驱动 ring 验证通过

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `rust/src/model/self_driving.rs`

本地 Mac CPU synthetic tensor，独立 libtorch。主例 N=3，历史 KV shard 长度 [2,5,3]，starter=1、assignee=2、finisher=0：Q 投影 1 次且归属 starter，current K/V 投影与 commit 各 1 次且归属 assignee，local partial=3，layer finish=1，hops=2；attention 最大误差 7.45e-9，整层最大误差 2.98e-8。通用性覆盖 N=1/2/4，route=successor 顺序且 hops=N-1。回归：focused 2/2、既有 ring attention 12/12、全量 tch 81/81 均通过。clippy 退出 0；只报告既有文件 warning，新代码无诊断。实验明确不包含真实网络、全模型、多请求或物理显存证明。

_updated: 2026-07-30 05:59:08_
### 单进程数值等价会掩盖跨节点角色归属错误

type: `lesson` · status: `held` · confidence: 1.0 · importance: 0.9 · source: `incident-2026-07-30-single-layer-role-boundary`

症状：第一版测试数值与单节点参考一致，但 starter 同时计算 Q/K/V 后借共享地址空间直接写入 assignee shard；这不能证明最终跨进程设计中的 assignee 自算自留，也会隐藏额外 K/V 传输。根因：测试只观察输出与写入份数，没有观察计算发生的逻辑 domain，单地址空间抹平了物理边界。修复：拆分 Q 与 current K/V 投影原语；Q 在 starter 执行，packet 抵达 assignee 时才投影并 commit K/V；统计实际执行计数并记录 projection domain。验证：新增角色断言先编译 RED，最小修复后 focused、ring 回归和全量 tch 测试均通过。可复用判据：涉及分布式 ownership 的 in-process 原型，输出等价不是充分证据，必须同时观察操作发生在哪个逻辑节点。

_updated: 2026-07-30 05:59:08_
### 执行 Rust 单层真实 tensor 自驱动 ring 实验

type: `task` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `docs/plans/2026-07-30-rust-single-layer-self-driving-experiment.md`

当前小节点。改动：提取现有 decode attention 数学原语；新增单层 in-process ring runner；用 uneven shards 验证 N=3，并覆盖 N=1/2/4。原因：直接验证 attention 与 residual/norm/MLP continuation 的组合，不让网络、runtime 或生产级 planner 混入。对计划贡献：若通过，证明自驱动 ring 的模型层核心可行；若失败，失败面被限制在投影、online-softmax merge 或 layer continuation。完成后必须停下汇报，不自动进入全模型或网络节点。

【完成证据 2026-07-30】实现单层真实 tensor runner：starter 唯一投影 Q；packet 按 successor 顺序访问 N 个互斥历史 KV shard；assignee 在 packet 抵达时唯一投影并持久保存 current K/V；finisher 唯一执行 O projection、attention residual、post-attention norm、MLP residual。N=3 uneven shards 与单节点参考最大整层误差 2.98e-8；N=1/2/4 路由与 N-1 hops 均通过。完整 tch 测试 81/81 通过，既有 ring-attention 回归 12/12 通过。边界：仍是单进程单层实验，不证明 QUIC、跨进程、全模型或物理显存 reservation。

_updated: 2026-07-30 05:59:08_
### Harness Auditor 拒绝第一版自驱动 decode 实施计划发布

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `harness-review-2026-07-29`

Pre-action verdict REJECTED。阻断点:1) H_i 只有公式，没有 resource profile/wire 来源及 zero/nonzero/concurrent 计费测试；2) cache.rs 的 Contiguous/BlockTable 路径使用 Tensor::cat，产生 O(local context) 临时整 shard，固定 H_i 无法覆盖；3) mode 由 worker 本地分支而非 coordinator 全员 capability/version 协商，mixed configuration 可能造成 response 合同不一致或 ring deadlock。要求修订计划与 graph memory 后重审。

_updated: 2026-07-29 06:03:41_
### 修订自驱动 decode 计划:reserved KV slab、完整 memory ledger 与全员 mode negotiation

type: `revision` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`

响应 Auditor REJECTED:resource profile 新增 B_i^K/B_i^V/G_i/H_i/W_i；C_i 来自模型加载后的可靠 device-free telemetry 与显式 KV budget 保守值，K/V 两个 slab 分别按 allocator granularity round。ledger 对 active requests 的持久 slab+metadata 求和、单线程 executor workspace 取 max；Task 1 覆盖 zero/nonzero share、并发重复计费、granularity 和 workspace release。Task 5 在 AdmitRequest 内物理预分配 ReservedKvCache，成功后才 ack；append copy 到 reserved slot、history narrow view，self-driving 禁止全 shard Tensor::cat。Task 3/8 使用 versioned WorkerHello 和 coordinator-only mode selection；先完成 control negotiation，再建立两个 peer data-plane streams，收齐 DataPlaneReady 后才 prefill，mixed-mode 提前拒绝。计划状态 ongoing，待重新审查。

_updated: 2026-07-29 17:24:29_
### 第一轮自驱动 decode 计划复核修正 overhead、prompt placement 和 feature gate

type: `evidence` · status: `superseded` · confidence: 1.0 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`

数学复核发现 fixed allocator/request overhead 若放进整请求 payload 后再乘份额，会在小份额节点低估精确 reservation；修正为先从 free KV bytes 整笔扣除 H_i，再计算 payload 上界并在整数计划后逐节点复核。长 prompt 的 durable history 决定整个 decode 的 attention scan，因此 prompt contiguous chunk 必须与 decode growth calendar 共享 bounded compute-balanced target。代码复核确认 worker_sdk 整体受 tch-backend gate，纯 scheduler 应放 distributed::decode_scheduler；model::decode 的纯 contract 可无条件编译，tensor driver 单独 feature gate。 后续 Auditor 指出该轮复核仍遗漏 context-sized 本地 Tensor::cat workspace、resource profile 字段来源和全员 mode negotiation，因此本节点只保留为阶段性证据，不作为计划安全闭环。

_updated: 2026-07-29 06:03:41_
### 修订 strict equal 1/N 默认:改为显存硬上界内 compute-balanced placement

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-07-29`

用户澄清 1/N 只强调显存压力可控和无 owner-collapse；异构设备必须按容量上界分配，并在上界内按 attention throughput 优化。旧 decision-self-driving-ring-theory-20260729 的 strict equal 默认与此冲突，现由 decision-self-driving-ring-v2-20260729 supersede。保留的部分:单 packet、N-1 hops、两 peer、exact-once、固定 sampler 可接受。改变的部分:equal 只在容量/吞吐相同时成立；默认 planner 使用 bounded water-filling，容量墙处退化为 pure capacity。

_updated: 2026-07-29 05:05:41_
### 用户确认显存 hard bound + bound 内 compute balance + 容量墙退化

type: `evidence` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-07-29`

用户明确确认:显存容量是 hard bound；在 hard bound 内按 attention throughput 做 compute-balanced 优化；请求逼近总容量时自然退化为纯 capacity 比例。并再次确认多请求各自异步跑 packet、request_id 分散初始 phase。

_updated: 2026-07-29 05:05:41_
### 实现计划代码审计:真正多请求 pipeline 必须替换阻塞 decode_request runtime

type: `evidence` · status: `held` · confidence: 0.98 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`

代码审计确认:RingPacket 仅 layer/Q/O/LSE/scale；LlamaModel.forward 同步执行 embedding->全部层->final norm->LM head；HcpRingAttentionBackend.forward 耦合 Q/K/V/O；WorkerRuntime 在 backend.decode_request 内阻塞；WorkerResponse.DecodeDone 默认每 worker 返回 logits；QUIC 已有 per-layer split-phase packet transport可复用。因此 R1 必须同时接 packet/projection/layer continuation/unique outcome；R2 必须新增 packet ingress、bounded ready queue、single-device executor、backpressure/fairness，单靠 coordinator DecodeBatch 不能证明 ring 内 pipeline。 计划可执行性复核补充: model 顶层可在 no-default 构建，纯 route/header 放无 Tensor 的 model::decode，tensor driver 后续用 tch feature gate；worker_sdk 整体受 tch-backend gate，纯 fairness/backpressure scheduler 必须放 distributed::decode_scheduler，再由 worker_sdk adapter 持有 Tensor。 第二轮代码复核:ContiguousKvCache.update 与 BlockTableKvCache.update/get_kv 都通过 Tensor::cat 物化本地完整 shard；当前 WorkerHandshake 固定 16 bytes，仅含 domain_id/capacity_mb；WorkerRuntime 无 negotiated mode 状态。这要求 reserved slab 和 versioned cluster negotiation 成为 R1 前置。

_updated: 2026-07-29 06:03:41_
### 准备并审查自驱动 decode ring 详细 TDD 实施计划

type: `task` · status: `superseded` · confidence: 0.98 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`

已生成 docs/plans/2026-07-29-self-driving-ring-decode-implementation.md。计划含动机六问/牺牲四问、bounded water-filling、冻结二维 calendar、14 个依赖 checkpoint、每步 red-green 命令、focused commit、R1/R2/R3/R4 审查门，以及 MPS/CUDA+HIP/三节点异构完成条件。本任务只准备计划，未修改生产代码。 第一轮复核补充:prompt contiguous chunk 与 decode calendar 共享 bounded target；fixed KV overhead 先整笔扣除再求 payload cap；纯 scheduler 放 ungated distributed 模块；已修正 sampler recurrence 测试参数和无效多 filter cargo 命令。 补充合同:throughput 缺失时 capacity-only 明确定义为 x_i=u_i/sum(u)，不混合猜测 rate；sampler logits 在同一 compute quantum 内消费释放，不进入 packet/ready queue/backlog。 [Auditor pre-action REJECTED] 计划需补齐:H_i/G_i/W_i 的 profile/wire 来源与计费测试；self-driving cache 改为 reservation-backed slab，禁止现有 Tensor::cat 全 shard 临时副本；coordinator 全员 version/mode/capability 协商并在 mixed-mode 时于 prefill 前拒绝。修订后重新审查。 [第二轮修订完成待审] K/V slab 分别按 allocator granularity 计费；C_i 不再从 capacity_mb 粗略换算；AdmitRequest 物理分配成功后才 ack；bootstrap 改为 control hello/negotiation -> peer streams -> DataPlaneReady -> admission/prefill。

_updated: 2026-07-29 17:24:29_
### 固定 sampler 不增加 durable KV,额外显存仅 O(batch*vocab) 瞬时 logits

type: `evidence` · status: `held` · confidence: 0.98 · importance: 0.9 · source: `analysis-2026-07-29`

在模型权重全节点复制且 kv_assignee(position) 独立 round-robin 的前提下,sampler 节点不新增 LM-head 权重,也不持有额外历史 KV。其额外状态是 final norm/LM-head/sampling 的临时激活与 logits:[batch,vocab]。152K vocab 的单个 fp32 logits row 约 0.58 MiB,与 context length 无关。风险是多请求 LM-head queue/算力热点,不是 KV 容量热点。

_updated: 2026-07-28 17:26:22_
### 自驱动环角色递推证明:层内自然轮转,token 间可能模数共振

type: `evidence` · status: `held` · confidence: 0.99 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-theory.md`

若每层结果停在 starter 的 predecessor 且 finisher 直接成为下一层 starter,则 s(t,l+1)=s(t,l)-1 mod N;若末层 finisher 直接启动下一 token,则 s(t+1,0)=s(t,0)-L mod N。当 L mod N=0 时,token starter/logits/sampler 永久落在同一节点。要满足 sampler 跨 token 轮转,至少需要一次 token 边界 phase shift;最小实现是 sampled token ID 沿 successor 多走 1 跳。

_updated: 2026-07-28 17:19:59_
### 插件 owner-return 单向环的物理下限是 N 跳,不是 N-1

type: `evidence` · status: `held` · confidence: 0.99 · importance: 1.0 · source: `docs/plans/2026-07-29-self-driving-ring-decode-revision.md`

代码证据:ring_transport.py 的拓扑固定为 i->(i+1)%N,owner(index 0)在 ring_decode_step 中先 send_packet,再从 predecessor recv_packet;所有 N-1 peer 的 RingDecodeNode 逐个 handle+forward。路径必为 0->1->...->N-1->0,恰 N 条物理边。N=3 时是 owner->A->B->owner=3 跳。把 owner local partial 从种子移到最后归并只改变计算顺序,不会删除任何边。N-1 只在结果可停在 predecessor finisher 时成立。

_updated: 2026-07-28 17:11:06_
### [2026-07-28] Rust 线 decode Q-ring 验证通过:Q+LSE 累积器环+增长零传输分片,跨节点 CUDA+HIP 闭环

type: `evidence` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `experiment`

decision-rust-decode-qring-20260728 的实现与验证(主仓 c4a3e7f,8 文件;subagent 实现+主 Agent 亲验+Reviewer APPROVE)。
实现:ring.rs 新增 seq_len==1 的 Q-ring 路径(decode_local_partial/decode_merge_packet/ring_decode_attention,复用既有 max-shifted 归并与 Phase 0/1/2 控制流);RingPacket 消息+QUIC/TCP 序列化;cache.update_sharded(keep/discard);五条件门(seq_len==1 && decode_ring && N>1 && prefill done && transport 支持 packet);legacy HCP_RING_DECODE_RING=0 保留。
验证阶梯:
1. cargo test --features tch-backend 68/68(主 Agent 亲跑;+6 新测试:3 域不均 chunk 正确性<1e-4、2 域线程化 forward 逐步对参考、legacy 回退、cache 分片精确计数);
2. Mac MPS 双 worker QUIC A/B:生成文本逐字相同;Q-ring 288/288 decode 事件(24层×2域×6步),packet 3640B(Qwen2-0.5B),decode 流量 5.68MB→1.05MB(11-token prompt,比值 O(seq) 增长);
3. 跨节点 white CUDA + pearl HIP(Qwen2.5-3B,SEQ=64,decode 6):两节点各 216/216 decode 事件(36层×6步),packet 8256B(16×128×4+64)精确 O(d);legacy 复跑同文本;
4. Reviewer APPROVE(自跑测试+自析 JSONL+diff 抽查不变量;判定 LinkedMock=false 为诚实工程非 test-rigging;保留项:coordinator exit code 未落盘、"+7"实为+6、perf 字段跨事件不同名)。
增长分片语义:全节点同算 forward → 新 KV 各节点自算,按 global_pos%N 自有即留/非自有即弃,零传输(plugin 线需捎带是因只有 owner 算 forward)——Rust 架构独有红利。
报告:reports/2domain-cuda-hip-20260728-115253/(含 perf JSONL×4 与 A/B 日志)。
ISSUE-007/008 resolved。遗留:全节点冗余 logits 计算(worker 0 以外被丢弃)列为后续评估项。

_updated: 2026-07-28 04:00:05_
### [2026-07-28] decode ≥2 请求并发验证通过(conc2b):按请求全隔离,增长分片在批量 decode 下保持

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

decision-decode-concurrent-20260727 的实现与验证(插件 cc733dd 并发隔离 + 453fa1f 流下载竞态修复)。
white RTX 4090 单卡 3 实例,2 并发 CP 请求(不同 prompt 变体,各 1536 token,chunks 512/512/512,decode 8):
1) 两请求 token 序列不同且各自与单节点参考全对(req0=[220,20,22,29514,...],req1=[220,17,323,279,...]);
2) 真并发:BATCH_STATS.max_reqs=2(同一 attention step 批量);
3) 跳池门精确化(get_forward_context 元数据逐请求判定):skipped=336/336(2×7×24),slots 1040≤1064;
4) 增长按请求隔离:ring packet 带 req tag(first_block_id:uuid,块 id 会被回收复用故需 uuid),A/B 各 calls=336、growth_appends=96(每请求每 peer 2 token×24 层=48,两请求共 96),per_req={tag1:2, tag2:2} 清晰隔离;
5) 流式窗口 6≤12(N×6);staging 0/0;ring map=2 后清空;triton 456/0、merge 48/0;decode 零 HTTP。
过程 bug:首次 conc2 死于两流并发下载同一 chunk 文件(safetensors "header too small")——加 per-(chunk,layer) stage lock+已 staged 去重(453fa1f)。
Reviewer 独立复核 APPROVE(8 项全过;答疑:growth_appends=96 语义为每请求每 peer 轮转分得 2 token,非滚动覆盖)。
报告:reports/ring-decode-conc2b-013813/。ISSUE-004 resolved。
意义:PoC 最小并发要求达成——多请求连续批下显存切分(prefill 流式+decode 增长分片)与按请求正确性同时成立。剩余:ISSUE-006(对等化,待探讨)、ISSUE-007/008(Rust decode,任务C)。

_updated: 2026-07-27 17:44:15_
### [2026-07-28] 从 N=1 到 N≥2 是竞态高发相变点:凡按"单实例/单请求/单下载者"假设写的路径都要重审

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `reflection`

conc2 首跑即死:两条 prefill 流并发下载同一 chunk 文件到同一本地路径,一方读到对方写了一半的文件(safetensors header too small)。单请求时代该路径天然无竞态,bug 潜伏。同类潜伏点清单(本轮已处理):n==1 启发式跳池门(批量失效)、peer growth 按层键(请求间混叠)、共享文件无锁下载。
教训:1) 单实例验证 PASS 后,把代码里所有"隐含的 1"列出来(键空间、启发式、文件路径、全局缓冲)逐一追问 N≥2 时怎样;2) 共享资源(文件/staging/buffer)的并发规则要在第一版就写明,不是等竞态发生再补;3) token 级正确性验证仍是最终裁判——混叠类 bug 在相同 prompt 下可能不可见,并发验证必须用不同 prompt 变体。

_updated: 2026-07-27 17:44:15_
### [2026-07-27] 流式 prefill+ring-only+邻接拉取三机验证通过(p2p3n-000004):瞬态显存有界,星形传输灭绝

type: `evidence` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `experiment`

decision-prefill-streaming-20260727 + decision-no-star-transport-20260727 的实现与验证(插件 6eacec3 流式 + fb4a2b0 删星形 + 7c059e1 验证器对齐;主仓驱动 190636d)。
验证阶梯全 PASS:
1. 单机 p2p4/p2p5(white):peak staged layers 5→6(旧=48 全量前缀);token 8/8 一致;decode-ring 默认开启(无 env 也跑);
2. 三机 p2p3n-000004(laptop A 4060 + white B 4090 + pearl C 9060XT,HEAD=7c059e1 三机一致):peak staged 4(≤6);token 8/8 一致(max diff 0.0215);memsplit/staging 0-0/pool-skip 168/168/slots 528≤552/triton 240-0;ring transport sent=recv=168;A/B RingDecodeNode 各 calls=168、growth 48/2 层;
3. 邻接 prefill 实证:pearl 只从 white 拉 c0+c1(48 GET),laptop 的 HTTP 客户端只有 white——无任何跨环直连,N>3 部分可达网络拓扑成立;
4. decode 零 HTTP(partial_attn 全日志 0 次),星形代码已删除(831 行)。
Reviewer 流程:p2p3n-235241 得 WARN(归档缺 A/B decode 统计,pkill 竞态)→ 驱动改优雅关停+stats 门(190636d)→ p2p3n-000004 复审 APPROVE(8 项全过)。
报告:reports/ring-decode-p2p-3node-p2p3n-000004/(对照 reports/ring-decode-p2p-3node-p2p3n-235241/)。
ISSUE-003(prefill 瞬态全量前缀)与 ISSUE-005(staging-then-compute 无重叠)已 resolve。
意义:真显存切分的最后一块瞬态缺口闭合——任何节点任何时刻(含 prefill 瞬态)持有的他人 KV ≤ (W+1) 层×chunks;decode 只剩 P2P 环;拓扑全程线性。剩余:ISSUE-004(decode 并发,任务B)、ISSUE-006(对等化,待探讨)、ISSUE-007/008(Rust decode,任务C)。

_updated: 2026-07-27 16:05:48_
### [2026-07-27] 归档证据必须自足:verdict 依赖的每项证据都要被门控,异步证据源要先优雅关停再归档

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `reflection`

p2p3n-235241 复审 WARN:驱动 cleanup 的 pkill 与 A/B 进程"打印 decode 统计→退出"竞态,归档日志缺 RingDecodeNode 统计行,verdict 实际只靠 C 侧证据。修法:1) 优雅关停(done-file 通知+等待退出,pkill 仅兜底);2) verdict 前对 A/B 日志 grep RingDecodeNode 门控,缺失即 FAIL(190636d)。
教训:1) 异步证据源(远程进程、后台线程)的生命周期结束动作(打印统计)必须先于归档/判定完成;2) verdict 引用的每一项证据都要有可机器检查的存在性门控,不能靠"应该在";3) Reviewer 的 WARN 价值——这是探针族之外的第四类观测缺陷:证据归档完整性,与"探针不可见/投影塌缩/通道不匹配"并列加入验证检查清单。

_updated: 2026-07-27 16:05:48_
### [2026-07-27] 三机异构 P2P decode Q-ring 验证通过(p2p3n-175719):CUDA+CUDA+ROCm 真环,Q+累积器跨机逐跳

type: `evidence` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `experiment`

blueprint-two-phase-ring-20260727 的跨节点闭环。驱动 scripts/run_3node_decode_p2p.sh(主仓 7f67b28),插件 HEAD=7bfe24b 三机一致。
拓扑:laptop(100.96.154.1, RTX 4060 Laptop CUDA)=A producer(c0=512, 环 idx1);white(100.118.253.68, RTX 4090 CUDA)=B relay(c1=512, 环 idx2);pearl(100.111.242.55, RX 9060 XT gfx1200 ROCm)=C owner(c2=512, 环 idx0)。prefill:C 经 HTTP 从 laptop 拉 c0、从 white 拉 c1(日志真实 IP 佐证);decode:TCP 环 C→A→B→C,Q+累积器逐跳,growth 捎带轮转。
判据全过(pearl 单节点参考对照):
1) token 8/8 一致 [220,20,22,29514,84253,916,16301,220],max|logit diff|=0.0215;
2) MEMSPLIT:staging 0/0(decode 开始释放),ring map finish 清;
3) owner ring transport sent=168 recv=168(7 步×24 层,跨 tailscale 真网络);
4) A(laptop)/B(white) RingDecodeNode 各 calls=168、growth_appends=48、growth=2 token/层——轮转预测跨机精确命中;
5) decode 零 HTTP(partial_attn 全日志 0 次);
6) pool-skip 168/168;slots 528≤552;triton 240/0。
延迟特征:owner generate 46.2s/8 token(~6s/token)——24 层×2 跳×~125ms tailscale RTT 与模型预测一致,decode 延迟完全由互联主导,CXL/类 RDMA 论据的实测定量点。
Reviewer 独立复核 APPROVE(8 项全有日志原文;保留:三机 vLLM dev build 日期略异 laptop d20260724 vs white/pearl d20260717、TCP 逐跳为三角印证证据)。
报告:reports/ring-decode-p2p-3node-p2p3n-175719/。
意义:两阶段统一 ring 架构在三机真异构(CUDA 两代 + ROCm)上端到端成立——prefill 传 KV、decode 传 Q+LSE,每节点只连 2 个 peer、只持自己 KV 份额,无 collective。后续开放:capacity-aware 不均等三档分片复验、更长 seq、decode 并发、owner 策略对照。

_updated: 2026-07-27 10:19:39_
### [2026-07-27] Decode Q-Ring 真 P2P 环拓扑验证通过(p2p3):Q+累积器逐跳绕环,decode 零 HTTP

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

task-d39 计划(blueprint-two-phase-ring-20260727)的实现落地:插件 ce70afc(实现)+41cdcd1(单例探针)+8696639(后台连接)+63852cd(就绪等待入 relay 线程)三枚审查/排障修复。
white RTX 4090 单卡 3 实例,vllm 0.23.1rc1,HCP_RING_DECODE_RING=1 + HCP_RING_DECODE_TRANSPORT=ring:环序 C(owner,0)→A(1)→B(2)→C,TCP 8950/8951/8952;prompt 1536 切 512/512/512,decode 8。
判据全过:
1) token 8/8 与单节点参考一致 [220,20,22,29514,84253,916,16301,220],max|logit diff|=0.0293(argmax 处 0);
2) MEMSPLIT 保持:staging decode 开始即释放(0/0),ring map finish 才清;
3) ring transport 真实承载:owner sent=168 recv=168(7 步×24 层,每包绕环一周);
4) A/B RingDecodeNode relay:calls=168,growth_appends=48(2 token×24 层),growth_tokens=2/层——轮转预测精确命中,且统计行在 A/B 自身日志(非 driver 转述);
5) decode 零 HTTP:POST /partial_attn 在全部日志中 0 次;prefill KV store 仍走 HTTP(GET /p2p3/* 符合预期)——两阶段传输对象分离成立;
6) pool-skip 168/168;slots 528≤552;triton 240/0、merge 24/0。
Reviewer 独立复核 APPROVE(7 项全有日志原文,反证扫描干净;保留:无字面 connect-ack 日志/主机名不在日志/driver 日志薄)。
报告:reports/ring-decode-p2p-p2p3-171607/。
意义:HCP 两阶段统一 ring 架构(blueprint-two-phase-ring-20260727)端到端闭环——prefill 传 KV、decode 传 Q+LSE,全程 P2P 逐跳,每节点仅 2 个连接,无任何 collective;显存切分(prefill chunk+growth 份额)与通信拓扑统一同时成立。下一步:三机真环 P2P decode(laptop A + white B + pearl C,inventory 拓扑)。

_updated: 2026-07-27 09:25:44_
### [2026-07-27] 环拓扑顺序启动下,init 期任何阻塞等待都可能死锁:连接/就绪等待一律后置到工作线程

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `reflection`

p2p1/p2p2 两连发同族 bug:
①p2p1:RingTransport.start() 同步阻塞连 successor(120s 超时),但环节点顺序启动(B 要等 A 就绪才拉起),A 连 B 必然超时,引擎 init 崩溃——修法:连接放后台线程,send 等待 connected 事件(8696639)。
②p2p2:connector __init__ 里 wait_ready 等 store _READY,但标记要 prefill 后才写、prefill 要引擎 init 完才跑——自死锁,A 挂到 driver 600s 强杀——修法:就绪等待移入 relay 线程(packet 只在 owner decode 时才来,store 早已就绪,63852cd)。
教训:1) 顺序启动是环/链拓扑的必然模式,init 期能拿到的只有自己的配置;对 peer 可达性、对"运行后才产生的数据"的等待必须后置到使用点或工作线程;2) 两次都是"验证脚本先跑起来才暴露"——纸面审查难以发现时序死锁,单节点 smoke 先行仍然必要;3) 与 lesson-decode-ring-callback-override-20260725(后初始化者覆盖回调)同属 init 顺序/生命周期族:框架初始化顺序是事实性前提,设计时先画出"谁在何时产生什么"。

_updated: 2026-07-27 09:25:44_
### [2026-07-27] Decode Q-Ring P2P 计划文档(task-d39,QoderCN):decode 传输从星形 HTTP 改为真 P2P TCP 环

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `user-direction + plan file`

计划文件:~/Library/Application Support/QoderCN/SharedClientCache/cache/plans/Decode_Ring_P2P_Transport_task-d39.md。内容:动机剖析六问+牺牲分析+算法设计(Q+累积器逐跳绕环,growth 捎带)+六阶段实现步骤(transport/ring node/backend/connector/validator/三机)。定位:只补充 decode 部分的传输拓扑目标,整体框架设计以 blueprint-two-phase-ring-20260727 为准。实现已落地:插件 ce70afc(ring_transport.py + RingDecodeNode + validate_ring_decode_p2p.py),审查修复 41cdcd1(单例探针)+8696639(后台连接 successor)。
[2026-07-27 更新] decode Q-ring P2P 已完成:单节点 p2p3 + 三机 p2p3n-175719 双 PASS。该任务的 decode 传输拓扑目标达成。

_updated: 2026-07-27 10:19:39_
### [2026-07-25] decode 增长分片验证通过(dsplit6):prefill/decode/增长全局显存切分语义闭合

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

decision-decode-growth-shard-20260725 的实现与验证(插件 cb424ad 实现 + a751de8 判据强化)。
white RTX 4090, vllm 0.23.1rc1, HCP_RING_DECODE_RING=1(默认 rr 轮转策略), validate_ring_decode_split --mode all:A producer(c0=512) + B relay(c1=512) + C owner(c2=512, full prompt 1536, decode 8)。
判据全过:
1) token 8/8 与单节点参考一致 [220,20,22,29514,84253,916,16301,220],max|logit diff|=0.0293(argmax 处 0);
2) MEMSPLIT DECODE 保持:generate 返回即 staging=0/0(前缀 decode 开始释放),ring map finish 才清;
3) 增长分片 peer 侧:A/B 各 calls=168(7 步×24 层) RPC 全 200,growth_appends=48(2 token×24 层),growth_tokens_per_layer=2——精确命中轮转预测(A={1536,1539},B={1537,1540});
4) 增长分片 owner 侧:decode 池写跳过 168/168((decode-1)×24 层)——增长不走 owner 池,自有份额在 backend 紧凑 buffer(owner={1538,1541});
5) slots 528≤552(c2+flush);triton 240/0、merge 24/0。
Reviewer 独立复核 APPROVE(8 项声明均有日志原文;A/B 统计行在各自日志原文中存在非 driver 转述;反证扫描 0 fallback/0 非 200;保留项:peer growth 未做落盘字节比对,单节点模拟拓扑)。
实现要点:append-then-serve 捎带保序(增长 KV 搭下一步 RPC,零额外跳);当前 token 始终瞬时参与本地段(因果尾);own_chunk_len 首 decode 步惰性捕获;HCP_RING_DECODE_GROWTH=rr|owner 策略缝(owner 退化供对照);HCP_RING_DECODE_RING 总开关不变。
报告:reports/ring-decode-growth-dsplit6-132831/。
意义:任何节点任何时刻只持久持有自己的 KV 份额(prefill chunk + 增长份额)——全局真正 KVCache 显存压力切分;显存压力全程可计算(chunk+growth/N);decode 慢(L×(P-1) RTT/token + 捎带)即 CXL 论据。后续开放:跨节点增长分片复验、capacity 加权指派策略、多请求并发 decode-ring、策略退化(owner)对照实验。

_updated: 2026-07-26 05:33:53_
### [2026-07-25] 槽位集合计数证明不了"未写入":释放槽位被复用后行为差异塌缩,计数型判据要直接测目标行为

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `reflection`

dsplit4(增长写池)与 dsplit5(增长跳池)的 WRITE_TRACK 槽位集合都是 528——主请求释放的槽位被 flush 请求复用,集合去重使两种相反行为数值相同,slots 核算无法区分。修法:WRITE_TRACK["skipped"] 显式计数被跳过的写((decode-1)×24 层=168 精确断言),行为判据直接测行为本身而非其副作用投影。
教训:1) 用"集合/计数投影"做判据时,先问复用/去重/缓存是否会让相反行为产生相同投影;2) 能直接计数目标行为(跳过次数、调用次数)就不用间接量;3) 与 lesson-plugin-logger-probe-20260725、lesson-closure-livelock-20260725 同族:探针有效性本身必须被质疑——这已经是第三次同族教训,验证脚本评审时应把"探针能否区分相反行为"列为固定检查项。

_updated: 2026-07-26 05:33:53_
### [2026-07-25] decode 显存切分(累积器绕环)验证通过(dsplit4):五项判据全过,decode 期 owner 只持自己 chunk

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

decision-memory-split-decode-20260725 的实现落地与验证(插件 2ad6aca 实现 + efe0bb0/4b92172 两枚修复)。
white RTX 4090, vllm 0.23.1rc1, HCP_RING_DECODE_RING=1, validate_ring_decode_split.py --mode all:A producer(c0=512) + B relay(c1=512) + C owner(c2=512, full prompt 1536, decode 8)。
五项判据全过:
1) C greedy 8 token 与单节点参考完全一致 [220,20,22,29514,84253,916,16301,220],max|logit diff|=0.0234(argmax 处 0.0);
2) MEMSPLIT DECODE:generate 返回即 PEER_KV_STAGING=0/PEER_REQ_MAP=0——前缀 KV 在 decode 开始释放而非请求结束;DECODE_RING_MAP 至 finish 才清(生命周期分离);
3) C 只写 528 池槽(c2 512+decode 8+slack≤584),前缀 chunk 从不进 owner 常驻池;
4) triton kernel 路径 attn 240/0、merge 24/0,0 回退;
5) A/B 各服务 168 次 POST /partial_attn 全 HTTP 200(7 decode 步×24 层,与预期精确吻合)。
Reviewer 独立复核 APPROVE(归档日志与 white 实时副本一致,无 fallback/错误迹象;保留意见:HCP_RING_DECODE_RING 环境变量未入日志、释放时点为单采样点推断)。
过程修复两个实现 bug:
①safetensors 0.8 load() 只收 bytes 不收 BytesIO——/partial_attn 每跳 HTTP 500(efe0bb0;既有 connector 走 load_file 所以未暴露);
②staging 释放回调被后初始化的调度侧 connector 实例覆盖(EngineCore 先建 model executor 后建 scheduler,core.py:118 vs :146),空 _live 静默空转,改 WORKER 角色门控(4b92172)。
排障弯路记录:dsplit2 曾据"served"日志 0 行误判 ring 分支未触发,实际分支一直在工作(HTTP 日志 168+168),探针不可见≠事件未发生。
报告:reports/ring-decode-split-dsplit4-032527/{driver,a,b,c}.log。
意义:decode 显存切分语义闭合——prefill/decode 对称,owner 全程只持自己 chunk+decode 增长;decode 每 token L×(P-1) RPC 跳的延迟代价即 CXL/类 RDMA 必要性论据。

_updated: 2026-07-25 19:32:22_
### [2026-07-25] 进程级全局回调被多实例"后初始化者"覆盖:静默失效,结果仍正确但语义判据失败

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `reflection`

ring connector 的 staging 释放回调(set_staging_release_fn)在 worker/scheduler 两个实例的 __init__ 无条件注册;vLLM EngineCore 先建 model executor 后建 scheduler(vllm/v1/engine/core.py:118 vs :146),调度侧实例后初始化,用自己空 _live 的绑定方法覆盖回调→_release_request_staging 永远空转→staging 全程不释放。危险之处:token 仍正确(旧 staged 路径兜底),只有显存切分语义判据能抓到。
教训:
1) 进程级全局 + 多实例注册必须考虑初始化顺序,谁后谁赢;按角色门控(WORKER only)或改实例无关的模块级函数;
2) "结果正确但语义判据失败"是定位信号——正确性路径与生命周期路径是两套代码,token 全对不能证明资源管理对;
3) 框架初始化顺序属于事实性前提,读源码确认,不靠记忆。

_updated: 2026-07-25 19:32:22_
### [2026-07-25] 插件 logger 输出不进捕获日志:探针不可见≠事件未发生,计数探针优先用服务侧 HTTP 请求行

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `reflection`

插件模块用 vllm init_logger(__name__)(非 vllm 命名空间),其 INFO 输出不会进 driver 捕获的日志文件(连既有的 "HcpRingKvConnector init"/"staged peer chunk" 也不出现)。dsplit2 据 "PartialAttentionService: served" 0 行误判 ring 分支未触发,实际 A/B 的 SimpleHTTPRequestHandler 请求行(stderr,必被捕获)显示 168+168 次 POST /partial_attn 全 200——观测通道不可见导致的假阴性。
教训:
1) 计数/计数率探针优先用服务侧 HTTP 请求行或 print(flush=True),不依赖 vllm logger;
2) 下"未发生"结论前,先验证观测通道本身可见(加一行启动必打印的校准日志);
3) 与 lesson-closure-livelock-20260725 同源:探针的假设(这里是"日志可达")也要被核对。

_updated: 2026-07-25 19:32:22_
### [2026-07-25] 环闭合三机验证通过:统一 ring 角色 + 邻接累积转发 + 轮转放置,(N+1)%N 字面成立

type: `evidence` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `experiment`

decision-ring-closure-123-20260725 的完整落地(插件 cefff57,主仓驱动 c77cb67)。
验证阶梯全 PASS:
1. N=2 回归 4 验证器(backend/connector/concurrent/relay)——环改动后向后兼容;
2. 单卡 3 实例闭合(white):3 引擎统一 ring_role=ring,3 请求轮转起始节点,position-2 token 全对,slots_written 1523/1552/1552(≤1604,各自只持有自己 chunk);
3. 三机环闭合(ringc-160010):laptop(4060 CUDA)=node0、white(4090 CUDA)=node1、pearl(9060XT ROCm)=node2,物理环 laptop→white→pearl→laptop,每节点只从物理前驱拉累积前缀(邻接 re-serve),3 并发请求轮转——node0 req1=[220,20,18,84253]、node1 req2=[220,20,18,84253]、node2 req0=[15,25009,220,20] 与各自单节点参考全对;staging 用后释放;triton 288 次 0 回退(三平台)。
语义:producer N 的 consumer 字面是 (N+1)%N(每节点在一个请求中生产、在另一个请求中从前驱消费);单序列数据面受因果约束不闭合,环在负载面闭合。
报告:reports/ring-closure-ringc-160010/。

_updated: 2026-07-25 08:04:18_
### [2026-07-25] 环闭合排障:stall 轮询风暴、url 补齐语义、块复用探针误报

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `reflection`

1. stall 轮询风暴:vLLM 调度器对 stalled 请求全速重试 get_num_new_matched_tokens(~1.6 万次/s/请求),每调用一次 HEAD 轮询 => 20 分钟 600 万连接打爆 loopback 临时端口,urlopen 失败后永远 stall(livelock)。修法:_prefix_ready 就绪结果永久缓存 + 失败按 chunk 0.5s 退避(插件 0cca5b8)。
2. url 补齐语义:_chunks_and_urls 短 url 列表原来补 ""(=全局/本地回退),邻接模型下第二个前缀 chunk 的就绪检查落到本地存储永远 False。修法:短列表补【最后一个 url】(前驱服务全部累积前缀,c85eff5)。教训:复数参数的默认值语义要逐组合(单 url×多 chunk)测试。
3. WRITE_TRACK 全局槽位集合 vs 每请求 block table 在并发下误报:释放块被分配器复用为后续请求的前缀区,旧主写入与新请求前缀区相交 => 假污染(61128/98304/110592)。并发下无法区分合法旧主与违规写入,改判据为 slots_written 核算(每节点≈自己 chunk token 数);token 匹配才是正确性主证据。
4. 排障方法论的胜处:先证 poll 计数(600 万 vs 0),再上 py-spy(被拒),最后 GNMT 探针计数定位到"第二个 chunk 从未被检查"——证据分级推进,不猜。

_updated: 2026-07-25 08:04:18_
### [2026-07-25] 三机真 ring 验证通过:laptop(4060 CUDA) + white(4090 CUDA relay) + pearl(9060XT ROCm)

type: `evidence` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `experiment`

run_id=ring3-033045,驱动 scripts/run_3node_ring.sh(主仓 commit ffab6c6),插件 HEAD=cd83a5b 三机一致。
拓扑:laptop(100.96.154.1, RTX 4060 Laptop CUDA)=A producer(c0=512);white(100.118.253.68, RTX 4090 CUDA)=B relay(c1=512,吃 laptop c0 产自己 c1,kv_both);pearl(100.111.242.55, RX 9060 XT gfx1200 ROCm)=C consumer(c2=512,复数 chunk_ids/peer_urls 从两个 peer stage)。
结果:1) C greedy 4 token 与 pearl 单节点参考完全一致 ref=cons=[220,20,22,29514](与单机 3 实例验证同 token,跨平台数值一致);max|logit diff|=0.023;
2) 显存切分:C 只写 528 池槽(自己 chunk+decode),前缀区域本地写入=0;
3) 2 前缀 chunk×24 层并发 staging,HTTP 日志证实跨节点传输(pearl 从 laptop GET c0 全部 24 层、从 white GET c1 全部 24 层);
4) staging 用后释放;triton kernel 216 次 0 回退(ROCm);
5) 就绪级联跨节点成立(B 的 _READY 依赖其完成 c0 staging+c1 计算)。
报告:reports/ring3node-ring3-033045/{driver,producer_a,relay_b,consumer_c}.log。
意义:HCP 首次实现 N=3 异构真 ring——三节点 worker 同级 peer,每节点只常驻自己 chunk,peer KV 瞬时借用;通用 N 框架下 N=2 是退化情形(回归全 PASS)。

_updated: 2026-07-24 19:33:45_
### [2026-07-25] 通用 N ring 插件实现:N=2 回归 + 单卡 3 实例 relay 全 PASS

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

task-generic-n-ring-impl-20260725 的实现(plugin commits 0a90c19 + cd83a5b)。
实现:1) connector ring_role=relay(消费前序+生产自己 chunk,kv_role=kv_both,store 侧 slot_mapping 跳过 external 前缀,就绪级联);2) backend PEER_REQ_MAP 改有序列表,多 chunk 连续前缀 cat 后单次 peer pass;3) hcp_ring 参数加 chunk_ids/peer_urls 复数(单数自动转单元素列表)。
验证(white RTX 4090,vllm 0.23.1rc1):
1. N=2 三件套回归全 PASS(向后兼容):validate_ring_backend/connector/concurrent,token 一致,memsplit 保持;
2. 新增 validate_ring_relay.py(3 实例:A producer c0=512 + B relay c1=512 + C consumer c2=512):就绪级联 A(8s)→B(8s);B 存储恰好 c1(512tok×24层);C 经复数参数从两个 peer stage 2 chunk×24 层并发,backend cat 连续前缀;C token 与单节点参考完全一致[220,20,22,29514],max|logit diff|=0.027;前缀区域本地写入=0;staging 用后释放;triton 216 次 0 回退。PASS。
下一步:三机真 ring(laptop A + white B relay + pearl C)。

_updated: 2026-07-24 19:26:53_
### [2026-07-25] laptop vLLM+CUDA 环境按 white 配方对齐完成,compat 6/6 + GPU smoke 通过

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.9 · source: `experiment`

laptop(100.96.154.1, RTX 4060 Laptop 8GB sm_89, Ubuntu 24.04)环境搭建完成并验证:
配方:miniconda + conda env vllm-v1(python 3.11)+ conda gcc/g++ 13.4 + torch 2.13.0+cu130/torchvision 0.28.0(pip 清华源)+ vllm 0.23.1rc1@3f99883d9 源码编译(应用 white 的 torch-unpin patch,与 white/pearl 同基线)。
验证:1) hcp-vllm-plugin pip install -e + compat_check 6/6 PASS(0 warnings);2) GPU smoke:Qwen2-0.5B-1M fp16 enforce_eager 生成连贯文本。注意 8GB 卡需 gpu_memory_utilization=0.75(默认 0.9 会在 KV 池预分配后激活 OOM)。
模型:/home/stark/models/Qwen2-0.5B-1M 已从 white 同步(md5 一致)。
GitHub:2 枚 read-only deploy key(插件仓 + 主仓走 github-main 别名),pip 走清华源。
laptop 至此具备 HCP ring 第三节点(真 CUDA worker)全部条件。

_updated: 2026-07-24 17:52:14_
### [2026-07-25] 无系统 CUDA 机器用 pip nvidia/cu13 工具链编译 vLLM 的四个坑 + OOM 教训

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `reflection`

laptop 无 /usr/local/cuda,纯 pip 工具链编译 vLLM 踩坑记录(white 有系统 cuda-13.1 所以没踩过):
1. pip 包装库只有带版本号文件(libcudart.so.13),缺无版本 dev 链接(libcudart.so),CMake find_library 找不到 => 批量 ln -s 建 dev 链接 + lib64→lib;
2. nvidia-cuda-nvcc 必须与 nvidia-nvvm/nvidia-cuda-crt 同版本,否则 ptxas 与 cicc 的 PTX ISA 不匹配(9.2 vs 9.3) => 三件统一 13.0.88,与 nvidia-cuda-runtime 头(13.0)一致,FindCUDA 的 nvcc/头版本检查才过;
3. CCCL 头(nv/thrust/cub/cuda)在独立包 nvidia-cuda-cccl-cu12 的 nvidia/cuda_cccl/include,要软链进 cu13/include(nvidia-cuda-cccl-cu13 只是 0.0.1 空壳);
4. laptop→GitHub HTTPS GnuTLS 抖动 => git config --global url."git@github.com:".insteadOf "https://github.com/",CMake FetchContent(cutlass/deepgemm/qutlass)全走 SSH;
5. 15GB 内存 + MAX_JOBS=8 => 编译 OOM(exit 137)且 tailscaled 被拖死整机掉线;MAX_JOBS=4 + nice -n 19 后 load 稳定 4.0,编译 ~2.5h 完成。教训:小内存机器编译先降并发保系统响应,CPU 限额非真凶、内存才是。

_updated: 2026-07-24 17:52:14_
### [2026-07-24] Mac 本地三仓同级布局落地,解耦在本机可开发

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.8 · source: `experiment`

decision-repo-decoupling-20260722 的收尾动作:本地 Mac 此前没有两个产品 repo 的 clone(只有主仓 + ~/VSCodeProjects/vllm 上游参考树)。现已在 ~/VSCodeProjects/ 下三仓同级:hetero-cp-ringattn(主仓)、hcp-vllm-plugin(HEAD 4c95561,与 GitHub 一致)、vllm-rocm-gfx1200(HEAD fbda49d,与 GitHub 一致)。清理主仓内拆仓遗留的 hcp_vllm_plugin/__pycache__ 陈旧缓存;主仓 README 新增三仓标准布局与边界说明(white/pearl 在 /home/stark/,Mac 在 ~/VSCodeProjects/;插件功能只在 hcp-vllm-plugin 改,gfx1200 兼容性只在 vllm-rocm-gfx1200 改,主仓只做调度核心/驱动/知识库),commit 1bf9b0c 已推送。后续 N>2 ring 的插件改动在 Mac 本地 hcp-vllm-plugin clone 进行,走 git 同步到 white/pearl。

_updated: 2026-07-24 08:34:57_
### [2026-07-22] gfx1200 适配 repo 整理完成:vllm-rocm-gfx1200(private),解耦全部落地

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.85 · source: `experiment`

github.com/stark-sim/vllm-rocm-gfx1200(private)。
内容:5 个补丁(从 pearl /home/stark/vllm 源码树 git diff 提取,base commit 3f99883d9 v0.23.1rc1.dev905):
0001 spinloop.cpp 改 x86intrin(ROCm Clang 23 编译错误);0002 禁用 GPTQ(HIP 缺 half2 atomicAdd);0003 ROCm 平台识别 torch.version.hip 兜底(amdsmi 不可用);0004 _get_gcn_arch 走 torch.cuda + HCP_ROCM_GCN_ARCH 覆盖;0005 pyproject 解除 torch==2.11.0 钉版。外加构建脚本(clone→checkout→apply→pip install -e)、LD_LIBRARY_PATH 运行 wrapper、README 兼容性矩阵。
pearl 迁移:插件 clone /home/stark/hcp-vllm-plugin + pip install -e 重装,compat_check 6/6 PASS;pearl GitHub 访问走已有用户 SSH key(known_hosts 补齐),跨网段 ssh 不稳时经 white(192.168.8.176)跳转。
两个产品 repo 至此全部独立: hcp-vllm-plugin + vllm-rocm-gfx1200;主仓=研究/驱动/知识库。

_updated: 2026-07-22 10:01:16_
### [2026-07-22] 第 1 步完成:vLLM 生态插件 v0.1 包装(entry point 自动注册双平台验证)

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.9 · source: `experiment`

决策 decision-vllm-plugin-packaging-20260721 的实施(commit c5e95a5)。
交付:
1. hcp_vllm_plugin/README.md:组件表、安装、兼容性(vLLM 0.23.1rc1 + torch 2.13,CUDA/ROCm 验证;依赖接口面明示:KVConnectorBase_V1 experimental / backend 注册表 / merge_attn_states / triton_utils)、快速开始(producer/consumer 配置 + 每请求 kv_transfer_params)、环境变量表、v0.1 限制清单(单 peer chunk、consumer 关 prefix caching、fp16/bf16、eager、kernel-hardening backlog 指向)、验证脚本索引;
2. hcp_vllm_plugin/compat_check.py:免 engine 冒烟——vllm 版本、KVConnectorBase_V1 方法面、CUSTOM 注册表、merge_attn_states、KVConnectorFactory、register() 执行、插件模块导入;pearl 与 white 均 PASS(0 warnings);
3. entry point 自动注册实证:探针 engine 不传 kv_connector_module_path,仅凭 kv_connector="HcpRingKvConnector" 解析成功并生成 token(pearl + white 均过),vllm.general_plugins 入口真正生效;
4. 包 docstring 更新为 ring memory-splitting 语义(原描述停留在全量 context-passing 时代)。
至此三步顺序(3 staging→2 kernel→1 packaging)全部完成,vLLM 线具备:可 pip install 的插件形态 + 双平台 triton kernel + 多请求连续批 CP + 跨节点异构闭环。

_updated: 2026-07-22 06:12:38_
### [2026-07-21] 第 2 步完成：ring attention 换自研 Triton kernel(带 LSE) + merge_attn_states,双平台验证通过

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

决策 decision-ring-paged-kernel-20260721 的实现(commit 0f7056c..18a1046)。
设计：vLLM 原生 triton kernel 不输出 LSE 且 TRITON_ATTN 不支持 cascade => 插件内自研 ring_triton_attn.py(fork vllm triton_prefill_attention,加 LSE 输出 + Q_OFFSET causal 偏移);同一 Triton kernel 覆盖 CUDA 与 ROCm(不再按平台分叉);local(causal+offset)/peer(non-causal) 两段都走它,merge 默认用 vllm merge_attn_states(triton);HCP_RING_IMPL/HCP_RING_MERGE 可切回 plain-torch 兜底;IMPL_STATS 计数器证明真实路径。
验证(全 PASS):
1. 数值探针(pearl gfx1200):kernel vs fp32 参考 max|diff|~1e-3(fp16 舍入),LSE ~1e-6;merge_attn_states vs plain merge 6.1e-5(无 inf);端到端两段合并 vs 全量 ~1e-4;
2. pearl 三件套(connector 单请求/并发/backend customst)triton 路径全 PASS,IMPL_STATS 216/408 次 triton 调用、0 torch 回退;
3. pearl 16k/8k 长上下文:PASS(24 层×8192 token staging、overlap 0);对照 HCP_RING_IMPL=torch 同规模 OOM 于 score 矩阵物化(exp(scores) 单次 3.50 GiB 分配失败)——kernel 化动机被实证;
4. white(RTX 4090 CUDA) 单请求+并发:PASS,同一 kernel,0 回退;
5. 跨节点并发复验 ringconc-014233(white producer 2 chunk + pearl consumer 2 并发):PASS。
过程修复:v load mask 转置 bug(Tk%BLOCK_N!=0 时越界键未清零,探针 Tq=37 暴露);validate_ring_backend customst 适配新 staging 签名(单 chunk 无映射回退)。
意义:score 矩阵不再物化,长上下文显存天花板消除,为 128K+/1M 的 vLLM 线扫清自实现障碍。

_updated: 2026-07-21 17:44:48_
### [2026-07-21] fork kernel 时 mask 的维度语义要逐行核对;数值探针必须覆盖非整除/边界形状

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `reflection`

ring_triton_attn fork 自 vllm triton_prefill_attention 时,v load 的 mask 被误写成与 qk mask 同形([1, BLOCK_N],而 v 布局是 [BLOCK_N, BLOCK_D]),Tk 为 BLOCK_N 整数倍时恰好全真不暴露,Tk=37 时越界键未清零产生垃圾/nan。教训:
1. fork kernel 时每一行 mask/stride 的维度语义都要与原布局核对,不能凭"形状能广播";
2. 数值探针形状集必须含非整除(Tq=37)、极小(Tq=1)、偏置(offset≠0)案例——本 bug 只有非整除案例暴露,整齐形状全过;
3. "LSE 全对但输出错"的定位价值:说明 online-softmax 记账正确,问题在数据装载(v mask)而非数学。

_updated: 2026-07-21 17:44:48_
### [2026-07-21] 多请求并发 CP 路径验证通过：staging 按 chunk 键 + 每请求 kv_transfer_params

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.9 · source: `experiment`

决策 decision-per-request-staging-20260721 的实现与验证(commit ec8e528..8bb2553)。
实现：
1. PEER_KV_STAGING 键从 layer 改为 (chunk_key, layer)；新增 PEER_REQ_MAP 以请求首块 id (生命周期内稳定)绑定请求→chunk，forward 按 batch 行查 peer KV；
2. 每请求参数走 vLLM 原生通道 SamplingParams.extra_args.kv_transfer_params.hcp_ring (chunk_id/prefix_len/peer_url)，全局 extra_config 保留为回退；显式 prefix_len=0 可退出 CP (存在性覆盖，非真值覆盖)——非 CP 请求与 CP 请求可同引擎混跑；
3. staged KV 按 chunk 引用计数，请求结束时释放；清理在携带 finished_req_ids 的那一步 forward 之前执行(connector metadata 携带)，shutdown() 兜底。
验证(pearl, ROCm)：
- validate_ring_concurrent.py：2 请求(各 1024 token、不同 prompt、各挂 peer chunk c0/c1)一次 generate(max_num_seqs=2)，token 与单节点参考全对；STAGING_STATS 显示 2 chunk×24 层并发 staging；BATCH_STATS.max_reqs=2(CP 路径真进连续批)；chunk-A 槽位本地写入 0；结束后 staging 清空。PASS。
- validate_ring_connector.py 单请求回归(2048/1024)：PASS。
跨节点复验(ringconc-232830, scripts/run_cross_node_ring_concurrent.sh, HEAD=8bb2553 双机一致)：
white(RTX 4090 CUDA) producer 算 2 个 chunk(c0/c1, 各 512 token)并经 HTTP 供取；pearl(ROCm gfx1200) consumer 一次 generate 提交 2 个全 prompt(1024 token),每请求经 kv_transfer_params 各挂各的 peer chunk。结果：双请求 token 与 pearl 单节点参考全对；
2 chunk×24 层并发 staging(经 HTTP 来自 white, producer 日志 GET 来源 100.111.242.55)；
BATCH_STATS.max_reqs=2(跨节点 CP 路径进连续批)；chunk-A 槽位本地写入 0；staging 用后释放。PASS。
排障记录(有复用价值)：
- 块 id 复用竞态(真修复)：finished 请求清理原在 forward 后(get_finished)，回收首块的新请求可能被绑到 过期 chunk；改为 metadata 携带 finished ids、start_load_kv 里 forward 前清理；
- 768 overlap 误报(非 bug)：验证脚本遗留 HCP_RING_SPLIT_TOKENS=1024 使短非 CP 请求落入 env-split 分支，WRITE_TRACK 把其自身写入误记为 peer 区域污染(32 槽×24 层)；attention 经 n_a 分支始终正确。connector 验证脚本已显式置 0。

_updated: 2026-07-21 15:30:29_
### [2026-07-21] 调试探针的假设要与所有激活路径核对，尤其遗留回退路径

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `reflection`

WRITE_TRACK 假设"peer 区域槽位绝不被本地写入"，该假设只对 connector-staged 路径成立；遗留的 HCP_RING_SPLIT_TOKENS env-split 路径(单进程 PoC 用)故意从本地 cache 读 peer,两条路径同时激活时探针把正常写入报成污染(768)。教训：
1. 新增验证手段时，穷举它会影响的所有代码路径(含遗留回退)，逐路径核对假设；
2. 验证脚本应显式固定行为相关环境变量(如 HCP_RING_SPLIT_TOKENS=0)，不依赖默认值；
3. 出现"数值异常但结果正确"时，先怀疑探针假设，再怀疑被测逻辑——token 全对 + overlap 异常 这个组合本身就是探针误报的特征。

_updated: 2026-07-21 15:30:29_
### [2026-07-21] 异构跨节点切分 CP 验证通过：white(CUDA) producer + pearl(ROCm) consumer 经 HcpRingKvConnector

type: `evidence` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `experiment`

run_id=ringx-210415，驱动 scripts/run_cross_node_ring_cp.sh，HEAD=cce069e（双机一致）。
拓扑：white(RTX 4090, vllm-v1) 以 CUSTOM backend(HcpRingAttentionBackend)+HcpRingKvConnector(role=producer) 只算 chunk A(2048-token prompt 的前 1024 token)，24 层 KV 存 safetensors 并经 HTTP(0.0.0.0:8901) 供取；
pearl(RX 9060 XT gfx1200, vllm-rocm) 以 CUSTOM backend+HcpRingKvConnector(role=consumer) 跑全 prompt，调度侧把 chunk A 标 external（全局 RoPE 位置、不重算），worker 侧经 HTTP 把 peer KV 拉进 ring backend 的 TRANSIENT PEER_KV_STAGING（不写 pearl 常驻 paged pool / block table），online softmax 合并 local(chunk B, causal)+peer(chunk A, transient)。
结果：
1. greedy 4 token 与 pearl 单节点参考完全一致：ref=[14579,220,22,21] cons=[14579,220,22,21]；
2. 末步 logits max|diff|=0.037（阈值 0.1，argmax 处 0.0）；
3. 显存切分证据：24/24 层 peer KV 经 HTTP 来自 white（producer 日志 GET 来自 100.111.242.55），1024 token/层；pearl 本地写 pool 槽位 1027（仅自身 chunk B），chunk-A 区域槽位本地写入 = 0；
4. report: reports/ring-cross-ringx-210415/{consumer,producer}.log。
意义：三步顺序（flash_attn→decode 充分验证→异构跨节点切分 CP）全部完成；vLLM worker 对 vLLM worker 组环 + KV connector 瞬时切分路线在真异构跨节点（CUDA↔ROCm）闭环。

_updated: 2026-07-21 13:08:24_
### [2026-07-21] HcpRingKvConnector：peer KV 以“切分瞬时”接入，2 进程显存切分验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `hcp_vllm_plugin/hcp_vllm_plugin/ring_connector.py + validate_ring_connector.py + /tmp/my_ring_val.log`

按用户约束（KV connector 默认是全量搬移，HCP 是切分瞬时）实现 HcpRingKvConnector（KVConnectorBase_V1）：调度侧 get_num_new_matched_tokens 仅把前序 chunk 标记为 external，给本 chunk 提供全局 RoPE 位置并阻止重复计算；worker 侧 start_load_kv 经 HTTP 从 producer 拉取 peer chunk 每层 KV，写入 ring_backend 的 PEER_KV_STAGING（瞬时），绝不写入常驻 paged pool——与 stock disaggregated-prefill 全量复制语义明确区分。ring_backend 增加 WRITE_TRACK 证明显存切分。验证（pearl 单机 2 个 vLLM 0.23 实例，CUSTOM backend + ring connector，2048-token prompt 切 1024+1024，greedy decode 4）：consumer tokens [14579,220,22,21] 与单节点一致，max|logit diff| 0.027（argmax 处 0.016），chunk-A 常驻池本地写入=0，peer KV 1024 tokens/layer×24 层全部经 HTTP 拉取（独立复跑通过，exit 0）。后续：跨节点（white CUDA producer + pearl ROCm consumer）、decode 充分性、性能（ROCm 无 flash_attn，目前 plain-PyTorch）。

_updated: 2026-07-21_
### [2026-07-21] flash_attn 平台现状：white CUDA 已可用，pearl ROCm 构建中

type: `evidence` · status: `closed` · confidence: 0.7 · importance: 0.8 · source: `white/pearl flash_attn probe`

flash_attn 双平台接通进展（下一步顺序第1步）：\n- white（CUDA，vLLM 0.23.1rc1）：无需单独装 flash_attn 包，vLLM vendored vllm_flash_attn 已可用，is_flash_attn_varlen_func_available()=True；实测 flash_attn_varlen_func(..., return_softmax_lse=True) 返回 (out [5,2,64], lse [2,5])，flash_attn+LSE 在 white 正常。\n- pearl（ROCm gfx1200，vLLM 0.23.1rc1）：is_flash_attn_varlen_func_available()=False（无 vendored ROCm flash_attn，也无 ROCm flash_attn 包，回退 Triton）。AMD 官方 index 无 gfx1200 预编译 flash-attn wheel。正在用 ROCm/flash-attention 的 main_perf 分支 + FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE（Triton 后端，硬件无关）源码构建，目标让 pearl 的 flash_attn 可用。\n注意：ROCm 的 flash_attn 是 ROCm/flash-attention fork，官方 flash-attn 为 CUDA-only；Triton 后端理论上可在 RDNA4 gfx1200 运行。

_updated: 2026-07-21_
### [2026-07-21] decode 充分验证：continuous batch + 多步 decode 全过（独立复跑）

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `hcp_vllm_plugin/validate_decode.py + /tmp/my_decode_val.log`

第2步 decode 充分验证（validate_decode.py，主 Agent 独立复跑确认，exit 0）：\n1) no-peer 退化 + 多步 decode：CUSTOM backend、HCP_RING_SPLIT_TOKENS=0，2048-token prompt greedy 16 token，全部匹配单节点参考（[220,23,15,74459,...]，max|logit diff| 0.023）。\n2) continuous batching：6 个长度 [64,200,350,700,1000,1500] 的 prompt 一次 generate 提交，BATCH_STATS.max_reqs=6 证明真在同一 attention step 批处理（非串行），6 个请求各 16 token 全部匹配单节点（diff 0.019–0.035）。证明 vLLM 连续批处理基础能力在 CUSTOM ring backend 下正常。\n3) CP 路径多步 decode：2 进程 ring-connector 切分（producer chunk A + consumer 全 prompt，HTTP 拉 peer KV），decode=8 与 decode=16 均 PASS，consumer 16 token [14579,220,22,21,...] 逐步匹配单节点；显存切分保持——consumer 写 1039 pool slots（1024 chunk-B prefill + 15 decode），chunk-A 常驻池本地写入=0。\n已知限制：PEER_KV_STAGING 按 layer 键，多并发 consumer 请求若 peer chunk 不同会互相覆盖，故 CP 路径限单并发（max_num_seqs=1）；no-peer 批处理无此限制。正确修法：staging 按 (request_id, layer) 键并把 request 身份经 attn_metadata 传入 forward（后续）。

_updated: 2026-07-21_
### [2026-07-17] vLLM 0.23.1rc1 源码编译补丁（gfx1200）

type: `evidence` · status: `held` · confidence: 0.75 · importance: 0.8 · source: `bash-i3gxwyr5 build log`

在 pearl（RX 9060 XT / gfx1200，ROCm 7.13，PyTorch 2.13.0a0+rocm7.13.0a20260416）上从 main 分支源码构建 vLLM 0.23.1rc1.dev905+g3f99883d9。为通过编译已打两个补丁：1) csrc/spinloop.cpp：将 <mwaitxintrin.h> 改为 <x86intrin.h>，修复 ROCm Clang 23 直接包含 mwaitxintrin 的编译错误；2) 禁用 GPTQ 路径：从 CMakeLists.txt 移除 csrc/libtorch_stable/quantization/gptq/q_gemm.cu，并在 csrc/libtorch_stable/torch_bindings.cpp 中注释 gptq_gemm/gptq_shuffle 的 ops.def/ops.impl，规避 HIP half2 atomicAdd 缺失导致的编译失败。当前构建仍在后台运行（task bash-i3gxwyr5），正在编译 HIP 对象。

_updated: 2026-07-17_
### [2026-07-17] vLLM 0.23.1rc1 源码编译成功并通过 ROCm gfx1200 prefill

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `bash-vjw890d3 build log + /tmp/test_vllm_prefill.py`

在 pearl（RX 9060 XT / gfx1200，ROCm 7.13，PyTorch 2.13.0a0+rocm7.13.0a20260416）上完成 vLLM 0.23.1rc1.dev905+g3f99883d9 源码编译。关键修复：1) csrc/spinloop.cpp 用 <x86intrin.h> 替代 <mwaitxintrin.h>；2) 禁用 GPTQ（CMakeLists 移除 q_gemm.cu，torch_bindings 注释 gptq_gemm/gptq_shuffle）规避 HIP half2 atomicAdd 缺失；3) 在 conda env bin 下把 clang/clang++/clang-cpp 软链到 amdclang/amdclang++/amdclang-cpp，修复 hipcc_cmake_linker_helper 链接失败。运行时用 LD_LIBRARY_PATH 覆盖 torch/lib 与 _rocm_sdk_{core,devel}/lib{,/host-math/lib,/rocm_sysdeps/lib}。验证：从 /tmp 运行脚本（避免 cwd=/home/stark 时 repo 目录名 vllm 把 import 变成 namespace package），LLM(model=/home/stark/models/Qwen2-0.5B-1M, dtype=float16, enforce_eager=True) 成功初始化并在 ROCm gfx1200 上 prefill+decode，输出 I am 类 token。

_updated: 2026-07-17_
### [2026-07-17] vLLM 0.23 V1 Block-Ring 插件在 pearl(gfx1200) 验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `pearl /tmp/poc_out.log + scripts/poc_vllm_block_ring_v1.py`

在 pearl（RX 9060 XT / gfx1200，自编译 vLLM 0.23.1rc1.dev905+g3f99883d9，ROCm 7.13）上实现并验证 V1 引擎版 Block-Ring 插件。实现：python/hcp_vllm_block_ring_plugin_v1.py 用 enable_multiprocessing=False 的 LLMEngine 同进程访问 model_executor/scheduler/KV cache，直接用 block_pool 分配物理块，手工构造 SchedulerOutput+NewRequestData 调 model_executor.execute_model（返回 None 时再 sample_tokens），KV cache 布局与 0.6.4 一致 [2, num_blocks, block_size, num_kv_heads, head_dim]。PoC：scripts/poc_vllm_block_ring_v1.py，Qwen2-0.5B-1M、fp32、block_size=16、chunk 16+16。结果：chunk A prefill + chunk B 带 context prefill + combined block table，最后位置 next token 与单节点 vLLM 参考一致（match=True），自回归 decode 第二个 token 也一致（match=True）。注意事项：1) ROCm attention 后端不支持 block_size=8，需用 16；2) 1M 模型默认 max_model_len=1048576 会导致 KV cache 初始化 OOM，插件/参考都需显式传 max_model_len（如 4096）；3) 运行 cwd 不能在含 vllm 子目录的路径（否则 import vllm 变 namespace package）；4) 运行需 LD_LIBRARY_PATH 覆盖 torch/lib 与 _rocm_sdk_{core,devel}/lib{,/host-math/lib,/rocm_sysdeps/lib}。

_updated: 2026-07-17_
### [2026-07-17] 跨节点 vLLM Block-Ring CP 验证通过（white CUDA + pearl ROCm）

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/vllm-cp-cuda-hip-20260717-134004 + scripts/run_cross_node_vllm_cp.sh`

首次实现 vLLM 跨节点 context-passing CP：white（RTX 4090 CUDA，vLLM 0.6.4 legacy 插件）作 domain 0，pearl（RX 9060 XT ROCm gfx1200，vLLM 0.23.1rc1 V1 插件）作 domain 1，经 Rust coordinator + QUIC KV ring 协作同一序列。关键设计：vLLM PagedAttention 的正确 CP 必须 context-passing——domain 1 先收 domain 0 的 chunk A KV 作 context 再 prefill chunk B（层 L 的 K/V 依赖层 L-1 的 context，先 prefill 再交换在数学上不正确）。新增：plugins 的 prefill_with_context_kv / set_global_seq_len / _local_seq_offset，decode/last_token 用 _global_seq_len（peer chunk 的 token id 不需要，早期 token 用占位符）；python/hcp_worker_sdk/cp_server.py（CpVllmWorkerServer，domain0 send-then-recv、domain1 recv-then-prefill-then-send）；python/hcp_vllm_cp_worker.py（自动识别 vLLM 0.6.x vs >=0.23）；scripts/run_cross_node_vllm_cp.sh。修复：domain 0 需按 prefill 时的 seq_len 上报，否则 coordinator 会错用其 chunk-local logits。验证：64-token 变化 prompt（alpha bravo ... qu），chunks 32+32，block_size 16，greedy 6 token，distributed 输出 ail rose rosemary rosewood 与单节点 vLLM 完全一致。已知限制：KvBlock 布局 [num_blocks, block_size, kv_heads, head_dim] 与 transformers/Rust 的 [batch, heads, seq, dim] 不同，故 vLLM worker 目前只能与 vLLM worker 组环。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector（KVConnectorBase_V1）单机 2 实例验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `scripts/poc_hcp_cp_connector.py + /tmp/poc_conn.log`

实现 vLLM 官方 KV connector 扩展点版本的 context-passing CP：hcp_vllm_plugin/ 包（pyproject + vllm.general_plugins 入口 + kv_connector_module_path），HcpCpConnector 以 ExampleConnector 为模板，producer 计算本 chunk 并共享存储 KV，consumer 把前序 chunk 标记为 external prefix（get_num_new_matched_tokens）只算本 chunk。关键修复：同步共享路径 load 必须返回 load_kv_async=False；get_finished 返回 (None,None) 避免 scheduler 断言。验证（pearl 单机 2 实例，vLLM 0.23.1rc1，Qwen2-0.5B-1M，64-token 变化 prompt，chunk 32+32）：consumer 首 token 604(ail) 与单节点参考一致，exit=0。该路线不打补丁、用官方稳定 API，故能跟进 vLLM 官方更新。注意：KV connector 仅 V1 引擎支持，跨节点异构需 white 也构建 V1 vLLM（当前 white 为 0.6.4）。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector HTTP 跨机传输验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `scripts/poc_hcp_cp_connector.py --http-port + /tmp/poc_http.log`

为 HcpCpConnector 增加 HTTP 跨机 KV 传输：producer 端仅 worker-side 起 ThreadingHTTPServer 共享 KV store（cp_serve_port），consumer 端 cp_peer_url 拉取（HEAD 探活 _READY，GET 拉 layer safetensors），带 5 次重试解决 IncompleteRead。修复：connector 按 role 实例化两次（scheduler+worker），HTTP server 只能 worker-side 绑定否则端口冲突。验证（pearl 单机 2 实例经 loopback HTTP，64-token，chunk 32+32）：0 个 fetch 失败，consumer 604(ail) 与单节点一致。至此 HCP 已成为一个不依赖补丁、基于 vLLM 官方 KVConnectorBase_V1 稳定 API 的生态插件，可跨机做 context-passing CP。异构跨节点（white CUDA + pearl ROCm）仍需 white 构建 V1 引擎 vLLM（当前 white 为 0.6.4 legacy）。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector 跨节点异构验证通过（white CUDA + pearl ROCm）

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/cp-plugin-cpplug-201341 + scripts/run_cross_node_cp_plugin.sh`

完成 HCP 作为 vLLM 生态插件的跨节点异构验证。先在 white（RTX 4090）构建 V1 引擎 vLLM 0.23.1rc1.dev905+g3f99883d9：新建 conda env vllm-v1，装 torch 2.13.0+cu126（下载慢约50min），clone 到 3f99883d9，装 build 依赖（cmake<4、ninja、setuptools-rust），关键是用 conda gcc-13 作为 nvcc host compiler 解决 Ubuntu 26.04 glibc 2.43 + CUDA 13.1 + gcc-15 的 rsqrt exception-spec 冲突；pip 装上 cu130 torch + torchvision 后 vLLM 0.23 在 white 跑通 prefill。随后跨节点：white producer（CUDA，HcpCpConnector，cp_serve_port=8899）算 chunk A 并经 HTTP 供 KV，pearl consumer（ROCm gfx1200，HcpCpConnector，cp_peer_url=http://white:8899）拉取 chunk A KV 作 external prefix 算 chunk B。验证（Qwen2-0.5B-1M，64-token 变化 prompt，chunk 32+32，greedy 4 token）：consumer [604,16009,16009,1534]=ail rose rosemary 与单节点参考完全一致。至此 HCP 是一个不打补丁、基于官方 KVConnectorBase_V1 稳定 API、可跨异构节点做 context-passing CP 的 vLLM 生态插件，能跟进 vLLM 官方更新。

_updated: 2026-07-17_
### [2026-07-17] HcpRingAttentionBackend：vLLM 显存切分 online softmax ring attention 验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `hcp_vllm_plugin/hcp_vllm_plugin/ring_backend.py + validate_ring_backend.py + /tmp/ring_val.log`

实现 vLLM 显存切分（memory-splitting）online softmax ring attention：自定义 attention backend HcpRingAttentionBackend（FlashAttentionBackend 子类，注册为 CUSTOM，vllm.general_plugins 入口）。每个 worker 只永久持有自己 chunk 的 KV，attention 时对 local chunk（causal）与 transient peer chunk（non-causal）分别计算 (O, LSE)，用 plain-PyTorch online softmax 合并，peer KV 经 PEER_KV_STAGING 瞬时暂存而不入 paged pool。RoPE 位置：单请求全 prompt，backend 按 HCP_RING_SPLIT_TOKENS 切分 peer/local，数学上等价 2-worker 分片。验证（pearl ROCm gfx1200，vLLM 0.23.1rc1，Qwen2-0.5B-1M，2048-token，split=1024，greedy）：ref/custom0/custom/customst 四种模式 sampled token 均 14579，top-5 集合一致，logits 差异在 fp16 噪声内（独立复跑通过）。ROCm 事实：flash_attn 未安装故用 plain-PyTorch attention（fp32 累加，correctness-grade）；merge_attn_states 未用（Triton 内核在 ROCm 有 inf 问题）。后续：KV connector 接线真实网络 peer KV、2 进程全局位置偏移、decode 阶段验证、性能优化。

_updated: 2026-07-17_
### [2026-07-02] vLLM Block Ring 插件骨架与 PoC 修正

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `git commit 3467cb4`

在 white 已可运行 vLLM 0.6.4 的基础上，继续完善插件实现并提交 commit 3467cb4。\n\n变更点：\n- python/hcp_vllm_block_ring_plugin.py：实现 VllmBlockRingPlugin.prefill / decode / get_kv_block / apply_peer_kv，直接调用 vLLM model_executor 绕过 scheduler。\n- 为 peer KV 在所有 attention 层复用同一组物理 block，避免 vLLM block table 跨层不一致。\n- 增加 _rope_delta_rotate_keys：对以 local position 预fill 的 peer key 做 RoPE delta 旋转，使其 global position 与 decode query 对齐。\n- scripts/poc_vllm_block_ring_2worker.py：修正 decode 输入为最后 prompt token，使用 set_global_tokens 同步完整序列，默认 prompt 长度满足 block_size 对齐断言。\n\n限制：\n- prefill() 目前返回 one-hot sampled token（非完整 last-token logits），与 HcpWorkerBackend 接口兼容但语义上是近似。\n- RoPE 校正目前只支持标准 Neox 配对 RoPE 与 rope_theta；尚未处理 rope_scaling / Yarn / NTK。\n- 需要等待 pearl 上 vLLM 源码编译完成后才能做真实 ROCm 硬件验证。

_updated: 2026-07-02 14:58:04_
### [2026-07-01] 搜索 vLLM RDNA4/gfx1200 社区轮子结果

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `web search`

搜索结论：\n\n1. 未发现可直接 pip install 的 vLLM 0.6.4 gfx1200 预编译 wheel。\n2. ROCm TheRock 提供 per-family nightly Python 包：gfx120X-all 索引（https://rocm.nightlies.amd.com/v2/gfx120X-all/），包含 PyTorch/ROCm 对 gfx1200/gfx1201 的支持。\n3. vLLM 上游 rocm/vllm-dev:base Docker 的 PYTORCH_ROCM_ARCH 已包含 gfx1200;gfx1201，说明源码构建的 arch list 已经支持。\n4. Step-Audio 的 Dockerfile 展示了在 gfx1151/gfx1200/gfx1201 上源码构建 vLLM 的 patch 路径。\n5. 社区 ROCmLibs 提供 gfx1201 的 hipblaslt/rocblas 库，但主要用于 Windows/koboldcpp。\n\n结论：没有现成轮子；最可行路径是用 TheRock gfx120X-all  nightly PyTorch + 源码编译 vLLM 0.6.4，目标 arch gfx1200。

_updated: 2026-07-02 14:15:47_
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
### [2026-06-17] 昇腾 910B NPU 控制面 E2E 打通

type: `session` · status: `closed` · confidence: 1.0 · importance: 0.75 · source: `memory-bank/progress.md`

单机 1× Ascend 910B4 (32 GB HBM) 上完成 Python vLLM worker ↔ Rust coordinator 控制面 E2E。Rust coordinator 脱离 libtorch feature 可编译运行，纯 Rust 采样替代 tch::Tensor。Coordinator 输出 generated: ! I'm。

_updated: 2026-06-29 05:34:19_
### 证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

White CUDA loopback 双 domain 3B max_diff=0.406，0.5B max_diff=0.344，argmax=10/10。跨平台单节点 0.438，异构分布式 0.484。证明跨平台 BLAS 仅贡献 ~0.1 额外差异，不是 logits 差异主导因素。

_updated: 2026-06-29 05:34:19_
