# Ring Attention Protocol Smoke

## 目标

本阶段把 Ring Attention / Context Parallel 中“ring 内流动的东西”从内存内抽象推进成可序列化、可发送、可接收、可检查的协议单元。

它不是性能实现。当前目标是固定低边界协议语义，并用本地 P2P 队列表达 point-to-point send / recv 语义：

- 每个 source domain 按自己的 `block_size` 切出 K/V block。
- K/V block 沿 ring 逐 hop 转发。
- 每个 receiver 可以从消息中恢复 source / sender / receiver / block metadata。
- softmax state 和 terminate 控制消息也走同一套编码路径。

## Context Parallel 数据流

当前 smoke 使用 3 个 domain：

```text
domain-0 -> domain-1 -> domain-2 -> domain-0
```

对任意 source domain 的一个 K/V block：

1. source domain 本地消费该 block。
2. source domain 将 block 发送给 ring next domain。
3. receiver 解码并验证消息。
4. receiver 继续将同一个 source block 转发给下一个 domain。
5. 直到其他 domain 都收到该 source block。

因此 `N` 个 domain 下，每个 source block 会产生 `N - 1` 条跨域 K/V block message。

## Message Schema

Rust 侧当前定义的最小 schema 包括：

- `schema_version`
- `sequence_id`
- `layer_index`
- `ring_step`
- `source_domain`
- `sender_domain`
- `receiver_domain`
- `message_kind`
- `payload_kind`
- `block`
- `tensor`
- `payload`

其中：

- `source_domain` 表示 K/V block 的原始归属 domain。
- `sender_domain` 表示当前 hop 的发送方。
- `receiver_domain` 表示当前 hop 的接收方。
- `block` 记录 `global_offset`、`block_len`、`source_seq_offset`。
- `tensor` 记录 dtype、head shape、payload byte size 和 checksum。

当前 wire format 使用 `serde_json`，因为本阶段优先可读性和可诊断性。后续如果需要性能，可以在 schema 稳定后替换成二进制格式。

## P2P Transport 语义

这里的 P2P 不是特指 IP/TCP。它表示跨 domain 只做点对点消息传递，不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主通信假设。

当前 smoke 使用 `local_p2p_queue`：

- 每个 domain 有独立 inbox。
- send 必须指定 sender / receiver。
- recv 会校验 expected sender。
- frame 内容仍是完整序列化后的 `RingAttnMessage`。

这一步先固定协议语义，不绑定到底层 IP/TCP。后续可以把同一个 `RingAttnMessage` 映射到 TCP、UCX/RDMA、NCCL send/recv、共享内存或自定义 GPU-direct transport。

远端 NVIDIA GPU 当前在 `192.168.8.172`。代码变更只通过 git 同步：本地提交并 push，远端只执行 `git pull` 和 smoke 命令，不在远端直接编辑源码。

## Remote P2P Pair Smoke

`tcp_remote_pair` 是一个双进程 / 双机器 smoke transport，用于确认同一套 `RingAttnMessage` 可以跨机器点对点发送、接收和校验。它不改变 P2P 的定义：P2P 仍是 point-to-point protocol 语义，TCP 只是本阶段最小可诊断的工程传输。

双机 smoke 不应使用 `127.0.0.1` 作为结论。远端 GPU 节点监听 `0.0.0.0:29172`，本机 client 连接 `192.168.8.172:29172`：

```bash
# 远端 GPU 节点执行，只通过 git 同步源码
cd ~/hetero-cp-ringattn
git pull
RUN_ID=rust-remote-p2p-$(date +%Y%m%d-%H%M%S) \
  BIND_ADDR=0.0.0.0:29172 \
  CARGO_OFFLINE=0 \
  bash scripts/run_rust_remote_p2p_server.sh
```

```bash
# 本机执行
RUN_ID=<same-run-id> \
  CONNECT_ADDR=192.168.8.172:29172 \
  CARGO_OFFLINE=0 \
  bash scripts/run_rust_remote_p2p_client.sh
```

当前 remote pair 会验证 3 条消息：

- client `mac-mps` 向 server `gpu-cuda` 发送 `kv_block`。
- server `gpu-cuda` 向 client `mac-mps` 返回 `softmax_state` ack。
- client `mac-mps` 向 server `gpu-cuda` 发送 `terminate`。

server report 预期 `sent=1 received=2`，client report 预期 `sent=2 received=1`。

当前 `tcp_remote_pair` 仍是最小双机握手，不是最终 CP runtime 拓扑。在真正的 Context Parallel P2P 模式中，每个节点都应该同时承担 inbound receiver 和 outbound sender 的职责：既监听上游 / peer 发来的 K/V block，也主动向 ring next domain 转发自己的或已接收的 K/V block。

## CP Ring Node Runtime Smoke

`cp_ring_node_runtime` 是当前的并发 CP ring 语义 smoke。它仍运行在单进程内，但每个 domain 对应一个 Rust thread，每个 thread 同时持有：

- inbound receiver：从 ring previous domain 接收 K/V block。
- outbound peer：向 ring next domain 发送本地 source block 或转发收到的 block。
- local compute update counter：收到每个远端 block 时计数，本地 source block 也计入本 domain 的 compute update。

默认 3 domain 配置下共有 10 个 source blocks。每个 block 跨域传递 `domain_count - 1` hop，因此预期 K/V message 数为 `10 * 2 = 20`。每个 domain 都会消费全部 10 个 source blocks，因此总 compute update 数为 `10 * 3 = 30`。

通过时 CLI 会显示：

```text
cp_ring_status=pass cp_ring_messages=20 cp_ring_compute_updates=30
```

这一步已经验证每节点双角色、多 block 持续转发和并发收发的协议语义；仍未验证真实双机多节点 TCP 拓扑。

`torch_attention_bridge` 已提供独立的 device-side attention compute smoke：C++ ATen 在请求设备上计算小尺寸 `softmax(QK^T / sqrt(d))V`，并与 CPU reference 对比。

`torch_block_update_bridge` 已将 `cp_ring_node_runtime` 的 `compute_updates=30` 接入 C++ ATen bridge：Rust 将 update 数传给 C++，C++ 在请求设备上执行同等次数的 attention block compute。通过时 CLI 会显示：

```text
torch_block_update_status=pass torch_block_update_code=2 torch_block_updates=30  # MPS
torch_block_update_status=pass torch_block_update_code=3 torch_block_updates=30  # CUDA
```

`torch_payload_block_bridge` 进一步把 `RingAttnMessage.payload` 改为 float32 K/V bytes，并让 `cp_ring_node_runtime` 捕获每次 compute update 的 payload block。Rust 逐块把这些 payload 交给 C++ ATen，C++ 在请求设备上执行 payload-backed attention compute。通过时 CLI 会显示：

```text
torch_payload_block_status=pass torch_payload_block_code=2 torch_payload_blocks=30/30  # MPS
torch_payload_block_status=pass torch_payload_block_code=3 torch_payload_blocks=30/30  # CUDA
```

这一步验证的是 CP ring update 事件已经能驱动真实消息 payload 的 device-side compute。`torch_payload_online_bridge` 进一步在设备上逐 block 维护 running max / running sum / output，并与 full attention CPU reference 对比。`torch_payload_chunk_bridge` 则把 online update 扩展到小尺寸 Q chunk，输出 `[query, head, dim]` chunk tensor。`torch_query_chunk_bridge` 在此基础上把 Q 改为 Rust/domain-side 显式 float32 payload，C++ bridge 不再为该路径内部构造 Q。

`DomainModelState` 是当前最小模型状态边界：每个 domain 持有自己的 Q chunk 和 K/V storage。source domain 发送 K/V block 时从自己的 K/V storage 切片；target domain 捕获 compute update 时带上自己的 Q payload。这样 query bridge 不再把不同 compute domain 的 blocks 混用同一个 Q，而是按 `compute_domain` 分组调用 C++ ATen bridge。

## Remote CP Node Smoke

`tcp_remote_cp_node` 是当前的 remote dual-role node smoke。每个进程同时做两件事：

- listener：接收 peer 发来的 K/V block。
- outbound peer：主动连接 peer listener，并发送本地 source blocks。
- forwarder：当收到的 K/V block 还没有走完 ring 时，继续转发给 outbound peer。

当前 2-domain remote 配置为 `mac-mps <-> gpu-cuda`，每个 domain 有 4 个 source blocks。由于只有两个 domain，每个 block 只跨一个 hop，因此每个 node 预期：

- `messages_sent=4`
- `messages_received=4`
- `compute_updates=8`
- `torch_payload_blocks=8/8`，当启用 `HCP_ENABLE_TORCH=1` 时每个 node 都在本地请求设备上消费本地与 peer 的 K/V payload。

已验证通过的双机命令形态：

```bash
# 本机先启动，避免 GPU 反连 Mac 时错过 listener
RUN_ID=rust-remote-cp-node-<timestamp> \
  NODE_INDEX=0 \
  BIND_ADDR=0.0.0.0:29176 \
  CONNECT_ADDR=192.168.8.172:29175 \
  CARGO_OFFLINE=0 \
  HCP_ENABLE_TORCH=1 \
  HCP_TORCH_DEVICE=mps \
  bash scripts/run_rust_remote_cp_node.sh
```

```bash
# GPU 节点随后启动；先用 `ifconfig | rg 'inet 192\.168\.8\.'` 确认当前 Mac 地址
PATH=/home/stark/.cargo/bin:$PATH \
  LIBTORCH=/home/stark/libtorch \
  LIBTORCH_INCLUDE=/home/stark/libtorch/include \
  LIBTORCH_LIB=/home/stark/libtorch/lib \
  LD_LIBRARY_PATH=/home/stark/libtorch/lib:$LD_LIBRARY_PATH \
  RUN_ID=rust-remote-cp-node-<timestamp> \
  NODE_INDEX=1 \
  BIND_ADDR=0.0.0.0:29175 \
  CONNECT_ADDR=<MAC_192_ADDR>:29176 \
  CARGO_OFFLINE=0 \
  HCP_ENABLE_TORCH=1 \
  HCP_TORCH_DEVICE=cuda:0 \
  bash scripts/run_rust_remote_cp_node.sh
```

这一步已经验证双机每节点双角色、双向多 block 持续收发，以及每个 node 对 8 个 captured K/V payload blocks 的本地 device-side ATen compute。由于 remote 版本目前是 2-domain，它没有中间节点，因此不覆盖 remote 多 hop forwarding；多 hop forwarding 仍由本地 3-domain `cp_ring_node_runtime` 覆盖。

macOS 上非阻塞 listener accept 出来的 stream 可能继承 nonblocking 状态；当前实现会在 `accept_with_retry` 成功后显式切回 blocking stream，避免大 payload frame 读取时出现 `WouldBlock`。

### 3-Node Remote CP Smoke

设置 `HCP_REMOTE_CP_DOMAINS=3` 后，remote 拓扑为：

```text
node0 mac-mps -> node1 gpu-cuda -> node2 mac-mps-2 -> node0 mac-mps
```

当前验证使用两台物理机器上的三个进程：node0 和 node2 在 Mac，node1 在 GPU host。每个 domain 仍有 4 个 source blocks；3-domain ring 中每个 block 跨 2 hop，因此每个 node 预期：

- `messages_sent=8`，其中 4 条本地 source blocks，4 条 forwarded peer blocks。
- `messages_received=8`，来自上游 peer 的本地 source blocks 和 forwarded blocks。
- `compute_updates=12`，本地 4 blocks + 远端 8 blocks。
- `torch_payload_blocks=12/12`，本地设备消费全部 captured payload blocks。
- `torch_payload_online_blocks=12/12`，本地设备逐 block 维护 online softmax state 并通过 CPU reference 对比。
- `torch_payload_chunk_blocks=12/12`，本地设备对小尺寸 Q chunk 维护 online softmax state 并输出 chunk tensor。
- `torch_query_chunk_blocks=12/12`，本地设备消费 Rust/domain-side Q payload 与 captured K/V payload blocks。
- `torch_query_output_blocks=12/12`，本地设备返回 Q chunk output digest，并由 Rust report 记录 output checksum / preview。

推荐使用统一 launcher，而不是手工分别启动三个节点：

```bash
RUN_ID=rust-remote-cp-3node-<timestamp> \
  PORT_BASE=29285 \
  bash scripts/run_rust_remote_cp_3node_smoke.sh
```

该脚本会：

- 自动发现当前 Mac 的 `192.168.8.x` 或 `100.x` 地址，也可用 `MAC_NODE_ADDR` 覆盖；`MAC_192_ADDR` 仅保留为兼容旧命令的别名。
- 在 GPU host 上执行 `git pull --ff-only`，保持远端源码只通过 git 同步；默认 GPU host 为 `192.168.8.172`，临时 VPN 路由可用 `GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138` 覆盖。
- 对本机和远端先做 cargo preflight build，降低节点启动后因编译耗时错过 retry 窗口的风险。
- 统一启动 node0、node2、node1，并把 launcher stdout/stderr 写入 `reports/<RUN_ID>/launch_node_<N>.log`。
- 默认本机节点使用 `HCP_TORCH_DEVICE=mps`，GPU 节点使用 `HCP_TORCH_DEVICE=cuda:0`。

如果需要手工排查，仍可直接调用 `scripts/run_rust_remote_cp_node.sh`，但正式 smoke 应优先使用统一 launcher，避免三节点启动时序由人工操作决定。

已通过的结果：

```text
node0: sent=8 received=8 compute_updates=12 torch_payload_block_code=2 torch_payload_blocks=12/12 torch_payload_online_code=2 torch_payload_online_blocks=12/12 torch_payload_chunk_code=2 torch_payload_chunk_blocks=12/12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12 torch_query_output_code=2 torch_query_output_blocks=12/12
node1: sent=8 received=8 compute_updates=12 torch_payload_block_code=3 torch_payload_blocks=12/12 torch_payload_online_code=3 torch_payload_online_blocks=12/12 torch_payload_chunk_code=3 torch_payload_chunk_blocks=12/12 torch_query_chunk_code=3 torch_query_chunk_blocks=12/12 torch_query_output_code=3 torch_query_output_blocks=12/12
node2: sent=8 received=8 compute_updates=12 torch_payload_block_code=2 torch_payload_blocks=12/12 torch_payload_online_code=2 torch_payload_online_blocks=12/12 torch_payload_chunk_code=2 torch_payload_chunk_blocks=12/12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12 torch_query_output_code=2 torch_query_output_blocks=12/12
```

`torch_query_chunk_bridge` 的主 smoke 已在本机 MPS 与远端 CUDA 验证通过：

```text
Mac MPS: torch_query_chunk_status=pass torch_query_chunk_code=2 torch_query_chunk_blocks=30/30
GPU CUDA: torch_query_chunk_status=pass torch_query_chunk_code=3 torch_query_chunk_blocks=30/30
```

2026-04-26 已用 `RUN_ID=rust-remote-cp-queryq-20260426-rerun2` 重跑 3-node remote CP query chunk smoke。上一轮失败根因是 Mac 的 `192.168.8.x` 地址从 `192.168.8.204` 变化到 `192.168.8.239`，node1/node2 仍在连接旧地址；使用当前地址后三节点均通过：

```text
node0: sent=8 received=8 compute_updates=12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12
node1: sent=8 received=8 compute_updates=12 torch_query_chunk_code=3 torch_query_chunk_blocks=12/12
node2: sent=8 received=8 compute_updates=12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12
```

2026-04-26 已用 `RUN_ID=rust-remote-cp-modelstate-20260426` 验证 `DomainModelState` 路径。结果同样通过：

```text
node0: sent=8 received=8 compute_updates=12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12
node1: sent=8 received=8 compute_updates=12 torch_query_chunk_code=3 torch_query_chunk_blocks=12/12
node2: sent=8 received=8 compute_updates=12 torch_query_chunk_code=2 torch_query_chunk_blocks=12/12
```

2026-04-26 已用 `RUN_ID=rust-remote-cp-output-unified-20260426` 验证统一 launcher 和 query output digest 路径。结果通过：

```text
node0: sent=8 received=8 compute_updates=12 torch_query_output_code=2 torch_query_output_blocks=12/12
node1: sent=8 received=8 compute_updates=12 torch_query_output_code=3 torch_query_output_blocks=12/12
node2: sent=8 received=8 compute_updates=12 torch_query_output_code=2 torch_query_output_blocks=12/12
```

2026-04-27 已用临时 VPN 地址 `GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138` 重跑 `RUN_ID=rust-remote-cp-modelstate-vpn-20260426`。远端先从 `d868522` fast-forward 到 `a84dc9c`，随后三节点均通过：

```text
node0: sent=8 received=8 compute_updates=12 torch_query_output_code=2 torch_query_output_blocks=12/12 output_seq_offset=0 output_slot_values=288
node1: sent=8 received=8 compute_updates=12 torch_query_output_code=3 torch_query_output_blocks=12/12 output_seq_offset=32 output_slot_values=288
node2: sent=8 received=8 compute_updates=12 torch_query_output_code=2 torch_query_output_blocks=12/12 output_seq_offset=64 output_slot_values=288
```

## Smoke Report

运行：

```bash
bash scripts/run_rust_ringattn_smoke.sh
```

本机硬件验证仍使用 MPS：

```bash
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh
```

通过时 CLI 会显示：

```text
protocol_status=pass protocol_messages=22
cp_ring_status=pass cp_ring_messages=20 cp_ring_compute_updates=30
torch_block_update_status=pass torch_block_update_code=<2|3> torch_block_updates=30
torch_payload_block_status=pass torch_payload_block_code=<2|3> torch_payload_blocks=30/30
torch_payload_online_status=pass torch_payload_online_code=<2|3> torch_payload_online_blocks=30/30
torch_payload_chunk_status=pass torch_payload_chunk_code=<2|3> torch_payload_chunk_blocks=30/30
```

JSON report 中新增 `protocol_smoke`：

```json
{
  "status": "pass",
  "transport": "local_p2p_queue",
  "schema_version": 1,
  "domains": 3,
  "source_blocks": 10,
  "summary": {
    "kv_block_messages": 20,
    "softmax_state_messages": 1,
    "terminate_messages": 1,
    "messages_sent": 22,
    "messages_received": 22
  }
}
```

`route_preview` 会记录前几条 K/V block 的 source / sender / receiver / block range，用于检查 ring order 是否符合 Context Parallel 预期。

`reports/**/*.json` 是生成产物，默认被 `.gitignore` 忽略；需要沉淀实验结论时，应优先写入文档或 memory-bank，而不是提交 raw report JSON。

## 后续

下一步应把 smoke 中的 synthetic Q chunk 换成 domain-local Q payload / state lifecycle，并抽出统一 transport trait。后续 transport 可以扩展到 UCX/RDMA、NCCL send/recv、共享内存或 GPU-direct 路线。
