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

当前 `tcp_remote_pair` 仍是最小双机握手，不是最终 CP runtime 拓扑。在真正的 Context Parallel P2P 模式中，每个节点都应该同时承担 inbound receiver 和 outbound sender 的职责：既监听上游 / peer 发来的 K/V block，也主动向 ring next domain 转发自己的或已接收的 K/V block。当前 smoke 只证明一条双机链路上双向 frame 可以互通和校验，还没有验证多节点 ring、并发收发、每节点同时 server/client、多 block 持续转发或 device-side attention compute。

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

下一步应把 `local_p2p_queue` 和 `tcp_remote_pair` 抽成统一 transport trait，并把 remote role 从单一 server/client 扩展成每个 domain 同时具备 listener + outbound peer 的 node runtime。后续 transport 可以扩展到 UCX/RDMA、NCCL send/recv、共享内存或 GPU-direct 路线。
