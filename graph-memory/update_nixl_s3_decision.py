import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"

TASK = "task-nixl-s3-cross-machine-transfer-20260816"
DECISION = "decision-nixl-s3-cross-machine-20260816"
OLD_DECISION = "decision-nixl-as-transport-20260816"
INQUIRY = "inquiry-nixl-as-hcp-transport-20260815"

def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence, source):
    conn.execute(
        "INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now')) ON CONFLICT(id) DO UPDATE SET type=excluded.type,layer=excluded.layer,project=excluded.project,title=excluded.title,content=excluded.content,importance=excluded.importance,confidence=excluded.confidence,status=excluded.status,source=excluded.source,updated_at=datetime('now')",
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, source),
    )

def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        "INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?) ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note",
        (source, target, edge_type, note),
    )

TASK_CONTENT = """NIXL block-direct transport S3：跨机 CUDA↔ROCm register→transfer→poll 全生命周期 + 双 agent side-channel + prefill KV ring 接线。

拆三个 checkpoint：
- S3a：跨机 white(CUDA)↔pearl(ROCm) 双向 register→metadata→transfer→poll→telemetry 全生命周期探针，数据正确性 + telemetry bytes 对齐 K10 wire-byte 口径。探针用文件交换 NIXL metadata + 应用层 block desc（临时 channel，非 HCP 架构 side channel）。
- S3b：双 agent side-channel 接 coordinator 控制面（复用 WorkerCommand/WorkerResponse，不新增端口/依赖，coordinator 是唯一拓扑知识源）。
- S3c：prefill KV ring 走 block 路径（KV micro-block 经 NixlBlockTransport register/transfer），N=2 CUDA↔ROCm 数值对照 single-node reference，QUIC/TCP 字节流回退保留。

前置：S1（KvBlockTransport trait + SerializedBlockTransport fallback）+ S2（NixlBlockTransport 官方 nixl-sys crate，white/pearl 双机单机 register/metadata 冒烟绿）均已完成。S3 是形态 B（block-direct）数据面的首次跨机真实验证，之后 S4 才做 paged-KV 化（block_size=16 + block_table）。"""

DECISION_CONTENT = """【NIXL block-direct S3 跨机验证的动机六问（2026-08-16）】

1. 问题：NIXL block-direct 目前只到"能建 agent、能注册、能拉 local metadata"的单机冒烟（S2，white/pearl 双机）；prefill KV ring 仍走 QUIC/TCP 字节流（序列化 + 拷贝），NIXL 的 GPU-direct 零拷贝价值没接进 HCP 数据面。手段层"网络自由"的 NIXL 候选停在半程，三期 vLLM PD 对比无法用 HCP 自己的 NIXL 路径裁决。

2. 现状：KvBlockTransport trait（S1）+ NixlBlockTransport 官方 nixl-sys crate（S2，stub-api dlopen libnixl_capi.so）已实现完整 trait 面（register_block/deregister_block/local_metadata/load_remote_metadata/submit_transfer/poll_transfers/wire_bytes_*），white(conda 轮 nixl_cu13)与 pearl(源码树 --with-rocm UCX) 均通过单机 register→local_metadata 冒烟。但从未跨机 transfer；metadata 交换 gap（remote block 的 addr/len/dev_id 运行时才知道，须经 side channel 交换）未走通；ring_attention 未接 block 路径。

3. 终态：N=2 white(CUDA)↔pearl(ROCm) 跨机：双向 register→transfer→poll 全生命周期，dest block 字节级 == src block；telemetry.total_bytes 填 K10 wire-byte 口径；side channel 复用 coordinator 控制面（不新增端口）；prefill KV ring 走 block 路径，数值对照 single-node reference 过既有门；QUIC/TCP 字节流回退保留；Mac 默认 feature-off 保持绿。

4. 他者：vLLM PD 用 NixlConnector 做 prefill→decode 整段 KV 一次搬移（block 级、GPU-direct）+ 独立 TCP side channel（VLLM_NIXL_SIDE_CHANNEL_HOST/PORT）交换 NIXL agent metadata。HCP 复用其底层 register→transfer→notif 生命周期，但数据流不同构：HCP 是 ring 逐 hop + capacity-weighted 不均等分片（每节点只永久持有自己 chunk），不是整段搬移；side channel 也不照搬（HCP 已有 coordinator 控制面 + 全拓扑知识，coordinator 是唯一 metadata 交换权威）。

5. 本方案：S3a（跨机 transfer 探针，文件交换 metadata + block desc）→ S3b（side-channel 接 coordinator 控制面）→ S3c（prefill KV ring 接 block 路径 + N=2 数值对照），每步一个 commit checkpoint。feature nixl-backend 门控 + QUIC 回退，Mac 保持 feature-off 绿；真实编译 + smoke 只在 white/pearl（git pull + rebuild）。

6. 为什么：NixlConnector 是"prefill 集群→decode 集群整段搬移"的同构形态，表达不了 HCP 的 ring 逐 hop + capacity-weighted 分片；HCP 只复用 NIXL 的 block 传输语义（K4 结论），ring 编排 + 控制面 side-channel 是 HCP 特有的。S3 是把"NIXL 作为 P2P ring 传输候选"从单机冒烟推进到跨机数据面真实验证的最小节点，也是后续 paged-KV 化（S4）的前置。

VERDICT: IMPLEMENT（无阻塞；按 S3a→S3b→S3c 三 checkpoint 推进）。"""

OLD_DECISION_CONTENT = """【NIXL 接入 transport trait 的动机六问（2026-08-16）】

1. 问题：HCP 的 KvTransport trait 目前只有 QUIC（生产）+ TCP（测试）+ Mock（单测）三个实现。用户要求把 NIXL 接上 transport trait，作为 QUIC/TCP 之外的第三种通信选择——服务"网络自由=手段"卖点（P2P ring 不绑定厂商 collective 栈，从 2.5GbE 到 CXL 都能跑）。这是 K4（评估 NIXL 作为 HCP ring 传输轮子）从"评估"升级为"实现"。

2. 现状：NIXL 是 C++/CUDA/ROCm 库（ai-dynamo/nixl），语义是 block 级 GPU 内存传输（Agent → register_memory → transfer → notification），有官方 Rust 绑定（nixl-sys FFI crate + 高层 nixl crate，crates.io 1.4.0，bindgen 构建）。KvTransport 是"序列化 tensor → frame bytes → 流式 send/recv"抽象。两者语义不同构：NIXL 是 block 传输 + 通知，不是字节流。NIXL 只能在 white(CUDA)/pearl(ROCm) 构建运行（已探明：pearl 有 /home/stark/build/nixl-1.4.0 源码构建 + libnixl.so + src/api/cpp/nixl.h；white 有 conda 轮内 libnixl.so），Mac 无 UCX/CUDA/ROCm，无法本地构建或运行 NIXL。

3. 终态：cargo feature（如 nixl-backend，默认 off）门控的 NixlKvTransport 实现 KvTransport，作为第三种传输；默认 Mac 构建保持绿（feature off）；feature on 只在 white/pearl 编译 + smoke。NIXL 侧的 wire_bytes_sent/recv 复用 K10 刚建的计量接口（用 getXferTelemetry 或 transfer 字节数填同字段）。

4. 他者：vLLM PD 用 NixlConnector 做 prefill→decode 整段 KV 搬移（block 级、GPU-direct）。Dynamo 的 block_manager/storage/nixl.rs 是同一 FFI 的官方用法。NIXL 的既有价值全在 GPU-direct block 传输（避免序列化+拷贝）；若只是"用 NIXL 搬序列化后的 frame bytes"，等于把 UCX 当字节管道，丢掉了 NIXL 的零拷贝优势。

5. 本方案（两形态）：
   - 形态 A（frame-carrier，薄适配）：serialize frame → 注册 host/device buffer → NIXL transfer → 对端 deserialize。复用 K10 wire-byte 口径，trait 不变，是"第三种传输"的字面实现（网络自由），但不兑现 NIXL 零拷贝价值。Mac 不可构建，验证在 white/pearl。
   - 形态 B（block-direct，新抽象）：直接注册 K/V tensor device 内存做 block 传输，绕开序列化。兑现 NIXL 价值，但 KvTransport trait 是字节流语义，需要新的 block-transport trait 面或并行抽象，改动远大于"接上 transport trait"。

6. 为什么：用户指令是"接上 transport trait"——字面对应形态 A。形态 A 是网络自由卖点的正确落点（第三种可插拔传输），且 K10 刚建的 wire-byte 接口让 NIXL 与 QUIC/TCP 同口径可对比。形态 B 是更大的架构重构（block 级数据面 + 调度改造），应作为独立后续节点，不混入本节点。硬约束：本节点代码 Mac 上只能 cargo check 门控状态（feature off 绿），真实编译+smoke 必须在 white/pearl 经 git pull + rebuild。

VERDICT（2026-08-16 用户裁定）: 走形态 B（block-direct）。S1（KvBlockTransport trait + SerializedBlockTransport fallback，358c4c5）+ S2（NixlBlockTransport 官方 nixl-sys crate，367fe04 + white 1b275c4）已落地；S3 为跨机全生命周期验证，S4 为 paged-KV 化。形态 A（frame-carrier）不实施——它把 UCX 当字节管道，丢掉 NIXL 零拷贝价值，与"网络自由=手段"卖点的真正落点（block-direct GPU 传输）相悖。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    conn.execute("UPDATE nodes SET status='active' WHERE id=?", (TASK,)) if False else None

    # 1) 更新旧 decision verdict（形态 B 已裁定）
    upsert_node(conn, OLD_DECISION, "decision", "active",
        "NIXL 接入 transport trait 决策：已裁定形态 B（block-direct，独立数据面）",
        OLD_DECISION_CONTENT, "held", 1.0, 1.0, "hetero-cp-ringattn@nixl-transport-decision-20260816")

    # 2) 新 S3 task + decision
    upsert_node(conn, TASK, "task", "active",
        "NIXL block-direct S3：跨机 CUDA↔ROCm transfer 全生命周期 + side-channel + prefill KV ring",
        TASK_CONTENT, "active", 0.95, 1.0, "hetero-cp-ringattn@nixl-s3-decision-20260816")

    upsert_node(conn, DECISION, "decision", "active",
        "NIXL block-direct S3 立项：跨机验证动机六问（VERDICT: IMPLEMENT）",
        DECISION_CONTENT, "held", 1.0, 1.0, "hetero-cp-ringattn@nixl-s3-decision-20260816")

    edges = [
        (DECISION, TASK, "PART_OF", "S3 decision governs the S3 task"),
        (DECISION, OLD_DECISION, "BUILDS_ON", "S3 is the cross-machine verification of form-B chosen in the transport decision"),
        (DECISION, INQUIRY, "PART_OF", "S3 advances K4 (NIXL as HCP ring transport) from evaluation to cross-machine data-plane"),
        (DECISION, "evidence-nixl-sys-white-cuda-verified-20260816", "BASED_ON", "S3 builds on the white/pearl single-machine register/metadata smoke"),
        (DECISION, "decision-k10-kv-byte-ledger-20260816", "BUILDS_ON", "S3a telemetry.total_bytes fills the K10 wire-byte caliber"),
        (DECISION, "decision-hcp-first-principles-value-20260815", "FOLLOWS", "cross-machine NIXL serves network-freedom = P2P ring over any transport"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision-nixl-s3-cross-machine-20260816 + task-nixl-s3-cross-machine-transfer-20260816")
    print("old verdict updated: decision-nixl-as-transport-20260816 -> form-B chosen")

if __name__ == "__main__":
    main()
