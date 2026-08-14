import sqlite3
import subprocess
import sys
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "phase2-audit-phase3-plan-2026-08-14"

AUDIT_TASK = "task-phase2-audit-phase3-white-offline-plan-20260814"
EVIDENCE = "evidence-phase2-audit-phase3-plan-20260814"
PHASE3 = "task-phase3-vllm-bench-and-ecosystem-20260812"
RECONNECT_EXISTING = "task-phase3-ring-reconnect-resilience-20260813"
WHITE_RECOVERY = "task-white-single-network-mode-20260814"

TASKS = {
    "task-phase3-9-evidence-harness-hygiene-20260814": (
        "三期 9：二期证据固化与 benchmark 驱动 endpoint 卫生",
        "目标：补齐二期 6c.1/6d 原始运行证据未进入 main 的长期复现缺口，并消除 phase3 驱动中的历史 endpoint 默认值。输入：边界文档、旧 route-B worktree 中 6d 报告、inventory authority 规则、现有 7a/8 驱动。产出：受版本控制的紧凑 evidence manifest（commit/topology/命令/关键断言/原始 artifact hash，不提交大日志）以及显式 endpoint 注入的驱动。验收：证据可从 commit 与 hash 追溯；所有远程运行参数来自当次 inventory 解析或显式环境变量；bash -n 与既有本地解析测试通过。失败判据：任一二期结论无法绑定 commit/拓扑/断言，或驱动仍静默回退到历史 IP。commit 边界：docs/evidence 与 harness 配置各自独立。white 不参与。",
    ),
    "task-phase3-10-transport-failure-isolation-20260814": (
        "三期 10：连接死亡请求级隔离、epoch 重连与 ring 重建",
        "目标：网络抖动不再通过 worker runtime expect/panic 清空整项服务。输入：7a 五次连接死亡证据、现有优雅退出日志、coordinator/worker/ring transport。产出分三小节点：(a) stream close 转结构化请求失败，active request 与 ledger 恰好清理一次且进程存活；(b) coordinator 以 domain_id+epoch 接受 worker 重连并拒绝 stale connection；(c) predecessor/successor peer 重拨并重建 ring。验收：注入 FIN/connection lost 后当前请求失败一次、无 panic/双重 release，health 进入 degraded；同 domain 新 epoch 重连后下一请求通过，旧 packet 被拒绝；最后在真实 N=2/N=3 做断链恢复门禁。失败判据：进程退出、旧 epoch 数据被接受、session/KV 泄漏或重复释放。commit 边界：failure isolation、coordinator reconnect、ring rebuild、真实 E2E 分开。前两项本地立即可做；真实 E2E 等在线节点。",
    ),
    "task-phase3-11-placement-ledger-integration-20260814": (
        "三期 11：placement/ledger WIP 分阶段接入",
        "目标：把当前未跟踪 placement.rs 中的 capacity-bounded frozen plan 作为三期素材审计后接入，不直接覆盖已验证调度。产出分三小节点：(a) 独立模块与属性/边界测试；(b) admission dry-run，仅 trace placement hash、per-worker reserved bytes 与 capacity-only fallback，生产行为不变；(c) owner 确认后再激活 prompt split/KV calendar。验收：同输入跨节点得到同 placement_hash；量化后各 worker 不越 byte bound；缺失任一 rate 时全体 capacity-only；ledger reserve/release exactly once；激活前后 dense/golden correctness 不变。附加门禁：Tch Tensor::zeros 的不可恢复 OOM 风险需以保守 allocatable budget 或隔离故障策略明确处理。失败判据：计划非确定、量化后越界、混用 guessed/measured rates、改变既有 golden。commit 边界：module tests、dry-run trace、behavior activation 分开。white 不参与前两项。",
    ),
    "task-phase3-12-session-lifecycle-continuation-queue-20260814": (
        "三期 12：session 生命周期与 barrier-scheduled continuation",
        "目标：补齐三期 8 明确排除的 resident KV 生命周期和并发 continuation 编排。采用最小方案：append 作为独占 barrier 排队，在既有 active decode 批次排空后执行；不在同一 layer 内混跑 continuation 与普通 decode。产出：显式 session release/TTL、过期清理、append queue/barrier 状态机及 trace。验收：客户端不再必须靠最终 append 才释放 KV；release/expiry/failure 恰好一次释放；多个普通请求存在时 append 可排队而非立即拒绝，排空后 golden PASS；未知/过期 session fail-closed。失败判据：resident KV 无界持有、barrier 饥饿、普通请求与 continuation 顺序不确定或 ledger 泄漏。commit 边界：lifecycle、barrier scheduler、本地 golden、真实 E2E 分开。依赖 transport failure isolation 与 placement dry-run。",
    ),
    "task-phase3-13-distributed-baseline-gate-20260814": (
        "三期 13：受控 vLLM 分布式 PP 基线可行性与公平测量",
        "目标：落实 owner 已修正的 baseline 口径，不用单机 vLLM 对比跨节点 HCP。先做可行性门禁：确认当前 vLLM 版本是否支持同一 PP 作业混合 CUDA/ROCm；若不支持，必须由 owner 选择同硬件同拓扑基线或拆分为机制基线与异构端到端绝对指标，禁止直接 speedup 宣称。产出：baseline ADR、统一 workload/网络元数据 schema、交错 A/B 运行 harness。验收：两侧均为分布式、模型/输入输出/采样/正确性门禁一致；network.json 等价；运行顺序交错以消除时间漂移；median 与置信带/既定噪声带共同报告。失败判据：跨不同 media/goodput 比较、单机对跨节点、不同硬件却声称算法 speedup。脚本与 ADR 可立即做；有效数值等 white 有线恢复或获得等价网络后重建。",
    ),
    "task-phase3-14-ecosystem-interface-decision-20260814": (
        "三期 14：vLLM backend/plugin/paged-KV 生态接口决策",
        "目标：在现有 OpenAI-compatible endpoint 已被 vllm bench 驱动的基础上，判断进一步嵌入 vLLM 的具体收益，避免为集成而集成。产出：比较外部独立 server、custom backend、KV connector/plugin 三种边界的 ADR；逐项列出可复用的 scheduler/block table/benchmark 能力与会破坏 HCP neighbor-only、跨 CUDA/ROCm、stationary KV 合同的部分。验收：选择能解决已验证缺口的最小接口；不反向改写 HCP core 数学；若无净收益则明确 defer。依赖 reconnect、placement dry-run 与 baseline feasibility 结论。white 不参与决策。",
    ),
    "task-phase3-15-white-recovery-hardware-gates-20260814": (
        "三期 15：white 恢复后的真实异构与有线性能门禁",
        "目标：white 物理恢复后只执行必须依赖其 RTX 4090/直连 2.5GbE 的硬件门禁。前置：单 networkd 模式、Wi-Fi/Tailscale、192.168.100.1/24 直连和 inventory 更新全部验证。产出：N=2 white+pearl 与 N=3 white+pearl+laptop correctness 回归；断链重连 E2E；有线 iperf/RTT network.json；废弃 WiFi provisional 表并重建性能基线；随后才运行公平 PP 对照。验收：所有节点同 commit，golden/trace/ledger 通过，network.json 明确为有线且可重复；旧 WiFi 数值不参与新比较。失败判据：恢复门禁未全绿就运行 benchmark，或把不同链路结果合并。commit 边界：恢复证据、correctness、reconnect E2E、wired baseline、PP comparison 分开。",
    ),
}

EVIDENCE_CONTENT = """2026-08-14 对二期与当前三期状态完成定点审计：
1. 二期 benchmark-readiness 的五项工程出口已真实落地并合 main：HTTP completions/SSE、prefill byte admission、active-request ledger、DecodeBatch FIFO、trace、6c.1 native baseline、6d N=3 服务；边界文档明确全部是 correctness+服务稳定性，不含性能结论。三期无需重做这些能力。
2. 二期主要不足分两类。证据保管缺口：6c.1/6d 的原始报告未被 main 跟踪，6d artifact 仍只存在旧 route-B worktree，长期复现依赖本机目录。工程边界缺口：连接死亡仍可在 worker runtime expect 路径升级为进程退出；这不推翻二期正确性，但使“可长跑服务”的口径不成立。另有 placement/ledger WIP 未集成、Tch reserved allocation OOM 不可恢复风险。
3. 三期实际已开始而非空白：7a 已用 vllm bench serve 驱动 N=2/N=3 三档 0 失败；8 已完成 HTTP keep_kv/append continuation 的 N=2 golden E2E。尚未完成的是受控分布式 vLLM baseline、生态接口评估、连接重连/重建、并发 continuation/session 生命周期和 placement 激活。
4. 现有 N=2 5-rep 性能表只能作为 provisional WiFi 环境记录。双端 WiFi 单流 TCP 约 44 Mbit/s 且重传严重，network.json 不等价时不可比较；white 恢复为 2.5GbE 直连后必须废弃旧带宽带并重建。
5. 当前驱动仍带历史 endpoint 默认值，和 inventory-authoritative 规则存在张力；在下一次远程运行前应先改成 inventory/显式参数注入。
6. 推荐执行顺序：9 证据与 harness 卫生 → 10 transport failure isolation/reconnect → 11 placement dry-run/激活决策 → 12 session 生命周期与 barrier-scheduled continuation → 13 分布式 PP baseline 可行性/公平方法 → 14 生态接口 ADR。15 是 white 恢复后的硬件门禁，不阻塞 9-14 的本地设计与前置实现。
结论：二期可以保持 completed，不应回滚或重开；其不足作为三期工程可信度与复现性节点补齐。推荐下一行为节点是三期 10，但先用三期 9 做一个很小的证据/驱动卫生 checkpoint。"""


def upsert_node(conn, node_id, node_type, title, content, status, importance=1.0):
    conn.execute(
        """
        INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,"active",?,?,?,?,1.0,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,
          updated_at=datetime('now')
        """,
        (node_id, node_type, PROJECT, title, content, importance, status, SOURCE),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source, target, edge_type, note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "二期缺口审计与 white 离线期间三期任务图",
        EVIDENCE_CONTENT,
        "verified",
    )
    conn.execute(
        "UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?",
        (AUDIT_TASK,),
    )
    for task_id, (title, content) in TASKS.items():
        upsert_node(conn, task_id, "task", title, content, "planned", 0.9)
        upsert_edge(conn, task_id, PHASE3, "PART_OF", "concrete phase-3 task node")
        upsert_edge(conn, task_id, EVIDENCE, "BASED_ON", "created from the 2026-08-14 audit")
    upsert_edge(conn, EVIDENCE, AUDIT_TASK, "CONFIRMS", "audit and plan task completed")
    upsert_edge(
        conn,
        "task-phase3-10-transport-failure-isolation-20260814",
        RECONNECT_EXISTING,
        "REFINES",
        "splits the existing reconnect task into independently verifiable checkpoints",
    )
    upsert_edge(
        conn,
        "task-phase3-15-white-recovery-hardware-gates-20260814",
        WHITE_RECOVERY,
        "DEPENDS_ON",
        "hardware gates begin only after white network recovery is verified",
    )
    upsert_edge(
        conn,
        "task-phase3-11-placement-ledger-integration-20260814",
        "task-phase3-10-transport-failure-isolation-20260814",
        "DEPENDS_ON",
        "placement activation should not precede failure isolation",
    )
    upsert_edge(
        conn,
        "task-phase3-12-session-lifecycle-continuation-queue-20260814",
        "task-phase3-11-placement-ledger-integration-20260814",
        "DEPENDS_ON",
        "multi-session lifecycle consumes the placement and ledger contract",
    )
    upsert_edge(
        conn,
        "task-phase3-13-distributed-baseline-gate-20260814",
        "task-phase3-10-transport-failure-isolation-20260814",
        "DEPENDS_ON",
        "long benchmark comparison requires transport resilience first",
    )
    upsert_edge(
        conn,
        "task-phase3-14-ecosystem-interface-decision-20260814",
        "task-phase3-13-distributed-baseline-gate-20260814",
        "DEPENDS_ON",
        "ecosystem choice follows baseline feasibility and measured gaps",
    )
    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("phase2_audit=verified")
    print("phase3_task_nodes=7")
    print("recommended_next=phase3-9-then-phase3-10")


if __name__ == "__main__":
    main()
