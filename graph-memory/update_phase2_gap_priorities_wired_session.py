import sqlite3
import subprocess
import sys
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-direction-2026-08-14-phase2-gaps"

OLD_DECISION = "decision-phase3-plan-around-white-offline-20260814"
PRIORITY = "decision-phase2-gap-priorities-20260814"
RECONNECT = "decision-bounded-transient-reconnect-20260814"
SESSION = "decision-session-kv-evictable-ownership-20260814"
WIRED = "evidence-white-pearl-wired-2p5g-20260814"
BASELINE = "task-phase3-wired-n2-baseline-20260814"
ADDRESSING = "decision-neighbor-only-addressing-deferred-20260814"
METHOD = "preference-motivation-analysis-20260721"
PHASE3 = "task-phase3-vllm-bench-and-ecosystem-20260812"


NODES = {
    PRIORITY: (
        "decision",
        "二期缺口按 bounded reconnect、session ownership、wired baseline 重排",
        """动机剖析六问：
1. 问题：上一轮把连接恢复、session KV、性能环境和地址默认值都列为三期缺口，但优先级与目标部署假设不够准确。
2. 现状：HCP 节点是专供设备，正常情况下应持续在线；同时支持 LAN/VPN，短时网络抖动客观存在。white 已恢复且与 pearl 建立 2.5GbE 直连。keep_kv session 当前强持有分布式 KV；地址默认值仍能服务当前 N=2/N=3 harness。
3. 终态：连接只在严格预算内恢复短时波动，超预算立即失败而不等待离线节点；session KV 采用可回收 ownership；性能在 2.5GbE 上重新建多轮基线；地址发现留到 N 增长时与 neighbor-only 部分可见拓扑一起设计。
4. 他者：vLLM 普通请求完成后释放 request block ownership；prefix caching 使完成块以 ref_cnt=0 进入 LRU free queue，可在内存压力下立即覆盖，而不是被 application session 永久 pin；运行中 ref_cnt>0 的块受保护，并提供 reset_prefix_cache/sleep 等运维清理。
5. 本方案：bounded retry + fail-fast topology unavailable；HCP idle session 进入分布式 LRU eviction 候选，append 执行期间 pin，显式 release/admin reset，压力驱逐后 append 返回 session miss；有线 baseline 记录 network.json 并跑 10 reps。
6. 为什么：长时间等待离线节点违背专供设备假设并拖垮请求尾延迟；固定 TTL 不是 vLLM 的核心机制，pressure-evictable ownership 更贴近真实内存管理；当前地址抽象尚未面对 N 节点部分可见图，过早泛化收益不足。
VERDICT: REPRIORITIZE。先完成 wired baseline 与 session ownership 设计，reconnect 仅实现短时 bounded recovery。""",
        "held",
    ),
    RECONNECT: (
        "decision",
        "连接恢复只覆盖短时 LAN/VPN 抖动，超预算 fail-fast",
        """部署假设：所有 worker 是专供且理应持续在线，不为长时间离线节点保留请求或无限等待。恢复合同：连接/stream 异常后在可配置的 attempt+wall-clock 双预算内重拨；退避必须有上限；预算耗尽后当前请求失败、KV/ledger 恰好释放一次、拓扑标记 unavailable，新请求 fail-closed，worker 后续重新注册可恢复服务。固定 N ring 不在请求中途缩容。默认次数与总时长不在本决策中拍死，由本地 fault injection 和 2.5GbE/VPN 抖动数据确定；目标是秒级而非分钟级。""",
        "held",
    ),
    SESSION: (
        "decision",
        "HCP idle session KV 改为可驱逐 ownership，不永久 pin",
        """vLLM 对照结论：请求完成时 scheduler free request ownership；启用 automatic prefix caching 时，完整缓存块保留 block hash，但 ref_cnt 降为 0 并进入 LRU free queue，既可命中复用，也可在下一次分配时被立即驱逐；活跃请求 touch 后 ref_cnt>0 并移出 free queue。该机制不依赖 TTL；另有 reset_prefix_cache 和 sleep(level>=1) 清空 KV。
HCP 采用对应边界而非照搬中心 block pool：keep_kv 完成后的 session 标记 idle+evictable，进入 coordinator 维护的分布式 LRU；append/continued decode 期间 pin，不可驱逐；admission 空间不足时按 LRU 原子驱逐 idle session，并向所有 domains 发送 ReleaseRequest，ledger exactly-once release；被驱逐 session 的 append 返回明确 session miss/expired，客户端可重新 prefill。补充显式 release 与 admin reset。TTL 仅作为可选部署上限，不作为主要正确性机制；不做跨 session prefix sharing。""",
        "held",
    ),
    WIRED: (
        "evidence",
        "white-pearl 2.5GbE 直连门禁通过：2.35 Gbit/s、0 retransmit",
        """2026-08-14 inventory 与实机验证：white enp10s0=192.168.100.1/24，pearl enp8s0=192.168.100.2/24；两端 ethtool 均为 2500Mb/s Full、link detected yes，路由明确走直连接口。双向 10-packet RTT：white→pearl avg 0.167ms，pearl→white avg 0.116ms，0% loss。iperf3 单流 5x10s receiver 全部 2.35 Gbit/s，sender 2.35-2.36 Gbit/s，全部 0 retransmit；4 streams receiver 2.35 Gbit/s、0 retransmit。相对旧 WiFi 单流约 44 Mbit/s 提升约 53x，链路波动显著收窄，满足重建性能基线的网络门禁。""",
        "verified",
    ),
    BASELINE: (
        "task",
        "在 white-pearl 2.5GbE 上重建 10-rep N=2 vllm bench 基线",
        """使用 phase3_8_perf_baseline_n2.sh，数据面固定 192.168.100.1↔192.168.100.2，REPS=10；每 rep 新建 coordinator/workers，L1/L2/L3 correctness gate 全过才进入统计。network.json 必须记录 2.5GbE 接口、RTT、iperf goodput/retransmits；控制脚本保存 sha256 与未提交 diff。旧 WiFi 表只保留历史环境证据，不与本轮数值合并。当前运行中。""",
        "active",
    ),
    ADDRESSING: (
        "decision",
        "当前地址默认值不列为二期缺口，N 增长时统一设计 neighbor-only 寻址",
        """当前 N=2/N=3 harness 的地址默认值可接受，不为通用化单独立项。未来 N 增长时，ring 上相邻节点可达但非相邻节点可能互相不可见，不能假设 coordinator 能把全体 endpoint 当作全互联地址表。届时把节点身份、控制面注册地址、每节点 predecessor/successor 可达 endpoint、NAT/VPN/LAN 多地址选择和 ring epoch 一起设计；当前只允许运行时显式覆盖数据面地址以选择 2.5GbE，不改变拓扑模型。""",
        "deferred",
    ),
}


def upsert_node(conn, node_id, node_type, title, content, status):
    conn.execute(
        """
        INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,'active',?,?,?,?,1.0,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,
          updated_at=datetime('now')
        """,
        (node_id, node_type, PROJECT, title, content, 1.0, status, SOURCE),
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
    for node_id, (node_type, title, content, status) in NODES.items():
        upsert_node(conn, node_id, node_type, title, content, status)
    conn.execute(
        "UPDATE nodes SET status='superseded', updated_at=datetime('now') WHERE id=?",
        (OLD_DECISION,),
    )
    for edge in (
        (METHOD, PRIORITY, "GOVERNS", "six-question reprioritization"),
        (RECONNECT, PRIORITY, "PART_OF", "bounded transient recovery scope"),
        (SESSION, PRIORITY, "PART_OF", "evictable session ownership scope"),
        (WIRED, PRIORITY, "SUPPORTS", "wired link removes the previous WiFi blocker"),
        (BASELINE, WIRED, "DEPENDS_ON", "benchmark starts only after wired network gate"),
        (BASELINE, PHASE3, "PART_OF", "rebuilds the phase-3 N=2 performance baseline"),
        (ADDRESSING, PRIORITY, "PART_OF", "explicitly deferred current non-gap"),
        (PRIORITY, OLD_DECISION, "SUPERSEDES", "white recovered and owner refined gap priorities"),
    ):
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("phase2_gap_priorities=updated")
    print("wired_2p5g=verified")
    print("session_kv=vllm-inspired-evictable")


if __name__ == "__main__":
    main()
