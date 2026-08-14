import sqlite3
import subprocess
import sys
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@wired-n2-baseline-20260814"
TASK = "task-phase3-wired-n2-baseline-20260814"
EVIDENCE = "evidence-phase3-wired-n2-baseline-20260814"
WIRED = "evidence-white-pearl-wired-2p5g-20260814"
OLD_WIFI = "evidence-phase3-8-wifi-link-ceiling-20260814"
LESSON = "lesson-benchmark-network-parser-unit-metadata-20260814"

EVIDENCE_CONTENT = """Run reports/routeb-p3-baseline-20260814-134121/ on main e07be07; white and pearl both synced to the same commit. N=2 topology: coordinator+worker0+vllm bench client on white RTX4090 CUDA, worker1 on pearl RX9060XT HIP; ring data path explicitly 192.168.100.1(enp10s0)↔192.168.100.2(enp8s0), both 2500Mb/s full duplex. network.json: RTT min/avg/max=0.108/0.172/0.241ms, iperf sender=2360Mbps receiver=2350Mbps, retransmits=0. REPS=10; every rep first-attempt PASS, each with 32/32 requests, trace ids/hops exact, reserved==released, metrics failed=0.
Aggregate median(min,max,spread):
L1 TTFT 334.44ms(324.03,342.06,5.4%), TPOT 69.94ms(68.98,72.48,5.0%), ITL 73.88ms(72.84,76.62,5.1%), output 15.18tok/s(15.10,15.22,0.8%).
L2 TTFT 149.92ms(145.86,158.77,8.6%), TPOT 55.79ms(54.93,56.20,2.3%), ITL 51.65ms(50.67,51.82,2.2%), output 31.68tok/s(31.31,32.31,3.2%).
L3 TTFT 279.81ms(275.51,297.45,7.8%), TPOT 112.18ms(111.03,117.25,5.5%), ITL 106.49ms(104.84,110.68,5.5%), output 31.02tok/s(29.94,31.21,4.1%).
Relative to superseded WiFi medians, wired is 26-66x lower TTFT, 20-31x lower TPOT and 10-23x higher output throughput; this is network-environment evidence, not algorithmic speedup. Current comparison baseline requires equivalent 2.5GbE network.json and at least 10 reps. Controller script sha256 and uncommitted diff are stored in the report."""

LESSON_CONTENT = """Incident: first wired baseline attempt produced network.json goodput=None even though raw iperf showed 2.35 Gbits/sec. Root cause: parser only matched Mbits/sec, a hidden assumption that held on the old 44 Mbit/s WiFi link but failed once the link crossed 1 Gbit/s. A second metadata check found aggregate topology strings still hard-coded to old 192.168.8.x even though the actual ring used 192.168.100.x. Resolution: normalize K/M/Gbits/sec to Mbps, preserve retransmits, parameterize data IPs, derive topology metadata from the same variables, save controller script hash+diff, abort/restart before collecting the ten-rep record. How to apply: benchmark environment metadata must be generated from the same runtime parameters as the workload and parsers must handle unit scaling; validate network.json before entering the repetition loop or accepting a baseline."""


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
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "white-pearl 2.5GbE N=2 10-rep vllm bench 基线全绿",
        EVIDENCE_CONTENT,
        "verified",
    )
    upsert_node(
        conn,
        LESSON,
        "lesson",
        "benchmark 网络解析必须支持单位缩放且元数据同源",
        LESSON_CONTENT,
        "held",
    )
    conn.execute(
        "UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?",
        (TASK,),
    )
    for edge in (
        (EVIDENCE, TASK, "CONFIRMS", "ten-repetition baseline completed"),
        (EVIDENCE, WIRED, "BASED_ON", "uses the verified 2.5GbE link"),
        (EVIDENCE, OLD_WIFI, "SUPERSEDES", "replaces the provisional WiFi table for comparisons"),
        (LESSON, EVIDENCE, "BASED_ON", "discovered and fixed before the accepted run"),
    ):
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("wired_baseline=verified")
    print("reps=10_all_first_attempt_pass")
    print("network=2350Mbps_0_retransmits")


if __name__ == "__main__":
    main()
