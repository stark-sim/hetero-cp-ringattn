import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@1b275c4"
OLD = "lesson-component-reuse-boundary-20260816"
NEW = "lesson-feature-implementation-record-20260816"
WHITE_EVIDENCE = "evidence-nixl-sys-white-cuda-verified-20260816"

TITLE = "feature 实现探索后记录：最终实现 + 选中方案 + 被弃方案 + 原因"

CONTENT = """【教训】实现一个 feature 经过探索后，要把「最终实现 + 选中方案 + 被弃方案 + 各自原因」作为持久记录，不能只记「做了什么」。

触发场景：NIXL 第三传输接入。探索了多个候选（手写 FFI vs 官方 crate、stub-api vs 非 stub、conda 轮 vs 源码树），最终实现是官方 nixl-sys crate + stub-api；但最初只记了「最终实现」，没记「抛弃了什么、为什么」。追问「是复用官方 crate 吗」暴露选择原因不持久，且措辞把复用边界写错（说「capi.so 退役」而 stub-api 仍在运行期 dlopen libnixl_capi.so）。

根因：探索结论被当成临时工作记忆，而非持久项目知识。「what」记了，「why / 被弃方案」没记。

第一条件可防：feature 落地提交时，按 skill feature-implementation-record 写一段持久记录——最终实现（文件/commit）、选中方案、每个被弃方案 + 具体原因（正确性/成本/不匹配/沉没成本/可维护性/可移植性/性能）。

本案例的选择（结果 + 原因）已记入 evidence-nixl-sys-white-cuda-verified-20260816；通用记录模式沉淀为 skill feature-implementation-record（~/.agents/skills/feature-implementation-record）。

边界：实现前分析 → motivation-analysis；实验对比 → route-experimentation；环境事实 → infrastructure-inventory；本节点只记录「探索后的选择结论 + 原因」。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    # Rename the lesson node id: insert NEW (full content), repoint edges, drop OLD.
    conn.execute(
        "INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now')) ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,updated_at=datetime('now')",
        (NEW, "lesson", "progress", PROJECT, TITLE, CONTENT, 0.9, 0.9, "held", SOURCE),
    )
    conn.execute("UPDATE edges SET source = ? WHERE source = ?", (NEW, OLD))
    conn.execute("UPDATE edges SET target = ? WHERE target = ?", (NEW, OLD))
    conn.execute("DELETE FROM nodes WHERE id = ?", (OLD,))

    # Correct the edge note's skill reference.
    conn.execute(
        "UPDATE edges SET note = ? WHERE source = ? AND target = ?",
        ("lesson extracted from the NIXL feature-exploration outcome; skill feature-implementation-record owns the general pattern", NEW, WHITE_EVIDENCE),
    )

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print(NEW)


if __name__ == "__main__":
    main()
