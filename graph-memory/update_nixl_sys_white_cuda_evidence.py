import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@1b275c4"
EVIDENCE = "evidence-nixl-sys-white-cuda-verified-20260816"
DECISION = "decision-nixl-as-transport-20260816"
TASK = "inquiry-nixl-as-hcp-transport-20260815"
PEARL_EVIDENCE = "evidence-nixl-sys-official-crate-verified-20260816"


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


EVIDENCE_CONTENT = """NIXL block transport 用官方 nixl-sys crate 在 white(CUDA/RTX4090) 真实验证通过（commit 1b275c4，pearl 之后第二台）。

用户裁决（2026-08-16）：white 也应像 pearl 一样用 nixl-sys 官方轮子直接跑通；能通就不必再管手写 capi.so FFI（避免沉没成本）。white 与 pearl 一样已解决 clang 问题，可用 nixl-sys 官方轮子继续推进。

环境：
- white = RTX 4090（24 GiB，driver 610.43.02），libtorch CUDA 13（libtorch_cuda.so + libcudart-*.so.13）。
- clang 21.1.8 + libclang-21-dev（/usr/lib/llvm-21/lib/libclang.so），bindgen 可识别。
- NIXL 来自 vllm-v1 conda 环境的 pip 轮 nixl_cu13（不是 pearl 的源码构建树）：libnixl_capi.so 在 .nixl_cu13.mesonpy.libs/，UCX 插件 libplugin_UCX.so 在 .nixl_cu13.mesonpy.libs/plugins/。轮内 RPATH 自包含（libnixl/libcore/libserdes/libucp/libuct/libucs/libucm 均轮内解析，ldd 无 not-found）。

同步与构建：
- white 仓库原落后 origin/main 19 个 commit（b40b351）；两处 untracked 脚本（decode_route_compare_n4_driver.sh / n4p_driver.sh）与 origin/main 逐字节相同（sha256 一致），rm 后 git merge --ff-only 到 1b275c4。
- cargo build --manifest-path rust/Cargo.toml --features tch-backend,nixl-backend --bin nixl-probe 真实编译通过（nixl-sys v1.4.0 bindgen + stub-api，26.1s，仅既有 unused-import warnings）。

运行（register→metadata smoke）：
- LD_LIBRARY_PATH=…/.nixl_cu13.mesonpy.libs:…/libtorch/lib + NIXL_PLUGIN_DIR=…/.nixl_cu13.mesonpy.libs/plugins，无 LD_PRELOAD（CUDA libtorch 直链，不需要 pearl 的 libtorch_hip.so）。
- 输出：agent created: hcp-probe-agent / registered block id=0 len=96 addr=1099985584128 / local metadata bytes=811 / OK，exit 0。addr 为 VRAM 地址（tensor 在 CUDA device），UCX backend 实例化 + VRAM register 成功。

scripts/nixl_transport_probe.sh 泛化：按 hostname 自动切 white/pearl（white=conda 轮 cu13，pearl=源码树 + libtorch_hip.so preload），并修复原脚本 cd 到仓库根却 cargo build 无 Cargo.toml 的 bug（改用 --manifest-path rust/Cargo.toml）。

证据边界：单机 white(CUDA) 的 register + local_metadata 运行时验证（官方 nixl-sys crate 版）。至此 white(CUDA)+pearl(ROCm) 两台均用官方 nixl-sys crate 通过 S2 单机 register/metadata 冒烟；手写 capi.so FFI 彻底退役。不覆盖跨机 CUDA↔ROCm register→transfer→poll 全生命周期、双 agent side-channel 交换、prefill KV ring 接线（S3）或 paged-KV 化（S4）。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "progress",
        "NIXL 官方 nixl-sys crate 在 white(CUDA) 真实验证：conda 轮 + bindgen 编译 + register/metadata 运行通过",
        EVIDENCE_CONTENT, "verified", 0.95, 0.95, SOURCE)

    edges = [
        (EVIDENCE, DECISION, "CONFIRMS", "official nixl-sys verified on BOTH platforms (white CUDA conda wheel + pearl ROCm source tree); hand-written capi.so FFI fully retired"),
        (EVIDENCE, TASK, "SUPPORTS", "S2 verified on white CUDA; S3 cross-node CUDA->ROCm register->transfer->poll remains"),
        (EVIDENCE, PEARL_EVIDENCE, "CONFIRMS", "white CUDA conda-wheel path confirms pearl ROCm source-tree path; both use official nixl-sys"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print(EVIDENCE)


if __name__ == "__main__":
    main()
