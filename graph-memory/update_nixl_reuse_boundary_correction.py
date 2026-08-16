import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@1b275c4"
WHITE_EVIDENCE = "evidence-nixl-sys-white-cuda-verified-20260816"
PEARL_EVIDENCE = "evidence-nixl-sys-official-crate-verified-20260816"
DECISION = "decision-nixl-as-transport-20260816"
TASK = "inquiry-nixl-as-hcp-transport-20260815"
LESSON = "lesson-component-reuse-boundary-20260816"


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


WHITE_CONTENT = """NIXL block transport 用官方 nixl-sys crate 在 white(CUDA/RTX4090) 真实验证通过（commit 1b275c4，pearl 之后第二台）。

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

组件选择（结果 + 原因）：
1. nixl-sys 官方 crate vs 手写 extern "C" → 官方 crate。原因：bindgen 生成绑定 + 安全封装（Agent/Backend/XferRequest/Drop/错误）由上游维护，删 ~500 行手写 FFI，类型/生命周期正确。
2. nixl-sys 的 stub-api vs 非 stub（真实 link）→ stub-api。原因：构建期免链 NIXL C++ 库（libnixl.so/libnixl_build.so/libnixl_common.so），且同一构建产物可 dlopen 不同布局的 libnixl_capi.so（pearl 源码树 / white conda 轮）。代价：运行期实现仍是 libnixl_capi.so，复用仅到绑定层。
3. 运行期绑定：dlopen（stub-api）vs 构建期 link（旧手写 #[link]）→ dlopen。原因：是 #2 的直接后果；代价：LD_LIBRARY_PATH 必须包含 libnixl_capi.so 所在目录，否则首次调用 dlopen 失败。
4. white 的 libnixl_capi.so 来源：conda 轮 nixl_cu13 vs 源码构建树 → conda 轮。原因：vllm-v1 环境已装、RPATH 自包含（ldd 无 not-found）、匹配 CUDA 13 的 libtorch。
5. pearl 的 libnixl_capi.so 来源：源码构建树 vs conda 轮 → 源码树。原因：pearl 是 ROCm，cu13/cu12 轮的 UCX 是 CUDA 构建、无法注册 HIP 指针（先前已证 NIXL_ERR_BACKEND），需 --with-rocm 源码构建 UCX。

复用边界澄清：复用的是官方 nixl-sys crate 的绑定层（bindgen）+ 安全封装层；NIXL 运行期实现仍由 libnixl_capi.so 提供——stub-api 在运行期 dlopen("libnixl_capi.so") 转发 nixl_capi_* 符号。退役的是「手写 extern "C" 声明 + 编译期 link」，不是 libnixl_capi.so 这个库。

证据边界：单机 white(CUDA) 的 register + local_metadata 运行时验证（官方 nixl-sys crate 版）。至此 white(CUDA)+pearl(ROCm) 两台均用官方 nixl-sys crate 通过 S2 单机 register/metadata 冒烟。不覆盖跨机 CUDA↔ROCm register→transfer→poll 全生命周期、双 agent side-channel 交换、prefill KV ring 接线（S3）或 paged-KV 化（S4）。"""


LESSON_CONTENT = """【教训】复用官方 crate/轮子要按层说清楚：绑定/声明层 vs 安全封装层 vs 运行期实现层，不能一句话混为一谈。

触发场景：NIXL 接入用官方 nixl-sys crate + stub-api feature。我先后写了「复用官方 crate」和「capi.so 彻底退役」，但 stub-api 的 stubs.cpp 在运行期 dlopen("libnixl_capi.so") 转发 nixl_capi_* 符号——libnixl_capi.so 根本没退役，只是从「手写 extern "C" + 编译期 #[link]」改成「官方 crate 的 stub 运行期 dlopen」。

根因：把「复用」当成单一 yes/no，而不是按层拆分。复用的是绑定层（bindgen）+ 安全封装层（Agent/Backend/XferRequest/Drop/错误）；运行期实现仍是同一个 libnixl_capi.so。

第一条件可防：在写「复用/退役」结论前，读 crate 的 build.rs / feature flag / stub 源码，确定符号是编译期 link 还是运行期 dlopen；每个组件选择写一行「结果 + 原因」。

组件选择（结果 + 原因）已记入 evidence-nixl-sys-white-cuda-verified-20260816；通用判断模式沉淀为 skill component-reuse-boundary（~/.agents/skills/component-reuse-boundary）。

边界：环境事实（哪个 .so 在哪）→ infrastructure-inventory；dlopen/link 调试 → systematic-debugging；本节点只记录判断模式与选择原因。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    # 1) Re-upsert white evidence with corrected wording + component-choice log.
    upsert_node(conn, WHITE_EVIDENCE, "evidence", "progress",
        "NIXL 官方 nixl-sys crate 在 white(CUDA) 真实验证：conda 轮 + bindgen 编译 + register/metadata 运行通过",
        WHITE_CONTENT, "verified", 0.95, 0.95, SOURCE)

    # 2) Correct the decision-edge note (was "hand-written capi.so FFI fully retired").
    upsert_edge(conn, WHITE_EVIDENCE, DECISION, "CONFIRMS",
        "official nixl-sys binding + safe-wrapper layer reused on BOTH platforms; hand-written extern C declarations retired, but libnixl_capi.so still provides the runtime implementation via stub-api dlopen")
    upsert_edge(conn, WHITE_EVIDENCE, TASK, "SUPPORTS",
        "S2 verified on white CUDA; S3 cross-node CUDA->ROCm register->transfer->poll remains")
    upsert_edge(conn, WHITE_EVIDENCE, PEARL_EVIDENCE, "CONFIRMS",
        "white CUDA conda-wheel path confirms pearl ROCm source-tree path; both use official nixl-sys")

    # 3) Fix the now-stale boundary sentence in the pearl evidence node.
    old = "white(CUDA) 端未用 nixl-sys 版跑（其 libnixl_capi.so 在 conda 轮内路径不同，S3 时统一）。"
    new = "white(CUDA) 端随后也已用 nixl-sys 版验证通过（conda 轮 nixl_cu13，见 evidence-nixl-sys-white-cuda-verified-20260816）。"
    row = conn.execute("SELECT content FROM nodes WHERE id = ?", (PEARL_EVIDENCE,)).fetchone()
    if row and old in row[0]:
        conn.execute("UPDATE nodes SET content = ?, updated_at = datetime('now') WHERE id = ?",
                     (row[0].replace(old, new), PEARL_EVIDENCE))

    # 4) Insert the general lesson node.
    upsert_node(conn, LESSON, "lesson", "progress",
        "组件复用边界：绑定层 vs 运行期实现层——复用官方 crate 不等于复用其实现",
        LESSON_CONTENT, "held", 0.9, 0.9, SOURCE)
    upsert_edge(conn, LESSON, WHITE_EVIDENCE, "SUPPORTS",
        "lesson extracted from the stub-api '复用/退役' wording correction; skill component-reuse-boundary owns the general pattern")

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print(LESSON)


if __name__ == "__main__":
    main()
