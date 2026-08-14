import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'hetero-cp-ringattn@pd-spike-20260814'
TASK = 'task-phase3-vllm-pd-baseline-20260814'
EV = 'evidence-phase3-pd-spike-green-20260814'

CONTENT = (
'[2026-08-14] step0 尖峰通过：vLLM PD 分离跨 CUDA+ROCm 异构 E2E 全链路绿。'
'拓扑：white(RTX4090,CUDA,prefill,NixlConnector kv_both,192.168.100.1:8100) -> KV 经 NIXL/UCX 过 2.5GbE 有线 -> pearl(RX9060XT,ROCm,decode,NixlPullConnector,192.168.100.2:8200) + disagg_proxy_demo(white:18000)。'
'E2E 证据：curl prompt=The capital of France is -> decode 生成 Paris. The population of Paris is（8 tokens, temperature=0, finish=length），token 语义正确。'
'打通路径上的三个真实障碍及解法：'
'1) vLLM 在 ROCm 平台硬编码 import rixl（vllm/distributed/nixl_utils.py），PyPI 的 rixl 是抢名包（已卸载）；ROCm/rixl GitHub 仓库已 DEPRECATED，AMD 支持已并入上游 ai-dynamo/nixl。'
'2) pip nixl(cu13) 在 pearl 可 import 但 register_memory(VRAM) 报 NIXL_ERR_BACKEND——CUDA 构建的 UCX 插件无法注册 HIP 指针（决定性否定证据）。'
'3) 解法：pearl 源码构建 UCX 1.19.1(--with-rocm=/opt/rocm, ~/ucx-1.19-rocm, 无 sudo) + nixl v1.4.0(-Ducx_path -Dwheel_variant=rocm, 轮内模块名 nixl_rocm)，site-packages 置 rixl.py 别名 shim 映射 rixl._api/_bindings -> nixl_rocm.*；探针 VRAM-REGISTER-OK。'
'4) 其他坑：white vllm-v1 需 PATH 含 env bin（ninja JIT）；pearl hipblaslt FULL cudagraph capture 失败 -> decode 侧 --enforce-eager；旧实例残留吃满 VRAM 导致 Free memory 0.0 GiB；误装抢名 rixl 时误删 httpx 致 huggingface_hub 崩，已补回。'
'边界：decode 侧 enforce-eager（cudagraph FULL 模式 hipblaslt stream-capture 冲突），prefill 侧 CUDA graphs 正常；正式 baseline 前需决定对编译档位的对齐策略。wire 字节计数未在本尖峰采集，战役 harness 中补。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
    VALUES (?,?,'progress',?,?,?,0.95,1.0,'verified',?,datetime('now'),datetime('now'))
    ON CONFLICT(id) DO UPDATE SET content=excluded.content,status=excluded.status,updated_at=datetime('now')''',
    (EV, 'evidence', PROJECT, '三期 A step0 尖峰绿：vLLM PD 跨 CUDA-ROCm 异构 E2E（NIXL/UCX 源码构建路径）', CONTENT, SOURCE))
conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
    ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''',
    (EV, TASK, 'CONFIRMS', 'step0 feasibility gate passed: PD cross-vendor KV transfer works on wired link'))
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('spike evidence recorded')