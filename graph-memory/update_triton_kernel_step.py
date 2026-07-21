import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='experiment'):
    c.execute('''
        INSERT OR REPLACE INTO nodes
        (id, type, layer, project, title, content, importance, confidence, status, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ''', (id_, type_, layer, PROJECT, title, content, importance, confidence, status, source))

def insert_edge(src, tgt, type_, weight=1.0, note=None):
    c.execute('''
        INSERT OR REPLACE INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, ?, ?)
    ''', (src, tgt, type_, weight, note))

insert_node(
    'ev-ring-triton-kernel-20260721', 'evidence', 'progress',
    '[2026-07-21] 第 2 步完成：ring attention 换自研 Triton kernel(带 LSE) + merge_attn_states,双平台验证通过',
    '决策 decision-ring-paged-kernel-20260721 的实现(commit 0f7056c..18a1046)。\n'
    '设计：vLLM 原生 triton kernel 不输出 LSE 且 TRITON_ATTN 不支持 cascade => 插件内自研 '
    'ring_triton_attn.py(fork vllm triton_prefill_attention,加 LSE 输出 + Q_OFFSET causal 偏移);'
    '同一 Triton kernel 覆盖 CUDA 与 ROCm(不再按平台分叉);local(causal+offset)/peer(non-causal) '
    '两段都走它,merge 默认用 vllm merge_attn_states(triton);HCP_RING_IMPL/HCP_RING_MERGE 可切回 '
    'plain-torch 兜底;IMPL_STATS 计数器证明真实路径。\n'
    '验证(全 PASS):\n'
    '1. 数值探针(pearl gfx1200):kernel vs fp32 参考 max|diff|~1e-3(fp16 舍入),LSE ~1e-6;'
    'merge_attn_states vs plain merge 6.1e-5(无 inf);端到端两段合并 vs 全量 ~1e-4;\n'
    '2. pearl 三件套(connector 单请求/并发/backend customst)triton 路径全 PASS,'
    'IMPL_STATS 216/408 次 triton 调用、0 torch 回退;\n'
    '3. pearl 16k/8k 长上下文:PASS(24 层×8192 token staging、overlap 0);'
    '对照 HCP_RING_IMPL=torch 同规模 OOM 于 score 矩阵物化(exp(scores) 单次 3.50 GiB 分配失败)'
    '——kernel 化动机被实证;\n'
    '4. white(RTX 4090 CUDA) 单请求+并发:PASS,同一 kernel,0 回退;\n'
    '5. 跨节点并发复验 ringconc-014233(white producer 2 chunk + pearl consumer 2 并发):PASS。\n'
    '过程修复:v load mask 转置 bug(Tk%BLOCK_N!=0 时越界键未清零,探针 Tq=37 暴露);'
    'validate_ring_backend customst 适配新 staging 签名(单 chunk 无映射回退)。\n'
    '意义:score 矩阵不再物化,长上下文显存天花板消除,为 128K+/1M 的 vLLM 线扫清自实现障碍。',
    importance=0.95, confidence=0.95, status='held', source='experiment'
)
insert_edge('ev-ring-triton-kernel-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('ev-ring-triton-kernel-20260721', 'decision-ring-paged-kernel-20260721', 'CONFIRMS', weight=0.95,
            note='按决策实施并双平台验证通过;顺序 3→2→1 的第 2 步完成')
insert_edge('ev-ring-triton-kernel-20260721', 'fact-triton-attn-rdna-fa-path-20260721', 'CONFIRMS', weight=0.9,
            note='Triton 跨 CUDA/ROCm 可移植性实证(同一 kernel 两平台 PASS)')
insert_edge('ev-ring-triton-kernel-20260721', 'belief-vllm-cascade-attn-20260721', 'CONFIRMS', weight=0.9,
            note='merge_attn_states 在 gfx1200 数值稳定(6.1e-5),cascade 式合并落地')
insert_edge('ev-ring-triton-kernel-20260721', 'decision-vllm-plugin-packaging-20260721', 'ENABLES', weight=0.8,
            note='kernel 与 staging 定型,插件配置面可冻结(第 1 步前置达成)')

# Close the paged-kernel decision as implemented
c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'decision-ring-paged-kernel-20260721'
""")

# Lesson: fork kernel 时逐行核对 mask/维度语义,探针要覆盖非整除形状
insert_node(
    'lesson-kernel-fork-mask-20260721', 'lesson', 'progress',
    '[2026-07-21] fork kernel 时 mask 的维度语义要逐行核对;数值探针必须覆盖非整除/边界形状',
    'ring_triton_attn fork 自 vllm triton_prefill_attention 时,v load 的 mask 被误写成与 qk mask '
    '同形([1, BLOCK_N],而 v 布局是 [BLOCK_N, BLOCK_D]),Tk 为 BLOCK_N 整数倍时恰好全真不暴露,'
    'Tk=37 时越界键未清零产生垃圾/nan。教训:\n'
    '1. fork kernel 时每一行 mask/stride 的维度语义都要与原布局核对,不能凭"形状能广播";\n'
    '2. 数值探针形状集必须含非整除(Tq=37)、极小(Tq=1)、偏置(offset≠0)案例——'
    '本 bug 只有非整除案例暴露,整齐形状全过;\n'
    '3. "LSE 全对但输出错"的定位价值:说明 online-softmax 记账正确,问题在数据装载(v mask)而非数学。',
    importance=0.8, confidence=0.9, status='held', source='reflection'
)
insert_edge('lesson-kernel-fork-mask-20260721', 'ev-ring-triton-kernel-20260721', 'BASED_ON', weight=0.9)

conn.commit()
conn.close()
print('triton kernel evidence + lesson recorded in graph.db')
