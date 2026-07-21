import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='code-reading'):
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
    'fact-triton-attn-rdna-fa-path-20260721', 'fact', 'active',
    'TRITON_ATTN 是 ROCm/RDNA 上 flash attention 算法的原生路径(非降级替代)',
    '[2026-07-21 源码+外部资料核实]\n'
    '1. TRITON_ATTN(vllm/v1/attention/backends/triton_attn.py)是 vLLM 一等后端:'
    'prefill 用 context_attention_fwd、decode 用 unified_attention 两个 Triton kernel,'
    '直读 block_table paged KV,分块 tiling + online softmax——与 flash_attn 同算法类,'
    'kernel 语言不同(Triton vs CK/CUDA)。\n'
    '2. RDNA 不走 flash_attn 包的根因是硬件矩阵指令集分裂:ROCm 的 flash_attn 包实体是 '
    'Composable Kernel tile kernel,专门为 Instinct/CDNA(gfx9, MFMA/matrix core, wave64)写'
    '(vllm#4514 原话);RDNA(gfx11/gfx12 消费卡)是 WMMA(AI acceleration, wave32),'
    'rocWMMA 文档支持矩阵分列两类指令集。CK kernel 不以 RDNA 为目标。\n'
    '3. Triton 从高层 IR 编译,ROCm 官方 Triton 后端原生支持 gfx11/gfx12(pearl 为 '
    'triton 3.7.0+rocm7.13);vLLM ROCm 安装文档历来要求装 ROCm Triton flash attention。\n'
    '4. 因此 rocm.py 的分层(gfx9→flash_attn 包/AITER,gfx1x→Triton)是硬件现实的直接映射;'
    'pearl(gfx1200)走 TRITON_ATTN 是设计意图。\n'
    '对第 2 步的含义:pearl 的"原生 kernel"即这套 Triton kernel,kernel 化=复用它并取 LSE。',
    importance=0.85, confidence=0.9, status='held', source='code-reading'
)
insert_edge('fact-triton-attn-rdna-fa-path-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('fact-triton-attn-rdna-fa-path-20260721', 'belief-vllm-cascade-attn-20260721', 'SUPPORTS', weight=0.85,
            note='坐实 pearl 侧 kernel 化路径: Triton kernel + LSE + merge')
insert_edge('fact-triton-attn-rdna-fa-path-20260721', 'decision-ring-paged-kernel-20260721', 'SUPPORTS', weight=0.8,
            note='pearl kernel 化的复用对象确定为 triton_attn 的 context_attention_fwd/unified_attention')

conn.commit()
conn.close()
print('TRITON_ATTN/RDNA fact recorded')
