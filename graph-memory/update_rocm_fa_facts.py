import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

# Refine belief-vllm-cascade-attn-20260721 with platform-specific facts
refined = (
    'vLLM cascade/LSE 机制存在但平台分层：CUDA 有 vendored FA(含 LSE),ROCm/RDNA 走 Triton + merge_attn_states 算子',
    'vLLM 的 cascade attention(共享前缀与各请求私有后缀分别算 attention 再按 LSE 合并)与 HCP ring backend 的 '
    'local(chunk B) + peer(chunk A) LSE merge 数学同构。但[2026-07-21 源码核实]平台能力分层：\n'
    '1. "vLLM 内置 flash_attn" 仅覆盖 CUDA(vllm.vllm_flash_attn vendored kernel)与 XPU;'
    'ROCm 在 fa_utils.py 里是 try: from flash_attn import ...(依赖用户自装上游包),'
    'pearl 的 vllm-rocm env 无 flash_attn/aiter 包 => FLASH_ATTN 后端不可用;\n'
    '2. vLLM 官方对 ROCm 分层(rocm.py):gfx9(CDNA) 预期 AITER FA / 上游 flash_attn 包;'
    'RDNA(gfx11xx/gfx12xx, pearl 9060 XT 为 gfx1200) 官方预期路径即 Triton 实现(注释原文);'
    '有 kv_connector 时 ROCM_ATTN 因 KV layout 不兼容被排除 => pearl connector 场景后端只有 TRITON_ATTN/CUSTOM;\n'
    '3. TRITON_ATTN 后端 assert attn_metadata.use_cascade is False(不接 cascade),'
    '但 vllm.v1.attention.ops.merge_attn_states(含 triton 版)输入正是 '
    '(prefix_out, prefix_lse, suffix_out, suffix_lse),形状与 HCP merge 一致;\n'
    '4. 推论(第 2 步平台策略):white 复用 vendored FA 的 LSE;pearl 用 triton kernel 算两段 + '
    'merge_attn_states——但该算子在 gfx1200 上须先做数值稳定性验证(HCP 团队曾在 ROCm 见过 inf),'
    '不可靠则保留已验证的 plain-PyTorch merge(3e-7)兜底。',
)

c.execute("""
    UPDATE nodes SET title = ?, content = ?, updated_at = datetime('now')
    WHERE id = 'belief-vllm-cascade-attn-20260721'
""", refined)

c.execute("""
    INSERT OR REPLACE INTO edges (source, target, type, weight, note)
    VALUES ('decision-ring-paged-kernel-20260721', 'belief-vllm-cascade-attn-20260721', 'DEPENDS_ON', 0.85,
            '第 2 步平台策略: white 用 vendored FA LSE; pearl 用 triton + merge_attn_states(先验证 gfx1200 数值稳定性)')
""")

conn.commit()
conn.close()
print('belief refined with ROCm FA platform facts')
