import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source=None):
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

# Deep-read evidence node
insert_node(
    'ev-stripe-attention-deep-read', 'evidence', 'active',
    '精读：Striped Attention 机制与 HCP 适配点',
    '来源：William Brandon 等，MIT，arXiv:2311.09431。\n\n'
    '核心机制：\n'
    '- 输入序列按 token 下标对 N（device 数）取模做 permutation，device i 持有下标满足 i mod N 的 token。\n'
    '- 因此每个 device 的 Q/K/V block 包含均匀散布在整个原始序列中的 token，而非连续 chunk。\n'
    '- 在每层 attention 开始前，Q/K/V 已经按此 layout 分好，不需要额外的 per-layer 通信。\n'
    '- Mask 调整：因果 mask 仍基于原始序列顺序；Striped 的 GetMask 保证每个 device 每轮遇到的上三角 mask 比例大致相同，从而负载均衡。\n'
    '- 对每轮 (Q_j, K_k, V_k)，若 j<k 则 mask 为下三角（含对角线以上全 -inf）；若 j≥k 则 mask 为上三角（含对角线以下全 -inf）。\n'
    '- Workload：i≥j 时约 c(c+1)/2，i<j 时约 c(c-1)/2；最大 workload 从 Ring 的 c² 降到接近 c²/2，理论极限 speedup 2×。\n\n'
    '实验结果：\n'
    '- 8×A100 80GB，256K 序列，最高端到端吞吐提升 1.45×；16×TPUv4，786K 序列，1.65×。\n'
    '- 序列越长、device 越多、block 越大，收益越明显。\n'
    '- 实现基于 JAX，使用 bfloat16 + float32 attention，tile-based skipping。\n\n'
    'HCP 适配关键点：\n'
    '- P2P-only 友好：仍然保持 Q 固定、KV 沿 ring P2P 传递，通信原语不需要 all-to-all / all-gather。\n'
    '- 非均等 chunk 兼容性：Striped 原始假设均分 block，但 permutation 本身可以推广到不均等 block（只要每个 device 的 token 在原始序列中均匀散布）。\n'
    '- RoPE/位置编码：必须对 position ids 同步 permutation；HCP 的 distributed RoPE 需要知道原始全局位置。\n'
    '- Online softmax：与 Ring Attention 完全一致，可直接复用 HCP 的 online softmax state 更新逻辑。\n'
    '- 当前 HCP 中 pearl（小/慢 domain）在 Phase 2 接收更多 remote block 的瓶颈，有望通过 striped 缓解。',
    importance=0.9, confidence=0.85, status='held', source='https://ar5iv.org/html/2311.09431'
)
insert_edge('ev-stripe-attention-deep-read', 'ev-stripe-attention-paper', 'BASED_ON', weight=1.0)
insert_edge('ev-stripe-attention-deep-read', 'hyp-stripe-ring', 'CONFIRMS', weight=0.9)

# Update hypothesis with more concrete content
c.execute('''
    UPDATE nodes SET content = ?, confidence = ?, importance = ?, updated_at = datetime('now')
    WHERE id = 'hyp-stripe-ring'
''', (
    '将 Striped Attention 的 striped permutation 引入 HCP，以缓解因果 attention 下 Ring Attention 的负载不均。'
    '具体需验证：1) 不均等 chunk size 下的 permutation 定义；2) RoPE position ids 的同步 permutation；3) 对 pearl 类慢节点的实际加速效果。'
    '预计对长序列、多 domain 场景收益最大。',
    0.75, 0.85
))

# P2P feasibility assessment
insert_node(
    'decision-p2p-feasibility', 'decision', 'active',
    'P2P-only 异构场景下的 Ring Attention 衍生方案筛选',
    '筛选标准：HCP 跨异构 domain 只支持 P2P send/recv，不支持 all-to-all / all-gather / reduce-scatter 等 collective。'
    '因此只保留可在纯 P2P ring 上实现的算法，排除依赖 NCCL/process-group 的方案。\n\n'
    '✅ 适合 P2P-only / HCP：\n'
    '- 原始 Ring Attention（Liu et al. 2023）：Q 固定，KV 沿 ring P2P 传递，online softmax。\n'
    '- Striped Attention（Brandon et al. 2023）：在 Ring 基础上只做输入 permutation + mask 调整，通信原语不变。\n'
    '- ZigZag Ring Attention（ring-flash-attention issue #2）：通过折叠 query 维度平衡负载，仍只需 P2P KV 传递。\n'
    '- Ring Flash Attention（zhuzilin 等开源）：将 FlashAttention kernel 与 Ring P2P 重叠，支持 ring/zigzag/stripe 模式。\n\n'
    '❌ 不适合 P2P-only（需要 collective 或与 HCP 假设冲突）：\n'
    '- DeepSpeed Ulysses：依赖 all-to-all 交换 Q/K/V，需要同构 NCCL process group。\n'
    '- USP（Tencent）：混合 Ulysses + Ring，Ulysses 段仍需 all-to-all，无法纯 P2P。\n'
    '- Llama3 flash_attn_varlen_func（ring-flash-attention）：技术上不是 ring attention，使用不同 CP 机制。\n'
    '- MoBA / XAttention / MTraining：稀疏/动态 attention 改变 attention 数学定义，HCP correctness-first 阶段不引入近似；且 MTraining 基于 Striped 但加入动态稀疏，需先验证基础 Striped。\n'
    '- LightSeq：优化 sequence-parallel 的 all-to-all / reduce-scatter 通信，非 P2P。\n'
    '- Mnemosyne：服务调度系统，非算法本身。',
    importance=0.85, confidence=0.8, status='held', source='web-survey + paper analysis'
)
insert_edge('decision-p2p-feasibility', 'bp-p2p-decision', 'PART_OF')
insert_edge('decision-p2p-feasibility', 'hyp-stripe-ring', 'SUPPORTS', weight=0.9,
            note='Striped 是 P2P-only 可行的优选方案')
insert_edge('decision-p2p-feasibility', 'claim-ring-derivatives', 'UPDATES', weight=0.8)

# Training possibility note
insert_node(
    'note-training-scope', 'claim', 'active',
    '训练场景评估：Striped Attention 训练收益对 HCP 当前目标意义有限',
    'Striped Attention 论文主要面向训练（forward + backward）。'
    'HCP 当前聚焦推理，且目标硬件是异构消费级设备（CUDA + HIP/MPS），互联带宽/延迟远低于训练集群。'
    '若扩展到训练，需要：\n'
    '- backward 阶段沿反方向传递梯度，并维护 ring 中的 activation/gradient buffer。\n'
    '- 跨 domain 的梯度同步（all-reduce 或类似机制），这与 P2P-only 假设冲突。\n'
    '- 消费级设备的 PCIe/Ethernet 互联难以支撑训练所需的高吞吐参数/梯度通信。\n'
    '结论：训练在理论上可行，但不是 HCP 当前阶段的高优先级方向；先把推理 + Striped 走通。',
    importance=0.7, confidence=0.75, status='held', source='paper-analysis + user-direction'
)
insert_edge('note-training-scope', 'hyp-stripe-ring', 'QUESTIONS', weight=0.5,
            note='训练不是当前重点，但可记录为低优先级')

# Update task to concrete next steps
c.execute('''
    UPDATE nodes SET content = ?, status = ?, updated_at = datetime('now')
    WHERE id = 'task-stripe-survey'
''', (
    '精读已完成。下一步：\n'
    '1. 在 HCP Rust correctness model 中实现 striped permutation（含 RoPE position id permutation）。\n'
    '2. 修改 online softmax mask 逻辑以匹配 Striped GetMask。\n'
    '3. 用同构/异构 uneven chunk 配置跑 correctness test，测量 per-domain 计算量。\n'
    '4. 输出设计文档：capacity-aware 不均等分片下如何应用 stripe，以及是否需同时调整 chunk 分配策略。',
    'ongoing'
))

conn.commit()
conn.close()
print('Deep-read findings recorded')
