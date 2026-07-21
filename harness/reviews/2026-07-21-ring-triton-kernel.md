# Review: 第 2 步 ring attention Triton kernel 化验证

- **日期**: 2026-07-21
- **对象**: `ring_triton_attn.py`(自研 Triton kernel,LSE 输出 + causal 偏移)+ `ring_backend.py` dispatch(triton 默认/torch 兜底)+ 全部验证套件
- **证据**: `validate_ring_triton_kernel.py`;pearl `/tmp/ring_conn_consumer.log`(long16k3, 16k/8k)、`/tmp/ring_conc_consumer.log`(tric1)、`/tmp/ring_conn_producer.log`(long16k_torch OOM);white `/tmp/ring_conn_consumer.log`(wtri2)、`/tmp/ring_conc_consumer.log`(wtric2);`reports/ring-conc-cross-ringconc-014233/`
- **Reviewer**: 独立子代理(只读核验,非主 Agent 自测)

## Verdict: APPROVE

### 关键核验点

1. **探针真实**: 6 组形状(含非整除 Tq=37、极小 Tq=1、offset≠0)+ GQA + merge 对比 + 端到端两段合并 vs 全量,失败 `sys.exit(1)`,容差 2e-3 非摆设。
2. **pearl 16k PASS**: tokens match,logit diff 0.023,attn triton 216/0,24 层×8192 token staging,overlap 0,staging 释放。vLLM jit_monitor 日志记录 `_ring_fwd_kernel` JIT 编译——kernel 真实执行的旁证。
3. **torch 对照 OOM 属实**: `/tmp/ring_conn_producer.log`(long16k_torch)`torch.exp(scores - m)` 处 3.50 GiB 分配失败;3.50 GiB 与 fp32 scores `[14,8192,8192]×4B` 精确吻合;日志无 "ring triton attn failed" 警告,确认是显式 `HCP_RING_IMPL=torch` 而非静默回退。
4. **white 双 PASS**: 216/0、408/0,同一 kernel 跨平台。
5. **跨节点 ringconc-014233**: 三方 HEAD 一致(18a1046),48 层 staging 经 HTTP 来自 white,GET 全部来自 pearl IP。
6. **代码审查**: LSE 自然对数换算正确(m_i×ln2+ln(l_i),m_i 为 exp2 域);Q_OFFSET 同时作用于查询位置与 causal end_n 剪枝;v load 掩码维度与 [BLOCK_N, BLOCK_D] 布局一致(转置 bug 已修);IMPL_STATS 双路径计数,验证脚本断言 `attn_triton>0 && attn_torch==0` 非硬编码。

### 备注(不影响结论)

- OOM traceback 在 producer 日志而非 wrapper 日志(声明小瑕疵,实质内容属实)。
- 建议今后探针 stdout tee 到 reports/ 留痕(已采纳为后续惯例)。
