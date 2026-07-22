# Review: v0.1 插件包装(hcp-vllm-plugin)

- **日期**: 2026-07-22
- **对象**: `hcp_vllm_plugin/`(pyproject entry point、`register()`、README、compat_check.py)
- **Reviewer**: 独立子代理(README↔代码逐条一致性审计)

## Verdict: WARN → 修正后视为 APPROVE(修正已在同一提交完成)

### 审查结论(Reviewer 原文要点)

- pyproject entry point、register() 三项注册、compat_check 六个检查面、五个验证脚本引用:全部属实 ✓
- README 的 9 个 config key、每请求 `hcp_ring` 三键、4 个环境变量及默认值、限制清单:全部与代码一致 ✓
- 版本声明(vLLM 0.23.1rc1 + torch 2.13)与 progress.md/引擎日志一致 ✓

### Reviewer 发现的两处文档错误(已修正)

1. **quickstart chunk key 不自洽**:producer 片段未设 `ring_chunk_id`(默认 `chunk0`),consumer 片段却写 `c0`,照抄会 `_prefix_ready("c0")` 永远为假、无限 stall。修正:producer 片段显式 `"ring_chunk_id": "c0"`。
2. **容差夸大 10 倍**:README 写"端到端 logits 差 ~1e-3",实测为 ~0.02–0.04(~1e-3 是 kernel 级 vs fp32)。修正为区分两级:kernel 级 ~1e-3,端到端 ~0.02–0.04 且 greedy token 逐 token 一致。
3. 次要:quickstart 补注 `disable_hybrid_kv_cache_manager=True`(所有验证运行均携带)。

### 结论

打包机制、注册逻辑、验证证据链全部真实;两处问题均为文档修正范畴,不涉及功能造假。修正提交后包装 v0.1 成立。
