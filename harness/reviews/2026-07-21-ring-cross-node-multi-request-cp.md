# Review: 跨节点多请求 ring 切分 CP 验证(ringconc-232830)

- **日期**: 2026-07-21
- **对象**: `scripts/run_cross_node_ring_concurrent.sh` 驱动的 white(CUDA) producer(2 chunk)+ pearl(ROCm) consumer(2 并发请求,连续批 CP 路径)验证
- **证据**: `reports/ring-conc-cross-ringconc-232830/{consumer,producer,driver}.log`、`hcp_vllm_plugin/validate_ring_concurrent.py`、`hcp_vllm_plugin/hcp_vllm_plugin/ring_connector.py`、`hcp_vllm_plugin/hcp_vllm_plugin/ring_backend.py`
- **Reviewer**: 独立子代理(只读核验,非主 Agent 自测)

## Verdict: APPROVE

### 关键核验点

1. **参考运行独立**: ref 引擎(默认 ROCM_ATTN,无 connector)与 consumer 引擎(CUSTOM + HcpRingKvConnector)为不同实例、不同时段,ref 先跑并显式释放。
2. **判定逻辑真实执行**: `validate_ring_concurrent.py:242-248` `ok = token_match and mem_ok` + `sys.exit(0/1)`;`mem_ok` 强制 `max_chunks==2, max_layers==48, staged_len==split, overlap==0, leftover==0, max_reqs>=2`,PASS 非硬编码。
3. **真多请求**: 两个不同 prompt(quick brown fox / curious hedgehog),SamplingParams 列表各挂 c0/c1;producer.log 精确计数 c0×24 + c1×24 = 48 个 layer GET,全部来自 pearl IP 100.111.242.55。
4. **显存切分**: overlap==0;staging 仅 transient;"staging freed" 非平凡通过(max_staged_layers==48 先证明填充,leftover==0 才有效)。
5. **连续批 CP**: `max requests in one attention step: 2`,CP 路径确在连续批内执行。
6. **环境一致**: driver.log 三方 git HEAD 一致(8bb2553),producer/consumer 分处 white/pearl 经 ssh 启动。

### 结果

- 双请求 greedy token 与 pearl 单节点参考逐 token 一致
- 2 chunk × 24 层并发 transient staging,512 token/层
- verdict: PASS

### 备注

- 两机日志时间戳差 8 小时系时区差异,事件顺序逻辑一致,不影响结论。
