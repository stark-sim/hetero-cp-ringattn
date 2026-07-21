# Review: 跨节点异构 ring 切分 CP 验证(ringx-210415)

- **日期**: 2026-07-21
- **对象**: `scripts/run_cross_node_ring_cp.sh` 驱动的 white(CUDA) producer + pearl(ROCm) consumer 切分 CP 验证
- **证据**: `reports/ring-cross-ringx-210415/{consumer,producer}.log`、`hcp_vllm_plugin/validate_ring_connector.py`、`hcp_vllm_plugin/hcp_vllm_plugin/ring_connector.py`、`hcp_vllm_plugin/hcp_vllm_plugin/ring_backend.py`
- **Reviewer**: 独立子代理(只读核验,非主 Agent 自测)

## Verdict: APPROVE

### 关键核验点

1. **参考引擎独立**: consumer.log 中 ref 引擎(默认 ROCM_ATTN 后端,无 connector)先于 consumer 引擎运行并显式 shutdown + gc;两者后端/配置/KV 池容量均不同,非同一运行。
2. **判定逻辑真实执行**: `validate_ring_connector.py:205-237` 强制 token 精确相等、max|logit diff|<0.1、`mem_ok=(24层 staged 且 staged_len==split 且 overlap==0)`,失败 `sys.exit(1)`,非硬编码 PASS。
3. **显存切分定量证据**: consumer 写 pool 槽位 1027 = 1024(chunk B)+ 3(decode 回写);若 chunk A 被本地重算应见 ~2051。chunk-A 区域槽位本地写入 = 0。
4. **跨节点传输真实**: producer.log 显示 layer 0-23 共 24 个 safetensors GET 全部来自 pearl IP 100.111.242.55,无 127.0.0.1;两机日志时间戳秒级对齐(8h 时区差)。
5. **结果**: ref=cons=`[14579, 220, 22, 21]`,max|logit diff|=0.037(argmax 处 0.0),verdict PASS。

### 次要观察(已处理/可忽略)

- driver stdout 原先未落盘 → 已在脚本中加 `exec > >(tee -a driver.log) 2>&1`。
- `staged_len` 仅抽样 staging 第一个 entry;有 24 层逐层 HTTP GET 日志互证,风险可忽略。
