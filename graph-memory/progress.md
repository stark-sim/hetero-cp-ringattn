# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

### [2026-07-21] HcpRingKvConnector：peer KV 以“切分瞬时”接入，2 进程显存切分验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `hcp_vllm_plugin/hcp_vllm_plugin/ring_connector.py + validate_ring_connector.py + /tmp/my_ring_val.log`

按用户约束（KV connector 默认是全量搬移，HCP 是切分瞬时）实现 HcpRingKvConnector（KVConnectorBase_V1）：调度侧 get_num_new_matched_tokens 仅把前序 chunk 标记为 external，给本 chunk 提供全局 RoPE 位置并阻止重复计算；worker 侧 start_load_kv 经 HTTP 从 producer 拉取 peer chunk 每层 KV，写入 ring_backend 的 PEER_KV_STAGING（瞬时），绝不写入常驻 paged pool——与 stock disaggregated-prefill 全量复制语义明确区分。ring_backend 增加 WRITE_TRACK 证明显存切分。验证（pearl 单机 2 个 vLLM 0.23 实例，CUSTOM backend + ring connector，2048-token prompt 切 1024+1024，greedy decode 4）：consumer tokens [14579,220,22,21] 与单节点一致，max|logit diff| 0.027（argmax 处 0.016），chunk-A 常驻池本地写入=0，peer KV 1024 tokens/layer×24 层全部经 HTTP 拉取（独立复跑通过，exit 0）。后续：跨节点（white CUDA producer + pearl ROCm consumer）、decode 充分性、性能（ROCm 无 flash_attn，目前 plain-PyTorch）。

_updated: 2026-07-21_
### [2026-07-21] flash_attn 平台现状：white CUDA 已可用，pearl ROCm 构建中

type: `evidence` · status: `closed` · confidence: 0.7 · importance: 0.8 · source: `white/pearl flash_attn probe`

flash_attn 双平台接通进展（下一步顺序第1步）：\n- white（CUDA，vLLM 0.23.1rc1）：无需单独装 flash_attn 包，vLLM vendored vllm_flash_attn 已可用，is_flash_attn_varlen_func_available()=True；实测 flash_attn_varlen_func(..., return_softmax_lse=True) 返回 (out [5,2,64], lse [2,5])，flash_attn+LSE 在 white 正常。\n- pearl（ROCm gfx1200，vLLM 0.23.1rc1）：is_flash_attn_varlen_func_available()=False（无 vendored ROCm flash_attn，也无 ROCm flash_attn 包，回退 Triton）。AMD 官方 index 无 gfx1200 预编译 flash-attn wheel。正在用 ROCm/flash-attention 的 main_perf 分支 + FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE（Triton 后端，硬件无关）源码构建，目标让 pearl 的 flash_attn 可用。\n注意：ROCm 的 flash_attn 是 ROCm/flash-attention fork，官方 flash-attn 为 CUDA-only；Triton 后端理论上可在 RDNA4 gfx1200 运行。

_updated: 2026-07-21_
### [2026-07-21] decode 充分验证：continuous batch + 多步 decode 全过（独立复跑）

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `hcp_vllm_plugin/validate_decode.py + /tmp/my_decode_val.log`

第2步 decode 充分验证（validate_decode.py，主 Agent 独立复跑确认，exit 0）：\n1) no-peer 退化 + 多步 decode：CUSTOM backend、HCP_RING_SPLIT_TOKENS=0，2048-token prompt greedy 16 token，全部匹配单节点参考（[220,23,15,74459,...]，max|logit diff| 0.023）。\n2) continuous batching：6 个长度 [64,200,350,700,1000,1500] 的 prompt 一次 generate 提交，BATCH_STATS.max_reqs=6 证明真在同一 attention step 批处理（非串行），6 个请求各 16 token 全部匹配单节点（diff 0.019–0.035）。证明 vLLM 连续批处理基础能力在 CUSTOM ring backend 下正常。\n3) CP 路径多步 decode：2 进程 ring-connector 切分（producer chunk A + consumer 全 prompt，HTTP 拉 peer KV），decode=8 与 decode=16 均 PASS，consumer 16 token [14579,220,22,21,...] 逐步匹配单节点；显存切分保持——consumer 写 1039 pool slots（1024 chunk-B prefill + 15 decode），chunk-A 常驻池本地写入=0。\n已知限制：PEER_KV_STAGING 按 layer 键，多并发 consumer 请求若 peer chunk 不同会互相覆盖，故 CP 路径限单并发（max_num_seqs=1）；no-peer 批处理无此限制。正确修法：staging 按 (request_id, layer) 键并把 request 身份经 attn_metadata 传入 forward（后续）。

_updated: 2026-07-21_
### [2026-07-17] vLLM 0.23.1rc1 源码编译补丁（gfx1200）

type: `evidence` · status: `held` · confidence: 0.75 · importance: 0.8 · source: `bash-i3gxwyr5 build log`

在 pearl（RX 9060 XT / gfx1200，ROCm 7.13，PyTorch 2.13.0a0+rocm7.13.0a20260416）上从 main 分支源码构建 vLLM 0.23.1rc1.dev905+g3f99883d9。为通过编译已打两个补丁：1) csrc/spinloop.cpp：将 <mwaitxintrin.h> 改为 <x86intrin.h>，修复 ROCm Clang 23 直接包含 mwaitxintrin 的编译错误；2) 禁用 GPTQ 路径：从 CMakeLists.txt 移除 csrc/libtorch_stable/quantization/gptq/q_gemm.cu，并在 csrc/libtorch_stable/torch_bindings.cpp 中注释 gptq_gemm/gptq_shuffle 的 ops.def/ops.impl，规避 HIP half2 atomicAdd 缺失导致的编译失败。当前构建仍在后台运行（task bash-i3gxwyr5），正在编译 HIP 对象。

_updated: 2026-07-17_
### [2026-07-17] vLLM 0.23.1rc1 源码编译成功并通过 ROCm gfx1200 prefill

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `bash-vjw890d3 build log + /tmp/test_vllm_prefill.py`

在 pearl（RX 9060 XT / gfx1200，ROCm 7.13，PyTorch 2.13.0a0+rocm7.13.0a20260416）上完成 vLLM 0.23.1rc1.dev905+g3f99883d9 源码编译。关键修复：1) csrc/spinloop.cpp 用 <x86intrin.h> 替代 <mwaitxintrin.h>；2) 禁用 GPTQ（CMakeLists 移除 q_gemm.cu，torch_bindings 注释 gptq_gemm/gptq_shuffle）规避 HIP half2 atomicAdd 缺失；3) 在 conda env bin 下把 clang/clang++/clang-cpp 软链到 amdclang/amdclang++/amdclang-cpp，修复 hipcc_cmake_linker_helper 链接失败。运行时用 LD_LIBRARY_PATH 覆盖 torch/lib 与 _rocm_sdk_{core,devel}/lib{,/host-math/lib,/rocm_sysdeps/lib}。验证：从 /tmp 运行脚本（避免 cwd=/home/stark 时 repo 目录名 vllm 把 import 变成 namespace package），LLM(model=/home/stark/models/Qwen2-0.5B-1M, dtype=float16, enforce_eager=True) 成功初始化并在 ROCm gfx1200 上 prefill+decode，输出 I am 类 token。

_updated: 2026-07-17_
### [2026-07-17] vLLM 0.23 V1 Block-Ring 插件在 pearl(gfx1200) 验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `pearl /tmp/poc_out.log + scripts/poc_vllm_block_ring_v1.py`

在 pearl（RX 9060 XT / gfx1200，自编译 vLLM 0.23.1rc1.dev905+g3f99883d9，ROCm 7.13）上实现并验证 V1 引擎版 Block-Ring 插件。实现：python/hcp_vllm_block_ring_plugin_v1.py 用 enable_multiprocessing=False 的 LLMEngine 同进程访问 model_executor/scheduler/KV cache，直接用 block_pool 分配物理块，手工构造 SchedulerOutput+NewRequestData 调 model_executor.execute_model（返回 None 时再 sample_tokens），KV cache 布局与 0.6.4 一致 [2, num_blocks, block_size, num_kv_heads, head_dim]。PoC：scripts/poc_vllm_block_ring_v1.py，Qwen2-0.5B-1M、fp32、block_size=16、chunk 16+16。结果：chunk A prefill + chunk B 带 context prefill + combined block table，最后位置 next token 与单节点 vLLM 参考一致（match=True），自回归 decode 第二个 token 也一致（match=True）。注意事项：1) ROCm attention 后端不支持 block_size=8，需用 16；2) 1M 模型默认 max_model_len=1048576 会导致 KV cache 初始化 OOM，插件/参考都需显式传 max_model_len（如 4096）；3) 运行 cwd 不能在含 vllm 子目录的路径（否则 import vllm 变 namespace package）；4) 运行需 LD_LIBRARY_PATH 覆盖 torch/lib 与 _rocm_sdk_{core,devel}/lib{,/host-math/lib,/rocm_sysdeps/lib}。

_updated: 2026-07-17_
### [2026-07-17] 跨节点 vLLM Block-Ring CP 验证通过（white CUDA + pearl ROCm）

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/vllm-cp-cuda-hip-20260717-134004 + scripts/run_cross_node_vllm_cp.sh`

首次实现 vLLM 跨节点 context-passing CP：white（RTX 4090 CUDA，vLLM 0.6.4 legacy 插件）作 domain 0，pearl（RX 9060 XT ROCm gfx1200，vLLM 0.23.1rc1 V1 插件）作 domain 1，经 Rust coordinator + QUIC KV ring 协作同一序列。关键设计：vLLM PagedAttention 的正确 CP 必须 context-passing——domain 1 先收 domain 0 的 chunk A KV 作 context 再 prefill chunk B（层 L 的 K/V 依赖层 L-1 的 context，先 prefill 再交换在数学上不正确）。新增：plugins 的 prefill_with_context_kv / set_global_seq_len / _local_seq_offset，decode/last_token 用 _global_seq_len（peer chunk 的 token id 不需要，早期 token 用占位符）；python/hcp_worker_sdk/cp_server.py（CpVllmWorkerServer，domain0 send-then-recv、domain1 recv-then-prefill-then-send）；python/hcp_vllm_cp_worker.py（自动识别 vLLM 0.6.x vs >=0.23）；scripts/run_cross_node_vllm_cp.sh。修复：domain 0 需按 prefill 时的 seq_len 上报，否则 coordinator 会错用其 chunk-local logits。验证：64-token 变化 prompt（alpha bravo ... qu），chunks 32+32，block_size 16，greedy 6 token，distributed 输出 ail rose rosemary rosewood 与单节点 vLLM 完全一致。已知限制：KvBlock 布局 [num_blocks, block_size, kv_heads, head_dim] 与 transformers/Rust 的 [batch, heads, seq, dim] 不同，故 vLLM worker 目前只能与 vLLM worker 组环。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector（KVConnectorBase_V1）单机 2 实例验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `scripts/poc_hcp_cp_connector.py + /tmp/poc_conn.log`

实现 vLLM 官方 KV connector 扩展点版本的 context-passing CP：hcp_vllm_plugin/ 包（pyproject + vllm.general_plugins 入口 + kv_connector_module_path），HcpCpConnector 以 ExampleConnector 为模板，producer 计算本 chunk 并共享存储 KV，consumer 把前序 chunk 标记为 external prefix（get_num_new_matched_tokens）只算本 chunk。关键修复：同步共享路径 load 必须返回 load_kv_async=False；get_finished 返回 (None,None) 避免 scheduler 断言。验证（pearl 单机 2 实例，vLLM 0.23.1rc1，Qwen2-0.5B-1M，64-token 变化 prompt，chunk 32+32）：consumer 首 token 604(ail) 与单节点参考一致，exit=0。该路线不打补丁、用官方稳定 API，故能跟进 vLLM 官方更新。注意：KV connector 仅 V1 引擎支持，跨节点异构需 white 也构建 V1 vLLM（当前 white 为 0.6.4）。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector HTTP 跨机传输验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `scripts/poc_hcp_cp_connector.py --http-port + /tmp/poc_http.log`

为 HcpCpConnector 增加 HTTP 跨机 KV 传输：producer 端仅 worker-side 起 ThreadingHTTPServer 共享 KV store（cp_serve_port），consumer 端 cp_peer_url 拉取（HEAD 探活 _READY，GET 拉 layer safetensors），带 5 次重试解决 IncompleteRead。修复：connector 按 role 实例化两次（scheduler+worker），HTTP server 只能 worker-side 绑定否则端口冲突。验证（pearl 单机 2 实例经 loopback HTTP，64-token，chunk 32+32）：0 个 fetch 失败，consumer 604(ail) 与单节点一致。至此 HCP 已成为一个不依赖补丁、基于 vLLM 官方 KVConnectorBase_V1 稳定 API 的生态插件，可跨机做 context-passing CP。异构跨节点（white CUDA + pearl ROCm）仍需 white 构建 V1 引擎 vLLM（当前 white 为 0.6.4 legacy）。

_updated: 2026-07-17_
### [2026-07-17] HcpCpConnector 跨节点异构验证通过（white CUDA + pearl ROCm）

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/cp-plugin-cpplug-201341 + scripts/run_cross_node_cp_plugin.sh`

完成 HCP 作为 vLLM 生态插件的跨节点异构验证。先在 white（RTX 4090）构建 V1 引擎 vLLM 0.23.1rc1.dev905+g3f99883d9：新建 conda env vllm-v1，装 torch 2.13.0+cu126（下载慢约50min），clone 到 3f99883d9，装 build 依赖（cmake<4、ninja、setuptools-rust），关键是用 conda gcc-13 作为 nvcc host compiler 解决 Ubuntu 26.04 glibc 2.43 + CUDA 13.1 + gcc-15 的 rsqrt exception-spec 冲突；pip 装上 cu130 torch + torchvision 后 vLLM 0.23 在 white 跑通 prefill。随后跨节点：white producer（CUDA，HcpCpConnector，cp_serve_port=8899）算 chunk A 并经 HTTP 供 KV，pearl consumer（ROCm gfx1200，HcpCpConnector，cp_peer_url=http://white:8899）拉取 chunk A KV 作 external prefix 算 chunk B。验证（Qwen2-0.5B-1M，64-token 变化 prompt，chunk 32+32，greedy 4 token）：consumer [604,16009,16009,1534]=ail rose rosemary 与单节点参考完全一致。至此 HCP 是一个不打补丁、基于官方 KVConnectorBase_V1 稳定 API、可跨异构节点做 context-passing CP 的 vLLM 生态插件，能跟进 vLLM 官方更新。

_updated: 2026-07-17_
### [2026-07-17] HcpRingAttentionBackend：vLLM 显存切分 online softmax ring attention 验证通过

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `hcp_vllm_plugin/hcp_vllm_plugin/ring_backend.py + validate_ring_backend.py + /tmp/ring_val.log`

实现 vLLM 显存切分（memory-splitting）online softmax ring attention：自定义 attention backend HcpRingAttentionBackend（FlashAttentionBackend 子类，注册为 CUSTOM，vllm.general_plugins 入口）。每个 worker 只永久持有自己 chunk 的 KV，attention 时对 local chunk（causal）与 transient peer chunk（non-causal）分别计算 (O, LSE)，用 plain-PyTorch online softmax 合并，peer KV 经 PEER_KV_STAGING 瞬时暂存而不入 paged pool。RoPE 位置：单请求全 prompt，backend 按 HCP_RING_SPLIT_TOKENS 切分 peer/local，数学上等价 2-worker 分片。验证（pearl ROCm gfx1200，vLLM 0.23.1rc1，Qwen2-0.5B-1M，2048-token，split=1024，greedy）：ref/custom0/custom/customst 四种模式 sampled token 均 14579，top-5 集合一致，logits 差异在 fp16 噪声内（独立复跑通过）。ROCm 事实：flash_attn 未安装故用 plain-PyTorch attention（fp32 累加，correctness-grade）；merge_attn_states 未用（Triton 内核在 ROCm 有 inf 问题）。后续：KV connector 接线真实网络 peer KV、2 进程全局位置偏移、decode 阶段验证、性能优化。

_updated: 2026-07-17_
### [2026-07-02] vLLM Block Ring 插件骨架与 PoC 修正

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `git commit 3467cb4`

在 white 已可运行 vLLM 0.6.4 的基础上，继续完善插件实现并提交 commit 3467cb4。\n\n变更点：\n- python/hcp_vllm_block_ring_plugin.py：实现 VllmBlockRingPlugin.prefill / decode / get_kv_block / apply_peer_kv，直接调用 vLLM model_executor 绕过 scheduler。\n- 为 peer KV 在所有 attention 层复用同一组物理 block，避免 vLLM block table 跨层不一致。\n- 增加 _rope_delta_rotate_keys：对以 local position 预fill 的 peer key 做 RoPE delta 旋转，使其 global position 与 decode query 对齐。\n- scripts/poc_vllm_block_ring_2worker.py：修正 decode 输入为最后 prompt token，使用 set_global_tokens 同步完整序列，默认 prompt 长度满足 block_size 对齐断言。\n\n限制：\n- prefill() 目前返回 one-hot sampled token（非完整 last-token logits），与 HcpWorkerBackend 接口兼容但语义上是近似。\n- RoPE 校正目前只支持标准 Neox 配对 RoPE 与 rope_theta；尚未处理 rope_scaling / Yarn / NTK。\n- 需要等待 pearl 上 vLLM 源码编译完成后才能做真实 ROCm 硬件验证。

_updated: 2026-07-02 14:58:04_
### [2026-07-01] 搜索 vLLM RDNA4/gfx1200 社区轮子结果

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `web search`

搜索结论：\n\n1. 未发现可直接 pip install 的 vLLM 0.6.4 gfx1200 预编译 wheel。\n2. ROCm TheRock 提供 per-family nightly Python 包：gfx120X-all 索引（https://rocm.nightlies.amd.com/v2/gfx120X-all/），包含 PyTorch/ROCm 对 gfx1200/gfx1201 的支持。\n3. vLLM 上游 rocm/vllm-dev:base Docker 的 PYTORCH_ROCM_ARCH 已包含 gfx1200;gfx1201，说明源码构建的 arch list 已经支持。\n4. Step-Audio 的 Dockerfile 展示了在 gfx1151/gfx1200/gfx1201 上源码构建 vLLM 的 patch 路径。\n5. 社区 ROCmLibs 提供 gfx1201 的 hipblaslt/rocblas 库，但主要用于 Windows/koboldcpp。\n\n结论：没有现成轮子；最可行路径是用 TheRock gfx120X-all  nightly PyTorch + 源码编译 vLLM 0.6.4，目标 arch gfx1200。

_updated: 2026-07-02 14:15:47_
### [2026-06-30] vLLM Block-Aware Ring 提取 PoC

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `white RTX 4090 vLLM 0.6.4 experiment`

在 white RTX 4090 上使用 vLLM 0.6.4 + Qwen2.5-3B 验证：\n\n1. 可以定位 CacheEngine.gpu_cache[layer] 的物理 block 布局：shape=(2, num_gpu_blocks, block_size, num_kv_heads, head_dim)。\n2. 可以读取任意物理 block 的 K/V：gpu_cache[layer][0/1, block_id]。\n3. 可以将序列化后的 block 写入新的未使用物理 slot，字节级一致。\n4. 通过 scheduler.block_manager.get_block_table(seq) 可以获取序列的 block table。\n\n结论：vLLM Block-Aware Ring 的 block 提取/写入路径可行，不需要修改 attention kernel。\n\n脚本：scripts/poc_vllm_block_extract.py, scripts/inspect_vllm_blocks.py

_updated: 2026-06-30 09:19:48_
### [2026-06-30] 正常规模工作负载对比：3B/7B，1K/4K

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node runs on white/pearl + white CPU/CUDA single-node benchmarks`

在 white+pearl 上对 Qwen2.5-3B / 7B 进行单节点与分布式对比，seq=1024/4096。\n\n单节点基线（white）：\n- 3B/1K CUDA 0.14s, CPU 7.78s\n- 3B/4K CUDA 0.27s, CPU 29.26s\n- 7B/1K CUDA 0.22s, CPU 17.58s\n- 7B/4K CUDA 0.52s, CPU 64.09s\n\n分布式 3B 策略对比（1:1 切分）：\n- 1K：Vanilla mean 12.2s, Striped 11.9s (-2.5%), ZigZag 11.5s (-5.5%)\n- 4K：Vanilla 39.8s, Striped 39.8s, ZigZag 39.6s (<1% 差异)\n\n关键结论：\n1. 在正常 3B/1K 场景下，ZigZag 比 Vanilla 有约 5% 收益，但方差与收益同量级。\n2. 在 3B/4K 下，跨节点传输主导，策略差异消失。\n3. 分布式 3B GPU 仍慢于单节点 CPU：1K 12s vs 7.8s；4K 40s vs 29s。\n4. 7B bf16 无法在 pearl 的 16GB HIP 卡上加载，分布式 7B 需要量化支持。\n\n报告：reports/normal-workloads-3b-20260630-142629/

_updated: 2026-06-30 06:27:31_
### [2026-06-30] 单节点 vs 分布式：4096 token 时间分解

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `local CPU/MPS benchmark + white CUDA single-node benchmark`

用同样的 Qwen2-0.5B 类模型对 4095-token prompt + 5 token decode 进行单节点基准测试，并与 HCP 分布式环结果对比。\n\n结果：\n- white RTX 4090 单节点 CUDA：0.12s\n- 本地 Mac CPU：4.5s\n- 本地 Mac MPS：5.2s\n- HCP 2-domain vanilla 1:1（RTX 4090 CUDA + RX 9060 XT HIP）：~15.1s\n- HCP 2-domain 100 Mbps：~206s\n\n关键结论：\n1. GPU 单节点速度远超 CPU（0.12s vs 4.5s）。\n2. HCP 分布式在 4K token 下比单节点 CPU 还慢（15s vs 4.5s），因为跨节点 KV 传输占主导。\n3. 这不是 CPU/GPU 问题，而是“单节点本地内存” vs “多节点网络”的问题。\n4. HCP 的价值在于打破超长上下文下的内存墙，而不是在小长度下加速。\n\n报告：reports/single-node-vs-distributed/

_updated: 2026-06-30 05:33:11_
### [2026-06-30] 100 Mbps 重复实验稳定结果

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node bandwidth experiment on white/pearl`

在 white+pearl 上对 Qwen2-0.5B-1M / seq=4096 / max_tokens=5 进行带宽稳定性复测。\n\n方法：\n- 使用 tc tbf 在 enp10s0 / enp8s0 上限制为 100 Mbps。\n- 每次运行前彻底清理进程并等待端口释放。\n- 基线（无 tc）跑 3 次，100 Mbps 跑 5 次。\n\n结果：\n- 基线：17s, 18s, 17s；均值 17.3s。\n- 100 Mbps：204s, 205s, 217s, 203s, 203s；均值 206.4s（方差 <3%）。\n\n结论：\n1. 单次 100 Mbps 测出的 38s 和 604s 是偶发离群值，不是真实分布。\n2. 稳定状态下 100 Mbps 带来约 11.9×  slowdown。\n3. 这进一步支持 hyp-net-speed：跨节点带宽是 HCP 性能的决定性因素。\n\n报告：reports/bw-stability-20260630-132311/

_updated: 2026-06-30 05:23:34_
### [2026-06-30] 1:1 chunk split derivative comparison on white+pearl

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `manual cross-node run on white/pearl`

在 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP) 上运行 --chunk-sizes 2048,2048 的等分切分，比较 Vanilla / Striped / ZigZag。\n\n配置：Qwen2-0.5B-1M，seq_len=4096，max_tokens=5。\n\n结果（perf log 聚合，单位 ms）：\n- Vanilla：domain0 total=15122 (recv 14423, local 146), domain1 total=14516 (recv 12804, local 656)；瓶颈 15122 ms。\n- Striped：domain0 total=15547 (recv 14795, local 133), domain1 total=14722 (recv 12601, local 662)；瓶颈 15547 ms。\n- ZigZag：domain0 total=15331 (recv 14675, local 132), domain1 total=14640 (recv 12919, local 651)；瓶颈 15331 ms。\n\n关键发现：\n1. 1:1 等分消除了 3:1 容量感知切分的负载不均，但三种策略差异仍在 <6%。\n2. 网络 recv 仍占绝对主导，1:1 并未改善端到端瓶颈。\n3. ZigZag 的理论优势（负载均衡 + 减少边界）在当前 tailscale 链路上无法体现。\n\n报告：reports/ring-derivatives-1to1-20260630-122906/

_updated: 2026-06-30 04:41:51_
### [2026-06-30] Ring Attention derivatives Phase 2: real white+pearl comparison

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `manual cross-node run on white/pearl`

在 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP) 真实异构硬件上运行 Vanilla / Striped / ZigZag 三种调度策略。\n\n配置：Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，tailscale 网络。\n\n结果（perf log 聚合，单位 ms）：\n- Vanilla：domain0 total=15077 (recv 14477, local 133), domain1 total=14392 (recv 12663, local 648)；瓶颈 15077 ms。\n- Striped：domain0 total=14759 (recv 14140, local 119), domain1 total=13948 (recv 12256, local 652)；瓶颈 14759 ms。\n- ZigZag：domain0 total=15578 (recv 14906, local 129), domain1 total=14773 (recv 13040, local 656)；瓶颈 15578 ms。\n\n关键发现：\n1. 三种策略在真实异构硬件上全部跑通，无 NaN / crash。\n2. 网络 recv 占绝对主导（domain0 >95%，domain1 ~88%），调度策略对负载均衡的改善被网络带宽完全掩盖。\n3. 三种策略端到端差异 <6%，说明当前 tailscale 链路已经是瓶颈。\n4. Striped 改变了生成 token 序列（与 vanilla/zigzag 不同），这在无意义重复 prompt 的小模型上是可接受的位置敏感性表现。\n\n报告：reports/ring-derivatives-manual-20260630-112010/

_updated: 2026-06-30 03:23:34_
### [2026-06-29] Ring Attention derivatives Phase 1: CPU mock correctness and load balance

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `cargo test --features tch-backend test_ring_attention_derivatives_uneven_perf`

在 Rust 中新增 RingSchedulingStrategy（Vanilla / Striped / ZigZag）和 assignment helper，并在 CPU mock 上验证 2-domain 3:1 不均等分片（seq=4096, num_heads=8, head_dim=128）。\n\n结果（单次 layer）：\n- Vanilla：domain0=74ms, domain1=47ms，瓶颈 domain0。\n- Striped：domain0=149ms, domain1=50ms，把 peer compute 推给 domain0，反而更慢。\n- ZigZag：domain0=64ms, domain1=39ms，两个 domain 都变快，负载更均衡。\n\n所有策略 correctness diff < 3e-8。\n\n结论：\n1. ZigZag 在 uneven 3:1 分片下有效改善了负载均衡。\n2. Striped 在当前加权 round-robin 实现下对 3:1 场景不适用（与之前挂起结论一致）。\n3. 需要真实硬件（white CUDA + pearl HIP）验证这些趋势是否保持。

_updated: 2026-06-29 16:01:43_
### 综述类支撑线必须有真实实现和硬件对比才有说服力

type: `lesson` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `user-direction`

在论证 CXL/RDMA 必要性时，最初计划用 Ring Attention 家族综述作为辅助证据。用户指出这不够：如果只是文献综述，没有基于 HCP 的真实实现和 white/pearl 硬件对比，无法形成有工作量、有说服力的论证。\n\n教训：\n1. 任何“方案对比”类 claim，必须有可运行的代码和可重复的测量。\n2. 当直接实验（hyp-net-speed）已经很强时，不要为了“显得完整”而引入高成本实现线。\n3. 文献综述只能作为背景，不能替代实验证据。

_updated: 2026-06-29 15:48:58_
### [2026-06-29] white-pearl 完整带宽矩阵：100 Mbps 下 HCP 慢 10-30x

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `reports/bw-matrix-20260629-220317 / harness operations`

实验：white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP)，Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，tc tbf 在 192.168.100.x 有线链路上限速，iperf3 验证实际带宽。\n\n结果（2 reps）：\n- baseline 2.35 Gbps：20.5 s avg（20/21 s）\n- 1000 Mbps：29.5 s avg（28/31 s）→ 1.44x slowdown\n- 500 Mbps：50.0 s avg（50/50 s）→ 2.44x slowdown\n- 100 Mbps：445 s avg（206/684 s）→ 21.7x slowdown（中位数 445 s）\n\n报告目录：reports/bw-matrix-20260629-220317/\n\n关键发现：\n1. 端到端时间随带宽下降呈非线性增长；100 Mbps 时通信成为绝对瓶颈。\n2. 100 Mbps 两次重复差异极大（206 s vs 684 s），提示低速下系统状态（热节流、设备调度、QUIC 拥塞控制）可能放大波动。\n3. 500 Mbps 已使 4K+5 token 任务慢约 2.4x；1 Gbps 仍慢约 1.4x。\n\n结论：P2P KV ring 对跨节点带宽极度敏感；要释放异构 CP 的实用性，需要远高于千兆以太网的互联带宽（CXL / RDMA / 高速 NVLink）。

_updated: 2026-06-29 14:32:15_
### [2026-06-29] white-pearl 限速 pilot：100M 带宽下 HCP 慢 10x

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `harness/operations/ (pending full matrix record)`

实验：white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP)，Qwen2-0.5B-1M，seq_len=4096，max_tokens=5，使用 tc tbf 在 192.168.100.x 有线链路上限速。\n\n结果：\n- 基线 2.35Gbps：总耗时 21s\n- 限速 100Mbps：总耗时 206s\n\n结论：\n1. 网络带宽对 HCP 跨节点异构推理有决定性影响。\n2. 当带宽从 2.35G 降到 100M 时，端到端时间增加约 10 倍，说明当前 P2P KV ring 在低速网络下通信成为绝对瓶颈。\n3. 这为 CXL / 类 RDMA 高速互联的必要性提供了直接实验证据。\n\n下一步：完整矩阵（baseline / 1000M / 500M / 100M × 2 reps）正在后台运行。

_updated: 2026-06-29 14:02:37_
### [2026-06-29] white RTX 4090 CUDA 上 Striped 未改善负载均衡

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `harness/operations/20260629-104712-stripe-real-hardware.yaml`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture\n主机：white (Tailscale 100.118.253.68), RTX 4090, libtorch CUDA\n配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1)\n\nVanilla：\n- domain 0 total=131.1ms (local=130.3ms, peer=0.03ms)\n- domain 1 total=54.6ms (local=5.5ms, peer=49.0ms)\n\nStriped：\n- domain 0 total=164.8ms (local=114.0ms, peer=50.1ms)\n- domain 1 total=57.0ms (local=7.8ms, peer=49.1ms)\n\ncorrectness diff 均 < 1.3e-8。\n\n结论：在 white CUDA 单进程 3:1 场景下，Striped 使瓶颈 domain 0 总耗时增加约 26%，未改善 wall-time。

_updated: 2026-06-29 12:44:16_
### [2026-06-29] pearl RX 9060 XT HIP 上 Striped 未改善负载均衡

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `harness/operations/20260629-104712-stripe-real-hardware.yaml`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture\n主机：pearl (Tailscale 100.111.242.55), RX 9060 XT, libtorch HIP\n配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1)\n\nVanilla：\n- domain 0 total=158.2ms (local=157.5ms, peer=0.05ms)\n- domain 1 total=89.1ms (local=13.7ms, peer=74.9ms)\n\nStriped：\n- domain 0 total=224.8ms (local=154.2ms, peer=70.3ms)\n- domain 1 total=87.4ms (local=11.6ms, peer=75.6ms)\n\ncorrectness diff 均 < 1.3e-8。\n\n结论：在 pearl HIP 单进程 3:1 场景下，Striped 使瓶颈 domain 0 总耗时增加约 42%，未改善 wall-time；pearl 整体比 white 慢约 1.2-1.4x。

_updated: 2026-06-29 12:44:16_
### CPU mock 只能验证语法和逻辑依赖，不能指导 LLM 服务架构设计

type: `lesson` · status: `held` · confidence: 0.95 · importance: 0.9

在 Striped Attention 原型验证中发现：CPU 上 correctness diff 和 perf 数字对 LLM 服务架构设计的实际作用几乎没有意义。\n\n原因：\n1. CPU 与加速卡（CUDA/HIP/MPS）的算力结构、memory bandwidth、kernel launch 开销完全不同。\n2. CPU mock 无法反映真实 heterogeneous 场景下各 domain 的计算速度差异、显存压力、P2P / 网络传输瓶颈。\n3. Striped 对负载均衡的影响取决于"慢 domain 到底有多慢"以及"peer compute 转移是否能被快 domain 吸收"，这些信息 CPU 无法提供。\n\n结论：代码逻辑层面的正确性可以在 CPU 快速验证；任何关于调度策略、overlap、分片比例、端到端吞吐/延迟的设计决策，必须在真实加速卡硬件上复跑后才能得出结论。

_updated: 2026-06-29 12:35:36_
### [2026-06-29] Striped correctness原型在CPU mock上验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `cargo test / rust/src/model/attention/ring.rs`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture
配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1), CPU Float32, mock transport。

Correctness：
- Vanilla diff = 2.8e-8
- Striped diff = 2.6e-8
均 < 1e-4，数值正确。

Perf（单次 layer，CPU mock）：
Vanilla：
- domain 0 total=118.5ms (local=117.0ms, peer=0.02ms)
- domain 1 total=46.3ms (local=15.8ms, peer=30.0ms)
Striped：
- domain 0 total=184.6ms (local=129.6ms, peer=53.3ms)
- domain 1 total=50.8ms (local=15.9ms, peer=34.6ms)

关键发现：在 homogenous CPU 上，Striped 把部分 peer compute 从 domain 1 转移到 domain 0，使原本就是瓶颈的 domain 0 更慢；domain 0/1 总耗时比从约 2.6x 扩大到约 3.6x。

_updated: 2026-06-29 10:46:05_
### [2026-06-19] 1M context 本地异构分布式推理成功

type: `session` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `memory-bank/progress.md`

white RTX 4090 CUDA + pearl RX 9060 XT HIP，2.5G 有线直连。Qwen2-0.5B-1M，capacity-aware 3:1 分片（white 750K / pearl 250K）。Prefill 24/24 全通，decode 5 tokens 全通，exit=0。总耗时 ~2h 11m，white 显存峰值 23,999 MB。攻克：KV channel buffer 512、QUIC timeout 14400s、max_position_embeddings=1048576 patch、pearl 碎片化 OOM 通过 3:1 分片缓解。

_updated: 2026-06-29 05:34:19_
### [2026-06-17] 昇腾 910B NPU 控制面 E2E 打通

type: `session` · status: `closed` · confidence: 1.0 · importance: 0.75 · source: `memory-bank/progress.md`

单机 1× Ascend 910B4 (32 GB HBM) 上完成 Python vLLM worker ↔ Rust coordinator 控制面 E2E。Rust coordinator 脱离 libtorch feature 可编译运行，纯 Rust 采样替代 tch::Tensor。Coordinator 输出 generated: ! I'm。

_updated: 2026-06-29 05:34:19_
### 证据：1:1/2:1/3:2 split 均导致 pearl OOM

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/progress.md`

在 1M context 尝试中，均分 500K 及 2:1、3:2 split 均在 layer 23/24 因 pearl 16GB 显存分配失败而 OOM。只有 3:1 split 成功。

_updated: 2026-06-29 05:34:19_
### 证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

White CUDA loopback 双 domain 3B max_diff=0.406，0.5B max_diff=0.344，argmax=10/10。跨平台单节点 0.438，异构分布式 0.484。证明跨平台 BLAS 仅贡献 ~0.1 额外差异，不是 logits 差异主导因素。

_updated: 2026-06-29 05:34:19_
