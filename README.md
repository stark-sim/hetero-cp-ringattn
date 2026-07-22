# hetero-cp-ringattn

HCP(Heterogeneous Context Parallelism)研究主仓:异构节点(不同显卡/显存)合作推理的
分布式调度框架、实验基础设施与知识库(graph-memory)。

## 仓库地图

本项目的两个产品级产出已解耦为独立仓库(private):

| 仓库 | 内容 | 生命周期 |
|------|------|----------|
| [stark-sim/hcp-vllm-plugin](https://github.com/stark-sim/hcp-vllm-plugin) | HCP 的 vLLM 生态插件:ring-attention CUSTOM backend + HcpRingKvConnector(显存切分 CP,KV 瞬时 staging)+ 自研 Triton kernel(LSE 输出,CUDA/ROCm 同一实现) | 跟随功能路线(N>2 ring、kernel hardening) |
| [stark-sim/vllm-rocm-gfx1200](https://github.com/stark-sim/vllm-rocm-gfx1200) | RX 9060 XT(gfx1200, RDNA4)适配 vLLM 的补丁与构建/运行脚本 | 跟随 ROCm/vLLM 版本 |

本仓保留:Rust/Python 分布式调度核心、transformers 线 worker、跨节点驱动脚本
(`scripts/run_cross_node_*.sh`,通过 `*_PLUGIN_REPO` 环境变量指向插件 clone,
默认 `/home/stark/hcp-vllm-plugin`)、graph-memory 知识库、docs/reports。

## 机器上的标准布局

```
/home/stark/hetero-cp-ringattn   # 本仓(研究/驱动)
/home/stark/hcp-vllm-plugin      # 插件 clone(pip install -e 从这里装)
```
