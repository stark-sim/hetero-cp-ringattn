# History and Lessons

## 目标

这个仓库虽然已经从 `honolulu` 中抽离，但它不是凭空出现的。

它继承的是一条逐步收敛的认知链：

- `phase1` 证明最早的序列维切分 + 跨机回传是可做的
- `phase2` 积累了 transport、report、correctness 记录方式
- `phase3` 积累了远端部署、路径解析、解释器固定、报告组织等真实运维经验

这个文档的目的，是说明：

- HCP 应该从历史资产中带走什么
- 又应该明确拒绝什么

## 来自 phase1 的经验

`phase1_poc_tcp/` 是最早的历史 HCP-like 尝试。

它验证过的核心问题是：

- 输入张量可以沿 `seq` 维切分
- 不同设备可以分别处理各自分片
- 跨机 bridge 可以把结果回传并汇总
- 汇总结果可以和单机 baseline 对比

对 HCP 来说，phase1 最值得继承的不是代码形态，而是三个思路：

- 先做最小可见的跨机闭环
- 始终保留单机 baseline 对照
- correctness 报告要优先于性能表演

## 来自 phase2 的经验

`phase2_native/` 的很多接口本身不适合作为 HCP 的长期边界，但它沉淀了两个很有价值的东西。

第一是 transport 视角：

- message 怎么走
- bytes / latency / failures 怎么记
- 失败时怎么把 transport 和 compute 拆开定位

第二是 report schema 视角：

- correctness metrics
- bridge / transport metrics
- performance metrics
- failure info

HCP 不需要把 phase2 整套 runtime 搬过来，但应该把这种“实验必须能被结构化记录”的习惯带过来。

## 来自 phase3 的经验

`phase3_layerwise/` 主要是 HLPP 主线，但它在真实双机运行里已经踩过很多远端部署问题。HCP 如果走向 remote heterogeneous smoke，这些经验不应该再重踩一次。

最值得直接继承的是：

- `path.root`
  - 相对路径必须由各 domain 在本机解释
- 解释器 pinning
  - 不能依赖调用者当前 shell 的 `python3`
- clean build 入口
  - build 环境也会被 `CONDA_*` / `PYTHONPATH` 污染
- report layout
  - 运行产物要固定落盘位置，方便对照与回溯

换句话说，phase3 给 HCP 的最大价值之一不是 layer-wise 逻辑，而是“远端系统真正怎么活下来”的经验。

## HCP 应该带走什么

建议明确迁移到 standalone repo 的内容：

- `phase1`
  - baseline compare 的验证习惯
  - 最小 controller / worker 可见性
- `phase2`
  - report schema 思路
  - transport metrics / failure 分类
- `phase3`
  - `path.root`
  - interpreter pinning
  - clean build
  - report directory discipline

## HCP 不该带走什么

同样重要的是拒绝历史包袱。

HCP 不应该继续带入：

- `HLPP` 的 layer plan / high-boundary 叙事
- `phase2` 的 page / KV oriented 长期接口
- 对 `honolulu` 主仓的源码级依赖

独立仓要复制的是：

- 原则
- 经验
- schema
- 验证方法

而不是把整个历史结构原样搬家。

## 当前建议

如果把 HCP 当成一个真正独立的仓库来建设，它应该继续遵守下面的迁移原则：

1. 文档优先迁移问题定义和经验总结
2. 代码只迁移最小必要公共类型和最小实验骨架
3. 每增加一个新组件，都要说明它是在继承哪段历史经验，而不是重新偶然长出来
