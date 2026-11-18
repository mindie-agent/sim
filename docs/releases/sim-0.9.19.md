# mindie-sim 0.9.19：Issues `#49`–`#60`、`#62`、`#63` 全量闭环、CANN 9.2.1 双硅基物理标定与跨仓 SSOT 统一

[English](sim-0.9.19.en.md)

`mindie-sim v0.9.19` 在唯一因果执行引擎上完成全部 14 个问题（`#49`–`#60`、`#62`、`#63`）闭环，端到端打穿 CANN 9.2.1 生产算子图与 `Ascend950PR` + `Ascend950DT` 双硅基物理标定，并以零静默回退与高风险接缝深度审计收敛 41 模型保真边界及跨仓 SSOT 契约：

## 1. Issues `#49`–`#57` 基础与架构闭环

- **Engram 主机侧暂存与两阶段流式拓扑扫描**：Engram 权重仅保留主机暂存边界，拓扑扫描按两阶段流式求解，避免整网驻留与重复构造。
- **压缩共享前缀索引器与解析式在线 UB TopK**：共享前缀 indexer 复用压缩索引，TopK 以解析式在线 UB 规划，保持候选平面与分块读取守恒。
- **mHC FP32 / 理想边界融合**：mHC 融合严格区分 FP32 边界与理想融合边界，保留 Sinkhorn 前置系数因果依赖，不跨边界借用私有 SRAM 或空闲 sibling core。
- **MC2 投机容量与 pack 量化器吸收**：MC2 投机容量继承真实 proposal 行；pack 量化器吸收进路由打包，不重复计价。
- **纯 SWA MQSMLA 与 128-head TP 支持**：纯滑动窗口 MQSMLA 与 128-head TP 切分保持 query/KV 几何一致。
- **`gathered_dcp` latent/draft KV 与 `cp>1` index-key 窗口**：`gathered_dcp` 统一 latent/draft KV 的 gather 边界，并修正 `cp>1` 时 index-key 窗口。
- **`kda_recurrent` 64-head 状态坍缩与 `recover_ssm_commit` 保留**：64-head 状态按真实 group 坍缩；投机解码保留 acceptance 之后的 `recover_ssm_commit`，普通探针不产生提交。
- **C8 稀疏 MLA 批处理与 `GLM-5.3-Flash` `W8A8_MXFP8`**：稀疏 MLA 按共享前缀批处理外部读取，`GLM-5.3-Flash` 全链路采用 `W8A8_MXFP8`。
- **A5 可持续 DRAM MBU 与 `a5_pod` `per_port` 并发协议**：A5 可持续 DRAM 带宽按真实 service mode 闭环；`a5_pod` 及 16 卡宽节点使用 `per_port` 节点内链路，并支持 `alltoall` / `alltoallv` 并发协议。

## 2. Issues `#58`–`#60`、`#62`、`#63`：CANN 9.2.1 算子图对齐与 `Ascend950PR` + `Ascend950DT` 双硅基标定

- **CANN 9.2.1 生产算子图结构泛化绑定（`#58`–`#60`、`#63`）**：全部按模型层结构、张量量化属性与硬件微架构泛化触发（零 `model_id` 硬编码）：
  - `aclnnMlaPrologV3WeightNz` / `aclnnMlaPrologV4WeightNz`、`aclnnMixedQuantSparseFlashMla`、`aclnnFFNWorkerBatching` / `aclnnTokenDispatch` 与 MXFP8 `engram_gate`；
  - 656B C8 紧凑 MLA KV 行、BF16 `KeyPool` 环形投影/权重驻留、LTP8 `o_proj_local` AllReduce 与 `MoeInitRoutingV3 + SwigluMxQuant`；
  - `QuantBatchMatmulV3` FP8、`LightningIndexer + QuantLightningIndexerGradKlLoss`、融合 `KDA` 与 `CausalConv1d UPDATE`。
- **`Ascend950PR_958b (1.4 TB/s)` + `Ascend950DT (4.0 TB/s)` 成对硅基标定（`#62`）**：新增 `Ascend950PR_958b` (`112 GiB`) 硬件规格，全部采用多阶段 `PipelineProgram` / `PlannedRecipe` 与声明有效 L2 容量推导 C8 SFA 稀疏 gather、`aclnnMixedQuantSparseFlashMla`、`KeyPool` RowShard、`arch35` QLI V2、`CausalConv1d UPDATE`、L2 推导 `MXFP4`/`MXFP8` GMM、`npu_swiglu_mx_quant`、窄 `N` Split-K BF16 Router、`aclnnMoeGatingTopK` 与单调 NZ Block-32 Prologue，零 `ResourceEnvelope`、100% 守恒 `plan.work`。

## 3. Zero-Silent-Fallback、`mindie-profiler` SSOT 统一与深度审计加固

- **零静默回退与 41 模型保真**：矩阵路径默认使用 `SteadyMatmul`；支持 `compressed-tensors` `W4A16`、无缓存双向 `gqa` FlashAttention、精确 `partial_rotary`/`swigluoai`/`noaux_tc`；所有计时入口对未建模结构统一抛出携带 `mindie-unmodeled-structure-hint/1` 的 `UnmodeledCapabilityError`。
- **跨仓 SSOT 单源统一**：`mindie_sim.observed_contracts` 与 `mindie_sim.subgraph_math` 统一服务 `mindie-sim` 与 `mindie-profiler v0.5.17+` 的观测合约与闭式子图算力/访存计算。
- **深度审计收口**：对齐 C11 `accel_*` 与 Python `Engine` 混合锚定耦合速率归一、首 tile 前驱跳过与 `cohort_size` 传递；消除 128 卡队列分析重复构图与 MC2 强制全图 DES 开销，完成全链路访存/算力守恒复核。
