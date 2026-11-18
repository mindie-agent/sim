# mindie-sim 0.9.19：Issues `#49`–`#57` 全量闭环、Zero-Silent-Fallback 41 模型保真与 `mindie-sim` / `mindie-profiler` SSOT 统一

[English](sim-0.9.19.en.md)

`mindie-sim v0.9.19` 在唯一因果执行引擎上完成 Issues `#49`–`#57` 全量闭环，并以 Zero-Silent-Fallback 为硬约束收敛 41 模型保真边界与跨仓 SSOT 契约：

## 1. Issues `#49`–`#57` 全量闭环

- **Engram 主机侧暂存与两阶段流式拓扑扫描**：Engram 权重仅保留主机暂存边界，拓扑扫描按两阶段流式求解，避免整网驻留与重复构造。
- **压缩共享前缀索引器与解析式在线 UB TopK**：共享前缀 indexer 复用压缩索引，TopK 以解析式在线 UB 规划，保持候选平面与分块读取守恒。
- **mHC FP32 / 理想边界融合**：mHC 融合严格区分 FP32 边界与理想融合边界，不跨边界借用私有 SRAM 或空闲 sibling core。
- **MC2 投机容量与 pack 量化器吸收**：MC2 投机容量继承真实 proposal 行；pack 量化器吸收进路由打包，不重复计价。
- **纯 SWA MQSMLA 与 128-head TP 支持**：纯滑动窗口 MQSMLA 与 128-head TP 切分保持 query/KV 几何一致。
- **`gathered_dcp` latent/draft KV 与 `cp>1` index-key 窗口**：`gathered_dcp` 统一 latent/draft KV 的 gather 边界，并修正 `cp>1` 时 index-key 窗口。
- **`kda_recurrent` 64-head 状态坍缩与 `recover_ssm_commit` 保留**：64-head 状态按真实 group 坍缩；投机解码保留 acceptance 之后的 `recover_ssm_commit`，普通探针不产生提交。
- **C8 稀疏 MLA 批处理与 `GLM-5.3-Flash` `W8A8_MXFP8`**：稀疏 MLA 按共享前缀批处理外部读取，`GLM-5.3-Flash` 全链路采用 `W8A8_MXFP8`。
- **A5 可持续 DRAM MBU `83.16%`**：A5 可持续 DRAM 带宽按真实 service mode 与 MBU 口径闭环。
- **`a5_pod` `per_port` 节点内链路与并发 `alltoall`/`alltoallv`**：`a5_pod` 使用 `per_port` 节点内链路，并支持 `alltoall` / `alltoallv` 并发协议。

## 2. Zero-Silent-Fallback 与 41 模型保真

- **`SteadyMatmul` 默认配方**：矩阵路径默认使用 `SteadyMatmul`，不再按测量存在性或模型名切换旁路。
- **`compressed-tensors` `W4A16`**：`kimi-k2-thinking` / `kimi-k2.5` 使用 INT4 group-quantized 权重与 BF16 计算，权重/激活格式显式建模。
- **无缓存双向 `gqa` FlashAttention**：`hunyuan-video-dit` 使用 cacheless 双向 `gqa` FlashAttention。
- **精确 `partial_rotary`、`swigluoai`、`noaux_tc`**：旋转位置仅作用于声明维度；`swigluoai` 与 `noaux_tc`（sigmoid + correction bias）按真实公式展开。
- **显式 `UnmodeledCapabilityError`**：所有 timing 入口在遇到未建模结构时抛出 `UnmodeledCapabilityError`，并携带 `mindie-unmodeled-structure-hint/1` 结构化提示，不再静默回退。

## 3. `mindie-sim` / `mindie-profiler` SSOT 统一与 Track 1/2 对齐

- **共享观测契约**：`mindie_sim.observed_contracts` 统一 profiler 与 sim 的观测边界、单位、来源与实现。
- **闭式子图数学**：`mindie_sim.subgraph_math` 以闭式表达 Cube FLOPs、Vector ops 与最小边界 HBM bytes，Track 1/2 复用同一公式。
- **组合与 SKU 对账**：Track 1/2 的组合并行、SKU 效率与 SLA 求解共享同一因果账本；投机分析几何在 `dp>1` cohort 中按 `q_len + draft_proposal_tokens` 计入验证行，并对低于验证宽度的探针保持普通目标行。
