# mindie-sim 0.9.18：Clean-Core 唯一机理主干、A2/A3/A5 物理配方统一与 `device-tool` 治理对齐

[English](sim-0.9.18.en.md)

`mindie-sim v0.9.18` 完成全栈 Clean-Core 机理化收敛与跨芯片（Ascend A2 / A3 双 Die / A5 `DAV_3510`）统一物理配方对齐：

- **唯一因果执行主干与禁用 `ResourceEnvelope` 兜底（Clean-Core Spine）**：单算子评估、子图、纯模型前向、4D 效率拆解、SLA 硬件缩放求解与 PD/拓扑扫描全部统一收敛至离散事件引擎 `runtime.Engine`（单次评估仅构造唯一 `EvaluationContext`）；彻底下线旁路公式器与 `ResourceEnvelope` 峰值除法兜底，未知算子机制显式抛出 `ModelingError`。
- **五类通用物理机理配方（Class A–E）与 A2/A3/A5 精确核归属（`PhysicalRecipe`）**：统一 `TiledFlashAndPagedAttentionRecipe`（Class A）、`TiledSparseIndexerRecipe`（Class B）、`ChunkedLinearAttentionRecipe`（Class C）、`AIVTiledVectorAndGatherRecipe`（Class D）与 `TopologyCollectiveRecipe`（Class E），以及 A2/A3 `paged_mla_arch22` / `w4a8` 与 A5 `DAV_3510`（`L0C = 256 KiB`、`l0c->ub` 直连、`K_8` UB 直连网）源码级配方，严格守恒 `mte2_aic` / `mte2_aiv` 分裂读管道字节与每 Die 唯一共享 `L2`（`buffer_units.l2 == 1`）。
- **37+ 模型目录独立守恒对账与增量魔法数字门禁**：通过 `scripts/verify_m1.py`（`hygiene` + `catalog-meta --require-models 37`）强制检验逻辑源参数、运行时预转换副本与有效/物理页对齐 KV 容量零误差对账，并启用 `scripts/check_no_magic_numbers.py` AST 增量字面量门禁。
- **跨代校准贡献总指南、文档归档与 `device-tool` 门禁对齐**：发布覆盖 A2 / A3 / A5 五域微基准与自定义 Attention 三级接入的 `CONTRIBUTING{.en}.md` 总指南，归档历史阶段性方案至 `docs/archive/`，并对齐 `--profile device-tool` PR 设计原则门禁。
