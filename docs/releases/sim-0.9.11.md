# mindie-sim 0.9.11

[English](sim-0.9.11.en.md)

本版本增强原生调用结构绑定与覆盖率核算契约、分段连续物理求解器机制守恒与独立原语评估接口，并随包交付 7 款多机集群模型的离线配置目录与编译期公式缓存优化：

- **原生调用结构绑定与覆盖率契约（`mindie_sim/prediction/native_binding.py`、`mindie_sim/prediction/native_coverage.py`、`mindie_sim/structure.py`）**：
  - 建立标准化原生调用绑定契约（`mindie-sim/native-call-binding/1`），将 `OpSpec` / `OperatorPlan` 算子边界与真实 ACLNN、Triton、CANN Kernel、`torch` 或 HCCL 入口对齐，携带类型化 ABI 签名摘要（`signature_hash`）与源码 `sha256` 溯源，严格校验空值字段、保持结构导出 `graph_id` 纯净并补齐 D0 混合注意力合法性判定。
  - 提供基于真实展开算子分母的覆盖率与 Profiler 对应关系核算（`native_call_coverage`、`coverage_from_records`、`join_profiler_rows`），严格区分 `bound`（已证实入口或算子核 + 源码摘要 + 类型化 ABI）与 `fully_paired`（入口与全部声明子核均具备正证据）。
  - 结构导出默认不附带 `native_calls` / `native_call_coverage`，需显式传入 `export_op_structure(..., native_evidence=True)`；默认导出保持与已发布 profiler 0.5.11 的 `graph_id` 校验兼容。
- **分段连续物理机制修正、独立原语评估与编译期缓存（`mindie_sim/prediction/piecewise/`）**：
  - 修正专家张量并行归约（`etp_allreduce`）边界语义：区分合并前专家局部归约（`local_experts`）与合并后词元归约（`tokens` / `moe_combine`），守恒真实通信字节量。
  - 统一 GQA / FlashAttention Decode 在不同 `kv_len` 与头组切分下的 Split-K 波次占用率单调性。
  - 新增规范化独立原语评估入口 `evaluate_primitive_spec`，复用完整子图编译链路的同源受控算子族编译器，严格守恒算子总耗时（`operator_ms`）与机制/核心聚合拆解。
  - 在分段算子编译期缓存静态公式文本，消除热路径重复字符串格式化开销。
- **离线集群模型目录随包分发与扩展架构下沉（`mindie_sim/catalog/fleet.py`、`mindie_sim/catalog/configs/*.json`、`mindie_sim/prediction/`）**：
  - 随 Wheel 与源码包内置 7 款多机集群模型离线配置与别名映射：`minimax-m3`、`minimax-m3.1-flash`、`gemma-4-26b-a4b`、`gemma-4-31b`、`qwen3.6-35b-a3b`、`qwen3.8-flash-next` 与 `step-5-preview`。
  - 完整接通 `GeGLU` 门控激活、稀疏 `GQA`（`index_topk`）注意力与索引器子图，以及 Gemma `K=V` 键值共享投影与 KV Cache 容量几何建模。

本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
