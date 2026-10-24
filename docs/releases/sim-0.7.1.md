# mindie-sim 0.7.1：实测算子画像计划、完成时间下界与声明工作量代理

[English](https://github.com/mindie-agent/sim/blob/v0.7.1/docs/releases/sim-0.7.1.en.md)

`mindie-sim 0.7.1` 新增 `ProfiledKernelProvider`（`backend="profiled_kernel"`），支持按显式配置的算子绑定（`bindings`）只读复用 `mindie-profiler` SQLite 资产库中的观测记录来生成算子执行计划（`KernelPlan`），将捕获的核函数耗时中位数作为单任务完成时间下界，同时保留算子声明的 Cube/Vector/HBM/MTE 工作量代理。

---

## 一、核心特性与行为变更

### 1. 显式 `bindings` 映射与严格 `profiled_device_kernel` 范围校验（`mindie_sim/profiled_kernel.py`，PR `#16`）

- `ProfiledKernelProvider` 接收配置中的 `db`、`environment_key` 与 `bindings` 列表，并以只读模式（`mode=ro`，`busy_timeout=5000`，带 1024 条有界内存缓存）按 `variant_id` 查询 `mindie-profiler` SQLite 数据库的 `variants` 与 `costs` 表。
- 算子匹配遵循严格的显式契约，不做跨 Shape、SKU、Recipe 或软件环境的通配：
  - 算子的 `(op.family, op.recipe, op.phase, op.launch_shape, op.dtype, op.layout)`（以及可选的 `op.metadata["profile_variant"]`）必须在配置的 `bindings` 中唯一命中；
  - 命中的 `binding["timing_scope"]` 必须精确等于 `"profiled_device_kernel"`（`CAPTURED_TIMING_SCOPE`）；
  - 数据库观测记录必须 `eligible=True`，且其 `hardware_key` 与 `environment_key` 与当前目标机器及配置环境完全一致。

### 2. 捕获耗时中位数下界与声明工作量代理（`mindie_sim/profiled_kernel.py`、`operator_plan.py`，PR `#16`）

- 捕获的 Device 核函数耗时中位数（`median_ns`）不包含 Cube、Vector、HBM、MTE2、MTE3 各子单元的独立硬件 occupancy。为防止同阶段并发算子无代价重叠，`ProfiledKernelProvider` 生成包含 `ExecutionStage("captured_service", demands=demands, minimum_duration_ms=observation["median_ns"] / 1e6)` 的 `KernelPlan`：
  - `minimum_duration_ms` 作为父阶段的单任务完成时间下界，保证算子服务耗时不低于实测中位数；
  - 算子自身声明的 `flops`、`vector_ops`、`hbm_bytes`、`mte2_bytes`、`mte3_bytes` 映射为共享阶段资源需求（`cube`/`vector`/`hbm`/`mte2`/`mte3`）；未声明任何工作量的算子会被拒绝（`profile_occupancy_undeclared_for_shared_stage`）；
  - 溯源信息（`provenance`）显式标记 `source_class="observed"`、`timing_scope="profiled_device_kernel"`、`resource_occupancy="not_captured"`、`occupancy_basis="declared_work_proxy"` 与 `independent_accuracy_acceptance=False`。

### 3. 成本模型优先级与已有观测复用（`mindie_sim/a3_cost.py`、`a3_measurement.py`，PR `#16`）

- **校准与通信优先级语义**：在 `PredictiveCostModel.cost()` 中，若算子已命中专用的实测校准或通信拓扑模型，系统保持原有校准/通信模型的定价优先级；若该算子同时携带 `op.metadata["profile_variant"]`，会在保留记录中显式标注 `superseded_profile_variant` 而非报错。仅当 `profile_variant` 与非执行节点、既有显式阶段计划或串行代价等自相矛盾的声明冲突（`PROFILE_VARIANT_CONTRADICTIONS`），或所绑定的 `profile_variant` 缺失/被解析后端静默替代（`analytic_backend_substituted`）时，才抛出 `A3InputError`。
- **复用已有观测跳过采样请求**：由 `profiled_kernel` 实测计划定价的算子标记为 `calibration.status = "profiled_kernel_observed"`，`a3_measurement.py` 的 `record_prediction_misses()` 将其与 `"calibrated_exact"`、`"calibrated_interpolated"` 一并视为已有实测依据，不再重复生成校准采样请求。

---

## 二、验证结果与证据边界

- **定向测试结果**：覆盖 `ProfiledKernelProvider` 绑定查询、完成时间下界、声明工作量保留、优先级覆盖/冲突处理与采样请求跳过的 **`22 passed`** 定向测试全部通过。
- **证据边界说明**：`profiled_kernel` 提供单算子核函数耗时中位数下界与声明工作量代理（`independent_accuracy_acceptance=False`），不代表实测硬件子管道 occupancy，亦不替代整网端到端精度或性能验收。
