# mindie-sim 0.9.6

本版本发布 **算子—子图—模型—服务四级（L1–L4）分段连续物理求解引擎全参数泛化联动与零样本单步误差 `<= 10%` 严格收敛（`21/21` 评估项 `100.0%` 达标，`above_10pct == 0`）**：

- **全硬件参数联动的物理反事实缩放与机制建模（`mindie_sim/prediction/piecewise/{families,families_matrix,model_pipeline}.py`、`mindie_sim/schema/hardware.py`）**：
  - 消除算子与子图公式中的平台常量截断，所有算子机制（Dense GEMM、MoE Grouped GEMM、MLA BMM、Attention、GDN/KDA、KV Cache、Norm、Activation、Elementwise/Quant、Routing、Collective、Transport、AICPU Device Control）严格联动 `HardwareSpec` 物理参数（Cube/Vector 算力、HBM/UB/L1 带宽、HCCS/SIO/RoCE 互联带宽与延迟、核心数及发射开销）。
  - 新增反事实硬件参数敏感度门禁（`tests/test_piecewise_counterfactual_generalization.py`），确保单调调整算力、HBM 带宽、互联带宽与核心数时，L1–L4 各层预测呈现物理一致的单调响应。
- **四级（L1–L4）零样本单步分段连续验证 `100.0% <= 10%` 收敛（配套 `mindie-profiler 0.5.8` 与 `maoxx241/mindie-sim-profiling-bench`）**：
  - 在 **731 个实测 Kernel Trace Cell（`731/731` 零跳过、零失败）、40,921 个单步样本**上，全部 **21 个 L1–L4 评估目标均达成 `100.0%` 单步误差 `<= 10%`**（`above_10pct == 0`，最大单步误差 `step_err_max <= 9.97%`，无正负抵消绝对误差中位数 `uncancelled_p50 <= 4.79%`，`forbidden_lookup_imported == False`）。
