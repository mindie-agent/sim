# mindie-sim 0.9.14

[English](sim-0.9.14.en.md)

本版本面向昇腾 A5（`Ascend950DT_9582`）多机专家并行（如 4 机 × 8 卡 `EP32`）与新一代混合架构大模型推理，系统落地多级互联与 MoE 偏斜/负载均衡求解、部署命令解析与硬件物理极值（SOL）对照、四路守恒残差归因（`diff-trace`）、XPyD 多池分离容量与显存六分量账本、单算子可解释理论公式与通用算子流（`OpFlow`）评估，并合入可选 C11 原生事件引擎加速核：

- **A5（`Ascend950DT_9582`）`EP32` 多级通信、HCCL 展开模式与 MoE 偏斜/EPLB 建模（`mindie_sim/fabric_topology.py`、`mindie_sim/fabric_calibrated.py`、`mindie_sim/observations/a5_comm_calibration.py`、`mindie_sim/prediction/hccl_*.py`、`mindie_sim/moe_skew_eplb.py`、`mindie_sim/parallel_ep_knee.py`）**：
  - 支持双模通信织网定价：Mode A（`HierarchicalFabricTree`）按显式 `die -> package -> node -> superpod -> pod -> spine_roce` 物理拓扑树定价跨机架 RoCE 与非均匀通信域；Mode B（`CalibratedFabricProfile` / `comm-w32`）消费 4 机 × 8 卡 A5 UB-mesh/URMA 实测分段曲线，对未标定范围或非均匀 `AllToAllV` 接收端拥塞执行显式边界拒止而非静默外推。
  - 精细区分 `HCCL_OP_EXPANSION_MODE` 下 `AIV`、`AICPU` 与 `CCU_SCHED`（含双 Die 协同与 Z 轴绕行开销）的通信展开机制，支持双微批（dual-microbatch）计算/通信重叠与 MC2 通算融合链路评估。
  - 新增 `moe_skew_eplb` 与 `parallel_ep_knee`：基于有限多项分布与 Dirichlet 专家热度浓度计算最慢 EP Rank 离散词元负载（`max_rank_tokens`），求解整数最小最大热专家副本放置与周期重平衡净收益（`eplb_effect`），并在显式显存与路由资格约束下扫描离散 EP 拐点（`sweep_ep_knee` / `settle_optima`）。
- **启动命令解析、硬件物理极值（SOL）对照与部署寻优（`mindie_sim/launch_*.py`、`mindie_sim/feature_cli.py`、`mindie_sim/extended_cli.py`、`mindie_sim/advise.py`、`mindie_sim/sweep.py`）**：
  - 新增 `mindie-sim inspect-launch`、`mindie-sim sol` 与 `mindie-sim feature-deltas` 子命令及 `parse_launch` / `compare_launch` API，直接解析 Shell、argv、JSON 或容器 YAML 启动配置，校验并行/特性组合合法性并输出相对硬件物理极值（Speed-of-Light）的差距分解与特性反事实增量。
  - 增强 `compose`（新增 `--recover-ssm` 开关）、`advise`、`sweep` 与 `skew` 子命令，并直接暴露 `optimize-graph`、`sweep-ep-knee`、`size-xpyd`、`memory-breakdown`、`kv-state`、`opflow` 与 `op-formula` 七个一级 CLI 子命令，使十支柱引擎能力均可通过 `mindie-sim <subcommand>` 直接调用。
- **实测残差四路守恒归因（`mindie-sim diff-trace`、`mindie_sim/diff_trace.py`）**：
  - 新增 `diff-trace` 子命令与 `attribute_diff` / `effective_operator_rows` 接口（`mindie-diff-trace/2`），通过累积反事实回放将预测与实测单步耗时残差严格守恒分解为单核机制、分块/Padding 膨胀、下发/图模式开销与多流重叠差异四项，自动校验固定算子 ABI 目录中的有效词元行数并生成可直接执行的 `mindie-profiler repro-op` 复现命令。
- **XPyD 分离池求解器、六分量显存账本与通用 7 族 `RecoverSSM` / 投机草稿稀疏状态物理（`mindie_sim/xpyd_*.py`、`mindie_sim/memory_breakdown.py`、`mindie_sim/hybrid_kv_state_*.py`、`mindie_sim/dspark_*.py`、`mindie_sim/linear_state_*.py`）**：
  - 新增 `solve_xpyd_pools`（`mindie-sim size-xpyd`）：在总卡数预算与严格利用率上限下联合裁剪 P-HBM、P-NIC、RoCE、D-NIC、D-HBM、KV 传输线速与暂存池容量，最大化满足 TTFT/TPOT SLO 的有效完成请求吞吐（Goodput）。
  - 新增 `MemoryLedger` / `analyze_memory_breakdown`（`mindie-sim memory-breakdown`）：按每 Rank 精确拆解分片权重、辅助权重（MTP/EPLB）、分页 KV 与稀疏/递推状态、通信域持久 Arena、ACLGraph 捕获池与运行时 Workspace 六大常驻 HBM 分量。
  - 泛化 7 类递推/线性注意力家族（`kda`、`gdn`、`mamba`、`gla`、`deltanet`、`lightning`、`retnet`）的 `RecoverSSM` 状态块/页分配与投机拒绝重放开销建模（`mindie-sim kv-state` / `compose --recover-ssm`），并支持投机草稿层（`dspark` / `mtp` / `eagle`）自动继承主模型稀疏 KV、滑动窗口（SWA）与 KV 压缩 `LayerPlan` 几何参数。
- **单算子理论公式引擎、通用算子流（`OpFlow`）、Day-0 图优化器、可选 C11 加速核与耦合资源仲裁加速（`mindie_sim/op_formulas/`、`mindie_sim/opflow/`、`mindie_sim/prediction/graph_optimizer.py`、`mindie_sim/prediction/weight_prefetch.py`、`mindie_sim/prediction/runtime/`）**：
  - 新增 `mindie_sim.op_formulas`（`mindie-sim op-formula`，覆盖 18+ 算子族代数公式卡、硬物理下界 `t_floor` 与参考下界 `t_ref`）与 `mindie_sim.opflow`（`mindie-sim opflow`，覆盖 UB/L1/L0 分块、Fixpipe 路径、Triton 降级门禁及 `ACLGraph` / `SuperKernel` 启动下界有序建模）。
  - 新增确定性 Day-0 图融合与多流调度优化器（`run_optimizer` / `mindie-sim optimize-graph`）、权重预取掩盖建模（`weight_prefetch`）以及 `mindie-sim build-accel` 可选 C11 原生事件引擎加速核（Windows 编译子进程默认携带 `CREATE_NO_WINDOW`），优化大规模耦合多单元资源仲裁与全量验证执行效率，并保留 `v0.9.11` 全部 7 款多机集群离线模型目录、`native-call-binding/1` 与 `evaluate_primitive_spec` 契约。
- **多代际原生特性扩展、自定义芯片派生与四时钟暴露耗时守恒分解（Issue `#34`、`mindie_sim/hardware_derivation.py`、`mindie_sim/prediction/exposed_breakdown.py`、`mindie_sim/prediction/local_tp_*.py`、`mindie_sim/catalog/expert_slots.py`）**：
  - 修复并原生建模 `collective_startup_mode`（`legacy` / `ub_mesh` 下 `AllToAll` 对数轮次启动开销）及 `matrix_compute_policy`（`auto` / `force_bf16`，自动识别 `MXA8W4` 在原生 FP8 Cube 硬件上的峰值映射并校验 `compute_peak_aliases` 闭环）。
  - 支持节点内局部并行宽度覆盖（`o_proj_tp`、`linear_proj_tp`、`lm_head_tp`）、跨请求共享前缀缓存池与单步批次前缀读取去重（`prefix_cache_mode` / `prefix_read_mode`）、EPLB 冗余专家槽位权重驻留（`redundant_experts_per_rank`）、基于 AIC/AIV 核数与 HBM/MTE2 参数派生自定义 SKU（`derive_hardware_sku` / `--hardware-derivation`）以及同构副本/语义图复用（`homogeneous_graph_reuse`）。
  - 在 `efficiency`、`sla`、`sweep` 与单步报告中输出基于 4 时钟热带（max-plus）关键路径自动机的 `exposed_breakdown_pct`（`cube` / `vector` / `memory` / `communication` 四分量严格非负且在活跃步守恒求和为 `100%`）。
- **Lean 4 形式化验证内核与差分不变量门禁（`formal/MindIESimSpec/`、`tests/test_formal_lean_invariants.py`）**：
  - 新增零外部定理库依赖的 Lean 4（`v4.16.0`）机器检验规范库 `formal/MindIESimSpec`，围绕四时钟 max-plus 守恒自动机（`FourClock`）、权重预取掩盖与六分量显存账本（`PrefetchAndLedger`）、非整除分片局部下界与 NZ 对齐（`ShardingAndLayout`）、7 族 `RecoverSSM` 阶梯页单调性（`RecoverSSM`）及 `UB-Mesh` 对数通信轮次（`AllToAllRounds`）完成 55 条核心定理证明（`0` `sorry` / `admit`），并配套随机 DAG 差分性质测试与路径过滤 CI 门禁。

本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
