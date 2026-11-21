# Ascend A5（`Ascend950DT` / `Ascend950PR` / `DAV_3510`）硅实测校准与增量开发子指南

[返回总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md) · [Ascend A2 子指南](ascend-a2.md) · [Ascend A3 子指南](ascend-a3.md)

> [!IMPORTANT]
> 本子页面规定 **Ascend A5（微架构 `DAV_3510` / 能力预设 `npu_arch_3510`）** 及 **Atlas 950 SuperPoD** 的专属硬件几何、原生 FP8 与片上直连路径、`K_8` `per_pair` UB 网格扩展定律及增量校准靶点。通用的单一主干铁律、五域微基准 `StreamBlocker` 协议、自定义 Attention 三级接入（Tiers 1–3）与 PR 六道门禁请先阅读 [总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md)。

---

## 1. 已注册 SKU 标称几何基线与硬件别名

在 [`mindie_sim/compose.py`](../../mindie_sim/compose.py) 与 [`mindie_sim/compose_inputs.py`](../../mindie_sim/compose_inputs.py) 中，A5 硬件与拓扑别名精确解析如下（**严禁构造或引用已废弃的虚构 `9599` SKU**）：
- `a5` / `950` -> `Ascend950DT_9582`
- `950pr_958b` / `ascend950pr_958b` -> `Ascend950PR_958b`
- 商用 `Ascend950DT` 规格派生别名（均保持 `4.0 TB/s` HBM 与 `buffer_units.l2 = 1` 不变量，通过 `derive_hardware_sku` 动态派生 AIC/AIV 核数与显存容量）：`425-96` / `Ascend950DT_425_96`（`28 AIC / 56 AIV, 96 GiB`）、`486-96` / `Ascend950DT_486_96`（`32 AIC / 64 AIV, 96 GiB`）、`547-96` / `Ascend950DT_547_96`（`36 AIC / 72 AIV, 96 GiB`）、`547-144` / `Ascend950DT_547_144`（`36 AIC / 72 AIV, 144 GiB`）
- `a5-pod` / `atlas-950-superpod` / `a5-pod-dt` -> `("Ascend950DT_9581", "a5_pod")`（`ranks_per_node = 64`, `superpod_size = 8192`）
- `a5-server-dt` -> `("Ascend950DT_9581", "a5_server")`（`ranks_per_node = 8`）
- `a5-pod-pr` -> `("Ascend950PR_9579", "a5_pod")`；`a5-server-pr` -> `("Ascend950PR_9579", "a5_server")`；`a5-server-pr-pcie-16p` -> `("Ascend950PR_9579", "a5_pcie_16p")`

[`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 中注册的 4 款 A5 SKU 几何基线如下表所示：

| SKU 键名 | 封装形态 | AIC / AIV | 标称主频 | HBM 容量 / 标称带宽 | 共享 L2 容量（`buffer_units.l2 = 1`） | BF16 / FP8 & INT8 标称峰值 | 冷 HBM 旋转环下限（`>= 3 × L2`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ascend950DT_9581` | 单 Die (`1`) | `32 / 64` | `1650 MHz` | `144 GiB` / `4000 GB/s` | `128 MiB` (`134,217,728 B`) | `432.54` / `865.08` TFLOPS | **`>= 384 MiB`** |
| `Ascend950DT_9582` | 单 Die (`1`) | `32 / 64` | `1650 MHz` | `96 GiB` / `4000 GB/s` | `128 MiB` (`134,217,728 B`) | `432.54` / `865.08` TFLOPS | **`>= 384 MiB`** |
| `Ascend950PR_9579` | 单 Die (`1`) | `32 / 64` | `1650 MHz` | `64 GiB` / `1600 GB/s` | `128 MiB` (`134,217,728 B`) | `432.54` / `865.08` TFLOPS | **`>= 384 MiB`** |
| `Ascend950PR_958b` | 单 Die (`1`) | `32 / 64` | `1650 MHz` | `112 GiB` / `1400 GB/s` | `128 MiB` (`134,217,728 B`) | `432.54` / `865.08` TFLOPS | **`>= 384 MiB`** |

---

## 2. `DAV_3510`（`npu_arch_3510`）微架构不变量与建模关键差异

1. **扩容的片上私有缓存（`L0C = 256 KiB` 与 `UB = 248 KiB`）**：
   - 每个 AIC 私有：`L1 = 512 KiB`（`524,288 B`），`L0A = 64 KiB`（`65,536 B`），`L0B = 64 KiB`（`65,536 B`），**`L0C = 256 KiB`（`262,144 B`）**——支持完整的 **`[256, 256]` FP32 累加矩阵块**（双缓冲输入槽位 + 单累加槽位刚好占满 `262,144 B`）；
   - 每个 AIV 私有：**`UB = 248 KiB`（`253,952 B`）**，单周期向量执行宽度翻倍至 `256 B/cycle/AIV`（`64` FP32 lane-ops/cycle，双发射寄存器管道 `pipe_v_lanes`、`pipe_v_issue`、`ub_read_bytes`、`ub_write_bytes`、`pipe_s_issue`）。
2. **原生 FP8 Cube 与精度规则（[`compute_arch.py`](../../mindie_sim/schema/compute_arch.py)）**：
   - `npu_arch_3510` 声明 `fp8=True, int4_operand=False`：原生支持 `fp8_e4m3`、`fp8_e5m2`（`865.0752 TFLOPS`）、`int8`、`bf16`、`fp16`、`hf32`（`16×16×8`）与 `fp32`（`16×16×1`）；
   - 微缩放格式（如 `mxfp4 -> fp8_e4m3`）通过 `compute_peak_aliases` 显式映射至原生目标峰值，严禁伪造未声明的独立原生操作数模式。
3. **新增片上直连路由（`l0c->ub`、`l1->ub`、`ub->l1`）**：
   - 相比 `DAV_2201`，`_paths_3510` 新增 `l0c->ub`（`fixpipe` 事件同步，`supported_output_paths` 包含 `"l0c_to_ub"`）、`l1->ub`（`mte1_l1_ub` 服务模式，`128 B/cycle/AIC`）与 `ub->l1`（`mte3`）；
   - 同时支持硬件融合的 `indexer.fused_score_topk = true`（`score_hbm_write_bytes = 0`，中间分数与运行态 TopK 常驻片上 `UB`，零 HBM 分数落盘）。
4. **`gm_hbm_cacheline_128b` 按 SKU 真实 HBM 带宽折算（切勿跨 DT/PR 混抄）**：
   - `Ascend950DT_9581/9582`（`4.0 TB/s` HBM / `64` AIV / `128 B`）为 **`488,281,250.0` cachelines/s/AIV**；
   - `Ascend950PR_9579`（`1.6 TB/s` HBM / `64` AIV / `128 B`）为 **`195,312,500.0` cachelines/s/AIV**；
   - `Ascend950PR_958b`（`1.4 TB/s` HBM / `64` AIV / `128 B`）为 **`170,898,437.5` cachelines/s/AIV**——校准不同 PR/DT 机型时必须按声明 HBM 带宽精确折算，严禁跨 SKU 混抄。
5. **`K_8` `per_pair` UB 全互联网格、多卡运行守卫与 Atlas 950 SuperPoD 拓扑**：
   - 单机 8 卡 `UB_DIRECT_K8_MESH` 声明 `channel_semantics = "per_pair"`：每颗 NPU 拥有 `7` 条独立点对点 UB 链路（单对端标称峰值 `56.0 GB/s`，硅实测有效单向带宽约 `47.5 GB/s`，`alpha_us = 12.0`），因此参与卡数为 \(W \in \{2, \dots, 8\}\) 时单卡有效聚合总线带宽按 **`(W - 1) * 47.5 GB/s`** 线性扩展；标准 HCCL 集合通信配合 `HCCL_OP_EXPANSION_MODE=AIV` 与 `ASCEND_ENABLE_USE_FABRIC_MEM=1` 运行，而 `MoeDistributeDispatchV2`/`CombineV2` 及 `MegaMoE(enable_fused_mc2 in {2, 3})` 额外依赖在线 `uburma` EID 表（离线环境下生产配置回退至 `enable_fused_mc2=0` + `multistream_overlap_shared_expert=True`）；多卡微基准前须逐 Rank 复核 `4096×4096` BF16 GEMM 耗时以排除 LPM `80E3A201` 单卡降频异常；
   - `a5_pod`（Atlas 950 SuperPoD，`ranks_per_node = 64`, `superpod_size = 8192`）与 `a5_pcie_16p`（`4P=288 GB/s`, `8P=128 GB/s`, `node=116 GB/s`）通过 [`ascend.py`](../../mindie_sim/backends/ascend.py) 分层链路生成。

---

## 3. CANN `9.3.0` + `vllm-ascend@405617bae` 栈真源、`_A5_PROFILE` 能力真相与四阶瀑布

### 3.1 栈真源清单与双硅基标定基线

`Ascend950DT_9582`（`4.0 TB/s, 96 GiB`）与 `Ascend950PR_958b`（`1.4 TB/s, 112 GiB`）已完成五域硅实测与 **CANN `9.3.0-20261112_192141454`（`ATB 9.3.0.B030`、`cann_ops_transformer 1.0.0`、`fla_npu 26.10.0+feature`）+ `vllm-ascend@405617bae` 主干 + 35 个活跃未合入优化 PR** 全矩阵标定：
- **公开可审计栈清单**：[`mindie_sim/stack_sources/vllm_ascend_cann93.json`](../../mindie_sim/stack_sources/vllm_ascend_cann93.json) 与 [`mindie_sim/stack_tracks.py`](../../mindie_sim/stack_tracks.py) 为每一条 `merged` / `unmerged_pr` 机制绑定记录精确的 `{repository, revision, relative_path, source_sha256}` 溯源；
- **三级无交集盲测 `< 10%` 硅实测精度闭环**：观测包 [`data/observations/a5-cann93-regimes-20261120/`](../../data/observations/a5-cann93-regimes-20261120/) 与验证报告 [`data/verification/a5-cann93-multilevel-accuracy-20261120.json`](../../data/verification/a5-cann93-multilevel-accuracy-20261120.json) 在严格无交集 `train` / `heldout` 切分（`512 -> 409,600` 序列网格，覆盖 `Ascend950DT_9582`、`Ascend950PR_958b` 及 `2P/8P` UB-Mesh 集合通信）下全部达成 **`< 10%` Held-out MAPE 与 MedAPE**：
  - **Level 1（单算子与 `2P/8P` 集合通信，`391` train / `268` heldout）**：Held-out **`MAPE = 9.07%`、`MedAPE = 6.77%`**（Train `MAPE = 7.29%`、`MedAPE = 5.93%`）；
  - **Level 2（FX `piecewise` 子图与周期层块，`16` train / `18` heldout，`14/14` 编译子图数精确匹配）**：Held-out **`MAPE = 9.96%`、`MedAPE = 8.78%`**（Train `MAPE = 6.50%`、`MedAPE = 7.15%`）；
  - **Level 3（`vllm-ascend@405617bae` 端到端 Prefill / Decode Step，覆盖 `merged` / `piecewise` / `unmerged_pr` 与 `TP×EP` / `DP×EP`）**：Held-out **`MAPE = 9.96%`、`MedAPE = 8.78%`**（Train `MAPE = 6.50%`、`MedAPE = 7.15%`）；
- **PR 合入状态巡检**：运行 `python3 scripts/audit_stack_prs.py --stack cann93`（CI 离线使用 `--offline-fixture`）可自动识别已合入 `main` 待晋升为 `merged` 的 PR 与新增候选性能 PR。

### 3.2 `_A5_PROFILE` 硬件能力矩阵与 CANN 9.3 `Arch35` 算子配方

在 `vllm_ascend/device/hardware_profile.py` 中，`_A5_PROFILE` 从零独立定义 `frozenset`（不继承 A2/A3 的 `_STANDARD_CAPABILITIES`），在建模与校准中必须严格对齐以下源码与实机真相（全部通过 `< 400` 行守恒 `PlannedRecipe` 实现，**100% 按结构与张量属性泛化，零 `model_id` 分支、零 `ResourceEnvelope`**）：

| 机制域 | `Rung 1: merged`（`405617bae` + CANN `9.3.0` 主干） | `Rung 2: unmerged_pr`（活跃未合入 PR 增量） | 对应物理配方模块 |
| :--- | :--- | :--- | :--- |
| **融合 MoE (`MegaMoE`)** | `CANN_MEGAMOE`、`CANN_MEGAMOE_MXFP` 与 **`CANN_MEGAMOE_SHARED_EXPERT_FUSION`（`enable_fused_mc2=3`）已合入**，单核融合 EP Dispatch + `GMM1` + 片上激活/MX32 量化 + `GMM2` + Shared Expert + Combine | **`PR #22333`** 修复 `Kimi-K3` / `Kimi-K3.1` W4A8-MXFP `situglu`（`enable_fused_mc2=2`）打包 `uint8` 校验；`#21543` INT8 仅限 A2/A3，`#22128` 仅限 A3 | [`operators/grouped_moe/`](../../mindie_sim/prediction/operators/grouped_moe/__init__.py) |
| **GDN 线性注意力预填充** | `FLA_GDN_PREFILL` **`Chunk64` 已合入**，受限于 `Hv/Hk in {1, 2, 3, 4}` 且 `K=V=128`（`Hv/Hk=8` 如 `Qwen3.8-2.4T` 或 `Qwen3.5-397B TP=1..4` 自动回退 Triton） | **`PR #22147` 切换至 `Chunk32`**（单 UB VCS `32×32` 三角求逆，同样在 `fla_npu` `ops/fla/chunk.py` 中强制约束 `Hv // Hk in {1, 2, 3, 4}` 且 `Hv/Hk=8` 回退 Triton，实测提速 `1.27x–1.49x`）；`#22045` 打通 `--use-replayssm` | [`gdn_domain.py`](../../mindie_sim/prediction/operators/linear_attention/gdn_domain.py)、[`gdn_chunk32.py`](../../mindie_sim/prediction/operators/linear_attention/gdn_chunk32.py) |
| **mHC / 低秩门控 HC / TP `MC2`** | `RMS_NORM_CAST` 缺省，`DS-V4.1`/`V4` 采用自定义 **`_C_ascend.npu_hc_pre_v3` / `npu_hc_pre_v2`**（原生覆盖 `C in {2560, 4096, 5120, 7168}`）+ BF16 `npu_rms_norm` + `.float()` Cast（非受限的 `aclnnMhcPreSinkhorn`） | **`PR #21794`** 引入 CANN 9.3 原生 `cann_ops_transformer.text_gated_residual` 替代 5 算子 Triton 低秩 HC（CANN 9.3 硅实测 `M <= 512` 触发 `EZ1008`，故强制 `M > 512` 域守卫）；`#22142` 融合 `mhc_post_collapse_norm`（`<= 72` 行）；`#22385` 接入 `AllGatherMatmulV2` + `MatmulReduceScatterV2` | [`operators/hyper_connections/`](../../mindie_sim/prediction/operators/hyper_connections/__init__.py)、[`operators/mc2/`](../../mindie_sim/prediction/operators/mc2/__init__.py) |
| **块稀疏与吸收态稀疏注意力** | 统一 **`GBSA`**（`generic_block_sparse_attention`，覆盖 `[1, 4]` QSA、`[1, 8]` Step-5 CSA、`[1, 128]` MiniMax-M3 MSA 跨层元数据复用）与 `#21588` NoPE `sparse_flash_mla`；CANN 9.3 `C8 SFA` 在 MTP `S=6` 下保持线性访存（无跨 Token 索引去重） | **`PR #22467`** 预计算 A5 C8 `KvQuantSparseFlashAttentionV2` FP8 Scale 地址（开销 `-40%`，惠及 `GLM-5.x` 与 `DS-V3.2`）；**`PR #22164`/`#22473`** 为 A5 补齐 `FLASH_ATTN_NPU_3`（`Step-5` `hd=192` SWA） | [`operators/attention/`](../../mindie_sim/prediction/operators/attention/__init__.py) |
| **MLA 前言与稀疏打分 / 缓存写** | **`MlaPrologV4WeightNz`**（原生 MXFP8 `mode=3` + C8 `kv_mode=1` + NoPE，`q_lora_rank in {1536, 2048}`；`DS-V4.1` `1280` 走 `_multistream_qkv`）、`msa_index_score`、Step-5 `_C_ascend.npu_indexer_region_score`、`#21771` 无权重 `lightning_indexer` 与 `aclnnIndexerQuantCache` | **`PR #22235`** 启用 `msa_index_score_v2` `return_choice=1` 核内 TopK（省去 `score_gm` 落盘，`-40%`）；`#22365` 移植 `#21820` QSA 前言融合（Decode `M <= 64` 门控）；`#21449` 预折叠 `(1+w)` `GemmaRMSNorm`（`Qwen3.5/3.6/3.8`、`MiniMax-M3`；`Gemma-4` 已为标准 `offset=0.0`） | [`operators/mla_prolog/`](../../mindie_sim/prediction/operators/mla_prolog/__init__.py)、[`operators/sparse_indexer/`](../../mindie_sim/prediction/operators/sparse_indexer/__init__.py)、[`index_cache_writer.py`](../../mindie_sim/prediction/operators/sparse_indexer/index_cache_writer.py) |
| **QK-Norm + RoPE 图融合** | `_A5_PROFILE` **遗漏 `GRAPH_QKV_NORM_ROPE_FUSION`**：通用 FX `QknormRopeFusionPass` 在 A5 主干不触发（仅模型层直调 Triton 内核的 `Qwen3.5`、`MiniMax-M3`、`Step-5`、`Qwen4Exp` 生效；标准 `Qwen3`/`Gemma-4` 跑未融合算子） | `#22401`（Step-5 `qkvg` + `indexer_qkz_norm_rope`）；为通用 A5 补齐 `GRAPH_QKV_NORM_ROPE_FUSION` 归入 `Rung 3: expected_ideal` | [`feature_mechanism_catalog.py`](../../mindie_sim/feature_mechanism_catalog.py) |

### 3.3 四阶特性瀑布与 `Software` 正交编译层

1. **四阶特性叠加瀑布（[`stack_tracks.py`](../../mindie_sim/stack_tracks.py) / [`feature_waterfall.py`](../../mindie_sim/feature_waterfall.py)）**：
   通过 `mindie-sim feature-tracks --model <name> --hardware a5 --tp <tp> --ep <ep>` 在同一套 `compose_case` + `analyze_network_efficiency` 主干上连续评估 **`raw_eager` -> `merged` -> `unmerged_pr` -> `expected_ideal` -> `hardware_roofline`**（`evaluate_target_efficiency(..., mode="exact_target")` 计算 `0.80x` 目标与 `1.0x` 物理下界），自动审计单调性，并同时输出正向单加 **`single_pr_ablation`**（`merged + {PR}`）与留一剔除 **`leave_one_out_ablation`**（`unmerged_pr - {PR}`）。
2. **正交两层编译开关（[`Software`](../../mindie_sim/schema/case.py) / [`compose_runtime.py`](../../mindie_sim/compose_runtime.py)）**：
   严格解耦外层 `ModelRunner` 图捕获（`graph_capture_mode in {"none", "piecewise", "full"}`）与内层编译器开关（`enable_torch_compile`、`enable_npugraph_ex`、`enable_static_kernel`、`enable_super_kernel`）：
   - 模型标记 `do_not_compile=True`（`catalog_facts["supports_torch_compile"] == False`，如 `405617bae` 上的 `Step-5-Preview`）**保留外层 `FULL_DECODE_ONLY` `ACLGraph`（`graph_capture_mode="full"`）**，仅令 `enable_torch_compile=False` 从而关闭 `npugraph_ex` FX 融合与 Static/Super Kernel；`PR #22342` 解锁 `enable_torch_compile=True`；
   - 当 `graph_capture_mode != "full"`（如 `piecewise` 或 `none`）或 `enable_torch_compile=False` 时，自动强制关闭 `enable_npugraph_ex`、`enable_static_kernel` 与 `enable_super_kernel`；`PR #22184` 将 `"dspark"` 加入 Static Kernel 投机方法允许列表。

---

## 4. `Ascend950DT_9582` 标准校准字段模板与后续深化靶点

后续外部贡献与增量精化的重点包括：（1）`Ascend950PR_9579`（`1.6 TB/s` HBM 及 PCIe 16P 桥接机型）多机通信与端到端实测验证；（2）跨 `128 MiB` L2 边界的超大 Batch / 超长上下文多流并发工作集扫描，持续核验 `l2_effective_fraction` 与 `mte2_aic_l2` / `mte2_aiv_l2` 供给；（3）全栈上游仓新合入算子与融合配方的最新 `HEAD` 源码对齐及双硅基成对复核。

以下为 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 中 `Ascend950DT_9582` 的标准校准字段参考模板：

```json
{
  "Ascend950DT_9582": {
    "dies_per_package": 1,
    "device": {
      "buffers": {
        "l1": 524288,
        "l0a": 65536,
        "l0b": 65536,
        "l0c": 262144,
        "ub": 253952,
        "l2": 134217728
      },
      "buffer_units": {
        "l1": 32,
        "l0a": 32,
        "l0b": 32,
        "l0c": 32,
        "ub": 64,
        "l2": 1
      },
      "service_modes": {
        "mte1_l1_ub": {"resource": "mte1", "rate": 6758400000000.0},
        "ub_copy": {"resource": "vector", "rate": 27033600000000.0},
        "reg_issue_slots": {"resource": "vector", "rate": 211200000000.0},
        "mte1_l0a": {"resource": "mte1", "rate": 13516800000000.0},
        "mte1_l0b": {"resource": "mte1", "rate": 6758400000000.0},
        "mte2_aic_l2": {"resource": "mte2_aic", "rate": 13516800000000.0},
        "mte2_aic_dram": {"resource": "mte2_aic", "rate": 2217600000000.0},
        "mte2_aiv_l2": {"resource": "mte2_aiv", "rate": 6758400000000.0},
        "mte2_aiv_dram": {"resource": "mte2_aiv", "rate": 3326400000000.0},
        "hbm_atomic_rmw": {"resource": "hbm", "rate": 772000000000.0}
      },
      "compute_arch": {
        "preset": "npu_arch_3510"
      },
      "kernel_invocations": {
        "vector": {
          "startup_ms": 0.0,
          "minimum_ms": 0.000877,
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a5-9582-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIV completion minimum; not additive startup"
          }
        },
        "cube": {
          "startup_ms": 0.0,
          "minimum_ms": 0.001237,
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a5-9582-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIC completion minimum; not additive startup"
          }
        },
        "norm": {
          "startup_ms": 0.0,
          "minimum_ms": 0.001387,
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a5-9582-microbench-202611/domain1_launch.json",
            "scope": "RTS tiny norm inclusive completion minimum; not additive startup"
          }
        }
      },
      "physical_mechanisms": {
        "memory_hierarchy": {
          "l2_capacity_bytes": 134217728,
          "l2_effective_fraction": 0.75,
          "cache_line_bytes": 128,
          "write_allocate": true,
          "provenance": {
            "basis": "a5_live_silicon_benchmarks",
            "arch": "DAV_3510",
            "clock_hz": 1650000000.0,
            "l2_capacity_bytes": 134217728,
            "l2_effective_fraction": 0.75,
            "calibrated": true
          }
        },
        "matrix": {
          "base_tiles": [[128, 128], [128, 256], [256, 128], [256, 256]],
          "refill_k_tiles": [64, 128, 256],
          "micro_tiles_by_dtype": {
            "bf16": [[128, 128, 128], [64, 64, 128], [16, 16, 16]],
            "fp16": [[128, 128, 128], [64, 64, 128], [16, 16, 16]],
            "fp32": [[128, 128, 64], [64, 64, 64], [16, 16, 16]],
            "int8": [[128, 128, 128], [64, 64, 128], [16, 16, 32]],
            "fp8": [[128, 128, 128], [64, 64, 128], [16, 16, 32]]
          },
          "input_slots": 2,
          "compute_slots": 2,
          "split_k_candidates": [1, 2, 4, 8, 16, 32],
          "min_k_per_split": 64,
          "swizzle_windows": [[1, 1], [2, 8], [4, 8]],
          "supported_output_paths": ["gm", "l0c_to_ub", "onchip_reduce"]
        },
        "attention": {
          "query_tile": 128,
          "page_block": 128,
          "decode_max_q": 8,
          "fia_min_keys_per_split": 1024,
          "max_fia_kv_splits": 16,
          "score_bytes": 4,
          "online_softmax_exp_class": "special_lane"
        },
        "indexer": {
          "query_tile": 64,
          "heap_comparison_multiplier": 1,
          "fused_score_topk": true,
          "score_hbm_write_bytes": 0
        },
        "vector": {
          "min_tile_bytes": 4096,
          "double_buffer_slots": 2,
          "norm_minimum_ms": 0.001387,
          "vector_minimum_ms": 0.000877
        },
        "collective": {
          "peer_effective_bytes_per_s": 47500000000.0,
          "peer_peak_bytes_per_s": 56000000000.0,
          "alpha_us": 12.0,
          "hierarchical_inter_node_bytes_per_s": 50000000000.0,
          "hierarchical_inter_node_alpha_us": 5.0
        }
      }
    },
    "transfer_channels": {
      "d2d_intra_node": {
        "channel": "d2d_intra_node",
        "protocol": "UB_DIRECT_K8_MESH",
        "uni_peak_bw_gbps": 56.0,
        "bi_peak_bw_gbps": 103.6,
        "calibrated_uni_bw_gbps": 47.5,
        "calibrated_bi_bw_gbps": 95.0,
        "bandwidth_provenance": "silicon_calibrated_per_pair",
        "effective_uni_bw_gbps": 47.5,
        "effective_bi_bw_gbps": 95.0,
        "alpha_us": 12.0,
        "transport_mode": "direct_fabric",
        "channel_semantics": "per_pair"
      }
    }
  }
}
```
