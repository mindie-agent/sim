# Ascend A5（`Ascend950DT` / `Ascend950PR` / `DAV_3510`）硅实测校准与增量开发子指南

[返回总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md) · [Ascend A2 子指南](ascend-a2.md) · [Ascend A3 子指南](ascend-a3.md)

> [!IMPORTANT]
> 本子页面规定 **Ascend A5（微架构 `DAV_3510` / 能力预设 `npu_arch_3510`）** 及 **Atlas 950 SuperPoD** 的专属硬件几何、原生 FP8 与片上直连路径、`K_8` `per_pair` UB 网格扩展定律及增量校准靶点。通用的单一主干铁律、五域微基准 `StreamBlocker` 协议、自定义 Attention 三级接入（Tiers 1–3）与 PR 六道门禁请先阅读 [总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md)。

---

## 1. 已注册 SKU 标称几何基线与硬件别名

在 [`mindie_sim/compose.py`](../../mindie_sim/compose.py) 中，A5 硬件与拓扑别名精确解析如下（**严禁构造或引用已废弃的虚构 `9599` SKU**）：
- `a5` / `950` -> `Ascend950DT_9582`
- `950pr_958b` / `ascend950pr_958b` -> `Ascend950PR_958b`
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
5. **`K_8` `per_pair` UB 全互联网格与 Atlas 950 SuperPoD 拓扑**：
   - 单机 8 卡 `UB_DIRECT_K8_MESH` 声明 `channel_semantics = "per_pair"`：每颗 NPU 拥有 `7` 条独立点对点 UB 链路（单对端标称峰值 `56.0 GB/s`，硅实测有效单向带宽约 `47.5 GB/s`，`alpha_us = 12.0`），因此参与卡数为 \(W \in \{2, \dots, 8\}\) 时单卡有效聚合总线带宽按 **`(W - 1) * 47.5 GB/s`** 线性扩展；
   - `a5_pod`（Atlas 950 SuperPoD，`ranks_per_node = 64`, `superpod_size = 8192`）与 `a5_pcie_16p`（`4P=288 GB/s`, `8P=128 GB/s`, `node=116 GB/s`）通过 [`ascend.py`](../../mindie_sim/backends/ascend.py) 分层链路生成。

---

## 3. A5 已合入双硅基标定基线、增量深化靶点与 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 模板

`Ascend950DT`（`4.0 TB/s`）与 `Ascend950PR_958b`（`1.4 TB/s`）已合入五域硅实测基线与 CANN 9.2.1 生产算子图成对标定（覆盖 C8 SFA 稀疏 gather、`aclnnMixedQuantSparseFlashMla`、`KeyPool` RowShard、`arch35` QLI V2、`CausalConv1d UPDATE`、L2 容量推导 `MXFP4/MXFP8` GMM、`npu_swiglu_mx_quant`、窄 `N` Split-K BF16 Router、`aclnnMoeGatingTopK` 与单调 NZ Block-32 Prologue，100% 按结构与张量属性泛化，零 `model_id` 绑定）。后续外部贡献与增量精化的重点包括：
1. `Ascend950PR_9579`（`1.6 TB/s` HBM 及 PCIe 16P 桥接机型）多机通信与端到端实测验证；
2. 跨 `128 MiB` L2 边界的超大 Batch / 超长上下文多流并发工作集扫描，持续核验 `l2_effective_fraction` 与 L2 源 `mte2_aic_l2` / `mte2_aiv_l2` 供给；
3. 上游 `vllm-ascend`、`cann/ops-transformer`、`cannbot-dsl` 新合入算子（如新型融合线性注意力与自定义稀疏 Attention Tiers 1–3）的最新 `HEAD` 源码对齐与双硅基成对复核。

以下为 `Ascend950DT_9582` 的标准校准字段参考模板：

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
