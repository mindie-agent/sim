# Ascend A3（`Ascend910_9362..9392` / CloudMatrix 384）硅实测校准与增量开发子指南

[返回总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md) · [Ascend A2 子指南](ascend-a2.md) · [Ascend A5 子指南](ascend-a5.md)

> [!IMPORTANT]
> 本子页面规定 **Ascend A3（单封装双 Die 架构，单 Die 计算核预设 `npu_arch_2201`）** 及 **CloudMatrix 384 超节点**的专属几何不变量、双 Die 拓扑边界与 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 校准补丁模板。通用的单一主干铁律、五域微基准 `StreamBlocker` 协议、自定义 Attention 三级接入（Tiers 1–3）与 PR 六道门禁请先阅读 [总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md)。

---

## 1. 双 Die 归属第一铁律（`dies_per_package = 2` / `Phy-ID 0..15`）

> [!CAUTION]
> **单颗逻辑 Rank = 单颗物理 Die（绝非整张双 Die 物理卡）！**
> 在 Ascend A3（如 8 张物理模组构成的 16-Die 节点，操作系统与 `npu-smi` 暴露 `Phy-ID 0..15`）中，`mindie-sim` 的每一个 `Device` 实例精确对应 **1 颗物理 Die**，且顶层声明 `"dies_per_package": 2`。
> - **严禁将整包双 Die 的显存（`128 GiB`）、核数（`40/48 AIC`）或共享 L2 容量翻倍填入 `device` 字段**！
> - `device.memory_bytes`、`compute_units`、`compute_peaks`、`rates` 与 `buffers.l2` 全部按 **单 Die** 计价，且单 Die 内仍严格保持 `buffer_units.l2 = 1`。

[`mindie_sim/compose.py`](../../mindie_sim/compose.py) 中的 A3 硬件与拓扑别名映射如下：
- `a3` / `910c` / `910_9392` -> `Ascend910_9392`
- `910_9382` -> `Ascend910_9382`
- `800i-a3-8p` -> `("Ascend910_9392", "a3_8p")`（4 物理封装 = `8` 逻辑 Die Rank）
- `800i-a3` / `800i-a3-16p` -> `("Ascend910_9392", "a3_16p")`（8 物理封装 = `16` 逻辑 Die Rank）
- `a3-pod` -> `("Ascend910_9392", "a3_superpod")`（**CloudMatrix 384**：`ranks_per_node = 16`, `superpod_size = 384`）

[`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 中注册的 6 款 A3 SKU **单 Die 基线**如下表所示：

| SKU 键名 | `dies_per_package` | 单 Die AIC / AIV | 标称主频 | 单 Die HBM / 带宽 | 单 Die 共享 L2（`buffer_units.l2 = 1`） | 单 Die BF16 / INT8 / INT4 峰值 | 冷 HBM 旋转环下限（`>= 3 × L2`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ascend910_9391` | `2` | `24 / 48` | `1850 MHz` | `64 GiB` / `1600 GB/s` | `192 MiB` (`201,326,592 B`) | `363.72` / `727.45` / `1454.90` TFLOPS | **`>= 576 MiB`** |
| `Ascend910_9392` | `2` | `24 / 48` | `1800 MHz` | `64 GiB` / `1600 GB/s` | `192 MiB` (`201,326,592 B`) | `353.89` / `707.79` / `1415.58` TFLOPS | **`>= 576 MiB`** |
| `Ascend910_9381` | `2` | `20 / 40` | `1800 MHz` | `64 GiB` / `1600 GB/s` | `192 MiB` (`201,326,592 B`) | `294.91` / `589.82` / `1179.65` TFLOPS | **`>= 576 MiB`** |
| `Ascend910_9382` | `2` | `20 / 40` | `1650 MHz` | `64 GiB` / `1600 GB/s` | `192 MiB` (`201,326,592 B`) | `270.34` / `540.67` / `1081.34` TFLOPS | **`>= 576 MiB`** |
| `Ascend910_9372` | `2` | `20 / 40` | `1650 MHz` | `64 GiB` / `1400 GB/s` | `192 MiB` (`201,326,592 B`) | `270.34` / `540.67` / `1081.34` TFLOPS | **`>= 576 MiB`** |
| `Ascend910_9362` | `2` | `20 / 40` | `1650 MHz` | `64 GiB` / `1200 GB/s` | **`168 MiB` (`176,160,768 B`)** | `270.34` / `540.67` / `1081.34` TFLOPS | **`>= 504 MiB`** |

---

## 2. A3 多级互联层次与单 Die 计算核边界

### 2.1 三层 Scale-Up / Scale-Out 物理互联边界（[`ascend.py`](../../mindie_sim/backends/ascend.py)）
A3 的集合通信与 MoE `AllToAll` / MC2 通算掩盖高度依赖多级链路拓扑，**校准时严禁把同封装 `SIO` 与跨封装 `HCCS_SW` 混拟合为单一带宽参数**：
1. **Tier 0 — 封装内兄弟 Die 互联（`d2d_intra_package`）**：
   - 连接同一物理封装内的偶数/奇数对 `(2k, 2k+1)`（如 `Phy-ID (0,1), (2,3), ..., (14,15)`）；
   - 协议为 `SIO`（标称 `uni_peak_bw_gbps = 448.0`，`bi_peak_bw_gbps = 828.8`，`alpha_us = 0.8`，`channel_semantics = "direct"`）。
   - 实测校准 `W = 2` 时，**必须分别采集 `(0, 1)` 同封装 `SIO` 对与 `(0, 2)` 跨封装 `HCCS_SW` 对**！
2. **Tier 1 — 节点内跨封装互联（`d2d_intra_node` 及 `4P` / `8P` 子域）**：
   - Profile 基础通道协议为 `HCCS_SW`（单通道基准 `uni_peak_bw_gbps = 196.0`，`alpha_us = 2.5`；拓扑预设 `a3_8p` / `a3_16p` 声明聚合对等容量 `intra_node_bw_gbps = 392.0`、`intra_bw_4p_gbps = 392.0`、`intra_bw_8p_gbps = 392.0`、`intra_node_bw_util = 0.80`、`intra_node_latency_us = 1.5`）；
   - 实测需完整覆盖 `W = 4, 8, 16` 下的 Ring / Mesh / Hierarchical AllReduce 与 DeepEP / MC2 Dispatch-Combine。
3. **Tier 2 — CloudMatrix 384 超节点域与跨 Pod 互联（`a3_superpod`）**：
   - 超节点内（`superpod_size = 384` Dies）通过高速交换平面互联（`superpod_bw_gbps = 392.0`，`inter_node_bw_util = 0.70`，`inter_node_latency_us = 3.5`）；
   - 跨超节点（`count > 384`）自动生成 `d2d_inter_pod` 链路（`inter_pod_bw_gbps = 50.0`，`inter_pod_latency_us = 10.0`）。

### 2.2 单 Die 计算核与片上缓存不变量（`npu_arch_2201`）
- A3 单颗 Die 的计算核指令集与片上私有缓冲容量与 `DAV_2201` 一致：`L1 = 512 KiB`、`L0A = 64 KiB`、`L0B = 64 KiB`、**`L0C = 128 KiB`（`131,072 B`，`256×256` FP32 累加块不可放入）**、**`UB = 192 KiB`（`196,608 B`）**。
- 原生支持 `bf16`、`fp16`、`fp32`、`hf32`、`int8`（W8A8）与 `int4`（W4A8 MSD / W4A16），**无原生 FP8 Cube**；AIC 与 AIV 经由共享 L2 / GM 交换中间数据，无片上 `l0c->ub` 直连管道。
- 注意 `Ascend910_9362` 的单 Die 共享 L2 容量为 **`168 MiB`（`176,160,768 B`）**，其余 5 款 A3 SKU 为 **`192 MiB`（`201,326,592 B`）**，切勿在 `9362` 上误填 `201,326,592`。

---

## 3. A3 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 精确校准补丁模板

以下展示 `Ascend910_9392` 的校准补丁模板（同时包含单 Die 计算/访存参数与 `d2d_intra_package` `SIO` + `d2d_intra_node` `HCCS_SW` 双层通道校准字段）：

```json
{
  "Ascend910_9392": {
    "dies_per_package": 2,
    "device": {
      "buffers": {
        "l1": 524288,
        "l0a": 65536,
        "l0b": 65536,
        "l0c": 131072,
        "ub": 196608,
        "l2": 201326592
      },
      "buffer_units": {
        "l1": 24,
        "l0a": 24,
        "l0b": 24,
        "l0c": 24,
        "ub": 48,
        "l2": 1
      },
      "service_modes": {
        "mte1_l0a": {
          "resource": "mte1",
          "rate": 11059200000000.0
        },
        "mte1_l0b": {
          "resource": "mte1",
          "rate": 5529600000000.0
        },
        "mte2_aic_l2": {
          "resource": "mte2_aic",
          "rate": "<MEASURED_OR_DECLARED_AIC_L2_BYTES_PER_S>"
        },
        "mte2_aic_dram": {
          "resource": "mte2_aic",
          "rate": "<MEASURED_AIC_COLD_DRAM_BYTES_PER_S>"
        },
        "mte2_aiv_l2": {
          "resource": "mte2_aiv",
          "rate": "<MEASURED_OR_DECLARED_AIV_L2_BYTES_PER_S>"
        },
        "mte2_aiv_dram": {
          "resource": "mte2_aiv",
          "rate": "<MEASURED_AIV_COLD_DRAM_BYTES_PER_S>"
        },
        "hbm_atomic_rmw": {
          "resource": "hbm",
          "rate": "<MEASURED_ATOMIC_RMW_BYTES_PER_S>"
        }
      },
      "compute_arch": {
        "preset": "npu_arch_2201"
      },
      "kernel_invocations": {
        "vector": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_AIV_ADD_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a3-9392-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIV completion minimum per Die; not additive startup"
          }
        },
        "cube": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_AIC_MATMUL_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a3-9392-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIC completion minimum per Die; not additive startup"
          }
        },
        "norm": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_TINY_NORM_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a3-9392-microbench-202611/domain1_launch.json",
            "scope": "RTS tiny norm inclusive completion minimum per Die"
          }
        }
      },
      "physical_mechanisms": {
        "memory_hierarchy": {
          "l2_capacity_bytes": 201326592,
          "l2_effective_fraction": 0.75,
          "cache_line_bytes": 128,
          "write_allocate": true,
          "provenance": {
            "basis": "a3_live_silicon_benchmarks",
            "arch": "A3",
            "clock_hz": 1800000000.0,
            "l2_capacity_bytes": 201326592,
            "l2_effective_fraction": 0.75,
            "capture": "capture://a3-9392-microbench-202611/summary.json",
            "calibrated": true
          }
        },
        "matrix": {
          "base_tiles": [[128, 128], [128, 256], [256, 128]],
          "refill_k_tiles": [64, 128, 256],
          "micro_tiles_by_dtype": {
            "bf16": [[128, 128, 128], [64, 64, 128], [16, 16, 16]],
            "fp16": [[128, 128, 128], [64, 64, 128], [16, 16, 16]],
            "fp32": [[128, 128, 64], [64, 64, 64], [16, 16, 16]],
            "int8": [[128, 128, 128], [64, 64, 128], [16, 16, 32]]
          },
          "input_slots": 2,
          "compute_slots": 2,
          "split_k_candidates": [1, 2, 4, 8, 16, 24],
          "min_k_per_split": 64,
          "swizzle_windows": [[1, 1], [2, 8], [4, 6]],
          "supported_output_paths": ["gm", "onchip_reduce"]
        },
        "attention": {
          "query_tile": 128,
          "page_block": 512,
          "decode_max_q": 16
        },
        "indexer": {
          "query_tile": 64,
          "fused_score_topk": false
        },
        "vector": {
          "min_tile_bytes": 4096,
          "double_buffer_slots": 2,
          "norm_minimum_ms": "<MEASURED_RTS_SATURATED_TINY_NORM_MS>",
          "vector_minimum_ms": "<MEASURED_RTS_SATURATED_AIV_ADD_MS>"
        },
        "collective": {
          "alpha_us": "<MEASURED_A3_COLLECTIVE_ALPHA_US>"
        }
      }
    },
    "transfer_channels": {
      "d2d_intra_package": {
        "channel": "d2d_intra_package",
        "protocol": "SIO",
        "uni_peak_bw_gbps": 448.0,
        "bi_peak_bw_gbps": 828.8,
        "calibrated_uni_bw_gbps": "<MEASURED_SIO_DIE2DIE_UNI_BW_GBPS>",
        "calibrated_bi_bw_gbps": "<MEASURED_SIO_DIE2DIE_BI_BW_GBPS>",
        "bandwidth_provenance": "calibrated",
        "effective_uni_bw_gbps": "<MEASURED_SIO_DIE2DIE_UNI_BW_GBPS>",
        "effective_bi_bw_gbps": "<MEASURED_SIO_DIE2DIE_BI_BW_GBPS>",
        "alpha_us": "<MEASURED_SIO_ALPHA_US>",
        "transport_mode": "die_to_die",
        "channel_semantics": "direct"
      },
      "d2d_intra_node": {
        "channel": "d2d_intra_node",
        "protocol": "HCCS_SW",
        "uni_peak_bw_gbps": 196.0,
        "bi_peak_bw_gbps": 362.6,
        "calibrated_uni_bw_gbps": "<MEASURED_HCCS_SW_UNI_BW_GBPS>",
        "calibrated_bi_bw_gbps": "<MEASURED_HCCS_SW_BI_BW_GBPS>",
        "bandwidth_provenance": "calibrated",
        "effective_uni_bw_gbps": "<MEASURED_HCCS_SW_UNI_BW_GBPS>",
        "effective_bi_bw_gbps": "<MEASURED_HCCS_SW_BI_BW_GBPS>",
        "alpha_us": "<MEASURED_HCCS_SW_ALPHA_US>",
        "transport_mode": "switched_fabric",
        "channel_semantics": "switched"
      }
    }
  }
}
```
