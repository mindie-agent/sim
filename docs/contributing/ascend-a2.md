# Ascend A2（`Ascend910B1..B4-1` / `DAV_2201`）硅实测校准与增量开发子指南

[返回总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md) · [Ascend A3 子指南](ascend-a3.md) · [Ascend A5 子指南](ascend-a5.md)

> [!IMPORTANT]
> 本子页面规定 **Ascend A2（微架构 `DAV_2201` / 能力预设 `npu_arch_2201`）** 的专属硬件几何基线、片上缓存与精度红线、互联拓扑语义及 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 校准补丁模板。通用的单一主干铁律、五域微基准 `StreamBlocker` 计时协议、自定义 Attention 三级接入（Tiers 1–3）与 PR 六道门禁请先阅读 [总指南 `CONTRIBUTING.md`](../../CONTRIBUTING.md)。

---

## 1. 已注册 SKU 标称几何基线与硬件别名

在 [`mindie_sim/compose/inputs.py`](../../mindie_sim/compose/inputs.py) 中，常用硬件别名映射为：`a2` / `910b` / `910b4` -> `Ascend910B4`；`800i-a2` -> `Ascend910B4` + `a2_standard`（8P HCCS）；`910b1` / `910b2` / `910b3` 分别对应各自精确 SKU。[`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 中注册的 6 款 A2 SKU 基线如下（采集前请先通过 `npu-smi info` 核对真机芯片型号）：

| SKU 键名 | 封装形态 | AIC / AIV | 标称主频 | HBM 容量 / 标称带宽 | 共享 L2 容量（`buffer_units.l2 = 1`） | BF16 / INT8 / INT4 标称峰值 | 冷 HBM 旋转环下限（`>= 3 × L2`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ascend910B1` | 单 Die (`1`) | `24 / 48` | `1850 MHz` | `64 GiB` / `1800 GB/s` | `192 MiB` (`201,326,592 B`) | `363.72` / `727.45` / `1454.90` TFLOPS | **`>= 576 MiB`** |
| `Ascend910B2` | 单 Die (`1`) | `24 / 48` | `1800 MHz` | `64 GiB` / `1800 GB/s` | `192 MiB` (`201,326,592 B`) | `353.89` / `707.79` / `1415.58` TFLOPS | **`>= 576 MiB`** |
| `Ascend910B2C` | 单 Die (`1`) | `24 / 48` | `1800 MHz` | `64 GiB` / `1800 GB/s` | `192 MiB` (`201,326,592 B`) | `353.89` / `707.79` / `1415.58` TFLOPS | **`>= 576 MiB`** |
| `Ascend910B3` | 单 Die (`1`) | `20 / 40` | `1800 MHz` | `64 GiB` / `1600 GB/s` | `192 MiB` (`201,326,592 B`) | `294.91` / `589.82` / `1179.65` TFLOPS | **`>= 576 MiB`** |
| `Ascend910B4` | 单 Die (`1`) | `20 / 40` | `1500 MHz` | `32 GiB` / `800 GB/s`* | `96 MiB` (`100,663,296 B`) | `245.76` / `491.52` / `983.04` TFLOPS | **`>= 288 MiB`** |
| `Ascend910B4-1` | 单 Die (`1`) | `20 / 40` | `1650 MHz` | `64 GiB` / `1200 GB/s` | `96 MiB` (`100,663,296 B`) | `270.34` / `540.67` / `1081.34` TFLOPS | **`>= 288 MiB`** |

> [!NOTE]
> `Ascend910B4` 当前在 `source.declaration` 中标记 `"hbm_peak_bw_verified": false`；全系 A2 的 `kernel_invocations`（`vector: 0.0025 ms`, `cube: 0.005 ms`）仍为从 `Ascend910_9362` 外推的基线值，`physical_mechanisms.memory_hierarchy.provenance.calibrated` 均为 `false`。这些是 A2 硅实测优先补齐的高价值校准项。

---

## 2. `DAV_2201`（`npu_arch_2201`）微架构不变量与建模红线

1. **片上私有缓存硬上限（尤其注意 `L0C = 128 KiB`）**：
   - 每个 AIC 私有：`L1 = 512 KiB`（`524,288 B`），`L0A = 64 KiB`（`65,536 B`），`L0B = 64 KiB`（`65,536 B`），**`L0C = 128 KiB`（`131,072 B`）**；
   - 每个 AIV 私有：**`UB = 192 KiB`（`196,608 B`）**，单周期向量执行宽度 `128 B/cycle/AIV`（`32` FP32 lane-ops/cycle）；
   - **分块铁律**：FP32 累加下 `256 × 256 × 4 B = 262,144 B` **超出 A2 的 `128 KiB` `L0C` 上限**！因此 A2 稠密/分组矩阵 `base_tiles` 只能使用 `[128, 128]`、`[128, 256]`、`[256, 128]` 及更小分块，切勿照搬 A5 的 `[256, 256]`。
2. **原生矩阵计算精度（支持 W8A8 / W4A8 / W4A16，严禁原生 FP8 Cube）**：
   - [`mindie_sim/schema/compute_arch.py`](../../mindie_sim/schema/compute_arch.py) 为 `npu_arch_2201` 固定声明 `fp8=False, int4_operand=True`：原生支持 `bf16` / `fp16`（`cubek16`）、`fp32`（`cubek4`）、`hf32`（`cubek8`）、`int8`（`cubek32`，`npu_quant_matmul` W8A8）及 **`int4`（`cubek64`，[`w4a8.py`](../../mindie_sim/prediction/operators/mx_quant_matrix/w4a8.py) MSD / `npu_weight_quant_batchmatmul` W4A16）**；
   - **严禁向 A2 的 `compute_peaks` 添加 `fp8` / `fp8_e4m3` / `fp8_e5m2`**，否则触发 `ModelingError("native_compute_mode_unsupported:matrix:fp8...")`。
3. **AIC <-> AIV 核间数据通路（无片上 `l0c->ub` 直连）**：
   - 在 `npu_arch_2201`（`_paths_2201`）中，独立的 AIC 与 AIV 核**必须经由 GM（共享 L2 或 HBM）交换中间张量**（`l0c->gm` 经 `fixpipe`，`gm->ub` 经 `mte2_aiv`，`ub->gm` 经 `mte3`）；
   - 编写 A2 融合算子配方（如 [`operators/paged_mla/`](../../mindie_sim/prediction/operators/paged_mla/__init__.py) 或自定义 AscendC/Triton Attention）时，`supported_output_paths` 不得声明 A5 专属的 `"l0c_to_ub"`，且跨核传递 Score / Probability / 中间激活必须计入 `fixpipe` / `mte2_aiv` / `mte3`（及未命中 L2 时的 `hbm`）流量。
4. **Attention 开箱默认基线与自定义内核扩展**：
   - 开箱默认基线覆盖 `npu_fusion_attention`（Prefill）、`npu_incre_flash_attention`（IFA）、`npu_fused_infer_attention_score`（FIA）及 `ascend_paged_mla_decode_arch22`（`KEY_TILE = 512`，`decode_max_q = 16`，物理头数 `8` 或 `16`）；
   - 若使用自研 AscendC / Triton-Ascend / TileLang / FlashMLA 内核，直接按 [总指南 `3.4` 节](../../CONTRIBUTING.md#34-专项实战如何接入自定义-attention-算子ascendc--triton--稀疏与融合变体) 的 Tiers 1–3 接入，无需迁就 IFA / FIA。
5. **机内互联拓扑（8P HCCS Mesh vs 16P PCIe/HCCS 分层桥接）**：
   - A2 为单封装单 Die（`dies_per_package = 1`，`d2d_intra_package.protocol = "NONE_SINGLE_DIE"`）；
   - 标准 8 卡机型（`a2_standard`）采用 HCCS Mesh（标称 `uni_peak_bw_gbps = 168.0`，`bi_peak_bw_gbps = 310.8`，在 [`ascend.py`](../../mindie_sim/backends/ascend.py) 中默认生成端口级 `per_port` 链路）；
   - 16 卡 PCIe / 4P-HCCS 桥接机型校准时，必须在机器声明或拓扑预设中显式通过 `intra_bw_4p_gbps`、`intra_bw_8p_gbps` 与 `intra_node_bw_gbps` 区分 4P/8P 环内 HCCS 带宽与跨组 PCIe Switch 带宽，并在 `W = 2, 4, 8, 16` 四个粒度分别采集集合通信 $(\alpha, \beta)$。

---

## 3. A2 [`ascend.json`](../../mindie_sim/backends/profiles/ascend.json) 精确校准补丁模板

以下展示 `Ascend910B1` 的精确校准结构（校准 `Ascend910B4` / `B4-1` 时请将 `buffers.l2` 与 `l2_capacity_bytes` 改为 `100663296`，核数改为 `20` / `40`；所有 `<MEASURED_*>` 必须来自真实硅实测拟合，未测项保留原值并维持 `calibrated: false`）：

```json
{
  "Ascend910B1": {
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
          "rate": 11366400000000.0
        },
        "mte1_l0b": {
          "resource": "mte1",
          "rate": 5683200000000.0
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
        "preset": "npu_arch_2201",
        "instruction_services": {
          "gm_hbm_cacheline_128b": {
            "unit": "cachelines",
            "owner_axis": "vector",
            "per_owner_rate": 292968750.0,
            "provenance": {
              "kind": "ascend_a2_hbm_128b_cacheline_transaction",
              "architecture": "A2/arch2201",
              "source": {
                "repository": "hicann/asc-devkit",
                "revision": "56500ea0dc219380f50f4a17ce73ef5264f830fc",
                "relative_path": "docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md",
                "source_sha256": "d16edc43d897cc4eba425802e0b7d1f61b1bc76fb50768f336d458ddd4927994"
              },
              "scope": "1.8e12 B/s HBM divided across 48 AIV at 128B line granularity"
            }
          }
        }
      },
      "kernel_invocations": {
        "vector": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_AIV_ADD_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a2-910b1-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIV completion minimum; not additive startup"
          }
        },
        "cube": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_AIC_MATMUL_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a2-910b1-microbench-202611/domain1_launch.json",
            "scope": "RTS inclusive AIC completion minimum; not additive startup"
          }
        },
        "norm": {
          "startup_ms": 0.0,
          "minimum_ms": "<MEASURED_RTS_SATURATED_TINY_NORM_MS>",
          "provenance": {
            "basis": "stream_saturated_event_median",
            "boundary": "device_kernel_completion",
            "source": "capture://a2-910b1-microbench-202611/domain1_launch.json",
            "scope": "RTS tiny norm inclusive completion minimum; not additive startup"
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
            "basis": "a2_live_silicon_benchmarks",
            "arch": "DAV_2201",
            "clock_hz": 1850000000.0,
            "l2_capacity_bytes": 201326592,
            "l2_effective_fraction": 0.75,
            "capture": "capture://a2-910b1-microbench-202611/summary.json",
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
          "alpha_us": "<MEASURED_HCCS_COLLECTIVE_ALPHA_US>"
        }
      }
    },
    "transfer_channels": {
      "d2d_intra_node": {
        "channel": "d2d_intra_node",
        "protocol": "HCCS",
        "uni_peak_bw_gbps": 168.0,
        "bi_peak_bw_gbps": 310.8,
        "calibrated_uni_bw_gbps": "<MEASURED_HCCS_UNI_BW_GBPS>",
        "calibrated_bi_bw_gbps": "<MEASURED_HCCS_BI_BW_GBPS>",
        "bandwidth_provenance": "calibrated",
        "effective_uni_bw_gbps": "<MEASURED_HCCS_UNI_BW_GBPS>",
        "effective_bi_bw_gbps": "<MEASURED_HCCS_BI_BW_GBPS>",
        "alpha_us": "<MEASURED_HCCS_ALPHA_US>",
        "transport_mode": "direct_fabric",
        "channel_semantics": "per_port"
      }
    }
  }
}
```
