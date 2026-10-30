"""Source-derived grouped expert BF16 ND GEMM planner.

[Role] Model non-transposed TILING_KEY=10 GMM_CUBE_IMP(false).
[Invariants] Semantic histograms compact positive bins while preserving expert
  IDs. The literal raw list instead terminates at the first nonpositive direct
  count; its declared weight-expert extent retains zero terminator padding.
  X/Y offsets are positive-count prefixes, and arithmetic is exact SetSingleShape
  work without invented fractal padding. Each block owns one actual source core;
  L1/L0 allocations and pipelines retain full physical owner counts.
[Timeouts & Locks] O(experts * units) spans, lazy output tiles; no waits or locks.
[Side-Effect Policy] No I/O, device runtime, measurements or global mutation.
"""
from __future__ import annotations

from bisect import bisect_right
import math
from typing import Any, Iterator, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import unit_group
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.operators import OpSpec

_IMPLEMENTATION = "ascendc_grouped_nd_bf16"
_SINGLE_M = 128
_SINGLE_N = 256
_BASE_N = 256
_MAX_BASE_M = 256
_L1_PART_A = 256 * 1024
_FP32_BYTES = 4
_MM_DTYPE_BYTES = 2
_ELEM_BYTES = 2.0

_SOURCE_IDENTITY = {
    "scope": "current_checkout_manifest_matched_not_historical_capture_image",
    "files": (
        {
            "path": "/Users/maoxx241/code/vllm-ascend/csrc/moe/moe_grouped_matmul/op_kernel/moe_grouped_matmul.cpp",
            "sha256": "7a6f14a3078351c8297915a9d452364a737fe6e41b15d562d169a22d2848214e",
            "bytes": 2443,
        },
        {
            "path": "/Users/maoxx241/code/vllm-ascend/csrc/moe/moe_grouped_matmul/op_kernel/moe_grouped_matmul.h",
            "sha256": "dfd18924241fa153f3b42c85e07a29733f575de95bb236c297a7a106ce35f628",
            "bytes": 7861,
        },
        {
            "path": "/Users/maoxx241/code/vllm-ascend/csrc/moe/moe_grouped_matmul/op_kernel/moe_grouped_matmul_tiling_data.h",
            "sha256": "4f23e438d69fa69bfdfddc82fef893f4886ef5be072bcf6f3c74199047bdd6cf",
            "bytes": 411,
        },
        {
            "path": "/Users/maoxx241/code/vllm-ascend/csrc/moe/moe_grouped_matmul/op_host/moe_grouped_matmul_tiling.h",
            "sha256": "ef061fc0afc726100a5b9ded5f4e648ea4a6d5674dd053e7941e3c19b01444c3",
            "bytes": 1267,
        },
        {
            "path": "/Users/maoxx241/code/vllm-ascend/csrc/moe/moe_grouped_matmul/op_host/moe_grouped_matmul_cpu.cpp",
            "sha256": "5953db15ff725024880f90982b531385b0187eed484c8b68a8c5d601bd792508",
            "bytes": 14500,
        },
    ),
}


def _source_identity() -> dict[str, Any]:
    """Return a per-op copy so metadata never aliases a mutable module global."""
    return {**_SOURCE_IDENTITY, "files": tuple(dict(item) for item in _SOURCE_IDENTITY["files"])}


_UNRESOLVED = (
    "group_list element width (T2 int32/int64) is defined outside the supplied files; scalar GetValue loads are not priced",
    "host tiling CPU instruction time is not declared by the supplied host source",
    "MatmulImpl internal baseM/baseK fractal padding and exact L1/L0 double-buffer schedule are not declared by the supplied files; the model uses host-derived one-panel residency",
    "MTE1 L1-to-L0 fill and Fixpipe output-path rates are not declared separately; L1/L0 residency is modeled, but only MTE2/MTE3/HBM service is charged",
    "idle launched blocks still execute the scalar group-list loop; that unpriced control work is excluded",
    "L2 residency of a repeated ND weight tile is not assumed, so weight bytes are charged per MMCompute task",
    "output bytes use the engine mte3 store role; the source path is MatmulImpl Fixpipe and no separate fixpipe rate is declared by the control hardware",
    "GetLibApiWorkSpaceSize() is not declared by the supplied files, so no workspace bytes are claimed",
    "the wrapper that builds the semantic histogram is not supplied; the adapter is the review-approved compaction of nonempty bins",
)


def _require_int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{reason}:{value!r}")
    if value < minimum:
        raise ValueError(f"{reason}:{value!r}")
    return value


def _require_count(value: Any, reason: str) -> int:
    """A literal raw count may be nonpositive; terminating it is the kernel's job."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{reason}:{value!r}")
    return value


def _require_sequence(value: Any, reason: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{reason}:{value!r}")
    return tuple(value)


def _select_dispatch(params: Mapping[str, Any]) -> dict[str, Any]:
    """Reject every non-selected kernel branch instead of silently substituting."""
    implementation = params.get("implementation", _IMPLEMENTATION)
    if implementation != _IMPLEMENTATION:
        raise ValueError(f"unsupported_grouped_gemm_implementation:{implementation!r}")
    weight_layout = params.get("weight_layout", "ND")
    if weight_layout != "ND":
        raise ValueError(f"unsupported_grouped_gemm_weight_layout:{weight_layout!r}")
    for key in ("weight_format", "format_weight"):
        value = params.get(key)
        if value is not None and str(value) not in {"ND", "CubeFormat::ND"}:
            raise ValueError(f"unsupported_grouped_gemm_{key}:{value!r}")
    transpose_weight = params.get("transpose_weight", False)
    if bool(transpose_weight):
        raise ValueError("unsupported_grouped_gemm_transposed_weight:TILING_KEY_11")
    tiling_key = params.get("tiling_key", 10)
    if isinstance(tiling_key, bool) or int(tiling_key) != 10:
        raise ValueError(f"unsupported_grouped_gemm_tiling_key:{tiling_key!r}")
    dtype = str(params.get("dtype", "bf16")).strip().lower()
    if dtype not in {"bf16", "bfloat16"}:
        raise ValueError(f"unsupported_grouped_gemm_dtype:{dtype!r}")
    for key in ("quantization", "quant_mode", "quant"):
        value = params.get(key)
        if value is not None and str(value).strip().lower() not in {"none", "no_quant", "unquantized"}:
            raise ValueError(f"unsupported_grouped_gemm_quantization:{key}={value!r}")
    return {
        "implementation": _IMPLEMENTATION, "transpose_weight": False,
        "weight_layout": "ND", "weight_format": "ND", "tiling_key": 10,
        "tiling_key_is": "TILING_KEY_IS(10UL)", "format_weight": "CubeFormat::ND",
        "transpose_template": False,
        "template": "KernelMoeGMMNoQuant<T,T2,CubeFormat::ND,false>",
        "kernel": "moe_grouped_matmul", "quantization": "none", "dtype": "bf16",
        "group_list_mode": "key_value_2d",
        "input_adapter": "semantic_histogram_compacted_preserve_expert_ids",
        "raw_group_list_terminator": "first_row_count_nonpositive_breaks_remaining_list",
    }


def _parse_geometry_params(
    params: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], int, int, int, str]:
    expert_rows = _require_sequence(params.get("expert_rows"), "expert_rows_must_be_sequence")
    if not expert_rows:
        raise ValueError("expert_rows_must_not_be_empty")
    rows = tuple(_require_int(value, "expert_rows_must_be_nonnegative_int", minimum=0)
                 for value in expert_rows)
    total_m = sum(rows)
    if total_m == 0:
        # Host CalMMTiling recomputes baseM from m_=0, obtaining 0, and returns
        # GRAPH_FAILED at moe_grouped_matmul_cpu.cpp:200-208.
        raise ValueError("grouped_gemm_total_rows_zero_invalid_host_tiling")

    group_ids_raw = params.get("group_ids")
    if group_ids_raw is None:
        group_ids = tuple(range(len(rows)))
    else:
        group_ids = tuple(_require_int(value, "group_ids_must_be_nonnegative_int", minimum=0)
                          for value in _require_sequence(group_ids_raw, "group_ids_must_be_sequence"))
        if len(group_ids) != len(rows):
            raise ValueError("group_ids_length_must_match_expert_rows")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("group_ids_must_be_unique_histogram_bins")

    weight_experts_raw = params.get("weight_experts")
    weight_experts = (len(rows) if weight_experts_raw is None else
                      _require_int(weight_experts_raw,
                                   "weight_experts_must_be_positive_int", minimum=1))
    if any(value >= weight_experts for value in group_ids):
        raise ValueError("group_ids_must_address_declared_weight_experts")

    n = _require_int(params.get("n"), "n_must_be_positive_int", minimum=1)
    k = _require_int(params.get("k"), "k_must_be_positive_int", minimum=1)

    x_rows_raw = params.get("x_rows")
    if x_rows_raw is not None:
        x_rows = _require_int(x_rows_raw, "x_rows_must_be_nonnegative_int", minimum=0)
        if x_rows != total_m:
            raise ValueError(
                f"grouped_gemm_x_rows_must_equal_histogram_sum:{x_rows}:{total_m}")

    phase = str(params.get("phase", "prefill"))
    if phase not in {"prefill", "decode", "mixed"}:
        raise ValueError(f"unsupported_grouped_gemm_phase:{phase!r}")
    return rows, group_ids, weight_experts, n, k, phase


def compact_histogram(expert_rows: tuple[int, ...],
                      group_ids: tuple[int, ...] | None = None
                      ) -> tuple[tuple[int, int], ...]:
    """Semantic histogram -> literal raw pairs for nonempty bins.

    Empty histogram bins are skipped while expert ids are preserved.  This is
    deliberately different from the kernel's literal raw-list terminator: a
    zero histogram bin is an empty expert, not the end of the expert bank.
    """
    rows = tuple(_require_int(value, "histogram_count_must_be_nonnegative_int", minimum=0)
                 for value in expert_rows)
    ids = tuple(range(len(rows))) if group_ids is None else tuple(
        _require_int(value, "histogram_expert_id_must_be_nonnegative_int", minimum=0)
        for value in group_ids)
    if len(ids) != len(rows):
        raise ValueError("histogram_ids_length_must_match_counts")
    return tuple((expert_id, count) for expert_id, count in zip(ids, rows) if count > 0)


def pad_raw_group_list(raw_group_list: tuple[tuple[int, int], ...],
                       extent: int) -> tuple[tuple[int, int], ...]:
    """Pad a compacted raw pair list with zero-count terminator entries."""
    pairs = tuple((_require_int(expert_id, "raw_expert_id_must_be_nonnegative_int", minimum=0),
                   _require_int(count, "raw_active_count_must_be_positive_int", minimum=1))
                  for expert_id, count in raw_group_list)
    extent = _require_int(extent, "raw_declared_extent_must_be_positive_int", minimum=1)
    if len(pairs) > extent:
        raise ValueError(f"raw_group_list_exceeds_declared_extent:{len(pairs)}:{extent}")
    return pairs + ((0, 0),) * (extent - len(pairs))


def interpret_raw_group_list(
    raw_group_list: tuple[tuple[int, int], ...], *, n: int | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Literal kernel interpretation of 2-D ``(expert_id, direct_row_count)``.

    Counts are direct row counts, never cumulative offsets.  The first count
    <= 0 breaks the remaining list before its expert id is read, matching
    ``moe_grouped_matmul.h:95-100``.  This interpreter is intentionally
    separate from :func:`compact_histogram`: a histogram zero must not truncate
    later positive experts.
    """
    raw = _require_sequence(raw_group_list, "raw_group_list_must_be_sequence")
    pairs = []
    for item in raw:
        pair = _require_sequence(item, "raw_group_list_entry_must_be_pair")
        if len(pair) != 2:
            raise ValueError(f"raw_group_list_entry_must_be_pair:{pair!r}")
        pairs.append((_require_count(pair[0], "raw_expert_id_must_be_int"),
                      _require_count(pair[1], "raw_row_count_must_be_int")))
    pairs_tuple = tuple(pairs)
    if n is not None:
        n = _require_int(n, "raw_n_must_be_positive_int", minimum=1)
    if k is not None:
        k = _require_int(k, "raw_k_must_be_positive_int", minimum=1)

    processed: list[dict[str, Any]] = []
    prefix_rows = 0
    stop_index = len(pairs_tuple)
    stop_reason = "end_of_list"
    for index, (expert_id, count) in enumerate(pairs_tuple):
        if count <= 0:
            stop_index = index
            stop_reason = "split_value_nonpositive"
            break
        if expert_id < 0:
            raise ValueError(f"raw_processed_expert_id_must_be_nonnegative_int:{expert_id!r}")
        record: dict[str, Any] = {
            "index": index,
            "expert_id": expert_id,
            "rows": count,
            "x_row_offset": prefix_rows,
            "y_row_offset": prefix_rows,
        }
        if k is not None:
            record["x_element_offset"] = prefix_rows * k
        if n is not None:
            record["y_element_offset"] = prefix_rows * n
        processed.append(record)
        prefix_rows += count
    unprocessed = tuple(
        {"index": i, "expert_id": pairs_tuple[i][0], "rows": pairs_tuple[i][1]}
        for i in range(stop_index, len(pairs_tuple)))
    return {
        "processed": tuple(processed),
        "unprocessed": unprocessed,
        "processed_rows": prefix_rows,
        "stop_index": stop_index,
        "stop_reason": stop_reason,
        "counts_are_direct_not_cumulative": True,
    }


def _entry_record(index: int, histogram_index: int, expert_id: int, rows: int,
                  prefix_rows: int, n: int, k: int, n_tiles: int) -> dict[str, Any]:
    m_tiles = math.ceil(rows / _SINGLE_M)
    tasks = m_tiles * n_tiles
    a_bytes = rows * k * _ELEM_BYTES * n_tiles
    b_bytes = m_tiles * k * n * _ELEM_BYTES
    c_bytes = rows * n * _ELEM_BYTES
    return {
        "index": index,
        "histogram_index": histogram_index,
        "expert_id": expert_id,
        "rows": rows,
        "x_row_offset": prefix_rows,
        "y_row_offset": prefix_rows,
        "x_element_offset": prefix_rows * k,
        "y_element_offset": prefix_rows * n,
        "w_element_offset": expert_id * k * n,
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "tasks": tasks,
        "flops": 2 * rows * n * k,
        "a_bytes": a_bytes,
        "b_bytes": b_bytes,
        "c_bytes": c_bytes,
        "mte2_bytes": a_bytes + b_bytes,
        "mte3_bytes": c_bytes,
        "hbm_bytes": a_bytes + b_bytes + c_bytes,
    }


def _build_entries(histogram: tuple[tuple[int, int], ...], n: int, k: int,
                   n_tiles: int) -> tuple[tuple[dict[str, Any], ...],
                                          tuple[dict[str, Any], ...], int]:
    """Compact nonempty bins in histogram order; prefix offsets follow counts."""
    processed: list[dict[str, Any]] = []
    empty: list[dict[str, Any]] = []
    prefix_rows = 0
    for histogram_index, (expert_id, rows) in enumerate(histogram):
        if rows <= 0:
            empty.append({"histogram_index": histogram_index,
                          "expert_id": expert_id, "rows": rows})
            continue
        processed.append(_entry_record(len(processed), histogram_index, expert_id,
                                       rows, prefix_rows, n, k, n_tiles))
        prefix_rows += rows
    return tuple(processed), tuple(empty), prefix_rows


def _align_down_16(value: int) -> int:
    return value & ~15


def _align_up_16(value: int) -> int:
    return (value + 15) & ~15


def _positive_buffer(buffers: Mapping[str, int], name: str) -> int | None:
    if name not in buffers:
        return None
    value = int(buffers[name])
    if value <= 0:
        raise ValueError(f"host_buffer_capacity_must_be_positive:{name}={value}")
    return value


def host_tiling(op: OpSpec, geometry: Geometry) -> dict[str, Any]:
    """Reproduce the host tiling fields that are derivable from geometry.

    Platform capacities are read from ``Geometry.buffers``.  If the L0 buffers
    are absent, baseM/baseK remain unresolved rather than guessed; the fixed
    single_m/single_n/core_num path still drives the plan.  When the L0 and L1
    capacities are present, the same host formulas also determine the per-unit
    one-panel residency sizes used by the planner.
    """
    meta = dict(op.metadata)
    host = dict(meta.get("host", {}))
    if not host:
        raise ValueError("grouped_gemm_op_requires_make_op_metadata")
    total_m = int(host["total_m"])
    processed_m = int(host.get("processed_m", total_m))
    if processed_m != total_m:
        raise ValueError(
            f"grouped_gemm_missing_rows_not_accepted:{processed_m}:{total_m}")
    n_tiles = int(host["n_tiles"])
    aic_num = int(geometry.compute_units)
    if aic_num < 1:
        raise ValueError("grouped_gemm_requires_positive_compute_units")
    core_num = min(aic_num, total_m * n_tiles)
    buffers = dict(geometry.buffers)
    l0a = _positive_buffer(buffers, "l0a")
    l0b = _positive_buffer(buffers, "l0b")
    l0c = _positive_buffer(buffers, "l0c")
    l1 = _positive_buffer(buffers, "l1")
    _positive_buffer(buffers, "ub")

    result: dict[str, Any] = {
        "aic_num": aic_num,
        "core_num": core_num,
        "base_n": _BASE_N,
        "single_m": _SINGLE_M,
        "single_n": _SINGLE_N,
        "mm_dtype_bytes": _MM_DTYPE_BYTES,
        "total_m": total_m,
        "processed_m": processed_m,
        "n_tiles": n_tiles,
        "base_m": None,
        "base_k": None,
        "step_ka": None,
        "step_kb": None,
        "depth_a1": None,
        "depth_b1": None,
        "allocation_bytes": {},
        "status": "unresolved_l0_buffers",
    }
    if l1 is not None and l1 < _L1_PART_A:
        # moe_grouped_matmul_cpu.cpp:87-92 rejects L1 below 256 KiB.
        raise ValueError("host_tiling_l1_below_256k")
    if l0a is None or l0b is None or l0c is None:
        return result

    base_k_raw = (l0b // 2) // (_BASE_N * _MM_DTYPE_BYTES)
    base_k = _align_down_16(base_k_raw)
    if base_k == 0:
        # The host would divide by zero while deriving baseM at lines 196-199.
        raise ValueError("host_tiling_base_k_zero")
    max_base_m = l0c // (_BASE_N * _FP32_BYTES)
    base_m = min((l0a // 2) // (base_k * _MM_DTYPE_BYTES), max_base_m)
    base_m = _align_up_16(total_m) if base_m > total_m else _align_down_16(base_m)
    if base_m > _MAX_BASE_M:
        base_m = _MAX_BASE_M
    if base_m == 0:
        raise ValueError("host_tiling_base_m_zero")

    # Per-unit L0 residency is one host-derived panel.  The MatmulImpl
    # double-buffer schedule is not declared, so it is not invented here.
    allocation_bytes = {
        "l0a": float(base_m * base_k * _MM_DTYPE_BYTES),
        "l0b": float(_BASE_N * base_k * _MM_DTYPE_BYTES),
        "l0c": float(base_m * _BASE_N * _FP32_BYTES),
    }
    result.update({
        "base_m": base_m,
        "base_k": base_k,
        "max_base_m": max_base_m,
        "allocation_bytes": allocation_bytes,
        "status": "l0_derived",
    })
    if l1 is None:
        return result

    l1_a_size = _L1_PART_A if base_m > _BASE_N else l1 - _L1_PART_A
    l1_b_size = l1 - l1_a_size
    step_ka = (l1_a_size // 2) // (base_m * base_k * _MM_DTYPE_BYTES)
    step_kb = (l1_b_size // 2) // (_BASE_N * base_k * _MM_DTYPE_BYTES)
    if step_ka == 0 or step_kb == 0:
        raise ValueError("host_tiling_step_ka_or_kb_zero")
    if step_ka > step_kb:
        step_ka = step_ka // step_kb * step_kb
    elif step_ka < step_kb:
        step_kb = step_kb // step_ka * step_ka
    # One declared K-panel residency: full A and B step groups, no double buffer.
    l1_panel = (step_ka * base_m * base_k * _MM_DTYPE_BYTES
                + step_kb * _BASE_N * base_k * _MM_DTYPE_BYTES)
    allocation_bytes["l1"] = float(l1_panel)
    result.update({
        "step_ka": step_ka,
        "step_kb": step_kb,
        "depth_a1": step_ka * 2,
        "depth_b1": step_kb * 2,
        "l1_a_size": l1_a_size,
        "l1_b_size": l1_b_size,
        "allocation_bytes": allocation_bytes,
        "status": "complete",
    })
    return result


def make_op(params: dict) -> OpSpec:
    if not isinstance(params, Mapping):
        raise ValueError("grouped_gemm_params_must_be_mapping")
    dispatch = _select_dispatch(params)
    expert_rows, group_ids, weight_experts, n, k, phase = _parse_geometry_params(params)
    histogram = tuple(zip(group_ids, expert_rows))
    active_pairs = compact_histogram(expert_rows, group_ids)
    kernel_group_list = pad_raw_group_list(active_pairs, weight_experts)
    if len(active_pairs) < weight_experts:
        terminator_index = len(active_pairs)
        terminator_reason = "split_value_nonpositive"
    else:
        terminator_index = weight_experts
        terminator_reason = "end_of_list"
    n_tiles = math.ceil(n / _SINGLE_N)
    processed, empty_entries, processed_m = _build_entries(histogram, n, k, n_tiles)
    total_m = sum(expert_rows)
    if processed_m != total_m:
        raise ValueError(f"grouped_gemm_missing_rows_not_accepted:{processed_m}:{total_m}")
    padded_m = sum(entry["m_tiles"] * _SINGLE_M for entry in processed)
    padded_n = math.ceil(n / _SINGLE_N) * _SINGLE_N
    kernel_task_num = sum(entry["tasks"] for entry in processed)

    metadata = (
        ("dispatch", dispatch),
        ("implementation", dict(dispatch)),
        ("source_identity", _source_identity()),
        ("host", {
            "base_n": _BASE_N,
            "single_m": _SINGLE_M,
            "single_n": _SINGLE_N,
            "total_m": total_m,
            "processed_m": processed_m,
            "missing_rows": total_m - processed_m,
            "n_tiles": n_tiles,
            "task_num": total_m * n_tiles,
            "kernel_task_num": kernel_task_num,
            "dtype_bytes": _ELEM_BYTES,
        }),
        ("histogram_entries", histogram),
        ("histogram_shape", (len(expert_rows), 2)),
        ("histogram_empty_entries", empty_entries),
        ("histogram_positive_entries", active_pairs),
        ("raw_group_list", kernel_group_list),
        ("raw_group_list_active", active_pairs),
        ("raw_group_list_extent", weight_experts),
        ("raw_group_list_padding_entries", weight_experts - len(active_pairs)),
        ("raw_group_list_terminator_index", terminator_index),
        ("raw_group_list_terminator_reason", terminator_reason),
        ("kernel_group_list", kernel_group_list),
        ("raw_group_list_semantics", {
            "pairs": "(expert_id, direct_row_count)",
            "counts_are_direct_not_cumulative": True,
            "break": "first_row_count_nonpositive_terminates_remaining_list",
            "row_offsets": "prefix_sum_of_preceding_positive_direct_counts",
        }),
        ("group_list_shape", (weight_experts, 2)),
        ("group_list_entries", weight_experts),
        ("group_list_element_bytes", None),
        ("expert_rows", expert_rows),
        ("group_ids", group_ids),
        ("weight_experts", weight_experts),
        ("input_rows", total_m),
        ("processed_m", processed_m),
        ("missing_rows", total_m - processed_m),
        ("stop_index", terminator_index),
        ("stop_reason", terminator_reason),
        ("processed_entries", processed),
        ("unprocessed_entries", ()),
        ("weight_shape", (weight_experts, k, n)),
        ("weight_stride_elements", (k * n, n, 1)),
        ("n", n),
        ("k", k),
        ("element_bytes", _ELEM_BYTES),
        ("unresolved", _UNRESOLVED),
    )
    notes = (
        "source scope: current checkout, manifest hash match; not a historical capture image",
        "semantic histogram adapter compacts nonempty bins preserving expert ids; input rows equal processed rows",
        "literal raw group list is direct counts and terminates at the first count <= 0; terminator padding retains the declared weight-bank extent",
        "exact logical 2*rows*cols*k per MMCompute tile; MatmulImpl fractal padding is unresolved",
        "each real source core id is mapped over the full physical compute width; L1/L0 residency uses the same physical partition of declared pool units",
        "weight bytes are charged per M tile; no L2 hit is assumed",
        "host tiling CPU time, group-list scalar loads and MTE1/Fixpipe service are not priced",
        "no historical grouped-GEMM capture row was found in the supplied evidence; no capture dispatch or timing is claimed",
    )
    return OpSpec(
        op_id="grouped_gemm",
        family="grouped_gemm",
        phase=phase,
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(total_m, n, k),
        local_shape=(processed_m, n, k),
        padded_shape=(padded_m, padded_n, k),
        launch_shape=(total_m, n, k),
        dtype="bf16",
        layout="ND",
        recipe="grouped_expert_gemm", source_boundary="kernel",
        metadata=metadata,
        notes=notes,
    )


def _total_work(entries: tuple[dict[str, Any], ...]) -> Work:
    flops = float(sum(entry["flops"] for entry in entries))
    hbm = float(sum(entry["hbm_bytes"] for entry in entries))
    mte2 = float(sum(entry["mte2_bytes"] for entry in entries))
    mte3 = float(sum(entry["mte3_bytes"] for entry in entries))
    resources = {role: value for role, value in (("hbm", hbm), ("mte2", mte2),
                                                ("mte3", mte3)) if value}
    return Work(matrix={"bf16": flops} if flops else {}, resources=resources)


def _core_spans(entries: tuple[dict[str, Any], ...],
                core_num: int) -> dict[int, tuple[tuple[int, int, int, int], ...]]:
    """Static kernel assignment: task t of a group goes to (t + count) % core."""
    if core_num <= 0:
        return {}
    spans: dict[int, list[tuple[int, int, int]]] = {}
    cumulative = 0
    for group_index, entry in enumerate(entries):
        tasks = int(entry["tasks"])
        start = cumulative % core_num
        # Every r < min(tasks, core) owns t=r, r+core, ...; at most core spans.
        for first in range(min(tasks, core_num)):
            core = (first + start) % core_num
            count = (tasks - 1 - first) // core_num + 1
            spans.setdefault(core, []).append((group_index, first, count))
        cumulative += tasks
    packed: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    for core, items in spans.items():
        rows, next_index = [], 0
        for group_index, first, count in items:
            rows.append((group_index, first, count, next_index))
            next_index += count
        packed[core] = tuple(rows)
    return packed


def _tile_geometry(entry: Mapping[str, Any], m_index: int, n_index: int,
                   n: int) -> tuple[int, int, int, int]:
    rows = min(_SINGLE_M, int(entry["rows"]) - m_index * _SINGLE_M)
    cols = min(_SINGLE_N, n - n_index * _SINGLE_N)
    if rows <= 0 or cols <= 0:
        raise ValueError("grouped_gemm_tile_geometry_invalid")
    return rows, cols, m_index, n_index


def _pin(context, role: str, core: int) -> dict[str, tuple[int, ...]]:
    """Map real source core ``core`` onto its full-width physical units.

    ``GetBlockIdx()`` is a launched id in ``0..core_num-1`` (``.h:72-74``),
    while the hardware partition is declared over ``context.core_count``.
    Mapping through the full width keeps a shared copy unit shared; only
    launched source ids are mapped, so unused physical units stay idle.
    """
    if context.core_count < 1:
        raise ValueError("grouped_gemm_physical_compute_units_must_be_positive")
    if role in context.resource_names:
        if int(context.resource_units.get(role, context.core_count)) < 1:
            raise ValueError(f"grouped_gemm_resource_units_must_be_positive:{role}")
    return context.owned(core, context.core_count, (role,))


def _buffer_units(context, pool: str, core: int) -> tuple[int, ...]:
    """Map source core ``core`` onto its declared physical pool units.

    Mirrors :func:`_pin`; every listed unit is charged the allocation size, so
    cores sharing one declared unit contend for that capacity.  Incompatible
    physical geometries raise instead of being silently averaged.
    """
    if context.core_count < 1:
        raise ValueError("grouped_gemm_physical_compute_units_must_be_positive")
    units = int(context.pool_units.get(pool, context.core_count))
    if units < 1:
        raise ValueError(f"grouped_gemm_buffer_units_must_be_positive:{pool}:{units}")
    try:
        return unit_group(core, context.core_count, units)
    except ValueError as exc:
        raise ValueError(
            f"grouped_gemm_buffer_unit_mapping_incompatible:"
            f"{pool}:{units}:{context.core_count}") from exc


def _allocation(context, pool: str, sizes: Mapping[str, float], core: int):
    size = sizes.get(pool)
    if not size or pool not in context.pool_names:
        return None
    return context.allocate(f"{pool}{core}", pool, size,
                            _buffer_units(context, pool, core))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    dispatch = dict(meta.get("dispatch", {}))
    if dispatch.get("implementation") != _IMPLEMENTATION:
        raise ValueError("grouped_gemm_build_requires_selected_make_op_dispatch")
    if (dispatch.get("tiling_key") != 10 or dispatch.get("transpose_weight") is not False
            or dispatch.get("weight_layout") != "ND" or dispatch.get("quantization") != "none"):
        raise ValueError("grouped_gemm_build_requires_selected_dispatch_flags")
    if op.dtype != "bf16" or op.layout != "ND":
        raise ValueError("grouped_gemm_build_requires_bf16_nd")
    if not {"processed_entries", "n", "k", "host", "input_rows"}.issubset(meta):
        raise ValueError("grouped_gemm_build_requires_make_op_metadata")
    entries = tuple(meta["processed_entries"])
    total_m = int(meta["input_rows"])
    processed_m = int(meta["processed_m"])
    if processed_m != total_m or int(meta.get("missing_rows", 0)) != 0:
        raise ValueError(
            f"grouped_gemm_missing_rows_not_accepted:{processed_m}:{total_m}")
    n = int(meta["n"])
    k = int(meta["k"])
    host = host_tiling(op, geometry)
    core_num = int(host["core_num"])
    if entries and core_num < 1:
        raise ValueError("grouped_gemm_requires_positive_core_num")
    sizes = dict(host.get("allocation_bytes", {}))
    work = _total_work(entries)
    spans = _core_spans(entries, core_num)

    def program(context):
        if context.core_count < core_num:
            raise ValueError("grouped_gemm_active_core_num_exceeds_physical:"
                             f"{core_num}:{context.core_count}")
        if not spans:
            return ParallelProgram(())
        programs = []
        for core in sorted(spans):
            core_spans = spans[core]
            starts = tuple(span[3] for span in core_spans)
            count = starts[-1] + core_spans[-1][2]

            def tile(index, core_spans=core_spans, starts=starts, core=core):
                position = bisect_right(starts, index) - 1
                group_index, first, _count, first_index = core_spans[position]
                task = first + (index - first_index) * core_num
                entry = entries[group_index]
                m_index, n_index = divmod(task, int(entry["n_tiles"]))
                rows, cols, _, _ = _tile_geometry(entry, m_index, n_index, n)
                a_bytes = rows * k * _ELEM_BYTES
                b_bytes = k * cols * _ELEM_BYTES
                c_bytes = rows * cols * _ELEM_BYTES
                flops = 2 * rows * cols * k
                l1 = _allocation(context, "l1", sizes, core)
                l0a = _allocation(context, "l0a", sizes, core)
                l0b = _allocation(context, "l0b", sizes, core)
                l0c = _allocation(context, "l0c", sizes, core)
                return (
                    Stage("load", context.bind(Work(resources={
                        "hbm": a_bytes + b_bytes,
                        "mte2": a_bytes + b_bytes,
                    })), acquire=(l1,) if l1 else (),
                        placement=_pin(context, "mte2", core)),
                    Stage("compute", context.bind(Work(matrix={"bf16": flops})),
                          after=("load",),
                          acquire=tuple(item for item in (l0a, l0b, l0c) if item),
                          free=tuple(item for item in (l1, l0a, l0b) if item),
                          placement=_pin(context, "cube", core)),
                    Stage("store", context.bind(Work(resources={
                        "hbm": c_bytes,
                        "mte3": c_bytes,
                    })), after=("compute",), free=(l0c,) if l0c else (),
                        placement=_pin(context, "mte3", core)),
                )

            programs.append(PipelineProgram(count, 1, tile, 0.0, 0.0))
        return ParallelProgram(tuple(programs))

    return OperatorPlan(work, program)


def iter_tiles(op: OpSpec, core_num: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield task-level addresses lazily for independent address oracles.

    This helper is intentionally O(output tiles) when consumed; ``build`` itself
    uses aggregate work and per-core spans and never calls it.
    """
    meta = dict(op.metadata)
    entries = tuple(meta["processed_entries"])
    n = int(meta["n"])
    k = int(meta["k"])
    core_count = None if core_num is None else int(core_num)
    if core_count is not None and core_count < 1:
        raise ValueError(f"grouped_gemm_core_num_must_be_positive:{core_num!r}")
    cumulative = 0
    for group_index, entry in enumerate(entries):
        tasks = int(entry["tasks"])
        start = 0 if core_count is None else cumulative % core_count
        for task in range(tasks):
            m_index, n_index = divmod(task, int(entry["n_tiles"]))
            rows, cols, _, _ = _tile_geometry(entry, m_index, n_index, n)
            yield {
                "group_index": group_index,
                "histogram_index": entry["histogram_index"],
                "expert_id": entry["expert_id"],
                "task_index": task,
                "core": None if core_count is None else (task + start) % core_count,
                "m_index": m_index,
                "n_index": n_index,
                "tile_rows": rows,
                "tile_cols": cols,
                "x_row_offset": entry["x_row_offset"] + m_index * _SINGLE_M,
                "y_row_offset": entry["y_row_offset"] + m_index * _SINGLE_M,
                "x_element_offset": (entry["x_row_offset"] + m_index * _SINGLE_M) * k,
                "y_element_offset": ((entry["y_row_offset"] + m_index * _SINGLE_M) * n
                                     + n_index * _SINGLE_N),
                "w_element_offset": entry["w_element_offset"] + n_index * _SINGLE_N,
                "weight_row_stride": n,
                "flops": 2 * rows * cols * k,
                "a_bytes": rows * k * _ELEM_BYTES,
                "b_bytes": k * cols * _ELEM_BYTES,
                "c_bytes": rows * cols * _ELEM_BYTES,
            }
        cumulative += tasks


__all__ = ["build", "compact_histogram", "host_tiling", "interpret_raw_group_list",
           "iter_tiles", "make_op", "pad_raw_group_list"]
