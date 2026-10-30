"""Independent routing, padding, address and work oracles for MoE alignment.

The planner is source-derived, not capture-fitted.  These checks establish the
source dispatch threshold, exact per-expert histogram and prefix, padded row
layout, small-batch thread-partition scatter order, expert-map filtering, the
small-batch three-barrier dependency, the source residual-fill writer coverage,
the two physical kernel launches on the large path, per-CTA ownership and the
public execution work ledger.  The unsupported post-gather wrapper path must be
rejected at ``make_op``; no unowned executable fallback is exercised.  These
are structural/arithmetic checks, not an accuracy claim against any device
measurement.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import ExecutionError
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe
from mindie_sim.prediction.operators import moe_align as moe_align_module
from mindie_sim.prediction.operators.moe_align import (
    build,
    iter_blocks,
    kernels,
    make_op,
    stage_work,
)
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeBook,
    RecipeContext,
    Stage,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.run_operator_cases import control_case, run

ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "tests/fixtures/operator_cases/moe_align.json"
MANIFEST_FILE = ROOT / "tests/fixtures/operator_sources/moe_align.json"
WRAPPER = Path(
    "/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/"
    "fused_moe/moe_align_block_size.py")
CUDA = Path(
    "/Users/maoxx241/code/vllm-v0.25.1/csrc/libtorch_stable/moe/"
    "moe_align_sum_kernels.cu")
MATH = Path("/Users/maoxx241/code/vllm-v0.25.1/vllm/utils/math_utils.py")

IMPLEMENTATION = "cuda_moe_align_block_size"
INT32 = 4
WARP = 32


def cases():
    return json.loads(CASE_FILE.read_text())["cases"]


def params_for(topk_ids, num_experts, block_size, **extra):
    values = {
        "topk_ids": list(topk_ids),
        "num_experts": num_experts,
        "block_size": block_size,
        "expert_map": None,
        "implementation": IMPLEMENTATION,
        "pad_sorted_ids": True,
    }
    values.update(extra)
    return values


def reference(params):
    """Independent NumPy routing/histogram/padding interpreter."""
    rows = list(params["topk_ids"])
    flat = np.asarray(rows, dtype=np.int64).reshape(-1)
    num_experts = int(params["num_experts"])
    block_size = int(params["block_size"])
    numel = int(flat.size)
    ignore = bool(params.get("ignore_invalid_experts", False))
    raw_map = params.get("expert_map")
    expert_map = None if raw_map is None else np.asarray(raw_map, dtype=np.int64)
    if expert_map is not None and not ignore:
        raise ValueError("post_gather_is_not_a_supported_oracle_input")
    has_map = expert_map is not None
    dtype_bytes = {
        "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
        "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
    }[params.get("topk_ids_dtype", "int32")]
    if has_map:
        mapped = expert_map[flat]
        valid = mapped >= 0
        expert_of = np.where(valid, mapped, -1)
    else:
        valid = np.ones(numel, dtype=bool)
        expert_of = flat.copy()
    counts = np.bincount(expert_of[valid], minlength=num_experts).astype(np.int64)
    padded = (counts + block_size - 1) // block_size * block_size
    cumsum = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
    total_padded = int(cumsum[-1])
    used_blocks = total_padded // block_size
    max_tokens_padded = numel + num_experts * (block_size - 1)
    if params.get("pad_sorted_ids", False):
        max_tokens_padded = (
            (max_tokens_padded + block_size - 1) // block_size * block_size)
    if numel < num_experts:
        max_tokens_padded = min(numel * block_size, max_tokens_padded)
    max_m_blocks = (max_tokens_padded + block_size - 1) // block_size
    small = numel < 1024 and num_experts <= 64
    stride = max(num_experts, WARP)
    tail_blocks = max_m_blocks - used_blocks
    # Source lines 272-276 are outside if (tid < num_experts); all stride
    # threads execute the residual loop.
    tail_written = tail_blocks
    valid_entries = int(valid.sum())
    sort_ctas = 0 if small else min((numel + 255) // 256, 65535)
    sort_ids = np.zeros(0, dtype=np.int64)
    sort_valid = np.zeros(0, dtype=np.int64)
    if sort_ctas:
        flat_index = np.arange(numel, dtype=np.int64)
        cta = (flat_index // 256) % sort_ctas
        sort_ids = np.bincount(cta, minlength=sort_ctas).astype(np.int64)
        sort_valid = np.bincount(
            cta[valid], minlength=sort_ctas).astype(np.int64)
    workspace = {
        "sorted_token_ids": max_tokens_padded * INT32,
        "expert_ids": max_m_blocks * INT32,
        "num_tokens_post_pad": INT32,
        "cumsum_buffer": 0 if small else (num_experts + 1) * INT32,
    }
    return {
        "flat": flat,
        "num_experts": num_experts,
        "block_size": block_size,
        "numel": numel,
        "counts": counts,
        "padded": padded,
        "cumsum": cumsum,
        "total_padded": total_padded,
        "used_blocks": used_blocks,
        "max_tokens_padded": max_tokens_padded,
        "max_m_blocks": max_m_blocks,
        "small": small,
        "stride": stride,
        "tail_blocks": tail_blocks,
        "tail_written": tail_written,
        "valid": valid,
        "expert_of": expert_of,
        "valid_entries": valid_entries,
        "expert_map": expert_map,
        "has_map": has_map,
        "dtype_bytes": dtype_bytes,
        "sort_ctas": sort_ctas,
        "sort_ids": sort_ids,
        "sort_valid": sort_valid,
        "workspace": workspace,
    }


def reference_stage_work(ref):
    """Independent byte/operation ledger mirroring the source loops."""
    numel = int(ref["numel"])
    num_experts = int(ref["num_experts"])
    max_tokens = int(ref["max_tokens_padded"])
    max_blocks = int(ref["max_m_blocks"])
    valid = int(ref["valid_entries"])
    topk_pass = numel * ref["dtype_bytes"]
    map_pass = numel * INT32 if ref["has_map"] else 0
    init = max_tokens * INT32
    scatter = valid * INT32
    if ref["small"]:
        expert_blocks = int(ref["used_blocks"]) + int(ref["tail_written"])
        expert_bytes = expert_blocks * INT32
        stages = {
            "init_sorted": {"hbm": init, "mte3": init},
            "clear_counts": {"vector": ref["stride"] * num_experts},
            "histogram": {
                "hbm": topk_pass + map_pass,
                "mte2": topk_pass + map_pass,
                "vector": numel,
            },
            "prefix": {
                "hbm": INT32, "mte3": INT32,
                "vector": ref["stride"] * num_experts + 2 * num_experts,
            },
            "expert_ids": {
                "hbm": expert_bytes, "mte3": expert_bytes,
                "vector": expert_blocks,
            },
            "scatter": {
                "hbm": topk_pass + map_pass + scatter,
                "mte2": topk_pass + map_pass,
                "mte3": scatter,
                "vector": numel,
            },
        }
    else:
        cumsum_write = (num_experts + 1) * INT32
        cumsum_fill = (2 * num_experts + 1024) * INT32
        expert_bytes = max_blocks * INT32
        stages = {
            "init_sorted": {"hbm": init, "mte3": init},
            "clear_counts": {
                "vector": ((num_experts + WARP - 1) // WARP) * WARP},
            "histogram": {
                "hbm": topk_pass + map_pass,
                "mte2": topk_pass + map_pass,
                "vector": numel,
            },
            # CUB BlockScan instruction work is deliberately not declared.
            "scan": {"hbm": cumsum_write + INT32, "mte3": cumsum_write + INT32},
            "expert_ids": {
                "hbm": expert_bytes + cumsum_fill,
                "mte2": cumsum_fill,
                "mte3": expert_bytes,
                "vector": max_blocks,
            },
            "sort": {
                "hbm": topk_pass + map_pass + 3 * scatter,
                "mte2": topk_pass + map_pass + scatter,
                "mte3": 2 * scatter,
                "vector": numel,
            },
        }
    return stages


def totals(stages):
    result = {}
    for stage in stages.values():
        for role, amount in stage.items():
            result[role] = result.get(role, 0) + amount
    return result


def reference_sorted_ids_small(ref, sentinel):
    """Small-batch deterministic scatter: thread residue, then iteration."""
    result = np.full(int(ref["max_tokens_padded"]), sentinel, dtype=np.int64)
    for expert in range(ref["num_experts"]):
        indices = np.flatnonzero(ref["valid"] & (ref["expert_of"] == expert))
        order = np.lexsort((indices // ref["stride"], indices % ref["stride"]))
        start = int(ref["cumsum"][expert])
        result[start:start + indices.size] = indices[order]
    return result


def reference_expert_ids(ref):
    """Block-to-expert map and the exact source residual-thread coverage."""
    block_experts = np.full(int(ref["max_m_blocks"]), -1, dtype=np.int64)
    for expert in range(ref["num_experts"]):
        first = int(ref["cumsum"][expert]) // ref["block_size"]
        blocks = int(ref["padded"][expert]) // ref["block_size"]
        block_experts[first:first + blocks] = expert
    written = np.zeros(int(ref["max_m_blocks"]), dtype=bool)
    written[:ref["used_blocks"]] = True
    writers = ref["stride"] if ref["small"] else 1024
    for tid in range(writers):
        position = ref["used_blocks"] + tid
        while position < ref["max_m_blocks"]:
            written[position] = True
            position += writers
    return block_experts, written


def collect_leaf_stages(program):
    """Static traversal of finite nested programs, including bounded tiles."""
    result = []
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            result.append(stage)
            if stage.program is not None:
                result.extend(collect_leaf_stages(stage.program))
    elif isinstance(program, ParallelProgram):
        for child in program.programs:
            result.extend(collect_leaf_stages(child))
    elif isinstance(program, PipelineProgram):
        indices = {0, program.count - 1} if program.count else set()
        for index in sorted(indices):
            for stage in program.tile(index):
                result.append(stage)
                if stage.program is not None:
                    result.extend(collect_leaf_stages(stage.program))
    return result


def recipe_context(op, *, owners=32, vector_units=64, mte_units=32):
    names = {"vector": "vector@r0:s0", "mte2": "mte2@r0:s0",
             "mte3": "mte3@r0:s0", "hbm": "memory@r0:s0"}
    return RecipeContext(
        op, {}, names,
        {name: 1e9 for name in names.values()}, owners,
        pool_names={}, resource_units={"vector": vector_units,
                                       "mte2": mte_units, "mte3": mte_units},
        compute_peaks={"bf16": 1e12, "fp32": 1e12}, pool_units={},
    )


def execution_for(op, *, launch_us=50, units=32, split_copy=True, shared=None,
                  buffer_units=32, scope="none"):
    raw = control_case(units=units, split_copy=split_copy)
    raw["machine"]["device"]["launch_us"] = launch_us
    if shared is not None:
        raw["machine"]["device"]["buffers"] = {"shared": shared}
        raw["machine"]["device"]["buffer_units"] = {"shared": buffer_units}
    case = parse_case(raw)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(build, kernels)})
    execution = Execution(case, CostModel(case), recipes=recipes, trace_limit=10000)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one",
                              replay_scope=scope)
    scheduled = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(scheduled.completion,))
    return execution, scheduled


def submitted_tasks(execution):
    return [entry for entry in execution.engine.trace
            if entry["task"].endswith(":submit")]


def unit_case(units):
    """Fixed per-unit vector rate; global memory is nonbinding."""
    return {
        "machine": {"name": "per-unit", "num_devices": 1, "device": {
            "memory_bytes": 1024**3, "compute_units": units,
            "compute_peaks": {"bf16": 1e12, "fp32": 1e12},
            "rates": {"memory": 1e18, "vector": 1e9 * units},
            "resource_units": {"vector": units},
        }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def test_campaign_cases_select_the_declared_source_path():
    loaded = cases()
    assert {case["split"] for case in loaded} == {"development", "heldout_geometry"}
    for case in loaded:
        op = make_op(case["params"])
        dispatch = dict(op.metadata)["dispatch"]
        assert {key: dispatch[key] for key in case["expected_dispatch"]} == \
            case["expected_dispatch"]
        assert dispatch["implementation"] == IMPLEMENTATION
        assert dispatch["pad_sorted_ids"] is True
        assert dispatch["path"] == "small_batch_expert"
        assert dispatch["symbol"].endswith("moe_align_block_size_small_batch_expert_kernel")
        assert dispatch["sort_symbol"] is None
        assert dispatch["output_dtype"] == "int32"
        assert dispatch["expert_map_dtype"] is None
        assert dispatch["kernel_phases"] == ("small_batch_expert",)
        assert dispatch["host_submissions"] == 1
        assert op.recipe == IMPLEMENTATION


def test_unselected_implementation_and_invalid_inputs_are_rejected():
    base = params_for([[0, 1]], 4, 16)
    for key, value in (
        ("implementation", "triton_moe_align"),
        ("pad_sorted_ids", 1),
        ("ignore_invalid_experts", 0),
        ("num_experts", 0),
        ("num_experts", 993),
        ("block_size", 0),
        ("phase", "other"),
        ("topk_ids_dtype", "float32"),
    ):
        with pytest.raises(ValueError):
            make_op({**base, key: value})
    with pytest.raises(ValueError, match="at_least_one_token"):
        make_op({**base, "topk_ids": []})
    with pytest.raises(ValueError, match="top_k_must_be_positive"):
        make_op({**base, "topk_ids": [[]]})
    with pytest.raises(ValueError, match="constant_top_k"):
        make_op({**base, "topk_ids": [[0, 1], [2]]})
    with pytest.raises(ValueError, match="below_num_experts"):
        make_op({**base, "topk_ids": [[0, 4]]})
    with pytest.raises(ValueError, match="nonnegative"):
        make_op({**base, "topk_ids": [[0, -1]]})
    with pytest.raises(ValueError, match="expert_map_length"):
        make_op({**base, "expert_map": [0]})
    with pytest.raises(ValueError, match="expert_map_local_id"):
        make_op({**base, "expert_map": [0, 1, 2, 4]})
    with pytest.raises(ValueError, match="expert_map_values"):
        make_op({**base, "expert_map": [0, 1, 2, -2]})
    # Wrapper lines 100-103 add an ATen advanced-indexing launch that is not
    # in the supplied native source.  It must be rejected, not left unowned.
    with pytest.raises(ValueError, match="post_gather_not_supported:requires_native_layout"):
        make_op({**base, "expert_map": [0, 1, 2, 3],
                 "ignore_invalid_experts": False})
    with pytest.raises(ValueError, match="params_must_be_mapping"):
        make_op([])  # type: ignore[arg-type]


def test_wrapper_buffer_sizes_padding_and_workspace_match_source_formulas():
    samples = (
        ([[0, 1]], 4, 16, True),
        ([[0, 1]], 4, 16, False),
        ([[0], [0], [0]], 2, 32, True),
        ([[0], [1], [2]], 4, 16, True),
        ([[i % 8] for i in range(33)], 8, 8, True),
        ([[i % 65] for i in range(1024)], 65, 16, True),
        ([[i % 4] for i in range(8)], 4, 8, False),
    )
    for topk_ids, experts, block, pad in samples:
        params = params_for(topk_ids, experts, block, pad_sorted_ids=pad)
        ref = reference(params)
        op = make_op(params)
        meta = dict(op.metadata)
        analysis = meta["analysis"]
        assert meta["counts"] == tuple(int(x) for x in ref["counts"])
        assert meta["padded_per_expert"] == tuple(int(x) for x in ref["padded"])
        assert meta["cumsum"] == tuple(int(x) for x in ref["cumsum"])
        assert analysis["total_padded"] == ref["total_padded"]
        assert analysis["max_tokens_padded"] == ref["max_tokens_padded"]
        assert analysis["max_m_blocks"] == ref["max_m_blocks"]
        assert analysis["used_blocks"] == ref["used_blocks"]
        assert analysis["sentinel"] == ref["numel"]
        assert analysis["workspace_bytes"] == ref["workspace"]
        assert op.padded_shape == (ref["total_padded"],)
        assert op.launch_shape == (ref["max_tokens_padded"],)
        assert ref["total_padded"] % block == 0
        assert ref["total_padded"] <= ref["max_tokens_padded"]
        assert ref["max_m_blocks"] >= ref["used_blocks"]
        assert sum(ref["counts"]) == ref["valid_entries"]


def test_histogram_prefix_and_small_scatter_match_numpy_address_oracle():
    samples = (
        ([[0, 1]], 4, 16),
        ([[2, 0], [2, 1], [2, 0]], 4, 16),
        ([[0], [0], [0]], 2, 32),
        ([[3, 0], [1, 3]], 8, 16),
        ([[1, 1], [0, 2], [2, 3], [3, 2]], 4, 32),
        ([[7], [7], [1], [0], [7], [2]], 8, 4),
    )
    for topk_ids, experts, block in samples:
        params = params_for(topk_ids, experts, block)
        ref = reference(params)
        op = make_op(params)
        meta = dict(op.metadata)
        assert meta["counts"] == tuple(int(x) for x in ref["counts"])
        assert meta["cumsum"] == tuple(int(x) for x in ref["cumsum"])
        assert dict(meta["analysis"])["valid_entries"] == ref["valid_entries"]
        expected = reference_sorted_ids_small(ref, ref["numel"])
        block_experts, written = reference_expert_ids(ref)
        assert written.all()
        assert int(written.sum()) == ref["max_m_blocks"]
        # Sentinel positions and each expert's exact prefix range.
        assert np.all(expected[ref["total_padded"]:] == ref["numel"])
        for expert in range(experts):
            start = int(ref["cumsum"][expert])
            end = start + int(ref["counts"][expert])
            expected_members = set(np.flatnonzero(
                ref["valid"] & (ref["expert_of"] == expert)).tolist())
            assert set(expected[start:end].tolist()) == expected_members
            assert np.all(expected[end:int(ref["cumsum"][expert + 1])] ==
                          ref["numel"])
            assert np.all(block_experts[ref["cumsum"][expert] // block:
                                        ref["cumsum"][expert + 1] // block] == expert)
        # The declared block ranges describe exactly the same prefix layout.
        declared = list(iter_blocks(op))
        expected_blocks = [
            (expert, start, int(ref["padded"][expert]))
            for expert, start in enumerate(ref["cumsum"][:-1])
            if ref["padded"][expert]
        ]
        assert [(row["expert"], row["start_row"], row["padded_rows"])
                for row in declared] == expected_blocks
        for row in declared:
            assert row["blocks"] == row["padded_rows"] // block
            assert row["first_block"] == row["start_row"] // block


def test_large_path_scatter_rank_ranges_and_sentinel_holes():
    topk_ids = [[index % 65] for index in range(4096)]
    params = params_for(topk_ids, 65, 16)
    ref = reference(params)
    assert not ref["small"]
    op = make_op(params)
    meta = dict(op.metadata)
    assert meta["counts"] == tuple(int(x) for x in ref["counts"])
    # The atomic scatter order is not claimed, but every expert's rank range
    # is fixed by the prefix and every non-counted slot must stay sentinel.
    layout = np.full(int(ref["max_tokens_padded"]), ref["numel"], dtype=np.int64)
    covered = []
    for expert in range(ref["num_experts"]):
        indices = np.flatnonzero(ref["valid"] & (ref["expert_of"] == expert))
        start = int(ref["cumsum"][expert])
        end = start + int(ref["counts"][expert])
        assert end == int(ref["cumsum"][expert + 1]) - (
            int(ref["padded"][expert]) - int(ref["counts"][expert]))
        # A valid atomic order exists in this fixed rank window; use the flat
        # order as one concrete witness and check every address is in range.
        layout[start:end] = indices
        assert np.all(layout[start:end] < ref["numel"])
        covered.extend(indices.tolist())
        assert np.all(layout[end:int(ref["cumsum"][expert + 1])] == ref["numel"])
    assert sorted(covered) == list(range(ref["numel"]))
    assert int(ref["cumsum"][-1]) <= int(ref["max_tokens_padded"])
    assert np.all(layout[int(ref["cumsum"][-1]):] == ref["numel"])
    block_experts, written = reference_expert_ids(ref)
    assert written.all()
    assert int(written.sum()) == ref["max_m_blocks"]
    for expert in range(ref["num_experts"]):
        first = int(ref["cumsum"][expert]) // ref["block_size"]
        blocks = int(ref["padded"][expert]) // ref["block_size"]
        assert np.all(block_experts[first:first + blocks] == expert)
    assert np.all(block_experts[ref["used_blocks"]:] == -1)


def test_small_batch_scatter_order_is_thread_partition_not_flat_token_order():
    # All 70 ids select expert 0.  Thread tid owns ids tid, tid+32, tid+64,
    # so the kernel-scattered order is 0,32,64,1,33,65,...
    params = params_for([[0] for _ in range(70)], 4, 4, pad_sorted_ids=False)
    ref = reference(params)
    assert ref["small"]
    expected = reference_sorted_ids_small(ref, ref["numel"])
    assert expected[:6].tolist() == [0, 32, 64, 1, 33, 65]
    assert expected[:6].tolist() != [0, 1, 2, 3, 4, 5]
    assert set(expected[:70].tolist()) == set(range(70))
    assert np.all(expected[70:] == 70)


def test_dispatch_thresholds_and_launch_geometry_follow_the_host_expression():
    generated = {
        "small_1023_64": (1023, 64, 1),
        "big_1024_64": (1024, 64, 1),
        "big_1023_65": (1023, 65, 1),
        "big_1_65": (1, 65, 1),
        "small_1_64": (1, 64, 1),
        "big_1024_992": (1024, 992, 1),
    }
    for label, (numel, experts, block) in generated.items():
        topk_ids = [[index % experts] for index in range(numel)]
        params = params_for(topk_ids, experts, block)
        op = make_op(params)
        dispatch = dict(op.metadata)["dispatch"]
        ref = reference(params)
        assert dispatch["small_batch_expert_mode"] == ref["small"], label
        assert dispatch["numel_below_1024"] == (numel < 1024), label
        assert dispatch["num_experts_at_most_64"] == (experts <= 64), label
        if ref["small"]:
            assert dispatch["kernel_phases"] == ("small_batch_expert",)
            assert dispatch["host_submissions"] == 1
            assert dispatch["grid"] == (1,)
            assert dispatch["threads"] == 256 + max(experts, 32)
            assert dispatch["fill_threads"] == 256
            assert dispatch["sort_grid"] is None
            assert dispatch["sort_symbol"] is None
            assert dispatch["symbol"].endswith(
                "moe_align_block_size_small_batch_expert_kernel")
            expected_shared = (
                (max(experts, 32) + 1) * experts + experts + 1) * INT32
        else:
            assert dispatch["kernel_phases"] == ("large_align", "large_sort")
            assert dispatch["host_submissions"] == 2
            assert dispatch["grid"] == (2,)
            assert dispatch["threads"] == 1024
            assert dispatch["fill_threads"] is None
            expected_sort = min((numel + 255) // 256, 65535)
            assert dispatch["sort_grid"] == (1, expected_sort)
            assert dispatch["sort_ctas"] == expected_sort
            assert dispatch["sort_block_threads"] == 256
            assert dispatch["sort_symbol"] == \
                "vllm::moe::count_and_sort_expert_tokens_kernel"
            assert dispatch["symbol"].endswith("moe_align_block_size_kernel")
            assert dispatch["scan_lanes"] == 1024
            expected_shared = ((experts + 31) // 32) * 32 * INT32
        analysis = dict(op.metadata)["analysis"]
        measured_shared = (analysis["small_shared_bytes"] if ref["small"]
                           else analysis["big_shared_bytes"])
        assert measured_shared == expected_shared
        assert dispatch["shared_bytes"] == expected_shared


def test_exact_histogram_is_not_replaced_by_a_mean_expert_count():
    experts, block, numel = 8, 16, 32
    skewed = [[0] for _ in range(numel)]
    balanced = [[index % experts] for index in range(numel)]
    for topk_ids, expected_counts, expected_padded in (
        (skewed, [32, 0, 0, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0, 0, 0]),
        (balanced, [4] * 8, [16] * 8),
    ):
        params = params_for(topk_ids, experts, block)
        op = make_op(params)
        meta = dict(op.metadata)
        assert meta["counts"] == tuple(expected_counts)
        assert meta["padded_per_expert"] == tuple(expected_padded)
        assert dict(meta["analysis"])["total_padded"] == sum(expected_padded)
        assert op.padded_shape == (sum(expected_padded),)
    skew_ref = reference(params_for(skewed, experts, block))
    balanced_ref = reference(params_for(balanced, experts, block))
    assert skew_ref["total_padded"] == 32
    assert balanced_ref["total_padded"] == 128
    assert [row["expert"] for row in iter_blocks(make_op(
        params_for(skewed, experts, block)))] == [0]
    assert [row["expert"] for row in iter_blocks(make_op(
        params_for(balanced, experts, block)))] == list(range(experts))


def test_native_expert_map_filtering_is_supported_post_gather_is_rejected():
    # CUDA lines 133-137/231-235/311-315 pass expert_map into the native
    # kernel and skip -1 locals.  Wrapper lines 100-103 instead index the
    # output tensor with an ATen kernel when ignore_invalid_experts is False;
    # that branch is outside the supplied source and must be rejected.
    topk_ids = [[0, 1], [2, 3], [1, 0], [3, 2], [0, 2]]
    expert_map = [1, -1, 2, 0]
    filtered_params = params_for(topk_ids, 4, 8, expert_map=expert_map,
                                 ignore_invalid_experts=True)
    ref = reference(filtered_params)
    op = make_op(filtered_params)
    meta = dict(op.metadata)
    analysis = dict(meta["analysis"])

    assert analysis["has_expert_map_kernel_arg"] is True
    assert analysis["expert_ids_post_gather"] is False
    assert analysis["expert_map"] == tuple(expert_map)
    assert analysis["post_gather_tail_value"] is None
    dispatch = dict(meta["dispatch"])
    assert dispatch["expert_map_dtype"] == "int32"
    assert dispatch["expert_ids_post_gather"] is False
    assert dispatch["expert_map_post_gather_supported"] is False
    assert analysis["valid_entries"] == ref["valid_entries"] == 8
    assert meta["counts"] == tuple(int(x) for x in ref["counts"])
    assert meta["cumsum"] == tuple(int(x) for x in ref["cumsum"])
    assert analysis["total_padded"] == 24

    blocks, written = reference_expert_ids(ref)
    assert blocks[:ref["used_blocks"]].tolist() == [0, 1, 2]
    assert np.all(blocks[ref["used_blocks"]:] == -1)
    assert written.all()

    stages = reference_stage_work(ref)
    assert "expert_map_gather" not in stages
    # 160 init + (40+40) filtered histogram + 4 total + 20 expert_ids
    # + (40+40+32) filtered scatter = 376 bytes.  Filtering is not a
    # free discount: the map is read in both passes.
    assert totals(stages)["hbm"] == 376
    assert set(dict(stage_work(op))) == set(stages)

    post_params = params_for(topk_ids, 4, 8, expert_map=expert_map,
                             ignore_invalid_experts=False)
    with pytest.raises(ValueError, match="requires_native_layout"):
        make_op(post_params)


def test_residual_fill_covers_all_stride_writers_1000_ids_10_valid_e4_b32():
    # Source lines 272-276 are outside if (tid < num_experts).  With
    # E=4 < 32 the expert group still has stride 32, so every residual block
    # is written by some tid; expert_ids has 36 initialized blocks, not 8.
    topk_ids = [[0] for _ in range(10)] + [[1] for _ in range(990)]
    params = params_for(topk_ids, 4, 32, expert_map=[0, -1, -1, -1],
                        ignore_invalid_experts=True)
    ref = reference(params)
    assert ref["small"]
    assert ref["numel"] == 1000
    assert ref["valid_entries"] == 10
    assert ref["total_padded"] == 32
    assert ref["used_blocks"] == 1
    assert ref["max_tokens_padded"] == 1152
    assert ref["max_m_blocks"] == 36
    assert ref["tail_blocks"] == 35
    assert ref["tail_written"] == 35
    op = make_op(params)
    meta = dict(op.metadata)
    analysis = dict(meta["analysis"])
    assert analysis["residual_writers"] == 32
    assert analysis["tail_blocks"] == 35
    assert analysis["tail_blocks_written"] == 35
    assert analysis["expert_ids_blocks_written"] == 36
    expert_ids = dict(stage_work(op))["expert_ids"]
    assert expert_ids.resources == {"hbm": 144.0, "mte3": 144.0,
                                    "vector": 36.0}
    assert dict(meta["work"]["stage_work"])["expert_ids"] == {
        "hbm": 144.0, "mte3": 144.0, "vector": 36.0}
    # Independent thread-loop oracle: all 36 blocks are initialized.
    _, written = reference_expert_ids(ref)
    assert written.all()
    assert int(written.sum()) == 36
    assert np.all(np.flatnonzero(written) == np.arange(36))
    # The used prefix is the local expert block; the rest is the -1 fill.
    assert ref["total_padded"] // ref["block_size"] == 1


def test_large_residual_fill_covers_more_than_one_1024_thread_cycle():
    # A filtering map can leave a large-batch tail far longer than the 1024
    # align threads.  The residue-class union still covers every block, so
    # expert_ids is fully initialized and charged at 4*MB bytes.
    experts, block, tokens = 65, 16, 20000
    topk_ids = [[0]] + [[1] for _ in range(tokens - 1)]
    expert_map = [0] + [-1] * (experts - 1)
    params = params_for(topk_ids, experts, block, expert_map=expert_map,
                        ignore_invalid_experts=True)
    ref = reference(params)
    assert not ref["small"]
    assert ref["tail_blocks"] > 1024
    op = make_op(params)
    analysis = dict(dict(op.metadata)["analysis"])
    assert analysis["residual_writers"] == 1024
    assert analysis["tail_blocks"] == ref["tail_blocks"]
    assert analysis["tail_blocks_written"] == ref["tail_blocks"]
    assert analysis["expert_ids_blocks_written"] == ref["max_m_blocks"]
    expert_ids = dict(stage_work(op))["expert_ids"]
    assert expert_ids.resources["mte3"] == ref["max_m_blocks"] * INT32
    assert expert_ids.resources["hbm"] == (
        ref["max_m_blocks"] * INT32 + (2 * experts + 1024) * INT32)
    _, written = reference_expert_ids(ref)
    assert written.all()
    assert int(written.sum()) == ref["max_m_blocks"]


def test_work_ledger_matches_independent_bytes_and_operation_oracle():
    samples = [
        params_for([[0, 1]], 4, 16),
        params_for([[2, 0], [2, 1], [2, 0]], 4, 16),
        params_for([[0], [0], [0]], 2, 32),
        params_for([[3, 0], [1, 3]], 8, 16),
        params_for([[1, 1], [0, 2], [2, 3], [3, 2]], 4, 32),
        params_for([[index % 65] for index in range(1024)], 65, 16),
        params_for([[0, 1], [2, 3], [1, 0], [3, 2]], 4, 8,
                   expert_map=[1, -1, 2, 0], ignore_invalid_experts=True),
        params_for([[0, 1]], 4, 16, topk_ids_dtype="int64"),
        params_for([[0, 1]], 4, 16, topk_ids_dtype="int8", pad_sorted_ids=False),
    ]
    for params in samples:
        ref = reference(params)
        expected = reference_stage_work(ref)
        op = make_op(params)
        meta = dict(op.metadata)
        actual = {
            name: {role: amount for role, amount in work.resources.items()}
            for name, work in stage_work(op)
        }
        assert actual == expected
        expected_totals = totals(expected)
        for role, amount in expected_totals.items():
            key = "vector_ops" if role == "vector" else f"{role}_bytes"
            assert meta["work"][key] == amount
        plan = build(op, Geometry(32, {}, {}))
        plan_resources = {role: plan.work.resources.get(role, 0)
                          for role in expected_totals}
        assert plan_resources == expected_totals
        assert not plan.work.matrix


def test_scan_stage_declares_only_cumsum_traffic_not_a_library_rate():
    params = params_for([[index % 128] for index in range(1024)], 128, 16)
    op = make_op(params)
    meta = dict(op.metadata)
    stages = dict(stage_work(op))
    assert "scan" in stages
    assert dict(meta["analysis"])["scan_lanes"] == 1024
    assert "vector" not in stages["scan"].resources
    assert stages["scan"].resources == {
        "hbm": (128 + 1) * INT32 + INT32,
        "mte3": (128 + 1) * INT32 + INT32,
    }
    # Two cumsum loads per expert thread plus one fill-start load by each of
    # the 1024 align threads; no cache reuse is assumed.
    assert stages["expert_ids"].resources["mte2"] == (2 * 128 + 1024) * INT32
    assert stages["expert_ids"].resources["hbm"] == \
        (2 * 128 + 1024) * INT32 + dict(meta["analysis"])["max_m_blocks"] * INT32
    analysis = dict(meta["analysis"])
    expected_vector = (
        2 * analysis["numel"] + analysis["padded_num_experts"] +
        analysis["max_m_blocks"])
    assert meta["work"]["vector_ops"] == expected_vector
    unresolved = " ".join(meta["unresolved"])
    assert "BlockScan" in unresolved
    assert "no scan-library instruction rate is fabricated" in unresolved
    assert all("scan" != name for name, _ in stage_work(
        make_op(params_for([[0, 1]], 4, 16))))


def test_kernels_expose_source_launch_boundaries_and_child_dispatch():
    small = make_op(params_for([[0, 1]], 4, 16))
    small_children = kernels(small)
    assert [(child.op_id, child.deps) for child in small_children] == [("align", ())]
    assert all(child.recipe == IMPLEMENTATION for child in small_children)
    assert dict(small_children[0].metadata)["kernel_phase"] == "small_batch_expert"
    assert dict(small_children[0].metadata)["dispatch"]["host_submissions"] == 1
    assert [name for name, _ in stage_work(small_children[0])] == [
        "init_sorted", "clear_counts", "histogram", "prefix", "expert_ids",
        "scatter"]

    large = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    large_children = kernels(large)
    assert [(child.op_id, child.deps) for child in large_children] == [
        ("align", ()), ("sort", ("align",))]
    assert [dict(child.metadata)["kernel_phase"] for child in large_children] == [
        "large_align", "large_sort"]
    assert all(child.recipe == IMPLEMENTATION for child in large_children)
    assert [name for name, _ in stage_work(large_children[0])] == [
        "init_sorted", "clear_counts", "histogram", "scan", "expert_ids"]
    assert [name for name, _ in stage_work(large_children[1])] == ["sort"]
    parent_work = build(large, Geometry(32, {}, {})).work
    child_work = build(large_children[0], Geometry(32, {}, {})).work + \
        build(large_children[1], Geometry(32, {}, {})).work
    assert parent_work.resources == child_work.resources

    # The native filtered map is a kernel argument, not a separate launch.
    filtered = make_op(params_for([[0, 1]], 4, 16, expert_map=[1, -1, 2, 0],
                                  ignore_invalid_experts=True))
    assert [(child.op_id, child.deps) for child in kernels(filtered)] == [
        ("align", ())]
    assert dict(filtered.metadata)["dispatch"]["host_submissions"] == 1
    filtered_large = make_op(params_for(
        [[i % 4] for i in range(1024)], 4, 32,
        expert_map=[1, -1, 2, 0], ignore_invalid_experts=True))
    assert [child.op_id for child in kernels(filtered_large)] == ["align", "sort"]
    assert dict(filtered_large.metadata)["dispatch"]["host_submissions"] == 2


def test_actual_host_submissions_follow_wrapper_dispatch():
    small = make_op(params_for([[0, 1]], 4, 16))
    execution, scheduled = execution_for(small)
    assert scheduled.operators == 2  # align + terminal join
    assert len(submitted_tasks(execution)) == 1
    assert execution.engine.serviced["host@r0:s0"] == pytest.approx(0.05)

    large = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    execution, scheduled = execution_for(large)
    assert scheduled.operators == 3  # align + sort + terminal join
    assert len(submitted_tasks(execution)) == 2
    assert execution.engine.serviced["host@r0:s0"] == pytest.approx(0.10)

    # A filtered map is inside the two native launches, not a third launch.
    filtered = make_op(params_for([[0, 1]], 4, 16, expert_map=[1, -1, 2, 0],
                                  ignore_invalid_experts=True))
    execution, scheduled = execution_for(filtered)
    assert scheduled.operators == 2  # align + terminal join
    assert len(submitted_tasks(execution)) == 1
    assert execution.engine.serviced["host@r0:s0"] == pytest.approx(0.05)


def _barrier_hardware():
    """The review's fixed per-unit rates; only vector and write are binding."""
    raw = control_case(units=1, split_copy=True)
    device = raw["machine"]["device"]
    device["launch_us"] = 0
    device["rates"].update(memory=1e20, read=1e20, write=128000,
                           vector=1360000)
    device["compute_peaks"] = dict.fromkeys(device["compute_peaks"], 1e20)
    device["resource_units"] = {"vector": 1, "mte2": 1, "mte3": 1}
    return raw


def _run_barrier_phase(child, build_fn):
    case = parse_case(_barrier_hardware())
    recipes = RecipeBook(recipes={child.recipe: PlannedRecipe(build_fn)})
    execution = Execution(case, CostModel(case), recipes=recipes,
                          trace_limit=4000)
    plan = IterationPlan((RequestView("r", 1, 0, True, "decode"),))
    iteration = IterationWork((child,), plan, 1, 1, 1, None, "decode",
                              "barrier")
    scheduled = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(scheduled.completion,))
    return execution


def _barrier_rows(execution):
    names = {"init_sorted", "histogram", "prefix"}
    return {row["task"].rsplit(":", 1)[-1]: row
            for row in execution.engine.trace
            if row["task"].rsplit(":", 1)[-1] in names}


def test_small_prefix_waits_for_histogram_and_init_sorted_source_barrier():
    # Source lines 205-216: the fill group initializes sorted_token_ids and
    # executes three __syncthreads(); the worker group's first barrier is
    # line 240 after the histogram.  The prefix reduction at lines 242-248
    # therefore cannot start before BOTH source groups reach that barrier.
    op = make_op(cases()[0]["params"])
    child = kernels(op)[0]
    execution = _run_barrier_phase(child, build)
    rows = _barrier_rows(execution)
    assert rows["prefix"]["start_ms"] >= rows["histogram"]["end_ms"]
    assert rows["prefix"]["start_ms"] >= rows["init_sorted"]["end_ms"]
    # Independent schedule: init_sorted 128B at 128000 B/s, then prefix
    # 136 vector ops at 1.36e6 ops/s (4B write is cheaper), then two 8B
    # writes.  No service count was changed to obtain this.
    expected_ms = 1000 * (
        128 / 128000 + 136 / 1360000 + 8 / 128000 + 8 / 128000)
    assert expected_ms == pytest.approx(1.225)
    assert execution.engine.summary()["time_ms"] == pytest.approx(expected_ms)

    # Negative control: restoring the previous prefix dependency (histogram
    # only) reproduces 1.15625 ms and lets prefix start before init_sorted
    # ends.  This isolates the cause without changing any service counts.
    def previous_dependency(oracle_op, geometry):
        plan = build(oracle_op, geometry)

        def program(context):
            original = plan.program(context)
            ctas = []
            for cta in original.stages:
                if cta.name != "cta0":
                    ctas.append(cta)
                    continue
                inner = tuple(
                    replace(inner, after=("histogram",))
                    if inner.name == "prefix" else inner
                    for inner in cta.program.stages)
                ctas.append(replace(cta, program=KernelProgram(inner)))
            return KernelProgram(tuple(ctas))

        return OperatorPlan(plan.work, program)

    previous = _run_barrier_phase(child, previous_dependency)
    old_rows = _barrier_rows(previous)
    assert old_rows["prefix"]["start_ms"] < old_rows["init_sorted"]["end_ms"]
    # With prefix overlapping init_sorted, its 4B write adds to the fill on
    # the same owner: 132B at 128000 B/s + two 8B writes = 1.15625 ms.
    previous_expected_ms = 1000 * (
        (128 + 4) / 128000 + 8 / 128000 + 8 / 128000)
    assert previous_expected_ms == pytest.approx(1.15625)
    assert previous.engine.summary()["time_ms"] == pytest.approx(
        previous_expected_ms)


def test_large_path_eager_and_capture_keep_align_then_sort_boundaries():
    # Source lines 638-678 launch align then sort.  Eager execution submits
    # each native launch; full graph capture submits the captured group once.
    op = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    assert dict(op.metadata)["dispatch"]["kernel_phases"] == (
        "large_align", "large_sort")
    eager, scheduled = execution_for(op, scope="none")
    eager_submits = submitted_tasks(eager)
    assert len(eager_submits) == 2
    assert [row["start_ms"] for row in eager_submits] == [0.0, 0.05]
    assert eager.engine.serviced["host@r0:s0"] == pytest.approx(0.10)
    # Device service adds a small finite tail; the launch boundary is 0.10.
    assert scheduled.completion.time == pytest.approx(0.10, abs=1e-3)

    captured, captured_scheduled = execution_for(op, scope="full")
    captured_submits = submitted_tasks(captured)
    assert len(captured_submits) == 1
    assert captured.engine.serviced["host@r0:s0"] == pytest.approx(0.05)
    assert captured_scheduled.completion.time == pytest.approx(0.05, abs=1e-3)


def test_small_single_cta_owns_one_physical_share_and_shares_it_between_stages():
    op = make_op(params_for([[0, 1]], 4, 16))
    child = kernels(op)[0]
    plan = build(child, Geometry(32, {"vector": 64, "mte2": 32, "mte3": 32},
                                 {}, {}))
    program = plan.program(recipe_context(child))
    leaves = [stage for stage in collect_leaf_stages(program)
              if stage.name in {"init_sorted", "clear_counts", "histogram",
                                "prefix", "expert_ids", "scatter"}]
    assert len(leaves) == 6
    assert all(not stage.caps for stage in leaves)
    vector_placements = {stage.placement.get("vector@r0:s0") for stage in leaves}
    mte2_placements = {stage.placement.get("mte2@r0:s0") for stage in leaves}
    mte3_placements = {stage.placement.get("mte3@r0:s0") for stage in leaves}
    assert vector_placements == {(0, 1)}
    assert mte2_placements == {(0,)}
    assert mte3_placements == {(0,)}
    assert all("memory@r0:s0" not in stage.placement for stage in leaves)

    # Fixed per-unit rate: one CTA's service does not grow with device size.
    durations = []
    for units in (1, 2, 32):
        report = verify_operator(unit_case(units), op, build, kernels=kernels)
        assert report["status"] == "passed"
        durations.append(report["result"]["duration_ms"])
    assert durations[0] == pytest.approx(durations[1])
    assert durations[0] == pytest.approx(durations[2])
    # All small-path vector work is on one dependency chain; the per-unit
    # rate bounds the kernel time independently of the unit count.
    vector_work = sum(work.resources.get("vector", 0)
                      for _, work in stage_work(op))
    assert durations[0] == pytest.approx(vector_work / 1e6)


def test_large_align_two_ctas_use_distinct_physical_owners():
    op = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    align = kernels(op)[0]
    plan = build(align, Geometry(32, {"vector": 64, "mte2": 32, "mte3": 32},
                                 {}, {}))
    program = plan.program(recipe_context(align))
    by_name = {stage.name: stage for stage in collect_leaf_stages(program)}
    cta0_names = {"clear_counts", "histogram", "scan", "expert_ids"}
    assert by_name["init_sorted"].placement["vector@r0:s0"] == (2, 3)
    for name in cta0_names:
        assert by_name[name].placement["vector@r0:s0"] == (0, 1)
    assert by_name["init_sorted"].placement["mte3@r0:s0"] == (1,)
    for name in cta0_names:
        assert by_name[name].placement["mte3@r0:s0"] == (0,)
    assert all(not stage.caps for stage in by_name.values())

    # With one physical unit, both CTAs map to the same owner and time-share.
    one = Geometry(1, {"vector": 1, "mte2": 1, "mte3": 1}, {}, {})
    program = build(align, one).program(
        recipe_context(align, owners=1, vector_units=1, mte_units=1))
    by_name = {stage.name: stage for stage in collect_leaf_stages(program)}
    assert by_name["init_sorted"].placement["vector@r0:s0"] == (0,)
    for name in cta0_names:
        assert by_name[name].placement["vector@r0:s0"] == (0,)


def test_large_sort_grid_is_lazy_per_unit_cta_streams():
    # 64 CTAs over 4 physical units: each unit receives a 16-CTA serial stream.
    topk_ids = [[index % 65] for index in range(64 * 256)]
    op = make_op(params_for(topk_ids, 65, 16))
    analysis = dict(dict(op.metadata)["analysis"])
    assert analysis["sort_ctas"] == 64
    sort = kernels(op)[1]
    plan = build(sort, Geometry(4, {"vector": 8, "mte2": 4, "mte3": 4}, {}, {}))
    context = recipe_context(sort, owners=4, vector_units=8, mte_units=4)
    program = plan.program(context)
    pipelines = [stage.program for stage in program.stages
                 if isinstance(stage.program, ParallelProgram)]
    assert len(pipelines) == 1
    streams = pipelines[0].programs
    assert len(streams) == 4
    assert all(isinstance(stream, PipelineProgram) for stream in streams)
    assert [stream.count for stream in streams] == [16, 16, 16, 16]
    assert sum(stream.count for stream in streams) == analysis["sort_ctas"]
    # Total tile work conserves the declared aggregate sort work.
    stage_work_total = dict(stage_work(op))["sort"]
    name_to_role = {name: role for role, name in context.resource_names.items()}
    tiles = []
    for stream in streams:
        for index in range(stream.count):
            tiles.append(stream.tile(index)[0])
    aggregate = {}
    for tile in tiles:
        assert not tile.caps
        for name, amount in tile.work.items():
            role = name_to_role[name]
            aggregate[role] = aggregate.get(role, 0) + amount
    assert aggregate == dict(stage_work_total.resources)
    # Every tile is pinned to exactly one physical CTA owner.
    for stream in streams:
        owner = stream.tile(0)[0].placement["vector@r0:s0"]
        assert len(owner) == 2  # 8 vector units / 4 owners


def test_sort_cta_streams_conserve_ids_and_filtered_valid_entries():
    # 1024 ids select 65 global experts (large path).  A filtering map keeps
    # half the experts and remaps them to local ids; per-CTA id and valid
    # counters must still sum exactly to the source loop totals.
    expert_map = [(-1 if expert % 2 else expert // 2) for expert in range(65)]
    topk_ids = [[index % 65] for index in range(1024)]
    params = params_for(topk_ids, 65, 16, expert_map=expert_map,
                        ignore_invalid_experts=True)
    ref = reference(params)
    assert not ref["small"]
    op = make_op(params)
    analysis = dict(dict(op.metadata)["analysis"])
    assert sum(analysis["sort_cta_ids"]) == ref["numel"]
    assert sum(analysis["sort_cta_valid"]) == ref["valid_entries"]
    assert tuple(analysis["sort_cta_ids"]) == tuple(int(x) for x in ref["sort_ids"])
    assert tuple(analysis["sort_cta_valid"]) == tuple(
        int(x) for x in ref["sort_valid"])
    sort = kernels(op)[1]
    context = recipe_context(sort, owners=4, vector_units=8, mte_units=4)
    program = build(
        sort, Geometry(4, {"vector": 8, "mte2": 4, "mte3": 4}, {}, {})
    ).program(context)
    parallel = next(stage.program for stage in program.stages
                    if isinstance(stage.program, ParallelProgram))
    name_to_role = {name: role for role, name in context.resource_names.items()}
    aggregate = {}
    for stream in parallel.programs:
        for index in range(stream.count):
            for name, amount in stream.tile(index)[0].work.items():
                role = name_to_role[name]
                aggregate[role] = aggregate.get(role, 0) + amount
    assert aggregate == dict(dict(stage_work(op))["sort"].resources)


def test_shared_memory_allocation_binds_to_the_same_cta_owner_and_releases():
    small = make_op(params_for([[0, 1]], 4, 16))
    small_shared = dict(dict(small.metadata)["analysis"])["align_shared_bytes"]
    execution, _ = execution_for(small, shared=small_shared)
    summary = execution.engine.summary()
    assert summary["peak_allocated"]["shared@r0:s0"] == small_shared
    assert summary["peak_allocated_per_unit"]["shared@r0:s0:unit0"] == small_shared
    assert not any(summary["live_allocated"].values())
    assert summary["live_tasks"] == 0

    large = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    large_shared = dict(dict(large.metadata)["analysis"])["align_shared_bytes"]
    execution, _ = execution_for(large, shared=large_shared)
    summary = execution.engine.summary()
    assert summary["peak_allocated"]["shared@r0:s0"] == large_shared
    assert summary["peak_allocated_per_unit"]["shared@r0:s0:unit0"] == large_shared
    assert summary["peak_allocated_per_unit"]["shared@r0:s0:unit1"] == large_shared
    assert not any(summary["live_allocated"].values())
    assert summary["live_tasks"] == 0

    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        execution_for(small, shared=small_shared - 4)


def test_shared_allocation_units_follow_each_cta_owner_slot():
    names = {"vector": "vector@r0:s0", "mte2": "mte2@r0:s0",
             "mte3": "mte3@r0:s0", "hbm": "memory@r0:s0"}

    def context_for(child, owners=4, shared_units=4):
        return RecipeContext(
            child, {}, names, {name: 1e9 for name in names.values()}, owners,
            pool_names={"shared": "shared@r0:s0"},
            resource_units={"vector": 8, "mte2": 4, "mte3": 4},
            compute_peaks={"bf16": 1e12, "fp32": 1e12},
            pool_units={"shared": shared_units},
        )

    small = make_op(params_for([[0, 1]], 4, 16))
    small_child = kernels(small)[0]
    small_shared = dict(dict(small.metadata)["analysis"])["align_shared_bytes"]
    small_program = build(
        small_child, Geometry(4, {"vector": 8, "mte2": 4, "mte3": 4},
                              {"shared": small_shared}, {"shared": 4})
    ).program(context_for(small_child))
    small_cta = small_program.stages[0]
    assert small_cta.name == "cta0"
    assert small_cta.acquire and small_cta.free
    assert small_cta.acquire == small_cta.free
    assert small_cta.acquire[0].pool == "shared@r0:s0"
    assert small_cta.acquire[0].size == small_shared
    assert small_cta.acquire[0].units == (0,)

    large = make_op(params_for([[i % 4] for i in range(1024)], 4, 32))
    large_align = kernels(large)[0]
    large_shared = dict(dict(large.metadata)["analysis"])["align_shared_bytes"]
    large_program = build(
        large_align, Geometry(4, {"vector": 8, "mte2": 4, "mte3": 4},
                              {"shared": large_shared}, {"shared": 4})
    ).program(context_for(large_align))
    by_name = {stage.name: stage for stage in large_program.stages}
    assert by_name["cta0"].acquire[0].units == (0,)
    assert by_name["cta1"].acquire[0].units == (1,)
    assert by_name["cta0"].acquire[0].size == large_shared
    assert by_name["cta1"].acquire[0].size == large_shared
    assert all(stage.acquire and stage.free
               for stage in (by_name["cta0"], by_name["cta1"]))


def test_actual_isolated_execution_conserves_declared_work_for_both_paths():
    selected = [
        make_op(case["params"]) for case in cases()
    ] + [
        make_op(params_for([[i % 4] for i in range(1024)], 4, 32)),
        make_op(params_for([[i % 65] for i in range(1024)], 65, 16)),
        make_op(params_for([[0, 1]], 4, 16, expert_map=[1, -1, 2, 0],
                           ignore_invalid_experts=True)),
        make_op(params_for([[i % 4] for i in range(1024)], 4, 32,
                           expert_map=[1, -1, 2, 0],
                           ignore_invalid_experts=True)),
    ]
    for op in selected:
        dispatch = dict(op.metadata)["dispatch"]
        expected_phases = len(dispatch["kernel_phases"])
        for split_copy in (False, True):
            hardware = control_case(split_copy=split_copy, units=32)
            report = verify_operator(hardware, op, build, kernels=kernels)
            assert report["status"] == "passed", (op.op_id, dispatch["path"],
                                                  report["work_mismatches"])
            assert report["finite_completion"] is True
            assert report["buffers_released"] is True
            assert report["work_mismatches"] == {}
            assert len(report["result"]["kernels"]) == expected_phases
            for row in report["result"]["kernels"]:
                assert row["live_tasks"] == 0
                assert not any(row["remaining_buffers"].values())
                assert row["service_delta"] == {}
            if split_copy:
                assert any(name.startswith("mte2@") for name in report["result"]["serviced"])
                assert any(name.startswith("mte3@") for name in report["result"]["serviced"])
            else:
                assert not any(name.startswith("mte2@") for name in report["result"]["serviced"])


def test_public_case_runner_preserves_splits_and_identity_invariance():
    result = run(moe_align_module, cases())
    assert result["status"] == "passed"
    assert result["scope"] == "structural_contract_only_no_accuracy_claim"
    assert len(result["cases"]) == 10
    assert {row["split"] for row in result["cases"]} == {
        "development", "heldout_geometry"}
    for row in result["cases"]:
        assert row["status"] == "passed"
        assert row["identity_invariant"] is True
        assert row["dispatch_mismatches"] == {}
        assert row["copy_resources"] in (False, True)


def test_large_batch_program_stays_lazy_and_bounded():
    experts, block, tokens, top_k = 128, 128, 20000, 8
    params = params_for(
        [[(index * 7 + j) % experts for j in range(top_k)]
         for index in range(tokens)], experts, block)
    op = make_op(params)
    analysis = dict(dict(op.metadata)["analysis"])
    assert analysis["numel"] == tokens * top_k
    assert analysis["small_batch_expert_mode"] is False
    assert len(stage_work(op)) == 6
    parsed = parse_case(control_case(units=32))
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(build, kernels)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes,
                          trace_limit=0)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    scheduled = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(scheduled.completion,))
    summary = execution.engine.summary()
    assert summary["live_tasks"] == 0
    assert summary["live_programs"] == 0
    assert execution.engine.peak_tasks < 256
    assert execution.engine.peak_allocated == {}
    assert execution.work["vector_ops"] > 0


def test_source_identity_hashes_lines_and_scope_match_manifest():
    manifest = json.loads(MANIFEST_FILE.read_text())
    task = manifest
    by_path = {item["path"]: item for item in task["sources"]}
    op = make_op(cases()[0]["params"])
    identity = dict(dict(op.metadata)["source_identity"])
    assert identity["scope"] == \
        "current_checkout_manifest_matched_not_historical_capture_image"
    assert {item["path"] for item in identity["files"]} == set(by_path)
    for item in identity["files"]:
        path = Path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"] == by_path[item["path"]]["sha256"]
        assert path.stat().st_size == item["bytes"] == \
            by_path[item["path"]]["bytes"]
    assert hashlib.sha256(MATH.read_bytes()).hexdigest() == \
        identity["helper"]["sha256"]
    lines = dict(dict(op.metadata)["source_lines"])
    assert "272-276" in lines["small_batch_residual_fill"]
    assert "205-216" in lines["small_batch_fill_barriers"]
    assert "240" in lines["small_batch_first_worker_barrier"]
    assert "179-185" in lines["align_residual_fill"]
    assert "613-614" in lines["small_batch_threshold"]
    assert "616-637" in lines["small_batch_launch"]
    assert "638-678" in lines["two_kernel_launch"]
    assert "582-680" in lines["host_dispatch_and_launch"]
    assert "74-88" in lines["wrapper_buffer_sizing"]
    assert "100-103" in lines["wrapper_expert_map_gather"]


def test_build_and_kernels_reject_unselected_or_malformed_ops():
    params = params_for([[0, 1]], 4, 16)
    op = make_op(params)
    geometry = Geometry(32, {}, {})
    wrong = replace(op, recipe="other")
    with pytest.raises(ValueError, match="selected_recipe"):
        build(wrong, geometry)
    with pytest.raises(ValueError, match="selected_recipe"):
        kernels(wrong)
    wrong_dispatch = replace(op, metadata=tuple(
        (key, {**value, "implementation": "other"} if key == "dispatch" else value)
        for key, value in op.metadata))
    with pytest.raises(ValueError, match="selected_implementation"):
        build(wrong_dispatch, geometry)
    with pytest.raises(ValueError, match="selected_implementation"):
        kernels(wrong_dispatch)
    missing = replace(op, metadata=tuple(
        (key, value) for key, value in op.metadata if key != "analysis"))
    with pytest.raises(ValueError, match="make_op_analysis"):
        build(missing, geometry)
    with pytest.raises(ValueError, match="make_op_analysis"):
        kernels(missing)
    with pytest.raises(ValueError, match="analysis"):
        stage_work(missing)
    child = kernels(op)[0]
    with pytest.raises(ValueError, match="parent_op"):
        kernels(child)
    bad_phase = replace(op, metadata=tuple(
        (key, {**value, "kernel_phases": ("bogus",)}
         if key == "dispatch" else value) for key, value in op.metadata))
    with pytest.raises(ValueError, match="button_mismatch"):
        build(bad_phase, geometry)


def test_phase_and_integer_width_are_explicit_axes():
    base = params_for([[0, 1]], 4, 16)
    assert make_op({**base, "phase": "decode"}).phase == "decode"
    assert make_op({**base, "phase": "mixed"}).phase == "mixed"
    int32 = make_op(base)
    int64 = make_op({**base, "topk_ids_dtype": "int64"})
    int8 = make_op({**base, "topk_ids_dtype": "int8"})
    assert dict(int32.metadata)["dispatch"]["topk_ids_dtype"] == "int32"
    assert dict(int64.metadata)["dispatch"]["topk_ids_dtype"] == "int64"
    assert dict(int8.metadata)["dispatch"]["topk_ids_dtype"] == "int8"
    assert dict(int64.metadata)["work"]["hbm_bytes"] > \
        dict(int32.metadata)["work"]["hbm_bytes"]
    assert dict(int8.metadata)["work"]["hbm_bytes"] < \
        dict(int32.metadata)["work"]["hbm_bytes"]
