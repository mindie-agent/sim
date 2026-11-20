"""Verify combine work, routing transport and late TP reduction.

[Role] Exercise tile and CTA work, DeepEP transfer and shared-expert ordering.
[Invariants] Tensor tails, weighted sums, and contiguous expert banks preserve
  exact work and dependencies across the operator and collective boundaries.
[Timeouts & Locks] Finite synthetic graphs and bounded lazy-window replay.
[Side-Effect Policy] In-memory modeling and Engine execution only.
"""
from collections import Counter
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import deepep
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.family_grouped_gemm import lower_op
from mindie_sim.prediction.implementations import produced_values
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_callers import LOCAL_EXPERT_REDUCE, LocalBankRouting
from mindie_sim.prediction.operators import moe_combine as mc
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0
ROLES = ("hbm", "mte2", "mte3", "vector")


def params(tokens=3, hidden=64, topk=2, *, dtype="bf16", weight_dtype="fp32",
           **overrides):
    value = {
        "tokens": tokens, "hidden": hidden, "topk": topk,
        "dtype": dtype, "weight_dtype": weight_dtype,
        "compute_capability": mc.COMPUTE_CAPABILITY,
        "implementation": mc.IMPLEMENTATION,
        "apply_router_weight_on_input": False,
        "has_expert_map": False,
    }
    value.update(overrides)
    return value


def context_for(op, units=32, *, vector_units=None, copy_roles=True):
    vector_units = units if vector_units is None else vector_units
    roles = ROLES if copy_roles else ("hbm", "vector")
    return RecipeContext(
        op=op, work={}, resource_names={role: role for role in roles},
        rates={role: 1.0 for role in roles},
        core_count=units, resource_units={"vector": vector_units},
    )


def iter_cta_tiles(program):
    """Yield ``(global_cta_index, stages)`` for a per-owner CTA program.

    This is used only by small oracles.  The huge-token test inspects stream
    counts without materializing any tile.
    """
    owners = len(program.programs)
    for core, stream in enumerate(program.programs):
        for local in range(stream.count):
            yield core + local * owners, stream.tile(local)


def reference_work(tokens, hidden, topk, input_bytes, weight_bytes,
                   block_m, block_k):
    """Independently enumerate every source CTA and its valid lanes."""
    m_tiles = (tokens + block_m - 1) // block_m
    k_tiles = (hidden + block_k - 1) // block_k
    a_upconvert = 1 if input_bytes < 4 else 0
    w_upconvert = 1 if weight_bytes < 4 else 0
    out_downconvert = 1 if input_bytes < 4 else 0
    per_program = (
        block_m * block_k
        + block_m * block_k * topk * (2 + a_upconvert)
        + block_m * topk * w_upconvert
        + block_m * block_k * out_downconvert
    )
    hbm = mte2 = mte3 = 0
    for pid_m in range(m_tiles):
        valid_m = min(block_m, tokens - pid_m * block_m)
        for pid_k in range(k_tiles):
            valid_k = min(block_k, hidden - pid_k * block_k)
            a_bytes = topk * valid_m * valid_k * input_bytes
            w_bytes = topk * valid_m * weight_bytes
            stored = valid_m * valid_k * input_bytes
            hbm += a_bytes + w_bytes + stored
            mte2 += a_bytes + w_bytes
            mte3 += stored
    return {
        "m_tiles": m_tiles, "k_tiles": k_tiles,
        "programs": m_tiles * k_tiles,
        "per_program": per_program,
        "hbm": hbm, "mte2": mte2, "mte3": mte3,
        "vector": m_tiles * k_tiles * per_program,
    }


def to_bf16(values):
    """Round FP32 to BF16 with round-to-nearest-even, without torch."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def weighted_sum_oracle(inputs, weights):
    """Explicit source-order FP32 accumulation and output downconvert."""
    a = np.asarray(inputs, dtype=np.float32)
    b = np.asarray(weights, dtype=np.float32)
    if a.ndim != 3 or b.shape != a.shape[:2]:
        raise ValueError("oracle_requires_inputs[m,topk,k]_and_weights[m,topk]")
    acc = np.zeros((a.shape[0], a.shape[2]), dtype=np.float32)
    for n in range(a.shape[1]):
        acc = acc + a[:, n, :] * b[:, n, None]
    return to_bf16(acc)


@pytest.mark.parametrize(
    "tokens,hidden,expected",
    [
        (1, 128, (1, 256, 4, 4)),    # BLOCK_M <= 4; BLOCK_K floor 256
        (4, 256, (1, 256, 4, 4)),
        (5, 257, (2, 512, 4, 4)),    # first BLOCK_M step; next pow2 <= 512
        (32, 512, (2, 512, 4, 4)),
        (33, 1024, (4, 1024, 16, 2)),  # BLOCK_M step and large-tile stages
        (128, 1024, (4, 1024, 16, 2)),
        (129, 1024, (16, 1024, 16, 2)),  # 16-row tile threshold
        (1024, 1024, (16, 1024, 16, 2)),
        (1025, 1024, (8, 1024, 16, 2)),  # final BLOCK_M branch
        (7, 512, (2, 512, 4, 4)),
        (7, 513, (2, 1024, 8, 4)),
        (7, 1025, (2, 1024, 8, 4)),   # BLOCK_K ceiling, not next pow2 2048
        (7, 2049, (2, 1024, 8, 4)),
    ],
)
def test_tile_dispatch_thresholds_follow_the_sm89_heuristic(tokens, hidden, expected):
    op = mc.make_op(params(tokens=tokens, hidden=hidden))
    selected = dict(op.metadata)["dispatch"]
    assert (
        selected["BLOCK_M"], selected["BLOCK_K"],
        selected["num_warps"], selected["num_stages"],
    ) == expected
    assert selected["is_fp32"] is False
    assert selected["grid"] == (
        math.ceil(hidden / expected[1]), math.ceil(tokens / expected[0]),
    )


@pytest.mark.parametrize(
    "tokens,hidden,expected",
    [
        (1, 128, (1, 256, 8, 4)),
        (5, 100, (2, 256, 8, 4)),
        (200, 64, (4, 256, 16, 4)),
    ],
)
def test_fp32_io_uses_the_source_fp32_tile_branch(tokens, hidden, expected):
    op = mc.make_op(params(tokens=tokens, hidden=hidden, dtype="fp32",
                           weight_dtype="fp32"))
    selected = dict(op.metadata)["dispatch"]
    assert (
        selected["BLOCK_M"], selected["BLOCK_K"],
        selected["num_warps"], selected["num_stages"],
    ) == expected
    assert selected["is_fp32"] is True


@pytest.mark.parametrize(
    "tokens,hidden,topk,dtype,weight_dtype",
    [
        (1, 1, 1, "bf16", "fp32"),
        (5, 130, 3, "bf16", "fp32"),
        (33, 1024, 2, "bf16", "fp32"),
        (17, 513, 4, "bf16", "bf16"),
        (9, 300, 5, "fp16", "fp16"),
        (7, 64, 2, "fp32", "fp32"),
        (7, 257, 3, "fp32", "bf16"),
    ],
)
def test_work_and_program_match_an_independent_tile_enumeration(
    tokens, hidden, topk, dtype, weight_dtype,
):
    op = mc.make_op(params(tokens=tokens, hidden=hidden, topk=topk,
                           dtype=dtype, weight_dtype=weight_dtype))
    selected = dict(op.metadata)["dispatch"]
    input_bytes = 4 if dtype == "fp32" else 2
    weight_bytes = 4 if weight_dtype == "fp32" else 2
    expected = reference_work(
        tokens, hidden, topk, input_bytes, weight_bytes,
        selected["BLOCK_M"], selected["BLOCK_K"],
    )
    plan = mc.build(op, Geometry(32, {"vector": 32}, {}))
    program = plan.program(context_for(op))
    assert isinstance(program, ParallelProgram)
    totals = Counter()
    for index, (load, compute, store) in iter_cta_tiles(program):
        assert compute.after == (load.name,) and store.after == (compute.name,)
        assert compute.placement["vector"] == (index % len(program.programs),)
        for stage in (load, compute, store):
            totals.update(stage.work)
    assert plan.work.resources == {
        "hbm": expected["hbm"], "mte2": expected["mte2"],
        "mte3": expected["mte3"], "vector": expected["vector"],
    }
    assert sum(stream.count for stream in program.programs) == expected["programs"]
    expected_owners = min(expected["programs"], 32)
    assert len(program.programs) == expected_owners
    assert all(isinstance(stream, PipelineProgram)
               for stream in program.programs)
    # Every source CTA appears exactly once across the owner streams.
    indices = [index for index, _ in iter_cta_tiles(program)]
    assert indices == list(range(expected["programs"]))
    assert totals == plan.work.resources


def test_program_is_lazy_for_a_huge_token_count():
    tokens, hidden, topk = 10**9, 1024, 2
    op = mc.make_op(params(tokens=tokens, hidden=hidden, topk=topk))
    plan = mc.build(op, Geometry(32, {"vector": 64}, {}))
    program = plan.program(context_for(op))
    expected_ctas = tokens // 8
    assert len(program.programs) == 32
    assert sum(stream.count for stream in program.programs) == expected_ctas
    assert all(stream.count == expected_ctas // 32
               for stream in program.programs)
    assert plan.work.resources["vector"] == expected_ctas * 8 * hidden * (2 + 3 * topk)
    assert plan.work.resources["hbm"] == tokens * (2 * hidden * (topk + 1) + 4 * topk)


def test_execution_keeps_live_cta_state_bounded_and_conserves_work():
    """Actual execution: at most one resident tile per owner stream."""
    op = mc.make_op(params(tokens=8192, hidden=1024, topk=1))
    geometry = Geometry(32, {"vector": 32}, {})
    plan = mc.build(op, geometry)
    program = plan.program(context_for(op, copy_roles=False))
    total_programs = sum(stream.count for stream in program.programs)
    assert total_programs == 1024

    engine = Engine([
        Resource("memory", 32e9, "bytes", units=32),
        Resource("vector", 32e6, "ops", units=32),
    ])
    context = RecipeContext(
        op=op, work={}, resource_names={"hbm": "memory", "vector": "vector"},
        rates={"memory": 32e9, "vector": 32e6},
        core_count=32, resource_units={"vector": 32},
    )
    ready = engine.submit("ready")
    complete = plan.lower(context).start(engine, "owned", ready)
    engine.advance(until=(complete,))
    summary = engine.summary()
    assert summary["live_tasks"] == 0 and summary["live_programs"] == 0
    # 1024 CTAs must not be submitted up front: only the 32 owner windows.
    assert summary["peak_live_tasks"] < 400
    assert engine.serviced["vector"] == plan.work.resources["vector"]
    assert engine.serviced["memory"] == plan.work.resources["hbm"]
    assert 0 < engine.completed_count <= 3 * total_programs + 64


def test_execution_trace_pins_each_cta_to_its_owner():
    """The running engine sees one owner placement per materialized fma."""
    op = mc.make_op(params(tokens=5, hidden=130, topk=3))
    plan = mc.build(op, Geometry(4, {"vector": 4}, {}))
    context = RecipeContext(
        op=op, work={}, resource_names={"hbm": "memory", "vector": "vector"},
        rates={"memory": 4e9, "vector": 4e6},
        core_count=4, resource_units={"vector": 4},
    )
    engine = Engine([
        Resource("memory", 4e9, "bytes", units=4),
        Resource("vector", 4e6, "ops", units=4),
    ], trace_limit=64)
    ready = engine.submit("ready")
    complete = plan.lower(context).start(engine, "op", ready)
    engine.advance(until=(complete,))
    fma = [entry for entry in engine.trace if entry["task"].endswith(":fma")]
    assert len(fma) == 3
    assert sorted(entry["placement"]["vector"] for entry in fma) == [
        (0,), (1,), (2,),
    ]
    assert engine.serviced["vector"] == pytest.approx(
        plan.work.resources["vector"], rel=1e-12,
    )
    assert engine.summary()["live_tasks"] == 0


def test_incompatible_physical_unit_partition_is_rejected():
    op = mc.make_op(params(tokens=1025, hidden=1024, topk=1))
    plan = mc.build(op, Geometry(32, {"vector": 48}, {}))
    program = plan.program(context_for(op, vector_units=48))
    with pytest.raises(ModelingError, match="ownership_geometry_incompatible"):
        program.programs[0].tile(0)


def test_build_rejects_stale_dispatch_metadata():
    op = mc.make_op(params(tokens=5, hidden=130, topk=3))
    metadata = dict(op.metadata)
    metadata["block_m"] = 99
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="metadata_mismatch:block_m"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))
    metadata = dict(op.metadata)
    dispatch = dict(metadata["dispatch"])
    dispatch["BLOCK_K"] = 1
    metadata["dispatch"] = dispatch
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="dispatch_mismatch:BLOCK_K"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))
    metadata = dict(op.metadata)
    dispatch = dict(metadata["dispatch"])
    dispatch["wrapper"] = "topk_weight_and_reduce"
    metadata["dispatch"] = dispatch
    stale = replace(op, metadata=tuple(sorted(metadata.items())))
    with pytest.raises(ModelingError, match="dispatch_mismatch:wrapper"):
        mc.build(stale, Geometry(32, {"vector": 64}, {}))


def test_numpy_weighted_sum_oracle_keeps_fp32_accumulation():
    rng = np.random.default_rng(20261028)
    inputs = to_bf16(rng.standard_normal((5, 4, 130)).astype(np.float32))
    weights = rng.standard_normal((5, 4)).astype(np.float32)
    actual = weighted_sum_oracle(inputs, weights)
    exact = np.einsum(
        "mtk,mt->mk", inputs.astype(np.float64), weights.astype(np.float64),
    )
    assert np.allclose(actual.astype(np.float64), exact,
                       rtol=2**-7, atol=2**-8)
    relative = np.abs(actual.astype(np.float64) - exact) / np.abs(exact)
    assert relative[np.abs(exact) > 0.1].max() < 2**-7
    assert actual.dtype == np.float32

    # A BF16 accumulator loses low-order contributions that the source keeps
    # until its final downconvert.
    a = to_bf16(np.array([[[1.0]] + [[0.001]] * 7], dtype=np.float32))
    b = np.ones((1, 8), dtype=np.float32)
    fp32_result = weighted_sum_oracle(a, b)
    bf16_acc = np.zeros((1, 1), dtype=np.float32)
    for n in range(8):
        bf16_acc = to_bf16(bf16_acc + a[:, n, :] * b[:, n, None])
    assert fp32_result[0, 0] != bf16_acc[0, 0]
    assert float(fp32_result[0, 0]) == pytest.approx(1.0078125)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"has_expert_map": True}, "expert_map_branch_not_selected"),
        ({"apply_router_weight_on_input": True},
         "apply_router_weight_on_input_not_supported"),
        ({"implementation": "moe_sum"}, "unsupported_implementation"),
        ({"compute_capability": 90}, "compute_capability_89"),
        ({"compute_capability": 80}, "compute_capability_89"),
        ({"compute_capability": 89.0}, "compute_capability_89"),
        ({"dtype": "fp8"}, "unsupported_dtype:dtype"),
        ({"weight_dtype": "int8"}, "unsupported_dtype:weight_dtype"),
        ({"tokens": 0}, "positive_integer:tokens"),
        ({"tokens": True}, "positive_integer:tokens"),
        ({"hidden": -1}, "positive_integer:hidden"),
        ({"topk": 0}, "positive_integer:topk"),
        ({"has_expert_map": 0}, "requires_explicit_boolean"),
        ({"apply_router_weight_on_input": 0}, "requires_explicit_boolean"),
        ({"topk_ids": np.zeros((1, 1), dtype=np.int64)},
         "unknown_parameters:topk_ids"),
    ],
)
def test_unselected_or_invalid_source_paths_are_rejected(overrides, match):
    with pytest.raises(ModelingError, match=match):
        mc.make_op(params(**overrides))


def test_missing_and_non_mapping_parameters_are_rejected():
    with pytest.raises(ModelingError, match="params_must_be_a_mapping"):
        mc.make_op([])
    missing = params()
    del missing["weight_dtype"]
    with pytest.raises(ModelingError, match="missing_parameters:weight_dtype"):
        mc.make_op(missing)


def moe_case(*, tp=1, dp=1, ep=1, experts=8, tokens=8, top_k=2, shared=0):
    ranks = tp * dp
    return {
        "machine": {
            "name": "declared", "num_devices": ranks,
            "device": {"memory_bytes": 1 << 30, "compute_units": 4,
                       "compute_peaks": {"bf16": 1e9, "fp32": 1e9, "int8": 1e9,
                                         "fp8_e4m3": 2e9, "mxfp8": 2e9},
                       "rates": {"memory": 1e9, "vector": 1e9}},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": list(range(ranks)), "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 1, "hidden_size": 32, "num_heads": 4,
            "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
            "ffn": "moe", "num_experts": experts, "top_k": top_k,
            "num_shared_experts": shared, "moe_intermediate_size": 64,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": tokens, "output_tokens": 1}]},
        "runtime": {"tp": tp, "dp": dp, "ep": ep, "max_num_batched_tokens": max(16, tokens)},
    }


def one(ops, recipe):
    (result,) = (op for op in ops if op.recipe == recipe)
    return result


@pytest.mark.parametrize("ep,node_width,remote_nodes", [(4, 4, 0), (8, 4, 1), (64, 8, 7)])
def test_deepep_two_stage_volume_deduplicates_remote_nodes(ep, node_width, remote_nodes):
    tokens, width, top_k, element_bytes = 100, 4096, 8, 2
    case = dict(ep=ep, ranks_per_node=node_width, dtype_bytes_val=element_bytes)
    group = deepep.deepep_group_volume(tokens, width, top_k, **case)
    intra = deepep.deepep_intra_volume(tokens, width, top_k, **case)
    assert group == tokens * width * element_bytes * remote_nodes
    assert intra == pytest.approx(tokens * width * element_bytes * top_k * (node_width - 1) / node_width)
    assert deepep.deepep_group_volume(tokens, width, top_k, eplb_peak_to_avg=1.25,
                                    **case) == pytest.approx(group * 1.25)


@pytest.mark.parametrize("ep,node_width", [(2, 2), (8, 4)])
def test_deepep_forward_and_inverse_transport_share_integer_row_cut(ep, node_width):
    raw = moe_case(dp=ep, ep=ep, tokens=4, top_k=2)
    plain = target_graphs(raw, phase="prefill", tokens=4, scope="model")[0]
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.25,
                                  "deepep_ranks_per_node": node_width}
    chunks = target_graphs(raw, phase="prefill", tokens=4, scope="model")[0]
    tokens, width, top_k = 4, 32, 2
    flat = tokens * width * 2 * top_k * (ep - 1) / ep
    two_stage = tokens * width * 2 * (ep // node_width - 1 + top_k * (node_width - 1) / node_width)
    for kind in ("dispatch", "combine"):
        recipe = f"expert_{kind}"
        full, tail = one(chunks, recipe), one(chunks, recipe + "_overlap")
        assert one(plain, recipe).link_bytes == pytest.approx(flat)
        assert (full.link_bytes, tail.link_bytes) == pytest.approx((two_stage * 0.75, two_stage * 0.25))
        assert dict(one(chunks, "expert_dispatch").metadata)["deepep_row_cut"] == (6, 2)
        assert tail.stream == "deepep_comm_stream"
    ids = {op.op_id for op in chunks} | {"analysis_input"}
    assert all(set(op.deps) <= ids for op in chunks)


@pytest.mark.parametrize("factor", [1, 1.25])
def test_block_scaled_dispatch_and_bf16_combine_conserve_transport(factor):
    raw = moe_case(dp=2, ep=2, experts=4, tokens=4)
    raw["model"]["quant"] = asdict(quant_preset("w8a8_mxfp8"))
    raw["runtime"]["kernels"] = {"moe_transport": {
        "kind": "separate_quantized_payload", "dispatch_dtype": "mxfp8",
        "scale_group": 32, "layer_scopes": ["body"], "eplb_peak_to_avg": factor,
    }}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    ops = graphs[0]
    dispatch, combine = one(ops, "expert_quantized_dispatch"), one(ops, "expert_combine")
    assert dict(dispatch.metadata)["row_bytes"] == 33
    assert dispatch.link_bytes == 8 * 33 / 2 * factor
    assert combine.link_bytes == 8 * 64 / 2 * factor
    producer = next(op for op in ops if op.op_id in dispatch.deps and op.family == "quantize")
    assert producer.mte3_bytes == 8 * 33
    result = graph_costs(raw, graphs)
    assert result["execution"]["live_tasks"] == 0


def test_shared_expert_stream_preserves_work_and_join():
    raw = moe_case(dp=2, ep=2, shared=1)
    raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
    summary = deepep.analyze_shared_expert_overlap(
        raw, phase="prefill", tokens=4, sequences=1, scope="model")
    assert summary["overlapped_step_ms"] <= summary["baseline_step_ms"]
    for work in ("total_flops", "total_hbm_bytes"):
        assert summary[f"baseline_{work}"] == summary[f"overlapped_{work}"]
    raw["runtime"]["metadata"] = {"overlap_shared_expert": True}
    ops = target_graphs(raw, phase="prefill", tokens=4, scope="model")[0]
    join = one(ops, "expert_add")
    parents = {op.recipe: op for op in ops if op.op_id in join.deps}
    assert parents["route_unpack"].stream != "shared_stream"
    assert parents["dense_down"].stream == "shared_stream"


def local_case(*, tokens=8, counts=None):
    raw = moe_case(tp=4, ep=4, tokens=tokens)
    runtime = raw["runtime"]
    runtime["moe_recipe"] = LOCAL_EXPERT_REDUCE
    runtime["routing"] = {"policy": "balanced"}
    if counts is not None:
        runtime["routing"] = {"realizations": [{"scope": "body", "layer": 0,
            "phase": "prefill", "tokens": tokens, "expert_rows": list(counts)}]}
        runtime["kernels"] = {"lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}}
    return raw


@pytest.mark.parametrize("counts,bank,physical_ids", [
    ((4, 8, 4, 0, 0, 0, 0, 0), (4, 8), (0, 1)),
    ((0, 0, 4, 8, 4, 0, 0, 0), (4, 8), (2, 3)),
    ((8, 0, 8, 0, 0, 0, 0, 0), (8, 0), (0, 1)),
    ((0, 0, 0, 0, 8, 0, 8, 0), (8, 0), (4, 5)),
])
def test_contiguous_expert_banks_preserve_zero_bins_and_physical_ids(counts, bank, physical_ids):
    raw = local_case(counts=counts)
    ops = target_operators(raw, phase="prefill", tokens=8)
    unpack = one(ops, "route_unpack")
    assert unpack.local_shape[0] == 8
    assert not any(op.family in {"alltoall", "alltoallv"} for op in ops)
    assert one(ops, "moe_output_reduce").deps == (unpack.op_id,)
    matrices = [op for op in ops if op.family == "grouped_gemm"]
    assert len(matrices) == 2
    for op in matrices:
        meta = dict(op.metadata)
        assert meta["expert_rows"] == meta["local_expert_rows"] == bank
        assert meta["representative_expert_ids"] == physical_ids
        assert meta["global_expert_rows"] == counts
        m, n, k = op.local_shape
        assert m == sum(bank)
        priced = operator_costs(raw, [op])["operators"][0]["operator"]
        assert priced["flops"] == 2 * m * n * k


def test_empty_local_bank_is_rejected_at_source_lowering():
    ops = target_operators(local_case(), phase="prefill", tokens=8)
    op = next(op for op in ops if op.family == "grouped_gemm")
    realization = LocalBankRouting(expert_rows=(0,) * 8, local_expert_rows=(0, 0),
                                   rank_rows=(0,) * 4, representative_rank=0,
                                   basis="declared_integer_counts", ep=4, parent={})
    meta = dict(op.metadata, routing_basis="declared_integer_counts",
                routing_realization=realization.to_dict())
    empty = replace(op, local_shape=(0, *op.local_shape[1:]), flops=0, metadata=tuple(meta.items()))
    with pytest.raises(ModelingError, match="local_bank_has_no_valid_rows"):
        lower_op(empty)


@pytest.mark.parametrize("shared,gate", [(0, False), (1, False), (1, True)])
def test_tp_reduction_follows_weighted_unpermute_and_shared_join(shared, gate):
    tokens, tp, width = 32, 4, 32
    raw = moe_case(tp=tp, shared=shared, tokens=tokens)
    raw["runtime"]["routing"] = {"policy": "balanced"}
    raw = parse_case(raw).to_dict()
    raw["model"]["layer_plans"][0]["shared_expert_gate"] = gate
    raw["runtime"]["moe_recipe"] = "alltoall"
    early = target_operators(raw, phase="prefill", tokens=tokens)
    raw["runtime"]["moe_recipe"] = "tp_join_reduce"
    late = target_operators(raw, phase="prefill", tokens=tokens)
    by_id = {op.op_id: op for op in late}
    reduce = one(late, "moe_output_reduce")
    unpack = one(late, "route_unpack")
    assert reduce.local_shape == (tokens, width)
    assert reduce.link_bytes == 2 * (tp - 1) / tp * tokens * width * 2
    parent = by_id[reduce.deps[0]]
    if shared:
        assert parent.recipe == "expert_add"
        assert unpack.op_id in parent.deps
        assert {by_id[key].recipe for key in parent.deps} == {
            unpack.recipe, "shared_gate_apply" if gate else "dense_down"}
    else:
        assert parent == unpack
    for field in ("flops", "vector_ops", "hbm_bytes"):
        assert sum(getattr(op, field) for op in early if op.family != "allreduce") == sum(
            getattr(op, field) for op in late if op.family != "allreduce")
    before, after = (sum(op.link_bytes for op in ops if op.family == "allreduce") for ops in (early, late))
    assert before - after == reduce.link_bytes * (1 + shared)
    result = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=tokens))
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["bounds"]["combined_ms"] <= result["duration_ms"] + 1e-9


@pytest.mark.parametrize("ep,latent,reason", [(4, None, "tp_only"), (1, 16, "routed_output_transform")])
def test_tp_reduction_rejects_invalid_reordering(ep, latent, reason):
    raw = parse_case(moe_case(tp=4, ep=ep, shared=1)).to_dict()
    raw["runtime"]["moe_recipe"] = "tp_join_reduce"
    if latent is not None:
        raw["model"]["layer_plans"][0]["expert_latent_size"] = latent
    with pytest.raises(ModelingError, match=reason):
        target_operators(raw, phase="prefill", tokens=8)


@pytest.mark.parametrize("overlap,cut", [(0.5, 4), (0.4, 5)])
def test_deepep_receive_views_alias_and_wait_for_their_chunks(overlap, cut):
    tokens = 8
    raw = moe_case(dp=2, ep=2, tokens=tokens, top_k=1)
    raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": overlap}
    ops = target_graphs(raw, phase="prefill", tokens=tokens, scope="model")[0]
    dispatch, tail = (one(ops, recipe) for recipe in ("expert_dispatch", "expert_dispatch_overlap"))
    by_id = {op.op_id: op for op in ops}
    for start, stop, ready in ((0, cut, (dispatch.op_id,)), (cut, tokens, (dispatch.op_id, tail.op_id))):
        view = by_id[f"s0.l0.expert_up.rows{start}_{stop}"]
        (value,) = produced_values(view)
        assert view.deps == ready and dict(view.metadata)["byte_offset"] == start * 64
        assert tuple(value["logical_shape"]) == (stop - start, 32)
        assert tuple(value["storage_shape"]) == (tokens, 32)
        assert value["alias_of"] == dispatch.op_id and value["write_bytes"] == 0
        conversion = next(op for op in ops if op.family == "quantize" and op.deps == (view.op_id,))
        assert conversion.mte3_bytes == (stop - start) * (32 + 4)
