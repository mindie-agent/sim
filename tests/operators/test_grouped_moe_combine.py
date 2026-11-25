"""Verify combine work, routing transport and late TP reduction.

[Role] Exercise tile and CTA work, DeepEP transfer and shared-expert ordering.
[Invariants] Tensor tails, weighted sums, and contiguous expert banks preserve
  exact work and dependencies across the operator and collective boundaries.
[Timeouts & Locks] Finite synthetic graphs and bounded lazy-window replay.
[Side-Effect Policy] In-memory modeling and Engine execution only.
"""
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from _support.axes import DT_SKUS, DUAL_DIE_SKU, GEN2_SKU, SKUS
from _support.hardware import machine, sku_case
from mindie_sim.schema.operators import OpSpec

from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.operators.moe_transport import deepep
from mindie_sim.prediction.implementations import produced_values
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators.moe_transport.moe_callers import LOCAL_EXPERT_REDUCE, RoutingPolicy
from mindie_sim.prediction.operators import grouped_moe
from mindie_sim.prediction.operators.grouped_moe.lowering import StageDemand
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0
WAVE_EXPERTS = 16
REMOTE_ROW_BYTES = 8192
WIRE_HIDDEN_WIDTH = 4096
GATE_UP_WIDTH = 64
DEQUANT_CHANNELS = 128


@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("tokens,hidden,topk", [(3, 32, 2), (5, 130, 3), (17, 64, 1)])
def test_combine_reads_routed_values_and_publishes_one_row_per_token(hardware, tokens, hidden, topk):
    geometry = Geometry.from_devices(machine(hardware).dies[:1])
    rows = (tokens,) * topk + (0,)
    n = 64
    shape = (sum(rows), n, hidden)
    op = OpSpec("routed", "grouped_moe", "decode", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", recipe="mega_moe",
                metadata=tuple(dict(group_rows=rows, n=n, k=hidden, tokens=tokens, top_k=topk,
                                    ep=2, dispatch_link_bytes=0, combine_link_bytes=0).items()))
    plan = grouped_moe.build(op, geometry)
    phase = next(item for item in plan.pipeline_spec.phases if item.phase_id == "combine:local")
    payload = 2 * tokens * hidden
    expected_read = topk * payload + 4 * tokens * topk + 8 * len(rows)
    assert {item.resource: item.amount for item in phase.work} == {
        "hbm": expected_read + payload, "l2": expected_read + payload,
        "mte2_aiv": expected_read, "mte3": payload, "vector": 2 * tokens * topk * hidden,
    }
    assert phase.after == ("gmm2",)


@pytest.mark.parametrize("hardware", SKUS[2:])
@pytest.mark.parametrize("remote_expert", [None, 0, WAVE_EXPERTS])
def test_wave_waits_for_the_whole_remote_row(hardware, remote_expert):
    geometry = Geometry.from_devices(machine(hardware).dies[:1])
    experts = 2 * WAVE_EXPERTS
    shape = (experts, GATE_UP_WIDTH, WIRE_HIDDEN_WIDTH)

    def build_transport(name, after, _amount, metadata):
        rows = dict(metadata)["wave_expert_rows"][remote_expert]
        return (StageDemand(name, after, Work(resources={"link": rows * REMOTE_ROW_BYTES})),) if rows else ()

    fields = dict(group_rows=(1,) * experts, n=GATE_UP_WIDTH, k=WIRE_HIDDEN_WIDTH, wave_rows=WAVE_EXPERTS, ep=2,
                  dispatch_link_bytes=REMOTE_ROW_BYTES, combine_link_bytes=REMOTE_ROW_BYTES)
    if remote_expert is not None:
        fields["cohort_transport_builder"] = build_transport
    op = OpSpec("routed", "grouped_moe", "decode", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", recipe="mega_moe", metadata=tuple(fields.items()))
    plan = grouped_moe.build(op, geometry)
    phases = {phase.phase_id: phase for phase in plan.pipeline_spec.phases}
    wire = {name: item.amount for name, phase in phases.items() for item in phase.work if item.resource == "link"}
    prefix = "" if remote_expert is None else f"wave{remote_expert // WAVE_EXPERTS}_"
    dispatch, combine = (prefix + f"{direction}:transport" for direction in ("dispatch", "combine"))
    assert wire == {dispatch: REMOTE_ROW_BYTES, combine: REMOTE_ROW_BYTES}
    ancestry = {}
    for name, phase in phases.items():
        ancestry[name] = set(phase.after).union(*(ancestry[parent] for parent in phase.after))
    assert prefix + "gmm2" in ancestry[combine]
    for wave in range(2):
        gmm1 = f"wave{wave}_gmm1"
        waits_for_remote = remote_expert is None or wave >= remote_expert // WAVE_EXPERTS
        assert (dispatch in ancestry[gmm1]) == waits_for_remote


@pytest.mark.parametrize("hardware", SKUS[:2])
@pytest.mark.parametrize("rows,owner_reads", [((1,), (1028,)), ((3, 0, 5), (3604, 2572))])
def test_int32_dequant_reads_channel_scales_once_per_participating_core_and_group(hardware, rows, owner_reads):
    geometry = Geometry.from_devices(machine(hardware).dies[:1])
    geometry = replace(geometry, stream_owners=StreamOwners((), (0, 1), 0, 2))
    shape = (sum(rows), DEQUANT_CHANNELS)
    op = OpSpec("dequant", "grouped_moe", "decode", 0, 0, (), shape, shape, shape, shape, dtype="int32",
                recipe="dequant_swiglu_quant", metadata=(("group_rows", rows),
                                                         ("requested_abi", "aclnnDequantSwigluQuant")))
    plan = grouped_moe.build(op, geometry)
    reads = [{item.resource: item.amount for item in phase.work}
             for phase in plan.pipeline_spec.phases if phase.phase_id.endswith(":read")]
    assert reads == [dict.fromkeys(("hbm", "l2", "mte2_aiv"), amount) for amount in owner_reads]
    assert plan.work.resources["mte2_aiv"] == sum(owner_reads)


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
    other = ep - node_width
    hit_prob = 1.0 if top_k > other else 1.0 - math.comb(other, top_k) / math.comb(ep, top_k)
    assert group == pytest.approx(tokens * width * element_bytes * remote_nodes * hit_prob)
    assert intra == pytest.approx(tokens * width * element_bytes * top_k * (node_width - 1) / node_width)
    assert deepep.deepep_group_volume(tokens, width, top_k, eplb_peak_to_avg=1.25,
                                    **case) == pytest.approx(group * 1.25)


@pytest.mark.parametrize("ep,node_width", [(2, 2), (8, 4)])
def test_deepep_forward_and_inverse_transport_share_integer_row_cut(ep, node_width):
    raw = moe_case(dp=ep, ep=ep, tokens=4, top_k=2)
    plain = target_graphs(raw, phase="prefill", tokens=4, scope="model")[0]
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": 0.25,
                                  "deepep_ranks_per_node": node_width}
    parsed = parse_case(raw)
    received = int(RoutingPolicy().local(
        parsed.model, parsed.model.layer_plans[0], parsed.runtime.parallel,
        4 * ep, runtime=parsed.runtime)[0])
    main_rows, tail_rows = deepep.overlap_row_cut(received, 0.25)
    chunks = target_graphs(raw, phase="prefill", tokens=4, scope="model")[0]
    tokens, width, top_k = 4, 32, 2
    flat = tokens * width * 2 * top_k * (ep - 1) / ep
    nodes = ep // node_width
    remote_hit = (
        (nodes - 1) * (1.0 - math.comb(ep - node_width, top_k) / math.comb(ep, top_k))
        if nodes > 1 else 0.0
    )
    two_stage = tokens * width * 2 * (remote_hit + top_k * (node_width - 1) / node_width)
    for kind in ("dispatch", "combine"):
        recipe = f"expert_{kind}"
        full, tail = one(chunks, recipe), one(chunks, recipe + "_overlap")
        assert one(plain, recipe).link_bytes == pytest.approx(flat)
        assert (full.link_bytes, tail.link_bytes) == pytest.approx(
            (two_stage * (main_rows / received), two_stage * (tail_rows / received)))
        assert dict(one(chunks, "expert_dispatch").metadata)["deepep_row_cut"] == (main_rows, tail_rows)
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
    dispatch = one(ops, "expert_quantized_dispatch")
    scale_dispatch = one(ops, "expert_scale_dispatch")
    combine = one(ops, "expert_combine")
    assert dict(dispatch.metadata)["row_bytes"] == 32
    assert dict(scale_dispatch.metadata)["row_bytes"] == 2
    assert dispatch.link_bytes == 8 * 32 / 2 * factor
    assert scale_dispatch.link_bytes == 8 * 2 / 2 * factor
    assert combine.link_bytes == 8 * 64 / 2 * factor
    producer = next(op for op in ops if op.op_id in dispatch.deps and op.family == "quantize")
    assert producer.mte3_bytes == 8 * (32 + 2)
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


@pytest.mark.parametrize("overlap", [0.5, 0.4])
def test_deepep_receive_views_alias_and_wait_for_their_chunks(overlap):
    tokens = 8
    raw = moe_case(dp=2, ep=2, tokens=tokens, top_k=1)
    raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
    raw["runtime"]["metadata"] = {"enable_deepep": True, "deepep_overlap_ratio": overlap}
    parsed = parse_case(raw)
    received = int(RoutingPolicy().local(
        parsed.model, parsed.model.layer_plans[0], parsed.runtime.parallel,
        2 * tokens, runtime=parsed.runtime)[0])
    cut, _ = deepep.overlap_row_cut(received, overlap)
    ops = target_graphs(raw, phase="prefill", tokens=tokens, scope="model")[0]
    dispatch, tail = (one(ops, recipe) for recipe in ("expert_dispatch", "expert_dispatch_overlap"))
    by_id = {op.op_id: op for op in ops}
    for start, stop, ready in ((0, cut, (dispatch.op_id,)), (cut, received, (dispatch.op_id, tail.op_id))):
        view = by_id[f"s0.l0.expert_up.rows{start}_{stop}"]
        (value,) = produced_values(view)
        assert view.deps == ready and dict(view.metadata)["byte_offset"] == start * 64
        assert tuple(value["logical_shape"]) == (stop - start, 32)
        assert tuple(value["storage_shape"]) == (received, 32)
        assert value["alias_of"] == dispatch.op_id and value["write_bytes"] == 0
        conversion = next(op for op in ops if op.family == "quantize" and op.deps == (view.op_id,))
        assert conversion.mte3_bytes == (stop - start) * (32 + 4)


def test_mx_panel_template_alias_topology_and_cache_invalidation():
    import json
    from math import ceil, lcm
    from pathlib import Path
    from mindie_sim.prediction.operators.mx_quant_matrix import mxfp4_grouped
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.mechanism import PlannedRecipe
    from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_geometry import panel_sizes
    from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_panels import compiled_panels, panel_parts
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.operators.matrix.refill_steady import compact_refill

    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root / "examples/cases/v41-a5-dp4-body-current.json").read_text())
    raw["runtime"]["routing"] = {"policy": "balanced"}
    raw["runtime"].setdefault("kernels", {}).setdefault("families", {})["grouped_gemm"] = {"kind": "ascend_mxa8w4_nznk"}
    gate = next(op for op in target_graphs(raw, scope="body", phase="decode", tokens=1, context=8, sequences=1)[0] if op.family == "grouped_gemm")
    op = replace(gate, local_shape=(1, 160, 320), semantic_shape=(1, 160, 320), launch_shape=(1, 160, 320),
                 metadata=tuple({**dict(gate.metadata), "matrix_group_rows": (1,),
                                 "matrix_weight_bank_groups": 1, "matrix_groups": 1,
                                 "activation_scale_shape": (1, 5, 2)}.items()))
    contexts = []

    def capture(cur_op, geom):
        plan = mxfp4_grouped.build(cur_op, geom)
        return replace(plan, program=lambda ctx: contexts.append(ctx) or plan.program(ctx))

    operator_costs(raw, (op,), recipes=RecipeBook(families={"grouped_gemm": PlannedRecipe(capture)}))
    base_ctx = contexts[0]

    def uncached(ctx, args):
        kb, ka, _ = panel_sizes(args["m"], args["n"])
        panels, period = ceil(args["k"] / kb), lcm(ka, 4096) // kb
        count = panels // period - (panels % period == 0 and args["k"] % kb != 0)
        opts = {k: v for k, v in args.items() if k != "shared_clients"}
        parts = [panel_parts(ctx, index=idx, **opts) for idx in range(panels)]
        return compact_refill(ctx, tuple(parts[:period]) if count else (), count, tuple(parts[count * period:]), shared_clients=args["shared_clients"])

    def run_ir(prog, ctx):
        units = {name: ctx.resource_units.get(role, 1) for role, name in ctx.resource_names.items()}
        eng = Engine([Resource(n, rate, units=units.get(n, 1)) for n, rate in ctx.rates.items()], trace_limit=100)
        done = prog.start(eng, "source", eng.submit("input", release=0))
        eng.advance(until=(done,))
        return done.time, dict(eng.serviced)

    # Full cache-key invalidation across rates, units, peaks, service_modes, owner, repeat, and shared_clients
    hbm, mte = base_ctx.resource_names["hbm"], base_ctx.resource_names["mte2_aic"]
    key_ctx = replace(base_ctx, service_modes={**base_ctx.service_modes, "mte1_l0a": ("mte1", 1e12), "mte1_l0b": ("mte1", 5e11)})
    key_args = dict(k=5120, m=1, n=160, repeat=1, owner=0, cores=32, aivs=64, directional=True, shared_clients={hbm: 1})
    cached = compiled_panels(key_ctx, **key_args)
    for variant_ctx, variant_args in (
        (replace(key_ctx, rates={**key_ctx.rates, hbm: key_ctx.rates[hbm] * 0.5}), key_args),
        (replace(key_ctx, rates={**key_ctx.rates, mte: key_ctx.rates[mte] * 0.5}), key_args),
        (replace(key_ctx, resource_units={**key_ctx.resource_units, "mte2_aic": 16}), key_args),
        (replace(key_ctx, compute_peaks={**key_ctx.compute_peaks, "fp8_e4m3": key_ctx.compute_peaks["fp8_e4m3"] * 0.5}), key_args),
        (replace(key_ctx, service_modes={**key_ctx.service_modes, "mte1_l0a": ("mte1", 1e10)}), key_args),
        (key_ctx, {**key_args, "owner": 1}),
        (key_ctx, {**key_args, "repeat": 2}),
        (key_ctx, {**key_args, "shared_clients": {hbm: 32}}),
    ):
        assert compiled_panels(variant_ctx, **variant_args) is not cached
    assert compiled_panels(replace(key_ctx, op=None, setup_ms=10, minimum_ms=100), **key_args) is cached

    # Resource sort-order normalization & AIC/AIV alias overlapping owner sets
    rev_names = {name: f"z{idx:03d}" for idx, name in enumerate(reversed(sorted(base_ctx.rates)))}
    sorted_ctx = replace(base_ctx, resource_names={role: rev_names[name] for role, name in base_ctx.resource_names.items()},
                         rates={rev_names[name]: rate for name, rate in base_ctx.rates.items()})
    sort_args = dict(k=320, m=3, n=160, repeat=1, owner=7, cores=32, aivs=64, directional=False, shared_clients={sorted_ctx.resource_names["hbm"]: 32})
    assert run_ir(compiled_panels(sorted_ctx, **sort_args), sorted_ctx) == run_ir(uncached(sorted_ctx, sort_args), sorted_ctx)

    alias = base_ctx.resource_names["mte2_aiv"]
    alias_ctx = replace(base_ctx, resource_names={**base_ctx.resource_names, "mte2_aic": alias}, resource_units={**base_ctx.resource_units, "mte2_aic": 64})
    compiled_panels.cache_clear()
    for owner in (0, 7, 31):
        alias_args = dict(k=320, m=3, n=160, repeat=1, owner=owner, cores=32, aivs=64, directional=False, shared_clients={alias_ctx.resource_names["hbm"]: 8})
        bound = compiled_panels(alias_ctx, **alias_args)
        assert bound == uncached(alias_ctx, alias_args)
        load_stages = bound.stages[0].program.stages
        assert load_stages[0].placement[alias] == (owner * 2, owner * 2 + 1)
        assert load_stages[1].placement[alias] == (owner * 2,) and load_stages[7].placement[alias] == (owner * 2 + 1,)
    assert compiled_panels.template_cache_info().misses == 1


def test_mx_swiglu_subblock_owner_timing_and_single_batch_invocation():
    from collections import Counter
    import json
    from pathlib import Path
    from mindie_sim.prediction.operators.mx_quant_matrix import mxfp8_swiglu

    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root / "examples/cases/high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c32_p4k.json").read_text())
    role, profile = mxfp8_swiglu.ROLE, mxfp8_swiglu.PROFILE
    raw["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion",
        "options": {"roles": [role], "program_profiles": {role: profile}, "program_options": {role: {"l1_k": 512, "compact": True}}},
    }
    first = next(o for o in target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=32, resource_fusion=True)[0] if o.recipe == profile)
    batch_row = operator_costs(raw, [first])["operators"][0]
    prog_md = batch_row["operator"]["metadata"]["program_execution"]
    assert prog_md["source_output_tiles"] == 288 and prog_md["active_aic_owners"] == 32
    assert max(val for key, val in batch_row["peak_buffers_per_unit"].items() if key.startswith("ub@")) == 184832

    meta = dict(first.metadata)
    meta["mxfp8_swiglu_config"] = dict(meta["mxfp8_swiglu_config"], rows=(3,) * 33, width=64, k=128, compact=False)
    unequal = replace(first, local_shape=(99, 64), deps=(), metadata=tuple(meta.items()))
    report = graph_costs(raw, [unequal], isolated=False, input_ready_ms=7, trace_limit=10000)
    vec = [e for e in report["events"] if e["task"].endswith(":swiglu_mx")]
    assert len(vec) == 66
    observed = Counter(e["placement"]["vector@r0:s0"][0] for e in vec)
    assert observed[0] == observed[1] == 2 and all(observed[idx] == 1 for idx in range(2, 64))
    owner0 = sorted((e for e in vec if tuple(e["placement"]["vector@r0:s0"]) == (0,)), key=lambda e: e["start_ms"])
    assert owner0[-1]["start_ms"] >= owner0[0]["end_ms"]


def _calibrated_case_and_eval(sku: str, recipes):
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.primitives import load_primitives

    cs = sku_case(sku)
    pt = load_primitives(sku)
    ev = EvaluationContext.create(cs, recipes, primitive_table=pt)
    ex = Execution(cs, ev.cost, recipes=recipes, bindings=ev.bindings)
    return cs, ev, lambda op: ex._context(op, 0, "none")


def test_calibrated_matrix_host_tiling_warmth_invariance_and_l2_speedup():
    from mindie_sim.prediction.call_builders import build_aclnn_call
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul

    for sku in (GEN2_SKU, DUAL_DIE_SKU, DT_SKUS[0]):
        cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 16, "n": 4096, "k": 4096})
        cs, ev, ctx_of = _calibrated_case_and_eval(sku, cd.recipes)
        op_cold = cd.ops[0]
        read_bytes = float(16 * 4096 * 2 + 4096 * 4096 * 2)
        op_warm = replace(
            op_cold, op_id=f"{op_cold.op_id}_warm", hbm_bytes=float(16 * 4096 * 2),
            metadata=tuple({**dict(op_cold.metadata), "admitted_l2_read_hit_bytes": read_bytes}.items()),
        )
        rec_cold, _ = SteadyMatmul().select(ctx_of(op_cold))
        rec_warm, _ = SteadyMatmul().select(ctx_of(op_warm))
        assert rec_cold == rec_warm
        t_cold = graph_costs(cs, {0: [op_cold]}, recipes=cd.recipes, evaluation=ev, isolated=True)["duration_ms"]
        t_warm = graph_costs(cs, {0: [op_warm]}, recipes=cd.recipes, evaluation=ev, isolated=True)["duration_ms"]
        assert 0.0 < t_warm < t_cold


def test_asymmetric_nd2nz_mte2_and_streamk_partial_l2_residency():
    from mindie_sim.prediction.call_builders import build_aclnn_call
    from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
        primitive_nd2nz_mte2_rate_bps_per_aic, streamk_fixup_latency_ms,
    )
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul

    cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 16, "n": 256, "k": 4096})
    op = cd.ops[0]
    cs_old, ev_old, ctx_old_of = _calibrated_case_and_eval(GEN2_SKU, cd.recipes)
    cs_new, ev_new, ctx_new_of = _calibrated_case_and_eval(DT_SKUS[0], cd.recipes)
    ctx_old, ctx_new = ctx_old_of(op), ctx_new_of(op)
    nd_sub = primitive_nd2nz_mte2_rate_bps_per_aic(ctx_old, 2, 16 * 4096 * 2, 128 * 4096 * 2, weight_layout="ND", row_bytes=256, stride_bytes=512)
    nd_full = primitive_nd2nz_mte2_rate_bps_per_aic(ctx_old, 2, 16 * 4096 * 2, 256 * 4096 * 2, weight_layout="ND", row_bytes=512, stride_bytes=512)
    nz_rate = primitive_nd2nz_mte2_rate_bps_per_aic(ctx_old, 2, 16 * 4096 * 2, 256 * 4096 * 2, weight_layout="FRACTAL_NZ", row_bytes=512, stride_bytes=512)
    sym_nd = primitive_nd2nz_mte2_rate_bps_per_aic(ctx_new, 2, 16 * 4096 * 2, 128 * 4096 * 2, weight_layout="ND", row_bytes=256, stride_bytes=512)
    sym_nz = primitive_nd2nz_mte2_rate_bps_per_aic(ctx_new, 2, 16 * 4096 * 2, 128 * 4096 * 2, weight_layout="FRACTAL_NZ", row_bytes=256, stride_bytes=512)
    assert 0.0 < nd_sub < nd_full < nz_rate
    assert sym_nd == pytest.approx(sym_nz)
    rec_old, _ = SteadyMatmul().select(ctx_old)
    rec_new, _ = SteadyMatmul().select(ctx_new)
    assert rec_old.k_splits == 1 and rec_new.k_splits > 1
    assert streamk_fixup_latency_ms(ctx_new, k_splits=rec_new.k_splits, used_cores=ctx_new.core_count) > 0.0
    rep_old = graph_costs(cs_old, {0: [op]}, recipes=cd.recipes, evaluation=ev_old, isolated=True)
    rep_new = graph_costs(cs_new, {0: [op]}, recipes=cd.recipes, evaluation=ev_new, isolated=True)
    assert rep_old["execution"]["work"]["flops"] == op.flops and rep_new["execution"]["work"]["flops"] == op.flops
    assert 0.0 < rep_new["duration_ms"] < rep_old["duration_ms"]


def test_grouped_matmul_aclnn_group_list_cache_disable_and_wave_hbm():
    from mindie_sim.prediction.call_builders import build_aclnn_call
    from mindie_sim.prediction.operators.grouped_moe.expected_grouped import _apply_grouped_weight_cache_policy
    from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
        matrix_kernel_setup_ms, primitive_cold_hbm_bps, primitive_wave_cold_hbm_per_owner_bps,
    )

    skew = [64, 0, 32, 0, 16, 0, 8, 0]
    cd = build_aclnn_call({"api": "aclnnGroupedMatmulV4", "dtype": "bf16", "group_list": skew, "n": 1024, "k": 1024})
    op_cold = cd.ops[0]
    w_bytes = float(len(skew) * 1024 * 1024 * 2)
    op_warm = replace(
        op_cold, op_id=f"{op_cold.op_id}_warm", hbm_bytes=float(sum(skew) * 1024 * 2 * 2),
        metadata=tuple({**dict(op_cold.metadata), "admitted_l2_read_hit_bytes": w_bytes}.items()),
    )
    for sku in (GEN2_SKU, DT_SKUS[0]):
        cs, ev, ctx_of = _calibrated_case_and_eval(sku, cd.recipes)
        ctx = ctx_of(op_cold)
        assert matrix_kernel_setup_ms(ctx, declared_groups=64) > matrix_kernel_setup_ms(ctx, declared_groups=8)
        meta_small = {"admitted_l2_read_hit_bytes": w_bytes}
        _apply_grouped_weight_cache_policy(
            ctx, meta_small, active_rows=(64, 32, 16, 8), declared_groups=8, rows=120, n=1024, k=1024, dtype="bf16",
        )
        if sku == GEN2_SKU:
            assert meta_small.get("l2_weight_cache_disable") is True and meta_small["admitted_l2_read_hit_bytes"] == pytest.approx(120 * 1024 * 2)
        else:
            assert "l2_weight_cache_disable" not in meta_small and meta_small["admitted_l2_read_hit_bytes"] == w_bytes
            meta_over = {"admitted_l2_read_hit_bytes": w_bytes}
            _apply_grouped_weight_cache_policy(ctx, meta_over, active_rows=(64,), declared_groups=128, rows=64, n=1024, k=1024, dtype="bf16")
            assert meta_over.get("l2_weight_cache_disable") is True
        t_cold = graph_costs(cs, {0: [op_cold]}, recipes=cd.recipes, evaluation=ev, isolated=True)["duration_ms"]
        t_warm = graph_costs(cs, {0: [op_warm]}, recipes=cd.recipes, evaluation=ev, isolated=True)["duration_ms"]
        assert 0.0 < t_warm <= t_cold
        cores = int(ctx.core_count)
        cold_tiles = cores + max(1, cores // 3)
        per_owner = primitive_wave_cold_hbm_per_owner_bps(ctx, cold_tiles=cold_tiles, row_bytes=256, stride_bytes=2048, is_grouped=True, is_full_row_coalesced=True, nn_tiles=8)
        full_agg = primitive_cold_hbm_bps(ctx, active_cores=cores, row_bytes=256, stride_bytes=2048, total_tiles=cold_tiles, is_grouped=True, is_full_row_coalesced=True)
        tail = cold_tiles % cores
        tail_agg = primitive_cold_hbm_bps(ctx, active_cores=tail, row_bytes=256, stride_bytes=2048, total_tiles=cold_tiles, is_grouped=True, is_full_row_coalesced=(tail >= 8))
        hbm_pool = float(ctx.rates[ctx.resource_names["hbm"]]) * 1000.0
        assert (cores / min(hbm_pool, cores * per_owner) + tail / min(hbm_pool, tail * per_owner)) == pytest.approx(cores / full_agg + tail / tail_agg, rel=1e-9)


def test_uncalibrated_matrix_helpers_return_neutral_defaults():
    from mindie_sim.prediction.call_builders import build_aclnn_call
    from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
    from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
        matrix_kernel_floor_ms, matrix_kernel_setup_ms, primitive_cold_hbm_bps,
        primitive_p05_stride_conflict_bps, primitive_wave_cold_hbm_per_owner_bps, streamk_fixup_latency_ms,
    )

    cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 16, "n": 256, "k": 1024})
    _, _, ctx_of = _calibrated_case_and_eval(GEN2_SKU, cd.recipes)
    cal_ctx = ctx_of(cd.ops[0])
    assert 0.0 < matrix_kernel_floor_ms(cal_ctx) < matrix_kernel_setup_ms(cal_ctx, is_streamk=True)
    assert primitive_p05_stride_conflict_bps(cal_ctx, 3 * 16384) == primitive_p05_stride_conflict_bps(cal_ctx, 16384)
    big_ctx = replace(cal_ctx, op=replace(cd.ops[0], hbm_bytes=512e6))
    kw = dict(active_cores=cal_ctx.core_count, row_bytes=512, stride_bytes=65536, total_tiles=cal_ctx.core_count, is_grouped=False)
    assert primitive_cold_hbm_bps(big_ctx, **kw) > primitive_cold_hbm_bps(cal_ctx, **kw)
    cd_lg = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 4, "n": 4096, "k": 4096})
    cal_lg = ctx_of(cd_lg.ops[0])
    cal_sel = MappedMatmul().select(cal_lg)[0]
    assert cal_sel.tile_n == 256 and MappedMatmul().select(replace(cal_lg, primitive_table=None))[0] == cal_sel
    cd_big = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 1024, "n": 32768, "k": 8192})
    _, _, ctx_a5 = _calibrated_case_and_eval(DT_SKUS[0], cd_big.recipes)
    ctx_big = ctx_a5(cd_big.ops[0])
    assert primitive_cold_hbm_bps(ctx_big, active_cores=ctx_big.core_count, row_bytes=512, stride_bytes=65536, total_tiles=128, is_grouped=False, m_tiles=4) < 1200e9
    assert primitive_cold_hbm_bps(ctx_big, active_cores=ctx_big.core_count, row_bytes=512, stride_bytes=65536, total_tiles=128, is_grouped=False, m_tiles=1) > 2000e9
    assert matrix_kernel_setup_ms(ctx_big, is_streamk=False) == pytest.approx(matrix_kernel_floor_ms(ctx_big))
    ctx = replace(cal_ctx, primitive_table=None)
    assert matrix_kernel_setup_ms(ctx, is_streamk=True, declared_groups=16) == 0.0
    assert matrix_kernel_floor_ms(ctx) == 0.0 and streamk_fixup_latency_ms(ctx, k_splits=4, used_cores=8) == 0.0
    assert primitive_wave_cold_hbm_per_owner_bps(ctx, cold_tiles=8, row_bytes=256, stride_bytes=512, is_grouped=False, is_full_row_coalesced=False) is None


def test_grouped_moe_abi_fp32_ub_expected_groups_and_a5_sector_eff_monotonicity():
    from _support.hardware import machine
    from mindie_sim.compose import compose_case
    from mindie_sim.hardware_derivation import derive_hardware_sku
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
    from mindie_sim.prediction.operators import grouped_moe as gm_pkg
    from mindie_sim.prediction.operators.grouped_moe.common import validate_contract
    from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.results import OpOccurrence
    from mindie_sim.primitives.loader import load_primitives

    # 1. Typed shipping activation predicates on grouped_moe arch22/arch35 + FP32 UB upcast.
    a5_geo = Geometry.from_devices(machine(DT_SKUS[0]).dies[:1])
    a3_geo = Geometry.from_devices(machine(GEN2_SKU).dies[:1])
    a5_ctx = OperatorPlanningContext(OpOccurrence(0, 0, 0, "gm_abi"), a5_geo, a5_geo.datapath)
    a3_ctx = OperatorPlanningContext(OpOccurrence(0, 0, 0, "gm_abi"), a3_geo, a3_geo.datapath)
    gm_arch35_strat = gm_pkg.strategy_for_arch(a5_ctx.compute_arch.arch_id)
    gm_arch22_strat = gm_pkg.strategy_for_arch(a3_ctx.compute_arch.arch_id)
    situ_meta = (
        ("mode", "mega_moe"), ("group_rows", (4, 4)), ("n", 128), ("k", 64),
        ("dtype_a", "bf16"), ("dtype_b", "bf16"), ("output_dtype", "bf16"),
        ("activation", "situ"), ("handoff_dtype", "bf16"),
        ("ep", 2), ("dispatch_link_bytes", 0), ("combine_link_bytes", 0),
    )
    situ_op = OpSpec("gm_situ", "grouped_moe", "decode", 0, 0, ("r0",), (8, 128, 64),
                     (8, 128, 64), (8, 128, 64), (8, 128, 64), dtype="bf16",
                     recipe="mega_moe", metadata=situ_meta)
    assert gm_arch35_strat is not None and not gm_arch35_strat.accepts(situ_op, a5_ctx)
    assert gm_arch22_strat is not None and not gm_arch22_strat.accepts(situ_op, a3_ctx)
    ext_op = replace(situ_op, metadata=situ_meta + (("abi_variant_extension", True),))
    assert gm_arch35_strat.accepts(ext_op, a5_ctx)
    assert gm_arch22_strat.accepts(ext_op, a3_ctx)
    _, slot_bytes = gm_pkg.fit_arch35_ub(validate_contract(ext_op, a5_ctx), a5_ctx)
    # BF16 handoff still budgets FP32 (4 B/elem) * 2 (gate+up) * rows * cols in UB.
    assert slot_bytes == 4 * 64 * 2 * 4

    # 2. ExpectedGroupedMatmul stamps declared_group_count & L2 cache policy on probability_moments.
    raw_k3 = compose_case("kimi-k3", DT_SKUS[0], tp=2, dp=1, ep=2, layers=2, prompt_tokens=1024, output_tokens=64)
    parsed_k3 = parse_case(raw_k3)
    g_prob = target_graphs(parsed_k3, phase="decode", tokens=4, context=1024, sequences=4, scope="body")[0]
    exp_up = next(op for op in g_prob if op.op_id.endswith(".expert_up"))
    assert dict(exp_up.metadata).get("matrix_weight_bank_groups", 0) > 1
    _, _, ctx_of = _calibrated_case_and_eval(DT_SKUS[0], RecipeBook.from_mapping(parsed_k3.runtime.kernels))
    captured_ops = []

    class _CaptureMatrix(SteadyMatmul):
        def lower(self, context, **kwargs):
            captured_ops.append(context.op)
            return super().lower(context, **kwargs)

    ExpectedGroupedMatmul(_CaptureMatrix()).lower(ctx_of(exp_up))
    assert dict(captured_ops[0].metadata)["declared_group_count"] >= dict(exp_up.metadata)["matrix_weight_bank_groups"]

    # 3. Calibrated PrimitiveTable A5 sector efficiency prevents M=1>M=4 inversion and 4->1.6 TB/s cliff.
    pt_a5 = load_primitives("Ascend950DT_9582")
    ev_k3 = EvaluationContext.create(parsed_k3, primitive_table=pt_a5)
    g1 = target_graphs(parsed_k3, phase="decode", tokens=1, context=1024, sequences=1, scope="body")[0]
    g4 = target_graphs(parsed_k3, phase="decode", tokens=4, context=1024, sequences=1, scope="body")[0]
    gc1 = graph_costs(parsed_k3, g1, evaluation=ev_k3, external_inputs={"analysis_input": 0})
    gc4 = graph_costs(parsed_k3, g4, evaluation=ev_k3, external_inputs={"analysis_input": 0})
    def d_of(gc, oid):
        return next(float(r.get("duration_ms", r["complete_ms"] - r["ready_ms"])) for r in gc["timeline"] if r["op_id"] == oid)
    assert d_of(gc4, "s0.l0.linear_qkvbfg_a") >= d_of(gc1, "s0.l0.linear_qkvbfg_a") - 1e-6
    assert gc4["duration_ms"] >= gc1["duration_ms"] - 1e-6

    def _scale_pt_hbm(pt, scale, sku):
        prims = {}
        for pid, cell in pt.primitives.items():
            if pid != "P01_datacopy":
                prims[pid] = cell
                continue
            metrics = {
                k: (replace(b, certified_upper_capacity=b.certified_upper_capacity * scale,
                            p05=None if b.p05 is None else b.p05 * scale,
                            p50=None if b.p50 is None else b.p50 * scale,
                            p95=None if b.p95 is None else b.p95 * scale,
                            p99=None if b.p99 is None else b.p99 * scale,
                            trimmed_mean=None if b.trimmed_mean is None else b.trimmed_mean * scale)
                    if "hbm" in k else b)
                for k, b in cell.metrics.items()
            }
            params = dict(cell.parameters or {})
            if "mte2_aic_cold_scaling_bps" in params:
                params["mte2_aic_cold_scaling_bps"] = {k: float(v) * scale for k, v in params["mte2_aic_cold_scaling_bps"].items()}
            if "strided_cold_surface" in params:
                params["strided_cold_surface"] = [{**e, "bps_p50": float(e["bps_p50"]) * scale} for e in params["strided_cold_surface"]]
            prims[pid] = replace(cell, metrics=metrics, parameters=params)
        return replace(pt, sku=sku, primitives=prims)

    gb = target_graphs(parsed_k3, phase="decode", tokens=1, context=1024, sequences=4, scope="body")[0]
    gate_durs = []
    for bw in (4.0e12, 1.6e12, 1.4e12):
        sku_name = f"s-{int(bw // 1e11)}"
        c_bw = replace(parsed_k3, machine=derive_hardware_sku(parsed_k3.machine, sku=sku_name, hbm_peak_bytes_per_s=bw))
        ev_bw = EvaluationContext.create(c_bw, primitive_table=_scale_pt_hbm(pt_a5, bw / 4.0e12, sku_name))
        gc_bw = graph_costs(c_bw, gb, evaluation=ev_bw, external_inputs={"analysis_input": 0})
        gate_durs.append(d_of(gc_bw, "s0.l0.gate_up"))
    assert 1.5 <= gate_durs[1] / gate_durs[0] <= 3.5
    assert 1.0 <= gate_durs[2] / gate_durs[1] <= 1.5
