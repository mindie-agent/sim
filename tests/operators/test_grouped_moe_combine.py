"""Verify combine work, routing transport and late TP reduction.

[Role] Exercise tile and CTA work, DeepEP transfer and shared-expert ordering.
[Invariants] Tensor tails, weighted sums, and contiguous expert banks preserve
  exact work and dependencies across the operator and collective boundaries.
[Timeouts & Locks] Finite synthetic graphs and bounded lazy-window replay.
[Side-Effect Policy] In-memory modeling and Engine execution only.
"""
from dataclasses import asdict, replace

import numpy as np
import pytest

from _support.axes import SKUS
from _support.hardware import machine
from mindie_sim.schema.operators import OpSpec

from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.operators.moe_transport import deepep
from mindie_sim.prediction.implementations import produced_values
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators.moe_transport.moe_callers import LOCAL_EXPERT_REDUCE
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
                metadata=tuple(dict(group_rows=rows, n=n, k=hidden, tokens=tokens, top_k=topk).items()))
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


def test_mx_panel_template_alias_topology_and_cache_invalidation():
    import json
    from math import ceil, lcm
    from pathlib import Path
    from mindie_sim.prediction.operators.mx_quant_matrix import mxfp4_grouped
    from mindie_sim.prediction.engine import Allocation, Engine, Resource
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
                 metadata=tuple({**dict(gate.metadata), "matrix_group_rows": (1,), "matrix_weight_bank_groups": 1, "matrix_groups": 1}.items()))
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
