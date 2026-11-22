"""MLA prolog, compressor, and KeyPool physical contracts.

[Role] Exercise governed plans and the native callers that produce their inputs.
[Invariants] Products, owners, cache rows, workspace and dependency edges conserve
  the work implied by each declared ABI and geometry.
[Timeouts & Locks] Bounded local simulation. No locks or device access.
[Side-Effect Policy] Reads packaged cases and uses test-local records.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import pytest
from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import sku_case
from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_format import block_quantized_row_bytes, cache_format
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.quant import projection_quant, quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.operators.mla_prolog import draft_context
from mindie_sim.prediction.operators.mla_prolog.compression import compressor
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.mla.mla_epilogs import index_head_weights, index_row_quantization
from mindie_sim.prediction.operators.mla.mla_native import index_layer_norm
from mindie_sim.prediction.operators.hyper_connections.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators import FamilyRegistry, OperatorPlanningContext
from mindie_sim.prediction.operators import mla_prolog as governed, paged_mla
from mindie_sim.prediction.operators.mla_prolog.common import (
    ABI_DSL, ABI_V3, ABI_V3_NZ, ABI_V4_NZ, WEIGHT_QUANT_BF16, WEIGHT_QUANT_INT8, WEIGHT_QUANT_MXFP8)
from mindie_sim.prediction.operators.paged_mla.common import DSL_MQS_ABI, MQS_ABI
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import StreamCoreBudget, StreamLane, StreamPlan
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, MlaSpec, ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.stack_registry import LoweringRequest, RankOps
from mindie_sim.structure import export_structure
from mindie_sim.structure_subgraphs import subgraph_contract

pytestmark = pytest.mark.tier0
CASES = Path(__file__).resolve().parents[2] / "examples/cases"
CONCURRENT = CASES / "high-concurrency-20261030"
PACKED_LAYOUT = "packed_int8_bf16_rope_fp32_scale"
PACKED_PROFILE = "mla_prolog_packed_int8_tile128_analytic"
FULL_QUANT_PROFILE = "mla_prolog_arch22_fullquant_a3925fd4"
SPLIT_N_SKU, SPLIT_M_SKU = REPRESENTATIVE_SKUS["a2"], REPRESENTATIVE_SKUS["a5"]
def _context(sku, op, device=None):
    geom = Geometry.from_devices(sku_case(sku, device).machine.dies)
    return OperatorPlanningContext(OpOccurrence(0, op.stage, 0, op.op_id), geom, geom.datapath)
def _prolog(tokens=5, **overrides):
    fields = dict(op_id="prolog", tokens=tokens, hidden_size=5120, num_heads=16,
                  q_lora_rank=1280, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64)
    return governed.make_op(**(fields | overrides))
@pytest.mark.parametrize("sku", SKUS)
def test_four_products_conserve_work_through_execution_and_projection(sku):
    op = _prolog(weight_quant_mode=WEIGHT_QUANT_BF16, kv_cache_quant_mode=0)
    assert op.flops == 2 * 5 * (5120 * 1280 + 5120 * (512 + 64)
                               + 1280 * 16 * (128 + 64) + 16 * 128 * 512)
    ctx = _context(sku, op)
    resolved = governed.resolve_operator_plan(op, ctx)
    plan, descriptor = resolved.recipe, resolved.descriptor
    assert isinstance(descriptor, FusedPipelineDescriptor) and descriptor.owner == ctx.occurrence
    cubes = {p.phase_id for p in descriptor.phases if p.phase_id.endswith("_cube")}
    assert descriptor.implementation == ABI_V3_NZ and len(cubes) == 4 and "mm4_w_uk_cube" in cubes
    amounts = Counter()
    for phase in descriptor.phases:
        amounts.update({item.resource: item.amount for item in phase.work})
    assert amounts == pytest.approx({**plan.work.resources, "cube": sum(plan.work.matrix.values())})
    delegated, replayed = governed.build(op, ctx.geometry), pickle.loads(pickle.dumps(plan))
    assert (delegated.work, delegated.pipeline_spec) == (replayed.work, replayed.pipeline_spec) == (plan.work, plan.pipeline_spec)
    assert verify_operator(sku_case(sku), op, governed.build, input_ready_ms=7)["status"] == "passed"
    spine = PredictionSpine(families=FamilyRegistry((governed.register_family(),)))
    step = spine.lower(sku_case(sku), LoweringRequest(declared_ops=(RankOps(0, (op,)),)))
    result = spine.run_step(pickle.loads(pickle.dumps(step)))
    assert result.span_ms > 0 and result.pipelines == (descriptor,) and pickle.loads(pickle.dumps(result)) == result
@pytest.mark.parametrize("heads,ownership", [(16, (1, 16)), (48, (2, 24))])
def test_absorption_assigns_every_head_to_one_matrix_owner(heads, ownership):
    op = _prolog(num_heads=heads)
    plan = governed.build(op, _context(SPLIT_M_SKU, op))
    assert (plan.metadata["heads_per_owner"], plan.metadata["w_uk_owners"]) == governed.w_uk_ownership(heads, 32) == ownership
    cube = next(p for p in plan.program.groups[0][0].phases if p.name == "mm4_w_uk_cube")
    assert {d.owner: dict(d.audit)["heads"] for d in cube.demands} == dict.fromkeys(range(ownership[1]), ownership[0])
@pytest.mark.parametrize("nope,step,slot", [(128, 128, 3_182_592), (192, 64, 1_984_512)])
def test_cq_scale_slot_is_published_and_gathered_across_chunk_tails(nope, step, slot):
    op = _prolog(step + 1, hidden_size=7168, q_lora_rank=1536, qk_nope_head_dim=nope,
                 weight_quant_mode=WEIGHT_QUANT_INT8, kv_cache_quant_mode=0)
    ctx = _context(SPLIT_N_SKU, op)
    plan = governed.build(op, ctx)
    assert plan.metadata["step_batch"] == step
    bytes_per_row = 4 * (1536 + 512 + 64 + 16 * (nope + 64)) + 2 * 16 * nope
    full, tail = plan.program.groups[0]
    assert full.workspace_bytes == plan.metadata["workspace_slot_bytes"] == step * (bytes_per_row + 32) == slot
    assert (full.count, tail.count, tail.workspace_bytes) == (1, 1, bytes_per_row + step * 32)
    for chunk, rows in ((full, step), (tail, 1)):
        phases = {phase.name: phase for phase in chunk.phases}
        writes = phases["mm1_cq_fixpipe_gm"]
        assert phases["cq_norm_mte2_aiv"].after == (writes.name,)
        assert _logical_bytes(writes) == rows * 1536 * 4
        assert _service_bytes(writes, "hbm") >= rows * 1536 * 4
        publish, gather = phases["cq_scale_mte3_gm"], phases["cq_scale_mte2_aiv"]
        assert publish.after == ("cq_norm_vector",)
        assert set(gather.after) == {publish.name, "cq_norm_mte3_gm"}
        assert phases["mm3_qc_qr_mte2_aic"].after == (gather.name,)
        assert _logical_bytes(publish) == _service_bytes(publish, "mte3") == rows * 32
        assert len(gather.demands) == ctx.effective_aiv_cores
        assert _service_bytes(gather, "mte2_aiv") == ctx.effective_aiv_cores * step * 32
    _execute(op, plan, ctx.geometry)
@pytest.mark.parametrize("weight,heads,cache,enabled", [
    (WEIGHT_QUANT_BF16, 16, 0, False), (WEIGHT_QUANT_INT8, 7, 0, False),
    (WEIGHT_QUANT_INT8, 8, 0, True), (WEIGHT_QUANT_INT8, 8, 1, True),
])
def test_cq_scale_workspace_follows_the_host_dequant_predicate(weight, heads, cache, enabled):
    op = _prolog(1, num_heads=heads, weight_quant_mode=weight, kv_cache_quant_mode=cache)
    spec = governed.build(op, _context(SPLIT_N_SKU, op)).pipeline_spec
    assert any(p.phase_id == "cq_scale_mte2_aiv" for p in spec.phases) is enabled
@pytest.mark.parametrize("abi", (ABI_V3, ABI_V3_NZ, ABI_V4_NZ))
def test_aclnn_rejects_illegal_cache_tiles_during_contract_validation(abi):
    with pytest.raises(ModelingError, match="invalid_tile_size:64"):
        _prolog(abi=abi, tile_size=64)
@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("abi", (ABI_V4_NZ, ABI_DSL))
def test_prolog_abi_availability_follows_declared_architecture(arch, abi):
    split_n = arch == ARCH_IDS[0].values[0]
    op = _prolog(abi=abi)
    ctx = _context(SPLIT_N_SKU if split_n else SPLIT_M_SKU, op)
    if ctx.compute_arch.arch_id == ARCH_IDS[0].values[0]:
        with pytest.raises(UnmodeledCapabilityError, match="unsupported_abi_or_weight_mode"):
            governed.build(op, ctx)
    else:
        assert governed.build(op, ctx).metadata["implementation"] == abi
@pytest.mark.parametrize("tokens,hidden,heads,weight,expected", [
    (8192, 7168, 128, WEIGHT_QUANT_MXFP8, True), (8192, 7680, 128, WEIGHT_QUANT_MXFP8, True),
    (8191, 7168, 128, WEIGHT_QUANT_MXFP8, False), (8192, 5120, 128, WEIGHT_QUANT_MXFP8, False),
    (8192, 7168, 64, WEIGHT_QUANT_MXFP8, False), (8192, 7168, 128, WEIGHT_QUANT_INT8, False),
])
def test_split_m_requires_every_host_tiling_predicate(tokens, hidden, heads, weight, expected):
    op = _prolog(tokens, hidden_size=hidden, num_heads=heads, weight_quant_mode=weight)
    assert governed.uses_split_m(governed.validate_contract(op)) is expected
    assert governed.build(op, _context(SPLIT_M_SKU, op)).metadata["split_mode"] == ("split_m" if expected else "split_n")
def _logical_bytes(phase):
    return sum(dict(d.audit)["logical_bytes"] for d in phase.demands)
def _service_bytes(phase, resource):
    return sum(d.work.resources.get(resource, 0) for d in phase.demands)
def _execute(op, plan, geometry):
    units = geometry.resource_units
    rates, pools = {role: geometry.rates.get(role, 1000) / 1000 for role in units}, {"hbm": 10**9, **geometry.buffers}
    binding = RecipeContext(plan.work.apply(op), {}, {r: r for r in units}, rates,
                            geometry.compute_units, resource_units=units,
                            compute_peaks=geometry.compute_peaks, pool_names={p: p for p in pools},
                            pool_capacities=pools, pool_units=geometry.buffer_units)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in rates.items()],
                    pools=pools, pool_units=geometry.buffer_units, trace_limit=10_000)
    done = plan.lower(binding).start(engine, op.op_id, engine.submit("input", release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert {k: v for k, v in engine.serviced.items() if v} == pytest.approx(binding.bind(plan.work), rel=1e-12)
    _drained({"execution": engine.summary()})
    return engine.summary()["peak_allocated"].get("hbm", 0)
@pytest.mark.parametrize("sku,abi,mode,size", (
    (SPLIT_N_SKU, ABI_V3_NZ, WEIGHT_QUANT_BF16, 2), (SPLIT_N_SKU, ABI_V3_NZ, WEIGHT_QUANT_INT8, 4),
    (SPLIT_M_SKU, ABI_V4_NZ, WEIGHT_QUANT_BF16, 2), (SPLIT_M_SKU, ABI_V4_NZ, WEIGHT_QUANT_INT8, 4),
    (SPLIT_M_SKU, ABI_V4_NZ, WEIGHT_QUANT_MXFP8, 4),
))
def test_projection_outputs_use_native_gm_element_widths(sku, abi, mode, size):
    op = _prolog(1, abi=abi, q_lora_rank=1536, num_heads=4, weight_quant_mode=mode,
                 kv_cache_quant_mode=0, cache_mode="BSND", do_rope=mode != WEIGHT_QUANT_BF16)
    ctx = _context(sku, op)
    plan = governed.build(op, ctx)
    phases = {p.name: p for p in plan.program.groups[0][0].phases}
    cq, kv, query = (size * n for n in (1536, 512 + 64, 4 * (128 + 64)))
    for name, expected in zip(("mm1_cq", "mm2_ckv_kr", "mm3_qc_qr"), (cq, kv, query), strict=True):
        write = phases[name + "_fixpipe_gm"]
        assert _logical_bytes(write) == _service_bytes(write, "fixpipe") == expected
    norm = tuple(d for d in phases["cq_norm_mte2_aiv"].demands if dict(d.audit)["tensor"] == "cq_norm")
    assert sum(d.work.resources["mte2_aiv"] for d in norm) == cq
    rope_tables = 2 * 64 * 4 if mode != WEIGHT_QUANT_BF16 else 0
    assert _logical_bytes(phases["kv_cache_mte2_aiv"]) == kv + 512 * 2 + rope_tables
    assert _logical_bytes(phases["query_rope_mte2_aiv"]) == 4 * 64 * size + rope_tables
    cast_bytes = 0 if mode == WEIGHT_QUANT_BF16 else 4 * 128 * 2
    if cast_bytes:
        assert _service_bytes(phases["qc_dequant_mte2_aiv"], "mte2_aiv") == 4 * 128 * size
    assert _execute(op, plan, ctx.geometry) == plan.metadata["workspace_slot_bytes"] == cq + kv + query + cast_bytes
@pytest.mark.parametrize("tokens,padded,splits,ub_kib,workspace", (
    (17, 32, 4, None, 917_504), (4096, 4128, 1, None, 29_589_504), (4096, 4128, 20, 72, 591_790_080)))
def test_dsl_split_k_exchanges_fp32_partials_and_retains_local_query(tokens, padded, splits, ub_kib, workspace):
    resident = tokens < 256
    op = _prolog(tokens, abi=ABI_DSL, dsl_d_width=512, phase="decode" if resident else "prefill")
    assert op.flops == 2 * tokens * (5120 * (1280 + 512) + 1280 * 16 * 512)
    ctx = _context(SPLIT_M_SKU, op, {"buffers": {"ub": ub_kib * 1024}} if ub_kib else None)
    plan = governed.build(op, ctx)
    assert plan.metadata["l1_reuse"] is resident and plan.metadata["k_splits"] == splits
    chunk, = plan.program.groups[0]
    phases = {p.name: p for p in chunk.phases}
    for product, tensor, width, ops in (("mm1_wqa", "qa_partials", 1280, 4), ("mm2_wkv", "kv_partials", 512, 1)):
        write, read = phases[product + "_fixpipe_gm"], phases[tensor + "_mte2_aiv"]
        assert write.name in read.after
        live = tokens * splits * width * 4
        assert _logical_bytes(write) == _service_bytes(read, "mte2_aiv") == _service_bytes(read, "hbm") == live
        assert _service_bytes(write, "fixpipe") == _service_bytes(write, "hbm") == padded * splits * width * 4
        post = "cq_norm" if tensor == "qa_partials" else "kv_cache"
        reduce = phases.get(tensor + "_reduce_vector")
        assert (reduce is not None) == (not resident and splits > 1)
        if reduce is not None:
            assert reduce.after == (read.name,)
            assert _service_bytes(reduce, "vector") == tokens * width * (splits - 1) * ops
        assert phases[post + "_mte2_aiv"].after == ((reduce or read).name,)
        if resident:
            assert {dict(d.audit)["split"] for d in write.demands} == set(range(splits))
            assert len({d.owner for d in write.demands}) > 1
        else:
            assert sum(dict(d.audit)["logical_flops"] for d in phases[product + "_cube"].demands) == 2 * tokens * width * 5120
    assert phases["mm3_wqb_mte2_aic"].after == ("cq_norm_mte3_gm",)
    assert all(set(d.work.resources) == {"fixpipe"} for d in phases["mm3_wqb_fixpipe_ub"].demands)
    assert _execute(op, plan, ctx.geometry) == plan.metadata["workspace_slot_bytes"] == workspace
    if resident:
        assert plan.metadata["step_batch"] == 32 and plan.work.matrix == {"fp8_e4m3": op.flops * 32 // 17}
        assert sum(dict(d.audit)["logical_flops"] for d in phases["a1_cube"].demands) == 2 * tokens * 5120 * (1280 + 512)
        no_l1 = replace(ctx, geometry=replace(ctx.geometry, buffers=dict(ctx.geometry.buffers) | {"l1": 1}))
        spill = governed.build(op, no_l1)
        assert not spill.metadata["l1_reuse"] and spill.metadata["logical_flops"] == plan.metadata["logical_flops"]
        paths = tuple(p for p in ctx.datapath.usable_paths if (p.source, p.target) != ("l0c", "ub"))
        with pytest.raises(ModelingError, match="l0c.*ub"):
            governed.build(op, replace(ctx, datapath=replace(ctx.datapath, usable_paths=paths)))
def test_disjoint_stream_uses_reachable_physical_owner_ids():
    op = replace(_prolog(17, abi=ABI_DSL, dsl_d_width=512), stream="prolog")
    streams = StreamPlan(0, (StreamLane("main", StreamCoreBudget(7, 14)),
                             StreamLane("prolog", StreamCoreBudget(3, 5)), StreamLane("rest")))
    report = _graph(sku_case(SPLIT_M_SKU), (op,), recipes=RecipeBook(governed.recipes()),
                    stream_plans=(streams,), trace_limit=3000)
    # Core 9 has only one peer in the lane. The remaining two pairs stay reachable.
    local = [e for e in report["events"] if ":mm3_wqb_fixpipe_ub:" in e["task"] and e["demand"]]
    assert {(e["placement"]["fixpipe@r0:s0"][0], e["metadata"]["aiv_destination"]) for e in local} == {
        (7, 14), (7, 15), (8, 16), (8, 17),
    }
    per_unit = report["execution"]["peak_allocated_per_unit"]
    assert {int(name.split(":unit")[1]) for name, size in per_unit.items() if name.startswith("ub@") and size} == set(range(14, 18))
    assert _drained(report)["fixpipe@r0:s0"] == pytest.approx(sum(e["demand"].get("fixpipe@r0:s0", 0) for e in report["events"]))
def _enable_fusion(raw, profile, prolog):
    raw["runtime"].setdefault("kernels", {})["lowerings"] = {
        "mla_prolog": prolog, "native_fusion": {"implementation": "declared_native_fusion",
                                               "options": {"roles": ["mla_prolog"], "traffic_profile": profile}}}
def _source_case(*, tiny=False):
    raw = json.loads((CASES / "glm53-a3-w4a8.json").read_text())
    _enable_fusion(raw, FULL_QUANT_PROFILE, "vllm_ascend_mla_prolog_v3")
    case = parse_case(raw)
    model, runtime = case.model, case.runtime
    plan = model.layer_plans[0]
    if tiny:
        plan = replace(plan, num_heads=8, q_lora_rank=1536, head_dim=192,
                       qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512)
        model = replace(model, hidden_size=1024, num_heads=8, head_dim=192, num_kv_heads=1)
        runtime = replace(runtime, parallel=ParallelSpec())
    return replace(case, model=replace(model, layers=1, layer_plans=(plan,)),
                   runtime=replace(runtime, pp_layer_partition=(1,)))
def _graph(case, ops, ready=7, **options):
    ids = {op.op_id for op in ops}
    return graph_costs(case, {0: tuple(ops)}, isolated=False, input_ready_ms=ready,
                       external_inputs={dep: ready for op in ops for dep in op.deps if dep not in ids},
                       **options)
def _drained(report):
    run = report["execution"]
    assert run["live_tasks"] == run["live_programs"] == 0 and not any(abs(v) > 1e-7 for v in run["live_allocated"].values())
    return run["serviced"]
def test_full_quant_source_preserves_live_outputs_and_one_launch():
    case = _source_case()
    ops = target_operators(case, phase="prefill", tokens=129)
    fused = next(op for op in ops if op.family == "native_fusion")
    source = dict(fused.metadata)["fusion_boundary"]["source_program"]["program"]
    assert any(name.endswith(".q_b.quantize") for name in source["fusion_external_outputs"])
    assert fused.op_id in next(op for op in ops if op.op_id.endswith(".index_q")).deps
    report = _graph(case, (fused,), include_submission=True)
    assert _drained(report)["host@r0:s0"] == pytest.approx(case.machine.dies[0].launch_us / 1000)
def test_source_plan_keeps_a_bounded_repeat_and_preserves_absorption_work():
    tokens = 128 * 10_000_000 + 1
    op = _prolog(tokens, num_heads=8, weight_quant_mode=WEIGHT_QUANT_INT8, kv_cache_quant_mode=0)
    plan = governed.build(op, _context(SPLIT_N_SKU, op))
    full, tail = plan.program.groups[0]
    assert (full.count, tail.count) == (10_000_000, 1)
    assert len(pickle.dumps(plan)) < 100_000
    assert plan.work.matrix["bf16"] == 2 * (tokens + 15) * 8 * 128 * 512
    plain = target_operators(json.loads((CASES / "glm53-a3-w4a8.json").read_text()), phase="prefill", tokens=8, resource_fusion=False)
    assert any(op.recipe == "q_absorb" for op in plain) and all(op.family != "native_fusion" for op in plain)
def test_repeated_source_groups_share_physical_identity_and_run_on_small_geometry():
    case = _source_case(tiny=True)
    assert _drained(_graph(case, target_operators(case, phase="prefill", tokens=32)))["vector@r0:s0"] > 0
    plan = case.model.layer_plans[0]
    case = replace(case, model=replace(case.model, layers=2, layer_plans=(plan, replace(plan, index=1))),
                   runtime=replace(case.runtime, pp_layer_partition=(2,)))
    first, second = [g for g in export_structure(case, phase="prefill", tokens=32, scope="body")["subgraphs"]
                     if g["role"] == "mla_prolog"]
    assert first["subgraph_id"] != second["subgraph_id"] and first["physical_signature"] == second["physical_signature"]
    assert first["physical_work"] == second["physical_work"] and first["physical_work"]["launches"] == 1
def _packed_case(layers):
    raw = json.loads((CONCURRENT / "glm78/a3_glm53_w4a8_tp16_tpot88_c8.json").read_text())
    raw["model"]["layers"] = layers
    _enable_fusion(raw, PACKED_PROFILE, {"implementation": "vllm_ascend_mla_prolog_v3",
                                  "options": {"cache_layout": PACKED_LAYOUT, "cache_group_size": 128}})
    return raw
def _packed_groups(raw, tokens=6, sequences=8):
    ops = target_graphs(raw, scope="body", phase="decode", tokens=tokens, sequences=sequences,
                        context=4104, apply_body_defaults=False)[0]
    return [op for op in ops if dict(op.metadata).get("fusion_role") == "mla_prolog"]
def test_packed_prolog_keeps_mixed_cache_planes_and_optional_query_scale():
    views = Counter()
    for op in _packed_groups(_packed_case(78)):
        meta = dict(op.metadata)
        view = meta["fusion_tensor_view"]
        enabled = view["query_norm_output"]
        views[enabled] += 1
        assert view["query_norm_scale_shape"] == ((48, 1) if enabled else ()) and view["query_norm_scale_dtype"] == "fp32"
        assert view["projection_weight_dtypes"] == (view["token_dtype"],) * 3 == ("int8",) * 3
        assert view["absorbed_weight_dtype"] == "bf16"
        assert view["packed_cache"]["component_offsets"] == {"latent": 0, "rope": 512, "scales": 640}
        epilog = next(p for p in meta["fusion_phases"] if p["recipe"] == "kv_store")
        assert (epilog["mte2_bytes"], epilog["mte3_bytes"]) == (48 * (576 * 2 + 8) + 4, 48 * (512 + 64 * 2 + 4 * 4))
        boundary = meta["fusion_boundary"]
        assert (boundary["submitted_kernels"], boundary["traffic"]["eliminated_writes"]) == (1, 0)
        cache, outputs = subgraph_contract({"role": "mla_prolog", "tensor_view": view})[4:6]
        assert cache[0] == ((1, 656), "int8", "ND") and outputs[2] == (((48, 2048), "int8", "ND") if enabled else ((), None, None))
    assert views == {False: 57, True: 21}
    wrong = _packed_case(1)
    wrong["runtime"]["kernels"]["lowerings"]["native_fusion"]["options"]["traffic_profile"] = FULL_QUANT_PROFILE
    with pytest.raises(ModelingError, match="bf16_uk_norm_cache"):
        _packed_groups(wrong, 1, 1)
def test_packed_store_matches_governed_row_and_persistent_capacity():
    case = parse_case(_packed_case(1))
    plan = case.model.layer_plans[0]
    op = next(op for op in target_operators(case, phase="decode", tokens=6, context=4104, resource_fusion=False)
              if op.recipe == "kv_store")
    attrs = dict(op.metadata)["packed_cache"]
    governed_op = _prolog(6, abi=ABI_V3_NZ, weight_quant_mode=WEIGHT_QUANT_INT8)
    c = governed.validate_contract(governed_op)
    assert attrs["row_bytes"] == c.kv_row_bytes == cache_format(case.model, "mla_kv").row_bytes(576) == 656
    assert attrs["component_offsets"] == {"latent": 0, "rope": 512, "scales": 640}
    assert (attrs["latent_dtype"], attrs["rope_dtype"], attrs["scale_dtype"]) == ("int8", "bf16", "fp32")
    assert op.mte3_bytes == 6 * c.kv_row_bytes and op.hbm_bytes == op.mte2_bytes + op.mte3_bytes
    assert dict(op.metadata)["cache_output_alias"]
    assert layer_cache_components(case.model, plan, case.runtime.parallel, 1024)["mla_kv"] == 1024 * 656
def test_native_boundary_service_preserves_work_and_responds_to_memory_rate():
    raw = _packed_case(1)
    selected = _packed_groups(raw, 1, 1)
    case = parse_case(raw)
    rows = []
    for rate in (5e8, 1e9, 2e9):
        dies = tuple(replace(d, rates=tuple((dict(d.rates) | {"hbm": rate}).items())) for d in case.machine.dies)
        rows.append(operator_costs(replace(case, machine=replace(case.machine, dies=dies)), selected)["operators"][0])
    assert rows[0]["duration_ms"] > rows[1]["duration_ms"] > rows[2]["duration_ms"]
    assert rows[0]["serviced"] == pytest.approx(rows[1]["serviced"]) == rows[2]["serviced"]
    before = operator_costs(raw, selected, input_ready_ms=7)["operators"][0]
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "explicit-call", "match": {"metadata.fusion_role": "mla_prolog"},
        "response": {"fixed_us": 1000, "terms": []}, "provenance": {"basis": "synthetic control"}}]
    after = operator_costs(raw, selected, input_ready_ms=7)["operators"][0]
    assert after["duration_ms"] == pytest.approx(1) and after["serviced"] == pytest.approx(before["serviced"])
    assert not after["service_before_input"] and after["live_tasks"] == 0
def _caller_case(**options):
    raw = json.loads((CONCURRENT / "matrix-extension/a3_glm53flash_tp16_aclgraph_c32.json").read_text())
    raw["model"]["layers"] = 4
    raw["runtime"]["kernels"]["programs"]["indexed_attention"] = {
        "kind": "indexed_attention_packed_int8", "scratch_policy": "offchip"}
    if options:
        raw["runtime"]["kernels"].setdefault("lowerings", {})["pool_key_indexer"] = {
            "implementation": "vllm_ascend_cann_keypool_caller",
            "options": {"tail_capacity": 9, "block_table_columns": 8195} | options}
    return raw
def _caller_layer(raw, batch):
    return {op.recipe: op for op in target_graphs(raw, scope="body", phase="decode", tokens=6,
                                                 context=131201, sequences=batch)[0] if op.layer == 3}
@pytest.mark.parametrize("batch", (3, 4, 8))
def test_keypool_caller_preserves_padding_projection_and_selection_geometry(batch):
    ops = _caller_layer(_caller_case(tail_capacity=9), batch)
    pool, score, weights = (ops[name] for name in ("cann_key_pool", "cann_pool_key_indexer", "index_weights"))
    rows, padded = batch * 6, max(24, batch * 6)
    assert pool.local_shape == (padded, 128, 4096) and pool.flops == 4 * padded * 128 * 4096
    pool_meta, meta = dict(pool.metadata), dict(score.metadata)
    assert pool_meta["projection_formats"] == dict.fromkeys(("hidden", "wk", "gate"), "bf16")
    assert (pool_meta["output_shape"], pool_meta["valid_output_rows"]) == ((padded // 4 + batch, 128), batch)
    assert (meta["valid_query_rows"], meta["padded_query_rows"], meta["native_query_rows"]) == (rows, padded, padded + 1)
    assert meta["dummy_request_queries"] == padded - rows + 1
    assert meta["scanned_pools"] == batch * sum((131201 + i) // 4 for i in range(1, 7))
    assert meta["output_shape"] == (padded + 1, 2051)
    assert dict(ops["cann_keypool_nonempty_inputs"].metadata)["block_table_shape"] == (batch + 1, 8195)
    gather = ops["cann_keypool_state_gather"]
    assert weights.dtype == "bf16" and weights.op_id in gather.deps
    assert dict(gather.metadata)["state_shape"] == (1 + 2 * batch, 4, 256)
    attention = next(op for op in ops.values() if op.family == "attention")
    topk = dict(ops["index_topk"].metadata)
    assert dict(attention.metadata)["indexed_gather_rows"] == topk["selected_token_rows"] == batch * (6 * 2048 + 11)
    assert not topk["pool_id_expansion_in_adapter"]
def test_keypool_rollback_keeps_tail_residency_and_cache_precedes_selection():
    raw = _caller_case(tail_capacity=9)
    ops = _caller_layer(raw, 4)
    write = dict(ops["cann_keypool_cache_write"].metadata)
    assert (write["tail_launch_rows"], write["valid_tail_rows"], write["masked_pooled_rows"]) == (24, 24, 6)
    short = _caller_layer(_caller_case(tail_capacity=4), 4)["cann_keypool_cache_write"]
    assert dict(short.metadata)["valid_tail_rows"] == 12
    key, gate, norm = (ops[name] for name in ("cann_keypool_rollback_key", "cann_keypool_rollback_gate", "index_norm"))
    assert key.local_shape == gate.local_shape == (24, 128, 4096)
    assert norm.deps == (key.op_id,) and norm.mte2_bytes == 24 * 128 * 2 + 2 * 128 * 4
    assert not {"cann_keypool_rollback_key_fp32", "index_norm_input_fp32", "index_norm_output_cast"} & ops.keys()
    case = parse_case(raw)
    plan = case.model.layer_plans[3]
    assert layer_cache_components(case.model, plan, case.runtime.parallel, 131201)["index_pool_tail"] == 9 * 2 * 128 * 4
    terms = {t.name: t for t in layer_weight_terms(plan, case.model)}
    assert term_bytes(terms["indexer_weights"], case.model.quant, case.runtime.parallel) == 32 * 4096 * 2
    report = _graph(raw, [op for name, op in ops.items() if name.startswith(("cann_key", "cann_pool", "index_"))])
    rows = {row["recipe"]: row for row in report["timeline"]}
    for before, after in (("index_weights", "cann_key_pool"), ("cann_keypool_cache_write", "cann_pool_key_indexer"),
                          ("cann_pool_key_indexer", "index_topk")):
        assert rows[after]["ready_ms"] >= rows[before]["complete_ms"]
    _drained(report)
def test_keypool_joint_service_charges_each_native_invocation():
    raw = _caller_case(tail_capacity=9)
    ops = _caller_layer(raw, 4)
    names = ("cann_key_pool", "cann_pool_key_indexer")
    raw["machine"]["device"] = {"rates": {"observed_pool": 1e9}, "work_units": {"observed_pool": "busy_ns"},
                                "resource_units": {"observed_pool": 1}}
    raw["runtime"]["kernels"]["service_regimes"] = [
        {"id": name, "match": {"recipe": name}, "response": {"fixed_us": us},
         "execution": {"kind": "joint_native", "resource": "observed_pool", "reference_rate": 1e9,
                       "native_kernel": dict(ops[name].metadata)["kernel"]}, "provenance": {"kind": "synthetic"}}
        for name, us in zip(names, (100, 200))]
    report = _graph(raw, [ops[name] for name in names])
    assert report["duration_ms"] == pytest.approx(.3) and _drained(report)["observed_pool@r0:s0"] == pytest.approx(300000)
    assert all(row["program"] == "JointNativeService" for row in report["timeline"])
def test_nope_cache_keeps_declared_width_and_quantized_store_traffic():
    assert MlaSpec.from_mapping(dict(q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=512,
                                     qk_rope_head_dim=64, v_head_dim=128, mla_use_nope=True)).to_dict()["mla_use_nope"]
    raw = _caller_case()
    raw["model"].setdefault("metadata", {})["cache_representations"] = {
        "mla_kv": {"dtype": "fp8_e4m3", "group_size": 128, "scale_dtype": "fp16"}}
    ops = target_graphs(raw, scope="body", phase="decode", tokens=6, context=0,
                        sequences=1, apply_body_defaults=False)[0]
    assert not {op.recipe for op in ops} & {"rotary", "q_rotary", "kv_rotary"}
    op = next(op for op in ops if op.recipe == "kv_store")
    meta = dict(op.metadata)
    assert meta["native_kind"] == "aclnnDynamicBlockQuant" and meta["cache_epilog"]["scatter"] == "aclnnScatterPaKvCache"
    assert op.mte3_bytes == 6 * meta["cache_row_bytes"] == 6 * block_quantized_row_bytes(512) == 6 * 520
def test_proposal_steps_keep_separate_pool_writes_and_share_selection():
    raw = _caller_case(tail_capacity=9)
    raw["runtime"].setdefault("metadata", {})["draft_proposal_tokens"] = 5
    ops = target_graphs(raw, scope="model", phase="decode", tokens=6, context=0, sequences=2)[0]
    counts = Counter(op.recipe for op in ops)
    assert [counts[name] for name in ("cann_key_pool", "cann_keypool_cache_write", "cann_pool_key_indexer", "index_topk")] == [6, 6, 2, 2]
    shared = next(op for op in ops if op.recipe == "index_topk" and op.op_id.startswith("draft0."))
    for step in range(5):
        attention = [op for op in ops if op.op_id.startswith(f"draft{step}.") and op.family == "attention"]
        assert attention and all(shared.op_id in op.deps for op in attention)
@pytest.mark.parametrize("scale,native", [("fp16", False), ("fp32", True)])
def test_keypool_fp8_cache_selection_requires_matching_scale_layout(scale, native):
    raw = _caller_case(tail_capacity=9)
    raw["model"]["quant"] = dict(name="bf16_fp8_index", weight_format="bf16", activation_format="bf16",
                                kv_dtype="int8", index_dtype="fp8_e4m3", index_scale_dtype=scale)
    raw["model"].setdefault("metadata", {})["cache_representations"] = {
        "index_k": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "fp16"}}
    ops = _caller_layer(raw, 4)
    if native:
        write = dict(ops["cann_keypool_cache_write"].metadata)
        score = ops["cann_pool_key_indexer"]
        meta = dict(score.metadata)
        assert (write["pooled_row_bytes"], write["pooled_scale_dtype"], meta["pool_row_bytes"]) == (132, "fp32", 132)
        assert score.flops == 2 * meta["safe_reference_scanned_pools"] * 32 * 128
    else:
        assert "index_pool_compress" in ops and "cann_keypool_cache_write" not in ops
def _layer(**changes):
    shape = dict(index=0, attention="mla", architecture="mla", ffn="dense", intermediate_size=1024,
                 num_heads=8, num_kv_heads=1, head_dim=576, q_lora_rank=1536, kv_lora_rank=512,
                 qk_nope_head_dim=512, qk_rope_head_dim=64, v_head_dim=128, mla_variant="latent_mla",
                 cache_kind="window_compressed", compress_ratio=4, compressor_width=1,
                 compressor_state_rows=8, compressor_kind="gated_ring", owns_long_kv=True,
                 kv_owner_layer_id=0, window=4096)
    return LayerPlan(**(shape | changes))
@pytest.mark.parametrize("ratio", (1, 4))
def test_compressor_prunes_gate_at_unit_ratio_and_fuses_valid_rows(ratio):
    plan = _layer(compress_ratio=ratio)
    model = ModelSpec("contract", 1, 4096, 8, 576, 1024, 1024, attention="mla", ffn="dense", layer_plans=(plan,), quant=QuantSpec("bf16"))
    builder = Builder("s0.l0", plan, 0, (), model, "decode")
    ready = compressor(builder, name="compress", source="hidden", tokens=8, padded_tokens=8,
                       valid_rows=2, width=576, plan=plan, model=model,
                       fused=ratio > 1, return_latent=ratio > 1)
    if ratio == 1:
        assert not {"compressor_gate", "compressor_ape"} & {term.name for term in layer_weight_terms(plan, model)}
        assert [op.recipe for op in builder.ops] == ["compress_values", "compress_norm", "compress_rotary"]
        assert ready == builder.ops[-1].op_id and builder.ops[0].flops == 2 * 8 * 576 * 4096
    else:
        op, = builder.ops
        assert ready == (op.op_id, op.op_id) and op.returns == ("latent",)
        assert op.flops == 4 * 8 * 576 * 4096 and op.vector_ops == 6 * 8 * 576 + 5 * 2 * 576 + 6 * 2 * 64
        assert op.mte2_bytes == 8 * 4096 * 2 + 4 * 576 * 4096 + (8 + ratio) * 576 * 4 + 576 * 2 + 2 * 64 * 4 * 2
        assert op.mte3_bytes == 2 * 2 * 576 * 2 + 8 * 576 * 4
def test_native_indexer_epilogs_keep_parameters_and_remove_intermediate_bytes():
    model = parse_case(_caller_case()).model
    plan = model.layer_plans[3]
    for native in (False, True):
        builder = Builder("s0.l0", plan, 0, (), model, "decode")
        index_layer_norm(builder, source="hidden", rows=8, width=128, native=native)
        if native:
            op, = builder.ops
            assert op.dtype == "bf16"
            assert (op.mte2_bytes, op.mte3_bytes, op.vector_ops) == (8 * 128 * 2 + 2 * 128 * 4, 8 * 128 * 2, 8 * 8 * 128)
        else:
            assert {op.family for op in builder.ops} == {"cast", "layernorm"}
        builder = Builder("s0.l0", plan, 0, (), model, "decode")
        index_head_weights(builder, plan=plan, model=model, quant=model.quant,
                           deps="hidden", pool_weights=None, tokens=8, native=native)
        assert len(builder.ops) == (1 if native else 2)
    builder = Builder("s0.l0", plan, 0, (), model, "decode")
    quant = builder.vector("index_key_quantize", ("index_norm",), 8 * 128, 4, family="quantize",
                           shape=(8, 128), read_bytes_override=8 * 128 * 2, write_bytes_override=8 * 132,
                           metadata=(("output_format", "int8"), ("scale_format", "fp16"),
                                     ("scale_shape", (8, 1)), ("scale_count", 1)))
    write = builder.store("index_store", (quant,), 8, 132, "int8")
    index_row_quantization(builder, write, quant, width=128, native=True)
    op, = builder.ops
    assert op.deps == ("index_norm",) and op.hbm_bytes == op.mte2_bytes + op.mte3_bytes
    assert (op.mte2_bytes, op.mte3_bytes, op.vector_ops) == (8 * 128 * 2, 8 * 132, 8 * 128 * 4)
def test_draft_context_prolog_shards_persisted_rows_and_orders_query_readiness():
    draft = SimpleNamespace(layers=2, hidden_size=4096, q_lora_rank=512, kv_lora_rank=512,
                            qk_rope_head_dim=64, activation_dtype="bf16")
    shape = (8, 4096, 4096)
    projection = OpSpec("projection", "gemm", "decode", 0, 0, (), shape, shape, shape, shape, dtype="bf16")
    ops = []
    ready = draft_context.prepare(ops, projection=projection, model=parse_case(_caller_case()).model, draft=draft, rows=8,
                                  query_rows=3, prefix="d:", stage=0, requests=(), parallel=ParallelSpec(cp=2))
    assert [op.recipe for op in ops] == ["draft_context_norm"] + [
        draft_context.PROLOG_RECIPE, draft_context.SCATTER_RECIPE] * 2 + ["draft_query_embedding", "draft_context_query_ready"]
    for prolog, scatter in ((ops[1], ops[2]), (ops[3], ops[4])):
        assert dict(prolog.metadata)["native_kind"] == ABI_V3_NZ and dict(prolog.metadata)["do_rope"]
        assert scatter.local_shape[0] == 4 and dict(scatter.metadata)["cache_epilog"]["kv_row_bytes"] == (512 + 64) * 2
    assert ready == ops[-1].op_id
def test_packed_query_quantization_requires_matching_projection_formats():
    op = _packed_groups(_packed_case(1), 1, 1)[0]
    view = dict(op.metadata)["fusion_tensor_view"]
    declaration = {"role": "mla_prolog", "tensor_view": view}
    assert apply_traffic_profile((), declaration, PACKED_PROFILE)[1]["profile"] == PACKED_PROFILE
    with pytest.raises(ModelingError, match="declared_token_projections"):
        apply_traffic_profile((), declaration | {"tensor_view": view | {"query_norm_dtype": "bf16"}}, PACKED_PROFILE)
@pytest.mark.parametrize("abi", (MQS_ABI, DSL_MQS_ABI))
def test_sparse_attention_conserves_branch_rows_and_guards_sparse_coverage(abi):
    # One draft row and a larger selective decode cohort use the same branch-local cache ABI.
    contract = paged_mla.PagedMlaContract(((1, 128),), 8, 64, abi=abi, n_ori=128, cmp_ratio=4)
    draft = paged_mla.make_op(contract, phase="draft")
    assert draft.local_shape == (1, 8, 512)
    assert paged_mla.build(draft, _context(SPLIT_M_SKU, draft)).work.matrix
    cohort = replace(contract, sequences=((1, 640),) * 72, n_cmp=512)
    op = paged_mla.make_op(cohort)
    planned = paged_mla.build(op, _context(SPLIT_M_SKU, op))
    assert planned.metadata["useful_matrix_flops"] == 2 * 72 * 640 * 8 * (512 + 512)
    assert planned.work.matrix
    with pytest.raises(ModelingError, match="sparse_pair_coverage"):
        paged_mla.make_op(replace(cohort, n_cmp=511))
def test_draft_layers_inherit_fusion_router_and_pool_ownership():
    case = parse_case(_caller_case())
    body = replace(case.model.layer_plans[3], mla_use_nope=True, packed_input_projection=True,
                   router_output_dtype="fp32", router_dtype="bf16", router_precast_weight_copy=False)
    model = replace(case.model, layer_plans=tuple(body for _ in case.model.layer_plans))
    plans = native_draft_layer_plans(model, model.auxiliary.mtp)
    assert len(plans) == model.auxiliary.mtp.layers
    for layer in plans:
        assert layer.packed_input_projection and layer.mla_use_nope and layer.owns_topk
        assert (layer.router_output_dtype, layer.router_dtype, layer.router_precast_weight_copy) == ("fp32", "bf16", False)
        assert layer.owns_index_k and layer.index_k_owner_layer_id == layer.index and layer.indexer_type == "shared"
def test_projection_quantization_preserves_absorption_index_and_draft_formats():
    from mindie_sim.catalog.draft import draft_quant
    from mindie_sim.catalog.draft_weights import _draft_attention_terms
    from mindie_sim.catalog.ops import compute_dtype_for
    from mindie_sim.compose import compose_case
    case = parse_case(compose_case("glm-5.3-flash", hardware=SPLIT_M_SKU, quant="w8a8_mxfp8",
                                  tp=1, layers=4, prompt_tokens=4096, output_tokens=16,
                                  concurrency=8, graph_capture_mode="none"))
    quant = case.model.quant
    assert {quant.format_for(k) for k in ("mla", "routed_expert", "shared_expert", "mtp")} == {"fp8_e4m3"}
    assert {quant.format_for(k) for k in ("kda", "mla_absorb", "indexer", "indexer_fp8", "dspark_input")} == {"bf16"}
    draft = case.model.auxiliary.mtp
    dq = draft_quant(case.model, draft)
    assert (dq.expert_weight_format, dq.format_for("mla_absorb")) == ("fp8_e4m3", "bf16")
    assert next(term for term in _draft_attention_terms(draft, dq) if term.name == "draft_kv_b").category == "mla_absorb"
    op = {op.recipe: dict(op.metadata) for op in target_graphs(case, scope="body", phase="decode", tokens=1, context=64)[0]
          if op.layer == 3}
    assert all(op[r]["weight_format"] == op[r]["activation_format"] == "fp8_e4m3"
               for r in ("qkv_a", "q_b", "output", "grouped_gate_up", "grouped_down"))
    index = projection_quant(quant_preset("w8a8_int8"), _layer(indexer=True, index_heads=32, index_dim=128, index_kpool=4))
    assert compute_dtype_for(index, "indexer.weights") == ("bf16", "bf16")
    assert compute_dtype_for(index, "indexer.pool_projection") == ("fp32", "fp32")
