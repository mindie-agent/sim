"""Block-scaled MX matrices and source-selected quantization programs.

[Role] Verify source geometry, panel scheduling, reuse, and fusion boundaries.
[Invariants] Work follows valid rows, real banks, and declared resource owners.
[Timeouts & Locks] Bounded local graph and Engine execution without locks.
[Side-Effect Policy] Read committed inputs and mutate only local data and caches.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import gc
from itertools import product
import json
from math import ceil, lcm
from pathlib import Path
import weakref

import pytest
from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.operators.mx_quant_matrix import family_mxfp8_swiglu, family_situ_mx, mxfp4_grouped, mxfp4_supply, mxfp8_grouped, mxfp8_swiglu
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.implementations import select
from mindie_sim.prediction.instructions import InstructionInventory
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe, Work
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_geometry import Config, activation_l2_fraction, columns, effective_l2_bytes, geometry_ledger, owner_tiles, panel_sizes, tile_inventory, tile_work, weight_l2_fraction
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_panels import compiled_panels, panel_parts
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_pipeline import Dependencies, Panel, Slots, SourcePipeline, pipeline_cycle
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_source_refill import source_panel
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.operators.matrix.refill_steady import compact_refill
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import InvocationTiming
CASES = Path(__file__).resolve().parents[2] / "examples/cases"
PIPES = ("hbm", "mte2_aic", "mte2_aiv", "mte3", "mte1", "vector", "fixpipe")
MATRIX_PIPES = ("hbm", "mte2_aic", "mte1", "fixpipe", "cube")
DIRECTIONAL = {"mte1_l0a": ("mte1", 1e10), "mte1_l0b": ("mte1", 2e10)}
MX4_ERROR = "mxa8w4_nznk_requires_fp8_mxfp4_group32_e8m0_to_bf16"
SITU = family_situ_mx.PROFILE
PREPARED = ("q_b", "grouped_output_a", "grouped_output_b")
pytestmark = pytest.mark.tier0
def read_case(name):
    return json.loads((CASES / f"{name}.json").read_text())
def body(case, *, tokens=1, context=8, sequences=1, fusion=False):
    return target_graphs(case, scope="body", phase="decode", tokens=tokens, context=context,
                         sequences=sequences, resource_fusion=fusion)[0]
def geometry(case):
    return Geometry.from_devices(parse_case(case).machine.dies)
def price(case, op, book=None, ready=7):
    return operator_costs(case, (op,), recipes=book, input_ready_ms=ready)["operators"][0]
def drained(row):
    assert not row["service_before_input"] and not row["live_tasks"]
    assert not any(row["remaining_buffers"].values())
def service(row, role):
    return row["serviced"].get(("memory" if role == "hbm" else role) + "@r0:s0", 0)
def book(recipe):
    return RecipeBook(families={"grouped_gemm": recipe})
def mx4(case, op, **options):
    return price(case, op, book(mxfp4_grouped.recipe(**options)))

@pytest.fixture(scope="module")
def mx4_source():
    raw = read_case("v41-a5-dp4-body-current")
    raw["runtime"]["routing"] = {"policy": "balanced"}
    raw["runtime"].setdefault("kernels", {}).setdefault("families", {})["grouped_gemm"] = {"kind": "ascend_mxa8w4_nznk"}
    ops = body(raw)
    return raw, [op for op in ops if op.family == "grouped_gemm"][:2], ops
def grouped_op(op, rows, n, k, *, wide=False):
    meta = dict(op.metadata, matrix_group_rows=rows, matrix_weight_bank_groups=len(rows),
                matrix_groups=len(rows) if wide else sum(r > 0 for r in rows),
                activation_scale_shape=(sum(rows), ceil(k / 64), 2))
    shape = (sum(rows), n, k)
    if wide:
        return replace(op, local_shape=(len(rows), *shape), metadata=tuple(meta.items()))
    return replace(op, local_shape=shape, semantic_shape=shape, launch_shape=shape, metadata=tuple(meta.items()))
def mx4_meta(row):
    return row["operator"]["metadata"]["program_execution"]["mxa8w4"]

def test_mx_grouped_frontend_preserves_rows_and_caller_boundaries(mx4_source):
    raw, group, ops = mx4_source
    assert {"swiglu", "quantize"} <= {op.family for op in ops}
    for op, row in zip(group, operator_costs(raw, group)["operators"], strict=True):
        md = mx4_meta(row)
        assert tuple(md["group_rows"]) == dict(op.metadata)["matrix_group_rows"]
        assert md["valid_rows"] == md["active_banks"] == sum(md["group_rows"]) == 6
        assert md["logical_flops"] == op.flops and row["operator"]["matrix_flops"] == {"fp8_e4m3":16*op.flops}
        drained(row)
@pytest.mark.parametrize("field,small", [("weight_format", True), ("weight_format", False), ("output_format", False)])
def test_mixed_grouped_rejects_incompatible_precision(mx4_source, field, small):
    raw, (gate, _), _ = mx4_source
    op = grouped_op(gate, (1,), 128, 64) if small else gate
    with pytest.raises(ModelingError, match=MX4_ERROR):
        mx4(raw, replace(op, metadata=tuple(dict(op.metadata, **{field:"fp8_e4m3"}).items())))
MX4_TAILS = (((1, 0, 2), 320, 320), ((0, 257, 0, 3), 544, 448), ((17,), 4608, 5120), ((1, 1), 5120, 2304))
DYNAMIC_TAILS = (((1, 0, 1), 32, 64), ((17, 0, 17), 128, 512), ((3, 0, 3), 160, 4160), ((1, 0, 1), 256, 5120))

@pytest.mark.parametrize("rows,n,k,producer", [(*shape, "serial") for shape in (*MX4_TAILS, ((3,), 160, 7680))]
                         + [(*shape, "dynamic_buffer4") for shape in DYNAMIC_TAILS]
                         + [((1,)*6+(0,)*42, n, k, "shared_hbm") for n, k in ((4608, 5120), (5120, 2304))])
def test_panels_conserve_tails_periodic_refills_and_issued_work(mx4_source, rows, n, k, producer):
    raw, (gate, _), _ = mx4_source
    op = grouped_op(gate, rows, n, k)
    if producer == "shared_hbm":
        raw = deepcopy(raw)
        raw["machine"]["device"]["rates"] = {"hbm": dict(parse_case(raw).machine.dies[0].rates)["hbm"]/2}
    options = {"producer_mode": producer} if producer != "shared_hbm" else {}
    a, b = (mx4(raw, op, compact_panels=flag, **options) for flag in (False, True))
    assert b["serviced"] == pytest.approx(a["serviced"], rel=1e-12, abs=1e-12)
    assert service(b, "fixpipe") == pytest.approx(2*sum(rows)*n)
    if producer == "shared_hbm":
        assert b["duration_ms"] == pytest.approx(a["duration_ms"], rel=.10)
        return
    for row in (a, b):
        drained(row)
    if producer == "dynamic_buffer4":
        assert all(mx4_meta(row)["producer_mode"] == "dynamic_buffer4" for row in (a, b))
        assert all(mx4_meta(row)["valid_rows"] == sum(rows) for row in (a, b))
        return
    cfg, l2 = Config(k, n, rows), int(effective_l2_bytes(geometry(raw)))
    l2_shares = dict(activation_l2=activation_l2_fraction(cfg, 32, l2), weight_l2=weight_l2_fraction(cfg, 32, l2))
    work = sum((tile_work(cfg, m, w, cores=32, **l2_shares) for o in range(32) for *_, m, w in owner_tiles(cfg, o, 32)),
               Work(resources={"hbm":64*256, "mte2_aiv":64*256}))
    assert b["operator"]["flops"] == work.matrix["fp8_e4m3"]
    for role, amount in work.resources.items():
        key, expected = ("vector", amount*32) if role == "reg_issue_slots" else (role, amount)
        assert service(b, key) == pytest.approx(expected)

@pytest.mark.parametrize("role", PIPES)
def test_grouped_producers_respond_to_each_physical_bottleneck(mx4_source, role):
    raw, (gate, _), _ = mx4_source
    raw = deepcopy(raw)
    modes = (("serial", False, 4, 256, 512), ("serial", True, 2, 160, 5120), ("dynamic_buffer4", False, 4, 256, 5120))
    for mode, compact, m, n, k in modes:
        if mode == "dynamic_buffer4" and role == "fixpipe":
            continue
        op = grouped_op(gate, (1,)*m, n, k)
        times = []
        for factor in (.5, 1, 2):
            raw["machine"]["device"]["rates"] = dict(dict.fromkeys(PIPES, 1e18), **{role:1e8*factor})
            times.append(mx4(raw, op, compact_panels=compact, producer_mode=mode)["duration_ms"])
        assert times[0] > times[1] > times[2]
def test_cube_issue_and_register_work_conserve_exact_payloads(mx4_source):
    raw, (gate, _), _ = mx4_source
    a, b = (mx4(raw, grouped_op(gate, (m, 0), 128, 64)) for m in (1, 16))
    first, full = map(mx4_meta, (a, b))
    assert first["issued_cube_flops"] == full["issued_cube_flops"] == full["logical_flops"] == 16*first["logical_flops"]
    assert service(a, "mte1") == service(b, "mte1") == (16+128)*66 and service(a, "mte3") == 16384+8192
    for opcode, field, value in (("ShiftRightB8", "calls", 64), ("ShiftRightB8", "active_lanes", 16384),
                               ("StoreB8", "calls", 32), ("StoreB8", "write_bytes", 8192), ("GatherUB_B16", "calls", 8)):
        assert first["register_inventory"]["instructions"][opcode][field] == value
    assert a["operator"]["vector_ops"] == 0 and service(a, "vector") == pytest.approx(302*32)
    raw = deepcopy(raw)
    raw["machine"]["device"].update(rates={"mte1": 1e9}, service_modes={
        role: {"resource": "mte1", "rate": rate} for role, rate in zip(DIRECTIONAL, (1e9, .5e9))})
    directional = mx4(raw, grouped_op(gate, (1, 0), 128, 64), mte1_mode="directional")
    assert service(directional, "mte1") == (16 + 2*128)*66 and service(directional, "hbm") == pytest.approx(service(a, "hbm"))
def panel_args(context, k=5120, m=1, n=160, repeat=1, owner=0, share=1, directional=False):
    return dict(k=k, m=m, n=n, repeat=repeat, owner=owner, cores=32, aivs=64, directional=directional, shared_clients={context.resource_names["hbm"]: share})
def uncached(context, options):
    k, m, n = (options[key] for key in ("k", "m", "n")); kb, ka, _ = panel_sizes(m, n)
    panels, period = ceil(k / kb), lcm(ka, 4096) // kb; count = panels // period - (panels % period == 0 and k % kb != 0)
    args = {key: value for key, value in options.items() if key != "shared_clients"}
    parts = [panel_parts(context, index=i, **args) for i in range(panels)]
    return compact_refill(context, tuple(parts[:period]) if count else (), count, tuple(parts[count * period:]), shared_clients=options["shared_clients"])
@pytest.fixture(scope="module")
def panel_context(mx4_source):
    raw, (gate, _), _ = mx4_source; contexts = []
    def capture(op, geometry):
        plan = mxfp4_grouped.build(op, geometry)
        return replace(plan, program=lambda ctx: contexts.append(ctx) or plan.program(ctx))
    price(raw, grouped_op(gate, (1,), 160, 320), book(PlannedRecipe(capture)))
    return contexts[0]
def engine_for(context, trace_limit=200):
    units = {name: context.resource_units.get(role, 1) for role, name in context.resource_names.items()}
    return Engine([Resource(n, rate, units=units.get(n, 1)) for n, rate in context.rates.items()], trace_limit=trace_limit)
def execute(program, context, copies=1):
    engine = engine_for(context, trace_limit=2000); ready = engine.submit("input", release=7)
    done = tuple(program.start(engine, f"copy{i}", ready, owner=f"caller{i}") for i in range(copies))
    engine.advance(until=done)
    return tuple(e.time for e in done), dict(engine.serviced), engine.completed_count, list(engine.trace)
def compare_ir(context, args, copies=1):
    cached, original = compiled_panels(context, **args), uncached(context, args)
    out = execute(cached, context, copies); assert cached == original and out == execute(original, context, copies)
    by_task = {e["task"]: e for e in out[3]}; assert any(t.endswith(":packed_read0") for t in by_task)
    for t, e in by_task.items():
        if t.endswith((":widen0", ":widen1")) and (rp := by_task.get(t.replace(":widen", ":packed_read"))): assert rp["end_ms"] <= e["start_ms"]
    return cached

@pytest.mark.parametrize("k,m,repeat", [(64, 1, 1), (320, 3, 1), (4096, 1, 1), (5120, 17, 1), (24576, 1, 1), (320, 1, 2)])
def test_panel_ir_matches_source_and_has_independent_events(panel_context, k, m, repeat):
    cached = compare_ir(panel_context, panel_args(panel_context, k=k, m=m, repeat=repeat), copies=2)
    for stage in cached.stages:
        with pytest.raises(TypeError): stage.work["injected"] = 1
    with pytest.raises(TypeError): cached.stages[0].program.stages[0].placement["injected"] = (0,)

def test_panel_cache_keys_and_lifetimes(panel_context):
    ctx = replace(panel_context, op=replace(panel_context.op), service_modes={**panel_context.service_modes, **DIRECTIONAL})
    args = panel_args(ctx, directional=True); base = compiled_panels(ctx, **args)
    assert compare_ir(ctx, dict(args, owner=1, repeat=2)) is not base
    refs = weakref.ref(ctx), weakref.ref(ctx.op); del ctx; gc.collect()
    assert all(ref() is None for ref in refs)
    compiled_panels.cache_clear()
    args = panel_args(panel_context, k=64); first = compiled_panels(panel_context, **args)
    for repeat in range(2, 1026): compiled_panels(panel_context, **dict(args, repeat=repeat))
    assert (compiled_panels.cache_info().currsize, compiled_panels.template_cache_info().currsize) == (1024, 128)
    assert compiled_panels(panel_context, **args) == first and compiled_panels(panel_context, **args) is not first

def test_panel_templates_rebind_real_units(panel_context):
    compiled_panels.cache_clear()
    for rank, owner, k in product(range(8), range(32), (5120, 256)):
        names = {n:n.replace("@r0:", f"@r{rank}:") for n in panel_context.rates}
        ctx = replace(panel_context, rates={names[n]:v for n, v in panel_context.rates.items()},
                      resource_names={r:names[n] for r, n in panel_context.resource_names.items()})
        compiled_panels(ctx, **panel_args(ctx, k=k, owner=owner, share=32))
    info = compiled_panels.template_cache_info()
    assert (compiled_panels.cache_info().misses, info.misses, info.hits) == (512, 2, 510)
    for owner, units in product((0, 15, 31), (16, 32, 64, 128)):
        pipes = dict.fromkeys(("mte1", "mte3", "mte2_aiv", "mte2_aic"), units)
        ctx = replace(panel_context, resource_units=dict(panel_context.resource_units, **pipes))
        compare_ir(ctx, panel_args(ctx, k=320, m=3, repeat=2, owner=owner, share=3))
def test_dynamic_slots_select_distinct_templates_and_match_eager_schedule(panel_context, mx4_source):
    compiled_panels.cache_clear()
    args = panel_args(panel_context); dynamic = dict(args, producer_mode="dynamic_buffer4")
    plans = [compiled_panels(panel_context, **opts) for opts in (args, dynamic, dict(dynamic, producer_slots=Slots(1, 1, 1, 1, 1)))]
    assert len({id(plan) for plan in plans}) == compiled_panels.template_cache_info().misses == 3
    def panel(i):
        return source_panel(panel_context, k=32768, m=1, n=128, repeat=1, index=i, owner=0, cores=32, aivs=64, directional=False)
    engine = engine_for(panel_context, trace_limit=2000)
    state = Dependencies(engine.submit("input", release=7), Slots())
    for i in range(64):
        done = state.advance(panel(i), lambda label, stage, deps: stage.submit(engine, f"panel{i}:{label}", deps, lambda n: n, ""))
    engine.advance(until=(done,))
    matches = lambda p, pfx: [e for e in list(engine.trace) if e["task"].startswith(f"panel{p}:{pfx}")]
    for p in range(6):
        pub = max(matches(p, f"publish{s}")[0]["end_ms"] for s in (0, 1))
        assert min(e["start_ms"] for e in matches(p, "operand")) >= pub
        for s in (0, 1): assert matches(p, f"packed_read{s}")[0]["end_ms"] <= matches(p, f"widen{s}")[0]["start_ms"]
        if p >= 2: assert matches(p, "operand0")[0]["start_ms"] >= matches(p - 2, "cube0")[0]["end_ms"]
        if p >= 4: assert matches(p, "packed_read0")[0]["start_ms"] >= matches(p - 4, "widen0")[0]["end_ms"]
    times, work, _, _ = execute(SourcePipeline(64, panel, activation_panels=4, scale_panels=8), panel_context)
    assert times == pytest.approx((done.time,), rel=1e-12) and work == pytest.approx(engine.serviced, rel=1e-12, abs=1e-12)
    raw, (gate, _), _ = mx4_source
    empty = mx4(raw, grouped_op(gate, (0, 0), 32, 64), producer_mode="dynamic_buffer4", compact_panels=False)
    assert [service(empty, r) for r in ("hbm", "mte2_aiv", "matrix")] == pytest.approx([64 * 256, 64 * 256, 0]); drained(empty)
    reg_roles = ("reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_shuffle_lanes")
    roles = {r: (1 if r == "cube" else 1e9) for r in ("hbm", "mte1", "mte2_aic", "mte2_aiv", "mte3", "vector", "fixpipe", "cube", *reg_roles)}
    units = {r: (2 if r in ("vector", "mte2_aiv", "mte3", *reg_roles) else 1) for r in roles}
    pools = {"ub": 256 * 1024, "l1": 512 * 1024, "l0a": 64 * 1024, "l0b": 64 * 1024, "l0c": 256 * 1024}
    geom = Geometry(1, units, pools, {k: (2 if k == "ub" else 1) for k in pools}); op4 = grouped_op(gate, (1,), 128, 2048)
    plan4 = mxfp4_grouped.build(op4, geom, compact=False, compact_panels=False)
    eng4 = Engine([Resource(r, v, units=units[r]) for r, v in roles.items()], pools=pools, pool_units=geom.buffer_units, trace_limit=1000)
    ctx4 = RecipeContext(plan4.work.apply(op4), {}, {r: r for r in roles}, roles, 1, resource_units=units, pool_names={k: k for k in pools}, compute_peaks={"fp8_e4m3": 1e6}, service_modes={"reg_issue_slots": ("vector", roles["vector"] * 1000 / 32)})
    done4 = plan4.lower(ctx4).start(eng4, "source", eng4.submit("input", release=7)); eng4.advance(horizon=6); assert not any(eng4.serviced.values()); eng4.advance(until=(done4,))
    ev4 = list(eng4.trace); m4 = lambda p, sfx: [e for e in ev4 if f":panel{p}:" in e["task"] and e["task"].endswith(sfx)]
    for p in range(4):
        assert min(e["start_ms"] for e in m4(p, ":operand")) >= max(m4(p, f":publish{s}")[0]["end_ms"] for s in (0, 1))
        for s in (0, 1): assert m4(p, f":packed_read{s}")[0]["end_ms"] <= m4(p, f":widen{s}")[0]["start_ms"]
        if p >= 2: assert m4(p, ":packed_read0")[0]["start_ms"] >= max(e["end_ms"] for e in m4(p - 2, ":cube"))
    assert not eng4._tasks and not eng4.allocations

@pytest.fixture(scope="module")
def mx8_source():
    raw = read_case("high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c32_p4k")
    plain = body(raw, tokens=4, context=4096, sequences=32, fusion=True)
    down = next(op for op in plain if op.op_id.endswith(".expert_down"))
    fused = deepcopy(raw)
    role, profile = mxfp8_swiglu.ROLE, mxfp8_swiglu.PROFILE
    fused["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion", "options": {"roles": [role],
            "program_profiles": {role: profile}, "program_options": {role: {"l1_k": 512, "compact": True}}}}
    return raw, down, plain, fused, body(fused, tokens=4, context=4096, sequences=32, fusion=True)

def test_mx_eight_frontend_runs_grouped_and_fused_source_kernels(mx8_source):
    raw, down, plain, case, fused = mx8_source
    raw = deepcopy(raw)
    raw["runtime"]["kernels"].setdefault("recipes", {})[down.recipe] = {"kind":"ascend_mxfp8_grouped_nz"}
    row = price(raw, down)
    md = row["operator"]["metadata"]["program_execution"]["source_mapping"]
    assert md["output_tiles_by_owner"] == (144,)*32 and Counter(md["group_rows"]) == {4:160, 3:128}
    assert row["operator"]["flops"] == 2*(288*16)*4096*256 and service(row, "fixpipe") == 2*1024*4096
    drained(row)
    calls = {o.op_id:o for o in fused if o.recipe == mxfp8_swiglu.PROFILE}
    downs = [[o for o in graph if o.op_id.endswith(".expert_down")] for graph in (plain, fused)]
    assert len(calls) == len(downs[0]) == len(downs[1]) == 42
    for a, b in zip(*downs, strict=True):
        assert same_semantics(a, replace(b, deps=a.deps)) and len(b.deps) == 1 and b.deps[0] in calls
    row = price(case, next(iter(calls.values())))
    assert row["operator"]["flops"] == 4*(288*16)*256*4096 and service(row, "mte3") == pytest.approx(1024*264, rel=1e-12)
def same_semantics(a, b):
    ignored = {"fused_origin_scopes", "input_conversion_subgraphs"}
    strip = lambda op: replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in ignored))
    return strip(a) == strip(b)

@pytest.mark.parametrize("swiglu,rows,n,k", [(False, *shape) for shape in (
    ((0, 3, 0, 17), 256, 256), ((1, 3), 160, 193), ((257, 7), 384, 320), ((0, 0), 128, 64),
    ((1, 3), 256, 64), ((3, 7), 384, 192), ((3, 7), 384, 193))]
    + [(True, *shape) for shape in (((0, 3, 9), 64, 128), ((1,), 256, 640), ((129, 2), 320, 256))])
def test_mx_eight_paired_stores_and_padded_panels_conserve_work(mx8_source, swiglu, rows, n, k):
    raw, down, _, fused, ops = mx8_source
    case = parse_case(fused if swiglu else raw)
    single_owner = not swiglu and (n,k) in ((256,64),(384,192),(384,193),(384,320))
    if single_owner:
        case = change_dies(case, lambda d:replace(d, compute_units=1, kernel_invocations=(),
            resource_units=tuple((r, 1) for r, _ in d.resource_units), buffer_units=tuple((r, 1) for r, _ in d.buffer_units)))
    align = lambda v, g: ((v + g - 1) // g) * g
    results = []
    for compact in (False, True):
        if swiglu:
            source = next(o for o in ops if o.recipe == mxfp8_swiglu.PROFILE)
            meta = dict(source.metadata)
            meta["mxfp8_swiglu_config"] = dict(meta["mxfp8_swiglu_config"], rows=rows, width=n, k=k, compact=compact)
            op = replace(source, local_shape=(sum(rows), n), metadata=tuple(meta.items()))
            row = price(case, op, ready=11)
            bm = align(min(sum(rows), 128), 16)
            bn = align(min(2 * n, 256), 16)
            tiles = [t for owner in mxfp8_swiglu.owner_tiles(rows, n, mxfp8_grouped.Mapping(bm, bn, 128, 256, 1), 32) for t in owner]
            expected_fixpipe = sum(8 * align(tm, 2) * tn for _, tm, tn, _, _ in tiles)
            expected_flops = sum(4 * align(tm, 16) * align(tn, 16) * align(k, 64) for _, tm, tn, _, _ in tiles)
            assert [service(row, r) for r in ("fixpipe", "mte3")] == pytest.approx([expected_fixpipe, sum(rows)*(n+n//32)])
        else:
            op = grouped_op(down, rows, n, k, wide=True)
            row = price(case, op, RecipeBook({op.recipe:mxfp8_grouped.recipe(compact=compact, l1_k=256)}))
            tiles = [t for owner in mxfp8_grouped.owner_tiles(rows, n, mxfp8_grouped.Mapping(256, 256, 128, 256, 1), 1 if single_owner else 32) for t in owner]
            expected_flops = sum(2 * align(tm, 16) * align(tn, 16) * align(k, 64) for _, tm, tn, _, _ in tiles)
        assert row["operator"]["flops"] == expected_flops
        drained(row)
        results.append(row)
    a, b = results
    assert a["serviced"] == pytest.approx(b["serviced"])
    if single_owner or swiglu:
        assert a["duration_ms"] == pytest.approx(b["duration_ms"], rel=.15 if swiglu else 1e-10)

@pytest.mark.parametrize("fault,code", [("extra", "requires_gate_up_SwiGLU_quantizer"),
                                      ("inner", "explicit_l1_k"), ("compact", "compact_must_be_boolean")])
def test_swiglu_requires_exact_members_and_valid_source_options(mx8_source, fault, code):
    case = parse_case(mx8_source[3])
    role = mxfp8_swiglu.ROLE
    ops = body(case, tokens=4, context=4096, sequences=32)
    declaration, members = next((d, m) for d, m in _groups(ops) if d["role"] == role)
    selection = select(case.runtime.kernels["lowerings"], "native_fusion")
    assert family_mxfp8_swiglu.capabilities()["scale_group_size"] == 32
    outputs = [members[-1].op_id]
    options = {"l1_k":100 if fault == "inner" else 512, "compact":1 if fault == "compact" else True}
    if fault == "extra":
        members = (*members, replace(members[0], op_id="additional_copy", family="copy", metadata=(), flops=0))
    with pytest.raises(ModelingError, match=code):
        family_mxfp8_swiglu.bind(declaration, members, outputs, selection, options)
def change_dies(case, change):
    return replace(case, machine=replace(case.machine, dies=tuple(map(change, case.machine.dies))))
def scale_dies(case, roles, factor):
    return change_dies(case, lambda d: replace(d, rates=tuple((r, v * factor if r in roles else v) for r, v in d.rates),
        compute_peaks=tuple((r, v * factor if "cube" in roles else v) for r, v in d.compute_peaks)))

@pytest.fixture(scope="module")
def prepared_cases():
    raw = read_case("high-concurrency-20261030/v41-dynamic-rotary-conditional/"
                    "a5_v41_fp8_tp1_dp8_ep8_c128_p2k-dynamic-rotary-source")
    raw["model"].update(layers=2, body_layers=2, layer_plans=raw["model"]["layer_plans"][:2])
    chosen = deepcopy(raw)
    chosen["runtime"]["kernels"]["lowerings"]["mx_matrix"] = {"implementation": "ascend_prepared_mx32_matrix",
        "options": {"projection_recipes": PREPARED, "scale_storage": "replace_checkpoint_scales"}}
    return parse_case(raw), parse_case(chosen)

@pytest.mark.parametrize("batch", (1, 16))
def test_prepared_projections_preserve_graph_and_physical_output(prepared_cases, batch):
    before, after = (body(c, context=2048, sequences=[batch]+[0]*7, fusion=True) for c in prepared_cases)
    assert sum(dict(o.metadata).get("semantic_recipe") in PREPARED for o in after) == 6
    for a, b in zip(before, after, strict=True):
        if a.recipe not in PREPARED:
            assert same_semantics(a, b)
            continue
        m, n, k = a.local_shape
        md, groups = dict(b.metadata), dict(a.metadata)["matrix_groups"]
        assert (b.op_id, b.deps, b.flops) == (a.op_id, a.deps, a.flops)
        assert md["packed_weight_bytes"] == groups*n*k*33//32 and md["fixpipe_bytes"] == 2*m*n
        row = price(prepared_cases[1], b)
        assert [service(row, r) for r in ("fixpipe", "hbm", "mte3", "vector")] == pytest.approx([2*m*n, b.mte2_bytes+2*m*n, 0, 0])
        drained(row)
        if a.recipe == "grouped_output_a":
            view = md["native_tensor_view"]
            assert (groups, m, view["input_stride"], view["output_stride"]) == (8, 8*batch, (8*k, k, 1), (8*n, n, 1))
def test_prepared_scale_residency_preserves_checkpoint_bytes_and_roundtrip(prepared_cases):
    raw, selected = prepared_cases
    terms = lambda c:{t.name:t for t in layer_weight_terms(c.model.layer_plans[0], c.model)}
    a, b = map(terms, prepared_cases)
    assert terms(parse_case(selected.to_dict())) == b
    for name, t in b.items():
        x, y = ((term, c.model.quant, c.runtime.parallel) for term, c in ((a[name], raw), (t, selected)))
        assert term_bytes(*x, source=True) == term_bytes(*y, source=True)
        delta = t.rows*t.cols*(1/32-1/1024) if name in ("q_b", "wo_a", "wo_b") else 0
        assert term_bytes(*y)-term_bytes(*x) == delta
        assert not delta or t.runtime_quant.weight_block == (1, 32)
    kernels = deepcopy(selected.runtime.kernels)
    del kernels["lowerings"]["mx_matrix"]
    assert terms(parse_case(replace(selected, runtime=replace(selected.runtime, kernels=kernels)))) == a

@pytest.mark.parametrize("axis", MATRIX_PIPES)
def test_grouped_and_prepared_matrix_timing_respond_to_scarce_pipes(mx8_source, prepared_cases, axis):
    raw, down, *_ = mx8_source
    case = prepared_cases[1]
    prepared = next(o for o in body(case, context=2048, sequences=[16]+[0]*7, fusion=True)
                    if dict(o.metadata).get("semantic_recipe") == "grouped_output_a")
    grouped = grouped_op(down, (2, 3), 512, 256, wide=True)
    for fixture, op, recipe in ((case, prepared, None), (parse_case(raw), grouped, RecipeBook({grouped.recipe:mxfp8_grouped.recipe()}))):
        slow, base = (price(scale_dies(fixture, (axis,), f), op, recipe) for f in (.001, 1))
        assert slow["duration_ms"] > base["duration_ms"]
        assert all(slow["operator"][field] == base["operator"][field] for field in ("flops", "hbm_bytes"))
        if recipe is None and axis == "cube":
            slow["serviced"]["matrix@r0:s0"] *= .001
        if recipe is None:
            assert slow["serviced"] == pytest.approx(base["serviced"])
def test_mx_eight_owner_tiles_partition_actual_output():
    for module, tile, rows, widths, cores in (
            (mxfp8_grouped, (256, 256, 128, 256, 1), ((3, 2), (257, 5), (0, 9, 0), (35, 0, 17), (7, 6, 11), (0, 0)),
             (32, 128, 160, 256, 384, 512, 768), (3, 7, 16, 32)),
            (mxfp8_swiglu, (128, 256, 128, 512, 2), ((0, 1, 0, 3), (129, 257), (2, 3, 4, 5)), (64, 256, 320), (3,))):
        for banks, n, owners in product(rows, widths, cores):
            tiles = module.owner_tiles(banks, n, mxfp8_grouped.Mapping(*tile), owners)
            cells = Counter((bank, y, x) for owner in tiles for bank, m, w, i, j in owner
                            for y in range(i, i+m) for x in range(j, j+w))
            assert len(cells) == sum(banks)*n and all(v == 1 for v in cells.values())
    load, steps = mxfp8_grouped.panel_work(3, 33, 128, 128)
    assert load.resources["mte2_aic"] == 3*128+64*128+(3+33)*4
    assert steps[0][0].resources["mte1"] == (16+64)*128+(16+48)*4
    assert steps[0][1].matrix["fp8_e4m3"] == 2*16*48*128
def test_grouped_device_minimum_applies_once(mx4_source, mx8_source):
    raw, (gate, _), _ = mx4_source
    raw = deepcopy(raw)
    op = grouped_op(gate, (1,)*4, 8192, 512)
    base = mx4(raw, op, compact=False, compact_panels=False)
    raw["runtime"]["kernels"]["invocations"] = [{"match":{"family":"grouped_gemm"}, "startup_ms":.003,
         "minimum_ms":10, "provenance":{"basis":"test_explicit_kernel_boundary"}}]
    for flag in (False, True):
        row = mx4(raw, op, compact=flag)
        assert row["serviced"] == pytest.approx(base["serviced"])
        assert row["duration_ms"] == pytest.approx(10) and len(row["basis"]["invocation_models"]) == 1
    raw, down, *_ = mx8_source
    case = change_dies(parse_case(raw), lambda d:replace(d, kernel_invocations=(("cube", InvocationTiming(minimum_ms=1)),)))
    op = grouped_op(down, (16, 16), 512, 320, wide=True)
    assert price(case, op, RecipeBook({op.recipe:mxfp8_grouped.recipe(l1_k=256)}))["duration_ms"] == pytest.approx(1)

@pytest.fixture(scope="module")
def situ_case():
    raw = read_case("kimi-k3-a5")
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant"]
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["rmsnorm_quant"] = "vllm_ascend_native_norm"
    case = parse_case(raw)
    quant = replace(case.model.quant, weight_format="fp8_e4m3", expert_weight_format="fp8_e4m3", weight_block=(32, 32),
                    activation_format="fp8_e4m3", activation_group_size=32, activation_scale_format="fp8_e8m0",
                    activation_scale_pack=2, exclusions=())
    return replace(case, model=replace(case.model, quant=quant))
def situ_group(case):
    return next((d, m) for d, m in _groups(body(case, context=16))
                if d["role"] == "situ_quant" and d["tensor_view"]["width"] == 768)
def situ_profile(case, **profile):
    kernels = deepcopy(case.runtime.kernels)
    kernels["lowerings"]["native_fusion"] = {"implementation": "declared_native_fusion",
                                            "options": {"roles": ["situ_quant"], **profile}}
    return replace(case, runtime=replace(case.runtime, kernels=kernels))

@pytest.mark.parametrize("fault,code", [("external", "external_activation_intermediate"), ("axis", "axis_minus_one")])
def test_situ_rejects_invalid_intermediate_outputs_and_tensor_axis(situ_case, fault, code):
    declaration, members = situ_group(situ_case)
    outputs = tuple(m.op_id for m in (members if fault == "external" else members[-1:]))
    if fault == "axis":
        declaration = dict(declaration, tensor_view=dict(declaration["tensor_view"], axis=0))
    with pytest.raises(ModelingError, match=code):
        family_situ_mx.config_from_group(declaration, members, outputs)
def test_situ_fusion_replaces_member_work_and_preserves_owned_reads(situ_case):
    _, members = situ_group(situ_case)
    assert lower_native_fusions(members, situ_case.runtime) == tuple(members)
    case = situ_profile(situ_case, program_profiles={"situ_quant":SITU})
    changed = tuple(replace(o, vector_ops=o.vector_ops+10**9, hbm_bytes=o.hbm_bytes+10**9,
                            mte3_bytes=o.mte3_bytes+10**9) for o in members)
    fused, = lower_native_fusions(members, case.runtime)
    recipe = RecipeBook.from_mapping(case.runtime.kernels)
    a, b = (recipe.prepare(lower_native_fusions(ops, case.runtime)[0], geometry(case))[0] for ops in (members, changed))
    assert a.hbm_bytes == b.hbm_bytes == 3072+792 and a.mte3_bytes == b.mte3_bytes == 792
    assert a.vector_ops == b.vector_ops > sum(o.vector_ops for o in members)
    report = graph_costs(case, (fused,), external_inputs=dict.fromkeys(fused.deps, 0), isolated=False, trace_limit=600)
    loads = [e for e in report["events"] if ":tile0:load" in e["task"]]
    assert {tuple(e["placement"]["mte2_aiv@r0:s0"]) for e in loads} == {(0,), (1,), (2,)}
    assert [e["serviced"]["mte2_aiv@r0:s0"] for e in loads] == [1024]*3
    case = situ_profile(situ_case, traffic_profile="situ_mx_quant_arch35_2a9bd1b0")
    fused, = lower_native_fusions(members, case.runtime)
    assert fused.vector_ops == sum(o.vector_ops for o in members)
    assert fused.hbm_bytes == sum(o.hbm_bytes for o in members)-2*next(o.mte3_bytes for o in members if o.family=="situ")

@pytest.mark.parametrize("m,n,beta,owners,lanes", [
    (1, 65, None, 1, 0), (1, 257, None, 2, 0), (1, 65, 25., 1, 0), (1, 768, None, 3, 32), (1, 768, 25., 3, 54), (130, 768, None, 64, 0)])
def test_situ_payload_and_vector_lanes_follow_actual_tiles(situ_case, m, n, beta, owners, lanes):
    cfg = family_situ_mx.Config(m, n, "fp8_e4m3", 4., beta)
    op, recipe = family_situ_mx.make_op(cfg), family_situ_mx.recipe_book()
    g = Geometry.from_devices(situ_case.machine.dies[:8])
    plan = family_situ_mx.tiling_for(cfg, g)
    assert family_situ_mx.owner_rows(cfg, plan) == ((3,)*2+(2,)*62 if m == 130 else (1,)*owners)
    prepared, _ = recipe.prepare(op, g)
    assert prepared.mte3_bytes == m*(n+2*ceil(n/64)) and prepared.hbm_bytes == 4*m*n+prepared.mte3_bytes
    assert n >= 768 or prepared.vector_ops >= 6*256*owners
    if n == 65 and beta:
        plain, _ = recipe.prepare(family_situ_mx.make_op(replace(cfg, linear_beta=None)), g)
        assert prepared.vector_ops-plain.vector_ops == 22*n
    if lanes:
        exp = 3*8*(32+16+15)
        result = graph_costs(situ_case, (op,), recipes=recipe, isolated=False, trace_limit=200)
        assert result["execution"]["serviced"]["vector@r0:s0"] == prepared.vector_ops == n*lanes+exp+24*13+3*(256+512+768)
        assert sum(e["serviced"]["vector@r0:s0"] for e in result["events"] if e["task"].endswith(":maxexp")) == exp

@pytest.mark.parametrize("k,width,rows", [(64, 64, (1, 0, 2)), (320, 128, (0, 3, 0, 1)), (3584, 3072, (1, 1)+(0,)*26),
    (256, 128, (2,)+(0,)*27), (3584, 3072, (0,)*28)])
def test_weight_supply_tracks_real_banks_and_ignores_capacity(k, width, rows):
    raw = read_case("kimi-k3-a5-current-checkpoint-conditional")
    cfg = mxfp4_supply.Config(k, width, rows, 512)
    if k == 3584 and any(rows):
        raw["runtime"]["routing"] = {"policy":"balanced"}
        d, m = next((d, m) for d, m in _groups(body(raw)) if d["role"] == "grouped_gate_up_situ_quant")
        assert mxfp4_supply.config_from_group(d, m, allocated_rows=512) == cfg
    recipe = mxfp4_supply.recipe_book()
    row = price(raw, mxfp4_supply.make_op(cfg), recipe)
    count = 2*width*k*sum(r>0 for r in rows)
    ntiles = sum(r > 0 for r in rows) * (width // cfg.half_tile)
    issue_slots = data_writes = ub_accesses = ub_stores = 0
    for offset in range(0, k, cfg.k_tile):
        for sub in (0, 1):
            packed = min(cfg.k_tile // 2, max(0, k - offset - sub * cfg.k_tile // 2)) * cfg.half_tile
            groups = packed // 256
            vf = max(1, ceil(groups / 64))
            issue_slots += ntiles * (4 * vf + 12 * groups)
            data_writes += ntiles * (3 * vf + 8 * groups)
            ub_accesses += ntiles * (3 * groups)
            ub_stores += ntiles * (2 * groups)
    assert [service(row, r) for r in ("hbm", "mte3", "vector", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "mte2_aic")] == pytest.approx([
        count/2, count, 32 * issue_slots, data_writes, ub_accesses, ub_stores, 0])
    drained(row)
    tiles = [(i, m, n) for i, m in enumerate(rows) if m for n in range(0, width, cfg.half_tile)]
    assert {t:o for o in range(32) for t in mxfp4_supply.tiles(cfg, o, 32)} == {t:i%32 for i, t in enumerate(tiles)}
    if k == 256:
        other = price(raw, mxfp4_supply.make_op(replace(cfg, allocated_rows=2)), recipe)
        assert (row["serviced"], row["duration_ms"]) == (other["serviced"], other["duration_ms"])

@pytest.mark.parametrize("role", ("hbm", "mte2_aiv", "mte3", "vector"))
def test_supply_service_uses_reciprocal_physical_rates(role):
    raw = read_case("kimi-k3-a5-current-checkpoint-conditional")
    op = mxfp4_supply.make_op(mxfp4_supply.Config(256, 128, (1,), 512))
    rates = dict.fromkeys(("hbm", "mte2_aiv", "mte3", "vector", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores"), 1e18)
    times = []
    for factor in (.5, 1, 2):
        raw["machine"].setdefault("device", {})["rates"] = dict(rates, **{role:1e8*factor})
        times.append(price(raw, op, mxfp4_supply.recipe_book())["duration_ms"])
    assert times == pytest.approx([2*times[1], times[1], times[1]/2], rel=1e-6) and times[1] > 0

@pytest.mark.parametrize("cfg", [Config(320, 320, (1, 0, 2)), Config(448, 544, (0, 257, 0, 3)),
                                 Config(5120, 4608, (17,)), Config(64, 32, (0, 0))])
def test_geometry_cache_preserves_integer_tiles_runs_and_work(cfg):
    for cores, compact, directional in product((1, 3, 32), (False, True), (False, True)):
        ledger = geometry_ledger(cfg, cores, compact, directional)
        shapes = [[(m, n) for *_, m, n in owner_tiles(cfg, o, cores)] for o in range(cores)]
        work = sum((tile_work(cfg, m, n, cores=cores, directional_mte1=directional) for ts in shapes for m, n in ts),
                   Work(resources={"hbm":512*cores, "mte2_aiv":512*cores}))
        inventory = sum((tile_inventory(cfg, m, n) for ts in shapes for m, n in ts), InstructionInventory({}))
        assert ledger.work == work and ledger.inventory == inventory
        assert ledger.counts == tuple(map(len, shapes))
        assert ledger.runs == tuple(sum(not compact or a!=b for a, b in zip([None, *ts], ts)) for ts in shapes)
def test_geometry_cache_bounds_and_immutable_work():
    cfg = Config(320, 320, (1, 0, 2))
    saved = geometry_ledger(cfg, 32)
    counts = saved.inventory.to_dict()["instructions"]
    for mapping, key in ((saved.work.resources, "hbm"), (saved.work.matrix, "fp8_e4m3"), (saved.inventory.counts, next(iter(counts)))):
        with pytest.raises(TypeError): mapping[key] = 0
    for i in range(1, 130): geometry_ledger(Config(i*64, 32, (0, 0)), 2)
    assert geometry_ledger.cache_info().currsize == 128 and geometry_ledger(cfg, 32) == saved and geometry_ledger(cfg, 32) is not saved
    cfg = Config(5120, 4608, (1, 0, 2)+(0,)*45)
    assert [n for _, n in columns(cfg, 32)] == [128]*16+[160]*16 and Counter(t[0] for o in range(32) for t in owner_tiles(cfg, o, 32)) == {0:32, 2:32}
    assert effective_l2_bytes(Geometry(1, {}, {"l2":1024}, {})) == 768

def test_source_pipeline_cadence_slots_and_cycle_jump():
    p = Panel(None, None, ((0, 0, 0, 3, 4, 2), (0, 0, 0, 0, 0, 0)), (False, False), ((0, 0),))
    assert [pipeline_cycle((p,), n).span_ms for n in (1, 2, 3)] == [9, 13, 17]
    tail = replace(p, activation=100, activation_scale=17, vectors=((8, 4, 3, 2, 1, 5), p.vectors[1]), scales=(True, False))
    for n in (1, 3, 5, 31, 100): assert pipeline_cycle((p,), n, tail=(tail,)).span_ms == max(117, 4*n+15)
    large = pipeline_cycle((p,), 100000)
    assert large.span_ms == 400005 and large.stepped_panels < 100 and large.jumped_panels > 99000

@pytest.mark.parametrize("kind", ("buffers", "minimum", "alias"))
def test_situ_runtime_keeps_owners_buffer_reuse_and_kernel_minimum(situ_case, kind):
    recipe = family_situ_mx.recipe_book(); m, n = (3, 256) if kind == "buffers" else (1, 768)
    op = family_situ_mx.make_op(family_situ_mx.Config(m, n, "fp8_e4m3", 4., 25. if kind=="minimum" else None))
    def hardware(d):
        units, rates = dict(d.resource_units), dict(d.rates)
        if kind == "buffers":
            units.update(vector=1, mte2_aiv=1, mte3=1)
            return replace(d, resource_units=tuple(units.items()), buffers=tuple(dict(d.buffers, ub=6656).items()), buffer_units=tuple(dict(d.buffer_units, ub=1).items()))
        if kind == "minimum": return replace(d, launch_us=7, kernel_invocations=(("vector", InvocationTiming(minimum_ms=.01)),))
        for pipe in ("mte2", "mte2_aic", "mte2_aiv", "mte3"): units.pop(pipe, None); rates.pop(pipe, None)
        modes = dict(d.service_modes, mte2=("dma", 4e12), mte3=("dma", 4e12))
        return replace(d, rates=tuple(dict(rates, dma=4e12).items()), resource_units=tuple(dict(units, dma=128).items()), service_modes=tuple(modes.items()))
    def run(case, submit=False):
        return graph_costs(case, (op,), recipes=recipe, isolated=False, trace_limit=200, include_submission=submit)
    case = change_dies(situ_case, hardware)
    report = run(case)
    work = report["execution"]["serviced"]
    if kind == "buffers":
        events = {e["task"].split(":tile")[-1]:e for e in report["events"] if ":tile" in e["task"]}
        assert events["1:situ"]["start_ms"] >= events["0:quantize"]["end_ms"]
        assert (work["vector@r0:s0"], work["memory@r0:s0"]) == (31008, 3864)
    elif kind == "minimum":
        host = run(case, True)
        assert (report["duration_ms"], host["duration_ms"]) == pytest.approx((.01, .017))
        assert work == run(situ_case)["execution"]["serviced"] and host["execution"]["serviced"]["host@r0:s0"] == .007
    else:
        transfers = [e for e in report["events"] if e["task"].endswith((":load", ":store"))]
        assert {tuple(e["placement"]["dma@r0:s0"]) for e in transfers} == {(0, 1), (2, 3), (4, 5)}
        for e in transfers:
            assert e["end_ms"]-e["start_ms"] == pytest.approx(e["serviced"]["dma@r0:s0"]/(4e9/64))
        assert work["dma@r0:s0"] == 3864
