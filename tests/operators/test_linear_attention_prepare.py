"""Linear-attention prepare, call binding, and WY work.

[Role] Exercise source prepare plans, caller bindings, and governed WY lowering.
[Invariants] Conserve typed arithmetic and tensor traffic at ragged boundaries.
[Timeouts & Locks] Bounded local planning and execution.
[Side-Effect Policy] Read-only fixture inputs and in-memory simulation.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.builders import linear_layer, linear_model
from _support.linear_cases import kda_hardware_case, sku_planning_context
from mindie_sim.prediction.operators.linear_attention.gdn_chunk32 import PR22147_SOURCE, STATE
from mindie_sim import compose_case, graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.observations.calibration import bind_observation
from mindie_sim.observations.comparison import call_observations
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.operators.kda.family_kda_prepare import IMPLEMENTATION as KDA_PREPARE
from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.operators.linear_attention.call_gdn import make_call, _gating_plan
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators.linear_attention import gdn_prepare as gdn; from mindie_sim.prediction.operators.kda import kda_prepare
from mindie_sim.prediction.operators.causal_conv import native_causal_conv_prefill as convolution
from mindie_sim.prediction.operators.linear_attention import build, compute_wy_chunk_work, resolve_operator_plan
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.operators.recurrent.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure import export_structure
from tools.run_operator_cases import control_case

pytestmark = pytest.mark.tier0

NEWEST_SKU = REPRESENTATIVE_SKUS["a5"]
OLDER_SKU = REPRESENTATIVE_SKUS["a3"]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GEOMETRY = dict(scope="body", phase="decode", tokens=4, sequences=2, context=128)
CALLER = "vllm_ascend_gdn_decode"
BT, K, FP32, BF16 = 64, 128, 4, 2
VCS_OPS = (32 * 31 // 2) * (64 + 64 + 4 + 2)
V, CHUNK = 128, 64
SCORE_HEAD = 2 * 2 * 32 * (32 + 64) * 128
FP32_HEAD = 4 * 2 * 16 ** 3 + 2 * 2 * 32 ** 3
DIAG_HEAD = 4 * (16 * (119 + 119 + 14) + 16)

def source_copy_oracle(params):
    """Enumerate the operand loads and Fixpipe calls in each source pack."""
    heads = params["heads"]
    ratio = heads // params["key_heads"]
    vdim = params["dv"]
    pack_size = 3 if ratio == 3 else 4
    tasks = [
        (head, min(BT, length - start))
        for length in params["lengths"] for start in range(0, length, BT)
        for head in range(heads)
    ]
    l0_tiles, fix_tiles, ub_tiles = [], [], []
    for start in range(0, len(tasks), pack_size):
        pack = tasks[start:start + pack_size]
        # Stage4_PrefillI loads I into L0 once, then reuses it for all heads.
        l0_tiles.extend([(BT, BT, FP32)] * 2)
        for head, rows in pack:
            # Stage3 copies I_vcs, then packs two diagonal leaves. The
            # CANN UB->UB DataCopy implementation uses PIPE_V, not MTE3.
            ub_tiles.extend(((32, 64, FP32), (32, 32, FP32), (32, 32, FP32)))
            if head % ratio == 0:
                l0_tiles.extend([(BT, K, BF16)] * 2)
                dual = ratio > 2 or (ratio == 2 and len(pack) < 4)
                fix_tiles.extend([(BT, BT, FP32)] * (2 if dual else 1))
            l0_tiles.extend([(BT, BT, FP32)] * 2)  # LeafLeft @ -L
            fix_tiles.append((BT, BT, FP32))       # Y -> GM
            l0_tiles.extend([(BT, BT, FP32)] * 4)  # I@Left and Y@Right
            if params.get("output_a", True):
                fix_tiles.append((rows, BT, BF16))
            if rows < BT or not params.get("output_a", True):
                fix_tiles.append((BT, BT, BF16))   # A -> L1
            # W and each 128-column U half reload both BF16 L0 operands.
            for _ in range(1 + vdim // K):
                l0_tiles.extend(((BT, BT, BF16), (BT, K, BF16)))
                fix_tiles.append((rows, K, BF16))
    return {
        "mte1": sum(rows * columns * width for rows, columns, width in l0_tiles),
        "fixpipe": sum(rows * columns * width for rows, columns, width in fix_tiles),
        "ub_copy": sum(rows * columns * width for rows, columns, width in ub_tiles),
    }

def layout_totals(params, cores=32):
    """Ledger from explicit chunk rows, projecting stage reads and stores."""
    hv, hk, dv = (params[key] for key in ("heads", "key_heads", "dv"))
    rows = [min(BT, n - start) for n in params["lengths"] for start in range(0, n, BT)]
    reads = writes = 0
    local = cores * 9 * BT**2 * FP32
    vector = cores * (2 * 32 * BT * 7 + 6 * BT**2)
    for n in rows:
        # Gate/beta, Q/K, inverse workspace, A reload, V and K-hat.
        reads += hv * (4*n + BT**2*FP32 + (n == BT)*BT**2*BF16 + n*(dv+K)*BF16)
        reads += hk * 2*n*K*BF16
        # Gate/beta, normalized Q/K and rstd, Y, A, W, U.
        writes += hv * (2*n*FP32 + BT**2*FP32 + n*(BT+K+dv)*BF16)
        writes += hk * 2*n*(K*BF16+FP32)
        local += hk*BT*K*BF16 + hv*(BT**2*FP32 + 2*(BT//2)**2*FP32 + BT*(dv+K)*BF16)
        vector += hv * (9*n + 8*BT**2 + VCS_OPS + BT*(3*dv+5*K))
        if n < BT:
            vector += hv * BT * (2+dv+K) + hk * 4*BT*K
    fix_gm = hv * (len(rows)*BT**2*FP32 + sum(rows)*(BT + (dv if dv > K else 0))*BF16)
    return dict(mte2=reads, mte3=writes-fix_gm+local, hbm=reads+writes, vector=vector,
                fp32=len(rows)*hv*4*2*BT**3, bf16=len(rows)*2*BT**2*(hk*K+hv*(K+dv)))
@pytest.mark.parametrize("output_a", (False, True))
def test_prepare_copy_destinations_and_typed_work(output_a):
    cases = json.loads((FIXTURES / "operator_cases/gdn_prepare.json").read_text())["cases"]
    for case in cases:
        params = dict(case["params"], output_a=output_a)
        op = gdn.make_op(params)
        prepared, plan = PlannedRecipe(gdn.build).prepare(op, Geometry(32, {"vector": 64}, {}))
        for role, amount in source_copy_oracle(params).items():
            assert plan.work.resources[role] == amount
        if output_a:
            expected = layout_totals(params)
            assert dict(prepared.matrix_flops) == {dtype: expected[dtype] for dtype in ("bf16", "fp32")}
            for role in ("mte2", "mte3", "hbm", "vector"):
                assert plan.work.resources[role] == expected[role]
            assert prepared.hbm_bytes == expected["hbm"]
            assert prepared.vector_ops == expected["vector"]

def test_prepare_normalizes_key_heads_and_pads_scaling_operands():
    params = dict(lengths=[65], heads=12, key_heads=4, dk=K, dv=K, chunk=BT, dtype="bf16",
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                  use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
                  use_exp2=False, gate_dtype="bf16", output_a=True)
    geo = Geometry(32, {"vector": 64}, {})
    norm = gdn.build(gdn.make_op(params), geo).work
    plain = gdn.build(gdn.make_op(dict(params, use_qk_l2norm_in_kernel=False)), geo).work
    assert norm.resources["hbm"] - plain.resources["hbm"] == 4 * 65 * (3 * K * BF16 + 2 * FP32)
    assert norm.resources["reg_reduce_lanes"] > plain.resources.get("reg_reduce_lanes", 0)
    hidden = gdn.build(gdn.make_op(dict(params, output_a=False)), geo).work
    assert norm.resources["hbm"] - hidden.resources["hbm"] == 12 * (2 * BT**2 + BT) * BF16
    assert norm.resources["fixpipe"] - hidden.resources["fixpipe"] == 12 * BT * BF16
    assert norm.resources["mte3"] == hidden.resources["mte3"] == layout_totals(params)["mte3"]

def test_ragged_chunk_addresses_and_grouped_pack_order():
    chunks = gdn.PackedChunks.from_lengths([63, 1, 65])
    spans = list(chunks.all_ranges())
    assert spans == [(0, 0, 0, 63), (1, 0, 63, 1), (2, 0, 64, 64), (2, 1, 128, 1)]
    for head in range(4):
        offsets = [((head * 129 + start) * K, (head * 129 + start + rows) * K)
                   for _, _, start, rows in spans]
        assert [right for _, right in offsets[:-1]] == [left for left, _ in offsets[1:]]
        assert offsets[-1][1] - offsets[0][0] == 129 * K
    assert [gdn._pack_work_id(0, 4, item, 2) for item in range(4)] == [0, 2, 1, 3]

def observed_calls():
    protocol = json.loads((FIXTURES / "observations/gdn-call-protocol.json").read_text())
    return {row["identity"]["conditions"]["role"]: row for row in call_observations(protocol)}

def test_historical_fp32_gating_has_two_fp32_outputs_and_ignores_timing_values():
    observed = observed_calls()["gating"]
    first = bind_observation(observed)
    changed = deepcopy(observed)
    changed["samples"] = [{"duration_ms": 1e12}]
    changed["statistics"] = {"sample_median_ms": 1e12}
    changed["canary"] = {"ok": False}
    second = bind_observation(changed)
    assert first["status"] == second["status"] == "bound"
    assert first["call"].ops == second["call"].ops
    assert first["call"].ledger["outputs"] == {
        "g": {"shape": [1, 4096, 48], "dtype": "fp32"},
        "beta": {"shape": [1, 4096, 48], "dtype": "fp32"},
    }
    changed["identity"]["conditions"]["dtype"] = "bf16"
    assert bind_observation(changed)["status"] == "semantic_mismatch"

def test_masked_gating_launch_keeps_parameters_and_conserves_call_work():
    call = make_call("gating", {"rows": 1, "heads": 9})
    raw = control_case(units=2, split_copy=True)
    result = verify_operator(raw, call.ops[0], _gating_plan, input_ready_ms=3)
    assert result["status"] == "passed"
    work = result["result"]["operator"]
    assert work["mte2_bytes"] == 9 * 8 * (1 + 4)
    assert work["mte3_bytes"] == 9 * 8
    assert work["hbm_bytes"] == work["mte2_bytes"] + work["mte3_bytes"]
    assert work["vector_ops"] == 4 * 2 * (13 * 64 * 8 + 2 * 8)
    assert result["input_dependency_obeyed"] and result["buffers_released"]
    raw["runtime"]["submission"] = {"scheduler_ms": 1000, "completion_ms": 1000}
    graph = graph_costs(raw, call.ops, recipes=call.recipes)
    row, = graph["isolated"]["operators"]
    assert row["declared_service"] == pytest.approx(row["serviced"])
    assert graph["duration_ms"] == pytest.approx(row["duration_ms"])

def test_historical_conv_call_retains_bias_device_flags_and_real_state_rows():
    bound = bind_observation(observed_calls()["conv_prefill"])
    assert bound["status"] == "bound"
    op, = bound["call"].ops
    metadata = dict(op.metadata)
    assert metadata["bias"] is True and metadata["has_initial_state"] == (False,)
    assert metadata["cache_indices"] == (1,)
    assert (metadata["state_rows"], metadata["state_history_slots"]) == (2, 3)
    assert convolution.host_tiling(op, core_num=64, ub_bytes=253952)["has_init_state_workspace"] == 1
    base = dict(conv_width=16, kernel=4, bias=True, cache_indices=(0, 1), state_cache_rows="batch+1")
    explicit = make_call("conv_prefill", dict(base, lengths=(1, 3), cu_seqlens=(0, 1, 4)))
    cumulative = make_call("conv_prefill", dict(base, cu_seqlens=(0, 1, 4)))
    assert explicit.ops == cumulative.ops
    assert dict(explicit.ops[0].metadata)["cache_indices"] == (0, 1)
    for offsets in ((1, 2, 4), (0, 3, 2), (0, 2.0, 4), (0, 2, 4)):
        with pytest.raises(ModelingError):
            make_call("conv_prefill", dict(base, lengths=(1, 3), cu_seqlens=offsets))

def test_wrapper_binding_preserves_complete_call_boundary():
    rows = observed_calls()
    for role in ("scan_decode", "conv_update", "scan_prefill"):
        bound = bind_observation(rows[role])
        assert bound["observation_boundary"] == rows[role]["boundary"]
        assert (bound["call"] is not None) == (role == "scan_prefill")
        assert bool(bound["mismatches"]) == (role != "scan_prefill")

def test_modified_producer_cannot_claim_reviewed_binding():
    observed = deepcopy(observed_calls()["gating"])
    observed["identity"]["producer_sha256"] = "modified-producer"
    binding = bind_observation(observed)
    assert binding["status"] == "semantic_mismatch" and binding["call"] is None
    assert "producer source" in binding["mismatches"][0]
def case(sku):
    return {
        "machine": {"backend": "ascend", "sku": sku, "num_devices": 8},
        "model": {"preset": "qwen3.6-27b", "quantization": "w8a8", "layers": 1},
        "runtime": {"tp": 8},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    }
def source_case(**options):
    raw = case(NEWEST_SKU)
    raw["runtime"]["kernels"] = {"lowerings": {"gdn_decode": {
        "implementation": CALLER, "options": options}}}
    return raw
def gating(raw, ops=None):
    ops = target_operators(raw, **GEOMETRY) if ops is None else ops
    return next(row for row in operator_costs(raw, ops)["operators"]
                if row["operator"]["recipe"] == "ascend_gdn_decode_gating")

def test_partial_source_binding_executes_same_store_and_survives_graph_roundtrip():
    raw = source_case(recurrent_implementation="arch35_fp32")
    before = deepcopy(raw)
    baseline, body = (predict(value, **GEOMETRY) for value in (case(NEWEST_SKU), raw))
    assert body["timeline"] == baseline["timeline"]
    graphs = deepcopy(target_graphs(raw, **GEOMETRY))
    binding = graphs.effective_kernels["lowerings"]["gdn_decode"]
    resolved = consume_target_declaration(parse_case(raw), graphs)
    assert resolved.runtime.kernels["lowerings"]["gdn_decode"] == binding
    exported = export_structure(raw, **GEOMETRY)["execution"]["software"]["kernels"]
    assert exported["lowerings"]["gdn_decode"] == binding
    graph = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False)
    assert graph["duration_ms"] == pytest.approx(body["body_forward_ms"])
    assert graph["execution"]["serviced"] == pytest.approx(body["execution"]["serviced"])
    assert raw == before

def test_empty_store_is_dma_opt_out_and_explicit_queue_is_software_owned():
    options = {"recurrent_implementation": "analytical"}
    queue = {"resource": "scalar_store_issue", "queue": {
        "outstanding": 1, "granule_bytes": 128, "response_ms": .001}}
    custom = source_case(**options, gating_beta_store=queue)
    inherited = gating(source_case(**options))
    plain = gating(source_case(**options, gating_beta_store={}))
    explicit = gating(custom)
    assert explicit["duration_ms"] > inherited["duration_ms"] > plain["duration_ms"]
    scalar = "scalar_store_issue@r0:s0"
    assert scalar not in plain["serviced"]
    assert explicit["serviced"][scalar] == inherited["serviced"][scalar] == 48
    ops = target_operators(custom, **GEOMETRY)
    custom["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .01}}}
    assert gating(custom, ops)["duration_ms"] == explicit["duration_ms"]

def test_inherited_hardware_reference_reprices_existing_explicit_source_graph():
    raw = source_case(recurrent_implementation="arch35_fp32")
    ops = target_operators(raw, **GEOMETRY)
    changed = deepcopy(raw)
    changed["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .001}}}
    before, reused, fresh = gating(raw, ops), gating(changed, ops), gating(changed)
    assert reused["duration_ms"] == fresh["duration_ms"] > before["duration_ms"]
    assert reused["serviced"] == pytest.approx(before["serviced"], rel=1e-12)
@pytest.mark.parametrize("sku", (NEWEST_SKU, OLDER_SKU))
def test_hardware_without_store_queue_keeps_gate_outputs_on_dma(sku):
    raw = source_case(recurrent_implementation="analytical")
    raw["machine"] = parse_case(case(sku)).machine.to_dict()
    for device in raw["machine"]["devices"]:
        device["store_queues"] = {}
    ops = target_operators(raw, **GEOMETRY)
    assert "scalar_store_issue@r0:s0" not in gating(raw, ops)["serviced"]
    assert "gating_beta_store" not in ops.effective_kernels["lowerings"]["gdn_decode"]["options"]

def test_policy_analytic_opt_out_and_source_leaf_keep_independent_semantics():
    raw = source_case(recurrent_implementation="analytical")
    binding = raw["runtime"]["kernels"]["lowerings"]["gdn_decode"]
    raw["runtime"]["kernels"]["lowerings"]["gdn_decode"] = {
        "cases": [{"domain": {"max_query_tokens": {"max": 1}}, "binding": binding}],
        "otherwise": "analytical"}
    analytic = target_operators(raw, **GEOMETRY)
    assert not any(op.recipe == "ascend_gdn_decode_gating" for op in analytic)
    native = target_operators(raw, **{**GEOMETRY, "tokens": 1})
    assert next(dict(op.metadata)["beta_store"] for op in native
                if op.recipe == "ascend_gdn_decode_gating")["from_hardware"]
    selected = consume_target_declaration(parse_case(raw), native)
    assert selected.runtime.kernels == native.effective_kernels
    source = next(op for op in native if op.family == "recurrent_vector")
    assert dict(source.metadata)["recurrence_selection"] == "analytical"

def recurrent_ops(kind, lengths, selection=None):
    model = linear_model(kind, layers=1)
    heads, dk, dv = (2, 128, 128) if selection else (3, 16, 32)
    plan = linear_layer(0, kind, linear_heads=heads, linear_key_heads=heads, linear_value_heads=heads,
                        linear_key_head_dim=dk, linear_head_dim=dv)
    views = tuple(RequestView(str(i), n, 0, False, "prefill", draft_prepare=False)
                  for i, n in enumerate(lengths))
    return linear(
        prefix="core", plan=plan, views=views, stage=0, deps="input", model=model,
        parallel=ParallelSpec(), quant=model.quant, mode="prefill", padded_tokens=sum(lengths),
        policy=AttentionPolicy(recurrent_chunk=64),
        runtime=Software.from_mapping({"kernels": {"lowerings": {"kda": selection}}}) if selection else None,
    ).ops
@pytest.mark.parametrize("kind,key_products", (("kda", 2), ("gdn", 1)))
@pytest.mark.parametrize("lengths", ((1,), (64,), (65,), (65, 127)))
def test_prepare_and_finalize_place_ragged_products(kind, key_products, lengths):
    by_recipe = {op.recipe: op for op in recurrent_ops(kind, lengths)}
    for index, tokens in enumerate(lengths):
        pairs = (tokens // 64) * 64**2 + (tokens % 64)**2
        prepare, final, scan, solve = (by_recipe[f"{name}{index}"] for name in (
            "chunk_products", "recurrent_output", "recurrent_scan", "triangular_solve"))
        assert prepare.flops == 2 * pairs * 3 * 16 * key_products
        assert final.flops == 2 * 3 * (tokens * 16 * 32 + pairs * (32 + 16 * (kind == "gdn")))
        assert prepare.op_id in solve.deps and scan.op_id in final.deps
def params(lengths=(64,), heads=2, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
        "gate_dtype": "fp32", "raw_g_dtype": "bf16", "beta_dtype": "fp32",
        "compact": False,
    }
    value.update(overrides)
    return value
def execute(value, case, *, trace_limit=0):
    op = kda_prepare.make_op(value)
    parsed = parse_case(case)
    book = RecipeBook(recipes={op.recipe: PlannedRecipe(kda_prepare.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=book,
                          trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution.engine

def oracle(lengths, heads):
    """Independent source interval arithmetic for GATE_PREPARE."""
    lengths = tuple(lengths)
    headchunks = sum(lengths) // CHUNK * heads
    rows = sum(lengths) * heads
    chunks = sum(lengths) // CHUNK
    gate_read = rows * 128 * 2 + len(lengths) * heads * (4 + 128 * 4 + 2 * 8)
    gate_write = rows * 128 * 4
    gate_vector = rows * 10 * 128 + chunks * heads * 128 + 2 * len(lengths) * heads
    gate_special = rows * 2 * 128 + len(lengths) * heads
    aic = headchunks * (57_344 + 24_576)
    aiv = gate_read + headchunks * (42_112 + 66_688 + 16_384)
    work = {
        "bf16": headchunks * SCORE_HEAD,
        "fp32": headchunks * FP32_HEAD,
        "mte2_aic": aic,
        "mte2_aiv": aiv,
        "mte2": aic + aiv,
        "mte3": gate_write + headchunks * (2 * 73_728 + 24_576 + 8_192),
        "fixpipe": headchunks * (24_576 + 12_288),
        "mte1": headchunks * (57_344 + 24_576),
        "vector": gate_vector + headchunks * (DIAG_HEAD + 64 * 64 * 7
            + 64 * 128 * 41 + 32 * 128 * 7 + 64 + 4096),
        "reg_special_lanes": gate_special + headchunks * (64 * 128 * 4 + 32 * 128),
    }
    work["hbm"] = (aic + aiv + work["mte3"]
                   + headchunks * 12_288)
    return work, headchunks

def plan(value, cores=2, *, split=False):
    op = kda_prepare.make_op(value)
    units = {"vector": 2 * cores}
    if split:
        units.update(mte2_aic=cores, mte2_aiv=2 * cores)
    return kda_prepare.build(op, Geometry(cores, units, {})), op
@pytest.mark.parametrize("lengths,heads", (((64,), 2), ((8192,), 12), ((64, 128), 2)))
def test_source_interval_work_is_conserved(lengths, heads):
    expected, headchunks = oracle(lengths, heads)
    for split in (False, True):
        built, op = plan(params(lengths, heads), 2, split=split)
        meta = dict(op.metadata)
        assert meta["headchunks"] == headchunks and meta["gate_tasks"] == len(lengths) * heads
        assert meta["small_fp32_gemms_per_head"] == 6
        assert built.work.matrix == {dtype: expected[dtype] for dtype in ("bf16", "fp32")}
        copy = ("mte2_aic", "mte2_aiv") if split else ("mte2",)
        for name in (*copy, "mte3", "fixpipe", "mte1", "vector", "reg_special_lanes", "hbm"):
            assert built.work.resources[name] == expected[name]
@pytest.mark.parametrize("lengths,heads,cores,expected_counts", [
    ((64,), 2, 2, (1,)),
    ((8192,), 12, 32, (24,) * 32),
    ((4096,), 4, 32, (4,) * 32),
])
def test_finite_paired_task_queue_and_stride(lengths, heads, cores, expected_counts):
    value = params(lengths=lengths, heads=heads)
    parsed = kda_prepare.parse_params(value)
    ownership = kda_prepare.task_ownership(parsed, cores)
    assert tuple(count for _core, count in ownership) == expected_counts
    assert sum(expected_counts) == parsed.paired_tasks
    assert parsed.metadata(cores)["paired_tasks_per_core"] == expected_counts

def test_selected_kda_prepare_replaces_local_products_and_rejects_partial_chunks():
    ops = recurrent_ops("kda", (64, 128), KDA_PREPARE)
    prepare, = [op for op in ops if op.recipe == kda_prepare.RECIPE]
    assert dict(prepare.metadata)["lengths"] == (64, 128)
    assert not any(op.recipe.startswith("chunk_products") for op in ops)
    assert any(prepare.op_id in op.deps for op in ops)
    with pytest.raises(ModelingError, match="partial_chunk_tail"):
        recurrent_ops("kda", (65,), KDA_PREPARE)

def local_case(cores=2, *, memory=1e30):
    case = kda_hardware_case(units=cores, split_copy=True)
    case["machine"]["device"]["rates"].update({
        "memory": memory, "mte2_aic": 1e12, "mte2_aiv": 1e12,
        "mte1": 4e12, "mte3": 1e12, "vector": 3e12,
        "reg_special_lanes": 8e11, "fixpipe": 2e12,
    })
    case["machine"]["device"]["resource_units"]["mte3"] = 2 * cores
    return case
@pytest.mark.parametrize("lengths,heads,cores", [
    ((64,), 2, 2), ((128,), 6, 2), ((64, 128, 64), 2, 2), ((128,), 66, 32),
])
def test_compact_work_and_noncontended_owner_schedule_match_expanded(lengths, heads, cores):
    case = local_case(cores)
    expanded = execute(params(lengths, heads, compact=False), case)
    compact = execute(params(lengths, heads, compact=True), case)
    assert compact.now == pytest.approx(expanded.now, rel=1e-10)
    assert compact.serviced == pytest.approx(expanded.serviced, rel=1e-10, abs=1e-12)
    assert compact.summary()["live_tasks"] == 0
    assert compact.summary()["live_programs"] == 0
    assert compact.completed_count < expanded.completed_count / 20

def test_gate_barrier_ready_and_empty_owners_survive_compaction():
    op = kda_prepare.make_op(params((64, 128), 2, compact=True))
    report = verify_operator(local_case(8), op, kda_prepare.build, input_ready_ms=7)
    assert report["status"] == "passed"
    engine = execute(params((64, 128), 2, compact=True),
                     local_case(8), trace_limit=300)
    gate = [row for row in engine.trace if ":gate:program:group" in row["task"]]
    prepare = [row for row in engine.trace if ":prepare:program:group" in row["task"]]
    assert len({row["task"].split(":gate:program:")[1].split(":")[0]
                for row in gate}) == 4
    assert len({row["task"].split(":prepare:program:")[1].split(":")[0]
                for row in prepare}) == 3
    assert min(row["start_ms"] for row in prepare) >= max(row["end_ms"] for row in gate)
    assert engine.summary()["live_tasks"] == 0

def _wy_op(**metadata):
    return OpSpec("wy", "gdn_prefill", "prefill", 0, 0, ("r",), (), (), (), (), recipe="gdn_prefill", dtype="bf16",
                  metadata=tuple(dict({"lengths": (64,), "dk": 64, "dv": 128}, **metadata).items()))
@pytest.mark.parametrize("sku", SKUS)
def test_wy_dispatch_conserves_contractions_and_launch_boundaries(sku):
    geometry, ctx = sku_planning_context(sku, "wy")
    op = _wy_op()
    resolved = resolve_operator_plan(op, ctx)
    plan = resolved.recipe
    assert build(op, geometry).work == plan.work
    phases = {phase.phase_id: phase for phase in plan.pipeline_spec.phases}
    if sku != NEWEST_SKU:
        assert plan.metadata["launch_boundaries"] == ("ChunkGatedDeltaRule::Process",)
        assert plan.metadata["launch_count"] == 1
        assert plan.metadata["sync_all_count"] == 4
        assert tuple(name for name in phases if name.startswith("SyncAll")) == (
            "SyncAll.entry", "SyncAll.prepare", "SyncAll.state", "SyncAll.output")
        return
    score, mixed = 2 * 64**2 * 128, 2 * 64 * 128**2
    contractions = {
        "KKT_QKT": 2 * score, "offdiagonal_inverse": 2 * 2 * 32**3,
        "Kcd_U": 2 * score, "R_O_inter": 2 * mixed,
        "delta_L0C_to_UB": mixed, "O_L0C_to_GM": score,
    }
    assert {name: sum(w.amount for w in phase.work if w.resource == "cube")
            for name, phase in phases.items() if any(w.resource == "cube" for w in phase.work)} == contractions
    assert plan.metadata["offdiag_inv_flops"] == 131_072
    assert tuple(name for name in phases if name in contractions or name == "S_next") == (
        *contractions, "S_next")
    assert phases["delta_L0C_to_UB"].after == ("R32_to_R16_UB_to_L1",)
    assert phases["S_next"].after == ("delta_L0C_to_UB",)
@pytest.mark.parametrize("lengths,dim,streamed,chunked,cube", (
    ((1,), 64, 1, 0, 0), ((65,), 128, 0, 1, 23_330_816), ((65, 1), 64, 1, 1, 23_330_816),
))
def test_gdn_dispatches_each_varlen_sequence_and_issues_padded_bf16(lengths, dim, streamed, chunked, cube):
    geom, _ = sku_planning_context(NEWEST_SKU, "wy")
    plan = build(_wy_op(lengths=lengths, dk=dim, dv=dim), geom)
    assert sum(plan.work.matrix.values()) == cube and set(plan.work.matrix) <= {"bf16"}
    assert (plan.metadata["streaming_exact_sequences"], plan.metadata["chunked_wy_sequences"]) == (streamed, chunked)
    assert plan.metadata["launch_count"] == 1 and plan.work.resources["vector"] > 0

def test_wy_chunk_width_requires_pinned_source_abi():
    from mindie_sim.public.errors import UnmodeledCapabilityError

    _, ctx = sku_planning_context(NEWEST_SKU, "wy")
    with pytest.raises(ModelingError, match="requires_pinned_abi"):
        resolve_operator_plan(_wy_op(bt=32), ctx)
    with pytest.raises(UnmodeledCapabilityError, match="gdn_chunk32_domain_unsupported"):
        resolve_operator_plan(_wy_op(bt=32, abi="gdn_chunk32", dk=64), ctx)
    plan = resolve_operator_plan(_wy_op(bt=32, abi="gdn_chunk32", dk=128), ctx).recipe
    assert plan.metadata["offdiag_inv_flops"] == compute_wy_chunk_work(32, 128, 128, 32).offdiag_inv_flops == 0
    assert plan.metadata["launch_count"] == len(plan.metadata["launch_boundaries"]) == 3
    assert compute_wy_chunk_work(64, 64, 128).offdiag_inv_flops == 2 * 2 * 32**3
    for leaf in ("prepare", "state", "output"):
        abi = f"gdn_{leaf}_arch35_chunk32"
        for hv in (1, 2, 3, 4):
            result = resolve_operator_plan(_wy_op(bt=32, abi=abi, dk=128, hv=hv, hk=1), ctx)
            assert result.recipe.metadata["launch_count"] == 1
            assert result.recipe.metadata["launch_boundaries"] == (abi,)
            assert [ref.to_dict() for ref in result.sources] == [PR22147_SOURCE]
        for field in ("key_heads", "hk", "num_k_heads", "q_heads"):
            with pytest.raises(ModelingError, match="unsupported_value_to_key_head_ratio"):
                resolve_operator_plan(_wy_op(bt=32, abi=abi, dk=128, heads=8, **{field: 1}), ctx)
    output = _wy_op(abi="gdn_output_arch35_chunk32", lengths=(32,), heads=1, key_heads=1, dk=128)
    assert resolve_operator_plan(output, ctx).recipe.work.matrix == {"bf16": 1_572_864}
@pytest.mark.parametrize("metadata", (
    {"required_routes": "l0c->ub"}, {"required_routes": (("ub", "l1"),)},
    {"abi": "gdn_chunk32"}, {"abi": "flash_kda_persistent"},
))
def test_staged_isa_rejects_direct_routes_and_foreign_abis(metadata):
    _, ctx = sku_planning_context(OLDER_SKU, "wy")
    op = _wy_op(**metadata)
    op = replace(op, family="kda_prefill", recipe="kda_prefill") if metadata.get("abi") == "flash_kda_persistent" else op
    with pytest.raises(ModelingError, match="unavailable"):
        resolve_operator_plan(op, ctx)
def test_direct_isa_rejects_synthetic_datapath_without_routes():
    _, ctx = sku_planning_context(NEWEST_SKU, "wy"); _, staged = sku_planning_context(OLDER_SKU, "wy")
    with pytest.raises(ModelingError, match="L0C to UB and UB to L1"):
        resolve_operator_plan(_wy_op(), replace(ctx, datapath=replace(ctx.datapath, usable_paths=staged.datapath.usable_paths)))
def test_chunk32_requires_exact_primitive_for_each_recurrence():
    from _support.hardware import sku_case
    from mindie_sim.primitives import UncertaintyBand, load_primitives
    sku = "Ascend950DT_9582"
    parsed = sku_case(sku)
    shipped = load_primitives(sku)
    pid, metric = "P13_crosscore_sync", "mode4_intra_block_ns"
    cells = dict(shipped.primitives)
    def band(ns):
        return UncertaintyBand(0, ns, ns, ns, ns, cv_pct=0, n_samples=1, bound_kind="latency_lower")
    mode2 = replace(cells[pid], status="calibrated", evidence_id="capture://test/sync/" + "0" * 64,
                    metrics={"mode2_paired_ns": band(424)})
    flag = cells["P12_intracore_flag"]
    def run(table, tokens):
        op = STATE.make_op({"lengths": (tokens,), "heads": 1, "key_heads": 1})
        book = RecipeBook(recipes={op.recipe: PlannedRecipe(STATE.build)})
        execution = Execution(parsed, CostModel(parsed), recipes=book, trace_limit=1024)
        execution.stage_contexts[0, 0] = replace(execution.stage_contexts[0, 0], primitive_table=table)
        work = IterationWork((op,), IterationPlan(()), 1, tokens, tokens, None, "prefill", "test")
        done = execution.add(work, 0, 0).completion
        execution.engine.advance(until=(done,))
        return done.time, execution.engine.trace
    op = STATE.make_op({"lengths": (512,), "heads": 1, "key_heads": 1})
    with pytest.raises(ModelingError, match=pid):
        STATE.build(op, Geometry(1, {"vector": 2})).program(RecipeContext(op, {}, {}, {}, 1))
    for table in (None, replace(shipped, primitives={**cells, pid: replace(cells[pid], status="unmeasured", metrics={})}), replace(shipped, primitives={**cells, pid: mode2}),
                  replace(shipped, primitives={**cells, pid: replace(mode2, metrics={
                      **mode2.metrics, metric: UncertaintyBand(140, bound_kind="latency_lower")})})):
        with pytest.raises(ModelingError, match=pid):
            run(table, 512)
    cells[pid] = replace(mode2, metrics={**mode2.metrics, metric: band(140)})
    duration, trace = run(replace(shipped, primitives=cells), 512)
    waits = [row["end_ms"] - row["start_ms"] for row in trace if row["task"].endswith(":recurrence_sync")]
    assert waits == pytest.approx([140e-6] * 16)
    cells["P12_intracore_flag"] = replace(flag, metrics={**flag.metrics, "flag_ns": band((flag.band("flag_ns").p50 or 0.0) + 100)})
    assert run(replace(shipped, primitives=cells), 512)[0] - duration == pytest.approx(100e-6)

def test_replay_lowers_mixed_gdn_and_preserves_ragged_attention_per_replica(monkeypatch):
    from mindie_sim.prediction.results import LoweringRequest
    from mindie_sim.prediction.spine import PredictionSpine
    raw = compose_case("qwen3.5-35b-a3b", NEWEST_SKU, tp=1, dp=2, ep=2, layers=4)
    mixed = (RequestView("prefill", 3, 7, False, "prefill"),
             RequestView("decode", 1, 512, True, "decode"),
             RequestView("pad", 1, 0, False, "decode", padding=True))
    small = (RequestView("other", 1, 31, True, "decode"),)
    iterations = ((0, IterationPlan(small)), (1, IterationPlan(mixed)))
    spine = PredictionSpine()
    prepared = spine.lower(parse_case(raw), LoweringRequest(iterations=iterations))
    seen = {}
    prepare = RecipeBook.prepare

    def record(book, op, geometry):
        if op.family == "attention":
            seen[geometry.replica] = (geometry.requests, geometry.padded_tokens)
        return prepare(book, op, geometry)
    monkeypatch.setattr(RecipeBook, "prepare", record)
    result = spine.run_step(prepared)
    assert result.span_ms > 0 and any(row.family == "gdn_prepare" for row in result.operators)
    assert seen == {0: (small, 1), 1: (mixed, 5)}
