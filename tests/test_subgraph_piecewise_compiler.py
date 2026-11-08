"""Mathematical and behavioural invariants of the three-level piecewise compiler.
The tests below establish independent arithmetic identities and structural
counterexamples from blueprint section 9.  They do not inspect class names,
fixed strings or documentation prose.  All values come from declared shapes,
tiles, state and resource supplies; no measurement enters a prediction branch.
[Role]
  Exercise bounded geometry, symbolic definedness, guard partition/crossover,
  domain enforcement and compiled subgraph state transitions.
[Invariants]
  Equivalent input orders select equivalent physical work; simplification
  preserves definedness; distinct roots stay distinct; declared analysis
  bounds are enforced.
[Timeouts & Locks]
  Pure CPU arithmetic; no waits, sleeps or locks.
[Side-Effect Policy]
  Read-only test code; no filesystem, network or device mutation.
"""
from __future__ import annotations
import math
from dataclasses import replace
from fractions import Fraction
import pytest
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.piecewise import (
    Add, AnalysisDomain, CeilDiv, Const, CoreWaveGuard, Crossover, Div,
    FloorDiv, GraphBucketGuard, ImplementationPolicy, Interval, Max, Min, Mul,
    OperatorPiecewiseCompiler, PipeDemand, PipeDominanceGuard, PipeEnvelope,
    PipeSupply, RequestGeometry, ServiceMode, SplitKGuard, SplitKPolicy,
    StepGuard, StructuralPartition, SubgraphCostFunction, Var, WorkloadGeometry,
    boundary_distance, compose_sequence, lower_fusion, pipeline_makespan,
    quant_ledger, run_engine_work,
)
from mindie_sim.prediction.piecewise.composition import ClosedForm
from mindie_sim.prediction.piecewise.compiler import OperatorCost
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, TemplateRegistry, instantiate_model,
)
from mindie_sim.prediction.subgraphs.registry import (
    KIND_BUILDERS, VOCABULARY, validate_combination,
)
from mindie_sim.prediction.subgraphs.state import (
    AcceptanceState, CompressionState, MoeRoutingState, ResidualSlotState,
    SharedKvContract, StateView,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import SUBGRAPH_KINDS
SUPPLY = PipeSupply((
    ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 2_000_000.0), ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
    ("fixpipe", 2_000_000.0), ("link", 1_000_000.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)
FEATURES = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                split_target=1000, max_splits=4)

def make_model(*, name="toy", plans=None, **overrides):
    base = dict(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16,
                intermediate_size=128, vocab_size=128)
    base.update(overrides)
    if plans is None:
        plans = [dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                      num_kv_heads=2, head_dim=16, intermediate_size=128,
                      qk_norm=True)]
    raw = dict(base, name=name, layers=len(plans), layer_plans=plans)
    return ModelSpec.from_mapping(raw)

def features(model, **overrides):
    values = dict(FEATURES)
    values.update(overrides)
    return FeatureConfig.from_model(model, **values)

def compile_kind(model, kind, *, layer=0, role=None, feature_overrides=None,
                 supply=SUPPLY, policy=POLICY):
    feature = features(model, **(feature_overrides or {}))
    instantiated = instantiate_model(model, feature)
    instance = (instantiated.find(kind, role=role) if layer is None
                else instantiated.find(kind, layer=layer, role=role))
    return OperatorPiecewiseCompiler(supply, policy).compile(instance)

def attention_cost(compiled, geometry, state=None):
    evaluation = compiled.evaluate(geometry, state)
    return next(cost for cost in evaluation.operators if cost.name == "attention")
# 1. Ragged attention pairs are enumerated, not approximated by B*mean(q)*mean(s).

def test_ragged_attention_matches_explicit_pair_enumeration():
    model = make_model()
    compiled = compile_kind(model, "A0")
    geometry = WorkloadGeometry((RequestGeometry(3, 5), RequestGeometry(1, 9)))
    cost = attention_cost(compiled, geometry)
    brute = 0
    for request in geometry.requests:
        for position in range(request.q_len):
            brute += request.context + position + 1
    assert brute == 31
    assert cost.get("causal_pairs") == brute
    assert cost.get("query_rows") == 4 * (3 + 1)
    # Same total tokens and total context, different ragged allocation.
    left = WorkloadGeometry((RequestGeometry(2, 4), RequestGeometry(2, 4)))
    right = WorkloadGeometry((RequestGeometry(1, 5), RequestGeometry(3, 3)))
    assert left.tokens == right.tokens == 4
    assert left.s_sum == right.s_sum == 12
    left_cost = attention_cost(compiled, left)
    right_cost = attention_cost(compiled, right)
    assert left_cost.get("causal_pairs") == 22
    assert right_cost.get("causal_pairs") == 21
    assert left_cost.get("causal_pairs") != right_cost.get("causal_pairs")
# 2. Tiling, tail rows and core waves at k*M_t-1, k*M_t, k*M_t+1.

def test_tiling_tails_and_core_waves_do_not_borrow_idle_cores():
    model = make_model(plans=[dict(index=0, attention="gqa", ffn="dense",
                                   num_heads=4, num_kv_heads=2, head_dim=16,
                                   intermediate_size=128)])
    compiled = compile_kind(model, "F0", feature_overrides={"kv_tile": 128})
    gate_up = compiled.operator("gate_up")
    seen = []
    for tokens, expected_tiles, expected_tail in ((31, 2, 15), (32, 2, 16),
                                                   (33, 3, 1)):
        cost = gate_up.evaluate(WorkloadGeometry.homogeneous(1, tokens, tokens + 1))
        regime = dict(cost.regime)
        assert regime["M_TILES"] == expected_tiles
        assert cost.get("tail_rows") == expected_tail
        assert cost.get("active_cores") == min(2, cost.get("j_tiles"))
        seen.append((regime["M_TILES"], cost.get("waves"), cost.get("tail_rows")))
    assert seen[0][0] == seen[1][0] == 2 and seen[2][0] == 3
    wave = CoreWaveGuard(4)
    assert wave.waves(1) == 1 and wave.active_cores(1) == 1
    assert wave.tail_cores(5) == 1
    assert wave.tail_cores(8) == 4
    audit = wave.verify(8)
    assert audit.unique and audit.checked == 8
    with pytest.raises(ModelingError):
        wave.waves(-1)
# 3. Symbolic partitions have one hit per lattice point and explicit ownership.

def test_symbolic_partition_is_unique_with_explicit_boundaries_and_misses():
    guard = StepGuard("query_tile", "qg", 4)
    audit = guard.verify(9)
    assert audit.unique
    assert audit.to_dict()["matches"] == {1: 4, 2: 4, 3: 1}
    assert guard.cell(1).contains(4) and not guard.cell(1).contains(5)
    assert guard.cell(2).contains(5)
    assert guard.boundaries(2) == (7, 8, 9)
    with pytest.raises(ModelingError):
        guard.select(0)
    with pytest.raises(ModelingError):
        guard.select(-3)
    bucket = GraphBucketGuard((1, 2, 4, 8))
    assert [bucket.select(value) for value in (1, 2, 3, 5, 8)] == [1, 2, 4, 8, 8]
    assert bucket.select_index(5) == 3 and bucket.pad(5) == 3
    assert bucket.verify().unique
    with pytest.raises(ModelingError):
        bucket.select(9)
    with pytest.raises(ModelingError):
        GraphBucketGuard((4, 2))
    partition = StructuralPartition((guard, StepGuard("kv", "s", 4)))
    report = partition.verify({"qg": (1, 9), "s": (1, 9)})
    assert report.unique and report.checked == 81
    with pytest.raises(ModelingError):
        partition.verify({"qg": (1, 9)})
    with pytest.raises(ModelingError):
        partition.verify({"qg": (1, 9), "s": (1, 9)}, limit=10)
# 4. Continuous pipe crossover equals at the root and has no fabricated jump.

def test_pipe_crossover_equality_and_dominance_flip():
    guard = PipeDominanceGuard((
        PipeDemand("cube", Var("T") * Const(4), 10),
        PipeDemand("mte2", Var("T") + Const(20), 5),
    ))
    roots = guard.crossover("cube", "mte2", "T", lower=0, upper=100)
    assert len(roots) == 1
    root = roots[0]
    assert root.exact and root.degree == 1 and root.value == 20
    assert guard.times({"T": 20})["cube"] == guard.times({"T": 20})["mte2"]
    assert guard.dominant({"T": 19}) == "mte2"
    assert guard.dominant({"T": 21}) == "cube"
    before = guard.envelope({"T": Fraction(19999, 1000)})
    after = guard.envelope({"T": Fraction(20001, 1000)})
    assert abs(float(before - after)) < 0.01
    never = PipeDominanceGuard((
        PipeDemand("cube", Div(Var("T"), Const(100)), 1),
        PipeDemand("mte2", Var("T") + Const(20), 1),
    ))
    assert never.crossover("cube", "mte2", "T", lower=0) == ()
    bilinear = PipeDominanceGuard((
        PipeDemand("cube", Var("T") * Var("S"), 1),
        PipeDemand("mte2", Var("T") * Const(10) * Var("T"), 1),
    ))
    roots = bilinear.crossover("cube", "mte2", "S", at={"T": 2})
    assert len(roots) == 1 and roots[0].value == 20
    with pytest.raises(ModelingError):
        bilinear.crossover("cube", "mte2", "S")
    tie = PipeDominanceGuard((
        PipeDemand("cube", Const(5), 1),
        PipeDemand("vector", Var("T"), 1),
    ))
    assert tie.dominant({"T": 5}) == "cube"
    assert set(tie.tied({"T": 5})) == {"cube", "vector"}
# 5. Pipeline fill/drain and the single-tile serial sum.

def test_pipeline_single_tile_sum_and_bounded_slots():
    assert pipeline_makespan([1, 2], tiles=1, slots=2) == 3
    assert pipeline_makespan([1, 2], tiles=3, slots=1) == 9
    assert pipeline_makespan([1, 2], tiles=3, slots=3) == 7
    assert pipeline_makespan([1, 2, 3], tiles=2, slots=2) == 9
    assert pipeline_makespan([1, 2, 3], tiles=1, slots=1) == 6
    with pytest.raises(ModelingError):
        pipeline_makespan([], 1, 1)
    with pytest.raises(ModelingError):
        pipeline_makespan([1], 0, 1)
# 6. Serial composition equals the graph and a late input adds a real gap.

def test_serial_composition_equals_engine_and_counts_late_input_gap():
    model = make_model()
    compiled = compile_kind(model, "A0")
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    function = SubgraphCostFunction(compiled)
    serial = function.evaluate_serial(geometry)
    assert serial.total_ms == sum(cost.duration_ms for cost in serial.costs)
    engine = run_engine_work(function.to_engine_work(geometry), SUPPLY)
    assert float(engine.makespan_ms) == pytest.approx(float(serial.total_ms), rel=1e-12)
    assert len(function.to_iteration_work(geometry)) == len(serial.costs)
    late = function.evaluate_serial(geometry, readiness={"qkv_projection": 2})
    assert late.gap_ms == 2
    assert float(late.total_ms) == pytest.approx(float(serial.total_ms) + 2)
    early = function.evaluate_serial(geometry, readiness={"qkv_projection": 0})
    assert early.gap_ms == 0
    assert early.total_ms == serial.total_ms
# 7. Shared-HBM contention rejects the isolated max and keeps one capacity.

def test_hbm_contention_total_bytes_not_isolated_max():
    shared = run_engine_work(
        {"A": [[("hbm", 1000)]], "B": [[("hbm", 1000)]]},
        PipeSupply((("hbm", 1000),)))
    assert shared.makespan_ms == 2
    assert dict(shared.resources)["hbm"] == 2
    isolated = run_engine_work(
        {"A": [[("hbm", 1000)]], "B": [[("hbm", 1000)]]},
        PipeSupply((("hbm", 1000), ("cube", 1_000_000))))
    assert isolated.makespan_ms == 2
    disjoint = run_engine_work(
        {"A": [[("hbm", 1000)]], "B": [[("cube", 1000)]]},
        PipeSupply((("hbm", 1000), ("cube", 1000))))
    assert disjoint.makespan_ms == 1
    model = make_model()
    compiled = compile_kind(model, "A0")
    function = SubgraphCostFunction(compiled)
    result = function.evaluate_fork_join_stages(
        {"A": [[("hbm", 1_000_000)]], "B": [[("hbm", 1_000_000)]]})
    assert result.isolated_max_ms == 1
    assert result.contention_ms == 2
    assert result.rejected_isolated_max
    assert result.total_ms == 2
    # Alias modes on one physical pipe add; distinct pipes may overlap.
    alias = PipeEnvelope((
        ServiceMode("copy", "hbm", Const(10), 10),
        ServiceMode("scale", "hbm", Const(10), 10),
    ))
    assert alias.duration({}) == 2
    assert alias.isolated_duration({}) == 1
    separate = PipeEnvelope((
        ServiceMode("copy", "hbm", Const(10), 10),
        ServiceMode("compute", "cube", Const(10), 10),
    ))
    assert separate.duration({}) == 1
# 8. Native fusion removes only internal materialisation and one startup.

def test_native_fusion_keeps_external_outputs_and_removes_internal_materialisation():
    model = make_model()
    compiled = compile_kind(model, "A0")
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    costs = compiled.evaluate(geometry).operators
    result = lower_fusion(compiled.instance, costs, "qkv_norm_rope",
                          supply=SUPPLY)
    assert result.startup_count == 1
    assert result.startup_count < len(result.semantic)
    assert "qkv_activation" in dict(result.removed)
    assert result.removed_bytes > 0
    kept = dict(result.kept)
    assert "qkv_activation" not in kept
    contract = next(item for item in compiled.instance.template.fusions
                    if item.name == "qkv_norm_rope")
    assert set(contract.keeps) <= set(kept)
    external = ({"query", "key", "value"} if "query" in kept
                else {"qk_norm_output", "rotary_output"})
    assert external <= set(kept)
    fused_serial = SubgraphCostFunction(compiled).evaluate_serial(
        geometry, fusion=("qkv_norm_rope",))
    assert any("attention_output" in dict(cost.materialization)
               for cost in fused_serial.costs)
    assert result.fused.duration_ms <= result.isolated_ms
    assert result.fused.fidelity == "fused_physical_kernel"
    # A declared stream boundary makes the lowering illegal.
    changed = tuple(
        replace(operator, template=replace(operator.template, stream="aux"))
        if operator.name == "qk_norm" else operator
        for operator in compiled.instance.operators)
    fake = replace(compiled.instance, operators=changed)
    with pytest.raises(ModelingError):
        lower_fusion(fake, costs, "qkv_norm_rope", supply=SUPPLY)
# 9. SharedKV owner/consumer ordering and per-consumer read accounting.

def test_shared_kv_owner_stores_once_and_consumer_reads_each_time():
    plans = [
        dict(index=0, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, owns_long_kv=True),
        dict(index=1, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, kv_owner_layer_id=0),
    ]
    model = make_model(plans=plans)
    feature = features(model, pp=2)
    instantiated = instantiate_model(model, feature)
    compiler = OperatorPiecewiseCompiler(SUPPLY, POLICY)
    store = compiler.compile(instantiated.find("K0", layer=0, role="kv"))
    read = compiler.compile(instantiated.find("K0", layer=1, role="kv"))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    store_cost = next(cost for cost in store.evaluate(geometry).operators
                      if cost.name == "kv_store")
    row_bytes = int(store.instance.operators[0].theta.require("row_bytes"))
    assert store_cost.get("kv_bytes") == 8 * row_bytes
    empty = StateView()
    with pytest.raises(ModelingError):
        read.evaluate(geometry, empty)
    produced = store.evaluate(geometry, empty)
    first = next(cost for cost in read.evaluate(geometry, produced.state_out).operators
                 if cost.name == "kv_read")
    second = next(cost for cost in read.evaluate(geometry, produced.state_out).operators
                  if cost.name == "kv_read")
    # Shared consumer K0 is a readiness port; A0 owns the physical KV read.
    assert (first.get("kv_bytes"), second.get("kv_bytes")) == (0, 0)
    assert first.pipe_work("mte2") == second.pipe_work("mte2") == 0
    contract = SharedKvContract(0, (1,), row_bytes, cross_stage=True,
                                transfer_bytes=row_bytes * 8)
    assert contract.store_bytes(8) == 8 * row_bytes
    assert contract.read_bytes(64) == 64 * row_bytes
    assert contract.transfer() == 8 * row_bytes
    contract.check_order(produced.state_out)
    with pytest.raises(ModelingError):
        contract.check_order(empty)
    assert any(cost.name == "stage_transfer" for cost in store.evaluate(geometry).operators)
# 10. K1 chunked compression telescopes without per-chunk ceil.

def test_k1_compression_telescopes_to_final_floor():
    state = CompressionState.start(4)
    chunks = [3, 5, 2, 7]
    per_chunk = []
    for query in chunks:
        per_chunk.append(state.new_rows(query))
        state = state.advance(query)
    assert per_chunk == [0, 2, 0, 2]
    assert sum(per_chunk) == 4 == 17 // 4
    assert sum((query + 3) // 4 for query in chunks) == 6  # independent ceil is wrong
    assert state.remainder == 1 and state.rows == 4
    assert CompressionState.telescoped_new_rows(4, 0, 17) == 4
    plans = [dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                  num_kv_heads=2, head_dim=16, intermediate_size=128,
                  compress_ratio=4, compressor_kind="weighted_pool",
                  owns_long_kv=True)]
    model = make_model(plans=plans)
    compiled = compile_kind(model, "K1")
    current = StateView()
    total = 0
    context = 0
    for query, expected in zip(chunks, per_chunk):
        geometry = WorkloadGeometry.homogeneous(1, query, context + query)
        evaluation = compiled.evaluate(geometry, current)
        compress = next(cost for cost in evaluation.operators
                        if cost.name == "compress")
        assert compress.get("new_rows") == expected
        total += int(compress.get("new_rows"))
        current = evaluation.state_out
        context += query
    assert total == 4
    assert current.compression == ((0, CompressionState(4, 17, 4, 1)),)
    assert current.compression_requests == (
        ((0, 0), CompressionState(4, 17, 4, 1)),)
    paired = compiled.evaluate(
        WorkloadGeometry((RequestGeometry(1, 3), RequestGeometry(1, 3))),
        StateView())
    assert (paired.operators[0].get("new_rows"),
            paired.operators[0].get("tail_rows")) == (2, 0)
    assert paired.state_out.compression_requests == (
        ((0, 0), CompressionState(4, 4, 1, 0)),
        ((0, 1), CompressionState(4, 4, 1, 0)))
# 11. MoE conservation, active experts, skew waves and zero-expert weight reads.

def test_moe_conservation_skew_waves_and_zero_expert_weight_reads():
    uniform = MoeRoutingState((2, 2, 2, 2), 2, 4)
    skew = MoeRoutingState((0, 0, 0, 8), 2, 4)
    assert sum(uniform.counts) == uniform.top_k * uniform.valid_tokens == 8
    assert uniform.active_experts == 4 and skew.active_experts == 1
    assert uniform.waves(4, 2) == 2 and skew.waves(4, 2) == 1
    assert abs(MoeRoutingState.expected_active_experts(4, 2, 4) - 3.75) < 1e-12
    assert skew.padded_rows(4) == 8
    with pytest.raises(ModelingError):
        MoeRoutingState((1, 1, 1, 1), 2, 4)
    plans = [dict(index=0, attention="gqa", ffn="moe", num_heads=4,
                  num_kv_heads=2, head_dim=16, intermediate_size=128,
                  num_experts=4, top_k=2, moe_intermediate_size=32,
                  num_shared_experts=1)]
    model = make_model(plans=plans, num_experts=4, top_k=2,
                       moe_intermediate_size=32, num_shared_experts=1)
    compiled = compile_kind(model, "F3")
    geometry = WorkloadGeometry.homogeneous(4, 1, 16)
    values = []
    for counts in ((2, 2, 2, 2), (0, 0, 0, 8)):
        state = StateView(routing=((0, MoeRoutingState(counts, 2, 4)),))
        evaluation = compiled.evaluate(geometry, state)
        gate = next(cost for cost in evaluation.operators
                    if cost.name == "expert_gate_up")
        values.append((gate.get("moe_rows"), gate.get("active_experts"),
                       gate.get("tiles"), gate.get("waves"),
                       gate.get("weight_bytes")))
    balanced, concentrated = values
    assert balanced[0] == concentrated[0] == 8
    assert balanced[1] == 4 and concentrated[1] == 1
    assert balanced[3] != concentrated[3]
    assert concentrated[4] < balanced[4]
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        OperatorPiecewiseCompiler(
            SUPPLY, POLICY, domain=AnalysisDomain(max_moe_rows=1)).compile(
                compiled.instance).evaluate(geometry)
# 12. Quantisation payload/scale/zero-point/residency accounting per role.

def test_quantization_ledger_separates_payload_scales_zero_and_residency():
    w4a8 = QuantSpec(name="w4a8", weight_format="int4",
                     activation_format="int8", weight_block=(1, 32),
                     scale_bytes=4)
    ledger = quant_ledger(w4a8, "dense_mlp", 256, 512)
    assert ledger.weight_format == "int4"
    assert ledger.payload_bytes == 256 * 256
    assert ledger.scale_bytes == 256 * 16 * 4
    assert ledger.zero_point_bytes == 0
    assert ledger.weight_bytes == ledger.payload_bytes + ledger.scale_bytes
    assert ledger.residency_bytes == ledger.weight_bytes
    w8a8 = QuantSpec(name="w8a8", weight_format="int8",
                     activation_format="int8", expert_weight_format="int8",
                     weight_block=(128, 128), scale_bytes=4,
                     int8_weight_metadata="scale_offset")
    int8 = quant_ledger(w8a8, "routed_expert", 256, 512)
    assert int8.payload_bytes == 256 * 512
    assert int8.zero_point_bytes == 256 * 4
    assert int8.scale_bytes + int8.zero_point_bytes == (
        int8.weight_bytes - int8.payload_bytes)
    overridden = QuantSpec(
        name="mixed", weight_format="int4", activation_format="int8",
        weight_block=(1, 32), scale_bytes=4,
        category_overrides={"attn.qkv": {"weight_format": "bf16"}},
    )
    assert quant_ledger(overridden, "attn.qkv", 128, 256).weight_format == "bf16"
    assert quant_ledger(overridden, "attn.o", 128, 256).weight_format == "int4"
    bf16_model = make_model()
    quant_model = make_model(
        quant=dict(name="w4a8", weight_format="int4", activation_format="int8",
                   weight_block=(1, 32), scale_bytes=4))
    bf16_gate = compile_kind(bf16_model, "F0").operator("gate_up").evaluate(
        WorkloadGeometry.homogeneous(1, 8, 9))
    quant_gate = compile_kind(quant_model, "F0").operator("gate_up").evaluate(
        WorkloadGeometry.homogeneous(1, 8, 9))
    assert bf16_gate.get("flops") == quant_gate.get("flops")
    assert quant_gate.get("weight_total") < bf16_gate.get("weight_total")
    assert quant_gate.pipe_work("vector") > bf16_gate.pipe_work("vector")
    assert quant_gate.residency_bytes == quant_gate.get("weight_total")
# 13. MTP verify work is not discounted by acceptance; commit progress moves.

def test_mtp_verify_work_is_independent_of_acceptance():
    raw = dict(
        name="mtp", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16, intermediate_size=128)],
        auxiliary=dict(mtp=dict(kind="mtp", layers=1, hidden_size=64,
                                intermediate_size=128, num_heads=4,
                                num_kv_heads=2, head_dim=16, q_width=64,
                                kv_width=32)))
    model = ModelSpec.from_mapping(raw)
    compiled = compile_kind(model, "D0", layer=None)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    worked = []
    for accepted in (0, 4):
        state = StateView(acceptance=AcceptanceState(accepted=accepted,
                                                     committed=accepted))
        evaluation = compiled.evaluate(geometry, state)
        verify = next(cost for cost in evaluation.operators
                      if cost.name == "target_verify")
        draft = next(cost for cost in evaluation.operators
                     if cost.name == "draft_heads")
        worked.append((verify.get("verify_flops"), draft.get("draft_flops"),
                       verify.get("acceptance_discount")))
    assert worked[0] == worked[1]
    assert worked[0][2] == 0
    progress = AcceptanceState()
    assert progress.commit(0, verified=1).progress == 0
    assert progress.commit(1, verified=1).progress == 1
    with pytest.raises(ModelingError):
        progress.commit(2, verified=1)
# 14. 0-shot structural generalisation, interning and model-label independence.

def test_zero_shot_generalisation_template_interning_and_label_independence():
    assert len(SUBGRAPH_KINDS) == 22 and len(set(SUBGRAPH_KINDS)) == 22
    assert set(KIND_BUILDERS) == set(SUBGRAPH_KINDS)
    assert {entry.kind for entry in VOCABULARY} == set(SUBGRAPH_KINDS)
    first = make_model(name="alpha")
    second = make_model(name="beta")
    first_features = features(first)
    first_inst = instantiate_model(first, first_features)
    second_inst = instantiate_model(second, features(second))
    assert first_inst.performance_keys() == second_inst.performance_keys()
    geometry = WorkloadGeometry.homogeneous(3, 2, 40)
    first_cost = compile_kind(first, "A0").evaluate(geometry).total_ms
    second_cost = compile_kind(second, "A0").evaluate(geometry).total_ms
    assert first_cost == second_cost
    # Repeated identical layers intern one template object.
    layer = dict(attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
                 head_dim=16, intermediate_size=128)
    twin = make_model(plans=[dict(layer, index=0), dict(layer, index=1)])
    registry = TemplateRegistry(256)
    instantiated = instantiate_model(twin, features(twin), registry=registry)
    assert len(registry) > 0 and registry.evictions == 0
    for left, right in zip(instantiated.layers[0].subgraphs,
                           instantiated.layers[1].subgraphs):
        if left.kind == right.kind and left.variant == right.variant:
            assert left.template is right.template
            assert left.performance_key() == right.performance_key()
    changed = make_model(name="alpha", hidden_size=128)
    changed_inst = instantiate_model(changed, features(changed))
    changed_cost = compile_kind(changed, "A0").evaluate(geometry).total_ms
    assert changed_cost != first_cost
    assert changed_inst.find("A0", layer=0).fingerprint() != \
        first_inst.find("A0", layer=0).fingerprint()
    validate_combination(("A0", "K0", "F0"))
    with pytest.raises(ModelingError):
        validate_combination(("I1",))
# 15. Bounded caches, lazy segment enumeration and bounded execution history.

def test_bounded_caches_lazy_segments_and_completed_task_release():
    plans = [dict(index=index, attention="gqa", ffn="dense", num_heads=4,
                  num_kv_heads=2, head_dim=16, intermediate_size=128)
             for index in range(4)]
    model = make_model(plans=plans)
    registry = TemplateRegistry(3)
    instantiated = instantiate_model(model, features(model), registry=registry)
    assert len(registry) <= 3
    assert registry.evictions > 0
    compiled = compile_kind(model, "A0")
    qkv = compiled.operator("qkv_projection")
    table = qkv.piecewise(AnalysisDomain(max_tokens=64, max_query_tiles=8,
                                        max_kv_tiles=8, max_work=100000),
                          limit=4)
    assert len(table) <= 4 and table.truncated and table.total_cells > 4
    before = len(registry)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    first = compiled.evaluate(geometry).total_ms
    for _ in range(50):
        assert compiled.evaluate(geometry).total_ms == first
    assert len(registry) == before
    assert len(compiled.operators) == 4
    result = run_engine_work(
        {"A": [[("hbm", 10)], [("hbm", 10)]],
         "B": [[("hbm", 10)], [("hbm", 10)]]},
        PipeSupply((("hbm", 1000),)))
    summary = dict(result.summary)
    assert summary["live_tasks"] == 0 and summary["live_programs"] == 0
    assert summary["peak_live_tasks"] <= 4
    leftover = ResidualSlotState(2)
    assert leftover.push(0).push(1).push(2).retained == (1, 2)
    assert ResidualSlotState(2).full is False
    assert ClosedForm("x").certified

def test_all_twenty_two_canonical_kinds_compile_from_one_model():
    plans = [
        dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, num_experts=4, top_k=2,
             moe_intermediate_size=32, num_shared_experts=1, qk_norm=True,
             hyper_connections=True, hc_mult=2, hc_sinkhorn_iters=2,
             engram=True, residual_block_size=2, owns_long_kv=True),
        dict(index=1, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1,
             head_dim=16, q_lora_rank=16, kv_lora_rank=16,
             qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
             intermediate_size=128, indexer=True, index_heads=2, index_dim=8,
             index_topk=4, candidate_filter=True, candidate_topk_blocks=2,
             candidate_block_size=4, kv_owner_layer_id=0),
        dict(index=2, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, linear_heads=2,
             linear_key_heads=2, linear_value_heads=2, linear_key_head_dim=8,
             linear_head_dim=8, conv_kernel=4, recurrent_state_elements=64,
             conv_state_elements=16, state_bytes=256),
        dict(index=3, attention="kda", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, linear_key_heads=2,
             linear_value_heads=2, linear_key_head_dim=8, linear_head_dim=8,
             conv_kernel=4, gate_rank=4, recurrent_state_elements=64,
             conv_state_elements=16, state_bytes=256, owns_long_kv=True),
        dict(index=4, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, cache_kind="window_compressed",
             compress_ratio=4, compressor_kind="weighted_pool", indexer=True,
             index_heads=1, index_dim=8, index_topk=2, owns_long_kv=True),
    ]
    raw = dict(
        name="all-kinds", layers=len(plans), hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        num_experts=4, top_k=2, moe_intermediate_size=32,
        num_shared_experts=1, layer_plans=plans,
        mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16,
                 qk_rope_head_dim=8, v_head_dim=16, index_topk=4,
                 index_head_dim=8, index_n_heads=2, compress_ratio=4),
        vision=dict(kind="vision", layers=1, hidden_size=64,
                    intermediate_size=128, num_heads=4, qkv_width=192,
                    patch_size=14),
        engram=dict(tables=[dict(layer_id=0, table_id=0, rows=64, width=64)],
                    max_ngram=2, heads=1, width=64),
        auxiliary=dict(mtp=dict(kind="mtp", layers=1, hidden_size=64,
                                intermediate_size=128, num_heads=4,
                                num_kv_heads=2, head_dim=16, q_width=64,
                                kv_width=32)))
    model = ModelSpec.from_mapping(raw)
    feature = features(model, pp=2)
    instantiated = instantiate_model(model, feature)
    assert {instance.kind for instance in instantiated.subgraphs} == set(SUBGRAPH_KINDS)
    compiler = OperatorPiecewiseCompiler(SUPPLY, POLICY)
    state = StateView(
        kv_ready=tuple((index, "kv_cache") for index in range(len(plans))),
        index_ready=tuple((index, "index") for index in range(len(plans))),
        selection_ready=tuple((index, "selection") for index in range(len(plans))),
        routing=((0, MoeRoutingState((2, 2, 2, 2), 2, 4)),))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32, media_tokens=16)
    totals = []
    for instance in instantiated.subgraphs:
        evaluation = compiler.compile(instance).evaluate(geometry, state)
        assert evaluation.total_ms >= 0
        totals.append(instance.kind)
    assert len(totals) == len(instantiated.subgraphs)
    index_store = compiler.compile(instantiated.find("I0", layer=1)).evaluate(
        geometry, StateView())
    assert (1, "index") in index_store.state_out.index_ready
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        OperatorPiecewiseCompiler(
            SUPPLY, POLICY, domain=AnalysisDomain(max_index_rows=1)).compile(
                instantiated.find("I1", layer=1)).evaluate(geometry)
def test_intervals_cover_unbounded_min_max_and_signed_floor_ceil():
    x = Var("x")
    lower = Min((x, Const(5))).interval({"x": Interval(None, 10)})
    upper = Max((x, Const(5))).interval({"x": Interval(0, None)})
    assert (lower.lower, lower.upper) == (None, 5)
    assert (upper.lower, upper.upper) == (5, None)
    assert Min((x, Const(5))).interval({"x": Interval(0, 10)}) == Interval(0, 5)
    assert Max((x, Const(5))).interval({"x": Interval(0, 10)}) == Interval(5, 10)
    bounds = {"x": Interval(-10, 10), "d": Interval(1, 2)}
    for kind in (FloorDiv, CeilDiv):
        expr = kind(x, Var("d"))
        interval = expr.interval(bounds)
        assert (interval.lower, interval.upper) == (-10, 10)
        for numerator in (-10, 10):
            for denominator in (1, 2):
                assert interval.contains(
                    expr.evaluate({"x": numerator, "d": denominator}))
    half = {"x": Fraction(3, 2)}
    assert FloorDiv(x, Const(1)).evaluate(half) == 1
    assert FloorDiv(x, Const(1)).simplify().evaluate(half) == 1
    assert CeilDiv(x, Const(1)).evaluate(half) == 2
    assert CeilDiv(x, Const(1)).simplify().evaluate(half) == 2
    for kind in (Div, FloorDiv, CeilDiv):
        simplified = kind(x, x).simplify()
        with pytest.raises(ModelingError):
            simplified.evaluate({"x": 0})
    for expression in (Div(x, x) - Div(x, x), 0 * Div(Const(1), x)):
        with pytest.raises(ModelingError):
            expression.simplify().evaluate({"x": 0})

def test_bucket_selection_index_and_distinct_exact_crossover_roots():
    partition = StructuralPartition((GraphBucketGuard((1, 2, 4, 8)),))
    assert partition.select({"T": 1}) == (0,)
    assert partition.select({"T": 3}) == (2,)
    assert partition.cell(partition.select({"T": 3}))[0].contains(3)
    assert not partition.cell(partition.select({"T": 1}))[0].contains(3)
    t = Var("T")
    a, b = 10**12, 10**12 + 100
    guard = PipeDominanceGuard((
        PipeDemand("cube", (t - Const(a)) * (t - Const(b)) + Const(10000), 1),
        PipeDemand("vector", Const(10000), 1),
    ))
    roots = guard.crossover("cube", "vector", "T", lower=a - 1, upper=b + 1)
    assert [root.value for root in roots] == [a, b]
    assert all(root.exact for root in roots)
    assert guard.dominant({"T": a - 1}) == "cube"
    assert guard.dominant({"T": a + 50}) == "vector"
    assert guard.dominant({"T": b + 1}) == "cube"
    for root in roots:
        assert guard.times({"T": root.value}) == {"cube": 10000, "vector": 10000}
    pairs = [(root.dominance_before, root.dominance_after) for root in roots]
    assert pairs == [("cube", "vector"), ("vector", "cube")]
    irrational = PipeDominanceGuard((
        PipeDemand("cube", (t - Const(a)) * (t - Const(a)) + Const(100), 1),
        PipeDemand("vector", Const(102), 1),
    ))
    roots = irrational.crossover("cube", "vector", "T", lower=a - 3, upper=a + 3)
    assert len(roots) == 2 and not any(root.exact for root in roots)
    assert roots[0].value == pytest.approx(a - math.sqrt(2), abs=1e-3)
    assert roots[1].value == pytest.approx(a + math.sqrt(2), abs=1e-3)
    assert [(root.dominance_before, root.dominance_after) for root in roots] == [
        ("cube", "vector"), ("vector", "cube")]

def test_compiler_domain_segments_limit_service_and_boundary_distance():
    model = make_model()
    instance = instantiate_model(model, features(model)).find("F0", layer=0)
    domain = AnalysisDomain(max_tokens=64)
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY, domain=domain).compile(
        instance)
    gate = compiled.operator("gate_up")
    geometry = WorkloadGeometry.homogeneous(1, 31, 32)
    cost = gate.evaluate(geometry)
    table = gate.piecewise(domain, limit=100)
    match = [s for s in table.segments if s.regime == cost.regime]
    assert len(match) == 1 and match[0].index == cost.segment_index
    export = AnalysisDomain(max_query_tiles=64, max_kv_tiles=64)
    exported = gate.piecewise(export, limit=100)
    assert exported.domain == export and all(
        segment.domain == export for segment in exported.segments)
    exported_match = [s for s in exported.segments if s.regime == cost.regime]
    expected = 0
    regime = dict(cost.regime)
    for axis in gate.axes:
        expected = (expected * max(1, len(axis.index_range(export)))
                    + axis.position(regime[axis.name], export))
    assert exported_match[0].index == expected
    restricted = OperatorPiecewiseCompiler(SUPPLY, POLICY,
                                           domain=AnalysisDomain(max_tokens=1))
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        restricted.compile(instance).evaluate(geometry)
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        restricted.compile(instance).operator("gate_up").evaluate(geometry)
    padded = WorkloadGeometry((RequestGeometry(1, 0),), physical_tokens=8)
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        padded.validate(AnalysisDomain(max_tokens=1))
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        OperatorPiecewiseCompiler(
            SUPPLY, POLICY, domain=AnalysisDomain(max_q=1)).compile(
                instance).evaluate(geometry)
    single = instantiate_model(model, features(model, max_splits=1)).find("F0", layer=0)
    limited = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile(
        single).piecewise(AnalysisDomain(max_tokens=16), limit=1)
    assert len(limited) == 1 and limited.truncated
    assert limited.total_cells == len(single.operators)
    assert cost.duration_ms == sum(cost.stage_times, Fraction(0))
    for pipe in ("cube", "hbm", "mte2"):
        service = sum((dict(stage).get(pipe, 0) / SUPPLY.effective(pipe)
                       for stage in cost.stage_work), Fraction(0))
        assert cost.pipe_time(pipe) == service
    assert cost.pipe_time("cube") > max(
        dict(stage).get("cube", 0) / SUPPLY.effective("cube")
        for stage in cost.stage_work)
    distance = boundary_distance(gate, "M_TILES", 31)
    assert (distance["lower"], distance["upper"]) == ("16", "32")
    assert (distance["distance_lower"], distance["distance_upper"]) == ("15", "1")

def test_default_domain_covers_long_sequences_and_enforces_derived_bounds():
    domain = AnalysisDomain()
    assert (domain.max_tokens, domain.max_batch, domain.max_q, domain.max_s_kv,
            domain.max_seq, domain.max_query_tiles, domain.max_kv_tiles,
            domain.max_work, domain.max_moe_rows, domain.max_index_rows,
            domain.max_media_tokens) == (
        1_048_576, 4096, 262_144, 1_048_576, 1_048_576, 65536, 65536,
        1_000_000_000_000, 1_048_576, 262_144, 65536)
    long = WorkloadGeometry((RequestGeometry(1, 131071),))
    long.validate(domain)
    compiled = compile_kind(make_model(), "A0")
    assert attention_cost(compiled, long).get("causal_pairs") == 131072
    restricted = OperatorPiecewiseCompiler(
        SUPPLY, POLICY, domain=AnalysisDomain(max_kv_tiles=1))
    with pytest.raises(ModelingError, match="analysis_domain_miss"):
        restricted.compile(compiled.instance).evaluate(
            WorkloadGeometry((RequestGeometry(1, 4096),)))

def test_request_query_ranges_select_non_causal_physical_rows():
    partial = RequestGeometry(4, 0, query_ranges=((0, 1),))
    assert partial.selected_q_len == 1
    assert (partial.pairs(), partial.pairs(causal=False)) == (1, 4)
    assert WorkloadGeometry((partial,)).pairs(causal=False) == 4
    assert WorkloadGeometry((partial,)).bindings()["PAIRS_NC"] == 4
    assert WorkloadGeometry((partial,)).selected_tokens == 1
    assert WorkloadGeometry((partial,)).bindings()["T_SELECTED"] == 1
    assert WorkloadGeometry((RequestGeometry(4, 0),
                             RequestGeometry(2, 0))).selected_tokens == 6
    assert RequestGeometry(4, 0).selected_q_len == 4
    assert RequestGeometry(4, 0).pairs(causal=False) == 16
    reordered = RequestGeometry(4, 0, query_ranges=((3, 4), (0, 1)))
    assert reordered.query_ranges == ((0, 1), (3, 4))
    assert RequestGeometry(4, 0, query_ranges=((2, 4), (0, 1),
                                               (1, 2))).query_ranges == ((0, 4),)
    compiled = compile_kind(make_model(), "A0")
    forward = attention_cost(compiled, WorkloadGeometry((RequestGeometry(
        4, 0, query_ranges=((0, 1), (3, 4))),)))
    backward = attention_cost(compiled, WorkloadGeometry((RequestGeometry(
        4, 0, query_ranges=((3, 4), (0, 1))),)))
    assert forward.get("physical_pairs") == backward.get("physical_pairs")
    assert forward.get("physical_flops") == backward.get("physical_flops")
    for invalid in ((-1, 1), (1, -1), (2, 1), (0, 5)):
        with pytest.raises(ModelingError):
            RequestGeometry(4, 0, query_ranges=(invalid,))
