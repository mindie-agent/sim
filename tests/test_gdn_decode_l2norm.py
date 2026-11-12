"""Decode GDN exposes the source q/k L2-normalization launches."""
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.subgraphs.builders_attention import _build_a0, _build_a3
from dataclasses import replace
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.prediction.piecewise.compiler import (ImplementationPolicy,
                                                       OperatorPiecewiseCompiler,
                                                       PipeSupply)
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.gdn_program_bridge import gating_duration_ms
from dataclasses import replace as dc_replace


def _ctx():
    model = load_model("qwen3.6-27b")
    layer = next(plan for plan in model.layer_plans if plan.linear_heads)
    features = FeatureConfig.from_model(model, tp=2, dp=4, ep=1)
    return model, layer, _Ctx(model, layer, features, stage=0, layers=len(model.layer_plans))


def test_gdn_decode_emits_separate_q_and_k_l2norm():
    model, layer, ctx = _ctx()
    built = _build_a3(ctx)
    names = [op.name for op in built.template.operators]
    assert names.count("q_l2norm") == 1
    assert names.count("k_l2norm") == 1
    for name in ("q_l2norm", "k_l2norm"):
        sig = built.signature(name)
        assert sig.find("source_kernel") == "l2norm_fwd_kernel2"
        assert sig.find("decode_only") == 1
        assert sig.require("elements") == (layer.linear_key_heads + 1) // 2 * layer.linear_head_dim


def test_non_gdn_attention_layer_has_no_decode_l2norm():
    model, layer, ctx = _ctx()
    regular = next(plan for plan in model.layer_plans if not plan.linear_heads)
    built = _build_a0(_Ctx(model, regular, ctx.features, stage=0,
                           layers=len(model.layer_plans)))
    names = [op.name for op in built.template.operators]
    assert "q_l2norm" not in names and "k_l2norm" not in names


def test_q_k_l2norm_not_emitted_in_chunked_prefill():
    model, layer, ctx = _ctx()
    built = _build_a3(replace(ctx, variant="chunked_prefill"))
    names = [op.name for op in built.template.operators]
    assert "q_l2norm" not in names and "k_l2norm" not in names


def test_q_k_l2norm_bytes_are_source_shaped():
    model, layer, ctx = _ctx()
    built = _build_a3(ctx)
    sig = built.signature("q_l2norm")
    local_key_heads = (layer.linear_key_heads + 1) // 2
    assert sig.require("elements") == local_key_heads * layer.linear_head_dim
    assert sig.find("read_bytes") == local_key_heads * layer.linear_head_dim * 2
    assert sig.find("write_bytes") == local_key_heads * layer.linear_head_dim * 2
    assert sig.find("ops_per_element") == 5


_SUPPLY = PipeSupply((("cube", 1e9), ("vector", 1e9), ("hbm", 1e9),
                     ("mte1", 1e9), ("mte2", 1e9), ("mte2_aiv", 1e9), ("mte2_aic", 1e9),
                     ("mte3", 1e9), ("fixpipe", 1e9), ("link", 1e9),
                     ("scalar_store_issue", 1e9), ("reg_data_writes", 1e9),
                     ("reg_ub_accesses", 1e9), ("reg_ub_stores", 1e9),
                     ("reg_alu_lanes", 1e9), ("reg_special_lanes", 1e9),
                     ("reg_reduce_lanes", 1e9), ("reg_shuffle_lanes", 1e9)),
                    resource_units=(("vector", 64), ("mte2_aiv", 64),
                                    ("mte2_aic", 64), ("mte3", 64), ("hbm", 64),
                                    ("scalar_store_issue", 64),
                                    ("reg_data_writes", 64), ("reg_ub_accesses", 64),
                                    ("reg_ub_stores", 64), ("reg_alu_lanes", 64),
                                    ("reg_special_lanes", 64), ("reg_reduce_lanes", 64),
                                    ("reg_shuffle_lanes", 64),
                                    ("mte1", 64), ("fixpipe", 64)),
                    service_modes=(("reg_issue_slots", ("vector", 211200000000.0)),
                                   ("ub_copy", ("vector", 27033600000000.0))),
                    compute_units=32,
                    buffers=(("ub", 253952),), pool_units=(("ub", 64),),
                    compute_peaks=(("bf16", 432537600000000.0),
                                   ("fp32", 27033600000000.0)))
_POLICY = ImplementationPolicy(mt=128, nt=16, cores=32, moe_row_tile=4,
                              split_target=1000, max_splits=4)


def _compiled_a3():
    from mindie_sim.prediction.subgraphs.builders_attention import _build_a3
    from mindie_sim.prediction.subgraphs.instantiate import _Ctx
    model = replace(load_model("qwen3.6-27b"), quant=QuantSpec(
        name="w8a8", weight_format="int8", activation_format="int8"))
    layer = next(plan for plan in model.layer_plans if plan.linear_heads)
    features = FeatureConfig.from_model(
        model, tp=2, dp=4, ep=1, pp=1, token_tile=128, kv_tile=16,
        cores=32, moe_row_tile=4, split_target=1000, max_splits=4)
    return _build_a3(_Ctx(model, layer, features, stage=0,
                          layers=len(model.layer_plans)))


def _compile(built, name):
    from mindie_sim.schema.subgraphs import BoundOperator
    template = next(op for op in built.template.operators if op.name == name)
    op = BoundOperator(template, built.signature(name))
    return OperatorPiecewiseCompiler(_SUPPLY, _POLICY).compile_operator(op, kind="A3")


def _compiled_qk():
    built = _compiled_a3()
    return _compile(built, "q_l2norm"), _compile(built, "k_l2norm")


def _geometry(*requests, phase):
    return WorkloadGeometry(tuple(requests), graph_bucket=sum(
        r.q_len for r in requests), phase=phase,
        local_batch=len(requests))


def test_qk_norm_activity_follows_decode_request_phase_and_rows():
    q, k = _compiled_qk()
    one = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    two = _geometry(RequestGeometry(1, 4096, phase="decode"),
                    RequestGeometry(1, 4096, phase="decode"), phase="decode")
    mixed = _geometry(RequestGeometry(1, 4096, phase="decode"),
                      RequestGeometry(4096, 0, phase="prefill"),
                      phase="decode")
    pure = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    for op in (q, k):
        c1 = op.evaluate(one)
        assert c1.pipe_work("hbm") == 4096
        assert c1.pipe_work("vector") == 8192
        c2 = op.evaluate(two)
        assert c2.pipe_work("hbm") == 8192
        assert c2.pipe_work("vector") == 10272
        cm = op.evaluate(mixed)
        assert cm.pipe_work("hbm") == 4096
        assert cm.pipe_work("vector") == 8192
        assert dict(cm.materialization)["q_l2norm_output" if op.name == "q_l2norm"
                                        else "k_l2norm_output"] == 2048
        cp = op.evaluate(pure)
        assert cp.work == () and cp.duration_ms == 0
        assert all(value == 0 for _, value in cp.materialization)
        assert cp.fidelity == "inactive"


def test_fused_gating_source_work_and_outputs():
    gating = _compile(_compiled_a3(), "gdn_gating")
    geometry = _geometry(RequestGeometry(1, 4096, phase="decode"),
                         RequestGeometry(1, 4096, phase="decode"),
                         phase="decode")
    cost = gating.evaluate(geometry)
    assert cost.pipe_work("hbm") == 12768
    assert cost.pipe_work("vector") == 1281024
    material = dict(cost.materialization)
    assert material["gdn_gating_decay_fp32"] == 24 * 4 * 2
    assert material["gdn_gating_beta_bf16"] == 24 * 2 * 2
    assert dict(cost.derived)["launched_blocks"] == 64
    assert dict(cost.derived)["active_lanes"] == 64


def test_prepare_activity_is_prefill_only():
    prepare = _compile(_compiled_a3(), "prepare")
    decode = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    prefill = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    assert prepare.evaluate(decode).fidelity == "inactive"
    assert prepare.evaluate(prefill).fidelity != "inactive"


def test_gating_plan_m1_m2_prefill_mixed_controls():
    gating = _compile(_compiled_a3(), "gdn_gating")
    m1 = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    m2 = _geometry(RequestGeometry(1, 4096, phase="decode"),
                   RequestGeometry(1, 4096, phase="decode"), phase="decode")
    prefill = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    mixed = _geometry(RequestGeometry(1, 4096, phase="decode"),
                      RequestGeometry(4096, 0, phase="prefill"), phase="decode")
    c1, c2 = gating.evaluate(m1), gating.evaluate(m2)
    assert c1.pipe_work("hbm") == 12528 and c2.pipe_work("hbm") == 12768
    assert c1.pipe_work("vector") == c2.pipe_work("vector") == 1281024
    assert dict(c1.materialization)["gdn_gating_decay_fp32"] == 96
    assert dict(c2.materialization)["gdn_gating_decay_fp32"] == 192
    cp, cm = gating.evaluate(prefill), gating.evaluate(mixed)
    assert cp.pipe_work("hbm") == 995328
    assert cm.pipe_work("hbm") == 1007856
    assert cm.pipe_work("vector") == 2562048
    assert dict(cm.materialization)["gdn_gating_beta_bf16"] == 196656


def test_gating_program_queue_latency_changes_time_not_work():
    base = gating_duration_ms(_SUPPLY, 2, 24)
    slow_queue = {"outstanding": 8, "granule_bytes": 128, "element_bytes": 2,
                  "response_ms": 9.125e-05, "row_control_ms": 7e-06}
    slow_queue["response_ms"] *= 8
    slow = gating_duration_ms(_SUPPLY, 2, 24, queue=slow_queue)
    assert slow > base
    # The plan work is geometry-only and is unchanged by the return queue.
    gating = _compile(_compiled_a3(), "gdn_gating")
    geometry = _geometry(RequestGeometry(1, 4096, phase="decode"),
                         RequestGeometry(1, 4096, phase="decode"), phase="decode")
    assert gating.evaluate(geometry).pipe_work("hbm") == 12768


def test_gating_no_queue_supply_uses_mte3_path_without_scalar():
    no_queue = dc_replace(_SUPPLY, store_queues=())
    duration = gating_duration_ms(no_queue, 2, 24)
    assert duration > 0
    gating = _compile(_compiled_a3(), "gdn_gating")
    geometry = _geometry(RequestGeometry(1, 4096, phase="decode"),
                         RequestGeometry(1, 4096, phase="decode"), phase="decode")
    cost = gating.evaluate(geometry)
    assert "scalar_store_issue" not in dict(cost.work)
    assert cost.pipe_work("mte3") == 2 * 24 * 6


def test_absent_scalar_resource_uses_no_queue_path():
    rates = tuple((k, v) for k, v in _SUPPLY.rates if k != "scalar_store_issue")
    units = tuple((k, v) for k, v in _SUPPLY.resource_units
                  if k != "scalar_store_issue")
    supply = dc_replace(_SUPPLY, rates=rates, resource_units=units,
                        store_queues=())
    result = gating_duration_ms(supply, 2, 24)
    assert result > 0


def test_declared_queue_flows_through_compiled_gating_price():
    queued = dc_replace(_SUPPLY, store_queues=((
        "scalar_store_issue",
        (("element_bytes", 2), ("granule_bytes", 128), ("outstanding", 8),
         ("response_ms", 9.125e-05), ("row_control_ms", 7e-06))),))
    built = _compiled_a3()
    template = next(op for op in built.template.operators if op.name == "gdn_gating")
    from mindie_sim.schema.subgraphs import BoundOperator
    compiled = OperatorPiecewiseCompiler(queued, _POLICY).compile_operator(
        BoundOperator(template, built.signature("gdn_gating")), kind="A3")
    geometry = _geometry(RequestGeometry(1, 4096, phase="decode"),
                         RequestGeometry(1, 4096, phase="decode"), phase="decode")
    cost = compiled.evaluate(geometry)
    assert cost.pipe_work("scalar_store_issue") == 48
    assert float(cost.duration_ms) > 0.001312


def test_runtime_store_binding_changes_time_with_fixed_work():
    queue = (("element_bytes", 2), ("granule_bytes", 128), ("outstanding", 8),
             ("response_ms", 9.125e-05), ("row_control_ms", 7e-06))
    base_store = (("resource", "scalar_store_issue"), ("queue", queue))
    base_supply = dc_replace(
        _SUPPLY, store_queues=(("scalar_store_issue", queue),),
        gdn_beta_store=base_store)
    slow_queue = tuple((k, (v * 8 if k == "response_ms" else v))
                       for k, v in queue)
    slow_supply = dc_replace(
        base_supply,
        gdn_beta_store=(("resource", "scalar_store_issue"),
                        ("queue", slow_queue)))
    built = _compiled_a3()
    template = next(op for op in built.template.operators if op.name == "gdn_gating")
    from mindie_sim.schema.subgraphs import BoundOperator
    operator = BoundOperator(template, built.signature("gdn_gating"))
    geometry = _geometry(RequestGeometry(1, 4096, phase="decode"),
                         RequestGeometry(1, 4096, phase="decode"), phase="decode")
    base = OperatorPiecewiseCompiler(base_supply, _POLICY).compile_operator(
        operator, kind="A3").evaluate(geometry)
    slow = OperatorPiecewiseCompiler(slow_supply, _POLICY).compile_operator(
        operator, kind="A3").evaluate(geometry)
    assert base.pipe_work("scalar_store_issue") == slow.pipe_work("scalar_store_issue") == 48
    assert slow.duration_ms > base.duration_ms


def test_native_recurrent_q1_q1x2_source_program_controls():
    from mindie_sim.schema.subgraphs import BoundOperator
    supply = dc_replace(_SUPPLY, gdn_decode_selected=True,
                        gdn_recurrent_implementation="arch35_fp32")
    built = _compiled_a3()
    compiler = OperatorPiecewiseCompiler(supply, _POLICY)
    recurrent = compiler.compile_operator(
        BoundOperator(next(op for op in built.template.operators
                           if op.name == "gated_delta"),
                      built.signature("gated_delta")), kind="A3")
    one = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    two = _geometry(RequestGeometry(1, 4096, phase="decode"),
                    RequestGeometry(1, 4096, phase="decode"), phase="decode")
    c1, c2 = recurrent.evaluate(one), recurrent.evaluate(two)
    assert c1.duration_ms > 0 and c2.duration_ms > 0
    assert c2.pipe_work("hbm") > c1.pipe_work("hbm")
    assert c2.pipe_work("mte3") == 2 * c1.pipe_work("mte3")


def test_mixed_recurrent_keeps_prefill_branch_not_decode_only():
    from mindie_sim.schema.subgraphs import BoundOperator
    supply = dc_replace(_SUPPLY, gdn_decode_selected=True,
                        gdn_recurrent_implementation="arch35_fp32")
    built = _compiled_a3()
    recurrent = OperatorPiecewiseCompiler(supply, _POLICY).compile_operator(
        BoundOperator(next(op for op in built.template.operators
                           if op.name == "gated_delta"),
                      built.signature("gated_delta")), kind="A3")
    one = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    mixed = _geometry(RequestGeometry(1, 4096, phase="decode"),
                      RequestGeometry(4096, 0, phase="prefill"), phase="decode")
    state_one = recurrent.evaluate(one).pipe_work("hbm")
    mixed_cost = recurrent.evaluate(mixed)
    assert mixed_cost.pipe_work("hbm") > state_one
    assert mixed_cost.duration_ms != recurrent.evaluate(one).duration_ms


def test_conv_prefill_and_mixed_use_native_program_not_decode_only():
    from mindie_sim.schema.subgraphs import BoundOperator
    supply = dc_replace(_SUPPLY, gdn_decode_selected=True)
    built = _compiled_a3()
    conv = OperatorPiecewiseCompiler(supply, _POLICY).compile_operator(
        BoundOperator(next(op for op in built.template.operators if op.name == "conv"),
                      built.signature("conv")), kind="A3")
    pure = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    mixed = _geometry(RequestGeometry(1, 4096, phase="decode"),
                      RequestGeometry(4096, 0, phase="prefill"), phase="decode")
    q1 = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    assert conv.program.program_duration(pure, None) is not None
    assert conv.program.program_duration(mixed, None) is not None
    assert conv.program.program_duration(q1, None) is not None
    assert conv.evaluate(pure).pipe_work("hbm") != conv.evaluate(q1).pipe_work("hbm")


def test_pure_prefill_conv_uses_history3_not_4098_and_speculative_is_explicit():
    from mindie_sim.schema.subgraphs import BoundOperator
    from mindie_sim.prediction.gdn_program_bridge import conv_program_result
    supply = dc_replace(_SUPPLY, gdn_decode_selected=True)
    built = _compiled_a3()
    conv = OperatorPiecewiseCompiler(supply, _POLICY).compile_operator(
        BoundOperator(next(op for op in built.template.operators if op.name == "conv"),
                      built.signature("conv")), kind="A3")
    pure = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    assert dict(conv.evaluate(pure).derived)["conv_history_rows"] == 3
    # Genuine speculative is an explicit q>1/history4 request, not no-draft
    # prefill; the native bridge must accept it only when asked.
    spec = conv_program_result(supply, 2, 5120, (2,), 4, True, 4)
    assert spec[0] > 0
