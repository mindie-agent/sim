"""Per-token DynamicQuantV2 is attached from the role's own activation.

The producer is the explicit dynamic int8 quant ahead of a qualifying GEMM.
BF16 role overrides, static int8, and FP8/MX do not take this schedule.
A fitting row is one GM load. Counts are the instantiated graph, not a
measured target.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import pytest
from dataclasses import replace
from pathlib import Path

from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.dynamic_quant_schedule import (
    ROW_SOURCE, host_source_copies, per_token_dynamic_int, quant_row_demand,
    quant_timeline,
)
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.families_matrix import UB_ALIGN
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.physical import ARCH3510_VECTOR_MIN_MS
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.builders_attention import _build_a0, _build_a3
from mindie_sim.prediction.subgraphs.builders_ffn import _build_f0
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, _Ctx, instantiate_model,
)
from mindie_sim.prediction.subgraphs.registry import TemplateRegistry
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"


def qwen_case():
    return parse_case(json.loads(CASE.read_text()))


def _ctx(model, layer, **features):
    config = FeatureConfig.from_model(model, tp=2, dp=4, ep=1, pp=1, **features)
    return _Ctx(model, layer, config, stage=0, layers=len(model.layer_plans))


def _layers(model):
    gdn = next(plan for plan in model.layer_plans if plan.linear_heads)
    attn = next(plan for plan in model.layer_plans if not plan.linear_heads)
    return gdn, attn


def test_preset_name_qualifies_dynamic_int8_without_repeated_scheme():
    case = qwen_case()
    quant = case.model.quant
    source = dict(quant.source)
    assert source.get("preset") == "w8a8_dynamic"
    assert "activation_scheme" not in source
    assert per_token_dynamic_int(quant, "gdn", "int8")
    assert per_token_dynamic_int(quant, "dense_mlp", "int8")
    assert per_token_dynamic_int(quant, "attn.qkv", "int8")
    assert per_token_dynamic_int(quant, "attn.o", "int8")
    assert not per_token_dynamic_int(quant, "gdn.o", "bf16")
    assert not per_token_dynamic_int(quant, "gdn.beta_alpha", "bf16")
    static = replace(quant, source=(
        ("activation_scheme", "static"),
        ("quant_method", "w8a8_dynamic"),
        ("preset", "w8a8_dynamic"),
    ))
    assert not per_token_dynamic_int(static, "dense_mlp", "int8")
    weighed = replace(quant, source=(
        ("activation_scheme", "dynamic"),
        ("quant_method", "weight_only"),
    ))
    assert not per_token_dynamic_int(weighed, "dense_mlp", "int8")
    mx = quant_preset("w8a8_mxfp8")
    assert not per_token_dynamic_int(mx, "dense_mlp", "fp8_e4m3")
    assert not per_token_dynamic_int(mx, "dense_mlp", "int8")


def _quant_names(built):
    return [op.name for op in built.template.operators if "Quant" in op.name]


def test_qwen_roles_precede_the_owning_gemm_once():
    case = qwen_case()
    model = case.model
    gdn, attn = _layers(model)
    a3 = _build_a3(_ctx(model, gdn))
    a0 = _build_a0(_ctx(model, attn))
    f0 = _build_f0(_ctx(model, attn))
    a3_names = [op.name for op in a3.template.operators]
    a0_names = [op.name for op in a0.template.operators]
    f0_names = [op.name for op in f0.template.operators]
    assert a3_names.index("qkvz_projection_QuantV2") + 1 == a3_names.index("qkvz_projection")
    assert a3_names.count("qkvz_projection_QuantV2") == 1
    assert "o_projection_QuantV2" not in a3_names
    assert a0_names.index("qkv_projection_QuantV2") + 1 == a0_names.index("qkv_projection")
    assert a0_names.index("o_projection_QuantV2") + 1 == a0_names.index("o_projection")
    assert f0_names.index("gate_up_QuantV2") + 1 == f0_names.index("gate_up")
    assert f0_names.index("down_QuantV2") + 1 == f0_names.index("down")
    hidden = model.hidden_size
    qkvz = a3.signature("qkvz_projection_QuantV2")
    assert qkvz.require("elements") == hidden
    assert qkvz.find("read_bytes") == hidden * 2
    assert qkvz.find("pass_read_bytes") == hidden * 2
    assert qkvz.find("write_bytes") == hidden + 4
    assert qkvz.find("quant_passes") == 2
    assert qkvz.find("row_scale_bytes") == 4
    assert qkvz.find("offset_gm_bytes") == 0
    assert qkvz.find("symmetrical") == 1
    assert qkvz.find("has_smooth") == 0
    assert qkvz.find("scale_format") == "fp32"
    assert qkvz.find("source_format") == "bf16"
    assert qkvz.find("block_axis") == "row"
    assert qkvz.find("category") == "gdn"
    assert qkvz.find("out_format") == "int8"
    assert f0.signature("gate_up_QuantV2").require("elements") == hidden
    assert f0.signature("down_QuantV2").require("elements") == f0.signature("down").require("K")
    assert a0.signature("qkv_projection_QuantV2").require("elements") == hidden
    assert a0.signature("o_projection_QuantV2").require("elements") == a0.signature("o_projection").require("K")
    assert a0.signature("o_projection_QuantV2").find("category") == "attn.o"
    for name in ("beta_slice", "alpha_slice", "core_zeros", "qkv_slice",
                 "q_slice", "k_slice", "v_slice", "q_l2norm", "k_l2norm"):
        assert a3_names.count(name) == 1


def test_gather_stays_ahead_of_the_quant_when_flashcomm_is_on():
    case = qwen_case()
    model = replace(case.model, metadata=case.model.metadata + (
        ("source_model_type", "qwen3_5_text"),))
    gdn, attn = _layers(model)
    features = dict(metadata=(
        ("source_model_type", "qwen3_5_text"),
        ("additional_config", (("enable_flashcomm1", "1"),)),
    ))
    f0_names = [op.name for op in _build_f0(_ctx(model, attn, **features)).template.operators]
    a0_names = [op.name for op in _build_a0(_ctx(model, attn, **features)).template.operators]
    a3_names = [op.name for op in _build_a3(_ctx(model, gdn, **features)).template.operators]
    assert f0_names.index("tp_all_gather") < f0_names.index("gate_up_QuantV2") < f0_names.index("gate_up")
    assert f0_names.index("down_QuantV2") + 1 == f0_names.index("down")
    assert f0_names.index("down") < f0_names.index("tp_reduce_scatter")
    assert "tp_allreduce" not in f0_names
    assert a0_names.index("tp_all_gather") < a0_names.index("qkv_projection_QuantV2")
    assert a0_names.index("qkv_projection_QuantV2") + 1 == a0_names.index("qkv_projection")
    assert a0_names.index("o_projection_QuantV2") + 1 == a0_names.index("o_projection")
    assert a0_names.index("o_projection") < a0_names.index("tp_reduce_scatter")
    assert a3_names.index("tp_all_gather") < a3_names.index("qkvz_projection_QuantV2")
    assert a3_names.index("qkvz_projection_QuantV2") + 1 == a3_names.index("qkvz_projection")
    assert "o_projection_QuantV2" not in a3_names
    assert a3_names.index("o_projection") < a3_names.index("tp_reduce_scatter")


def test_static_mx_and_bf16_do_not_grow_the_dynamic_producer():
    case = qwen_case()
    model = case.model
    _gdn, attn = _layers(model)
    static = replace(model, quant=replace(model.quant, source=(
        ("activation_scheme", "static"),
        ("quant_method", "w8a8_dynamic"),
        ("preset", "w8a8_dynamic"),
    )))
    assert _quant_names(_build_f0(_ctx(static, attn))) == []
    assert _quant_names(_build_a0(_ctx(static, attn))) == []
    mx = replace(model, quant=quant_preset("w8a8_mxfp8"))
    gdn = next(plan for plan in mx.layer_plans if plan.linear_heads)
    assert _quant_names(_build_f0(_ctx(mx, attn))) == []
    assert _quant_names(_build_a3(_ctx(mx, gdn))) == []
    # The pre-existing attn.o guard still serves an FP8 role. It does not
    # gain the int8 two-pass scale.
    mx_o = _build_a0(_ctx(mx, attn)).signature("o_projection_QuantV2")
    assert mx_o.find("quant_passes") is None
    assert mx_o.find("out_format") == "fp8_e4m3"
    plain = replace(model, quant=None)
    assert _quant_names(_build_f0(_ctx(plain, attn))) == []
    assert _quant_names(_build_a0(_ctx(plain, attn))) == []
    assert _quant_names(_build_a3(_ctx(plain, gdn))) == []


def test_instantiated_qwen_graph_has_the_five_dynamic_roles():
    case = qwen_case()
    model = case.model
    features = FeatureConfig.from_runtime(case.runtime, model)
    instantiated = instantiate_model(model, features, registry=TemplateRegistry())
    counts = Counter()
    views = Counter()
    for instance in instantiated.subgraphs:
        for op in instance.operators:
            if "Quant" in op.name:
                counts[(instance.kind, op.name)] += 1
            if op.name in {"beta_slice", "q_l2norm"}:
                views[op.name] += 1
    assert counts[("F0", "gate_up_QuantV2")] == 64
    assert counts[("F0", "down_QuantV2")] == 64
    assert counts[("A3", "qkvz_projection_QuantV2")] == 48
    assert counts[("A0", "qkv_projection_QuantV2")] == 16
    assert counts[("A0", "o_projection_QuantV2")] == 16
    assert counts[("A3", "o_projection_QuantV2")] == 0
    assert sum(counts.values()) == 208
    assert views["beta_slice"] == 48
    assert views["q_l2norm"] == 48


def _align(value):
    return int(-(-int(value) // UB_ALIGN)) * UB_ALIGN


def _quant_operator(elements, *, cores=32, tp=1):
    source = 2.0
    scale = 4.0
    return BoundOperator(
        OperatorTemplate(name="gate_up_QuantV2", family="quantize", role="quantize"),
        StaticSignature(
            role="quantize",
            matrix=(("elements", elements),),
            quant=(("act_bytes", 1.0),),
            parallel=(("tp", tp),),
            tiles=(("mt", 128), ("cores", cores)),
            extras=(
                ("ops_per_element", 2),
                ("read_bytes", elements * source),
                ("write_bytes", elements * 1 + scale),
                ("quant_op", "QuantV2"),
                ("out_format", "int8"),
                ("category", "dense_mlp"),
                ("quant_passes", 2),
                ("pass_read_bytes", elements * source),
                ("source_bytes", source),
                ("source_format", "bf16"),
                ("row_scale_bytes", scale),
                ("scale_format", "fp32"),
                ("offset_gm_bytes", 0),
                ("symmetrical", 1),
                ("has_smooth", 0),
                ("block_axis", "row"),
            ),
        ),
    )


def _machine_supply():
    # 9579 record: 64 AIV and 253952 B UB. cores=32 is the AIC cap and must
    # not become BlockNum. The vector invocation is present and must not
    # become this kernel's elapsed time.
    return PipeSupply(
        (("cube", 1e18), ("vector", 1e18), ("hbm", 1e18),
         ("mte1", 1e18), ("mte2", 1e18), ("mte3", 1e18),
         ("fixpipe", 1e18), ("link", 1e18)),
        invocations=(("vector", float(ARCH3510_VECTOR_MIN_MS)),
                     ("ub_bytes", 253952.0)),
        resource_units=(("vector", 64), ("mte2_aiv", 64), ("mte3", 64)),
    )


def _profile_supply(*, aiv=64, ub=253952):
    """9579 die rates in work per millisecond. Unit count is the published record."""
    return PipeSupply(
        (("cube", 1e18), ("vector", 6.7584e9), ("hbm", 1.6e9),
         ("mte1", 1e18), ("mte2", 4.224e9), ("mte3", 1.35168e10),
         ("fixpipe", 1e18), ("link", 1e18)),
        invocations=(("vector", float(ARCH3510_VECTOR_MIN_MS)),
                     ("ub_bytes", float(ub))),
        resource_units=(("vector", aiv), ("mte2_aiv", aiv), ("mte3", aiv)),
    )


def _policy():
    return ImplementationPolicy(mt=128, nt=16, cores=32, moe_row_tile=4,
                                split_target=1000, max_splits=4)


def _compile(elements, supply, *, tp=1, kind=""):
    return OperatorPiecewiseCompiler(supply, _policy()).compile_operator(
        _quant_operator(elements, tp=tp), kind=kind)


def test_fitting_row_is_one_gm_load_and_engine_makespan():
    elements = 5120
    ub = 253952
    for width in (3072, 5120, 8704):
        plan = host_source_copies(width, ub)
        assert plan["mode"] == "full_db"
        assert plan["gm_source_reads"] == 1
        assert plan["copies"] == (width,)
    supply = _machine_supply()
    compiled = _compile(elements, supply)
    cost = compiled.evaluate(WorkloadGeometry.homogeneous(1, 2, 3))
    assert cost.stage_times == ()
    assert cost.get("quant_passes") == 2
    assert cost.get("row_scale_bytes") == 4
    assert cost.get("offset_gm_bytes") == 0
    assert cost.get("ub_mode") == 2
    assert cost.get("use_db") == 1
    assert cost.get("gm_source_reads") == 1
    assert cost.get("aiv_owners") == 64
    assert cost.get("launched_blocks") == 2
    assert cost.get("rows_per_head_core") == 1
    assert cost.get("rows_per_tail_core") == 0
    assert cost.get("physical_rows") == 2
    assert cost.get("valid_rows") == 2
    assert cost.get("y_gm_row_bytes") == elements
    assert cost.get("scale_gm_row_bytes") == 4
    assert cost.get("ub_row_bytes") == _align(elements * 2)
    assert cost.get("ub_write_row_bytes") == _align(elements + 4)
    assert cost.get("gm_read_row_bytes") == _align(elements * 2)
    mte2 = sum(dict(stage).get("mte2", 0) for stage in cost.stage_work)
    mte3 = sum(dict(stage).get("mte3", 0) for stage in cost.stage_work)
    hbm = sum(dict(stage).get("hbm", 0) for stage in cost.stage_work)
    assert mte2 == 2 * _align(elements * 2)
    assert mte3 == 2 * (elements + 4)
    assert hbm == mte2 + mte3
    per_row = round(elements * 59 / 8) + 1024 * math.ceil(elements / 256)
    assert dict(cost.work)["vector"] == 2 * per_row
    report = quant_timeline(
        rates=dict(supply.rates), units=dict(supply.resource_units),
        rows=2, row_len=elements, source_bytes=2, y_bytes=1, scale_bytes=4,
        ub_bytes=ub, has_smooth=False, aiv=64)
    assert float(cost.duration_ms) == pytest.approx(report["makespan_ms"])
    assert float(cost.duration_ms) < float(ARCH3510_VECTOR_MIN_MS)
    assert report["mte2_bytes"] == mte2
    assert report["mte3_bytes"] == mte3
    assert report["vector_ops"] == 2 * per_row


def test_aiv_block_count_saturates_at_the_machine_not_the_aic_cap():
    supply = _machine_supply()
    compiled = _compile(5120, supply)

    def blocks(requests):
        geometry = WorkloadGeometry(tuple(requests), phase=requests[0].phase)
        return compiled.evaluate(geometry).get("launched_blocks")

    prefill = RequestGeometry(4096, 0, phase="prefill")
    assert blocks((prefill,)) == 64
    # Two decode rows, no tp pad: two owners, not the AIC cap of 32.
    assert blocks((RequestGeometry(1, 4160), RequestGeometry(1, 4160))) == 2


def test_mixed_q_is_4097_valid_and_only_f0_pads():
    supply = _machine_supply()
    mixed = WorkloadGeometry((
        RequestGeometry(1, 0, phase="decode"),
        RequestGeometry(4096, 0, phase="prefill"),
    ), phase="prefill")
    f0 = _compile(5120, supply, tp=2, kind="F0").evaluate(mixed)
    other = _compile(5120, supply, tp=2, kind="A0").evaluate(mixed)
    assert f0.get("valid_rows") == 4097
    assert other.get("valid_rows") == 4097
    assert f0.get("physical_rows") == 4098
    assert other.get("physical_rows") == 4097
    assert f0.get("launched_blocks") == 64
    assert other.get("launched_blocks") == 64
    f0_mte2 = sum(dict(stage).get("mte2", 0) for stage in f0.stage_work)
    other_mte2 = sum(dict(stage).get("mte2", 0) for stage in other.stage_work)
    assert other_mte2 == 4097 * _align(5120 * 2)
    assert f0_mte2 - other_mte2 == _align(5120 * 2)
    demand = quant_row_demand(mixed, kind="F0", tp=2)
    assert demand.source == ROW_SOURCE
    assert quant_row_demand(mixed, kind="A3", tp=2).rows == 4097
    # An already published physical count is not padded a second time.
    published = WorkloadGeometry(
        (RequestGeometry(1, 0, phase="decode"),), physical_tokens=4, phase="decode")
    assert quant_row_demand(published, kind="F0", tp=2).rows == 4
    assert quant_row_demand(published, kind="F0", tp=2).source == "physical_tokens"


def test_large_shape_rereads_only_full_chunks():
    ub = 253952
    # Double-buffer does not fit, single-buffer does: still one GM load.
    narrow = host_source_copies(11520, ub)
    assert narrow["mode"] == "full_single"
    assert narrow["gm_source_reads"] == 1
    # Misses full-load but the large chunk still covers the row once.
    covered = host_source_copies(23040, ub)
    assert covered["mode"] == "large"
    assert covered["copies"] == (23040,)
    # One full chunk is copied twice; the tail is copied once.
    wide = host_source_copies(32768, ub)
    assert wide["mode"] == "large"
    assert wide["gm_source_reads"] == 2
    assert wide["copies"].count(31488) == 2
    assert wide["copies"][-1] == 1280
    supply = _machine_supply()
    compiled = _compile(32768, supply)
    cost = compiled.evaluate(WorkloadGeometry.homogeneous(1, 2, 3))
    assert cost.get("ub_mode") == 3
    assert cost.get("gm_source_reads") == 2
    per_row = 2 * _align(31488 * 2) + _align(1280 * 2)
    assert cost.get("gm_read_row_bytes") == per_row
    mte2 = sum(dict(stage).get("mte2", 0) for stage in cost.stage_work)
    mte3 = sum(dict(stage).get("mte3", 0) for stage in cost.stage_work)
    assert mte2 == 2 * per_row
    assert mte3 == 2 * (32768 + 4)
    vector = round(32768 * 59 / 8) + 1024 * math.ceil(32768 / 256)
    assert dict(cost.work)["vector"] == 2 * vector
    report = quant_timeline(
        rates=dict(supply.rates), units=dict(supply.resource_units),
        rows=2, row_len=32768, source_bytes=2, y_bytes=1, scale_bytes=4,
        ub_bytes=ub, has_smooth=False, aiv=64)
    assert report["vector_ops"] == 2 * vector
    assert report["mte2_bytes"] == mte2
    assert report["mode"] == "large"
    covered_cost = _compile(23040, supply).evaluate(WorkloadGeometry.homogeneous(1, 2, 3))
    assert covered.get("gm_source_reads") == 1
    covered_vector = round(23040 * 59 / 8) + 1024 * math.ceil(23040 / 256)
    assert covered_cost.get("gm_source_reads") == 1
    assert dict(covered_cost.work)["vector"] == 2 * covered_vector
    assert float(cost.duration_ms) < float(ARCH3510_VECTOR_MIN_MS)


def test_plain_quantized_elementwise_stays_one_stage():
    supply = _machine_supply()
    policy = ImplementationPolicy(mt=128, nt=16, cores=32, moe_row_tile=4,
                                  split_target=1000, max_splits=4)
    plain = BoundOperator(
        OperatorTemplate(name="AddRmsNormQuant", family="elementwise", role="norm"),
        StaticSignature(
            role="norm",
            matrix=(("elements", 100),),
            quant=(("act_bytes", 2.0),),
            tiles=(("mt", 16), ("cores", 1),),
            extras=(("ops_per_element", 5), ("read_bytes", 400.0),
                    ("write_bytes", 200.0), ("quantized", 1)),
        ),
    )
    plain_cost = OperatorPiecewiseCompiler(supply, policy).compile_operator(plain).evaluate(
        WorkloadGeometry.homogeneous(1, 3, 4))
    assert len(plain_cost.stage_times) == 1
    assert "quant_passes" not in dict(plain_cost.derived)
    assert plain_cost.get("ub_write_row_bytes") == 128
    assert float(plain_cost.duration_ms) == pytest.approx(float(ARCH3510_VECTOR_MIN_MS))


def _profile_report(rows, row_len, *, aiv=64, ub=253952):
    supply = _profile_supply(aiv=aiv, ub=ub)
    return quant_timeline(
        rates=dict(supply.rates), units=dict(supply.resource_units),
        rows=rows, row_len=row_len, source_bytes=2, y_bytes=1, scale_bytes=4,
        ub_bytes=ub, has_smooth=False, aiv=aiv)


def test_plain_m1_is_below_the_invocation_floor():
    supply = _profile_supply()
    cost = _compile(5120, supply).evaluate(WorkloadGeometry.homogeneous(1, 1, 2))
    report = _profile_report(1, 5120)
    assert cost.get("physical_rows") == 1
    assert cost.get("launched_blocks") == 1
    assert cost.stage_times == ()
    assert float(cost.duration_ms) == pytest.approx(report["makespan_ms"])
    assert 0 < float(cost.duration_ms) < float(ARCH3510_VECTOR_MIN_MS)
    vector_only = report["vector_ops"] / report["local_rates"]["vector"]
    assert float(cost.duration_ms) > vector_only
    # One tile has no cross-tile overlap, so the makespan is the serial sum.
    assert report["makespan_ms"] == pytest.approx(report["head_serial_ms"], rel=1e-6)


def test_full_db_tiles_overlap_and_one_owner_keeps_one_unit():
    # One AIV, four rows, full-db multiRow 2: two resident tiles. The next
    # CopyIn overlaps the current vector pipe, so the makespan is below the
    # serial sum of the same stages.
    overlapped = _profile_report(4, 5120, aiv=1)
    assert overlapped["slots"] == 2
    assert overlapped["tiles_per_head"] == 2
    assert overlapped["makespan_ms"] < overlapped["head_serial_ms"]
    # Two launched owners each keep one pinned unit. They do not split the
    # die vector rate between only the active owners.
    one = _profile_report(1, 5120, aiv=64)
    two = _profile_report(2, 5120, aiv=64)
    assert two["launched"] == 2
    assert two["makespan_ms"] == pytest.approx(one["makespan_ms"], rel=1e-6)
    assert two["local_rates"]["vector"] == pytest.approx(6.7584e9 / 64)


def test_unbalanced_head_core_is_the_makespan():
    head = _profile_report(65, 5120, aiv=64)
    even = _profile_report(64, 5120, aiv=64)
    assert head["rows_per_head"] == 2
    assert head["rows_per_tail"] == 1
    assert head["launched"] == 64
    assert head["head_ms"] > head["tail_ms"]
    assert head["makespan_ms"] == pytest.approx(head["head_ms"])
    assert head["makespan_ms"] > even["makespan_ms"]


def test_c1_physical_rows_ignore_the_graph_bucket():
    supply = _profile_supply()
    # Valid head is one decode row. tp 2 pads every quant kernel to 2.
    # graph_bucket 1 is not that tensor.
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 4096, phase="decode"),),
        graph_bucket=1, phase="decode")
    for kind in ("F0", "A0", "A3"):
        cost = _compile(5120, supply, tp=2, kind=kind).evaluate(geometry)
        assert cost.get("valid_rows") == 1
        assert cost.get("physical_rows") == 2
        assert cost.get("launched_blocks") == 2
        mte2 = sum(dict(stage).get("mte2", 0) for stage in cost.stage_work)
        assert mte2 == 2 * _align(5120 * 2)
    prefill = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")
    padded = _compile(5120, supply, tp=2, kind="F0").evaluate(prefill)
    assert padded.get("valid_rows") == 4096
    assert padded.get("physical_rows") == 4096
    assert quant_row_demand(prefill, kind="F0", tp=2).source == ROW_SOURCE
