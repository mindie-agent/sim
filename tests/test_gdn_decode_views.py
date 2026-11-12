"""GDN Slice and ZerosLike launches follow the v0.30.0 materialization rule."""
from dataclasses import replace
from fractions import Fraction

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.gdn_decode_views import decode_views
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
import json

from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry, deployment_reject, lanes_from_bound_deployment,
    lanes_from_case,
)
from mindie_sim.prediction.piecewise.physical import ARCH3510_VECTOR_MIN_MS
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.builders_attention import _build_a0, _build_a3
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.schema.common import ModelingError
import pytest


def _ctx():
    model = load_model("qwen3.6-27b")
    layer = next(plan for plan in model.layer_plans if plan.linear_heads)
    features = FeatureConfig.from_model(model, tp=2, dp=4, ep=1)
    return model, layer, _Ctx(model, layer, features, stage=0, layers=len(model.layer_plans))


def _plan(layer):
    key_heads = (layer.linear_key_heads + 1) // 2
    value_heads = (layer.linear_value_heads + 1) // 2
    d_k = layer.linear_key_head_dim or layer.linear_head_dim
    d_v = layer.linear_head_dim
    return key_heads, value_heads, d_k, d_v


def test_decode_views_follow_the_packed_qkvz_layout():
    _model, layer, ctx = _ctx()
    built = _build_a3(ctx)
    key_heads, value_heads, d_k, d_v = _plan(layer)
    expected = decode_views(key_heads, value_heads, d_k, d_v)
    names = [op.name for op in built.template.operators]
    families = {op.name: op.family for op in built.template.operators}
    for item in expected:
        assert names.count(item.name) == 1
        assert families[item.name] == "copy"
        sig = built.signature(item.name)
        assert sig.role == item.role
        assert sig.find("source_kernel") == item.source_kernel
        assert sig.find("execution") == item.execution
        assert sig.find("transfer") == item.transfer
        assert sig.find("row_source") == item.row_source
        assert sig.find("owner_axis") == "sequence"
        assert sig.find("decode_only") in (None, 0)
        assert sig.require("elements") == item.elements
        assert sig.require("rank") == item.rank
        assert sig.require("source_extent") == item.source_extent
        assert sig.require("offset_elements") == item.offset_elements
        assert sig.find("read_bytes") == item.read_elements * 2
        assert sig.find("write_bytes") == item.write_elements * 2
    assert [item.name for item in expected] == [
        "beta_slice", "alpha_slice", "core_zeros", "qkv_slice",
        "q_slice", "k_slice", "v_slice", "z_slice",
    ]
    assert names.index("gated_delta") < names.index("z_slice") < names.index("gated_norm")
    query = key_heads * d_k
    value = value_heads * d_v
    packed = 2 * query + 2 * value
    qkv = 2 * query + value
    # Qwen3.6-27B TP=2: 64 = 8192/128 and 24 = 3072/128. The Z copy is the
    # trailing packed subview, not the 2375x24x128x128 recurrent state.
    assert (key_heads, value_heads, d_k, d_v) == (8, 24, 128, 128)
    assert packed == 8192 and qkv == 5120 and value == 3072
    z = built.signature("z_slice")
    assert z.require("rank") == 3
    assert z.require("source_heads") == packed // d_v == 64
    assert z.require("kept_heads") == value // d_v == 24
    assert z.require("head_dim") == d_v
    assert z.require("offset_elements") == qkv
    assert z.require("elements") == value
    assert z.require("offset_elements") + z.require("elements") == packed
    assert z.find("execution") == "copy"
    assert built.signature("core_zeros").find("execution") == "zero_fill"
    assert built.signature("core_zeros").find("read_bytes") == 0
    assert built.signature("q_slice").require("offset_elements") == 0
    assert built.signature("k_slice").require("offset_elements") == query
    assert built.signature("v_slice").require("offset_elements") == 2 * query


def test_projection_roles_stay_on_the_quant_binder():
    _model, _layer, ctx = _ctx()
    built = _build_a3(ctx)
    assert built.signature("qkvz_projection").role == "qkvz"
    assert built.signature("ba_projection").role == "ba"
    assert built.signature("o_projection").role == "output"
    assert all(item.role not in {"gdn", "gdn.o", "gdn.beta_alpha"}
               for item in decode_views(1, 1, 1, 1))


def test_views_absent_from_attention_and_chunked_prefill():
    model, _layer, ctx = _ctx()
    regular = next(plan for plan in model.layer_plans if not plan.linear_heads)
    attention = _build_a0(_Ctx(model, regular, ctx.features, stage=0,
                               layers=len(model.layer_plans)))
    prefill = _build_a3(replace(ctx, variant="chunked_prefill"))
    banned = {item.name for item in decode_views(1, 1, 1, 1)}
    for built in (attention, prefill):
        assert banned.isdisjoint(op.name for op in built.template.operators)


def test_decode_view_extent_rejects_non_positive_and_unaligned_z():
    with pytest.raises(ModelingError):
        decode_views(0, 1, 1, 1)
    with pytest.raises(ModelingError):
        decode_views(1, 1, 1, 3)


_FLOOR = Fraction(1, 4)
_SUPPLY = PipeSupply(
    (("cube", 1e18), ("vector", 1e18), ("hbm", 1e18),
     ("mte1", 1e18), ("mte2", 1e18), ("mte3", 1e18),
     ("fixpipe", 1e18), ("link", 1e18)),
    invocations=(("vector", float(_FLOOR)),),
)
_POLICY = ImplementationPolicy(mt=128, nt=16, cores=64, moe_row_tile=4,
                              split_target=1000, max_splits=4)
_SLICES = ("beta_slice", "alpha_slice", "qkv_slice", "q_slice", "k_slice",
           "v_slice", "z_slice")


def _compiled(name):
    model = load_model("qwen3.6-27b")
    features = FeatureConfig.from_model(
        model, tp=2, dp=4, ep=1, pp=1, token_tile=128, kv_tile=16,
        cores=64, moe_row_tile=4, split_target=1000, max_splits=4)
    pipe = assemble_model(model, supply=_SUPPLY, policy=_POLICY, features=features)
    a3 = pipe.instantiated.layers[0].find("A3")
    compiler = OperatorPiecewiseCompiler(_SUPPLY, _POLICY)
    return compiler.compile_operator(a3.operator(name), kind="A3")


def _geometry(*requests, phase, bucket=None, **lanes):
    tokens = sum(request.q_len for request in requests)
    return WorkloadGeometry(
        tuple(requests), graph_bucket=tokens if bucket is None else bucket,
        phase=phase, local_batch=len(requests), **lanes)


_DECLARED = dict(
    tp_width=2, dense_sequence_parallel=True, graph_dispatch="full_decode_only",
    capture_sizes=(1, 2, 4, 8, 16, 32),
)


def _rows(cost):
    return int(cost.get("copy_rows"))


def test_copies_materialize_only_on_the_source_row_condition():
    c1 = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode")
    c8 = _geometry(RequestGeometry(1, 4096, phase="decode"),
                   RequestGeometry(1, 4096, phase="decode"), phase="decode")
    mixed = _geometry(RequestGeometry(1, 4096, phase="decode"),
                      RequestGeometry(4096, 0, phase="prefill"), phase="decode")
    pure = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill")
    padded = _geometry(RequestGeometry(1, 4096, phase="decode"),
                       phase="decode", bucket=8)
    expected = {
        "beta_slice": (0, 2, 4097, 4096),
        "alpha_slice": (0, 2, 4097, 4096),
        "qkv_slice": (0, 2, 4096, 4096),
        "q_slice": (0, 0, 0, 0),
        "k_slice": (0, 0, 0, 0),
        "v_slice": (0, 0, 0, 0),
        "z_slice": (0, 2, 4097, 4096),
    }
    # q and k on two decode rows are the only late slices with rows in c8.
    expected["q_slice"] = (0, 2, 0, 0)
    expected["k_slice"] = (0, 2, 0, 0)
    expected["v_slice"] = (0, 2, 0, 0)
    for name in _SLICES:
        cost = _compiled(name)
        assert cost.family == "copy"
        seen = []
        for geometry in (c1, c8, mixed, pure):
            result = cost.evaluate(geometry)
            seen.append(_rows(result))
            if _rows(result) == 0:
                assert result.fidelity == "inactive"
                assert result.duration_ms == 0
                assert result.work == ()
            else:
                assert result.fidelity == "isolated_device_time"
                assert result.pipe_work("vector") == 0
                assert result.pipe_work("mte2") == _rows(result) * cost.operator.theta.require("read_bytes")
                assert result.pipe_work("mte3") == _rows(result) * cost.operator.theta.require("write_bytes")
                assert result.duration_ms < _FLOOR
        assert tuple(seen) == expected[name]
        held = cost.evaluate(padded)
        assert held.fidelity == "inactive"
        assert _rows(held) == 0
    zeros = _compiled("core_zeros")
    for geometry, rows in ((c1, 1), (c8, 2), (mixed, 4097), (pure, 4096), (padded, 1)):
        result = zeros.evaluate(geometry)
        assert result.fidelity == "isolated_device_time"
        assert _rows(result) == rows
        assert result.pipe_work("vector") > 0
        assert result.pipe_work("mte2") == 0
        assert result.duration_ms == _FLOOR
    prepare = _compiled("prepare")
    kept = prepare.evaluate(pure)
    assert kept.fidelity != "inactive"
    assert kept.duration_ms > 0


def _declared_decode(*requests, **overrides):
    fields = dict(_DECLARED)
    fields.update(overrides)
    return _geometry(*requests, phase="decode", **fields)


def test_declared_static_lanes_pad_operators_without_a_second_sequence():
    one = RequestGeometry(1, 4096, phase="decode")
    declared = _declared_decode(one)
    plain = _geometry(one, phase="decode")
    pair = _declared_decode(one, RequestGeometry(1, 4096, phase="decode"))
    prefill = _geometry(RequestGeometry(4096, 0, phase="prefill"), phase="prefill", **_DECLARED)
    mixed = _geometry(one, RequestGeometry(4096, 0, phase="prefill"), phase="decode", **_DECLARED)
    held = _geometry(one, phase="decode", bucket=8)
    eager = _geometry(one, phase="decode", tp_width=2, dense_sequence_parallel=True,
                      graph_dispatch="none", capture_sizes=(1, 2, 4, 8, 16, 32))
    narrow = _declared_decode(one, tp_width=1)
    off = _declared_decode(one, dense_sequence_parallel=False)
    uncovered = _declared_decode(one, capture_sizes=(1,))
    skipped = _declared_decode(one, capture_sizes=(1, 3, 4))
    bare = _declared_decode(one, capture_sizes=())
    bucketed = _geometry(one, phase="decode", bucket=8, **_DECLARED)

    lanes = declared.token_lanes()
    for key, value in lanes.items():
        assert declared.bindings()[key] == value
    assert lanes["lane_declaration"] == "static_dense_sp"
    assert lanes["static_decode"] is True
    assert lanes["scheduled_sequences"] == 1
    assert lanes["valid_tokens"] == 1
    assert lanes["head_tokens"] == 1 == declared.producing_lm_head_rows()
    assert lanes["gathered_tokens"] == lanes["embedding_tokens"] == lanes["mlp_tokens"] == 2
    assert lanes["shard_tokens"] == 1
    assert lanes["tp_rounded_tokens"] == 2
    assert len(declared.requests) == 1
    assert declared.requests[0].kv_len == 4097
    assert declared.graph_bucket == 1
    assert declared.effective_tokens == 1
    for name in (*_SLICES, "core_zeros"):
        assert _rows(_compiled(name).evaluate(declared)) == 2

    assert plain.token_lanes()["lane_declaration"] == "dense_sp_not_declared"
    assert plain.token_lanes()["gathered_tokens"] == 1
    assert plain.physical_gdn_rows() == (1, 1, 0)
    assert all(_rows(_compiled(name).evaluate(plain)) == 0 for name in _SLICES)
    assert _rows(_compiled("core_zeros").evaluate(plain)) == 1

    both = pair.token_lanes()
    assert both["scheduled_sequences"] == both["head_tokens"] == both["gathered_tokens"] == 2
    assert both["gathered_tokens"] not in (3, 4)
    assert all(_rows(_compiled(name).evaluate(pair)) == 2 for name in (*_SLICES, "core_zeros"))

    pf = prefill.token_lanes()
    assert pf["static_decode"] is False
    assert pf["lane_declaration"] == "not_uniform_decode"
    assert pf["head_tokens"] == 1
    assert pf["gathered_tokens"] == pf["mlp_tokens"] == 4096
    assert pf["shard_tokens"] == 2048
    assert prefill.physical_gdn_rows() == (4096, 0, 4096)
    for name in ("q_slice", "k_slice", "v_slice"):
        assert _rows(_compiled(name).evaluate(prefill)) == 0
    for name in ("beta_slice", "alpha_slice", "qkv_slice", "z_slice", "core_zeros"):
        assert _rows(_compiled(name).evaluate(prefill)) == 4096

    mx = mixed.token_lanes()
    assert mx["static_decode"] is False
    assert mx["gathered_tokens"] == 4097
    assert mx["embedding_tokens"] == mx["mlp_tokens"] == 4098
    assert mx["shard_tokens"] == 2049
    assert mx["head_tokens"] == mixed.producing_lm_head_rows()
    assert len(mixed.requests) == 2
    assert mixed.physical_gdn_rows() == (4097, 1, 4096)
    for name in ("q_slice", "k_slice", "v_slice"):
        assert _rows(_compiled(name).evaluate(mixed)) == 0
    for name in ("beta_slice", "alpha_slice", "z_slice", "core_zeros"):
        assert _rows(_compiled(name).evaluate(mixed)) == 4097
    assert _rows(_compiled("qkv_slice").evaluate(mixed)) == 4096

    assert held.token_lanes()["gathered_tokens"] == 1
    assert all(_rows(_compiled(name).evaluate(held)) == 0 for name in _SLICES)
    assert _rows(_compiled("core_zeros").evaluate(held)) == 1
    assert bucketed.graph_bucket == 8
    assert bucketed.effective_tokens == 8
    assert bucketed.token_lanes()["gathered_tokens"] == 2
    assert all(_rows(_compiled(name).evaluate(bucketed)) == 2 for name in _SLICES)

    eg = eager.token_lanes()
    assert eg["lane_declaration"] == "graph_dispatch_not_static"
    assert eg["static_decode"] is False
    assert eg["gathered_tokens"] == 1
    assert eg["embedding_tokens"] == eg["mlp_tokens"] == 2
    assert eg["shard_tokens"] == 1
    assert all(_rows(_compiled(name).evaluate(eager)) == 0 for name in _SLICES)
    assert _rows(_compiled("core_zeros").evaluate(eager)) == 1

    assert narrow.token_lanes()["lane_declaration"] == "tp_width_not_declared"
    assert narrow.token_lanes()["gathered_tokens"] == 1
    assert off.token_lanes()["lane_declaration"] == "dense_sp_not_declared"
    assert off.token_lanes()["gathered_tokens"] == 1
    assert all(_rows(_compiled(name).evaluate(off)) == 0 for name in _SLICES)
    assert uncovered.token_lanes()["lane_declaration"] == "no_capture_size_covers_tp_rounded_tokens"
    assert uncovered.token_lanes()["gathered_tokens"] == 1
    assert uncovered.token_lanes()["embedding_tokens"] == 2
    assert all(_rows(_compiled(name).evaluate(uncovered)) == 0 for name in _SLICES)
    assert skipped.token_lanes()["gathered_tokens"] == 4
    assert skipped.token_lanes()["lane_declaration"] == "static_dense_sp"
    assert bare.token_lanes()["gathered_tokens"] == 2
    assert bare.capture_sizes == ()


def test_case_adapter_keeps_an_undeclared_capture_logical():
    enrolled = {
        "runtime": {
            "tp": 8, "dp": 1, "ep": 1, "pp": 1, "graph_mode": "none",
            "graph_capture_mode": "full",
            "metadata": {
                "geometry_policy": (
                    "supply actual query rows and sequence count from profiler; "
                    "capture padding must be stated separately"),
            },
        },
    }
    fields = lanes_from_case(enrolled, tp=2, pp=1, capture_sizes=(1, 2, 4, 8, 16, 32))
    assert "dense_sequence_parallel" not in fields
    assert fields["graph_dispatch"] == "none"
    assert fields["tp_width"] == 2
    assert fields["capture_sizes"] == (1, 2, 4, 8, 16, 32)
    logical = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode", **fields)
    assert logical.token_lanes()["static_decode"] is False
    assert logical.token_lanes()["lane_declaration"] == "dense_sp_not_declared"
    assert logical.token_lanes()["gathered_tokens"] == 1
    assert logical.producing_lm_head_rows() == 1
    captured = {
        "model": {"hf_config": {"model_type": "qwen3_5_text"}},
        "runtime": {
            "graph_mode": "none",
            "metadata": {
                "source_model_type": "qwen3_5_text",
                "captured_execution_mode": "FULL_DECODE_ONLY",
                "additional_config": {"enable_flashcomm1": "1"},
            },
        },
    }
    admitted = lanes_from_case(captured, tp=2, pp=1, capture_sizes=(1, 2, 4, 8, 16, 32))
    assert admitted["dense_sequence_parallel"] is True
    assert admitted["graph_dispatch"] == "full_decode_only"
    padded = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode", **admitted)
    assert padded.token_lanes()["lane_declaration"] == "static_dense_sp"
    assert padded.token_lanes()["gathered_tokens"] == 2
    assert len(padded.requests) == 1
    other = dict(captured)
    other["model"] = {"hf_config": {"model_type": "qwen3_5_moe_text"}}
    other["runtime"] = {
        "metadata": {
            "source_model_type": "qwen3_5_moe_text",
            "captured_execution_mode": "FULL_DECODE_ONLY",
            "additional_config": {"enable_flashcomm1": "1"},
        },
    }
    refused = lanes_from_case(other, tp=2, pp=1, capture_sizes=(1, 2, 4, 8, 16, 32))
    assert refused["dense_sequence_parallel"] is False
    silent = lanes_from_case(
        {"runtime": {"graph_capture_mode": "full"}}, tp=2, pp=1)
    assert "graph_dispatch" not in silent


def test_declared_c1_copy_duration_follows_declared_bytes():
    supply = PipeSupply(
        (("cube", 865075200000), ("vector", 6758400000), ("hbm", 4000000000),
         ("mte1", 4224000000), ("mte2", 4224000000), ("mte3", 13516800000),
         ("fixpipe", 13516800000), ("link", 392000000)),
        invocations=(("vector", float(ARCH3510_VECTOR_MIN_MS)),),
    )
    declared = _declared_decode(RequestGeometry(1, 4096, phase="decode"))
    floor = as_fraction(ARCH3510_VECTOR_MIN_MS)
    hbm_rate = 4000000000
    for name in _SLICES:
        cost = _compiled_on(name, supply)
        result = cost.evaluate(declared)
        read = cost.operator.theta.require("read_bytes")
        write = cost.operator.theta.require("write_bytes")
        assert _rows(result) == 2
        assert result.pipe_work("vector") == 0
        assert result.pipe_work("hbm") == 2 * (read + write)
        assert result.dominant_pipe == "hbm"
        assert result.duration_ms == as_fraction(2 * (read + write)) / hbm_rate
        assert result.duration_ms < floor
    zeros = _compiled_on("core_zeros", supply).evaluate(declared)
    assert _rows(zeros) == 2
    assert zeros.pipe_work("mte2") == 0
    assert zeros.pipe_work("vector") > 0
    assert zeros.duration_ms == floor


def test_unlisted_status_withholds_the_lane_without_discarding_the_rows():
    raw = {
        "model": {"hf_config": {
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
        }},
        "runtime": {"graph_mode": "none", "metadata": {}},
    }
    sizes = [1, 2, 4, 8, 16, 32]
    status = {
        "spec": {
            "tp": 2, "dp": 4,
            "additional_config": {"enable_flashcomm1": True},
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": sizes,
            },
        },
        "compilation_used": json.dumps({
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sizes,
        }),
        "cells": [{"tag": "listed_decode", "rc": 0}],
    }
    listed = lanes_from_bound_deployment(
        raw, status, tp=2, dp=4, pp=1, capture_sizes=sizes, cell_id="listed_decode")
    one = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode", **listed)
    assert listed["graph_dispatch"] == "full_decode_only"
    assert listed["dense_sequence_parallel"] is True
    assert one.token_lanes()["scheduled_sequences"] == one.token_lanes()["head_tokens"] == 1
    assert one.token_lanes()["gathered_tokens"] == 2
    assert one.physical_gdn_rows() == (2, 2, 0)
    assert len(one.requests) == 1
    missed = lanes_from_bound_deployment(
        raw, status, tp=2, dp=4, pp=1, capture_sizes=sizes, cell_id="historical_decode")
    logical = _geometry(RequestGeometry(1, 4096, phase="decode"), phase="decode", **missed)
    assert deployment_reject(status, cell_id="historical_decode", tp=2, dp=4, pp=1) == "source_cell_not_in_deployment"
    assert missed.get("graph_dispatch") == "none"
    assert "dense_sequence_parallel" not in missed
    assert logical.physical_gdn_rows() == (1, 1, 0)
    clash = lanes_from_bound_deployment(
        raw, status, tp=4, dp=4, pp=1, capture_sizes=sizes, cell_id="listed_decode")
    assert deployment_reject(status, cell_id="listed_decode", tp=4, dp=4, pp=1) == "source_parallel_contradicts_deployment"
    assert clash.get("graph_dispatch") == "none"
    assert "dense_sequence_parallel" not in clash
    assert deployment_reject(None, cell_id="historical_decode", tp=2, dp=4, pp=1) == ""
    untouched = dict(raw)
    lanes_from_bound_deployment(
        untouched, status, tp=2, dp=4, pp=1, capture_sizes=sizes, cell_id="historical_decode")
    assert untouched == raw


def _compiled_on(name, supply):
    model = load_model("qwen3.6-27b")
    features = FeatureConfig.from_model(
        model, tp=2, dp=4, ep=1, pp=1, token_tile=128, kv_tile=16,
        cores=64, moe_row_tile=4, split_target=1000, max_splits=4)
    pipe = assemble_model(model, supply=supply, policy=_POLICY, features=features)
    a3 = pipe.instantiated.layers[0].find("A3")
    compiler = OperatorPiecewiseCompiler(supply, _POLICY)
    return compiler.compile_operator(a3.operator(name), kind="A3")
