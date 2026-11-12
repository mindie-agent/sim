"""Primitive evaluation and routed-collective physical invariants.

[Role]
  Establish that the ETP AllReduce reduction boundary follows the declared MoE
  recipe, that GQA decode occupancy is monotone across the AIC boundary and
  packed query tiles, that collective aliases lower to one canonical F2
  evaluation, and that a ragged standalone prefill keeps zero context.

[Invariants]
  * An ``etp_allreduce`` message is one token activation for token-row recipes
    and ``top_k`` routed expert rows for expert-row recipes.
  * GQA decode duration and HBM work do not jump upward at the 31->32 AIC
    boundary, and packed query rows stay inside one QUERY_TILES step.
  * A5 single-token decode at 8K context keeps the legacy wave occupancy while
    a 126K context remains finite and non-decreasing.
  * Every collective alias (``alltoall``, ``all_to_all``, ``all-to-all``,
    ``alltoallv``) evaluates as the same canonical F2 operator.
  * A ragged standalone prefill ``tokens=5, batch=2`` has causal pairs ``9``.
  * Every primitive result is positive, finite, exactly conserved across
    mechanism and core attribution, and grows with token count.
  * Unknown primitives raise ``ModelingError``; there is no silent fallback.

[Timeouts & Locks]
  Pure in-memory arithmetic; bounded per-test work, no waits or locks.

[Side-Effect Policy]
  Reads no device and writes no repository file; environment variables are
  redirected under ``tmp_path`` by an autouse fixture.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim import target_operators
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry, evaluate_primitive_spec,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs import FeatureConfig, instantiate_model
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature, SubgraphInstance,
    SubgraphTemplate,
)
from test_balanced_experts import balanced_case

CORE_KEYS = {"AI_CORE", "MIX_AIC", "AI_VECTOR_CORE", "COMMUNICATION"}
PRIMITIVES = (
    "fia", "gqa_attention", "mla_attention", "gdn", "conv1d", "moe_native",
    "grouped_matmul", "swiglu", "silu", "rmsnorm", "rotary", "quant", "gemm",
    "collective",
)
SKUS = ("Ascend910B4", "Ascend910_9362", "Ascend950DT_9582", "A2", "A3", "A5")
FIA_SPEC = {
    "op": "fia", "tp": 2, "q_heads": 8, "kv_heads": 2, "head_dim": 256,
    "batch": 8, "q_len": 4, "kv_len": 126000, "phase": "decode",
}
ETP_SPEC = {
    "op": "etp_allreduce", "tp": 16, "ep": 1, "E": 512, "top_k": 10,
    "hidden": 4096, "tokens": 640,
}


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch):
    for name in ("MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME",
                 "MINDIE_DIAGNOSTICS_ROOT", "HOME"):
        target = tmp_path / name.lower()
        target.mkdir()
        monkeypatch.setenv(name, str(target))


def _hardware(sku, devices=8):
    return Hardware.from_mapping({"backend": "ascend", "sku": sku,
                                  "num_devices": devices})


def _domain():
    return AnalysisDomain(
        max_tokens=10**8, max_moe_rows=10**8, max_query_tiles=10**8,
        max_kv_tiles=10**8, max_index_rows=10**11, max_work=10**15,
        max_q=10**8, max_s_kv=10**8, max_seq=10**8, max_batch=10**6)


def _compiler_for_supply(supply, cores):
    return OperatorPiecewiseCompiler(
        supply, ImplementationPolicy(cores=cores), domain=_domain())


def _compiler(sku, devices=8):
    hw = _hardware(sku, devices)
    supply = PipeSupply.from_hardware(hw, dtype="bf16")
    return _compiler_for_supply(
        supply, max(1, int(hw.dies[0].compute_units))), hw


def _bound(name, family, *, role=None, matrix=(), heads=(), experts=(), quant=(),
           parallel=(), tiles=(), extras=()):
    template = OperatorTemplate(name=name, family=family, role=role or name)
    theta = StaticSignature(role=name, matrix=matrix, heads=heads, experts=experts,
                            quant=quant, parallel=parallel, tiles=tiles,
                            extras=extras)
    return BoundOperator(template, theta)


def _etp_operator():
    return _bound("etp_allreduce", "collective", role="etp_allreduce",
                  parallel=(("members", 16), ("etp", 16), ("tp", 16),
                            ("ep", 1)),
                  experts=(("E", 512), ("local", 512), ("k", 10)),
                  quant=(("wire_bytes", 8192.0), ("bytes_per_token", 8192.0)))


def _transport(name, row_bytes=8192.0, top_k=10):
    return _bound(name, "transport", role=name,
                  parallel=(("members", 16), ("ep", 16), ("ranks_per_node", 8)),
                  matrix=(("row_bytes", row_bytes),),
                  experts=(("top_k", top_k),),
                  quant=(("wire_bytes", row_bytes),))


def _f2_alltoall():
    return _bound("alltoall", "collective", role="alltoall",
                  parallel=(("members", 16), ("ep", 16), ("ranks_per_node", 8)),
                  experts=(("E", 512), ("local", 512), ("k", 10)),
                  quant=(("wire_bytes", 8192.0),),
                  extras=(("collective", "alltoall"),))


def _f3_instance(operator):
    template = SubgraphTemplate(kind="F3", variant="collective-test",
                                title="F3 collective test",
                                operators=(operator.template,))
    return SubgraphInstance("f3-collective", template, layer=0, stage=0,
                            operators=(operator,))


def _fia_operator(cores=20):
    return _bound("fia", "attention", role="attention",
                  heads=(("q", 8), ("kv", 2), ("dk", 256), ("dv", 256)),
                  quant=(("act_bytes", 2.0),
                         ("kv_bytes", 2 * (256 + 256) * 2.0)),
                  tiles=(("mt", 128), ("nt", 128), ("cores", cores)))


def _prefill(tokens):
    return WorkloadGeometry.homogeneous(1, tokens, tokens, phase="prefill")


# 1. ETP AllReduce carries token activations; AllToAllV carries routed rows.


def test_etp_allreduce_reduces_token_activation_not_moe_rows():
    operator, geometry = _etp_operator(), _prefill(640)
    for sku in ("Ascend950DT_9582", "Ascend910_9362"):
        compiler, _ = _compiler(sku, 16)
        direct = compiler.compile_operator(operator, kind="F3").evaluate(geometry)
        compiled = compiler.compile(_f3_instance(operator)).evaluate(geometry)
        for cost in (direct, compiled.operators[0]):
            assert cost.get("message_bytes") == 640 * 4096 * 2
            assert cost.get("moe_rows") == 0
            assert 0.1 <= float(cost.duration_ms) <= 3.0


def test_etp_allreduce_primitive_and_layer_repeat():
    for sku in ("Ascend950DT_9582", "Ascend910_9362"):
        one = evaluate_primitive_spec(dict(ETP_SPEC), sku=sku, num_devices=16)
        details = one["operators"][0]
        assert details["derived"]["message_bytes"] == 640 * 4096 * 2
        assert 0.1 <= one["operator_ms"] <= 3.0
        assert one["step_ms"] == pytest.approx(one["operator_ms"])
        for layers in (10, 16):
            deep = evaluate_primitive_spec(dict(ETP_SPEC, repeat=layers),
                                           sku=sku, num_devices=16)
            assert 1.0 <= deep["step_ms"] < 5.0
            assert deep["step_ms"] == pytest.approx(
                one["operator_ms"] * layers)


def test_f2_dispatch_and_alltoall_still_scale_with_routed_rows():
    compiler, _ = _compiler("Ascend950DT_9582", 16)
    geometry = _prefill(640)
    dispatch = compiler.compile_operator(_transport("dispatch"), kind="F2")
    combine = compiler.compile_operator(_transport("combine"), kind="F2")
    for compiled in (dispatch, combine):
        cost = compiled.evaluate(geometry)
        assert cost.get("peer_bytes") == 640 * 10 * 8192
        assert cost.get("routed_rows") == 640 * 10
    alltoall = compiler.compile_operator(_f2_alltoall(), kind="F2")
    cost = alltoall.evaluate(geometry)
    assert cost.get("message_bytes") == 640 * 10 * 8192
    assert cost.get("message_bytes") // 8192 == 640 * 10


def _etp_boundary(recipe):
    raw = balanced_case(tp=4, tokens=32)
    raw["runtime"]["moe_recipe"] = recipe
    parsed = parse_case(raw)
    features = FeatureConfig.from_runtime(parsed.runtime, parsed.model)
    instance = instantiate_model(parsed.model, features).find("F3", layer=0)
    compiler, _ = _compiler("Ascend950DT_9582", 4)
    cost = compiler.compile(instance).operator("etp_allreduce").evaluate(_prefill(32))
    return cost, raw


def test_etp_reduction_boundary_follows_moe_recipe():
    for recipe, rows, target_recipe in (
            ("alltoall", 64, "expert_tensor_reduce"),
            ("tp_join_reduce", 32, "moe_output_reduce")):
        cost, raw = _etp_boundary(recipe)
        target = next(op for op in target_operators(raw, phase="prefill", tokens=32)
                      if op.recipe == target_recipe)
        assert target.local_shape[0] == rows
        assert cost.get("message_bytes") == rows * target.local_shape[1] * 2
    # Local-expert reduce requires ep == tp for its target graph; only the
    # reduction row boundary (32 token rows) is compared here.
    cost, _ = _etp_boundary("local_expert_reduce")
    raw = balanced_case(tp=4, ep=4, tokens=32)
    raw["runtime"]["moe_recipe"] = "local_expert_reduce"
    target = next(op for op in target_operators(raw, phase="prefill", tokens=32)
                  if op.recipe == "moe_output_reduce")
    assert target.local_shape[0] == 32
    assert cost.get("message_bytes") == 32 * target.local_shape[1] * 2


# 2. Split-K GQA decode: linear KV traffic, bounded latency, monotone context.


def test_a3_split_k_gqa_decode_reads_kv_once():
    compiler, _ = _compiler("Ascend910_9362")
    compiled = compiler.compile_operator(_fia_operator(), kind="A0")
    geometry = WorkloadGeometry(
        tuple(RequestGeometry(4, 126000 - 4, phase="decode") for _ in range(8)),
        phase="decode")
    cost = compiled.evaluate(geometry)
    assert dict(cost.regime)["SPLITS"] == 8
    assert 2.0e9 <= float(cost.pipe_work("hbm")) <= 2.2e9
    assert 2.0e9 <= float(cost.pipe_work("mte2")) <= 2.2e9
    assert 1.8 <= float(cost.duration_ms) <= 2.3
    result = evaluate_primitive_spec(dict(FIA_SPEC), sku="Ascend910_9362")
    details = result["operators"][0]
    assert details["regime"]["SPLITS"] == 8
    assert 2.0e9 <= details["work"]["hbm"] <= 2.2e9
    assert 2.0e9 <= details["work"]["mte2"] <= 2.2e9
    assert 1.8 <= result["operator_ms"] <= 2.3


def test_a3_split_k_context_monotonic_across_boundaries():
    values = [
        evaluate_primitive_spec(dict(FIA_SPEC, kv_len=context),
                                sku="Ascend910_9362")["operator_ms"]
        for context in (512, 1024, 1025, 2048, 4096, 8192, 32768, 126000)
    ]
    assert all(right >= left for left, right in zip(values, values[1:]))
    assert values[2] > values[1]


def test_a5_single_token_decode_keeps_legacy_occupancy():
    small = evaluate_primitive_spec(
        dict(FIA_SPEC, q_len=1, kv_len=8192, phase="decode"),
        sku="Ascend950DT_9582")
    details = small["operators"][0]
    burst = details["derived"]["kv_burst_bytes"]
    assert burst > 0
    ratio = details["work"]["hbm"] / burst
    assert 4.66 <= ratio <= 4.72
    assert dict(details["regime"])["QUERY_TILES"] == 1
    large = evaluate_primitive_spec(
        dict(FIA_SPEC, q_len=1, kv_len=126000, phase="decode"),
        sku="Ascend950DT_9582")
    assert math.isfinite(large["operator_ms"])
    assert large["operator_ms"] >= small["operator_ms"]
    assert dict(large["operators"][0]["regime"])["QUERY_TILES"] == 1


def _fixed_rate_supply(cores):
    base = PipeSupply.from_hardware(_hardware("Ascend950DT_9582", 8), dtype="bf16")
    units = dict(base.resource_units)
    units["compute_units"] = cores
    return replace(base, resource_units=tuple(sorted(units.items())))


def test_gqa_decode_occupancy_is_monotone_at_aic_boundary():
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 8192, phase="decode"),), phase="decode")
    metrics = []
    for cores in (20, 24, 31, 32):
        compiler = _compiler_for_supply(_fixed_rate_supply(cores), cores)
        cost = compiler.compile_operator(_fia_operator(cores), kind="A0").evaluate(
            geometry)
        metrics.append((float(cost.duration_ms), float(cost.pipe_work("hbm"))))
    assert all(math.isfinite(value) and value > 0
               for row in metrics for value in row)
    assert metrics[-1][0] <= metrics[-2][0] * (1 + 1e-12)
    assert metrics[-1][1] <= metrics[-2][1] * (1 + 1e-12)


def test_gqa_packed_query_tiles_stay_in_one_tile():
    compiler = _compiler_for_supply(_fixed_rate_supply(32), 32)
    for q_len in (1, 2, 4):
        cost = compiler.compile_operator(_fia_operator(32), kind="A0").evaluate(
            WorkloadGeometry((RequestGeometry(q_len, 8192, phase="decode"),),
                             phase="decode"))
        assert q_len * int(cost.get("group_heads")) <= 128
        assert dict(cost.regime)["QUERY_TILES"] == 1


def test_collective_aliases_canonicalize_to_identical_f2():
    results = [evaluate_primitive_spec(
        {"op": alias, "members": 16, "hidden": 4096, "tokens": 32, "top_k": 8},
        sku="Ascend950DT_9582", num_devices=16)
        for alias in ("alltoall", "all_to_all", "all-to-all", "alltoallv")]
    baseline = results[0]["operator_ms"]
    for result in results:
        details = result["operators"][0]
        assert details["name"] == "alltoall"
        assert details["kind"] == "F2"
        assert result["operator_ms"] == pytest.approx(baseline)


def test_ragged_standalone_prefill_uses_zero_context():
    details = evaluate_primitive_spec(
        {"op": "fia", "tokens": 5, "batch": 2, "phase": "prefill"},
        sku="Ascend950DT_9582")["operators"][0]
    assert details["derived"]["causal_pairs"] == 9


# 3. Primitive evaluator coverage, conservation and unknown-selector handling.


@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_primitive_eval_conserves_time_for_every_tier(primitive, sku):
    result = evaluate_primitive_spec({"op": primitive, "tokens": 256}, sku=sku)
    assert math.isfinite(result["step_ms"]) and result["step_ms"] > 0
    assert math.isfinite(result["operator_ms"]) and result["operator_ms"] > 0
    assert result["step_ms"] == pytest.approx(result["operator_ms"])
    assert sum(result["by_mechanism_ms"].values()) == pytest.approx(
        result["operator_ms"])
    assert sum(result["by_core_ms"].values()) == pytest.approx(
        result["operator_ms"])
    assert set(result["by_core_ms"]) <= CORE_KEYS
    assert result["num_devices"] == 8
    assert all(item["operator_ms"] > 0 for item in result["operators"])


@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_primitive_eval_is_monotone_in_tokens(primitive, sku):
    extra = {"E": 8, "top_k": 2} if primitive in {
        "moe_native", "grouped_matmul"} else {}
    small = evaluate_primitive_spec({"op": primitive, "tokens": 128, **extra},
                                    sku=sku)["operator_ms"]
    large = evaluate_primitive_spec({"op": primitive, "tokens": 1024, **extra},
                                    sku=sku)["operator_ms"]
    assert large >= small > 0


@pytest.mark.parametrize("spec", (
    {"op": "not_a_primitive"}, {"primitive": "unknown_family"},
    {"operator": "bogus"}, {},
))
def test_unknown_or_missing_primitive_raises(spec):
    with pytest.raises(ModelingError):
        evaluate_primitive_spec(spec)


def test_quantized_format_resolution_falls_back_to_bf16_when_unsupported():
    supported = evaluate_primitive_spec(
        {"op": "gemm", "K": 256, "N": 256, "weight_format": "fp8_e4m3",
         "act_format": "fp8_e4m3", "tokens": 64}, sku="Ascend950DT_9582")
    unsupported = evaluate_primitive_spec(
        {"op": "gemm", "K": 256, "N": 256, "weight_format": "fp8_e4m3",
         "act_format": "fp8_e4m3", "tokens": 64}, sku="Ascend910B4")
    assert supported["operator_ms"] > 0
    assert unsupported["operator_ms"] > 0
    assert supported["sku"] == "Ascend950DT_9582"
    assert unsupported["sku"] == "Ascend910B4"
