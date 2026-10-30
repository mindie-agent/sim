"""Explicit arch22 MLA Prolog source phases through the public paths."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mla_prolog_arch22 import (
    CHUNK, PROFILE, describe, geometry, owners, phase_io, source_chunk_work,
    selected_plan, validate_declared_view,
)
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions, recipes
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure


ROOT = Path(__file__).resolve().parents[1]
CASE = "glm53-a3-w4a8"


def _raw():
    raw = json.loads((ROOT / "examples/cases" / (CASE + ".json")).read_text())
    lowerings = raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault("lowerings", {})
    lowerings.update({
        "mla_prolog": "vllm_ascend_mla_prolog_v3",
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["mla_prolog"], "traffic_profile": PROFILE},
        },
    })
    return raw


def _case(*, small=False):
    case = parse_case(_raw())
    plan = case.model.layer_plans[0]
    if small:
        plan = replace(
            plan, index=0, num_heads=8, q_lora_rank=1536, head_dim=192,
            qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512)
        model = replace(
            case.model, layers=1, hidden_size=1024, num_heads=8, head_dim=192,
            num_kv_heads=1, layer_plans=(plan,))
        parallel = replace(
            case.runtime.parallel, dp=1, tp=1, ep=1, pp=1, cp=1, dcp=1,
            pcp=1, cp_plan="auto")
        runtime = replace(case.runtime, pp_layer_partition=(1,), parallel=parallel)
    else:
        plan = replace(plan, index=0)
        model = replace(case.model, layers=1, layer_plans=(plan,))
        runtime = replace(case.runtime, pp_layer_partition=(1,))
    return replace(case, model=model, runtime=runtime)


def _group(case, tokens):
    ops = target_operators(case, phase="prefill", tokens=tokens, resource_fusion=False)
    declaration, members = next(
        (declaration, members) for declaration, members in _groups(ops)
        if declaration["role"] == "mla_prolog")
    return ops, declaration, members


def _fused(case, tokens):
    ops, declaration, members = _group(case, tokens)
    fused = next(op for op in lower_native_fusions(ops, case.runtime)
                 if op.family == "native_fusion")
    device = case.machine.dies[0]
    geometry_value = Geometry(
        device.compute_units, {"vector": device.units("vector")},
        dict(device.buffers), dict(device.buffer_units))
    fused, _ = recipes()[fused.recipe].prepare(fused, geometry_value)
    return declaration, members, fused, geometry_value


def _independent_scratch(tokens, *, hcq=2048, hckv=512, rope=64, heads=8, nope=192):
    width = nope + rope
    return {
        "cq_int32_write": tokens * hcq * 4,
        "cq_int32_read": tokens * hcq * 4,
        "ckvkr_int32_write": tokens * (hckv + rope) * 4,
        "ckv_int32_read": tokens * hckv * 4,
        "kr_int32_read": tokens * rope * 4,
        "qcqr_int32_write": tokens * heads * width * 4,
        "qc_int32_read": tokens * heads * nope * 4,
        "qr_int32_read": tokens * heads * rope * 4,
        "qc_bf16_write": tokens * heads * nope * 2,
        "qc_bf16_read": tokens * heads * nope * 2,
        "cq_scale_write": tokens * 32,
        "qnorm_scale_write": tokens * 4,
        "cq_scale_reads": tokens * 32 * 40,
    }


def test_source_profile_executes_exact_work_through_operator_and_cohort_paths():
    case = _case()
    declaration, members, fused, device_geometry = _fused(case, 128)
    view = declaration["tensor_view"]
    plan = selected_plan(fused, device_geometry)
    expected = selected_plan(fused, device_geometry).work
    source = dict(fused.metadata)["fusion_boundary"]["source_program"]
    assert fused.hbm_bytes == pytest.approx(expected.resources["hbm"])
    assert fused.vector_ops == pytest.approx(expected.resources["vector"])
    assert dict(fused.matrix_flops) == expected.matrix
    assert fused.flops == pytest.approx(sum(expected.matrix.values()))
    assert source["program"]["work"]["hbm"] == pytest.approx(expected.resources["hbm"])
    assert source["program"]["work"]["matrix"] == expected.matrix

    # Entirely public operator pricing: one selected source program.
    report = operator_costs(case, (fused,), input_ready_ms=17)
    row = report["operators"][0]
    assert all(abs(value) < 1e-6 for value in row["service_delta"].values())
    assert row["service_before_input"] == {}
    assert row["operator"]["hbm_bytes"] == pytest.approx(expected.resources["hbm"])
    assert row["operator"]["vector_ops"] == pytest.approx(expected.resources["vector"])
    assert dict(row["operator"]["matrix_flops"]) == expected.matrix
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(expected.resources["hbm"])
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(expected.resources["vector"])
    assert row["live_tasks"] == 0
    assert not any(row["remaining_buffers"].values())

    # The same plan is used by the ordinary graph path, including one launch.
    result = graph_costs(case, {0: (fused,)}, include_submission=True,
                         external_inputs={dep: 0 for dep in fused.deps},
                         isolated=False, trace_limit=1000)
    execution = result["execution"]
    assert execution["serviced"]["memory@r0:s0"] == pytest.approx(expected.resources["hbm"])
    assert execution["serviced"]["vector@r0:s0"] == pytest.approx(expected.resources["vector"])
    assert execution["serviced"]["host@r0:s0"] == pytest.approx(
        case.machine.dies[0].launch_us / 1000)
    submits = [event for event in result["events"]
               if event["task"].endswith(f"{fused.op_id}:submit")]
    assert len(submits) == 1
    assert execution["live_tasks"] == execution["live_programs"] == 0

    # Cohort pricing resolves the same fused recipe, not member metadata.
    cohort = operator_costs(case, {0: (fused,)}, input_ready_ms=0)
    assert cohort["boundary"] == "sum_of_isolated_device_programs"
    assert cohort["operators"][0]["serviced"]["memory@r0:s0"] == pytest.approx(
        expected.resources["hbm"])
    assert source_chunk_work(view)["qn"].matrix == {"bf16": 2 * 128 * 8 * 192 * 512}


def test_int32_scratch_write_read_deltas_and_128_129_257_tails():
    for total, chunks in ((128, (128,)), (129, (128, 1)), (257, (128, 128, 1))):
        case = _case()
        declaration, _, fused, device_geometry = _fused(case, total)
        view = declaration["tensor_view"]
        g = geometry(view, device_geometry)
        assert tuple(g.chunk_sizes) == chunks
        assert g.tokens == total

        seen = {key: 0 for key in _independent_scratch(1)}
        for index, tokens in enumerate(g.chunk_sizes):
            io = phase_io(view, tokens, g=g, initial=index == 0)
            expected = _independent_scratch(tokens, hcq=g.hcq, hckv=g.hckv,
                                            rope=g.rope, heads=g.heads, nope=g.nope)
            # These are exactly the source-produced/consumed INT32/BF16 GM
            # edges, separated from operand and scale traffic.
            assert io["cq"]["write"] == expected["cq_int32_write"]
            assert io["ckvkr"]["write"] == expected["ckvkr_int32_write"]
            assert io["qcqr"]["write"] == expected["qcqr_int32_write"]
            assert io["rms_cq"]["read"] - tokens * 4 == \
                expected["cq_int32_read"]
            assert io["rms_ckv"]["read"] - (
                tokens * 4 + tokens * g.rope * 4 + tokens * 8) == \
                expected["ckv_int32_read"]
            assert io["rope_kr"]["read"] - tokens * 8 == \
                expected["kr_int32_read"]
            assert io["dequant_qc"]["read"] - g.heads * g.nope * 4 == expected["qc_int32_read"]
            assert io["rope_qr"]["read"] - g.heads * g.rope * 8 == expected["qr_int32_read"]
            assert io["dequant_qc"]["write"] == expected["qc_bf16_write"]
            assert io["qn"]["read"] - g.heads * g.nope * g.hckv * 2 == \
                expected["qc_bf16_read"]
            assert io["rms_cq"]["write"] - (
                tokens * g.hcq + tokens * 32 +
                (expected["qnorm_scale_write"] if view["query_norm_output"] else 0)) == 0
            assert io["scale_gather"]["read"] == expected["cq_scale_reads"]
            assert io["cq"]["read"] == tokens * g.hidden + g.hidden * g.hcq
            assert ("params" in io) == (index == 0)
            for key in seen:
                seen[key] += expected[key]

        plan = selected_plan(fused, device_geometry)
        expected_hbm = sum(
            sum(work.resources.get("hbm", 0)
                for work in source_chunk_work(view, tokens, g=g, initial=index == 0).values())
            for index, tokens in enumerate(chunks))
        assert plan.work.resources["hbm"] == pytest.approx(expected_hbm)
        expected_matrix = {}
        for index, tokens in enumerate(chunks):
            for work in source_chunk_work(view, tokens, g=g, initial=index == 0).values():
                for key, value in work.matrix.items():
                    expected_matrix[key] = expected_matrix.get(key, 0) + value
        assert plan.work.matrix == expected_matrix
        full = _independent_scratch(total, hcq=g.hcq, hckv=g.hckv,
                                    rope=g.rope, heads=g.heads, nope=g.nope)
        assert seen == full
        assert describe(view, g)["total"]["hbm"] == pytest.approx(
            plan.work.resources["hbm"])
        assert describe(view, g)["chunks"] == len(chunks)


def test_perstage_active_owners_track_heads_and_two_device_geometries():
    case = _case()
    declaration, _, _, _ = _fused(case, 128)
    base = dict(declaration["tensor_view"])
    geometries = (
        (Geometry(20, {"vector": 40}), 8, (16, 9, 8, 8, 40, 16)),
        (Geometry(20, {"vector": 40}), 16, (16, 9, 16, 16, 40, 32)),
        (Geometry(24, {"vector": 48}), 8, (22, 9, 8, 8, 48, 16)),
        (Geometry(24, {"vector": 48}), 16, (22, 9, 16, 16, 48, 32)),
    )
    for device_geometry, heads, expected in geometries:
        view = {**base, "heads": heads}
        g = geometry(view, device_geometry)
        ledger = describe(view, g)["owners"]
        actual = (
            len(ledger["cq"]["aic"]), len(ledger["ckvkr"]["aic"]),
            len(ledger["qcqr"]["aic"]), len(ledger["qn"]["aic"]),
            ledger["vector"]["aiv"], len(ledger["qc_dequant_vector"]["aiv"]),
        )
        assert actual == expected
        # A 1-token tail owns one AIV, never a whole-device reservation.
        assert owners(g, 1)["rms"] == (0,)
        assert owners(g, 32)["rms"] == tuple(range(32))
        assert phase_io(view, 1)["rms_cq"]["vector"] > 0


def test_delayed_input_readiness_and_external_query_norm_live_producer():
    case = _case()
    declaration, _, fused, _ = _fused(case, 129)
    metadata = dict(fused.metadata)
    boundary = metadata["fusion_boundary"]
    source = boundary["source_program"]
    assert metadata["fusion_source_profile"] == PROFILE
    assert source["selected_domain"]["heads"] == 8
    assert source["source"]["revision"] == "a3925fd49cd256ad478d2faf983bf2961b8ae2c8"
    assert source["source"]["program"]["sha256"].startswith("ea32ad7c8e4af216")
    assert source["branch"]["kv_cache_quant_mode"] == "NO_QUANT"
    assert source["branch"]["duration_used"] is False
    assert source["program"]["chunks"] == 2
    assert source["program"]["chunk_runs"] == [{"tokens": 128, "count": 1},
                                              {"tokens": 1, "count": 1}]
    assert source["program"]["launch_count"] == 1

    # The int8 normalized-query producer is an external group output
    # consumed by the indexer; it is the dynamic-quantized q_norm result.
    qnorm = next(member for member in source["program"]["fusion_semantic_ids"]
                 if member.endswith(".q_b.quantize"))
    assert qnorm in source["program"]["fusion_external_outputs"]
    recipe = next(item for item in source["program"]["fusion_semantic_recipes"]
                  if item["op_id"] == qnorm)
    assert recipe["recipe"] == "dynamic_quantize"
    consumer = next(op for op in target_operators(case, phase="prefill", tokens=129)
                    if op.op_id.endswith(".index_q"))
    assert fused.op_id in consumer.deps

    report = operator_costs(case, (fused,), input_ready_ms=23)
    row = report["operators"][0]
    assert row["service_before_input"] == {}
    assert row["input_ready_ms"] == 23
    assert row["duration_ms"] > 0
    assert row["live_tasks"] == 0


def test_small_geometry_whole_graph_uses_source_and_clears_all_live_state():
    case = _case(small=True)
    ops = target_operators(case, phase="prefill", tokens=32)
    selected = [op for op in ops if op.family == "native_fusion"]
    assert len(selected) == 1
    fused = selected[0]
    view = dict(fused.metadata)["fusion_tensor_view"]
    assert view["hidden"] == 1024 and view["query_rank"] == 1536
    assert dict(fused.metadata)["fusion_source_profile"] == PROFILE

    ids = {op.op_id for op in ops}
    external = {dep: 0 for op in ops for dep in op.deps if dep not in ids}
    result = graph_costs(case, {0: ops}, include_submission=False,
                         external_inputs=external, isolated=False)
    execution = result["execution"]
    assert result["duration_ms"] > 0
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert execution["serviced"]["memory@r0:s0"] > 0
    assert execution["serviced"]["vector@r0:s0"] > 0

    manifest = export_structure(case, phase="prefill", tokens=32, scope="body")
    group = next(item for item in manifest["subgraphs"] if item["role"] == "mla_prolog")
    assert group["physical_work"]["launches"] == 1
    assert group["resource_boundary"]["source_program"]["profile"] == PROFILE
    assert group["resource_boundary"]["source_program"]["selected_domain"]["hidden"] == 1024
    device = case.machine.dies[0]
    independent = selected_plan(fused, Geometry(
        device.compute_units, {"vector": device.units("vector")},
        dict(device.buffers), dict(device.buffer_units))).work
    assert group["physical_work"]["hbm_bytes"] == pytest.approx(
        independent.resources["hbm"])
    assert group["physical_work"]["vector_ops"] == pytest.approx(
        independent.resources["vector"])
    assert group["physical_work"]["matrix_flops"] == independent.matrix


def test_selected_profile_rejects_other_quant_cache_and_dequant_opt_branches():
    case = _case()
    declaration, _, _, _ = _fused(case, 128)
    view = declaration["tensor_view"]
    with pytest.raises(ValueError, match="int8_token_projection"):
        validate_declared_view({**view, "cache_dtype": "int8"})
    with pytest.raises(ValueError, match="enableDequantOpt_n_ge_8"):
        validate_declared_view({**view, "heads": 4})
    with pytest.raises(ValueError, match="int8_token_projection"):
        validate_declared_view({**view, "projection_weight_dtypes": ("int8", "bf16", "int8")})

    # Default semantic lowering is untouched when no native-fusion profile is
    # selected; it still declares the standalone BF16 UK/UV group.
    plain = json.loads((ROOT / "examples/cases" / (CASE + ".json")).read_text())
    plain_case = parse_case(plain)
    semantic = target_operators(plain_case, phase="prefill", tokens=8,
                                resource_fusion=False)
    assert not any(op.family == "native_fusion" for op in semantic)
    assert any(op.recipe == "q_absorb" for op in semantic)


def test_source_aiv_order_and_odd_token_rope_ownership_execute_on_engine():
    case = _case()
    _, _, fused, _ = _fused(case, 1)
    report = graph_costs(
        case, {0: (fused,)}, external_inputs={dep: 11 for dep in fused.deps},
        isolated=False, trace_limit=2000)
    events = report["events"]
    children = lambda name: [e for e in events
                            if f":tile0:{name}:program:" in e["task"] and e["demand"]]
    for predecessor, consumer in (("rope_kr", "dequant_qc"),
                                  ("dequant_qc", "rope_qr"),
                                  ("scale_gather", "qcqr")):
        assert min(e["start_ms"] for e in children(consumer)) >= \
            max(e["end_ms"] for e in children(predecessor))
    # Source uses (M + subBlockIdx)/2: at M=1 only odd AIVs do Qr work.
    assert {next(iter(e["placement"].values())) for e in children("rope_qr")} == \
        {(index,) for index in range(1, 16, 2)}
    assert all(e["start_ms"] >= 11 for e in events if e["demand"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_source_metadata_and_arithmetic_remain_bounded_for_many_chunks():
    case = _case()
    declaration, _, _, device = _fused(case, 128)
    view = {**declaration["tensor_view"], "tokens": 128 * 10_000_000 + 1}
    g = geometry(view, device)
    ledger = describe(view, g)
    assert g.tokens_at(10_000_000) == 1
    assert len(ledger["chunk_runs"]) == 3
    assert len(ledger["phases"]) < 40
    assert len(json.dumps(ledger)) < 30_000
    # A single initial parameter upload, not 10M copies of each AIV's gamma.
    parameter = [p for p in ledger["phases"] if p["name"] == "params"]
    assert len(parameter) == 1 and parameter[0]["chunk_count"] == 1
    assert parameter[0]["hbm"] == 40 * (2048 * 6 + 512 * 2 + 576 * 4)
    assert ledger["total"]["matrix"]["bf16"] == 2 * view["tokens"] * 8 * 192 * 512
    with pytest.raises(ValueError, match="full_quant_requires_aiv_aic_ratio_2"):
        geometry(view, Geometry(20, {"vector": 20}))


def test_repeated_source_groups_have_one_physical_identity_per_tensor_contract():
    case = _case(small=True)
    plan = case.model.layer_plans[0]
    case = replace(
        case, model=replace(case.model, layers=2,
                            layer_plans=(plan, replace(plan, index=1))),
        runtime=replace(case.runtime, pp_layer_partition=(2,)))
    manifest = export_structure(case, phase="prefill", tokens=1, scope="body")
    groups = [group for group in manifest["subgraphs"] if group["role"] == "mla_prolog"]
    assert len(groups) == 2
    assert groups[0]["subgraph_id"] != groups[1]["subgraph_id"]
    assert groups[0]["physical_work"] == groups[1]["physical_work"]
    assert groups[0]["physical_signature"] == groups[1]["physical_signature"]
