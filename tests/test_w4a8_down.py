"""Ordinary A8W4 MSD grouped-down checks against an independent work oracle."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.w4a8_down import RECIPE, lower_op
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/source-families/w4a8-down-msd-control.json"
GATE_RECIPE = "ascend_w4a8_msd"


def _raw(**overrides):
    raw = json.loads(CASE.read_text())
    for key, value in overrides.items():
        if key == "routing":
            raw["runtime"]["routing"] = value
        elif key == "remove_int4":
            raw["machine"]["device"]["compute_peaks"].pop("int4", None)
        elif key == "remove_down_binding":
            raw["runtime"]["kernels"]["lowerings"].pop("grouped_gemm")
        elif key == "remove_gate_profile":
            raw["runtime"]["kernels"]["lowerings"].pop("native_fusion")
        elif key == "fixpipe_rate":
            raw["machine"]["device"]["rates"]["fixpipe"] = value
        else:
            raise AssertionError(key)
    return raw


def _ops(raw, *, resource_fusion=False):
    return target_operators(parse_case(raw), phase="prefill", tokens=4,
                            resource_fusion=resource_fusion)


def _down(ops):
    return next(op for op in ops if op.recipe == RECIPE)


def _oracle(m, n, k, rows):
    active = sum(row > 0 for row in rows)
    # This four-expert control selects CV11: four physical AIVs each copy
    # and reduce all four group counts before their strided input rows.
    metadata_bytes = 4 * len(rows) * 8
    metadata_vector = 4 * 2 * len(rows)
    pre_read, pre_write = m * k + metadata_bytes, m * k
    matrix_read = m * k + 0.5 * n * k * active + 8 * n * active
    matrix_write = 4 * m * n
    post_read = 4 * m * n + 4 * n * active + 4 * m
    out_write = 2 * m * n
    return {
        "flops": 4 * m * n * k,
        "hbm_bytes": (pre_read + pre_write + matrix_read + matrix_write
                      + post_read + out_write),
        "mte2_bytes": pre_read + matrix_read + post_read,
        "mte3_bytes": pre_write + out_write,
        "vector_ops": (6.5 * m * k + 2 * m * n + 6 * m * n + m * n
                       + metadata_vector),
        "weight_bytes": 0.5 * n * k * active,
        "encoded_scale_bytes": 8 * n * active,
    }


def test_ordinary_binding_keeps_input_quantizer_and_bf16_output():
    raw = _raw()
    ops = _ops(raw)
    down = _down(ops)
    by_id = {op.op_id: op for op in ops}
    quant = by_id[down.deps[0]]
    metadata = dict(down.metadata)

    assert down.source_boundary == "kernel"
    assert down.dtype == "int4"
    assert dict(down.matrix_flops) == {"int4": 4 * 8 * 64 * 64}
    assert metadata["output_format"] == "bf16"
    assert metadata["w4a8_msd_down"]["output_quantization"] is False
    assert quant.op_id == down.op_id + ".quantize"
    assert quant.family == "quantize" and quant.recipe == "dynamic_quantize"
    assert quant.deps == ("s0.l0.expert_act",)
    assert not [op for op in ops if op.family == "quantize" and down.op_id in op.deps]

    semantic_raw = _raw(remove_down_binding=True)
    semantic = next(op for op in _ops(semantic_raw)
                    if op.recipe == "grouped_down")
    for field in ("op_id", "family", "phase", "stage", "layer", "requests",
                  "semantic_shape", "stage", "stream"):
        assert getattr(down, field) == getattr(semantic, field)
    lowered_meta = dict(down.metadata)
    for key, value in dict(semantic.metadata).items():
        if key != "compute_format":
            assert lowered_meta[key] == value


def test_exact_phase_work_matches_independent_arithmetic_oracle():
    raw = _raw(remove_gate_profile=True)
    case = parse_case(raw)
    ops = _ops(raw)
    down = _down(ops)
    expected = _oracle(8, 64, 64, (3, 0, 3, 2))
    report = graph_costs(case, [down], external_inputs={down.deps[0]: 0},
                         include_submission=True, isolated=True)
    applied = report["isolated"]["operators"][0]["operator"]
    execution = report["execution"]

    assert execution["work"]["flops"] == pytest.approx(expected["flops"])
    assert applied["mte2_bytes"] == pytest.approx(expected["mte2_bytes"])
    assert applied["mte3_bytes"] == pytest.approx(expected["mte3_bytes"])
    assert applied["hbm_bytes"] == pytest.approx(expected["hbm_bytes"])
    assert applied["vector_ops"] == pytest.approx(expected["vector_ops"])
    assert applied["matrix_flops"] == {"int4": expected["flops"]}
    assert execution["recipes"] == {RECIPE: 1}
    assert execution["serviced"]["matrix@r0:s0"] == pytest.approx(
        1000 * expected["flops"] / 500_000_000.0)
    assert execution["live_tasks"] == execution["live_programs"] == 0
    ledger = dict(down.metadata)["w4a8_msd_down_ledger"]
    assert ledger["matrix"]["weight_bytes"] == pytest.approx(expected["weight_bytes"])
    assert ledger["matrix"]["encoded_scale_bytes"] == pytest.approx(
        expected["encoded_scale_bytes"])


def test_zero_expert_reads_no_bank_and_forced_balance_is_exact():
    uneven_raw = _raw(remove_gate_profile=True)
    uneven_case = parse_case(uneven_raw)
    uneven = _down(_ops(uneven_raw))
    uneven_report = graph_costs(
        uneven_case, [uneven], external_inputs={uneven.deps[0]: 0},
        include_submission=True, isolated=True)
    uneven_op = uneven_report["isolated"]["operators"][0]["operator"]

    balanced_raw = _raw(routing={"policy": "balanced"}, remove_gate_profile=True)
    balanced_case = parse_case(balanced_raw)
    balanced = _down(_ops(balanced_raw))
    balanced_meta = dict(balanced.metadata)["w4a8_msd_down"]
    assert balanced_meta["expert_rows"] == [2, 2, 2, 2]
    assert balanced_meta["active_groups"] == 4
    assert balanced_meta["routing_basis"] == "forced_balanced_integer_counts"
    balanced_report = graph_costs(
        balanced_case, [balanced], external_inputs={balanced.deps[0]: 0},
        include_submission=True, isolated=True)
    balanced_op = balanced_report["isolated"]["operators"][0]["operator"]

    # One active bank adds packed weights, encoded column scales and the
    # post-dequant FP32 assist row; the empty bank reads none of them.
    extra_bank = 0.5 * 64 * 64 + 8 * 64 + 4 * 64
    assert balanced_op["hbm_bytes"] - uneven_op["hbm_bytes"] == pytest.approx(extra_bank)
    assert balanced_op["mte2_bytes"] - uneven_op["mte2_bytes"] == pytest.approx(extra_bank)
    assert balanced_op["vector_ops"] == pytest.approx(uneven_op["vector_ops"])
    assert balanced_op["flops"] == pytest.approx(uneven_op["flops"])
    assert dict(uneven.metadata)["w4a8_msd_down"]["expert_rows"] == [3, 0, 3, 2]


def test_delayed_input_and_absent_int4_rate_are_explicit():
    raw = _raw(remove_gate_profile=True)
    case = parse_case(raw)
    down = _down(_ops(raw))
    baseline = graph_costs(case, [down], external_inputs={down.deps[0]: 0},
                           include_submission=True, isolated=True)
    late = graph_costs(case, [down], external_inputs={down.deps[0]: 5},
                       include_submission=True, isolated=True)
    assert late["timeline"][0]["ready_ms"] == pytest.approx(5)
    assert late["timeline"][0]["complete_ms"] - baseline["timeline"][0]["complete_ms"] \
        == pytest.approx(5)

    missing = parse_case(_raw(remove_gate_profile=True, remove_int4=True))
    with pytest.raises(ModelingError, match="compute_precision_rate_required:int4"):
        graph_costs(missing, [down], external_inputs={down.deps[0]: 0},
                    include_submission=True)


def test_fixpipe_publication_is_optional_and_hbm_retained():
    raw = _raw(remove_gate_profile=True)
    case = parse_case(raw)
    down = _down(_ops(raw))
    without = graph_costs(case, [down], external_inputs={down.deps[0]: 0},
                         include_submission=True, isolated=True)
    assert "fixpipe@r0:s0" not in without["execution"]["serviced"]

    raw_fixpipe = _raw(remove_gate_profile=True, fixpipe_rate=1e9)
    case_fixpipe = parse_case(raw_fixpipe)
    down_fixpipe = _down(_ops(raw_fixpipe))
    with_fixpipe = graph_costs(
        case_fixpipe, [down_fixpipe], external_inputs={down_fixpipe.deps[0]: 0},
        include_submission=True, isolated=True)
    applied = with_fixpipe["isolated"]["operators"][0]["operator"]
    assert with_fixpipe["execution"]["serviced"]["fixpipe@r0:s0"] == pytest.approx(4 * 8 * 64)
    assert applied["mte3_bytes"] == 8 * 64 + 2 * 8 * 64
    assert applied["hbm_bytes"] == without["isolated"]["operators"][0]["operator"]["hbm_bytes"]


def test_source_domain_rejects_non_down_probability_and_bad_geometry():
    semantic_raw = _raw(remove_down_binding=True)
    semantic_ops = _ops(semantic_raw)
    semantic = next(op for op in semantic_ops if op.recipe == "grouped_down")
    gate = next(op for op in semantic_ops if op.recipe == "grouped_gate_up")
    with pytest.raises(ModelingError, match="unsupported_recipe"):
        lower_op(gate)

    bad_n = replace(semantic, local_shape=(8, 32, 64), semantic_shape=(8, 32, 64),
                    flops=2 * 8 * 32 * 64)
    with pytest.raises(ModelingError, match="n_multiple64"):
        lower_op(bad_n)
    bad_k = replace(semantic, local_shape=(8, 64, 96), semantic_shape=(8, 64, 96),
                    flops=2 * 8 * 64 * 96)
    with pytest.raises(ModelingError, match="divisible_by_64"):
        lower_op(bad_k)
    bad_output = replace(
        semantic,
        metadata=tuple((key, "int8" if key == "output_format" else value)
                       for key, value in semantic.metadata))
    with pytest.raises(ModelingError, match="requires_bf16_output"):
        lower_op(bad_output)

    probability = _raw()
    probability["runtime"].pop("routing")
    with pytest.raises(ModelingError, match="rejects_probability_moment_routing"):
        _ops(probability)


def test_gate_msd_fusion_and_down_source_plan_coexist():
    raw = _raw()
    ops = _ops(raw, resource_fusion=True)
    gate = next(op for op in ops if op.recipe == GATE_RECIPE)
    down = _down(ops)
    assert gate.family == "native_fusion"
    assert dict(gate.metadata)["fusion_role"] == "grouped_gate_up_activation_quant"

    report = predict(raw)
    assert report["status"] == "modeled"
    assert report["metrics"]["ttft_ms"]["mean"] > 0
    assert report["execution"]["recipes"][GATE_RECIPE] == 3
    assert report["execution"]["recipes"][RECIPE] == 3


def test_binding_waits_for_normal_non_bf16_producer_and_requires_the_pass():
    raw = _raw(remove_gate_profile=True)
    raw["model"]["activation_dtype"] = "fp32"
    ops = _ops(raw)
    down = _down(ops)
    quant = next(op for op in ops if op.op_id == down.deps[0])
    assert quant.dtype == "fp32"
    assert quant.mte2_bytes == 8 * 64 * 4
    assert quant.mte3_bytes == 8 * 64 + 8 * 4
    assert len([op for op in ops if op.op_id == quant.op_id]) == 1
    raw["runtime"]["lowering_passes"] = []
    with pytest.raises(ModelingError, match="implementation_requires_activation_quantization"):
        _ops(raw)


def test_cross_dp_balanced_source_consumes_each_local_expert_histogram():
    raw = _raw(routing={"policy": "balanced"}, remove_gate_profile=True)
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    raw["runtime"].update(dp=2, ep=2)
    case = parse_case(raw)
    graphs = target_graphs(case, phase="prefill", tokens=4, scope="body")
    assert set(graphs) == {0, 1}
    for ops in graphs.values():
        down = _down(ops)
        config = dict(down.metadata)["w4a8_msd_down"]
        assert config["expert_rows"] == [4, 4]
        quant = next(op for op in ops if op.op_id == down.deps[0])
        assert quant.local_shape == (8, 64)
        assert down.flops == 4 * 8 * 64 * 64
    report = graph_costs(case, graphs, external_inputs={"analysis_input": 0})
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["execution"]["recipes"][RECIPE] == 2


def test_draft_scope_binding_keeps_request_identity_and_actual_source_dtype():
    from mindie_sim.prediction.transforms import lower_implementations

    raw = _raw(remove_down_binding=True, remove_gate_profile=True)
    semantic = next(op for op in _ops(raw) if op.recipe == "grouped_down")
    metadata = dict(semantic.metadata)
    metadata.pop("activation_producer")
    draft = replace(
        semantic, layer=-1, phase="draft", requests=("draft:r0",),
        deps=("draft:activation",), metadata=tuple(metadata.items()))
    runtime = parse_case(_raw(remove_gate_profile=True)).runtime
    ops = lower_implementations((draft,), runtime, "fp16")
    down, quant = _down(ops), next(op for op in ops if op.family == "quantize")
    assert down.layer == quant.layer == -1
    assert down.phase == quant.phase == "draft"
    assert down.requests == quant.requests == ("draft:r0",)
    assert quant.dtype == "fp16" and quant.deps == ("draft:activation",)
    assert down.deps == (quant.op_id,)


def test_source_phases_use_existing_split_read_pipes_without_capacity_aliases():
    raw = _raw(remove_gate_profile=True)
    raw["machine"]["device"]["rates"].update(mte2_aic=1e9, mte2_aiv=2e9)
    case = parse_case(raw)
    down = _down(_ops(raw))
    report = graph_costs(case, [down], external_inputs={down.deps[0]: 0})
    serviced = report["execution"]["serviced"]
    m, n, k, active = 8, 64, 64, 3
    assert serviced["mte2_aic@r0:s0"] == pytest.approx(
        m * k + 0.5 * n * k * active + 8 * n * active)
    assert serviced["mte2_aiv@r0:s0"] == pytest.approx(
        m * k + 4 * m * n + 4 * n * active + 4 * m + 4 * 4 * 8)
    assert "mte2@r0:s0" not in serviced
    split_service = dict(serviced)
    raw["machine"]["device"]["rates"]["mte2"] = 3e9
    case = parse_case(raw)
    report = graph_costs(case, [_down(_ops(raw))], external_inputs={down.deps[0]: 0})
    serviced = report["execution"]["serviced"]
    # An unused generic pipe cannot steal work from declared AIC/AIV owners.
    assert serviced.get("mte2@r0:s0", 0) == 0
    assert {key: value for key, value in serviced.items() if value} == pytest.approx(
        {key: value for key, value in split_service.items() if value})
    read_bytes = _oracle(m, n, k, (3, 0, 3, 2))["mte2_bytes"]
    assert serviced["mte2_aic@r0:s0"] + serviced["mte2_aiv@r0:s0"] == read_bytes
    # A genuine shared-pipe profile still uses its one declared resource.
    del raw["machine"]["device"]["rates"]["mte2_aic"]
    del raw["machine"]["device"]["rates"]["mte2_aiv"]
    generic = graph_costs(raw, [_down(_ops(raw))], external_inputs={down.deps[0]: 0})
    assert generic["execution"]["serviced"]["mte2@r0:s0"] == read_bytes
