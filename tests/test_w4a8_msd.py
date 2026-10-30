"""Public A8W4 MSD grouped fusion checks against an independent work oracle."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/source-families/w4a8-msd-control.json"
ROLE = "grouped_gate_up_activation_quant"


def _raw(**overrides):
    raw = json.loads(CASE.read_text())
    for key, value in overrides.items():
        if key == "roles":
            raw["runtime"]["kernels"]["lowerings"]["native_fusion"]["options"]["roles"] = value
        elif key == "routing":
            raw["runtime"]["routing"] = value
        elif key == "remove_int4":
            raw["machine"]["device"]["compute_peaks"].pop("int4", None)
        else:
            raise AssertionError(key)
    return raw


def _case(raw):
    return parse_case(raw)


def _fused(case, *, tokens=4):
    ops = target_operators(case, phase="prefill", tokens=tokens)
    return ops, [op for op in ops if op.family == "native_fusion"]


def _oracle(m, n, k, rows, aivs=2):
    active, h = sum(row > 0 for row in rows), n // 2
    labels = [expert for expert, count in enumerate(rows) for _ in range(count)]
    quotient, remainder = divmod(m, min(m, aivs))
    offset, assist_loads = 0, 0
    for owner in range(min(m, aivs)):
        size = quotient + (owner < remainder)
        group = labels[offset:offset + size]
        assist_loads += len(set(group))
        offset += size
    pre_read, pre_write = m * k, m * k
    matrix_read = m * k + 0.5 * n * k * active + 8 * n * active
    matrix_write = 4 * m * n
    post_read = 4 * m * n + 4 * n * assist_loads + 4 * m
    post_write = m * h + 4 * m
    vector = (6.5 * m * k + 2 * m * n + 6 * m * n
              + 5 * m * h + 5 * m * h + m)
    return {
        "flops": 4 * m * n * k,
        "vector_ops": vector,
        "hbm_bytes": pre_read + pre_write + matrix_read + matrix_write
                     + post_read + post_write,
        "mte2_bytes": pre_read + matrix_read + post_read,
        "mte3_bytes": pre_write + post_write,
        "weight_bytes": 0.5 * n * k * active,
        "scale_bytes": 12 * n * active + 8 * m,
    }


def test_uneven_histogram_zero_expert_and_no_double_counted_member_work():
    case = _case(_raw())
    semantic = target_operators(case, phase="prefill", tokens=4, resource_fusion=False)
    declaration, members = next((d, m) for d, m in _groups(semantic) if d["role"] == ROLE)
    old = _oracle(8, 64, 32, [3, 0, 3, 2])
    assert dict(members[0].metadata)["routing_realization"]["expert_rows"] == (3, 0, 3, 2)
    assert members[0].local_shape == (8, 64, 32)

    ops, fused_ops = _fused(case)
    assert len(fused_ops) == 1
    fused = fused_ops[0]
    metadata = dict(fused.metadata)
    assert fused.recipe == "ascend_w4a8_msd"
    assert metadata["fusion_semantic_ids"] == [op.op_id for op in members]
    assert metadata["fusion_group_key"] == declaration["key"]
    assert metadata["w4a8_msd"]["expert_rows"] == [3, 0, 3, 2]
    assert metadata["w4a8_msd"]["m"] == 8
    assert metadata["fusion_boundary"]["traffic"]["eliminated_reads"] == 0
    for field in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
        assert getattr(fused, field) == pytest.approx(old[field])
    # Exact arithmetic: the replaced member recipes (gate 2MNK, activation
    # and quantizer work) are not added to the new physical ledger.
    member_vector = sum(op.vector_ops for op in members)
    assert fused.vector_ops == pytest.approx(old["vector_ops"])
    assert fused.vector_ops != pytest.approx(member_vector + old["vector_ops"])
    assert fused.flops == 4 * 8 * 64 * 32
    down = next(op for op in ops if op.op_id.endswith("expert_down"))
    assert down.deps == (fused.op_id,)
    assert fused.requests == members[-1].requests


def test_graph_executes_exact_phase_work_and_late_input_dependency():
    case = _case(_raw())
    _, fused_ops = _fused(case)
    fused = fused_ops[0]
    oracle = _oracle(8, 64, 32, [3, 0, 3, 2], aivs=8)
    external = {dep: 0 for dep in fused.deps}
    report = graph_costs(case, [fused], external_inputs=external,
                         include_submission=True, isolated=True)
    execution = report["execution"]
    assert execution["work"]["flops"] == pytest.approx(oracle["flops"])
    assert execution["work"]["vector_ops"] == pytest.approx(oracle["vector_ops"])
    # The actual four-AIC/eight-AIV example gives each row its own owner.
    # Each of those owners loads its own assist row.
    assert execution["work"]["hbm_bytes"] == pytest.approx(oracle["hbm_bytes"])
    applied = report["isolated"]["operators"][0]["operator"]
    assert applied["mte2_bytes"] == pytest.approx(oracle["mte2_bytes"])
    assert applied["hbm_bytes"] == pytest.approx(oracle["hbm_bytes"])
    assert execution["recipes"] == {"ascend_w4a8_msd": 1}
    assert execution["serviced"]["matrix@r0:s0"] == pytest.approx(
        1000 * oracle["flops"] / 500_000_000.0)
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert report["timeline"][0]["ready_ms"] == pytest.approx(0)

    late = graph_costs(case, [fused], external_inputs={dep: 5 for dep in fused.deps},
                       include_submission=True, isolated=True)
    assert late["timeline"][0]["ready_ms"] == pytest.approx(5)
    assert late["timeline"][0]["complete_ms"] > 5
    assert late["timeline"][0]["complete_ms"] - report["timeline"][0]["complete_ms"] \
        == pytest.approx(5)


def test_m1_forced_balanced_counts_and_zero_groups_conserve_rows():
    raw = _raw()
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 1,
        "expert_rows": [1, 1, 0, 0]}]}
    fused = _fused(_case(raw), tokens=1)[1][0]
    assert dict(fused.metadata)["w4a8_msd"]["m"] == 2
    assert fused.flops == 4 * 2 * 64 * 32
    assert fused.mte2_bytes == pytest.approx(
        2 * 32 + 2 * 32 + 0.5 * 64 * 32 * 2 + 8 * 64 * 2
        + 4 * 2 * 64 + 4 * 64 * 2 + 4 * 2)
    assert fused.mte2_bytes < _oracle(4, 64, 32, [1, 1, 1, 1])["mte2_bytes"]

    balanced = _raw(routing={"policy": "balanced"})
    fused = _fused(_case(balanced))[1][0]
    config = dict(fused.metadata)["w4a8_msd"]
    assert config["expert_rows"] == [2, 2, 2, 2]
    assert config["active_groups"] == 4
    assert fused.flops == 4 * 8 * 64 * 32
    # The unresolved two-AIV reference gives four assist segments in both
    # histograms. Only the newly active weight bank and column scale differ.
    uneven = _fused(_case(_raw()))[1][0]
    assert fused.mte2_bytes - uneven.mte2_bytes == pytest.approx(0.5 * 64 * 32 + 8 * 64)


def test_absent_int4_rate_is_visible_and_probability_routing_is_rejected():
    raw = _raw(remove_int4=True)
    case = _case(raw)
    _, fused_ops = _fused(case)
    with pytest.raises(ModelingError, match="compute_precision_rate_required:int4"):
        graph_costs(case, [fused_ops[0]],
                    external_inputs={dep: 0 for dep in fused_ops[0].deps},
                    include_submission=True)

    probability = _raw()
    probability["runtime"].pop("routing")
    with pytest.raises(ModelingError, match="w4a8_msd_requires_explicit_integer_expert_rows"):
        _fused(_case(probability))


def test_default_native_fusion_role_survives_mixed_selection():
    case = _case(_raw(roles=[ROLE, "router_selection"]))
    semantic = target_operators(case, phase="prefill", tokens=4, resource_fusion=False)
    _, router_members = next((d, m) for d, m in _groups(semantic)
                             if d["role"] == "router_selection")
    ops, fused_ops = _fused(case)
    by_role = {dict(op.metadata).get("fusion_role"): op for op in fused_ops}
    assert set(by_role) == {ROLE, "router_selection"}
    assert by_role[ROLE].recipe == "ascend_w4a8_msd"
    assert by_role["router_selection"].recipe == "declared_native_fusion"
    assert by_role["router_selection"].hbm_bytes == pytest.approx(
        sum(op.hbm_bytes for op in router_members))


def test_public_predict_executes_the_selected_program_profile():
    report = predict(_raw())
    assert report["status"] == "modeled"
    assert report["metrics"]["ttft_ms"]["mean"] > 0
    assert report["metrics"]["tpot_ms"]["mean"] > 0
    assert report["execution"]["recipes"]["ascend_w4a8_msd"] == 3


def test_external_gate_output_is_not_erased_by_source_fusion():
    case = _case(_raw())
    semantic = target_operators(case, phase="prefill", tokens=4, resource_fusion=False)
    _, members = next((d, m) for d, m in _groups(semantic) if d["role"] == ROLE)
    gate = members[0]
    # A second user of the BF16 gate output makes it a real live-out. The MSD
    # kernel's only outputs are INT8 SwiGLU and its FP32 scale, so IDs alone
    # cannot preserve this required materialization.
    semantic = tuple(replace(op, deps=op.deps + (gate.op_id,))
                     if op.recipe == "grouped_down" else op for op in semantic)
    with pytest.raises(ModelingError, match="intermediate_external_outputs"):
        lower_native_fusions(semantic, case.runtime)


def test_block_scale_quantizer_cannot_reuse_per_row_source_contract():
    case = _case(_raw())
    semantic = target_operators(case, phase="prefill", tokens=4, resource_fusion=False)
    _, members = next((d, m) for d, m in _groups(semantic) if d["role"] == ROLE)
    quant = members[-1]
    meta = dict(quant.metadata)
    meta.update(activation_group_size=16, activation_scale_shape=(8, 2),
                activation_scale_bytes=64)
    changed = replace(quant, metadata=tuple(meta.items()),
                      mte3_bytes=quant.mte3_bytes + 32, hbm_bytes=quant.hbm_bytes + 32)
    semantic = tuple(changed if op.op_id == quant.op_id else op for op in semantic)
    with pytest.raises(ModelingError, match="one_fp32_output_scale"):
        lower_native_fusions(semantic, case.runtime)


def test_matrix_publication_uses_declared_fixpipe_not_vector_store():
    raw = _raw()
    raw["machine"]["device"]["rates"]["fixpipe"] = 1e9
    case = _case(raw)
    fused = _fused(case)[1][0]
    report = graph_costs(case, [fused], external_inputs={dep: 0 for dep in fused.deps})
    assert report["execution"]["serviced"]["fixpipe@r0:s0"] == pytest.approx(4 * 8 * 64)
    assert fused.mte3_bytes == 8 * 32 + 8 * 32 + 4 * 8
