"""NR-S7 linear-state page, RecoverSSM commit and replay tests.

[Role]
  Prove the canonical RecoverSSM page geometry, the exact accepted-length PMF,
  the single-counted boundary checkpoint, the source commit tile/launch
  geometry, the align pre-copy bytes and launches, the RecoverSSM-only page
  stride and the per-stage memory reconciliation.

[Invariants]
  Assertions are exact slot/byte/arithmetic identities or typed rejections;
  no test asserts a private intermediate.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device, subprocess or network access.

[Side-Effect Policy]
  Builds immutable records; creates nothing outside pytest's temporary root.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from mindie_sim.hybrid_kv_state_options import _stage_partition
from mindie_sim.linear_state_physics import (
    GLM_KDA_HEAD_DIM, LinearLayer, RecoverSsmConfig, accepted_token_pmf,
    boundary_crossings, compare_linear_state_memory,
    expected_accepted_tokens, linear_layers_from_model, linear_state_memory,
    validate_glm_kda_execution, validate_recoverssm)
from mindie_sim.linear_state_prefix_replay import (
    prefix_replay_cost, prefix_replay_plan)
from mindie_sim.linear_state_replay import (
    precopy_cost, recoverssm_crossover, verify_commit_cost)
from mindie_sim.memory_breakdown import (analyze_memory_breakdown,
                                         apply_state_pool_delta)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import WorkloadSpec


def _layer(name="layer0", **overrides):
    values = dict(name=name, kind="kda", index=0, num_heads=64, key_dim=128,
                  value_dim=128, conv_kernel=4, conv_width=24576,
                  activation_dtype="bf16", recurrent_elements=64 * 128 * 128,
                  conv_history_elements=24576 * 3)
    values.update(overrides)
    return LinearLayer(**values)


def _config(**overrides):
    values = dict(use_replayssm=True, speculative_k=4,
                  enable_prefix_caching=False, mamba_cache_mode="none",
                  tp=8, model_type="glm5_next", kv_block_size=128)
    values.update(overrides)
    return RecoverSsmConfig(**values)


def _align(**overrides):
    values = dict(use_replayssm=True, speculative_k=4,
                  enable_prefix_caching=True, mamba_cache_mode="align",
                  tp=8, model_type="glm5_next", kv_block_size=128)
    values.update(overrides)
    return RecoverSsmConfig(**values)


def _case(*, tp=8, dp=1, pp=1, partition=None, max_model_len=4096, max_seqs=8,
          graph=False):
    runtime = {"tp": tp, "dp": dp, "ep": 1, "pp": pp, "pcp": 1, "cp": 1,
               "max_model_len": max_model_len, "max_num_seqs": max_seqs,
               "max_num_batched_tokens": 16384, "kv_block_size": 128}
    if partition is not None:
        runtime["pp_layer_partition"] = list(partition)
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 64, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"catalog_key": "glm-5.3-flash"},
        "runtime": runtime,
        "workload": {"requests": [{"request_id": "r",
                                   "prompt_tokens": 1024,
                                   "output_tokens": 16}]}})


# ------------------------------------------------------- canonical page


def test_recover_page_is_state_conv_and_packed_records():
    layer = _layer()
    page = layer.page_geometry(8, 5)
    assert page.recurrent_state_bytes == 8 * 128 * 128 * 4 == 524_288
    assert page.conv_rows == 3 + 4 == 7
    assert page.conv_state_bytes == 24576 * 7 / 8 * 2 == 43_008
    assert page.record_bytes_per_token == 4_096 + 4_096
    assert page.packed_record_bytes == 5 * 8_192 == 40_960
    assert page.page_bytes == 524_288 + 43_008 + 40_960 == 608_256
    plan = linear_state_memory(_config(), [layer], sequence_tokens=2048)
    assert plan.total_bytes_per_sequence == 608_256
    assert plan.matrix_to_record_ratio == 64.0
    assert plan.transient_verify_bytes_per_sequence == 0.0


def test_align_multi_page_checkpoint_is_exact():
    layer = _layer()
    plan = linear_state_memory(_align(), [layer], sequence_tokens=2048)
    assert plan.pages_per_sequence == 16
    assert plan.total_bytes_per_sequence == 16 * 608_256 == 9_732_096
    assert plan.state_slots_per_sequence == 16
    assert plan.packed_record_bytes_per_sequence == 16 * 40_960


def test_standard_slots_store_history_plus_k_conv_rows():
    layer = _layer()
    config = _config(use_replayssm=False)
    plan = linear_state_memory(config, [layer], sequence_tokens=2048)
    assert plan.state_slots_per_sequence == 5
    assert layer.conv_rows(5) == 7
    slot = 524_288 + 43_008
    assert slot == 567_296
    assert plan.total_bytes_per_sequence == 5 * slot == 2_836_480


def test_record_kg_dtype_override_is_explicit():
    bf16 = _layer()
    fp32 = _layer(name="layer1", record_kg_dtype="fp32")
    assert bf16.kg_bytes(8) == 4_096
    assert fp32.kg_bytes(8) == 8_192
    assert fp32.page_geometry(8, 5).packed_record_bytes == 5 * 12_288
    plan = linear_state_memory(_config(), [fp32], sequence_tokens=128)
    assert plan.matrix_to_record_ratio == pytest.approx(524_288 / 12_288)


# ------------------------------------------------------- GLM execution


def test_glm_execution_contract_is_separate_from_generic_math():
    assert validate_glm_kda_execution(
        kind="kda", activation_dtype="bf16", key_dim=128, value_dim=128,
        conv_kernel=4, speculative_k=4, cache_mode="none",
        block_size=128).ok
    report = validate_glm_kda_execution(
        kind="gdn", activation_dtype="fp16", key_dim=64, value_dim=64,
        conv_kernel=2, speculative_k=4, cache_mode="align", block_size=4)
    assert "glm_kda_requires_kda_layer" in report.codes
    assert "glm_kda_requires_bf16_activation" in report.codes
    assert "glm_kda_requires_head_dim_128" in report.codes
    assert "glm_kda_speculation_requires_conv_kernel_4" in report.codes
    assert "glm_kda_align_block_smaller_than_spec_window" in report.codes
    assert GLM_KDA_HEAD_DIM == 128


def test_align_block_smaller_than_spec_window_rejects_before_use():
    with pytest.raises(ModelingError, match="align_block_smaller"):
        _align(mamba_block_size=1, kv_block_size=1)
    report = validate_recoverssm(_config())
    assert report.ok


# ------------------------------------------------------- acceptance PMF


def test_accepted_pmf_is_exact_and_sums_to_one():
    pmf = accepted_token_pmf(4, 0.8)
    assert pmf == ((1, pytest.approx(0.2)), (2, pytest.approx(0.16)),
                   (3, pytest.approx(0.128)), (4, pytest.approx(0.1024)),
                   (5, pytest.approx(0.4096)))
    assert sum(probability for _, probability in pmf) == pytest.approx(1.0)
    assert expected_accepted_tokens(4, 0.8) == pytest.approx(3.3616)
    assert expected_accepted_tokens(4, 1.0) == 5.0
    assert expected_accepted_tokens(4, 0.0) == 1.0


def test_boundary_probability_at_2044_is_exactly_alpha_cubed():
    cost = verify_commit_cost(
        _align(), [_layer()], batch=1, sequence_tokens=2044,
        hbm_bytes_per_s=1.2e12, accept_rate=0.8)
    assert cost["boundary_probability"] == pytest.approx(0.8 ** 3) == pytest.approx(0.512)
    assert cost["boundary_commits"] == pytest.approx(0.512)
    assert cost["full_state_writes"] == pytest.approx(1.512)
    assert cost["commit_len"] == pytest.approx(3.3616)


def test_commit_len_is_not_rounded_before_cost():
    cost = verify_commit_cost(
        _align(), [_layer()], batch=2, sequence_tokens=2044,
        hbm_bytes_per_s=1.2e12, accept_rate=0.8)
    record_read = 12_288  # value-tiled: correction 4,096 + 2 x kg 4,096
    # Committed page: state 524,288 + conv_history 3 rows = 18,432.
    page = 542_720
    assert cost["commit_len"] == pytest.approx(3.3616)
    assert cost["normal_record_read_bytes_per_token"] == record_read
    assert cost["stage2_read_bytes"] == pytest.approx(
        2 * page + 2 * 3.3616 * record_read
        + 2 * (4 * 0.2 * 0.8 ** 3 + 5 * 0.8 ** 4) * 8_192
        + 2 * 0.512 * 18_432)
    assert cost["compact_record_bytes"] == 2 * 5 * 8_192


def test_boundary_checkpoint_is_counted_once():
    layer = _layer()
    crossing = verify_commit_cost(
        _align(), [layer], batch=2, sequence_tokens=2047,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    inside = verify_commit_cost(
        _align(), [layer], batch=2, sequence_tokens=2048,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    page = 542_720
    assert crossing["boundary_commits"] == 2
    # A crossing request loads one extra conv history window and writes the
    # boundary state-plus-history page; the initial state is loaded once.
    assert crossing["boundary_read_bytes"] == 2 * 18_432
    assert crossing["boundary_write_bytes"] == 2 * page == 1_085_440
    assert crossing["boundary_extra_bytes"] == 1_122_304
    assert crossing["write_bytes"] - inside["write_bytes"] == 2 * page
    assert crossing["hbm_bytes"] - inside["hbm_bytes"] == 1_122_304 + 2 * 3 * 8_192
    assert crossing["stage2_matrix_flops"] - inside["stage2_matrix_flops"] == 2 * 2 * 131_072 * 4
    assert crossing["full_state_writes"] == 3
    assert inside["full_state_writes"] == 1
    assert inside["boundary_extra_bytes"] == 0


def test_commit_tile_and_launch_geometry_match_the_source():
    crossing = verify_commit_cost(
        _align(), [_layer()], batch=2, sequence_tokens=2047,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    inside = verify_commit_cost(
        _config(), [_layer()], batch=2, sequence_tokens=2048,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    assert crossing["commit_block_v"] == 64
    assert crossing["boundary_block_v"] == 32
    assert crossing["ub_tile_rows"] == (190, 62)
    assert crossing["stage2_program_count"] == 32 + 64 + 2 + 2
    assert crossing["kernel_launches"] == 1 + 4
    assert inside["kernel_launches"] == 1 + 3
    assert inside["stage2_program_count"] == 32 + 2 + 2
    no_dispatch = verify_commit_cost(
        _align(), [_layer()], batch=2, sequence_tokens=2047,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3,
        program_dispatch_us=0.0)
    assert no_dispatch["stage2_timing"]["program_grid_ms"] == 0.0
    assert no_dispatch["total_ms"] == pytest.approx(
        no_dispatch["stage1_timing"]["total_ms"]
        + no_dispatch["stage2_timing"]["total_ms"])


def test_crossover_distinguishes_search_limit_from_never():
    exceeds = recoverssm_crossover(
        _config(), [_layer()], hbm_bytes_per_s=1.2e12,
        compute_flops_per_s=1.0e14, kernel_launch_ms=0.5,
        sequence_tokens=2048, max_batch=10, accept_rate=0.8)
    assert exceeds["reason"] == "recoverssm_crossover_exceeds_max_batch"
    never = recoverssm_crossover(
        _config(), [_layer()], hbm_bytes_per_s=1.2e12,
        compute_flops_per_s=1.0e14, kernel_launch_ms=0.0, max_batch=2000,
        accept_rate=0.8, program_dispatch_us=1.0e6)
    assert never["reason"] == "recoverssm_stage2_launch_never_amortized"
    zero_launch = recoverssm_crossover(
        _config(), [_layer()], hbm_bytes_per_s=1.2e12,
        compute_flops_per_s=1.0e14, kernel_launch_ms=0.0, max_batch=10,
        accept_rate=0.8)
    assert zero_launch["crossover_batch"] == 1


# ------------------------------------------------------- pre-copy (F12)


def test_precopy_is_zero_outside_align_mode():
    cost = precopy_cost(_config(), [_layer()], sequence_start=0,
                        sequence_end=128, batch=1, hbm_bytes_per_s=1.0e12,
                        kernel_launch_ms=0.01)
    assert cost["applies"] is False
    assert cost["copy_bytes"] == 0.0
    assert cost["kernel_launches"] == 0
    assert cost["bulk_ms"] == 0.0


def test_precopy_includes_state_and_conv_window_and_fast_exit_launches():
    config = _align()
    crossing = precopy_cost(
        config, [_layer()], sequence_start=2047, sequence_end=2175,
        batch=2, hbm_bytes_per_s=1.0e12, kernel_launch_ms=0.01)
    assert crossing["crossings"] == 1
    assert crossing["copy_bytes"] == 2 * 1 * 2 * (524_288 + 18_432)
    assert crossing["conv_copy_bytes"] == 2 * 1 * 18_432
    assert crossing["kernel_launches"] == 2
    non_crossing = precopy_cost(
        config, [_layer()], sequence_start=2048, sequence_end=2053,
        batch=2, hbm_bytes_per_s=1.0e12, kernel_launch_ms=0.01)
    assert non_crossing["copy_bytes"] == 0.0
    assert non_crossing["kernel_launches"] == 2
    assert non_crossing["bulk_ms"] == pytest.approx(0.02)
    assert boundary_crossings(2048, 2053, 128) == 0


# ------------------------------------------------------- stride / replay


def test_page_stride_is_propagated_only_to_recoverssm():
    comparison = compare_linear_state_memory(
        _align(), [_layer()], sequence_tokens=256,
        page_stride_bytes=1_000_000)
    assert comparison["state_page_stride_bytes_applied_to"] == "recoverssm"
    assert comparison["standard"]["page_stride_bytes"] is None
    assert comparison["recoverssm"]["page_stride_bytes"] == 1_000_000
    assert comparison["recoverssm"]["padding_bytes_per_sequence"] == 2 * (
        1_000_000 - 608_256)
    with pytest.raises(ModelingError, match="page_stride_requires_replayssm"):
        linear_state_memory(_config(use_replayssm=False), [_layer()],
                            sequence_tokens=256,
                            page_stride_bytes=1_000_000)
    with pytest.raises(ModelingError, match="page_stride_smaller_than_page"):
        linear_state_memory(_align(), [_layer()], sequence_tokens=256,
                            page_stride_bytes=100_000)


def test_prefix_replay_loads_the_canonical_page_and_reports_replay_ms():
    plan = prefix_replay_plan(_align(), [_layer()], isl=2048,
                              prefix_hit=1000)
    assert plan["checkpoint_tokens"] == 896
    assert plan["replay_tokens"] == 104
    assert plan["linear_layer_tokens"] == 1152
    cost = prefix_replay_cost(_align(), [_layer()], plan=plan,
                              hbm_bytes_per_s=1.2e12,
                              compute_flops_per_s=1.0e14)
    assert cost["checkpoint_bytes"] == 608_256
    assert cost["replay_ms"] > 0
    assert cost["prompt_ms"] > cost["replay_ms"]
    exact = prefix_replay_plan(_align(), [_layer()], isl=2048,
                               prefix_hit=1024)
    assert exact["replay_tokens"] == 0


# ------------------------------------------------------- stage ownership


def test_stage_ownership_uses_the_declared_pp_partition():
    case = _case(tp=8, pp=2, partition=(1, 44))
    partition = _stage_partition(case.runtime, case.model)
    assert partition == (1, 44)
    layers = linear_layers_from_model(case.model, tp=8)
    assert [layer.index for layer in layers[:1]] == [0]
    stage0 = [layer for layer in layers if layer.index < 1]
    stage1 = [layer for layer in layers if layer.index >= 1]
    assert len(stage0) == 1 and len(stage1) == 33
    page = _layer().page_geometry(8, 5).page_bytes
    assert sum(layer.page_geometry(8, 5).page_bytes for layer in stage0) == page
    assert sum(layer.page_geometry(8, 5).page_bytes for layer in stage1) == (
        33 * page)


def test_state_pool_delta_charges_each_stage_its_own_value():
    case = _case(tp=8, pp=2, partition=(1, 44))
    base = analyze_memory_breakdown(case, sequence_tokens=2048, sequences=8,
                                    graph_analytical_bound=True)
    adjusted = apply_state_pool_delta(
        base, delta_bytes=0, per_stage_bytes=(0, 1_000))
    assert adjusted.stage_peak_bytes[0] == base.stage_peak_bytes[0]
    assert adjusted.stage_peak_bytes[1] == base.stage_peak_bytes[1] + 1_000
    for row, old in zip(adjusted.per_rank_bytes, base.per_rank_bytes):
        extra = 1_000 if int(row[1]) == 1 else 0
        assert row[3] == old[3] + extra
    assert sum(component.bytes for component in adjusted.components) == (
        adjusted.peak_hbm_bytes)
    for stage in range(2):
        assert sum(component.per_stage_bytes[stage]
                   for component in adjusted.components) == (
                       adjusted.stage_peak_bytes[stage])
    assert adjusted.peak_hbm_bytes == base.peak_hbm_bytes + 1_000


def test_dp_replicas_receive_the_same_stage_delta():
    case = _case(tp=8, dp=2, pp=2, partition=(1, 44))
    base = analyze_memory_breakdown(case, sequence_tokens=2048, sequences=8,
                                    graph_analytical_bound=True)
    adjusted = apply_state_pool_delta(
        base, per_stage_bytes=(0, 1_000))
    stage1_rows = [row for row in adjusted.per_rank_bytes
                   if int(row[1]) == 1]
    assert len({int(row[0]) for row in stage1_rows}) == 2
    assert all(row[3] == old[3] + 1_000
               for row, old in zip(stage1_rows,
                                   [row for row in base.per_rank_bytes
                                    if int(row[1]) == 1]))
    assert all(row[3] == old[3]
               for row, old in zip(adjusted.per_rank_bytes,
                                   base.per_rank_bytes)
               if int(row[1]) == 0)


def test_signed_replacement_can_release_baseline_bytes():
    case = _case(tp=8, pp=2, partition=(1, 44))
    base = analyze_memory_breakdown(case, sequence_tokens=2048, sequences=8,
                                    graph_analytical_bound=True)
    adjusted = apply_state_pool_delta(
        base, per_stage_bytes=(0, 0), replaced_per_stage_bytes=(0, 1_000),
        allow_negative=True)
    assert adjusted.stage_peak_bytes[1] == base.stage_peak_bytes[1] - 1_000
    assert adjusted.peak_hbm_bytes == base.peak_hbm_bytes - 1_000
    assert sum(component.bytes for component in adjusted.components) == (
        adjusted.peak_hbm_bytes)
    with pytest.raises(ModelingError, match="state_delta_negative"):
        apply_state_pool_delta(
            base, per_stage_bytes=(0, 0),
            replaced_per_stage_bytes=(0, 1_000))


# ------------------------------------------------------- pre-coercion (F14)


def test_linear_layer_rejects_inconsistent_conv_and_tp():
    with pytest.raises(ModelingError, match="conv_width_invalid"):
        LinearLayer(name="l", index=0, conv_width=0,
                    conv_history_elements=99)
    layer = _layer(conv_width=5, conv_history_elements=15)
    with pytest.raises(ModelingError, match="conv_width_not_divisible_by_tp"):
        layer.conv_bytes(2)
    with pytest.raises(ModelingError, match="tp_invalid"):
        _layer().conv_bytes(True)


@pytest.mark.parametrize("value", [64.9, True, "64", None])
def test_catalog_heads_are_not_coerced(value):
    plan = SimpleNamespace(index=0, attention="kda", linear_heads=value,
                           linear_key_head_dim=128, linear_head_dim=128,
                           conv_kernel=4, conv_width=24576,
                           conv_state_elements=24576 * 3,
                           recurrent_state_elements=64 * 128 * 128,
                           recurrent_state_dtype="fp32",
                           conv_state_dtype="bf16")
    model = SimpleNamespace(layer_plans=(plan,), activation_dtype="bf16")
    with pytest.raises(ModelingError):
        linear_layers_from_model(model, tp=8)


@pytest.mark.parametrize("overrides", [
    {"speculative_k": True}, {"pp": 1.0}, {"mamba_cache_mode": "ALIGN"},
    {"speculative_k": 7}, {"kv_role": "producer"},
])
def test_config_rejects_types_and_limits(overrides):
    with pytest.raises(ModelingError):
        _config(**overrides)


def test_accept_rate_nan_or_over_one_rejects():
    with pytest.raises(ModelingError, match="accept_rate_invalid"):
        expected_accepted_tokens(4, 1.5)
    with pytest.raises(ModelingError, match="accept_rate_invalid"):
        expected_accepted_tokens(4, float("nan"))


def test_verify_zero_bandwidth_is_a_typed_rejection():
    with pytest.raises(ModelingError, match="hbm_bytes_per_s_invalid"):
        verify_commit_cost(_config(), [_layer()], batch=1,
                           sequence_tokens=128, hbm_bytes_per_s=0)
    with pytest.raises(ModelingError, match="hbm_bytes_per_s_invalid"):
        verify_commit_cost(_config(), [_layer()], batch=1,
                           sequence_tokens=128, hbm_bytes_per_s=True)


# R11: a leaf declaration cannot replace the registered base field contract.
FIX11_FORMS = ("classvar", "any", "object", "union", "unannotated", "property")


def fix11_subtype(record, name, item, form, depth, *, forge=False):
    import dataclasses as dc
    from typing import Any, ClassVar, Union
    annotations = {"extra_contract": int}
    namespace = {"__module__": __name__, "__annotations__": annotations,
                 "extra_contract": dc.field(default=1, kw_only=True)}
    if form in ("classvar", "any", "object", "union", "property"):
        annotations[name] = {"classvar": ClassVar[object], "any": Any,
            "object": object, "union": Union[int, object],
            "property": ClassVar[object]}[form]
    if form in ("classvar", "unannotated", "property"):
        namespace[name] = property(lambda self: item) if form == "property" else item
    cls = dc.dataclass(type("Fix11" + type(record).__name__, (type(record),), namespace),
                       frozen=type(record).__dataclass_params__.frozen, kw_only=True)
    for level in range(1, depth):
        cls = type(f"Fix11Level{level}", (cls,), {})
    values = {field.name: (item if field.name == name else
              getattr(record, field.name, field.default)) for field in dc.fields(cls)}
    if not forge:
        return cls(**values)
    value = object.__new__(cls)
    for key, entry in values.items():
        object.__setattr__(value, key, entry)
    return value


def _fix11_consumers(form, depth, capture):
    from nr_s7_fix2_support import _case as case_factory
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.model import WorkloadSpec
    from test_nr_s7_fix2_regressions_r2 import fix9_consumer_calls, fix10_reject_before_body
    case = case_factory()
    case = replace(case, runtime=replace(case.runtime, graph_capture_mode=capture))
    mutations = {"classvar": (("model", "num_heads", 0), ("runtime", "max_num_seqs", 0),
                             ("case", "workload", WorkloadSpec._pure_model_sentinel())),
                 "any": (("model", "hidden_size", "invalid"),
                         ("case", "workload", WorkloadSpec._pure_model_sentinel())),
                 "object": (("case", "workload", None),),
                 "union": (("model", "hidden_size", "invalid"),),
                 "unannotated": (("model", "hidden_size", "invalid"),),
                 "property": (("case", "workload", WorkloadSpec._pure_model_sentinel()),)}[form]
    calls, controls = [], []
    for slot, name, bad in mutations:
        original = case if slot == "case" else getattr(case, slot)
        record = fix11_subtype(original, name, bad, form, depth)
        with pytest.raises(ModelingError):
            _validate_nested(record, "record")
        broken = record if slot == "case" else replace(case, **{slot: record})
        calls.extend(fix9_consumer_calls(broken).values())
        if form in ("classvar", "any") and depth == 1:
            base_bad = replace(original, **{name: bad})
            base_case = base_bad if slot == "case" else replace(case, **{slot: base_bad})
            controls.extend(fix9_consumer_calls(base_case).values())
    fix10_reject_before_body((*calls, *controls))
    valid = replace(case,
        model=fix11_subtype(case.model, "num_heads", case.model.num_heads, form, depth),
        runtime=fix11_subtype(case.runtime, "max_num_seqs", case.runtime.max_num_seqs, form, depth))
    valid = fix11_subtype(valid, "workload", valid.workload, form, depth)
    _validate_nested(valid, "valid")
    valid_calls = fix9_consumer_calls(valid)
    for call in valid_calls.values():
        assert call() is not None
    return {"consumer_rejections": len(calls), "base_controls": len(controls),
            "nested_rejections": len(mutations), "valid_consumers": len(valid_calls)}


@pytest.mark.parametrize(
    "form,depth,capture",
    [(f, 1, "piecewise") for f in FIX11_FORMS]
    + [("classvar", 3, "piecewise"), ("any", 3, "none"), ("classvar", 1, "full")],
)
def test_fix11_redeclared_contracts_reject_before_consumer_body(form, depth, capture):
    _fix11_consumers(form, depth, capture)


def test_fix11_workload_precedes_descriptors_and_checks_subclass_additions():
    import dataclasses as dc
    from typing import Any, ClassVar
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.case import Case
    from mindie_sim.schema.model import WorkloadSpec
    case = _case()
    for form in FIX11_FORMS:
        bad = fix11_subtype(case, "workload", WorkloadSpec._pure_model_sentinel(), form, 3)
        def unreachable(self):
            raise AssertionError("machine accessed before invalid workload")
        leaf = type("WorkloadFirst", (type(bad),), {"machine": property(unreachable)})
        broken = object.__new__(leaf)
        broken.__dict__.update(vars(bad))
        with pytest.raises(ModelingError, match="requests_invalid"):
            _validate_nested(broken, "case")
    # Different annotations at intermediate and leaf levels still retain Case.workload.
    @dc.dataclass(frozen=True)
    class Intermediate(Case):
        workload: ClassVar[WorkloadSpec] = WorkloadSpec._pure_model_sentinel()
    @dc.dataclass(frozen=True, kw_only=True)
    class Leaf(Intermediate):
        workload: Any = None
        extra_contract: int = 1
    valid = Leaf(**{field.name: getattr(case, field.name, field.default) for field in dc.fields(Leaf)})
    _validate_nested(valid, "valid")
    for changes in ({"workload": WorkloadSpec._pure_model_sentinel()}, {"extra_contract": "invalid"}):
        with pytest.raises(ModelingError):
            _validate_nested(replace(valid, **changes), "leaf")
    @dc.dataclass(frozen=True)
    class Extension(Case):
        extra_request: WorkloadSpec | None = None
    extended = Extension(**{field.name: getattr(case, field.name) for field in dc.fields(case)},
                         extra_request=WorkloadSpec._pure_model_sentinel())
    with pytest.raises(ModelingError, match="requests_invalid"):
        _validate_nested(extended, "extension")


def fix11_witnesses():
    from collections import Counter
    from test_nr_s7_fix3_regressions import (
        fix11_domain_witnesses, test_fix11_multiple_registered_bases_and_missing_optional_property)
    totals = Counter()
    for form in FIX11_FORMS:
        for depth in (1, 3):
            for capture in ("none", "full", "piecewise"):
                totals.update(_fix11_consumers(form, depth, capture))
    test_fix11_workload_precedes_descriptors_and_checks_subclass_additions()
    test_fix11_multiple_registered_bases_and_missing_optional_property()
    return {**dict(totals), "consumer_body_entries": 0, "invalid_accepts": 0,
            "raw_type_errors": 0, "forms": list(FIX11_FORMS), "depths": [1, 3],
            "capture_modes": ["none", "full", "piecewise"], "consumer_forms": 10,
            "domains": fix11_domain_witnesses()}
