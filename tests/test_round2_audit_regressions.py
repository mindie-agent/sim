"""Tier 0/1 regression counterexamples for the Round-2 P0/P1 audit findings.

[Role]
  Pin the exact physical counterexamples from the Round-2 review: DBO lowering,
  selected-SKU attention hierarchy, standalone aclnn declarations, paged causal
  masking, pooled-index HBM accounting, family target overrides, sweep candidate
  identity, checkpoint persistence, and channel-semantics validation.

[Invariants]
  Each test reproduces one reviewed defect and asserts an independent physical
  invariant (work conservation, dtype-declared traffic, mask-independent FLOPs,
  distinct evaluation identity), never a static string or documentation claim.

[Timeouts & Locks]
  Bounded in-memory cases (<= 4 layers, <= 8 devices); no I/O except pytest
  temporary checkpoint files; no devices or network.

[Side-Effect Policy]
  Writes only under pytest's temporary directory; never mutates repository state.
"""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction import frontend as frontend_module
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency,
    evaluate_target_efficiency,
)
from mindie_sim.prediction.index_pool import _visited_pool_rows
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep import sweep_topologies_iter
from mindie_sim.sweep_engine import compute_sweep_candidate_id


def _disable_homogeneous_lowering(monkeypatch):
    """Force the direct body path so the homogeneous fast path can be compared."""
    monkeypatch.setattr(
        frontend_module.Frontend, "_try_lower_homogeneous",
        lambda self, **kwargs: None)


# --------------------------------------------------------------------------- #
# Finding 1 (P0): homogeneous lowering must not bypass DBO partitioning.
# --------------------------------------------------------------------------- #
def test_dbo_partitioning_is_never_bypassed_by_homogeneous_lowering(monkeypatch):
    case = compose_case(
        "qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=4,
        enable_dbo=True, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    optimized = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=2)
    _disable_homogeneous_lowering(monkeypatch)
    direct = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=2)

    assert len(optimized["operators"]) == len(direct["operators"])
    assert optimized["graph_duration_ms"] == pytest.approx(
        direct["graph_duration_ms"])
    assert optimized["total_flops"] == pytest.approx(direct["total_flops"])
    prefixes = {op["op_id"].split(".", 1)[0] for op in optimized["operators"]}
    assert not ({"dbo0", "dbo1"} & prefixes)


# --------------------------------------------------------------------------- #
# Finding 2 (P0): homogeneous attention must use the selected SKU hierarchy.
# --------------------------------------------------------------------------- #
def test_homogeneous_attention_uses_selected_machine_capacity(monkeypatch):
    case = compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=1, layers=4,
        prompt_tokens=512, output_tokens=2)
    optimized = analyze_network_efficiency(
        case, phase="prefill", tokens=512, context=30000, sequences=1)
    _disable_homogeneous_lowering(monkeypatch)
    direct = analyze_network_efficiency(
        case, phase="prefill", tokens=512, context=30000, sequences=1)

    assert optimized["total_hbm_bytes"] == pytest.approx(
        direct["total_hbm_bytes"])
    assert optimized["graph_duration_ms"] == pytest.approx(
        direct["graph_duration_ms"])


# --------------------------------------------------------------------------- #
# Finding 3 (P1): valid standalone aclnn declarations must not raise NameError.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("declaration", [
    {"api": "aclnnMatmul", "tensors": {
        "x": {"shape": [32, 128], "dtype": "bf16"},
        "w": {"shape": [128, 32], "dtype": "bf16"},
        "out": {"shape": [32, 32], "dtype": "bf16"}}, "phase": "decode"},
    {"api": "aclnnFlashAttentionScore", "tensors": {
        "query": {"shape": [1, 8, 64], "dtype": "bf16"},
        "key": {"shape": [1, 8, 64], "dtype": "bf16"},
        "value": {"shape": [1, 8, 32], "dtype": "bf16"}},
        "attrs": {"causal": True}, "phase": "prefill"},
    {"api": "aclnnRmsNorm", "tensors": {
        "x": {"shape": [16, 128], "dtype": "bf16"},
        "gamma": {"shape": [128], "dtype": "bf16"},
        "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
    {"api": "aclnnAddRmsNorm", "tensors": {
        "x": {"shape": [16, 128], "dtype": "bf16"},
        "residual": {"shape": [16, 128], "dtype": "bf16"},
        "gamma": {"shape": [128], "dtype": "bf16"},
        "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
    {"api": "aclnnSwiGlu", "tensors": {
        "x": {"shape": [16, 256], "dtype": "bf16"},
        "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
    {"api": "aclnnDynamicQuant", "tensors": {
        "x": {"shape": [16, 128], "dtype": "bf16"},
        "y": {"shape": [16, 128], "dtype": "int8"}}, "phase": "decode"},
])
def test_standalone_tensor_shaped_aclnn_declarations_resolve(declaration):
    call = build_aclnn_call(declaration)
    assert call.ops
    assert call.ops[0].hbm_bytes > 0.0


# --------------------------------------------------------------------------- #
# Finding 4 (P0): standalone Matmul must honor tensor-declared dtypes.
# --------------------------------------------------------------------------- #
def test_standalone_matmul_honors_tensor_declared_fp32_dtypes():
    declaration = {
        "api": "aclnnMatmul",
        "tensors": {
            "x": {"shape": [32, 128], "dtype": "fp32"},
            "w": {"shape": [128, 32], "dtype": "fp32"},
            "out": {"shape": [32, 32], "dtype": "fp32"},
        },
        "phase": "decode",
    }
    op = build_aclnn_call(declaration).ops[0]
    assert op.dtype == "fp32"
    assert op.mte2_bytes == pytest.approx(32 * 128 * 4 + 128 * 32 * 4)
    assert op.mte3_bytes == pytest.approx(32 * 32 * 4)
    assert op.hbm_bytes == pytest.approx(36864.0)


# --------------------------------------------------------------------------- #
# Finding 5 (P0): KV paging must never remove the causal attention mask.
# --------------------------------------------------------------------------- #
def test_paged_causal_attention_keeps_causal_pair_count():
    base = {
        "api": "aclnnFlashAttentionScore",
        "tensors": {
            "query": {"shape": [1, 8, 64], "dtype": "bf16"},
            "key": {"shape": [1, 8, 64], "dtype": "bf16"},
            "value": {"shape": [1, 8, 32], "dtype": "bf16"},
        },
        "attrs": {"causal": True},
        "phase": "prefill",
    }
    dense = build_aclnn_call(base).ops[0]
    paged = build_aclnn_call({**base, "block_size": 128}).ops[0]
    assert dense.flops == pytest.approx(6912.0)
    assert paged.flops == pytest.approx(dense.flops)
    assert paged.vector_ops == pytest.approx(dense.vector_ops)


# --------------------------------------------------------------------------- #
# Finding 6 (P0): pooled-index HBM accounting separates private and shared rows.
# --------------------------------------------------------------------------- #
def _pool_view(index, key, ranges=(), q_len=2):
    return SimpleNamespace(
        q_len=q_len, context=8192, kv_len=8192, prefix_tokens=4096,
        prefix_cache_key=key, request_id=f"req-{index}", query_ranges=ranges)


def test_visited_pool_rows_counts_private_suffixes_and_prefix_union():
    plan = SimpleNamespace(index_kpool=4, noncausal_attention=False)
    private = SimpleNamespace(prefix_cache_mode="private", prefix_read_mode="auto")
    shared = SimpleNamespace(prefix_cache_mode="shared", prefix_read_mode="auto")

    assert _visited_pool_rows(
        [_pool_view(0, "a"), _pool_view(1, "b")], plan,
        runtime=private) == 4096
    assert _visited_pool_rows(
        [_pool_view(0, "a"), _pool_view(1, "b")], plan,
        runtime=shared) == 4096
    assert _visited_pool_rows(
        [_pool_view(0, "a"), _pool_view(1, "a")], plan,
        runtime=shared) == 3072


def test_visited_pool_rows_honors_pcp_query_ranges_and_noncausal_extent():
    causal = SimpleNamespace(index_kpool=4, noncausal_attention=False)
    noncausal = SimpleNamespace(index_kpool=4, noncausal_attention=True)
    shared = SimpleNamespace(prefix_cache_mode="shared", prefix_read_mode="auto")
    lates = _pool_view(0, "a", ((4000, 4100),))
    base = _pool_view(0, "a")
    assert _visited_pool_rows([lates], causal, runtime=shared) > _visited_pool_rows(
        [base], causal, runtime=shared)
    assert _visited_pool_rows(
        [SimpleNamespace(q_len=2, context=8192, kv_len=16384, prefix_tokens=0,
                         prefix_cache_key="", request_id="r", query_ranges=())],
        noncausal, runtime=shared) == 4096


# --------------------------------------------------------------------------- #
# Finding 7 (P0): exact-target aggregation must retain family overrides.
# --------------------------------------------------------------------------- #
def test_exact_target_retains_per_family_hbm_requirement():
    case = compose_case(
        "qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=4,
        prompt_tokens=64, output_tokens=2)
    base = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=1,
        mode="exact_target")
    slower = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=1,
        mode="exact_target",
        family_targets={"gate_up_projection": {"target_mbu": 0.1}})

    assert slower["physical_ideal_ms"] == pytest.approx(base["physical_ideal_ms"])
    assert slower["target_step_duration_ms"] > base["target_step_duration_ms"] * 4.0
    assert slower["target_step_duration_ms"] == pytest.approx(
        slower["raw_target_duration_ms"])
    expected_mbu = (slower["current_network_mbu"]
                    * slower["current_step_duration_ms"]
                    / slower["target_step_duration_ms"])
    assert slower["target_network_mbu"] == pytest.approx(expected_mbu)


# --------------------------------------------------------------------------- #
# Finding 8 (P0): candidate identity covers model, runtime, workload, machine.
# --------------------------------------------------------------------------- #
def _candidate_id(raw):
    parsed = parse_case(raw)
    return compute_sweep_candidate_id(
        parsed, ParallelSpec(tp=1, dp=1, ep=1, pp=1), cand_devices=1,
        ranks_per_node=1, phase="decode", scope="body", context=8,
        q_tokens=1, emit_tokens=1.0, output_budget=2, sla_ms=50.0,
        max_sequences_per_dp=1, lock_parallel=False)


def test_candidate_id_separates_memory_budget_and_runtime_bindings():
    case = compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=1, layers=2,
        prompt_tokens=8, output_tokens=2, concurrency=1)
    baseline = _candidate_id(case)
    tiny = deepcopy(case)
    tiny["runtime"]["gpu_memory_utilization"] = 0.001
    rebatched = deepcopy(case)
    rebatched["runtime"]["kv_block_size"] = 64
    rebound = deepcopy(case)
    # A binding the structural defaults never install, so the identity
    # difference is the declaration and not an already-selected default.
    rebound["runtime"]["kernels"] = {"lowerings": {"mla_head_padding": "vllm_ascend_mla_head_padding"}}
    assert len({baseline, _candidate_id(tiny), _candidate_id(rebatched),
                _candidate_id(rebound)}) == 4
    assert _candidate_id(case) == baseline


# --------------------------------------------------------------------------- #
# Finding 9 (P0): checkpointing keeps every grid/derivation row and upgrades B1.
# --------------------------------------------------------------------------- #
def _sweep_case(layers=2):
    return compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, pp=1, num_devices=1,
        layers=layers, prompt_tokens=8, output_tokens=2, concurrency=1,
        max_num_seqs=1)


def _run_sweep(case, path, **kwargs):
    return list(sweep_topologies_iter(
        case, context=8, output_budget=2, max_sequences_per_dp=1,
        two_stage=False, output_jsonl=path, **kwargs))


def test_sweep_checkpoint_persists_every_device_grid(tmp_path):
    path = tmp_path / "grid.jsonl"
    rows = _run_sweep(_sweep_case(), path, num_devices_grid=(1, 2))
    persisted = [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 1
    assert len(persisted) == len(rows)
    assert len({row["candidate_id"] for row in persisted}) == len(rows)


def test_sweep_checkpoint_persists_hardware_derivation_metadata(tmp_path):
    path = tmp_path / "derived.jsonl"
    derivations = [
        {"sku": "derived-a", "hbm_peak_bytes_per_s": 1.5e12},
        {"sku": "derived-b", "hbm_peak_bytes_per_s": 1.2e12},
    ]
    rows = _run_sweep(_sweep_case(), path, num_devices=1,
                      hardware_derivations=derivations)
    persisted = [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert len(persisted) == 2
    assert {row["derived_sku"] for row in persisted} == {"derived-a", "derived-b"}
    assert all(row["derivation"]["sku"] == row["derived_sku"] for row in persisted)


def test_sweep_resume_upgrades_screening_checkpoints(tmp_path):
    path = tmp_path / "resume.jsonl"
    rows = list(sweep_topologies_iter(
        _sweep_case(), num_devices=1, context=8, output_budget=2,
        max_sequences_per_dp=1, two_stage=True, output_jsonl=path))
    assert rows
    demoted = []
    for row in rows:
        item = dict(row)
        item["evaluation_stage"] = "b1_screen"
        demoted.append(item)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in demoted), encoding="utf-8")

    resumed = list(sweep_topologies_iter(
        _sweep_case(), num_devices=1, context=8, output_budget=2,
        max_sequences_per_dp=1, two_stage=True, output_jsonl=path, resume=True))
    assert resumed
    assert not any(row.get("resumed_from_checkpoint") for row in resumed)
    assert any(row.get("evaluation_stage") == "full_sla" for row in resumed)


# --------------------------------------------------------------------------- #
# Finding 10 (P0): invalid channel semantics fail with explicit links present.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ("channel_semantics", "semantics"))
def test_invalid_channel_semantics_rejected_with_explicit_links(field):
    raw = compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=8, layers=1,
        ranks_per_node=8)["machine"]
    links = [link.to_dict() for link in Hardware.from_mapping(raw).links]
    assert links
    raw["links"] = links
    raw["transfer_channels"]["d2d_inter_node"][field] = "invented"
    with pytest.raises(ModelingError, match="transfer_channel_semantics_unsupported"):
        Hardware.from_mapping(raw)


def test_composed_neutral_machine_rejects_invalid_channel_semantics():
    raw = compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=8, layers=1,
        ranks_per_node=8)
    raw["machine"]["transfer_channels"]["d2d_intra_node"]["semantics"] = "invented"
    with pytest.raises(ModelingError, match="transfer_channel_semantics_unsupported"):
        parse_case(raw)
