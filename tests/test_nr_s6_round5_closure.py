"""Round-5 closure: executable F1..F11 boundary regressions.

[Role]
  Re-run the Round-5 review counterexamples against the repaired trace replay,
  half-open expert range, physical KV/batch validation, uint1 value contract,
  shared-gate probe protocol, idempotent calibration parsing, short artifact
  storage and the independent typed callable stub for all 13 ready recipes.

[Invariants]
  * Every assertion executes product code; no source-string grep stands in for
    a runtime deadline, cleanup, protocol or retrieval check.
  * Probe tests use the real two-thread rendezvous broker and give both ranks
    independent clocks and output paths.
  * The callable table is transcribed from the external signature, not read
    from the catalog during the negative checks.

[Timeouts & Locks]
  In-memory work plus two bounded broker threads; no NPU, HCCL, process or
  device-subprocess launch.

[Side-Effect Policy]
  Writes only pytest temporary files and short content-addressed reproducer
  directories created by the product.
"""

from __future__ import annotations

import json
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import shutil
import tempfile

import pytest
from mindie_profiler.events import ProfilerUsageError

from mindie_profiler.fabric_probe import build_fabric_probe_plan, main as fabric_main
from mindie_profiler.fabric_probe_fit import (
    fit_calibrated_fabric_profile, parse_fabric_probe_samples,
)
from mindie_profiler.op_api_registry import bind_captured_occurrence, catalog_entry
from mindie_profiler.repro_op import (
    _compact_occurrence, call_mcp_tool, generate_single_op_reproducer,
)
from mindie_profiler.repro_preconditions import precondition_problems
from mindie_profiler.repro_storage import resolve_repro_artifact, short_root
from mindie_profiler.repro_values import validate_captured_values
from mindie_profiler.token_budget import estimate_tokens
from mindie_sim.diff_trace import attribute_diff

from nr_s6_round4_probe import run_ranks
from nr_s6_support import exec_script, paged_binding
from test_nr_s6_round2_closure import _ready_bindings
from test_nr_s6_round3_closure import _bsh_fa_row, _fit_samples, _moe_row
from test_nr_s6_round4_closure import _probe_plan


def _records(results):
    return [item for _rank, (_code, rows) in results.items() for item in rows]


def _samples(results):
    return [item for item in _records(results)
            if item.get("schema") == "mindie-fabric-probe-sample/2"]


def _codes(results):
    return sorted(code for code, _rows in results.values())


# ----------------------------------------------------------------------- F1
def test_f1_identical_trace_with_explicit_gap_is_zero_difference():
    trace = {
        "total_ms": 3,
        "gaps": [{"stream": 0, "start_ms": 1, "duration_ms": 1}],
        "ops": [{"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
                {"name": "B", "stream": 0, "start_ms": 2, "duration_ms": 1}],
    }
    result = attribute_diff(trace, trace, launch_overhead_us=0)
    assert result["stages"] == {"R0": 3.0, "R1": 3.0, "R2": 3.0,
                                "R3": 3.0, "R4_model": 3.0, "O": 3.0}
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(0.0)
    assert result["attribution"]["graph_delta_ms"] == pytest.approx(0.0)
    assert result["attribution"]["op_delta_ms"] == pytest.approx(0.0)
    assert result["attribution"]["residual_ms"] == pytest.approx(0.0)
    assert result["reconciliation"]["reconciled"] is True


def test_f1_inserted_occurrence_transfers_release_by_captured_index():
    predicted = {"total_ms": 2, "ops": [
        {"name": "A", "stream": 0, "start_ms": 0, "duration_ms": 1},
        {"name": "B", "stream": 0, "start_ms": 1, "duration_ms": 1,
         "deps": [0]}]}
    captured = {"total_ms": 4,
                "gaps": [{"stream": 0, "start_ms": 2, "duration_ms": 1}],
                "ops": [{"name": "X", "stream": 0, "start_ms": 0,
                         "duration_ms": 1},
                        {"name": "A", "stream": 0, "start_ms": 1,
                         "duration_ms": 1},
                        {"name": "B", "stream": 0, "start_ms": 3,
                         "duration_ms": 1, "deps": [0]}]}
    result = attribute_diff(predicted, captured, launch_overhead_us=0)
    assert result["stages"] == {"R0": 2.0, "R1": 3.0, "R2": 4.0,
                                "R3": 4.0, "R4_model": 4.0, "O": 4.0}
    assert result["attribution"]["sched_delta_ms"] == pytest.approx(1.0)
    assert result["attribution"]["graph_delta_ms"] == pytest.approx(1.0)
    assert result["attribution"]["op_delta_ms"] == pytest.approx(0.0)
    assert result["reconciliation"]["reconciled"] is False


# ----------------------------------------------------------------------- F2
def test_f2_half_open_range_endpoint_and_arity():
    legal = _moe_row()
    legal["attrs"]["active_expert_range"] = [0, 4]
    assert _compact_occurrence(bind_captured_occurrence(legal))["status"] == "ready"
    for bad in ([0, 5], [4, 4]):
        row = _moe_row()
        row["attrs"]["active_expert_range"] = bad
        binding = bind_captured_occurrence(row)
        assert binding["status"] == "bound"
        with pytest.raises(Exception) as excinfo:
            _compact_occurrence(binding)
        assert "invalid_precondition" in str(excinfo.value)
    for bad in ([0, 1, 2], ["bad"], [True, 1]):
        row = _moe_row()
        row["attrs"]["active_expert_range"] = bad
        assert bind_captured_occurrence(row)["status"] != "bound"

# ----------------------------------------------------------------------- F3
def test_f3_every_kv_tensor_is_physically_validated():
    row = _bsh_fa_row([3, 2])
    row["input_shapes"][2] = "[2,1,1024]"
    binding = bind_captured_occurrence(row)
    assert binding["padding"]["valid"] is False
    assert any(item["code"] == "sequence_kv_exceeds_per_sequence_extent"
               for item in binding["padding"]["violations"])
    with pytest.raises(Exception):
        _compact_occurrence(binding)

def test_f3_qkv_batch_cardinality_and_tnd_vector_arity():
    row = _bsh_fa_row([3, 2])
    row["input_shapes"][0] = "[2,4,1024]"
    row["input_shapes"][1] = "[3,4,1024]"
    row["input_shapes"][2] = "[3,4,1024]"
    row["attrs"]["actual_seq_q_len"] = [3, 2]
    row["attrs"]["actual_seq_kv_len"] = [1, 1, 1]
    binding = bind_captured_occurrence(row)
    assert binding["padding"]["valid"] is False
    assert any(item["code"] == "qkv_batch_mismatch"
               for item in binding["padding"]["violations"])
    occurrence = {
        "preconditions": {"kind": "attention_geometry"}, "call_plan": {},
        "attrs": {"input_layout": "BSH"}, "allocations": [
            {"name": "query", "shape": [2, 4, 1024], "dtype": "bf16"},
            {"name": "key", "shape": [3, 4, 1024], "dtype": "bf16"},
            {"name": "value", "shape": [3, 4, 1024], "dtype": "bf16"}]}
    problems = precondition_problems(occurrence)
    assert any("batch cardinality" in item["detail"] for item in problems)
    tnd = dict(occurrence)
    tnd["attrs"] = {"input_layout": "TND", "actual_seq_q_len": [1, 2],
                    "actual_seq_kv_len": [1]}
    tnd["allocations"] = [
        {"name": "query", "shape": [3, 4, 128], "dtype": "bf16"},
        {"name": "key", "shape": [3, 4, 128], "dtype": "bf16"},
        {"name": "value", "shape": [3, 4, 128], "dtype": "bf16"}]
    assert any("vector arity" in item["detail"]
               for item in precondition_problems(tnd))

# ----------------------------------------------------------------------- F5
def test_f5_uint1_value_contract_and_unknown_dtype_rejection():
    for value in ([2], [-1], [1e100], [True], [0.5]):
        values, problems = validate_captured_values([1], "uint1", value)
        assert values is None and problems, value
    assert validate_captured_values([1], "uint1", [1])[0] == [1]
    values, problems = validate_captured_values([1], "mystery", [1])
    assert values is None
    assert any(item["code"] == "unsupported_value_dtype:mystery"
               for item in problems)

def test_f5_uint1_descriptor_is_blocked_before_ready():
    import ast
    row = _bsh_fa_row([3, 2])
    row["input_shapes"] = [
        {"shape": ast.literal_eval(shape), "dtype": dtype, "layout": layout}
        for shape, dtype, layout in zip(row["input_shapes"],
                                        row["input_dtypes"],
                                        row["input_formats"])]
    row["input_shapes"][4] = {"shape": [1], "dtype": "DT_UINT1",
                              "layout": "ND", "values": [2]}
    binding = bind_captured_occurrence(row)
    assert binding["status"] == "bound"
    occurrence = _compact_occurrence(binding)
    assert occurrence["status"] == "blocked"
    assert occurrence["checks"]["allocatable_torch_dtype"] is False
    assert occurrence["fidelity"] != "value_captured"
    assert any(item["code"] == "allocatable_torch_dtype"
               for item in occurrence["precondition_problems"])
    catalog = json.loads(
        (Path(checked_io.__file__).resolve().parent / "op_api_catalog.json").read_text())
    operators = list(catalog["operators"].values())
    assert len(operators) == 32
    assert sum(1 for entry in operators
               if (entry.get("repro_recipe") or {}).get("supported")) == 13
    assert sum(1 for entry in operators
               if not (entry.get("repro_recipe") or {}).get("supported")) == 19
    assert sum(1 for entry in operators
               if entry.get("capture_verified") is True) == 0

# ------------------------------------------------------------------ F6 / F7
def test_f6_late_local_clock_cannot_break_the_shared_protocol():
    plan = _probe_plan(iterations=2, warmup=0)
    plan["budget_s"] = 1.0
    plan["deadline"]["measurement_budget_s"] = 0.9
    plan["deadline"]["max_wall_s"] = 1.0
    plan["deadline"]["cleanup_allowance_s"] = 0.1
    world, results = run_ranks(
        plan, decision_delay=(1, 6, 2.0))
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert _codes(results) == [2, 2]
    assert world.broker.decision_logs[0][-1] == 0
    assert world.broker.decision_logs[1][-1] == 0

def test_f6_latency_excludes_min_gate_coordination_time():
    world, results = run_ranks(
        _probe_plan(iterations=2, warmup=0), decision_advance=0.010)
    samples = _samples(results)
    assert world.errors == {}
    assert _codes(results) == [0, 0]
    assert len(samples) == 4
    assert {item["latency_us"] for item in samples} == {0.0}

def test_f6_data_failure_signals_peer_abort_without_timeout():
    plan = _probe_plan(iterations=2, warmup=0)
    world, results = run_ranks(plan, data_actions={1: {2: "raise"}})
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert world.broker.aborted is True
    assert _codes(results) == [2, 2]
    statuses = [item.get("status") for item in _records(results)
                if item.get("kind") == "failed"]
    assert "collective_failed" in statuses
    assert "peer_aborted" in statuses
    assert world.abort_record is not None
    assert world.abort_record["rank"] == 1

def test_f7_late_telemetry_disqualifies_completion_on_both_ranks():
    plan = _probe_plan(iterations=1, warmup=0)
    plan["budget_s"] = 1.0
    plan["deadline"]["measurement_budget_s"] = 0.9
    plan["deadline"]["max_wall_s"] = 1.0
    world, results = run_ranks(plan, clock_advance=2.0)
    samples = _samples(results)
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert _codes(results) == [2, 2]
    assert samples
    assert all(item["validated"] is False for item in samples)
    assert all(item.get("status") == "deadline_exceeded" for item in samples)

def test_f7_cleanup_failure_is_recorded_and_returns_two():
    plan = _probe_plan(iterations=1, warmup=0)
    world, results = run_ranks(
        plan, fail_destroy={0: True, 1: True})
    assert world.errors == {}
    assert world.broker.mismatches == []
    assert _codes(results) == [2, 2]
    cleanup = [item for item in _records(results)
               if item.get("kind") == "cleanup_failed"]
    assert len(cleanup) == 2
    assert all(item["cleanup_failed"] is True
               and item["status"] == "cleanup_failed" for item in cleanup)

# ----------------------------------------------------------------------- F8
def _fabric_plan():
    return build_fabric_probe_plan(
        world_size=8, ranks_per_node=8, duration_s=60,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": list(range(8)),
                             "protocol": "HCCS"}]})

def test_f8_invalid_group_size_survives_repeated_parsing():
    for bad in (False, 0.9, "garbage", 0, -1, 2):
        samples = _fit_samples(validated=True, rank=0,
                               group_ranks=list(range(8)))
        for sample in samples:
            sample["group_size"] = bad
        if bad != 2:
            with pytest.raises(ProfilerUsageError):
                parse_fabric_probe_samples(samples)
            continue
        first = parse_fabric_probe_samples(samples)
        second = parse_fabric_probe_samples(first)
        assert all(item["group_size"] is None for item in second)
        assert all(item["_group_size_invalid"] is not None for item in second)
        profile = fit_calibrated_fabric_profile(_fabric_plan(), first)
        assert profile["status"] == "partial"
        assert profile["usable_for_link_override"] is False
        assert {item["reason"] for item in profile["rejected_samples"]} == {
            "group_size_invalid"}

def test_f8_cli_rejects_invalid_group_size(tmp_path):
    for bad in (False, 0.9, "garbage"):
        samples = _fit_samples(validated=True, rank=0,
                               group_ranks=list(range(8)))
        for sample in samples:
            sample["group_size"] = bad
        samples_path = tmp_path / f"samples-{str(bad)}.jsonl"
        samples_path.write_text("\n".join(json.dumps(item) for item in samples))
        topology_path = tmp_path / "topology.json"
        topology_path.write_text(json.dumps(
            {"tiers": [{"tier": "intra_node_hccs", "ranks": list(range(8)),
                        "protocol": "HCCS"}]}))
        output_path = tmp_path / f"profile-{str(bad)}.json"
        with pytest.raises(ProfilerUsageError):
            fabric_main([
                "--world-size", "8", "--ranks-per-node", "8",
                "--duration-s", "60", "--topology", str(topology_path),
                "--samples", str(samples_path), "--output", str(output_path)])

# ---------------------------------------------------------------------- F10
def test_f10_long_default_temp_root_still_returns_resolvable_short_paths(
        monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir",
                        lambda: "/tmp/" + "a" * 180)
    result = call_mcp_tool("repro_op", {"occurrence": paged_binding()})
    try:
        assert result["status"] == "ready"
        assert estimate_tokens(result) <= 250
        assert result["artifact"]["retrievable"] is True
        script = Path(result["artifact_path"])
        manifest = Path(result["manifest_path"])
        assert checked_io.path_is_file(script) and checked_io.path_is_file(manifest)
        assert len(str(script)) < 80 and len(str(manifest)) < 80
        assert script.read_text().startswith("#!")
        assert "repro_sha256" in manifest.read_text()
        resolved = resolve_repro_artifact(result["artifact_id"])
        assert resolved["resolved"] is True
        assert resolved["artifact_path"] == result["artifact_path"]
        assert resolved["manifest_path"] == result["manifest_path"]
        assert resolved["script_sha256"] == result["artifact_id"]
        via_tool = call_mcp_tool(
            "repro_op", {"resolve_artifact_id": result["artifact_id"]})
        assert via_tool["artifact_path"] == result["artifact_path"]
        assert str(short_root()) in result["artifact_path"]
    finally:
        shutil.rmtree(Path(result["artifact_path"]).parent, ignore_errors=True)

def test_f10_long_caller_path_is_still_resolvable(monkeypatch):
    requested = "/tmp/" + "b" * 180
    result = call_mcp_tool("repro_op", {
        "occurrence": paged_binding(), "output_dir": requested})
    try:
        assert result["status"] == "ready"
        assert len(str(Path(result["artifact_path"]))) < 80
        assert checked_io.path_is_file(Path(result["artifact_path"]))
        assert resolve_repro_artifact(result["artifact_id"])["retrievable"] is True
    finally:
        shutil.rmtree(Path(result["artifact_path"]).parent, ignore_errors=True)

# ---------------------------------------------------------------------- F11
# Independent callable table transcribed from the external signatures (not
# derived from the catalog at assertion time).  Every parameter carries
# (kind, required, nullable): optional tensors are nullable because the real
# wrapper may receive None for an absent tensor, while every required parameter
# and every optional scalar/array rejects an explicit None.
_READY_TYPES = {
    "ChunkGatedDeltaRule": (
        [("q", "tensor", True, False), ("k", "tensor", True, False),
         ("v", "tensor", True, False)],
        {"beta": ("tensor", False, True),
         "initial_state": ("tensor", False, True),
         "actual_seq_lengths": ("tensor", False, True),
         "scale": ("float", False, False),
         "g": ("tensor", False, True)}),
    "MoeGatingTopKSoftmax": (
        [("x", "tensor", True, False)],
        {"finished": ("tensor", False, True),
         "k": ("int", False, False)}),
    "MoeInitRoutingV2": (
        [("x", "tensor", True, False), ("expert_idx", "tensor", True, False)],
        {"scale": ("tensor", False, True),
         "offset": ("tensor", False, True),
         "active_num": ("int", False, False),
         "expert_capacity": ("int", False, False),
         "expert_num": ("int", False, False),
         "drop_pad_mode": ("int", False, False),
         "expert_tokens_num_type": ("int", False, False),
         "expert_tokens_num_flag": ("bool", False, False),
         "quant_mode": ("int", False, False),
         "active_expert_range": ("int_array", False, False),
         "row_idx_type": ("int", False, False),
         "x_dtype": ("int", False, False)}),
    "QuantLightningIndexer": (
        [("query", "tensor", True, False), ("key", "tensor", True, False),
         ("weights", "tensor", True, False),
         ("query_dequant_scale", "tensor", True, False),
         ("key_dequant_scale", "tensor", True, False),
         ("query_quant_mode", "int", True, False),
         ("key_quant_mode", "int", True, False)],
        {"actual_seq_lengths_query": ("tensor", False, True),
         "actual_seq_lengths_key": ("tensor", False, True),
         "block_table": ("tensor", False, True),
         "layout_query": ("str", False, False),
         "layout_key": ("str", False, False),
         "sparse_count": ("int", False, False),
         "sparse_mode": ("int", False, False),
         "pre_tokens": ("int", False, False),
         "next_tokens": ("int", False, False),
         "query_dtype": ("int", False, False),
         "key_dtype": ("int", False, False)}),
    "aclnnAddRmsNorm": (
        [("x1", "tensor", True, False), ("x2", "tensor", True, False),
         ("gamma", "tensor", True, False)],
        {"epsilon": ("float", False, False)}),
    "aclnnApplyRotaryPosEmbV2": (
        [("query_ref", "tensor", True, False),
         ("key_ref", "tensor", True, False), ("cos", "tensor", True, False),
         ("sin", "tensor", True, False)],
        {"layout": ("str", False, False),
         "rotary_mode": ("str", False, False)}),
    "aclnnDynamicQuant": (
        [("x", "tensor", True, False)],
        {"smooth_scales": ("tensor", False, True),
         "group_index": ("tensor", False, True),
         "dst_type": ("int", False, False),
         "quant_mode": ("str", False, False),
         "dst_type_max": ("float", False, False)}),
    "aclnnFlashAttentionVarLenScore": (
        [("query", "tensor", True, False), ("key", "tensor", True, False),
         ("value", "tensor", True, False)],
        {"head_num": ("int", True, False),
         "input_layout": ("str", True, False),
         "pse": ("tensor", False, True),
         "padding_mask": ("tensor", False, True),
         "atten_mask": ("tensor", False, True),
         "scale": ("float", False, False),
         "keep_prob": ("float", False, False),
         "pre_tockens": ("int", False, False),
         "next_tockens": ("int", False, False),
         "inner_precise": ("int", False, False),
         "prefix": ("int_array", False, False),
         "actual_seq_qlen": ("int_array", False, False),
         "actual_seq_kvlen": ("int_array", False, False),
         "sparse_mode": ("int", False, False),
         "gen_mask_parallel": ("bool", False, False),
         "sync": ("bool", False, False),
         "softmax_layout": ("str", False, False),
         "sink": ("tensor", False, True),
         "dropout_mask": ("tensor", False, True),
         "seed": ("int", False, False),
         "offset": ("int", False, False)}),
    "aclnnFusedInferAttentionScoreV2": (
        [("query", "tensor", True, False), ("key", "tensor", True, False),
         ("value", "tensor", True, False)],
        {"pse_shift": ("tensor", False, True),
         "atten_mask": ("tensor", False, True),
         "actual_seq_lengths": ("int_array", False, False),
         "actual_seq_lengths_kv": ("int_array", False, False),
         "dequant_scale1": ("tensor", False, True),
         "quant_scale1": ("tensor", False, True),
         "dequant_scale2": ("tensor", False, True),
         "quant_scale2": ("tensor", False, True),
         "quant_offset2": ("tensor", False, True),
         "antiquant_scale": ("tensor", False, True),
         "antiquant_offset": ("tensor", False, True),
         "key_antiquant_scale": ("tensor", False, True),
         "key_antiquant_offset": ("tensor", False, True),
         "value_antiquant_scale": ("tensor", False, True),
         "value_antiquant_offset": ("tensor", False, True),
         "block_table": ("tensor", False, True),
         "query_padding_size": ("tensor", False, True),
         "kv_padding_size": ("tensor", False, True),
         "key_shared_prefix": ("tensor", False, True),
         "value_shared_prefix": ("tensor", False, True),
         "actual_shared_prefix_len": ("int_array", False, False),
         "query_rope": ("tensor", False, True),
         "key_rope": ("tensor", False, True),
         "key_rope_antiquant_scale": ("tensor", False, True),
         "num_heads": ("int", False, False),
         "scale": ("float", False, False),
         "pre_tokens": ("int", False, False),
         "next_tokens": ("int", False, False),
         "input_layout": ("str", False, False),
         "num_key_value_heads": ("int", False, False),
         "sparse_mode": ("int", False, False),
         "inner_precise": ("int", False, False),
         "block_size": ("int", False, False),
         "antiquant_mode": ("int", False, False),
         "key_antiquant_mode": ("int", False, False),
         "value_antiquant_mode": ("int", False, False),
         "softmax_lse_flag": ("bool", False, False)}),
    "aclnnGroupedMatmulV4": (
        [],
        {"x": ("tensor_list", True, False),
         "weight": ("tensor_list", True, False),
         "bias": ("tensor_list", False, True),
         "scale": ("tensor_list", False, True),
         "offset": ("tensor_list", False, True),
         "antiquant_scale": ("tensor_list", False, True),
         "antiquant_offset": ("tensor_list", False, True),
         "per_token_scale": ("tensor_list", False, True),
         "group_list": ("tensor", False, True),
         "activation_input": ("tensor_list", False, True),
         "activation_quant_scale": ("tensor_list", False, True),
         "activation_quant_offset": ("tensor_list", False, True),
         "split_item": ("int", False, False),
         "group_type": ("int", False, False),
         "group_list_type": ("int", False, False),
         "act_type": ("int", False, False)}),
    "aclnnPagedAttention": (
        [],
        {"query": ("tensor", True, False),
         "key_cache": ("tensor", True, False),
         "value_cache": ("tensor", True, False),
         "num_kv_heads": ("int", True, False),
         "num_heads": ("int", True, False),
         "scale_value": ("float", True, False),
         "block_table": ("tensor", True, False),
         "context_lens": ("tensor", True, False),
         "out": ("tensor", True, False)}),
    "aclnnRmsNorm": (
        [("x", "tensor", True, False), ("gamma", "tensor", True, False)],
        {"epsilon": ("float", False, False)}),
    "aclnnSwiGlu": (
        [("x", "tensor", True, False)],
        {"dim": ("int", False, False)}),
}
_FIXED_LENGTHS = {("MoeInitRoutingV2", "active_expert_range"): 2}

class _FakeTensor:
    def __init__(self, shape=(1,)):
        self.shape = tuple(shape)

def _check_value(name, kind, value, fixed=None):
    if value is None:
        raise TypeError(f"{name}: None is not accepted")
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{name}: expected bool, got {value!r}")
    elif kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name}: expected non-bool int, got {value!r}")
    elif kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name}: expected non-bool number, got {value!r}")
    elif kind == "str":
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected str, got {value!r}")
    elif kind == "tensor":
        if not isinstance(value, _FakeTensor):
            raise TypeError(f"{name}: expected tensor, got {type(value)!r}")
    elif kind == "tensor_list":
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, _FakeTensor) for item in value):
            raise TypeError(f"{name}: expected tensor[], got {value!r}")
    elif kind in {"int_array", "array"}:
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value):
            raise TypeError(f"{name}: expected strict int[], got {value!r}")
        if fixed is not None and len(value) != fixed:
            raise ValueError(f"{name}: expected length {fixed}, got {len(value)}")
    elif kind == "float_array":
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value):
            raise TypeError(f"{name}: expected float[], got {value!r}")
    else:
        raise ValueError(f"{name}: unsupported kind {kind!r}")

def _strict_stub(positional, keyword, fixed_lengths=None):
    fixed_lengths = fixed_lengths or {}
    required_positional = sum(1 for _n, _k, required, _null in positional
                              if required)

    def stub(*args, **kwargs):
        if len(args) < required_positional or len(args) > len(positional):
            raise TypeError(
                f"expected {required_positional}..{len(positional)} positional "
                f"args, got {len(args)}")
        for (name, kind, _required, nullable), value in zip(positional, args):
            if value is None:
                if not nullable:
                    raise TypeError(f"{name}: None is not accepted")
                continue
            _check_value(name, kind, value, fixed_lengths.get(name))
        supplied = set()
        for name, value in kwargs.items():
            if name not in keyword:
                raise TypeError(f"unknown keyword {name!r}")
            kind, _required, nullable = keyword[name]
            if value is None:
                if not nullable:
                    raise TypeError(f"{name}: None is not accepted")
            else:
                _check_value(name, kind, value, fixed_lengths.get(name))
            supplied.add(name)
        missing = sorted(name for name, (_k, required, _null) in keyword.items()
                         if required and name not in supplied)
        if missing:
            raise TypeError(f"missing required keyword(s): {', '.join(missing)}")
        return ("y", "scale")

    return stub

def test_f11_callable_table_covers_all_ready_recipes():
    assert len(_READY_TYPES) == 13
    for name, (positional, keyword) in _READY_TYPES.items():
        recipe = catalog_entry(name)["repro_recipe"]
        params = recipe["signature"]["params"]
        assert set(params) == ({item[0] for item in positional}
                               | set(keyword)), name
        assert [(item[0], item[1]) for item in positional] == [
            (item, params[item]["kind"]) for item in params
            if item in {entry[0] for entry in positional}], name
        for param, kind, required, nullable in positional:
            assert params[param]["kind"] == kind, (name, param)
            assert params[param]["required"] is required, (name, param)
            assert isinstance(nullable, bool) and not (required and nullable)
        for param, (kind, required, nullable) in keyword.items():
            assert params[param]["kind"] == kind, (name, param)
            assert params[param]["required"] is required, (name, param)
            assert isinstance(nullable, bool)
            assert nullable == (not required
                                and kind in {"tensor", "tensor_list"}), (
                                    name, param)

def test_f11_strict_type_negative_self_tests():
    positional, keyword = _READY_TYPES["MoeInitRoutingV2"]
    moe = _strict_stub(positional, keyword, {"active_expert_range": 2})
    x, idx = _FakeTensor(), _FakeTensor()
    for bad in ([0, 1, 2], ["bad"], [True, 1]):
        with pytest.raises((TypeError, ValueError)):
            moe(x, idx, active_expert_range=bad)
    with pytest.raises((TypeError, ValueError)):
        moe(x, idx, expert_tokens_num_flag="bad")
    with pytest.raises(TypeError):
        moe(x, idx, expert_tokens_num_flag=None)
    with pytest.raises(TypeError):
        moe(None, None)
    fa = _strict_stub(*_READY_TYPES["aclnnFlashAttentionVarLenScore"])
    with pytest.raises(TypeError):
        fa(_FakeTensor(), _FakeTensor(), _FakeTensor())
    int_flag = _strict_stub(
        [], {"expert_tokens_num_flag": ("int", False, False)})
    with pytest.raises((TypeError, ValueError)):
        int_flag(expert_tokens_num_flag=True)
    with pytest.raises(TypeError):
        int_flag(expert_tokens_num_flag=None)
    scalar_int = _strict_stub([], {"k": ("int", False, False)})
    for bad in ("bad", True):
        with pytest.raises((TypeError, ValueError)):
            scalar_int(k=bad)

def _ready_invoke_setup(name):
    binding = _ready_bindings()[name]
    occurrence = _compact_occurrence(binding)
    positional, keyword = _READY_TYPES[name]
    fixed = {param: length for (op, param), length in _FIXED_LENGTHS.items()
             if op == name}
    stub = _strict_stub(positional, keyword, fixed)
    namespace = exec_script(
        generate_single_op_reproducer(binding, stage=False)["script"])
    namespace["_resolve"] = lambda _path: stub
    inputs, outputs, lists = {}, {}, {}
    for spec in occurrence["allocations"]:
        item = _FakeTensor(spec["shape"])
        if spec.get("tensor_list"):
            lists.setdefault(spec["tensor_list"], []).append(
                (spec.get("list_index", 0), item))
        elif spec["direction"] == "input":
            inputs[spec["name"]] = item
        else:
            outputs[spec["name"]] = item
    lists = {key: [item for _idx, item in sorted(values)]
             for key, values in lists.items()}
    metadata = {param: _FakeTensor()
                for param, spec in occurrence["call_plan"]["keyword"].items()
                if spec.get("kind") == "meta"}
    return (namespace, occurrence["call_plan"], inputs, outputs, lists,
            occurrence["attrs"], metadata)

def _invoke_all_ready():
    invoked = []
    for name in _ready_bindings():
        (namespace, plan, inputs, outputs, lists, attrs,
         metadata) = _ready_invoke_setup(name)
        namespace["_invoke"](plan, inputs, outputs, lists, attrs, metadata)
        invoked.append(name)
    return invoked

def test_f11_all_thirteen_ready_recipes_execute_through_strict_stub():
    invoked = _invoke_all_ready()
    assert set(invoked) == set(_READY_TYPES)
    assert len(invoked) == 13
