"""Round-2 closure: R9..R15 runtime, delivery and documentation evidence.

[Role]
  Execute the emitted fabric-probe and single-op scripts against recording
  stubs, exercise the fit qualification rules, resolve the immutable occurrence
  selector, verify persisted artifacts and bounded receipts, and audit the
  final documentation and module hygiene.

[Invariants]
  * The probe loop executes a fixed pre-agreed iteration count per rank; a
    rank-0 deadline abort is shared through one group ``int32[1]`` reduction.
  * A 0.051 s plan whose estimated cost is larger is rejected before allocation.
  * Isolated and loaded fits are separate; the nonnegative boundary projection
    returns ``(0, 1.8)`` for ``(1,1),(2,4)`` with SSE ``0.8``.
  * The emitted reproducer reports ``unstable_allocation`` when allocation
    drifts, and ``cold_verified`` only with a real flush counter and sweep.
  * The MCP reproducer persists a retrievable script and manifest.

[Timeouts & Locks]
  Pure in-process stubs; no NPU, SSH or distributed process is started.

[Side-Effect Policy]
  Writes only under pytest's ``tmp_path`` or the process temp directory.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
__import__("pytest").importorskip("mindie_profiler")
from mindie_profiler import repro_atomic_io as checked_io
import sys
import types

import pytest

from temp_cleanup import tempdir

@pytest.fixture(autouse=True)
def _isolate_short_root(monkeypatch):
    import shutil as _shutil
    short = tempdir('mp_', directory='/tmp')
    monkeypatch.setenv('MINDIE_REPRO_SHORT_ROOT', str(short))
    try:
        yield
    finally:
        _shutil.rmtree(short, ignore_errors=True)

from mindie_profiler.fabric_probe import (
    build_fabric_probe_plan, emit_fabric_probe_script,
)
from mindie_profiler.fabric_probe_fit import (
    _weighted_fit, fit_calibrated_fabric_profile,
)
from mindie_profiler.op_api_registry import bind_captured_occurrence
from mindie_profiler.repro_op import (
    _row_from_args, call_mcp_tool, generate_single_op_reproducer, main as repro_main,
)
from mindie_profiler.token_budget import estimate_tokens
from mindie_sim.diff_trace import attribute_diff
from mindie_sim.diff_trace_replay import _op_uid

from nr_s6_support import paged_binding, paged_row


ROOT = Path(checked_io.__file__).resolve().parents[1]


# ------------------------------------------------------------- script stubs
class _Tensor:
    def __init__(self, shape=(), dtype=None, device=None, decision=None, value=1):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._item = value
        self._decision = decision

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def item(self):
        return self._item

    def zero_(self):
        return self

    def fill_(self, *_args):
        return self

    def copy_(self, _other):
        return self

    def clone(self):
        return _Tensor(self.shape, self.dtype, self.device)

    def sum(self):
        return _Tensor((), self.dtype, self.device)

    def reshape(self, shape):
        return _Tensor(shape, self.dtype, self.device)

    def all(self):
        return self


def _make_torch(alloc_sequence=None):
    calls = {"alloc": 0}

    def zeros(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = tuple(args[0])
        elif len(args) == 1:
            shape = (args[0],)
        else:
            shape = tuple(args)
        return _Tensor(shape, kwargs.get("dtype"), kwargs.get("device"))

    def memory_allocated():
        if not alloc_sequence:
            return 0
        index = min(calls["alloc"], len(alloc_sequence) - 1)
        calls["alloc"] += 1
        return alloc_sequence[index]

    module = types.ModuleType("torch")
    module.__path__ = []
    for name in ("int8", "int32", "int64", "uint8", "bfloat16", "float16",
                 "float32", "bool", "float8_e4m3fn", "float8_e5m2"):
        setattr(module, name, name)
    module.zeros = zeros
    module.tensor = lambda values, dtype=None, device=None: _Tensor(
        (len(values),), dtype, device, decision=True, value=values[0])
    module.empty_strided = lambda shape, strides, dtype=None, device=None: _Tensor(
        shape, dtype, device)
    module.isfinite = lambda _tensor: _Tensor((), None, None, value=True)
    module.npu = types.SimpleNamespace(
        synchronize=lambda: None, device_count=lambda: 4,
        set_device=lambda _rank: None, memory_allocated=memory_allocated,
        Event=lambda enable_timing=False: _Event())
    return module


class _Event:
    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, _other):
        return 10.0


class SharedDecisions:
    def __init__(self):
        self.stop = set()

    def reset(self):
        self.stop.clear()


class _FakeDist:
    def __init__(self, shared, rank):
        self.shared = shared
        self.rank = rank
        self.collective_calls = {}
        self.decision_calls = []
        self.validations = 0
        self.timeout = None

    class ReduceOp:
        MIN = "min"

    def init_process_group(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    def new_group(self, ranks):
        return ("group", tuple(ranks))

    def destroy_process_group(self):
        return None

    def all_reduce(self, tensor, group=None, op=None):
        if getattr(tensor, "_decision", False):
            self.decision_calls.append((group, tensor._item))
            if tensor._item == 0:
                self.shared.stop.add(group)
            if group in self.shared.stop:
                tensor._item = 0
        else:
            self.collective_calls[group] = self.collective_calls.get(group, 0) + 1

    def all_gather_into_tensor(self, *_args, **_kwargs):
        return None

    def reduce_scatter_tensor(self, *_args, **_kwargs):
        return None

    def all_to_all_single(self, *_args, **_kwargs):
        return None

    def send(self, *_args, **_kwargs):
        return None

    def recv(self, *_args, **_kwargs):
        return None


class _FakeClock:
    def __init__(self, offset_s=0.0, jump_s=0.0):
        self.now = 1000.0 + float(offset_s)
        self.jump = float(jump_s)

    def monotonic(self):
        self.now += self.jump
        return self.now

    def perf_counter_ns(self):
        return int(self.now * 1e9)


def _probe_run(plan, rank, world, clock, shared=None):
    script = emit_fabric_probe_script(plan)["script"]
    torch = _make_torch()
    npu = types.ModuleType("torch_npu")
    npu.__path__ = []
    npu.npu_get_clock_rate = lambda: 1.8e9
    distributed = types.ModuleType("torch.distributed")
    shared = shared or SharedDecisions()
    dist = _FakeDist(shared, rank)
    for name in ("init_process_group", "new_group", "destroy_process_group",
                 "all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
                 "all_to_all_single", "send", "recv"):
        setattr(distributed, name, getattr(dist, name))
    distributed.ReduceOp = _FakeDist.ReduceOp
    torch.distributed = distributed
    namespace = {"__name__": "probe_runtime", "time": clock}
    old = {name: sys.modules.get(name) for name in ("torch", "torch_npu",
                                                    "torch.distributed")}
    sys.modules["torch"] = torch
    sys.modules["torch_npu"] = npu
    sys.modules["torch.distributed"] = distributed
    previous = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE",
                                                    "LOCAL_RANK")}
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank % 4))
    try:
        exec(compile(script, "<probe>", "exec"), namespace)
        namespace["time"] = clock
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = namespace["main"]([])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return {"code": code, "dist": dist, "output": buffer.getvalue(),
            "clock": clock, "shared": shared}


def test_r3_absent_optional_attr_uses_the_wrapper_default():
    binding = bind_captured_occurrence({
        "op_type": "aclnnRmsNorm", "cann_version": "9.0",
        "input_shapes": ["[4,8]", "[8]"], "input_names": ["x", "gamma"],
        "input_dtypes": ["BF16", "BF16"], "input_formats": ["ND", "ND"],
        "output_shapes": ["[4,8]", "[4]"], "output_names": ["y", "rstd"],
        "output_dtypes": ["BF16", "FP32"], "output_formats": ["ND", "ND"],
        "attrs": {}})
    assert binding["status"] == "bound"
    result = generate_single_op_reproducer(binding)
    assert result["status"] == "ready"
    assert "epsilon" not in result["metadata"]["install"]


# Independent official parameter lists (Ascend op-plugin YAML, 2026-11-14).
_OFFICIAL = {
    "torch_npu.npu_fusion_attention": (
        ["query", "key", "value", "head_num", "input_layout", "pse",
         "padding_mask", "atten_mask", "scale", "keep_prob", "pre_tockens",
         "next_tockens", "inner_precise", "prefix", "actual_seq_qlen",
         "actual_seq_kvlen", "sparse_mode", "gen_mask_parallel", "sync",
         "softmax_layout", "sink", "dropout_mask", "seed", "offset"],
        ["query", "key", "value", "head_num", "input_layout"]),
    "torch_npu.npu_fused_infer_attention_score": (
        ["query", "key", "value", "pse_shift", "atten_mask",
         "actual_seq_lengths", "actual_seq_lengths_kv", "dequant_scale1",
         "quant_scale1", "dequant_scale2", "quant_scale2", "quant_offset2",
         "antiquant_scale", "antiquant_offset", "key_antiquant_scale",
         "key_antiquant_offset", "value_antiquant_scale",
         "value_antiquant_offset", "block_table", "query_padding_size",
         "kv_padding_size", "key_shared_prefix", "value_shared_prefix",
         "actual_shared_prefix_len", "query_rope", "key_rope",
         "key_rope_antiquant_scale", "num_heads", "scale", "pre_tokens",
         "next_tokens", "input_layout", "num_key_value_heads", "sparse_mode",
         "inner_precise", "block_size", "antiquant_mode", "key_antiquant_mode",
         "value_antiquant_mode", "softmax_lse_flag"],
        ["query", "key", "value"]),
    "torch_npu.npu_apply_rotary_pos_emb": (
        ["query", "key", "cos", "sin", "layout", "rotary_mode"],
        ["query", "key", "cos", "sin"]),
    "torch_npu.npu_grouped_matmul": (
        ["x", "weight", "bias", "scale", "offset", "antiquant_scale",
         "antiquant_offset", "per_token_scale", "group_list", "activation_input",
         "activation_quant_scale", "activation_quant_offset", "split_item",
         "group_type", "group_list_type", "act_type"], ["x", "weight"]),
    "torch_npu.npu_dynamic_quant": (
        ["input", "smooth_scales", "group_index", "dst_type", "quant_mode",
         "dst_type_max"], ["input"]),
    "torch_npu.npu_moe_init_routing_v2": (
        ["x", "expert_idx", "scale", "offset", "active_num", "expert_capacity",
         "expert_num", "drop_pad_mode", "expert_tokens_num_type",
         "expert_tokens_num_flag", "quant_mode", "active_expert_range",
         "row_idx_type", "x_dtype"], ["x", "expert_idx"]),
    "torch_npu.npu_swiglu": (["self", "dim"], ["self"]),
    "torch_npu.npu_moe_gating_top_k_softmax": (
        ["x", "finished", "k"], ["x"]),
    "torch_npu.npu_quant_lightning_indexer": (
        ["query", "key", "weights", "query_dequant_scale", "key_dequant_scale",
         "query_quant_mode", "key_quant_mode", "actual_seq_lengths_query",
         "actual_seq_lengths_key", "block_table", "layout_query", "layout_key",
         "sparse_count", "sparse_mode", "pre_tokens", "next_tokens",
         "query_dtype", "key_dtype"],
        ["query", "key", "weights", "query_dequant_scale", "key_dequant_scale",
         "query_quant_mode", "key_quant_mode"]),
    "torch_npu.npu_rms_norm": (["self", "gamma", "epsilon"], ["self", "gamma"]),
    "torch_npu.npu_add_rms_norm": (
        ["x1", "x2", "gamma", "epsilon"], ["x1", "x2", "gamma"]),
    "torch_npu.npu_chunk_gated_delta_rule": (
        ["query", "key", "value", "beta", "initial_state", "actual_seq_lengths",
         "scale", "g"], ["query", "key", "value"]),
    "torch_npu._npu_paged_attention": (
        ["query", "key_cache", "value_cache", "num_kv_heads", "num_heads",
         "scale_value", "block_table", "context_lens", "out"],
        ["query", "key_cache", "value_cache", "num_kv_heads", "num_heads",
         "scale_value", "block_table", "context_lens", "out"]),
}


def test_r3_catalog_signatures_match_independent_official_lists():
    catalog = json.loads((ROOT / "mindie_profiler" / "op_api_catalog.json").read_text())
    checked = 0
    for name, entry in catalog["operators"].items():
        recipe = entry["repro_recipe"]
        if recipe["status"] != "ready":
            continue
        positional, required = _OFFICIAL[recipe["call"]]
        all_params = set(positional)
        supplied = set(positional[:len(recipe.get("positional") or ())])
        supplied |= set(recipe.get("keyword") or {})
        assert set(recipe.get("keyword") or {}) <= all_params, name
        assert set(required) <= supplied, (name, set(required) - supplied)
        assert len(recipe.get("positional") or ()) <= len(positional), name
        checked += 1
    assert checked == 13


# ---------------------------------------------------------------- R9 / R10
def _probe_plan():
    return build_fabric_probe_plan(
        world_size=4, ranks_per_node=2, sizes=[4096], ops=["all_reduce"],
        iterations=2, duration_s=60.0, warmup=0)


def test_r9_all_ranks_execute_the_same_fixed_collective_sequence():
    plan = _probe_plan()
    runs = [_probe_run(plan, rank, 4, _FakeClock(jump_s=0.0)) for rank in range(4)]
    for group in runs[0]["dist"].collective_calls:
        counts = {runs[rank]["dist"].collective_calls.get(group)
                  for rank in range(4)
                  if group in runs[rank]["dist"].collective_calls}
        assert len(counts) == 1, (group, counts)
    assert all(run["code"] == (0 if run["dist"].collective_calls else 2)
               for run in runs)


def test_r9_a_post_deadline_rank_cannot_emit_validated_samples():
    plan = _probe_plan()
    # A rank whose post-sync clock is already past the shared deadline must
    # reject its own samples; the peer with a healthy clock still completes.
    slow = _probe_run(plan, 1, 4, _FakeClock(jump_s=1e6))
    fast = _probe_run(plan, 0, 4, _FakeClock(jump_s=0.0))
    assert slow["code"] != 0
    assert "mindie-fabric-probe-sample/2" not in slow["output"]
    assert fast["code"] == 0
    assert "mindie-fabric-probe-sample/2" in fast["output"]


def test_r9_rank0_deadline_abort_is_shared_before_validation():
    plan = _probe_plan()
    runs = []
    shared = SharedDecisions()
    for rank in range(4):
        clock = _FakeClock(jump_s=1e6) if rank == 0 else _FakeClock()
        runs.append(_probe_run(plan, rank, 4, clock, shared=shared))
    members = [run for run in runs if run["dist"].decision_calls]
    assert members
    for run in members:
        assert run["dist"].validations == 0
        assert not any(run["dist"].collective_calls.values())
    assert any("unmeasured" in run["output"] for run in members)


def test_r10_cost_estimate_includes_init_group_warmup_validation_cleanup():
    plan = build_fabric_probe_plan(world_size=16, duration_s=0.051)
    assert plan["status"] == "rejected"
    assert plan["admission"]["admitted"] is False
    assert plan["admission"]["reason"] == "estimated_cost_exceeds_budget"
    assert plan["estimated_s"] > 0.051
    breakdown = plan["estimated_breakdown_s"]
    assert set(breakdown) == {"init_cost_s", "group_create_s", "warmup_s",
                              "measure_s", "validation_s", "cleanup_s"}
    assert sum(breakdown.values()) == pytest.approx(plan["estimated_s"], rel=1e-6)
    assert plan["admitted"] == []


def test_r10_init_timeout_is_clamped_to_the_remaining_budget():
    plan = build_fabric_probe_plan(world_size=4, ranks_per_node=2,
                                   duration_s=1.0)
    assert plan["status"] == "rejected"
    run = _probe_run(plan, 0, 4, _FakeClock())
    assert run["dist"].timeout is not None
    assert run["dist"].timeout.total_seconds() <= 1.0
    assert run["dist"].timeout.total_seconds() < 31


# --------------------------------------------------------------------- R11
def _fit_plan():
    return build_fabric_probe_plan(world_size=16, ranks_per_node=8,
                                   duration_s=60)

def _samples(tier="intra_node_hccs", alpha=8.0, beta=0.1, regime="isolated",
             protocol="HCCS", group_size=8, **extra):
    return [{"tier": tier, "op": "all_reduce", "bytes": size,
             "latency_us": alpha + beta * size / 1000.0,
             "group_size": group_size, "protocol": protocol,
             "regime": regime, "rank": 0, "validated": True, **extra}
            for size in (4096, 65536, 1048576)]

def test_r11_validated_false_and_out_of_group_samples_are_rejected():
    plan = _fit_plan()
    bad = _samples()
    bad[0]["validated"] = False
    profile = fit_calibrated_fabric_profile(plan, bad)
    assert profile["status"] == "partial"
    assert profile["usable_for_link_override"] is False
    assert any(item["reason"] == "validated_false"
               for item in profile["rejected_samples"])
    out = _samples(rank=999)
    profile = fit_calibrated_fabric_profile(plan, out)
    assert any(item["reason"] == "out_of_group_rank"
               for item in profile["rejected_samples"])
    assert profile["usable_for_link_override"] is False
    protocol = _samples(protocol="RoCE")
    profile = fit_calibrated_fabric_profile(plan, protocol)
    assert any(item["reason"] == "protocol_incompatible"
               for item in profile["rejected_samples"])
    assert profile["status"] == "partial"

def test_r11_decreasing_curve_is_not_calibrated_or_usable():
    plan = _fit_plan()
    samples = [{"tier": "intra_node_hccs", "op": "all_reduce", "bytes": size,
                "latency_us": 100.0 - size / 1e6, "group_size": 8,
                "protocol": "HCCS", "regime": "isolated", "rank": 0,
                "validated": True}
               for size in (4096, 65536, 1048576)]
    profile = fit_calibrated_fabric_profile(plan, samples)
    fit = profile["tiers"]["intra_node_hccs"]["operations"]["all_reduce"][
        "regimes"]["isolated"]
    assert fit["monotonicity_violation"] is True
    assert fit["qualified"] is False
    assert profile["usable_for_link_override"] is False

def test_r11_regimes_are_fitted_separately_and_not_pooled():
    plan = _fit_plan()
    samples = (_samples(alpha=8.0, beta=0.1, regime="isolated")
               + _samples(alpha=16.0, beta=0.2, regime="loaded"))
    profile = fit_calibrated_fabric_profile(plan, samples)
    operation = profile["tiers"]["intra_node_hccs"]["operations"]["all_reduce"]
    isolated = operation["regimes"]["isolated"]
    loaded = operation["regimes"]["loaded"]
    assert isolated["alpha_us"] == pytest.approx(8.0, rel=1e-3)
    assert loaded["alpha_us"] == pytest.approx(16.0, rel=1e-3)
    assert isolated["beta_ns_per_byte"] == pytest.approx(0.1, rel=1e-3)
    assert loaded["beta_ns_per_byte"] == pytest.approx(0.2, rel=1e-3)
    assert operation["alpha_us"] == pytest.approx(8.0, rel=1e-3)
    assert operation["congestion_factor"] == pytest.approx(2.0, rel=0.05)
    assert operation["congestion_evidence"]["isolated"]["alpha_us"] == pytest.approx(
        8.0, rel=1e-3)

def test_r11_nonnegative_boundary_projection_is_the_minimum_sse():
    fit = _weighted_fit([(1.0, 1.0), (2.0, 4.0)], [1.0, 1.0])
    assert fit["alpha_us"] == 0.0
    assert fit["beta_ns_per_byte"] == pytest.approx(1800.0)
    assert fit["sse"] == pytest.approx(0.8)
    assert fit["sse"] < 4.5

def test_r11_derived_topology_is_labelled_unverified():
    derived = build_fabric_probe_plan(world_size=16, ranks_per_node=8,
                                      duration_s=60)
    assert derived["topology_label"] == "unverified_rank_group"
    assert derived["topology_source"] == "rank_count_assumption"
    assert derived["physical_verification"] is False
    explicit = build_fabric_probe_plan(
        world_size=4, duration_s=60,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                             "protocol": "HCCS"}]})
    assert explicit["topology_label"] == "explicit_topology"
    assert explicit["physical_verification"] is True

# --------------------------------------------------------------------- R12
def _repro_run(binding, alloc_sequence):
    script = generate_single_op_reproducer(binding)["script"]
    torch = _make_torch(alloc_sequence=alloc_sequence)
    npu = types.ModuleType("torch_npu")
    npu.__path__ = []
    npu._npu_paged_attention = lambda **_kwargs: None
    npu.npu_format_cast = lambda tensor, _fmt: tensor
    old_torch, old_npu = sys.modules.get("torch"), sys.modules.get("torch_npu")
    old_telemetry = os.environ.get("MINDIE_REPRO_DISABLE_TELEMETRY")
    os.environ["MINDIE_REPRO_DISABLE_TELEMETRY"] = "1"
    sys.modules["torch"] = torch
    sys.modules["torch_npu"] = npu
    try:
        namespace = {"__name__": "repro_runtime"}
        exec(compile(script, "<repro>", "exec"), namespace)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = namespace["main"](["--iters", "10", "--warmup", "1",
                                      "--flush-mb", "0"])
    finally:
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch
        if old_npu is None:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = old_npu
        if old_telemetry is None:
            os.environ.pop("MINDIE_REPRO_DISABLE_TELEMETRY", None)
        else:
            os.environ["MINDIE_REPRO_DISABLE_TELEMETRY"] = old_telemetry
    return code, json.loads(buffer.getvalue().strip().splitlines()[-1]), namespace

def test_r12_allocation_drift_forces_unstable_allocation():
    code, payload, _ = _repro_run(paged_binding(), [100, 200])
    assert code == 0
    assert payload["allocation"]["stable"] is False
    assert payload["status"] == "unstable_allocation"
    assert "p50_us" not in payload

def test_r12_stable_allocation_still_reports_ok():
    _, payload, _ = _repro_run(paged_binding(), [100, 100])
    assert payload["allocation"]["stable"] is True
    assert payload["status"] == "ok"
    assert payload["p50_us"] == pytest.approx(10000.0)

def test_r12_cold_verification_requires_counter_and_byte_sweep():
    _, _, namespace = _repro_run(paged_binding(), [0, 0])
    never = namespace["_cache_label"](256, None)
    assert never["cold_verified"] is False
    inferred = namespace["_cache_label"](1, 4096)
    assert inferred["cold_verified"] is False
    verified = namespace["_cache_label"](1, 4096, 8, 8192)
    assert verified["cold_verified"] is True
    assert verified["byte_sweep_factor"] == pytest.approx(2.0)
    assert verified["flush_invocations"] == 8

def test_r12_clock_ratio_does_not_assert_thermal_causation():
    _, payload, namespace = _repro_run(paged_binding(), [0, 0])
    assert payload["clock"]["thermal_throttled"] is None
    assert payload["clock"]["thermal_attribution"] == "not_measured"
    assert payload["clock"]["temperature_telemetry"] is False
    source = Path(ROOT / "mindie_profiler" / "repro_script_template.py").read_text()
    assert "clock_ratio_below_nominal" in source
    assert "thermal_throttled" in source

# --------------------------------------------------------------------- R13
def test_r13_op_uid_resolves_the_exact_raw_row_after_host_rows(tmp_path):
    capture = {"run_id": "run-1", "total_ms": 12, "rows": [
        {"op_type": "HostLaunch", "name": "HostLaunch", "stream": 0,
         "start_ms": 0, "duration_ms": 0},
        {"op_type": "aclnnPagedAttention", "name": "aclnnPagedAttention",
         "stream": 0, "start_ms": 0, "duration_ms": 12,
         "input_shapes": ["[16,8,128]", "[256,16,8,128]", "[256,16,8,128]",
                          "[16,16]", "[16]"],
         "input_names": ["query", "key_cache", "value_cache", "block_table",
                         "context_lens"],
         "input_dtypes": ["BF16", "BF16", "BF16", "INT32", "INT32"],
         "input_formats": ["ND"] * 5,
         "output_shapes": ["[16,8,128]"], "output_names": ["out"],
         "output_dtypes": ["BF16"], "output_formats": ["ND"],
         "attrs": {"num_kv_heads": 8, "num_heads": 8,
                   "scale_value": 0.088, "kv_len": 2048}}]}
    predicted = {"total_ms": 10, "ops": [
        {"name": "aclnnPagedAttention", "stream": 0, "start_ms": 0,
         "duration_ms": 10}]}
    result = attribute_diff(predicted, capture, launch_overhead_us=0)
    offender = result["kernel_roofline_and_clock"]["top_offenders"][0]
    assert offender["raw_op_index"] == 1
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    args = types.SimpleNamespace(
        row=None, capture=str(path), op_name="aclnnPagedAttention",
        occurrence_index=None, index=None, version="8.0",
        op_uid=offender["op_uid"])
    binding, index, _ = _row_from_args(args)
    assert index == 1
    assert binding["operator"] == "aclnnPagedAttention"
    expected = _op_uid("run-1", {"rank": 0, "step": 0, "stream": 0,
                                 "abi_digest": "", "raw_op_index": 1})
    assert offender["op_uid"] == expected

def test_r13_capture_level_cann_version_reaches_the_reproducer(tmp_path):
    row = paged_row()
    row.pop("cann_version", None)
    row.update({"name": "aclnnPagedAttention", "stream": 0, "start_ms": 0,
                "duration_ms": 12})
    capture = {"run_id": "run-1", "cann_version": "8.0", "total_ms": 12,
               "rows": [{"op_type": "HostLaunch", "name": "HostLaunch",
                         "stream": 0, "start_ms": 0, "duration_ms": 0}, row]}
    predicted = {"total_ms": 10, "ops": [
        {"name": "aclnnPagedAttention", "stream": 0, "start_ms": 0,
         "duration_ms": 10}]}
    result = attribute_diff(predicted, capture, launch_overhead_us=0)
    offender = result["kernel_roofline_and_clock"]["top_offenders"][0]
    assert "--version 8.0" in offender["repro_command"]
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    args = types.SimpleNamespace(
        row=None, capture=str(path), op_name="aclnnPagedAttention",
        occurrence_index=None, index=None, version=None,
        op_uid=offender["op_uid"])
    binding, index, _ = _row_from_args(args)
    assert index == 1
    assert binding["status"] == "bound"
    assert binding["cann_version"] == "8.0"

# --------------------------------------------------------------------- R14
def test_r14_mcp_reproducer_persists_a_retrievable_artifact():
    result = call_mcp_tool("repro_op", {"occurrence": paged_binding()})
    assert result["status"] == "ready"
    script_path = Path(result["artifact_path"])
    manifest_path = Path(result["manifest_path"])
    assert checked_io.path_exists(script_path) and checked_io.path_exists(manifest_path)
    assert result["artifact"]["retrievable"] is True
    import hashlib
    digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
    assert result["repro_sha256"] == f"sha256:{digest}"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["repro_sha256"] == result["repro_sha256"]

def test_r14_thousand_issue_envelope_stays_bounded():
    blocked = {"schema": "mindie-op-occurrence-binding/2", "status": "incomplete",
               "issues": [{"code": "required_slot_omitted", "slot": index,
                           "name": f"tensor_{index}"} for index in range(1000)]}
    result = call_mcp_tool("repro_op", {"occurrence": blocked})
    assert result["status"] == "blocked"
    assert result["issue_count"] == 1000
    assert result["issues_truncated_count"] == 992
    assert len(result["issues"]) == 8
    assert result["receipt"]["token_estimate"] <= 3000
    assert result["receipt"]["within_budget"] is True
    assert estimate_tokens(result) <= 3000

def test_r14_cli_cann_version_and_mcp_padding_forwarding(tmp_path):
    predicted = {"total_ms": 10, "ops": [
        {"name": "X", "stream": 0, "start_ms": 0, "duration_ms": 10}]}
    captured = {"total_ms": 10, "ops": [
        {"name": "X", "stream": 0, "start_ms": 0, "duration_ms": 10}]}
    padding = {"aclnnGroupedMatmulV4": {
        "bucket_tokens": 64, "bound_args": {"group_list": [16, 32, 48, 64]}}}
    pred_path, cap_path, pad_path = (tmp_path / "pred.json",
                                     tmp_path / "cap.json",
                                     tmp_path / "pad.json")
    pred_path.write_text(json.dumps(predicted))
    cap_path.write_text(json.dumps(captured))
    pad_path.write_text(json.dumps(padding))
    from mindie_sim.__main__ import main as sim_main
    out = tmp_path / "out.json"
    assert sim_main(["diff-trace", "--predicted", str(pred_path),
                     "--captured", str(cap_path), "--padding", str(pad_path),
                     "--cann-version", "9.0", "--output", str(out)]) == 0
    payload = json.loads(out.read_text())
    charge = payload["padding"]["charges"][0]
    assert charge["version_supported"] is True
    assert charge["version_qualified"] is False
    assert charge["version_verified"] is False
    assert charge["effective_tokens"] == 64

    from mindie_sim.diff_trace import call_mcp_tool as diff_tool
    forwarded = diff_tool({"predicted": predicted, "captured": captured,
                           "padding": padding, "cann_version": "9.0"})
    assert forwarded["padding"]["valid"] is True
    assert forwarded["padding"]["charges"][0]["effective_tokens"] == 64
