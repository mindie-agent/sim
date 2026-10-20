"""Graph capture, correctness preparation and per-condition device setup.

[Role]
  Own the single explicit measurement stream, build deterministic static
  buffers, run the untimed eager pointer diagnostic, capture the actual
  ``torch.npu.NPUGraph`` with global capture error mode, replay once for
  correctness, mutate/restore the static input in place, warm the selected
  callable and hand a ready artifact to the timing mixin.  Per-trial rewarm lives
  in the timing mixin so it happens before every trial.  Torch is imported only
  by ``setup`` for a real run.

[Invariants]
  * Exactly one explicit measurement stream is created and reused.
  * RMSNorm/SwiGLU use installed default outputs only; copy uses one same
    preallocated destination.  No native ``.out`` API is claimed.
  * Capture always passes ``stream``, ``auto_dispatch_capture=False`` and
    ``capture_error_mode="global"``.  A capture failure never becomes eager
    timing and never relaxes the error mode.
  * The capture body contains only exact operator calls plus host pointer
    bookkeeping.  Setup/synchronize/empty_cache and warmup are outside samples.
  * Only the last native output reference is retained.  Capture and eager
    pointer sequences are recorded separately, without causal claims.
  * Static input changes and reference checks occur outside timing.  Eager
    diagnostic/correctness outputs are released before capture; only the captured
    graph final output is retained with the graph.  Per-trial rewarm is outside
    timing and recorded by the timing mixin.

[Timeouts & Locks]
  Cooperative case/global/setup checks around stages.  One stream, one graph at
  a time, no locks/threads; native calls remain bounded by the outer supervisor.

[Side-Effect Policy]
  Allocates NPU tensors, captures/replays graphs and mutates the declared static
  input only in a validated real run.  Writes an optional debug dump only to the
  explicit debug directory.
"""
import importlib
import re
import time
from pathlib import Path
from typing import Any, Mapping

from .a3_operator_graph_reference import (
    build_condition_tensors,
    check_against_reference,
    output_facts,
    result_tensors,
    tensor_facts,
    unique_pointer_count,
)
from .a3_operator_graph_device_common import (
    BudgetClock,
    CaseBudgetExceeded,
    ConditionArtifact,
    ConditionFailure,
    DeviceSetupError,
    GlobalWorkBudgetExceeded,
    PendingCondition,
    _collect_environment,
    _memory_facts,
    exception_info,
)
from .a3_operator_graph_timing import TimingMixin

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

class DeviceRunner(TimingMixin):
    """One explicit stream plus per-condition graph/correctness/timing state."""

    def __init__(self, *, device: str, device_index: int, budgets: Mapping[str, Any],
                 debug_dir: str | None) -> None:
        self.device = device
        self.device_index = device_index
        self.budgets = {key: float(value) for key, value in budgets.items()
                        if key in ("case", "global_work", "setup", "cleanup")}
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.torch: Any = None
        self.torch_npu: Any = None
        self.device_obj: Any = None
        self.stream: Any = None
        self.environment: dict[str, Any] = {}
        self.clock: BudgetClock | None = None
        self._graphs: list[tuple[str, Any]] = []
        self._debug_dir_ready = False

    def setup(self, clock: BudgetClock) -> dict[str, Any]:
        start = time.monotonic()
        self.clock = clock
        clock.check_setup("import")
        try:
            self.torch = importlib.import_module("torch")
            self.torch_npu = importlib.import_module("torch_npu")
        except Exception as exc:
            raise DeviceSetupError(
                f"cannot import installed torch/torch_npu stack: {exc}") from exc
        clock.check_setup("availability")
        torch = self.torch
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise DeviceSetupError("torch has no torch.npu namespace")
        if not bool(npu.is_available()):
            raise DeviceSetupError("torch.npu.is_available() is false")
        clock.check_setup("set_device")
        npu.set_device(self.device_index)
        self.device_obj = torch.device(self.device)
        npu.synchronize()
        clock.check_setup("stream")
        self.stream = npu.Stream(device=self.device_index)
        npu.synchronize()
        clock.check_setup("environment")
        self.environment = _collect_environment(
            torch, self.torch_npu, self.device_obj, self.stream, self.device_index)
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_dir_ready = True
        clock.check_setup("setup_complete")
        return {
            "status": "ok",
            "duration_s": time.monotonic() - start,
            "torch_imported": True,
            "torch_npu_imported": True,
            "device_set": True,
            "measurement_stream_created_once": True,
            "initial_synchronize": True,
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
            "torch_inference_mode": bool(torch.is_inference_mode_enabled()),
        }

    def _sync(self) -> None:
        self.stream.synchronize()

    def _check_case(self, start: float, remaining_s: float, stage: str) -> None:
        elapsed = time.monotonic() - start
        if elapsed > remaining_s:
            raise CaseBudgetExceeded(stage, elapsed, remaining_s)

    def _check_deadlines(self, start: float, remaining_s: float,
                         stage: str) -> None:
        if self.clock is not None:
            self.clock.check_work(stage)
        self._check_case(start, remaining_s, stage)

    def _metadata(self, condition: Mapping[str, Any],
                  protocol: Mapping[str, Any]) -> dict[str, Any]:
        graph_mode = condition["mode"] != "eager_block"
        return {
            "execution": {
                "measurement_stream": repr(self.stream),
                "stream_policy": "one_explicit_measurement_stream_created_once",
                "same_stream_for_all_modes": True,
                "inference_mode": bool(
                    self.torch.is_inference_mode_enabled()),
                "timing_events": {"status": "not_created",
                                  "enable_timing": True,
                                  "precreated": True},
                "cache_policy": {
                    "graph_context_entry": (
                        "installed torch.npu.graph context performs "
                        "torch.npu.synchronize() and empty_cache() before "
                        "capture_begin" if graph_mode else
                        "not_applicable_eager_block"),
                    "explicit_empty_cache_calls": 0,
                    "setup_and_capture_excluded_from_samples": True,
                    "warmup_iterations": int(protocol["warmup_iterations"]),
                    "warmup_per_trial": True,
                    "warmup_scope": (
                        "exact callable/replay rewarmed before every trial; "
                        "excluded from timing arrays and counted in budgets"),
                    "warmup_excluded_from_samples": True,
                    "between_measured_blocks":
                        "measurement-stream synchronize outside all clocks",
                },
                "timed_body_extra_work": False,
                "output_lifetime": "retain_last_only",
                "output_lifetime_policy": {
                    "native": "retain_only_the_last_native_output_reference",
                    "eager": (
                        "only the current timed block's final output is retained; "
                        "the previous owner is released before the next block and "
                        "no warmup/diagnostic output is retained"),
                    "graph": (
                        "the captured final output is retained with the graph until "
                        "post-work cleanup"),
                    "warmup_output_retained": False,
                    "diagnostic_output_released": True,
                },
                "pool_policy": ("graph_private_pool_no_sharing" if graph_mode
                                else "no_graph_eager_block"),
                "callable_source": condition["callable_source"],
                "capture_error_mode": "global" if graph_mode else "not_applicable",
                "auto_dispatch_capture": False if graph_mode else None,
            },
            "graph_capture": {
                "attempted": False,
                "status": "not_attempted",
                "timing_excluded": True,
                "no_eager_fallback": True,
            },
            "allocation": {
                "output_policy": condition["output_policy"],
                "native_out_overload": False,
                "preallocated_output": bool(condition["preallocated_output"]),
                "output_lifetime": "retain_last_only",
                "pool_sharing": {
                    "shared": False,
                    "policy": "none" if graph_mode else "not_applicable_eager_block",
                    "per_condition_graph": graph_mode,
                },
                "capture_output_pointer_sequence": [],
                "eager_diagnostic": {"performed": False},
                "declared_capture_output_bytes": int(
                    condition["declared_capture_output_bytes"]),
                "declared_capture_output_bytes_kind":
                    condition["declared_capture_output_bytes_kind"],
            },
            "correctness": None,
            "debug_dump": {
                "requested": self.debug_dir is not None,
                "status": ("not_attempted" if self.debug_dir is not None
                           else "not_requested"),
                "timing_excluded": True,
                "substitutes_for_measured_task_count": False,
            },
            "active_case_seconds": 0.0,
        }

    def _ensure_operator_available(self, condition: Mapping[str, Any]) -> None:
        name = ("npu_rms_norm" if condition["operator"] == "rms_norm"
                else "npu_swiglu" if condition["operator"] == "swiglu" else None)
        if name is not None and not callable(getattr(self.torch_npu, name, None)):
            raise PendingCondition(
                f"torch_npu.{name} is unavailable in the installed stack")
        if condition["mode"] != "eager_block":
            if not callable(getattr(self.torch.npu, "graph", None)):
                raise PendingCondition("torch.npu.graph context is unavailable")
            if getattr(self.torch.npu, "NPUGraph", None) is None:
                raise PendingCondition("torch.npu.NPUGraph is unavailable")

    def _eager_diagnostic(self, condition: Mapping[str, Any], tensors: Any,
                          warmup_floor: int) -> tuple[Any, dict[str, Any]]:
        block_calls = int(condition["block_invocations"])
        calls = max(block_calls, int(warmup_floor))
        pointers: list[list[int]] = []
        last = None
        with self.torch.npu.stream(self.stream):
            for _ in range(calls):
                last = tensors.callable()
                pointers.append([
                    int(tensor.data_ptr())
                    for tensor in result_tensors(self.torch, last)
                ])
        self._sync()
        return last, {
            "performed": True,
            "phase": "before_capture",
            "timed": False,
            "calls": calls,
            "declared_block_invocations": block_calls,
            "warmup_floor": int(warmup_floor),
            "output_pointer_sequence": pointers,
            "unique_address_count": unique_pointer_count(pointers),
            "last_output": output_facts(self.torch, last),
            "synchronized": True,
            "note": (
                "eager allocator diagnostic only; graph private-pool reuse is "
                "reported separately and no causal tax is inferred"),
        }

    def _debug_dump(self, graph: Any, condition: Mapping[str, Any],
                    metadata: dict[str, Any]) -> None:
        info = metadata["debug_dump"]
        if self.debug_dir is None:
            return
        dumper = getattr(graph, "debug_dump", None)
        if not callable(dumper):
            info.update({"status": "unsupported",
                         "reason": "torch.npu.NPUGraph.debug_dump is unavailable"})
            return
        name = _SAFE_NAME_RE.sub("_", condition["condition_id"]) + ".json"
        path = self.debug_dir / name
        started = time.monotonic()
        try:
            dumper(str(path))
            info.update({
                "status": "ok",
                "path": str(path),
                "duration_s": time.monotonic() - started,
                "file_present": path.exists(),
                "file_bytes": path.stat().st_size if path.exists() else None,
                "note": "structural dump; not a measured task count",
            })
        except Exception as exc:
            info.update({
                "status": "error",
                "path": str(path),
                "duration_s": time.monotonic() - started,
                "error": exception_info(exc),
                "note": (
                    "optional dump failure does not invalidate otherwise good "
                    "measured replay timing"),
            })

    def _capture(self, condition: Mapping[str, Any], tensors: Any,
                 metadata: dict[str, Any], start: float,
                 remaining_s: float) -> tuple[Any, Any, dict[str, Any]]:
        graph = self.torch.npu.NPUGraph()
        self._graphs.append((condition["condition_id"], graph))
        capture_invocations = int(condition["capture_invocations"])
        pointers: list[list[int]] = []
        last = None
        capture_meta = metadata["graph_capture"]
        capture_meta.update({
            "attempted": True,
            "status": "capturing",
            "npu_graph_class": "torch.npu.NPUGraph",
            "context": (
                "torch.npu.graph(g, stream=<measurement stream>, "
                "auto_dispatch_capture=False, capture_error_mode='global')"),
            "auto_dispatch_capture": False,
            "capture_error_mode": "global",
            "captured_invocations": capture_invocations,
            "timing_excluded": True,
            "setup_and_entry_sync_empty_cache_excluded": True,
        })
        try:
            with self.torch.npu.graph(
                    graph, stream=self.stream, auto_dispatch_capture=False,
                    capture_error_mode="global"):
                for _ in range(capture_invocations):
                    last = tensors.callable()
                    pointers.append([
                        int(tensor.data_ptr())
                        for tensor in result_tensors(self.torch, last)
                    ])
        except Exception as exc:
            capture_meta.update({
                "status": "failed",
                "error": exception_info(exc),
                "captured_output_pointer_sequence": pointers,
                "timing_excluded": True,
                "no_eager_fallback": True,
                "capture_error_mode_relaxed": False,
            })
            raise ConditionFailure("capture", exc, metadata) from exc
        self._sync()
        self._check_deadlines(start, remaining_s, "capture")
        pool: dict[str, Any] = {"shared": False, "policy": "none"}
        try:
            pool["token_repr"] = repr(graph.pool())[:500]
            pool["queried"] = True
        except Exception as exc:
            pool.update({"queried": True, "error": exception_info(exc)})
        primary = [group[0] for group in pointers if group]
        capture_meta.update({
            "status": "captured",
            "captured_output_pointer_sequence": pointers,
            "primary_output_pointer_sequence": primary,
            "primary_unique_address_count": len(set(primary)),
            "unique_address_count": unique_pointer_count(pointers),
            "sequence_length": len(pointers),
            "addresses_reused_within_capture": (
                unique_pointer_count(pointers) < sum(len(group) for group in pointers)),
            "last_output": output_facts(self.torch, last),
            "final_output_pointer": int(
                result_tensors(self.torch, last)[0].data_ptr()),
            "pool": pool,
            "timing_excluded": True,
        })
        self._debug_dump(graph, condition, metadata)
        self._check_deadlines(start, remaining_s, "debug_dump")
        return graph, last, capture_meta

    def _replay_once(self, graph: Any) -> None:
        with self.torch.npu.stream(self.stream):
            graph.replay()
        self._sync()

    def _call_once(self, tensors: Any) -> Any:
        with self.torch.npu.stream(self.stream):
            result = tensors.callable()
        self._sync()
        return result

    def _mutate_and_check(self, condition: Mapping[str, Any], tensors: Any,
                          graph: Any, capture_last: Any,
                          metadata: dict[str, Any],
                          protocol: Mapping[str, Any], start: float,
                          remaining_s: float) -> None:
        correctness = metadata["correctness"]
        static_pointer_before = int(tensors.static_device.data_ptr())
        stale: dict[str, Any] = {
            "performed": True,
            "input_pointer_before": static_pointer_before,
            "input_bytes_sha256_before": tensors.data_sha256,
            "mutated_bytes_sha256": tensors.mutation_sha256,
            "restored_bytes_sha256_expected": tensors.data_sha256,
            "timing_excluded": True,
        }
        restored = False
        try:
            with self.torch.npu.stream(self.stream):
                tensors.static_device.copy_(tensors.mutation_cpu)
            self._sync()
            after_mutation = int(tensors.static_device.data_ptr())
            stale["input_pointer_after_mutation"] = after_mutation
            stale["mutation_preserved_address"] = (
                after_mutation == static_pointer_before)
            self._replay_once(graph)
            mutated_check = check_against_reference(
                self.torch, condition, tensors, capture_last,
                protocol, mutated=True)
            stale_compare = check_against_reference(
                self.torch, condition, tensors, capture_last,
                protocol, mutated=False)
            stale.update({
                "mutated_output_check": mutated_check,
                "mutated_output_matches_original_reference":
                    bool(stale_compare["passed"]),
                "mutated_output_matches_mutated_reference":
                    bool(mutated_check["passed"]),
                "stale_output_detected": bool(
                    stale_compare["passed"] and not mutated_check["passed"]),
            })
            correctness["stale_output_check"] = stale
            if not stale["mutation_preserved_address"]:
                raise ConditionFailure(
                    "mutation", RuntimeError("in-place input mutation changed "
                                             "the device address"), metadata)
            if not mutated_check["passed"]:
                raise ConditionFailure(
                    "stale_output", RuntimeError(
                        "mutated-input replay did not match the mutated reference "
                        f"(stale_output_detected={stale['stale_output_detected']})"),
                    metadata)
            self._check_deadlines(start, remaining_s, "stale_output_check")
        finally:
            correctness["stale_output_check"] = stale
            try:
                with self.torch.npu.stream(self.stream):
                    tensors.static_device.copy_(tensors.static_cpu)
                self._sync()
                restored_pointer = int(tensors.static_device.data_ptr())
                restored = True
            except Exception as exc:
                stale["restore_error"] = exception_info(exc)
                stale["input_pointer_after_restore"] = None
                stale["restore_preserved_address"] = False
                correctness["stale_output_check"] = stale
                raise ConditionFailure(
                    "restore", exc, metadata) from exc
            stale["input_pointer_after_restore"] = restored_pointer
            stale["restore_preserved_address"] = (
                restored_pointer == static_pointer_before)
            stale["restored"] = restored
            correctness["stale_output_check"] = stale
            if not stale["restore_preserved_address"]:
                raise ConditionFailure(
                    "restore", RuntimeError(
                        "in-place restore changed the device address"), metadata)

    def prepare(self, condition: Mapping[str, Any],
                protocol: Mapping[str, Any],
                active_case_before: float) -> ConditionArtifact:
        case_budget = float(self.budgets["case"])
        remaining = case_budget - float(active_case_before)
        if remaining <= 0.0:
            raise ConditionFailure(
                "case_budget", CaseBudgetExceeded(
                    "prepare", 0.0, case_budget),
                {"active_case_seconds": float(active_case_before)},
                reason="case budget exhausted before condition setup")
        start = time.monotonic()
        metadata = self._metadata(condition, protocol)
        phase = "prepare"
        metadata["active_case_seconds"] = float(active_case_before)
        try:
            self._check_deadlines(start, remaining, phase)
            self._ensure_operator_available(condition)
            phase = "allocation"
            memory_before = _memory_facts(self.torch)
            with self.torch.npu.stream(self.stream):
                tensors = build_condition_tensors(
                    self.torch, self.torch_npu, self.device_obj, condition)
            self._sync()
            self._check_deadlines(start, remaining, phase)
            allocation = metadata["allocation"]
            allocation.update({
                "static_input": tensor_facts(tensors.static_device),
                "input_cpu_sha256": tensors.data_sha256,
                "input_hash_method": tensors.data_hash_method,
                "mutation_cpu_sha256": tensors.mutation_sha256,
                "input_read_only_during_timing": True,
                "same_static_input_for_all_trials": True,
                "pairing_basis": {
                    "source_bytes": "identical seeded CPU bytes for every mode/N "
                                    "of this case",
                    "shared_device_address_across_modes": False,
                    "read_only_during_timing": True,
                    "note": "data identity is recorded by SHA-256; each condition "
                            "owns a fixed device address",
                },
                "memory_before": memory_before,
                "memory_after_allocation": _memory_facts(self.torch),
            })
            if tensors.gamma_device is not None:
                allocation["gamma"] = tensor_facts(tensors.gamma_device)
                allocation["gamma_cpu_sha256"] = tensors.gamma_sha256
            if tensors.target_device is not None:
                allocation["preallocated_target"] = tensor_facts(
                    tensors.target_device)
                allocation["target_written_by_copy"] = True
            phase = "eager_diagnostic"
            diagnostic_last, diagnostic = self._eager_diagnostic(
                condition, tensors, int(protocol["warmup_iterations"]))
            allocation["eager_diagnostic"] = diagnostic
            allocation["eager_diagnostic_last_output"] = output_facts(
                self.torch, diagnostic_last)
            del diagnostic_last
            self._check_deadlines(start, remaining, phase)
            graph = None
            capture_last = None
            if condition["mode"] == "eager_block":
                phase = "eager_correctness"
                output = self._call_once(tensors)
                original = check_against_reference(
                    self.torch, condition, tensors, output, protocol)
                metadata["correctness"] = {
                    "status": original["status"],
                    "passed": original["passed"],
                    "original_output": original,
                    "stale_output_check": {
                        "performed": False,
                        "reason": "eager_block_has_no_captured_output",
                        "timing_excluded": True,
                    },
                    "reference_dtype": "float32_cpu",
                    "comparison_outside_timing": True,
                    "data_repair": "none",
                }
                allocation["correctness_output"] = output_facts(
                    self.torch, output)
                if not original["passed"]:
                    raise ConditionFailure(
                        "eager_correctness", RuntimeError(
                            "eager output did not match FP32 CPU reference"),
                        metadata)
                del output
            else:
                phase = "capture"
                allocation["memory_before_capture"] = _memory_facts(self.torch)
                graph, capture_last, _ = self._capture(
                    condition, tensors, metadata, start, remaining)
                allocation["memory_after_capture"] = _memory_facts(self.torch)
                phase = "post_capture_replay"
                self._replay_once(graph)
                original = check_against_reference(
                    self.torch, condition, tensors, capture_last, protocol)
                metadata["correctness"] = {
                    "status": original["status"],
                    "passed": original["passed"],
                    "original_output": original,
                    "stale_output_check": {
                        "performed": False,
                        "reason": "not_yet_run",
                        "timing_excluded": True,
                    },
                    "reference_dtype": "float32_cpu",
                    "comparison_outside_timing": True,
                    "data_repair": "none",
                }
                if not original["passed"]:
                    raise ConditionFailure(
                        "correctness", RuntimeError(
                            "post-capture replay output did not match FP32 CPU "
                            "reference"), metadata)
                self._check_deadlines(start, remaining, phase)
                phase = "stale_output_check"
                self._mutate_and_check(
                    condition, tensors, graph, capture_last, metadata,
                    protocol, start, remaining)
                self._check_deadlines(start, remaining, phase)
            phase = "allocation_metadata"
            if capture_last is not None:
                allocation["final_output"] = output_facts(
                    self.torch, capture_last)
                allocation["final_output_retained"] = True
                allocation["final_output_capture_last"] = True
            elif condition["mode"] == "eager_block":
                allocation["final_output"] = None
                allocation["final_output_retained"] = False
                allocation["final_output_note"] = (
                    "eager native output exists only for the current timed block; "
                    "no warmup/correctness/diagnostic output is retained")
                if tensors.target_device is not None:
                    allocation["preallocated_destination_retained"] = True
            allocation["capture_output_pointer_sequence"] = (
                metadata["graph_capture"].get(
                    "captured_output_pointer_sequence", []))
            allocation["capture_unique_address_count"] = (
                metadata["graph_capture"].get("unique_address_count", 0))
            allocation["eager_unique_address_count"] = diagnostic[
                "unique_address_count"]
            allocation["eager_vs_capture"] = {
                "eager_unique_addresses": diagnostic["unique_address_count"],
                "capture_unique_addresses": allocation[
                    "capture_unique_address_count"],
                "capture_applicability": (
                    "not_applicable_eager_block"
                    if condition["mode"] == "eager_block"
                    else "captured_graph_private_pool"),
                "note": (
                    "counts are recorded independently; a difference does not "
                    "identify a pure host-dispatch cause"),
            }
            phase = "timing_events"
            start_event = self.torch.npu.Event(enable_timing=True)
            end_event = self.torch.npu.Event(enable_timing=True)
            metadata["execution"]["timing_events"] = {
                "status": "created",
                "enable_timing": True,
                "precreated": True,
                "count": 2,
                "stream": "same_explicit_measurement_stream",
                "empty_event_control": {
                    "measured": False,
                    "subtracted": False,
                    "policy": "no automatic subtraction",
                },
            }
            allocation["memory_after_prepare"] = _memory_facts(self.torch)
            phase = "prepare_complete"
            self._check_deadlines(start, remaining, phase)
            active = float(active_case_before) + (time.monotonic() - start)
            if active > case_budget:
                raise CaseBudgetExceeded(
                    "prepare_complete", active, case_budget)
            metadata["active_case_seconds"] = active
            return ConditionArtifact(
                condition=dict(condition),
                tensors=tensors,
                mode=condition["mode"],
                graph=graph,
                events=(start_event, end_event),
                capture_last=capture_last,
                final_output=capture_last,
                metadata=metadata,
                active_case_seconds=active,
            )
        except ConditionFailure as exc:
            elapsed = time.monotonic() - start
            exc.metadata["active_case_seconds"] = (
                float(active_case_before) + elapsed)
            raise
        except GlobalWorkBudgetExceeded as exc:
            elapsed = time.monotonic() - start
            metadata["active_case_seconds"] = float(active_case_before) + elapsed
            raise ConditionFailure("budget", exc, metadata, stop_run=True) from exc
        except CaseBudgetExceeded as exc:
            elapsed = time.monotonic() - start
            metadata["active_case_seconds"] = float(active_case_before) + elapsed
            raise ConditionFailure("budget", exc, metadata) from exc
        except PendingCondition as exc:
            elapsed = time.monotonic() - start
            metadata["active_case_seconds"] = float(active_case_before) + elapsed
            raise ConditionFailure("availability", exc, metadata,
                                   status="pending") from exc
        except Exception as exc:
            elapsed = time.monotonic() - start
            metadata["active_case_seconds"] = float(active_case_before) + elapsed
            raise ConditionFailure(phase, exc, metadata) from exc

