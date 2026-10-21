"""Opt-in instruction-flow Vector plans and an optional CANNBot CLI adapter.

This module imports no device libraries.  The built-in profile is an explicitly
assumed RegBase design scenario, not a measured A3/A5 calibration.  CANNBot
constants stay in an external, revision-checked checkout; they are never copied
or retuned here.  Instruction issue/pipe limits are service floors, not additive
latencies on top of resource work.  Host scheduling/submission is outside scope.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .operator_plan import EstimateContext, ExecutionStage, KernelEstimate, KernelPlan

CANNBOT_REVISION = "2e2e4ae0fdc7445b114ed707e2ebd39fe4712eb0"
_CANNBOT_SKILL = Path("ops/ops-theoretical-performance-estimation")
_WIDTHS = {"fp16": 2, "bf16": 2, "fp32": 4}
_ALIASES = {"float16": "fp16", "half": "fp16", "float32": "fp32", "float": "fp32", "bfloat16": "bf16"}


def _dtype(value: Any) -> str:
    key = str(value).lower()
    return _ALIASES.get(key, key)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}:numeric_required")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name}:invalid")
    return result


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    result = _number(value, name, positive=positive)
    if int(result) != result:
        raise ValueError(f"{name}:integer_required")
    return int(result)


def _metadata(op: Any) -> dict[str, Any]:
    result = dict(getattr(op, "metadata", ()))
    for name in ("vector_flow", "cannbot_case"):
        if isinstance(result.get(name), str):
            result[name] = json.loads(result[name])
    return result


@dataclass(frozen=True)
class VectorInstruction:
    """One instruction class. Rates are per-core, per-SIMD instruction.

    ``throughput_cycles`` is reciprocal pipe throughput; ``issue_slots`` is
    how many issues the shared unit can accept in a cycle.  Latency is used
    only for an explicit dependency, never inferred from throughput.
    """

    name: str
    unit: str = "alu"
    pipe: str = "alu"
    issue_slots: float = 2.0
    throughput_cycles: float = 1.0
    latency_cycles: float = 1.0
    dtypes: tuple[str, ...] = ("fp16", "fp32")

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.pipe or not self.dtypes:
            raise ValueError("vector_instruction:missing_identity")
        for key in ("issue_slots", "throughput_cycles", "latency_cycles"):
            _number(getattr(self, key), key, positive=True)


@dataclass(frozen=True)
class VectorProfile:
    """Explicit ISA/rate/capacity hypothesis for one supported architecture.

    Built-in defaults are an assumed 950 RegBase scenario. Custom profiles
    should provide their own name/source/architecture and instruction table.
    Memory bandwidth always comes from the case's machine, not this profile.
    """

    name: str = "assumed_regbase_v1"
    source: str = "assumed: RegBase issue/pipe design scenario; not hardware calibration"
    architectures: tuple[str, ...] = ("A5", "ASCEND950", "DAV_3510")
    frequency_hz: float = 1.65e9
    simd_bytes: int = 256
    vector_cores: int = 64
    ub_bytes_per_core: int = 253952
    tile_elements: int = 4096
    buffers: int = 2
    max_tiles: int = 2048
    instructions: tuple[VectorInstruction, ...] = field(default_factory=lambda: (
        VectorInstruction("Add"), VectorInstruction("Sub"), VectorInstruction("Mul"),
        VectorInstruction("Neg"),
        VectorInstruction("Cast", dtypes=("fp16", "bf16", "fp32")),
        VectorInstruction("Exp", pipe="transcendental", throughput_cycles=4, latency_cycles=4),
        VectorInstruction("Div", pipe="transcendental", throughput_cycles=4, latency_cycles=4),
    ))

    def __post_init__(self) -> None:
        if not self.name or not self.source or not self.architectures:
            raise ValueError("vector_profile:missing_provenance")
        _number(self.frequency_hz, "frequency_hz", positive=True)
        for key in ("simd_bytes", "vector_cores", "ub_bytes_per_core", "tile_elements", "buffers", "max_tiles"):
            _integer(getattr(self, key), key, positive=True)
        if len({item.name for item in self.instructions}) != len(self.instructions):
            raise ValueError("vector_profile:duplicate_instruction")
        slots: dict[str, float] = {}
        for item in self.instructions:
            if item.unit in slots and slots[item.unit] != item.issue_slots:
                raise ValueError("vector_profile:inconsistent_unit_issue_slots")
            slots[item.unit] = item.issue_slots

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VectorProfile":
        data = dict(value)
        if "instructions" in data:
            data["instructions"] = tuple(VectorInstruction(**{
                **item, "dtypes": tuple(item.get("dtypes", ("fp16", "fp32"))),
            }) for item in data["instructions"])
        if "architectures" in data:
            data["architectures"] = tuple(data["architectures"])
        return cls(**data)


def _auto_flow(op: Any, context: EstimateContext) -> Mapping[str, Any] | None:
    meta = _metadata(op)
    if "vector_flow" in meta:
        if not isinstance(meta["vector_flow"], Mapping):
            raise ValueError("vector_flow:mapping_required")
        return meta["vector_flow"]
    family = str(getattr(op, "family", "")).lower().replace("_", "")
    kind = family if family in {"silu", "swiglu"} else str(meta.get("activation", "")).lower().replace("_", "")
    if kind not in {"silu", "swiglu"}:
        return None
    shape = getattr(op, "local_shape", ()) or getattr(op, "launch_shape", ())
    if not shape:
        raise ValueError("vector_flow:shape_required")
    elements = math.prod(shape)
    dtype = _dtype(getattr(op, "dtype", "") or context.activation_dtype or context.compute_dtype)
    if dtype not in _WIDTHS:
        raise ValueError("vector_flow:unsupported_dtype")
    lowp = dtype != "fp32"
    inst = ([{"op": "Cast", "src_dtype": dtype, "dst_dtype": "fp32",
              "count": 2 if kind == "swiglu" else 1}] if lowp else [])
    inst += [{"op": name, "dtype": "fp32"} for name in ("Neg", "Exp", "Add", "Div")]
    if kind == "swiglu":
        inst.append({"op": "Mul", "dtype": "fp32"})
    if lowp:
        inst.append({"op": "Cast", "src_dtype": "fp32", "dst_dtype": dtype})
    return {
        "elements": elements, "instructions": inst,
        "input_bytes_per_element": _WIDTHS[dtype] * (2 if kind == "swiglu" else 1),
        "output_bytes_per_element": _WIDTHS[dtype],
        # Explicit storage hypothesis: input/output and four FP32 temporaries.
        "ub_bytes_per_element": _WIDTHS[dtype] * (3 if kind == "swiglu" else 2) + 16,
        "mapping": f"assumed_{kind}_fp32_intermediates",
        "dependency_assumption": "independent element bundles provide throughput ILP; no measured instruction latency trace",
    }


def _flow_cycles(flow: Mapping[str, Any], elements: int, profile: VectorProfile) -> tuple[float, float, dict[str, Any]]:
    table = {item.name: item for item in profile.instructions}
    rows = flow.get("instructions")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("vector_flow:instructions_required")
    issue: dict[str, float] = {}
    pipes: dict[str, float] = {}
    ends: dict[str, float] = {}
    details = []
    operations = 0.0
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("vector_flow:instruction_mapping_required")
        name = str(raw.get("op", ""))
        if name not in table:
            raise ValueError(f"vector_instruction:unsupported:{name}")
        inst = table[name]
        dtype = _dtype(raw.get("dtype", "fp32"))
        if name == "Cast":
            src, dst = _dtype(raw.get("src_dtype", "")), _dtype(raw.get("dst_dtype", ""))
            if src not in inst.dtypes or dst not in inst.dtypes:
                raise ValueError(f"vector_instruction:unsupported_cast:{src}:{dst}")
            width = max(_WIDTHS[src], _WIDTHS[dst])
        else:
            if dtype not in inst.dtypes or dtype not in _WIDTHS:
                raise ValueError(f"vector_instruction:unsupported_dtype:{name}:{dtype}")
            width = _WIDTHS[dtype]
        count = _integer(raw.get("count", 1), "instruction.count", positive=True)
        lanes = profile.simd_bytes // width
        if lanes < 1:
            raise ValueError("vector_profile:simd_width_too_small")
        bundles = math.ceil(elements / lanes) * count
        dependency = str(raw.get("dependency", "independent"))
        if dependency not in {"independent", "dependent"}:
            raise ValueError("vector_instruction:dependency_invalid")
        issue_cycles = bundles / inst.issue_slots
        pipe_cycles = bundles * inst.throughput_cycles / inst.issue_slots
        if dependency == "dependent":
            # Explicit same-operation recurrence, not a made-up cross-pipe chain.
            pipe_cycles = max(pipe_cycles, bundles * inst.latency_cycles)
        issue[inst.unit] = issue.get(inst.unit, 0.0) + issue_cycles
        pipe_key = f"{inst.unit}/{inst.pipe}"
        pipes[pipe_key] = pipes.get(pipe_key, 0.0) + pipe_cycles
        label = str(raw.get("name", f"i{index}"))
        if label in ends:
            raise ValueError("vector_instruction:duplicate_name")
        deps = raw.get("deps", ())
        if not isinstance(deps, (tuple, list)) or any(dep not in ends for dep in deps):
            raise ValueError("vector_instruction:dependency_must_precede_consumer")
        # A declared full-instruction ordering is deliberately conservative;
        # independent bundles retain their throughput overlap by default.
        ends[label] = max((ends[dep] for dep in deps), default=0.0) + pipe_cycles
        operations += elements * count
        details.append({"name": label, "op": name, "bundles": bundles,
                        "issue_cycles": issue_cycles, "pipe_cycles": pipe_cycles,
                        "dependency_end_cycles": ends[label], "dependency": dependency})
    cycles = max(*issue.values(), *pipes.values(), *ends.values())
    return cycles, operations, {"issue_cycles": issue, "pipe_cycles": pipes,
                                "dependency_cycles": max(ends.values()), "instructions": details}


class VectorInstructionProvider:
    """Produce finite read/compute/write tile DAGs with shared issue limits."""

    name = "vector_instruction"

    def __init__(self, profile: VectorProfile | None = None) -> None:
        self.profile = profile or VectorProfile()

    def estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate:
        try:
            return self._estimate(op, machine, context)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return KernelEstimate(unsupported_reason=str(exc))

    def _estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate:
        profile = self.profile
        if machine.arch_family.upper() not in {item.upper() for item in profile.architectures}:
            return KernelEstimate(unsupported_reason="vector_profile:architecture_mismatch")
        if getattr(op, "flops", 0) > 0 or getattr(op, "link_bytes", 0) > 0:
            return KernelEstimate(unsupported_reason="vector_profile:pure_vector_only")
        flow = _auto_flow(op, context)
        if flow is None:
            return KernelEstimate(unsupported_reason="vector_flow:not_declared")
        elements = _integer(flow.get("elements"), "vector_flow.elements", positive=True)
        stage_index = _integer(getattr(op, "stage", 0), "operator.stage")
        die = machine.dies[stage_index % len(machine.dies)]
        cores = min(profile.vector_cores, die.aiv_count or profile.vector_cores)
        if "active_cores" in flow:
            cores = _integer(flow["active_cores"], "active_cores", positive=True)
            if cores > profile.vector_cores or (die.aiv_count and cores > die.aiv_count):
                raise ValueError("vector_flow:active_cores_exceed_profile_or_machine")
        ub = min(profile.ub_bytes_per_core, die.ub_bytes or profile.ub_bytes_per_core)
        buffers = _integer(flow.get("buffers", profile.buffers), "buffers", positive=True)
        per_element = _number(flow.get("ub_bytes_per_element"), "ub_bytes_per_element", positive=True)
        capacity = int(ub // (per_element * buffers))
        tile = min(_integer(flow.get("tile_elements", profile.tile_elements), "tile_elements", positive=True), capacity)
        if tile < 1:
            raise ValueError("vector_flow:ub_capacity_exceeded")
        read_bpe = _number(flow.get("input_bytes_per_element"), "input_bytes_per_element")
        write_bpe = _number(flow.get("output_bytes_per_element"), "output_bytes_per_element")
        q, r = divmod(elements, cores)
        heaviest = q + bool(r)
        tiles = math.ceil(heaviest / tile)
        if tiles > profile.max_tiles:
            raise ValueError("vector_flow:tile_limit_exceeded")
        stages = []
        breakdown = []
        for index in range(tiles):
            offset = index * tile
            active_elements = r * max(0, min(tile, q + 1 - offset)) + (cores - r) * max(0, min(tile, q - offset))
            local_elements = min(tile, heaviest - offset)
            cycles, _, details = _flow_cycles(flow, local_elements, profile)
            # Count semantic operations across actual cores, not padded SIMD lanes.
            operations = active_elements * sum(_integer(item.get("count", 1), "count", positive=True) for item in flow["instructions"])
            read, compute, write = (f"t{index}.{kind}" for kind in ("read", "vector", "write"))
            read_deps = ([f"t{index - 1}.read"] if index else [])
            if index >= buffers:
                read_deps.append(f"t{index - buffers}.write")
            compute_deps = [read] + ([f"t{index - 1}.vector"] if index else [])
            write_deps = [compute] + ([f"t{index - 1}.write"] if index else [])
            stages.extend((
                ExecutionStage(read, (("hbm", active_elements * read_bpe), ("mte2", active_elements * read_bpe)), tuple(read_deps)),
                ExecutionStage(compute, (("vector", operations),), tuple(compute_deps), cycles / profile.frequency_hz * 1000,
                               ("instruction issue/pipe/dependency service floor; not additive",)),
                ExecutionStage(write, (("hbm", active_elements * write_bpe), ("mte3", active_elements * write_bpe)), tuple(write_deps)),
            ))
            breakdown.append({"tile": index, "elements": active_elements,
                              "heaviest_core_elements": local_elements, "cycles": cycles, **details})
        return KernelEstimate(plan=KernelPlan(
            backend=self.name, version="1", stages=tuple(stages),
            provenance={"profile": profile.name, "source": profile.source,
                        "calibration": "assumed", "frequency_hz": profile.frequency_hz,
                        "vector_cores": cores, "simd_bytes": profile.simd_bytes,
                        "mapping": flow.get("mapping", "explicit_instruction_flow"),
                        "dependency_assumption": flow.get("dependency_assumption", "explicit instruction deps and recurrences only"),
                        "tile_breakdown": breakdown, "timing_scope": context.timing_scope,
                        "limitation": "balanced element partition; no bank conflicts or instruction frontend tracing; issue/SFU floors are per kernel, not shared across concurrent kernels (aggregate vector ops still contend)"},
            buffer_certificate={"ub_bytes_per_core": ub, "ub_bytes_used": tile * per_element * buffers,
                                "tile_elements": tile, "buffers": buffers, "tiles_per_heaviest_core": tiles},
            includes_dispatch=False,
        ))


def _native_hardware_contract(case: Mapping[str, Any], machine: Any, op: Any,
                              hardware: Mapping[str, Any]) -> dict[str, Any]:
    """Compare measurable rate/count fields; an alias alone is not a match."""
    variant_name = case.get("hardware_variant", "ascend950pr_9589")
    variant = hardware["variants"][variant_name]
    stage = _integer(getattr(op, "stage", 0), "operator.stage")
    die = machine.dies[stage % len(machine.dies)]
    cores = int(variant["core_counts"]["vector_core"])
    frequency = float(variant["frequency"]["vector_freq_mhz"]) * 1e6
    hbm = float(variant["peak"]["gm_bandwidth_bytes_per_s"])
    declared_frequency = _number(dict(machine.metadata).get("vector_frequency_hz", 0), "vector_frequency_hz")
    if die.aiv_count != cores:
        raise ValueError("hardware_mismatch:vector_core_count")
    if not math.isclose(declared_frequency, frequency, rel_tol=1e-9, abs_tol=0):
        raise ValueError("hardware_mismatch:explicit_vector_frequency_hz_required")
    if not math.isclose(die.hbm_bw_bytes_per_s, hbm, rel_tol=1e-9, abs_tol=0):
        raise ValueError("hardware_mismatch:hbm_bandwidth")
    gm = (float(variant["derived_from_ini"]["vector_core_ddr_bandwidth_bytes_per_s"])
          if case.get("gm_bandwidth_model") == "vector_core_conservative" else hbm)
    if die.mte_bw_bytes_per_s < gm:
        raise ValueError("hardware_mismatch:mte_bandwidth_below_native_gm_rate")
    if "tiling" in case and die.ub_bytes != int(hardware["common"]["memory_capacity_bytes_per_core"]["ub"]):
        raise ValueError("hardware_mismatch:tiling_ub_capacity")
    return {
        "scope": "declared fields matched to pinned native profile, not device-probed",
        "variant": variant_name, "stage": stage, "vector_cores": cores,
        "vector_frequency_hz": frequency, "hbm_bw_bytes_per_s": hbm,
        "machine_mte_bw_bytes_per_s": die.mte_bw_bytes_per_s,
        "machine_vector_ops_per_s": die.vector_ops_per_s,
        "limitation": "generic vector_ops ceiling is a separate machine declaration; native ISA rates remain external profile inputs",
    }


class CannbotVectorProvider:
    """Run an external fixed-version native CLI without importing its modules.

    Caller explicitly supplies ``metadata.cannbot_case`` and declares matching
    ``machine.metadata.cannbot_hardware_variant`` / ``vector_frequency_hz``.
    Native AIV count and HBM bandwidth must match the target stage. Results stay a single
    opaque kernel stage; no unsupported internal timeline is fabricated.
    """

    name = "cannbot_vector"

    def __init__(self, repo: str | Path, *, revision: str = CANNBOT_REVISION,
                 python_executable: str | Path = sys.executable, timeout_s: float = 30.0) -> None:
        self.repo = Path(repo).resolve()
        self.revision = revision
        self.python_executable = str(python_executable)
        self.timeout_s = _number(timeout_s, "timeout_s", positive=True)
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("cannbot:full_revision_required")
        self._cache: dict[str, KernelEstimate] = {}

    def estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate:
        try:
            return self._estimate(op, machine, context)
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            return KernelEstimate(unsupported_reason=f"cannbot:{exc}")

    def _estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate:
        case = _metadata(op).get("cannbot_case")
        if not isinstance(case, Mapping):
            return KernelEstimate(unsupported_reason="cannbot:case_not_declared")
        if getattr(op, "flops", 0) > 0 or getattr(op, "link_bytes", 0) > 0:
            return KernelEstimate(unsupported_reason="cannbot:pure_vector_only")
        if case.get("gm_bandwidth_model", "chip_peak") not in {"chip_peak", "vector_core_conservative"}:
            return KernelEstimate(unsupported_reason="cannbot:cache_aware_memory_requires_l2_resource_contract")
        variant = case.get("hardware_variant", "ascend950pr_9589")
        declared = dict(machine.metadata).get("cannbot_hardware_variant")
        if declared != variant or machine.arch_family.upper() not in {"A5", "ASCEND950", "DAV_3510"}:
            return KernelEstimate(unsupported_reason="cannbot:hardware_variant_not_explicitly_matched")
        revision = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True, timeout=self.timeout_s).stdout.strip()
        if revision != self.revision:
            return KernelEstimate(unsupported_reason="cannbot:revision_mismatch")
        skill = self.repo / _CANNBOT_SKILL
        paths = [_CANNBOT_SKILL / "scripts/evaluate_a5_vector_roofline.py",
                 _CANNBOT_SKILL / "references/hardware/ascend950pr.json",
                 _CANNBOT_SKILL / "references/vector_instruction_types.json"]
        # Check pinned content including staged edits; Git on Windows may only
        # change LF to CRLF. Record actual file hashes and normalize only CRLF.
        hashes = {}
        for relative in paths:
            pinned = subprocess.run(["git", "-C", str(self.repo), "show", f"{self.revision}:{relative.as_posix()}"],
                                    capture_output=True, check=True, timeout=self.timeout_s).stdout
            current = (self.repo / relative).read_bytes()
            if current.replace(b"\r\n", b"\n") != pinned.replace(b"\r\n", b"\n"):
                return KernelEstimate(unsupported_reason="cannbot:modified_engine_or_calibration")
            hashes[relative.as_posix()] = hashlib.sha256(current).hexdigest()
        hardware = json.loads((skill / "references/hardware/ascend950pr.json").read_text(encoding="utf-8"))
        binding = _native_hardware_contract(case, machine, op, hardware)
        allowed_elements = set()
        for key in ("local_shape", "launch_shape"):
            shape = getattr(op, key, ())
            if shape:
                allowed_elements.add(math.prod(_integer(value, "operator.shape", positive=True) for value in shape))
        if not allowed_elements:
            return KernelEstimate(unsupported_reason="cannbot:operator_shape_required")
        payload = json.dumps(dict(case), ensure_ascii=False, sort_keys=True)
        identity = json.dumps({"case": dict(case), "binding": binding,
                               "allowed_elements": sorted(allowed_elements),
                               "timing_scope": context.timing_scope}, sort_keys=True)
        cache_key = hashlib.sha256(identity.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        with tempfile.TemporaryDirectory(prefix="mindie-sim-cannbot-") as directory:
            source = Path(directory) / "case.jsonl"
            source.write_text(payload + "\n", encoding="utf-8")
            process = subprocess.run([
                self.python_executable, str(skill / "scripts/evaluate_a5_vector_roofline.py"),
                "--hardware-json", str(skill / "references/hardware/ascend950pr.json"),
                "--instruction-json", str(skill / "references/vector_instruction_types.json"),
                "--cases", str(source),
            ], capture_output=True, text=True, check=True, timeout=self.timeout_s)
        rows = json.loads(process.stdout)["results"]
        if len(rows) != 1:
            raise ValueError("exactly_one_result_required")
        row = rows[0]
        instruction_elements = [_number(item["elements"], "instruction.elements", positive=True)
                                for item in row["instruction_breakdown"]]
        if not instruction_elements or max(instruction_elements) not in allowed_elements:
            return KernelEstimate(unsupported_reason="cannbot:case_operator_element_count_mismatch")
        duration = _number(row["theoretical_time_us"], "native_duration") / 1000
        fixed = _number(row.get("fixed_overhead_us", 0), "native_fixed_overhead")
        floor = _number(row.get("launch_floor_us", 0), "native_launch_floor")
        if fixed and context.timing_scope != "device_event_warm":
            return KernelEstimate(unsupported_reason="cannbot:host_wall_overhead_not_supported")
        memory = row["memory"]
        if any(_number(memory.get(key, 0), key) > 0 for key in ("l2_read_bytes", "l2_write_bytes")):
            return KernelEstimate(unsupported_reason="cannbot:explicit_l2_work_requires_l2_resource_contract")
        # Native gm_read already contains repeated reads. gm_reread is a
        # classification of that traffic, never additional bytes to add again.
        read = _number(memory.get("gm_read_bytes", 0), "gm_read_bytes")
        write = _number(memory.get("gm_write_bytes", 0), "gm_write_bytes")
        operations = sum(_number(item["elements"], "elements") * _number(item.get("count", 1), "count")
                         for item in row["instruction_breakdown"])
        plan = KernelPlan(
            backend=self.name, version="1", stages=(ExecutionStage(
                "native.kernel", (("hbm", read + write), ("mte2", read), ("mte3", write), ("vector", operations)),
                minimum_duration_ms=duration,
                notes=("opaque native kernel service floor; no inferred microstage overlap",)),),
            provenance={"revision": self.revision, "hardware_variant": variant,
                        "case_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "source_sha256": hashes, "native_result": row,
                        "hardware_contract": binding,
                        "shape_contract": {"operator_allowed_elements": sorted(allowed_elements),
                                           "native_instruction_max_elements": max(instruction_elements),
                                           "scope": "element cardinality only; caller owns flow semantics"},
                        "calibration": "external_native_profile", "timing_scope": context.timing_scope,
                        "limitation": "whole-kernel stage; aggregate resource contention may increase native standalone duration; no cross-kernel shared instruction issue/SFU ports"},
            includes_dispatch=bool(fixed or floor),
        )
        estimate = KernelEstimate(plan=plan)
        self._cache[cache_key] = estimate
        return estimate
