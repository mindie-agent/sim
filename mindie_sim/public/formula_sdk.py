"""Two-Layer Operator SDK and Simulation-vs-Modeling honesty boundary.

[Role] Expose the canonical two-layer operator workflow and enforce the strict
  separation between Layer-2 Simulation Mode (``allow_unbound_formula=False``)
  and Layer-1 Formula-Driven Hardware Modeling Mode (``allow_unbound_formula=True``):
  - Layer 1 (Modeling Mode ONLY): ``FormulaTensorSpec``, ``FormulaKernelSpec``,
    ``FormulaKernelEvaluation``, ``evaluate_formula_kernel`` /
    ``simulate_formula_kernel``, and ``model_target_ceiling``;
  - Layer 2 (Simulation Mode Binding): ``bind_kernel_implementation`` promoting
    a formula specification or custom operator into a real ``Extension`` /
    ``Provider`` bound to a concrete implementation, recipe, or ``aclnn`` call.
[Invariants]
  * Simulation Mode (``simulate_operator``, ``simulate_native_call``,
    ``simulate_aclnn_call``, ``simulate_graph``, ``simulate_model``,
    ``simulate_service``) hard-locks ``allow_unbound_formula=False`` and fails
    fast with ``CapabilityGap(code="unbound_operator_implementation")`` whenever
    an unbound ``FormulaKernelSpec`` or unbound formula ``OpSpec`` is passed
    without a bound Layer-2 provider.
  * Layer-1 formula projections ALWAYS stamp ``binding_state="unbound_formula_projection"``
    and ``has_unbound_formula_ops=True``, and compute latencies purely from
    hardware ``datapath_for_die`` SRAM/pipe limits (zero heuristic multipliers).
  * ``bind_kernel_implementation`` rejects empty/placeholder bindings so Layer 2
    cannot be bypassed without an executable implementation, recipe, or ABI call.
[Timeouts & Locks] Pure bounded in-memory arithmetic; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Pure functional builders and evaluations; zero global
  registry mutation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.operators import OpSpec

from .contracts import Extension, Provider
from .environment import ExecutionEnvironment
from .errors import CapabilityGap, ExtensionError

BINDING_STATE_BOUND = "bound_implementation"
BINDING_STATE_UNBOUND_FORMULA = "unbound_formula_projection"
MODE_MODELING = "modeling"

_FP32_BYTES = 4
_DEFAULT_UB_BYTES = 256 * 1024
_DEFAULT_L1_BYTES = 512 * 1024
_DEFAULT_L0_BYTES = 64 * 1024
_US_PER_S = 1e6
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DTYPE_BYTES: dict[str, float] = {
    "fp64": 8.0, "float64": 8.0, "int64": 8.0,
    "fp32": 4.0, "float32": 4.0, "int32": 4.0, "uint32": 4.0,
    "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0, "int16": 2.0,
    "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "mxfp8": 1.0, "int8": 1.0, "uint8": 1.0,
    "fp4": 0.5, "mxfp4": 0.5, "int4": 0.5, "w4a8": 0.5,
}
_DOUBLE_BUFFER_FACTOR = 2
_ROW_REDUCTION_SCRATCH_MULTIPLIER = 3
_ELEMENTWISE_SCRATCH_MULTIPLIER = 2


def _dtype_nbytes(dtype: str) -> float:
    return _DTYPE_BYTES.get(str(dtype or "bf16").strip().lower(), 2.0)


def _tensor_elements(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        val = int(dim)
        if val <= 0:
            return 0
        total *= val
    return total


try:
    from mindie_sim.optimization.kernel_synthesis import (  # type: ignore[import-not-found]
        FormulaKernelEvaluation as FormulaKernelEvaluation,
        FormulaKernelSpec as FormulaKernelSpec,
        FormulaTensorSpec as FormulaTensorSpec,
        evaluate_formula_kernel as _canonical_evaluate_formula_kernel,
    )
    _HAS_CANONICAL_SYNTHESIS = True
except ImportError:
    _HAS_CANONICAL_SYNTHESIS = False

    @dataclass(frozen=True, slots=True)
    class FormulaTensorSpec:  # type: ignore[no-redef]
        """Declarative tensor contract for a Layer-1 formula-driven kernel."""

        name: str
        shape: tuple[int, ...]
        dtype: str = "bf16"
        role: str = "input"
        hbm_resident: bool = True

        @property
        def element_count(self) -> int:
            return _tensor_elements(self.shape)

        @property
        def nbytes(self) -> float:
            return float(self.element_count) * _dtype_nbytes(self.dtype)

        def to_dict(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "shape": list(self.shape),
                "dtype": self.dtype,
                "role": self.role,
                "hbm_resident": self.hbm_resident,
            }

    @dataclass(frozen=True, slots=True)
    class FormulaKernelSpec:  # type: ignore[no-redef]
        """Layer-1 mathematical & tensor-contract specification for an unimplemented kernel."""

        name: str
        equations: tuple[str, ...]
        inputs: tuple[FormulaTensorSpec, ...] = ()
        outputs: tuple[FormulaTensorSpec, ...] = ()
        weights: tuple[FormulaTensorSpec, ...] = ()
        matrix_flops: tuple[tuple[str, float], ...] = ()
        vector_ops_fp32: float = 0.0
        link_bytes: float = 0.0
        row_reduction: bool = False
        uses_simt: bool = False
        atomic_ops: tuple[str, ...] = ()
        abi_signature_id: str = ""
        family: str = "formula_kernel"
        activation: str = ""
        activation_params: tuple[tuple[str, float], ...] = ()

        @classmethod
        def from_ops(
            cls,
            ops: Sequence[OpSpec],
            *,
            name: str = "",
            equations: Sequence[str] = (),
            family: str = "formula_kernel",
            uses_simt: bool = False,
        ) -> FormulaKernelSpec:
            op_list = tuple(ops)
            if not op_list:
                raise ExtensionError("formula_kernel_spec_requires_non_empty_ops")
            spec_name = name or f"Fused_{op_list[0].family}_{len(op_list)}ops"
            in_specs: list[FormulaTensorSpec] = []
            out_specs: list[FormulaTensorSpec] = []
            weight_specs: list[FormulaTensorSpec] = []
            mat_flops: dict[str, float] = {}
            vec_ops = 0.0
            link_bytes = 0.0
            for idx, op in enumerate(op_list):
                for t_idx, tensor in enumerate(op.inputs):
                    in_specs.append(FormulaTensorSpec(
                        name=f"{op.op_id}.in{t_idx}",
                        shape=tuple(int(d) for d in getattr(tensor, "shape", ()) or (1,)),
                        dtype=str(getattr(tensor, "dtype", "bf16") or "bf16"),
                        role="input",
                        hbm_resident=(idx == 0),
                    ))
                for t_idx, tensor in enumerate(op.outputs):
                    out_specs.append(FormulaTensorSpec(
                        name=f"{op.op_id}.out{t_idx}",
                        shape=tuple(int(d) for d in getattr(tensor, "shape", ()) or (1,)),
                        dtype=str(getattr(tensor, "dtype", "bf16") or "bf16"),
                        role="output",
                        hbm_resident=(idx == len(op_list) - 1),
                    ))
                meta = dict(op.metadata)
                w_bytes = float(meta.get("weight_bytes", 0.0) or 0.0)
                if w_bytes > 0.0:
                    weight_specs.append(FormulaTensorSpec(
                        name=f"{op.op_id}.weight",
                        shape=(max(1, int(round(w_bytes))),),
                        dtype="uint8",
                        role="weight",
                        hbm_resident=True,
                    ))
                flops = float(meta.get("flops", 0.0) or 0.0)
                if flops > 0.0:
                    mode = str(meta.get("compute_dtype") or meta.get("dtype") or "bf16")
                    mat_flops[mode] = mat_flops.get(mode, 0.0) + flops
                v_ops = float(meta.get("vector_ops", 0.0) or 0.0)
                if v_ops > 0.0:
                    vec_ops += v_ops
                elif flops <= 0.0:
                    vec_ops += float(sum(t.element_count for t in out_specs[-len(op.outputs):])) * 4.0
                link_bytes += float(meta.get("comm_bytes", 0.0) or 0.0)
            eqs = tuple(equations) or tuple(f"{op.op_id}:{op.family}" for op in op_list)
            return cls(
                name=spec_name,
                equations=eqs,
                inputs=tuple(in_specs),
                outputs=tuple(out_specs),
                weights=tuple(weight_specs),
                matrix_flops=tuple(sorted(mat_flops.items())),
                vector_ops_fp32=vec_ops,
                link_bytes=link_bytes,
                uses_simt=uses_simt,
                atomic_ops=tuple(op.op_id for op in op_list),
                family=family,
            )

    @dataclass(frozen=True, slots=True)
    class FormulaKernelEvaluation:  # type: ignore[no-redef]
        """Hardware-grounded Layer-1 projection result (always stamped unbound)."""

        name: str
        abi_signature_id: str
        composed_equations: tuple[str, ...]
        projected_us: float
        l_star_us: float
        tile_spec: Mapping[str, Any]
        ub_peak_bytes: int
        ub_budget_bytes: int
        l1_peak_bytes: int
        l0_peak_bytes: int
        sram_verified: bool
        binding_state: str = BINDING_STATE_UNBOUND_FORMULA
        has_unbound_formula_ops: bool = True
        dominant_pipe: str = ""
        hbm_bytes: float = 0.0
        mte2_bytes: float = 0.0
        mte3_bytes: float = 0.0
        launch_us: float = 0.0

        def to_dict(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "abi_signature_id": self.abi_signature_id,
                "composed_equations": list(self.composed_equations),
                "projected_us": round(self.projected_us, 4),
                "l_star_us": round(self.l_star_us, 4),
                "tile_spec": dict(self.tile_spec),
                "ub_peak_bytes": int(self.ub_peak_bytes),
                "ub_budget_bytes": int(self.ub_budget_bytes),
                "l1_peak_bytes": int(self.l1_peak_bytes),
                "l0_peak_bytes": int(self.l0_peak_bytes),
                "sram_verified": bool(self.sram_verified),
                "binding_state": self.binding_state,
                "has_unbound_formula_ops": bool(self.has_unbound_formula_ops),
                "dominant_pipe": self.dominant_pipe,
                "hbm_bytes": float(self.hbm_bytes),
                "mte2_bytes": float(self.mte2_bytes),
                "mte3_bytes": float(self.mte3_bytes),
                "launch_us": float(self.launch_us),
            }

    def _canonical_evaluate_formula_kernel(  # type: ignore[misc]
        spec: FormulaKernelSpec,
        hardware: Any,
        *,
        primitive_table: Any = None,
        active_aic: int | None = None,
        active_aiv: int | None = None,
    ) -> FormulaKernelEvaluation:
        return _standalone_evaluate_formula_kernel(
            spec, hardware, primitive_table=primitive_table,
            active_aic=active_aic, active_aiv=active_aiv,
        )


def _resolve_device(hardware: Any) -> Device:
    """Resolve Hardware / Device / Case / ExecutionEnvironment / SKU string into one Device."""
    if isinstance(hardware, Device):
        return hardware
    if isinstance(hardware, Hardware):
        return hardware.dies[0]
    if isinstance(hardware, Case):
        return hardware.machine.dies[0]
    if isinstance(hardware, ExecutionEnvironment):
        return hardware.machine.dies[0]
    if isinstance(hardware, str):
        from mindie_sim.compose import resolve_hardware_sku
        from mindie_sim.prediction.structure import parse_hardware
        raw_hw = resolve_hardware_sku(hardware)
        return parse_hardware(raw_hw).dies[0]
    if isinstance(hardware, Mapping):
        from mindie_sim.prediction.structure import parse_case, parse_hardware
        if "model" in hardware and "hardware" in hardware:
            return parse_case(dict(hardware)).machine.dies[0]
        raw_hw = hardware.get("hardware") or hardware.get("machine") or hardware
        return parse_hardware(dict(raw_hw)).dies[0]
    raise ExtensionError("formula_kernel_hardware_invalid:" + type(hardware).__name__)


def _lookup_peak(peaks: Mapping[str, float], aliases: Mapping[str, str], mode: str) -> float:
    low = str(mode or "bf16").strip().lower()
    if low in peaks and peaks[low] > 0.0:
        return float(peaks[low])
    alias = aliases.get(low, "")
    if alias and alias in peaks and peaks[alias] > 0.0:
        return float(peaks[alias])
    for candidate in ("bf16", "fp16", "fp8", "cube"):
        if candidate in peaks and peaks[candidate] > 0.0:
            return float(peaks[candidate])
    return max((float(v) for v in peaks.values() if float(v) > 0.0), default=1e12)


def _standalone_evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Pure physical SRAM/pipe evaluator matching ``datapath_for_die`` capacities."""
    del primitive_table  # Reserved for post-merge primitive micro-kernel calibration.
    if not isinstance(spec, FormulaKernelSpec):
        raise ExtensionError("formula_kernel_requires_FormulaKernelSpec")
    die = _resolve_device(hardware)
    dp = datapath_for_die(die)
    buffers = {str(k).lower(): int(v) for k, v in dp.buffers}
    rates = {str(k).lower(): float(v) for k, v in die.rates}
    peaks = {str(k).lower(): float(v) for k, v in dp.compute_peaks}
    aliases = {str(k).lower(): str(v).lower() for k, v in dp.compute_peak_aliases}

    total_aic = max(1, int(dp.aic_count))
    total_aiv = max(1, int(dp.aiv_count))
    eff_aic = max(1, min(total_aic, int(active_aic))) if active_aic is not None else total_aic
    eff_aiv = max(1, min(total_aiv, int(active_aiv))) if active_aiv is not None else total_aiv
    aic_scale = float(eff_aic) / float(total_aic)
    aiv_scale = float(eff_aiv) / float(total_aiv)

    ub_raw = buffers.get("ub", _DEFAULT_UB_BYTES)
    ub_budget = max(0, ub_raw - (int(dp.simt_ub_carveout_bytes) if spec.uses_simt else 0))
    l1_budget = buffers.get("l1", _DEFAULT_L1_BYTES)
    l0_budget = buffers.get("l0a", _DEFAULT_L0_BYTES)

    in_hbm = sum(t.nbytes for t in spec.inputs if t.hbm_resident)
    w_hbm = sum(t.nbytes for t in spec.weights if t.hbm_resident)
    out_hbm = sum(t.nbytes for t in spec.outputs if t.hbm_resident)
    read_bytes = in_hbm + w_hbm
    write_bytes = out_hbm
    total_hbm = read_bytes + write_bytes

    all_tensors = (*spec.inputs, *spec.outputs)
    row_width = max((int(t.shape[-1]) for t in all_tensors if t.shape), default=1)
    total_rows = max(
        (
            _tensor_elements(t.shape[:-1]) if len(t.shape) > 1 else 1
            for t in all_tensors if t.shape
        ),
        default=1,
    )
    scratch_mult = _ROW_REDUCTION_SCRATCH_MULTIPLIER if spec.row_reduction else _ELEMENTWISE_SCRATCH_MULTIPLIER
    per_row_ub_bytes = row_width * _FP32_BYTES * scratch_mult * _DOUBLE_BUFFER_FACTOR
    max_rows_in_ub = max(1, ub_budget // max(1, per_row_ub_bytes)) if ub_budget > 0 else 1
    tile_rows = max(1, min(total_rows, max_rows_in_ub))
    ub_peak = tile_rows * per_row_ub_bytes

    has_matrix = bool(spec.matrix_flops)
    l1_peak = min(l1_budget, max(0, int(round(min(w_hbm + in_hbm, float(l1_budget)))))) if has_matrix else 0
    l0_peak = min(l0_budget, max(0, l1_peak // 4)) if has_matrix else 0
    sram_ok = (ub_peak <= ub_budget) and (l1_peak <= l1_budget) and (l0_peak <= l0_budget)

    hbm_read_rate = max(1.0, float(dp.hbm_read_bytes_per_s or rates.get("hbm", 1e12)))
    hbm_write_rate = max(1.0, float(dp.hbm_write_bytes_per_s or rates.get("hbm", 1e12)))
    hbm_alpha = max(0.0, float(dp.hbm_rw_alpha))
    hbm_us = (
        (read_bytes / hbm_read_rate + write_bytes / hbm_write_rate)
        * (1.0 + hbm_alpha * (1.0 if (read_bytes > 0.0 and write_bytes > 0.0) else 0.0))
        * _US_PER_S
    )
    mte2_rate = max(1.0, float(rates.get("mte2") or rates.get("mte2_aiv") or hbm_read_rate))
    mte3_rate = max(1.0, float(rates.get("mte3") or hbm_write_rate))
    mte2_us = (read_bytes / (mte2_rate * aiv_scale)) * _US_PER_S if read_bytes > 0.0 else 0.0
    mte3_us = (write_bytes / (mte3_rate * aiv_scale)) * _US_PER_S if write_bytes > 0.0 else 0.0

    cube_us = 0.0
    for mode, flops in spec.matrix_flops:
        if float(flops) > 0.0:
            peak = max(1.0, _lookup_peak(peaks, aliases, mode) * aic_scale)
            cube_us += (float(flops) / peak) * _US_PER_S

    vec_rate = max(1.0, float(rates.get("vector") or peaks.get("fp32") or 1e12) * aiv_scale)
    vec_us = (float(spec.vector_ops_fp32) / vec_rate) * _US_PER_S if spec.vector_ops_fp32 > 0.0 else 0.0

    link_rate = max(1.0, float(rates.get("link") or 1e11))
    link_us = (float(spec.link_bytes) / link_rate) * _US_PER_S if spec.link_bytes > 0.0 else 0.0

    pipe_times = {
        "hbm": hbm_us, "mte2": mte2_us, "mte3": mte3_us,
        "cube": cube_us, "vector": vec_us, "link": link_us,
    }
    dominant_pipe, l_star_us = max(pipe_times.items(), key=lambda kv: kv[1])
    launch_us = max(0.0, float(die.launch_us or 0.0))
    projected_us = l_star_us + launch_us
    abi_id = spec.abi_signature_id or f"formula::{_SLUG_RE.sub('_', spec.name.lower()).strip('_') or 'kernel'}"
    return FormulaKernelEvaluation(
        name=spec.name,
        abi_signature_id=abi_id,
        composed_equations=tuple(spec.equations),
        projected_us=projected_us,
        l_star_us=l_star_us,
        tile_spec={
            "tile_rows": tile_rows,
            "row_width": row_width,
            "double_buffered": True,
            "active_aic": eff_aic,
            "active_aiv": eff_aiv,
            "ub_row_bytes_fp32": per_row_ub_bytes,
            "pipe_us": {k: round(v, 4) for k, v in pipe_times.items()},
        },
        ub_peak_bytes=int(ub_peak),
        ub_budget_bytes=int(ub_budget),
        l1_peak_bytes=int(l1_peak),
        l0_peak_bytes=int(l0_peak),
        sram_verified=bool(sram_ok),
        binding_state=BINDING_STATE_UNBOUND_FORMULA,
        has_unbound_formula_ops=True,
        dominant_pipe=dominant_pipe,
        hbm_bytes=float(total_hbm),
        mte2_bytes=float(read_bytes),
        mte3_bytes=float(write_bytes),
        launch_us=float(launch_us),
    )


def evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Evaluate a Layer-1 ``FormulaKernelSpec`` directly against hardware SRAM & pipes."""
    return _canonical_evaluate_formula_kernel(
        spec, hardware, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def simulate_formula_kernel(
    spec_or_hardware: Any,
    hardware_or_spec: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Layer-1 Modeling Mode entrypoint accepting ``(spec, hardware)`` or ``(hardware, spec)``."""
    if isinstance(spec_or_hardware, FormulaKernelSpec):
        spec, hardware = spec_or_hardware, hardware_or_spec
    elif isinstance(hardware_or_spec, FormulaKernelSpec):
        hardware, spec = spec_or_hardware, hardware_or_spec
    else:
        raise ExtensionError("simulate_formula_kernel_requires_FormulaKernelSpec")
    return evaluate_formula_kernel(
        spec, hardware, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def _iter_ops_items(ops: Any) -> tuple[Any, ...]:
    if ops is None:
        return ()
    if isinstance(ops, (FormulaKernelSpec, OpSpec, Mapping)):
        return (ops,)
    if isinstance(ops, Sequence) and not isinstance(ops, (str, bytes, bytearray)):
        return tuple(ops)
    return (ops,)


def _is_unbound_op_item(item: Any) -> str | None:
    """Return offending kernel/op name if ``item`` represents an unbound Layer-1 formula op."""
    if isinstance(item, FormulaKernelSpec):
        return item.name or "formula_kernel"
    if isinstance(item, OpSpec):
        meta = dict(item.metadata)
        if (
            str(meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(meta.get("unbound_formula_op"))
        ):
            return item.op_id or item.family
        return None
    if isinstance(item, Mapping):
        meta = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        if (
            str(item.get("binding_state") or meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(item.get("unbound_formula_op") or meta.get("unbound_formula_op"))
        ):
            return str(item.get("op_id") or item.get("name") or "unbound_op")
    return None


def enforce_simulation_binding_guard(
    ops: Any,
    *,
    capability: str,
    entry: str,
    allow_unbound_formula: bool = False,
    has_bound_provider: bool = False,
) -> None:
    """Hard-lock Simulation Mode (``allow_unbound_formula=False``) against unbound Layer-1 ops."""
    if allow_unbound_formula:
        raise ExtensionError(f"simulation_mode_forbids_unbound_formula_ops:{entry}")
    if has_bound_provider:
        return
    for item in _iter_ops_items(ops):
        offender = _is_unbound_op_item(item)
        if offender is not None:
            raise CapabilityGap(
                capability=capability,
                entry=entry,
                kind="unmodeled_capability",
                code="unbound_operator_implementation",
                message=(
                    f"unbound_operator_implementation:{offender}: Simulation Mode "
                    "(allow_unbound_formula=False) requires a Layer-2 bound implementation "
                    "registered via bind_kernel_implementation(); for Layer-1 formula-driven "
                    "ceiling projection use simulate_formula_kernel() or model_target_ceiling()."
                ),
                affected_results=("operator_sum_ms", "duration_ms", "timeline"),
                minimal_replay_input={"entry": entry, "unbound_op": offender},
                fallback="not_attempted",
            )


def _strip_unbound_metadata(op: OpSpec, *, kernel_name: str) -> OpSpec:
    meta = dict(op.metadata)
    meta.pop("unbound_formula_op", None)
    meta["binding_state"] = BINDING_STATE_BOUND
    meta.setdefault("bound_kernel_name", kernel_name)
    return replace(op, metadata=meta)


def bind_kernel_implementation(
    spec_or_name: FormulaKernelSpec | str,
    *,
    implementation: Callable[..., Any] | Mapping[str, Any] | None = None,
    extension_id: str | None = None,
    version: str = "1.0.0",
    provider_name: str | None = None,
    capability: str = "operator",
    recipe: Any = None,
    aclnn_api: str | None = None,
    aclnn_symbol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    default: bool = True,
) -> Extension:
    """Layer-2 SDK: bind a formula kernel or custom op to a real ``Extension`` / ``Provider``.

    The returned immutable ``Extension`` can be loaded via ``load_extensions(extensions=[ext])``
    and passed to ``simulate_operator`` / ``simulate_native_call`` / ``simulate_graph`` in
    strict Simulation Mode (``allow_unbound_formula=False``).
    """
    kernel_name = spec_or_name.name if isinstance(spec_or_name, FormulaKernelSpec) else str(spec_or_name or "").strip()
    if not kernel_name:
        raise ExtensionError("bind_kernel_implementation_requires_kernel_name")
    if implementation is None and recipe is None and not (aclnn_api or aclnn_symbol):
        raise ExtensionError(f"bind_kernel_implementation_requires_real_binding:{kernel_name}")

    slug = _SLUG_RE.sub("_", kernel_name.lower()).strip("_") or "custom_kernel"
    ext_id = extension_id or f"bound_{slug}"
    prov_name = provider_name or f"bound_{slug}"
    binding_meta: dict[str, Any] = {
        "binding_state": BINDING_STATE_BOUND,
        "bound_kernel_name": kernel_name,
        "layer": "layer2_bound_implementation",
    }
    if isinstance(spec_or_name, FormulaKernelSpec):
        binding_meta["equations"] = list(spec_or_name.equations)
        binding_meta["abi_signature_id"] = spec_or_name.abi_signature_id or f"bound::{slug}"
    if aclnn_api or aclnn_symbol:
        binding_meta["aclnn_api"] = str(aclnn_api or aclnn_symbol)
    if metadata:
        binding_meta.update(dict(metadata))

    if callable(implementation):
        purpose = "report" if capability in {"operator", "native", "graph"} and recipe is None and not (aclnn_api or aclnn_symbol) else "builder"

        def _callable_factory(*, case: Any, ops: Any, **kwargs: Any) -> Any:
            return implementation(case=case, ops=ops, **kwargs)

        provider = Provider(
            capability=capability,
            name=prov_name,
            factory=_callable_factory,
            purpose=purpose,
            version=version,
            default=default,
            description=f"Layer-2 bound implementation for {kernel_name}",
            metadata=binding_meta,
        )
        return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)

    def _builder_factory(*, case: Any, ops: Any, **kwargs: Any) -> Mapping[str, Any]:
        del case, kwargs
        if isinstance(implementation, Mapping):
            out = dict(implementation)
            if recipe is not None and "recipes" not in out:
                out["recipes"] = recipe
            raw_ops = out.get("ops", out.get("operators", ops))
            out["ops"] = tuple(
                _strip_unbound_metadata(item, kernel_name=kernel_name) if isinstance(item, OpSpec) else item
                for item in _iter_ops_items(raw_ops)
            )
            return out
        if aclnn_api or aclnn_symbol:
            from mindie_sim.prediction.call_builders import build_call
            built = build_call({"builder": "aclnn", "arguments": {"api": str(aclnn_api or aclnn_symbol)}})
            bound_ops = tuple(_strip_unbound_metadata(op, kernel_name=kernel_name) for op in built.ops)
            return {"ops": bound_ops, "recipes": recipe or built.recipes}
        converted: list[OpSpec] = []
        for item in _iter_ops_items(ops):
            if isinstance(item, OpSpec):
                converted.append(_strip_unbound_metadata(item, kernel_name=kernel_name))
            else:
                raise ExtensionError(f"bind_kernel_implementation_requires_op_spec_for_recipe:{kernel_name}")
        return {"ops": tuple(converted), "recipes": recipe}

    provider = Provider(
        capability=capability,
        name=prov_name,
        factory=_builder_factory,
        purpose="builder",
        version=version,
        default=default,
        description=f"Layer-2 recipe/ABI binding for {kernel_name}",
        metadata=binding_meta,
    )
    return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)


def model_target_ceiling(
    model: Any,
    hardware: Any = "a5",
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """First-class Modeling Mode entrypoint (``allow_unbound_formula=True``).

    Independently solves ``Before (Parallel0, B0)`` (current-stack simulation)
    vs ``After (Parallel1*, B1*)`` (target ceiling modeling with unimplemented
    algebraic kernel synthesis) under the declared SLA and HBM budget.
    """
    from mindie_sim.solve import solve_deployment

    call_kwargs = dict(kwargs)
    call_kwargs["mode"] = MODE_MODELING
    result = solve_deployment(model, hardware, **call_kwargs)
    result["analysis_mode"] = MODE_MODELING
    result["allow_unbound_formula"] = True
    return result


__all__ = [
    "FormulaKernelEvaluation",
    "FormulaKernelSpec",
    "FormulaTensorSpec",
    "bind_kernel_implementation",
    "enforce_simulation_binding_guard",
    "evaluate_formula_kernel",
    "model_target_ceiling",
    "simulate_formula_kernel",
]
