"""Pure-English outward multi-file operator matrix writer & hard sanitization gate.

[Role] Emit the structured outward-facing multi-file English deliverable package
  (``README.md``, ``hardware_nominal_specs.csv``, ``summary/{Executive_Summary.xlsx,
  model_step_summary.csv}``, ``models/<model_id>/{Operator_Breakdown.xlsx,
  op_breakdown_matrix.csv,scaling_curves.csv}``, ``operators/unique_kernel_catalog.csv``,
  and standalone interactive ``charts/scaling_explorer.html``) plus optional
  ``.tar.gz`` packaging and strictly separated internal ``--emit-bench-plan``
  outputs (never inside the outward bundle).
[Invariants]
  * Verbatim CANN ``msprof`` (``op_summary_*.csv``) English column headers on
    all operator tables; live Excel formulas in ``Op_Breakdown`` referencing
    ``Hardware_Nominal_Specs``; native OOXML line charts (``xl/charts/chart*.xml``)
    in ``Scaling_Curves``.
  * ``assert_zero_internal_jargon`` scans EVERY emitted file (including every XML
    entry inside every ``.xlsx`` ZIP) and fails closed if any internal token,
    private path/IP, or CJK character appears anywhere.
  * Atomic writes via temporary sibling files + ``os.replace``; ``TimeoutError``
    is never masked by ``OSError``.
[Timeouts & Locks] Bounded local filesystem serialization; no locks or device calls.
[Side-Effect Policy] Writes exclusively into caller-specified ``out_dir``,
  optional ``<out_dir>.tar.gz``, and optional separate ``emit_bench_plan_dir``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Any
import zipfile

from mindie_sim.schema.bench_matrix import (
    BYTES_PER_GB,
    OUTWARD_FORBIDDEN_INTERNAL_TOKENS,
    CampaignManifest,
    HardwareNominalSpec,
    HbmAdmissibilityReceipt,
    MatrixSpec,
    OperatorMatrixManifest,
    OperatorMatrixRow,
    StepSummary,
    assert_no_outward_internal_tokens,
    canonical_json,
)
from mindie_sim.schema.common import ModelingError

from .ooxml import (
    ChartSeries,
    LineChartSpec,
    STYLE_ALERT_HIGH,
    STYLE_FLOAT2,
    STYLE_FLOAT3,
    STYLE_HEADER,
    STYLE_INT,
    STYLE_TEXT,
    Workbook,
    Worksheet,
    column_letter,
)
from .operator_matrix import (
    build_bench_plans_by_sku,
    build_operator_matrix,
    render_subgraph_plan_v1_compat,
)

# ---------------------------------------------------------------------------
# Constants, Verbatim msprof Headers & Sanitization Patterns
# ---------------------------------------------------------------------------

OUTWARD_OP_COLUMNS: tuple[str, ...] = (
    "Model",
    "Hardware SKU",
    "Phase",
    "Sequence Length",
    "Schedule Cliff",
    "Batch Size",
    "Parallel Config",
    "Submodule",
    "Op Name",
    "OP Type",
    "Kernel Name",
    "Accelerator Core",
    "Block Num",
    "Mix Block Num",
    "Stream ID",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Analytical Formula",
    "Formula Parameters",
    "FLOPs",
    "HBM Read(Bytes)",
    "HBM Write(Bytes)",
    "HBM Total(Bytes)",
    "L2 Traffic(Bytes)",
    "UB Working Set(Bytes)",
    "Comm Volume(Bytes)",
    "Arithmetic Intensity(FLOPs/Byte)",
    "Invocation Count",
    "Theoretical Roofline(us)",
    "Predicted Task Duration(us)",
    "Predicted Exposed Step Time(us)",
    "Step Time Share(%)",
    "msprof Task Duration P50(us)",
    "msprof Task Duration P99(us)",
    "Prediction APE(%)",
    "MFU(%)",
    "HBM Bandwidth Utilization(%)",
    "HBM Read MBU(%)",
    "Comm Bandwidth Utilization(%)",
    "Bound Type",
)

HW_NOMINAL_COLUMNS: tuple[str, ...] = (
    "Hardware SKU",
    "Architecture",
    "AIC Cores",
    "AIV Cores",
    "HBM Capacity(GiB)",
    "L2 Cache(MiB)",
    "UB per Core(KiB)",
    "Nominal Clock(MHz)",
    "BF16 Cube Peak(TFLOPS)",
    "INT8 Cube Peak(TOPS)",
    "FP8 Cube Peak(TFLOPS)",
    "Vector Peak(TOPS)",
    "HBM Peak(GB/s)",
    "Intra-Node Link Peak(GB/s)",
    "BF16 Cube Peak(FLOPs/s)",
    "FP32 Cube Peak(FLOPs/s)",
    "INT8 Cube Peak(Ops/s)",
    "FP8 Cube Peak(FLOPs/s)",
    "INT4 Cube Peak(Ops/s)",
    "Vector Peak(Ops/s)",
    "HBM Peak(Bytes/s)",
    "Intra-Pkg SIO Peak(Bytes/s)",
    "Intra-Node Link Peak(Bytes/s)",
    "Inter-Node Link Peak(Bytes/s)",
)

STEP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "Model",
    "Hardware SKU",
    "Architecture",
    "Phase",
    "Sequence Length",
    "Batch Size",
    "Parallel Config",
    "Device Count",
    "HBM Admissible",
    "Admissibility Status",
    "Predicted Step Duration(us)",
    "Predicted Step Duration(ms)",
    "msprof Step Duration(us)",
    "Step Delta(us)",
    "Step MFU(%)",
    "Model MBU(%)",
    "Physical HBM Bandwidth Utilization(%)",
    "Step Comm Bandwidth Utilization(%)",
    "msprof Step MFU(%)",
    "msprof Model MBU(%)",
    "msprof Physical HBM Util(%)",
    "Active Weights(Bytes)",
    "KV Cache Read(Bytes)",
    "New KV Write(Bytes)",
    "Total HBM Traffic(Bytes)",
    "Total FLOPs",
    "Runtime Peak HBM(GiB)",
    "Load Peak HBM(GiB)",
    "Usable HBM per Die(GiB)",
    "Max Admissible Batch",
)

SCALING_CURVE_COLUMNS: tuple[str, ...] = (
    "Model",
    "Hardware SKU",
    "Phase",
    "Parallel Config",
    "Operator Scope",
    "Sequence Length",
    "Batch Size",
    "Schedule Cliff",
    "Theoretical Roofline(us)",
    "Predicted Task Duration(us)",
    "msprof Task Duration P50(us)",
    "msprof Task Duration P99(us)",
    "MFU(%)",
    "HBM Bandwidth Utilization(%)",
    "HBM Read MBU(%)",
)

CATALOG_COLUMNS: tuple[str, ...] = (
    "Catalog ID",
    "Hardware SKU",
    "Architecture",
    "OP Type",
    "Kernel Name",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Analytical Formula",
    "Formula Parameters",
    "FLOPs",
    "HBM Read(Bytes)",
    "HBM Write(Bytes)",
    "HBM Total(Bytes)",
    "Theoretical Roofline(us)",
    "Predicted Task Duration(us)",
    "msprof Task Duration P50(us)",
    "MFU(%)",
    "HBM Bandwidth Utilization(%)",
    "HBM Read MBU(%)",
    "Bound Type",
    "Model Count",
    "Total Invocations",
    "Aggregate Exposed Time(us)",
)

_CLIFF_DISPLAY_LABELS: Mapping[str, str] = {
    "anchor": "Anchor",
    "wave_cliff": "Wave Boundary",
    "ub_cliff": "UB Reload Knee",
    "l2_cliff": "L2 Residency Knee",
    "kv_block_cliff": "KV Block Boundary",
    "index_pool_cliff": "Sparse Index Threshold",
}

_GIB_FLOAT = float(1024 ** 3)
_MIB_FLOAT = float(1024 ** 2)
_KIB_FLOAT = 1024.0
_TERA_FLOAT = 1e12
_US_PER_MS = 1000.0
_MAX_SCAN_BYTES = 64 * 1024 * 1024

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PRIVATE_PATH_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_])/(?:Users|home|root|mnt|dev/davinci)[^\s\"',;<>|)]*|"
    r"[A-Za-z]:\\(?:Users|Windows|ProgramData|Program Files)[^\s\"',;<>|)]*)"
)
_IP_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.:])(?:\d{1,3}(?:\.\d{1,3}){3}|"
    r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})(?![A-Za-z0-9_.:])"
)
_FORMULA_LEAD_CHARS = frozenset({"=", "+", "@", "\t", "\r"})
_MODEL_DIR_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_INTERNAL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("mindie_sim", "analytical_model"),
    ("mindie-sim", "analytical-model"),
    ("mindie_profiler", "msprof_bench"),
    ("Stage 0", "Baseline"),
    ("Stage 1", "Config"),
    ("Stage 2", "Framework"),
    ("Stage 3", "Optimized"),
    ("L*", "Roofline"),
    ("l_star", "roofline"),
    ("dual_cert", "bound_check"),
    ("Farkas", "Linear"),
    ("OpOccurrence", "Invocation"),
    ("LayerRunProgram", "LayerSchedule"),
    ("PlannedRecipe", "KernelPlan"),
    ("PrimitiveTable", "ProfileTable"),
    ("arch22", "gen2"),
    ("arch35", "gen5"),
    ("cohort", "batch_group"),
    ("rep_dedup", "layer_repeat"),
    ("unbound_formula", "analytical_formula"),
    ("cell_id", "point_key"),
    ("k2-", "KRN-"),
    ("sig-", "SIG_"),
)


@dataclass(frozen=True, slots=True)
class MatrixExportReceipt:
    """Immutable receipt describing exported outward bundle & optional internal plans."""

    out_dir: str
    tar_gz_path: str | None
    bench_plan_paths: tuple[str, ...]
    compat_v1_paths: tuple[str, ...]
    model_count: int
    sku_count: int
    row_count: int
    step_count: int
    unique_kernel_count: int
    sanitization_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "tar_gz_path": self.tar_gz_path,
            "bench_plan_paths": list(self.bench_plan_paths),
            "compat_v1_paths": list(self.compat_v1_paths),
            "model_count": self.model_count,
            "sku_count": self.sku_count,
            "row_count": self.row_count,
            "step_count": self.step_count,
            "unique_kernel_count": self.unique_kernel_count,
            "sanitization_passed": self.sanitization_passed,
        }


# ---------------------------------------------------------------------------
# Atomic Write & Outward Pure-English Sanitization Gate
# ---------------------------------------------------------------------------

def _is_private_ip_literal(token: str) -> bool:
    if ":" in token and not any(c in token for c in "0123456789abcdefABCDEF"):
        return False
    if token.count(":") == 1:
        return False
    try:
        addr = ipaddress.ip_address(token)
    except ValueError:
        return False
    return bool(addr.is_private or addr.is_loopback or addr.is_link_local)


def _redact_private_ips(text: str) -> str:
    return _IP_TOKEN_RE.sub(lambda m: "<addr>" if _is_private_ip_literal(m.group(0)) else m.group(0), text)


def _find_private_ip(text: str) -> str | None:
    for match in _IP_TOKEN_RE.finditer(text):
        tok = match.group(0)
        if _is_private_ip_literal(tok):
            return tok
    return None


def _sanitize_outward_str(value: Any) -> str:
    text = str(value if value is not None else "")
    for old, new in _INTERNAL_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
    for tok in ("P01_", "P06_", "P18_"):
        if tok in text:
            text = text.replace(tok, "K_")
    if _CJK_RE.search(text):
        text = _CJK_RE.sub("", text).strip()
    text = _PRIVATE_PATH_RE.sub("<path>", text)
    text = _redact_private_ips(text)
    return text


def _neutralize_csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str) or not value:
        return value
    lead = value[0]
    if lead in _FORMULA_LEAD_CHARS:
        return "'" + value
    if lead == "-":
        try:
            float(value)
        except ValueError:
            return "'" + value
    return value


def _safe_model_dirname(model_id: str) -> str:
    cleaned = _MODEL_DIR_SAFE_RE.sub("_", _sanitize_outward_str(model_id)).strip("._-")
    return cleaned or "model"


def _write_atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except TimeoutError:
        raise
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _write_atomic_text(path: Path, text: str) -> None:
    _write_atomic_bytes(path, text.replace("\r\n", "\n").encode("utf-8"))


def _write_csv_atomic(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_neutralize_csv_cell(v) for v in row])
    _write_atomic_text(path, buf.getvalue())


def _scan_text_for_violations(text: str, *, context: str) -> None:
    assert_no_outward_internal_tokens(text, context=context)
    if _CJK_RE.search(text):
        match = _CJK_RE.search(text)
        sample = match.group(0) if match else ""
        raise ModelingError(f"outward_cjk_character_leak:{context}:{sample!r}")
    if _PRIVATE_PATH_RE.search(text):
        raise ModelingError(f"outward_private_path_leak:{context}")
    if _find_private_ip(text) is not None:
        raise ModelingError(f"outward_private_ip_leak:{context}")


def assert_zero_internal_jargon(bundle_dir: str | Path) -> None:
    """Scan EVERY file in ``bundle_dir`` (including XML entries inside ``.xlsx``) and fail closed."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise ModelingError(f"outward_bundle_dir_missing:{root.name}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "bench_plan" in rel.lower() or rel.endswith(".bench_plan_v2.json"):
            raise ModelingError(f"outward_bundle_contains_internal_bench_plan:{rel}")
        if path.suffix.lower() == ".xlsx":
            with zipfile.ZipFile(path, "r") as zf:
                for member in sorted(zf.namelist()):
                    if member.endswith((".xml", ".rels")):
                        raw = zf.read(member)
                        if len(raw) > _MAX_SCAN_BYTES:
                            raise ModelingError(f"outward_xml_too_large:{rel}!{member}")
                        _scan_text_for_violations(
                            raw.decode("utf-8", errors="strict"),
                            context=f"{rel}!{member}",
                        )
        else:
            raw = path.read_bytes()
            if len(raw) > _MAX_SCAN_BYTES:
                raise ModelingError(f"outward_file_too_large:{rel}")
            _scan_text_for_violations(raw.decode("utf-8", errors="strict"), context=rel)


# ---------------------------------------------------------------------------
# Row Projectors (Nominal Specs, Step Summary, msprof Operator Breakdown, Catalog)
# ---------------------------------------------------------------------------

def _nominal_spec_row(spec: HardwareNominalSpec) -> list[Any]:
    bf16_flops = spec.cube_peaks_flops_per_s.get("bf16", spec.cube_peaks_flops_per_s.get("fp16", 0.0))
    fp32_flops = spec.cube_peaks_flops_per_s.get("fp32", spec.cube_peaks_flops_per_s.get("hf32", 0.0))
    int8_ops = spec.cube_peaks_flops_per_s.get("int8", 0.0)
    fp8_flops = spec.cube_peaks_flops_per_s.get("fp8_e4m3", spec.cube_peaks_flops_per_s.get("mxfp8", 0.0))
    int4_ops = spec.cube_peaks_flops_per_s.get("int4", spec.cube_peaks_flops_per_s.get("mxfp4", 0.0))
    sio_b_s = spec.raw_link_peaks_bytes_per_s.get("d2d_intra_pkg", spec.raw_link_peaks_bytes_per_s.get("sio", 0.0))
    link_b_s = spec.raw_link_peaks_bytes_per_s.get(
        "d2d_intra_node", next(iter(spec.raw_link_peaks_bytes_per_s.values()), 0.0),
    )
    inter_link_b_s = spec.raw_link_peaks_bytes_per_s.get(
        "roce_inter_node",
        spec.raw_link_peaks_bytes_per_s.get("d2d_inter_node", link_b_s),
    )
    return [
        _sanitize_outward_str(spec.sku),
        _sanitize_outward_str(spec.soc_arch),
        int(spec.aic_count),
        int(spec.aiv_count),
        round(float(spec.hbm_capacity_bytes) / _GIB_FLOAT, 2),
        round(float(spec.l2_capacity_bytes) / _MIB_FLOAT, 2),
        round(float(spec.ub_capacity_bytes) / _KIB_FLOAT, 2),
        round(float(spec.nominal_clock_mhz), 2),
        round(bf16_flops / _TERA_FLOAT, 3),
        round(int8_ops / _TERA_FLOAT, 3),
        round(fp8_flops / _TERA_FLOAT, 3),
        round(float(spec.vector_peak_ops_per_s) / _TERA_FLOAT, 3),
        round(float(spec.hbm_peak_bytes_per_s) / BYTES_PER_GB, 2),
        round(link_b_s / BYTES_PER_GB, 2),
        float(bf16_flops),
        float(fp32_flops),
        float(int8_ops),
        float(fp8_flops),
        float(int4_ops),
        float(spec.vector_peak_ops_per_s),
        float(spec.hbm_peak_bytes_per_s),
        float(sio_b_s),
        float(link_b_s),
        float(inter_link_b_s),
    ]


def _display_arm_label(parallel_arm: str, *, stage: str = "before", multi_stage: bool = False) -> str:
    base = _sanitize_outward_str(parallel_arm)
    if not multi_stage:
        return base
    tag = "Optimized" if str(stage).strip().lower() == "after" else "Baseline"
    return f"{base} [{tag}]"


def _step_summary_row(
    step: StepSummary,
    receipt: HbmAdmissibilityReceipt | None,
    *,
    stage: str = "before",
    multi_stage: bool = False,
) -> list[Any]:
    rt_gib = round(float(receipt.max_runtime_peak_hbm_bytes) / _GIB_FLOAT, 3) if receipt else ""
    ld_gib = round(float(receipt.max_load_peak_hbm_bytes) / _GIB_FLOAT, 3) if receipt else ""
    usable_gib = round(float(receipt.usable_hbm_bytes_per_die) / _GIB_FLOAT, 3) if receipt else ""
    max_conc = int(receipt.max_admissible_concurrency) if receipt else ""
    status_str = _sanitize_outward_str(receipt.reason if receipt else ("ADMISSIBLE" if step.admissible else "INFEASIBLE"))
    return [
        _sanitize_outward_str(step.model_id),
        _sanitize_outward_str(step.sku),
        _sanitize_outward_str(step.soc_arch),
        "Prefill" if step.phase == "prefill" else "Decode",
        int(step.seq_len),
        int(step.concurrency),
        _display_arm_label(step.parallel_arm, stage=stage, multi_stage=multi_stage),
        int(step.num_dies),
        "Yes" if step.admissible else "No",
        status_str,
        round(float(step.sim_step_us), 3),
        round(float(step.sim_step_us) / _US_PER_MS, 4),
        round(float(step.measured_step_us), 3) if step.measured_step_us is not None else "",
        round(float(step.step_delta_us), 3) if step.step_delta_us is not None else "",
        round(float(step.step_sim_mfu_pct), 2),
        round(float(step.model_mbu_pct), 2),
        round(float(step.hbm_bus_util_pct), 2),
        round(float(step.step_sim_link_bu_pct), 2),
        round(float(step.step_measured_mfu_pct), 2) if step.step_measured_mfu_pct is not None else "",
        round(float(step.measured_model_mbu_pct), 2) if step.measured_model_mbu_pct is not None else "",
        round(float(step.measured_hbm_bus_util_pct), 2) if step.measured_hbm_bus_util_pct is not None else "",
        int(step.active_weight_bytes),
        int(step.kv_read_bytes),
        int(step.new_kv_write_bytes),
        int(step.total_step_hbm_rw_bytes),
        int(step.total_step_cube_flops),
        rt_gib,
        ld_gib,
        usable_gib,
        max_conc,
    ]


def _format_formula_params(terms: Mapping[str, Any]) -> str:
    parts = [f"{_sanitize_outward_str(k)}={_sanitize_outward_str(v)}" for k, v in sorted(terms.items())]
    return ", ".join(parts)


def _outward_op_row_values(row: OperatorMatrixRow, *, multi_stage: bool = False) -> list[Any]:
    cliff_label = _CLIFF_DISPLAY_LABELS.get(row.seq_point_kind, "Anchor")
    submod = _sanitize_outward_str(row.layer_archetype_id or row.model_scope)
    return [
        _sanitize_outward_str(row.model_id),
        _sanitize_outward_str(row.sku),
        "Prefill" if row.phase == "prefill" else "Decode",
        int(row.seq_len),
        cliff_label,
        int(row.concurrency),
        _display_arm_label(row.parallel_arm, stage=row.stage, multi_stage=multi_stage),
        submod,
        _sanitize_outward_str(row.op_name or row.op_id),
        _sanitize_outward_str(row.op_type or row.family),
        _sanitize_outward_str(row.kernel_name or row.upstream_abi),
        _sanitize_outward_str(row.accelerator_core),
        int(row.block_dim),
        int(row.mix_block_dim),
        int(row.stream_id),
        _sanitize_outward_str(row.input_shapes),
        _sanitize_outward_str(row.input_data_types),
        _sanitize_outward_str(row.input_formats),
        _sanitize_outward_str(row.output_shapes),
        _sanitize_outward_str(row.output_data_types),
        _sanitize_outward_str(row.output_formats),
        _sanitize_outward_str(row.formula_symbolic),
        _format_formula_params(row.formula_terms),
        int(row.cube_flops),
        int(row.hbm_read_bytes),
        int(row.hbm_write_bytes),
        int(row.hbm_total_bytes),
        int(row.l2_bytes),
        int(row.ub_bytes),
        int(row.link_bytes),
        round(float(row.arithmetic_intensity), 3),
        int(row.occurrence_count),
        round(float(row.nominal_theo_us), 3),
        round(float(row.per_call_stage0_us), 3),
        round(float(row.model_exposed_stage0_us), 3),
        round(float(row.step_share_pct), 2),
        round(float(row.measured_us), 3) if row.measured_us is not None else "",
        round(float(row.measured_p99_us), 3) if row.measured_p99_us is not None else "",
        round(float(row.ape_pct), 2) if row.ape_pct is not None else "",
        round(float(row.sim_mfu_pct), 2),
        round(float(row.sim_mbu_rw_pct), 2),
        round(float(row.sim_mbu_read_pct), 2),
        round(float(row.sim_link_bu_pct), 2),
        _sanitize_outward_str(row.bound_type),
    ]


def _build_scaling_curve_rows(
    model_rows: Sequence[OperatorMatrixRow],
    model_steps: Sequence[StepSummary],
    *,
    stage_by_cell: Mapping[str, str] | None = None,
    multi_stage: bool = False,
) -> list[list[Any]]:
    out: list[list[Any]] = []
    # 1. Step-level scaling rows (one per admissible step point; graph-derived bound, honest empty P99)
    by_cell_rows: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    for r in model_rows:
        by_cell_rows[r.cell_id].append(r)
    for step in sorted(model_steps, key=lambda s: (s.sku, s.phase, s.parallel_arm, s.concurrency, s.seq_len)):
        if not step.admissible:
            continue
        c_rows = by_cell_rows.get(step.cell_id, [])
        st_lbl = (stage_by_cell or {}).get(step.cell_id, c_rows[0].stage if c_rows else "before")
        step_theo = float(step.l_star_step_us) if float(step.l_star_step_us) > 0.0 else max(
            (r.nominal_theo_us for r in c_rows), default=0.0,
        )
        cliff_lbl = _CLIFF_DISPLAY_LABELS.get(c_rows[0].seq_point_kind, "Anchor") if c_rows else "Anchor"
        out.append([
            _sanitize_outward_str(step.model_id),
            _sanitize_outward_str(step.sku),
            "Prefill" if step.phase == "prefill" else "Decode",
            _display_arm_label(step.parallel_arm, stage=st_lbl, multi_stage=multi_stage),
            "Step Total",
            int(step.seq_len),
            int(step.concurrency),
            cliff_lbl,
            round(step_theo, 3),
            round(float(step.sim_step_us), 3),
            round(float(step.measured_step_us), 3) if step.measured_step_us is not None else "",
            "",
            round(float(step.step_sim_mfu_pct), 2),
            round(float(step.hbm_bus_util_pct), 2),
            round(float(step.model_mbu_pct), 2),
        ])
    # 2. Per-operator scaling rows (sorted deterministically)
    for r in sorted(model_rows, key=lambda x: (x.sku, x.phase, x.parallel_arm, x.stage, x.op_type, x.kernel_name, x.concurrency, x.seq_len)):
        op_scope = _sanitize_outward_str(f"{r.op_type}: {r.kernel_name}")
        out.append([
            _sanitize_outward_str(r.model_id),
            _sanitize_outward_str(r.sku),
            "Prefill" if r.phase == "prefill" else "Decode",
            _display_arm_label(r.parallel_arm, stage=r.stage, multi_stage=multi_stage),
            op_scope,
            int(r.seq_len),
            int(r.concurrency),
            _CLIFF_DISPLAY_LABELS.get(r.seq_point_kind, "Anchor"),
            round(float(r.nominal_theo_us), 3),
            round(float(r.per_call_stage0_us), 3),
            round(float(r.measured_us), 3) if r.measured_us is not None else "",
            round(float(r.measured_p99_us), 3) if r.measured_p99_us is not None else "",
            round(float(r.sim_mfu_pct), 2),
            round(float(r.sim_mbu_rw_pct), 2),
            round(float(r.sim_mbu_read_pct), 2),
        ])
    return out


def _build_unique_kernel_catalog(manifest: OperatorMatrixManifest) -> list[list[Any]]:
    grouped: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    for row in manifest.rows:
        grouped[row.bench_case_id].append(row)
    ranked = sorted(
        grouped.values(),
        key=lambda grp: (-sum(r.model_exposed_stage0_us for r in grp), grp[0].sku, grp[0].kernel_name, grp[0].op_type),
    )
    out: list[list[Any]] = []
    for idx, grp in enumerate(ranked, start=1):
        rep = grp[0]
        models_count = len({r.model_id for r in grp})
        total_inv = sum(r.occurrence_count for r in grp)
        agg_exposed = sum(r.model_exposed_stage0_us for r in grp)
        meas_p50 = next((r.measured_us for r in grp if r.measured_us is not None), None)
        out.append([
            f"KRN-{idx:04d}",
            _sanitize_outward_str(rep.sku),
            _sanitize_outward_str(rep.soc_arch),
            _sanitize_outward_str(rep.op_type or rep.family),
            _sanitize_outward_str(rep.kernel_name or rep.upstream_abi),
            _sanitize_outward_str(rep.accelerator_core),
            _sanitize_outward_str(rep.input_shapes),
            _sanitize_outward_str(rep.input_data_types),
            _sanitize_outward_str(rep.input_formats),
            _sanitize_outward_str(rep.output_shapes),
            _sanitize_outward_str(rep.output_data_types),
            _sanitize_outward_str(rep.output_formats),
            _sanitize_outward_str(rep.formula_symbolic),
            _format_formula_params(rep.formula_terms),
            int(rep.cube_flops),
            int(rep.hbm_read_bytes),
            int(rep.hbm_write_bytes),
            int(rep.hbm_total_bytes),
            round(float(rep.nominal_theo_us), 3),
            round(float(rep.per_call_stage0_us), 3),
            round(float(meas_p50), 3) if meas_p50 is not None else "",
            round(float(rep.sim_mfu_pct), 2),
            round(float(rep.sim_mbu_rw_pct), 2),
            round(float(rep.sim_mbu_read_pct), 2),
            _sanitize_outward_str(rep.bound_type),
            int(models_count),
            int(total_inv),
            round(agg_exposed, 3),
        ])
    return out


# ---------------------------------------------------------------------------
# OOXML Workbook Builders (Live Formulas + Native Line Charts)
# ---------------------------------------------------------------------------

def _populate_table_sheet(
    ws: Worksheet, headers: Sequence[str], rows: Sequence[Sequence[Any]],
) -> None:
    ws.freeze_panes(1, 0)
    ws.set_row_height(1, 24.0)
    for col_idx, heading in enumerate(headers, start=1):
        ws.set_cell(1, col_idx, heading, style=STYLE_HEADER)
        ws.set_col_width(col_idx, max(14.0, min(38.0, float(len(heading)) + 4.0)))
    for row_offset, rvals in enumerate(rows, start=2):
        for col_idx, val in enumerate(rvals, start=1):
            if val == "" or val is None:
                ws.set_cell(row_offset, col_idx, "", style=STYLE_TEXT)
            elif isinstance(val, bool):
                ws.set_cell(row_offset, col_idx, "Yes" if val else "No", style=STYLE_TEXT)
            elif isinstance(val, int):
                ws.set_cell(row_offset, col_idx, val, style=STYLE_INT)
            elif isinstance(val, float):
                ws.set_cell(row_offset, col_idx, val, style=STYLE_FLOAT3)
            else:
                ws.set_cell(row_offset, col_idx, str(val), style=STYLE_TEXT)


def _write_executive_summary_xlsx(
    path: Path,
    nominal_rows: Sequence[Sequence[Any]],
    step_rows: Sequence[Sequence[Any]],
    catalog_rows: Sequence[Sequence[Any]],
) -> None:
    wb = Workbook()
    ws_step = wb.create_sheet("Executive_Summary")
    _populate_table_sheet(ws_step, STEP_SUMMARY_COLUMNS, step_rows)
    adm_col = STEP_SUMMARY_COLUMNS.index("HBM Admissible") + 1
    for r_idx, row in enumerate(step_rows, start=2):
        if row[adm_col - 1] == "No":
            ws_step.set_cell(r_idx, adm_col, "No", style=STYLE_ALERT_HIGH)
    ws_hw = wb.create_sheet("Hardware_Nominal_Specs")
    _populate_table_sheet(ws_hw, HW_NOMINAL_COLUMNS, nominal_rows)
    ws_cat = wb.create_sheet("Unique_Kernel_Catalog")
    _populate_table_sheet(ws_cat, CATALOG_COLUMNS, catalog_rows)
    wb.save(path)


def _hw_col_range(col_name: str, hw_last_row: int) -> str:
    col_l = column_letter(HW_NOMINAL_COLUMNS.index(col_name) + 1)
    return f"Hardware_Nominal_Specs!${col_l}$2:${col_l}${hw_last_row}"


def _write_model_operator_breakdown_xlsx(
    path: Path,
    nominal_rows: Sequence[Sequence[Any]],
    model_step_rows: Sequence[Sequence[Any]],
    model_op_rows: Sequence[OperatorMatrixRow],
    scaling_rows: Sequence[Sequence[Any]],
    *,
    multi_stage: bool = False,
) -> None:
    """[Cohesive-Pipeline] Build per-model 4-sheet OOXML workbook with live formulas & line charts."""
    wb = Workbook()
    # Sheet 1: Hardware_Nominal_Specs (un-derated datasheet denominators)
    ws_hw = wb.create_sheet("Hardware_Nominal_Specs")
    _populate_table_sheet(ws_hw, HW_NOMINAL_COLUMNS, nominal_rows)
    hw_last_row = max(2, len(nominal_rows) + 1)

    # Sheet 2: Step_Summary
    ws_step = wb.create_sheet("Step_Summary")
    _populate_table_sheet(ws_step, STEP_SUMMARY_COLUMNS, model_step_rows)

    # Sheet 3: Op_Breakdown (with LIVE Excel formulas referencing Hardware_Nominal_Specs!)
    ws_op = wb.create_sheet("Op_Breakdown")
    ws_op.freeze_panes(1, 0)
    ws_op.set_row_height(1, 24.0)
    col_idx_map = {name: i + 1 for i, name in enumerate(OUTWARD_OP_COLUMNS)}
    for name, cidx in col_idx_map.items():
        ws_op.set_cell(1, cidx, name, style=STYLE_HEADER)
        ws_op.set_col_width(cidx, max(14.0, min(38.0, float(len(name)) + 4.0)))

    sku_col = column_letter(col_idx_map["Hardware SKU"])
    flops_col = column_letter(col_idx_map["FLOPs"])
    rd_col = column_letter(col_idx_map["HBM Read(Bytes)"])
    wr_col = column_letter(col_idx_map["HBM Write(Bytes)"])
    tot_col = column_letter(col_idx_map["HBM Total(Bytes)"])
    comm_col = column_letter(col_idx_map["Comm Volume(Bytes)"])
    pred_col = column_letter(col_idx_map["Predicted Task Duration(us)"])
    meas_col = column_letter(col_idx_map["msprof Task Duration P50(us)"])

    hw_sku_rng = _hw_col_range("Hardware SKU", hw_last_row)
    hw_bf16_rng = _hw_col_range("BF16 Cube Peak(FLOPs/s)", hw_last_row)
    hw_fp32_rng = _hw_col_range("FP32 Cube Peak(FLOPs/s)", hw_last_row)
    hw_int8_rng = _hw_col_range("INT8 Cube Peak(Ops/s)", hw_last_row)
    hw_fp8_rng = _hw_col_range("FP8 Cube Peak(FLOPs/s)", hw_last_row)
    hw_int4_rng = _hw_col_range("INT4 Cube Peak(Ops/s)", hw_last_row)
    hw_vec_rng = _hw_col_range("Vector Peak(Ops/s)", hw_last_row)
    hw_hbm_rng = _hw_col_range("HBM Peak(Bytes/s)", hw_last_row)
    hw_sio_rng = _hw_col_range("Intra-Pkg SIO Peak(Bytes/s)", hw_last_row)
    hw_intra_rng = _hw_col_range("Intra-Node Link Peak(Bytes/s)", hw_last_row)
    hw_inter_rng = _hw_col_range("Inter-Node Link Peak(Bytes/s)", hw_last_row)

    for r_num, op_row in enumerate(model_op_rows, start=2):
        vals = _outward_op_row_values(op_row, multi_stage=multi_stage)
        sku_match = f"MATCH({sku_col}{r_num},{hw_sku_rng},0)"
        dtype_low = str(op_row.cube_dtype or "bf16").strip().lower()
        if dtype_low in {"fp32", "float32", "hf32"}:
            cube_rng = hw_fp32_rng
        elif dtype_low in {"int4", "fp4", "fp4_e2m1", "mxfp4", "mxfp4_e2m1"}:
            cube_rng = hw_int4_rng
        elif dtype_low in {"int8", "w8a8", "w4a8"}:
            cube_rng = hw_int8_rng
        elif dtype_low.startswith(("fp8", "mxfp8", "w8a8_fp8")):
            cube_rng = hw_fp8_rng
        else:
            cube_rng = hw_bf16_rng
        cube_peak_expr = f"INDEX({cube_rng},{sku_match})"
        vec_peak_expr = f"INDEX({hw_vec_rng},{sku_match})"
        hbm_peak_expr = f"INDEX({hw_hbm_rng},{sku_match})"
        tier_low = str(op_row.link_tier or "").lower()
        if "sio" in tier_low or "intra_pkg" in tier_low:
            link_rng = hw_sio_rng
        elif "inter" in tier_low or "roce" in tier_low:
            link_rng = hw_inter_rng
        else:
            link_rng = hw_intra_rng
        link_peak_expr = f"INDEX({link_rng},{sku_match})"
        t_cube = f"IF(AND({flops_col}{r_num}>0,{cube_peak_expr}>0),{flops_col}{r_num}/{cube_peak_expr},0)"
        t_hbm = f"IF(AND({tot_col}{r_num}>0,{hbm_peak_expr}>0),{tot_col}{r_num}/{hbm_peak_expr},0)"
        t_link = f"IF(AND({comm_col}{r_num}>0,{link_peak_expr}>0),{comm_col}{r_num}/{link_peak_expr},0)"
        t_vec = f",IF({vec_peak_expr}>0,{int(op_row.vector_ops)}/{vec_peak_expr},0)" if int(op_row.vector_ops) > 0 else ""
        formulas: dict[str, tuple[str, Any, int]] = {
            "HBM Total(Bytes)": (f"={rd_col}{r_num}+{wr_col}{r_num}", int(op_row.hbm_total_bytes), STYLE_INT),
            "Arithmetic Intensity(FLOPs/Byte)": (
                f"=IF({tot_col}{r_num}>0,{flops_col}{r_num}/{tot_col}{r_num},0)",
                round(float(op_row.arithmetic_intensity), 3),
                STYLE_FLOAT3,
            ),
            "Theoretical Roofline(us)": (
                f"=MAX({t_cube}{t_vec},{t_hbm},{t_link})*1000000",
                round(float(op_row.nominal_theo_us), 3),
                STYLE_FLOAT3,
            ),
            "Prediction APE(%)": (
                f"=IF(AND(ISNUMBER({meas_col}{r_num}),{meas_col}{r_num}>0),ABS({pred_col}{r_num}-{meas_col}{r_num})/{meas_col}{r_num}*100,\"\")",
                round(float(op_row.ape_pct), 2) if op_row.ape_pct is not None else "",
                STYLE_FLOAT2,
            ),
            "MFU(%)": (
                f"=IF(AND({pred_col}{r_num}>0,{flops_col}{r_num}>0,{cube_peak_expr}>0),{flops_col}{r_num}/(({pred_col}{r_num}/1000000)*{cube_peak_expr})*100,0)",
                round(float(op_row.sim_mfu_pct), 2),
                STYLE_FLOAT2,
            ),
            "HBM Bandwidth Utilization(%)": (
                f"=IF(AND({pred_col}{r_num}>0,{tot_col}{r_num}>0,{hbm_peak_expr}>0),{tot_col}{r_num}/(({pred_col}{r_num}/1000000)*{hbm_peak_expr})*100,0)",
                round(float(op_row.sim_mbu_rw_pct), 2),
                STYLE_FLOAT2,
            ),
            "HBM Read MBU(%)": (
                f"=IF(AND({pred_col}{r_num}>0,{rd_col}{r_num}>0,{hbm_peak_expr}>0),{rd_col}{r_num}/(({pred_col}{r_num}/1000000)*{hbm_peak_expr})*100,0)",
                round(float(op_row.sim_mbu_read_pct), 2),
                STYLE_FLOAT2,
            ),
            "Comm Bandwidth Utilization(%)": (
                f"=IF(AND({pred_col}{r_num}>0,{comm_col}{r_num}>0,{link_peak_expr}>0),{comm_col}{r_num}/(({pred_col}{r_num}/1000000)*{link_peak_expr})*100,0)",
                round(float(op_row.sim_link_bu_pct), 2),
                STYLE_FLOAT2,
            ),
        }
        for col_name, val in zip(OUTWARD_OP_COLUMNS, vals):
            cidx = col_idx_map[col_name]
            if col_name in formulas:
                expr, cached, st = formulas[col_name]
                ws_op.set_cell(r_num, cidx, cached, formula=expr, style=st)
            elif val == "" or val is None:
                ws_op.set_cell(r_num, cidx, "", style=STYLE_TEXT)
            elif isinstance(val, int):
                ws_op.set_cell(r_num, cidx, val, style=STYLE_INT)
            elif isinstance(val, float):
                ws_op.set_cell(r_num, cidx, val, style=STYLE_FLOAT3)
            else:
                ws_op.set_cell(r_num, cidx, str(val), style=STYLE_TEXT)

    # Sheet 4: Scaling_Curves + Native OOXML Line Charts (xl/charts/chart*.xml)
    ws_sc = wb.create_sheet("Scaling_Curves")
    _populate_table_sheet(ws_sc, SCALING_CURVE_COLUMNS, scaling_rows)
    step_curve_count = sum(1 for row in scaling_rows if row[4] == "Step Total")
    chart_end = max(2, (step_curve_count if step_curve_count >= 1 else len(scaling_rows)) + 1)
    if scaling_rows:
        seq_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Sequence Length") + 1)
        theo_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Theoretical Roofline(us)") + 1)
        pred_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Predicted Task Duration(us)") + 1)
        p50_col_l = column_letter(SCALING_CURVE_COLUMNS.index("msprof Task Duration P50(us)") + 1)
        p99_col_l = column_letter(SCALING_CURVE_COLUMNS.index("msprof Task Duration P99(us)") + 1)
        mfu_col_l = column_letter(SCALING_CURVE_COLUMNS.index("MFU(%)") + 1)
        mbu_col_l = column_letter(SCALING_CURVE_COLUMNS.index("HBM Bandwidth Utilization(%)") + 1)
        mbu_rd_col_l = column_letter(SCALING_CURVE_COLUMNS.index("HBM Read MBU(%)") + 1)
        cat_ref = f"Scaling_Curves!${seq_col_l}$2:${seq_col_l}${chart_end}"
        ws_sc.add_line_chart(LineChartSpec(
            title="Latency Scaling: Theoretical Roofline vs Predicted vs msprof",
            x_title="Sequence Length (Tokens)",
            y_title="Duration (us)",
            anchor_from=(16, 1),
            anchor_to=(26, 18),
            series=(
                ChartSeries("Theoretical Roofline(us)", cat_ref, f"Scaling_Curves!${theo_col_l}$2:${theo_col_l}${chart_end}", "1F4E78"),
                ChartSeries("Predicted Task Duration(us)", cat_ref, f"Scaling_Curves!${pred_col_l}$2:${pred_col_l}${chart_end}", "E67E22"),
                ChartSeries("msprof P50(us)", cat_ref, f"Scaling_Curves!${p50_col_l}$2:${p50_col_l}${chart_end}", "2E7D32"),
                ChartSeries("msprof P99(us)", cat_ref, f"Scaling_Curves!${p99_col_l}$2:${p99_col_l}${chart_end}", "C62828"),
            ),
        ))
        ws_sc.add_line_chart(LineChartSpec(
            title="Hardware Efficiency Scaling: MFU(%) & HBM Utilization(%)",
            x_title="Sequence Length (Tokens)",
            y_title="Utilization (%)",
            anchor_from=(16, 20),
            anchor_to=(26, 37),
            series=(
                ChartSeries("MFU(%)", cat_ref, f"Scaling_Curves!${mfu_col_l}$2:${mfu_col_l}${chart_end}", "1F4E78"),
                ChartSeries("HBM Bandwidth Utilization(%)", cat_ref, f"Scaling_Curves!${mbu_col_l}$2:${mbu_col_l}${chart_end}", "2E7D32"),
                ChartSeries("HBM Read MBU(%)", cat_ref, f"Scaling_Curves!${mbu_rd_col_l}$2:${mbu_rd_col_l}${chart_end}", "8E44AD"),
            ),
        ))
    wb.save(path)


# ---------------------------------------------------------------------------
# Pure-English README.md & Interactive Zero-Dependency HTML Viewer
# ---------------------------------------------------------------------------

def _render_readme_md(
    manifest: OperatorMatrixManifest,
    nominal_rows: Sequence[Sequence[Any]],
) -> str:
    models = sorted({_sanitize_outward_str(s.model_id) for s in manifest.step_summaries} | {
        _sanitize_outward_str(r.model_id) for r in manifest.rows
    })
    lines: list[str] = [
        "# Ascend NPU Operator Performance Breakdown & Roofline Package",
        "",
        "## Package Layout",
        "",
        "- `hardware_nominal_specs.csv`: Un-derated public datasheet hardware nominal specifications per Ascend SKU.",
        "- `summary/Executive_Summary.xlsx` & `summary/model_step_summary.csv`: Cross-model step latency, MFU(%), Model MBU(%), Physical HBM Utilization(%), and dual-peak HBM admissibility.",
        "- `models/<model_id>/Operator_Breakdown.xlsx`: Per-model workbook containing `Hardware_Nominal_Specs`, `Step_Summary`, `Op_Breakdown` (live Excel formulas referencing `Hardware_Nominal_Specs`), and `Scaling_Curves` (native OOXML line charts).",
        "- `models/<model_id>/op_breakdown_matrix.csv` & `models/<model_id>/scaling_curves.csv`: Verbatim CANN `msprof`-aligned CSV tables.",
        "- `operators/unique_kernel_catalog.csv`: Cross-model deduplicated unique kernel catalog ranked by aggregate exposed step time.",
        "- `charts/scaling_explorer.html`: Standalone zero-dependency interactive SVG chart viewer.",
        "",
        "## Hardware Nominal Datasheet Specifications (Un-Derated, Utilization = 1.0)",
        "",
        "| Hardware SKU | Arch | AIC | AIV | HBM (GiB) | L2 (MiB) | UB (KiB) | Clock (MHz) | BF16 (TFLOPS) | INT8 (TOPS) | FP8 (TFLOPS) | Vector (TOPS) | HBM Peak (GB/s) | Intra-Node Link (GB/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nr in nominal_rows:
        lines.append(
            f"| {nr[0]} | {nr[1]} | {nr[2]} | {nr[3]} | {nr[4]} | {nr[5]} | {nr[6]} | {nr[7]} | "
            f"{nr[8]} | {nr[9]} | {nr[10]} | {nr[11]} | {nr[12]} | {nr[13]} |"
        )
    lines.extend([
        "",
        "## Roofline, MFU & MBU Definitions",
        "",
        "- **Theoretical Roofline(us)**: `max(FLOPs / P_cube_nominal, VectorOps / P_vec_nominal, (HBM_Read + HBM_Write) / B_hbm_nominal, CommBytes / B_link_raw_nominal) * 1e6` using strictly un-derated nominal datasheet peaks (`utilization = 1.0, launch_us = 0`).",
        "- **MFU(%)**: `FLOPs / (Duration_s * P_cube_nominal) * 100`.",
        "- **HBM Bandwidth Utilization(%)**: `(HBM_Read + HBM_Write) / (Duration_s * B_hbm_nominal) * 100` (physical HBM DQ pin throughput matching `msprof`).",
        "- **HBM Read MBU(%)**: `HBM_Read / (Duration_s * B_hbm_nominal) * 100` (read-only convention).",
        "- **Model MBU(%) (Step Level)**: `(ActiveWeights + KV_Read + New_KV_Write) / (NumDies * StepDuration_s * B_hbm_nominal) * 100` (MosaicML irreducible memory traffic definition; gap vs Physical HBM Utilization quantifies intermediate activation traffic).",
        "- **Comm Bandwidth Utilization(%)**: `CommBytes / (Duration_s * B_link_raw_nominal) * 100`.",
        "",
        "## Verbatim CANN `msprof` Column Glossary",
        "",
        "- `Op Name`, `OP Type`, `Kernel Name`, `Accelerator Core` (`AI_CORE`, `AI_VECTOR_CORE`, `MIX_AIC`, `MIX_AIV`, `HCCL`, `DSA`), `Block Num`, `Mix Block Num`, `Stream ID`, `Input Shapes`, `Input Data Types`, `Input Formats`, `Output Shapes`, `Output Data Types`, `Output Formats`: Verbatim CANN `msprof` (`op_summary_*.csv`) field conventions.",
        "- `Predicted Task Duration(us)` vs `msprof Task Duration P50(us)` / `P99(us)`: Analytical duration compared against silicon `msprof` timing when attached.",
        "",
        f"## Included Models ({len(models)})",
        "",
        *[f"- `{m}`" for m in models],
        "",
    ])
    return "\n".join(lines)


def _render_scaling_explorer_html(all_curve_rows: Sequence[Sequence[Any]]) -> str:
    """[Cohesive-Pipeline] Render standalone zero-dependency pure-English SVG scaling explorer."""
    payload: list[dict[str, Any]] = []
    for r in all_curve_rows:
        payload.append({
            "model": str(r[0]),
            "sku": str(r[1]),
            "phase": str(r[2]),
            "arm": str(r[3]),
            "op": str(r[4]),
            "seq": int(r[5]),
            "batch": int(r[6]),
            "cliff": str(r[7]),
            "theo": float(r[8]),
            "pred": float(r[9]),
            "p50": float(r[10]) if r[10] != "" else None,
            "p99": float(r[11]) if r[11] != "" else None,
            "mfu": float(r[12]),
            "mbu": float(r[13]),
            "mbuRd": float(r[14]),
        })
    data_json = (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ascend NPU Operator Scaling Explorer</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --fg:#f8fafc; --muted:#94a3b8; --accent:#38bdf8; --border:#334155; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); padding:20px; }}
  h1 {{ margin:0 0 12px 0; font-size:20px; font-weight:600; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:12px; background:var(--card); padding:14px; border-radius:8px; border:1px solid var(--border); margin-bottom:16px; }}
  .ctrl {{ display:flex; flex-direction:column; gap:4px; min-width:150px; }}
  .ctrl label {{ font-size:11px; text-transform:uppercase; color:var(--muted); letter-spacing:0.04em; }}
  .ctrl select {{ background:#0f172a; color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 8px; font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
  .panel {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; }}
  .panel h2 {{ margin:0 0 8px 0; font-size:15px; color:#e2e8f0; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); margin-bottom:8px; }}
  .swatch {{ display:inline-block; width:12px; height:3px; vertical-align:middle; margin-right:5px; }}
  svg {{ width:100%; height:310px; display:block; }}
  table {{ width:100%; border-collapse:collapse; margin-top:16px; font-size:12px; background:var(--card); border-radius:8px; overflow:hidden; }}
  th, td {{ padding:6px 10px; border-bottom:1px solid var(--border); text-align:right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; background:#162032; }}
</style>
</head>
<body>
<h1>Ascend NPU Operator &amp; Step Scaling Explorer</h1>
<div class="controls">
  <div class="ctrl"><label for="selModel">Model</label><select id="selModel"></select></div>
  <div class="ctrl"><label for="selSku">Hardware SKU</label><select id="selSku"></select></div>
  <div class="ctrl"><label for="selPhase">Phase</label><select id="selPhase"></select></div>
  <div class="ctrl"><label for="selOp">Operator Scope</label><select id="selOp"></select></div>
  <div class="ctrl"><label for="selAxis">X-Axis</label><select id="selAxis"><option value="seq">Sequence Length</option><option value="batch">Batch Size</option></select></div>
</div>
<div class="grid">
  <div class="panel">
    <h2>Latency Scaling (us): Theoretical Roofline vs Predicted vs msprof</h2>
    <div class="legend">
      <span><i class="swatch" style="background:#38bdf8"></i>Theoretical Roofline(us)</span>
      <span><i class="swatch" style="background:#f59e0b"></i>Predicted Task Duration(us)</span>
      <span><i class="swatch" style="background:#22c55e"></i>msprof P50(us)</span>
      <span><i class="swatch" style="background:#ef4444"></i>msprof P99(us)</span>
    </div>
    <svg id="svgLatency" viewBox="0 0 600 310"></svg>
  </div>
  <div class="panel">
    <h2>Hardware Efficiency Scaling (%): MFU &amp; HBM Bandwidth Utilization</h2>
    <div class="legend">
      <span><i class="swatch" style="background:#38bdf8"></i>MFU(%)</span>
      <span><i class="swatch" style="background:#22c55e"></i>HBM Bandwidth Utilization(%)</span>
      <span><i class="swatch" style="background:#a855f7"></i>HBM Read MBU(%)</span>
    </div>
    <svg id="svgUtil" viewBox="0 0 600 310"></svg>
  </div>
</div>
<table>
  <thead>
    <tr><th>Parallel Config</th><th>Schedule Cliff</th><th>Seq Len</th><th>Batch</th><th>Theoretical(us)</th><th>Predicted(us)</th><th>msprof P50(us)</th><th>msprof P99(us)</th><th>MFU(%)</th><th>HBM Util(%)</th><th>HBM Read(%)</th></tr>
  </thead>
  <tbody id="tblBody"></tbody>
</table>
<script>
const ROWS = {data_json};
const uniq = (arr) => Array.from(new Set(arr));
const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs, text) => {{
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {{}})) el.setAttribute(k, String(v));
  if (text !== undefined) el.textContent = String(text);
  return el;
}};
const fillSel = (el, vals) => {{
  const prev = el.value;
  el.replaceChildren();
  for (const v of vals) {{
    const opt = document.createElement("option");
    opt.value = String(v);
    opt.textContent = String(v);
    el.appendChild(opt);
  }}
  if (vals.includes(prev)) el.value = prev;
}};
const selModel = document.getElementById("selModel");
const selSku = document.getElementById("selSku");
const selPhase = document.getElementById("selPhase");
const selOp = document.getElementById("selOp");
const selAxis = document.getElementById("selAxis");

function drawChart(svg, pts, xKey, seriesDefs) {{
  const W = 600, H = 310, L = 58, R = 18, T = 14, B = 40;
  svg.replaceChildren();
  if (!pts.length) {{
    svg.appendChild(svgEl("text", {{ x: 300, y: 155, fill: "#94a3b8", "text-anchor": "middle", "font-size": 13 }}, "No matching data points"));
    return;
  }}
  const sorted = [...pts].sort((a, b) => a[xKey] - b[xKey]);
  let yMax = 0;
  for (const s of seriesDefs) for (const p of sorted) if (p[s.key] !== null && p[s.key] > yMax) yMax = p[s.key];
  yMax = yMax > 0 ? yMax * 1.12 : 1.0;
  const xPos = (i) => sorted.length === 1 ? (L + W - R) / 2 : L + (i / (sorted.length - 1)) * (W - L - R);
  const yPos = (v) => H - B - (v / yMax) * (H - T - B);
  for (let g = 0; g <= 4; g++) {{
    const val = (yMax * g) / 4, y = yPos(val);
    svg.appendChild(svgEl("line", {{ x1: L, y1: y, x2: W - R, y2: y, stroke: "#334155", "stroke-dasharray": "3,3" }}));
    svg.appendChild(svgEl("text", {{ x: L - 6, y: y + 4, fill: "#94a3b8", "text-anchor": "end", "font-size": 10 }}, val.toFixed(1)));
  }}
  sorted.forEach((p, i) => {{
    svg.appendChild(svgEl("text", {{ x: xPos(i), y: H - B + 16, fill: "#94a3b8", "text-anchor": "middle", "font-size": 10 }}, p[xKey]));
  }});
  for (const s of seriesDefs) {{
    const valid = sorted.map((p, i) => [i, p[s.key]]).filter(([, v]) => v !== null && v !== undefined);
    if (!valid.length) continue;
    const d = valid.map(([i, v], k) => `${{k === 0 ? "M" : "L"}}${{xPos(i).toFixed(1)}},${{yPos(v).toFixed(1)}}`).join(" ");
    svg.appendChild(svgEl("path", {{ d, fill: "none", stroke: s.color, "stroke-width": 2.2 }}));
    for (const [i, v] of valid) {{
      const c = svgEl("circle", {{ cx: xPos(i).toFixed(1), cy: yPos(v).toFixed(1), r: 3.2, fill: s.color }});
      c.appendChild(svgEl("title", {{}}, `${{s.label}}: ${{v.toFixed(2)}}`));
      svg.appendChild(c);
    }}
  }}
}}

function refreshSelectors() {{
  fillSel(selModel, uniq(ROWS.map(r => r.model)));
  const mRows = ROWS.filter(r => r.model === selModel.value);
  fillSel(selSku, uniq(mRows.map(r => r.sku)));
  const sRows = mRows.filter(r => r.sku === selSku.value);
  fillSel(selPhase, uniq(sRows.map(r => r.phase)));
  const pRows = sRows.filter(r => r.phase === selPhase.value);
  fillSel(selOp, uniq(pRows.map(r => r.op)));
  render();
}}

function render() {{
  const pts = ROWS.filter(r => r.model === selModel.value && r.sku === selSku.value && r.phase === selPhase.value && r.op === selOp.value);
  const xKey = selAxis.value;
  drawChart(document.getElementById("svgLatency"), pts, xKey, [
    {{ key: "theo", label: "Theoretical Roofline(us)", color: "#38bdf8" }},
    {{ key: "pred", label: "Predicted Task Duration(us)", color: "#f59e0b" }},
    {{ key: "p50", label: "msprof P50(us)", color: "#22c55e" }},
    {{ key: "p99", label: "msprof P99(us)", color: "#ef4444" }},
  ]);
  drawChart(document.getElementById("svgUtil"), pts, xKey, [
    {{ key: "mfu", label: "MFU(%)", color: "#38bdf8" }},
    {{ key: "mbu", label: "HBM Bandwidth Utilization(%)", color: "#22c55e" }},
    {{ key: "mbuRd", label: "HBM Read MBU(%)", color: "#a855f7" }},
  ]);
  const sorted = [...pts].sort((a, b) => a[xKey] - b[xKey]);
  const tbody = document.getElementById("tblBody");
  tbody.replaceChildren();
  for (const p of sorted) {{
    const tr = document.createElement("tr");
    const cells = [
      p.arm, p.cliff, p.seq, p.batch, p.theo.toFixed(2), p.pred.toFixed(2),
      p.p50 === null ? "-" : p.p50.toFixed(2), p.p99 === null ? "-" : p.p99.toFixed(2),
      p.mfu.toFixed(2), p.mbu.toFixed(2), p.mbuRd.toFixed(2),
    ];
    for (const val of cells) {{
      const td = document.createElement("td");
      td.textContent = String(val);
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
}}

[selModel, selSku, selPhase].forEach(el => el.addEventListener("change", refreshSelectors));
[selOp, selAxis].forEach(el => el.addEventListener("change", render));
refreshSelectors();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main Export Entrypoint (Outward Package + Optional .tar.gz + Separate Plan)
# ---------------------------------------------------------------------------

def _assert_separate_bench_plan_dir(out_dir: Path, bench_plan_dir: Path) -> None:
    resolved_out = out_dir.resolve()
    resolved_plan = bench_plan_dir.resolve()
    if resolved_plan == resolved_out or resolved_out in resolved_plan.parents:
        raise ModelingError(
            "bench_plan_dir_must_be_outside_outward_bundle:pass_separate_directory"
        )


def _package_tar_gz(bundle_dir: Path, files: Sequence[Path]) -> Path:
    tar_path = bundle_dir.with_name(f"{bundle_dir.name}.tar.gz")
    tmp_path = bundle_dir.with_name(f".{bundle_dir.name}.tmp.{os.getpid()}.tar.gz")
    try:
        with tarfile.open(tmp_path, mode="w:gz") as tf:
            for item in sorted({p.resolve() for p in files}, key=lambda p: p.as_posix()):
                if item.is_file():
                    arcname = f"{bundle_dir.name}/{item.relative_to(bundle_dir.resolve()).as_posix()}"
                    tf.add(item, arcname=arcname, recursive=False)
        os.replace(tmp_path, tar_path)
    except TimeoutError:
        raise
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return tar_path


def export_operator_matrix(
    spec_or_manifest: MatrixSpec | OperatorMatrixManifest,
    out_dir: str | Path,
    *,
    campaign: CampaignManifest | Mapping[str, Any] | str | Path | None = None,
    plan_only: bool = False,
    compat_v1: bool = False,
    emit_bench_plan_dir: str | Path | None = None,
    include_fabric: bool = True,
    tar_gz: bool = False,
    layers: int | None = None,
    feature_track: str = "shipping",
) -> MatrixExportReceipt:
    """[Cohesive-Pipeline] Export pure-English multi-file operator matrix package and optional separate bench plans."""
    manifest = (
        spec_or_manifest if isinstance(spec_or_manifest, OperatorMatrixManifest)
        else build_operator_matrix(
            spec_or_manifest, campaign=campaign, layers=layers, feature_track=feature_track,
        )
    )
    out_root = Path(out_dir)
    if emit_bench_plan_dir is not None:
        _assert_separate_bench_plan_dir(out_root, Path(emit_bench_plan_dir))
    bench_paths: list[str] = []
    v1_paths: list[str] = []

    if plan_only:
        plan_dir = Path(emit_bench_plan_dir) if emit_bench_plan_dir is not None else out_root
        plan_dir.mkdir(parents=True, exist_ok=True)
        plans = build_bench_plans_by_sku(manifest, include_fabric=include_fabric)
        for sku, plan_v2 in sorted(plans.items()):
            p_path = plan_dir / f"{sku}.bench_plan_v2.json"
            _write_atomic_text(p_path, json.dumps(plan_v2.to_dict(include_seal=True), indent=2, sort_keys=True) + "\n")
            bench_paths.append(str(p_path))
            if compat_v1:
                v1_path = plan_dir / f"{sku}.subgraph_plan_v1.json"
                v1_doc = render_subgraph_plan_v1_compat(plan_v2)
                _write_atomic_text(v1_path, json.dumps(v1_doc, indent=2, sort_keys=True) + "\n")
                v1_paths.append(str(v1_path))
        return MatrixExportReceipt(
            out_dir=str(plan_dir),
            tar_gz_path=None,
            bench_plan_paths=tuple(bench_paths),
            compat_v1_paths=tuple(v1_paths),
            model_count=len({s.model_id for s in manifest.step_summaries}),
            sku_count=len(manifest.nominal_specs),
            row_count=len(manifest.rows),
            step_count=len(manifest.step_summaries),
            unique_kernel_count=len({r.bench_case_id for r in manifest.rows}),
            sanitization_passed=True,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    emitted_files: list[Path] = []
    multi_stage = len(manifest.spec.stages) > 1 or len({r.stage for r in manifest.rows}) > 1
    stage_by_cell = {r.cell_id: r.stage for r in manifest.rows}
    receipt_by_cell = {r.cell_id: r for r in manifest.hbm_receipts if r.cell_id}
    nominal_rows = [_nominal_spec_row(spec) for _, spec in sorted(manifest.nominal_specs.items())]
    step_rows = [
        _step_summary_row(
            step, receipt_by_cell.get(step.cell_id),
            stage=stage_by_cell.get(step.cell_id, "before"), multi_stage=multi_stage,
        )
        for step in manifest.step_summaries
    ]
    catalog_rows = _build_unique_kernel_catalog(manifest)

    # 1. Root README.md & hardware_nominal_specs.csv
    readme_p = out_root / "README.md"
    hw_csv_p = out_root / "hardware_nominal_specs.csv"
    _write_atomic_text(readme_p, _render_readme_md(manifest, nominal_rows))
    _write_csv_atomic(hw_csv_p, HW_NOMINAL_COLUMNS, nominal_rows)
    emitted_files.extend([readme_p, hw_csv_p])

    # 2. summary/Executive_Summary.xlsx & summary/model_step_summary.csv
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    step_csv_p = summary_dir / "model_step_summary.csv"
    exec_xlsx_p = summary_dir / "Executive_Summary.xlsx"
    _write_csv_atomic(step_csv_p, STEP_SUMMARY_COLUMNS, step_rows)
    _write_executive_summary_xlsx(exec_xlsx_p, nominal_rows, step_rows, catalog_rows)
    emitted_files.extend([step_csv_p, exec_xlsx_p])

    # 3. models/<model_id>/{Operator_Breakdown.xlsx, op_breakdown_matrix.csv, scaling_curves.csv}
    rows_by_model: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    steps_by_model: dict[str, list[StepSummary]] = defaultdict(list)
    for row in manifest.rows:
        rows_by_model[row.model_id].append(row)
    for step in manifest.step_summaries:
        steps_by_model[step.model_id].append(step)
    model_ids = sorted(set(rows_by_model) | set(steps_by_model))

    all_curve_rows: list[list[Any]] = []
    for mid in model_ids:
        mdir = out_root / "models" / _safe_model_dirname(mid)
        mdir.mkdir(parents=True, exist_ok=True)
        m_rows = rows_by_model.get(mid, [])
        m_steps = steps_by_model.get(mid, [])
        m_step_rows = [
            _step_summary_row(
                s, receipt_by_cell.get(s.cell_id),
                stage=stage_by_cell.get(s.cell_id, "before"), multi_stage=multi_stage,
            )
            for s in m_steps
        ]
        m_op_csv_rows = [_outward_op_row_values(r, multi_stage=multi_stage) for r in m_rows]
        m_curve_rows = _build_scaling_curve_rows(
            m_rows, m_steps, stage_by_cell=stage_by_cell, multi_stage=multi_stage,
        )
        all_curve_rows.extend(m_curve_rows)
        op_csv_p = mdir / "op_breakdown_matrix.csv"
        sc_csv_p = mdir / "scaling_curves.csv"
        op_xlsx_p = mdir / "Operator_Breakdown.xlsx"
        _write_csv_atomic(op_csv_p, OUTWARD_OP_COLUMNS, m_op_csv_rows)
        _write_csv_atomic(sc_csv_p, SCALING_CURVE_COLUMNS, m_curve_rows)
        _write_model_operator_breakdown_xlsx(
            op_xlsx_p, nominal_rows, m_step_rows, m_rows, m_curve_rows, multi_stage=multi_stage,
        )
        emitted_files.extend([op_csv_p, sc_csv_p, op_xlsx_p])

    # 4. operators/unique_kernel_catalog.csv
    ops_dir = out_root / "operators"
    ops_dir.mkdir(parents=True, exist_ok=True)
    cat_csv_p = ops_dir / "unique_kernel_catalog.csv"
    _write_csv_atomic(cat_csv_p, CATALOG_COLUMNS, catalog_rows)
    emitted_files.append(cat_csv_p)

    # 5. charts/scaling_explorer.html
    charts_dir = out_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    html_p = charts_dir / "scaling_explorer.html"
    _write_atomic_text(html_p, _render_scaling_explorer_html(all_curve_rows))
    emitted_files.append(html_p)

    # 6. HARD SANITIZATION GATE (scans every file + every XML entry inside every .xlsx!)
    assert_zero_internal_jargon(out_root)

    # 7. Optional .tar.gz archive (packages ONLY artifacts emitted by THIS export!)
    tar_path_str: str | None = None
    if tar_gz:
        tar_path_str = str(_package_tar_gz(out_root, emitted_files))

    # 8. Optional SEPARATE internal bench plans (--emit-bench-plan <dir>)
    if emit_bench_plan_dir is not None:
        plan_dir = Path(emit_bench_plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)
        plans = build_bench_plans_by_sku(manifest, include_fabric=include_fabric)
        for sku, plan_v2 in sorted(plans.items()):
            p_path = plan_dir / f"{sku}.bench_plan_v2.json"
            _write_atomic_text(p_path, json.dumps(plan_v2.to_dict(include_seal=True), indent=2, sort_keys=True) + "\n")
            bench_paths.append(str(p_path))
            if compat_v1:
                v1_path = plan_dir / f"{sku}.subgraph_plan_v1.json"
                v1_doc = render_subgraph_plan_v1_compat(plan_v2)
                _write_atomic_text(v1_path, json.dumps(v1_doc, indent=2, sort_keys=True) + "\n")
                v1_paths.append(str(v1_path))

    return MatrixExportReceipt(
        out_dir=str(out_root),
        tar_gz_path=tar_path_str,
        bench_plan_paths=tuple(bench_paths),
        compat_v1_paths=tuple(v1_paths),
        model_count=len(model_ids),
        sku_count=len(manifest.nominal_specs),
        row_count=len(manifest.rows),
        step_count=len(manifest.step_summaries),
        unique_kernel_count=len(catalog_rows),
        sanitization_passed=True,
    )


__all__ = [
    "CATALOG_COLUMNS",
    "HW_NOMINAL_COLUMNS",
    "OUTWARD_OP_COLUMNS",
    "SCALING_CURVE_COLUMNS",
    "STEP_SUMMARY_COLUMNS",
    "MatrixExportReceipt",
    "assert_zero_internal_jargon",
    "export_operator_matrix",
]
