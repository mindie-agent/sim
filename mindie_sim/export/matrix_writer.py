"""Pure-English outward multi-file operator matrix writer & hard sanitization gate.

[Role] Emit the structured outward-facing multi-file English deliverable package
  (``README.md``, ``manifest.json``, ``hardware_nominal_specs.csv``,
  ``summary/{Executive_Summary.xlsx,model_step_summary.csv}``,
  ``models/<model_id>/{Operator_Breakdown.xlsx,op_breakdown_matrix.csv,
  scaling_curves.csv}``, ``operators/{unique_kernel_catalog.csv,
  Operator_Scaling_Curves.xlsx,operator_scaling_curves.csv}``, and standalone
  interactive ``charts/scaling_explorer.html``) plus optional ``.tar.gz``
  packaging and strictly separated internal ``--emit-bench-plan`` outputs
  (never inside the outward bundle).
[Invariants]
  * Verbatim CANN ``msprof`` (``op_summary_*.csv``) English column headers on
    all operator tables; live Excel formulas in ``Op_Breakdown`` referencing
    ``Hardware_Nominal_Specs``; native OOXML line charts (``xl/charts/chart*.xml``)
    in ``Scaling_Curves``, ``Operator_Scaling_Charts``, and ``Operator_Charts``.
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
    FabricOperatorCurvePoint,
    build_bench_plans_by_sku,
    build_fabric_operator_curve_rows,
    build_operator_matrix,
    enrich_manifest_with_campaign,
    load_campaign_manifest,
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
    "Nominal Cube Peak(TFLOPS)",
    "Nominal Vector Peak(TFLOPS)",
    "Nominal Total Peak Cube+Vector(TFLOPS)",
    "INT8 Cube Peak(TOPS)",
    "FP8 Cube Peak(TFLOPS)",
    "Vector FP32 Lane Peak(TOPS)",
    "HBM Peak(GB/s)",
    "Intra-Node Link Peak(GB/s)",
    "BF16 Cube Peak(FLOPs/s)",
    "HF32 Cube Peak(FLOPs/s)",
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

OPERATOR_SCALING_CURVE_COLUMNS: tuple[str, ...] = (
    "OP Type",
    "Kernel Name",
    "Accelerator Core",
    "Ascend SKU",
    "Model",
    "Phase",
    "Parallel Config",
    "Fixed Slice",
    "X Axis",
    "X Value",
    "Input Shapes",
    "Output Shapes",
    "Theoretical Roofline(us)",
    "Predicted Task Duration(us)",
    "msprof P50(us)",
    "msprof P99(us)",
    "Roofline Bound",
    "HBM Read(MB)",
    "HBM Write(MB)",
    "Cube Utilization(%)",
    "Vector Utilization(%)",
    "Achieved Bandwidth(GB/s)",
)

OPERATOR_CATALOG_INDEX_COLUMNS: tuple[str, ...] = (
    "OP Type",
    "Kernel Name",
    "Accelerator Core",
    "Scope",
    "SKU Count",
    "Model Count",
    "Curve Series Count",
    "Data Points",
    "X Axis",
    "Min X",
    "Max X",
    "Min Predicted(us)",
    "Max Predicted(us)",
    "Mean Roofline Efficiency(%)",
    "Measured Points",
    "Mean Prediction APE(%)",
    "Primary Roofline Bound",
)

OPERATOR_CHART_INDEX_COLUMNS: tuple[str, ...] = (
    "Chart #",
    "OP Type",
    "Kernel Name",
    "Ascend SKU",
    "Model",
    "Phase",
    "Parallel Config",
    "Fixed Slice",
    "X Axis",
    "Points",
    "Data Range",
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
_BYTES_PER_MB = 1_000_000.0
_US_PER_SECOND = 1_000_000.0
_PCT_SCALE = 100.0
_TERA_FLOAT = 1e12
_US_PER_MS = 1000.0
_MAX_SCAN_BYTES = 64 * 1024 * 1024
# Datasheet FP16/BF16 Vector FMA peak = 8x unified FP32 single-instruction lane rate
# (2x full-register width * 2x FP16-vs-FP32 lane packing * 2 FLOPs/FMA).
_VECTOR_FP16_FMA_PER_FP32_LANE = 8.0
_PEAK_TFLOPS_DECIMALS = 4
_HEADER_ROW_HEIGHT_PT = 24.0
_MIN_COL_WIDTH_CHARS = 14.0
_MAX_COL_WIDTH_CHARS = 38.0
_COL_WIDTH_PAD_CHARS = 4.0
_STEP_CHART_ANCHOR_COL_START = 16
_STEP_CHART_ANCHOR_COL_END = 26
_STEP_CHART_LAT_ROW_START = 1
_STEP_CHART_LAT_ROW_END = 18
_STEP_CHART_UTIL_ROW_START = 20
_STEP_CHART_UTIL_ROW_END = 37
_OP_CHART_GRID_COLS = 2
_OP_CHART_COL_SPAN = 10
_OP_CHART_COL_GAP = 1
_OP_CHART_ROW_SPAN = 17
_OP_CHART_ROW_STRIDE = 19
_OP_CHART_DATA_TABLE_PAD_COLS = 1
_MAX_OPERATOR_CHARTS_PER_SHEET = 80
_OUTWARD_BUNDLE_SCHEMA = "ascend-operator-matrix-bundle/1"
_FABRIC_MODEL_LABEL = "Fabric Interconnect"

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
    unique_operator_count: int = 0
    fabric_curve_point_count: int = 0
    operator_curves_xlsx_path: str | None = None
    operator_curves_csv_path: str | None = None
    manifest_json_path: str | None = None

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
            "unique_operator_count": self.unique_operator_count,
            "fabric_curve_point_count": self.fabric_curve_point_count,
            "operator_curves_xlsx_path": self.operator_curves_xlsx_path,
            "operator_curves_csv_path": self.operator_curves_csv_path,
            "manifest_json_path": self.manifest_json_path,
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
    hf32_flops = spec.cube_peaks_flops_per_s.get("hf32", fp32_flops)
    int8_ops = spec.cube_peaks_flops_per_s.get("int8", 0.0)
    fp8_flops = spec.cube_peaks_flops_per_s.get(
        "fp8_e4m3", spec.cube_peaks_flops_per_s.get("mxfp8", spec.cube_peaks_flops_per_s.get("hifloat8", 0.0)),
    )
    int4_ops = spec.cube_peaks_flops_per_s.get(
        "int4", spec.cube_peaks_flops_per_s.get("mxfp4", spec.cube_peaks_flops_per_s.get("fp4_e2m1", 0.0)),
    )
    vec_fma_flops = float(spec.vector_peak_ops_per_s) * _VECTOR_FP16_FMA_PER_FP32_LANE
    total_cube_vec_flops = float(bf16_flops) + vec_fma_flops
    sio_b_s = spec.raw_link_peaks_bytes_per_s.get(
        "d2d_intra_package",
        spec.raw_link_peaks_bytes_per_s.get("d2d_intra_pkg", spec.raw_link_peaks_bytes_per_s.get("sio", 0.0)),
    )
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
        round(float(bf16_flops) / _TERA_FLOAT, _PEAK_TFLOPS_DECIMALS),
        round(vec_fma_flops / _TERA_FLOAT, _PEAK_TFLOPS_DECIMALS),
        round(total_cube_vec_flops / _TERA_FLOAT, _PEAK_TFLOPS_DECIMALS),
        round(int8_ops / _TERA_FLOAT, 3),
        round(fp8_flops / _TERA_FLOAT, 3),
        round(float(spec.vector_peak_ops_per_s) / _TERA_FLOAT, 3),
        round(float(spec.hbm_peak_bytes_per_s) / BYTES_PER_GB, 2),
        round(link_b_s / BYTES_PER_GB, 2),
        float(bf16_flops),
        float(hf32_flops),
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


def _dominant_row_rank(row: OperatorMatrixRow) -> tuple[int, float, int, float, str]:
    return (
        1 if row.measured_us is not None else 0,
        float(row.model_exposed_stage0_us),
        int(row.occurrence_count),
        float(row.per_call_stage0_us),
        str(row.op_id),
    )


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
    # 2. Per-operator scaling rows (deduplicated per cell & op_scope to eliminate duplicate-X sawtooth)
    best_by_point: dict[tuple[str, str, str, str, str, int, int], OperatorMatrixRow] = {}
    for r in model_rows:
        op_scope = _sanitize_outward_str(f"{r.op_type or r.family}: {r.kernel_name or r.upstream_abi}")
        arm_disp = _display_arm_label(r.parallel_arm, stage=r.stage, multi_stage=multi_stage)
        pt_key = (
            _sanitize_outward_str(r.model_id),
            _sanitize_outward_str(r.sku),
            "Prefill" if r.phase == "prefill" else "Decode",
            arm_disp,
            op_scope,
            int(r.concurrency),
            int(r.seq_len),
        )
        prev = best_by_point.get(pt_key)
        if prev is None or _dominant_row_rank(r) > _dominant_row_rank(prev):
            best_by_point[pt_key] = r
    for pt_key in sorted(best_by_point):
        model_s, sku_s, phase_s, arm_disp, op_scope, conc_i, seq_i = pt_key
        r = best_by_point[pt_key]
        out.append([
            model_s,
            sku_s,
            phase_s,
            arm_disp,
            op_scope,
            seq_i,
            conc_i,
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


def _operator_row_vector_util_and_bw(
    row: OperatorMatrixRow,
    nominal_specs: Mapping[str, HardwareNominalSpec],
) -> tuple[float, float]:
    dur_s = max(float(row.per_call_stage0_us) / _US_PER_SECOND, 1e-12)
    nom_spec = nominal_specs.get(row.sku)
    vec_peak = float(nom_spec.vector_peak_ops_per_s) if nom_spec is not None else 0.0
    vec_util = (
        (float(row.vector_ops) / (dur_s * vec_peak)) * _PCT_SCALE
        if (int(row.vector_ops) > 0 and vec_peak > 0.0)
        else 0.0
    )
    hbm_total = float(row.hbm_read_bytes + row.hbm_write_bytes)
    link_bytes = float(row.link_bytes)
    use_link = (row.bound_type == "Communication Bound" and link_bytes > 0.0) or (
        hbm_total <= 0.0 and link_bytes > 0.0
    )
    bw_bytes = link_bytes if use_link else hbm_total
    achieved_bw = (bw_bytes / dur_s) / BYTES_PER_GB if bw_bytes > 0.0 else 0.0
    return round(vec_util, 2), round(achieved_bw, 2)


def _build_operator_scaling_curves(
    op_rows: Sequence[OperatorMatrixRow],
    nominal_specs: Mapping[str, HardwareNominalSpec],
    *,
    fabric_points: Sequence[FabricOperatorCurvePoint] = (),
    multi_stage: bool = False,
) -> list[list[Any]]:
    """[Cohesive-Pipeline] Build deduplicated contiguous per-operator scaling curves across models and fabric."""
    seqs_by_grp: dict[tuple[str, str, str, str, str, str], set[int]] = defaultdict(set)
    batches_by_grp: dict[tuple[str, str, str, str, str, str], set[int]] = defaultdict(set)
    for r in op_rows:
        op_type = _sanitize_outward_str(r.op_type or r.family)
        kern = _sanitize_outward_str(r.kernel_name or r.upstream_abi)
        arm_disp = _display_arm_label(r.parallel_arm, stage=r.stage, multi_stage=multi_stage)
        grp_k = (
            op_type, kern, _sanitize_outward_str(r.sku), _sanitize_outward_str(r.model_id),
            "Prefill" if r.phase == "prefill" else "Decode", arm_disp,
        )
        seqs_by_grp[grp_k].add(int(r.seq_len))
        batches_by_grp[grp_k].add(int(r.concurrency))

    dedup_model: dict[tuple[str, str, str, str, str, str, str, str, int], OperatorMatrixRow] = {}
    for r in op_rows:
        op_type = _sanitize_outward_str(r.op_type or r.family)
        kern = _sanitize_outward_str(r.kernel_name or r.upstream_abi)
        sku_s = _sanitize_outward_str(r.sku)
        model_s = _sanitize_outward_str(r.model_id)
        phase_s = "Prefill" if r.phase == "prefill" else "Decode"
        arm_disp = _display_arm_label(r.parallel_arm, stage=r.stage, multi_stage=multi_stage)
        grp_k = (op_type, kern, sku_s, model_s, phase_s, arm_disp)
        sweep_batch = len(seqs_by_grp[grp_k]) <= 1 and len(batches_by_grp[grp_k]) >= 2
        if sweep_batch:
            x_axis = "Batch Size"
            x_val = int(r.concurrency)
            fixed_slice = f"Seq={int(r.seq_len)}"
        else:
            x_axis = "Sequence Length"
            x_val = int(r.seq_len)
            fixed_slice = f"Batch={int(r.concurrency)}"
        dedup_key = (op_type, kern, sku_s, model_s, phase_s, arm_disp, fixed_slice, x_axis, x_val)
        prev = dedup_model.get(dedup_key)
        if prev is None or _dominant_row_rank(r) > _dominant_row_rank(prev):
            dedup_model[dedup_key] = r

    merged_rows: dict[tuple[str, str, str, str, str, str, str, str, int], list[Any]] = {}
    for dedup_key, r in dedup_model.items():
        op_type, kern, sku_s, model_s, phase_s, arm_disp, fixed_slice, x_axis, x_val = dedup_key
        vec_util, achieved_bw = _operator_row_vector_util_and_bw(r, nominal_specs)
        merged_rows[dedup_key] = [
            op_type,
            kern,
            _sanitize_outward_str(r.accelerator_core),
            sku_s,
            model_s,
            phase_s,
            arm_disp,
            fixed_slice,
            x_axis,
            x_val,
            _sanitize_outward_str(r.input_shapes),
            _sanitize_outward_str(r.output_shapes),
            round(float(r.nominal_theo_us), 3),
            round(float(r.per_call_stage0_us), 3),
            round(float(r.measured_us), 3) if r.measured_us is not None else "",
            round(float(r.measured_p99_us), 3) if r.measured_p99_us is not None else "",
            _sanitize_outward_str(r.bound_type),
            round(float(r.hbm_read_bytes) / _BYTES_PER_MB, 4),
            round(float(r.hbm_write_bytes) / _BYTES_PER_MB, 4),
            round(float(r.sim_mfu_pct), 2),
            vec_util,
            achieved_bw,
        ]

    for pt in fabric_points:
        op_type = _sanitize_outward_str(pt.op_type)
        kern = _sanitize_outward_str(pt.kernel_name)
        sku_s = _sanitize_outward_str(pt.sku)
        arm_disp = _sanitize_outward_str(f"{pt.route} ({pt.participants}P)")
        fixed_slice = _sanitize_outward_str(f"Ranks={pt.participants} ({pt.route})")
        x_axis = "Message Bytes"
        x_val = int(pt.payload_bytes)
        dedup_key = (op_type, kern, sku_s, _FABRIC_MODEL_LABEL, "Fabric", arm_disp, fixed_slice, x_axis, x_val)
        shape_tag = f"[{x_val}B x {int(pt.participants)}P]"
        merged_rows[dedup_key] = [
            op_type,
            kern,
            "HCCL",
            sku_s,
            _FABRIC_MODEL_LABEL,
            "Fabric",
            arm_disp,
            fixed_slice,
            x_axis,
            x_val,
            shape_tag,
            shape_tag,
            round(float(pt.nominal_theo_us), 3),
            round(float(pt.predicted_us), 3),
            round(float(pt.measured_p50_us), 3) if pt.measured_p50_us is not None else "",
            round(float(pt.measured_p99_us), 3) if pt.measured_p99_us is not None else "",
            _sanitize_outward_str(pt.bound_type),
            0.0,
            0.0,
            0.0,
            0.0,
            round(float(pt.achieved_bw_gb_s), 2),
        ]

    return [merged_rows[k] for k in sorted(merged_rows)]


def _build_operator_catalog_index(curve_rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    idx_map = {name: i for i, name in enumerate(OPERATOR_SCALING_CURVE_COLUMNS)}
    grouped: dict[tuple[str, str], list[Sequence[Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(str(row[idx_map["OP Type"]]), str(row[idx_map["Kernel Name"]]))].append(row)

    out: list[list[Any]] = []
    for (op_type, kern), grp in sorted(grouped.items()):
        cores = sorted({str(r[idx_map["Accelerator Core"]]) for r in grp})
        phases = {str(r[idx_map["Phase"]]) for r in grp}
        scope = "Fabric Collective" if phases == {"Fabric"} else (
            "Model + Fabric" if "Fabric" in phases else "Model Operator"
        )
        skus = {str(r[idx_map["Ascend SKU"]]) for r in grp}
        models = {str(r[idx_map["Model"]]) for r in grp}
        series_keys = {
            (
                str(r[idx_map["Ascend SKU"]]),
                str(r[idx_map["Model"]]),
                str(r[idx_map["Phase"]]),
                str(r[idx_map["Parallel Config"]]),
                str(r[idx_map["Fixed Slice"]]),
                str(r[idx_map["X Axis"]]),
            )
            for r in grp
        }
        x_axes = sorted({str(r[idx_map["X Axis"]]) for r in grp})
        xs = [int(r[idx_map["X Value"]]) for r in grp]
        preds = [float(r[idx_map["Predicted Task Duration(us)"]]) for r in grp]
        effs = [
            (float(r[idx_map["Theoretical Roofline(us)"]]) / float(r[idx_map["Predicted Task Duration(us)"]])) * _PCT_SCALE
            for r in grp
            if float(r[idx_map["Predicted Task Duration(us)"]]) > 0.0 and float(r[idx_map["Theoretical Roofline(us)"]]) > 0.0
        ]
        apes = [
            abs(float(r[idx_map["Predicted Task Duration(us)"]]) - float(r[idx_map["msprof P50(us)"]]))
            / float(r[idx_map["msprof P50(us)"]]) * _PCT_SCALE
            for r in grp
            if r[idx_map["msprof P50(us)"]] != "" and float(r[idx_map["msprof P50(us)"]]) > 0.0
        ]
        bound_counts: dict[str, int] = defaultdict(int)
        for r in grp:
            bound_counts[str(r[idx_map["Roofline Bound"]])] += 1
        primary_bound = max(sorted(bound_counts.items()), key=lambda kv: kv[1])[0]
        mean_eff = round(sum(effs) / float(len(effs)), 2) if effs else 0.0
        mean_ape: Any = round(sum(apes) / float(len(apes)), 2) if apes else ""
        out.append([
            op_type,
            kern,
            ", ".join(cores),
            scope,
            len(skus),
            len(models),
            len(series_keys),
            len(grp),
            " / ".join(x_axes),
            min(xs),
            max(xs),
            round(min(preds), 3),
            round(max(preds), 3),
            mean_eff,
            len(apes),
            mean_ape,
            primary_bound,
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
    ws.set_row_height(1, _HEADER_ROW_HEIGHT_PT)
    for col_idx, heading in enumerate(headers, start=1):
        ws.set_cell(1, col_idx, heading, style=STYLE_HEADER)
        ws.set_col_width(
            col_idx,
            max(_MIN_COL_WIDTH_CHARS, min(_MAX_COL_WIDTH_CHARS, float(len(heading)) + _COL_WIDTH_PAD_CHARS)),
        )
    for row_offset, rvals in enumerate(rows, start=2):
        for col_idx, val in enumerate(rvals, start=1):
            if val == "" or val is None:
                ws.set_cell(row_offset, col_idx, None, style=STYLE_TEXT)
            elif isinstance(val, bool):
                ws.set_cell(row_offset, col_idx, "Yes" if val else "No", style=STYLE_TEXT)
            elif isinstance(val, int):
                ws.set_cell(row_offset, col_idx, val, style=STYLE_INT)
            elif isinstance(val, float):
                ws.set_cell(row_offset, col_idx, val, style=STYLE_FLOAT3)
            else:
                ws.set_cell(row_offset, col_idx, str(val), style=STYLE_TEXT)


def _attach_operator_line_charts(
    chart_ws: Worksheet,
    data_sheet_name: str,
    curve_rows: Sequence[Sequence[Any]],
    *,
    anchor_col_start: int,
    max_charts: int = _MAX_OPERATOR_CHARTS_PER_SHEET,
) -> list[list[Any]]:
    """Attach non-overlapping native OOXML ``<c:lineChart>`` charts per unique canonical operator."""
    if not curve_rows:
        return []
    col_i = {name: i for i, name in enumerate(OPERATOR_SCALING_CURVE_COLUMNS)}
    x_col = column_letter(col_i["X Value"] + 1)
    theo_col = column_letter(col_i["Theoretical Roofline(us)"] + 1)
    pred_col = column_letter(col_i["Predicted Task Duration(us)"] + 1)
    p50_col = column_letter(col_i["msprof P50(us)"] + 1)
    p99_col = column_letter(col_i["msprof P99(us)"] + 1)
    last_col_l = column_letter(len(OPERATOR_SCALING_CURVE_COLUMNS))

    # Identify contiguous series spans (start_row, end_row) in 1-based Excel row indices.
    spans_by_op: dict[tuple[str, str], list[tuple[int, int, Sequence[Any], bool, float]]] = defaultdict(list)
    cur_key: tuple[str, str, str, str, str, str, str, str] | None = None
    span_start = 2
    span_has_meas = False
    span_max_pred = 0.0
    span_first: Sequence[Any] = curve_rows[0]

    def _flush_span(end_row: int) -> None:
        if cur_key is None:
            return
        spans_by_op[(cur_key[0], cur_key[1])].append(
            (span_start, end_row, span_first, span_has_meas, span_max_pred),
        )

    for offset, row in enumerate(curve_rows, start=2):
        series_key = (
            str(row[col_i["OP Type"]]),
            str(row[col_i["Kernel Name"]]),
            str(row[col_i["Ascend SKU"]]),
            str(row[col_i["Model"]]),
            str(row[col_i["Phase"]]),
            str(row[col_i["Parallel Config"]]),
            str(row[col_i["Fixed Slice"]]),
            str(row[col_i["X Axis"]]),
        )
        has_m = row[col_i["msprof P50(us)"]] != ""
        pred_v = float(row[col_i["Predicted Task Duration(us)"]])
        if series_key != cur_key:
            _flush_span(offset - 1)
            cur_key = series_key
            span_start = offset
            span_first = row
            span_has_meas = has_m
            span_max_pred = pred_v
        else:
            span_has_meas = span_has_meas or has_m
            span_max_pred = max(span_max_pred, pred_v)
    _flush_span(len(curve_rows) + 1)

    selected: list[tuple[tuple[str, str], tuple[int, int, Sequence[Any], bool, float]]] = []
    for op_key, spans in sorted(spans_by_op.items()):
        best = max(
            spans,
            key=lambda s: (1 if s[3] else 0, s[1] - s[0] + 1, s[4], -s[0]),
        )
        selected.append((op_key, best))

    selected.sort(
        key=lambda item: (
            0 if (item[1][1] - item[1][0] + 1) >= 2 else 1,
            -item[1][4],
            item[0][0],
            item[0][1],
        ),
    )
    chart_index_rows: list[list[Any]] = []
    for chart_idx, ((op_type, kern), (r_start, r_end, rep, _has_m, _max_p)) in enumerate(selected[:max_charts]):
        sku_s = str(rep[col_i["Ascend SKU"]])
        model_s = str(rep[col_i["Model"]])
        phase_s = str(rep[col_i["Phase"]])
        arm_s = str(rep[col_i["Parallel Config"]])
        slice_s = str(rep[col_i["Fixed Slice"]])
        x_axis_s = str(rep[col_i["X Axis"]])
        grid_col = chart_idx % _OP_CHART_GRID_COLS
        grid_row = chart_idx // _OP_CHART_GRID_COLS
        c1 = anchor_col_start + grid_col * (_OP_CHART_COL_SPAN + _OP_CHART_COL_GAP)
        c2 = c1 + _OP_CHART_COL_SPAN
        r1 = 1 + grid_row * _OP_CHART_ROW_STRIDE
        r2 = r1 + _OP_CHART_ROW_SPAN
        cat_ref = f"{data_sheet_name}!${x_col}${r_start}:${x_col}${r_end}"
        chart_ws.add_line_chart(LineChartSpec(
            title=f"{op_type} ({kern}) | {sku_s} {phase_s} {slice_s}",
            x_title=x_axis_s,
            y_title="Duration (us)",
            anchor_from=(c1, r1),
            anchor_to=(c2, r2),
            series=(
                ChartSeries("Theoretical Roofline(us)", cat_ref, f"{data_sheet_name}!${theo_col}${r_start}:${theo_col}${r_end}", "1F4E78"),
                ChartSeries("Predicted Task Duration(us)", cat_ref, f"{data_sheet_name}!${pred_col}${r_start}:${pred_col}${r_end}", "E67E22"),
                ChartSeries("msprof P50(us)", cat_ref, f"{data_sheet_name}!${p50_col}${r_start}:${p50_col}${r_end}", "2E7D32"),
                ChartSeries("msprof P99(us)", cat_ref, f"{data_sheet_name}!${p99_col}${r_start}:${p99_col}${r_end}", "C62828"),
            ),
        ))
        chart_index_rows.append([
            chart_idx + 1,
            op_type,
            kern,
            sku_s,
            model_s,
            phase_s,
            arm_s,
            slice_s,
            x_axis_s,
            r_end - r_start + 1,
            f"{data_sheet_name}!A{r_start}:{last_col_l}{r_end}",
        ])
    return chart_index_rows


def _write_operator_scaling_curves_xlsx(
    path: Path,
    catalog_index_rows: Sequence[Sequence[Any]],
    all_op_curve_rows: Sequence[Sequence[Any]],
) -> None:
    """Build standalone 3-sheet ``operators/Operator_Scaling_Curves.xlsx`` with native OOXML charts."""
    wb = Workbook()
    ws_idx = wb.create_sheet("Operator_Catalog_Index")
    _populate_table_sheet(ws_idx, OPERATOR_CATALOG_INDEX_COLUMNS, catalog_index_rows)
    ws_all = wb.create_sheet("All_Operator_Curves")
    _populate_table_sheet(ws_all, OPERATOR_SCALING_CURVE_COLUMNS, all_op_curve_rows)
    ws_charts = wb.create_sheet("Operator_Charts")
    chart_anchor_col = len(OPERATOR_CHART_INDEX_COLUMNS) + _OP_CHART_DATA_TABLE_PAD_COLS
    chart_idx_rows = _attach_operator_line_charts(
        ws_charts,
        "All_Operator_Curves",
        all_op_curve_rows,
        anchor_col_start=chart_anchor_col,
    )
    _populate_table_sheet(ws_charts, OPERATOR_CHART_INDEX_COLUMNS, chart_idx_rows)
    wb.save(path)


def _populate_hardware_nominal_sheet(
    ws: Worksheet, nominal_rows: Sequence[Sequence[Any]],
) -> None:
    _populate_table_sheet(ws, HW_NOMINAL_COLUMNS, nominal_rows)
    cube_idx = HW_NOMINAL_COLUMNS.index("Nominal Cube Peak(TFLOPS)") + 1
    vec_idx = HW_NOMINAL_COLUMNS.index("Nominal Vector Peak(TFLOPS)") + 1
    tot_idx = HW_NOMINAL_COLUMNS.index("Nominal Total Peak Cube+Vector(TFLOPS)") + 1
    cube_col = column_letter(cube_idx)
    vec_col = column_letter(vec_idx)
    for r_idx, row in enumerate(nominal_rows, start=2):
        ws.set_cell(r_idx, cube_idx, row[cube_idx - 1], style=STYLE_FLOAT2)
        ws.set_cell(r_idx, vec_idx, row[vec_idx - 1], style=STYLE_FLOAT2)
        ws.set_cell(
            r_idx,
            tot_idx,
            row[tot_idx - 1],
            formula=f"={cube_col}{r_idx}+{vec_col}{r_idx}",
            style=STYLE_FLOAT2,
        )


def _write_executive_summary_xlsx(
    path: Path,
    nominal_rows: Sequence[Sequence[Any]],
    step_rows: Sequence[Sequence[Any]],
    catalog_rows: Sequence[Sequence[Any]],
    operator_curve_rows: Sequence[Sequence[Any]] = (),
) -> None:
    wb = Workbook()
    ws_step = wb.create_sheet("Executive_Summary")
    _populate_table_sheet(ws_step, STEP_SUMMARY_COLUMNS, step_rows)
    adm_col = STEP_SUMMARY_COLUMNS.index("HBM Admissible") + 1
    for r_idx, row in enumerate(step_rows, start=2):
        if row[adm_col - 1] == "No":
            ws_step.set_cell(r_idx, adm_col, "No", style=STYLE_ALERT_HIGH)
    ws_hw = wb.create_sheet("Hardware_Nominal_Specs")
    _populate_hardware_nominal_sheet(ws_hw, nominal_rows)
    ws_cat = wb.create_sheet("Unique_Kernel_Catalog")
    _populate_table_sheet(ws_cat, CATALOG_COLUMNS, catalog_rows)
    ws_op_charts = wb.create_sheet("Operator_Scaling_Charts")
    _populate_table_sheet(ws_op_charts, OPERATOR_SCALING_CURVE_COLUMNS, operator_curve_rows)
    _attach_operator_line_charts(
        ws_op_charts,
        "Operator_Scaling_Charts",
        operator_curve_rows,
        anchor_col_start=len(OPERATOR_SCALING_CURVE_COLUMNS) + _OP_CHART_DATA_TABLE_PAD_COLS,
    )
    wb.save(path)


def _hw_col_range(col_name: str, hw_last_row: int) -> str:
    col_l = column_letter(HW_NOMINAL_COLUMNS.index(col_name) + 1)
    return f"Hardware_Nominal_Specs!${col_l}$2:${col_l}${hw_last_row}"


def _first_contiguous_step_span(scaling_rows: Sequence[Sequence[Any]]) -> tuple[int, int]:
    """Return 1-based ``(start_row, end_row)`` of the first clean single-slice Step Total curve."""
    first_slice: tuple[Any, ...] | None = None
    start_row = 2
    end_row = 2
    for offset, row in enumerate(scaling_rows, start=2):
        if row[4] != "Step Total":
            if first_slice is not None:
                break
            continue
        slice_key = (row[0], row[1], row[2], row[3], row[6])
        if first_slice is None:
            first_slice = slice_key
            start_row = offset
            end_row = offset
        elif slice_key == first_slice:
            end_row = offset
        else:
            break
    if first_slice is None:
        return 2, max(2, len(scaling_rows) + 1)
    return start_row, end_row


def _write_model_operator_breakdown_xlsx(
    path: Path,
    nominal_rows: Sequence[Sequence[Any]],
    model_step_rows: Sequence[Sequence[Any]],
    model_op_rows: Sequence[OperatorMatrixRow],
    scaling_rows: Sequence[Sequence[Any]],
    *,
    operator_curve_rows: Sequence[Sequence[Any]] = (),
    multi_stage: bool = False,
) -> None:
    """[Cohesive-Pipeline] Build per-model OOXML workbook with live formulas & step + per-operator line charts."""
    wb = Workbook()
    # Sheet 1: Hardware_Nominal_Specs (un-derated datasheet denominators)
    ws_hw = wb.create_sheet("Hardware_Nominal_Specs")
    _populate_hardware_nominal_sheet(ws_hw, nominal_rows)
    hw_last_row = max(2, len(nominal_rows) + 1)

    # Sheet 2: Step_Summary
    ws_step = wb.create_sheet("Step_Summary")
    _populate_table_sheet(ws_step, STEP_SUMMARY_COLUMNS, model_step_rows)

    # Sheet 3: Op_Breakdown (with LIVE Excel formulas referencing Hardware_Nominal_Specs!)
    ws_op = wb.create_sheet("Op_Breakdown")
    ws_op.freeze_panes(1, 0)
    ws_op.set_row_height(1, _HEADER_ROW_HEIGHT_PT)
    col_idx_map = {name: i + 1 for i, name in enumerate(OUTWARD_OP_COLUMNS)}
    for name, cidx in col_idx_map.items():
        ws_op.set_cell(1, cidx, name, style=STYLE_HEADER)
        ws_op.set_col_width(
            cidx,
            max(_MIN_COL_WIDTH_CHARS, min(_MAX_COL_WIDTH_CHARS, float(len(name)) + _COL_WIDTH_PAD_CHARS)),
        )

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
    hw_hf32_rng = _hw_col_range("HF32 Cube Peak(FLOPs/s)", hw_last_row)
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
        if dtype_low == "hf32":
            cube_rng = hw_hf32_rng
        elif dtype_low in {"fp32", "float32"}:
            cube_rng = hw_fp32_rng
        elif dtype_low in {"int4", "fp4", "fp4_e2m1", "mxfp4", "mxfp4_e2m1"}:
            cube_rng = hw_int4_rng
        elif dtype_low in {"int8", "w8a8", "w4a8"}:
            cube_rng = hw_int8_rng
        elif dtype_low.startswith(("fp8", "mxfp8", "w8a8_fp8", "hif8", "hifloat8")):
            cube_rng = hw_fp8_rng
        else:
            cube_rng = hw_bf16_rng
        cube_peak_expr = f"INDEX({cube_rng},{sku_match})"
        vec_peak_expr = f"INDEX({hw_vec_rng},{sku_match})"
        hbm_peak_expr = f"INDEX({hw_hbm_rng},{sku_match})"
        tier_low = str(op_row.link_tier or "").lower()
        if "sio" in tier_low or "intra_pkg" in tier_low or "intra_package" in tier_low:
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

    # Sheet 4: Scaling_Curves + Native OOXML Step-Level Line Charts (xl/charts/chart1..2.xml)
    ws_sc = wb.create_sheet("Scaling_Curves")
    _populate_table_sheet(ws_sc, SCALING_CURVE_COLUMNS, scaling_rows)
    if scaling_rows:
        chart_start, chart_end = _first_contiguous_step_span(scaling_rows)
        seq_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Sequence Length") + 1)
        theo_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Theoretical Roofline(us)") + 1)
        pred_col_l = column_letter(SCALING_CURVE_COLUMNS.index("Predicted Task Duration(us)") + 1)
        p50_col_l = column_letter(SCALING_CURVE_COLUMNS.index("msprof Task Duration P50(us)") + 1)
        p99_col_l = column_letter(SCALING_CURVE_COLUMNS.index("msprof Task Duration P99(us)") + 1)
        mfu_col_l = column_letter(SCALING_CURVE_COLUMNS.index("MFU(%)") + 1)
        mbu_col_l = column_letter(SCALING_CURVE_COLUMNS.index("HBM Bandwidth Utilization(%)") + 1)
        mbu_rd_col_l = column_letter(SCALING_CURVE_COLUMNS.index("HBM Read MBU(%)") + 1)
        cat_ref = f"Scaling_Curves!${seq_col_l}${chart_start}:${seq_col_l}${chart_end}"
        ws_sc.add_line_chart(LineChartSpec(
            title="Latency Scaling: Theoretical Roofline vs Predicted vs msprof",
            x_title="Sequence Length (Tokens)",
            y_title="Duration (us)",
            anchor_from=(_STEP_CHART_ANCHOR_COL_START, _STEP_CHART_LAT_ROW_START),
            anchor_to=(_STEP_CHART_ANCHOR_COL_END, _STEP_CHART_LAT_ROW_END),
            series=(
                ChartSeries("Theoretical Roofline(us)", cat_ref, f"Scaling_Curves!${theo_col_l}${chart_start}:${theo_col_l}${chart_end}", "1F4E78"),
                ChartSeries("Predicted Task Duration(us)", cat_ref, f"Scaling_Curves!${pred_col_l}${chart_start}:${pred_col_l}${chart_end}", "E67E22"),
                ChartSeries("msprof P50(us)", cat_ref, f"Scaling_Curves!${p50_col_l}${chart_start}:${p50_col_l}${chart_end}", "2E7D32"),
                ChartSeries("msprof P99(us)", cat_ref, f"Scaling_Curves!${p99_col_l}${chart_start}:${p99_col_l}${chart_end}", "C62828"),
            ),
        ))
        ws_sc.add_line_chart(LineChartSpec(
            title="Hardware Efficiency Scaling: MFU(%) & HBM Utilization(%)",
            x_title="Sequence Length (Tokens)",
            y_title="Utilization (%)",
            anchor_from=(_STEP_CHART_ANCHOR_COL_START, _STEP_CHART_UTIL_ROW_START),
            anchor_to=(_STEP_CHART_ANCHOR_COL_END, _STEP_CHART_UTIL_ROW_END),
            series=(
                ChartSeries("MFU(%)", cat_ref, f"Scaling_Curves!${mfu_col_l}${chart_start}:${mfu_col_l}${chart_end}", "1F4E78"),
                ChartSeries("HBM Bandwidth Utilization(%)", cat_ref, f"Scaling_Curves!${mbu_col_l}${chart_start}:${mbu_col_l}${chart_end}", "2E7D32"),
                ChartSeries("HBM Read MBU(%)", cat_ref, f"Scaling_Curves!${mbu_rd_col_l}${chart_start}:${mbu_rd_col_l}${chart_end}", "8E44AD"),
            ),
        ))

    # Sheet 5: Operator_Scaling_Charts (contiguous per-operator curves + native OOXML line charts)
    ws_op_charts = wb.create_sheet("Operator_Scaling_Charts")
    _populate_table_sheet(ws_op_charts, OPERATOR_SCALING_CURVE_COLUMNS, operator_curve_rows)
    _attach_operator_line_charts(
        ws_op_charts,
        "Operator_Scaling_Charts",
        operator_curve_rows,
        anchor_col_start=len(OPERATOR_SCALING_CURVE_COLUMNS) + _OP_CHART_DATA_TABLE_PAD_COLS,
    )
    wb.save(path)


# ---------------------------------------------------------------------------
# Pure-English README.md & Interactive Zero-Dependency HTML Viewer
# ---------------------------------------------------------------------------

def _render_readme_md(
    manifest: OperatorMatrixManifest,
    nominal_rows: Sequence[Sequence[Any]],
    *,
    unique_operator_count: int = 0,
    fabric_curve_point_count: int = 0,
) -> str:
    models = sorted({_sanitize_outward_str(s.model_id) for s in manifest.step_summaries} | {
        _sanitize_outward_str(r.model_id) for r in manifest.rows
    })
    lines: list[str] = [
        "# Ascend NPU Operator Performance Breakdown & Roofline Package",
        "",
        "## Package Layout",
        "",
        "- `manifest.json`: Machine-readable pure-English index of all workbooks, CSV tables, operator curves, and interactive explorer modes.",
        "- `hardware_nominal_specs.csv`: Un-derated public datasheet hardware nominal specifications per Ascend SKU.",
        "- `summary/Executive_Summary.xlsx` & `summary/model_step_summary.csv`: Cross-model step latency, MFU(%), Model MBU(%), Physical HBM Utilization(%), dual-peak HBM admissibility, and `Operator_Scaling_Charts` (per-operator native OOXML line charts).",
        "- `models/<model_id>/Operator_Breakdown.xlsx`: Per-model workbook containing `Hardware_Nominal_Specs`, `Step_Summary`, `Op_Breakdown` (live Excel formulas referencing `Hardware_Nominal_Specs`), `Scaling_Curves` (step-level native OOXML line charts), and `Operator_Scaling_Charts` (per-operator native OOXML line charts).",
        "- `models/<model_id>/op_breakdown_matrix.csv` & `models/<model_id>/scaling_curves.csv`: Verbatim CANN `msprof`-aligned CSV tables.",
        f"- `operators/Operator_Scaling_Curves.xlsx` & `operators/operator_scaling_curves.csv`: Standalone all-operators scaling package covering all {unique_operator_count} unique canonical operators (including {fabric_curve_point_count} fabric interconnect collective ladder points over Message Bytes) with `Operator_Catalog_Index`, `All_Operator_Curves`, and `Operator_Charts` (native OOXML line charts per operator).",
        "- `operators/unique_kernel_catalog.csv`: Cross-model deduplicated unique kernel catalog ranked by aggregate exposed step time.",
        "- `charts/scaling_explorer.html`: Standalone zero-dependency interactive SVG chart viewer featuring both `By Model -> Operator` mode (with Parallel Config and Fixed Slice multi-series selectors) and `All Operators Catalog` mode (searchable across all model operators and fabric collectives).",
        "",
        "## Hardware Nominal Datasheet Specifications (Un-Derated, Utilization = 1.0)",
        "",
        "| Hardware SKU | Arch | AIC | AIV | HBM (GiB) | L2 (MiB) | UB (KiB) | Clock (MHz) | Nominal Cube Peak (TFLOPS) | Nominal Vector Peak (TFLOPS) | Nominal Total Peak Cube+Vector (TFLOPS) | INT8 Cube (TOPS) | FP8 Cube (TFLOPS) | HBM Peak (GB/s) | Intra-Node Link (GB/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    hw_idx = {name: i for i, name in enumerate(HW_NOMINAL_COLUMNS)}
    for nr in nominal_rows:
        lines.append(
            f"| {nr[hw_idx['Hardware SKU']]} | {nr[hw_idx['Architecture']]} | {nr[hw_idx['AIC Cores']]} | "
            f"{nr[hw_idx['AIV Cores']]} | {nr[hw_idx['HBM Capacity(GiB)']]} | {nr[hw_idx['L2 Cache(MiB)']]} | "
            f"{nr[hw_idx['UB per Core(KiB)']]} | {nr[hw_idx['Nominal Clock(MHz)']]} | "
            f"{float(nr[hw_idx['Nominal Cube Peak(TFLOPS)']]):.2f} | "
            f"{float(nr[hw_idx['Nominal Vector Peak(TFLOPS)']]):.2f} | "
            f"{float(nr[hw_idx['Nominal Total Peak Cube+Vector(TFLOPS)']]):.2f} | "
            f"{nr[hw_idx['INT8 Cube Peak(TOPS)']]} | {nr[hw_idx['FP8 Cube Peak(TFLOPS)']]} | "
            f"{nr[hw_idx['HBM Peak(GB/s)']]} | {nr[hw_idx['Intra-Node Link Peak(GB/s)']]} |"
        )
    lines.extend([
        "",
        "## Roofline, MFU & MBU Definitions",
        "",
        "- **Nominal Cube Peak(TFLOPS)**: Pure BF16/FP16 Cube matrix peak (`AIC` units only); used as `P_cube_nominal` for Cube Roofline and `MFU(%)`.",
        "- **Nominal Vector Peak(TFLOPS)**: Full-register FP16/BF16 Vector FMA peak (`AIV` units only, `8x` unified FP32 single-instruction lane rate `Vector Peak(Ops/s)`); Vector Roofline uses pure Vector Peak (`P_vec_nominal`), never combined `Cube + Vector`.",
        "- **Nominal Total Peak Cube+Vector(TFLOPS)**: `Nominal Cube Peak(TFLOPS) + Nominal Vector Peak(TFLOPS)`, reconciling pure unit peaks to headline Ascend datasheet totals (for example, `432.54 + 54.07 = 486.60 TFLOPS` on `Ascend950DT_9582` at `1650 MHz`, `270.34 + 16.90 = 287.23 TFLOPS` on `Ascend910B4-1` / `Ascend910_9362` at `1650 MHz`, `313.34 TFLOPS` on `Ascend910B3`, and `376.01 TFLOPS` on `Ascend910B2`).",
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


def _render_bundle_manifest_json(
    *,
    model_ids: Sequence[str],
    skus: Sequence[str],
    unique_operator_count: int,
    unique_kernel_count: int,
    fabric_curve_point_count: int,
    campaign_attached: bool,
) -> str:
    doc = {
        "schema": _OUTWARD_BUNDLE_SCHEMA,
        "readme_md": "README.md",
        "hardware_nominal_specs_csv": "hardware_nominal_specs.csv",
        "summary": {
            "executive_summary_xlsx": "summary/Executive_Summary.xlsx",
            "model_step_summary_csv": "summary/model_step_summary.csv",
            "sheets": [
                "Executive_Summary",
                "Hardware_Nominal_Specs",
                "Unique_Kernel_Catalog",
                "Operator_Scaling_Charts",
            ],
        },
        "operators": {
            "unique_kernel_catalog_csv": "operators/unique_kernel_catalog.csv",
            "operator_scaling_curves_xlsx": "operators/Operator_Scaling_Curves.xlsx",
            "operator_scaling_curves_csv": "operators/operator_scaling_curves.csv",
            "sheets": [
                "Operator_Catalog_Index",
                "All_Operator_Curves",
                "Operator_Charts",
            ],
            "unique_operator_count": int(unique_operator_count),
            "unique_kernel_count": int(unique_kernel_count),
            "fabric_curve_point_count": int(fabric_curve_point_count),
        },
        "charts": {
            "scaling_explorer_html": "charts/scaling_explorer.html",
            "modes": ["By Model -> Operator", "All Operators Catalog"],
        },
        "models": [
            {
                "model_id": _sanitize_outward_str(mid),
                "operator_breakdown_xlsx": f"models/{_safe_model_dirname(mid)}/Operator_Breakdown.xlsx",
                "op_breakdown_matrix_csv": f"models/{_safe_model_dirname(mid)}/op_breakdown_matrix.csv",
                "scaling_curves_csv": f"models/{_safe_model_dirname(mid)}/scaling_curves.csv",
            }
            for mid in model_ids
        ],
        "skus": [_sanitize_outward_str(s) for s in skus],
        "campaign_attached": bool(campaign_attached),
    }
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _escape_script_json(obj: Any) -> str:
    return (
        json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_scaling_explorer_html(
    all_curve_rows: Sequence[Sequence[Any]],
    all_operator_curve_rows: Sequence[Sequence[Any]] = (),
    operator_catalog_index: Sequence[Sequence[Any]] = (),
) -> str:
    """[Cohesive-Pipeline] Render standalone zero-dependency pure-English SVG scaling explorer."""
    model_payload: list[dict[str, Any]] = []
    for r in all_curve_rows:
        model_payload.append({
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
    op_col = {name: i for i, name in enumerate(OPERATOR_SCALING_CURVE_COLUMNS)}
    op_payload: list[dict[str, Any]] = []
    for r in all_operator_curve_rows:
        op_type = str(r[op_col["OP Type"]])
        kern = str(r[op_col["Kernel Name"]])
        op_payload.append({
            "opType": op_type,
            "kernel": kern,
            "opKey": f"{op_type} | {kern}",
            "core": str(r[op_col["Accelerator Core"]]),
            "sku": str(r[op_col["Ascend SKU"]]),
            "model": str(r[op_col["Model"]]),
            "phase": str(r[op_col["Phase"]]),
            "arm": str(r[op_col["Parallel Config"]]),
            "slice": str(r[op_col["Fixed Slice"]]),
            "xAxis": str(r[op_col["X Axis"]]),
            "xVal": int(r[op_col["X Value"]]),
            "inShapes": str(r[op_col["Input Shapes"]]),
            "outShapes": str(r[op_col["Output Shapes"]]),
            "theo": float(r[op_col["Theoretical Roofline(us)"]]),
            "pred": float(r[op_col["Predicted Task Duration(us)"]]),
            "p50": float(r[op_col["msprof P50(us)"]]) if r[op_col["msprof P50(us)"]] != "" else None,
            "p99": float(r[op_col["msprof P99(us)"]]) if r[op_col["msprof P99(us)"]] != "" else None,
            "bound": str(r[op_col["Roofline Bound"]]),
            "hbmRdMb": float(r[op_col["HBM Read(MB)"]]),
            "hbmWrMb": float(r[op_col["HBM Write(MB)"]]),
            "cubeUtil": float(r[op_col["Cube Utilization(%)"]]),
            "vecUtil": float(r[op_col["Vector Utilization(%)"]]),
            "bwGbS": float(r[op_col["Achieved Bandwidth(GB/s)"]]),
        })
    idx_col = {name: i for i, name in enumerate(OPERATOR_CATALOG_INDEX_COLUMNS)}
    idx_payload: list[dict[str, Any]] = []
    for r in operator_catalog_index:
        op_type = str(r[idx_col["OP Type"]])
        kern = str(r[idx_col["Kernel Name"]])
        idx_payload.append({
            "opType": op_type,
            "kernel": kern,
            "opKey": f"{op_type} | {kern}",
            "core": str(r[idx_col["Accelerator Core"]]),
            "scope": str(r[idx_col["Scope"]]),
            "skuCount": int(r[idx_col["SKU Count"]]),
            "modelCount": int(r[idx_col["Model Count"]]),
            "seriesCount": int(r[idx_col["Curve Series Count"]]),
            "pointCount": int(r[idx_col["Data Points"]]),
            "xAxis": str(r[idx_col["X Axis"]]),
            "minX": int(r[idx_col["Min X"]]),
            "maxX": int(r[idx_col["Max X"]]),
            "minPred": float(r[idx_col["Min Predicted(us)"]]),
            "maxPred": float(r[idx_col["Max Predicted(us)"]]),
            "meanEff": float(r[idx_col["Mean Roofline Efficiency(%)"]]),
            "measCount": int(r[idx_col["Measured Points"]]),
            "meanApe": float(r[idx_col["Mean Prediction APE(%)"]]) if r[idx_col["Mean Prediction APE(%)"]] != "" else None,
            "bound": str(r[idx_col["Primary Roofline Bound"]]),
        })
    rows_json = _escape_script_json(model_payload)
    op_rows_json = _escape_script_json(op_payload)
    op_index_json = _escape_script_json(idx_payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ascend NPU Operator Scaling Explorer</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --fg:#f8fafc; --muted:#94a3b8; --accent:#38bdf8; --border:#334155; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); padding:20px; }}
  .topbar {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; }}
  h1 {{ margin:0; font-size:20px; font-weight:600; }}
  .mode-tabs {{ display:inline-flex; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:3px; gap:4px; }}
  .mode-btn {{ background:transparent; color:var(--muted); border:none; border-radius:6px; padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer; }}
  .mode-btn.active {{ background:#0284c7; color:#ffffff; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:12px; background:var(--card); padding:14px; border-radius:8px; border:1px solid var(--border); margin-bottom:16px; }}
  .hidden {{ display:none !important; }}
  .ctrl {{ display:flex; flex-direction:column; gap:4px; min-width:148px; }}
  .ctrl.wide {{ min-width:240px; flex:1 1 240px; }}
  .ctrl label {{ font-size:11px; text-transform:uppercase; color:var(--muted); letter-spacing:0.04em; }}
  .ctrl select, .ctrl input {{ background:#0f172a; color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 8px; font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
  .panel {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; }}
  .panel h2 {{ margin:0 0 8px 0; font-size:15px; color:#e2e8f0; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:12px; font-size:11px; color:var(--muted); margin-bottom:8px; min-height:18px; }}
  .swatch {{ display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:5px; }}
  .swatch.dashed {{ border-top:2px dashed; height:0; background:transparent !important; }}
  svg {{ width:100%; height:320px; display:block; }}
  .section-hdr {{ margin:18px 0 8px 0; font-size:14px; color:#cbd5e1; display:flex; justify-content:space-between; align-items:center; }}
  .table-wrap {{ overflow-x:auto; border-radius:8px; border:1px solid var(--border); background:var(--card); }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; background:var(--card); }}
  th, td {{ padding:6px 10px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; background:#162032; }}
  tr.clickable {{ cursor:pointer; }}
  tr.clickable:hover {{ background:#26354d; }}
  tr.selected-row {{ background:#1d4ed833; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Ascend NPU Operator &amp; Step Scaling Explorer</h1>
  <div class="mode-tabs" role="tablist">
    <button id="btnModeModel" type="button" class="mode-btn active" data-mode="model">By Model -&gt; Operator</button>
    <button id="btnModeCatalog" type="button" class="mode-btn" data-mode="catalog">All Operators Catalog</button>
  </div>
</div>

<div id="modelControls" class="controls">
  <div class="ctrl"><label for="selModel">Model</label><select id="selModel"></select></div>
  <div class="ctrl"><label for="selSku">Hardware SKU</label><select id="selSku"></select></div>
  <div class="ctrl"><label for="selPhase">Phase</label><select id="selPhase"></select></div>
  <div class="ctrl"><label for="selArm">Parallel Config</label><select id="selArm"></select></div>
  <div class="ctrl wide"><label for="selOp">Operator Scope</label><select id="selOp"></select></div>
  <div class="ctrl"><label for="selAxis">X-Axis</label><select id="selAxis"><option value="seq">Sequence Length</option><option value="batch">Concurrency (Batch)</option></select></div>
  <div class="ctrl"><label id="lblFixedSlice" for="selFixedSlice">Batch Size</label><select id="selFixedSlice"></select></div>
</div>

<div id="catalogControls" class="controls hidden">
  <div class="ctrl wide"><label for="opSearch">Search Operators</label><input id="opSearch" type="search" placeholder="Filter OP Type, Kernel Name, Core, Bound..."></div>
  <div class="ctrl wide"><label for="catOp">Operator (OP Type | Kernel Name)</label><select id="catOp"></select></div>
  <div class="ctrl"><label for="catSku">Hardware SKU</label><select id="catSku"></select></div>
  <div class="ctrl"><label for="catPhase">Phase</label><select id="catPhase"></select></div>
  <div class="ctrl wide"><label for="catModelArm">Model / Parallel Config</label><select id="catModelArm"></select></div>
  <div class="ctrl"><label for="catSlice">Fixed Slice</label><select id="catSlice"></select></div>
</div>

<div class="grid">
  <div class="panel">
    <h2 id="latTitle">Latency Scaling (us): Theoretical Roofline vs Predicted vs msprof</h2>
    <div id="latLegend" class="legend"></div>
    <svg id="svgLatency" viewBox="0 0 640 320"></svg>
  </div>
  <div class="panel">
    <h2 id="utilTitle">Hardware Efficiency Scaling (%): MFU &amp; HBM Bandwidth Utilization</h2>
    <div id="utilLegend" class="legend"></div>
    <svg id="svgUtil" viewBox="0 0 640 320"></svg>
  </div>
</div>

<div id="catalogSummarySection" class="hidden">
  <div class="section-hdr">
    <span>All Operators Catalog Summary (click any operator to plot its scaling curves)</span>
    <span id="catSummaryCount"></span>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>OP Type</th><th>Kernel Name</th><th>Core</th><th>Scope</th><th>Curves</th>
          <th>Points</th><th>X Axis</th><th>Min Dur(us)</th><th>Max Dur(us)</th>
          <th>Roofline Eff(%)</th><th>Measured Pts</th><th>Mean APE(%)</th><th>Primary Bound</th>
        </tr>
      </thead>
      <tbody id="catSummaryBody"></tbody>
    </table>
  </div>
</div>

<div class="section-hdr">
  <span id="detailTableTitle">Active Scaling Points</span>
  <span id="detailTableCount"></span>
</div>
<div class="table-wrap">
  <table>
    <thead id="tblHead"></thead>
    <tbody id="tblBody"></tbody>
  </table>
</div>

<script>
const ROWS = {rows_json};
const OP_ROWS = {op_rows_json};
const OP_INDEX = {op_index_json};
const ALL_TOKEN = "__ALL__";
const ALL_MODELS_TOKEN = "__ALL_MODELS__";
const PALETTE = ["#f59e0b", "#38bdf8", "#22c55e", "#a855f7", "#ef4444", "#14b8a6", "#f472b6", "#eab308", "#60a5fa", "#fb7185"];
const uniq = (arr) => Array.from(new Set(arr));
const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs, text) => {{
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {{}})) el.setAttribute(k, String(v));
  if (text !== undefined) el.textContent = String(text);
  return el;
}};
const fillSel = (el, items) => {{
  const prev = el.value;
  el.replaceChildren();
  const normalized = items.map(item => typeof item === "object" ? item : {{ value: String(item), label: String(item) }});
  for (const item of normalized) {{
    const opt = document.createElement("option");
    opt.value = item.value;
    opt.textContent = item.label;
    el.appendChild(opt);
  }}
  if (normalized.some(it => it.value === prev)) el.value = prev;
}};
const fmtX = (v) => {{
  const n = Number(v);
  if (n >= 1048576 && n % 1048576 === 0) return `${{n / 1048576}}M`;
  if (n >= 1024 && n % 1024 === 0) return `${{n / 1024}}K`;
  return String(n);
}};

const btnModeModel = document.getElementById("btnModeModel");
const btnModeCatalog = document.getElementById("btnModeCatalog");
const modelControls = document.getElementById("modelControls");
const catalogControls = document.getElementById("catalogControls");
const catalogSummarySection = document.getElementById("catalogSummarySection");

const selModel = document.getElementById("selModel");
const selSku = document.getElementById("selSku");
const selPhase = document.getElementById("selPhase");
const selArm = document.getElementById("selArm");
const selOp = document.getElementById("selOp");
const selAxis = document.getElementById("selAxis");
const selFixedSlice = document.getElementById("selFixedSlice");
const lblFixedSlice = document.getElementById("lblFixedSlice");

const opSearch = document.getElementById("opSearch");
const catOp = document.getElementById("catOp");
const catSku = document.getElementById("catSku");
const catPhase = document.getElementById("catPhase");
const catModelArm = document.getElementById("catModelArm");
const catSlice = document.getElementById("catSlice");

let activeMode = "model";

function setLegend(el, seriesList) {{
  el.replaceChildren();
  for (const s of seriesList) {{
    if (!s.points || !s.points.length) continue;
    const span = document.createElement("span");
    const sw = document.createElement("i");
    sw.className = s.dashed ? "swatch dashed" : "swatch";
    if (s.dashed) sw.style.borderTopColor = s.color;
    else sw.style.background = s.color;
    span.appendChild(sw);
    span.appendChild(document.createTextNode(s.label));
    el.appendChild(span);
  }}
}}

function dedupSortedSeries(rawPts) {{
  const byX = new Map();
  for (const pt of rawPts) {{
    if (pt.y === null || pt.y === undefined || Number.isNaN(pt.y)) continue;
    const prev = byX.get(pt.x);
    if (!prev || pt.y > prev.y) byX.set(pt.x, pt);
  }}
  return Array.from(byX.values()).sort((a, b) => a.x - b.x);
}}

function drawMultiSeriesChart(svg, seriesList, xAxisLabel, yAxisLabel) {{
  const W = 640, H = 320, L = 66, R = 18, T = 14, B = 48;
  svg.replaceChildren();
  const cleanSeries = seriesList.map(s => ({{ ...s, points: dedupSortedSeries(s.points || []) }})).filter(s => s.points.length > 0);
  if (!cleanSeries.length) {{
    svg.appendChild(svgEl("text", {{ x: W / 2, y: H / 2, fill: "#94a3b8", "text-anchor": "middle", "font-size": 13 }}, "No matching data points"));
    return;
  }}
  const xTicks = Array.from(new Set(cleanSeries.flatMap(s => s.points.map(p => p.x)))).sort((a, b) => a - b);
  const xIndex = new Map(xTicks.map((v, idx) => [v, idx]));
  let yMax = 0;
  for (const s of cleanSeries) for (const p of s.points) if (p.y > yMax) yMax = p.y;
  yMax = yMax > 0 ? yMax * 1.12 : 1.0;
  const xPos = (xv) => xTicks.length === 1 ? (L + W - R) / 2 : L + (xIndex.get(xv) / (xTicks.length - 1)) * (W - L - R);
  const yPos = (yv) => H - B - (yv / yMax) * (H - T - B);

  for (let g = 0; g <= 4; g++) {{
    const val = (yMax * g) / 4, y = yPos(val);
    svg.appendChild(svgEl("line", {{ x1: L, y1: y, x2: W - R, y2: y, stroke: "#334155", "stroke-dasharray": "3,3" }}));
    svg.appendChild(svgEl("text", {{ x: L - 6, y: y + 4, fill: "#94a3b8", "text-anchor": "end", "font-size": 10 }}, val >= 100 ? val.toFixed(0) : val.toFixed(2)));
  }}
  const tickStep = Math.max(1, Math.ceil(xTicks.length / 10));
  xTicks.forEach((xv, i) => {{
    if (i % tickStep !== 0 && i !== xTicks.length - 1) return;
    svg.appendChild(svgEl("text", {{ x: xPos(xv), y: H - B + 16, fill: "#94a3b8", "text-anchor": "middle", "font-size": 10 }}, fmtX(xv)));
  }});
  svg.appendChild(svgEl("text", {{ x: (L + W - R) / 2, y: H - 8, fill: "#94a3b8", "text-anchor": "middle", "font-size": 11 }}, xAxisLabel));
  svg.appendChild(svgEl("text", {{ x: 14, y: (T + H - B) / 2, fill: "#94a3b8", "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 14 ${{(T + H - B) / 2}})` }}, yAxisLabel));

  for (const s of cleanSeries) {{
    const d = s.points.map((p, k) => `${{k === 0 ? "M" : "L"}}${{xPos(p.x).toFixed(1)}},${{yPos(p.y).toFixed(1)}}`).join(" ");
    const pathAttrs = {{ d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, opacity: s.opacity || 1 }};
    if (s.dashed) pathAttrs["stroke-dasharray"] = "5,4";
    svg.appendChild(svgEl("path", pathAttrs));
    for (const p of s.points) {{
      const c = svgEl("circle", {{ cx: xPos(p.x).toFixed(1), cy: yPos(p.y).toFixed(1), r: s.radius || 3.2, fill: s.color, opacity: s.opacity || 1 }});
      c.appendChild(svgEl("title", {{}}, `${{s.label}} @ ${{fmtX(p.x)}}: ${{p.y.toFixed(3)}}`));
      svg.appendChild(c);
    }}
  }}
}}

function setTableHeader(cols) {{
  const thead = document.getElementById("tblHead");
  thead.replaceChildren();
  const tr = document.createElement("tr");
  for (const c of cols) {{
    const th = document.createElement("th");
    th.textContent = c;
    tr.appendChild(th);
  }}
  thead.appendChild(tr);
}}

function refreshModelSelectors() {{
  fillSel(selModel, uniq(ROWS.map(r => r.model)));
  const mRows = ROWS.filter(r => r.model === selModel.value);
  fillSel(selSku, uniq(mRows.map(r => r.sku)));
  const sRows = mRows.filter(r => r.sku === selSku.value);
  fillSel(selPhase, uniq(sRows.map(r => r.phase)));
  const pRows = sRows.filter(r => r.phase === selPhase.value);
  fillSel(selArm, uniq(pRows.map(r => r.arm)));
  const aRows = pRows.filter(r => r.arm === selArm.value);
  fillSel(selOp, uniq(aRows.map(r => r.op)));
  refreshModelFixedSlice();
}}

function refreshModelFixedSlice() {{
  const xKey = selAxis.value;
  const sliceKey = xKey === "seq" ? "batch" : "seq";
  lblFixedSlice.textContent = xKey === "seq" ? "Batch Size (Fixed Slice)" : "Sequence Length (Fixed Slice)";
  const base = ROWS.filter(r =>
    r.model === selModel.value &&
    r.sku === selSku.value &&
    r.phase === selPhase.value &&
    r.arm === selArm.value &&
    r.op === selOp.value
  );
  const slices = uniq(base.map(r => r[sliceKey])).sort((a, b) => a - b);
  const allLabel = xKey === "seq" ? "All Batches (Multi-Series)" : "All Seq Lengths (Multi-Series)";
  const prefix = xKey === "seq" ? "Batch" : "Seq";
  const opts = slices.map(v => ({{ value: String(v), label: `${{prefix}} = ${{fmtX(v)}}` }}));
  if (slices.length > 1) opts.unshift({{ value: ALL_TOKEN, label: allLabel }});
  else if (slices.length === 0) opts.push({{ value: ALL_TOKEN, label: allLabel }});
  fillSel(selFixedSlice, opts);
  renderModelMode();
}}

function renderModelMode() {{
  const xKey = selAxis.value;
  const sliceKey = xKey === "seq" ? "batch" : "seq";
  const xLabel = xKey === "seq" ? "Sequence Length (Tokens)" : "Concurrency (Batch Size)";
  const prefix = xKey === "seq" ? "Batch" : "Seq";
  const base = ROWS.filter(r =>
    r.model === selModel.value &&
    r.sku === selSku.value &&
    r.phase === selPhase.value &&
    r.arm === selArm.value &&
    r.op === selOp.value
  );
  const sliceVal = selFixedSlice.value;
  const latSeries = [];
  const utilSeries = [];
  let tableRows = [];

  if (sliceVal === ALL_TOKEN) {{
    const slices = uniq(base.map(r => r[sliceKey])).sort((a, b) => a - b);
    slices.forEach((sv, idx) => {{
      const col = PALETTE[idx % PALETTE.length];
      const sub = base.filter(r => r[sliceKey] === sv).sort((a, b) => a[xKey] - b[xKey]);
      tableRows.push(...sub);
      latSeries.push({{ label: `${{prefix}}=${{fmtX(sv)}} Predicted(us)`, color: col, points: sub.map(p => ({{ x: p[xKey], y: p.pred }})) }});
      latSeries.push({{ label: `${{prefix}}=${{fmtX(sv)}} Roofline(us)`, color: col, dashed: true, width: 1.5, opacity: 0.65, radius: 2.2, points: sub.map(p => ({{ x: p[xKey], y: p.theo }})) }});
      const meas = sub.filter(p => p.p50 !== null);
      if (meas.length) latSeries.push({{ label: `${{prefix}}=${{fmtX(sv)}} msprof P50(us)`, color: "#22c55e", dashed: true, width: 1.8, radius: 4.0, points: meas.map(p => ({{ x: p[xKey], y: p.p50 }})) }});
      utilSeries.push({{ label: `${{prefix}}=${{fmtX(sv)}} HBM Util(%)`, color: col, points: sub.map(p => ({{ x: p[xKey], y: p.mbu }})) }});
      utilSeries.push({{ label: `${{prefix}}=${{fmtX(sv)}} MFU(%)`, color: col, dashed: true, width: 1.5, opacity: 0.65, radius: 2.2, points: sub.map(p => ({{ x: p[xKey], y: p.mfu }})) }});
    }});
  }} else {{
    const targetSlice = Number(sliceVal);
    const sub = base.filter(r => r[sliceKey] === targetSlice).sort((a, b) => a[xKey] - b[xKey]);
    tableRows = sub;
    latSeries.push(
      {{ label: "Theoretical Roofline(us)", color: "#38bdf8", dashed: true, points: sub.map(p => ({{ x: p[xKey], y: p.theo }})) }},
      {{ label: "Predicted Task Duration(us)", color: "#f59e0b", points: sub.map(p => ({{ x: p[xKey], y: p.pred }})) }},
      {{ label: "msprof P50(us)", color: "#22c55e", points: sub.filter(p => p.p50 !== null).map(p => ({{ x: p[xKey], y: p.p50 }})) }},
      {{ label: "msprof P99(us)", color: "#ef4444", points: sub.filter(p => p.p99 !== null).map(p => ({{ x: p[xKey], y: p.p99 }})) }},
    );
    utilSeries.push(
      {{ label: "MFU(%)", color: "#38bdf8", points: sub.map(p => ({{ x: p[xKey], y: p.mfu }})) }},
      {{ label: "HBM Bandwidth Utilization(%)", color: "#22c55e", points: sub.map(p => ({{ x: p[xKey], y: p.mbu }})) }},
      {{ label: "HBM Read MBU(%)", color: "#a855f7", points: sub.map(p => ({{ x: p[xKey], y: p.mbuRd }})) }},
    );
  }}

  document.getElementById("latTitle").textContent = `Latency Scaling (us): ${{selOp.value || ""}}`;
  document.getElementById("utilTitle").textContent = "Hardware Efficiency Scaling (%): MFU & HBM Bandwidth Utilization";
  setLegend(document.getElementById("latLegend"), latSeries);
  setLegend(document.getElementById("utilLegend"), utilSeries);
  drawMultiSeriesChart(document.getElementById("svgLatency"), latSeries, xLabel, "Duration (us)");
  drawMultiSeriesChart(document.getElementById("svgUtil"), utilSeries, xLabel, "Utilization (%)");

  setTableHeader(["Parallel Config", "Schedule Cliff", "Seq Len", "Batch", "Theoretical(us)", "Predicted(us)", "msprof P50(us)", "msprof P99(us)", "MFU(%)", "HBM Util(%)", "HBM Read(%)"]);
  document.getElementById("detailTableTitle").textContent = `Model Operator Points: ${{selModel.value}} / ${{selOp.value}}`;
  document.getElementById("detailTableCount").textContent = `${{tableRows.length}} points`;
  const tbody = document.getElementById("tblBody");
  tbody.replaceChildren();
  for (const p of tableRows) {{
    const tr = document.createElement("tr");
    const cells = [
      p.arm, p.cliff, p.seq, p.batch, p.theo.toFixed(3), p.pred.toFixed(3),
      p.p50 === null ? "-" : p.p50.toFixed(3), p.p99 === null ? "-" : p.p99.toFixed(3),
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

function filteredCatalogIndex() {{
  const q = (opSearch.value || "").trim().toLowerCase();
  if (!q) return OP_INDEX;
  return OP_INDEX.filter(item =>
    item.opType.toLowerCase().includes(q) ||
    item.kernel.toLowerCase().includes(q) ||
    item.core.toLowerCase().includes(q) ||
    item.scope.toLowerCase().includes(q) ||
    item.bound.toLowerCase().includes(q)
  );
}}

function renderCatalogSummaryTable(items) {{
  document.getElementById("catSummaryCount").textContent = `${{items.length}} / ${{OP_INDEX.length}} canonical operators`;
  const tbody = document.getElementById("catSummaryBody");
  tbody.replaceChildren();
  for (const item of items) {{
    const tr = document.createElement("tr");
    tr.className = item.opKey === catOp.value ? "clickable selected-row" : "clickable";
    const cells = [
      item.opType, item.kernel, item.core, item.scope, item.seriesCount,
      item.pointCount, item.xAxis, item.minPred.toFixed(3), item.maxPred.toFixed(3),
      item.meanEff.toFixed(2), item.measCount, item.meanApe === null ? "-" : item.meanApe.toFixed(2), item.bound,
    ];
    for (const val of cells) {{
      const td = document.createElement("td");
      td.textContent = String(val);
      tr.appendChild(td);
    }}
    tr.addEventListener("click", () => {{
      catOp.value = item.opKey;
      refreshCatalogCascades();
    }});
    tbody.appendChild(tr);
  }}
}}

function refreshCatalogOperators() {{
  const items = filteredCatalogIndex();
  fillSel(catOp, items.map(it => ({{ value: it.opKey, label: `${{it.opType}} | ${{it.kernel}} (${{it.pointCount}} pts)` }})));
  refreshCatalogCascades();
}}

function refreshCatalogCascades() {{
  const opRows = OP_ROWS.filter(r => r.opKey === catOp.value);
  fillSel(catSku, uniq(opRows.map(r => r.sku)));
  const sRows = opRows.filter(r => r.sku === catSku.value);
  const phases = uniq(sRows.map(r => r.phase));
  const phaseOpts = phases.map(p => ({{ value: p, label: p }}));
  if (phases.length > 1) phaseOpts.unshift({{ value: ALL_TOKEN, label: "All Phases" }});
  fillSel(catPhase, phaseOpts);
  const pRows = catPhase.value === ALL_TOKEN ? sRows : sRows.filter(r => r.phase === catPhase.value);
  const maKeys = uniq(pRows.map(r => `${{r.model}} | ${{r.arm}}`));
  const maOpts = maKeys.map(k => ({{ value: k, label: k }}));
  if (maKeys.length > 1) maOpts.unshift({{ value: ALL_MODELS_TOKEN, label: "All Models Overlay" }});
  fillSel(catModelArm, maOpts);
  const maRows = catModelArm.value === ALL_MODELS_TOKEN ? pRows : pRows.filter(r => `${{r.model}} | ${{r.arm}}` === catModelArm.value);
  const slices = uniq(maRows.map(r => r.slice));
  const sliceOpts = slices.map(s => ({{ value: s, label: s }}));
  if (slices.length > 1) sliceOpts.unshift({{ value: ALL_TOKEN, label: "All Slices (Multi-Series)" }});
  fillSel(catSlice, sliceOpts);
  renderCatalogMode();
}}

function renderCatalogMode() {{
  renderCatalogSummaryTable(filteredCatalogIndex());
  let pts = OP_ROWS.filter(r => r.opKey === catOp.value && r.sku === catSku.value);
  if (catPhase.value && catPhase.value !== ALL_TOKEN) pts = pts.filter(r => r.phase === catPhase.value);
  if (catModelArm.value && catModelArm.value !== ALL_MODELS_TOKEN) pts = pts.filter(r => `${{r.model}} | ${{r.arm}}` === catModelArm.value);
  if (catSlice.value && catSlice.value !== ALL_TOKEN) pts = pts.filter(r => r.slice === catSlice.value);

  const groups = new Map();
  for (const p of pts) {{
    const gKey = `${{p.model}} | ${{p.arm}} | ${{p.phase}} | ${{p.slice}}`;
    if (!groups.has(gKey)) groups.set(gKey, []);
    groups.get(gKey).push(p);
  }}
  const xAxes = uniq(pts.map(p => p.xAxis));
  const xAxisTitle = xAxes.length ? xAxes.join(" / ") : "Tokens / Message Bytes";
  const latSeries = [];
  const utilSeries = [];

  if (groups.size === 1) {{
    const onlyPts = Array.from(groups.values())[0].sort((a, b) => a.xVal - b.xVal);
    latSeries.push(
      {{ label: "Theoretical Roofline(us)", color: "#38bdf8", dashed: true, points: onlyPts.map(p => ({{ x: p.xVal, y: p.theo }})) }},
      {{ label: "Predicted Task Duration(us)", color: "#f59e0b", points: onlyPts.map(p => ({{ x: p.xVal, y: p.pred }})) }},
      {{ label: "msprof P50(us)", color: "#22c55e", points: onlyPts.filter(p => p.p50 !== null).map(p => ({{ x: p.xVal, y: p.p50 }})) }},
      {{ label: "msprof P99(us)", color: "#ef4444", points: onlyPts.filter(p => p.p99 !== null).map(p => ({{ x: p.xVal, y: p.p99 }})) }},
    );
    utilSeries.push(
      {{ label: "Cube Utilization(%)", color: "#38bdf8", points: onlyPts.map(p => ({{ x: p.xVal, y: p.cubeUtil }})) }},
      {{ label: "Vector Utilization(%)", color: "#a855f7", points: onlyPts.map(p => ({{ x: p.xVal, y: p.vecUtil }})) }},
      {{ label: "Achieved Bandwidth(GB/s)", color: "#22c55e", points: onlyPts.map(p => ({{ x: p.xVal, y: p.bwGbS }})) }},
    );
  }} else {{
    let gIdx = 0;
    for (const [gKey, gPts] of groups.entries()) {{
      const col = PALETTE[gIdx % PALETTE.length];
      gIdx += 1;
      const sorted = [...gPts].sort((a, b) => a.xVal - b.xVal);
      latSeries.push({{ label: `${{gKey}} Pred(us)`, color: col, points: sorted.map(p => ({{ x: p.xVal, y: p.pred }})) }});
      latSeries.push({{ label: `${{gKey}} Roofline(us)`, color: col, dashed: true, width: 1.4, opacity: 0.55, radius: 2.0, points: sorted.map(p => ({{ x: p.xVal, y: p.theo }})) }});
      const meas = sorted.filter(p => p.p50 !== null);
      if (meas.length) latSeries.push({{ label: `${{gKey}} msprof P50(us)`, color: "#22c55e", dashed: true, width: 1.8, radius: 4.0, points: meas.map(p => ({{ x: p.xVal, y: p.p50 }})) }});
      utilSeries.push({{ label: `${{gKey}} Achieved BW(GB/s)`, color: col, points: sorted.map(p => ({{ x: p.xVal, y: p.bwGbS }})) }});
    }}
  }}

  document.getElementById("latTitle").textContent = `Operator Latency Scaling (us): ${{catOp.value || ""}}`;
  document.getElementById("utilTitle").textContent = groups.size === 1 ? "Operator Utilization (%) & Achieved Bandwidth (GB/s)" : "Achieved Bandwidth Scaling (GB/s) Across Overlay Series";
  setLegend(document.getElementById("latLegend"), latSeries);
  setLegend(document.getElementById("utilLegend"), utilSeries);
  drawMultiSeriesChart(document.getElementById("svgLatency"), latSeries, xAxisTitle, "Duration (us)");
  drawMultiSeriesChart(document.getElementById("svgUtil"), utilSeries, xAxisTitle, groups.size === 1 ? "Util (%) / BW (GB/s)" : "Bandwidth (GB/s)");

  setTableHeader(["Model", "Phase", "Parallel Config", "Fixed Slice", "X Axis", "X Value", "Theoretical(us)", "Predicted(us)", "msprof P50(us)", "msprof P99(us)", "Cube Util(%)", "Vector Util(%)", "Achieved BW(GB/s)", "Roofline Bound"]);
  document.getElementById("detailTableTitle").textContent = `Operator Curve Points: ${{catOp.value || ""}}`;
  const sortedAll = [...pts].sort((a, b) => a.model.localeCompare(b.model) || a.arm.localeCompare(b.arm) || a.slice.localeCompare(b.slice) || a.xVal - b.xVal);
  document.getElementById("detailTableCount").textContent = `${{sortedAll.length}} points`;
  const tbody = document.getElementById("tblBody");
  tbody.replaceChildren();
  for (const p of sortedAll) {{
    const tr = document.createElement("tr");
    const cells = [
      p.model, p.phase, p.arm, p.slice, p.xAxis, fmtX(p.xVal),
      p.theo.toFixed(3), p.pred.toFixed(3),
      p.p50 === null ? "-" : p.p50.toFixed(3), p.p99 === null ? "-" : p.p99.toFixed(3),
      p.cubeUtil.toFixed(2), p.vecUtil.toFixed(2), p.bwGbS.toFixed(2), p.bound,
    ];
    for (const val of cells) {{
      const td = document.createElement("td");
      td.textContent = String(val);
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
}}

function setMode(mode) {{
  activeMode = mode;
  btnModeModel.classList.toggle("active", mode === "model");
  btnModeCatalog.classList.toggle("active", mode === "catalog");
  modelControls.classList.toggle("hidden", mode !== "model");
  catalogControls.classList.toggle("hidden", mode !== "catalog");
  catalogSummarySection.classList.toggle("hidden", mode !== "catalog");
  if (mode === "model") refreshModelSelectors();
  else refreshCatalogOperators();
}}

btnModeModel.addEventListener("click", () => setMode("model"));
btnModeCatalog.addEventListener("click", () => setMode("catalog"));
[selModel, selSku, selPhase, selArm].forEach(el => el.addEventListener("change", refreshModelSelectors));
[selOp, selAxis].forEach(el => el.addEventListener("change", refreshModelFixedSlice));
selFixedSlice.addEventListener("change", renderModelMode);

opSearch.addEventListener("input", refreshCatalogOperators);
[catOp, catSku, catPhase, catModelArm].forEach(el => el.addEventListener("change", refreshCatalogCascades));
catSlice.addEventListener("change", renderCatalogMode);

setMode("model");
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
    verified_campaign: CampaignManifest | None = None
    if isinstance(spec_or_manifest, OperatorMatrixManifest):
        manifest = spec_or_manifest
        if campaign is not None:
            manifest, verified_campaign = enrich_manifest_with_campaign(manifest, campaign)
    else:
        manifest = build_operator_matrix(
            spec_or_manifest, campaign=campaign, layers=layers, feature_track=feature_track,
        )
        if campaign is not None:
            verified_campaign = load_campaign_manifest(campaign)
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

    fabric_measured: dict[str, tuple[float, float]] = {}
    if verified_campaign is not None:
        for c_rec in verified_campaign.cases:
            if c_rec.status == "ok" and c_rec.qualified:
                fabric_measured[c_rec.case_id] = (float(c_rec.p50_us), float(c_rec.p99_us))
    fabric_points = (
        build_fabric_operator_curve_rows(manifest.nominal_specs, measured_by_case_id=fabric_measured)
        if include_fabric else []
    )
    all_operator_curve_rows = _build_operator_scaling_curves(
        manifest.rows,
        manifest.nominal_specs,
        fabric_points=fabric_points,
        multi_stage=multi_stage,
    )
    operator_catalog_index = _build_operator_catalog_index(all_operator_curve_rows)

    rows_by_model: dict[str, list[OperatorMatrixRow]] = defaultdict(list)
    steps_by_model: dict[str, list[StepSummary]] = defaultdict(list)
    for row in manifest.rows:
        rows_by_model[row.model_id].append(row)
    for step in manifest.step_summaries:
        steps_by_model[step.model_id].append(step)
    model_ids = sorted(set(rows_by_model) | set(steps_by_model))

    # 1. Root README.md, manifest.json & hardware_nominal_specs.csv
    readme_p = out_root / "README.md"
    manifest_json_p = out_root / "manifest.json"
    hw_csv_p = out_root / "hardware_nominal_specs.csv"
    _write_atomic_text(
        readme_p,
        _render_readme_md(
            manifest,
            nominal_rows,
            unique_operator_count=len(operator_catalog_index),
            fabric_curve_point_count=len(fabric_points),
        ),
    )
    _write_atomic_text(
        manifest_json_p,
        _render_bundle_manifest_json(
            model_ids=model_ids,
            skus=sorted(manifest.nominal_specs),
            unique_operator_count=len(operator_catalog_index),
            unique_kernel_count=len(catalog_rows),
            fabric_curve_point_count=len(fabric_points),
            campaign_attached=(manifest.campaign_id is not None or verified_campaign is not None),
        ),
    )
    _write_csv_atomic(hw_csv_p, HW_NOMINAL_COLUMNS, nominal_rows)
    emitted_files.extend([readme_p, manifest_json_p, hw_csv_p])

    # 2. summary/Executive_Summary.xlsx & summary/model_step_summary.csv
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    step_csv_p = summary_dir / "model_step_summary.csv"
    exec_xlsx_p = summary_dir / "Executive_Summary.xlsx"
    _write_csv_atomic(step_csv_p, STEP_SUMMARY_COLUMNS, step_rows)
    _write_executive_summary_xlsx(
        exec_xlsx_p, nominal_rows, step_rows, catalog_rows, all_operator_curve_rows,
    )
    emitted_files.extend([step_csv_p, exec_xlsx_p])

    # 3. models/<model_id>/{Operator_Breakdown.xlsx, op_breakdown_matrix.csv, scaling_curves.csv}
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
        m_op_curve_rows = _build_operator_scaling_curves(
            m_rows, manifest.nominal_specs, multi_stage=multi_stage,
        )
        all_curve_rows.extend(m_curve_rows)
        op_csv_p = mdir / "op_breakdown_matrix.csv"
        sc_csv_p = mdir / "scaling_curves.csv"
        op_xlsx_p = mdir / "Operator_Breakdown.xlsx"
        _write_csv_atomic(op_csv_p, OUTWARD_OP_COLUMNS, m_op_csv_rows)
        _write_csv_atomic(sc_csv_p, SCALING_CURVE_COLUMNS, m_curve_rows)
        _write_model_operator_breakdown_xlsx(
            op_xlsx_p,
            nominal_rows,
            m_step_rows,
            m_rows,
            m_curve_rows,
            operator_curve_rows=m_op_curve_rows,
            multi_stage=multi_stage,
        )
        emitted_files.extend([op_csv_p, sc_csv_p, op_xlsx_p])

    # 4. operators/{unique_kernel_catalog.csv, operator_scaling_curves.csv, Operator_Scaling_Curves.xlsx}
    ops_dir = out_root / "operators"
    ops_dir.mkdir(parents=True, exist_ok=True)
    cat_csv_p = ops_dir / "unique_kernel_catalog.csv"
    op_curves_csv_p = ops_dir / "operator_scaling_curves.csv"
    op_curves_xlsx_p = ops_dir / "Operator_Scaling_Curves.xlsx"
    _write_csv_atomic(cat_csv_p, CATALOG_COLUMNS, catalog_rows)
    _write_csv_atomic(op_curves_csv_p, OPERATOR_SCALING_CURVE_COLUMNS, all_operator_curve_rows)
    _write_operator_scaling_curves_xlsx(
        op_curves_xlsx_p, operator_catalog_index, all_operator_curve_rows,
    )
    emitted_files.extend([cat_csv_p, op_curves_csv_p, op_curves_xlsx_p])

    # 5. charts/scaling_explorer.html
    charts_dir = out_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    html_p = charts_dir / "scaling_explorer.html"
    _write_atomic_text(
        html_p,
        _render_scaling_explorer_html(
            all_curve_rows, all_operator_curve_rows, operator_catalog_index,
        ),
    )
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
        unique_operator_count=len(operator_catalog_index),
        fabric_curve_point_count=len(fabric_points),
        operator_curves_xlsx_path=str(op_curves_xlsx_p),
        operator_curves_csv_path=str(op_curves_csv_p),
        manifest_json_path=str(manifest_json_p),
    )


__all__ = [
    "CATALOG_COLUMNS",
    "HW_NOMINAL_COLUMNS",
    "OPERATOR_CATALOG_INDEX_COLUMNS",
    "OPERATOR_CHART_INDEX_COLUMNS",
    "OPERATOR_SCALING_CURVE_COLUMNS",
    "OUTWARD_OP_COLUMNS",
    "SCALING_CURVE_COLUMNS",
    "STEP_SUMMARY_COLUMNS",
    "MatrixExportReceipt",
    "assert_zero_internal_jargon",
    "export_operator_matrix",
]
