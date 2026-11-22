"""Pragmatic productization exporters for ``mindie_sim`` (``msprof`` Trace & OOXML XLSX).

[Role] Expose the single cohesive export domain package providing:
  (1) ``build_export_bundle`` (shared ``runtime.Engine`` evaluation, Lean-4
      certified ``L*`` physical lower bound & Gap Radar, 1-to-N ``kernel_details.csv``
      reconciliation, and closed-form Memory/Quant/PP ledger),
  (2) ``export_msprof_trace`` / ``iter_msprof_trace_events`` (1:1 streaming
      Ascend ``msprof`` ``trace_view.json`` for MindStudio Insight ``msinsight``),
  (3) ``export_comparison_xlsx`` (zero-dependency OOXML ``.xlsx`` workbook with
      real ``merge_cells``, 1-to-N kernel reconciliation, and Memory/Quant/PP
      calculator).
[Invariants] Strictly a top-layer consumer in the unidirectional import DAG;
  lower layers (``schema``, ``catalog``, ``prediction``, ``optimization``) never
  import ``mindie_sim.export``. Zero ``0.80x`` heuristic fudge factors and zero
  third-party OOXML dependencies.
[Timeouts & Locks] Delegates bounded execution and streaming writes to submodules;
  no locks or device contexts.
[Side-Effect Policy] Package import has zero filesystem or runtime side effects.
"""
from __future__ import annotations

from .analysis import (
    CacheLedgerRow,
    DEFAULT_GAP_RADAR_TOP_K,
    EXPORT_BUNDLE_SCHEMA,
    ExportBundle,
    GIB_BYTES,
    GapRadarRow,
    KernelObservationRow,
    MIB_BYTES,
    MXFP_GROUP_SIZE,
    MemoryCalculatorLedger,
    OperatorExportRow,
    RECONCILE_ABS_FLOOR_US,
    RECONCILE_APE_THRESHOLD_PCT,
    ReconciledKernelGroup,
    ReconciliationSummary,
    SUPPORTED_CACHE_DTYPES,
    SUPPORTED_WEIGHT_QUANTS,
    StageBudgetRow,
    US_PER_MS,
    WEIGHT_QUANT_FORMATS,
    WeightLedgerRow,
    build_export_bundle,
    classify_submodule,
    compute_memory_quant_ledger,
    evaluate_weight_storage_bytes,
    extract_layer_index,
    load_kernel_details_csv,
    load_observation_bundle,
    reconcile_operators_with_kernels,
)
from .trace_exporter import (
    MSPROF_PROCESS_SPEC,
    TRACE_EXPORT_RECEIPT_SCHEMA,
    TraceExportSummary,
    export_msprof_trace,
    iter_msprof_trace_events,
)
from .xlsx_exporter import (
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_SUMMARY,
    Workbook,
    Worksheet,
    XLSX_EXPORT_RECEIPT_SCHEMA,
    XlsxExportSummary,
    cell_ref,
    column_letter,
    export_comparison_xlsx,
)

__all__ = [
    "CacheLedgerRow",
    "DEFAULT_GAP_RADAR_TOP_K",
    "EXPORT_BUNDLE_SCHEMA",
    "ExportBundle",
    "GIB_BYTES",
    "GapRadarRow",
    "KernelObservationRow",
    "MIB_BYTES",
    "MSPROF_PROCESS_SPEC",
    "MXFP_GROUP_SIZE",
    "MemoryCalculatorLedger",
    "OperatorExportRow",
    "RECONCILE_ABS_FLOOR_US",
    "RECONCILE_APE_THRESHOLD_PCT",
    "ReconciledKernelGroup",
    "ReconciliationSummary",
    "SHEET_BREAKDOWN",
    "SHEET_MEMORY",
    "SHEET_SUMMARY",
    "SUPPORTED_CACHE_DTYPES",
    "SUPPORTED_WEIGHT_QUANTS",
    "StageBudgetRow",
    "TRACE_EXPORT_RECEIPT_SCHEMA",
    "TraceExportSummary",
    "US_PER_MS",
    "WEIGHT_QUANT_FORMATS",
    "WeightLedgerRow",
    "Workbook",
    "Worksheet",
    "XLSX_EXPORT_RECEIPT_SCHEMA",
    "XlsxExportSummary",
    "build_export_bundle",
    "cell_ref",
    "classify_submodule",
    "column_letter",
    "compute_memory_quant_ledger",
    "evaluate_weight_storage_bytes",
    "export_comparison_xlsx",
    "export_msprof_trace",
    "extract_layer_index",
    "iter_msprof_trace_events",
    "load_kernel_details_csv",
    "load_observation_bundle",
    "reconcile_operators_with_kernels",
]
