"""Zero-dependency OOXML ``.xlsx`` exporter with Before-vs-After ``merge_cells`` & New-Kernel Backlog.

[Role] Generate executive-ready ECMA-376 OOXML ``.xlsx`` workbooks directly via
  Python's stdlib ``zipfile`` + XML serializer (zero third-party runtime deps)
  with real ``worksheet.merge_cells(...)`` (``<mergeCells><mergeCell .../>``),
  frozen panes, data validation dropdowns, live Excel formulas + cached values,
  and human-native Ascend optimization sheets:
  1. ``Summary`` (Whole-net KPIs, Before-vs-After delta, Gap Radar),
  2. ``Layer_Operator_Breakdown`` (Archetype ×N grouping, side-by-side Before vs
     After with N->1 ``merge_cells`` fusion, MSO Stream-1 overlap formulas,
     Source Refs, Backend/Status, and Before/After Bounds),
  3. ``Memory_Quant_Calculator`` (closed-form per-die Weight/Quant/Cache/PP HBM
     calculator honoring per-row ``QuantSpec.exclusions``),
  4. ``New_Kernel_Design_Backlog`` (pre-filled objective hardware specs +
     Agent-fillable design & acceptance columns),
  5. Optional ``Measured_msprof_Calibration`` (ONLY when ``observations_path``
     is supplied — zero dead ``-`` measured columns in Before-vs-After mode!).
[Invariants] Every timing value comes strictly from ``runtime.Engine`` via
  ``ExportBundle`` (zero ``0.80x`` heuristic fudge factors and zero banned
  internal Python recipe class names). Merged cell ranges are validated
  non-overlapping and store numeric/formula operands exclusively in the top-left
  anchor cell so Excel ``SUM`` formulas never double-count. Every ``<worksheet>``
  follows strict ECMA-376 child element order. Atomic writes use same-directory
  temp files + ``os.replace``.
[Timeouts & Locks] Bounded O(Archetypes) in-memory XML serialization; no locks,
  network, or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` output path and
  cleans up private temp files on error.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from xml.sax.saxutils import escape as xml_escape
import zipfile

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest
from .analysis import (
    AlignedComparisonRow,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    ExportBundle,
    FP32_BYTES,
    GIB_BYTES,
    KV_QUANT_GROUP_SIZE,
    MIB_BYTES,
    MemoryCalculatorLedger,
    NewKernelBacklogItem,
    OperatorExportRow,
    SUPPORTED_CACHE_DTYPES,
    SUPPORTED_WEIGHT_QUANTS,
    US_PER_MS,
    WEIGHT_QUANT_FORMATS,
    align_before_after_operators,
    build_export_bundle,
    build_new_kernel_backlog,
    compose_export_case,
)

#: Schema tag for XLSX export receipts.
XLSX_EXPORT_RECEIPT_SCHEMA = "mindie-sim.xlsx-export-receipt/2"
#: OOXML SpreadsheetML and Relationships XML namespaces.
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"

#: Excel column alphabet radix and ASCII anchor.
EXCEL_COL_RADIX = 26
ASCII_UPPER_A = 65
#: Default row height (pt), wrapped data row height (pt), header height (pt), and Excel calc ID.
DEFAULT_ROW_HEIGHT_PT = 19.0
HEADER_ROW_HEIGHT_PT = 28.0
WRAP_ROW_HEIGHT_PT = 30.0
BACKLOG_ROW_HEIGHT_PT = 44.0
EXCEL_CALC_ID = 191029
#: Page margin constants (inches).
PAGE_MARGIN_SIDE = 0.65
PAGE_MARGIN_TOP_BOTTOM = 0.75
PAGE_MARGIN_HEADER_FOOTER = 0.3
#: Column width bounds in character units.
MIN_COL_WIDTH = 8.0
MAX_COL_WIDTH = 56.0
#: Gap-ratio visual severity thresholds for Gap Radar highlighting.
GAP_RATIO_SEVERE = 3.0
GAP_RATIO_MODERATE = 1.5
#: Cache bytes/elem approximations matching closed-form Excel formulas.
C8_CACHE_FACTOR = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
#: Percentage divisor for Excel percentage number format (0.0%).
PCT_DIVISOR = 100.0
#: MiB per GiB divisor for live Excel formulas.
MIB_PER_GIB = 1024
#: Regex matching illegal XML 1.0 control characters.
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")

#: Style indices registered in ``xl/styles.xml`` (must match ``_styles_xml``).
STYLE_DEFAULT = 0
STYLE_TITLE = 1
STYLE_SECTION = 2
STYLE_HEADER = 3
STYLE_TEXT = 4
STYLE_MERGED = 5
STYLE_INT = 6
STYLE_FLOAT2 = 7
STYLE_FLOAT3 = 8
STYLE_PCT = 9
STYLE_RATIO = 10
STYLE_INPUT_NUM = 11
STYLE_INPUT_TEXT = 12
STYLE_TOTAL_TEXT = 13
STYLE_TOTAL_NUM = 14
STYLE_TOTAL_PCT = 15
STYLE_ALERT_HIGH = 16
STYLE_ALERT_MID = 17
STYLE_ALERT_OK = 18
STYLE_TEXT_WRAP = 19
STYLE_INPUT_TEXT_WRAP = 20
STYLE_NEW_KERNEL = 21
STYLE_SUBTOTAL_TEXT = 22
STYLE_SUBTOTAL_NUM = 23
STYLE_SUBTOTAL_RATIO = 24
STYLE_OVERLAP = 25
STYLE_TOTAL_FLOAT3 = 26

#: Canonical sheet names of the human-native Ascend optimization workbook.
SHEET_SUMMARY = "Summary"
SHEET_BREAKDOWN = "Layer_Operator_Breakdown"
SHEET_MEMORY = "Memory_Quant_Calculator"
SHEET_NEW_KERNELS = "New_Kernel_Design_Backlog"
SHEET_MEASURED = "Measured_msprof_Calibration"

#: Sheet 1 (Summary) structural layout constants.
SUMMARY_COLS = 12
SUMMARY_SECTION_ROW = 4
SUMMARY_HEADER_ROW = 5
SUMMARY_DATA_START_ROW = 6
SUMMARY_LEFT_END_COL = 5
SUMMARY_RIGHT_START_COL = 6
SUMMARY_BASIS_START_COL = 4
SUMMARY_META_PAR_START_COL = 4
SUMMARY_META_PAR_END_COL = 6
SUMMARY_META_CTX_COL = 7
SUMMARY_META_GRAPH_START_COL = 10
SUMMARY_GRAPH_PREVIEW_LEN = 16
SUMMARY_FREEZE_ROWS = 2

#: Sheet 2 Before-vs-After mode column indices (21 columns A..U).
CMP_COLS = 21
CMP_DATA_START_ROW = 3
CMP_FREEZE_ROWS = 2
CMP_FREEZE_COLS = 4
CMP_COL_ARCH = 1
CMP_COL_REP = 2
CMP_COL_SUB = 3
CMP_COL_ROLE = 4
CMP_COL_SHAPE = 5
CMP_COL_BEF_KERN = 6
CMP_COL_BEF_SRC = 7
CMP_COL_BEF_US = 8
CMP_COL_BEF_MS = 9
CMP_COL_BEF_BOUND = 10
CMP_COL_AFT_KERN = 11
CMP_COL_STATUS = 12
CMP_COL_AFT_US = 13
CMP_COL_STREAM = 14
CMP_COL_AFT_MS = 15
CMP_COL_SAVED_MS = 16
CMP_COL_SPEEDUP = 17
CMP_COL_AFT_BOUND = 18
CMP_COL_LSTAR_MS = 19
CMP_COL_GAP_MS = 20
CMP_COL_ACTION = 21
CMP_AFTER_MERGE_COLS = (
    CMP_COL_AFT_KERN, CMP_COL_STATUS, CMP_COL_AFT_US, CMP_COL_STREAM,
    CMP_COL_AFT_MS, CMP_COL_SAVED_MS, CMP_COL_SPEEDUP, CMP_COL_AFT_BOUND,
    CMP_COL_LSTAR_MS, CMP_COL_GAP_MS, CMP_COL_ACTION,
)
CMP_SUBTOTAL_LABEL_END_COL = 8

#: Sheet 2 Standalone mode column indices (15 columns A..O).
STD_COLS = 15
STD_DATA_START_ROW = 3
STD_COL_ARCH = 1
STD_COL_REP = 2
STD_COL_SUB = 3
STD_COL_ROLE = 4
STD_COL_SHAPE = 5
STD_COL_KERN = 6
STD_COL_STATUS = 7
STD_COL_SRC = 8
STD_COL_STREAM = 9
STD_COL_US = 10
STD_COL_MS = 11
STD_COL_LSTAR_US = 12
STD_COL_LSTAR_MS = 13
STD_COL_GAP_MS = 14
STD_COL_BOUND = 15
STD_SUBTOTAL_LABEL_END_COL = 10

#: Sheet 3 (Memory_Quant_Calculator) structural layout constants.
MEMORY_COLS = 22
MEMORY_TABLE_END_COL = 14
MEMORY_WEIGHT_MERGE_END_COL = 9
MEMORY_WEIGHT_SUM_START_COL = 10
MEMORY_CACHE_BASIS_START_COL = 7
MEMORY_CACHE_MERGE_END_COL = 4
MEMORY_CACHE_REQ_COL = 5
MEMORY_CACHE_BATCH_COL = 6
MEMORY_STAGE_HEADROOM_COL = 12
MEMORY_STAGE_STATUS_COL = 13
MEMORY_STAGE_RAW_ENGRAM_COL = 14
MEMORY_STAGE_AGG_START_COL = 3
MEMORY_CONTROL_COL_LABEL = 17
MEMORY_CONTROL_COL_VALUE = 18
MEMORY_LOOKUP_START_COL = 19
MEMORY_LOOKUP_HEADER_ROW = 3
MEMORY_LOOKUP_FIRST_ROW = 4
MEMORY_LOOKUP_LAST_ROW = 10
MEMORY_TABLE_HEADER_ROW = 3
MEMORY_FREEZE_ROWS = 3
MEMORY_QUANT_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "W8A8", "W4A8", "MXFP8", "MXFP4")

#: Sheet 4 (New_Kernel_Design_Backlog) structural layout constants (10 columns A..J).
BACKLOG_COLS = 10
BACKLOG_HEADER_ROW = 3
BACKLOG_DATA_START_ROW = 4
BACKLOG_COL_GAIN = 7
BACKLOG_LABEL_MERGE_END = 6

#: Conditional Measured_msprof_Calibration sheet layout constants (14 columns A..N).
MEAS_COLS = 14
MEAS_DATA_START_ROW = 3
MEAS_SIM_MERGE_COLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14)
MEAS_COL_SIM_US = 9
MEAS_COL_KERN_NAME = 10
MEAS_COL_KERN_US = 11
MEAS_COL_SUM_US = 12
MEAS_COL_APE = 13
MEAS_COL_BOUND = 14
MEAS_TOTAL_SUM_COLS = (8, 9, 11, 12)

#: Preset column widths across sheets.
SUMMARY_COL_WIDTHS = (8.0, 26.0, 8.0, 22.0, 26.0, 26.0, 24.0, 20.0, 14.0, 14.0, 14.0, 12.0)
CMP_COL_WIDTHS = (
    32.0, 10.0, 24.0, 28.0, 34.0, 28.0, 30.0, 13.0, 14.0, 22.0,
    28.0, 28.0, 13.0, 24.0, 15.0, 13.0, 11.0, 22.0, 14.0, 13.0, 40.0,
)
STD_COL_WIDTHS = (
    32.0, 10.0, 24.0, 28.0, 34.0, 28.0, 28.0, 30.0, 24.0, 13.0,
    15.0, 13.0, 14.0, 13.0, 22.0,
)
MEMORY_COL_WIDTHS = (
    16.0, 24.0, 12.0, 12.0, 18.0, 12.0, 12.0, 12.0, 13.0, 15.0, 15.0,
    13.0, 14.0, 28.0, 4.0, 4.0, 26.0, 30.0, 12.0, 12.0, 11.0, 12.0,
)
BACKLOG_COL_WIDTHS = (10.0, 30.0, 26.0, 28.0, 30.0, 24.0, 15.0, 38.0, 46.0, 36.0)
MEAS_COL_WIDTHS = (
    28.0, 22.0, 24.0, 24.0, 24.0, 28.0, 12.0, 14.0, 14.0, 28.0, 14.0, 14.0, 12.0, 20.0,
)


@dataclass(frozen=True, slots=True)
class XlsxExportSummary:
    """Compact receipt returned after writing an ``.xlsx`` workbook."""

    schema: str
    output_path: str
    model_name: str
    hardware_sku: str
    phase: str
    graph_id: str
    sheets: tuple[str, ...]
    merge_cell_counts: dict[str, int]
    operator_rows: int
    weight_rows: int
    has_comparison: bool
    has_measured: bool
    bytes_written: int
    new_kernel_backlog_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_path": self.output_path,
            "model_name": self.model_name,
            "hardware_sku": self.hardware_sku,
            "phase": self.phase,
            "graph_id": self.graph_id,
            "sheets": list(self.sheets),
            "merge_cell_counts": dict(self.merge_cell_counts),
            "operator_rows": self.operator_rows,
            "weight_rows": self.weight_rows,
            "has_comparison": self.has_comparison,
            "has_measured": self.has_measured,
            "bytes_written": self.bytes_written,
            "new_kernel_backlog_count": self.new_kernel_backlog_count,
        }


@dataclass(slots=True)
class _CellRecord:
    value: Any = None
    formula: str | None = None
    style: int = STYLE_DEFAULT


@dataclass(frozen=True, slots=True)
class _DataValidation:
    sqref: str
    val_type: str
    formula1: str
    formula2: str | None = None


def column_letter(col_index: int) -> str:
    """Convert 1-based Excel column index to letters (``1 -> 'A'``, ``27 -> 'AA'``)."""
    if col_index < 1:
        raise ValueError(f"invalid_excel_col_index:{col_index}")
    chars: list[str] = []
    n = int(col_index)
    while n > 0:
        n, rem = divmod(n - 1, EXCEL_COL_RADIX)
        chars.append(chr(ASCII_UPPER_A + rem))
    return "".join(reversed(chars))


def cell_ref(row: int, col: int) -> str:
    """Format 1-based ``(row, col)`` as an A1 cell reference."""
    return f"{column_letter(col)}{int(row)}"


def _parse_cell_ref(ref: str) -> tuple[int, int]:
    match = _CELL_REF_RE.match(ref.strip().upper())
    if match is None:
        raise ValueError(f"invalid_cell_ref:{ref!r}")
    col_str, row_str = match.group(1), match.group(2)
    col = 0
    for ch in col_str:
        col = col * EXCEL_COL_RADIX + (ord(ch) - ASCII_UPPER_A + 1)
    return int(row_str), col


def _sanitize_xml_text(text: str) -> str:
    return _ILLEGAL_XML_RE.sub("", text)


class Worksheet:
    """One ECMA-376 OOXML worksheet with ``merge_cells``, formulas, and validations."""

    def __init__(self, title: str) -> None:
        self.title = str(title)
        self._cells: dict[tuple[int, int], _CellRecord] = {}
        self._merges: list[tuple[int, int, int, int, str]] = []
        self._validations: list[_DataValidation] = []
        self._col_widths: dict[int, float] = {}
        self._row_heights: dict[int, float] = {}
        self._freeze: tuple[int, int] | None = None

    @property
    def merged_ranges(self) -> tuple[str, ...]:
        return tuple(item[4] for item in self._merges)

    def set_col_width(self, col: int, width: float) -> None:
        self._col_widths[int(col)] = max(MIN_COL_WIDTH, min(MAX_COL_WIDTH, float(width)))

    def set_row_height(self, row: int, height: float) -> None:
        self._row_heights[int(row)] = float(height)

    def freeze_panes(self, row: int, col: int = 0) -> None:
        self._freeze = (max(0, int(row)), max(0, int(col)))

    def set_cell(
        self,
        row: int,
        col: int,
        value: Any = None,
        *,
        formula: str | None = None,
        style: int = STYLE_DEFAULT,
    ) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non_finite_xlsx_cell_value:{row}:{col}:{value!r}")
        clean_formula = None
        if formula is not None:
            clean_formula = formula[1:] if formula.startswith("=") else formula
        self._cells[(int(row), int(col))] = _CellRecord(
            value=value, formula=clean_formula, style=int(style),
        )

    def merge_cells(self, cell_range: str) -> None:
        """Register a validated non-overlapping OOXML ``<mergeCell ref="..."/>``."""
        parts = cell_range.strip().upper().split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid_merge_range:{cell_range!r}")
        r1, c1 = _parse_cell_ref(parts[0])
        r2, c2 = _parse_cell_ref(parts[1])
        if r1 > r2 or c1 > c2:
            raise ValueError(f"inverted_merge_range:{cell_range!r}")
        if r1 == r2 and c1 == c2:
            return
        for mr1, mc1, mr2, mc2, existing in self._merges:
            if not (r2 < mr1 or r1 > mr2 or c2 < mc1 or c1 > mc2):
                raise ValueError(f"overlapping_merge_range:{cell_range}:{existing}")
        canonical = f"{cell_ref(r1, c1)}:{cell_ref(r2, c2)}"
        self._merges.append((r1, c1, r2, c2, canonical))
        anchor = self._cells.get((r1, c1))
        anchor_style = anchor.style if anchor is not None else STYLE_MERGED
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) != (r1, c1) and (r, c) not in self._cells:
                    self._cells[(r, c)] = _CellRecord(value=None, style=anchor_style)

    def merge_range(self, r1: int, c1: int, r2: int, c2: int) -> None:
        if r1 == r2 and c1 == c2:
            return
        self.merge_cells(f"{cell_ref(r1, c1)}:{cell_ref(r2, c2)}")

    def add_data_validation(
        self,
        sqref: str,
        *,
        val_type: str = "list",
        formula1: str,
        formula2: str | None = None,
    ) -> None:
        self._validations.append(_DataValidation(sqref, val_type, formula1, formula2))

    def _dimension_ref(self) -> str:
        if not self._cells:
            return "A1:A1"
        max_r = max(r for r, _ in self._cells)
        max_c = max(c for _, c in self._cells)
        return f"A1:{cell_ref(max_r, max_c)}"

    def to_xml(self, shared_strings: _SharedStringTable, *, tab_selected: bool = False) -> str:
        parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            f'<worksheet xmlns="{NS_MAIN}" xmlns:r="{NS_REL_DOC}">',
            f'<dimension ref="{self._dimension_ref()}"/>',
            self._sheet_views_xml(tab_selected=tab_selected),
            f'<sheetFormatPr defaultRowHeight="{DEFAULT_ROW_HEIGHT_PT:.1f}"/>',
        ]
        cols_xml = self._cols_xml()
        if cols_xml:
            parts.append(cols_xml)
        parts.append(self._sheet_data_xml(shared_strings))
        if self._merges:
            m_tags = "".join(f'<mergeCell ref="{ref}"/>' for *_, ref in self._merges)
            parts.append(f'<mergeCells count="{len(self._merges)}">{m_tags}</mergeCells>')
        if self._validations:
            parts.append(self._validations_xml())
        parts.append(
            f'<pageMargins left="{PAGE_MARGIN_SIDE}" right="{PAGE_MARGIN_SIDE}" '
            f'top="{PAGE_MARGIN_TOP_BOTTOM}" bottom="{PAGE_MARGIN_TOP_BOTTOM}" '
            f'header="{PAGE_MARGIN_HEADER_FOOTER}" footer="{PAGE_MARGIN_HEADER_FOOTER}"/>'
        )
        parts.append("</worksheet>")
        return "".join(parts)

    def _sheet_views_xml(self, *, tab_selected: bool) -> str:
        sel = ' tabSelected="1"' if tab_selected else ""
        if self._freeze is None or self._freeze == (0, 0):
            return f'<sheetViews><sheetView{sel} showGridLines="1" workbookViewId="0"/></sheetViews>'
        y_split, x_split = self._freeze
        top_left = cell_ref(y_split + 1, x_split + 1)
        pane_attrs = []
        if x_split > 0:
            pane_attrs.append(f'xSplit="{x_split}"')
        if y_split > 0:
            pane_attrs.append(f'ySplit="{y_split}"')
        active = "bottomRight" if (x_split > 0 and y_split > 0) else ("bottomLeft" if y_split > 0 else "topRight")
        pane_attrs.extend([f'topLeftCell="{top_left}"', f'activePane="{active}"', 'state="frozen"'])
        return (
            f'<sheetViews><sheetView{sel} showGridLines="1" workbookViewId="0">'
            f'<pane {" ".join(pane_attrs)}/></sheetView></sheetViews>'
        )

    def _cols_xml(self) -> str:
        if not self._col_widths:
            return ""
        tags = [
            f'<col min="{col}" max="{col}" width="{width:.2f}" customWidth="1"/>'
            for col, width in sorted(self._col_widths.items())
        ]
        return f'<cols>{"".join(tags)}</cols>'

    def _sheet_data_xml(self, shared_strings: _SharedStringTable) -> str:
        rows_map: dict[int, list[tuple[int, _CellRecord]]] = {}
        for (r, c), rec in self._cells.items():
            rows_map.setdefault(r, []).append((c, rec))
        row_chunks = ["<sheetData>"]
        for r in sorted(rows_map):
            ht = self._row_heights.get(r)
            ht_attr = f' ht="{ht:.1f}" customHeight="1"' if ht is not None else ""
            row_chunks.append(f'<row r="{r}"{ht_attr}>')
            for c, rec in sorted(rows_map[r], key=lambda item: item[0]):
                row_chunks.append(_cell_to_xml(r, c, rec, shared_strings))
            row_chunks.append("</row>")
        row_chunks.append("</sheetData>")
        return "".join(row_chunks)

    def _validations_xml(self) -> str:
        tags = [f'<dataValidations count="{len(self._validations)}">']
        for dv in self._validations:
            f1 = xml_escape(_sanitize_xml_text(dv.formula1))
            f2_xml = (
                f"<formula2>{xml_escape(_sanitize_xml_text(dv.formula2))}</formula2>"
                if dv.formula2 is not None else ""
            )
            tags.append(
                f'<dataValidation type="{dv.val_type}" allowBlank="1" showDropDown="0" '
                f'sqref="{dv.sqref}"><formula1>{f1}</formula1>{f2_xml}</dataValidation>'
            )
        tags.append("</dataValidations>")
        return "".join(tags)


class _SharedStringTable:
    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._items: list[str] = []

    def add(self, text: str) -> int:
        clean = _sanitize_xml_text(text)
        idx = self._index.get(clean)
        if idx is None:
            idx = len(self._items)
            self._index[clean] = idx
            self._items.append(clean)
        return idx

    def to_xml(self) -> str:
        count = len(self._items)
        items_xml = "".join(
            f'<si><t xml:space="preserve">{xml_escape(s)}</t></si>' for s in self._items
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<sst xmlns="{NS_MAIN}" count="{count}" uniqueCount="{count}">{items_xml}</sst>'
        )


def _cell_to_xml(row: int, col: int, rec: _CellRecord, sst: _SharedStringTable) -> str:
    ref = cell_ref(row, col)
    s_attr = f' s="{rec.style}"' if rec.style else ""
    if rec.formula is not None:
        f_xml = f"<f>{xml_escape(_sanitize_xml_text(rec.formula))}</f>"
        if rec.value is None:
            return f'<c r="{ref}"{s_attr}>{f_xml}</c>'
        if isinstance(rec.value, bool):
            return f'<c r="{ref}" t="b"{s_attr}>{f_xml}<v>{1 if rec.value else 0}</v></c>'
        if isinstance(rec.value, (int, float)):
            return f'<c r="{ref}"{s_attr}>{f_xml}<v>{rec.value}</v></c>'
        v_str = xml_escape(_sanitize_xml_text(str(rec.value)))
        return f'<c r="{ref}" t="str"{s_attr}>{f_xml}<v>{v_str}</v></c>'
    if rec.value is None:
        return f'<c r="{ref}"{s_attr}/>'
    if isinstance(rec.value, bool):
        return f'<c r="{ref}" t="b"{s_attr}><v>{1 if rec.value else 0}</v></c>'
    if isinstance(rec.value, (int, float)):
        return f'<c r="{ref}"{s_attr}><v>{rec.value}</v></c>'
    idx = sst.add(str(rec.value))
    return f'<c r="{ref}" t="s"{s_attr}><v>{idx}</v></c>'


class Workbook:
    """Zero-dependency OOXML ``.xlsx`` workbook builder and atomic ZIP serializer."""

    def __init__(self) -> None:
        self.worksheets: list[Worksheet] = []

    def create_sheet(self, title: str) -> Worksheet:
        sheet = Worksheet(title)
        self.worksheets.append(sheet)
        return sheet

    def save(self, output_path: str | Path) -> int:
        if not self.worksheets:
            raise ValueError("workbook_has_no_worksheets")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sst = _SharedStringTable()
        sheet_xmls = [
            ws.to_xml(sst, tab_selected=(idx == 0))
            for idx, ws in enumerate(self.worksheets)
        ]
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("[Content_Types].xml", _content_types_xml(len(self.worksheets)))
                zf.writestr("_rels/.rels", _root_rels_xml())
                zf.writestr("xl/workbook.xml", _workbook_xml(self.worksheets))
                zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(self.worksheets)))
                zf.writestr("xl/styles.xml", _styles_xml())
                zf.writestr("xl/sharedStrings.xml", sst.to_xml())
                for idx, xml_text in enumerate(sheet_xmls, start=1):
                    zf.writestr(f"xl/worksheets/sheet{idx}.xml", xml_text)
            size = int(tmp_path.stat().st_size)
            os.replace(tmp_path, target)
            return size
        except TimeoutError:
            _unlink_quiet(tmp_path)
            raise
        except Exception:
            _unlink_quiet(tmp_path)
            raise


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TimeoutError:
        raise
    except OSError:
        pass


def _content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{NS_CONTENT_TYPES}">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}</Types>'
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{NS_REL_PKG}">'
        f'<Relationship Id="rId1" Type="{NS_REL_DOC}/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheets: Sequence[Worksheet]) -> str:
    entries = "".join(
        f'<sheet name="{xml_escape(ws.title)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, ws in enumerate(sheets, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{NS_MAIN}" xmlns:r="{NS_REL_DOC}">'
        f"<sheets>{entries}</sheets>"
        f'<calcPr calcId="{EXCEL_CALC_ID}" fullCalcOnLoad="1"/>'
        "</workbook>"
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = [
        f'<Relationship Id="rId{idx}" Type="{NS_REL_DOC}/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    ]
    style_id = sheet_count + 1
    sst_id = sheet_count + 2
    rels.append(f'<Relationship Id="rId{style_id}" Type="{NS_REL_DOC}/styles" Target="styles.xml"/>')
    rels.append(f'<Relationship Id="rId{sst_id}" Type="{NS_REL_DOC}/sharedStrings" Target="sharedStrings.xml"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{NS_REL_PKG}">{"".join(rels)}</Relationships>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{NS_MAIN}">'
        '<numFmts count="5">'
        '<numFmt numFmtId="164" formatCode="#,##0"/>'
        '<numFmt numFmtId="165" formatCode="#,##0.00"/>'
        '<numFmt numFmtId="166" formatCode="#,##0.000"/>'
        '<numFmt numFmtId="167" formatCode="0.0%"/>'
        '<numFmt numFmtId="168" formatCode="0.00&quot;x&quot;"/>'
        "</numFmts>"
        '<fonts count="7">'
        '<font><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><sz val="12"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1F2937"/><name val="Arial"/></font>'
        '<font><sz val="10"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF9C3412"/><name val="Arial"/></font>'
        "</fonts>"
        '<fills count="11">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9E1F2"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF2F5597"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFF2F5FA"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFE9EDF4"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFFCE4D6"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFE2EFDA"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFE8EEF5"/></patternFill></fill>'
        "</fills>"
        '<borders count="3">'
        "<border><left/><right/><top/><bottom/><diagonal/></border>"
        '<border>'
        '<left style="thin"><color rgb="FFD9D9D9"/></left>'
        '<right style="thin"><color rgb="FFD9D9D9"/></right>'
        '<top style="thin"><color rgb="FFD9D9D9"/></top>'
        '<bottom style="thin"><color rgb="FFD9D9D9"/></bottom>'
        "<diagonal/></border>"
        '<border>'
        '<left style="thin"><color rgb="FFA6A6A6"/></left>'
        '<right style="thin"><color rgb="FFA6A6A6"/></right>'
        '<top style="thin"><color rgb="FF44546A"/></top>'
        '<bottom style="double"><color rgb="FF44546A"/></bottom>'
        "<diagonal/></border>"
        "</borders>"
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="27">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="3" fillId="4" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="4" fillId="5" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="165" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="166" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="167" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="168" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="164" fontId="5" fillId="6" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="5" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="4" fillId="7" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>'
        '<xf numFmtId="165" fontId="4" fillId="7" borderId="2" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="167" fontId="4" fillId="7" borderId="2" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="168" fontId="4" fillId="8" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="168" fontId="4" fillId="6" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="168" fontId="4" fillId="9" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="5" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="6" fillId="8" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="4" fillId="10" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>'
        '<xf numFmtId="166" fontId="4" fillId="10" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="168" fontId="4" fillId="10" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="9" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="166" fontId="4" fillId="7" borderId="2" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def _apply_col_widths(ws: Worksheet, widths: Sequence[float]) -> None:
    for idx, width in enumerate(widths, start=1):
        ws.set_col_width(idx, width)


def _write_banner(ws: Worksheet, row: int, col_end: int, text: str, style: int) -> None:
    ws.set_cell(row, 1, text, style=style)
    ws.merge_range(row, 1, row, col_end)


def _gap_ratio_style(ratio: float | None) -> int:
    if ratio is None:
        return STYLE_RATIO
    if ratio >= GAP_RATIO_SEVERE:
        return STYLE_ALERT_HIGH
    if ratio >= GAP_RATIO_MODERATE:
        return STYLE_ALERT_MID
    return STYLE_ALERT_OK


def _write_row(
    ws: Worksheet,
    row: int,
    entries: Sequence[tuple[Any, str | None, int]],
    *,
    start_col: int = 1,
) -> None:
    for offset, (val, formula, style) in enumerate(entries):
        ws.set_cell(row, start_col + offset, val, formula=formula, style=style)


def _write_headers(
    ws: Worksheet, row: int, labels: Sequence[str], *, start_col: int = 1,
) -> None:
    ws.set_row_height(row, HEADER_ROW_HEIGHT_PT)
    for offset, label in enumerate(labels):
        ws.set_cell(row, start_col + offset, label, style=STYLE_HEADER)


def _extend_span(spans: list[tuple[Any, int, int]], key: Any, r1: int, r2: int) -> None:
    if spans and spans[-1][0] == key:
        spans[-1] = (key, spans[-1][1], r2)
    else:
        spans.append((key, r1, r2))


def _stream_overlap_label(op: OperatorExportRow) -> str:
    if op.is_overlapped:
        return "Stream 1 (Overlapped, 0 exposed)"
    if op.stream_index == 1:
        return "Stream 1 (MSO Aux)"
    if op.stream_index == 2 or op.task_type == "HCCL":
        return "Stream 2 (HCCL / MC2)"
    return "Stream 0 (Compute)"


def _kernel_with_backend(op: OperatorExportRow) -> str:
    kern = op.ascend_kernel or op.implementation
    return f"{kern} [{op.kernel_backend}]" if op.kernel_backend else kern


def _lookup_agent_design(
    designs: Mapping[str, Mapping[str, str]] | None,
    kernel_key: str,
    target_kernel: str = "",
) -> tuple[str | None, str | None]:
    if not designs:
        return None, None
    entry = designs.get(kernel_key)
    if entry is None:
        low_key = kernel_key.lower()
        low_tgt = target_kernel.lower()
        for k, val in designs.items():
            kl = str(k).lower()
            if kl == low_key or (kl and (kl in low_key or kl in low_tgt)):
                entry = val
                break
    if not isinstance(entry, Mapping):
        return None, None
    design = str(entry.get("design") or entry.get("proposal") or "").strip() or None
    accept = str(entry.get("acceptance") or entry.get("criteria") or "").strip() or None
    return design, accept


def _populate_summary_sheet(
    ws: Worksheet,
    bundle: ExportBundle,
    *,
    compare_bundle: ExportBundle | None = None,
    backlog_count: int = 0,
) -> None:
    _apply_col_widths(ws, SUMMARY_COL_WIDTHS)
    ws.freeze_panes(SUMMARY_FREEZE_ROWS, 0)
    track_note = (
        f" | Before({bundle.feature_track or 'base'}) -> After({compare_bundle.feature_track or 'cand'})"
        if compare_bundle is not None else (f" | Track={bundle.feature_track}" if bundle.feature_track else "")
    )
    _write_banner(
        ws, 1, SUMMARY_COLS,
        f"MindIE-Sim Ascend Performance & Optimization Summary | {bundle.model_name} @ {bundle.hardware_sku}{track_note}",
        STYLE_TITLE,
    )
    mem = bundle.memory_ledger
    par_label = f"DP={mem.dp} TP={mem.tp} EP={mem.ep} PP={mem.pp} DCP={mem.dcp} PCP={mem.pcp} LocalTP={mem.local_tp}"
    _write_row(ws, 2, (
        ("Phase", None, STYLE_MERGED),
        (bundle.phase.upper(), None, STYLE_TEXT),
        ("Parallelism", None, STYLE_MERGED),
        (par_label, None, STYLE_TEXT),
    ))
    ws.merge_range(2, SUMMARY_META_PAR_START_COL, 2, SUMMARY_META_PAR_END_COL)
    _write_row(ws, 2, (
        ("Context/Seq/Tok", None, STYLE_MERGED),
        (f"{bundle.context_tokens}/{bundle.sequences}/{bundle.tokens_per_seq}", None, STYLE_TEXT),
        ("Layers / Graph", None, STYLE_MERGED),
        (
            f"Target={bundle.total_target_layers}L Draft={bundle.total_draft_layers}L | {bundle.graph_id[:SUMMARY_GRAPH_PREVIEW_LEN]}",
            None, STYLE_TEXT,
        ),
    ), start_col=SUMMARY_META_CTX_COL)
    ws.merge_range(2, SUMMARY_META_GRAPH_START_COL, 2, SUMMARY_COLS)

    _write_banner(ws, SUMMARY_SECTION_ROW, SUMMARY_LEFT_END_COL, "1. Whole-Net Step Performance & Lean 4 Certified Lower Bound", STYLE_SECTION)
    ws.set_cell(SUMMARY_SECTION_ROW, SUMMARY_RIGHT_START_COL, "2. Hardware Bottleneck Mix & Exposed Clocks", style=STYLE_SECTION)
    ws.merge_range(SUMMARY_SECTION_ROW, SUMMARY_RIGHT_START_COL, SUMMARY_SECTION_ROW, SUMMARY_COLS)

    latency_label = "Prefill TTFT (ms)" if bundle.phase == "prefill" else "Decode TPOT (ms)"
    step_ms = bundle.step_span_us / US_PER_MS
    l_star_ms = bundle.step_l_star_lean_us / US_PER_MS
    overlap_ratio = (bundle.operator_sum_us / bundle.step_span_us) if bundle.step_span_us > 0.0 else 1.0
    phys_eff = (bundle.step_l_star_lean_us / bundle.step_span_us) if bundle.step_span_us > 0.0 else 0.0

    _write_headers(ws, SUMMARY_HEADER_ROW, ("Metric", "Value", "Unit", "Basis"))
    ws.merge_range(SUMMARY_HEADER_ROW, SUMMARY_BASIS_START_COL, SUMMARY_HEADER_ROW, SUMMARY_LEFT_END_COL)
    kpi_rows: tuple[tuple[str, Any, str | None, int, str, str], ...] = (
        (latency_label, step_ms, None, STYLE_FLOAT3, "ms", "runtime.Engine (Full Model)"),
        ("Step Latency", bundle.step_span_us, "=B6*1000", STYLE_FLOAT2, "us", f"{len(bundle.archetypes)} Unique Archetypes"),
        ("Operator Sum", bundle.operator_sum_us, None, STYLE_FLOAT2, "us", "Serial Model Op Sum"),
        ("Overlap Speedup", overlap_ratio, "=IF(B7>0,B8/B7,1)", STYLE_RATIO, "x", "OpSum / Step"),
        ("Throughput (Total)", bundle.throughput_tokens_per_s, None, STYLE_FLOAT2, "tok/s", "Cluster"),
        ("Throughput (Per Die)", bundle.throughput_per_die_tokens_per_s, None, STYLE_FLOAT2, "tok/s/die", f"{mem.total_dies} dies"),
        ("Lean 4 L* Floor", bundle.step_l_star_lean_us, None, STYLE_FLOAT2, "us", f"Cert={bundle.step_certificate_valid}"),
        ("Step Gap (T_pred - L*)", bundle.step_span_us - bundle.step_l_star_lean_us, "=B7-B12", STYLE_FLOAT2, "us", ",".join(bundle.step_tight_resources) or "bound"),
        ("Physical-Limit Eff (L*/T)", phys_eff, "=IF(B7>0,B12/B7,0)", STYLE_PCT, "%", f"L*={l_star_ms:.3f} ms"),
        ("Network Cube MFU", bundle.network_cube_mfu / PCT_DIVISOR, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network Vector MFU", bundle.network_vec_mfu / PCT_DIVISOR, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network HBM MBU", bundle.network_mbu / PCT_DIVISOR, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network Link Util", bundle.network_link_util / PCT_DIVISOR, None, STYLE_PCT, "%", bundle.hardware_die_summary),
    )
    for offset, (label, val, formula, st, unit, basis) in enumerate(kpi_rows):
        r = SUMMARY_DATA_START_ROW + offset
        _write_row(ws, r, ((label, None, STYLE_TEXT), (val, formula, st), (unit, None, STYLE_TEXT), (basis, None, STYLE_TEXT)))
        ws.merge_range(r, SUMMARY_BASIS_START_COL, r, SUMMARY_LEFT_END_COL)

    b_tot_row = _populate_bottleneck_table(ws, bundle)
    next_row = _populate_summary_delta_block(ws, b_tot_row + 2, bundle, compare_bundle, backlog_count=backlog_count)
    radar_start = max(SUMMARY_DATA_START_ROW + len(kpi_rows) + 1, next_row + 1)
    _populate_gap_radar_block(ws, radar_start, compare_bundle or bundle)


def _populate_bottleneck_table(ws: Worksheet, bundle: ExportBundle) -> int:
    _write_headers(
        ws, SUMMARY_HEADER_ROW,
        ("Bottleneck Class", "Model Calls", "Duration (ms)", "Share", "Roofline Eff", "Exposed (ms)", "Focus"),
        start_col=SUMMARY_RIGHT_START_COL,
    )
    bound_keys = ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")
    exposed_map = {
        "memory_bound": bundle.exposed_breakdown_us.get("memory", 0.0) / US_PER_MS,
        "cube_bound": bundle.exposed_breakdown_us.get("cube", 0.0) / US_PER_MS,
        "vector_bound": bundle.exposed_breakdown_us.get("vector", 0.0) / US_PER_MS,
        "link_bound": bundle.exposed_breakdown_us.get("communication", 0.0) / US_PER_MS,
        "launch_bound": 0.0,
    }
    focus_map = {
        "memory_bound": "Quant / Fusion / UB Reuse",
        "cube_bound": "AIC Tiling / Format",
        "vector_bound": "AIV Fusion / C++ DSL",
        "link_bound": "MC2 / Overlap / Comm Split",
        "launch_bound": "aclGraph / Op Fusion",
    }
    b_start = SUMMARY_DATA_START_ROW
    b_end = b_start + len(bound_keys) - 1
    b_tot_row = b_end + 1
    tot_b_ms = sum(float((bundle.bottleneck_mix.get(k) or {}).get("duration_us", 0.0)) for k in bound_keys) / US_PER_MS
    tot_b_ops = sum(int((bundle.bottleneck_mix.get(k) or {}).get("count", 0)) for k in bound_keys)
    tot_exp_ms = sum(exposed_map.values())
    for offset, b_key in enumerate(bound_keys):
        r = b_start + offset
        info = bundle.bottleneck_mix.get(b_key) or {}
        dur_ms = float(info.get("duration_us", 0.0)) / US_PER_MS
        share = (dur_ms / tot_b_ms) if tot_b_ms > 0.0 else 0.0
        _write_row(ws, r, (
            (b_key, None, STYLE_TEXT),
            (int(info.get("count", 0)), None, STYLE_INT),
            (dur_ms, None, STYLE_FLOAT3),
            (share, f"=IF(H{b_tot_row}>0,H{r}/$H${b_tot_row},0)", STYLE_PCT),
            (float(info.get("roofline_eff_pct", 0.0)) / PCT_DIVISOR, None, STYLE_PCT),
            (exposed_map[b_key], None, STYLE_FLOAT3),
            (focus_map[b_key], None, STYLE_TEXT),
        ), start_col=SUMMARY_RIGHT_START_COL)
    _write_row(ws, b_tot_row, (
        ("Total", None, STYLE_TOTAL_TEXT),
        (tot_b_ops, f"=SUM(G{b_start}:G{b_end})", STYLE_TOTAL_NUM),
        (tot_b_ms, f"=SUM(H{b_start}:H{b_end})", STYLE_TOTAL_FLOAT3),
        (1.0 if tot_b_ms > 0.0 else 0.0, f"=SUM(I{b_start}:I{b_end})", STYLE_TOTAL_PCT),
        (bundle.network_roofline_efficiency / PCT_DIVISOR, None, STYLE_TOTAL_PCT),
        (tot_exp_ms, f"=SUM(K{b_start}:K{b_end})", STYLE_TOTAL_FLOAT3),
        ("", None, STYLE_TOTAL_TEXT),
    ), start_col=SUMMARY_RIGHT_START_COL)
    return b_tot_row


def _populate_summary_delta_block(
    ws: Worksheet,
    start_row: int,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
    *,
    backlog_count: int,
) -> int:
    c_start = SUMMARY_RIGHT_START_COL
    if compare_bundle is not None:
        ws.set_cell(start_row, c_start, "3. Before vs After Optimization Comparison Delta", style=STYLE_SECTION)
        ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
        _write_headers(ws, start_row + 1, ("Metric", "Before", "After", "Delta", "Delta %", "Speedup", "Note"), start_col=c_start)
        pairs = (
            ("Step Latency (ms)", bundle.step_span_us / US_PER_MS, compare_bundle.step_span_us / US_PER_MS, "Lower is better"),
            ("Lean 4 L* Floor (ms)", bundle.step_l_star_lean_us / US_PER_MS, compare_bundle.step_l_star_lean_us / US_PER_MS, "Physical floor"),
            ("Throughput (tok/s)", bundle.throughput_tokens_per_s, compare_bundle.throughput_tokens_per_s, "Higher is better"),
            ("Max Stage HBM (GiB)", bundle.memory_ledger.max_stage_hbm_bytes / GIB_BYTES,
             compare_bundle.memory_ledger.max_stage_hbm_bytes / GIB_BYTES, bundle.memory_ledger.weight_quant_display),
        )
        for idx, (label, before, after, note) in enumerate(pairs):
            r = start_row + 2 + idx
            delta = after - before
            pct = (delta / before) if before > 0.0 else 0.0
            spd = (before / after) if after > 0.0 else 0.0
            _write_row(ws, r, (
                (label, None, STYLE_TEXT),
                (before, None, STYLE_FLOAT3),
                (after, None, STYLE_FLOAT3),
                (delta, f"=H{r}-G{r}", STYLE_FLOAT3),
                (pct, f"=IF(G{r}>0,(H{r}-G{r})/G{r},0)", STYLE_PCT),
                (spd, f"=IF(H{r}>0,G{r}/H{r},0)", STYLE_RATIO),
                (note, None, STYLE_TEXT),
            ), start_col=c_start)
        r_bk = start_row + 2 + len(pairs)
        _write_row(ws, r_bk, (
            ("New/Extended Kernels", None, STYLE_TEXT),
            (0, None, STYLE_INT),
            (backlog_count, None, STYLE_INT),
            (backlog_count, None, STYLE_INT),
            (None, None, STYLE_TEXT),
            (None, None, STYLE_TEXT),
            (f"See {SHEET_NEW_KERNELS}", None, STYLE_NEW_KERNEL if backlog_count > 0 else STYLE_TEXT),
        ), start_col=c_start)
        return r_bk + 1

    if bundle.reconciliation is not None:
        rec = bundle.reconciliation
        ws.set_cell(start_row, c_start, f"3. Sim Pred vs Measured Profiler ({rec.source_label})", style=STYLE_SECTION)
        ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
        _write_headers(ws, start_row + 1, ("Reconcile Metric", "Sim Pred", "Measured", "Delta", "APE %", "Coverage", "Sheet"), start_col=c_start)
        r = start_row + 2
        delta = rec.matched_sim_sum_us - rec.matched_measured_sum_us
        ape = (abs(delta) / rec.matched_measured_sum_us) if rec.matched_measured_sum_us > 0.0 else 0.0
        _write_row(ws, r, (
            ("Matched Groups Sum (us)", None, STYLE_TEXT),
            (rec.matched_sim_sum_us, None, STYLE_FLOAT2),
            (rec.matched_measured_sum_us, None, STYLE_FLOAT2),
            (delta, f"=G{r}-H{r}", STYLE_FLOAT2),
            (ape, f"=IF(H{r}>0,ABS(G{r}-H{r})/H{r},0)", STYLE_PCT),
            (f"{rec.matched_kernels}/{rec.total_kernels} kernels", None, STYLE_TEXT),
            (SHEET_MEASURED, None, STYLE_TEXT),
        ), start_col=c_start)
        return r + 1

    ws.set_cell(start_row, c_start, "3. Physical-Limit Headroom Summary (Standalone)", style=STYLE_SECTION)
    ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
    _write_headers(ws, start_row + 1, ("Metric", "Sim Pred (ms)", "Lean L* (ms)", "Gap (ms)", "Gap Ratio", "Certificate", "Quant Policy"), start_col=c_start)
    r = start_row + 2
    pred_ms = bundle.step_span_us / US_PER_MS
    lstar_ms = bundle.step_l_star_lean_us / US_PER_MS
    ratio = (pred_ms / lstar_ms) if lstar_ms > 0.0 else 0.0
    _write_row(ws, r, (
        ("Whole Step (ms)", None, STYLE_TEXT),
        (pred_ms, None, STYLE_FLOAT3),
        (lstar_ms, None, STYLE_FLOAT3),
        (pred_ms - lstar_ms, f"=G{r}-H{r}", STYLE_FLOAT3),
        (ratio, f"=IF(H{r}>0,G{r}/H{r},0)", _gap_ratio_style(ratio)),
        ("VERIFIED" if bundle.step_certificate_valid else "UNVERIFIED", None, STYLE_TEXT),
        (bundle.memory_ledger.weight_quant_display, None, STYLE_TEXT),
    ), start_col=c_start)
    return r + 1


def _populate_gap_radar_block(ws: Worksheet, start_row: int, bundle: ExportBundle) -> None:
    _write_banner(
        ws, start_row, SUMMARY_COLS,
        "4. Top Physical-Limit Gap Radar (Model-Scaled ms, Ranked by T_pred - L*_Lean)",
        STYLE_SECTION,
    )
    headers = (
        "Rank", "Archetype", "×N", "Submodule", "Logical Role", "Ascend Kernel",
        "Kernel Status", "Bottleneck", "Model L* (ms)", "Model Pred (ms)",
        "Model Gap (ms)", "Gap Ratio",
    )
    hdr_row = start_row + 1
    _write_headers(ws, hdr_row, headers)
    for offset, entry in enumerate(bundle.gap_radar):
        r = hdr_row + 1 + offset
        l_ms = entry.l_star_lean_us / US_PER_MS
        p_ms = entry.pred_us / US_PER_MS
        g_ms = entry.gap_us / US_PER_MS
        _write_row(ws, r, (
            (entry.rank, None, STYLE_INT),
            (entry.layer_label, None, STYLE_TEXT_WRAP),
            (entry.repeat_count, None, STYLE_INT),
            (entry.submodule, None, STYLE_TEXT_WRAP),
            (entry.logical_role or entry.op_id, None, STYLE_TEXT_WRAP),
            (entry.implementation, None, STYLE_TEXT_WRAP),
            (entry.kernel_status, None, STYLE_NEW_KERNEL if "[" in entry.kernel_status else STYLE_TEXT_WRAP),
            (entry.dominant_bound, None, STYLE_TEXT),
            (l_ms, None, STYLE_FLOAT3),
            (p_ms, None, STYLE_FLOAT3),
            (g_ms, f"=J{r}-I{r}", STYLE_FLOAT3),
            (entry.gap_ratio if entry.gap_ratio is not None else 0.0, f"=IF(I{r}>0,J{r}/I{r},0)", _gap_ratio_style(entry.gap_ratio)),
        ))


def _populate_comparison_breakdown_sheet(
    ws: Worksheet,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    aligned_rows: Sequence[AlignedComparisonRow],
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Populate Sheet 2 with side-by-side Before vs After N->1 ``merge_cells`` & live formulas."""
    _apply_col_widths(ws, CMP_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, CMP_COLS,
        (
            f"Before ({before_bundle.feature_track or 'baseline'}) vs After ({after_bundle.feature_track or 'candidate'}) "
            f"Layer-Archetype Operator Breakdown | {before_bundle.model_name} @ {before_bundle.hardware_sku} ({before_bundle.phase.upper()})"
        ),
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role", "Rank-Local Shapes & Dtype",
        "Before Ascend Kernel [Backend]", "Before Source Ref",
        "Before Single (us)", "Before Model (ms)", "Before Bound",
        "After Ascend Kernel [Backend]", "Kernel Status (算子状态/开发需求)",
        "After Single (us)", "After Stream / Overlap", "After Model Exposed (ms)",
        "Saved Δ (ms)", "Speedup", "After Bound",
        "Lean L* Model (ms)", "Gap vs L* (ms)",
        "Optimization Action & New-Kernel Design Slot",
    )
    _write_headers(ws, 2, headers)

    # Group aligned rows by archetype so each archetype ends with a crisp Subtotal row.
    by_arch: list[tuple[str, str, int, list[AlignedComparisonRow]]] = []
    for item in aligned_rows:
        if not by_arch or by_arch[-1][0] != item.archetype_id:
            by_arch.append((item.archetype_id, item.archetype_title, item.repeat_count, []))
        by_arch[-1][3].append(item)

    row_cursor = CMP_DATA_START_ROW
    subtotal_rows: list[int] = []
    tot_bef_ms = tot_aft_ms = tot_lstar_ms = 0.0

    for _, arch_title, rep_count, arch_items in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_bef_ms = arch_aft_ms = arch_lstar_ms = 0.0

        for item in arch_items:
            row_count = max(1, len(item.before_ops), len(item.after_ops))
            r_start = row_cursor
            r_end = row_cursor + row_count - 1
            b_ms, a_ms, l_ms = _write_comparison_group(ws, r_start, r_end, item, agent_designs)
            arch_bef_ms += b_ms
            arch_aft_ms += a_ms
            arch_lstar_ms += l_ms
            _extend_span(sub_spans, item.submodule, r_start, r_end)
            row_cursor = r_end + 1

        arch_end = row_cursor - 1
        ws.set_cell(arch_start, CMP_COL_ARCH, arch_title, style=STYLE_MERGED)
        ws.merge_range(arch_start, CMP_COL_ARCH, arch_end, CMP_COL_ARCH)
        for sub_label, s1, s2 in sub_spans:
            ws.set_cell(s1, CMP_COL_SUB, sub_label, style=STYLE_MERGED)
            ws.merge_range(s1, CMP_COL_SUB, s2, CMP_COL_SUB)

        r_sub = row_cursor
        _write_comparison_subtotal_row(
            ws, r_sub, arch_title, arch_start, arch_end, arch_bef_ms, arch_aft_ms, arch_lstar_ms,
        )
        subtotal_rows.append(r_sub)
        tot_bef_ms += arch_bef_ms
        tot_aft_ms += arch_aft_ms
        tot_lstar_ms += arch_lstar_ms
        row_cursor += 1

    _write_comparison_grand_total(ws, row_cursor, subtotal_rows, tot_bef_ms, tot_aft_ms, tot_lstar_ms)


def _write_comparison_group(
    ws: Worksheet,
    r_start: int,
    r_end: int,
    item: AlignedComparisonRow,
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> tuple[float, float, float]:
    """Write one aligned 1->1 / N->1 / eliminated / added group with vertical ``merge_cells``."""
    rep = max(1, int(item.repeat_count))
    before_ms_sum = 0.0
    if item.before_ops:
        for offset, b_op in enumerate(item.before_ops):
            r = r_start + offset
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            b_ms = b_op.dur_us * rep / US_PER_MS
            before_ms_sum += b_ms
            _write_row(ws, r, (
                (rep, None, STYLE_INT),
            ), start_col=CMP_COL_REP)
            _write_row(ws, r, (
                (b_op.logical_role or b_op.op_id, None, STYLE_TEXT_WRAP),
                (b_op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(b_op), None, STYLE_TEXT_WRAP),
                (b_op.source_ref, None, STYLE_TEXT_WRAP),
                (b_op.dur_us, None, STYLE_FLOAT3),
                (b_ms, f"=H{r}*B{r}/1000", STYLE_FLOAT3),
                (b_op.bound_display, None, STYLE_TEXT),
            ), start_col=CMP_COL_ROLE)
    else:
        a0 = item.after_ops[0]
        ws.set_row_height(r_start, WRAP_ROW_HEIGHT_PT)
        ws.set_cell(r_start, CMP_COL_REP, rep, style=STYLE_INT)
        _write_row(ws, r_start, (
            (a0.logical_role or a0.op_id, None, STYLE_TEXT_WRAP),
            (a0.tensor_contract, None, STYLE_TEXT_WRAP),
            ("(Not in Before)", None, STYLE_TEXT),
            ("-", None, STYLE_TEXT),
            (0.0, None, STYLE_FLOAT3),
            (0.0, f"=H{r_start}*B{r_start}/1000", STYLE_FLOAT3),
            ("-", None, STYLE_TEXT),
        ), start_col=CMP_COL_ROLE)

    if item.after_ops:
        a_op = item.after_ops[0]
        stream_lbl = _stream_overlap_label(a_op)
        aft_ms = 0.0 if a_op.is_overlapped else (a_op.dur_us * rep / US_PER_MS)
        lstar_ms = 0.0 if a_op.is_overlapped else (a_op.l_star_lean_us * rep / US_PER_MS)
        saved_ms = before_ms_sum - aft_ms
        speedup = (before_ms_sum / aft_ms) if aft_ms > 0.0 else 0.0
        gap_ms = max(0.0, aft_ms - lstar_ms)
        design_override, _ = _lookup_agent_design(agent_designs, a_op.ascend_kernel, _kernel_with_backend(a_op))
        action_txt = (
            f"[Agent Design] {design_override}" if design_override
            else (f"{item.action_note} -> See {SHEET_NEW_KERNELS}" if a_op.needs_new_kernel else item.action_note)
        )
        _write_row(ws, r_start, (
            (_kernel_with_backend(a_op), None, STYLE_TEXT_WRAP),
            (a_op.kernel_status, None, STYLE_NEW_KERNEL if a_op.needs_new_kernel else STYLE_TEXT_WRAP),
            (a_op.dur_us, None, STYLE_FLOAT3),
            (stream_lbl, None, STYLE_OVERLAP if a_op.is_overlapped else STYLE_TEXT_WRAP),
            (aft_ms, f'=IF(ISNUMBER(SEARCH("Overlapped",N{r_start})),0,M{r_start}*B{r_start}/1000)', STYLE_FLOAT3),
            (saved_ms, f"=SUM(I{r_start}:I{r_end})-O{r_start}", STYLE_FLOAT3),
            (speedup, f"=IF(O{r_start}>0,SUM(I{r_start}:I{r_end})/O{r_start},0)", STYLE_RATIO),
            (a_op.bound_display, None, STYLE_OVERLAP if a_op.is_overlapped else STYLE_TEXT),
            (lstar_ms, None, STYLE_FLOAT3),
            (gap_ms, f"=MAX(0,O{r_start}-S{r_start})", STYLE_FLOAT3),
            (action_txt, None, STYLE_INPUT_TEXT_WRAP if (a_op.needs_new_kernel and not design_override) else STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)
    else:
        aft_ms = lstar_ms = 0.0
        _write_row(ws, r_start, (
            ("(Eliminated in After)", None, STYLE_OVERLAP),
            ("Eliminated", None, STYLE_OVERLAP),
            (0.0, None, STYLE_FLOAT3),
            ("Eliminated (0 exposed)", None, STYLE_OVERLAP),
            (0.0, "=0", STYLE_FLOAT3),
            (before_ms_sum, f"=SUM(I{r_start}:I{r_end})-O{r_start}", STYLE_FLOAT3),
            (0.0, f"=IF(O{r_start}>0,SUM(I{r_start}:I{r_end})/O{r_start},0)", STYLE_RATIO),
            ("Eliminated", None, STYLE_OVERLAP),
            (0.0, None, STYLE_FLOAT3),
            (0.0, f"=MAX(0,O{r_start}-S{r_start})", STYLE_FLOAT3),
            (item.action_note, None, STYLE_TEXT_WRAP),
        ), start_col=CMP_COL_AFT_KERN)

    if r_end > r_start:
        for col in CMP_AFTER_MERGE_COLS:
            ws.merge_range(r_start, col, r_end, col)
    return before_ms_sum, aft_ms, lstar_ms


def _write_comparison_subtotal_row(
    ws: Worksheet,
    r_sub: int,
    arch_title: str,
    r_start: int,
    r_end: int,
    bef_ms: float,
    aft_ms: float,
    lstar_ms: float,
) -> None:
    ws.set_cell(r_sub, 1, f"Subtotal — {arch_title}", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, 1, r_sub, CMP_SUBTOTAL_LABEL_END_COL)
    ws.set_cell(r_sub, CMP_COL_BEF_MS, bef_ms, formula=f"=SUM(I{r_start}:I{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_BEF_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.merge_range(r_sub, CMP_COL_BEF_BOUND, r_sub, CMP_COL_STREAM)
    ws.set_cell(r_sub, CMP_COL_AFT_MS, aft_ms, formula=f"=SUM(O{r_start}:O{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_SAVED_MS, bef_ms - aft_ms, formula=f"=I{r_sub}-O{r_sub}", style=STYLE_SUBTOTAL_NUM)
    spd = (bef_ms / aft_ms) if aft_ms > 0.0 else 0.0
    ws.set_cell(r_sub, CMP_COL_SPEEDUP, spd, formula=f"=IF(O{r_sub}>0,I{r_sub}/O{r_sub},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(r_sub, CMP_COL_AFT_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
    ws.set_cell(r_sub, CMP_COL_LSTAR_MS, lstar_ms, formula=f"=SUM(S{r_start}:S{r_end})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_GAP_MS, max(0.0, aft_ms - lstar_ms), formula=f"=MAX(0,O{r_sub}-S{r_sub})", style=STYLE_SUBTOTAL_NUM)
    ws.set_cell(r_sub, CMP_COL_ACTION, "", style=STYLE_SUBTOTAL_TEXT)


def _write_comparison_grand_total(
    ws: Worksheet,
    tot_row: int,
    subtotal_rows: Sequence[int],
    tot_bef_ms: float,
    tot_aft_ms: float,
    tot_lstar_ms: float,
) -> None:
    ws.set_cell(tot_row, 1, "WHOLE-MODEL TOTAL (Exposed Sum across All Archetypes)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, CMP_SUBTOTAL_LABEL_END_COL)
    sum_i = "=" + "+".join(f"I{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_o = "=" + "+".join(f"O{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_s = "=" + "+".join(f"S{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(tot_row, CMP_COL_BEF_MS, tot_bef_ms, formula=sum_i, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_BEF_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, CMP_COL_BEF_BOUND, tot_row, CMP_COL_STREAM)
    ws.set_cell(tot_row, CMP_COL_AFT_MS, tot_aft_ms, formula=sum_o, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_SAVED_MS, tot_bef_ms - tot_aft_ms, formula=f"=I{tot_row}-O{tot_row}", style=STYLE_TOTAL_FLOAT3)
    spd = (tot_bef_ms / tot_aft_ms) if tot_aft_ms > 0.0 else 0.0
    ws.set_cell(tot_row, CMP_COL_SPEEDUP, spd, formula=f"=IF(O{tot_row}>0,I{tot_row}/O{tot_row},0)", style=STYLE_SUBTOTAL_RATIO)
    ws.set_cell(tot_row, CMP_COL_AFT_BOUND, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(tot_row, CMP_COL_LSTAR_MS, tot_lstar_ms, formula=sum_s, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_GAP_MS, max(0.0, tot_aft_ms - tot_lstar_ms), formula=f"=MAX(0,O{tot_row}-S{tot_row})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(tot_row, CMP_COL_ACTION, "Live formula totals (zero double-counting)", style=STYLE_TOTAL_TEXT)


def _populate_standalone_breakdown_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    """Populate Sheet 2 in standalone mode (zero dead ``-`` Measured columns!)."""
    _apply_col_widths(ws, STD_COL_WIDTHS)
    ws.freeze_panes(CMP_FREEZE_ROWS, CMP_FREEZE_COLS)
    _write_banner(
        ws, 1, STD_COLS,
        f"Layer-Archetype -> Submodule -> Ascend Kernel Breakdown | {bundle.model_name} @ {bundle.hardware_sku} ({bundle.phase.upper()})",
        STYLE_TITLE,
    )
    headers = (
        "Archetype (Scope & Layer Range)", "Repeats (×N)", "Submodule",
        "Logical Role", "Rank-Local Shapes & Dtype", "Ascend Kernel [Backend]",
        "Kernel Status (算子状态/开发需求)", "Source Ref", "Stream / Overlap",
        "Single Pred (us)", "Model Exposed (ms)", "Lean L* Single (us)",
        "Lean L* Model (ms)", "Gap vs L* (ms)", "Bottleneck Bound",
    )
    _write_headers(ws, 2, headers)

    by_arch: list[tuple[str, str, list[OperatorExportRow]]] = []
    for op in bundle.operators:
        aid = op.archetype_id or op.layer_label
        if not by_arch or by_arch[-1][0] != aid:
            by_arch.append((aid, op.archetype_title or op.layer_label, []))
        by_arch[-1][2].append(op)

    row_cursor = STD_DATA_START_ROW
    subtotal_rows: list[int] = []
    tot_ms = tot_lstar_ms = 0.0
    for _, arch_title, ops in by_arch:
        arch_start = row_cursor
        sub_spans: list[tuple[str, int, int]] = []
        arch_ms = arch_lstar_ms = 0.0
        for op in ops:
            r = row_cursor
            ws.set_row_height(r, WRAP_ROW_HEIGHT_PT)
            rep = max(1, int(op.repeat_count))
            stream_lbl = _stream_overlap_label(op)
            exp_ms = 0.0 if op.is_overlapped else (op.dur_us * rep / US_PER_MS)
            l_ms = op.l_star_lean_us * rep / US_PER_MS
            arch_ms += exp_ms
            arch_lstar_ms += l_ms
            _write_row(ws, r, (
                (rep, None, STYLE_INT),
            ), start_col=STD_COL_REP)
            _write_row(ws, r, (
                (op.logical_role or op.op_id, None, STYLE_TEXT_WRAP),
                (op.tensor_contract, None, STYLE_TEXT_WRAP),
                (_kernel_with_backend(op), None, STYLE_TEXT_WRAP),
                (op.kernel_status, None, STYLE_NEW_KERNEL if op.needs_new_kernel else STYLE_TEXT_WRAP),
                (op.source_ref, None, STYLE_TEXT_WRAP),
                (stream_lbl, None, STYLE_OVERLAP if op.is_overlapped else STYLE_TEXT_WRAP),
                (op.dur_us, None, STYLE_FLOAT3),
                (exp_ms, f'=IF(ISNUMBER(SEARCH("Overlapped",I{r})),0,J{r}*B{r}/1000)', STYLE_FLOAT3),
                (op.l_star_lean_us, None, STYLE_FLOAT3),
                (l_ms, f"=L{r}*B{r}/1000", STYLE_FLOAT3),
                (max(0.0, exp_ms - l_ms), f"=MAX(0,K{r}-M{r})", STYLE_FLOAT3),
                (op.bound_display, None, STYLE_OVERLAP if op.is_overlapped else STYLE_TEXT),
            ), start_col=STD_COL_ROLE)
            _extend_span(sub_spans, op.submodule, r, r)
            row_cursor += 1
        arch_end = row_cursor - 1
        ws.set_cell(arch_start, STD_COL_ARCH, arch_title, style=STYLE_MERGED)
        ws.merge_range(arch_start, STD_COL_ARCH, arch_end, STD_COL_ARCH)
        for sub_label, s1, s2 in sub_spans:
            ws.set_cell(s1, STD_COL_SUB, sub_label, style=STYLE_MERGED)
            ws.merge_range(s1, STD_COL_SUB, s2, STD_COL_SUB)

        r_sub = row_cursor
        ws.set_cell(r_sub, 1, f"Subtotal — {arch_title}", style=STYLE_SUBTOTAL_TEXT)
        ws.merge_range(r_sub, 1, r_sub, STD_SUBTOTAL_LABEL_END_COL)
        ws.set_cell(r_sub, STD_COL_MS, arch_ms, formula=f"=SUM(K{arch_start}:K{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_LSTAR_US, "", style=STYLE_SUBTOTAL_TEXT)
        ws.set_cell(r_sub, STD_COL_LSTAR_MS, arch_lstar_ms, formula=f"=SUM(M{arch_start}:M{arch_end})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_GAP_MS, max(0.0, arch_ms - arch_lstar_ms), formula=f"=MAX(0,K{r_sub}-M{r_sub})", style=STYLE_SUBTOTAL_NUM)
        ws.set_cell(r_sub, STD_COL_BOUND, "", style=STYLE_SUBTOTAL_TEXT)
        subtotal_rows.append(r_sub)
        tot_ms += arch_ms
        tot_lstar_ms += arch_lstar_ms
        row_cursor += 1

    ws.set_cell(row_cursor, 1, "WHOLE-MODEL TOTAL (Exposed Sum)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(row_cursor, 1, row_cursor, STD_SUBTOTAL_LABEL_END_COL)
    sum_k = "=" + "+".join(f"K{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    sum_m = "=" + "+".join(f"M{r}" for r in subtotal_rows) if subtotal_rows else "=0"
    ws.set_cell(row_cursor, STD_COL_MS, tot_ms, formula=sum_k, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_LSTAR_US, "", style=STYLE_TOTAL_TEXT)
    ws.set_cell(row_cursor, STD_COL_LSTAR_MS, tot_lstar_ms, formula=sum_m, style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_GAP_MS, max(0.0, tot_ms - tot_lstar_ms), formula=f"=MAX(0,K{row_cursor}-M{row_cursor})", style=STYLE_TOTAL_FLOAT3)
    ws.set_cell(row_cursor, STD_COL_BOUND, f"{len(bundle.operators)} archetype ops", style=STYLE_TOTAL_TEXT)


def _populate_memory_control_panel(ws: Worksheet, mem: MemoryCalculatorLedger) -> None:
    c_lbl = MEMORY_CONTROL_COL_LABEL
    c_val = MEMORY_CONTROL_COL_VALUE
    ws.set_cell(2, c_lbl, "Parallel & Quant Control Panel", style=STYLE_SECTION)
    ws.merge_range(2, c_lbl, 2, c_val)
    controls: tuple[tuple[str, Any, str | None, int], ...] = (
        ("Weight Quant Preset", mem.weight_quant, None, STYLE_INPUT_TEXT),
        ("KV/Indexer/KeyPool Dtype", mem.cache_dtype, None, STYLE_INPUT_TEXT),
        ("Engram / PLE Host Offload", "ON" if mem.engram_host_offload else "OFF", None, STYLE_INPUT_TEXT),
        ("DP", mem.dp, None, STYLE_INPUT_NUM),
        ("TP", mem.tp, None, STYLE_INPUT_NUM),
        ("EP", mem.ep, None, STYLE_INPUT_NUM),
        ("PP", mem.pp, None, STYLE_INPUT_NUM),
        ("DCP", mem.dcp, None, STYLE_INPUT_NUM),
        ("PCP", mem.pcp, None, STYLE_INPUT_NUM),
        ("Local-TP (o/q_proj)", mem.local_tp, None, STYLE_INPUT_NUM),
        ("Shared-TP", mem.shared_tp, None, STYLE_INPUT_NUM),
        ("Total Cluster Dies", mem.total_dies, "=R6*R7*R9*R11", STYLE_INT),
        ("Sequence Length (tok)", mem.seq_len, None, STYLE_INPUT_NUM),
        ("Concurrency (per DP)", mem.concurrency, None, STYLE_INPUT_NUM),
        ("Cache Block Size", mem.block_size, None, STYLE_INPUT_NUM),
        ("Spec / MTP Tokens", mem.spec_tokens, None, STYLE_INPUT_NUM),
        ("Local Cache Tokens / Rank", mem.local_cache_tokens, "=ROUNDUP(R15/(R17*R10),0)*R17", STYLE_INT),
        ("Die HBM Capacity (GiB)", mem.die_hbm_capacity_bytes / float(GIB_BYTES), None, STYLE_FLOAT2),
        ("GPU Memory Utilization", mem.gpu_memory_utilization, None, STYLE_PCT),
        ("Usable Die HBM Budget (GiB)", mem.usable_hbm_bytes / float(GIB_BYTES), "=R20*R21", STYLE_TOTAL_NUM),
        ("Quant Exclusion Policy", mem.weight_quant_display or mem.weight_quant, None, STYLE_TEXT_WRAP),
    )
    for offset, (label, val, formula, style) in enumerate(controls):
        r = MEMORY_TABLE_HEADER_ROW + offset
        ws.set_cell(r, c_lbl, label, style=STYLE_TEXT)
        ws.set_cell(r, c_val, val, formula=formula, style=style)

    ws.add_data_validation("R3", formula1='"' + ",".join(SUPPORTED_WEIGHT_QUANTS) + '"')
    ws.add_data_validation("R4", formula1='"' + ",".join(SUPPORTED_CACHE_DTYPES) + '"')
    ws.add_data_validation("R5", formula1='"ON,OFF"')

    ws.set_cell(2, MEMORY_LOOKUP_START_COL, "Quant Format Storage Rules", style=STYLE_SECTION)
    ws.merge_range(2, MEMORY_LOOKUP_START_COL, 2, MEMORY_COLS)
    _write_headers(ws, MEMORY_LOOKUP_HEADER_ROW, ("Format", "B/Elem", "MX Group", "Aux B/Row"), start_col=MEMORY_LOOKUP_START_COL)
    for idx, fmt in enumerate(MEMORY_QUANT_LOOKUP_ORDER):
        r = MEMORY_LOOKUP_FIRST_ROW + idx
        spec = WEIGHT_QUANT_FORMATS[fmt]
        _write_row(ws, r, (
            (fmt, None, STYLE_TEXT),
            (float(spec["bytes"]), None, STYLE_FLOAT2),
            (int(spec["group"]), None, STYLE_INT),
            (int(spec["aux"]), None, STYLE_INT),
        ), start_col=MEMORY_LOOKUP_START_COL)


def _populate_memory_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    mem = bundle.memory_ledger
    _apply_col_widths(ws, MEMORY_COL_WIDTHS)
    ws.freeze_panes(MEMORY_FREEZE_ROWS, 2)
    _write_banner(
        ws, 1, MEMORY_COLS,
        f"Closed-Form Memory / Quant / Cache / Engram-PLE / PP HBM Calculator | {mem.model_name} @ {mem.hardware_sku} | {mem.weight_quant_display or mem.weight_quant}",
        STYLE_TITLE,
    )
    _populate_memory_control_panel(ws, mem)

    ws.set_cell(2, 1, "1. Per-Die Weight + Quant Scale Closed-Form Table (Post-TP Local Shapes, Respecting quant.exclusions)", style=STYLE_SECTION)
    ws.merge_range(2, 1, 2, MEMORY_TABLE_END_COL)
    w_headers = (
        "Group", "Tensor Name", "Global Rows", "Global Cols", "Shard Rule",
        "Local Rows", "Local Cols", "Max Stage Rep", "Quant Format",
        "Payload (B)", "MX Scale (B)", "Channel Aux (B)", "Per-Die (GiB)", "Residency & Quant Policy",
    )
    _write_headers(ws, MEMORY_TABLE_HEADER_ROW, w_headers)

    lookup_ref = f"$S${MEMORY_LOOKUP_FIRST_ROW}:$V${MEMORY_LOOKUP_LAST_ROW}"
    w_start = MEMORY_TABLE_HEADER_ROW + 1
    group_spans: list[tuple[str, int, int]] = []
    for idx, row in enumerate(mem.weight_rows):
        r = w_start + idx
        _extend_span(group_spans, row.group, r, r)
        if row.is_engram_or_ple:
            res_text = "Host DRAM" if mem.engram_host_offload else "NPU HBM"
            res_formula: str | None = '=IF($R$5="ON","Host DRAM","NPU HBM")'
        else:
            res_text = f"NPU HBM ({row.quant_note})" if row.quant_note else "NPU HBM"
            res_formula = None
        _write_row(ws, r, (
            (row.name, None, STYLE_TEXT),
            (row.global_rows, None, STYLE_INT),
            (row.global_cols, None, STYLE_INT),
            (row.shard_rule, None, STYLE_TEXT),
            (row.local_rows, None, STYLE_INT),
            (row.local_cols, None, STYLE_INT),
            (row.max_stage_repeats, None, STYLE_INT),
            (row.quant_format, None, STYLE_INPUT_TEXT),
            (row.payload_bytes, f"=F{r}*ROUNDUP(G{r}*VLOOKUP(I{r},{lookup_ref},2,FALSE),0)*H{r}", STYLE_INT),
            (row.scale_bytes, f"=IF(VLOOKUP(I{r},{lookup_ref},3,FALSE)>0,F{r}*ROUNDUP(G{r}/VLOOKUP(I{r},{lookup_ref},3,FALSE),0)*H{r},0)", STYLE_INT),
            (row.aux_bytes, f"=F{r}*VLOOKUP(I{r},{lookup_ref},4,FALSE)*H{r}", STYLE_INT),
            (float(row.per_die_bytes) / float(GIB_BYTES), f"=(J{r}+K{r}+L{r})/{GIB_BYTES}", STYLE_FLOAT3),
            (res_text, res_formula, STYLE_TEXT),
        ), start_col=2)

    for grp, r1, r2 in group_spans:
        ws.set_cell(r1, 1, grp, style=STYLE_MERGED)
        ws.merge_range(r1, 1, r2, 1)

    w_end = w_start + len(mem.weight_rows) - 1
    if w_end >= w_start:
        ws.add_data_validation(f"I{w_start}:I{w_end}", formula1='"' + ",".join(MEMORY_QUANT_LOOKUP_ORDER) + '"')
    w_tot = w_end + 1
    ws.set_cell(w_tot, 1, "Weights Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(w_tot, 1, w_tot, MEMORY_WEIGHT_MERGE_END_COL)
    weight_totals = (
        sum(x.payload_bytes for x in mem.weight_rows),
        sum(x.scale_bytes for x in mem.weight_rows),
        sum(x.aux_bytes for x in mem.weight_rows),
        sum(x.per_die_bytes for x in mem.weight_rows) / float(GIB_BYTES),
    )
    for offset, tot_v in enumerate(weight_totals):
        col = MEMORY_WEIGHT_SUM_START_COL + offset
        c_let = column_letter(col)
        formula = f"=SUM({c_let}{w_start}:{c_let}{w_end})" if w_end >= w_start else "=0"
        ws.set_cell(w_tot, col, tot_v, formula=formula, style=STYLE_TOTAL_NUM)
    ws.set_cell(w_tot, MEMORY_TABLE_END_COL, f"{len(mem.weight_rows)} tensors", style=STYLE_TOTAL_TEXT)

    cache_end = _populate_cache_block(ws, w_tot + 2, mem)
    _populate_stage_budget_block(ws, cache_end + 2, mem)


def _populate_cache_block(ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger) -> int:
    ws.set_cell(start_row, 1, "2. KV / Indexer / KeyPool / Recurrent State Cache Breakdown", style=STYLE_SECTION)
    ws.merge_range(start_row, 1, start_row, MEMORY_TABLE_END_COL)
    hdr_row = start_row + 1
    headers = (
        "Component", "Cache Kind", "Max Stage Layers", "Bytes/Tok/Layer",
        "Per-Request (MiB)", "Batch Total (GiB)", "Scaling Basis",
    )
    _write_headers(ws, hdr_row, headers)
    ws.merge_range(hdr_row, MEMORY_CACHE_BASIS_START_COL, hdr_row, MEMORY_TABLE_END_COL)

    c_start = hdr_row + 1
    for offset, row in enumerate(mem.cache_rows):
        r = c_start + offset
        if row.scales_with_tokens:
            req_formula = f"=C{r}*D{r}*$R$19/{MIB_BYTES}"
            basis = f"Scales with Local Cache Tokens (BF16/C8/FP8={mem.cache_dtype}, ratio={C8_CACHE_FACTOR:.4f}x for C8)"
        else:
            req_formula = None
            basis = "Fixed per-request recurrent/conv/compressor state"
        _write_row(ws, r, (
            (row.component, None, STYLE_TEXT),
            (row.cache_kind, None, STYLE_TEXT),
            (row.max_stage_layers, None, STYLE_INT),
            (row.bytes_per_token_per_layer, None, STYLE_FLOAT2),
            (row.per_request_bytes / float(MIB_BYTES), req_formula, STYLE_FLOAT3),
            (row.batch_total_bytes / float(GIB_BYTES), f"=E{r}*$R$16/{MIB_PER_GIB}", STYLE_FLOAT3),
            (basis, None, STYLE_TEXT),
        ))
        ws.merge_range(r, MEMORY_CACHE_BASIS_START_COL, r, MEMORY_TABLE_END_COL)

    c_end = c_start + len(mem.cache_rows) - 1
    c_tot = c_end + 1
    ws.set_cell(c_tot, 1, "Cache Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(c_tot, 1, c_tot, MEMORY_CACHE_MERGE_END_COL)
    ws.set_cell(
        c_tot, MEMORY_CACHE_REQ_COL, sum(x.per_request_bytes for x in mem.cache_rows) / float(MIB_BYTES),
        formula=f"=SUM(E{c_start}:E{c_end})" if c_end >= c_start else "=0", style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BATCH_COL, sum(x.batch_total_bytes for x in mem.cache_rows) / float(GIB_BYTES),
        formula=f"=SUM(F{c_start}:F{c_end})" if c_end >= c_start else "=0", style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BASIS_START_COL,
        f"Local tokens={mem.local_cache_tokens}, Concurrency={mem.concurrency}", style=STYLE_TOTAL_TEXT,
    )
    ws.merge_range(c_tot, MEMORY_CACHE_BASIS_START_COL, c_tot, MEMORY_TABLE_END_COL)
    return c_tot


def _populate_stage_budget_block(ws: Worksheet, start_row: int, mem: MemoryCalculatorLedger) -> int:
    ws.set_cell(start_row, 1, "3. Per-PP-Stage HBM vs Host Offload Budget (DP/TP/EP/PP/DCP/PCP/Local-TP)", style=STYLE_SECTION)
    ws.merge_range(start_row, 1, start_row, MEMORY_TABLE_END_COL)
    hdr_row = start_row + 1
    headers = (
        "PP Stage", "Layers", "Weights Payload (GiB)", "Quant Scale/Aux (GiB)",
        "Engram/PLE HBM (GiB)", "Engram/PLE Host (GiB)", "Cache (GiB)",
        "Act & Workspace (GiB)", "Reserves (GiB)", "Total HBM (GiB)",
        "HBM Budget (GiB)", "Headroom (GiB)", "HBM Status", "Raw Engram (GiB)",
    )
    _write_headers(ws, hdr_row, headers)

    s_start = hdr_row + 1
    for offset, st in enumerate(mem.stage_budgets):
        r = s_start + offset
        raw_engram_gib = (st.engram_ple_hbm_bytes + st.engram_ple_host_bytes) / float(GIB_BYTES)
        _write_row(ws, r, (
            (f"Stage {st.stage}", None, STYLE_MERGED),
            (st.owned_layers, None, STYLE_INT),
            (st.weight_payload_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.weight_scale_aux_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.engram_ple_hbm_bytes / float(GIB_BYTES), f'=IF($R$5="ON",0,N{r})', STYLE_FLOAT3),
            (st.engram_ple_host_bytes / float(GIB_BYTES), f'=IF($R$5="ON",N{r},0)', STYLE_FLOAT3),
            (st.cache_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.workspace_peak_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.reserve_bytes / float(GIB_BYTES), None, STYLE_FLOAT3),
            (st.total_hbm_bytes / float(GIB_BYTES), f"=SUM(C{r}:E{r},G{r}:I{r})", STYLE_FLOAT3),
            (st.usable_hbm_budget_bytes / float(GIB_BYTES), "=$R$22", STYLE_FLOAT3),
            (st.headroom_bytes / float(GIB_BYTES), f"=K{r}-J{r}", STYLE_FLOAT3),
            ("PASS" if st.fits_hbm else "OOM", f'=IF(J{r}<=K{r},"PASS","OOM")', STYLE_ALERT_OK if st.fits_hbm else STYLE_ALERT_HIGH),
            (raw_engram_gib, None, STYLE_FLOAT3),
        ))

    s_end = s_start + len(mem.stage_budgets) - 1
    max_row = s_end + 1
    ws.set_cell(max_row, 1, "Max Stage (Bottleneck)", style=STYLE_TOTAL_TEXT)
    ws.set_cell(
        max_row, 2, sum(s.owned_layers for s in mem.stage_budgets),
        formula=f"=SUM(B{s_start}:B{s_end})" if s_end >= s_start else "=0", style=STYLE_TOTAL_NUM,
    )
    agg_series = (
        [s.weight_payload_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.weight_scale_aux_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.engram_ple_hbm_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.engram_ple_host_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.cache_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.workspace_peak_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.reserve_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.total_hbm_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.usable_hbm_budget_bytes / GIB_BYTES for s in mem.stage_budgets],
        [s.headroom_bytes / GIB_BYTES for s in mem.stage_budgets],
    )
    for offset, vals in enumerate(agg_series):
        col = MEMORY_STAGE_AGG_START_COL + offset
        c_let = column_letter(col)
        is_headroom = col == MEMORY_STAGE_HEADROOM_COL
        agg = "MIN" if is_headroom else "MAX"
        val = (min(vals) if is_headroom else max(vals)) if vals else 0.0
        ws.set_cell(
            max_row, col, val,
            formula=f"={agg}({c_let}{s_start}:{c_let}{s_end})" if s_end >= s_start else "=0",
            style=STYLE_TOTAL_NUM,
        )
    ws.set_cell(
        max_row, MEMORY_STAGE_STATUS_COL, "PASS" if mem.fits_hbm else "OOM",
        formula=f'=IF(J{max_row}<=K{max_row},"PASS","OOM")',
        style=STYLE_ALERT_OK if mem.fits_hbm else STYLE_ALERT_HIGH,
    )
    raw_engram_vals = [(s.engram_ple_hbm_bytes + s.engram_ple_host_bytes) / GIB_BYTES for s in mem.stage_budgets]
    ws.set_cell(
        max_row, MEMORY_STAGE_RAW_ENGRAM_COL, max(raw_engram_vals, default=0.0),
        formula=f"=MAX(N{s_start}:N{s_end})" if s_end >= s_start else "=0", style=STYLE_TOTAL_NUM,
    )
    return max_row


def _populate_new_kernel_backlog_sheet(
    ws: Worksheet,
    backlog: Sequence[NewKernelBacklogItem],
    target_bundle: ExportBundle,
    agent_designs: Mapping[str, Mapping[str, str]] | None,
) -> None:
    """Populate Sheet 4 with objective hardware constraints + Agent-fillable design slots."""
    _apply_col_widths(ws, BACKLOG_COL_WIDTHS)
    ws.freeze_panes(BACKLOG_HEADER_ROW, 2)
    _write_banner(
        ws, 1, BACKLOG_COLS,
        f"New & Extended Ascend Kernel Design Backlog | {target_bundle.model_name} @ {target_bundle.hardware_sku} ({target_bundle.hardware_die_summary})",
        STYLE_TITLE,
    )
    _write_banner(
        ws, 2, BACKLOG_COLS,
        "Columns A-H are pre-filled deterministically by MindIE-Sim; Columns I-J are reserved Agent design/acceptance slots (--kernel-designs-json).",
        STYLE_SECTION,
    )
    headers = (
        "Priority", "Target Kernel & Suggested Backend", "Development Status",
        "Replaces (Before Op Chain)", "Before Source Ref",
        "Archetypes & Calls/Step", "Expected Gain (ms)",
        "Rank-Local Shapes, Dtype & Hardware Spec",
        "Agent 新算子设计方案 / Tiling & 流水排布草稿 (Agent Fillable)",
        "风险与验收标准 (Agent Fillable)",
    )
    _write_headers(ws, BACKLOG_HEADER_ROW, headers)

    if not backlog:
        ws.set_cell(
            BACKLOG_DATA_START_ROW, 1,
            "No new or extended kernels required — all After kernels map to existing shipping CANN/HCCL ops.",
            style=STYLE_OVERLAP,
        )
        ws.merge_range(BACKLOG_DATA_START_ROW, 1, BACKLOG_DATA_START_ROW, BACKLOG_COLS)
        return

    for idx, item in enumerate(backlog):
        r = BACKLOG_DATA_START_ROW + idx
        ws.set_row_height(r, BACKLOG_ROW_HEIGHT_PT)
        design_ov, accept_ov = _lookup_agent_design(agent_designs, item.kernel_key, item.target_kernel)
        spec_txt = f"{item.tensor_contract}\n{item.microarch_target}"
        _write_row(ws, r, (
            (f"P{item.priority}", None, STYLE_MERGED),
            (item.target_kernel, None, STYLE_TEXT_WRAP),
            (item.status, None, STYLE_NEW_KERNEL),
            (item.replaces_chain, None, STYLE_TEXT_WRAP),
            (item.source_refs or "-", None, STYLE_TEXT_WRAP),
            (f"{item.archetype_summary} ({item.total_calls_per_step} calls/step)", None, STYLE_TEXT_WRAP),
            (item.expected_gain_ms, None, STYLE_FLOAT3),
            (spec_txt, None, STYLE_TEXT_WRAP),
            (design_ov or item.default_design_slot, None, STYLE_INPUT_TEXT_WRAP),
            (accept_ov or item.default_acceptance_slot, None, STYLE_INPUT_TEXT_WRAP),
        ))

    r_end = BACKLOG_DATA_START_ROW + len(backlog) - 1
    r_tot = r_end + 1
    ws.set_cell(r_tot, 1, f"TOTAL ({len(backlog)} kernel items)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, 1, r_tot, BACKLOG_LABEL_MERGE_END)
    ws.set_cell(
        r_tot, BACKLOG_COL_GAIN,
        sum(x.expected_gain_ms for x in backlog),
        formula=f"=SUM(G{BACKLOG_DATA_START_ROW}:G{r_end})",
        style=STYLE_TOTAL_FLOAT3,
    )
    ws.set_cell(r_tot, BACKLOG_COL_GAIN + 1, "Pass agent_kernel_designs or --kernel-designs-json to populate I-J", style=STYLE_TOTAL_TEXT)
    ws.merge_range(r_tot, BACKLOG_COL_GAIN + 1, r_tot, BACKLOG_COLS)


def _populate_measured_calibration_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    """Conditional sheet emitted ONLY when ``observations_path`` is supplied."""
    _apply_col_widths(ws, MEAS_COL_WIDTHS)
    ws.freeze_panes(2, 2)
    rec = bundle.reconciliation
    src_lbl = rec.source_label if rec is not None else "msprof"
    _write_banner(
        ws, 1, MEAS_COLS,
        f"Measured msprof 1-to-N Kernel Calibration ({src_lbl}) | {bundle.model_name} @ {bundle.hardware_sku}",
        STYLE_TITLE,
    )
    headers = (
        "Archetype", "Submodule", "Sim Op ID", "Logical Role", "Ascend Kernel",
        "Rank-Local Shapes & Dtype", "L2 Hit %", "L* Floor (us)", "Sim Pred (us)",
        "Measured Kernel (msprof)", "Kernel Measured (us)", "Measured Sum (us)",
        "APE %", "Bottleneck",
    )
    _write_headers(ws, 2, headers)

    row_cursor = MEAS_DATA_START_ROW
    for op in bundle.operators:
        k_count = max(1, len(op.measured_kernels))
        r_start = row_cursor
        r_end = row_cursor + k_count - 1
        _write_row(ws, r_start, (
            (op.archetype_title or op.layer_label, None, STYLE_TEXT_WRAP),
            (op.submodule, None, STYLE_TEXT_WRAP),
            (op.op_id, None, STYLE_TEXT),
            (op.logical_role, None, STYLE_TEXT_WRAP),
            (op.ascend_kernel, None, STYLE_TEXT_WRAP),
            (op.tensor_contract, None, STYLE_TEXT_WRAP),
            (op.l2_hit_pct / PCT_DIVISOR, None, STYLE_PCT),
            (op.l_star_lean_us, None, STYLE_FLOAT3),
            (op.dur_us, None, STYLE_FLOAT3),
        ))
        if op.measured_kernels:
            for offset, k_row in enumerate(op.measured_kernels):
                ws.set_cell(r_start + offset, MEAS_COL_KERN_NAME, k_row.name, style=STYLE_TEXT)
                ws.set_cell(r_start + offset, MEAS_COL_KERN_US, k_row.duration_us, style=STYLE_FLOAT3)
            ws.set_cell(
                r_start, MEAS_COL_SUM_US, op.measured_sum_us,
                formula=f"=SUM(K{r_start}:K{r_end})", style=STYLE_FLOAT3,
            )
            ws.set_cell(
                r_start, MEAS_COL_APE, (op.ape_pct / PCT_DIVISOR) if op.ape_pct is not None else 0.0,
                formula=f"=IF(L{r_start}>0,ABS(I{r_start}-L{r_start})/L{r_start},0)", style=STYLE_PCT,
            )
        else:
            _write_row(ws, r_start, (
                ("(Unmatched)", None, STYLE_TEXT), (None, None, STYLE_FLOAT3),
                (None, None, STYLE_FLOAT3), (None, None, STYLE_PCT),
            ), start_col=MEAS_COL_KERN_NAME)
        ws.set_cell(r_start, MEAS_COL_BOUND, op.bound_display, style=STYLE_TEXT)
        if k_count > 1:
            for col in MEAS_SIM_MERGE_COLS:
                ws.merge_range(r_start, col, r_end, col)
        row_cursor = r_end + 1

    data_end = row_cursor - 1
    tot_row = row_cursor
    ws.set_cell(tot_row, 1, "TOTAL", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, MEAS_COL_SIM_US - 2)
    total_vals = (
        sum(op.l_star_lean_us for op in bundle.operators),
        sum(op.dur_us for op in bundle.operators),
        sum(k.duration_us for op in bundle.operators for k in op.measured_kernels),
        sum(op.measured_sum_us or 0.0 for op in bundle.operators),
    )
    for col, val in zip(MEAS_TOTAL_SUM_COLS, total_vals):
        c_let = column_letter(col)
        formula = f"=SUM({c_let}{MEAS_DATA_START_ROW}:{c_let}{data_end})" if data_end >= MEAS_DATA_START_ROW else "=0"
        ws.set_cell(tot_row, col, val, formula=formula, style=STYLE_TOTAL_FLOAT3)
    meas_tot = sum(op.measured_sum_us or 0.0 for op in bundle.operators)
    sim_tot = sum(op.dur_us for op in bundle.operators if op.measured_sum_us is not None)
    agg_ape = (abs(sim_tot - meas_tot) / meas_tot) if meas_tot > 0.0 else 0.0
    ws.set_cell(
        tot_row, MEAS_COL_APE, agg_ape,
        formula=f"=IF(L{tot_row}>0,ABS(I{tot_row}-L{tot_row})/L{tot_row},0)",
        style=STYLE_TOTAL_PCT,
    )


def _resolve_xlsx_bundles(
    source: ExportBundle | CaseInput | None,
    compare_source: ExportBundle | CaseInput | None,
    *,
    request: StepRequest | None,
    step_eval: StepEvaluation | None,
    replay: ReplayOptions,
    observations_path: str | Path | None,
    weight_quant: str | None,
    cache_dtype: str | None,
    engram_host_offload: bool | None,
    model: str | None,
    hardware: str | None,
    topology_preset: str | None,
    dp: int,
    tp: int,
    dcp: int,
    ep: int,
    pp: int,
    pcp: int,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    mtp_proposal_tokens: int,
    mtp_acceptance_length: float,
    compare_track: str | None,
    target_track: str | None,
) -> tuple[ExportBundle, ExportBundle | None]:
    if source is None:
        if not model or not hardware:
            raise ValueError("export_comparison_xlsx_requires_source_or_model_and_hardware")
        base_track = compare_track or "merged"
        raw_bef, auto_req = compose_export_case(
            model, hardware, topology_preset=topology_preset,
            dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
            mtp_acceptance_length=mtp_acceptance_length, feature_track=base_track,
        )
        eff_req = request or auto_req
        source = raw_bef
        if compare_source is None and target_track:
            raw_aft, _ = compose_export_case(
                model, hardware, topology_preset=topology_preset,
                dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
                prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
                mtp_acceptance_length=mtp_acceptance_length, feature_track=target_track,
            )
            compare_source = raw_aft
    else:
        eff_req = request

    bundle = (
        source if isinstance(source, ExportBundle)
        else build_export_bundle(
            source,
            request=eff_req,
            step_eval=step_eval,
            replay=replay,
            observations_path=observations_path,
            weight_quant=weight_quant,
            cache_dtype=cache_dtype,
            engram_host_offload=engram_host_offload,
        )
    )
    compare_bundle: ExportBundle | None = None
    if compare_source is not None:
        compare_bundle = (
            compare_source if isinstance(compare_source, ExportBundle)
            else build_export_bundle(
                compare_source,
                request=eff_req or StepRequest(
                    phase=bundle.phase,  # type: ignore[arg-type]
                    tokens=bundle.tokens_per_seq,
                    context=bundle.context_tokens,
                    sequences=bundle.sequences,
                    scope=bundle.scope,  # type: ignore[arg-type]
                ),
                replay=replay,
                weight_quant=weight_quant,
                cache_dtype=cache_dtype,
                engram_host_offload=engram_host_offload,
            )
        )
    return bundle, compare_bundle


def export_comparison_xlsx(
    source: ExportBundle | CaseInput | None = None,
    output_path: str | Path = "",
    *,
    compare_source: ExportBundle | CaseInput | None = None,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    agent_kernel_designs: Mapping[str, Mapping[str, str]] | None = None,
    model: str | None = None,
    hardware: str | None = None,
    topology_preset: str | None = None,
    dp: int = 1,
    tp: int = 1,
    dcp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS,
    concurrency: int = 1,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    compare_track: str | None = None,
    target_track: str | None = None,
) -> XlsxExportSummary:
    """Build and atomically save the human-native Ascend comparison ``.xlsx`` workbook."""
    if not str(output_path):
        raise ValueError("export_comparison_xlsx_requires_output_path")
    bundle, compare_bundle = _resolve_xlsx_bundles(
        source, compare_source,
        request=request, step_eval=step_eval, replay=replay,
        observations_path=observations_path,
        weight_quant=weight_quant, cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload,
        model=model, hardware=hardware, topology_preset=topology_preset,
        dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
        mtp_acceptance_length=mtp_acceptance_length,
        compare_track=compare_track, target_track=target_track,
    )

    aligned_rows: tuple[AlignedComparisonRow, ...] = ()
    backlog: tuple[NewKernelBacklogItem, ...] = ()
    if compare_bundle is not None:
        aligned_rows = align_before_after_operators(bundle, compare_bundle)
        backlog = build_new_kernel_backlog(aligned_rows, compare_bundle)
    else:
        standalone_aligned = tuple(
            AlignedComparisonRow(
                op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count,
                op.submodule, (op,), (op,), "1->1", "",
            )
            for op in bundle.operators if op.needs_new_kernel
        )
        if standalone_aligned:
            backlog = build_new_kernel_backlog(standalone_aligned, bundle)

    wb = Workbook()
    ws_summary = wb.create_sheet(SHEET_SUMMARY)
    ws_breakdown = wb.create_sheet(SHEET_BREAKDOWN)
    ws_memory = wb.create_sheet(SHEET_MEMORY)

    _populate_summary_sheet(ws_summary, bundle, compare_bundle=compare_bundle, backlog_count=len(backlog))
    if compare_bundle is not None:
        _populate_comparison_breakdown_sheet(ws_breakdown, bundle, compare_bundle, aligned_rows, agent_kernel_designs)
    else:
        _populate_standalone_breakdown_sheet(ws_breakdown, bundle)
    _populate_memory_sheet(ws_memory, bundle)

    sheet_names = [SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY]
    merge_counts = {
        SHEET_SUMMARY: len(ws_summary.merged_ranges),
        SHEET_BREAKDOWN: len(ws_breakdown.merged_ranges),
        SHEET_MEMORY: len(ws_memory.merged_ranges),
    }

    if compare_bundle is not None or backlog:
        ws_backlog = wb.create_sheet(SHEET_NEW_KERNELS)
        _populate_new_kernel_backlog_sheet(ws_backlog, backlog, compare_bundle or bundle, agent_designs=agent_kernel_designs)
        sheet_names.append(SHEET_NEW_KERNELS)
        merge_counts[SHEET_NEW_KERNELS] = len(ws_backlog.merged_ranges)

    if bundle.reconciliation is not None:
        ws_meas = wb.create_sheet(SHEET_MEASURED)
        _populate_measured_calibration_sheet(ws_meas, bundle)
        sheet_names.append(SHEET_MEASURED)
        merge_counts[SHEET_MEASURED] = len(ws_meas.merged_ranges)

    bytes_written = wb.save(output_path)
    return XlsxExportSummary(
        schema=XLSX_EXPORT_RECEIPT_SCHEMA,
        output_path=str(output_path),
        model_name=bundle.model_name,
        hardware_sku=bundle.hardware_sku,
        phase=bundle.phase,
        graph_id=bundle.graph_id,
        sheets=tuple(sheet_names),
        merge_cell_counts=merge_counts,
        operator_rows=len(aligned_rows) if compare_bundle is not None else len(bundle.operators),
        weight_rows=len(bundle.memory_ledger.weight_rows),
        has_comparison=compare_bundle is not None,
        has_measured=bundle.reconciliation is not None,
        bytes_written=bytes_written,
        new_kernel_backlog_count=len(backlog),
    )


__all__ = [
    "SHEET_BREAKDOWN",
    "SHEET_MEASURED",
    "SHEET_MEMORY",
    "SHEET_NEW_KERNELS",
    "SHEET_SUMMARY",
    "Workbook",
    "Worksheet",
    "XLSX_EXPORT_RECEIPT_SCHEMA",
    "XlsxExportSummary",
    "cell_ref",
    "column_letter",
    "export_comparison_xlsx",
]
