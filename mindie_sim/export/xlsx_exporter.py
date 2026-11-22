"""Zero-dependency OOXML ``.xlsx`` exporter with ``merge_cells`` & Memory/Quant Calculator.

[Role] Generate executive-ready ECMA-376 OOXML ``.xlsx`` workbooks directly via
  Python's stdlib ``zipfile`` + XML serializer (zero third-party runtime deps)
  with real ``worksheet.merge_cells(...)`` (``<mergeCells><mergeCell .../>``),
  frozen panes, data validation dropdowns, live Excel formulas + cached values,
  and three authoritative sheets: ``Summary``, ``Layer_Operator_Breakdown``,
  and ``Memory_Quant_Calculator``.
[Invariants] Every timing value comes strictly from ``runtime.Engine`` via
  ``ExportBundle`` (zero ``0.80x`` heuristic fudge factors). Merged cell ranges
  are checked non-overlapping and store numeric/formula operands exclusively in
  the top-left anchor cell so Excel ``SUM`` formulas never double-count 1-to-N
  fused-to-unfused groups. Every ``<worksheet>`` follows strict ECMA-376 child
  element order (``dimension``, ``sheetViews``, ``sheetFormatPr``, ``cols``,
  ``sheetData``, ``mergeCells``, ``dataValidations``, ``pageMargins``). Atomic
  writes use same-directory temp files + ``os.replace``.
[Timeouts & Locks] Finite in-memory XML serialization bounded by operator and
  weight rows; no locks, network, or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` output path and
  cleans up private temp files on error.
"""
from __future__ import annotations

from collections.abc import Sequence
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
    ExportBundle,
    FP32_BYTES,
    GIB_BYTES,
    KV_QUANT_GROUP_SIZE,
    MIB_BYTES,
    MemoryCalculatorLedger,
    OperatorExportRow,
    SUPPORTED_CACHE_DTYPES,
    SUPPORTED_WEIGHT_QUANTS,
    US_PER_MS,
    WEIGHT_QUANT_FORMATS,
    build_export_bundle,
)

#: Schema tag for XLSX export receipts.
XLSX_EXPORT_RECEIPT_SCHEMA = "mindie-sim.xlsx-export-receipt/1"
#: OOXML SpreadsheetML and Relationships XML namespaces.
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"

#: Excel column alphabet radix and ASCII anchor.
EXCEL_COL_RADIX = 26
ASCII_UPPER_A = 65
#: Default row height (pt) and Excel calc engine revision identifier.
DEFAULT_ROW_HEIGHT_PT = 18
EXCEL_CALC_ID = 191029
#: Page margin constants (inches).
PAGE_MARGIN_SIDE = 0.7
PAGE_MARGIN_TOP_BOTTOM = 0.75
PAGE_MARGIN_HEADER_FOOTER = 0.3
#: Default/min/max column width bounds in character units.
MIN_COL_WIDTH = 10.0
MAX_COL_WIDTH = 44.0
COL_WIDTH_PADDING = 3.0
#: Gap-ratio visual severity thresholds for Gap Radar highlighting.
GAP_RATIO_SEVERE = 3.0
GAP_RATIO_MODERATE = 1.5
#: Cache bytes/elem approximations matching closed-form Excel formulas.
C8_CACHE_FACTOR = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
#: Regex matching illegal XML 1.0 control characters.
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")

#: Style indices registered in ``xl/styles.xml`` (must match ``_build_styles_xml``).
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

#: Sheet names of the 3-sheet executive workbook.
SHEET_SUMMARY = "Summary"
SHEET_BREAKDOWN = "Layer_Operator_Breakdown"
SHEET_MEMORY = "Memory_Quant_Calculator"

#: Sheet 1 (Summary) structural layout constants.
SUMMARY_COLS = 11
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

#: Sheet 2 (Layer_Operator_Breakdown) structural layout constants.
BREAKDOWN_COLS = 18
BREAKDOWN_DATA_START_ROW = 3
BREAKDOWN_SIM_START_COL = 3
BREAKDOWN_KERNEL_NAME_COL = 14
BREAKDOWN_KERNEL_US_COL = 15
BREAKDOWN_SUM_US_COL = 16
BREAKDOWN_APE_COL = 17
BREAKDOWN_BOTTLENECK_COL = 18
BREAKDOWN_SIM_MERGE_COLS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18)
BREAKDOWN_TOTAL_SUM_COLS = (8, 9, 10, 12, 13, 15, 16)

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

#: Ordered quant format lookup rows emitted in Sheet 3 ``S4:V10``.
MEMORY_QUANT_LOOKUP_ORDER = ("BF16", "FP32", "FP8", "W8A8", "W4A8", "MXFP8", "MXFP4")

#: Summary & Breakdown column width presets.
SUMMARY_COL_WIDTHS = (8.0, 14.0, 16.0, 28.0, 18.0, 24.0, 16.0, 15.0, 15.0, 16.0, 16.0)
BREAKDOWN_COL_WIDTHS = (
    12.0, 14.0, 28.0, 16.0, 24.0, 18.0, 10.0, 15.0, 15.0,
    15.0, 11.0, 16.0, 14.0, 28.0, 16.0, 16.0, 12.0, 16.0,
)
MEMORY_COL_WIDTHS = (
    16.0, 24.0, 12.0, 12.0, 18.0, 12.0, 12.0, 12.0, 13.0, 15.0, 15.0,
    13.0, 14.0, 16.0, 4.0, 4.0, 26.0, 14.0, 12.0, 12.0, 11.0, 12.0,
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
            f'<sheetFormatPr defaultRowHeight="{DEFAULT_ROW_HEIGHT_PT}"/>',
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
        '<fonts count="6">'
        '<font><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><sz val="13"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="11"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1F2937"/><name val="Arial"/></font>'
        '<font><sz val="10"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        "</fonts>"
        '<fills count="10">'
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
        '<cellXfs count="19">'
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
    for offset, label in enumerate(labels):
        ws.set_cell(row, start_col + offset, label, style=STYLE_HEADER)


def _populate_summary_sheet(
    ws: Worksheet,
    bundle: ExportBundle,
    *,
    compare_bundle: ExportBundle | None = None,
) -> None:
    _apply_col_widths(ws, SUMMARY_COL_WIDTHS)
    ws.freeze_panes(2, 0)
    _write_banner(
        ws, 1, SUMMARY_COLS,
        f"MindIE-Sim Executive Performance & Physical-Limit Gap Summary | {bundle.model_name} @ {bundle.hardware_sku}",
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
        ("Graph ID", None, STYLE_MERGED),
        (bundle.graph_id[:SUMMARY_GRAPH_PREVIEW_LEN], None, STYLE_TEXT),
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
        (latency_label, step_ms, None, STYLE_FLOAT3, "ms", "runtime.Engine"),
        ("Step Latency", bundle.step_span_us, "=B6*1000", STYLE_FLOAT2, "us", "Exact Engine Span"),
        ("Operator Sum", bundle.operator_sum_us, None, STYLE_FLOAT2, "us", "Serial Op Sum"),
        ("Overlap Speedup", overlap_ratio, "=IF(B7>0,B8/B7,1)", STYLE_RATIO, "x", "OpSum / Step"),
        ("Throughput (Total)", bundle.throughput_tokens_per_s, None, STYLE_FLOAT2, "tok/s", "Cluster"),
        ("Throughput (Per Die)", bundle.throughput_per_die_tokens_per_s, None, STYLE_FLOAT2, "tok/s/die", f"{mem.total_dies} dies"),
        ("Lean 4 L* Floor", bundle.step_l_star_lean_us, None, STYLE_FLOAT2, "us", f"Cert={bundle.step_certificate_valid}"),
        ("Step Gap (T_pred - L*)", bundle.step_span_us - bundle.step_l_star_lean_us, "=B7-B12", STYLE_FLOAT2, "us", ",".join(bundle.step_tight_resources) or "bound"),
        ("Physical-Limit Eff (L*/T)", phys_eff, "=IF(B7>0,B12/B7,0)", STYLE_PCT, "%", f"L*={l_star_ms:.3f} ms"),
        ("Network Cube MFU", bundle.network_cube_mfu / 100.0, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network Vector MFU", bundle.network_vec_mfu / 100.0, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network HBM MBU", bundle.network_mbu / 100.0, None, STYLE_PCT, "%", "4D Efficiency"),
        ("Network Link Util", bundle.network_link_util / 100.0, None, STYLE_PCT, "%", "4D Efficiency"),
    )
    for offset, (label, val, formula, st, unit, basis) in enumerate(kpi_rows):
        r = SUMMARY_DATA_START_ROW + offset
        _write_row(ws, r, (
            (label, None, STYLE_TEXT),
            (val, formula, st),
            (unit, None, STYLE_TEXT),
            (basis, None, STYLE_TEXT),
        ))
        ws.merge_range(r, SUMMARY_BASIS_START_COL, r, SUMMARY_LEFT_END_COL)

    _write_headers(
        ws, SUMMARY_HEADER_ROW,
        ("Bottleneck Class", "Ops", "Duration (us)", "Share", "Roofline Eff", "Exposed (us)"),
        start_col=SUMMARY_RIGHT_START_COL,
    )
    bound_keys = ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")
    exposed_map = {
        "memory_bound": bundle.exposed_breakdown_us.get("memory", 0.0),
        "cube_bound": bundle.exposed_breakdown_us.get("cube", 0.0),
        "vector_bound": bundle.exposed_breakdown_us.get("vector", 0.0),
        "link_bound": bundle.exposed_breakdown_us.get("communication", 0.0),
        "launch_bound": 0.0,
    }
    b_start = SUMMARY_DATA_START_ROW
    b_end = b_start + len(bound_keys) - 1
    b_tot_row = b_end + 1
    tot_b_us = sum(float((bundle.bottleneck_mix.get(k) or {}).get("duration_us", 0.0)) for k in bound_keys)
    tot_b_ops = sum(int((bundle.bottleneck_mix.get(k) or {}).get("count", 0)) for k in bound_keys)
    tot_exp_us = sum(exposed_map.values())
    for offset, b_key in enumerate(bound_keys):
        r = b_start + offset
        info = bundle.bottleneck_mix.get(b_key) or {}
        dur_u = float(info.get("duration_us", 0.0))
        share = (dur_u / tot_b_us) if tot_b_us > 0.0 else 0.0
        _write_row(ws, r, (
            (b_key, None, STYLE_TEXT),
            (int(info.get("count", 0)), None, STYLE_INT),
            (dur_u, None, STYLE_FLOAT2),
            (share, f"=IF(H{b_tot_row}>0,H{r}/$H${b_tot_row},0)", STYLE_PCT),
            (float(info.get("roofline_eff_pct", 0.0)) / 100.0, None, STYLE_PCT),
            (exposed_map[b_key], None, STYLE_FLOAT2),
        ), start_col=SUMMARY_RIGHT_START_COL)
    _write_row(ws, b_tot_row, (
        ("Total", None, STYLE_TOTAL_TEXT),
        (tot_b_ops, f"=SUM(G{b_start}:G{b_end})", STYLE_TOTAL_NUM),
        (tot_b_us, f"=SUM(H{b_start}:H{b_end})", STYLE_TOTAL_NUM),
        (1.0 if tot_b_us > 0.0 else 0.0, f"=SUM(I{b_start}:I{b_end})", STYLE_TOTAL_PCT),
        (bundle.network_roofline_efficiency / 100.0, None, STYLE_TOTAL_PCT),
        (tot_exp_us, f"=SUM(K{b_start}:K{b_end})", STYLE_TOTAL_NUM),
    ), start_col=SUMMARY_RIGHT_START_COL)

    next_row = _populate_summary_delta_block(ws, b_tot_row + 2, bundle, compare_bundle)
    radar_start = max(SUMMARY_DATA_START_ROW + len(kpi_rows) + 1, next_row + 1)
    _populate_gap_radar_block(ws, radar_start, bundle)


def _populate_summary_delta_block(
    ws: Worksheet,
    start_row: int,
    bundle: ExportBundle,
    compare_bundle: ExportBundle | None,
) -> int:
    c_start = SUMMARY_RIGHT_START_COL
    if compare_bundle is not None:
        ws.set_cell(start_row, c_start, "3. Before vs After Comparison Delta", style=STYLE_SECTION)
        ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
        _write_headers(ws, start_row + 1, ("Metric", "Before", "After", "Delta", "Delta %", "Speedup"), start_col=c_start)
        pairs = (
            ("Step Latency (us)", bundle.step_span_us, compare_bundle.step_span_us),
            ("Lean 4 L* Floor (us)", bundle.step_l_star_lean_us, compare_bundle.step_l_star_lean_us),
            ("Throughput (tok/s)", bundle.throughput_tokens_per_s, compare_bundle.throughput_tokens_per_s),
            ("Max Stage HBM (GiB)", bundle.memory_ledger.max_stage_hbm_bytes / GIB_BYTES,
             compare_bundle.memory_ledger.max_stage_hbm_bytes / GIB_BYTES),
        )
        for idx, (label, before, after) in enumerate(pairs):
            r = start_row + 2 + idx
            delta = after - before
            pct = (delta / before) if before > 0.0 else 0.0
            spd = (before / after) if after > 0.0 else 0.0
            _write_row(ws, r, (
                (label, None, STYLE_TEXT),
                (before, None, STYLE_FLOAT2),
                (after, None, STYLE_FLOAT2),
                (delta, f"=H{r}-G{r}", STYLE_FLOAT2),
                (pct, f"=IF(G{r}>0,(H{r}-G{r})/G{r},0)", STYLE_PCT),
                (spd, f"=IF(H{r}>0,G{r}/H{r},0)", STYLE_RATIO),
            ), start_col=c_start)
        return start_row + 2 + len(pairs)

    if bundle.reconciliation is not None:
        rec = bundle.reconciliation
        ws.set_cell(start_row, c_start, f"3. Sim Pred vs Measured Profiler ({rec.source_label})", style=STYLE_SECTION)
        ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
        _write_headers(ws, start_row + 1, ("Reconcile Metric", "Sim Pred", "Measured", "Delta", "APE %", "Coverage"), start_col=c_start)
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
        ), start_col=c_start)
        r2 = r + 1
        _write_row(ws, r2, (
            ("Median / P90 APE (%)", None, STYLE_TEXT),
            ((rec.median_ape_pct or 0.0) / 100.0, None, STYLE_PCT),
            ((rec.p90_ape_pct or 0.0) / 100.0, None, STYLE_PCT),
            (rec.unmatched_measured_us, None, STYLE_FLOAT2),
            ((rec.within_gate_pct or 0.0) / 100.0, None, STYLE_PCT),
            (f"{rec.within_gate_count}/{rec.matched_sim_ops} <=10%/2us", None, STYLE_TEXT),
        ), start_col=c_start)
        return r2 + 1

    ws.set_cell(start_row, c_start, "3. Physical-Limit Headroom Summary (Standalone)", style=STYLE_SECTION)
    ws.merge_range(start_row, c_start, start_row, SUMMARY_COLS)
    _write_headers(ws, start_row + 1, ("Metric", "Sim Pred", "Lean L*", "Gap (us)", "Gap Ratio", "Certificate"), start_col=c_start)
    r = start_row + 2
    gap_us = bundle.step_span_us - bundle.step_l_star_lean_us
    ratio = (bundle.step_span_us / bundle.step_l_star_lean_us) if bundle.step_l_star_lean_us > 0.0 else 0.0
    _write_row(ws, r, (
        ("Whole Step (us)", None, STYLE_TEXT),
        (bundle.step_span_us, None, STYLE_FLOAT2),
        (bundle.step_l_star_lean_us, None, STYLE_FLOAT2),
        (gap_us, f"=G{r}-H{r}", STYLE_FLOAT2),
        (ratio, f"=IF(H{r}>0,G{r}/H{r},0)", _gap_ratio_style(ratio)),
        ("VERIFIED" if bundle.step_certificate_valid else "UNVERIFIED", None, STYLE_TEXT),
    ), start_col=c_start)
    return r + 1


def _populate_gap_radar_block(ws: Worksheet, start_row: int, bundle: ExportBundle) -> None:
    _write_banner(
        ws, start_row, SUMMARY_COLS,
        "4. Top Physical-Limit Gap Radar (Ranked by T_pred - L*_Lean and T_pred / L*_Lean)",
        STYLE_SECTION,
    )
    headers = (
        "Rank", "Layer", "Submodule", "Op ID", "Family", "Implementation",
        "Bottleneck", "L*_Lean (us)", "Sim Pred (us)", "Gap (T_pred - L*, us)", "Gap Ratio (T/L*)",
    )
    hdr_row = start_row + 1
    _write_headers(ws, hdr_row, headers)
    for offset, entry in enumerate(bundle.gap_radar):
        r = hdr_row + 1 + offset
        _write_row(ws, r, (
            (entry.rank, None, STYLE_INT),
            (entry.layer_label, None, STYLE_TEXT),
            (entry.submodule, None, STYLE_TEXT),
            (entry.op_id, None, STYLE_TEXT),
            (entry.family, None, STYLE_TEXT),
            (entry.implementation, None, STYLE_TEXT),
            (entry.dominant_bound, None, STYLE_TEXT),
            (entry.l_star_lean_us, None, STYLE_FLOAT3),
            (entry.pred_us, None, STYLE_FLOAT3),
            (entry.gap_us, f"=I{r}-H{r}", STYLE_FLOAT3),
            (entry.gap_ratio if entry.gap_ratio is not None else 0.0, f"=IF(H{r}>0,I{r}/H{r},0)", _gap_ratio_style(entry.gap_ratio)),
        ))


def _format_shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(d)) for d in shape) if shape else "scalar"


def _populate_breakdown_sheet(ws: Worksheet, bundle: ExportBundle) -> None:
    _apply_col_widths(ws, BREAKDOWN_COL_WIDTHS)
    ws.freeze_panes(2, 2)
    _write_banner(
        ws, 1, BREAKDOWN_COLS,
        f"Layer -> Submodule -> 1-to-N Kernel Breakdown | {bundle.model_name} @ {bundle.hardware_sku} ({bundle.phase})",
        STYLE_TITLE,
    )
    headers = (
        "Layer", "Submodule", "Sim Op ID", "Family", "Implementation", "Shapes",
        "Dtype", "FLOPs", "HBM Read (B)", "HBM Write (B)", "L2 Hit %",
        "L* Physical Floor (us)", "Sim Pred (us)", "Measured Kernel (msprof)",
        "Kernel Measured (us)", "Measured Sum (us)", "APE %", "Bottleneck",
    )
    _write_headers(ws, 2, headers)

    row_cursor = BREAKDOWN_DATA_START_ROW
    layer_spans: list[tuple[str, int, int]] = []
    sub_spans: list[tuple[tuple[str, str], int, int]] = []
    for op in bundle.operators:
        k_count = max(1, len(op.measured_kernels))
        r_start = row_cursor
        r_end = row_cursor + k_count - 1
        _write_breakdown_op_anchor(ws, r_start, r_end, op)
        if k_count > 1:
            for col in BREAKDOWN_SIM_MERGE_COLS:
                ws.merge_range(r_start, col, r_end, col)
        _extend_span(layer_spans, op.layer_label, r_start, r_end)
        _extend_span(sub_spans, (op.layer_label, op.submodule), r_start, r_end)
        row_cursor = r_end + 1

    for label, r1, r2 in layer_spans:
        ws.set_cell(r1, 1, label, style=STYLE_MERGED)
        ws.merge_range(r1, 1, r2, 1)
    for (_, sub_label), r1, r2 in sub_spans:
        ws.set_cell(r1, 2, sub_label, style=STYLE_MERGED)
        ws.merge_range(r1, 2, r2, 2)

    data_end = row_cursor - 1
    tot_row = row_cursor
    ws.set_cell(tot_row, 1, "TOTAL", style=STYLE_TOTAL_TEXT)
    ws.merge_range(tot_row, 1, tot_row, 2)
    ws.set_cell(tot_row, BREAKDOWN_SIM_START_COL, f"{len(bundle.operators)} sim ops", style=STYLE_TOTAL_TEXT)
    total_vals = (
        sum(op.flops for op in bundle.operators),
        sum(op.hbm_read_bytes for op in bundle.operators),
        sum(op.hbm_write_bytes for op in bundle.operators),
        sum(op.l_star_lean_us for op in bundle.operators),
        sum(op.dur_us for op in bundle.operators),
        sum(k.duration_us for op in bundle.operators for k in op.measured_kernels),
        sum(op.measured_sum_us or 0.0 for op in bundle.operators),
    )
    for col, total_val in zip(BREAKDOWN_TOTAL_SUM_COLS, total_vals):
        c_let = column_letter(col)
        formula = (
            f"=SUM({c_let}{BREAKDOWN_DATA_START_ROW}:{c_let}{data_end})"
            if data_end >= BREAKDOWN_DATA_START_ROW else "=0"
        )
        ws.set_cell(tot_row, col, total_val, formula=formula, style=STYLE_TOTAL_NUM)
    meas_tot = sum(op.measured_sum_us or 0.0 for op in bundle.operators)
    sim_matched_tot = sum(op.dur_us for op in bundle.operators if op.measured_sum_us is not None)
    agg_ape = (abs(sim_matched_tot - meas_tot) / meas_tot) if meas_tot > 0.0 else 0.0
    ws.set_cell(
        tot_row, BREAKDOWN_APE_COL, agg_ape,
        formula=f"=IF(P{tot_row}>0,ABS(M{tot_row}-P{tot_row})/P{tot_row},0)",
        style=STYLE_TOTAL_PCT,
    )


def _write_breakdown_op_anchor(
    ws: Worksheet, r_start: int, r_end: int, op: OperatorExportRow,
) -> None:
    _write_row(ws, r_start, (
        (op.op_id, None, STYLE_TEXT),
        (op.family, None, STYLE_TEXT),
        (op.implementation, None, STYLE_TEXT),
        (_format_shape_text(op.shape), None, STYLE_TEXT),
        (op.dtype, None, STYLE_TEXT),
        (op.flops, None, STYLE_INT),
        (op.hbm_read_bytes, None, STYLE_INT),
        (op.hbm_write_bytes, None, STYLE_INT),
        (op.l2_hit_pct / 100.0, None, STYLE_PCT),
        (op.l_star_lean_us, None, STYLE_FLOAT3),
        (op.dur_us, None, STYLE_FLOAT3),
    ), start_col=BREAKDOWN_SIM_START_COL)
    if op.measured_kernels:
        for offset, k_row in enumerate(op.measured_kernels):
            ws.set_cell(r_start + offset, BREAKDOWN_KERNEL_NAME_COL, k_row.name, style=STYLE_TEXT)
            ws.set_cell(r_start + offset, BREAKDOWN_KERNEL_US_COL, k_row.duration_us, style=STYLE_FLOAT3)
        ws.set_cell(
            r_start, BREAKDOWN_SUM_US_COL, op.measured_sum_us,
            formula=f"=SUM(O{r_start}:O{r_end})", style=STYLE_FLOAT3,
        )
        ws.set_cell(
            r_start, BREAKDOWN_APE_COL, (op.ape_pct / 100.0) if op.ape_pct is not None else 0.0,
            formula=f"=IF(P{r_start}>0,ABS(M{r_start}-P{r_start})/P{r_start},0)",
            style=STYLE_PCT,
        )
    else:
        _write_row(ws, r_start, (
            ("-", None, STYLE_TEXT),
            (None, None, STYLE_FLOAT3),
            (None, None, STYLE_FLOAT3),
            (None, None, STYLE_PCT),
        ), start_col=BREAKDOWN_KERNEL_NAME_COL)
    ws.set_cell(r_start, BREAKDOWN_BOTTLENECK_COL, op.dominant_bound, style=STYLE_TEXT)


def _extend_span(spans: list[tuple[Any, int, int]], key: Any, r1: int, r2: int) -> None:
    if spans and spans[-1][0] == key:
        spans[-1] = (key, spans[-1][1], r2)
    else:
        spans.append((key, r1, r2))


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
        f"Closed-Form Memory / Quant / Cache / Engram-PLE / PP HBM Calculator | {mem.model_name} @ {mem.hardware_sku}",
        STYLE_TITLE,
    )
    _populate_memory_control_panel(ws, mem)

    ws.set_cell(2, 1, "1. Per-Die Weight + Quant Scale Closed-Form Table (Post-TP Local Shapes)", style=STYLE_SECTION)
    ws.merge_range(2, 1, 2, MEMORY_TABLE_END_COL)
    w_headers = (
        "Group", "Tensor Name", "Global Rows", "Global Cols", "Shard Rule",
        "Local Rows", "Local Cols", "Max Stage Rep", "Quant Format",
        "Payload (B)", "MX Scale (B)", "Channel Aux (B)", "Per-Die (GiB)", "Residency",
    )
    _write_headers(ws, MEMORY_TABLE_HEADER_ROW, w_headers)

    lookup_ref = f"$S${MEMORY_LOOKUP_FIRST_ROW}:$V${MEMORY_LOOKUP_LAST_ROW}"
    w_start = MEMORY_TABLE_HEADER_ROW + 1
    group_spans: list[tuple[str, int, int]] = []
    for idx, row in enumerate(mem.weight_rows):
        r = w_start + idx
        _extend_span(group_spans, row.group, r, r)
        res_text = ("Host DRAM" if mem.engram_host_offload else "NPU HBM") if row.is_engram_or_ple else "NPU HBM"
        res_formula = '=IF($R$5="ON","Host DRAM","NPU HBM")' if row.is_engram_or_ple else None
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
            (row.batch_total_bytes / float(GIB_BYTES), f"=E{r}*$R$16/1024", STYLE_FLOAT3),
            (basis, None, STYLE_TEXT),
        ))
        ws.merge_range(r, MEMORY_CACHE_BASIS_START_COL, r, MEMORY_TABLE_END_COL)

    c_end = c_start + len(mem.cache_rows) - 1
    c_tot = c_end + 1
    ws.set_cell(c_tot, 1, "Cache Total (Max Stage)", style=STYLE_TOTAL_TEXT)
    ws.merge_range(c_tot, 1, c_tot, MEMORY_CACHE_MERGE_END_COL)
    ws.set_cell(
        c_tot, MEMORY_CACHE_REQ_COL, sum(x.per_request_bytes for x in mem.cache_rows) / float(MIB_BYTES),
        formula=f"=SUM(E{c_start}:E{c_end})" if c_end >= c_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BATCH_COL, sum(x.batch_total_bytes for x in mem.cache_rows) / float(GIB_BYTES),
        formula=f"=SUM(F{c_start}:F{c_end})" if c_end >= c_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    ws.set_cell(
        c_tot, MEMORY_CACHE_BASIS_START_COL,
        f"Local tokens={mem.local_cache_tokens}, Concurrency={mem.concurrency}",
        style=STYLE_TOTAL_TEXT,
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
        formula=f"=MAX(N{s_start}:N{s_end})" if s_end >= s_start else "=0",
        style=STYLE_TOTAL_NUM,
    )
    return max_row


def export_comparison_xlsx(
    source: ExportBundle | CaseInput,
    output_path: str | Path,
    *,
    compare_source: ExportBundle | CaseInput | None = None,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
) -> XlsxExportSummary:
    """Build and atomically save the 3-sheet ``merge_cells`` comparison ``.xlsx`` workbook."""
    bundle = (
        source if isinstance(source, ExportBundle)
        else build_export_bundle(
            source,
            request=request,
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
                request=request or StepRequest(
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

    wb = Workbook()
    ws_summary = wb.create_sheet(SHEET_SUMMARY)
    ws_breakdown = wb.create_sheet(SHEET_BREAKDOWN)
    ws_memory = wb.create_sheet(SHEET_MEMORY)

    _populate_summary_sheet(ws_summary, bundle, compare_bundle=compare_bundle)
    _populate_breakdown_sheet(ws_breakdown, bundle)
    _populate_memory_sheet(ws_memory, bundle)

    bytes_written = wb.save(output_path)
    return XlsxExportSummary(
        schema=XLSX_EXPORT_RECEIPT_SCHEMA,
        output_path=str(output_path),
        model_name=bundle.model_name,
        hardware_sku=bundle.hardware_sku,
        phase=bundle.phase,
        graph_id=bundle.graph_id,
        sheets=(SHEET_SUMMARY, SHEET_BREAKDOWN, SHEET_MEMORY),
        merge_cell_counts={
            SHEET_SUMMARY: len(ws_summary.merged_ranges),
            SHEET_BREAKDOWN: len(ws_breakdown.merged_ranges),
            SHEET_MEMORY: len(ws_memory.merged_ranges),
        },
        operator_rows=len(bundle.operators),
        weight_rows=len(bundle.memory_ledger.weight_rows),
        has_comparison=compare_bundle is not None,
        has_measured=bundle.reconciliation is not None,
        bytes_written=bytes_written,
    )


__all__ = [
    "SHEET_BREAKDOWN",
    "SHEET_MEMORY",
    "SHEET_SUMMARY",
    "Workbook",
    "Worksheet",
    "XLSX_EXPORT_RECEIPT_SCHEMA",
    "XlsxExportSummary",
    "cell_ref",
    "column_letter",
    "export_comparison_xlsx",
]
