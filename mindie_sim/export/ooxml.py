"""Zero-dependency ECMA-376 OOXML ``.xlsx`` workbook builder and atomic serializer.

[Role] Provide stdlib-only OOXML SpreadsheetML primitives (``Workbook``,
  ``Worksheet``, validated non-overlapping ``merge_cells``, live formulas with
  cached values, frozen panes, data validation dropdowns, shared strings,
  executive style sheet, and same-directory atomic ZIP persistence).
[Invariants] Merged cell ranges are validated non-overlapping and store numeric
  or formula operands exclusively in the top-left anchor cell so Excel ``SUM``
  formulas never double-count. Every ``<worksheet>`` follows strict ECMA-376
  child element order (``dimension``, ``sheetViews``, ``sheetFormatPr``,
  ``cols``, ``sheetData``, ``mergeCells``, ``dataValidations``, ``pageMargins``).
  ``TimeoutError`` is never swallowed by ``except OSError:``.
[Timeouts & Locks] Pure in-memory XML serialization + atomic ``os.replace``;
  no locks or device contexts.
[Side-Effect Policy] Writes only the caller-requested ``.xlsx`` output path and
  cleans up its private same-directory temporary file on error.
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

#: OOXML SpreadsheetML and Relationships XML namespaces.
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"

#: Excel column alphabet radix, ASCII anchor, and formula limits.
EXCEL_COL_RADIX = 26
ASCII_UPPER_A = 65
EXCEL_MAX_FORMULA_ARGS = 255
EXCEL_MAX_FORMULA_CHARS = 8192
#: Row heights (pt) and Excel calc ID.
DEFAULT_ROW_HEIGHT_PT = 19.0
HEADER_ROW_HEIGHT_PT = 28.0
WRAP_ROW_HEIGHT_PT = 30.0
BACKLOG_ROW_HEIGHT_PT = 34.0
CARD_DESIGN_ROW_HEIGHT_PT = 46.0
CARD_ACCEPT_ROW_HEIGHT_PT = 36.0
EXCEL_CALC_ID = 191029
#: Page margin constants (inches).
PAGE_MARGIN_SIDE = 0.65
PAGE_MARGIN_TOP_BOTTOM = 0.75
PAGE_MARGIN_HEADER_FOOTER = 0.3
#: Column width bounds in character units.
MIN_COL_WIDTH = 8.0
MAX_COL_WIDTH = 56.0
#: Regex matching illegal XML 1.0 control characters and A1 cell refs.
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
STYLE_KPI_HIGHLIGHT = 27
STYLE_OPT_BADGE = 28
STYLE_CARD_HDR = 29
STYLE_MUTED_TEXT = 30
STYLE_PARTIAL_OVERLAP = 31
STYLE_L1_TEXT = 32
STYLE_L1_NUM = 33
STYLE_L1_PCT = 34
STYLE_MASKED_NUM = 35


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

    @property
    def max_row(self) -> int:
        cell_max = max((r for r, _ in self._cells), default=0)
        merge_max = max((r2 for _, _, r2, _, _ in self._merges), default=0)
        return max(cell_max, merge_max)

    @property
    def max_column(self) -> int:
        cell_max = max((c for _, c in self._cells), default=0)
        merge_max = max((c2 for _, _, _, c2, _ in self._merges), default=0)
        width_max = max(self._col_widths, default=0)
        return max(cell_max, merge_max, width_max)

    def validate_max_column(self, limit: int) -> None:
        """Raise ``ValueError`` if populated cells, merges, or widths exceed ``limit``."""
        col_lim = int(limit)
        if self.max_column > col_lim:
            raise ValueError(
                f"worksheet_column_limit_exceeded:{self.title}:{self.max_column}>{col_lim}"
            )

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
        r_idx, c_idx = int(row), int(col)
        if r_idx < 1 or c_idx < 1:
            raise ValueError(f"invalid_xlsx_cell_coord:({r_idx},{c_idx})")
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"non_primitive_xlsx_cell_value:{r_idx}:{c_idx}:{type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non_finite_xlsx_cell_value:{r_idx}:{c_idx}:{value!r}")
        clean_formula = None
        if formula is not None:
            if not isinstance(formula, str):
                raise TypeError(
                    f"non_string_xlsx_formula:{r_idx}:{c_idx}:{type(formula).__name__}"
                )
            clean_formula = formula[1:] if formula.startswith("=") else formula
        self._cells[(r_idx, c_idx)] = _CellRecord(
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
    rels.append(f'<Relationship Id="rId{sheet_count + 1}" Type="{NS_REL_DOC}/styles" Target="styles.xml"/>')
    rels.append(f'<Relationship Id="rId{sheet_count + 2}" Type="{NS_REL_DOC}/sharedStrings" Target="sharedStrings.xml"/>')
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
        '<fonts count="9">'
        '<font><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><sz val="12"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1F2937"/><name val="Arial"/></font>'
        '<font><sz val="10"/><color rgb="FF1F4E78"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF9C3412"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><color rgb="FF1E40AF"/><name val="Arial"/></font>'
        '<font><sz val="9"/><color rgb="FF6B7280"/><name val="Arial"/></font>'
        "</fonts>"
        '<fills count="15">'
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
        '<fill><patternFill patternType="solid"><fgColor rgb="FFDBEAFE"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF334155"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFCCFBF1"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFE0E7FF"/></patternFill></fill>'
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
        '<cellXfs count="36">'
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
        '<xf numFmtId="165" fontId="4" fillId="11" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="7" fillId="11" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="3" fillId="12" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="8" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="13" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="14" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="166" fontId="2" fillId="14" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="167" fontId="2" fillId="14" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="166" fontId="0" fillId="9" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )
