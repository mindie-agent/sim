"""npu-smi snapshot types and parsers for Layer 1 hardware analysis.

[Role]
  Parse ``npu-smi info``, ``npu-smi info -t common`` and board captures into
  typed snapshots.  Runtime capacity and chip metadata stay separate from CANN
  platform metadata owned by ``hardware``.

[Invariants]
  * Parsing never fabricates a device row or a runtime capacity; unsupported
    lines are skipped and an empty info capture raises a stable error.
  * ``hbm_total_mib`` is evidence from npu-smi, not the platform .ini value.
  * A3 dual-die ``Chip ID``/``Phy-ID`` pairs remain distinct per device row.

[Timeouts & Locks]
  Pure in-memory parsing.  File reads happen only in ``load_npu_smi_snapshot``
  over the caller's explicit path; no locks, sleeps, network or subprocesses.

[Side-Effect Policy]
  Reads only explicit paths supplied by the caller.  Never writes files,
  changes clocks, or invokes npu-smi itself.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .hardware_common import HardwareError

_BUS_RE = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.")
_PAIR_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass(frozen=True)
class NpuSmiDevice:
    npu_id: int
    chip_id: int
    phy_id: int | None
    name: str
    health: str | None
    power_w: float | None
    temp_c: float | None
    aicore_pct: float | None
    memory_used_mib: int | None
    memory_total_mib: int | None
    hbm_used_mib: int | None
    hbm_total_mib: int | None
    bus_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NpuSmiChip:
    npu_id: int
    chip_id: int
    hbm_usage_rate_pct: float | None
    aicore_freq_mhz: float | None
    aicore_current_freq_mhz: float | None
    aicore_count: int | None
    temp_c: float | None
    power_w: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NpuSmiSnapshot:
    devices: tuple[NpuSmiDevice, ...]
    chips: tuple[NpuSmiChip, ...]
    board: Mapping[str, str]
    source: str = "<text>"

    @property
    def chip_count(self) -> int:
        raw = self.board.get("Chip Count", "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return max((device.chip_id for device in self.devices), default=-1) + 1

    @property
    def dual_die(self) -> bool:
        return self.chip_count > 1

    @property
    def phy_ids(self) -> tuple[int, ...]:
        return tuple(sorted({d.phy_id for d in self.devices if d.phy_id is not None}))

    @property
    def hbm_totals_mib(self) -> tuple[int, ...]:
        return tuple(sorted({d.hbm_total_mib for d in self.devices if d.hbm_total_mib}))

    @property
    def min_hbm_mib(self) -> float | None:
        values = self.hbm_totals_mib
        return float(min(values)) if values else None

    @property
    def max_hbm_mib(self) -> float | None:
        values = self.hbm_totals_mib
        return float(max(values)) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chip_count": self.chip_count,
            "dual_die": self.dual_die,
            "phy_ids": list(self.phy_ids),
            "verified_runtime_hbm_mib": self.min_hbm_mib,
            "verified_runtime_hbm_mib_max": self.max_hbm_mib,
            "devices": [device.to_dict() for device in self.devices],
            "chips": [chip.to_dict() for chip in self.chips],
            "board": dict(sorted(self.board.items())),
        }


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_npu_smi_info(text: str, *, source: str = "<npu-smi-info>") -> NpuSmiSnapshot:
    devices: list[NpuSmiDevice] = []
    pending: tuple[int, str, str | None, float | None, float | None] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        first = cells[0].split()
        if not first or not first[0].isdigit():
            continue
        if first[0].lower() in {"npu", "chip"}:
            continue
        if len(first) >= 2 and not first[1].isdigit():
            stats = cells[2].split()
            pending = (
                int(first[0]),
                first[1],
                cells[1] or None,
                _number(stats[0]) if stats else None,
                _number(stats[1]) if len(stats) > 1 else None,
            )
            continue
        bus_index = next((i for i, cell in enumerate(cells) if _BUS_RE.match(cell)), None)
        if bus_index is None or pending is None:
            continue
        chip_id = int(first[0])
        phy_id = int(first[1]) if len(first) > 1 and first[1].isdigit() else None
        stats = cells[bus_index + 1].split() if len(cells) > bus_index + 1 else []
        pairs = _PAIR_RE.findall(cells[bus_index + 1]) if len(cells) > bus_index + 1 else []
        memory = (int(pairs[0][0]), int(pairs[0][1])) if pairs else (None, None)
        hbm = (int(pairs[1][0]), int(pairs[1][1])) if len(pairs) > 1 else (None, None)
        devices.append(
            NpuSmiDevice(
                npu_id=pending[0],
                chip_id=chip_id,
                phy_id=phy_id,
                name=pending[1],
                health=pending[2],
                power_w=pending[3],
                temp_c=pending[4],
                aicore_pct=_number(stats[0]) if stats else None,
                memory_used_mib=memory[0],
                memory_total_mib=memory[1],
                hbm_used_mib=hbm[0],
                hbm_total_mib=hbm[1],
                bus_id=cells[bus_index],
            )
        )
    if not devices:
        raise HardwareError("npu_smi_info_no_devices")
    return NpuSmiSnapshot(devices=tuple(devices), chips=(), board={}, source=source)


def parse_npu_smi_common(text: str, *, source: str = "<npu-smi-common>") -> tuple[NpuSmiChip, ...]:
    chips: list[NpuSmiChip] = []
    current: dict[str, str] | None = None
    npu_id = 0
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", 1))
        if key == "NPU ID":
            try:
                npu_id = int(value)
            except ValueError:
                npu_id = 0
            continue
        if key == "Chip ID":
            if current is not None:
                chips.append(_chip_from_fields(npu_id, current))
            current = {"Chip ID": value}
            continue
        if current is not None:
            current[key] = value
    if current is not None:
        chips.append(_chip_from_fields(npu_id, current))
    if not chips:
        raise HardwareError("npu_smi_common_no_chips")
    return tuple(chips)


def _chip_from_fields(npu_id: int, fields: Mapping[str, str]) -> NpuSmiChip:
    def value(key: str) -> float | None:
        return _number(fields.get(key))

    def integer(key: str) -> int | None:
        raw = fields.get(key)
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    return NpuSmiChip(
        npu_id=npu_id,
        chip_id=integer("Chip ID") or 0,
        hbm_usage_rate_pct=value("HBM Usage Rate(%)"),
        aicore_freq_mhz=value("Aicore Freq(MHZ)"),
        aicore_current_freq_mhz=value("Aicore curFreq(MHZ)"),
        aicore_count=integer("Aicore Count"),
        temp_c=value("Temperature(C)"),
        power_w=value("NPU Real-time Power(W)"),
    )


def parse_npu_smi_board(text: str) -> dict[str, str]:
    board: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key and value:
            board[key] = value
    if not board:
        raise HardwareError("npu_smi_board_empty")
    return board


def parse_npu_smi_snapshot(
    info_text: str,
    *,
    common_text: str | None = None,
    board_text: str | None = None,
    source: str = "<npu-smi>",
) -> NpuSmiSnapshot:
    info = parse_npu_smi_info(info_text, source=source)
    chips = parse_npu_smi_common(common_text) if common_text else ()
    board = parse_npu_smi_board(board_text) if board_text else {}
    return NpuSmiSnapshot(devices=info.devices, chips=chips, board=board, source=source)


def load_npu_smi_snapshot(path: str | Path) -> NpuSmiSnapshot:
    root = Path(path).expanduser()
    if not root.exists():
        raise HardwareError("npu_smi_path_missing")
    if root.is_dir():
        texts = {p.name: p.read_text(encoding="utf-8-sig", errors="replace") for p in root.glob("*.txt")}
        info = next((text for name, text in texts.items() if "npu-smi" in text and _BUS_RE.search(text)), None)
        common = next((text for name, text in texts.items() if "Chip ID" in text and "Aicore Freq" in text), None)
        board = next((text for name, text in texts.items() if "Product Name" in text and "Chip Count" in text), None)
        if info is None:
            raise HardwareError("npu_smi_info_missing")
        return parse_npu_smi_snapshot(info, common_text=common, board_text=board, source=str(root))
    text = root.read_text(encoding="utf-8-sig", errors="replace")
    if "Product Name" in text and "Chip Count" in text and "npu-smi" not in text:
        return NpuSmiSnapshot(devices=(), chips=(), board=parse_npu_smi_board(text), source=str(root))
    if "Chip ID" in text and "Aicore Freq" in text and "npu-smi" not in text:
        return NpuSmiSnapshot(devices=(), chips=parse_npu_smi_common(text), board={}, source=str(root))
    return parse_npu_smi_info(text, source=str(root))


__all__ = [
    "NpuSmiDevice",
    "NpuSmiChip",
    "NpuSmiSnapshot",
    "parse_npu_smi_info",
    "parse_npu_smi_common",
    "parse_npu_smi_board",
    "parse_npu_smi_snapshot",
    "load_npu_smi_snapshot",
]
