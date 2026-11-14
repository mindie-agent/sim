from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from mindie_sim import __main__ as cli_main
from mindie_sim import custom_hardware, model_catalog


class _Cp936Stdout:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, s: str) -> int:
        self.buf.extend(s.encode("cp936"))
        return len(s)

    def flush(self) -> None:
        pass


def test_cli_read_json_and_emit_utf8_lf_and_cp936_stdout_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "case_utf8.json"
    # Contains characters outside CP936 (e.g. U+26A1 HIGH VOLTAGE, U+03BC GREEK MU)
    payload = {"note": "latency \u03bcs \u26a1 — UTF-8", "value": 42}
    src.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    loaded = cli_main.read_json(src)
    assert loaded == payload

    out_file = tmp_path / "out.json"
    cli_main.emit(payload, out_file)
    raw = out_file.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8")) == payload

    fake_out = _Cp936Stdout()
    monkeypatch.setattr(sys, "stdout", fake_out)
    cli_main.emit(payload, None)
    assert json.loads(fake_out.buf.decode("cp936")) == payload


def test_timeout_error_not_swallowed_in_catalog_and_custom_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sku_file = tmp_path / "skus.json"
    sku_file.write_text("{}", encoding="utf-8")

    def raise_timeout(*_a, **_kw):
        raise TimeoutError("io_deadline")

    monkeypatch.setattr(Path, "open", raise_timeout)
    with pytest.raises(TimeoutError):
        custom_hardware.load_custom_hardware_skus_from_file(sku_file)

    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(sku_file))
    with pytest.raises(TimeoutError):
        custom_hardware.list_custom_hardware_skus()

    monkeypatch.setattr(Path, "read_text", raise_timeout)
    with pytest.raises(TimeoutError):
        model_catalog.load_architecture_profile(next(iter(model_catalog.ARCHITECTURE_PRESET_CATALOG)))
