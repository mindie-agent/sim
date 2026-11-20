"""Shared parametrize axes: the only home of SKU / ISA / phase literals in tests.

[Role] Give every domain one canonical way to sweep hardware generations and
  ISAs, so test files and functions never carry ``a2`` / ``a5`` / ``arch35``
  tokens in their names (``check_test_orthogonality.py`` TO001/TO009/TO012).
[Invariants] ``SKUS`` ids are the short generation names (``a2``/``a3``/``a5``)
  so ``pytest -k a5`` keeps working; ``ALL_SKUS`` mirrors every profile in
  ``mindie_sim/backends/profiles/ascend.json`` (16 entries, one per id).
[Timeouts & Locks] None.
[Side-Effect Policy] Reads the packaged profile JSON once at import.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROFILES = Path(__file__).resolve().parents[2] / "mindie_sim" / "backends" / "profiles" / "ascend.json"

#: One representative SKU per hardware generation (DAV_2201 x2, DAV_3510).
REPRESENTATIVE_SKUS = {
    "a2": "Ascend910B4",
    "a3": "Ascend910_9391",
    "a5": "Ascend950DT_9581",
}
SKUS = tuple(pytest.param(name, id=gen) for gen, name in REPRESENTATIVE_SKUS.items())

#: Compute-architecture identifiers declared by ``ComputeArchSpec`` profiles.
ARCH_IDS = (pytest.param("npu_arch_2201", id="arch22"), pytest.param("npu_arch_3510", id="arch35"))

PHASES = (pytest.param("prefill", id="prefill"), pytest.param("decode", id="decode"))


def all_sku_names() -> tuple[str, ...]:
    """Every packaged Ascend profile name, in declaration order."""
    return tuple(json.loads(_PROFILES.read_text(encoding="utf-8")))


ALL_SKUS = tuple(pytest.param(name, id=name) for name in all_sku_names())
