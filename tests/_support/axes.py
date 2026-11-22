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

#: Stream-plan core-partitioning sweep across DAV_2201 (A2/A3) and DAV_3510 (A5).
STREAM_CORE_CASES = (
    ("Ascend910B2", 24, 48, 16, 32, 16),
    ("Ascend910_9382", 24, 48, 16, 32, 16),
    ("Ascend950DT_9581", 32, 64, 24, 48, 16),
)

#: CLI/SDK catalog compose matrix across generations and models.
CATALOG_COMPOSE_CASES = (
    ("qwen3.5-35b-a3b", 4, 2),
    ("glm-5.3-flash", 4, 4),
    ("deepseek-v4-flash", 8, 2),
)
COMPOSE_HARDWARE = (
    ("a2", "Ascend910B4", 1),
    ("a3", "Ascend910_9392", 2),
    ("a5", "Ascend950DT_9582", 2),
)
ALIASES_BY_SKU = {
    "Ascend910B4": ("a2", "910b"),
    "Ascend910B1": ("910b1",),
    "Ascend910B2": ("910b2",),
    "Ascend910B3": ("910b3",),
    "Ascend910_9392": ("a3", "910c"),
    "Ascend910_9382": ("910_9382",),
    "Ascend950DT_9582": ("a5", "950", "Ascend950DT_9582", "910_9599"),
    "Ascend950DT_9581": ("a5-pod", "atlas-950-superpod"),
    "Ascend950PR_9579": ("a5-pod-pr", "950pr_a5"),
}

#: Topology/SLA sweep SKU pairs for DAV_3510 DT/PR and per-port link models.
DT_SKUS = ("Ascend950DT_9582", "Ascend950PR_9579")
PER_PORT_SKUS = ("Ascend910B1", "Ascend910_9362")

#: Canonical SoC family & generation SKU constants for parametrized sweeps.
SOC_950 = "Ascend950"
SOC_910_93 = "Ascend910_93"
DUAL_DIE_SKU = "Ascend910_9362"
GEN2_SKU = "Ascend910B4"
GEN3_SKU = "Ascend910_9392"
GEN5_PR_SKU = "Ascend950PR_9579"
GEN5_DT_SKU = "Ascend950DT_9582"
GEN5_DT_SMALL_SKU = "Ascend950DT_9581"

BODY_DEFAULT_SKUS = ("Ascend910_9362", "Ascend950DT_9582")
WIDE_SKUS = ("Ascend910B2", "Ascend910_9362", "Ascend950DT_9582")
KDA_SKUS = (
    ("Ascend910_9392", "vllm_ascend_recurrent_kda_arch22"),
    ("Ascend950DT_9582", "vllm_ascend_recurrent_kda_arch35"),
)
KDA_ONLY_SKUS = ("Ascend910_9392", "Ascend950DT_9582")

A2_SKUS = {
    "Ascend910B1": (1.85e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2C": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B3": (1.80e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910B4": (1.50e9, 20, 40, 34359738368, 0.8e12),
    "Ascend910B4-1": (1.50e9, 20, 40, 34359738368, 0.8e12),
}

COMPOSITE_ACCURACY_CAPTURES = (
    ("a2-qwen35-w8a8-ep4-sn", "qwen3.5-35b-a3b", "a2", "w8a8", "Ascend910B4", 2, 2, 4, None, 2, 8, "decode", 1, 128, 1, 1.474388080420759),
    ("a3-9362-qwen35-w8a8-ep8-sn", "qwen3.5-35b-a3b", "Ascend910_9362", "w8a8", "Ascend910_9362", 4, 2, 8, None, 2, 8, "decode", 1, 256, 1, 0.6844563585607575),
    ("a3-9392-glm53-w4a8-ep2-sn", "glm-5.3", "Ascend910_9392", "w4a8", "Ascend910_9392", 2, 1, 2, None, 4, 8, "decode", 1, 128, 1, 0.9199233984208398),
    ("a5-9582-qwen35-fp8-prefill", "qwen3.5-35b-a3b", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 2, 1, 2, None, 2, 8, "prefill", 128, 0, 1, 0.6480762050980904),
    ("a5-9582-qwen397-w4a8", "qwen3.5-397b-a17b", "Ascend950DT_9582", "w4a8", "Ascend950DT_9582", 2, 1, 2, None, 2, 8, "decode", 1, 128, 1, 0.6609402926303547),
    ("a2-dsv32-w8a8-ep4", "deepseek-v3.2", "a2", "w8a8", "Ascend910B4", 2, 2, 4, None, 4, 8, "decode", 1, 128, 1, 6.087898982343473),
    ("a5-v41-fp8-prefill", "deepseek-v4.1", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 2, 1, 2, None, 4, 8, "prefill", 128, 0, 1, 1.5976127185425135),
    ("a3-9362-ep16-sn", "qwen3.5-35b-a3b", "Ascend910_9362", "w8a8", "Ascend910_9362", 4, 4, 16, 16, 2, 16, "decode", 1, 64, 1, 0.7644720422342266),
    ("a2-ep16-mn", "qwen3.5-35b-a3b", "a2", "w8a8", "Ascend910B4", 4, 4, 16, 8, 2, 16, "decode", 1, 64, 1, 2.214654517369641),
    ("a5-9582-ep32-mn-fp8", "qwen3.5-397b-a17b", "Ascend950DT_9582", "fp8", "Ascend950DT_9582", 4, 8, 32, 8, 2, 32, "decode", 1, 64, 1, 2.046621835955975),
    ("a5-9582-ep64-mn", "qwen3.5-35b-a3b", "Ascend950DT_9582", "w8a8", "Ascend950DT_9582", 8, 8, 64, 8, 2, 64, "prefill", 128, 0, 1, 3.5328551527254435),
)
