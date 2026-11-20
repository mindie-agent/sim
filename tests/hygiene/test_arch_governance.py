"""Architecture governance AST & layout gate (AG001..AG005) verification.

[Role] Verify ``scripts/check_arch_governance.py`` catches flat ISA/CANN module
  sprawl, direct/dynamic/re-exported leaf access, ``service_modes`` and
  ``compute_peaks`` capability sniffing, and missing ISA leaf provenance while
  exempting model catalog JSON configs and legal ``service_rate`` lookups.
[Invariants] Real repository has zero non-allowlisted violations and zero
  violations under ``mindie_sim/prediction/operators/<family>/**``. Multiset
  differential subtraction never waives governed operator packages.
[Timeouts & Locks] Pure Tier-0 local AST tests (<3s); no network or locks.
[Side-Effect Policy] Writes synthetic fixtures only inside pytest ``tmp_path``.
"""
from __future__ import annotations

from pathlib import Path

import scripts.check_arch_governance as arch_gov

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_real_repository_passes_arch_governance_gate():
    real_all = arch_gov.scan_repository(REPO_ROOT)
    assert arch_gov.filter_allowlisted(real_all) == ()
    assert not any(
        len(v.path.parts) == 5 and v.path.parts[:3] == ("mindie_sim", "prediction", "operators")
        for v in real_all
    )


def test_arch_governance_rules_ag001_through_ag005_and_differential_mode(tmp_path):
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "models" / "qwen3.5-35b-a3b.json").write_text("{}", encoding="utf-8")

    pkg = tmp_path / "mindie_sim" / "prediction"
    fam = pkg / "operators" / "demo"
    fam.mkdir(parents=True)
    (pkg / "bad_arch22.py").write_text("X = 1\n", encoding="utf-8")
    (fam / "__init__.py").write_text(
        "from . import arch22\n__all__ = ['arch22']\n", encoding="utf-8"
    )
    (fam / "arch22.py").write_text('"""Missing abi and sources."""\nVAL = 1\n', encoding="utf-8")
    (pkg / "bad_caller.py").write_text(
        "import importlib\n"
        "from mindie_sim.prediction.operators.demo import arch22\n"
        "from mindie_sim.prediction.operators import demo as d\n"
        "def f(ctx):\n"
        "    _ = d.arch22\n"
        "    _ = getattr(d, 'arch35')\n"
        "    importlib.import_module('mindie_sim.prediction.operators.demo.arch22')\n"
        "    if 'mte2_aic_dram' in ctx.service_modes:\n"
        "        return 1\n"
        "    if ctx.service_modes.get('mte2_aiv_dram'):\n"
        "        return 2\n"
        "    if 'fp8' in ctx.compute_peaks:\n"
        "        return 3\n"
        "    if ctx.peak('bf16') > 100.0:\n"
        "        return 4\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (pkg / "good_caller.py").write_text(
        "def ok(ctx):\n"
        "    return ctx.service_rate('mte2_aiv_dram')\n",
        encoding="utf-8",
    )

    found = arch_gov.scan_repository(tmp_path)
    rules_by_file: dict[str, set[str]] = {}
    for item in found:
        rules_by_file.setdefault(item.path.as_posix(), set()).add(item.rule)

    assert "configs/models/qwen3.5-35b-a3b.json" not in rules_by_file
    assert "mindie_sim/prediction/good_caller.py" not in rules_by_file
    assert "AG001" in rules_by_file["mindie_sim/prediction/bad_arch22.py"]
    assert "AG002" in rules_by_file["mindie_sim/prediction/operators/demo/__init__.py"]
    assert "AG005" in rules_by_file["mindie_sim/prediction/operators/demo/arch22.py"]
    assert {"AG002", "AG003", "AG004"} <= rules_by_file["mindie_sim/prediction/bad_caller.py"]

    delta = arch_gov.new_violations(found, found)
    assert len(delta) > 0
    assert all(
        v.path.as_posix().startswith("mindie_sim/prediction/operators/demo/")
        for v in delta
    )
