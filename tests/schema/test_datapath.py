"""Datapath capability and ComputeArch plumbing invariants across Ascend ISAs.

[Role] Verify ``ascend.json`` datapath declarations, ``DatapathCapabilitySpec``
  derivation from ``ComputeArchSpec`` + ``Device``, homogeneity checks in
  ``Geometry.from_devices``, and propagation into ``ExecutionBindings`` and
  ``RecipeContext``.
[Invariants] All 12 ``npu_arch_2201`` SKUs forbid ``l0c_to_ub`` / ``ub_to_l1``;
  all 4 ``npu_arch_3510`` SKUs expose both. Mixed-ISA stages raise
  ``ModelingError``.
[Timeouts & Locks] Pure Tier-0 in-memory tests (<2s); no I/O writes or locks.
[Side-Effect Policy] Read-only over repository profile JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import mindie_sim.api as sdk_api
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.evaluation import ExecutionBindings
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

REPO_ROOT = Path(__file__).resolve().parents[2]
ASCEND_PROFILES_PATH = REPO_ROOT / "mindie_sim" / "backends" / "profiles" / "ascend.json"
REPRESENTATIVE_SKUS = (
    ("a2", "Ascend910B2", "npu_arch_2201", False),
    ("a3", "Ascend910_9382", "npu_arch_2201", False),
    ("a5", "Ascend950DT_9581", "npu_arch_3510", True),
)


def _case_for(sku: str):
    return sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": 1}
    )


def test_ascend_profiles_l0c_to_ub_datapath_contract():
    profiles = json.loads(ASCEND_PROFILES_PATH.read_text(encoding="utf-8"))
    assert len(profiles) == 16

    arch22_skus = [
        sku for sku, cfg in profiles.items()
        if cfg["device"]["compute_arch"]["preset"] == "npu_arch_2201"
    ]
    arch35_skus = [
        sku for sku, cfg in profiles.items()
        if cfg["device"]["compute_arch"]["preset"] == "npu_arch_3510"
    ]
    assert len(arch22_skus) == 12
    assert len(arch35_skus) == 4

    for sku in arch22_skus:
        out_paths = profiles[sku]["device"]["physical_mechanisms"]["matrix"]["supported_output_paths"]
        assert "l0c_to_ub" not in out_paths
        dp = datapath_for_die(_case_for(sku).machine.dies[0])
        assert dp.arch_id == "npu_arch_2201"
        assert dp.has_l0c_to_ub is False
        assert dp.has_ub_to_l1 is False
        assert dp.has_path("gm", "ub") is True

    for sku in arch35_skus:
        out_paths = profiles[sku]["device"]["physical_mechanisms"]["matrix"]["supported_output_paths"]
        assert "l0c_to_ub" in out_paths
        dp = datapath_for_die(_case_for(sku).machine.dies[0])
        assert dp.arch_id == "npu_arch_3510"
        assert dp.has_l0c_to_ub is True
        assert dp.has_ub_to_l1 is True


@pytest.mark.parametrize("tier,sku,arch_id,has_direct_ub", REPRESENTATIVE_SKUS)
def test_compute_arch_and_datapath_plumbing_through_bindings_and_context(
    tier, sku, arch_id, has_direct_ub
):
    del tier
    case = _case_for(sku)
    die = case.machine.dies[0]
    geom = Geometry.from_devices(case.machine.dies)
    assert geom.compute_arch == die.compute_arch
    assert geom.datapath is not None
    assert geom.datapath.arch_id == arch_id
    assert geom.datapath.has_l0c_to_ub is has_direct_ub
    assert geom.datapath.has_ub_to_l1 is has_direct_ub

    bindings = ExecutionBindings.create(case)
    assert bindings.compute_arch == die.compute_arch
    assert bindings.datapath_for_rank(0).arch_id == arch_id
    assert bindings.datapath_for_stage(0, 0).arch_id == arch_id

    execution = Execution(case, CostModel(case), bindings=bindings)
    stage_ctx = execution.stage_contexts[(0, 0)]
    assert stage_ctx.compute_arch == die.compute_arch
    assert stage_ctx.datapath == bindings.datapath_for_stage(0, 0)
    assert stage_ctx.effective_aic_cores == die.aic_count
    assert stage_ctx.effective_aiv_cores == die.units("vector")


def test_geometry_rejects_heterogeneous_compute_arch_stage():
    die22 = _case_for("Ascend910B2").machine.dies[0]
    die35 = _case_for("Ascend950DT_9581").machine.dies[0]
    with pytest.raises(ModelingError, match="stage_devices_heterogeneous_compute_arch"):
        Geometry.from_devices((die22, die35))
