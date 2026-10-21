"""Campaign machine declaration resolution (blocker P15).

[Role]
  Turn a campaign's ``machine`` declaration into the machine it actually
  names.  ``a3_operator_campaign_conditions.DEFAULT_MACHINE`` is the declared
  **A3** campaign baseline; a declaration that names a SKU of another family
  must come from that SKU's capability profile instead of silently inheriting
  A3 device numbers under a non-A3 key.

[Invariants]
  * No SKU declaration means the A3 baseline (``{**DEFAULT_MACHINE, **declared}``)
    unchanged, so the published A3 campaign numbers do not move.
  * An A3 SKU keeps the tuned A3 baseline; any other family derives every rate
    (dies, cube peak, fp8 peak, vector peak, HBM bandwidth, links) from its own
    profile.
  * An explicit family or cube/vector peak that contradicts the SKU profile is
    a stable ``A3InputError``, never a silent merge.
  * An unresolvable SKU is ``campaign_machine_sku_unresolved:<sku>``.

[Timeouts & Locks]
  Pure in-memory resolution (local SKU catalog arithmetic).  No device, file,
  subprocess, network, lock or sleep.

[Side-Effect Policy]
  No artifact or registry mutation; the caller owns the returned declaration.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_operator_campaign_conditions import (
    DEFAULT_MACHINE, _as_int, _as_positive,
)
from .a3_types import A3InputError, MachineSpec


def resolve_campaign_machine(
    declared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a campaign machine declaration to the machine it names.

    ``DEFAULT_MACHINE`` is the declared **A3** campaign baseline.  A declaration
    naming a SKU of another family cannot inherit it: ``{"sku":
    "Ascend950DT_9582"}`` merged over the A3 literal used to yield an A5 key on
    A3 device numbers and ``fp8 == bf16`` peaks (blocker P15).  Such a
    declaration derives the machine from its SKU profile; an A3 SKU keeps the
    tuned baseline, and a family/peak contradiction is refused.
    """
    override = dict(declared or {})
    sku = str(override.get("sku") or override.get("soc") or "").strip()
    if not sku:
        return {**DEFAULT_MACHINE, **override}
    try:
        profile_machine = MachineSpec.from_mapping({"sku": sku, "num_devices": 1})
    except Exception as exc:  # noqa: BLE001 - one campaign-facing reason
        raise A3InputError(f"campaign_machine_sku_unresolved:{sku}") from exc
    family = str(profile_machine.arch_family or "").strip().upper()
    declared_family = str(override.get("arch_family") or "").strip().upper()
    if declared_family and family and declared_family != family:
        raise A3InputError(
            f"campaign_machine_sku_family_conflict:{sku}:{declared_family}!={family}"
        )
    if family == str(DEFAULT_MACHINE["arch_family"]).upper():
        # Same architecture as the baseline: the tuned A3 declaration stands.
        return {**DEFAULT_MACHINE, **override}
    _reject_sku_peak_conflict(sku, profile_machine, override)
    derived: dict[str, Any] = {
        "name": override.get("name") or f"declared-{sku}-operator-campaign",
        "sku": sku,
        "arch_family": family,
        "dies_per_package": profile_machine.dies_per_package,
        "num_devices": _as_int(
            override.get("num_devices", DEFAULT_MACHINE["num_devices"]),
            "campaign.machine.num_devices_invalid",
        ),
        "ranks_per_node": _as_int(
            override.get("ranks_per_node", DEFAULT_MACHINE["ranks_per_node"]),
            "campaign.machine.ranks_per_node_invalid",
        ),
    }
    if override.get("device"):
        derived["device"] = dict(override["device"])
    declared_peaks = dict(override.get("compute_peaks") or override.get("dtype_peaks") or {})
    if declared_peaks:
        # The profile owns every peak it declares; a caller may only add a
        # dtype the SKU does not declare (conflicts are rejected above).
        derived["compute_peaks"] = {**dict(profile_machine.compute_peaks),
                                    **{str(k).strip().lower(): float(v)
                                       for k, v in declared_peaks.items()}}
    return derived


def _reject_sku_peak_conflict(
    sku: str, profile_machine: MachineSpec, override: Mapping[str, Any],
) -> None:
    """Refuse explicit rates that contradict the SKU's own declared peaks."""
    expected = dict(profile_machine.compute_peaks)
    declared_cube = dict(override.get("device") or {}).get("cube_flops_per_s")
    bf16 = expected.get("bf16")
    if declared_cube is not None and bf16:
        value = _as_positive(declared_cube, "campaign.machine.cube_flops_per_s_invalid")
        if abs(value - float(bf16)) > 1e-3 * float(bf16):
            raise A3InputError(
                "campaign_machine_sku_peak_conflict:%s:cube_flops_per_s:%r!=%r"
                % (sku, value, float(bf16))
            )
    for key in ("compute_peaks", "dtype_peaks"):
        for dtype, peak in dict(override.get(key) or {}).items():
            name = str(dtype).strip().lower()
            if name not in expected:
                continue
            value = _as_positive(peak, "campaign.machine.compute_peak_invalid")
            if abs(value - float(expected[name])) > 1e-3 * float(expected[name]):
                raise A3InputError(
                    "campaign_machine_sku_peak_conflict:%s:%s:%r!=%r"
                    % (sku, name, value, float(expected[name]))
                )


__all__ = ["resolve_campaign_machine"]
