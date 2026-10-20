"""Typed input contracts for the A3 predictive / normative modeling paths.

[Role]
  Validate one case into explicit machine (per-die resources and link
  topology), model (Dense/GQA/MLA and Dense/MoE FFN), workload, runtime and
  parallel-placement records.  Convert semantic shapes into local shards while
  keeping package totals out of per-die resource fields.

[Invariants]
  * Every numeric resource has a unit named in its field.  ``hbm_bytes`` is a
    per-die capacity, never a package or node total.
  * A parallel plan is ``dp * pp * tp`` ranks; EP is a sub-group inside each TP
    group and is never multiplied into the world size as an independent axis.
  * Links are a projection of the declared transfer channels
    (``comm.default_transfer_channels``): a single-die A2 package has an
    intra-node HCCS fabric and no intra-package SIO; a dual-die A3 package has
    one dedicated SIO wire per package plus the intra-node switch.  No family
    string selects a rate here.
  * A link declares its contention semantics (``mesh``/``switched``/``direct``
    /``unknown``); an explicit fixture that omits it is a shared pipe, never a
    silently assumed mesh.
  * Revisions and source paths are metadata: they do not affect semantic input
    keys, cost formulas or compatibility.  Compatibility is validated by
    shape/feature contracts in this module.
  * Unsupported or missing fields raise ``A3InputError`` with a stable reason;
    zero/negative sizes never become a modeled plan.

[Timeouts & Locks]
  Pure in-memory validation.  Optional SKU resolution is local arithmetic; no
  device, subprocess, network, lock or sleep is used.

[Side-Effect Policy]
  No file writes, environment mutation, provider imports or device probes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .comm import TransferChannelSpec, default_transfer_channels

_CASE_PROVENANCE_FIELDS = (
    "source", "revision", "commit", "git_commit", "source_sha", "name",
    "description",
)
# Document prose that is not business input.  Unknown top-level fields are
# rejected; there is no format-version allowance.
_IGNORED_DOCUMENT_FIELDS = {
    "metadata", "notes", "assumptions",
}
_DTYPE_BYTES = {
    "fp32": 4.0, "float32": 4.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0,
    "float16": 2.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int8": 1.0, "int4": 0.5, "int32": 4.0, "bool": 1.0,
}
_DTYPE_PEAK = {
    "bf16": 1.0, "bfloat16": 1.0, "fp16": 1.0, "float16": 1.0,
    "fp32": 0.5, "float32": 0.5, "fp8": 2.0, "fp8_e4m3": 2.0,
    "fp8_e5m2": 2.0, "int8": 2.0, "int4": 4.0,
}


class A3InputError(ValueError):
    """Stable A3 input validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise A3InputError(reason)
    return value


def _unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    for key in mapping:
        if (key not in allowed and key not in _CASE_PROVENANCE_FIELDS
                and key not in _IGNORED_DOCUMENT_FIELDS
                and not str(key).startswith("x_")):
            raise A3InputError(f"unknown_field:{context}.{key}")


def _float(value: Any, reason: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError(reason) from exc
    if not math.isfinite(result) or result < minimum:
        raise A3InputError(reason)
    return result


def _int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise A3InputError(reason)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError(reason) from exc
    if result < minimum or float(value) != float(result):
        raise A3InputError(reason)
    return result


def _positive_float(value: Any, reason: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise A3InputError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError(reason) from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise A3InputError(reason)
    return result


def _bool(value: Any, reason: str) -> bool:
    if not isinstance(value, bool):
        raise A3InputError(reason)
    return value


def _text(value: Any, reason: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise A3InputError(reason)
    return value.strip()


def dtype_bytes(dtype: str) -> float:
    key = str(dtype).strip().lower()
    if key not in _DTYPE_BYTES:
        raise A3InputError(f"unsupported_dtype:{dtype}")
    return _DTYPE_BYTES[key]


def dtype_peak_scale(dtype: str) -> float:
    key = str(dtype).strip().lower()
    if key not in _DTYPE_PEAK:
        raise A3InputError(f"unsupported_compute_dtype:{dtype}")
    return _DTYPE_PEAK[key]


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return (int(value) + alignment - 1) // alignment * alignment


def _normalize_dtype(value: Any, reason: str) -> str:
    text = _text(value, reason).lower().replace("-", "_")
    if text not in _DTYPE_BYTES:
        raise A3InputError(f"{reason}:{text}")
    return text


def _metadata(mapping: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    items = []
    for key in _CASE_PROVENANCE_FIELDS:
        if mapping.get(key) is not None:
            items.append((key, str(mapping[key])))
    if isinstance(mapping.get("metadata"), Mapping):
        items.extend(sorted(
            (str(k), v if isinstance(v, str) else json.dumps(v, sort_keys=True))
            for k, v in mapping["metadata"].items()
        ))
    return tuple(sorted(items))


@dataclass(frozen=True)
class LinkSpec:
    name: str
    channel: str
    capacity_bytes_per_ms: float
    latency_us: float
    members: tuple[int, ...]
    scope: str = "shared"
    source: str = "declared"
    # Declared contention semantics of the physical fabric: ``mesh`` (every
    # pair owns a wire, A2 HCCS), ``switched`` (one aggregate pipe, A3
    # HCCS_SW / RoCE), ``direct`` (dedicated point-to-point, SIO die pair),
    # ``unknown`` (an explicit fixture that did not declare it -- treated as a
    # shared pipe, never silently as a mesh).
    semantics: str = "unknown"
    protocol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channel": self.channel,
            "capacity_bytes_per_ms": self.capacity_bytes_per_ms,
            "capacity_gbps": self.capacity_bytes_per_ms / 1e6,
            "latency_us": self.latency_us,
            "members": list(self.members),
            "scope": self.scope,
            "source": self.source,
            "transport_semantics": self.semantics,
            "protocol": self.protocol,
        }


#: Scope tokens that address a physical fabric rather than a link instance.
#: ``d2d`` is the legacy generic token used by every pre-A2 workload op; it
#: means "the machine's intra-node D2D fabric", not "the first link in a list".
LINK_SCOPE_ALIASES: dict[str, tuple[str, ...]] = {
    "d2d": ("d2d_intra_node",),
    "hccs": ("d2d_intra_node",),
    "hccssw": ("d2d_intra_node",),
    "hccs_sw": ("d2d_intra_node",),
    "intra_node": ("d2d_intra_node",),
    "d2d_intra_node": ("d2d_intra_node",),
    "sio": ("d2d_intra_package",),
    "intra_package": ("d2d_intra_package",),
    "d2d_intra_package": ("d2d_intra_package",),
    "die_pair": ("d2d_intra_package",),
    "inter_node": ("d2d_inter_node",),
    "roce": ("d2d_inter_node",),
    "d2d_inter_node": ("d2d_inter_node",),
}


@dataclass(frozen=True)
class DieResources:
    rank: int
    device_id: int
    package_id: int
    die_in_package: int
    hbm_bytes: int
    cube_flops_per_s: float
    vector_ops_per_s: float
    hbm_bw_bytes_per_s: float
    mte_bw_bytes_per_s: float
    launch_us: float
    l2_bytes: int = 0
    l1_bytes: int = 0
    l0a_bytes: int = 0
    l0b_bytes: int = 0
    l0c_bytes: int = 0
    ub_bytes: int = 0
    aic_count: int = 0
    aiv_count: int = 0
    mte1_bw_bytes_per_s: float = 0.0
    source: str = "declared"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "device_id": self.device_id,
            "package_id": self.package_id,
            "die_in_package": self.die_in_package,
            "hbm_bytes": self.hbm_bytes,
            "hbm_gib": self.hbm_bytes / (1024 ** 3),
            "cube_flops_per_s": self.cube_flops_per_s,
            "vector_ops_per_s": self.vector_ops_per_s,
            "hbm_bw_bytes_per_s": self.hbm_bw_bytes_per_s,
            "mte_bw_bytes_per_s": self.mte_bw_bytes_per_s,
            "launch_us": self.launch_us,
            "l2_bytes": self.l2_bytes,
            "l1_bytes": self.l1_bytes, "l0a_bytes": self.l0a_bytes,
            "l0b_bytes": self.l0b_bytes, "l0c_bytes": self.l0c_bytes,
            "ub_bytes": self.ub_bytes,
            "aic_count": self.aic_count, "aiv_count": self.aiv_count,
            "mte1_bw_bytes_per_s": self.mte1_bw_bytes_per_s,
            "source": self.source,
        }


@dataclass(frozen=True)
class MachineSpec:
    name: str
    arch_family: str
    sku: str
    dies_per_package: int
    ranks_per_node: int
    dies: tuple[DieResources, ...]
    links: tuple[LinkSpec, ...]
    compute_peaks: tuple[tuple[str, float], ...] = ()
    hardware_key: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    #: The declared transfer-channel view this machine's ``links`` were derived
    #: from (``comm.TransferChannelSpec`` values).  Kept as a tuple so the
    #: machine stays hashable/frozen; consumers index it by ``.channel``.
    transfer_channels: tuple[TransferChannelSpec, ...] = ()

    def channel(self, name: str) -> TransferChannelSpec | None:
        key = str(name).strip().lower()
        for item in self.transfer_channels:
            if item.channel.lower() == key:
                return item
        return None

    def link_for_scope(self, scope: str) -> LinkSpec | None:
        """Resolve a fabric scope token to exactly one physical link.

        Resolution order: an exact link name / scope / channel, then a declared
        channel alias (``sio`` -> ``d2d_intra_package``, ``hccs`` ->
        ``d2d_intra_node``), then -- only for an *unrecognised* token on a
        single-fabric machine -- that one fabric.  A token that names a fabric
        the machine does not have (``sio`` on A2) is an explicit error instead
        of silently pricing traffic on an unrelated link.
        """
        key = str(scope or "").strip().lower()
        links = list(self.links)
        if not key:
            raise A3InputError("link_scope_missing")
        if not links:
            raise A3InputError(f"machine_has_no_links:{key}")
        exact = [
            link for link in links
            if key in {
                link.name.lower(), link.scope.lower(), link.channel.lower(),
            }
        ]
        if not exact:
            channels = LINK_SCOPE_ALIASES.get(key)
            if channels is not None:
                exact = [link for link in links if link.channel.lower() in channels]
                if not exact and key == "d2d" and len(links) == 1:
                    # The generic legacy token means "this machine's D2D
                    # fabric"; a single-fabric machine has exactly one.
                    return links[0]
                if not exact:
                    raise A3InputError(f"link_scope_unresolved:{key}")
            elif len(links) == 1:
                # A machine that declares exactly one fabric has no ambiguity.
                return links[0]
            else:
                raise A3InputError(f"link_scope_unknown:{key}")
        # Deterministic and conservative: among equal-token matches the
        # smallest capacity wins, with the name as tie-break, so declaration
        # order never decides which wire is priced.
        return min(exact, key=lambda link: (link.capacity_bytes_per_ms, link.name))

    @classmethod
    def from_mapping(cls, value: Any) -> "MachineSpec":
        raw = _mapping(value, "machine_invalid")
        allowed = {
            "name", "sku", "soc", "arch_family", "dies_per_package", "ranks_per_node",
            "num_devices", "device", "links", "hbm_gib", "hbm_bytes",
            "cube_flops_per_s", "vector_ops_per_s", "hbm_bw_bytes_per_s",
            "mte_bw_bytes_per_s", "launch_us", "l2_bytes", "l1_bytes",
            "l0a_bytes", "l0b_bytes", "l0c_bytes", "ub_bytes", "aic_count",
            "aiv_count", "mte1_bw_bytes_per_s", "transfer_channels",
            "compute_peaks", "dtype_peaks", "hardware_key", "dies", "note",
        }
        _unknown(raw, allowed, "machine")
        sku = str(raw.get("sku", raw.get("soc", ""))).strip()
        profile = None
        if sku:
            try:
                from .hardware import resolve_device_capability_profile

                profile = resolve_device_capability_profile(soc=sku)
            except Exception as exc:  # noqa: BLE001 - hardware boundary is optional
                raise A3InputError(f"machine_sku_unresolved:{sku}:{type(exc).__name__}") from exc
        explicit_dies: list[DieResources] = []
        if raw.get("dies") is not None:
            if not isinstance(raw["dies"], Sequence) or isinstance(raw["dies"], (str, bytes)) or not raw["dies"]:
                raise A3InputError("machine.dies_invalid")
            for item in raw["dies"]:
                die = _mapping(item, "machine.die_invalid")
                hbm = die.get("hbm_bytes")
                if hbm is None and die.get("hbm_gib") is not None:
                    hbm = int(round(float(die["hbm_gib"]) * 1024 ** 3))
                explicit_dies.append(DieResources(
                    rank=_int(die.get("rank"), "machine.die.rank_invalid"),
                    device_id=_int(die.get("device_id", die.get("rank", 0)), "machine.die.device_invalid"),
                    package_id=_int(die.get("package_id", 0), "machine.die.package_invalid"),
                    die_in_package=_int(die.get("die_in_package", 0), "machine.die.package_die_invalid"),
                    hbm_bytes=_int(hbm, "machine.die.hbm_invalid", minimum=1),
                    cube_flops_per_s=_positive_float(die.get("cube_flops_per_s"), "machine.die.cube_invalid"),
                    vector_ops_per_s=_positive_float(die.get("vector_ops_per_s"), "machine.die.vector_invalid"),
                    hbm_bw_bytes_per_s=_positive_float(die.get("hbm_bw_bytes_per_s"), "machine.die.hbm_bw_invalid"),
                    mte_bw_bytes_per_s=_positive_float(die.get("mte_bw_bytes_per_s"), "machine.die.mte_invalid"),
                    launch_us=_positive_float(die.get("launch_us", 0.0), "machine.die.launch_invalid", allow_zero=True),
                    l2_bytes=_int(die.get("l2_bytes", 0), "machine.die.l2_invalid"),
                    l1_bytes=_int(die.get("l1_bytes", 0), "machine.die.l1_invalid"),
                    l0a_bytes=_int(die.get("l0a_bytes", 0), "machine.die.l0a_invalid"),
                    l0b_bytes=_int(die.get("l0b_bytes", 0), "machine.die.l0b_invalid"),
                    l0c_bytes=_int(die.get("l0c_bytes", 0), "machine.die.l0c_invalid"),
                    ub_bytes=_int(die.get("ub_bytes", 0), "machine.die.ub_invalid"),
                    aic_count=_int(die.get("aic_count", 0), "machine.die.aic_invalid"),
                    aiv_count=_int(die.get("aiv_count", 0), "machine.die.aiv_invalid"),
                    mte1_bw_bytes_per_s=_positive_float(
                        die.get("mte1_bw_bytes_per_s", 0.0),
                        "machine.die.mte1_invalid", allow_zero=True,
                    ),
                    source=str(die.get("source", "declared")),
                ))
            if not explicit_dies:
                raise A3InputError("machine.dies_invalid")
        device = raw.get("device")
        if device is not None and not isinstance(device, Mapping):
            raise A3InputError("machine.device_invalid")
        merged: dict[str, Any] = dict(device or {})
        if explicit_dies:
            first = explicit_dies[0]
            merged.update({
                "hbm_bytes": first.hbm_bytes, "cube_flops_per_s": first.cube_flops_per_s,
                "vector_ops_per_s": first.vector_ops_per_s,
                "hbm_bw_bytes_per_s": first.hbm_bw_bytes_per_s,
                "mte_bw_bytes_per_s": first.mte_bw_bytes_per_s,
                "launch_us": first.launch_us, "l2_bytes": first.l2_bytes,
                "l1_bytes": first.l1_bytes, "l0a_bytes": first.l0a_bytes,
                "l0b_bytes": first.l0b_bytes, "l0c_bytes": first.l0c_bytes,
                "ub_bytes": first.ub_bytes,
                "aic_count": first.aic_count, "aiv_count": first.aiv_count,
                "mte1_bw_bytes_per_s": first.mte1_bw_bytes_per_s,
            })
        for key in (
            "hbm_gib", "hbm_bytes", "cube_flops_per_s", "vector_ops_per_s",
            "hbm_bw_bytes_per_s", "mte_bw_bytes_per_s", "launch_us", "l2_bytes",
            "l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes", "ub_bytes",
            "aic_count", "aiv_count", "mte1_bw_bytes_per_s",
        ):
            if raw.get(key) is not None:
                merged[key] = raw[key]
        if profile is not None:
            merged = _profile_defaults(profile, merged)
        hbm_bytes = merged.get("hbm_bytes")
        if hbm_bytes is None and merged.get("hbm_gib") is not None:
            hbm_bytes = int(round(_positive_float(merged["hbm_gib"], "machine_hbm_invalid") * 1024 ** 3))
        if hbm_bytes is None:
            raise A3InputError("machine_hbm_bytes_required")
        hbm_bytes = _int(hbm_bytes, "machine_hbm_invalid", minimum=1)
        cube = _positive_float(merged.get("cube_flops_per_s"), "machine_cube_flops_required")
        vector = _positive_float(merged.get("vector_ops_per_s"), "machine_vector_ops_required")
        hbm_bw = _positive_float(merged.get("hbm_bw_bytes_per_s"), "machine_hbm_bw_required")
        mte = _positive_float(merged.get("mte_bw_bytes_per_s"), "machine_mte_bw_required")
        launch = _positive_float(merged.get("launch_us", 20.0), "machine_launch_us_invalid", allow_zero=True)
        l2 = _int(merged.get("l2_bytes", 0), "machine_l2_invalid")
        l1 = _int(merged.get("l1_bytes", 0), "machine_l1_invalid")
        l0a = _int(merged.get("l0a_bytes", 0), "machine_l0a_invalid")
        l0b = _int(merged.get("l0b_bytes", 0), "machine_l0b_invalid")
        l0c = _int(merged.get("l0c_bytes", 0), "machine_l0c_invalid")
        ub = _int(merged.get("ub_bytes", 0), "machine_ub_invalid")
        aic_count = _int(merged.get("aic_count", 0), "machine_aic_invalid")
        aiv_count = _int(merged.get("aiv_count", 0), "machine_aiv_invalid")
        mte1_bw = float(merged.get("mte1_bw_bytes_per_s", 0.0) or 0.0)
        dies_per_package = _int(raw.get("dies_per_package", merged.get("dies_per_package", 1)), "machine_dies_per_package_invalid", minimum=1)
        ranks_per_node = _int(raw.get("ranks_per_node", merged.get("ranks_per_node", 8)), "machine_ranks_per_node_invalid", minimum=1)
        num_devices = len(explicit_dies) if explicit_dies else _int(
            raw.get("num_devices", merged.get("num_devices", dies_per_package)),
            "machine_num_devices_invalid", minimum=1,
        )
        arch = str(raw.get("arch_family", merged.get("arch_family", "custom"))).strip().upper()
        name = str(raw.get("name", sku or f"custom-{num_devices}x")).strip()
        memory = getattr(profile, "memory_hierarchy_spec", None) if profile is not None else None
        channel_overrides: dict[str, Any] | None = None
        if raw.get("transfer_channels") is not None:
            raw_channels = raw["transfer_channels"]
            if (not isinstance(raw_channels, Sequence)
                    or isinstance(raw_channels, (str, bytes))):
                raise A3InputError("machine.transfer_channels_invalid")
            channel_overrides = {}
            for index, entry in enumerate(raw_channels):
                item = _mapping(entry, "machine.transfer_channel_invalid")
                channel_name = str(item.get("channel") or "").strip()
                if not channel_name:
                    raise A3InputError(
                        f"machine.transfer_channel_invalid:{index}"
                    )
                channel_overrides[channel_name.lower()] = dict(item)
        channels = default_transfer_channels(
            arch,
            dies_per_package=dies_per_package,
            intra_package_die_bw_gbps=getattr(memory, "intra_package_die_bw_gbps", None),
            intra_node_hccs_bw_gbps=getattr(memory, "intra_node_hccs_bw_gbps", None),
            inter_node_roce_bw_gbps=getattr(memory, "inter_node_roce_bw_gbps", None),
            overrides=channel_overrides,
        )
        dies: list[DieResources] = list(explicit_dies)
        if not explicit_dies:
            for rank in range(num_devices):
                devices_per_pkg = max(1, dies_per_package)
                package_id = rank // devices_per_pkg
                dies.append(DieResources(
                    rank=rank,
                    device_id=rank,
                    package_id=package_id,
                    die_in_package=rank % devices_per_pkg,
                    hbm_bytes=hbm_bytes,
                    cube_flops_per_s=cube,
                    vector_ops_per_s=vector,
                    hbm_bw_bytes_per_s=hbm_bw,
                    mte_bw_bytes_per_s=mte,
                    launch_us=launch,
                    l2_bytes=l2, l1_bytes=l1, l0a_bytes=l0a,
                    l0b_bytes=l0b, l0c_bytes=l0c, ub_bytes=ub,
                    aic_count=aic_count, aiv_count=aiv_count,
                    mte1_bw_bytes_per_s=mte1_bw,
                    source="sku_profile" if profile is not None else "declared",
                ))
        links = _parse_links(
            raw.get("links"), num_devices, arch,
            dies_per_package=dies_per_package, ranks_per_node=ranks_per_node,
            channels=channels,
        )
        peaks_raw = raw.get("compute_peaks", raw.get("dtype_peaks", {}))
        peaks: list[tuple[str, float]] = []
        if peaks_raw is not None:
            if not isinstance(peaks_raw, Mapping):
                raise A3InputError("machine.compute_peaks_invalid")
            for key, value in peaks_raw.items():
                peaks.append((str(key).strip().lower(), _positive_float(value, "machine.compute_peak_invalid")))
        return cls(
            name=name, arch_family=arch, sku=sku, dies_per_package=dies_per_package,
            ranks_per_node=ranks_per_node, dies=tuple(dies), links=tuple(links),
            compute_peaks=tuple(sorted(peaks)),
            hardware_key=str(raw.get("hardware_key", sku or name)).strip(),
            metadata=_metadata(raw),
            transfer_channels=tuple(channels.values()),
        )

    def validate_placement(self, parallel: "ParallelSpec") -> None:
        if parallel.world_size > len(self.dies):
            raise A3InputError(
                f"world_size_exceeds_devices:{parallel.world_size}>{len(self.dies)}"
            )
        if parallel.tp > len(self.dies):
            raise A3InputError(f"tp_exceeds_devices:{parallel.tp}")

    def peak_flops_for(self, dtype: str) -> float:
        key = str(dtype).strip().lower()
        declared = dict(self.compute_peaks)
        if key in declared:
            return declared[key]
        base = self.dies[0].cube_flops_per_s
        if key in {"bf16", "bfloat16", "fp16", "float16"}:
            return base
        # Do not infer a 2x/4x low-precision multiplier from the dtype.  A3
        # fp8/int4 is only modeled at a speedup when the caller declares the
        # corresponding hardware/kernel peak.
        return base

    def die_for(self, dp_rank: int, stage: int, tp_rank: int, pp: int = 1, tp: int = 1) -> DieResources:
        index = ((dp_rank * max(1, pp) + stage) * max(1, tp) + tp_rank) % len(self.dies)
        return self.dies[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sku": self.sku,
            "arch_family": self.arch_family,
            "dies_per_package": self.dies_per_package,
            "ranks_per_node": self.ranks_per_node,
            "num_devices": len(self.dies),
            "dies": [die.to_dict() for die in self.dies],
            "links": [link.to_dict() for link in self.links],
            "compute_peaks": dict(self.compute_peaks),
            "hardware_key": self.hardware_key,
            "metadata": dict(self.metadata),
            "transfer_channels": [item.to_dict() for item in self.transfer_channels],
            "note": "hbm_bytes is per die; package/node totals are not stored here",
        }

    @classmethod
    def from_profile(
        cls,
        profile: Any,
        *,
        name: str = "",
        sku: str = "",
        hardware_key: str = "",
        num_devices: int | None = None,
        ranks_per_node: int | None = None,
        clock_mhz: float | None = None,
        launch_us: float | None = None,
        vector_ops_per_s: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MachineSpec":
        """Build a machine from a resolved ``DeviceCapabilityProfile``.

        Every per-die quantity is derived from the declared capability (ops per
        cycle, HBM, L2, buffers).  ``clock_mhz`` defaults to the profile's
        documented clock precedence (explicit override -> verified runtime
        clock -> configured clock); a caller may pass the clock it used so the
        configured-vs-runtime discrepancy stays readable in ``metadata``.

        ``vector_ops_per_s`` is a *caller declaration* for the one capability
        A2/A3 do not publish (B8: no ``ops_per_cycle_per_aiv`` exists, and the
        ``.ini`` ``vec_calc_size`` is a SIMD byte width, not an element-op
        rate).  When the profile does declare it, that value wins; when the
        caller declares nothing either, ``_profile_defaults`` keeps the
        historical ``aiv x simd_width x clock`` nominal and the caller is
        expected to label it unverified in ``metadata``.  A quantity nothing
        can supply (launch/dispatch intercept) stays an explicit argument.
        """
        topology = profile.clock_and_topology_spec
        resolved_soc = str(topology.soc_version or sku or "").strip()
        effective_clock = getattr(profile, "effective_clock_mhz", None)
        resolved_clock = float(clock_mhz) if clock_mhz is not None else (
            effective_clock() if callable(effective_clock) else topology.configured_clock_mhz
        )
        if resolved_clock is None:
            raise A3InputError("machine_clock_undeclared")
        merged: dict[str, Any] = {}
        if launch_us is not None:
            merged["launch_us"] = float(launch_us)
        defaults = _profile_defaults(profile, merged, clock_mhz=resolved_clock)
        if vector_ops_per_s is not None:
            defaults["vector_ops_per_s"] = float(vector_ops_per_s)
        devices = int(
            num_devices
            or max(1, int(topology.dies_per_package or 1))
        )
        device = {
            key: defaults[key] for key in (
                "hbm_bytes", "cube_flops_per_s", "vector_ops_per_s",
                "hbm_bw_bytes_per_s", "mte_bw_bytes_per_s", "launch_us",
                "l2_bytes", "l1_bytes", "l0a_bytes", "l0b_bytes",
                "l0c_bytes", "ub_bytes", "aic_count", "aiv_count",
                "mte1_bw_bytes_per_s",
            )
        }
        mapping: dict[str, Any] = {
            "name": name or f"profile-{resolved_soc or 'unknown'}",
            "arch_family": defaults["arch_family"],
            "dies_per_package": defaults["dies_per_package"],
            "num_devices": devices,
            "ranks_per_node": int(ranks_per_node or devices),
            "device": device,
            "hardware_key": hardware_key or resolved_soc,
        }
        if resolved_soc:
            mapping["sku"] = resolved_soc
        if metadata:
            mapping["metadata"] = dict(metadata)
        return cls.from_mapping(mapping)


def _profile_defaults(
    profile: Any, merged: dict[str, Any], *, clock_mhz: float | None = None,
) -> dict[str, Any]:
    clock_mhz = (
        float(clock_mhz) if clock_mhz is not None
        else profile.clock_and_topology_spec.configured_clock_mhz
    )
    hz = float(clock_mhz) * 1e6
    cube = profile.cube_spec
    bf16_ops_per_die = cube.ops_per_cycle_per_die("bf16")
    if bf16_ops_per_die is None or cube.aic_count_per_die is None:
        raise A3InputError("machine_sku_cube_capability_missing")
    memory = profile.memory_hierarchy_spec
    hbm_mib = memory.verified_runtime_hbm_mib or memory.platform_config_hbm_mib
    if hbm_mib is None:
        raise A3InputError("machine_sku_hbm_capability_missing")
    hbm_bw_gbps = memory.sustained_hbm_bw_gbps or memory.hbm_peak_bw_gbps
    if hbm_bw_gbps is None:
        raise A3InputError("machine_sku_hbm_bandwidth_missing")
    defaults = {
        "hbm_bytes": int(round(float(hbm_mib) * 1024 ** 2)),
        "cube_flops_per_s": float(bf16_ops_per_die) * hz,
        # Declared per-AIV ops/cycle wins when the profile publishes it.  A2/A3
        # currently declare nothing (B8), so the historical
        # ``aiv x simd_width_bytes x clock`` nominal is kept -- that nominal is
        # *unverified*, not physical (bytes/cycle is not element-ops/cycle).
        # ``MachineSpec.from_profile(vector_ops_per_s=...)`` exists so a caller
        # that owns a measured effective rate can replace it explicitly.
        "vector_ops_per_s": _profile_vector_ops_per_s(profile, hz),
        "hbm_bw_bytes_per_s": float(hbm_bw_gbps) * 1e9,
        "mte_bw_bytes_per_s": 2.0 * float(hbm_bw_gbps) * 1e9,
        "launch_us": 20.0,
        "l2_bytes": int(round(float(memory.l2_capacity_mib or 0.0) * 1024 ** 2)),
        "l1_bytes": int(round(float(cube.l1_kib) * 1024)),
        "l0a_bytes": int(round(float(cube.l0a_kib) * 1024)),
        "l0b_bytes": int(round(float(cube.l0b_kib) * 1024)),
        "l0c_bytes": int(round(float(cube.l0c_kib) * 1024)),
        "ub_bytes": int(round(float(profile.vector_spec.allocatable_ub_kib) * 1024)),
        "aic_count": int(cube.aic_count_per_die or 0),
        "aiv_count": int(profile.vector_spec.aiv_count_per_die or 0),
        "mte1_bw_bytes_per_s": float(hbm_bw_gbps) * 1e9,
        "dies_per_package": profile.clock_and_topology_spec.dies_per_package,
        "arch_family": profile.clock_and_topology_spec.arch_family,
    }
    defaults.update(merged)
    return defaults


def _profile_vector_ops_per_s(profile: Any, hz: float) -> float:
    """Per-die vector peak for a profile: declared, else the legacy nominal."""
    declared = getattr(profile, "peak_vector_ops_per_s", None)
    if callable(declared):
        value = declared("bf16")
        if value is not None:
            return float(value)
    # No A2/A3 profile publishes ``ops_per_cycle_per_aiv``.  Keep the rate the
    # existing machines were built with so switching the fixture path cannot
    # halve the vector engine; the value is an unverified nominal and a caller
    # with a measured effective rate overrides it via ``from_profile``.
    return (
        float(profile.vector_spec.aiv_count_per_die or 1)
        * float(profile.vector_spec.simd_width_bytes or 32) * hz
    )


def _channel_links(
    channels: Mapping[str, TransferChannelSpec],
    *,
    num_devices: int,
    dies_per_package: int,
    ranks_per_node: int,
) -> list[LinkSpec]:
    """Materialize links from the declared transfer channels.

    This is the single place a fabric rate becomes a ``LinkSpec``.  It encodes
    only physical structure, never a family literal:

    * ``d2d_intra_package`` (SIO) exists only when the package really has two
      dies; each package gets its own dedicated pair wire, so a single-die A2
      part has **no** SIO link at all.
    * ``d2d_intra_node`` exists for every rank set and carries the declared
      protocol (A2 ``HCCS`` mesh, A3 ``HCCS_SW`` switch).
    * ``d2d_inter_node`` exists only when devices exceed one node.
    """
    links: list[LinkSpec] = []
    intra_package = channels.get("d2d_intra_package")
    if (intra_package is not None and intra_package.effective_uni_bw_gbps
            and dies_per_package > 1):
        packages = math.ceil(num_devices / dies_per_package)
        for package in range(packages):
            members = tuple(range(
                package * dies_per_package,
                min(num_devices, (package + 1) * dies_per_package),
            ))
            if len(members) < 2:
                continue
            name = (
                "d2d_intra_package" if packages == 1
                else f"d2d_intra_package:package{package}"
            )
            links.append(LinkSpec(
                name=name, channel="d2d_intra_package",
                capacity_bytes_per_ms=float(intra_package.effective_uni_bw_gbps) * 1e6,
                latency_us=float(intra_package.alpha_us),
                members=members,
                scope="fabric:d2d_intra_package",
                source="comm_default_transfer_channels_not_measured",
                semantics=str(intra_package.channel_semantics or "direct"),
                protocol=str(intra_package.protocol),
            ))
    intra_node = channels.get("d2d_intra_node")
    if intra_node is not None and intra_node.effective_uni_bw_gbps:
        links.append(LinkSpec(
            name="d2d_intra_node", channel="d2d_intra_node",
            capacity_bytes_per_ms=float(intra_node.effective_uni_bw_gbps) * 1e6,
            latency_us=float(intra_node.alpha_us),
            members=tuple(range(num_devices)),
            scope="fabric:d2d_intra_node",
            source="comm_default_transfer_channels_not_measured",
            semantics=str(intra_node.channel_semantics or "unknown"),
            protocol=str(intra_node.protocol),
        ))
    inter_node = channels.get("d2d_inter_node")
    if (inter_node is not None and inter_node.effective_uni_bw_gbps
            and num_devices > max(1, ranks_per_node)):
        links.append(LinkSpec(
            name="d2d_inter_node", channel="d2d_inter_node",
            capacity_bytes_per_ms=float(inter_node.effective_uni_bw_gbps) * 1e6,
            latency_us=float(inter_node.alpha_us),
            members=tuple(range(num_devices)),
            scope="fabric:d2d_inter_node",
            source="comm_default_transfer_channels_not_measured",
            semantics=str(inter_node.channel_semantics or "switched"),
            protocol=str(inter_node.protocol),
        ))
    return links


def _parse_links(
    raw: Any,
    num_devices: int,
    arch: str,
    *,
    dies_per_package: int = 1,
    ranks_per_node: int | None = None,
    channels: Mapping[str, TransferChannelSpec] | None = None,
) -> list[LinkSpec]:
    if channels is None:
        channels = default_transfer_channels(
            arch, dies_per_package=dies_per_package,
        )
    if raw is None:
        return _channel_links(
            channels, num_devices=num_devices,
            dies_per_package=dies_per_package,
            ranks_per_node=int(ranks_per_node or num_devices),
        )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise A3InputError("machine.links_invalid")
    links: list[LinkSpec] = []
    for index, item in enumerate(raw):
        entry = _mapping(item, "machine.link_invalid")
        allowed = {
            "name", "channel", "capacity_bytes_per_ms", "capacity_gbps",
            "latency_us", "members", "scope", "source", "semantics",
            "transport_semantics", "protocol",
        }
        _unknown(entry, allowed, f"machine.links[{index}]")
        capacity = entry.get("capacity_bytes_per_ms")
        if capacity is None and entry.get("capacity_gbps") is not None:
            capacity = _positive_float(entry["capacity_gbps"], "machine.link_capacity_invalid") * 1e6
        members_raw = entry.get("members", list(range(num_devices)))
        if not isinstance(members_raw, Sequence) or isinstance(members_raw, (str, bytes)):
            raise A3InputError("machine.link_members_invalid")
        members = tuple(sorted({_int(v, "machine.link_member_invalid") for v in members_raw}))
        if not members or any(member >= num_devices for member in members):
            raise A3InputError("machine.link_member_out_of_range")
        channel = _text(entry.get("channel", "d2d_intra_node"), "machine.link_channel_invalid")
        declared_protocol = channels.get(channel)
        protocol = str(
            entry.get("protocol")
            or (declared_protocol.protocol if declared_protocol is not None else "")
        )
        # An explicit fixture must *declare* mesh vs switched; when it names a
        # channel the machine itself declares, that channel's semantics carry
        # over (A2 HCCS mesh, A3 HCCS_SW switch, SIO pair).  A link on no
        # declared channel is a shared pipe instead of being silently promoted
        # to a dedicated-pair fabric.
        semantics = str(
            entry.get("semantics")
            or entry.get("transport_semantics")
            or (
                declared_protocol.channel_semantics
                if declared_protocol is not None
                and declared_protocol.channel_semantics in {
                    "mesh", "direct", "switched",
                }
                else "unknown"
            )
        ).strip().lower()
        if semantics not in {"mesh", "switched", "direct", "unknown"}:
            raise A3InputError(f"machine.link_semantics_invalid:{semantics}")
        links.append(LinkSpec(
            name=_text(entry.get("name", f"link{index}"), "machine.link_name_invalid"),
            channel=channel,
            capacity_bytes_per_ms=_positive_float(capacity, "machine.link_capacity_invalid"),
            latency_us=_positive_float(entry.get("latency_us", 0.0), "machine.link_latency_invalid", allow_zero=True),
            members=members,
            scope=_text(entry.get("scope", f"fabric:{channel}"), "machine.link_scope_invalid"),
            source=str(entry.get("source", "declared")),
            semantics=semantics,
            protocol=protocol,
        ))
    return links


@dataclass(frozen=True)
class ParallelSpec:
    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    dcp: int = 1

    @classmethod
    def from_mapping(cls, value: Any, prefix: str = "parallel") -> "ParallelSpec":
        raw = _mapping(value, f"{prefix}_invalid")
        allowed = {"dp", "tp", "ep", "pp", "dcp", "world_size"}
        _unknown(raw, allowed, prefix)
        return cls(
            dp=_int(raw.get("dp", 1), f"{prefix}.dp_invalid", minimum=1),
            tp=_int(raw.get("tp", 1), f"{prefix}.tp_invalid", minimum=1),
            ep=_int(raw.get("ep", 1), f"{prefix}.ep_invalid", minimum=1),
            pp=_int(raw.get("pp", 1), f"{prefix}.pp_invalid", minimum=1),
            dcp=_int(raw.get("dcp", 1), f"{prefix}.dcp_invalid", minimum=1),
        )

    @property
    def world_size(self) -> int:
        return self.dp * self.pp * self.tp

    @property
    def ep_domain(self) -> int:
        return self.dp * self.tp

    @property
    def ep_spans_dp(self) -> bool:
        return self.ep > self.tp

    @property
    def etp(self) -> int:
        # Expert tensor-parallel shard width.  EP groups live in the DP*TP
        # domain; when EP is smaller than TP the remaining TP factor may shard
        # each expert.  EP itself is never added as another world axis.
        if self.ep <= 1:
            return self.tp
        if self.ep <= self.tp and self.tp % self.ep == 0:
            return self.tp // self.ep
        return 1

    def validate(self, model: "ModelSpec") -> None:
        if self.ep > self.ep_domain:
            raise A3InputError(f"ep_exceeds_dp_tp_domain:{self.ep}>{self.ep_domain}")
        if self.ep_domain % self.ep != 0:
            raise A3InputError(f"ep_not_divisor_of_dp_tp_domain:{self.ep_domain}%{self.ep}")
        if self.ep > 1 and model.num_experts and model.num_experts % self.ep != 0:
            raise A3InputError(f"experts_not_divisible_by_ep:{model.num_experts}%{self.ep}")
        if model.layers % self.pp != 0:
            raise A3InputError(f"layers_not_divisible_by_pp:{model.layers}%{self.pp}")
        if model.num_heads % self.tp != 0:
            raise A3InputError(f"heads_not_divisible_by_tp:{model.num_heads}%{self.tp}")
        if self.ep > 1 and self.ep <= self.tp and self.tp % self.ep != 0:
            raise A3InputError(f"ep_not_divisor_of_tp:{self.ep}%{self.tp}")
        if self.dcp > self.dp * self.tp and self.dcp > 1:
            raise A3InputError(f"dcp_exceeds_dp_tp_domain:{self.dcp}>{self.dp * self.tp}")
        if self.ep > 1 and not model.num_experts:
            raise A3InputError("ep_requires_moe_model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dp": self.dp, "tp": self.tp, "ep": self.ep, "pp": self.pp, "dcp": self.dcp,
            "world_size": self.world_size, "ep_scope": "dp_tp_domain",
            "ep_domain": self.ep_domain, "ep_spans_dp": self.ep_spans_dp,
            "etp": self.etp, "attention_tp": self.tp,
            "rank_formula": "rank = ((dp_id * pp + stage) * tp + tp_rank)",
            "moe_ep_rank_formula": "ep_rank = (dp_id * tp + tp_rank) % ep",
        }


from .a3_model_types import (  # noqa: E402  (defined after primitive validators)
    MlaSpec, ModelSpec, RequestSpec, SloSpec, WorkloadSpec,
)


@dataclass(frozen=True)
class KvTransferSpec:
    enabled: bool = False
    channel: str = "d2d_intra_node"
    chunks: int = 1
    bytes_per_token: int = 0
    ready_policy: str = "first_chunk"

    @classmethod
    def from_mapping(cls, value: Any) -> "KvTransferSpec":
        if value is None:
            return cls()
        raw = _mapping(value, "runtime.kv_transfer_invalid")
        _unknown(raw, {"enabled", "channel", "chunks", "bytes_per_token", "ready_policy"}, "runtime.kv_transfer")
        return cls(
            enabled=_bool(raw.get("enabled", False), "runtime.kv_transfer.enabled_invalid"),
            channel=_text(raw.get("channel", "d2d_intra_node"), "runtime.kv_transfer.channel_invalid"),
            chunks=_int(raw.get("chunks", 1), "runtime.kv_transfer.chunks_invalid", minimum=1),
            bytes_per_token=_int(raw.get("bytes_per_token", 0), "runtime.kv_transfer.bytes_invalid"),
            ready_policy=_text(raw.get("ready_policy", "first_chunk"), "runtime.kv_transfer.ready_invalid"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "channel": self.channel, "chunks": self.chunks,
            "bytes_per_token": self.bytes_per_token, "ready_policy": self.ready_policy,
        }


@dataclass(frozen=True)
class PrepareSpec:
    """Explicit ahead-of-forward preparation work.

    ``known_at`` is semantic, not a priority hint: startup/arrival/previous
    forward work may be scheduled early; an unknown future token, routing
    result or speculative acceptance decision is rejected.
    """
    name: str
    kind: str = "static_metadata"
    bytes: int = 0
    duration_ms: float = 0.0
    known_at: str = "startup"
    reusable: bool = True
    request_id: str | None = None
    deadline_ms: float | None = None
    lifetime_ms: float | None = None
    link_bytes: int = 0
    mte_bytes: int = 0
    consumer: str = "background"
    resource: str = "prep"
    consumer_op_ids: tuple[str, ...] = ()
    first_consumer_op_ids: tuple[str, ...] = ()
    replace_op_ids: tuple[str, ...] = ()
    owner_stage: int = -1
    owner_rank: str = ""
    iteration: int = -1
    capacity_bytes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    service_bytes: int = 0
    mte2_bytes: int = 0
    mte3_bytes: int = 0
    cost_once_key: str = ""
    lifetime: str = ""
    arrival_ms: float = 0.0
    accounted_elsewhere: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "PrepareSpec":
        raw = _mapping(value, "runtime.prepare_invalid")
        allowed = {
            "name", "kind", "bytes", "duration_ms", "known_at", "reusable",
            "request_id", "deadline_ms", "lifetime_ms", "link_bytes", "mte_bytes",
            "consumer", "resource", "consumer_op_ids", "first_consumer_op_ids",
            "replace_op_ids", "owner_stage", "owner_rank", "iteration",
            "capacity_bytes", "read_bytes", "write_bytes", "service_bytes",
            "mte2_bytes", "mte3_bytes", "cost_once_key", "lifetime",
            "accounted_elsewhere",
        }
        _unknown(raw, allowed, "runtime.prepare")
        known = str(raw.get("known_at", "startup"))
        if known not in {"startup", "arrival", "previous_forward"}:
            raise A3InputError(f"runtime.prepare_known_at_invalid:{known}")
        consumer = str(raw.get("consumer", "background"))
        if consumer not in {"forward", "sample", "background"}:
            raise A3InputError(f"runtime.prepare_consumer_invalid:{consumer}")
        name = str(raw.get("name", "prepare"))
        if not name:
            raise A3InputError("runtime.prepare_name_empty")
        return cls(
            name=name, kind=str(raw.get("kind", "static_metadata")),
            bytes=_int(raw.get("bytes", 0), "runtime.prepare_bytes_invalid", minimum=0),
            duration_ms=_float(raw.get("duration_ms", 0.0), "runtime.prepare_duration_invalid", minimum=0.0),
            known_at=known, reusable=bool(raw.get("reusable", True)),
            request_id=(str(raw["request_id"]) if raw.get("request_id") is not None else None),
            deadline_ms=(_float(raw["deadline_ms"], "runtime.prepare_deadline_invalid", minimum=0.0)
                         if raw.get("deadline_ms") is not None else None),
            lifetime_ms=(_float(raw["lifetime_ms"], "runtime.prepare_lifetime_invalid", minimum=0.0)
                         if raw.get("lifetime_ms") is not None else None),
            link_bytes=_int(raw.get("link_bytes", 0), "runtime.prepare_link_invalid", minimum=0),
            mte_bytes=_int(raw.get("mte_bytes", 0), "runtime.prepare_mte_invalid", minimum=0),
            consumer=consumer,
            resource=str(raw.get("resource", "prep")),
            consumer_op_ids=tuple(str(v) for v in (raw.get("consumer_op_ids") or ())),
            first_consumer_op_ids=tuple(
                str(v) for v in (raw.get("first_consumer_op_ids") or ())
            ),
            replace_op_ids=tuple(str(v) for v in (raw.get("replace_op_ids") or ())),
            owner_stage=int(raw.get("owner_stage", -1)),
            owner_rank=str(raw.get("owner_rank", "")),
            iteration=int(raw.get("iteration", -1)),
            capacity_bytes=_int(raw.get("capacity_bytes", 0), "runtime.prepare_capacity_invalid", minimum=0),
            read_bytes=_int(raw.get("read_bytes", 0), "runtime.prepare_read_invalid", minimum=0),
            write_bytes=_int(raw.get("write_bytes", 0), "runtime.prepare_write_invalid", minimum=0),
            service_bytes=_int(raw.get("service_bytes", 0), "runtime.prepare_service_invalid", minimum=0),
            mte2_bytes=_int(raw.get("mte2_bytes", 0), "runtime.prepare_mte2_invalid", minimum=0),
            mte3_bytes=_int(raw.get("mte3_bytes", 0), "runtime.prepare_mte3_invalid", minimum=0),
            cost_once_key=str(raw.get("cost_once_key", "")),
            lifetime=str(raw.get("lifetime", "")),
            arrival_ms=float(raw.get("arrival_ms", 0.0) or 0.0),
            accounted_elsewhere=bool(raw.get("accounted_elsewhere", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "bytes": self.bytes,
            "duration_ms": self.duration_ms, "known_at": self.known_at,
            "reusable": self.reusable, "request_id": self.request_id,
            "deadline_ms": self.deadline_ms, "lifetime_ms": self.lifetime_ms,
            "link_bytes": self.link_bytes, "mte_bytes": self.mte_bytes,
            "consumer": self.consumer, "resource": self.resource,
            "consumer_op_ids": list(self.consumer_op_ids),
            "first_consumer_op_ids": list(self.first_consumer_op_ids),
            "replace_op_ids": list(self.replace_op_ids),
            "owner_stage": self.owner_stage, "owner_rank": self.owner_rank,
            "iteration": self.iteration, "capacity_bytes": self.capacity_bytes,
            "read_bytes": self.read_bytes, "write_bytes": self.write_bytes,
            "service_bytes": self.service_bytes,
            "mte2_bytes": self.mte2_bytes, "mte3_bytes": self.mte3_bytes,
            "cost_once_key": self.cost_once_key, "lifetime": self.lifetime,
            "arrival_ms": self.arrival_ms,
            "accounted_elsewhere": self.accounted_elsewhere,
        }


@dataclass(frozen=True)
class RuntimeSpec:
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_model_len: int = 32768
    enable_chunked_prefill: bool = True
    max_num_input_tokens: int | None = None
    kv_block_size: int = 128
    long_prefill_token_threshold: int = 0
    gpu_memory_utilization: float = 0.9
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
    graph_buckets: tuple[int, ...] = ()
    graph_mode: str = "auto"
    curve_allow_primitive_transfer: bool = False
    moe_capacity_factor: float = 1.0
    moe_recipe: str = "auto"
    mc2_capacity: int = 65536
    mc2_chunks: int = 4
    quant_recipe: str = "auto"
    kv_transfer: KvTransferSpec = field(default_factory=KvTransferSpec)
    media_tokens: int = 0
    engram_transfer_bytes: int = 0
    fusion: bool = False
    fusion_worker_count: int = 1
    prepare: tuple[PrepareSpec, ...] = ()
    cv_query: tuple[int, int, int, int] | None = None
    model_prepare: bool = False
    prepare_host_rates: tuple[tuple[str, float], ...] = ()
    communication_backend: str = "unspecified"
    communication_observed_backend: str = "unknown"
    communication_route: str = ""
    communication_timing_completion: str = ""
    communication_requested_aiv_cores: int = 0
    communication_observed_aiv_cores: int = 0
    communication_algorithm_regime: str = ""
    communication_topology_class: str = ""
    communication_in_place: bool = False
    scheduler: str = "vllm_like_token_budget"
    max_inflight_iterations: int = 1
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeSpec":
        raw = _mapping(value, "runtime_invalid") if value is not None else {}
        allowed = {
            "max_num_batched_tokens", "max_num_seqs", "max_model_len", "enable_chunked_prefill",
            "chunked_prefill", "max_num_input_tokens", "kv_block_size",
            "long_prefill_token_threshold", "long_prefill_threshold", "gpu_memory_utilization",
            "parallel", "tp", "ep", "dp", "pp", "dcp", "graph_buckets", "graph_mode",
            "curve_allow_primitive_transfer",
            "moe_capacity_factor", "moe_recipe", "mc2_capacity", "mc2_chunks", "quant_recipe",
            "kv_transfer", "scheduler", "max_inflight_iterations",
            "media_tokens", "engram_transfer_bytes", "fusion",
            "fusion_worker_count", "prepare",
            "cv_query", "model_prepare", "prepare_host_rates",
            "communication_backend", "communication_observed_backend",
            "communication_route", "communication_timing_completion",
            "communication_requested_aiv_cores",
            "communication_observed_aiv_cores",
            "communication_algorithm_regime", "communication_topology_class",
            "communication_in_place", "metadata",
        }
        _unknown(raw, allowed, "runtime")
        parallel_raw = raw.get("parallel")
        if isinstance(parallel_raw, Mapping):
            parallel = ParallelSpec.from_mapping(parallel_raw)
        else:
            parallel = ParallelSpec(
                dp=_int(raw.get("dp", 1), "runtime.dp_invalid", minimum=1),
                tp=_int(raw.get("tp", 1), "runtime.tp_invalid", minimum=1),
                ep=_int(raw.get("ep", 1), "runtime.ep_invalid", minimum=1),
                pp=_int(raw.get("pp", 1), "runtime.pp_invalid", minimum=1),
                dcp=_int(raw.get("dcp", 1), "runtime.dcp_invalid", minimum=1),
            )
        buckets_raw = raw.get("graph_buckets", ())
        if isinstance(buckets_raw, Sequence) and not isinstance(buckets_raw, (str, bytes)):
            buckets = tuple(sorted({_int(v, "runtime.graph_bucket_invalid", minimum=1) for v in buckets_raw}))
        else:
            raise A3InputError("runtime.graph_buckets_invalid")
        graph_mode = _text(raw.get("graph_mode", "auto"), "runtime.graph_mode_invalid").lower()
        if graph_mode not in {"auto", "eager", "force"}:
            raise A3InputError(f"runtime.graph_mode_unsupported:{graph_mode}")
        inflight = _int(raw.get("max_inflight_iterations", 1), "runtime.max_inflight_invalid", minimum=1)
        if inflight != 1:
            raise A3InputError("runtime.max_inflight_iterations_only_1_supported")
        runtime = cls(
            max_num_batched_tokens=_int(raw.get("max_num_batched_tokens", 16384), "runtime.batch_budget_invalid", minimum=1),
            max_num_seqs=_int(raw.get("max_num_seqs", 256), "runtime.max_num_seqs_invalid", minimum=1),
            max_model_len=_int(raw.get("max_model_len", 32768), "runtime.max_model_len_invalid", minimum=1),
            enable_chunked_prefill=_bool(raw.get("enable_chunked_prefill", raw.get("chunked_prefill", True)), "runtime.chunked_prefill_invalid"),
            max_num_input_tokens=(_int(raw["max_num_input_tokens"], "runtime.input_budget_invalid", minimum=1)
                                   if raw.get("max_num_input_tokens") is not None else None),
            kv_block_size=_int(raw.get("kv_block_size", 128), "runtime.kv_block_size_invalid", minimum=1),
            long_prefill_token_threshold=_int(
                raw.get("long_prefill_token_threshold", raw.get("long_prefill_threshold", 0)),
                "runtime.long_prefill_threshold_invalid", minimum=0,
            ),
            gpu_memory_utilization=_positive_float(raw.get("gpu_memory_utilization", 0.9), "runtime.memory_utilization_invalid"),
            parallel=parallel, graph_buckets=buckets, graph_mode=graph_mode,
            curve_allow_primitive_transfer=_bool(
                raw.get("curve_allow_primitive_transfer", False),
                "runtime.curve_allow_primitive_transfer_invalid",
            ),
            moe_capacity_factor=_positive_float(raw.get("moe_capacity_factor", 1.0), "runtime.moe_capacity_factor_invalid"),
            moe_recipe=_text(raw.get("moe_recipe", "auto"), "runtime.moe_recipe_invalid").lower(),
            mc2_capacity=_int(raw.get("mc2_capacity", 65536), "runtime.mc2_capacity_invalid", minimum=1),
            mc2_chunks=_int(raw.get("mc2_chunks", 4), "runtime.mc2_chunks_invalid", minimum=1),
            quant_recipe=_text(raw.get("quant_recipe", "auto"), "runtime.quant_recipe_invalid").lower(),
            kv_transfer=KvTransferSpec.from_mapping(raw.get("kv_transfer")),
            media_tokens=_int(raw.get("media_tokens", 0), "runtime.media_tokens_invalid", minimum=0),
            engram_transfer_bytes=_int(raw.get("engram_transfer_bytes", 0), "runtime.engram_bytes_invalid", minimum=0),
            fusion=bool(raw.get("fusion", False)),
            fusion_worker_count=_int(
                raw.get("fusion_worker_count", 1),
                "runtime.fusion_worker_count_invalid", minimum=1,
            ),
            prepare=tuple(
                PrepareSpec.from_mapping(item)
                for item in (raw.get("prepare") or ())
            ),
            cv_query=(
                tuple(_int(v, "runtime.cv_query_invalid", minimum=1)
                      for v in raw["cv_query"])
                if raw.get("cv_query") is not None else None
            ),
            model_prepare=bool(raw.get("model_prepare", False)),
            prepare_host_rates=tuple(
                (str(item["name"]), float(item["ops_per_s"]))
                for item in (raw.get("prepare_host_rates") or ())
            ),
            communication_backend=str(raw.get("communication_backend", "unspecified")),
            communication_observed_backend=str(
                raw.get("communication_observed_backend", "unknown")
            ),
            communication_route=str(raw.get("communication_route", "")),
            communication_timing_completion=str(
                raw.get("communication_timing_completion", "")
            ),
            communication_requested_aiv_cores=int(
                raw.get("communication_requested_aiv_cores", 0) or 0
            ),
            communication_observed_aiv_cores=int(
                raw.get("communication_observed_aiv_cores", 0) or 0
            ),
            communication_algorithm_regime=str(
                raw.get("communication_algorithm_regime", "")
            ),
            communication_topology_class=str(
                raw.get("communication_topology_class", "")
            ),
            communication_in_place=bool(raw.get("communication_in_place", False)),
            scheduler=_text(raw.get("scheduler", "vllm_like_token_budget"), "runtime.scheduler_invalid"),
            max_inflight_iterations=inflight,
            metadata=tuple(sorted(
                (str(key), value if isinstance(value, str)
                 else json.dumps(value, sort_keys=True))
                for key, value in dict(_metadata(raw)).items()
            )),
        )
        if runtime.gpu_memory_utilization > 1.0:
            raise A3InputError("runtime.memory_utilization_above_one")
        if runtime.max_num_batched_tokens > runtime.max_model_len:
            # This is legal (budget above model length), but never a reason to
            # allocate more KV than max_model_len.
            pass
        if runtime.moe_recipe not in {"auto", "mc2", "alltoall"}:
            raise A3InputError(f"runtime.moe_recipe_unsupported:{runtime.moe_recipe}")
        return runtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_num_input_tokens": self.max_num_input_tokens,
            "kv_block_size": self.kv_block_size,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "parallel": self.parallel.to_dict(),
            "graph_buckets": list(self.graph_buckets),
            "graph_mode": self.graph_mode,
            "curve_allow_primitive_transfer": self.curve_allow_primitive_transfer,
            "moe_capacity_factor": self.moe_capacity_factor,
            "moe_recipe": self.moe_recipe,
            "mc2_capacity": self.mc2_capacity,
            "mc2_chunks": self.mc2_chunks,
            "quant_recipe": self.quant_recipe,
            "kv_transfer": self.kv_transfer.to_dict(),
            "media_tokens": self.media_tokens,
            "engram_transfer_bytes": self.engram_transfer_bytes,
            "fusion": self.fusion,
            "fusion_worker_count": self.fusion_worker_count,
            "prepare": [item.to_dict() for item in self.prepare],
            "cv_query": list(self.cv_query) if self.cv_query is not None else None,
            "model_prepare": self.model_prepare,
            "prepare_host_rates": [list(item) for item in self.prepare_host_rates],
            "communication_backend": self.communication_backend,
            "communication_observed_backend": self.communication_observed_backend,
            "communication_route": self.communication_route,
            "communication_timing_completion": self.communication_timing_completion,
            "communication_requested_aiv_cores": self.communication_requested_aiv_cores,
            "communication_observed_aiv_cores": self.communication_observed_aiv_cores,
            "communication_algorithm_regime": self.communication_algorithm_regime,
            "communication_topology_class": self.communication_topology_class,
            "communication_in_place": self.communication_in_place,
            "scheduler": self.scheduler,
            "max_inflight_iterations": self.max_inflight_iterations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CaseSpec:
    machine: MachineSpec
    model: ModelSpec
    workload: WorkloadSpec
    runtime: RuntimeSpec
    slo: SloSpec = field(default_factory=SloSpec)
    calibration: Any = None
    communication_calibration: Any = None
    cv_calibration: Any = None
    curve_artifact: Any = None
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Any) -> "CaseSpec":
        raw = _mapping(value, "case_invalid")
        allowed = {
            "machine", "model", "workload", "runtime", "slo", "calibration",
            "search", "should", "curve_artifact",
        }
        _unknown(raw, allowed, "case")
        machine = MachineSpec.from_mapping(raw.get("machine"))
        model = ModelSpec.from_mapping(raw.get("model"))
        workload = WorkloadSpec.from_mapping(raw.get("workload"))
        runtime = RuntimeSpec.from_mapping(raw.get("runtime", {}))
        # Media positions are expanded in RequestSpec at parse time.  Merge the
        # catalog-owned declarative metadata so graph lowering knows which
        # requests still need one tower and that positions are pre-expanded.
        try:
            from .a3_catalog_media import media_runtime_metadata
        except ImportError:
            media_runtime_metadata = None
        if media_runtime_metadata is not None:
            media_meta = media_runtime_metadata(workload)
            if media_meta:
                merged = dict(runtime.metadata)
                merged.update(media_meta)
                runtime = replace(
                    runtime, metadata=tuple(sorted(merged.items())),
                )
        runtime.parallel.validate(model)
        machine.validate_placement(runtime.parallel)
        workload.validate_model(model, runtime)
        slo_raw = raw.get("slo")
        slo = SloSpec.from_mapping(slo_raw) if slo_raw is not None else SloSpec()
        return cls(
            machine=machine, model=model, workload=workload, runtime=runtime,
            slo=slo, calibration=raw.get("calibration"),
            communication_calibration=None, cv_calibration=None,
            curve_artifact=raw.get("curve_artifact"),
            metadata=_metadata(raw),
        )

    def semantic_payload(self) -> dict[str, Any]:
        """Canonical semantics; source/revision/commit are deliberately absent."""
        return {
            "machine": {
                "arch_family": self.machine.arch_family,
                "dies_per_package": self.machine.dies_per_package,
                "dies": [
                    {k: v for k, v in die.to_dict().items() if k != "source"}
                    for die in self.machine.dies
                ],
                "links": [
                    {k: v for k, v in link.to_dict().items() if k != "source"}
                    for link in self.machine.links
                ],
            },
            "model": self.model.to_dict(),
            "workload": self.workload.to_dict(),
            "runtime": self.runtime.to_dict(),
            "slo": self.slo.to_dict(),
        }

    def semantic_signature(self) -> str:
        payload = self.semantic_payload()
        payload["model"].pop("metadata", None)
        payload["workload"].pop("metadata", None)
        payload["runtime"].pop("metadata", None)
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_plan(self) -> dict[str, Any]:
        return {
            "semantic_signature": self.semantic_signature(),
            "machine": self.machine.to_dict(),
            "model": self.model.to_dict(),
            "workload": self.workload.to_dict(),
            "runtime": self.runtime.to_dict(),
            "slo": self.slo.to_dict(),
            "parallel_semantics": {
                "world_size": self.runtime.parallel.world_size,
                "rank_formula": "rank = ((dp_id * pp + stage) * tp + tp_rank)",
                "ep_scope": "ep ranks are a sub-group of the tp group; not multiplied again",
            },
            "metadata": dict(self.metadata),
        }


__all__ = [
    "A3InputError", "LinkSpec", "DieResources", "MachineSpec", "ParallelSpec",
    "MlaSpec", "ModelSpec", "SloSpec", "RequestSpec", "WorkloadSpec",
    "KvTransferSpec", "RuntimeSpec", "CaseSpec", "dtype_bytes", "dtype_peak_scale",
    "align_up",
]
