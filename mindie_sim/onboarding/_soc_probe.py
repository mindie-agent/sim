"""Silicon SoC probe, CANN INI / ``npu-smi`` parsers, and profile validator/scaffolder.

[Role] Probe ``aclrtGetSocName``, CANN ``platform_config/<SocName>.ini``, and
  ``npu-smi`` topology (via live SSH or deterministic ``--probe-json`` replay),
  validate existing ``ascend.json`` profiles + ``DatapathCapabilitySpec`` +
  fabric channels (failing closed on ``silicon_profile_conflict``), or scaffold
  a verified new SKU profile entry.
[Invariants] Zero raw IPv4 quads, hostnames, usernames, or ``$HOME`` paths in
  emitted records; ``buffer_units.l2 == 1`` enforced on every profile;
  ``TimeoutError`` always propagates before ``OSError``.
[Timeouts & Locks] Bounded SSH subprocesses; no locks or device contexts in
  the local process.
[Side-Effect Policy] Read-only unless ``scaffold_out`` is explicitly requested.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from mindie_sim.backends import ascend_profiles
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die, scale_explicit_placement_device
from mindie_sim.schema.hardware import Hardware
from mindie_sim.stack_registry import default_stack_registry

from .contracts import (
    DEFAULT_PROBE_TIMEOUT_S,
    DEFAULT_SSH_CONNECT_TIMEOUT_S,
    GIB_BYTES,
    MIB_BYTES,
    SOC_PROBE_SCHEMA,
    SiliconProfileConflict,
    SocProbeRecord,
    assert_sanitized_payload,
    build_capture_uri,
    canonical_json_bytes,
    read_bounded_json,
    sanitize_public_alias,
    subprocess_window_kwargs,
    write_atomic_json,
)

MAX_IDLE_AICORE_UTIL_PCT = 2
MAX_IDLE_HBM_UTIL_PCT = 10
DEFAULT_SSH_PORT = 22
ARCH_PRESET_22 = "npu_arch_2201"
ARCH_PRESET_35 = "npu_arch_3510"
KNOWN_ARCH_PRESETS = frozenset({ARCH_PRESET_22, ARCH_PRESET_35})
_ISA_TOKEN_TO_PRESET = {
    "npu_arch_2201": ARCH_PRESET_22,
    "arch22": ARCH_PRESET_22,
    "dav-c220": ARCH_PRESET_22,
    "npu_arch_3510": ARCH_PRESET_35,
    "arch35": ARCH_PRESET_35,
    "dav-c310": ARCH_PRESET_35,
}
_SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")

_SIZE_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmg]?i?b)?\s*$", re.IGNORECASE)
_UNIT_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": MIB_BYTES,
    "mib": MIB_BYTES,
    "gb": GIB_BYTES,
    "gib": GIB_BYTES,
}
_SAFE_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_size_bytes(raw: Any, *, default_unit: str = "b") -> int:
    if raw is None or raw == "":
        return 0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(round(float(raw) * _UNIT_MULTIPLIERS[default_unit]))
    match = _SIZE_UNIT_RE.match(str(raw))
    if not match:
        return 0
    unit = (match.group(2) or default_unit).lower()
    return int(round(float(match.group(1)) * _UNIT_MULTIPLIERS.get(unit, 1)))


def parse_platform_config_ini(ini_text: str) -> dict[str, Any]:
    """Parse CANN ``platform_config/<SocName>.ini`` key/value hardware facts."""
    values: dict[str, str] = {}
    for line in ini_text.splitlines():
        clean = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not clean or clean.startswith("[") or "=" not in clean:
            continue
        key, val = clean.split("=", 1)
        values[key.strip().lower()] = val.strip()

    def _first(*keys: str) -> str:
        for key in keys:
            if key in values and values[key]:
                return values[key]
        return ""

    def _int_of(*keys: str) -> int:
        raw = _first(*keys)
        if not raw:
            return 0
        digits = re.match(r"(\d+)", raw)
        return int(digits.group(1)) if digits else 0

    soc_ver = _first("soc_version", "soc_name")
    isa_raw = _first("arch_preset", "compute_arch", "isa_version", "cube_arch", "ai_core_arch").lower()
    arch_preset = ""
    for token, preset in _ISA_TOKEN_TO_PRESET.items():
        if token in isa_raw:
            arch_preset = preset
            break
    ini_sha256 = hashlib.sha256(ini_text.encode("utf-8")).hexdigest() if ini_text.strip() else ""
    return {
        "soc_version": soc_ver,
        "arch_preset": arch_preset,
        "architecture": _first("architecture", "soc_family", "product_family"),
        "ini_sha256": ini_sha256,
        "aic_count": _int_of("ai_core_cnt", "aic_count", "cube_core_cnt"),
        "aiv_count": _int_of("vector_core_cnt", "aiv_count", "ai_vector_core_cnt"),
        "aicore_freq_mhz": _int_of("aic_freq", "aicore_freq", "cube_freq", "ai_core_freq"),
        "l2_bytes": _parse_size_bytes(_first("l2_size", "l2_cache_size")),
        "ub_bytes": _parse_size_bytes(_first("ub_size", "unified_buffer_size")),
        "l1_bytes": _parse_size_bytes(_first("l1_size", "l1_buffer_size")),
        "l0a_bytes": _parse_size_bytes(_first("l0a_size", "l0a_buffer_size")),
        "l0b_bytes": _parse_size_bytes(_first("l0b_size", "l0b_buffer_size")),
        "l0c_bytes": _parse_size_bytes(_first("l0c_size", "l0c_buffer_size")),
        "hbm_bytes": _parse_size_bytes(_first("hbm_size", "memory_size")),
    }


def parse_npu_smi_topology(common_text: str, topo_text: str = "") -> dict[str, Any]:
    """Extract AIC count, frequency, HBM bytes, and topology hints from ``npu-smi`` text."""
    def _int_field(label: str) -> int:
        match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", common_text)
        return int(match.group(1)) if match else 0

    hbm_mb = _int_field("HBM Capacity(MB)")
    chip_ids = set(re.findall(r"(?m)^\s*Chip ID\s*:\s*(\d+)", common_text))
    dies_per_pkg = max(1, len(chip_ids))
    ranks = max(dies_per_pkg, len(re.findall(r"(?m)^\s*NPU\d+\b", topo_text)))
    fabric: dict[str, float] = {}
    if "SIO" in topo_text:
        fabric["d2d_intra_package"] = 1.0
    if "HCCS" in topo_text:
        fabric["d2d_intra_node"] = 1.0
    return {
        "aic_count": _int_field("Aicore Count"),
        "aicore_freq_mhz": _int_field("Aicore Freq(MHZ)"),
        "hbm_bytes": hbm_mb * MIB_BYTES if hbm_mb > 0 else 0,
        "dies_per_package": dies_per_pkg,
        "ranks_per_node": ranks,
        "fabric_hints": fabric,
    }


def verify_idle_die_telemetry(
    info_text: str,
    common_text: str,
    *,
    npu_id: int = 0,
) -> dict[str, Any]:
    """Fail closed unless target NPU die has zero processes and idle AICore/HBM."""
    if not re.search(rf"No running processes found in NPU {npu_id}(?!\d)", info_text):
        raise ModelingError(f"onboarding_npu_not_idle:npu={npu_id}")

    def _int_field(label: str) -> int | None:
        match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", common_text)
        return int(match.group(1)) if match else None

    aic_util = _int_field("Aicore Usage Rate(%)")
    hbm_util = _int_field("HBM Usage Rate(%)")
    aic_cnt = _int_field("Aicore Count")
    aic_freq = _int_field("Aicore Freq(MHZ)")
    if aic_util is None or hbm_util is None:
        raise ModelingError("onboarding_npu_telemetry_missing")
    if aic_util > MAX_IDLE_AICORE_UTIL_PCT:
        raise ModelingError(f"onboarding_npu_busy:aicore_util={aic_util}")
    if hbm_util > MAX_IDLE_HBM_UTIL_PCT:
        raise ModelingError(f"onboarding_npu_busy:hbm_util={hbm_util}")
    return {
        "npu_id": npu_id,
        "aicore_util_pct": aic_util,
        "hbm_util_pct": hbm_util,
        "observed_aic_count": aic_cnt or 0,
        "observed_aicore_freq_mhz": aic_freq or 0,
    }


def _ssh_exec(
    endpoint_alias: str,
    remote_cmd: str,
    *,
    timeout_s: int,
    endpoints_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    ep_file = endpoints_path or (Path.home() / ".mindie" / "endpoints.json")
    raw = read_bounded_json(ep_file)
    eps = raw.get("endpoints", raw) if isinstance(raw, Mapping) else {}
    if endpoint_alias not in eps:
        raise ModelingError(
            f"onboarding_endpoint_unknown:{sanitize_public_alias(endpoint_alias, 'ep')}"
        )
    ep = dict(eps[endpoint_alias])
    cmd = [
        "ssh", "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_S}",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "LogLevel=ERROR",
        "-p", str(int(ep.get("port", DEFAULT_SSH_PORT))),
    ]
    if ep.get("identity_file"):
        cmd.extend(["-i", str(ep["identity_file"])])
    cmd.extend(["-l", str(ep.get("user", "root")), "--", str(ep["host"]), remote_cmd])
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"onboarding_ssh_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_ssh_failed:{exc}") from exc


def run_ssh_command(
    endpoint_alias: str,
    remote_cmd: str,
    timeout_s: int,
    *,
    endpoints_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``remote_cmd`` on ``endpoint_alias`` via sanctioned ``endpoints.json`` SSH."""
    return _ssh_exec(
        endpoint_alias,
        remote_cmd,
        timeout_s=int(timeout_s),
        endpoints_path=endpoints_path,
    )


def _resolve_explicit_arch(
    raw_arch_preset: str,
    raw_architecture: str,
) -> tuple[str, str]:
    """Normalize explicit architecture declarations without SKU-name sniffing."""
    preset_raw = raw_arch_preset.strip().lower()
    arch_preset = _ISA_TOKEN_TO_PRESET.get(preset_raw, "")
    architecture = raw_architecture.strip()
    return arch_preset, architecture


def _missing_scaffold_capabilities(
    probe: SocProbeRecord,
    base_template_sku: str,
) -> list[str]:
    missing: list[str] = []
    if probe.arch_preset not in KNOWN_ARCH_PRESETS:
        missing.append("arch_preset")
    if not probe.architecture:
        missing.append("architecture")
    if probe.aic_count < 1:
        missing.append("aic_count")
    if probe.aiv_count < 1:
        missing.append("aiv_count")
    if probe.hbm_bytes < 1:
        missing.append("hbm_bytes")
    if probe.l2_bytes < 1:
        missing.append("l2_bytes")
    if probe.ub_bytes < 1:
        missing.append("ub_bytes")
    if not base_template_sku.strip():
        missing.append("base_template_sku")
    return missing


def _derive_canonical_bindings(
    soc_name: str,
    arch_id: str,
) -> dict[str, Any]:
    slug = _SAFE_SLUG_RE.sub("_", soc_name.lower()).strip("_")
    arch_tag = "arch35" if arch_id == ARCH_PRESET_35 else "arch22"
    stack_names = sorted(default_stack_registry().list_stacks())
    return {
        "compose_sku": soc_name,
        "primitive_table_relpath": f"data/primitives/{arch_tag}_{slug}.json",
        "registered_stacks": stack_names,
    }


def _scaffold_sku_profile(
    probe: SocProbeRecord,
    all_profiles: Mapping[str, Any],
    *,
    base_template_sku: str,
) -> dict[str, Any]:
    if base_template_sku not in all_profiles:
        raise ModelingError(f"silicon_template_sku_unknown:{base_template_sku}")
    base = deepcopy(all_profiles[base_template_sku])
    dev = base["device"]
    base_preset = str(dev.get("compute_arch", {}).get("preset") or "")
    if base_preset != probe.arch_preset:
        raise ModelingError(
            f"silicon_template_arch_mismatch:template={base_template_sku}"
            f":template_arch={base_preset}:probe_arch={probe.arch_preset}"
        )
    base_aic = int(dev["compute_units"])
    base_aiv = int(dev["resource_units"].get("vector", base_aic * 2))
    target_aiv = probe.aiv_count
    dev["compute_units"] = probe.aic_count
    dev["resource_units"]["vector"] = target_aiv
    scale_explicit_placement_device(
        dev,
        {"compute_units": probe.aic_count, "resource_units": {"vector": target_aiv}},
        probe.aic_count,
        base_aic,
        base_aiv,
    )
    dev["memory_bytes"] = probe.hbm_bytes
    for pool, val in (
        ("l2", probe.l2_bytes),
        ("ub", probe.ub_bytes),
        ("l1", probe.l1_bytes),
        ("l0a", probe.l0a_bytes),
        ("l0b", probe.l0b_bytes),
        ("l0c", probe.l0c_bytes),
    ):
        if val > 0:
            dev["buffers"][pool] = val
    dev["buffer_units"]["l2"] = 1
    mem_hier = dev.get("physical_mechanisms", {}).get("memory_hierarchy")
    if isinstance(mem_hier, dict):
        mem_hier["l2_capacity_bytes"] = probe.l2_bytes
        prov = mem_hier.get("provenance")
        if isinstance(prov, dict):
            prov["l2_capacity_bytes"] = probe.l2_bytes
            prov["calibrated"] = False
    base["architecture"] = probe.architecture
    base["dies_per_package"] = probe.dies_per_package
    for chan_name, bw_gbps in probe.fabric_channels:
        if chan_name in base.get("transfer_channels", {}) and bw_gbps > 0.0:
            base["transfer_channels"][chan_name]["uni_peak_bw_gbps"] = float(bw_gbps)
    base["source"] = {
        "calibration": {
            "calibrated": False,
            "rate_provenance": "declared_nominal",
            "source_sku": probe.soc_name,
            "anchor_sku": base_template_sku,
            "capture_uri": probe.capture_uri,
        }
    }
    return base


_FABRIC_BW_MIN_RATIO = 0.5
_FABRIC_BW_MAX_RATIO = 1.25


def _validate_profile_against_probe(
    probe: SocProbeRecord,
    profile_entry: Mapping[str, Any],
) -> tuple[list[SiliconProfileConflict], dict[str, Any]]:
    raw_machine = {
        "sku": probe.soc_name,
        "architecture": profile_entry["architecture"],
        "dies_per_package": profile_entry["dies_per_package"],
        "ranks_per_node": max(1, probe.ranks_per_node),
        "num_devices": 1,
        "device": deepcopy(profile_entry["device"]),
        "transfer_channels": deepcopy(profile_entry.get("transfer_channels", {})),
    }
    hw = Hardware.from_mapping(raw_machine)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    bufs = dict(die.buffers)
    buf_units = dict(die.buffer_units)
    conflicts: list[SiliconProfileConflict] = []

    def _cmp(field: str, observed: Any, declared: Any) -> None:
        if observed and observed != declared:
            conflicts.append(
                SiliconProfileConflict(
                    field=field,
                    observed=observed,
                    declared=declared,
                    reason=f"silicon_profile_conflict:{field}:observed={observed}:declared={declared}",
                )
            )

    _cmp("arch_preset", probe.arch_preset, str(dp.arch_id))
    _cmp("aic_count", probe.aic_count, int(dp.aic_count))
    _cmp("aiv_count", probe.aiv_count, int(dp.aiv_count))
    _cmp("dies_per_package", probe.dies_per_package, int(profile_entry["dies_per_package"]))
    _cmp("hbm_bytes", probe.hbm_bytes, int(die.memory_bytes))
    _cmp("l2_bytes", probe.l2_bytes, int(bufs.get("l2", 0)))
    _cmp("ub_bytes", probe.ub_bytes, int(bufs.get("ub", 0)))
    _cmp("l1_bytes", probe.l1_bytes, int(bufs.get("l1", 0)))
    _cmp("l0a_bytes", probe.l0a_bytes, int(bufs.get("l0a", 0)))
    _cmp("l0b_bytes", probe.l0b_bytes, int(bufs.get("l0b", 0)))
    _cmp("l0c_bytes", probe.l0c_bytes, int(bufs.get("l0c", 0)))
    declared_channels = dict(profile_entry.get("transfer_channels") or {})
    for chan_name, obs_bw in probe.fabric_channels:
        if chan_name not in declared_channels:
            conflicts.append(
                SiliconProfileConflict(
                    field=f"fabric_channels.{chan_name}",
                    observed=obs_bw,
                    declared="absent",
                    reason=f"silicon_profile_conflict:fabric_channel_undeclared:{chan_name}",
                )
            )
            continue
        chan_spec = declared_channels[chan_name]
        if isinstance(chan_spec, Mapping):
            uni_peak = chan_spec.get("uni_peak_bw_gbps")
            eff_bw = chan_spec.get("effective_uni_bw_gbps") or uni_peak
            bi_peak = chan_spec.get("bi_peak_bw_gbps")
            if uni_peak is None and eff_bw is None:
                if obs_bw > 0.0:
                    conflicts.append(
                        SiliconProfileConflict(
                            field=f"fabric_channels.{chan_name}",
                            observed=obs_bw,
                            declared=0.0,
                            reason=f"silicon_profile_conflict:fabric_channel_unavailable:{chan_name}",
                        )
                    )
            else:
                lo_ref = float(eff_bw if eff_bw is not None else uni_peak) * _FABRIC_BW_MIN_RATIO
                hi_ref = float(max(v for v in (uni_peak, eff_bw, bi_peak) if v is not None)) * _FABRIC_BW_MAX_RATIO
                if obs_bw <= 0.0 or obs_bw < lo_ref or obs_bw > hi_ref:
                    conflicts.append(
                        SiliconProfileConflict(
                            field=f"fabric_channels.{chan_name}",
                            observed=obs_bw,
                            declared=float(uni_peak if uni_peak is not None else eff_bw),
                            reason=(
                                f"silicon_profile_conflict:fabric_channel_bw_mismatch:{chan_name}"
                                f":observed={obs_bw}:declared_peak={uni_peak if uni_peak is not None else eff_bw}"
                            ),
                        )
                    )
    if int(buf_units.get("l2", 0)) != 1:
        conflicts.append(
            SiliconProfileConflict(
                field="buffer_units.l2",
                observed=1,
                declared=int(buf_units.get("l2", 0)),
                reason="silicon_profile_conflict:buffer_units.l2_must_equal_1",
            )
        )
    summary = {
        "arch_id": str(dp.arch_id),
        "aic_count": int(dp.aic_count),
        "aiv_count": int(dp.aiv_count),
        "has_l0c_to_ub": bool(dp.has_l0c_to_ub),
        "has_ub_to_l1": bool(dp.has_ub_to_l1),
        "buffer_units_l2": int(buf_units.get("l2", 0)),
        "fabric_channels": sorted(k for k, _ in hw.transfer_channels),
    }
    return conflicts, summary


def probe_soc(
    *,
    endpoint_alias: str = "",
    probe_json: str | Path | None = None,
    profiles_path: str | Path | None = None,
    scaffold_out: str | Path | None = None,
    allow_scaffold: bool = False,
    base_template_sku: str = "",
    container: str = "",
    npu_id: int = 0,
    campaign_id: str = "camp-probe-soc",
    timeout_s: int = DEFAULT_PROBE_TIMEOUT_S,
    ssh_runner: Callable[[str, str, int], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Stage 1: Probe silicon SoC truth and validate or scaffold ``ascend.json`` profile."""
    if probe_json is None and not endpoint_alias:
        raise ModelingError("probe_soc_requires_endpoint_or_probe_json")
    if container and not _SAFE_CONTAINER_RE.fullmatch(container):
        raise ModelingError("probe_soc_container_invalid")

    if probe_json is not None:
        raw_probe = read_bounded_json(probe_json)
        if not isinstance(raw_probe, Mapping):
            raise ModelingError("probe_soc_json_must_be_mapping")
    else:
        probe_script = (
            "SOC=$(python3 -c 'import torch_npu,torch; print(torch.npu.get_device_name(0))' 2>/dev/null || true); "
            "echo \"__SOC__=$SOC\"; "
            "echo '__COMMON__'; npu-smi info -t common -i " + str(int(npu_id)) + "; "
            "echo '__TOPO__'; npu-smi info -t topo 2>/dev/null || true; "
            "INI=$(find /usr/local/Ascend -name \"${SOC}.ini\" 2>/dev/null | head -n 1); "
            "echo '__INI__'; [ -n \"$INI\" ] && cat \"$INI\" || true"
        )
        remote_cmd = (
            f"docker exec -i {shlex.quote(container)} bash -lc {shlex.quote(probe_script)}"
            if container
            else f"bash -lc {shlex.quote(probe_script)}"
        )
        done = (
            ssh_runner(endpoint_alias, remote_cmd, timeout_s)
            if ssh_runner is not None
            else _ssh_exec(endpoint_alias, remote_cmd, timeout_s=timeout_s)
        )
        if done.returncode != 0:
            raise ModelingError(f"probe_soc_ssh_failed:rc={done.returncode}")
        out = done.stdout or ""
        soc_m = re.search(r"__SOC__=(.+)", out)
        common_part = out.split("__COMMON__", 1)[1].split("__TOPO__", 1)[0] if "__COMMON__" in out else ""
        topo_part = out.split("__TOPO__", 1)[1].split("__INI__", 1)[0] if "__TOPO__" in out else ""
        ini_part = out.split("__INI__", 1)[1] if "__INI__" in out else ""
        raw_probe = {
            "aclrt_soc_name": soc_m.group(1).strip() if soc_m else "",
            "platform_config_ini": ini_part,
            "npu_smi_common": common_part,
            "npu_smi_topo": topo_part,
            "endpoint_alias": endpoint_alias,
        }

    ini_parsed = parse_platform_config_ini(str(raw_probe.get("platform_config_ini") or ""))
    smi_common_text = str(raw_probe.get("npu_smi_common") or "")
    smi_parsed = parse_npu_smi_topology(smi_common_text, str(raw_probe.get("npu_smi_topo") or ""))
    source_conflicts: list[SiliconProfileConflict] = []

    def _agree(field: str, candidates: tuple[tuple[str, Any], ...]) -> Any:
        present = [(src, val) for src, val in candidates if val not in (None, "", 0)]
        if not present:
            return "" if field in ("soc_name", "arch_preset", "architecture") else 0
        first_src, first_val = present[0]
        for other_src, other_val in present[1:]:
            if other_val != first_val:
                source_conflicts.append(
                    SiliconProfileConflict(
                        field=f"{field}:{other_src}",
                        observed=other_val,
                        declared=first_val,
                        reason=f"silicon_profile_conflict:source_mismatch:{field}:{other_src}={other_val}:{first_src}={first_val}",
                    )
                )
        return first_val

    soc_name = str(
        _agree(
            "soc_name",
            (
                ("raw.soc_name", str(raw_probe.get("soc_name") or "").strip()),
                ("aclrt_soc_name", str(raw_probe.get("aclrt_soc_name") or "").strip()),
                ("platform_config_ini.soc_version", str(ini_parsed.get("soc_version") or "").strip()),
            ),
        )
    ).strip()
    if not soc_name:
        raise ModelingError("probe_soc_name_unresolvable")

    aic = int(_agree("aic_count", (("raw", int(raw_probe.get("aic_count") or 0)), ("ini", int(ini_parsed["aic_count"] or 0)), ("npu_smi", int(smi_parsed["aic_count"] or 0)))))
    aiv = int(_agree("aiv_count", (("raw", int(raw_probe.get("aiv_count") or 0)), ("ini", int(ini_parsed["aiv_count"] or 0)))))
    freq = int(_agree("aicore_freq_mhz", (("raw", int(raw_probe.get("aicore_freq_mhz") or 0)), ("ini", int(ini_parsed["aicore_freq_mhz"] or 0)), ("npu_smi", int(smi_parsed["aicore_freq_mhz"] or 0)))))
    hbm_bytes = int(_agree("hbm_bytes", (("raw", int(raw_probe.get("hbm_bytes") or 0)), ("ini", int(ini_parsed["hbm_bytes"] or 0)), ("npu_smi", int(smi_parsed["hbm_bytes"] or 0)))))
    l2_bytes = int(_agree("l2_bytes", (("raw", int(raw_probe.get("l2_bytes") or 0)), ("ini", int(ini_parsed["l2_bytes"] or 0)))))
    ub_bytes = int(_agree("ub_bytes", (("raw", int(raw_probe.get("ub_bytes") or 0)), ("ini", int(ini_parsed["ub_bytes"] or 0)))))
    l1_bytes = int(_agree("l1_bytes", (("raw", int(raw_probe.get("l1_bytes") or 0)), ("ini", int(ini_parsed["l1_bytes"] or 0)))))
    l0a_bytes = int(_agree("l0a_bytes", (("raw", int(raw_probe.get("l0a_bytes") or 0)), ("ini", int(ini_parsed["l0a_bytes"] or 0)))))
    l0b_bytes = int(_agree("l0b_bytes", (("raw", int(raw_probe.get("l0b_bytes") or 0)), ("ini", int(ini_parsed["l0b_bytes"] or 0)))))
    l0c_bytes = int(_agree("l0c_bytes", (("raw", int(raw_probe.get("l0c_bytes") or 0)), ("ini", int(ini_parsed["l0c_bytes"] or 0)))))
    smi_dies = int(smi_parsed["dies_per_package"]) if smi_common_text.strip() else 0
    dies_pkg = int(_agree("dies_per_package", (("raw", int(raw_probe.get("dies_per_package") or 0)), ("npu_smi", smi_dies))) or 1)
    ranks_node = int(raw_probe.get("ranks_per_node") or smi_parsed["ranks_per_node"] or dies_pkg)
    raw_arch_norm, _ = _resolve_explicit_arch(str(raw_probe.get("arch_preset") or ""), "")
    arch_preset = str(_agree("arch_preset", (("raw", raw_arch_norm), ("ini", str(ini_parsed["arch_preset"] or "")))))
    architecture = str(_agree("architecture", (("raw", str(raw_probe.get("architecture") or "").strip()), ("ini", str(ini_parsed.get("architecture") or "").strip()))))
    resolved_template_sku = str(base_template_sku or raw_probe.get("base_template_sku") or "").strip()
    clean_alias = sanitize_public_alias(
        str(raw_probe.get("endpoint_alias") or endpoint_alias or soc_name),
        soc_name,
    )
    fabric_map = raw_probe.get("fabric_channels")
    fabric_channels = (
        tuple(sorted((str(k), float(v)) for k, v in fabric_map.items()))
        if isinstance(fabric_map, Mapping)
        else ()
    )
    sanitized_body = {
        "schema": SOC_PROBE_SCHEMA, "soc_name": soc_name, "arch_preset": arch_preset,
        "architecture": architecture, "ini_sha256": ini_parsed["ini_sha256"],
        "aic_count": aic, "aiv_count": aiv, "aicore_freq_mhz": freq,
        "hbm_bytes": hbm_bytes, "l2_bytes": l2_bytes, "ub_bytes": ub_bytes,
        "l1_bytes": l1_bytes, "l0a_bytes": l0a_bytes, "l0b_bytes": l0b_bytes, "l0c_bytes": l0c_bytes,
        "dies_per_package": dies_pkg, "ranks_per_node": ranks_node,
        "cann_version": str(raw_probe.get("cann_version") or "unknown"),
        "endpoint_alias": clean_alias, "fabric_channels": {k: v for k, v in fabric_channels},
    }
    capture_uri, probe_sha256 = build_capture_uri(
        clean_alias, campaign_id, canonical_json_bytes(sanitized_body)
    )
    probe = SocProbeRecord(
        soc_name=soc_name, arch_preset=arch_preset, architecture=architecture,
        aic_count=aic, aiv_count=aiv, aicore_freq_mhz=freq,
        hbm_bytes=hbm_bytes, l2_bytes=l2_bytes, ub_bytes=ub_bytes,
        l1_bytes=l1_bytes, l0a_bytes=l0a_bytes, l0b_bytes=l0b_bytes, l0c_bytes=l0c_bytes,
        dies_per_package=dies_pkg, ranks_per_node=ranks_node,
        cann_version=str(raw_probe.get("cann_version") or "unknown"),
        endpoint_alias=clean_alias, capture_uri=capture_uri, probe_sha256=probe_sha256,
        fabric_channels=fabric_channels, topology_preset=str(raw_probe.get("topology_preset") or ""),
    )

    all_profiles = read_bounded_json(profiles_path) if profiles_path is not None else ascend_profiles()
    probe_cann_raw = str(raw_probe.get("cann_version") or "").strip()
    probe_cann = "" if probe_cann_raw.lower() in {"", "unknown", "unmeasured"} else probe_cann_raw
    if soc_name in all_profiles:
        profile_conflicts, datapath_summary = _validate_profile_against_probe(probe, all_profiles[soc_name])
        conflicts = [*source_conflicts, *profile_conflicts]
        missing_evidence: list[str] = []
        if not ini_parsed["ini_sha256"]:
            missing_evidence.append("platform_config_ini")
        if probe.aiv_count < 1:
            missing_evidence.append("aiv_count")
        if probe.l2_bytes < 1:
            missing_evidence.append("l2_bytes")
        if probe.ub_bytes < 1:
            missing_evidence.append("ub_bytes")
        ok = not conflicts and not missing_evidence
        err_code = (
            ""
            if ok
            else ("silicon_profile_conflict" if conflicts else "silicon_probe_incomplete")
        )
        result: dict[str, Any] = {
            "schema": SOC_PROBE_SCHEMA,
            "stage": "probe-soc",
            "mode": "validated_existing",
            "sku": soc_name,
            "probe_cann_version": probe_cann,
            "ok": ok,
            "status": "ok" if ok else "failed",
            "error": err_code,
            "missing_probe_fields": missing_evidence,
            "ini_sha256": ini_parsed["ini_sha256"],
            "probe": probe.to_dict(),
            "datapath_summary": datapath_summary,
            "canonical_bindings": _derive_canonical_bindings(soc_name, datapath_summary["arch_id"]),
            "conflicts": [c.to_dict() for c in conflicts],
        }
        assert_sanitized_payload(result, field_name="probe_soc_result")
        return result

    missing_caps = _missing_scaffold_capabilities(probe, resolved_template_sku)
    if missing_caps:
        result = {
            "schema": SOC_PROBE_SCHEMA,
            "stage": "probe-soc",
            "mode": "capabilities_incomplete",
            "sku": soc_name,
            "probe_cann_version": probe_cann,
            "ok": False,
            "status": "failed",
            "error": "silicon_capabilities_incomplete",
            "missing_capabilities": missing_caps,
            "ini_sha256": ini_parsed["ini_sha256"],
            "probe": probe.to_dict(),
            "conflicts": [c.to_dict() for c in source_conflicts],
        }
        assert_sanitized_payload(result, field_name="probe_soc_result")
        return result

    scaffolded = _scaffold_sku_profile(
        probe,
        all_profiles,
        base_template_sku=resolved_template_sku,
    )
    profile_conflicts, datapath_summary = _validate_profile_against_probe(probe, scaffolded)
    conflicts = [*source_conflicts, *profile_conflicts]
    if scaffold_out is not None:
        write_atomic_json(scaffold_out, {soc_name: scaffolded})
    can_succeed = (allow_scaffold or scaffold_out is not None) and not conflicts
    result = {
        "schema": SOC_PROBE_SCHEMA,
        "stage": "probe-soc",
        "mode": "scaffolded_new" if can_succeed else "profile_missing",
        "sku": soc_name,
        "probe_cann_version": probe_cann,
        "ok": can_succeed,
        "status": "scaffolded" if can_succeed else "failed",
        "error": "" if can_succeed else ("silicon_profile_conflict" if conflicts else "sku_profile_missing"),
        "ini_sha256": ini_parsed["ini_sha256"],
        "probe": probe.to_dict(),
        "datapath_summary": datapath_summary,
        "canonical_bindings": _derive_canonical_bindings(soc_name, datapath_summary["arch_id"]),
        "conflicts": [c.to_dict() for c in conflicts],
        "scaffold_profile": {soc_name: scaffolded},
    }
    assert_sanitized_payload(result, field_name="probe_soc_result")
    return result


__all__ = [
    "ARCH_PRESET_22",
    "ARCH_PRESET_35",
    "parse_npu_smi_topology",
    "parse_platform_config_ini",
    "probe_soc",
    "run_ssh_command",
    "verify_idle_die_telemetry",
]
