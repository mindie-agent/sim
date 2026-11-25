"""Turnkey 4-stage New-SKU onboarding pipeline (probe -> calibrate -> 512..400K -> gates).

[Role] Orchestrate standardized onboarding of a new or existing Ascend SKU:
  * Stage 1 (``probe_soc``): probe ``aclrtGetSocName``, CANN
    ``platform_config/<SocName>.ini``, and ``npu-smi`` topology via SSH or
    ``--probe-json`` replay, validate existing ``ascend.json`` +
    ``DatapathCapabilitySpec`` + fabric channels (failing closed on
    ``silicon_profile_conflict``), or scaffold a verified profile entry.
  * Stage 2 (``calibrate_l0_and_fabric``): enforce ``TimeoutBudget``, acquire a
    ``HostQueue`` lease on live runs, run ``tools/bench_l0_primitives.py`` or
    seal raw JSON via the single canonical ``mindie_sim.primitives.sealer``,
    and emit ``data/primitives/<arch>_<sku>.json`` + ``train`` manifest.
  * Stage 3 (``validate_512_400k``): synthesize the disjoint 512..400K Anchor +
    Off-Grid Cliff + 2D + L1/L2/L3 plan and score via ``score_sealed_gate_cases``.
  * Stage 4 (``verify_sku_gates`` & ``run_sku_onboarding``): enforce pointwise
    ``max(10%, 2.0 us)`` multilevel gate, formal ``OptimalityCertificate``
    non-inversion, ``check_arch_governance.py --strict``,
    ``check_test_orthogonality.py``, auto-trigger ``diagnose_tail_cluster.py``
    on failure, and seal a SHA-256 ``OnboardingReceipt``.
[Invariants] Zero duplicate primitive sealers; zero raw IPv4/hostnames/usernames
  in emitted payloads; ``calibrated_count == 0`` returns ``ok=False``;
  ``TimeoutError`` always propagates before ``OSError``.
[Timeouts & Locks] Explicit ``TimeoutBudget`` enforcing
  ``outer >= sum(inner) + cleanup_grace``; bounded subprocesses; leases unwind
  in ``finally``.
[Side-Effect Policy] Writes only caller-requested output paths via atomic
  ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Any

from mindie_sim.backends import ascend_profiles
from mindie_sim.observations.multilevel import FrozenFitManifest
from mindie_sim.primitives import sealer as _canonical_sealer
from mindie_sim.primitives.loader import default_primitives_dir
from mindie_sim.primitives.schema import PrimitiveTable, parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die, scale_explicit_placement_device
from mindie_sim.schema.hardware import Hardware
from mindie_sim.feature_mechanisms import default_stack_registry

from .contracts import (
    ARCH_PRESET_22, ARCH_PRESET_35, COLD_RING_L2_MULTIPLIER, DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_FABRIC_BUDGET_S, DEFAULT_HELDOUT_APE_THRESHOLD_PCT, DEFAULT_L0_BUDGET_S,
    DEFAULT_PHYSICAL_ATOL_NS, DEFAULT_PROBE_TIMEOUT_S, DEFAULT_SSH_CONNECT_TIMEOUT_S,
    GIB_BYTES, MIB_BYTES, SHORT_DIGEST_LEN, SKU_ONBOARDING_RECEIPT_SCHEMA,
    TRAIN_OBSERVATION_MANIFEST_SCHEMA, SiliconProfileConflict, SocProbeRecord, TimeoutBudget,
    assert_sanitized_payload, build_capture_uri, collect_receipt_warnings, evaluate_abi_rejections,
    finalize_stage4_certification, load_frozen_fit_manifest, parse_lineage_edges, read_bounded_json, resolve_repo_root,
    run_python_script, sanitize_lease_for_receipt, sanitize_public_alias,
    sanitize_public_campaign_id, sha256_bytes, subprocess_window_kwargs,
    synthesize_sku_512_400k_plan, validate_strict_atol_ns, validate_strict_threshold_pct,
    verify_fits_disjoint_from_residuals, write_atomic_json,
)
from .profiler_normalizer import (
    acquire_coordinator_host_queue_lease, acquire_stage3_observations, assign_plan_point_level,
    validate_host_queue_grant,
)
from .whitebox_runner import (
    materialize_sku_hardware, resolve_sku_profile, score_sealed_gate_cases, temporary_scaffold_profile,
)

MAX_IDLE_AICORE_UTIL_PCT = 2
MAX_IDLE_HBM_UTIL_PCT = 10
_DEFAULT_SSH_PORT = 22
_FABRIC_BW_MIN_RATIO = 0.5
_FABRIC_BW_MAX_RATIO = 1.25
_ABSENT_CANN_TOKENS = frozenset({"", "unmeasured", "unknown"})
_KNOWN_ARCH_PRESETS = frozenset({ARCH_PRESET_22, ARCH_PRESET_35})
_ISA_TOKEN_TO_PRESET = {
    "npu_arch_2201": ARCH_PRESET_22, "arch22": ARCH_PRESET_22, "dav-c220": ARCH_PRESET_22,
    "npu_arch_3510": ARCH_PRESET_35, "arch35": ARCH_PRESET_35, "dav-c310": ARCH_PRESET_35,
}
_SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SIZE_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmg]?i?b)?\s*$", re.IGNORECASE)
_UNIT_MULTIPLIERS = {"": 1, "b": 1, "kb": 1024, "kib": 1024, "mb": MIB_BYTES, "mib": MIB_BYTES, "gb": GIB_BYTES, "gib": GIB_BYTES}
_SAFE_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_canonical_l0_sealer(repo_root: str | Path | None = None) -> Any:
    """Return the single canonical installed L0 primitive sealer ``mindie_sim.primitives.sealer``."""
    del repo_root
    return _canonical_sealer


def _parse_size_bytes(raw: Any, *, default_unit: str = "b") -> int:
    if raw is None or raw == "":
        return 0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(round(float(raw) * _UNIT_MULTIPLIERS[default_unit]))
    m = _SIZE_UNIT_RE.match(str(raw))
    return int(round(float(m.group(1)) * _UNIT_MULTIPLIERS.get((m.group(2) or default_unit).lower(), 1))) if m else 0


def parse_platform_config_ini(ini_text: str) -> dict[str, Any]:
    """Parse CANN ``platform_config/<SocName>.ini`` hardware facts."""
    vals: dict[str, str] = {}
    for line in ini_text.splitlines():
        clean = line.split("#", 1)[0].split(";", 1)[0].strip()
        if clean and not clean.startswith("[") and "=" in clean:
            k, v = clean.split("=", 1)
            vals[k.strip().lower()] = v.strip()

    def _first(*keys: str) -> str:
        return next((vals[k] for k in keys if vals.get(k)), "")

    def _int_of(*keys: str) -> int:
        m = re.match(r"(\d+)", _first(*keys))
        return int(m.group(1)) if m else 0

    isa = _first("arch_preset", "compute_arch", "isa_version", "cube_arch", "ai_core_arch").lower()
    preset = next((p for tok, p in _ISA_TOKEN_TO_PRESET.items() if tok in isa), "")
    return {
        "soc_version": _first("soc_version", "soc_name"), "arch_preset": preset,
        "architecture": _first("architecture", "soc_family", "product_family"),
        "ini_sha256": sha256_bytes(ini_text.encode("utf-8")) if ini_text.strip() else "",
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
    def _int_f(label: str) -> int:
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", common_text)
        return int(m.group(1)) if m else 0

    hbm_mb = _int_f("HBM Capacity(MB)")
    dies = max(1, len(set(re.findall(r"(?m)^\s*Chip ID\s*:\s*(\d+)", common_text))))
    ranks = max(dies, len(re.findall(r"(?m)^\s*NPU\d+\b", topo_text)))
    fab = {k: 1.0 for tok, k in (("SIO", "d2d_intra_package"), ("HCCS", "d2d_intra_node")) if tok in topo_text}
    return {"aic_count": _int_f("Aicore Count"), "aicore_freq_mhz": _int_f("Aicore Freq(MHZ)"), "hbm_bytes": hbm_mb * MIB_BYTES if hbm_mb > 0 else 0, "dies_per_package": dies, "ranks_per_node": ranks, "fabric_hints": fab}


def verify_idle_die_telemetry(info_text: str, common_text: str, *, npu_id: int = 0) -> dict[str, Any]:
    """Fail closed unless target NPU die has zero running processes and idle AICore/HBM."""
    if not re.search(rf"No running processes found in NPU {npu_id}(?!\d)", info_text):
        raise ModelingError(f"onboarding_npu_not_idle:npu={npu_id}")

    def _opt_int(label: str) -> int | None:
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", common_text)
        return int(m.group(1)) if m else None

    aic_u, hbm_u = _opt_int("Aicore Usage Rate(%)"), _opt_int("HBM Usage Rate(%)")
    if aic_u is None or hbm_u is None:
        raise ModelingError("onboarding_npu_telemetry_missing")
    if aic_u > MAX_IDLE_AICORE_UTIL_PCT:
        raise ModelingError(f"onboarding_npu_busy:aicore_util={aic_u}")
    if hbm_u > MAX_IDLE_HBM_UTIL_PCT:
        raise ModelingError(f"onboarding_npu_busy:hbm_util={hbm_u}")
    return {"npu_id": npu_id, "aicore_util_pct": aic_u, "hbm_util_pct": hbm_u, "observed_aic_count": _opt_int("Aicore Count") or 0, "observed_aicore_freq_mhz": _opt_int("Aicore Freq(MHZ)") or 0}


def _ssh_exec(endpoint_alias: str, remote_cmd: str, *, timeout_s: int, endpoints_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    raw = read_bounded_json(endpoints_path or (Path.home() / ".mindie" / "endpoints.json"))
    eps = raw.get("endpoints", raw) if isinstance(raw, Mapping) else {}
    if endpoint_alias not in eps:
        raise ModelingError(f"onboarding_endpoint_unknown:{sanitize_public_alias(endpoint_alias, 'ep')}")
    ep = dict(eps[endpoint_alias])
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_S}", "-o", "StrictHostKeyChecking=yes", "-o", "LogLevel=ERROR", "-p", str(int(ep.get("port", _DEFAULT_SSH_PORT)))]
    if ep.get("identity_file"):
        cmd.extend(["-i", str(ep["identity_file"])])
    cmd.extend(["-l", str(ep.get("user", "root")), "--", str(ep["host"]), remote_cmd])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False, **subprocess_window_kwargs())
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"onboarding_ssh_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_ssh_failed:{exc}") from exc


def _validate_profile_against_probe(probe: SocProbeRecord, entry: Mapping[str, Any]) -> tuple[list[SiliconProfileConflict], dict[str, Any]]:
    hw = Hardware.from_mapping({"sku": probe.soc_name, "architecture": entry["architecture"], "dies_per_package": entry["dies_per_package"], "ranks_per_node": max(1, probe.ranks_per_node), "num_devices": 1, "device": deepcopy(entry["device"]), "transfer_channels": deepcopy(entry.get("transfer_channels", {}))})
    die = hw.dies[0]
    dp = datapath_for_die(die)
    bufs = dict(die.buffers)
    conflicts: list[SiliconProfileConflict] = []

    def _cmp(field: str, obs: Any, decl: Any) -> None:
        if obs and obs != decl:
            conflicts.append(SiliconProfileConflict(field, obs, decl, f"silicon_profile_conflict:{field}:observed={obs}:declared={decl}"))

    _cmp("arch_preset", probe.arch_preset, str(dp.arch_id))
    _cmp("aic_count", probe.aic_count, int(dp.aic_count))
    _cmp("aiv_count", probe.aiv_count, int(dp.aiv_count))
    _cmp("dies_per_package", probe.dies_per_package, int(entry["dies_per_package"]))
    _cmp("hbm_bytes", probe.hbm_bytes, int(die.memory_bytes))
    for fname, val, bk in (("l2_bytes", probe.l2_bytes, "l2"), ("ub_bytes", probe.ub_bytes, "ub"), ("l1_bytes", probe.l1_bytes, "l1"), ("l0a_bytes", probe.l0a_bytes, "l0a"), ("l0b_bytes", probe.l0b_bytes, "l0b"), ("l0c_bytes", probe.l0c_bytes, "l0c")):
        _cmp(fname, val, int(bufs.get(bk, 0)))
    if int(dict(die.buffer_units).get("l2", 1)) != 1:
        conflicts.append(SiliconProfileConflict("buffer_units.l2", 1, dict(die.buffer_units).get("l2"), "silicon_profile_conflict:buffer_units.l2_must_be_1"))
    chans = dict(entry.get("transfer_channels") or {})
    for cname, bw in probe.fabric_channels:
        if cname not in chans:
            conflicts.append(SiliconProfileConflict(f"fabric_channels.{cname}", bw, None, f"silicon_profile_conflict:fabric_channel_missing:{cname}"))
        elif bw > 0.0:
            decl_bw = float(chans[cname].get("uni_peak_bw_gbps") or 0.0)
            if decl_bw > 0.0 and not (_FABRIC_BW_MIN_RATIO * decl_bw <= bw <= _FABRIC_BW_MAX_RATIO * decl_bw):
                conflicts.append(SiliconProfileConflict(f"fabric_channels.{cname}", bw, decl_bw, f"silicon_profile_conflict:fabric_bw_drift:{cname}"))
    return conflicts, {"arch_id": str(dp.arch_id), "aic_count": int(dp.aic_count), "aiv_count": int(dp.aiv_count), "l2_bytes": int(bufs.get("l2", 0)), "hbm_bytes": int(die.memory_bytes), "fabric_channels": sorted(chans)}


def _scaffold_sku_profile(probe: SocProbeRecord, all_profiles: Mapping[str, Any], *, base_template_sku: str) -> dict[str, Any]:
    if base_template_sku not in all_profiles:
        raise ModelingError(f"silicon_template_sku_unknown:{base_template_sku}")
    base = deepcopy(all_profiles[base_template_sku])
    dev = base["device"]
    base_preset = str(dev.get("compute_arch", {}).get("preset") or "")
    if base_preset != probe.arch_preset:
        raise ModelingError(f"silicon_template_arch_mismatch:template={base_template_sku}:template_arch={base_preset}:probe_arch={probe.arch_preset}")
    base_aic = int(dev["compute_units"])
    base_aiv = int(dev["resource_units"].get("vector", base_aic * 2))
    dev["compute_units"], dev["resource_units"]["vector"] = probe.aic_count, probe.aiv_count
    scale_explicit_placement_device(dev, {"compute_units": probe.aic_count, "resource_units": {"vector": probe.aiv_count}}, probe.aic_count, base_aic, base_aiv)
    dev["memory_bytes"] = probe.hbm_bytes
    for pool, val in (("l2", probe.l2_bytes), ("ub", probe.ub_bytes), ("l1", probe.l1_bytes), ("l0a", probe.l0a_bytes), ("l0b", probe.l0b_bytes), ("l0c", probe.l0c_bytes)):
        if val > 0:
            dev["buffers"][pool] = val
    if probe.l2_bytes > 0 and isinstance(dev.get("physical_mechanisms"), dict):
        mem_h = dev["physical_mechanisms"].get("memory_hierarchy")
        if isinstance(mem_h, dict):
            mem_h["l2_capacity_bytes"] = probe.l2_bytes
            if isinstance(mem_h.get("provenance"), dict):
                mem_h["provenance"]["l2_capacity_bytes"] = probe.l2_bytes
    dev["buffer_units"]["l2"] = 1
    base["architecture"], base["dies_per_package"] = probe.architecture, probe.dies_per_package
    for cname, bw in probe.fabric_channels:
        if cname in base.get("transfer_channels", {}) and bw > 0.0:
            base["transfer_channels"][cname]["uni_peak_bw_gbps"] = float(bw)
    base["source"] = {"calibration": {"calibrated": False, "rate_provenance": "declared_nominal", "source_sku": probe.soc_name, "anchor_sku": base_template_sku, "capture_uri": probe.capture_uri}}
    return base


def probe_soc(
    *, endpoint_alias: str = "", probe_json: str | Path | None = None, profiles_path: str | Path | None = None,
    scaffold_out: str | Path | None = None, allow_scaffold: bool = False, base_template_sku: str = "",
    container: str = "", npu_id: int = 0, campaign_id: str = "camp-probe-soc",
    timeout_s: int = DEFAULT_PROBE_TIMEOUT_S, ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Stage 1: Probe silicon SoC facts and validate or scaffold an ``ascend.json`` SKU profile."""
    if probe_json is not None:
        raw_doc = read_bounded_json(probe_json)
        if not isinstance(raw_doc, Mapping):
            raise ModelingError("soc_probe_json_not_mapping")
    elif endpoint_alias.strip():
        if container and not _SAFE_CONTAINER_RE.fullmatch(container):
            raise ModelingError(f"soc_probe_container_invalid:{container}")
        py_cmd = (
            "python3 -c \"import ctypes,json,os,subprocess; "
            "soc=os.environ.get('ASCEND_SOC_NAME',''); "
            f"print(json.dumps({{'aclrt_soc_name':soc,'npu_smi_common':subprocess.run(['npu-smi','info','-t','common','-i','{int(npu_id)}'],capture_output=True,text=True).stdout}}))\""
        )
        remote = f"docker exec {shlex.quote(container)} sh -lc {shlex.quote(py_cmd)}" if container else py_cmd
        done = ssh_runner(endpoint_alias, remote, int(timeout_s)) if ssh_runner else _ssh_exec(endpoint_alias, remote, timeout_s=int(timeout_s))
        if int(getattr(done, "returncode", 1)) != 0:
            raise ModelingError(f"soc_probe_ssh_failed:rc={getattr(done, 'returncode', 1)}")
        raw_doc = json.loads(getattr(done, "stdout", "") or "{}")
    else:
        raise ModelingError("soc_probe_requires_probe_json_or_endpoint_alias")

    ini_f = parse_platform_config_ini(str(raw_doc.get("platform_config_ini") or ""))
    smi_f = parse_npu_smi_topology(str(raw_doc.get("npu_smi_common") or ""), str(raw_doc.get("npu_smi_topo") or ""))
    acl_soc, decl_soc = str(raw_doc.get("aclrt_soc_name") or "").strip(), str(raw_doc.get("soc_name") or "").strip()
    soc_name = acl_soc or decl_soc or ini_f["soc_version"]
    early_conflicts: list[SiliconProfileConflict] = []
    if acl_soc and decl_soc and acl_soc != decl_soc:
        early_conflicts.append(SiliconProfileConflict("soc_name", acl_soc, decl_soc, f"silicon_profile_conflict:soc_name:aclrt={acl_soc}:declared={decl_soc}"))
    if acl_soc and ini_f["soc_version"] and acl_soc != ini_f["soc_version"]:
        early_conflicts.append(SiliconProfileConflict("platform_config_ini.soc_version", ini_f["soc_version"], acl_soc, "silicon_profile_conflict:ini_soc_mismatch"))
    if not soc_name:
        raise ModelingError("soc_probe_missing_soc_name")
    raw_fab = raw_doc.get("fabric_channels")
    fab_pairs = tuple(sorted((str(k), float(v)) for k, v in raw_fab.items())) if isinstance(raw_fab, Mapping) else ()
    raw_preset = str(raw_doc.get("arch_preset") or ini_f["arch_preset"] or "").strip().lower()
    preset = _ISA_TOKEN_TO_PRESET.get(raw_preset, "")
    probe = SocProbeRecord(
        soc_name=soc_name, arch_preset=preset, architecture=str(raw_doc.get("architecture") or ini_f["architecture"] or "").strip(),
        aic_count=int(raw_doc.get("aic_count") or ini_f["aic_count"] or smi_f["aic_count"] or 0),
        aiv_count=int(raw_doc.get("aiv_count") or ini_f["aiv_count"] or 0),
        hbm_bytes=int(raw_doc.get("hbm_bytes") or smi_f["hbm_bytes"] or ini_f["hbm_bytes"] or 0),
        l2_bytes=int(raw_doc.get("l2_bytes") or ini_f["l2_bytes"] or 0),
        ub_bytes=int(raw_doc.get("ub_bytes") or ini_f["ub_bytes"] or 0),
        dies_per_package=max(1, int(raw_doc.get("dies_per_package") or smi_f["dies_per_package"] or 1)),
        ranks_per_node=max(1, int(raw_doc.get("ranks_per_node") or smi_f["ranks_per_node"] or 1)),
        aicore_freq_mhz=int(raw_doc.get("aicore_freq_mhz") or smi_f["aicore_freq_mhz"] or ini_f["aicore_freq_mhz"] or 0),
        l1_bytes=int(raw_doc.get("l1_bytes") or ini_f["l1_bytes"] or 0), l0a_bytes=int(raw_doc.get("l0a_bytes") or ini_f["l0a_bytes"] or 0),
        l0b_bytes=int(raw_doc.get("l0b_bytes") or ini_f["l0b_bytes"] or 0), l0c_bytes=int(raw_doc.get("l0c_bytes") or ini_f["l0c_bytes"] or 0),
        endpoint_alias=sanitize_public_alias(endpoint_alias or str(raw_doc.get("endpoint_alias") or ""), "local"),
        cann_version=str(raw_doc.get("cann_version") or "unmeasured").strip(),
        platform_ini_sha256=ini_f["ini_sha256"],
        capture_uri=build_capture_uri(f"soc-probe/{sanitize_public_campaign_id(campaign_id)}", json.dumps(dict(raw_doc), sort_keys=True).encode("utf-8")),
        fabric_channels=fab_pairs,
    )
    all_profiles = read_bounded_json(profiles_path) if profiles_path is not None else ascend_profiles()
    action, scaffold_profile, datapath_summary, conflicts = "validated_existing", None, {}, list(early_conflicts)
    if soc_name in all_profiles:
        c2, datapath_summary = _validate_profile_against_probe(probe, all_profiles[soc_name])
        conflicts.extend(c2)
    elif allow_scaffold:
        missing = [k for k, v in (("arch_preset", probe.arch_preset in _KNOWN_ARCH_PRESETS), ("architecture", bool(probe.architecture)), ("aic_count", probe.aic_count > 0), ("aiv_count", probe.aiv_count > 0), ("hbm_bytes", probe.hbm_bytes > 0), ("l2_bytes", probe.l2_bytes > 0), ("ub_bytes", probe.ub_bytes > 0), ("base_template_sku", bool(base_template_sku.strip()))) if not v]
        if missing:
            raise ModelingError(f"soc_probe_scaffold_missing_fields:{','.join(missing)}")
        entry = _scaffold_sku_profile(probe, all_profiles, base_template_sku=base_template_sku)
        c2, datapath_summary = _validate_profile_against_probe(probe, entry)
        conflicts.extend(c2)
        action, scaffold_profile = "scaffolded_new", {soc_name: entry}
        if scaffold_out is not None and not conflicts:
            write_atomic_json(scaffold_out, scaffold_profile)
    elif early_conflicts:
        action = "conflict_detected"
    else:
        raise ModelingError(f"silicon_profile_unknown:{soc_name}")

    slug = _SAFE_SLUG_RE.sub("_", soc_name.lower()).strip("_")
    arch_tag = "arch35" if (probe.arch_preset or datapath_summary.get("arch_id")) == ARCH_PRESET_35 else "arch22"
    result: dict[str, Any] = {
        "stage": "probe-soc", "ok": not conflicts, "status": "pass" if not conflicts else "fail",
        "action": action, "sku": soc_name, "probe": probe.to_dict(),
        "datapath": datapath_summary, "conflicts": [c.to_dict() for c in conflicts],
        "canonical_bindings": {"compose_sku": soc_name, "primitive_table_relpath": f"data/primitives/{arch_tag}_{slug}.json", "registered_stacks": sorted(default_stack_registry().list_stacks())},
    }
    if conflicts:
        result["error"] = "silicon_profile_conflict"
    if scaffold_profile is not None:
        result["scaffold_profile"] = scaffold_profile
    assert_sanitized_payload(result, field_name="probe_soc_result")
    return result


def _run_live_l0_bench(sku: str, endpoint_alias: str, campaign_id: str, cann_version: str, repo_root: Path, timeout_s: int, visible_devices: Sequence[int], dies_per_pkg: int, profile_override: Mapping[str, Any] | None) -> dict[str, Any]:
    if not endpoint_alias.strip():
        raise ModelingError("live_l0_bench_requires_endpoint_alias")
    script = repo_root / "tools" / "bench_l0_primitives.py"
    if not script.is_file():
        raise ModelingError(f"bench_l0_primitives_not_found:{script}")
    eff_npu, eff_chip = divmod(int(visible_devices[0]), max(1, dies_per_pkg)) if visible_devices else (0, 0)
    with tempfile.TemporaryDirectory(prefix="msim-l0-") as tmp_dir:
        tmp = Path(tmp_dir)
        out_t, out_m = tmp / "l0.json", tmp / "manifest.json"
        args = ["--endpoint-alias", endpoint_alias, "--sku", sku, "--campaign-id", campaign_id, "--cann-version", cann_version, "--npu-id", str(eff_npu), "--chip-id", str(eff_chip), "--out", str(out_t), "--manifest", str(out_m)]
        if profile_override is not None:
            args.extend(["--profile-json", str(write_atomic_json(tmp / "profile.json", {sku: dict(profile_override)}))])
        if visible_devices:
            args.extend(["--visible-devices", ",".join(str(int(d)) for d in visible_devices)])
        done = run_python_script(script, args, cwd=repo_root, timeout_s=timeout_s)
        if done.returncode != 0:
            raise ModelingError(f"bench_l0_primitives_failed:rc={done.returncode}")
        cap = read_bounded_json(tmp / "captures" / f"{out_t.stem}.capture.json")
        if not isinstance(cap, Mapping) or not isinstance(cap.get("payload"), Mapping):
            raise ModelingError("bench_l0_primitives_missing_capture_dict")
        return dict(cap["payload"])


def _manifest_cann_by_digest(repo_root: str | Path | None, manifest_path: str | Path | None) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for p in (*( [Path(manifest_path)] if manifest_path else [] ), resolve_repo_root(repo_root) / "data/primitives/manifest.json", default_primitives_dir() / "manifest.json"):
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError):
            continue
        for entry in (doc.get("tables") or {}).values() if isinstance(doc, Mapping) else ():
            if isinstance(entry, Mapping):
                cann = str(entry.get("cann_version") or "").strip()
                if cann.lower() in _ABSENT_CANN_TOKENS:
                    continue
                if entry.get("capture_uri") and "/" in str(entry["capture_uri"]):
                    mapping.setdefault(str(entry["capture_uri"]).rsplit("/", 1)[-1].lower(), set()).add(cann)
                if entry.get("envelope_sha256"):
                    mapping.setdefault(str(entry["envelope_sha256"]).lower(), set()).add(cann)
    return mapping


def calibrate_l0_and_fabric(
    *, sku: str, endpoint_alias: str = "", raw_l0_json: str | Path | None = None, raw_fabric_json: str | Path | None = None,
    out_primitives: str | Path | None = None, manifest_path: str | Path | None = None, train_manifest_out: str | Path | None = None,
    campaign_id: str = "camp-onboard-l0", cann_version: str = "unmeasured",
    l0_budget_s: int = DEFAULT_L0_BUDGET_S, fabric_budget_s: int = DEFAULT_FABRIC_BUDGET_S,
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S, outer_timeout_s: int | None = None,
    live_bench: bool = False, repo_root: str | Path | None = None, profiler_root: str | Path | None = None,
    coordinator_bin: str = "", profile_override: Mapping[str, Any] | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    l0_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    fabric_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage 2: Calibrate P01..P20 L0 + fabric via single canonical ``mindie_sim.primitives.sealer``."""
    del profiler_root
    eff_outer = outer_timeout_s if outer_timeout_s is not None else int(l0_budget_s) + int(fabric_budget_s) + int(cleanup_grace_s)
    budget = TimeoutBudget((("l0_primitives", int(l0_budget_s)), ("fabric_dma", int(fabric_budget_s))), cleanup_grace_s=int(cleanup_grace_s), outer_timeout_s=int(eff_outer))
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    clean_camp = sanitize_public_campaign_id(campaign_id, "camp-onboard-l0")
    sealer = load_canonical_l0_sealer(repo_root)

    require_grant = bool(live_bench or lease_acquirer is not None)
    lease_ctx: AbstractContextManager[Mapping[str, Any]] = (
        lease_acquirer(sku, budget.outer_timeout_s) if lease_acquirer is not None
        else (acquire_coordinator_host_queue_lease(endpoint_alias, npu_count=1, ttl_seconds=budget.outer_timeout_s, min_subprocess_timeout_s=budget.total_inner_s, cleanup_grace_s=budget.cleanup_grace_s, coordinator_bin=coordinator_bin) if live_bench
              else nullcontext({"lease_mode": "offline_replay" if (raw_l0_json or l0_runner) else "uncalibrated_dry_run"}))
    )
    cap_digests: list[str] = []
    with lease_ctx as raw_lease:
        if require_grant:
            _, _, granted_devs, _ = validate_host_queue_grant(raw_lease, required_devices=1, min_ttl_seconds=budget.total_inner_s + budget.cleanup_grace_s)
        else:
            granted_devs = []
        raw_l0: dict[str, Any] | None = None
        if raw_l0_json is not None:
            p_l0 = Path(raw_l0_json)
            cap_digests.append(sha256_bytes(p_l0.read_bytes()))
            parsed = read_bounded_json(p_l0)
            raw_l0 = dict(parsed["payload"]) if isinstance(parsed, Mapping) and set(parsed.keys()) == {"tag", "payload"} and isinstance(parsed.get("payload"), Mapping) else dict(parsed)
        elif l0_runner is not None:
            raw_l0 = dict(l0_runner(sku, int(l0_budget_s)))
        elif live_bench:
            prof = resolve_sku_profile(sku, profile_override)
            raw_l0 = _run_live_l0_bench(sku, endpoint_alias, clean_camp, cann_version, resolve_repo_root(repo_root), int(l0_budget_s), granted_devs, int((prof or {}).get("dies_per_package", 1)), profile_override)
        raw_fab: dict[str, Any] | None = None
        if raw_fabric_json is not None:
            p_f = Path(raw_fabric_json)
            cap_digests.append(sha256_bytes(p_f.read_bytes()))
            parsed_f = read_bounded_json(p_f)
            raw_fab = dict(parsed_f["payload"]) if isinstance(parsed_f, Mapping) and set(parsed_f.keys()) == {"tag", "payload"} else dict(parsed_f)
        elif fabric_runner is not None:
            raw_fab = dict(fabric_runner(sku, int(fabric_budget_s)))

    for label, src in (("raw_l0", raw_l0), ("raw_fabric", raw_fab)):
        if isinstance(src, Mapping) and src.get("device_name") and str(src["device_name"]).strip() != sku:
            raise ModelingError(f"bench_sample_sku_mismatch:{label}:expected={sku}:got={src['device_name']}")
    merged = dict(raw_l0 or {}) if (raw_l0 is not None or raw_fab is not None) else None
    if merged is not None and raw_fab:
        for k, v in raw_fab.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = {**dict(merged[k]), **dict(v)}
            else:
                merged[k] = v

    # Cross-check CANN provenance across L0 capture, fabric capture, manifest, and request
    cann_map = _manifest_cann_by_digest(repo_root, manifest_path)
    ev_canns = {
        c for c in (
            *(str((src or {}).get(k) or "").strip() for src in (raw_l0, raw_fab) for k in ("cann_version", "runtime_cann_version")),
            *(c for d in cap_digests for c in cann_map.get(d, ())),
        )
        if c.lower() not in _ABSENT_CANN_TOKENS
    }
    if len(ev_canns) > 1:
        raise ModelingError(f"cann_version_provenance_conflict:observed={sorted(ev_canns)}")
    req_cann = str(cann_version or "").strip()
    if req_cann.lower() not in _ABSENT_CANN_TOKENS and ev_canns and req_cann not in ev_canns:
        raise ModelingError(f"cann_version_provenance_conflict:requested={req_cann}:observed={sorted(ev_canns)}")
    eff_cann = next(iter(sorted(ev_canns)), req_cann if req_cann.lower() not in _ABSENT_CANN_TOKENS else ("unmeasured" if merged is None else "unknown"))

    clean_ep = sanitize_public_alias(endpoint_alias or sku, "local")
    with temporary_scaffold_profile(sku, profile_override):
        table_payload, conflicts, cov_matrix = sealer.build_primitive_table_with_diagnostics(
            sku=sku, cann_version=eff_cann, endpoint_alias=clean_ep, campaign_id=clean_camp, raw_measurements=merged,
        )
    parsed_tbl = parse_primitive_table(table_payload, verify_envelope=True)
    cap_uri = str(table_payload["provenance"]["capture_uri"])
    cal_cnt = sum(1 for v in table_payload["primitives"].values() if v.get("status") == "calibrated")
    if out_primitives is not None:
        write_atomic_json(out_primitives, table_payload)
    train_root = f"{clean_camp}-{sha256_bytes(f'l0-train:{sku}:{parsed_tbl.envelope_sha256}'.encode('utf-8'))[:SHORT_DIGEST_LEN]}"
    train_manifest = {"schema": TRAIN_OBSERVATION_MANIFEST_SCHEMA, "sku": sku, "campaign_id": clean_camp, "cann_version": eff_cann, "train_lineage_root": train_root, "primitives_envelope_sha256": parsed_tbl.envelope_sha256, "capture_uri": cap_uri, "calibrated_count": cal_cnt, "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER, "l2_bytes": int(dict(die.buffers).get("l2", 0))}
    if train_manifest_out is not None:
        write_atomic_json(train_manifest_out, train_manifest)
    lease_pub = sanitize_lease_for_receipt(raw_lease)
    rec_u, rec_w = collect_receipt_warnings(lease_pub)
    ok = cal_cnt > 0
    out: dict[str, Any] = {
        "stage": "calibrate-l0-and-fabric", "ok": ok, "status": "calibrated" if ok else "uncalibrated",
        "sku": sku, "cann_version": eff_cann, "calibrated_count": cal_cnt, "budget": budget.to_dict(),
        "lease": lease_pub, "primitives_envelope_sha256": parsed_tbl.envelope_sha256,
        "primitives_payload": table_payload, "train_manifest": train_manifest,
        "metric_conflicts": conflicts, "coverage_matrix": cov_matrix,
    }
    if rec_u or rec_w:
        out["receipt_unconfirmed"], out["receipt_warning"] = rec_u, rec_w
    assert_sanitized_payload(out, field_name="calibrate_l0_result")
    return out


def validate_512_400k(
    *, sku: str, observations_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    train_manifest_json: str | Path | Mapping[str, Any] | None = None,
    primitives_json: str | Path | Mapping[str, Any] | None = None,
    fits_json: str | Path | Mapping[str, Any] | FrozenFitManifest | None = None,
    lineage_edges: Sequence[Any] = (),
    plan_out: str | Path | None = None, report_out: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT, atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
    campaign_id: str = "camp-onboard-512-400k", live_profiler: bool = False,
    endpoint_alias: str = "", profiler_root: str | Path | None = None, coordinator_bin: str = "",
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    profile_override: Mapping[str, Any] | None = None, runner: Any = None,
    profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stage 3: Score disjoint 512..400K Anchor + Off-Grid Cliff + 2D + L1/L2/L3 benchmark suite."""
    cap_pct, eff_atol = validate_strict_threshold_pct(threshold_pct), validate_strict_atol_ns(atol_ns)
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    plan_doc = synthesize_sku_512_400k_plan(sku=sku, core_count=int(die.compute_units), l2_bytes=int(dict(die.buffers).get("l2", 0)), campaign_id=campaign_id, profile_override=profile_override).to_dict()
    if plan_out is not None:
        write_atomic_json(plan_out, plan_doc)
    prim_tbl: PrimitiveTable | None = None
    if primitives_json is not None:
        prim_tbl = parse_primitive_table(read_bounded_json(primitives_json) if isinstance(primitives_json, (str, Path)) else primitives_json)
    ext_train: list[str] = [str(plan_doc["train_lineage_root"])]
    if train_manifest_json is not None:
        tm = read_bounded_json(train_manifest_json) if isinstance(train_manifest_json, (str, Path)) else train_manifest_json
        if isinstance(tm, Mapping) and tm.get("train_lineage_root"):
            ext_train.append(str(tm["train_lineage_root"]))

    rows, abi_obs, lease_pub, req_npus = acquire_stage3_observations(
        plan_doc, sku=sku, raw_source=observations_json, custom_runner=profiler_runner,
        live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root,
        coordinator_bin=coordinator_bin, capture_root=capture_root, profile_override=profile_override,
        abi_rejection_results=abi_rejection_results, lease_acquirer=lease_acquirer,
    )
    exp_levels = {str(p["point_id"]): assign_plan_point_level(plan_doc, p) for sk in ("train_points", "heldout_points") for p in plan_doc[sk]}
    scored = score_sealed_gate_cases(
        rows, sku=sku, runner=runner, threshold_pct=cap_pct, atol_ns=eff_atol,
        primitive_table=prim_tbl, profile_override=profile_override, external_train_roots=ext_train,
        manifest_lineage_edges=lineage_edges, fits_manifest=fits_json, expected_points=plan_doc["heldout_points"],
        expected_train_points=plan_doc["train_points"], expected_levels=exp_levels,
        capture_root=capture_root, require_capture_files=require_capture_files,
    )
    abi_chk = evaluate_abi_rejections(plan_doc.get("abi_rejection_points") or (), abi_obs)
    ok = bool(scored["ok"]) and bool(abi_chk["ok"])
    status = "insufficient_evidence" if scored["status"] == "insufficient_evidence" else ("pass" if ok else "fail")
    report: dict[str, Any] = {
        "stage": "validate-512-400k", "ok": ok, "status": status, "sku": sku,
        "threshold_pct": cap_pct, "atol_ns": eff_atol, "plan_sha256": plan_doc["plan_sha256"],
        "required_ranks": req_npus, "acquisition_mode": "live_silicon" if live_profiler else "offline_reverify",
        "capture_files_verified": bool(scored["capture_files_verified"]), "lease": lease_pub,
        "primitives_envelope_sha256": prim_tbl.envelope_sha256 if prim_tbl else "",
        "fits_sha256": str(scored.get("fits_sha256") or ""),
        "missing_heldout_points": scored["missing_plan_points"], "missing_train_points": scored["missing_train_points"],
        "over_budget_points": scored["over_budget_points"], "failing_cases": scored["failing_cases"],
        "abi_rejection_check": abi_chk, "lineage": scored["lineage"],
        "heldout_summary": scored["heldout_summary"], "train_summary": scored["train_summary"], "residuals": scored["residuals"],
    }
    rec_u, rec_w = collect_receipt_warnings(lease_pub)
    if rec_u or rec_w:
        report["receipt_unconfirmed"], report["receipt_warning"] = rec_u, rec_w
    assert_sanitized_payload(report, field_name="validate_512_400k_report")
    if report_out is not None:
        write_atomic_json(report_out, report)
    return report


def verify_sku_gates(
    *, sku: str, validation_report_json: str | Path | Mapping[str, Any] | None = None,
    multilevel_manifest_json: str | Path | Mapping[str, Any] | None = None,
    fits_json: str | Path | Mapping[str, Any] | FrozenFitManifest | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    primitives_json: str | Path | Mapping[str, Any] | None = None, expected_cann_version: str = "",
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    repo_root: str | Path | None = None, threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS, run_repo_gates: bool = True, orthogonality_base: str | None = None,
    profile_override: Mapping[str, Any] | None = None, runner: Any = None,
    out_receipt: str | Path | None = None, diagnosis_out: str | Path | None = None,
    stage2_lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 4: Enforce multilevel <10% / max(10%, 2us), certified lower-bound non-inversion, governance & orthogonality gates."""
    cap_pct, eff_atol = validate_strict_threshold_pct(threshold_pct), validate_strict_atol_ns(atol_ns)
    if validation_report_json is None and multilevel_manifest_json is None:
        raise ModelingError("verify_sku_gates_requires_validation_report_or_manifest")
    prim_tbl: PrimitiveTable | None = None
    if primitives_json is not None:
        prim_tbl = parse_primitive_table(read_bounded_json(primitives_json) if isinstance(primitives_json, (str, Path)) else primitives_json)
        if prim_tbl.sku != sku:
            raise ModelingError(f"primitives_sku_mismatch:expected={sku}:got={prim_tbl.sku}")
        if expected_cann_version.strip() and str(prim_tbl.cann_version or "").strip() != expected_cann_version.strip():
            raise ModelingError(f"primitives_cann_version_mismatch:expected={expected_cann_version}:got={prim_tbl.cann_version}")

    fits_obj, fits_sha = (
        load_frozen_fit_manifest(fits_json, default_fit_id=f"onboarding-fits-{sanitize_public_alias(sku, 'sku')}")
        if fits_json is not None
        else (None, "")
    )
    if validation_report_json is not None:
        val_doc = dict(read_bounded_json(validation_report_json) if isinstance(validation_report_json, (str, Path)) else validation_report_json)
        report_fits_sha = str(val_doc.get("fits_sha256") or "").strip().lower()
        if fits_obj is not None:
            if report_fits_sha and report_fits_sha != fits_sha.lower():
                raise ModelingError(f"fits_manifest_digest_mismatch:report={report_fits_sha}:supplied={fits_sha}")
            verify_fits_disjoint_from_residuals(fits_obj, val_doc)
    else:
        mf = read_bounded_json(multilevel_manifest_json) if isinstance(multilevel_manifest_json, (str, Path)) else multilevel_manifest_json
        assert isinstance(mf, Mapping)
        mf_edges = parse_lineage_edges(mf.get("lineage_edges") or ())
        val_doc = validate_512_400k(sku=sku, observations_json=list(mf.get("cases") or ()), abi_rejection_results=list(mf.get("abi_rejection_results") or ()), primitives_json=primitives_json, fits_json=fits_obj, lineage_edges=mf_edges, capture_root=capture_root, require_capture_files=require_capture_files, profile_override=profile_override, runner=runner, threshold_pct=cap_pct, atol_ns=eff_atol)

    artifacts = {
        "primitives_envelope_sha256": prim_tbl.envelope_sha256 if prim_tbl else str(val_doc.get("primitives_envelope_sha256") or ""),
        "plan_sha256": str(val_doc.get("plan_sha256") or ""),
    }
    eff_fits_sha = fits_sha or str(val_doc.get("fits_sha256") or "")
    if eff_fits_sha:
        artifacts["fits_sha256"] = eff_fits_sha
    all_ok, status, receipt, diagnosis = finalize_stage4_certification(
        schema=SKU_ONBOARDING_RECEIPT_SCHEMA, target_kind="sku", target_id=f"sku:{sku}", sku=sku,
        score_doc=val_doc, lean_bounds_json=lean_bounds_json, cap_pct=cap_pct, eff_atol=eff_atol,
        ape_gate_name="multilevel_512_400k",
        artifacts=artifacts,
        extra_warning_records=(sanitize_lease_for_receipt(stage2_lease),),
        capture_root=capture_root, require_capture_files=require_capture_files,
        repo_root=repo_root, run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base,
        out_receipt=out_receipt, diagnosis_out=diagnosis_out,
    )
    out: dict[str, Any] = {"stage": "verify-gates", "ok": all_ok, "status": status, "sku": sku, "receipt": receipt}
    if diagnosis is not None:
        out["diagnosis"] = diagnosis
    return out


def run_sku_onboarding(
    *, sku: str | None = None, endpoint_alias: str = "", probe_json: str | Path | None = None,
    raw_l0_json: str | Path | None = None, raw_fabric_json: str | Path | None = None,
    observations_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None = None,
    out_dir: str | Path | None = None, repo_root: str | Path | None = None,
    allow_scaffold: bool = False, base_template_sku: str = "", run_repo_gates: bool = True,
    orthogonality_base: str | None = None, threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    cann_version: str = "unmeasured", live_bench: bool = False, live_profiler: bool = False,
    profiler_root: str | Path | None = None, coordinator_bin: str = "",
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    runner: Any = None, profiler_runner: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute all 4 New-SKU onboarding stages end-to-end."""
    base_out = Path(out_dir) if out_dir is not None else None
    s1 = probe_soc(endpoint_alias=endpoint_alias, probe_json=probe_json, allow_scaffold=allow_scaffold, base_template_sku=base_template_sku, scaffold_out=(base_out / "scaffold_profile.json") if base_out else None)
    eff_sku = sku or str(s1["sku"])
    prof_ov = (s1.get("scaffold_profile") or {}).get(eff_sku) if isinstance(s1.get("scaffold_profile"), Mapping) else None
    if not s1["ok"]:
        return {"stage": "run-all", "ok": False, "status": "failed", "sku": eff_sku, "stages": {"probe-soc": s1}}
    s2 = calibrate_l0_and_fabric(sku=eff_sku, endpoint_alias=endpoint_alias, raw_l0_json=raw_l0_json, raw_fabric_json=raw_fabric_json, out_primitives=(base_out / "primitives.json") if base_out else None, train_manifest_out=(base_out / "train_manifest.json") if base_out else None, cann_version=cann_version, live_bench=live_bench, repo_root=repo_root, profiler_root=profiler_root, coordinator_bin=coordinator_bin, profile_override=prof_ov)
    s3 = validate_512_400k(sku=eff_sku, observations_json=observations_json, train_manifest_json=s2["train_manifest"], primitives_json=s2["primitives_payload"], plan_out=(base_out / "cliff_plan.json") if base_out else None, report_out=(base_out / "validation_report.json") if base_out else None, threshold_pct=threshold_pct, live_profiler=live_profiler, endpoint_alias=endpoint_alias, profiler_root=profiler_root, coordinator_bin=coordinator_bin, capture_root=capture_root, require_capture_files=require_capture_files, profile_override=prof_ov, runner=runner, profiler_runner=profiler_runner, abi_rejection_results=abi_rejection_results)
    s4 = verify_sku_gates(sku=eff_sku, validation_report_json=s3, lean_bounds_json=lean_bounds_json, primitives_json=s2["primitives_payload"], capture_root=capture_root, require_capture_files=require_capture_files, repo_root=repo_root, threshold_pct=threshold_pct, run_repo_gates=run_repo_gates, orthogonality_base=orthogonality_base, profile_override=prof_ov, runner=runner, out_receipt=(base_out / "sku_onboarding_receipt.json") if base_out else None, diagnosis_out=(base_out / "tail_diagnosis.json") if base_out else None, stage2_lease=s2.get("lease"))
    ok = bool(s1["ok"] and s2["ok"] and s3["ok"] and s4["ok"])
    return {"stage": "run-all", "ok": ok, "status": s4["status"] if ok else "failed", "sku": eff_sku, "stages": {"probe-soc": s1, "calibrate-l0-and-fabric": s2, "validate-512-400k": s3, "verify-gates": s4}}
