#!/usr/bin/env python3
"""Remote SSH execution, idle-die verification, and manifest helpers for L0 bench.

[Role] Plane-1 helper extracted from ``tools/bench_l0_primitives.py`` owning SSH
  transport, strict ``npu-smi`` idle-die validation, remote probe staging, and
  ``data/primitives/manifest.json`` index updates.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Verifies
  idle NPU state and physical-to-logical mapping before executing any probe;
  unconditionally cleans remote temporary directories; never emits raw IPs,
  hostnames, usernames, or host paths.
[Timeouts & Locks] Enforces bounded SSH connect/command timeouts and retries
  only on SSH transport exit 255; escalates TimeoutError cleanly.
[Side-Effect Policy] Stages temporary scripts strictly under remote ``/tmp``
  with EXIT trap cleanup and updates only the requested manifest path locally.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from _bench_l0_probes import RESULT_MARKER

SSH_CONNECT_TIMEOUT_S = 15
SSH_MAX_RETRIES = 3
SSH_FAIL_RC = 255
NPU_SMI_TIMEOUT_S = 20
REMOTE_BENCH_TIMEOUT_S = 720
DEFAULT_SSH_PORT = 22
MAX_IDLE_AICORE_UTIL_PCT = 2
MAX_IDLE_HBM_UTIL_PCT = 10
MIN_FREE_HBM_MIB = 2560
MIN_MAP_COLS = 5
JSON_INDENT = 2
SLOT_CHIP_COL = 2
SLOT_LOGIC_COL = 3
NO_SLOT_CHIP_COL = 1
NO_SLOT_LOGIC_COL = 2
SPLIT_PAIR_COUNT = 2
MIN_CURVE_SAMPLES = 8
MAX_CURVE_CV_PCT = 5.0
PRIMITIVE_SCHEMA_VERSION = "mindie-l0-primitives/1"

SAFE_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_VISIBLE_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
IPV4_QUAD_RE = re.compile(r"(?<!\d)\d{1,3}(?:[._-]\d{1,3}){3}(?!\d)")
PRIVATE_CAPTURE_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}[._-]){3}\d{1,3}(?![\d.])"
    r"|/(?:Users|home|root)/[^\s\"']*"
    r"|[^\s\"']*@[^\s\"']+"
    r"|\$\{?HOME\}?"
    r"|(?:^|[\s\"'/:@])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:invalid|local|internal|corp|lan|arpa|com|net|org|edu|gov|io|cn)\b",
    re.IGNORECASE,
)
SAFE_ALIAS_RE = re.compile(r"[^a-z0-9_-]+")
NEUTRAL_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
KNOWN_RAW_KEYS = frozenset({
    "device_name", "observed_silicon", "status", "sku",
    "P01_datacopy", "P02_load_data", "P03_fixpipe", "P04_mte3_store", "P05_l2_residency_reuse",
    "P06_mmad", "P07_vec_alu", "P08_vec_sfu", "P09_vec_reduce_cast", "P10_sort_topk",
    "P11_simt", "P12_intracore_flag", "P13_crosscore_sync", "P14_mutex_block_mmad",
    "P15_rts_stars", "P16_multistream_core_control", "P17_aicpu_metadata_launch",
    "P18_hccl_collective_band", "P19_direct_peer_window", "P20_host_dma_sync",
})


def sanitize_record_alias(alias: str, sku: str) -> str:
    """Return a clean public endpoint alias with zero raw IPs, usernames, domains, or paths."""
    raw = str(alias or "").strip().lower()
    fallback = f"lab-{SAFE_ALIAS_RE.sub('-', sku.lower()).strip('-')}"
    if not raw or "@" in raw or "/" in raw or "\\" in raw or "." in raw or IPV4_QUAD_RE.search(raw):
        return fallback
    cleaned = SAFE_ALIAS_RE.sub("-", raw).strip("-")
    if not cleaned or cleaned in {"a2", "a3", "a5", "host", "default"} or not NEUTRAL_ALIAS_RE.fullmatch(cleaned):
        return fallback
    return cleaned


def sanitize_capture_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return PRIVATE_CAPTURE_RE.sub("redacted", obj)
    if isinstance(obj, dict):
        return {PRIVATE_CAPTURE_RE.sub("redacted", str(k)): sanitize_capture_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_capture_obj(v) for v in obj]
    return obj


def assert_no_privacy_leak(text: str, *, label: str) -> None:
    if PRIVATE_CAPTURE_RE.search(text):
        raise ValueError(f"bench_privacy_leak_detected:{label}")


def canonical_capture_bytes(sku: str, raw_measurements: dict[str, Any] | None) -> bytes:
    """Serialize sanitized raw calibration samples/summary to canonical JSON bytes."""
    payload = sanitize_capture_obj(raw_measurements) if raw_measurements else {"sku": sku, "status": "unmeasured"}
    body = {"tag": "campaign", "payload": payload}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    assert_no_privacy_leak(encoded, label="capture")
    return encoded.encode("utf-8")


def unwrap_raw_measurements(loaded: Any) -> dict[str, Any] | None:
    """Unwrap a campaign capture envelope at most once and validate its structure."""
    if not isinstance(loaded, dict):
        raise ValueError("bench_raw_json_invalid_shape:not_mapping")
    data: Any = loaded
    if data.get("tag") == "campaign" and isinstance(data.get("payload"), dict):
        data = data["payload"]
        if data.get("tag") == "campaign":
            raise ValueError("bench_raw_json_invalid_shape:double_wrapped_campaign")
    if data.get("status") == "unmeasured" and not any(k.startswith("P") for k in data):
        return None
    if not (KNOWN_RAW_KEYS & set(data.keys())):
        raise ValueError("bench_raw_json_invalid_shape:no_known_measurement_keys")
    return dict(data)


def validate_quantitative_curves(
    curves: Any, *, max_bw_bps: float, max_cv_pct: float = MAX_CURVE_CV_PCT, min_samples: int = MIN_CURVE_SAMPLES,
) -> dict[str, Any]:
    """Validate optional P18/P19/P20 curve points against sample, CV, and physical bounds."""
    if not isinstance(curves, dict) or not curves:
        return {}

    def _check_entry(entry: dict[str, Any]) -> dict[str, Any]:
        out = dict(entry)
        bw_keys = [k for k, v in entry.items() if isinstance(v, (int, float)) and ("bps" in k or "bw" in k)]
        has_bw = bool(bw_keys)
        if not has_bw:
            return out
        n_val = entry.get("n_samples", entry.get("n"))
        cv_val = entry.get("cv_pct")
        issues: list[str] = []
        if not isinstance(n_val, int) or n_val < min_samples:
            issues.append("missing_or_low_n_samples")
        if not isinstance(cv_val, (int, float)) or not math.isfinite(float(cv_val)) or float(cv_val) > max_cv_pct:
            issues.append("missing_or_high_cv_pct")
        for bk in bw_keys:
            bv = float(entry[bk])
            if not math.isfinite(bv) or bv <= 0.0 or (max_bw_bps > 0.0 and bv > max_bw_bps):
                issues.append(f"bound_conflict:{bk}")
        out["calibration_gate_status"] = "calibrated" if not issues else "unverified_diagnostic:" + ",".join(issues)
        return out

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if any(isinstance(v, (int, float)) and ("bps" in k or "bw" in k) for k, v in node.items()):
                return _check_entry(node)
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    return dict(_walk(curves))


def load_endpoint(alias: str) -> dict[str, Any]:
    data = json.loads((Path.home() / ".mindie" / "endpoints.json").read_text(encoding="utf-8"))
    eps = data.get("endpoints", data)
    if alias not in eps:
        raise SystemExit(f"Unknown endpoint alias: {alias}")
    return dict(eps[alias])


def ssh_run(
    ep: dict[str, Any], remote_cmd: str, *, stdin_text: str | None = None, timeout: int = REMOTE_BENCH_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "StrictHostKeyChecking=yes", "-o", "LogLevel=ERROR",
        "-p", str(ep.get("port", DEFAULT_SSH_PORT)),
    ]
    if ep.get("identity_file"):
        cmd.extend(["-i", str(ep["identity_file"])])
    cmd.extend(["-l", str(ep.get("user", "root")), "--", str(ep["host"]), remote_cmd])
    last: subprocess.CompletedProcess[str] | None = None
    for _ in range(SSH_MAX_RETRIES):
        try:
            last = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"bench_ssh_timeout:timeout_s={timeout}") from None
        if last.returncode != SSH_FAIL_RC:
            return last
    assert last is not None
    return last


def parse_npu_smi_idle_die(
    info_text: str, map_text: str, common_text: str, *,
    npu_id: int, chip_id: int, device_id: int, visible_devices: str = "",
) -> dict[str, Any]:
    """Strictly verify physical die idle state, mapping, util, HBM, and clock."""
    if not re.search(rf"No running processes found in NPU {npu_id}(?!\d)", info_text):
        raise RuntimeError(f"bench_npu_not_idle:npu={npu_id}")
    logic_id: int | None = None
    has_slot = "Slot ID" in map_text
    c_idx, l_idx = (SLOT_CHIP_COL, SLOT_LOGIC_COL) if has_slot else (NO_SLOT_CHIP_COL, NO_SLOT_LOGIC_COL)
    for line in map_text.splitlines():
        cols = line.split()
        if len(cols) > l_idx and cols[0].isdigit() and cols[c_idx].isdigit() and cols[l_idx].isdigit():
            if int(cols[0]) == npu_id and int(cols[c_idx]) == chip_id:
                logic_id = int(cols[l_idx])
                break
    if logic_id is None:
        raise ValueError(f"bench_npu_mapping_missing:npu={npu_id}:chip={chip_id}")
    if visible_devices:
        if not SAFE_VISIBLE_RE.fullmatch(visible_devices):
            raise ValueError("bench_visible_devices_invalid")
        vis_list = [int(x) for x in visible_devices.split(",")]
        if device_id < 0 or device_id >= len(vis_list):
            raise ValueError("bench_npu_die_mismatch")
        target_logic = vis_list[device_id]
    else:
        target_logic = device_id
    if target_logic != logic_id:
        raise ValueError(f"bench_npu_die_mismatch:expected_logic={logic_id}:got={target_logic}")
    block = common_text
    if "Chip ID" in common_text:
        parts = re.split(r"(?m)^\s*Chip ID\s*:\s*", common_text)
        matched = [part for part in parts[1:] if re.match(rf"^{chip_id}(?!\d)", part.strip())]
        if not matched:
            raise ValueError(f"bench_npu_common_missing:chip={chip_id}")
        block = matched[0]

    def _int_field(label: str) -> int | None:
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", block)
        return int(m.group(1)) if m else None

    aic_util, hbm_util = _int_field("Aicore Usage Rate(%)"), _int_field("HBM Usage Rate(%)")
    hbm_cap_mb = _int_field("HBM Capacity(MB)")
    aic_cnt = _int_field("Aicore Count")
    aic_freq, cur_freq = _int_field("Aicore Freq(MHZ)"), _int_field("Aicore curFreq(MHZ)")
    if aic_util is None or hbm_util is None:
        raise RuntimeError("bench_npu_telemetry_missing")
    if aic_cnt is None or aic_cnt <= 0 or aic_freq is None or aic_freq <= 0:
        raise RuntimeError("bench_npu_clock_missing")
    if aic_util > MAX_IDLE_AICORE_UTIL_PCT:
        raise RuntimeError(f"bench_npu_busy:aicore_util={aic_util}")
    if hbm_util > MAX_IDLE_HBM_UTIL_PCT:
        raise RuntimeError(f"bench_npu_busy:hbm_util={hbm_util}")
    if hbm_cap_mb is not None and hbm_cap_mb > 0:
        free_mib = hbm_cap_mb * (100 - hbm_util) / 100.0
        if free_mib < MIN_FREE_HBM_MIB:
            raise RuntimeError(f"bench_npu_low_free_hbm:free_mib={int(free_mib)}")
    hbm_pair = re.search(rf"\|\s*{npu_id}\s+\S+.*?(\d+)\s*/\s*(\d+)\s*\|", info_text)
    if hbm_pair:
        used_mib, total_mib = int(hbm_pair.group(1)), int(hbm_pair.group(2))
        if total_mib > 0 and (total_mib - used_mib) < MIN_FREE_HBM_MIB:
            raise RuntimeError(f"bench_npu_low_free_hbm:free_mib={total_mib - used_mib}")
    return {
        "npu_id": npu_id, "chip_id": chip_id, "chip_logic_id": target_logic,
        "aicore_util_pct": aic_util, "hbm_util_pct": hbm_util,
        "observed_aic_count": aic_cnt, "observed_aicore_freq_mhz": aic_freq, "observed_cur_freq_mhz": cur_freq,
    }


def _parse_npu_smi_bundle(stdout: str, npu_id: int, chip_id: int, device_id: int, visible_devices: str) -> dict[str, Any]:
    parts = stdout.split("__MAP__", 1)
    if len(parts) != SPLIT_PAIR_COUNT or "__COMMON__" not in parts[1]:
        raise RuntimeError("bench_npu_smi_parse_failed")
    info_text = parts[0]
    map_text, common_text = parts[1].split("__COMMON__", 1)
    return parse_npu_smi_idle_die(
        info_text, map_text, common_text,
        npu_id=int(npu_id), chip_id=int(chip_id), device_id=int(device_id), visible_devices=visible_devices,
    )


def check_idle_die(
    ep: dict[str, Any], npu_id: int, chip_id: int, device_id: int, visible_devices: str,
) -> dict[str, Any]:
    cmd = f"npu-smi info; echo '__MAP__'; npu-smi info -m; echo '__COMMON__'; npu-smi info -t common -i {int(npu_id)}"
    res = ssh_run(ep, cmd, timeout=NPU_SMI_TIMEOUT_S)
    if res.returncode != 0:
        raise RuntimeError(f"bench_npu_smi_failed:rc={res.returncode}")
    return _parse_npu_smi_bundle(res.stdout, npu_id, chip_id, device_id, visible_devices)


def check_local_idle_die(
    npu_id: int, chip_id: int, device_id: int, visible_devices: str = "",
) -> dict[str, Any]:
    cmd = f"npu-smi info; echo '__MAP__'; npu-smi info -m; echo '__COMMON__'; npu-smi info -t common -i {int(npu_id)}"
    try:
        res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=NPU_SMI_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"bench_local_npu_smi_timeout:timeout_s={NPU_SMI_TIMEOUT_S}") from None
    if res.returncode != 0:
        raise RuntimeError(f"bench_npu_smi_failed:rc={res.returncode}")
    return _parse_npu_smi_bundle(res.stdout, npu_id, chip_id, device_id, visible_devices)


def run_remote_calibration(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    ep = load_endpoint(args.endpoint_alias)
    vis = str(args.visible_devices or "")
    obs = check_idle_die(ep, int(args.npu_id), int(args.chip_id), int(args.device_id), vis)
    if int(args.peer_device_id) >= 0:
        check_idle_die(ep, int(args.peer_npu_id), int(args.peer_chip_id), int(args.peer_device_id), vis)
    if args.container and not SAFE_CONTAINER_RE.fullmatch(args.container):
        raise ValueError("bench_container_invalid")
    tools_dir = Path(__file__).resolve().parent
    probe_b64 = base64.b64encode((tools_dir / "_bench_l0_probes.py").read_bytes()).decode("ascii")
    hccl_b64 = base64.b64encode((tools_dir / "_bench_l0_hccl.py").read_bytes()).decode("ascii")
    ac_b64 = base64.b64encode((tools_dir / "_bench_l0_ascendc.py").read_bytes()).decode("ascii")
    ack_b64 = base64.b64encode((tools_dir / "_bench_l0_ascendc_kernels.py").read_bytes()).decode("ascii")
    env_line = f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(vis)}\n" if vis else ""
    py_cmd = " ".join([
        shlex.quote(args.python_bin), '"$TMPD/_bench_l0_probes.py"',
        "--device-id", str(int(args.device_id)), "--peer-device-id", str(int(args.peer_device_id)),
        "--l2-bytes", str(int(spec["l2_bytes"])), "--aic-count", str(int(spec["aic_count"])),
        "--aiv-count", str(int(spec["aiv_count"])), "--arch", shlex.quote(spec["arch_preset"]),
        "--warmup", str(int(args.warmup)), "--repeats", str(int(args.repeats)),
        "--primitives", shlex.quote(str(getattr(args, "primitives", "all") or "all")),
    ])
    bash_script = (
        "set -euo pipefail\n"
        "TMPD=$(mktemp -d /tmp/mindie-l0-bench.XXXXXX)\n"
        "trap 'rm -rf \"$TMPD\"' EXIT\n"
        f"cat <<'EOF_B64' | base64 -d > \"$TMPD/_bench_l0_probes.py\"\n{probe_b64}\nEOF_B64\n"
        f"cat <<'EOF_HC' | base64 -d > \"$TMPD/_bench_l0_hccl.py\"\n{hccl_b64}\nEOF_HC\n"
        f"cat <<'EOF_AC' | base64 -d > \"$TMPD/_bench_l0_ascendc.py\"\n{ac_b64}\nEOF_AC\n"
        f"cat <<'EOF_ACK' | base64 -d > \"$TMPD/_bench_l0_ascendc_kernels.py\"\n{ack_b64}\nEOF_ACK\n"
        "for f in /usr/local/Ascend/cann-9.3.0/set_env.sh /usr/local/Ascend/cann-9.2.1/set_env.sh "
        "/usr/local/Ascend/cann-9.2.0/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do "
        "[ -f \"$f\" ] && source \"$f\" >/dev/null 2>&1 && break; done\n"
        f"{env_line}{py_cmd}\n"
    )
    remote_cmd = f"docker exec -i {shlex.quote(args.container)} bash -s" if args.container else "bash -s"
    completed = ssh_run(ep, remote_cmd, stdin_text=bash_script, timeout=REMOTE_BENCH_TIMEOUT_S)
    if completed.returncode != 0:
        raise RuntimeError(f"remote_calibration_failed:rc={completed.returncode}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            data = json.loads(line[len(RESULT_MARKER):])
            data["observed_silicon"] = obs
            return data
    raise RuntimeError("remote_calibration_missing_result")


def coverage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    prims = payload["primitives"]
    cal_m = sup_m = conf_m = 0
    for cell in prims.values():
        if cell["status"] == "unsupported":
            continue
        diags = cell.get("parameters", {}).get("metric_diagnostics", {})
        for d in diags.values():
            sup_m += 1
            st = d.get("status")
            if st == "calibrated":
                cal_m += 1
            elif st == "p99_exceeds_nominal_conflict":
                conf_m += 1
    p06_cov = prims.get("P06_mmad", {}).get("parameters", {}).get("mode_coverage", {})
    return {
        "calibrated_metrics": cal_m, "supported_metrics": sup_m, "conflict_metrics": conf_m,
        "calibrated_modes": list(p06_cov.get("calibrated_modes", [])),
        "unmeasured_modes": list(p06_cov.get("unmeasured_modes", [])),
        "unsupported_modes": list(p06_cov.get("unsupported_modes", [])),
        "conflict_modes": list(p06_cov.get("conflict_modes", [])),
    }


def update_manifest(manifest_path: Path, table_path: Path, payload: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "tables": {}}
    sku = payload["sku"]
    prims = payload["primitives"]
    ini_sha = str(prims.get("P01_datacopy", {}).get("parameters", {}).get("ini_sha256") or "")
    manifest.setdefault("tables", {})[sku] = {
        "file": table_path.name, "sku": sku, "compatible_skus": [sku],
        "arch_preset": payload["arch_preset"], "cann_version": payload["cann_version"],
        "ini_sha256": ini_sha,
        "calibrated": bool(payload["provenance"]["calibrated"]),
        "calibrated_primitive_count": sum(1 for c in prims.values() if c["status"] == "calibrated"),
        "unsupported_primitive_count": sum(1 for c in prims.values() if c["status"] == "unsupported"),
        "unmeasured_primitive_count": sum(1 for c in prims.values() if c["status"] == "unmeasured"),
        "total_primitives": len(prims),
        "coverage_summary": coverage_summary(payload),
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": payload["provenance"]["capture_uri"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=JSON_INDENT, sort_keys=True) + "\n", encoding="utf-8")
