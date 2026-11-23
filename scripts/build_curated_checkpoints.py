#!/usr/bin/env python3
"""Deterministic single-publisher generator for ``curated_checkpoints.json``.

[Role] Stream raw ModelSlim ``quant_model_description*.json`` descriptors and
  merge HF ``config.json["quantization_config"]`` records from scan metadata
  into the packaged ``mindie_sim/catalog/data/curated_checkpoints.json`` asset.
[Invariants] Uses streaming ``checkpoint_reader`` parsing so ``> 50 MiB``
  descriptors (e.g. 92.4 MB Kimi-K3.1) never require whole-file ``json.loads``;
  verifies round-trip digest equality for every emitted checkpoint; rejects
  host home paths, private intranet IPs, and oversized output payloads.
[Timeouts & Locks] Bounded local streaming disk reads and atomic ``os.replace``
  output write; zero network calls or locks.
[Side-Effect Policy] Writes only the requested ``--output`` JSON path (or
  verifies without writing under ``--check``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim.catalog.checkpoint_reader import (
    CURATED_CHECKPOINTS_SCHEMA,
    deserialize_curated_entry,
    read_hf_quantization_config,
    serialize_curated_entry,
    stream_modelslim_description_file,
)
from mindie_sim.schema.checkpoint import CHECKPOINT_QUANT_MAP_SCHEMA

DEFAULT_OUTPUT_REL = Path("mindie_sim/catalog/data/curated_checkpoints.json")
MAX_CURATED_JSON_BYTES = 250 * 1024
PRIVATE_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
)
HOST_PATH_RE = re.compile(r"(?:/Users/|/home/|/root/|[A-Za-z]:\\\\Users\\\\)")


def _repo_id_from_filename(path: Path) -> str:
    parts = path.name.split("__")
    if len(parts) < 3:
        raise ValueError(f"unexpected_scanned_filename:{path.name}")
    return f"{parts[0]}/{parts[1]}"


def _load_meta_index(meta_json: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if meta_json is None or not meta_json.is_file():
        return {}, []
    raw = json.loads(meta_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("meta_json_requires_list")
    by_repo: dict[str, dict[str, Any]] = {}
    hf_rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("repo_id") or "").strip()
        if not rid:
            continue
        by_repo[rid] = row
        if row.get("source_file") == "config.json[quantization_config]":
            hf_rows.append(row)
    return by_repo, hf_rows


def build_curated_payload(
    scan_dir: Path,
    meta_json: Path | None = None,
    *,
    force_stdlib: bool = False,
) -> dict[str, Any]:
    """Stream all raw descriptors and produce the lossless curated payload."""
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"scan_dir_not_found:{scan_dir}")
    meta_by_repo, hf_rows = _load_meta_index(meta_json)
    checkpoints: dict[str, dict[str, Any]] = {}

    for json_path in sorted(scan_dir.glob("*.json")):
        repo_id = _repo_id_from_filename(json_path)
        meta = meta_by_repo.get(repo_id) or {}
        dtype_hint = meta.get("config_torch_dtype")
        base_cfg = {"torch_dtype": dtype_hint} if isinstance(dtype_hint, str) and dtype_hint.strip() else None
        qmap, non_weights = stream_modelslim_description_file(
            json_path,
            checkpoint_id=repo_id,
            base_config=base_cfg,
            force_stdlib=force_stdlib,
        )
        # Preserve canonical descriptor filename without owner/repo prefix.
        descriptor_name = json_path.name.split("__", 2)[2]
        serialized = serialize_curated_entry(qmap, non_weight_patterns=non_weights)
        serialized["descriptor_path"] = descriptor_name

        # Round-trip verification against lossless L0 CheckpointQuantMap.
        roundtrip_qmap, _ = deserialize_curated_entry(repo_id, serialized)
        expected_qmap = qmap
        if expected_qmap.provenance.descriptor_path != descriptor_name:
            from dataclasses import replace
            expected_qmap = replace(
                expected_qmap,
                provenance=replace(expected_qmap.provenance, descriptor_path=descriptor_name),
            )
        if roundtrip_qmap.digest() != expected_qmap.digest():
            raise RuntimeError(f"roundtrip_digest_mismatch:{repo_id}")
        checkpoints[repo_id] = serialized

    for row in sorted(hf_rows, key=lambda item: str(item.get("repo_id") or "")):
        repo_id = str(row["repo_id"]).strip()
        if repo_id in checkpoints:
            continue
        qcfg = row.get("quantization_config")
        if not isinstance(qcfg, dict) or not qcfg:
            continue
        dtype_hint = row.get("config_torch_dtype")
        base_cfg = {"torch_dtype": dtype_hint} if isinstance(dtype_hint, str) and dtype_hint.strip() else None
        qmap = read_hf_quantization_config(
            qcfg,
            checkpoint_id=repo_id,
            base_config=base_cfg,
            descriptor_path="config.json",
        )
        serialized = serialize_curated_entry(qmap)
        roundtrip_qmap, _ = deserialize_curated_entry(repo_id, serialized)
        if roundtrip_qmap.digest() != qmap.digest():
            raise RuntimeError(f"roundtrip_digest_mismatch:{repo_id}")
        checkpoints[repo_id] = serialized

    return {
        "schema_version": CURATED_CHECKPOINTS_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_QUANT_MAP_SCHEMA,
        "count": len(checkpoints),
        "checkpoints": dict(sorted(checkpoints.items())),
    }


def _render_and_validate(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    raw_bytes = rendered.encode("utf-8")
    if len(raw_bytes) > MAX_CURATED_JSON_BYTES:
        raise RuntimeError(
            f"curated_checkpoints_exceeds_budget:{len(raw_bytes)}>{MAX_CURATED_JSON_BYTES}"
        )
    if HOST_PATH_RE.search(rendered):
        raise RuntimeError("curated_checkpoints_leaks_host_home_path")
    if PRIVATE_IP_RE.search(rendered):
        raise RuntimeError("curated_checkpoints_leaks_private_ip")
    return rendered


def _write_atomic(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        os.replace(tmp_path, target)
    except TimeoutError:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    except OSError:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True, help="Directory of raw scanned *.json files")
    parser.add_argument("--meta-json", type=Path, default=None, help="Path to all_checkpoints_quant_scan.json")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT_REL,
        help="Target path for curated_checkpoints.json",
    )
    parser.add_argument("--force-stdlib", action="store_true", help="Force stdlib chunked JSON streamer")
    parser.add_argument("--check", action="store_true", help="Verify output matches without writing")
    args = parser.parse_args(argv)

    payload = build_curated_payload(
        args.scan_dir, args.meta_json, force_stdlib=args.force_stdlib,
    )
    rendered = _render_and_validate(payload)
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if existing != rendered:
            print(f"DRIFT: {args.output} differs from generated curated payload", file=sys.stderr)
            return 1
        print(f"OK: {args.output} matches ({payload['count']} checkpoints, {len(rendered.encode()) / 1024:.1f} KiB)")
        return 0

    _write_atomic(args.output, rendered)
    print(f"WROTE {payload['count']} checkpoints -> {args.output} ({len(rendered.encode()) / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
