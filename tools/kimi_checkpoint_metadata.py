"""Read Kimi checkpoint declarations and bounded headers without weight payloads.

[Role] Produce independent per-module quantization evidence from a checkpoint.
[Invariants] Stream the quant/index files; read only selected tensor headers.
  Never load weight payloads, import a tensor runtime, or initialize a device.
[Timeouts & Locks] The caller bounds remote execution; no locks or subprocesses.
[Side-Effect Policy] Read the checkpoint; write only the explicitly requested report.
"""
import argparse
import collections
import hashlib
import json
import re
import resource
import socket
import struct
import time
from pathlib import Path


def collect(checkpoint):
    started = time.monotonic()
    root = Path(checkpoint)
    decoder = json.JSONDecoder()
    groups, selected, globals_ = {}, {}, {}
    all_schemes = collections.Counter()
    quant_sha = hashlib.sha256()
    line_count = 0
    with (root / "quant_model_description.json").open("rb") as handle:
        for line_count, raw in enumerate(handle, 1):
            quant_sha.update(raw)
            line = raw.decode().strip().rstrip(",")
            if not line.startswith('"'):
                continue
            key, pos = decoder.raw_decode(line)
            value_text = line[pos:].lstrip()[1:].strip()
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                if key.startswith("language_model."):
                    raise
                continue
            if not key.startswith(("language_model.", "vision_tower.", "mm_projector.")):
                globals_[key] = value
                continue
            all_schemes[str(value)] += 1
            if not key.startswith("language_model."):
                continue
            normalized = re.sub(r"\.layers\.\d+\.", ".layers.L.", key)
            normalized = re.sub(r"\.experts\.\d+\.", ".experts.E.", normalized)
            row = groups.setdefault(normalized, {
                "schemes": collections.Counter(), "samples": [],
            })
            row["schemes"][str(value)] += 1
            if len(row["samples"]) < 2:
                row["samples"].append({"key": key, "scheme": value})
            layer = re.search(r"\.layers\.(\d+)\.", key)
            if (layer and int(layer.group(1)) in (0, 1, 3)
                    and (".experts." not in key or ".experts.0." in key)
                    or key in ("language_model.lm_head.weight",
                               "language_model.model.embed_tokens.weight")):
                selected[key] = value

    config_path = root / "config.json"
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("config.json exceeds the 1MiB metadata budget")
    config_raw = config_path.read_bytes()
    config_info = {
        "path": str(config_path), "sha256": hashlib.sha256(config_raw).hexdigest(),
        "bytes": len(config_raw), "mtime_ns": config_path.stat().st_mtime_ns,
        "content": json.loads(config_raw),
    }
    index_path = root / "quant_model_weights.safetensors.index.json"
    index_sha = hashlib.sha256()
    selected_shards = {}
    with index_path.open("rb") as handle:
        for raw in handle:
            index_sha.update(raw)
            line = raw.decode().strip().rstrip(",")
            if not line.startswith('"'):
                continue
            key, pos = decoder.raw_decode(line)
            if key not in selected:
                continue
            selected_shards[key] = json.loads(line[pos:].lstrip()[1:].strip())

    by_shard = collections.defaultdict(list)
    for key, filename in selected_shards.items():
        by_shard[filename].append(key)
    headers = []
    for filename, keys in sorted(by_shard.items()):
        path = root / filename
        with path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if header_bytes > 8 * 1024 * 1024:
                raise ValueError(f"Header exceeds 8MiB budget: {filename}:{header_bytes}")
            raw = handle.read(header_bytes)
        if len(raw) != header_bytes:
            raise ValueError(f"Incomplete header: {filename}")
        header = json.loads(raw)
        headers.append({
            "path": str(path), "file_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "header_bytes": header_bytes,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "tensors": {key: header[key] for key in keys},
        })

    return {
        "schema": "kimi-checkpoint-module-quant-audit/v1",
        "host": socket.gethostname(), "checkpoint": str(root),
        "observed_at_unix_s": time.time(), "config": config_info,
        "quant_description": {
            "path": str(root / "quant_model_description.json"),
            "bytes": (root / "quant_model_description.json").stat().st_size,
            "mtime_ns": (root / "quant_model_description.json").stat().st_mtime_ns,
            "sha256": quant_sha.hexdigest(), "lines": line_count,
            "globals": globals_, "all_scheme_counts": all_schemes,
            "normalized_language_module_groups": groups,
            "selected_entries": selected,
        },
        "weight_index": {
            "path": str(index_path), "bytes": index_path.stat().st_size,
            "mtime_ns": index_path.stat().st_mtime_ns,
            "sha256": index_sha.hexdigest(),
            "selected_tensor_shards": selected_shards,
            "selected_entries_missing_in_index": sorted(set(selected) - set(selected_shards)),
        },
        "selected_safetensors_headers": headers,
        "limits": {
            "weight_payload_bytes_read": 0,
            "config_and_quant_are_current_checkpoint_metadata": True,
            "config_hash_matches_retained_campaign": config_info["sha256"] ==
                "07677f14bc468c16a788fc6517f7fa715701fa1c3ff74b3f7ef3449a5bc8c735",
            "historical_quant_hash_available": False,
            "float_scheme_requires_header_dtype_interpretation": True,
        },
        "usage": {
            "wall_seconds": time.monotonic() - started,
            "max_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_rss_unit": "operating-system ru_maxrss units",
            "header_count": len(headers),
            "header_bytes_read": sum(row["header_bytes"] + 8 for row in headers),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="Existing Kimi checkpoint directory")
    parser.add_argument("-o", "--output", type=Path, help="Report path; default is stdout")
    args = parser.parse_args(argv)
    text = json.dumps(collect(args.checkpoint), sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
