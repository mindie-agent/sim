#!/usr/bin/env python3
"""Regenerate machine-readable catalog reference packages.

Writes structure JSON and reproducible full-model cases from the checked-in
configs.  It never stores latency as a replay key: predictions must be rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

from mindie_sim.a3_catalog_cache import request_cache_breakdown
from mindie_sim.a3_catalog_weights import stage_weight_bytes, weight_breakdown
from mindie_sim.a3_types import MachineSpec, ParallelSpec
from mindie_sim.model_catalog import load_model

ROOT = Path(__file__).resolve().parent
TOPOLOGIES = {
    "deepseek-v4-flash-0731": dict(tp=8, dp=1, ep=8, pp=1, devices=8),
    "deepseek-v4-pro-0813": dict(tp=16, dp=1, ep=16, pp=1, devices=16),
    "deepseek-v4-10t": dict(tp=16, dp=1, ep=16, pp=1, devices=16),
    "deepseek-v4.1-flash": dict(tp=16, dp=1, ep=16, pp=4, devices=64),
    "glm-5.3": dict(tp=16, dp=1, ep=16, pp=2, devices=32),
    "kimi-k3": dict(tp=16, dp=1, ep=16, pp=3, devices=48),
    "kimi-k3-dspark": dict(tp=1, dp=1, ep=1, pp=1, devices=8),
    "qwen3.8-2.4t-a95b": dict(tp=32, dp=1, ep=32, pp=4, devices=128),
    "qwen3.8-27b": dict(tp=8, dp=1, ep=1, pp=1, devices=8),
}
PROVENANCE = {
    "deepseek-v4-flash-0731": {"hf": "deepseek-ai/DeepSeek-V4-Flash-0731", "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062"},
    "deepseek-v4-pro-0813": {"hf": "deepseek-ai/DeepSeek-V4-Pro-0813", "revision": "72e1d3230f6c080a530b0a1d46f8eb4602340597"},
    "deepseek-v4-10t": {"hf": "user-provided attachment", "revision": ""},
    "deepseek-v4.1-flash": {"hf": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "dba1be0a40aa45a94ad051997016db3960a90277"},
    "glm-5.3": {"hf": "zai-org/GLM-5.3", "revision": "187fb9fff6319062325ff825627ef6db084d9bc6"},
    "kimi-k3": {"hf": "moonshotai/Kimi-K3", "revision": "f831ab66814297da540d832a5235f8e904f29d06"},
    "kimi-k3-dspark": {"hf": "RadixArk/Kimi-K3-DSpark", "revision": "3c5bac301d9cf392706189d82ed947feca6c2f0f"},
    "qwen3.8-2.4t-a95b": {"hf": "Qwen/Qwen3.8-2.4T-A95B", "revision": "207bd685a7e3696cfaff12ded7c6a7ea0f88c996"},
    "qwen3.8-27b": {"hf": "Qwen/Qwen3.8-27B", "revision": ""},
}
A3_DEVICE = {
    "hbm_gib": 64, "cube_flops_per_s": 2.70336e14,
    "vector_ops_per_s": 1.6896e13, "hbm_bw_bytes_per_s": 1.2e12,
    "mte_bw_bytes_per_s": 2.4e12, "launch_us": 20.0,
    # Declared Ascend910_9391 A3 physical-core/capacity facts used by the
    # accepted public matrix. These enable finite fusion buffer capacity
    # evaluation; they do not alter compute/HBM peaks.
    "l2_bytes": 192 * 1024 * 1024,
    "l1_bytes": 512 * 1024,
    "l0a_bytes": 64 * 1024,
    "l0b_bytes": 64 * 1024,
    "l0c_bytes": 128 * 1024,
    "ub_bytes": 192 * 1024,
    "aic_count": 24, "aiv_count": 48,
}


def machine(topology):
    return {
        "name": f"declared-a3-{topology['devices']}-die",
        "arch_family": "A3", "dies_per_package": 2,
        "sku": "Ascend910_9391",
        "num_devices": topology["devices"], "ranks_per_node": 8,
        "device": dict(A3_DEVICE),
        "compute_peaks": {"bf16": 2.70336e14, "fp8_e4m3": 2.70336e14},
        "links": [{
            "name": "d2d", "channel": "d2d_intra_node",
            "capacity_gbps": 196, "latency_us": 2.5,
            "members": list(range(topology["devices"])),
        }],
    }


def case(key, topology):
    return {
        "name": f"catalog-full-model-{key}",
        "machine": machine(topology),
        "model": {"catalog_key": key},
        "workload": {"requests": [{"prompt_tokens": 8, "output_tokens": 1}]},
        "runtime": {
            "max_num_batched_tokens": 16, "max_num_seqs": 2,
            "max_model_len": 256, "enable_chunked_prefill": True,
            "graph_mode": "eager", "tp": topology["tp"], "dp": topology["dp"],
            "ep": topology["ep"], "pp": topology["pp"],
        },
    }


def main() -> None:
    (ROOT / "structure").mkdir(parents=True, exist_ok=True)
    (ROOT / "cases").mkdir(parents=True, exist_ok=True)
    index = []
    for key, topo in TOPOLOGIES.items():
        model = load_model(key)
        parallel = ParallelSpec(tp=topo["tp"], dp=topo["dp"], ep=topo["ep"], pp=topo["pp"])
        stages = stage_weight_bytes(model, parallel, 8)
        payload = {
            "model_key": key,
            "provenance": {
                "config": f"mindie_sim/model_configs/{key}.json",
                "se_tools_snapshot": "ae8779fdd5a602bd3a4358bca93b2161d134a47b",
                **PROVENANCE[key],
                "scope": "attribution only; not a runtime admission gate",
            },
            "model": model.to_dict(),
            "reference_placement": {k: topo[k] for k in ("tp", "dp", "ep", "pp")},
            "per_rank_stage_weight_bytes": stages,
            "weight_category_bytes": weight_breakdown(model, parallel, 8),
            "cache_at_max_model_len": request_cache_breakdown(model, parallel, 256),
            "declared_device": dict(A3_DEVICE),
            "scope": "declared semantic geometry; not a measured timing row",
        }
        (ROOT / "structure" / f"{key}.json").write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8",
        )
        case_payload = case(key, topo)
        (ROOT / "cases" / f"{key}.json").write_text(
            json.dumps(case_payload, indent=1) + "\n", encoding="utf-8",
        )
        search_spec = {
            "objective": "goodput", "max_candidates": 2,
            "tp": [topo["tp"]], "dp": [topo["dp"]], "ep": [topo["ep"]],
            "pp": [topo["pp"]], "max_num_batched_tokens": [16],
            "fusion": [False, True], "graph_buckets": [],
        }
        (ROOT / "cases" / f"{key}.search.json").write_text(
            json.dumps(search_spec, indent=1) + "\n", encoding="utf-8",
        )
        should_constraints = {
            "tp": [topo["tp"]], "dp": [topo["dp"]], "ep": [topo["ep"]],
            "pp": [topo["pp"]], "fusion": [False, True],
            "graph_padding": False, "search_budget": 3,
        }
        (ROOT / "cases" / f"{key}.should.json").write_text(
            json.dumps(should_constraints, indent=1) + "\n", encoding="utf-8",
        )
        index.append({
            "model_key": key, "body_layers": model.layers,
            "hidden_size": model.hidden_size,
            "parameter_count": model.parameter_count(),
            "logical_body_weight_bytes": model.resident_weight_bytes_override,
            "reference_placement": payload["reference_placement"],
            "case": f"cases/{key}.json", "search": f"cases/{key}.search.json",
            "should_constraints": f"cases/{key}.should.json",
            "structure": f"structure/{key}.json",
        })
        print(f"wrote {key}: {len(stages)} stages, {stages[max(range(len(stages)), key=lambda i: stages[i])]/1e9:.1f}GB max")
    media_case = case("qwen3.8-27b", TOPOLOGIES["qwen3.8-27b"])
    media_case["name"] = "catalog-qwen3.8-27b-media-smoke"
    media_case["workload"] = {"requests": [{
        "request_id": "r0", "prompt_tokens": 1, "media_tokens": 4,
        "output_tokens": 2,
    }]}
    media_case["runtime"]["metadata"] = {
        "media_positions_pre_expanded": "true",
        "media_tokens_by_request": '{"r0": 4}',
    }
    (ROOT / "cases" / "qwen3.8-27b-media.json").write_text(
        json.dumps(media_case, indent=1) + "\n", encoding="utf-8",
    )
    (ROOT / "index.json").write_text(
        json.dumps({"models": index,
                    "extra_cases": ["cases/qwen3.8-27b-media.json"]}, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
