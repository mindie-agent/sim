"""Declared raw cases and probes shared by collective / communication tests.

[Role] Single definition of the small declared-GPU raw case, the explicit
  ``exchange`` collective operator, parsed ``Topology`` builders, the cross-DP
  pairwise cohort case, the HCCL source-expansion case and the joint native
  service case, so collective and caller tests never import ``test_*`` modules.
[Invariants] Every builder returns a fresh mutable raw mapping or immutable
  schema record; no catalog scan, file I/O or device access at construction.
[Timeouts & Locks] Pure in-memory construction; no I/O, sleeps or locks.
[Side-Effect Policy] Side-effect-free; zero global mutation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

from mindie_sim import target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.operators import OpSpec

from _support.hardware import hardware_case


def case(*, dp=1, tp=1, pp=1, ep=1, prompt=4, output=3, requests=None):
    """Declared ``dp*tp*pp``-rank GPU with one shared 1000 B/ms fabric link."""
    ranks = dp * tp * pp
    return {
        "machine": {
            "name": "declared-gpu", "architecture": "declared", "num_devices": ranks,
            "device": {"memory_bytes": 1024 ** 3, "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                       "rates": {"memory": 1e9, "vector": 1e9}, "compute_units": 4},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": list(range(ranks)), "semantics": "shared"}],
        },
        "model": {
            "name": "geometry", "layers": 2, "hidden_size": 32, "num_heads": 4,
            "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
        },
        "workload": {"requests": requests or [
            {"id": "r", "prompt_tokens": prompt, "output_tokens": output}]},
        "runtime": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "max_num_batched_tokens": 16},
    }


def exchange(volume, *, family="alltoall", axis="tp", stage=0, metadata=()):
    """One explicit collective of ``volume`` per-rank link bytes on ``axis``."""
    return OpSpec("exchange", family, "prefill", stage, 0, (), (1,), (1,), (1,), (1,),
                  link_bytes=volume, metadata=(("group_axis", axis),) + metadata)


def topology(raw):
    parsed = parse_case(raw)
    return Topology(parsed.machine, parsed.runtime.parallel)


def per_pair_topology(ranks, *, capacity=1, latency_us=0, framing=None):
    """A complete direct graph so every ordered pair owns one directed link."""
    raw = case(tp=ranks)
    raw["model"].update(num_heads=2 * ranks, num_kv_heads=2)
    raw["machine"]["links"] = [
        {"name": f"p{source}{destination}", "members": [source, destination],
         "semantics": "per_pair", "capacity_bytes_per_ms": capacity,
         "latency_us": latency_us, **(framing or {})}
        for source, destination in combinations(range(ranks), 2)
    ]
    return topology(raw)


def ready_events(engine, delays):
    return {rank: engine.submit(f"ready{rank}", latency=delay)
            for rank, delay in enumerate(delays)}


def cohort_case(count):
    """``count`` DP replicas joined by an EP MoE pairwise per-pair 1 B/ms fabric."""
    raw = case(dp=count, ep=count)
    raw["model"].update(ffn="moe", num_experts=count * 2, top_k=2,
                        moe_intermediate_size=32)
    raw["machine"]["links"][0].update(
        semantics="per_pair", capacity_bytes_per_ms=1)
    raw["runtime"]["kernels"] = {"families": {
        "alltoall": {"kind": "pairwise_collective"},
        "expert_dispatch": {"kind": "pairwise_collective"},
        "expert_combine": {"kind": "pairwise_collective"},
    }}
    return raw


def wire(report):
    """Nonzero serviced link bytes of an execution report."""
    return {key: value for key, value in report["execution"]["serviced"].items()
            if key.startswith("link:") and value}


def declared(expansion="AIV", template="mesh_oneshot", *, ranks=2, blocks=64):
    """TP allreduce case whose HCCL source expansion and template are declared."""
    raw = case(tp=ranks)
    raw["machine"]["device"].update(
        rates={"memory": 1e9, "vector": 1e9, "mte2_aiv": 1e9, "mte2_aic": 1e9, "mte3": 1e9},
        buffers={"ub": 256 * 1024},
        resource_units={"vector": 64, "mte2_aiv": 64, "mte3": 64},
        buffer_units={"ub": 64})
    raw["machine"]["links"][0].update(capacity_bytes_per_ms=1e6, latency_us=5)
    raw["runtime"]["kernels"] = {"hccl": {"allreduce": {
        "expansion": expansion, "template": template,
        **({"block_limit": blocks} if expansion == "AIV" else {})}}}
    return raw


def isolated_calls(raw):
    """Decode body ops carrying ``hccl_source_group``, deps restricted to the kept set."""
    graphs = target_graphs(raw, phase="decode", tokens=1, context=16, scope="body")
    selected = {}
    for rank, ops in graphs.items():
        keep = [op for op in ops if "hccl_source_group" in dict(op.metadata)]
        ids = {op.op_id for op in keep}
        selected[rank] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids))
                               for op in keep)
    return selected


def copy_op(work=2000, **changes):
    """Contiguous bf16 copy of ``work`` HBM bytes with recipe ``copy``."""
    return replace(OpSpec(
        "copy", "copy", "prefill", 0, 0, ("r",),
        (work // 2,), (work // 2,), (work // 2,), (work // 2,),
        hbm_bytes=work, dtype="bf16", layout="contiguous", recipe="copy",
    ), **changes)


def joint_case():
    """One-device case whose copy recipe is served by a joint native queue."""
    rule = {"id": "observed-copy", "match": {"recipe": copy_op(4000).recipe},
            "coordinates": {"bytes": {"path": "work.hbm"}},
            "domain": {"bytes": {"min": 1000, "max": 4000}},
            "response": {"fixed_us": 2000},
            "provenance": {"kind": "synthetic_causal_control"},
            "execution": {"kind": "joint_native", "resource": "observed_copy",
                          "reference_rate": 1e9, "native_kernel": "DeclaredNativeCopy"}}
    raw = deepcopy(hardware_case())
    raw.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [rule]
    device = raw["machine"]["device"]
    device["rates"]["observed_copy"] = 1e9
    device["work_units"] = {"observed_copy": "busy_ns"}
    device["resource_units"] = {"observed_copy": 1}
    return raw
