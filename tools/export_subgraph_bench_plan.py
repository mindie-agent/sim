"""Export a CPU-only, full-catalog subgraph benchmark plan across hardware SKUs.

[Role]
  Compile catalog models through ``compose_case`` and ``export_structure``,
  deduplicate operator/subgraph replay cases, retain reverse indices, preserve
  all collective cosets, separate human display labels from certified upstream
  ABIs, and align expert banks with ``expert_slot_plan``.
[Invariants]
  * ``CORE_CATALOG_ENTRIES`` is dynamic; every body/MTP/DSpark layer exports a
    node signature; MoE replay uses integer routing aligned with
    ``mindie_sim.catalog.expert_slots.expert_slot_plan`` (supporting padded
    slots when ``E % EP != 0``).
  * Train/heldout freezes from geometry before timing; every non-control op
    maps 1-to-1 with zero fallback; ``display_name`` is never conflated with
    certified ``upstream_abi``.
[Timeouts & Locks]
  Bounded CPU work; one model per worker process; no locks.
[Side-Effect Policy]
  Reads local catalog JSON and writes only ``--out`` plan files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mindie_sim.catalog import fleet
from mindie_sim.compose import compose_case, resolve_hardware_sku, va_recommended_config
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.routing_realizations import select_realization
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.structure import export_structure
from tools.bench_plan_builders import (
    BUILDERS, COLLECTIVE_BUILDERS, CONTROL_FAMILIES, UnmodeledCapabilityError,
    attention_shape_dims, collapse_device_kernel_groups, fabric_route_ranks,
    is_ladder_kv_node, resolve_builder_strict, resolve_collective_cosets,
    resolve_collective_group_ranks, resolve_fabric_collective_abi,
)
from tools.bench_plan_parallel import (
    ARCHETYPES, DEFAULT_TP, DENSE_EP, GEOMETRY_ARMS, HELDOUT_BATCH,
    HELDOUT_FABRIC_BYTES, HELDOUT_GEOMETRY_PROBES, HELDOUT_KV_LENGTH,
    HELDOUT_MATRIX_ROWS, HELDOUT_ROUTING_DISTRIBUTION, HELDOUT_ROUTING_ZERO_FRACTION,
    MAX_EXPORT_MODEL_LEN, MIB, PlanError, ROUTING_DISTRIBUTIONS,
    SEQUENCE_LADDER_TOKENS, ZERO_EXPERT_FRACTIONS, add_case as _add_case,
    add_routing_arms as _add_routing_arms, assign_split, canonical_json as _canonical,
    case_identities as _identities, collective_group_size,
    compact_ledger as _compact_ledger, compact_model as _compact_model,
    compact_signature as _compact_signature, expert_bank_bytes as _expert_bank_bytes,
    explicit_ledger as _explicit_ledger, layer_coverage as _layer_coverage,
    layer_signatures, merge_case as _merge_case, merge_occurrences as _merge_occurrences,
    model_archetype, multi_axis_parallel_arms, occurrences_for_layers as _occurrences,
    parallel_policy, parameter_count as _parameter_count,
    positive_experts as _positive_experts, probe_geometries_for,
    register_routing_case as _register_routing_case, representative_layer_depth,
    reverse_index_cases as _reverse_index, routing_block as _routing_block,
    routing_counts as _routing_counts, service_identity as _service_identity,
    short_digest as _digest, slot_plan_for as _slot_plan_for,
    split_fields as _split_fields, topk_ids_for_counts as _topk_ids,
    validate_routing_payload as _validate_routing,
)

SCHEMA, NODE_SCHEMA, PLAN_SUFFIX, MANIFEST_NAME = (
    "mindie-bench-subgraph-plan/1", "mindie-bench-subgraph-node/1", ".plan.json", "manifest.json",
)
DEFAULT_SKUS = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382")
DEFAULT_GEOMETRIES, DEFAULT_TRACKS = ("prefill", "decode"), ("merged", "unmerged_pr")
FABRIC_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "all_to_all_v")
GIB = 1024 ** 3
S0_REPLAY_CEILING_BYTES = 8 * GIB
FABRIC_COLD_RING_FACTOR, FABRIC_SLOT_BYTES = 3, 8 * MIB
FABRIC_ROUTES = ("intra_package_die", "intra_node_hccs", "inter_node_roce")
FABRIC_LADDER_BYTES = tuple(sorted({*(1 << n for n in (12, 14, 18, 22, 26)), *HELDOUT_FABRIC_BYTES}))
FABRIC_PARTICIPANTS = {"intra_package_die": 2, "intra_node_hccs": 4, "inter_node_roce": 8}
ELEMENT_BYTES, MISSING_LAYER = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "int4": 1}, -1
GEMM_FAMILIES = frozenset({
    "gemm", "down_projection", "gate_up_projection", "output_projection", "qkv_projection",
    "projection", "logits", "router", "index_projection", "gate_projection", "vision_merge",
    "attention_qk", "attention_pv",
})
KV_FAMILIES = frozenset({
    "attention", "compressed_attention", "indexer", "indexer_quant_cache",
    "kv_compress_epilog", "cache_store", "key_pool", "attn_res_prefix",
})
KV_ONLY_LADDER_PROBES = frozenset(HELDOUT_GEOMETRY_PROBES) - {"decode_b16", "decode_kv16384", "prefill_s2048"}
METADATA_KEYS = frozenset("""kernel kernel_name native_kind native_implementation caller_implementation
implementation implementation_binding program_kind semantic_family group_axis matrix_group_rows matrix_groups matrix_epilogue
matrix_mapping_domain routing_realization routing_basis moe_init_routing top_k num_experts ep_degree ep_rank
graph_replay_scope graph_captured graph_segment weight_format weight_dtype activation_format input_layout
output_format cache_format cache_kind cache_layout cache_tensor_view stream collective collective_id peer_traffic
payload_rows row_bytes state_version dequant packed_layout quant_category ffn_role ffn_category
device_kernel_group fusion_boundary fusion_group_key fused_from fused_members output_shape input_rows
issued_rows matrix_weight_bank_groups topk_ids_dtype topk_ids_producer topk_weights_producer
payload_row_bytes payload_row_elements payload_semantics weight_representation weight_payload_bytes
weight_scale_bytes activation_input_bytes cache_resolution cache_visible heads head_dim key_heads group_size
participants nbytes per_token_scale collective_op mc2_mode qli_abi onchip_handoff_to sp_collective_kind
quantized_all_reduce quantized_all_reduce_applied recover_ssm pipeline_gather pipeline_scatter physical_rank""".split())


def _catalog_entries() -> dict[str, str]:
    return dict(fleet.CORE_CATALOG_ENTRIES)


def _first_dim(shape):
    return int(shape[0]) if isinstance(shape, (list, tuple)) and shape and type(shape[0]) is int else None


def _case_key(payload):
    return ("case", payload["kind"], payload["phase"], payload["physical_signature"], _canonical(payload["split_axes"]))


def _kv_dependent(family, recipe):
    del recipe
    return str(family or "") in KV_FAMILIES


def _node_kind(family, builder):
    return ("boundary" if str(family or "") in CONTROL_FAMILIES
            else ("collective" if builder in COLLECTIVE_BUILDERS else "operator"))


def _metadata(operator):
    return {key: deepcopy(value) for key, value in (operator.get("metadata") or {}).items()
            if key in METADATA_KEYS or key.startswith(("routing_", "moe_", "group_", "matrix_"))}


def _collective_nbytes(operator, metadata, dtype):
    for key in ("nbytes", "payload_bytes", "peer_bytes"):
        value = metadata.get(key)
        if type(value) is int and value > 0:
            return value
    elements = 1
    for value in operator.get("launch_shape") or operator.get("local_shape") or ():
        if type(value) is not int:
            return None
        elements *= value
    return elements * ELEMENT_BYTES.get(str(dtype), 4)


def _shape_dims(family, builder, recipe, operator, metadata, parallel, *, raw_meta=None, batch=1, kv_length=None, phase="decode"):
    del recipe
    local = [int(value) for value in (operator.get("local_shape") or ()) if type(value) is int]
    dims = local or [int(value) for value in (operator.get("launch_shape") or ()) if type(value) is int]
    shape: dict[str, Any] = {}
    groups = metadata.get("matrix_group_rows") or ()
    if builder in {"matmul", "quant_matmul", "all_gather_matmul", "matmul_reduce_scatter"} or (family in GEMM_FAMILIES and len(dims) >= 2):
        if len(dims) >= 3:
            shape.update(m=dims[0], n=dims[1], k=dims[2])
        elif dims:
            shape.update(m=dims[0], k=dims[-1])
    elif builder in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
        shape.update(m=dims[0] if dims else 0, n=dims[1] if len(dims) > 1 else 0, k=dims[2] if len(dims) > 2 else 0,
                     e=int(metadata.get("num_experts") or len(groups) or metadata.get("matrix_groups") or 0))
    elif builder in {"fusion_attention", "fused_infer_attention", "incre_flash_attention", "absorbed_mla",
                     "sparse_flash_attention", "mixed_quant_sparse_mla", "block_sparse_attention_c8"}:
        shape.update(attention_shape_dims(raw_meta if raw_meta is not None else metadata, dims,
                                          sequences=batch, kv_length=kv_length, phase=phase))
    elif builder in {"topk", "moe_gating_topk", "moe_gating_topk_hash", "moe_init_routing"}:
        shape.update(tokens=dims[0] if dims else 0, e=int(metadata.get("num_experts") or 0), top_k=int(metadata.get("top_k") or 0))
    elif builder in COLLECTIVE_BUILDERS:
        shape.update(bytes=_collective_nbytes(operator, metadata, operator.get("dtype")),
                     group_size=collective_group_size(metadata, parallel))
    elif builder in {"ple_lookup", "strided_gather", "index_add"}:
        shape.update(tokens=dims[0] if dims else 0, width=dims[-1] if dims else 0)
    elif dims:
        shape.update(tokens=dims[0], width=dims[-1])
    return {key: int(value) for key, value in shape.items() if value is not None}


def _declared_implementation(metadata: Mapping[str, Any]) -> str:
    """Return an explicitly declared implementation/ABI from metadata, or empty."""
    for key in ("native_kind", "native_implementation", "kernel", "kernel_name", "implementation", "caller_implementation"):
        value = metadata.get(key)
        if isinstance(value, str) and value and value not in {"static", "declared"}:
            return value
    binding = metadata.get("implementation_binding")
    if isinstance(binding, Mapping):
        for key in ("implementation", "selector", "program_recipe"):
            if binding.get(key):
                return str(binding[key])
    return ""


def _display_name(family: Any, recipe: Any, fallback: str = "") -> str:
    """Human-readable display label (never certified as an upstream ABI)."""
    return f"{family or 'operator'}:{recipe or fallback}"


def _node_fields(*, node_id, family, recipe, operator, metadata, deps, outputs, parallel,
                 model_id="", layer_idx=MISSING_LAYER, topology=None, stage_idx=0,
                 replica_idx=0, batch=1, kv_length=None, phase="decode"):
    raw_meta = dict(operator.get("metadata") or {})
    dtype = str(operator.get("dtype") or "")
    raw_layout = str(operator.get("layout") or "").strip()
    layout = "contiguous" if raw_layout in {"", "declared"} else raw_layout
    declared_impl = _declared_implementation(raw_meta)
    display_name = _display_name(family, recipe, str(recipe or family or "operator"))
    dtypes = [d for d in (dtype, str(raw_meta.get("output_dtype") or ""), str(raw_meta.get("activation_format") or "")) if d]
    builder, upstream_abi = resolve_builder_strict(
        str(family or ""), str(recipe or ""), declared_impl, dtypes, raw_meta,
        model_id=model_id, layer_idx=int(layer_idx), local_shape=tuple(operator.get("local_shape") or ()),
    )
    kind, routing = _node_kind(family, builder), _routing_block(metadata)
    group = routing["group_list"] if routing else metadata.get("matrix_group_rows")
    collective = builder in COLLECTIVE_BUILDERS
    topo_meta = {**metadata, **{k: raw_meta[k] for k in ("group_ranks", "physical_rank", "rank") if k in raw_meta}}
    cosets = resolve_collective_cosets(topology, topo_meta, stage_idx) if (collective and topology is not None) else None
    if cosets is not None:
        has_explicit = any(k in topo_meta for k in ("group_ranks", "physical_rank", "rank"))
        rep_ranks = () if has_explicit else tuple(topology.ranks(int(replica_idx), stage_idx))
        granks = resolve_collective_group_ranks(
            topology, topo_meta, stage_idx,
            rank=(None if has_explicit else (int(rep_ranks[0]) if rep_ranks else 0)),
        )
    else:
        granks = None
    gsize = len(granks) if granks else (collective_group_size(metadata, parallel) if collective else None)
    shape = _shape_dims(family, builder, recipe, operator, metadata, parallel,
                        raw_meta=raw_meta, batch=batch, kv_length=kv_length, phase=phase)
    if collective and gsize is not None:
        shape["group_size"] = int(gsize)
    return {
        "schema": NODE_SCHEMA, "node_id": str(node_id), "kind": kind,
        "op": str(recipe or family or "unknown"), "display_name": display_name,
        "builder": builder, "upstream_abi": upstream_abi, "phase": str(phase),
        "dtype": dtype or "fp32", "layout": layout,
        "output_dtype": str(metadata.get("output_dtype") or dtype or "fp32"),
        "group_list_type": 1, "shape": shape, "inputs": [str(value) for value in deps],
        "outputs": [str(value) for value in outputs],
        "implementation": declared_impl or upstream_abi,
        "implementation_source": "metadata" if declared_impl else "builder_upstream_abi",
        "kernel": str(metadata.get("kernel") or metadata.get("native_kind") or upstream_abi),
        "group_list": [int(value) for value in group] if group else None,
        "per_token_scale": bool(metadata["per_token_scale"]) if "per_token_scale" in metadata else None,
        "format_code": 29 if layout == "FRACTAL_NZ" else None,
        "nbytes": _collective_nbytes(operator, metadata, dtype) if collective else None,
        "group_size": gsize,
        "group": str(metadata.get("group_axis") or metadata.get("collective") or "tp") if collective else None,
        **({"group_ranks": granks, "group_cosets": cosets} if granks else {}),
    }


def _stage_and_boundary(nodes):
    include = [node["node_id"] for node in nodes if node["kind"] != "collective"]
    return ("s0_local", {"include": include}) if include else ("s1_joint", {"include": [node["node_id"] for node in nodes]})


def _case_base(*, model_id, archetype, parallel, arm, phase, kind, family, signature, work, matrix_rows, batch, kv_length, routing, scope):
    exec_mode = "aclgraph" if str(phase) in {"decode", "draft"} else "eager"
    return {
        "kind": kind, "model_id": model_id, "archetype_id": archetype["archetype_id"],
        "archetype_name": archetype["name"], "arm": arm, "family": family, "phase": phase,
        "scope": scope, "parallel": dict(parallel), "physical_signature": signature,
        "execution_mode": exec_mode, "cudagraph_mode": "FULL_DECODE_ONLY" if exec_mode == "aclgraph" else "NONE",
        "matrix_rows": matrix_rows, "physical_work": deepcopy(work or {}), "batch": batch,
        "kv_length": kv_length, "routing": routing,
        "physical_rank_role": "virtual_rank_0_local_slice" if parallel.get("virtual_rank_zero") else "physical_rank_0",
    }


def _node_payload(node, *, model_id, archetype, parallel, arm, batch, kv_length, phase, pr_overlay=False, topology=None):
    operator = node.get("operator") or {}
    metadata = _metadata(operator)
    routing, layer_idx = _routing_block(metadata), int(node.get("layer", MISSING_LAYER))
    node_phase, matrix_rows = str(node.get("phase", phase)), _first_dim(operator.get("local_shape"))
    kv = kv_length if _kv_dependent(node.get("family"), operator.get("recipe")) else None
    split = _split_fields(matrix_rows=matrix_rows, batch=batch, kv_length=kv,
                          routing_distribution="balanced" if routing else None,
                          zero_expert_fraction=0.0 if routing else None, pr_overlay=pr_overlay)
    built = _node_fields(
        node_id=node["node_id"], family=node.get("family"), recipe=operator.get("recipe"),
        operator=operator, metadata=metadata, deps=node.get("deps") or (), outputs=(),
        parallel=parallel, model_id=model_id, layer_idx=layer_idx, topology=topology,
        stage_idx=int(node.get("stage", 0)), replica_idx=int(node.get("replica", 0)),
        batch=batch, kv_length=kv_length, phase=node_phase,
    )
    stage, boundary = _stage_and_boundary([built])
    return {
        **_case_base(model_id=model_id, archetype=archetype, parallel=parallel, arm=arm, phase=node_phase,
                     kind="operator", family=node.get("family"), signature=node.get("physical_signature"),
                     work=node.get("physical_work"), matrix_rows=matrix_rows, batch=batch, kv_length=kv,
                     routing=routing, scope="body" if layer_idx >= 0 else "global"),
        "recipe": operator.get("recipe"), "pipeline_stage": int(node.get("stage", 0)),
        "nodes": [built], "edges": [],
        **({"group_ranks": built["group_ranks"], "group_cosets": built["group_cosets"]} if built.get("group_ranks") else {}),
        "timed_boundary": boundary, "stage": stage,
        "timing": {"boundary": "resource_phase" if metadata.get("device_kernel_group") else "device_kernel",
                   "protocol": "declared_graph_replay_5_warmup_3_trials_20_events"}, **split,
    }


def _subgraph_payload(subgraph, members, *, model_id, archetype, parallel, arm, batch, kv_length, matrix_rows, phase, pr_overlay=False, topology=None):
    members_set, sub_phase = {m["node_id"] for m in members}, str(subgraph.get("phase", phase))
    edges = sorted({(str(dep), str(m["node_id"])) for m in members for dep in m.get("deps") or () if dep in members_set})
    dependents = defaultdict(list)
    for source, target in edges:
        dependents[source].append(target)
    nodes = [_node_fields(
        node_id=m["node_id"], family=m.get("family"), recipe=(m.get("operator") or {}).get("recipe"),
        operator=m.get("operator") or {}, metadata=_metadata(m.get("operator") or {}),
        deps=m.get("deps") or (), outputs=dependents.get(m["node_id"], ()), parallel=parallel,
        model_id=model_id, layer_idx=int(m.get("layer", subgraph.get("layer", MISSING_LAYER))),
        topology=topology, stage_idx=int(m.get("stage", 0)), replica_idx=int(m.get("replica", 0)),
        batch=batch, kv_length=kv_length,
        phase=str(m.get("phase", sub_phase)),
    ) for m in members]
    kv = kv_length if any(_kv_dependent(m.get("family"), (m.get("operator") or {}).get("recipe")) for m in members) else None
    stage, boundary = _stage_and_boundary(nodes)
    return {
        **_case_base(model_id=model_id, archetype=archetype, parallel=parallel, arm=arm, phase=sub_phase,
                     kind="subgraph", family=subgraph.get("family"), signature=subgraph.get("physical_signature"),
                     work=subgraph.get("physical_work"), matrix_rows=matrix_rows, batch=batch, kv_length=kv,
                     routing=None, scope="body" if int(subgraph.get("layer", MISSING_LAYER)) >= 0 else "global"),
        "role": subgraph.get("role"), "nodes": nodes,
        "edges": [{"from": s, "to": t, "kind": "data"} for s, t in edges],
        "timed_boundary": boundary, "stage": stage,
        "timing": {"boundary": "resource_phase", "protocol": "declared_subgraph_boundary_replay_5_warmup_3_trials_20_events"},
        **_split_fields(matrix_rows=matrix_rows, batch=batch, kv_length=kv, pr_overlay=pr_overlay),
    }


def _harvest(cases, *, model_id, archetype, parallel, report, arm, geometry, pr_overlay=False, topology=None, kv_only=False, merged_sigs=None):
    batch = int(geometry["sequences"])
    kv_length = int(geometry["context"]) if geometry["phase"] == "decode" else int(geometry["context"] + geometry["tokens"])
    raw_nodes, raw_subs = collapse_device_kernel_groups(report["nodes"], report["subgraphs"])
    nodes_by_id, groups = {node["node_id"]: node for node in raw_nodes}, defaultdict(list)
    for node in raw_nodes:
        if str(node.get("family") or "") not in CONTROL_FAMILIES and (not kv_only or is_ladder_kv_node(node)):
            groups[node["physical_signature"]].append(node)
    for sig, occurrences in groups.items():
        eff_pr = bool(pr_overlay and (merged_sigs is None or ("op", sig) not in merged_sigs))
        if not pr_overlay and merged_sigs is not None:
            merged_sigs.add(("op", sig))
        payload = _node_payload(occurrences[0], model_id=model_id, archetype=archetype, parallel=parallel, arm=arm,
                                batch=batch, kv_length=kv_length, phase=str(geometry["phase"]), pr_overlay=eff_pr, topology=topology)
        _add_case(cases, _case_key(payload), payload,
                  _occurrences(model_id, archetype, arm, (int(n.get("layer", MISSING_LAYER)) for n in occurrences)))
    groups = defaultdict(list)
    for sub in raw_subs:
        groups[sub["physical_signature"]].append(sub)
    for sig, occurrences in groups.items():
        rep = occurrences[0]
        members = [nodes_by_id[k] for k in rep.get("node_ids") or () if k in nodes_by_id]
        if not members or (kv_only and not any(is_ladder_kv_node(m) for m in members)):
            continue
        eff_pr = bool(pr_overlay and (merged_sigs is None or ("sub", sig) not in merged_sigs))
        if not pr_overlay and merged_sigs is not None:
            merged_sigs.add(("sub", sig))
        rows = [v for v in (_first_dim((n.get("operator") or {}).get("local_shape")) for n in members) if v is not None]
        payload = _subgraph_payload(rep, members, model_id=model_id, archetype=archetype, parallel=parallel, arm=arm,
                                    batch=batch, kv_length=kv_length, matrix_rows=max(rows) if rows else None,
                                    phase=str(geometry["phase"]), pr_overlay=eff_pr, topology=topology)
        layers = ([int(v) for s in occurrences for v in (s.get("layer_ids") or ()) if int(v) >= 0]
                  or [int(rep.get("layer", MISSING_LAYER))])
        _add_case(cases, _case_key(payload), payload, _occurrences(model_id, archetype, arm, layers))
    return {"nodes": len(raw_nodes), "subgraphs": len(raw_subs)}


def _proposal_tokens(model):
    auxiliary = getattr(model, "auxiliary", None)
    if auxiliary is None or auxiliary.standalone_draft:
        return 0
    mtp = getattr(auxiliary, "mtp", None)
    if mtp is not None:
        return max(1, int(mtp.layers)) if (mtp.kind == "mtp" and mtp.execution_mode == "cyclic_nextn"
                                           and not mtp.noncausal_query_block) else 1
    dspark = getattr(auxiliary, "dspark", None)
    return max(1, int(getattr(dspark, "block_size", 0) or 1)) if dspark is not None else 0


def _compile_one(cases, *, model_id, model, archetype, parallel, geometry_name, arm_label, case_spec, proposals, pr_overlay=False, kv_only=False, merged_sigs=None):
    del model
    geometry = dict(GEOMETRY_ARMS[geometry_name])
    active_proposals = proposals if geometry["phase"] == "decode" else 0
    if geometry["phase"] == "decode":
        geometry["tokens"] += active_proposals
    report = export_structure(
        case_spec, phase=geometry["phase"], tokens=geometry["tokens"],
        context=geometry["context"], sequences=geometry["sequences"], scope="model",
        replicas=[0] if parallel["dp"] > 1 else None,
    )
    declared = set(range(int(case_spec.model.layers)))
    seen = {int(n.get("layer", MISSING_LAYER)) for n in report["nodes"] if n.get("phase") != "draft"}
    if not declared.issubset(seen):
        raise PlanError(f"structure_export_did_not_cover_full_depth:{model_id}:{sorted(declared - seen)}")
    if int(case_spec.model.num_experts) > 0 and any(p.num_experts > 0 for p in case_spec.model.layer_plans):
        layer = next(plan for plan in case_spec.model.layer_plans if plan.num_experts > 0)
        sel = select_realization(case_spec.runtime, layer, phase=geometry["phase"], tokens=geometry["tokens"],
                                 valid_tokens=geometry["tokens"] * geometry["sequences"],
                                 global_tokens=geometry["tokens"] * geometry["sequences"] * parallel["dp"], token_shards=1, replica=0)
        if sel is None or any(int(v) != v for v in sel.local_expert_rows):
            raise PlanError(f"balanced_routing_did_not_produce_integer_expert_rows:{model_id}")
    topo = Topology(case_spec.machine, case_spec.runtime.parallel)
    stats = _harvest(cases, model_id=model_id, archetype=archetype, parallel=parallel, report=report, arm=arm_label,
                     geometry=geometry, pr_overlay=pr_overlay, topology=topo, kv_only=kv_only, merged_sigs=merged_sigs)
    drafts = {int(n.get("layer")) for n in report["nodes"] if n.get("phase") == "draft" and int(n.get("layer", MISSING_LAYER)) >= 0}
    arm = {"arm": arm_label, "geometry": dict(geometry), "phase": geometry["phase"], "nodes": stats["nodes"],
           "subgraphs": stats["subgraphs"], "graph_id": report.get("graph_id"),
           "layers_covered": len(seen & declared), "proposal_tokens": active_proposals}
    return arm, drafts


def _compile_model_task(task):
    model_id, structure_sku, track_geometries, probe_ladder = task
    model = load_model(model_id)
    archetype, proposals = model_archetype(model_id, model), _proposal_tokens(model)
    cases, arms, drafts, case_cache = {}, [], set(), {}
    va_prof = va_recommended_config(model_id, structure_sku)
    engram_off = True if va_prof.get("engram_cpu_offload") else None
    base_geoms = tuple(dict.fromkeys(g for _t, g in track_geometries if g in {"prefill", "decode"}))
    base_tracks = tuple(dict.fromkeys(t for t, _g in track_geometries))
    rep_depth = representative_layer_depth(model, layer_signatures(model))
    for axis_tag, parallel, extra_pairs in multi_axis_parallel_arms(model, geometries=base_geoms, tracks=base_tracks, probe_ladder=probe_ladder):
        pairs = track_geometries if axis_tag is None else extra_pairs
        mso = bool(va_prof.get("multistream_overlap_shared_expert")) if parallel["ep"] > 1 else None
        arm_proposals, merged_sigs_by_geom = (proposals if axis_tag is None else 0), defaultdict(set)
        for track, geometry_name in pairs:
            phase, is_probe = str(GEOMETRY_ARMS[geometry_name]["phase"]), geometry_name in HELDOUT_GEOMETRY_PROBES
            eff_layers = None if (axis_tag is None and not is_probe) else rep_depth
            eff_proposals = arm_proposals if (not is_probe and phase == "decode") else 0
            cache_key = (axis_tag or "tp_x_ep", track, phase, eff_layers, eff_proposals)
            if cache_key not in case_cache:
                case = compose_case(
                    model_id, structure_sku, tp=parallel["tp"], dp=parallel["dp"], ep=parallel["ep"],
                    pp=parallel["pp"], pcp=parallel["pcp"], dcp=parallel["dcp"], sp=parallel["sp"],
                    layers=eff_layers, num_devices=parallel["num_devices"], va_recommended=True,
                    multistream_overlap_shared_expert=mso, engram_cpu_offload=engram_off,
                    recover_ssm=False, mtp_proposal_tokens=eff_proposals,
                    graph_capture_mode=("full" if phase == "decode" else "none"),
                    max_model_len=MAX_EXPORT_MODEL_LEN, runtime_stack="vllm-ascend",
                    feature_track=track, _stack_phase=phase,
                )
                if int(model.num_experts) > 0:
                    case["runtime"]["routing"] = {"policy": "balanced"}
                case_cache[cache_key] = parse_case(case)
            suffix = ("" if axis_tag is None else f":{axis_tag}") + ("" if track == "merged" else f":{track}")
            kv_only = bool(geometry_name in KV_ONLY_LADDER_PROBES or (axis_tag is not None and is_probe))
            arm, layers = _compile_one(
                cases, model_id=model_id, model=model, archetype=archetype, parallel=parallel,
                geometry_name=geometry_name, arm_label=f"{geometry_name}{suffix}",
                case_spec=case_cache[cache_key], proposals=eff_proposals,
                pr_overlay=(track != "merged"), kv_only=kv_only, merged_sigs=merged_sigs_by_geom[geometry_name],
            )
            arms.append({**arm, "parallel_axis": axis_tag or ("tp_x_ep" if int(model.num_experts) > 0 else "tp")})
            if axis_tag is None:
                drafts |= layers
    return {"model_id": model_id, "arms": arms, "cases": list(cases.values()), "draft_layers": sorted(drafts)}


def _run_tasks(tasks, workers):
    if workers <= 1:
        return [_compile_model_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_compile_model_task, tasks))


def _capability_tokens(die):
    cap = die.capability
    return tuple(sorted({
        f"preset:{cap.preset}", *(f"representation:{i.name}" for i in cap.representations),
        *(f"mode:{i.name}" for i in cap.native_modes), *(f"path:{i.name}" for i in cap.paths),
        *(f"runtime:{i}" for i in cap.runtimes),
    }))


def _hardware_machine(sku):
    probe = parse_case(compose_case("qwen3-32b", sku, tp=DEFAULT_TP, dp=1, ep=DENSE_EP)).machine
    n = DEFAULT_TP * max(1, int(probe.dies_per_package))
    machine = parse_case(compose_case("qwen3-32b", sku, tp=DEFAULT_TP, dp=1, ep=DENSE_EP, num_devices=n, ranks_per_node=n)).machine
    return machine, machine.dies[0]


def _fingerprint(die):
    return f"{die.capability.preset}:{_digest(_capability_tokens(die))}"


def _sku_compile_key(sku):
    machine, die = _hardware_machine(sku)
    return (machine.arch_family, int(die.compute_units), int(machine.dies_per_package),
            int(machine.ranks_per_node), int(dict(die.buffers).get("l2", 0)), int(die.memory_bytes))


def _hardware_view(sku, structure_sku):
    machine, die = _hardware_machine(sku)
    l2_bytes = int(dict(die.buffers).get("l2", 0))
    cold_ring = FABRIC_COLD_RING_FACTOR * l2_bytes
    node_ids = sorted({int(getattr(d, "node", 0)) for d in machine.dies})
    pkg_keys = sorted({(int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) for d in machine.dies})
    rank_node = {int(d.rank): int(getattr(d, "node", 0)) for d in machine.dies}
    has_inter = any(
        len({rank_node[int(r)] for r in lnk.members if int(r) in rank_node}) >= 2
        and any(m in f"{lnk.channel} {lnk.name}".lower() for m in ("inter_node", "inter_pod", "roce", "rdma", "scaleout", "scale_out"))
        for lnk in machine.links
    )
    return {
        "sku": sku, "architecture": machine.arch_family, "capability_preset": die.capability.preset,
        "dies_per_node": int(machine.ranks_per_node), "capability_digest": _digest(_capability_tokens(die)),
        "dies_per_package": int(machine.dies_per_package), "l2_bytes_per_die": l2_bytes,
        "hbm_bytes_per_die": int(die.memory_bytes), "fabric_cold_ring_bytes": cold_ring,
        "structure_identity_sku": structure_sku,
        "nodes": len(node_ids), "total_dies": len(machine.dies),
        "ranks": [int(d.rank) for d in machine.dies],
        "ranks_by_node": [sorted(int(d.rank) for d in machine.dies if int(getattr(d, "node", 0)) == n) for n in node_ids],
        "package_ranks": [sorted(int(d.rank) for d in machine.dies if (int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) == k) for k in pkg_keys],
        "has_inter_node_link": has_inter,
        "declared_links": [{"name": lnk.name, "channel": lnk.channel, "members": sorted(int(r) for r in lnk.members)} for lnk in machine.links],
        "fabric_cold_ring_8mib_slots": int(math.ceil(cold_ring / FABRIC_SLOT_BYTES)),
        "structure_identity_is_sku_independent": False,
    }


def _fabric_cases(hardware):
    routes = [
        r for r in FABRIC_ROUTES
        if (r != "intra_package_die" or hardware["dies_per_package"] >= 2)
        and (r != "intra_node_hccs" or hardware["dies_per_node"] // max(1, hardware["dies_per_package"]) >= 2)
        and (r != "inter_node_roce" or (int(hardware.get("nodes") or 1) >= 2 and bool(hardware.get("has_inter_node_link", True))))
    ]
    cases = []
    for collective_op in FABRIC_COLLECTIVES:
        builder, upstream_abi = resolve_fabric_collective_abi(collective_op)
        for route in routes:
            for payload in FABRIC_LADDER_BYTES:
                participants = FABRIC_PARTICIPANTS[route]
                ranks = fabric_route_ranks(route, participants, hardware)
                node = {
                    "schema": NODE_SCHEMA, "node_id": "fab0", "kind": "collective", "op": collective_op,
                    "display_name": f"collective:{collective_op}", "builder": builder,
                    "upstream_abi": upstream_abi, "dtype": "fp32", "output_dtype": "fp32",
                    "layout": "contiguous", "inputs": [], "outputs": [], "group_list": None,
                    "group_list_type": 1, "per_token_scale": None, "format_code": None,
                    "nbytes": int(payload), "group_size": participants, "group": route,
                    "group_ranks": ranks, "group_cosets": [ranks], "implementation": upstream_abi,
                    "implementation_source": "builder_upstream_abi", "kernel": upstream_abi,
                    "shape": {"bytes": int(payload), "group_size": participants},
                }
                cases.append({
                    "case_id": _digest(("fabric", collective_op, route, payload), "fabric-"),
                    "kind": "fabric", "family": "collective", "recipe": collective_op, "phase": "fabric",
                    "scope": "interconnect", "sku": hardware["sku"], "route": route, "participants": participants,
                    "group_ranks": ranks, "group_cosets": [ranks], "fabric_bytes": int(payload),
                    "work_bytes": int(payload), "matrix_rows": None, "batch": None, "kv_length": None,
                    "stage": "s1_joint", "nodes": [node], "edges": [], "timed_boundary": {"include": ["fab0"]},
                    "occurrence_count": 0,
                    "timing": {"boundary": "device_kernel", "protocol": "collective_completion_events_2_5ms_blocker"},
                    "memory_budget": {"expanded_allocation_bytes": int(payload), "replay_ceiling_bytes": S0_REPLAY_CEILING_BYTES},
                    **_split_fields(fabric_bytes=payload),
                })
    return cases


def _heldout_probe_models(models):
    by_arch = defaultdict(list)
    for model_id, rec in models.items():
        by_arch[rec["archetype"]["archetype_id"]].append(model_id)
    return sorted(min(cands, key=lambda m: (bool(getattr(getattr(models[m]["_model"], "auxiliary", None), "standalone_draft", False)),
                                            models[m]["parameter_count"], m)) for cands in by_arch.values())


def _model_record(model_id):
    model = load_model(model_id)
    parallel = parallel_policy(model)
    slots = (_slot_plan_for(int(model.num_experts), int(parallel["ep"]),
                            redundant=int(parallel.get("redundant_experts_per_rank") or 0))
             if int(model.num_experts) > 0 else None)
    return {
        "model_id": model_id, "archetype": model_archetype(model_id, model), "parallel": parallel,
        "parameter_count": _parameter_count(model), "layers": int(model.layers), "attention": model.attention,
        "ffn": model.ffn, "num_experts": int(model.num_experts), "top_k": int(model.top_k),
        "quantization": model.quant.name, "expert_bank": _expert_bank_bytes(model, parallel, slots=slots),
        "layer_signatures": layer_signatures(model), "proposal_tokens": _proposal_tokens(model),
        "compiled_arms": [], "_model": model,
    }


def compile_structural_catalog(model_ids=None, *, skus=DEFAULT_SKUS, geometries=DEFAULT_GEOMETRIES,
                               tracks=DEFAULT_TRACKS, heldout_geometry_probes=True, workers=None):
    """Compile one deduplicated structural case table per die capability group."""
    catalog = _catalog_entries()
    selected = sorted(model_ids or catalog)
    missing = [m for m in selected if m not in catalog]
    if missing:
        raise PlanError(f"unknown_catalog_models:{','.join(missing)}")
    if not selected:
        raise PlanError("plan_requires_at_least_one_model")
    resolved = tuple(resolve_hardware_sku(sku) for sku in skus)
    for track in tracks:
        if track not in {"merged", "unmerged_pr"}:
            raise PlanError(f"unknown_feature_track:{track}")
    probe_geoms = probe_geometries_for(geometries, enabled=heldout_geometry_probes)
    for geometry in (*geometries, *probe_geoms):
        if geometry not in GEOMETRY_ARMS:
            raise PlanError(f"unknown_geometry_arm:{geometry}")
    has_decode = any(GEOMETRY_ARMS[g]["phase"] == "decode" for g in (*geometries, *probe_geoms))
    worker_count = int(workers) if workers is not None else min(4, os.cpu_count() or 1) if len(selected) >= 4 else 1
    if worker_count < 1:
        raise PlanError(f"workers_requires_positive_integer:{worker_count}")
    groups: dict[str, dict[str, Any]] = {}
    sku_groups = {}
    for sku in resolved:
        machine, die = _hardware_machine(sku)
        fp = _fingerprint(die)
        sku_groups[sku] = fp
        group = groups.setdefault(fp, {"fingerprint": fp, "capability_preset": die.capability.preset,
                                       "capability_tokens": list(_capability_tokens(die)), "architectures": [],
                                       "skus": [], "structure_sku": sku, "models": {}, "cases": {}})
        group["architectures"].append(machine.arch_family)
        group["skus"].append(sku)
    structural = {
        "schema": SCHEMA, "skus": list(resolved), "geometries": list(geometries), "tracks": list(tracks),
        "groups": groups, "hardware": {}, "sku_groups": sku_groups,
        "sequence_ladder_tokens": list(SEQUENCE_LADDER_TOKENS),
        "catalog": {"source": "CORE_CATALOG_ENTRIES", "model_count": len(catalog), "model_ids": sorted(catalog)},
        "heldout_geometry_probes": list(probe_geoms),
        "diagnostics": {"routing_arms_skipped": [], "compile_errors": []},
    }
    for group in groups.values():
        group["sku_cases"], group["sku_models"], sub_cache = {}, {}, {}
        for sku in group["skus"]:
            sub_key = _sku_compile_key(sku)
            if sub_key not in sub_cache:
                models, cases = {m: _model_record(m) for m in selected}, {}
                probes = set(_heldout_probe_models(models)) if heldout_geometry_probes else set()
                tasks = [(m, sku, [*((t, g) for t in tracks for g in geometries),
                          *(("merged", g) for g in (probe_geoms if m in probes else ()))], m in probes) for m in selected]
                for res in _run_tasks(tasks, worker_count):
                    rec = models[res["model_id"]]
                    rec["compiled_arms"] = res["arms"]
                    rec["layer_coverage"] = _layer_coverage(rec["_model"], rec["layer_signatures"], res["draft_layers"], has_decode_arm=has_decode)
                    for case in res["cases"]:
                        _merge_case(cases, case)
                _add_routing_arms({"models": models, "cases": cases, "diagnostics": structural["diagnostics"]})
                for rec in models.values():
                    cov = rec["layer_coverage"]
                    if not cov["body"]["complete"] or any(not i["complete"] and not i.get("declared_only") for i in cov["drafts"].values()):
                        raise PlanError(f"layer_signature_coverage_incomplete:{rec['model_id']}")
                    rec.pop("_model", None)
                sub_cache[sub_key] = (sku, models, cases)
            sub_sku, sub_models, sub_cases = sub_cache[sub_key]
            group["sku_models"][sku], group["sku_cases"][sku] = sub_models, sub_cases
            structural["hardware"][sku] = _hardware_view(sku, sub_sku)
        group["models"], group["cases"] = group["sku_models"][group["structure_sku"]], group["sku_cases"][group["structure_sku"]]
    structural["archetypes"] = [{
        "archetype_id": n, "name": name, "declared_model_ids": ids.split(),
        "compiled_model_ids": sorted({m for g in groups.values() for m, r in g["models"].items() if r["archetype"]["archetype_id"] == n}),
    } for n, name, ids in ARCHETYPES]
    return structural


def _group_for(structural, sku):
    try:
        return structural["groups"][structural["sku_groups"][sku]]
    except KeyError as error:
        raise PlanError(f"sku_has_no_structural_group:{sku}") from error


def render_plan(structural, sku):
    """Materialize one SKU plan from its capability group's structural cases."""
    resolved = resolve_hardware_sku(sku)
    hardware, group = structural["hardware"][resolved], _group_for(structural, resolved)
    sku_cases = group.get("sku_cases", {}).get(resolved, group["cases"])
    sku_models = group.get("sku_models", {}).get(resolved, group["models"])
    cases = []
    for case in sorted(sku_cases.values(), key=lambda i: i["case_id"]):
        entry = {**case["payload"], "sku": resolved, "occurrence_count": case["occurrence_count"]}
        entry["identities"] = _identities(entry, resolved, hardware)
        cases.append(entry)
    for entry in _fabric_cases(hardware):
        entry["identities"] = _identities(entry, resolved, hardware)
        cases.append(entry)
    reverse_index = _reverse_index(sku_cases)
    for entry in cases:
        reverse_index.setdefault(entry["case_id"], [])
    split_ids, reasons = {"train": [], "heldout": []}, Counter()
    for entry in cases:
        split_ids[entry["split"]].append(entry["case_id"])
        reasons.update(entry["split_reasons"])
    digest = _digest([entry["case_id"] for entry in cases])
    return {
        "schema": SCHEMA, "plan_id": f"plan-{resolved}-{digest}", "stage": "s0_local", "rank": None,
        "world_size": 1, "execution_modes": ["eager", "aclgraph"], "sku": resolved,
        "catalog_model_count": structural["catalog"]["model_count"], "catalog_models": structural["catalog"]["model_ids"],
        "compiled_models": sorted(sku_models), "hardware": hardware, "archetypes": structural["archetypes"],
        "models": [_compact_model(sku_models[m]) for m in sorted(sku_models)], "cases": cases, "reverse_index": reverse_index,
        "split_manifest": {
            "policy": "heldout_if_any_reserved_axis_before_timing", "heldout_matrix_rows": sorted(HELDOUT_MATRIX_ROWS),
            "heldout_batch": HELDOUT_BATCH, "heldout_kv_length": HELDOUT_KV_LENGTH,
            "heldout_fabric_bytes": sorted(HELDOUT_FABRIC_BYTES),
            "heldout_routing": {"distribution": HELDOUT_ROUTING_DISTRIBUTION, "zero_expert_fraction": HELDOUT_ROUTING_ZERO_FRACTION},
            "real_weights_heldout": True, "pr_overlay_heldout": True, "train_case_ids": split_ids["train"],
            "heldout_case_ids": split_ids["heldout"], "counts": {k: len(v) for k, v in split_ids.items()},
            "reasons": dict(sorted(reasons.items())),
        },
        "summary": {
            "cases": len(cases), "train": sum(c["split"] == "train" for c in cases),
            "heldout": sum(c["split"] == "heldout" for c in cases),
            "by_kind": dict(sorted(Counter(c["kind"] for c in cases).items())), "reverse_index_cases": len(reverse_index),
            "by_stage": dict(sorted(Counter(c["stage"] for c in cases).items())), "case_id_digest": digest,
            "splits": dict(sorted(Counter(c["split"] for c in cases).items())),
            "occurrences": sum(i["count"] for g in reverse_index.values() for i in g),
        },
    }


def export_plan(output, *, skus=DEFAULT_SKUS, models=None, geometries=DEFAULT_GEOMETRIES,
                tracks=DEFAULT_TRACKS, heldout_geometry_probes=True, workers=None):
    """Compile and write the manifest plus one plan file per SKU."""
    structural = compile_structural_catalog(models, skus=skus, geometries=geometries, tracks=tracks,
                                            heldout_geometry_probes=heldout_geometry_probes, workers=workers)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = []
    for sku in structural["skus"]:
        plan = render_plan(structural, sku)
        path = out_dir / f"{sku}{PLAN_SUFFIX}"
        path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        plans.append({"sku": sku, "path": path.name, "cases": plan["summary"]["cases"], "train": plan["summary"]["train"],
                      "heldout": plan["summary"]["heldout"], "stage": plan["summary"]["by_stage"],
                      "case_id_digest": plan["summary"]["case_id_digest"]})
    primary, primary_plan = next(iter(structural["groups"].values())), render_plan(structural, structural["skus"][0])
    manifest = {
        "schema": SCHEMA, "catalog": structural["catalog"], "compiled_models": sorted(primary["models"]),
        "skus": list(structural["skus"]), "structure_identity_is_sku_independent": False,
        "structure_groups": [{
            "fingerprint": g["fingerprint"], "capability_preset": g["capability_preset"], "skus": g["skus"],
            "architectures": sorted(set(g["architectures"])), "structure_sku": g["structure_sku"], "cases": len(g["cases"]),
            "case_id_digest": _digest([c["case_id"] for c in sorted(g["cases"].values(), key=lambda i: i["case_id"])]),
        } for g in structural["groups"].values()],
        "geometries": structural["geometries"], "heldout_geometry_probes": structural["heldout_geometry_probes"],
        "archetypes": structural["archetypes"], "models": [_compact_model(primary["models"][m]) for m in sorted(primary["models"])],
        "summary": {
            "compiled_models": len(primary["models"]), "catalog_models": structural["catalog"]["model_count"],
            "cases": primary_plan["summary"]["cases"], "structure_groups": len(structural["groups"]),
            "occurrences": primary_plan["summary"]["occurrences"], "by_kind": primary_plan["summary"]["by_kind"],
            "splits": primary_plan["summary"]["splits"],
            "routing_arms_skipped": len(structural["diagnostics"]["routing_arms_skipped"]),
        },
        "routing_arms_skipped": structural["diagnostics"]["routing_arms_skipped"], "plans": plans,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", choices=("maintained",), default="maintained")
    parser.add_argument("--skus", nargs="+", default=list(DEFAULT_SKUS))
    parser.add_argument("--models", nargs="+", default=None, help="Optional model subset for a scoped local compile")
    parser.add_argument("--geometries", nargs="+", default=list(DEFAULT_GEOMETRIES), choices=sorted(GEOMETRY_ARMS))
    parser.add_argument("--tracks", nargs="+", default=list(DEFAULT_TRACKS), choices=("merged", "unmerged_pr"))
    parser.add_argument("--workers", type=int, default=None, help="Per-model compile workers (default: bounded by catalog size)")
    parser.add_argument("--no-heldout-geometry-probes", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = export_plan(args.out, skus=args.skus, models=args.models, geometries=args.geometries, tracks=args.tracks,
                               workers=args.workers, heldout_geometry_probes=not args.no_heldout_geometry_probes)
    except (PlanError, UnmodeledCapabilityError, ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "ok", "schema": manifest["schema"], "catalog_models": manifest["catalog"]["model_count"],
        "splits": manifest["summary"]["splits"], "compiled_models": len(manifest["compiled_models"]),
        "cases": manifest["summary"]["cases"], "occurrences": manifest["summary"]["occurrences"],
        "plans": [plan["path"] for plan in manifest["plans"]],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
