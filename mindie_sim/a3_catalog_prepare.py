"""Prepare-eligible catalog work for cross-forward asynchronous scheduling.

[Role]
  Expose real model-derived preparation candidates (quant scale/layout, KDA
  convolution layout, V4.1 compressor ring capacity, request-known Engram ID
  metadata and vision position-table preparation) with exact tensor shapes,
  byte/operation counts, known-at classification, lifetime, owner, resource
  demands and consumer selectors.

[Invariants]
  * Every candidate is a semantic operation or an explicit capacity
    reservation derived from the model config; no latency placeholder or
    invented measured coefficient is emitted.
  * ``known_at`` is one of model_load / request_arrival / runtime_static.
    Future decode tokens, QKV, router results and speculative acceptance are
    never marked known early.
  * Engram row values are not precomputed: only ID/metadata preparation is
    eligible before the consumer; row fetch/gather remains dependent on the
    request IDs and the consumer op.
  * Vision preparation is eligible only when the processed grid is supplied in
    runtime metadata (or the media token count is already fixed); raw-pixel
    processor semantics remain an explicit gap.
  * CPU/H2D service rates are requested as explicit inputs; missing rates are
    a measurement gap, not zero time.
  * ``cost_once_key`` prevents double charging only for genuinely
    model-resident objects. Request/iteration-scoped candidates are never
    globally reusable.
  * ``request_id``/``arrival_ms``/``iteration`` scope preparation; ``known_at``
    is ``previous_forward`` for decode metadata derived from a sampled token.
  * ``owner_stage``/``owner_rank`` and per-rank tensor shapes partition physical
    work by TP/EP/PP; whole-model movement is not a live-capacity quantity.
  * ``replace_op_ids`` names the exact semantic ops replaced by preparation;
    ``first_consumer_op_ids`` names where the readiness edge attaches. Capacity
    candidates expose ``capacity_bytes`` with zero ``service_bytes``; MTE read
    and write directions remain separate.

[Timeouts & Locks]
  Pure arithmetic bounded by layers/tables/terms; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .a3_catalog_types import QuantSpec
from .a3_catalog_weights import layer_weight_terms, stage_weight_terms
from .a3_model_types import ModelSpec
from .a3_types import A3InputError, ParallelSpec, RuntimeSpec, dtype_bytes


@dataclass(frozen=True)
class PrepareCandidate:
    candidate_id: str
    kind: str
    phase: str
    known_at: str
    lifetime: str
    owner: str
    consumer_selector: str
    consumer_op_ids: tuple[str, ...] = ()
    first_consumer_op_ids: tuple[str, ...] = ()
    replace_op_ids: tuple[str, ...] = ()
    tensor_shape: tuple[int, ...] = ()
    bytes: float = 0.0
    operations: float = 0.0
    resources: tuple[tuple[str, float], ...] = ()
    request_id: str = ""
    arrival_ms: float = 0.0
    iteration: int = -1
    reusable: bool = True
    owner_stage: int = -1
    owner_rank: str = ""
    capacity_bytes: float = 0.0
    read_bytes: float = 0.0
    write_bytes: float = 0.0
    service_bytes: float = 0.0
    depends_on: tuple[str, ...] = ()
    cost_once_key: str = ""
    host_rates_required: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "kind": self.kind,
            "phase": self.phase, "known_at": self.known_at,
            "lifetime": self.lifetime, "owner": self.owner,
            "consumer_selector": self.consumer_selector,
            "consumer_op_ids": list(self.consumer_op_ids),
            "first_consumer_op_ids": list(self.first_consumer_op_ids),
            "replace_op_ids": list(self.replace_op_ids),
            "tensor_shape": list(self.tensor_shape),
            "bytes": self.bytes, "operations": self.operations,
            "resources": dict(self.resources),
            "request_id": self.request_id,
            "arrival_ms": self.arrival_ms,
            "iteration": self.iteration,
            "reusable": self.reusable,
            "owner_stage": self.owner_stage,
            "owner_rank": self.owner_rank,
            "capacity_bytes": self.capacity_bytes,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "service_bytes": self.service_bytes,
            "depends_on": list(self.depends_on),
            "cost_once_key": self.cost_once_key,
            "host_rates_required": list(self.host_rates_required),
            "metadata": dict(self.metadata),
        }


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _work_ops(work: Any | None) -> Iterable[Any]:
    return tuple(getattr(work, "ops", ()) or ())


def _stage_of(layer_id: int, pp: int, layers: int) -> int:
    return min(max(1, pp) - 1, int(layer_id) * max(1, pp) // max(1, layers))


def _consumers(
    work: Any | None, *, families: Sequence[str] = (),
    quantized: bool = False, op_suffixes: Sequence[str] = (),
    request_id: str = "", stage: int | None = None,
    pp: int = 1, layers: int = 0,
) -> tuple[str, ...]:
    result: list[str] = []
    for op in _work_ops(work):
        op_id = str(getattr(op, "op_id", ""))
        if request_id:
            op_requests = tuple(str(x) for x in getattr(op, "requests", ()) or ())
            if request_id not in op_requests:
                continue
        if stage is not None and layers > 0:
            if _stage_of(int(getattr(op, "layer", -1)), pp, layers) != stage:
                continue
        meta = dict(getattr(op, "metadata", ()) or ())
        family = str(meta.get("semantic_family", "")) or str(getattr(op, "family", ""))
        if families and not any(token in family for token in families):
            continue
        if op_suffixes and not any(op_id.endswith(suffix) for suffix in op_suffixes):
            continue
        if quantized:
            fmt = str(meta.get("weight_format", ""))
            if fmt in {"bf16", "", "fp32", "float32"}:
                continue
        result.append(op_id)
    return tuple(sorted(set(result)))


def _scale_candidates(
    model: ModelSpec, parallel: ParallelSpec, work: Any | None,
) -> list[PrepareCandidate]:
    from .a3_catalog_weights import _sharded_shape

    quant = model.quant
    if quant is None:
        return []
    pp = max(1, parallel.pp)
    stages: dict[int, dict[str, float]] = {}
    block_rows, block_cols = quant.weight_block
    for plan in model.layer_plans:
        stage = _stage_of(plan.index, pp, model.layers)
        row = stages.setdefault(stage, {
            "fp8_scale": 0.0, "fp4_scale": 0.0, "fp4_data": 0.0,
        })
        for term in layer_weight_terms(plan, model):
            fmt = quant.format_for(term.category)
            rows, cols, count = _sharded_shape(term, parallel, 1)
            if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"} and block_rows > 0:
                row["fp8_scale"] += (
                    _ceil_div(rows, block_rows)
                    * _ceil_div(cols, block_cols) * count
                )
            elif fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
                row["fp4_scale"] += (
                    rows * _ceil_div(cols, max(1, quant.expert_block)) * count
                )
                row["fp4_data"] += rows * _ceil_div(cols, 2) * count
    candidates: list[PrepareCandidate] = []
    scale_bytes = float(quant.scale_bytes)
    for stage, row in sorted(stages.items()):
        stage_consumers = _consumers(
            work, quantized=True, stage=stage, pp=pp, layers=model.layers,
        )
        fp8_scale = int(row["fp8_scale"])
        if fp8_scale:
            payload = fp8_scale * scale_bytes
            candidates.append(PrepareCandidate(
                candidate_id=f"prepare.quant_fp8_scale_layout.s{stage}",
                kind="quant_scale_layout", phase="prepare",
                known_at="model_load", lifetime="model_resident",
                owner=f"stage{stage}_rank_local", owner_stage=stage,
                owner_rank="rank_local",
                consumer_selector="quantized_gemms",
                consumer_op_ids=stage_consumers,
                first_consumer_op_ids=stage_consumers[:1],
                tensor_shape=(fp8_scale,), bytes=payload,
                operations=float(fp8_scale),
                resources=(
                    ("hbm_bytes", 2.0 * payload),
                    ("mte2_bytes", payload), ("mte3_bytes", payload),
                    ("launch", 1.0),
                ),
                capacity_bytes=payload, read_bytes=payload,
                write_bytes=payload, service_bytes=2.0 * payload,
                cost_once_key=f"{model.name}:fp8_scale_layout:s{stage}:{fp8_scale}",
                metadata=(("scale_block", (block_rows, block_cols)),
                          ("scale_dtype", quant.scale_format),
                          ("source_and_runtime_distinct", True),
                          ("owner_stage", stage),
                          ("whole_model_quantity", False)),
            ))
        fp4_scale = int(row["fp4_scale"])
        if fp4_scale:
            fp4_data = int(row["fp4_data"])
            payload = fp4_scale
            read_bytes = fp4_data + payload
            write_bytes = fp4_data + payload
            service_bytes = read_bytes + write_bytes
            candidates.append(PrepareCandidate(
                candidate_id=f"prepare.quant_fp4_scale_layout.s{stage}",
                kind="quant_scale_layout", phase="prepare",
                known_at="model_load", lifetime="model_resident",
                owner=f"stage{stage}_expert_owner", owner_stage=stage,
                owner_rank="expert_owner",
                consumer_selector="routed_expert_gemms",
                consumer_op_ids=_consumers(
                    work, families=("expert",), quantized=True,
                    stage=stage, pp=pp, layers=model.layers,
                ),
                first_consumer_op_ids=_consumers(
                    work, families=("expert",), quantized=True,
                    stage=stage, pp=pp, layers=model.layers,
                )[:1],
                tensor_shape=(fp4_scale,), bytes=write_bytes,
                operations=float(fp4_scale),
                resources=(
                    ("hbm_bytes", service_bytes),
                    ("mte2_bytes", read_bytes), ("mte3_bytes", write_bytes),
                    ("launch", 1.0),
                ),
                capacity_bytes=write_bytes,
                read_bytes=read_bytes, write_bytes=write_bytes,
                service_bytes=service_bytes,
                cost_once_key=f"{model.name}:fp4_scale_layout:s{stage}:{fp4_scale}",
                metadata=(("group_size", quant.expert_block),
                          ("scale_dtype", "fp8_e8m0"),
                          ("packed_data_bytes", fp4_data),
                          ("owner_stage", stage),
                          ("whole_model_quantity", False)),
            ))
    return candidates


def _kda_conv_candidates(
    model: ModelSpec, parallel: ParallelSpec, work: Any | None,
) -> list[PrepareCandidate]:
    pp = max(1, parallel.pp)
    stages: dict[int, dict[str, float]] = {}
    layers_by_stage: dict[int, int] = {}
    for plan in model.layer_plans:
        if plan.attention != "kda" or plan.conv_width <= 0:
            continue
        stage = _stage_of(plan.index, pp, model.layers)
        row = stages.setdefault(stage, {"elements": 0.0, "source": 0.0})
        elements = int(plan.conv_width) * int(plan.conv_kernel)
        row["elements"] += elements
        row["source"] += float(elements * 4)
        layers_by_stage[stage] = layers_by_stage.get(stage, 0) + 1
    candidates: list[PrepareCandidate] = []
    for stage, row in sorted(stages.items()):
        elements = int(row["elements"])
        source_bytes = float(row["source"])
        prepared_bytes = float(elements * 2)
        consumers = _consumers(
            work, families=("kda",), op_suffixes=(".conv",),
            stage=stage, pp=pp, layers=model.layers,
        )
        candidates.append(PrepareCandidate(
            candidate_id=f"prepare.kimi_kda_conv_layout.s{stage}",
            kind="conv_kernel_layout", phase="prepare",
            known_at="model_load", lifetime="model_resident",
            owner=f"stage{stage}_rank_local", owner_stage=stage,
            owner_rank="rank_local",
            consumer_selector="kda_conv",
            consumer_op_ids=consumers,
            first_consumer_op_ids=consumers[:1],
            tensor_shape=(elements,), bytes=source_bytes,
            operations=float(elements),
            resources=(
                ("hbm_bytes", source_bytes + prepared_bytes),
                ("mte2_bytes", source_bytes),
                ("mte3_bytes", prepared_bytes), ("launch", 1.0),
            ),
            capacity_bytes=prepared_bytes,
            read_bytes=source_bytes, write_bytes=prepared_bytes,
            service_bytes=source_bytes + prepared_bytes,
            cost_once_key=f"{model.name}:kda_conv_layout:s{stage}:{elements}",
            metadata=(("layers", layers_by_stage[stage]),
                      ("source_dtype", "fp32"), ("prepared_dtype", "bf16"),
                      ("source_bytes_preserved", source_bytes),
                      ("prepared_bytes", prepared_bytes),
                      ("owner_stage", stage),
                      ("whole_model_quantity", False)),
        ))
    return candidates


def _compressor_buffer_candidates(
    model: ModelSpec, parallel: ParallelSpec, work: Any | None,
    workload: Any | None,
) -> list[PrepareCandidate]:
    facts = dict(model.catalog_facts)
    if str(facts.get("profile", "")) != "deepseek_v41":
        return []
    requests = tuple(getattr(workload, "requests", ()) or ())
    request_rows = requests or (None,)
    pp = max(1, parallel.pp)
    candidates: list[PrepareCandidate] = []
    for plan in model.layer_plans:
        if not plan.owns_long_kv or plan.compress_ratio != 2:
            continue
        stage = _stage_of(plan.index, pp, model.layers)
        rows, d = 32, plan.head_dim
        reserved = float(rows * 2 * d * 4)
        for request in request_rows:
            request_id = str(getattr(request, "request_id", "")) if request else ""
            consumers = _consumers(
                work, families=("recurrent_state",),
                op_suffixes=(".c2_ring",), request_id=request_id,
                stage=stage, pp=pp, layers=model.layers,
            )
            if not consumers:
                consumers = _consumers(
                    work, families=("recurrent_state",),
                    op_suffixes=(".c2_ring",),
                )
            suffix = f".r{request_id}" if request_id else ""
            candidates.append(PrepareCandidate(
                candidate_id=f"prepare.compressor_ring.l{plan.index}{suffix}",
                kind="compressor_buffer_capacity", phase="prepare",
                known_at="request_arrival", lifetime="request_resident",
                owner=f"stage{stage}_request_ring", owner_stage=stage,
                owner_rank="rank_local",
                consumer_selector="c2_ring",
                consumer_op_ids=consumers,
                first_consumer_op_ids=consumers[:1],
                tensor_shape=(rows, 2, d), bytes=0.0, operations=0.0,
                resources=(("hbm_reserved_bytes", reserved), ("launch", 0.0)),
                request_id=request_id,
                arrival_ms=float(getattr(request, "arrival_ms", 0.0)) if request else 0.0,
                iteration=0, reusable=False,
                capacity_bytes=reserved, read_bytes=0.0,
                write_bytes=0.0, service_bytes=0.0,
                cost_once_key=(
                    f"{model.name}:c2_ring_capacity:l{plan.index}{suffix}"
                ),
                metadata=(("capacity_rows", rows),
                          ("valid_rows_rule", "min(S,32)"),
                          ("ratio", plan.compress_ratio),
                          ("fp32_kv_and_score_ring", True),
                          ("capacity_not_service_bytes", True),
                          ("request_resident", True)),
            ))
    return candidates


def _engram_candidates(
    model: ModelSpec, runtime: RuntimeSpec, work: Any | None,
    workload: Any | None, parallel: ParallelSpec,
) -> list[PrepareCandidate]:
    engram = model.engram
    if engram is None:
        return []
    requests = tuple(getattr(workload, "requests", ()) or ())
    if requests:
        request_rows = requests
        basis = "request_prompt_tokens_known_at_request_arrival"
    else:
        request_rows = (None,)
        basis = "runtime_batch_budget_upper_bound"
    pp = max(1, parallel.pp)
    candidates: list[PrepareCandidate] = []
    for request in request_rows:
        request_id = str(getattr(request, "request_id", "")) if request else ""
        arrival = float(getattr(request, "arrival_ms", 0.0)) if request else 0.0
        prompt_tokens = (
            int(getattr(request, "prompt_tokens", 0)) if request
            else int(runtime.max_num_batched_tokens)
        )
        output_tokens = int(getattr(request, "output_tokens", 0)) if request else 0
        for table in engram.tables:
            stage = _stage_of(table.layer_id, pp, model.layers)
            lookups = max(0, prompt_tokens * engram.lookups_per_token_per_table)
            ids_bytes = float(lookups * engram.id_bytes)
            ids_ops = _consumers(
                work, families=("engram",),
                op_suffixes=(f"engram_ids_t{table.table_id}",),
                request_id=request_id,
            )
            query_ops = _consumers(
                work, families=("engram",),
                op_suffixes=(f"engram_query_t{table.table_id}",),
                request_id=request_id,
            )
            if not query_ops and ids_ops:
                query_ops = tuple(
                    op_id.replace("engram_ids_t", "engram_query_t")
                    for op_id in ids_ops
                )
            suffix = f".r{request_id}" if request_id else ""
            candidates.append(PrepareCandidate(
                candidate_id=f"prepare.engram_ids.t{table.table_id}{suffix}",
                kind="engram_id_metadata", phase="prepare",
                known_at="request_arrival" if request is not None else "runtime_static",
                lifetime="request_prompt_iteration",
                owner=f"stage{stage}_tp_leader", owner_stage=stage,
                owner_rank="tp_leader",
                consumer_selector="engram_query",
                consumer_op_ids=query_ops,
                first_consumer_op_ids=query_ops[:1],
                replace_op_ids=ids_ops,
                tensor_shape=(lookups,), bytes=ids_bytes,
                operations=float(lookups * 2),
                resources=(
                    ("cpu_ops", float(lookups * 2)), ("hbm_bytes", ids_bytes),
                    ("mte3_bytes", ids_bytes), ("launch", 1.0),
                ),
                request_id=request_id, arrival_ms=arrival, iteration=0,
                reusable=False, read_bytes=ids_bytes, write_bytes=ids_bytes,
                service_bytes=ids_bytes,
                host_rates_required=("cpu_ops_per_s",),
                cost_once_key=(
                    f"{model.name}:engram_ids:t{table.table_id}{suffix}:i0"
                ),
                metadata=(("table_id", table.table_id), ("lookups", lookups),
                          ("lookups_per_token", engram.lookups_per_token_per_table),
                          ("token_basis", basis),
                          ("row_values_not_precomputed", True),
                          ("consumer_still_fetches_rows", True),
                          ("replaces_exact_metadata_op", True),
                          ("request_scoped", True)),
            ))
            if output_tokens > 1 and request is not None:
                decode_lookups = engram.lookups_per_token_per_table
                decode_bytes = float(decode_lookups * engram.id_bytes)
                for step in range(1, output_tokens):
                    candidates.append(PrepareCandidate(
                        candidate_id=(
                            f"prepare.engram_ids_decode.t{table.table_id}"
                            f"{suffix}.step{step}"
                        ),
                        kind="engram_id_metadata", phase="prepare",
                        known_at="previous_forward", lifetime="iteration",
                        owner=f"stage{stage}_tp_leader", owner_stage=stage,
                        owner_rank="tp_leader",
                        consumer_selector="engram_query",
                        consumer_op_ids=(),
                        first_consumer_op_ids=(),
                        tensor_shape=(decode_lookups,),
                        bytes=decode_bytes,
                        operations=float(decode_lookups * 2),
                        resources=(
                            ("cpu_ops", float(decode_lookups * 2)),
                            ("hbm_bytes", decode_bytes),
                            ("mte3_bytes", decode_bytes), ("launch", 1.0),
                        ),
                        request_id=request_id, arrival_ms=arrival,
                        iteration=step, reusable=False,
                        read_bytes=decode_bytes, write_bytes=decode_bytes,
                        service_bytes=decode_bytes,
                        host_rates_required=("cpu_ops_per_s",),
                        cost_once_key="",
                        metadata=(
                            ("table_id", table.table_id),
                            ("lookups", decode_lookups),
                            ("token_basis", "decoded_token_after_previous_forward"),
                            ("materialize_after_previous_forward", True),
                            ("token_ids_known_only_then", True),
                            ("template_not_scheduled_at_startup", True),
                            ("row_values_not_precomputed", True),
                        ),
                    ))
    return candidates


def _vision_candidates(
    model: ModelSpec, runtime: RuntimeSpec, work: Any | None,
    workload: Any | None = None,
) -> list[PrepareCandidate]:
    vision = model.vision
    if vision is None or int(runtime.media_tokens or 0) <= 0:
        return []
    from .a3_catalog_media import media_geometry
    patch_tokens, grid_h, grid_w, frames = media_geometry(runtime, vision)
    if patch_tokens <= 0:
        return []
    requests = tuple(getattr(workload, "requests", ()) or ())
    media_request = next(
        (request for request in requests
         if int(getattr(request, "media_tokens", 0) or 0) > 0),
        None,
    )
    request_id = str(getattr(media_request, "request_id", "")) if media_request else ""
    positions = max(1, vision.pos_embed_tokens or patch_tokens)
    elements = positions * vision.hidden_size
    payload = float(elements * dtype_bytes("bf16"))
    return [PrepareCandidate(
        candidate_id="prepare.vision_pos_table",
        kind="vision_position_table", phase="prepare",
        known_at="request_arrival", lifetime="request_resident",
        owner="stage0_rank_local", owner_stage=0, owner_rank="rank_local",
        request_id=request_id,
        arrival_ms=float(getattr(media_request, "arrival_ms", 0.0)) if media_request else 0.0,
        iteration=0,
        consumer_selector="vision_patch_pos",
        consumer_op_ids=_consumers(work, families=("vision",),
                                   request_id=request_id,
                                   op_suffixes=("vision.patch_embed", "vision.pos_embed")),
        tensor_shape=(positions, vision.hidden_size), bytes=payload,
        operations=float(elements),
        resources=(
            ("hbm_bytes", payload), ("mte2_bytes", payload),
            ("mte3_bytes", payload), ("launch", 1.0),
        ),
        capacity_bytes=payload, read_bytes=payload, write_bytes=payload,
        service_bytes=payload, reusable=False,
        cost_once_key=f"{model.name}:vision_pos:{positions}:{patch_tokens}",
        metadata=(("patch_tokens", patch_tokens), ("grid", (grid_h, grid_w, frames)),
                  ("processed_grid_known", grid_h > 0),
                  ("raw_processor_semantics_gap", grid_h == 0)),
    )]


def _workload_work(
    model: ModelSpec, runtime: RuntimeSpec, parallel: ParallelSpec,
    workload: Any,
) -> Any | None:
    """Derive the real prefill lowering when a caller has no iteration work."""
    from .a3_graph_lowering import infer_catalog_iteration
    from .a3_workload import IterationPlan, RequestView

    requests = tuple(getattr(workload, "requests", ()) or ())
    if not requests:
        return None
    views = tuple(
        RequestView(
            request_id=str(request.request_id),
            q_len=max(1, int(request.prompt_tokens)), context=0,
            samples=False, phase="prefill", semantic_phase="prefill",
        )
        for request in requests
    )
    return infer_catalog_iteration(IterationPlan(views), model, runtime, parallel)


def catalog_prepare_candidates(
    model: ModelSpec, runtime: RuntimeSpec, parallel: ParallelSpec,
    *, work: Any | None = None, workload: Any | None = None,
) -> tuple[PrepareCandidate, ...]:
    """Return explicit prepare-eligible candidates; no invented timing."""
    if model.quant is None:
        return ()
    parallel.validate(model)
    if work is None and workload is not None:
        work = _workload_work(model, runtime, parallel, workload)
    candidates: list[PrepareCandidate] = []
    candidates.extend(_scale_candidates(model, parallel, work))
    candidates.extend(_kda_conv_candidates(model, parallel, work))
    candidates.extend(_compressor_buffer_candidates(model, parallel, work, workload))
    candidates.extend(_engram_candidates(model, runtime, work, workload, parallel))
    candidates.extend(_vision_candidates(model, runtime, work, workload))
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


__all__ = ["PrepareCandidate", "catalog_prepare_candidates"]
