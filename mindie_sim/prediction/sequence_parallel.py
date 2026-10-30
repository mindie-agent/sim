"""Kimi K3 AttnRes sequence-parallel planning and communication boundaries.

[Role] Validate the source-selected sequence-parallel declaration, retain the
  local/global token split, and emit ordinary allgather, reduce-scatter and
  view records for the existing graph/engine path.
[Invariants] A plan is immutable and is never rescaled after graph construction.
  The explicit source branch keeps valid query rows separate from physical
  graph/SP padding. Allgather output is sliced to the incoming positions rows
  before attention; reduce-scatter pads its output back to TP. AllToAll forced
  routing executes padded rows. Expert counts stay cohort-global, not /TP.
[Timeouts & Locks] Pure bounded validation and record construction; no I/O.
[Side-Effect Policy] No engine, topology, runtime or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from mindie_sim.catalog.ops import barrier_op, comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec


__all__ = [
    "DECLARATION", "IMPLEMENTATION", "SOURCE_METADATA", "SequenceParallelPlan",
    "allgather", "attention_input", "close_output", "input_shard", "output_trim",
    "reduce_scatter", "select",
]

IMPLEMENTATION = "vllm_ascend_kimi_sequence_parallel"

#: Exact selected source declaration. Unknown or unsupported options are errors,
#: not silently ignored alternatives.
DECLARATION = MappingProxyType({
    "implementation": IMPLEMENTATION,
    "enable_expert_parallel": True,
    "padding": "none",
    "attention_output": "separate_reduce_scatter",
    "shard_materialization": "view",
    "shared_expert_layout": "tp_sharded",
})

#: Provenance is mechanism-level and contains no occurrence identity.
SOURCE_METADATA = (
    ("source_implementation", IMPLEMENTATION),
    ("source_file", "/vllm-workspace/vllm-ascend/vllm_ascend/models/kimi_k3.py"),
    ("source_sha256", "be9adc792774f919f443caf483eb2e04e4453617465381f47a482dd4736ec63f"),
    ("source_image_id", "sha256:883670fcc91cbd75a7070b00501be0ecd81a266b81366a21bf02139d8e6a33e1"),
    ("source_selection_lines", "877-880"),
    ("source_boundary_lines", "239-260,686-694,786-794,1010-1036,1089-1130"),
)

PADDING_SOURCE_METADATA = (
    ("attention_row_source", {
        "repository": "vllm-project/vllm-ascend",
        "path": "vllm_ascend/models/kimi_k3.py",
        "sha256": "c069c65685a8f40ad97624e23554c1c9235abf31bbc36f39b908e5074f23ce73",
        "lines": "499-507",
        "scope": "allgather_slice_to_positions_then_attention_then_sp_reduce_scatter",
        "capture_binary_identity": "not_established; retained_local_source",
    }),
    ("padding_helper_source", {
        "repository": "vllm-project/vllm",
        "revision": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "path": "vllm/models/common/ops/sequence_parallel.py",
        "sha256": "699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd",
        "capture_helper_identity": "not_established; independent_local_source",
    }),
    ("padded_router_source", {
        "repository": "vllm-project/vllm-ascend",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
        "sha256": "86d316b1daa1b5cd633b9e74717b883eb562c4318c047a9e7110d620accee06c",
        "symbol": "TokenDispatcherWithAll2AllV._preprocess",
        "scope": "forced_IDs_all_rows; no_MC2_mask; capture_binary_not_established",
    }),
)

_ASSUMPTIONS = (
    "source branch: PP==1 and enable_expert_parallel and TP>1; no prefill/decode conditional",
    "no padding: tokens equal sum(q_len), nonempty q_len positive, aggregate tokens divisible by TP",
    "an idle DP replica may have no local rows while its EP cohort still has expert work",
    "the incoming sequence-parallel tensor is already TP-sharded; source_partitions=1",
    "the input shard is a declared view; no pinned primitive sp_shard payload is modeled",
    "attention consumes incoming positions rows after allgather slice; reduce-scatter repads to TP",
    "dense MLP gathers local rows, runs full-token TP matrices, then reduce-scatters",
    "MoE router and latent transforms consume local rows; expert routing uses cohort_tokens",
    "MoE expert counts remain cohort-global and are not divided by TP",
    "AttnRes and normalization consume local rows",
    "the final/source-stage close-output allgather supplies full rows; no second allreduce plus reduce-scatter",
    "shared TP-sharded experts explicitly gather full rows and reduce-scatter local outputs; factory dispatch is not inferred",
)


@dataclass(frozen=True)
class SequenceParallelPlan:
    """One validated source branch for this DP replica's iteration rows."""

    implementation: str
    tokens_global: int
    tokens_local: int
    cohort_tokens: int
    source_partitions: int
    tp: int
    ep: int
    dp: int
    phases: tuple[str, ...]
    source: tuple[tuple[str, Any], ...]
    assumptions: tuple[str, ...]
    padding: str = "none"
    input_rows: int = 0
    valid_global: int = 0
    valid_per_tp: tuple[int, ...] = ()

    @property
    def physical_global(self):
        return self.tokens_global

    def ffn_tokens(self, layer) -> int:
        """Dense TP matrices need full rows; routed MoE consumes local rows."""
        if layer.ffn == "dense":
            return self.tokens_global
        if layer.ffn == "moe":
            return self.tokens_local
        raise ModelingError(f"sequence_parallel_ffn_kind_unsupported:{layer.ffn}")

    @property
    def attention_tokens(self) -> int:
        # positions.shape[0] belongs to the incoming graph tensor. SP padding
        # does not expand it; any separately declared graph bucket still does.
        return self.input_rows

    @property
    def attn_res_tokens(self) -> int:
        return self.tokens_local

    @property
    def normalize_tokens(self) -> int:
        return self.tokens_local

    @property
    def router_tokens(self) -> int:
        return self.tokens_local

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "padding": self.padding,
            "input_rows": self.input_rows,
            "valid_global": self.valid_global,
            "physical_global": self.physical_global,
            "attention_tokens": self.attention_tokens,
            "valid_per_tp": list(self.valid_per_tp),
            "router_padding_policy": "execute_padded_rows",
            "tokens_global": self.tokens_global,
            "tokens_local": self.tokens_local,
            "cohort_tokens": self.cohort_tokens,
            "source_partitions": self.source_partitions,
            "tp": self.tp,
            "ep": self.ep,
            "dp": self.dp,
            "phases": list(self.phases),
            "source": dict(self.source),
            "assumptions": list(self.assumptions),
        }


def _kernels(runtime) -> Mapping:
    if isinstance(runtime, Mapping):
        raw = runtime.get("kernels", {})
    else:
        raw = getattr(runtime, "kernels", {})
    if not isinstance(raw, Mapping):
        raise ModelingError("sequence_parallel_runtime_kernels_invalid")
    return raw


def _positive_int(value, reason: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingError(f"{reason}:{value!r}")
    return value


def _parallel(runtime):
    if not isinstance(runtime, Mapping):
        return runtime.parallel
    raw = runtime.get("parallel")
    if not isinstance(raw, Mapping):
        raise ModelingError("sequence_parallel_runtime_parallel_invalid")
    return ParallelSpec.from_mapping(raw)


def _tuple_names(value, reason: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ModelingError(f"{reason}:{value!r}")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ModelingError(f"{reason}:{value!r}")
    return result


def _validate_declaration(raw: Any) -> Mapping:
    if not isinstance(raw, Mapping):
        raise ModelingError("sequence_parallel_declaration_must_be_mapping")
    unknown = set(raw) - set(DECLARATION)
    if unknown:
        raise ModelingError("sequence_parallel_declaration_unknown:" + ",".join(sorted(unknown)))
    missing = set(DECLARATION) - set(raw)
    if missing:
        raise ModelingError("sequence_parallel_declaration_missing:" + ",".join(sorted(missing)))
    for key, expected in DECLARATION.items():
        actual = raw[key]
        if key == "enable_expert_parallel":
            if actual is not True:
                raise ModelingError(
                    f"sequence_parallel_option_unsupported:{key}={actual!r}")
        elif key == "padding" and actual in {"none", "right_to_tp"}:
            continue
        elif actual != expected:
            raise ModelingError(
                f"sequence_parallel_option_unsupported:{key}={actual!r}")
    return raw


def _validate_views(views, tokens, *, padding="none") -> tuple[Any, ...]:
    try:
        values = tuple(views)
    except TypeError as exc:
        raise ModelingError("sequence_parallel_views_must_be_iterable") from exc
    for view in values:
        q_len = getattr(view, "q_len", None)
        if not isinstance(q_len, int) or isinstance(q_len, bool) or q_len <= 0:
            raise ModelingError(f"sequence_parallel_requires_positive_q_len:{q_len!r}")
        if getattr(view, "padding", False):
            raise ModelingError("sequence_parallel_no_padding_domain")
        for field in ("draft_proposal_tokens", "draft_accepted_tokens"):
            if getattr(view, field, 0):
                raise ModelingError(
                    f"sequence_parallel_speculative_field_unsupported:{field}")
        # The ordinary decode scheduler records one committed token here even
        # when no draft exists. Only a multi-token advance needs speculative ABI.
        if getattr(view, "draft_advanced_tokens", 0) not in (0, 1):
            raise ModelingError(
                "sequence_parallel_speculative_field_unsupported:draft_advanced_tokens")
    total = sum(view.q_len for view in values)
    if tokens < total or (padding == "none" and tokens != total):
        raise ModelingError(
            f"sequence_parallel_tokens_must_equal_sum_q_len:{tokens}!={total}")
    return values


def _validate_model(model) -> None:
    plans = tuple(getattr(model, "layer_plans", ()))
    if not plans:
        raise ModelingError("sequence_parallel_requires_attn_residual_layer_plans")
    if any(getattr(layer, "scope", "body") != "body" for layer in plans):
        raise ModelingError("sequence_parallel_requires_body_scope")
    if any(int(getattr(layer, "residual_block_size", 0)) <= 0 for layer in plans):
        raise ModelingError("sequence_parallel_requires_source_attn_residual_block_size")
    if any(bool(getattr(layer, "hyper_connections", False))
           or int(getattr(layer, "hc_mult", 0) or 0) > 0 for layer in plans):
        raise ModelingError("sequence_parallel_hyperconnections_unsupported")


def select(runtime, model, views, *, tokens, global_tokens=None):
    """Return the immutable plan, or ``None`` when no declaration is present.

    ``none`` retains the divisible-input contract. ``right_to_tp`` pads the
    physical input, including prior graph padding, while preserving valid
    views. Cross-DP totals must already include each replica's physical rows.
    """
    kernels = _kernels(runtime)
    if "sequence_parallel" not in kernels:
        return None
    declaration = _validate_declaration(kernels["sequence_parallel"])
    padding = declaration["padding"]
    if type(tokens) is not int or tokens < 0:
        raise ModelingError(f"sequence_parallel_requires_nonnegative_tokens:{tokens!r}")
    values = _validate_views(views, tokens, padding=padding)
    valid = sum(view.q_len for view in values)
    input_rows = tokens
    parallel = _parallel(runtime)
    pp, cp, pcp = int(parallel.pp), int(parallel.cp), int(parallel.pcp)
    tp, ep, dp = int(parallel.tp), int(parallel.ep), int(parallel.dp)
    if pp != 1:
        raise ModelingError(f"sequence_parallel_requires_pp1:{pp}")
    if cp != 1:
        raise ModelingError(f"sequence_parallel_requires_cp1:{cp}")
    if pcp != 1:
        raise ModelingError(f"sequence_parallel_requires_pcp1:{pcp}")
    if tp <= 1:
        raise ModelingError(f"sequence_parallel_requires_tp_gt_1:{tp}")
    if ep < tp or ep % tp:
        raise ModelingError(f"sequence_parallel_requires_ep_ge_tp_and_divisible:{ep}:{tp}")
    _validate_model(model)
    if padding == "right_to_tp":
        tokens += (-tokens) % tp
    if tokens % tp:
        raise ModelingError(
            f"sequence_parallel_requires_tokens_divisible_by_tp:{tokens}%{tp}")
    local = tokens // tp
    if ep > tp:
        if global_tokens is None:
            raise ModelingError("sequence_parallel_cross_dp_requires_complete_cohort_tokens")
        cohort = _positive_int(
            global_tokens, "sequence_parallel_global_tokens_invalid")
        if cohort < tokens:
            raise ModelingError(
                f"sequence_parallel_cohort_tokens_below_local_full:{cohort}<{tokens}")
        if cohort % tp:
            raise ModelingError(
                f"sequence_parallel_cohort_tokens_not_divisible_by_tp:{cohort}%{tp}")
    else:
        cohort = tokens
        if not cohort:
            raise ModelingError("sequence_parallel_requires_nonempty_ep_cohort")
        if global_tokens is not None and global_tokens != tokens:
            raise ModelingError(
                f"sequence_parallel_global_tokens_outside_ep_cohort:"
                f"{global_tokens}!={tokens}")
    phases = tuple(sorted({str(getattr(view, "phase", "")) for view in values}))
    return SequenceParallelPlan(
        implementation=IMPLEMENTATION,
        tokens_global=tokens,
        tokens_local=local,
        cohort_tokens=cohort,
        source_partitions=1,
        tp=tp,
        ep=ep,
        dp=dp,
        phases=phases,
        source=SOURCE_METADATA + (PADDING_SOURCE_METADATA if padding == "right_to_tp" else ()),
        assumptions=(_ASSUMPTIONS if padding == "none" else tuple(
            item for item in _ASSUMPTIONS
            if not item.startswith(("no padding:", "the input shard"))) + (
                "right-pad the incoming physical tensor to TP; shard is a view of that padded allocation",
                "AllToAll force-EPLB executes padded router rows; they are not valid request queries",
                "valid_per_tp is diagnostic and does not mask expert dispatch",
                "SP helper padding is local-source verified; captured helper binary identity is not established",
            )),
        padding=padding, input_rows=input_rows, valid_global=valid,
        valid_per_tp=tuple(max(0, min(local, valid - rank * local)) for rank in range(tp)),
    )


def _shape(plan: SequenceParallelPlan, width: int, *, row_multiplier: int = 1,
           channel_multiplier: int = 1) -> tuple[int, int, int]:
    width = _positive_int(width, "sequence_parallel_width_invalid")
    rows = _positive_int(row_multiplier, "sequence_parallel_row_multiplier_invalid")
    if not isinstance(channel_multiplier, int) or isinstance(channel_multiplier, bool) \
            or channel_multiplier <= 0:
        raise ModelingError(
            f"sequence_parallel_channel_multiplier_invalid:{channel_multiplier!r}")
    global_rows = plan.tokens_global * rows
    if global_rows % plan.tp:
        raise ModelingError(
            f"sequence_parallel_multiplied_rows_not_divisible_by_tp:"
            f"{global_rows}%{plan.tp}")
    return global_rows // plan.tp, global_rows, width * channel_multiplier


def _wire(plan: SequenceParallelPlan, rows: int, width: int, dtype: str) -> float:
    return float(rows * width * dtype_bytes(dtype) * (plan.tp - 1) / plan.tp)


def _common(plan: SequenceParallelPlan, *, role: str, input_shape: tuple[int, ...],
            output_shape: tuple[int, ...], payload: float | None = None,
            collective: str | None = None) -> tuple[tuple[str, Any], ...]:
    items = [
        ("sequence_parallel", plan.implementation),
        ("sequence_parallel_role", role),
        ("source_partitions", plan.source_partitions),
        ("source_assignment_partitions", plan.source_partitions),
        ("padding", plan.padding),
        ("valid_global", plan.valid_global),
        ("physical_global", plan.physical_global),
        ("valid_per_tp", plan.valid_per_tp),
        ("router_padding_policy", "execute_padded_rows"),
        ("input_shape", tuple(input_shape)),
        ("output_shape", tuple(output_shape)),
        ("tokens_global", plan.tokens_global),
        ("tokens_local", plan.tokens_local),
        ("cohort_tokens", plan.cohort_tokens),
    ]
    if collective is not None:
        items.extend((("collective", collective), ("group_axis", "tp"),
                      ("steps", plan.tp - 1)))
    if payload is not None:
        items.append(("wire_payload_bytes", payload))
    items.extend(plan.source)
    return tuple(items)


def input_shard(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
                stage: int, layer: int, requests: tuple[str, ...], phase: str,
                width: int, dtype: str, row_multiplier: int = 1,
                channel_multiplier: int = 1) -> OpSpec:
    """Right-padding materializes once; the subsequent TP slice is a view."""
    local_rows, global_rows, width = _shape(
        plan, width, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier)
    deps = _tuple_names(deps, "sequence_parallel_input_shard_deps_invalid")
    if not deps:
        raise ModelingError("sequence_parallel_input_shard_requires_full_embedding_dependency")
    requests = _tuple_names(requests, "sequence_parallel_requests_invalid")
    metadata = _common(
        plan, role="input_shard",
        input_shape=(global_rows, width), output_shape=(local_rows, width),
        payload=0.0,
    ) + (
        ("shard_axis", "tp"),
        ("materialization", "view"),
        ("shard_materialization", "view"),
        ("input_dependency", "full_embedding"),
        ("physical_payload_bytes", 0.0),
        ("zero_payload_reason", "declared_view_no_padding"),
    )
    if plan.physical_global > plan.input_rows:
        input_rows = plan.input_rows * row_multiplier
        read_bytes = input_rows * width * dtype_bytes(dtype)
        write_bytes = global_rows * width * dtype_bytes(dtype)
        pad_metadata = dict(metadata)
        pad_metadata.update({
            "input_shape": (input_rows, width),
            "materialization": "right_padding_then_shard_view",
            "shard_materialization": "view",
            "physical_payload_bytes": read_bytes + write_bytes,
            "padding_rows": global_rows - input_rows,
            "full_allocation_shape": (global_rows, width),
            "mapping_basis": "analytical_pad_copy_and_zero; native_tiling_not_inferred",
        })
        pad_metadata.pop("zero_payload_reason")
        return vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, dtype=str(dtype), family="pad", recipe="sequence_parallel_right_pad",
            elements=(global_rows - input_rows) * width, ops_per_element=1,
            shape=(local_rows, width), read_bytes_override=read_bytes,
            write_bytes_override=write_bytes, metadata=tuple(pad_metadata.items()),
            notes=("source right-padding copies existing physical rows and zeros only added rows; TP slicing is a view",),
        )
    return replace(
        barrier_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, recipe="sequence_parallel_view",
            shape=(local_rows, width), metadata=metadata,
            notes=("full rows enter as a declared TP view; no copy payload is modeled",),
        ),
        dtype=str(dtype),
    )


def output_trim(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
                stage: int, layer: int, requests: tuple[str, ...], phase: str,
                width: int, dtype: str) -> OpSpec:
    """The caller slices gathered SP padding back to its incoming tensor rows."""
    metadata = _common(
        plan, role="output_trim", input_shape=(plan.physical_global, width),
        output_shape=(plan.input_rows, width), payload=0.0,
    ) + (("materialization", "view"), ("physical_payload_bytes", 0.0))
    return replace(barrier_op(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, recipe="sequence_parallel_output_view",
        shape=(plan.input_rows, width), metadata=metadata,
    ), dtype=str(dtype))


def attention_input(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
                    stage: int, layer: int, requests: tuple[str, ...], phase: str,
                    width: int, dtype: str) -> OpSpec:
    """The source's leading-row slice aliases the gathered allocation."""
    op = output_trim(plan=plan, op_id=op_id, deps=deps, stage=stage, layer=layer,
                     requests=requests, phase=phase, width=width, dtype=dtype)
    metadata = dict(op.metadata)
    metadata.update(sequence_parallel_role="attention_input_view",
                    attention_tokens=plan.attention_tokens,
                    source_expression="hidden_states[:positions.shape[0]]")
    return replace(op, recipe="sequence_parallel_attention_view",
                   metadata=tuple(metadata.items()))


def allgather(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
              stage: int, layer: int, requests: tuple[str, ...], phase: str,
              width: int, dtype: str, row_multiplier: int = 1,
              channel_multiplier: int = 1) -> OpSpec:
    """Emit an ordinary TP allgather from local rows to full rows."""
    if not plan.tokens_local:
        raise ModelingError("sequence_parallel_empty_replica_has_no_tp_collective")
    local_rows, global_rows, width = _shape(
        plan, width, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier)
    deps = _tuple_names(deps, "sequence_parallel_allgather_deps_invalid")
    requests = _tuple_names(requests, "sequence_parallel_requests_invalid")
    payload = _wire(plan, global_rows, width, dtype)
    metadata = _common(
        plan, role="allgather", input_shape=(local_rows, width),
        output_shape=(global_rows, width), payload=payload,
        collective="allgather",
    )
    return replace(
        comm_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            family="allgather", phase=phase, recipe="ring_allgather",
            link_bytes=payload, shape=(global_rows, width), metadata=metadata,
            notes=("sequence-parallel gather: local rows -> full rows",),
        ),
        dtype=str(dtype),
    )


def reduce_scatter(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
                   stage: int, layer: int, requests: tuple[str, ...], phase: str,
                   width: int, dtype: str, row_multiplier: int = 1,
                   channel_multiplier: int = 1) -> OpSpec:
    """Emit an ordinary TP reduce-scatter from full rows to local rows."""
    if not plan.tokens_local:
        raise ModelingError("sequence_parallel_empty_replica_has_no_tp_collective")
    local_rows, global_rows, width = _shape(
        plan, width, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier)
    deps = _tuple_names(deps, "sequence_parallel_reduce_scatter_deps_invalid")
    requests = _tuple_names(requests, "sequence_parallel_requests_invalid")
    payload = _wire(plan, global_rows, width, dtype)
    metadata = _common(
        plan, role="reduce_scatter", input_shape=(global_rows, width),
        output_shape=(local_rows, width), payload=payload,
        collective="reduce_scatter",
    )
    return replace(
        comm_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            family="reduce_scatter", phase=phase, recipe="ring_reduce_scatter",
            link_bytes=payload, shape=(local_rows, width), metadata=metadata,
            notes=("sequence-parallel reduction: full rows -> local rows",),
        ),
        dtype=str(dtype),
    )


def close_output(ops, last, *, plan: SequenceParallelPlan, width: int, dtype: str,
                 op_id: str | None = None, row_multiplier: int = 1,
                 channel_multiplier: int = 1,
                 input_rows: int | None = None) -> tuple[tuple[OpSpec, ...], str]:
    """Replace only the exact current tail TP allreduce with reduce-scatter.

    The tail must be the last record, an ``allreduce`` on ``group_axis=tp``,
    with its sole producer dependency. No operator-id prefix is inspected.
    """
    ops = tuple(ops)
    if not ops or ops[-1].op_id != last:
        raise ModelingError("sequence_parallel_close_output_requires_exact_tail")
    tail = ops[-1]
    metadata = dict(tail.metadata)
    if tail.family != "allreduce" or metadata.get("group_axis") != "tp":
        raise ModelingError("sequence_parallel_close_output_requires_tp_allreduce_tail")
    if len(tail.deps) != 1:
        raise ModelingError("sequence_parallel_close_output_requires_sole_producer")
    replacement_id = op_id or tail.op_id + ".reduce_scatter"
    if any(op.op_id == replacement_id for op in ops[:-1]):
        raise ModelingError(f"sequence_parallel_close_output_duplicate_id:{replacement_id}")
    prefix, dependencies = ops[:-1], tail.deps
    if input_rows is not None:
        _, global_rows, physical_width = _shape(
            plan, width, row_multiplier=row_multiplier,
            channel_multiplier=channel_multiplier)
        source_rows = input_rows * row_multiplier
        if source_rows != global_rows:
            # sp_reduce_scatter materializes F.pad before either custom or
            # ordinary collective; it copies live rows and zeroes the tail.
            if source_rows < 0 or source_rows > global_rows:
                raise ModelingError("sequence_parallel_reduce_scatter_rows_outside_allocation")
            payload_read = source_rows * physical_width * dtype_bytes(dtype)
            payload_write = global_rows * physical_width * dtype_bytes(dtype)
            pad = vector_op(
                op_id=replacement_id + ".pad", deps=dependencies,
                stage=tail.stage, layer=tail.layer, requests=tail.requests,
                phase=tail.phase, dtype=str(dtype), family="pad",
                recipe="sequence_parallel_reduce_scatter_pad",
                shape=(global_rows, physical_width),
                elements=(global_rows - source_rows) * physical_width, ops_per_element=1,
                read_bytes_override=payload_read, write_bytes_override=payload_write,
                metadata=_common(
                    plan, role="reduce_scatter_pad", input_shape=(source_rows, physical_width),
                    output_shape=(global_rows, physical_width), payload=0.0) + (
                        ("materialization", "right_padding"),
                        ("padding_rows", global_rows - source_rows),
                        ("physical_payload_bytes", payload_read + payload_write),
                        ("mapping_basis", "analytical_F_pad_copy_and_zero")),
            )
            prefix += (replace(pad, stream=tail.stream),)
            dependencies = (pad.op_id,)
    replacement = reduce_scatter(
        plan=plan, op_id=replacement_id, deps=dependencies, stage=tail.stage,
        layer=tail.layer, requests=tail.requests, phase=tail.phase,
        width=width, dtype=dtype, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier,
    )
    return prefix + (replace(replacement, stream=tail.stream),), replacement_id
