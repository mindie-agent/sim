"""Explicit MoE caller TP token slicing and reconstruction.

[Role] Keep replicated router/shared work separate from local source token rows.
[Invariants] A TP view neither copies bytes nor changes global expert counts.
  Selection uses allocated local rows; masked packing uses only valid rows.
  Explicit MC2 padding occurs after router/shared work; finalize gathers once
  and trims its allocation back to the original input rows.
  Caller ownership is independent of the selected transport implementation.
  The ALLTOALL caller currently requires divisible input without padding.
  This contract is not sequence parallelism or ALLGATHER dispatch.
[Timeouts & Locks] Constant-sized graph descriptions; no I/O, locks or histories.
[Side-Effect Policy] Immutable caller metadata and operator descriptions only.
"""
from dataclasses import dataclass, replace
import math
from typing import Literal

from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.catalog.ops import barrier_op, comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.runtime_features import split_routing_features

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/ops/fused_moe/prepare_finalize.py",
        source_sha256="88560c839172f5b433753c796d17a5d41278204f33b2e9ca9d4b2af9ccbe16d8",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/ops/fused_moe/routed_experts.py",
        source_sha256="c5d0bce5c8ef7ceebc0711d45d919a21dca7ea5c3a36822a60977a8120a67006",
    ),
)

KIND = "vllm_ascend_mc2_tp_slice"
ALLTOALL_KIND = "vllm_ascend_alltoall_tp_slice"
_PAD_V3_PADDING_SHAPE = (4,)  # One before/after pair for each of the two input axes.
_UNIFORM_SHARE_TOL = 1e-12
_LOGITS_DTYPE_ALIASES = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "path": "vllm_ascend/ops/fused_moe/prepare_finalize.py",
    "prepare_lines": (291, 306), "finalize_lines": (196, 228, 316, 325),
    "routing_path": "vllm_ascend/ops/fused_moe/routed_experts.py",
    "routing_lines": (656, 675, 713, 717),
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
ALLTOALL_SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "vllm_ascend/ops/fused_moe/prepare_finalize.py",
    "sha256": "88560c839172f5b433753c796d17a5d41278204f33b2e9ca9d4b2af9ccbe16d8",
    "symbol": "PrepareAndFinalizeWithAll2All",
    "prepare_lines": (158, 180), "finalize_lines": (210, 225),
    "scope": "TP-divisible input; independent_source_semantics; not_capture_binary_identity",
}
PAD_V3_SOURCE = {
    "repository": ALLTOALL_SOURCE["repository"],
    "revision": ALLTOALL_SOURCE["revision"],
    "path": ALLTOALL_SOURCE["path"],
    "sha256": ALLTOALL_SOURCE["sha256"],
    "symbol": "PrepareAndFinalizeWithMC2.prepare",
    "prepare_lines": (279, 294),
    "scope": "functional constant-zero padding; not_capture_binary_identity",
}


@dataclass(frozen=True)
class TokenPartition:
    full_rows: int
    local_rows: int
    tp: int
    gather_output_materialization: str = "view"
    padding: str = "none"
    kind: str = KIND
    padding_implementation: str = "concat"
    ep: int = 0
    logits_dtype: str = ""
    activation_tensor_rows: int = 0
    activation_owners: int = 0
    producer_logits_dtype: str = ""
    destination_rank_shares: tuple[float, ...] = ()

    @property
    def caller(self):
        return "mc2" if self.kind == KIND else "alltoall"

    @property
    def source(self):
        return SOURCE if self.kind == KIND else ALLTOALL_SOURCE

    @property
    def allocated_rows(self):
        return self.local_rows * self.tp

    @property
    def valid_per_rank(self):
        return tuple(max(0, min(self.local_rows, self.full_rows - r * self.local_rows))
                     for r in range(self.tp))

    def metadata(self):
        return (("moe_token_partition", {
            "kind": self.kind, "full_rows": self.full_rows,
            "local_rows": self.local_rows, "source_local_rows": self.local_rows, "tp": self.tp,
            "routing_realization_token_scope": "replica_before_caller_tp_slice",
            "source_partitions": 1, "padding": self.padding,
            "allocated_rows": self.allocated_rows, "valid_rows": self.full_rows,
            **({"padding_implementation": self.padding_implementation}
               if self.padding_implementation != "concat" else {}),
            "valid_rows_per_rank": self.valid_per_rank,
            "gather_output_materialization": self.gather_output_materialization,
            "source": self.source,
            **({"activation_tensor_rows": self.activation_tensor_rows}
               if self.activation_tensor_rows else {}),
            **({"activation_owners": self.activation_owners}
               if self.activation_owners else {}),
        }),)

    def prepare(self, *, prefix, input_id, logits_id, width, experts,
                stage, layer, requests, phase, dtype, logits_dtype):
        # Zero-copy TP views inherit the producer tensor's output format.
        # Padded paths emit an explicit cast op whenever target logits_dtype
        # differs from the producer's output format instead of silently retyping.
        producer_dt = _canonical_logits_dtype(self.producer_logits_dtype)
        target_logits_dtype = self.logits_dtype or logits_dtype
        target_dt = _canonical_logits_dtype(target_logits_dtype)
        if self.allocated_rows == self.full_rows:
            resolved_logits_dtype = self.producer_logits_dtype or logits_dtype
        else:
            resolved_logits_dtype = target_logits_dtype
        prepared = []
        if (
            self.allocated_rows != self.full_rows
            and producer_dt
            and target_dt
            and producer_dt != target_dt
        ):
            cast_id = prefix + f".{self.caller}_logits_cast"
            prepared.append(vector_op(
                op_id=cast_id, deps=(logits_id,), stage=stage, layer=layer,
                requests=requests, phase=phase, family="cast",
                recipe=f"{self.caller}_logits_cast", dtype=resolved_logits_dtype,
                elements=self.full_rows * experts, ops_per_element=1,
                shape=(self.full_rows, experts),
                read_bytes_override=self.full_rows * experts * dtype_bytes(producer_dt),
                write_bytes_override=self.full_rows * experts * dtype_bytes(target_dt),
                metadata=self.metadata() + (
                    ("input_format", producer_dt),
                    ("output_format", target_dt),
                    ("invocation_class", "vector"),
                ),
            ))
            logits_id = cast_id
        def view(suffix, dependency, columns, element_dtype):
            if self.allocated_rows != self.full_rows:
                # Source revisions use either a persistent zero tensor or
                # functional padding. Only concatenation reads the zero rows;
                # functional padding zero-initializes the full allocation and
                # copies the valid slice inside one caller operator boundary.
                pad_id = prefix + suffix.replace("_view", "_pad")
                payload = self.allocated_rows * columns * dtype_bytes(element_dtype)
                functional = self.padding_implementation == "pad_v3"
                source_bytes = self.full_rows * columns * dtype_bytes(element_dtype)
                prepared.append(vector_op(
                    op_id=pad_id, deps=(dependency,), stage=stage, layer=layer,
                    requests=requests, phase=phase, family="tensor_materialization",
                    recipe="tensor_pad" if functional else "tensor_concat", dtype=element_dtype,
                    elements=self.allocated_rows * columns,
                    ops_per_element=0,
                    shape=(self.allocated_rows, columns),
                    read_bytes_override=source_bytes if functional else payload,
                    write_bytes_override=payload,
                    metadata=self.metadata() + (("native_kind", "PadV3" if functional else "ConcatD"),
                        ("input_shapes", (((self.full_rows, columns), _PAD_V3_PADDING_SHAPE, ()) if functional
                            else ((self.full_rows, columns),
                                  (self.allocated_rows-self.full_rows, columns)))),
                        ("output_shape", (self.allocated_rows, columns)),
                        ("padding_value", 0),
                        ("padding_source", PAD_V3_SOURCE if functional else SOURCE),
                        ("invocation_class", "vector"))))
                dependency = pad_id
            return replace(barrier_op(
                op_id=prefix + suffix, deps=(dependency,), stage=stage, layer=layer,
                requests=requests, phase=phase, recipe=f"{self.caller}_tp_token_view",
                shape=(self.local_rows, columns),
                metadata=self.metadata() + (("materialization", "view"),
                    ("input_shape", (self.allocated_rows, columns)),
                    ("output_shape", (self.local_rows, columns)),
                    ("physical_payload_bytes", 0.0))), dtype=element_dtype)
        hidden = view(f".{self.caller}_input_view", input_id, width, dtype)
        logits = view(f".{self.caller}_logits_view", logits_id, experts, resolved_logits_dtype)
        return (*prepared, hidden, logits)

    def finalize(self, *, prefix, source, width, stage, layer, requests, phase, dtype):
        local_bytes = self.local_rows * width * dtype_bytes(dtype)
        gathered = replace(comm_op(
            op_id=prefix + f".{self.caller}_tp_gather", deps=(source,), stage=stage, layer=layer,
            requests=requests, phase=phase, family="allgather", recipe="ring_allgather",
            link_bytes=(self.tp - 1) * local_bytes,
            shape=(self.allocated_rows, width),
            metadata=self.metadata() + (("group_axis", "tp"), ("steps", self.tp - 1),
                ("input_shape", (self.local_rows, width)),
                ("output_shape", (self.allocated_rows, width)),
                ("collective_role", f"{self.caller}_finalize_token_reconstruction"))), dtype=dtype)
        if self.gather_output_materialization == "view":
            return self._trim((gathered,), prefix, width, stage, layer, requests, phase, dtype)
        # Some callers materialize each gathered rank slice. The capture-backed
        # choice is explicit; no batch threshold or hidden copy multiplier.
        copies = tuple(vector_op(
            op_id=prefix + f".{self.caller}_gather_copy{rank}", deps=(gathered.op_id,),
            stage=stage, layer=layer, requests=requests, phase=phase, dtype=dtype,
            family="tensor_materialization", recipe="tensor_copy",
            elements=self.local_rows * width, ops_per_element=0,
            shape=(self.local_rows, width), read_bytes_override=local_bytes,
            write_bytes_override=local_bytes,
            metadata=self.metadata() + (("native_kind", "TensorMove"),
                ("native_tensor_layout", "ND"),
                ("gathered_slice", rank), ("invocation_class", "vector")))
            for rank in range(self.tp))
        complete = replace(barrier_op(
            op_id=prefix + f".{self.caller}_gather_complete", deps=tuple(op.op_id for op in copies),
            stage=stage, layer=layer, requests=requests, phase=phase,
            recipe=f"{self.caller}_gather_copies_complete", shape=(self.allocated_rows, width),
            metadata=self.metadata()), dtype=dtype)
        return self._trim((gathered, *copies, complete), prefix, width,
                          stage, layer, requests, phase, dtype)

    def _trim(self, ops, prefix, width, stage, layer, requests, phase, dtype):
        if self.allocated_rows == self.full_rows:
            return ops
        trimmed = replace(barrier_op(
            op_id=prefix + f".{self.caller}_output_trim", deps=(ops[-1].op_id,),
            stage=stage, layer=layer, requests=requests, phase=phase,
            recipe=f"{self.caller}_tp_output_view", shape=(self.full_rows, width),
            metadata=self.metadata() + (("materialization", "view"),
                ("input_shape", (self.allocated_rows, width)),
                ("output_shape", (self.full_rows, width)),
                ("physical_payload_bytes", 0.0))), dtype=dtype)
        return (*ops, trimmed)

    def expected_traffic(self, top_k):
        return MaskedTraffic(self, top_k)


@dataclass(frozen=True)
class MaskedTraffic:
    """Known source mask plus canonical expert-to-rank destination probabilities."""
    partition: TokenPartition
    top_k: int

    def _shares(self, ep: int) -> tuple[float, ...]:
        shares = self.partition.destination_rank_shares
        if len(shares) == ep:
            return shares
        return (1.0 / float(ep),) * ep

    @property
    def basis(self) -> str:
        p = self.partition
        ep = p.ep or p.tp
        shares = self._shares(ep)
        uniform = 1.0 / float(ep)
        if all(abs(share - uniform) <= _UNIFORM_SHARE_TOL for share in shares):
            return "masked_source_uniform_destination_expectation"
        return "masked_source_routed_destination_expectation"

    @property
    def valid_local_tokens(self):
        return self.partition.valid_per_rank[0]

    def payload_matrix(self, row_bytes, *, reverse=False):
        p = self.partition
        ep = p.ep or p.tp
        shares = self._shares(ep)
        return tuple(
            tuple(
                0 if src == dst else (
                    row_bytes * self.top_k
                    * p.valid_per_rank[(dst if reverse else src) % p.tp]
                    * shares[src if reverse else dst]
                )
                for dst in range(ep)
            )
            for src in range(ep)
        )

    def to_dict(self):
        return {"basis": self.basis, "valid_tokens_per_rank": self.partition.valid_per_rank,
                "allocated_tokens_per_rank": self.partition.local_rows,
                "source": SOURCE, "destination_expert_ids": "unobserved"}


def _canonical_logits_dtype(value: str) -> str:
    low = str(value or "").strip().lower()
    return _LOGITS_DTYPE_ALIASES.get(low, low)


def _destination_rank_shares(*, model, plan, parallel, runtime) -> tuple[float, ...]:
    """Derive normalized per-EP-rank destination shares matching grouped compute placement."""
    ep = int(getattr(parallel, "ep", 0) or getattr(parallel, "tp", 0))
    experts = int(getattr(plan, "num_experts", 0) or 0)
    top_k = int(getattr(plan, "top_k", 0) or 0)
    if ep < 1 or experts < 1 or top_k < 1:
        return ()
    from mindie_sim.prediction.operators.moe_transport.moe_callers import RoutingPolicy

    rows_per_token, _ = RoutingPolicy().expert_moments(model, plan, 1)
    total_mass = math.fsum(rows_per_token)
    if total_mass <= 0.0:
        return (1.0 / float(ep),) * ep
    runtime_routing = dict(getattr(runtime, "routing", {}) or {})
    has_slots = (
        int(getattr(parallel, "redundant_experts_per_rank", 0) or 0) > 0
        or "expert_slot_map" in runtime_routing
        or (experts > 0 and ep > 0 and experts % ep != 0)
    )
    rank_shares = [0.0] * ep
    if has_slots:
        _, slot_routing = split_routing_features(runtime_routing)
        slot_plan = expert_slot_plan(model, plan, parallel, routing=slot_routing)
        copies = slot_plan.copies_per_expert()
        for rank, expert_ids in enumerate(slot_plan.expert_ids_by_rank):
            for expert in expert_ids:
                count = copies.get(int(expert), 1)
                rank_shares[rank] += (float(rows_per_token[int(expert)]) / total_mass) / float(max(1, count))
    elif ep <= experts:
        for expert, rows in enumerate(rows_per_token):
            rank_shares[expert % ep] += float(rows) / total_mass
    else:
        for rank in range(ep):
            expert = rank % experts
            replicas = 1 + (ep - 1 - expert) // experts
            rank_shares[rank] += (float(rows_per_token[expert]) / total_mass) / float(replicas)
    return tuple(rank_shares)



def select(runtime, *, parallel, plan, model, tokens, sequence_plan=None,
           reduce_after_join=False):
    kernels = getattr(runtime, "kernels", {}) or {}
    raw = kernels.get("moe_token_partition")
    if raw is None:
        return None
    if (not isinstance(raw, dict) or raw.get("kind") not in {KIND, ALLTOALL_KIND}
            or set(raw) - {"kind", "gather_output_materialization", "padding", "layer_scopes",
                           "padding_implementation", "logits_dtype",
                           "activation_tensor_rows", "activation_owners"}):
        raise ModelingError("moe_token_partition_requires_explicit_vllm_ascend_tp_slice")
    scopes = raw.get("layer_scopes")
    if scopes is not None:
        if (not isinstance(scopes, (list, tuple)) or not scopes
                or any(scope not in {"body", "draft"} for scope in scopes)):
            raise ModelingError("moe_token_partition_layer_scopes_require_body_or_draft")
        if plan.scope not in scopes:
            return None
    materialization = raw.get("gather_output_materialization", "view")
    if materialization not in {"view", "per_rank_copy"}:
        raise ModelingError("mc2_gather_output_materialization_requires_view_or_per_rank_copy")
    if parallel.tp < 2:
        # TP1 has no TP reconstruction; ordinary/observed code emits no gather.
        return None
    if (parallel.dp != 1 or parallel.ep != parallel.tp
            or parallel.etp != 1 or parallel.cp != 1
            or sequence_plan is not None or reduce_after_join
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        raise ModelingError("mc2_tp_slice_requires_dp1_ep_equals_tp_without_sequence_parallel")
    if (plan.expert_latent_size not in (0, None, model.hidden_size)
            or (raw["kind"] == KIND and kernels.get("moe_transport"))):
        raise ModelingError("mc2_tp_slice_requires_direct_hidden_rows_without_separate_transport")
    padding = raw.get("padding", "none")
    padding_implementation = raw.get("padding_implementation", "concat")
    if padding_implementation not in {"concat", "pad_v3"}:
        raise ModelingError("mc2_padding_implementation_requires_concat_or_pad_v3")
    if padding_implementation == "pad_v3" and (raw["kind"] != KIND or padding != "right_to_tp"):
        raise ModelingError("pad_v3_requires_mc2_right_to_tp_padding")
    if raw["kind"] == ALLTOALL_KIND and padding != "none":
        raise ModelingError("alltoall_tp_slice_requires_divisible_rows_without_padding")
    if padding not in {"none", "right_to_tp"}:
        raise ModelingError("mc2_tp_slice_padding_requires_none_or_right_to_tp")
    logits_dtype = raw.get("logits_dtype", "")
    if logits_dtype not in {"", "bf16", "float32"}:
        raise ModelingError("mc2_tp_slice_logits_dtype_requires_bf16_or_float32")
    activation_tensor_rows = raw.get("activation_tensor_rows", 0)
    activation_owners = raw.get("activation_owners", 0)
    for name, value in (("activation_tensor_rows", activation_tensor_rows),
                        ("activation_owners", activation_owners)):
        if value not in (0, None) and (not isinstance(value, int)
                                       or isinstance(value, bool) or value < 1):
            raise ModelingError(f"mc2_tp_slice_{name}_requires_positive_integer")
    if tokens <= 0 or (tokens % parallel.tp and padding == "none"):
        raise ModelingError("mc2_tp_slice_requires_positive_tp_divisible_physical_rows")
    local = (tokens + parallel.tp - 1) // parallel.tp
    producer_logits_dtype = _canonical_logits_dtype(
        getattr(plan, "router_output_dtype", "")
        or getattr(plan, "router_dtype", "")
        or getattr(model, "activation_dtype", "")
    )
    destination_rank_shares = _destination_rank_shares(
        model=model, plan=plan, parallel=parallel, runtime=runtime,
    )
    return TokenPartition(tokens, local, parallel.tp, materialization, padding, raw["kind"],
                          padding_implementation, ep=parallel.ep,
                          logits_dtype=logits_dtype,
                          activation_tensor_rows=int(activation_tensor_rows or 0),
                          activation_owners=int(activation_owners or 0),
                          producer_logits_dtype=producer_logits_dtype,
                          destination_rank_shares=destination_rank_shares)
