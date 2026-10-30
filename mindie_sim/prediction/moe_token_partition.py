"""MC2 caller TP token slicing and reconstruction.

[Role] Keep replicated router/shared work separate from MC2 source token rows.
[Invariants] A TP view neither copies bytes nor changes global expert counts.
  Selection uses allocated local rows; masked packing uses only valid rows.
  Explicit MC2 padding occurs after router/shared work; finalize gathers once
  and trims its allocation back to the original input rows.
  Caller ownership is independent of the selected transport implementation.
  This contract is not sequence parallelism or ALLGATHER dispatch.
[Timeouts & Locks] Constant-sized graph descriptions; no I/O, locks or histories.
[Side-Effect Policy] Immutable caller metadata and operator descriptions only.
"""
from dataclasses import dataclass, replace

from mindie_sim.catalog.ops import barrier_op, comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes

KIND = "vllm_ascend_mc2_tp_slice"
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "path": "vllm_ascend/ops/fused_moe/prepare_finalize.py",
    "prepare_lines": (291, 306), "finalize_lines": (196, 228, 316, 325),
    "routing_path": "vllm_ascend/ops/fused_moe/routed_experts.py",
    "routing_lines": (656, 675, 713, 717),
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


@dataclass(frozen=True)
class TokenPartition:
    full_rows: int
    local_rows: int
    tp: int
    gather_output_materialization: str = "view"
    padding: str = "none"

    @property
    def allocated_rows(self):
        return self.local_rows * self.tp

    @property
    def valid_per_rank(self):
        return tuple(max(0, min(self.local_rows, self.full_rows - r * self.local_rows))
                     for r in range(self.tp))

    def metadata(self):
        return (("moe_token_partition", {
            "kind": KIND, "full_rows": self.full_rows,
            "local_rows": self.local_rows, "source_local_rows": self.local_rows, "tp": self.tp,
            "routing_realization_token_scope": "replica_before_caller_tp_slice",
            "source_partitions": 1, "padding": self.padding,
            "allocated_rows": self.allocated_rows, "valid_rows": self.full_rows,
            "valid_rows_per_rank": self.valid_per_rank,
            "gather_output_materialization": self.gather_output_materialization,
            "source": SOURCE,
        }),)

    def prepare(self, *, prefix, input_id, logits_id, width, experts,
                stage, layer, requests, phase, dtype, logits_dtype):
        prepared = []
        def view(suffix, dependency, columns, element_dtype):
            if self.allocated_rows != self.full_rows:
                # The source concatenates a persistent zero tensor. Its read
                # and the full output write are real; zero creation is outside
                # this repeated caller boundary.
                pad_id = prefix + suffix.replace("_view", "_pad")
                payload = self.allocated_rows * columns * dtype_bytes(element_dtype)
                prepared.append(vector_op(
                    op_id=pad_id, deps=(dependency,), stage=stage, layer=layer,
                    requests=requests, phase=phase, family="tensor_materialization",
                    recipe="tensor_concat", dtype=element_dtype,
                    elements=self.allocated_rows * columns, ops_per_element=0,
                    shape=(self.allocated_rows, columns), read_bytes_override=payload,
                    write_bytes_override=payload,
                    metadata=self.metadata() + (("native_kind", "ConcatD"),
                        ("input_shapes", ((self.full_rows, columns),
                                         (self.allocated_rows-self.full_rows, columns))),
                        ("output_shape", (self.allocated_rows, columns)),
                        ("padding_value", 0), ("invocation_class", "vector"))))
                dependency = pad_id
            return replace(barrier_op(
                op_id=prefix + suffix, deps=(dependency,), stage=stage, layer=layer,
                requests=requests, phase=phase, recipe="mc2_tp_token_view",
                shape=(self.local_rows, columns),
                metadata=self.metadata() + (("materialization", "view"),
                    ("input_shape", (self.allocated_rows, columns)),
                    ("output_shape", (self.local_rows, columns)),
                    ("physical_payload_bytes", 0.0))), dtype=element_dtype)
        hidden = view(".mc2_input_view", input_id, width, dtype)
        logits = view(".mc2_logits_view", logits_id, experts, logits_dtype)
        return (*prepared, hidden, logits)

    def finalize(self, *, prefix, source, width, stage, layer, requests, phase, dtype):
        local_bytes = self.local_rows * width * dtype_bytes(dtype)
        gathered = replace(comm_op(
            op_id=prefix + ".mc2_tp_gather", deps=(source,), stage=stage, layer=layer,
            requests=requests, phase=phase, family="allgather", recipe="ring_allgather",
            link_bytes=(self.tp - 1) * local_bytes,
            shape=(self.allocated_rows, width),
            metadata=self.metadata() + (("group_axis", "tp"), ("steps", self.tp - 1),
                ("input_shape", (self.local_rows, width)),
                ("output_shape", (self.allocated_rows, width)),
                ("collective_role", "mc2_finalize_token_reconstruction"))), dtype=dtype)
        if self.gather_output_materialization == "view":
            return self._trim((gathered,), prefix, width, stage, layer, requests, phase, dtype)
        # Some callers materialize each gathered rank slice. The capture-backed
        # choice is explicit; no batch threshold or hidden copy multiplier.
        copies = tuple(vector_op(
            op_id=prefix + f".mc2_gather_copy{rank}", deps=(gathered.op_id,),
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
            op_id=prefix + ".mc2_gather_complete", deps=tuple(op.op_id for op in copies),
            stage=stage, layer=layer, requests=requests, phase=phase,
            recipe="mc2_gather_copies_complete", shape=(self.allocated_rows, width),
            metadata=self.metadata()), dtype=dtype)
        return self._trim((gathered, *copies, complete), prefix, width,
                          stage, layer, requests, phase, dtype)

    def _trim(self, ops, prefix, width, stage, layer, requests, phase, dtype):
        if self.allocated_rows == self.full_rows:
            return ops
        trimmed = replace(barrier_op(
            op_id=prefix + ".mc2_output_trim", deps=(ops[-1].op_id,),
            stage=stage, layer=layer, requests=requests, phase=phase,
            recipe="mc2_tp_output_view", shape=(self.full_rows, width),
            metadata=self.metadata() + (("materialization", "view"),
                ("input_shape", (self.allocated_rows, width)),
                ("output_shape", (self.full_rows, width)),
                ("physical_payload_bytes", 0.0))), dtype=dtype)
        return (*ops, trimmed)

    def expected_traffic(self, top_k):
        return MaskedTraffic(self, top_k)


@dataclass(frozen=True)
class MaskedTraffic:
    """Known source mask plus declared uniform destinations, not captured IDs."""
    partition: TokenPartition
    top_k: int
    basis = "masked_source_uniform_destination_expectation"

    @property
    def valid_local_tokens(self):
        return self.partition.valid_per_rank[0]

    def payload_matrix(self, row_bytes, *, reverse=False):
        p = self.partition
        return tuple(tuple(0 if src == dst else row_bytes * self.top_k
                           * p.valid_per_rank[dst if reverse else src] / p.tp
                           for dst in range(p.tp)) for src in range(p.tp))

    def to_dict(self):
        return {"basis": self.basis, "valid_tokens_per_rank": self.partition.valid_per_rank,
                "allocated_tokens_per_rank": self.partition.local_rows,
                "source": SOURCE, "destination_expert_ids": "unobserved"}



def select(runtime, *, parallel, plan, model, tokens, sequence_plan=None,
           reduce_after_join=False):
    kernels = getattr(runtime, "kernels", {}) or {}
    raw = kernels.get("moe_token_partition")
    if raw is None:
        return None
    if (not isinstance(raw, dict) or raw.get("kind") != KIND
            or set(raw) - {"kind", "gather_output_materialization", "padding"}):
        raise ModelingError("moe_token_partition_requires_vllm_ascend_mc2_tp_slice")
    materialization = raw.get("gather_output_materialization", "view")
    if materialization not in {"view", "per_rank_copy"}:
        raise ModelingError("mc2_gather_output_materialization_requires_view_or_per_rank_copy")
    if (parallel.dp != 1 or parallel.ep != parallel.tp or parallel.tp < 2
            or parallel.etp != 1 or parallel.cp != 1 or parallel.pcp != 1
            or sequence_plan is not None or reduce_after_join
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        raise ModelingError("mc2_tp_slice_requires_dp1_ep_equals_tp_without_sequence_parallel")
    if (plan.expert_latent_size not in (0, None, model.hidden_size)
            or getattr(runtime, "kernels", {}).get("moe_transport")):
        raise ModelingError("mc2_tp_slice_requires_direct_hidden_rows_without_separate_transport")
    padding = raw.get("padding", "none")
    if padding not in {"none", "right_to_tp"}:
        raise ModelingError("mc2_tp_slice_padding_requires_none_or_right_to_tp")
    if tokens <= 0 or (tokens % parallel.tp and padding == "none"):
        raise ModelingError("mc2_tp_slice_requires_positive_tp_divisible_physical_rows")
    local = (tokens + parallel.tp - 1) // parallel.tp
    return TokenPartition(tokens, local, parallel.tp, materialization, padding)
