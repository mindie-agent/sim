"""AttnRes and sequence-parallel planning and communication boundaries.

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
    "allgather", "attention_input", "close_output", "emit_mc2_mask_pad",
    "emit_reduce_scatter_pad", "input_shard", "mmrs_eligible", "output_trim",
    "reduce_scatter", "select", "sequence_parallel_enabled",
]

IMPLEMENTATION = "vllm_ascend_sequence_parallel"
_VALID_SP_SCOPES = frozenset({"off", "stage_entry_all_layers", "moe_flashcomm1_only"})
_VALID_SP_PAD_PROVIDERS = frozenset({"none", "constant_pad_nd", "cat_zeros"})
_OPTIONAL_DECLARATION_KEYS = frozenset({
    "sp_scope",
    "sp_pad_provider",
    "capture_bucket_rows",
    "mc2_active_mask_active",
    "mmrs_fused",
    "allow_draft_scope",
    "rejects_sp_with_pp",
})
_MMRS_ALLOWED_TP = frozenset({2, 4, 8})
_MMRS_CANONICAL_DTYPES = MappingProxyType({
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
})
_MMRS_ALLOWED_DTYPES = frozenset(_MMRS_CANONICAL_DTYPES)
_MMRS_K_MIN = 256
_MMRS_K_MAX_EXCLUSIVE = 65535
_MMRS_PRODUCER_FAMILIES = frozenset({
    "gemm",
    "projection",
    "output_projection",
    "o_projection",
    "down_projection",
    "gate_up_projection",
    "gate_projection",
    "qkv_projection",
    "index_projection",
    "router",
    "logits",
    "vision_projector",
})
_MC2_MASK_CAST_COUNT = 3
_EAGER_GRAPH_MODES = frozenset({"none", "off", "eager", "disabled"})
_MMRS_INTERNAL_CHUNKS = 1
_MMRS_REQUIRED_DIMS = 3
_MMRS_REQUIRED_RATES = (
    "hbm", "l2", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", "vector",
)
_ARCH22_PRESET = "npu_arch_2201"
_ARCH35_PRESET = "npu_arch_3510"

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
    ("source_repository", "vllm-project/vllm-ascend"),
    ("source_file", "vllm_ascend/models/common/ops/sequence_parallel.py"),
    ("source_manifest", "mindie_sim/stacks/vllm_ascend/manifest.json"),
)

PADDING_SOURCE_METADATA = (
    ("attention_row_source", {
        "repository": "vllm-project/vllm-ascend",
        "path": "vllm_ascend/models/common/ops/sequence_parallel.py",
        "scope": "allgather_slice_to_positions_then_attention_then_sp_reduce_scatter",
    }),
    ("padding_helper_source", {
        "repository": "vllm-project/vllm",
        "revision": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "path": "vllm/models/common/ops/sequence_parallel.py",
        "sha256": "699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd",
    }),
    ("padded_router_source", {
        "repository": "vllm-project/vllm-ascend",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
        "sha256": "86d316b1daa1b5cd633b9e74717b883eb562c4318c047a9e7110d620accee06c",
        "symbol": "TokenDispatcherWithAll2AllV._preprocess",
        "scope": "mc2_mask_gating_or_padded_rows_by_active_plan",
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
    valid_cohort_tokens: int = 0
    sp_scope: str = "stage_entry_all_layers"
    sp_pad_provider: str = "constant_pad_nd"
    capture_bucket_rows: int = 0
    mc2_active_mask_active: bool = False
    mmrs_fused: bool = False
    pp: int = 1
    eager_execution: bool = False
    pcp: int = 1

    @property
    def physical_global(self):
        return self.tokens_global

    @property
    def stage_entry_active(self) -> bool:
        """True only when stage-entry TP shard + full-layer SP is active (not ``moe_flashcomm1_only``)."""
        return self.sp_scope == "stage_entry_all_layers"

    @property
    def applies_to_attention(self) -> bool:
        """True only when attention consumes SP-gathered/scattered activations."""
        return self.sp_scope == "stage_entry_all_layers"

    @property
    def applies_to_dense_ffn(self) -> bool:
        """True only when dense FFN participates in stage-wide sequence parallelism."""
        return self.sp_scope == "stage_entry_all_layers"

    @property
    def applies_to_moe(self) -> bool:
        """True for both stage-wide SP and MoE-only FlashComm1 SP."""
        return self.sp_scope in {"stage_entry_all_layers", "moe_flashcomm1_only"}

    def reduce_scatter_pad_rows(self, global_input_rows: int) -> int:
        """Exact G1 per-occurrence SP padding law for a global-input ``sp_reduce_scatter``."""
        if self.sp_scope == "off" or self.tp <= 1 or global_input_rows <= 0:
            return 0
        rem = int(global_input_rows) % self.tp
        return 0 if rem == 0 else self.tp - rem

    def directional_remote_link_bytes(
        self, collective: str, rows: int, hidden: int, elem_bytes: int,
    ) -> int:
        """Exact directional remote link bytes per rank (strictly 0 when ``tp == 1``)."""
        if self.tp <= 1 or rows <= 0 or hidden <= 0 or elem_bytes <= 0:
            return 0
        total = int(rows) * int(hidden) * int(elem_bytes)
        if collective in {"allgather", "reduce_scatter"}:
            return (total * (self.tp - 1)) // self.tp
        if collective == "allreduce":
            return (2 * total * (self.tp - 1)) // self.tp
        raise ModelingError(f"sequence_parallel_unknown_collective:{collective!r}")

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
            "sp_scope": self.sp_scope,
            "stage_entry_active": self.stage_entry_active,
            "applies_to_attention": self.applies_to_attention,
            "applies_to_dense_ffn": self.applies_to_dense_ffn,
            "applies_to_moe": self.applies_to_moe,
            "sp_pad_provider": self.sp_pad_provider,
            "capture_bucket_rows": self.capture_bucket_rows,
            "mc2_active_mask_active": self.mc2_active_mask_active,
            "mmrs_fused": self.mmrs_fused,
            "pp": self.pp,
            "input_rows": self.input_rows,
            "valid_global": self.valid_global,
            "physical_global": self.physical_global,
            "attention_tokens": self.attention_tokens,
            "valid_per_tp": list(self.valid_per_tp),
            "router_padding_policy": "execute_padded_rows",
            "tokens_global": self.tokens_global,
            "tokens_local": self.tokens_local,
            "cohort_tokens": self.cohort_tokens,
            "valid_cohort_tokens": self.valid_cohort_tokens,
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


def _validate_declaration(raw: Any, *, has_moe_layers: bool = True) -> Mapping:
    if not isinstance(raw, Mapping):
        raise ModelingError("sequence_parallel_declaration_must_be_mapping")
    unknown = set(raw) - set(DECLARATION) - _OPTIONAL_DECLARATION_KEYS
    if unknown:
        raise ModelingError("sequence_parallel_declaration_unknown:" + ",".join(sorted(unknown)))
    missing = set(DECLARATION) - set(raw)
    if missing:
        raise ModelingError("sequence_parallel_declaration_missing:" + ",".join(sorted(missing)))
    for key, expected in DECLARATION.items():
        actual = raw[key]
        if key == "enable_expert_parallel":
            if has_moe_layers and actual is not True:
                raise ModelingError(
                    f"sequence_parallel_option_unsupported:{key}={actual!r}")
            if not isinstance(actual, bool):
                raise ModelingError(
                    f"sequence_parallel_option_unsupported:{key}={actual!r}")
        elif key == "implementation" and actual == IMPLEMENTATION:
            continue
        elif key == "padding" and actual in {"none", "right_to_tp"}:
            continue
        elif actual != expected:
            raise ModelingError(
                f"sequence_parallel_option_unsupported:{key}={actual!r}")
    if "sp_scope" in raw and str(raw["sp_scope"]) not in _VALID_SP_SCOPES:
        raise ModelingError(f"sequence_parallel_option_unsupported:sp_scope={raw['sp_scope']!r}")
    if "sp_pad_provider" in raw and str(raw["sp_pad_provider"]) not in _VALID_SP_PAD_PROVIDERS:
        raise ModelingError(
            f"sequence_parallel_option_unsupported:sp_pad_provider={raw['sp_pad_provider']!r}"
        )
    return raw


def _validate_views(views, tokens, *, padding: str = "none", pcp: int = 1) -> tuple[Any, ...]:
    try:
        values = tuple(views)
    except TypeError as exc:
        raise ModelingError("sequence_parallel_views_must_be_iterable") from exc
    padded = []
    logical = []
    for view in values:
        q_len = getattr(view, "q_len", None)
        pcp_idle_peer = (
            pcp > 1
            and padding == "right_to_tp"
            and isinstance(q_len, int)
            and not isinstance(q_len, bool)
            and q_len == 0
            and int(getattr(view, "global_q_len", 0) or 0) > 0
        )
        if not pcp_idle_peer and (not isinstance(q_len, int) or isinstance(q_len, bool) or q_len <= 0):
            raise ModelingError(f"sequence_parallel_requires_positive_q_len:{q_len!r}")
        if getattr(view, "padding", False):
            if padding != "right_to_tp":
                raise ModelingError("sequence_parallel_no_padding_domain")
            padded.append(view)
            continue
        logical.append(view)
        # Speculative draft fields are rejected only for the unpadded contract.
        # right_to_tp serving may carry a draft proposal on the same views.
        if padding == "none":
            for field in ("draft_proposal_tokens", "draft_accepted_tokens"):
                if getattr(view, field, 0):
                    raise ModelingError(
                        f"sequence_parallel_speculative_field_unsupported:{field}")
            # The ordinary decode scheduler records one committed token here even
            # when no draft exists. Only a multi-token advance needs speculative ABI.
            if getattr(view, "draft_advanced_tokens", 0) not in (0, 1):
                raise ModelingError(
                    "sequence_parallel_speculative_field_unsupported:draft_advanced_tokens")
    logical_total = sum(view.q_len for view in logical)
    physical_total = sum(view.q_len for view in padded)
    total = physical_total + logical_total if padded else logical_total
    if padding == "none" and padded:
        raise ModelingError("sequence_parallel_no_padding_domain")
    if tokens < total or (padding == "none" and tokens != total):
        raise ModelingError(
            f"sequence_parallel_tokens_must_equal_sum_q_len:{tokens}!={total}")
    return values


def _validate_model(
    model, *, padding: str = "none", allow_draft_scope: bool = True,
) -> None:
    plans = tuple(getattr(model, "layer_plans", ()))
    if not plans:
        raise ModelingError("sequence_parallel_requires_attn_residual_layer_plans")
    allowed_scopes = {"body", "draft"} if allow_draft_scope else {"body"}
    if any(getattr(layer, "scope", "body") not in allowed_scopes for layer in plans):
        raise ModelingError("sequence_parallel_requires_body_scope")
    # Standard residual models have no AttnRes block.
    # That layout is valid once SP right-pads to TP; the unpadded contract
    # still requires a positive source residual block.
    if padding == "none" and any(
            int(getattr(layer, "residual_block_size", 0)) <= 0 for layer in plans):
        raise ModelingError("sequence_parallel_requires_source_attn_residual_block_size")
    if padding == "none" and any(
            bool(getattr(layer, "hyper_connections", False))
            or int(getattr(layer, "hc_mult", 0) or 0) > 0 for layer in plans):
        raise ModelingError("sequence_parallel_hyperconnections_unsupported")


def sequence_parallel_enabled(metadata: Mapping | None, tp: int) -> bool:
    """One source capability decision for the declared SP implementation.

    The builders must not duplicate this condition.  It is exactly the already
    accepted declaration: implementation name plus TP>1; no model/path name or
    timing constant is added.
    """
    decl = dict(metadata or {}).get("sequence_parallel")
    return (isinstance(decl, Mapping)
            and str(decl.get("implementation", "")) == IMPLEMENTATION
            and int(tp) > 1)


def mmrs_eligible(
    *, tp: int, dtype: str, has_bias: bool, k_dim: int, enable_mmrs_fusion: bool = False,
) -> bool:
    """Exact CANN ``MatmulReduceScatterV2`` eligibility predicate (G1 / D1 §2.1)."""
    return (
        bool(enable_mmrs_fusion)
        and int(tp) in _MMRS_ALLOWED_TP
        and str(dtype or "").strip().lower() in _MMRS_ALLOWED_DTYPES
        and not bool(has_bias)
        and _MMRS_K_MIN <= int(k_dim) < _MMRS_K_MAX_EXCLUSIVE
    )


def _is_eager_execution(runtime, values: tuple[Any, ...], *, capture_bucket_rows: int) -> bool:
    """Resolve whether this invocation runs eagerly rather than inside a captured graph."""
    if capture_bucket_rows > 0:
        return False
    is_mapping = isinstance(runtime, Mapping)
    graph_mode = str(
        runtime.get("graph_mode", "") if is_mapping
        else getattr(runtime, "graph_mode", "") or ""
    ).strip().lower()
    capture_mode = str(
        runtime.get("graph_capture_mode", "") if is_mapping
        else getattr(runtime, "graph_capture_mode", "") or ""
    ).strip().lower()
    if graph_mode in _EAGER_GRAPH_MODES or capture_mode in _EAGER_GRAPH_MODES:
        return True
    if not is_mapping and hasattr(runtime, "graph_mode") and hasattr(runtime, "graph_buckets"):
        if values and all(hasattr(v, "phase") and hasattr(v, "padding") for v in values):
            from mindie_sim.prediction.graph import GraphPolicy

            _, scope = GraphPolicy().select(values, runtime)
            return scope == "none"
        has_buckets = bool(
            getattr(runtime, "graph_buckets", ())
            or getattr(runtime, "graph_capture_keys", ())
            or getattr(runtime, "piecewise_capture_keys", ())
        )
        return graph_mode == "auto" and not has_buckets
    if is_mapping and graph_mode == "auto":
        has_buckets = any(
            runtime.get(key)
            for key in ("graph_buckets", "graph_capture_keys", "piecewise_capture_keys")
        )
        return not has_buckets
    return False


def select(runtime, model, views, *, tokens, global_tokens=None, rank_ledger=None,
           valid_global_tokens=None, allow_draft_scope: bool = True):
    """Return the immutable plan, or ``None`` when no declaration is present.

    ``none`` retains the divisible-input contract. ``right_to_tp`` pads the
    physical input, including prior graph padding, while preserving valid
    views. Cross-DP totals must already include each replica's physical rows.
    """
    kernels = _kernels(runtime)
    if "sequence_parallel" not in kernels:
        return None
    has_moe_layers = any(
        getattr(layer, "ffn", "") == "moe"
        for layer in tuple(getattr(model, "layer_plans", ()))
    )
    declaration = _validate_declaration(
        kernels["sequence_parallel"], has_moe_layers=has_moe_layers)
    padding = declaration["padding"]
    sp_scope = str(declaration.get("sp_scope", "stage_entry_all_layers"))
    if sp_scope == "off":
        return None
    capture_bucket_rows = int(declaration.get("capture_bucket_rows", 0) or 0)
    mc2_active_mask_active = bool(declaration.get("mc2_active_mask_active", False))
    mmrs_fused = bool(declaration.get("mmrs_fused", False))
    effective_allow_draft = bool(declaration.get("allow_draft_scope", allow_draft_scope))
    if type(tokens) is not int or tokens < 0:
        raise ModelingError(f"sequence_parallel_requires_nonnegative_tokens:{tokens!r}")
    parallel = _parallel(runtime)
    values = _validate_views(views, tokens, padding=padding, pcp=int(parallel.pcp))
    eager_execution = _is_eager_execution(
        runtime, values, capture_bucket_rows=capture_bucket_rows,
    )
    default_provider = (
        "none" if padding == "none"
        else ("cat_zeros" if (sp_scope == "moe_flashcomm1_only" or eager_execution) else "constant_pad_nd")
    )
    sp_pad_provider = str(declaration.get("sp_pad_provider", default_provider))
    # A padding view is a physical graph row, not a logical request.  Its
    # q_len remains the source allocation; valid_global excludes it.
    valid = sum(view.q_len for view in values
                if not getattr(view, "padding", False))
    input_rows = max(tokens, capture_bucket_rows) if capture_bucket_rows > 0 else tokens
    tokens = input_rows
    pp, cp, pcp = int(parallel.pp), int(parallel.cp), int(parallel.pcp)
    tp, ep, dp = int(parallel.tp), int(parallel.ep), int(parallel.dp)
    if pp < 1:
        raise ModelingError(f"sequence_parallel_requires_pp1:{pp}")
    if pp > 1 and bool(declaration.get("rejects_sp_with_pp", False)):
        raise ModelingError(f"sequence_parallel_pp_gt_1_unsupported:{pp}")
    # Unpadded SP is the cp=1 source path. right_to_tp serving allows cp >= 1.
    if padding == "none" and cp != 1:
        raise ModelingError(f"sequence_parallel_requires_cp1:{cp}")
    if pcp < 1:
        raise ModelingError(f"sequence_parallel_requires_pcp1:{pcp}")
    if tp <= 1:
        raise ModelingError(f"sequence_parallel_requires_tp_gt_1:{tp}")
    dense_flashcomm = ep == 1 and not has_moe_layers
    if not dense_flashcomm and (ep < tp or ep % tp):
        raise ModelingError(f"sequence_parallel_requires_ep_ge_tp_and_divisible:{ep}:{tp}")
    _validate_model(model, padding=padding, allow_draft_scope=effective_allow_draft)
    if padding == "right_to_tp":
        tokens += (-tokens) % tp
    if tokens % tp:
        raise ModelingError(
            f"sequence_parallel_requires_tokens_divisible_by_tp:{tokens}%{tp}")
    local = tokens // tp
    if rank_ledger is not None:
        cohort = rank_ledger.physical
    elif ep > tp:
        if global_tokens is None:
            if padding == "right_to_tp" and dp == 1:
                pcp_in_ep = min(pcp, max(1, ep // tp)) if getattr(parallel, "ep_spans_pcp", False) else 1
                global_tokens = tokens * pcp_in_ep if any(getattr(v, "phase", "") == "prefill" for v in values) else tokens
            else:
                raise ModelingError("sequence_parallel_cross_dp_requires_complete_cohort_tokens")
        cohort = _positive_int(
            global_tokens, "sequence_parallel_global_tokens_invalid")
        if padding == "right_to_tp":
            cohort = max(cohort, tokens)
            cohort += (-cohort) % tp
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
        if (global_tokens is not None and global_tokens != tokens
                and not (padding == "right_to_tp" and global_tokens in {input_rows, valid})
                and not (dp > 1 and global_tokens >= tokens)):
            raise ModelingError(
                f"sequence_parallel_global_tokens_outside_ep_cohort:"
                f"{global_tokens}!={tokens}")
    if rank_ledger is not None:
        valid_cohort = int(sum(rank_ledger.valid))
    elif ep <= tp:
        valid_cohort = valid
    elif valid_global_tokens is not None:
        valid_cohort = min(cohort, max(0, int(valid_global_tokens)))
    elif padding == "right_to_tp":
        if dp == 1:
            pcp_in_ep = (min(pcp, max(1, ep // tp))
                         if getattr(parallel, "ep_spans_pcp", False) else 1)
            valid_cohort = (valid * pcp_in_ep
                            if any(getattr(v, "phase", "") == "prefill" for v in values)
                            else valid)
        elif tokens > 0:
            valid_cohort = min(cohort, valid * max(1, cohort // tokens))
        else:
            valid_cohort = cohort
    else:
        valid_cohort = cohort
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
        valid_cohort_tokens=valid_cohort,
        sp_scope=sp_scope, sp_pad_provider=sp_pad_provider,
        capture_bucket_rows=capture_bucket_rows,
        mc2_active_mask_active=mc2_active_mask_active,
        mmrs_fused=mmrs_fused, pp=pp, eager_execution=eager_execution, pcp=pcp,
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
        ("router_padding_policy", (
            "skip_padded_rows_via_mc2_mask" if plan.mc2_active_mask_active
            else "execute_padded_rows"
        )),
        ("input_shape", tuple(input_shape)),
        ("output_shape", tuple(output_shape)),
        ("tokens_global", plan.tokens_global),
        ("tokens_local", plan.tokens_local),
        ("cohort_tokens", plan.cohort_tokens),
        ("valid_cohort_tokens", plan.valid_cohort_tokens),
    ]
    if collective is not None:
        items.extend((("collective", collective), ("group_axis", "tp"),
                      ("comm_axis", "tp_sp"), ("comm_group_size", int(plan.tp)),
                      ("steps", plan.tp - 1)))
    if payload is not None:
        items.append(("wire_payload_bytes", payload))
    items.extend(plan.source)
    return tuple(items)


def input_shard(*, plan: SequenceParallelPlan, op_id: str, deps: tuple[str, ...],
                stage: int, layer: int, requests: tuple[str, ...], phase: str,
                width: int, dtype: str, row_multiplier: int = 1,
                channel_multiplier: int = 1, clone_aligned: bool = True) -> OpSpec:
    """Right-padding materializes once; aligned eager SP shards clone their local slice."""
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
            "sp_pad_provider": "constant_pad_nd",
            "aclnn_symbol": "aclnnConstantPadNd",
            "native_kinds": ("MemSet", "PadV3"),
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
    if clone_aligned and plan.padding == "right_to_tp" and plan.eager_execution and local_rows > 0:
        copy_bytes = float(local_rows * width * dtype_bytes(dtype))
        clone_metadata = dict(metadata)
        clone_metadata.update({
            "materialization": "aligned_clone",
            "shard_materialization": "clone",
            "physical_payload_bytes": 2.0 * copy_bytes,
            "native_kinds": ("Copy",),
            "source_expression": "out.clone() if sp_pad == 0 else out",
        })
        clone_metadata.pop("zero_payload_reason")
        return vector_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            phase=phase, dtype=str(dtype), family="copy",
            recipe="sequence_parallel_aligned_clone",
            elements=local_rows * width, ops_per_element=0,
            shape=(local_rows, width), read_bytes_override=copy_bytes,
            write_bytes_override=copy_bytes, metadata=tuple(clone_metadata.items()),
            notes=("eager aligned sp_shard clones the local TP slice (vllm_ascend sequence_parallel.py:43)",),
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


def _close_replicated_output(
    ops: tuple[OpSpec, ...], tail: OpSpec, *, plan: SequenceParallelPlan,
    width: int, dtype: str, op_id: str | None, row_multiplier: int,
    channel_multiplier: int, input_rows: int | None,
) -> tuple[tuple[OpSpec, ...], str]:
    """Slice an already TP-replicated output into local SP rows without reduction."""
    if not plan.tokens_local:
        raise ModelingError("sequence_parallel_empty_replica_has_no_tp_collective")
    shard_id = op_id or tail.op_id + ".sp_shard"
    if any(op.op_id == shard_id for op in ops):
        raise ModelingError(f"sequence_parallel_close_output_duplicate_id:{shard_id}")
    _, global_rows, _ = _shape(
        plan, width, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier)
    source_rows = (
        input_rows * row_multiplier if input_rows is not None
        else plan.input_rows * row_multiplier
    )
    if source_rows < 0 or source_rows > global_rows:
        raise ModelingError("sequence_parallel_reduce_scatter_rows_outside_allocation")
    shard_plan = (
        plan if input_rows is None or input_rows == plan.input_rows
        else replace(plan, input_rows=input_rows)
    )
    shard_op = input_shard(
        plan=shard_plan, op_id=shard_id, deps=(tail.op_id,), stage=tail.stage,
        layer=tail.layer, requests=tail.requests, phase=tail.phase,
        width=width, dtype=dtype, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier, clone_aligned=False,
    )
    shard_meta = dict(shard_op.metadata)
    shard_meta["sequence_parallel_role"] = "output_shard"
    shard_meta["input_dependency"] = "replicated_tp_output"
    shard_meta["tp_output_placement"] = "replicated"
    shard_op = replace(
        shard_op, stream=tail.stream, metadata=tuple(shard_meta.items()),
    )
    return ops + (shard_op,), shard_id


def emit_reduce_scatter_pad(
    *,
    plan: SequenceParallelPlan,
    op_id: str,
    deps: tuple[str, ...],
    stage: int,
    layer: int,
    requests: tuple[str, ...],
    phase: str,
    width: int,
    dtype: str,
    source_rows: int,
    target_rows: int | None = None,
    stream: str = "",
) -> OpSpec | None:
    """Emit G1 exact per-occurrence SP pad matching downstream ``reduce_scatter`` rows."""
    target_global = int(target_rows) if target_rows is not None else int(plan.tokens_global)
    src_rows = int(source_rows)
    if src_rows < 0 or src_rows > target_global:
        raise ModelingError(
            f"sequence_parallel_reduce_scatter_rows_outside_allocation:{src_rows}>{target_global}"
        )
    if plan.padding == "right_to_tp":
        pad_rows = max(0, target_global - src_rows)
    else:
        pad_rows = plan.reduce_scatter_pad_rows(src_rows)
    if pad_rows <= 0:
        return None
    padded_rows = src_rows + pad_rows
    elem_bytes = dtype_bytes(dtype)
    is_cat_zeros = plan.sp_pad_provider == "cat_zeros"
    if is_cat_zeros:
        # Eager vllm-ascend materializes x.new_zeros([pad, H]) then torch.cat([x, zeros], dim=0).
        payload_read = padded_rows * int(width) * elem_bytes
        payload_write = (pad_rows + padded_rows) * int(width) * elem_bytes
    else:
        payload_read = src_rows * int(width) * elem_bytes
        payload_write = padded_rows * int(width) * elem_bytes
    recipe = "sequence_parallel_cat_zeros_pad" if is_cat_zeros else "sequence_parallel_reduce_scatter_pad"
    expansion = ("ZerosLike", "ConcatD") if is_cat_zeros else ("MemSet", "PadV3")
    abi_meta: tuple[tuple[str, Any], ...] = (
        (("aclnn_symbols", ("aclnnInplaceZero", "aclnnCat")),)
        if is_cat_zeros else (("aclnn_symbol", "aclnnConstantPadNd"),)
    )
    pad = vector_op(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, dtype=str(dtype), family="pad", recipe=recipe,
        shape=(padded_rows, int(width)), elements=pad_rows * int(width), ops_per_element=1,
        read_bytes_override=payload_read, write_bytes_override=payload_write,
        metadata=_common(
            plan, role="reduce_scatter_pad", input_shape=(src_rows, int(width)),
            output_shape=(padded_rows, int(width)), payload=0.0,
        ) + abi_meta + (
            ("materialization", "cat_zeros" if is_cat_zeros else "right_padding"),
            ("sp_pad_provider", plan.sp_pad_provider),
            ("native_kinds", expansion),
            ("padding_rows", pad_rows),
            ("physical_payload_bytes", payload_read + payload_write),
            ("mapping_basis", "analytical_F_pad_copy_and_zero"),
        ),
    )
    return replace(pad, stream=stream) if stream else pad


def emit_mc2_mask_pad(
    *,
    plan: SequenceParallelPlan,
    op_id: str,
    deps: tuple[str, ...],
    stage: int,
    layer: int,
    requests: tuple[str, ...],
    phase: str,
    stream: str = "",
) -> OpSpec | None:
    """Emit G1 ``mc2_mask`` boolean pad (``3xCast + MemSet + PadV3``) when ``Q_bucket % T != 0``."""
    if not plan.mc2_active_mask_active:
        return None
    pad_rows = plan.reduce_scatter_pad_rows(plan.input_rows)
    if pad_rows <= 0:
        return None
    padded_rows = int(plan.input_rows) + pad_rows
    mask_kernels = ("Cast", "MemSet", "PadV3", "Cast", "Cast")
    op = vector_op(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, dtype="int8", family="pad", recipe="sequence_parallel_reduce_scatter_pad",
        shape=(padded_rows,), elements=padded_rows * _MC2_MASK_CAST_COUNT, ops_per_element=1,
        read_bytes_override=int(plan.input_rows), write_bytes_override=padded_rows,
        metadata=_common(
            plan, role="mc2_mask_pad", input_shape=(int(plan.input_rows),),
            output_shape=(padded_rows,), payload=0.0,
        ) + (
            ("aclnn_symbols", ("aclnnCast", "aclnnConstantPadNd", "aclnnCast", "aclnnCast")),
            ("native_kinds", mask_kernels),
            ("mc2_mask_cast_launches", _MC2_MASK_CAST_COUNT),
            ("padding_rows", pad_rows),
        ),
    )
    return replace(op, stream=stream) if stream else op


def _resolve_mmrs_ranks(
    machine: Any,
    plan: SequenceParallelPlan,
    tail: OpSpec,
    tail_meta: Mapping[str, Any],
    *,
    replica: int | None,
    physical_rank: int | None,
    group_ranks: tuple[int, ...] | None,
) -> tuple[bool, tuple[int, ...] | None]:
    """Resolve the physical TP group for MC2 MMRS, failing closed on ambiguous non-contiguous layouts."""
    die_ranks = tuple(int(die.rank) for die in machine.dies)
    die_set = set(die_ranks)
    tp = int(plan.tp)
    raw_group = group_ranks if group_ranks is not None else tail_meta.get("group_ranks")
    if raw_group is not None:
        resolved = tuple(int(rk) for rk in raw_group)
        if len(resolved) == tp and len(set(resolved)) == tp and set(resolved) <= die_set:
            return True, resolved
        return False, None
    pcp = max(1, int(plan.pcp))
    pp = max(1, int(plan.pp))
    stage = int(tail.stage)
    width = tp * pcp
    replica_span = pp * width
    replica_count = len(die_ranks) // replica_span if replica_span > 0 else 0
    stage_tp_groups: list[tuple[int, ...]] = []
    for rep in range(replica_count):
        base = (rep * pp + stage) * width
        for copy_idx in range(pcp):
            grp = die_ranks[base + copy_idx * tp : base + (copy_idx + 1) * tp]
            if len(grp) == tp:
                stage_tp_groups.append(grp)
    raw_rank = physical_rank if physical_rank is not None else tail_meta.get("physical_rank")
    if raw_rank is not None:
        target = int(raw_rank)
        for grp in stage_tp_groups:
            if target in grp:
                return True, grp
        return False, None
    raw_replica = replica if replica is not None else tail_meta.get("replica")
    if raw_replica is not None and pcp == 1:
        start = (int(raw_replica) * pp + stage) * width
        if 0 <= start and start + tp <= len(die_ranks):
            return True, die_ranks[start : start + tp]
        return False, None
    if len(stage_tp_groups) == 1:
        return True, stage_tp_groups[0]
    if not stage_tp_groups:
        return False, None
    contiguous = all(
        set(grp) == set(range((rk // tp) * tp, (rk // tp) * tp + tp))
        for grp in stage_tp_groups
        for rk in grp
    )
    return (True, None) if contiguous else (False, None)


def _build_executable_mmrs(
    producer_op: OpSpec,
    replacement: OpSpec,
    tail: OpSpec,
    *,
    plan: SequenceParallelPlan,
    dtype: str,
    tail_meta: Mapping[str, Any],
    machine: Any | None,
    replica: int | None = None,
    physical_rank: int | None = None,
    group_ranks: tuple[int, ...] | None = None,
) -> OpSpec | None:
    """Build a governed ``mc2_matmul_reduce_scatter`` op iff it has an executable matrix+collective plan."""
    if (producer_op.plan_required or producer_op.family not in _MMRS_PRODUCER_FAMILIES
            or producer_op.stage != tail.stage
            or (producer_op.stream or "") != (tail.stream or "")):
        return None
    prod_meta = dict(producer_op.metadata)
    if int(prod_meta.get("matrix_groups", 1) or 1) != 1:
        return None
    dims: tuple[int, int, int] | None = None
    for shp in (producer_op.local_shape, producer_op.semantic_shape):
        if len(shp) >= _MMRS_REQUIRED_DIMS:
            m_dim, n_dim, k_dim = int(shp[0]), int(shp[1]), int(shp[2])
            if m_dim > 0 and n_dim > 0 and k_dim > 0:
                dims = (m_dim, n_dim, k_dim)
                break
    if dims is None:
        return None
    m_dim, n_dim, k_dim = dims
    if m_dim % plan.tp != 0:
        return None
    has_bias = bool(prod_meta.get("has_bias", tail_meta.get("has_bias", False)))
    comp_dt = str(prod_meta.get("compute_format") or producer_op.dtype or dtype).strip().lower()
    out_dt = str(prod_meta.get("output_format") or comp_dt).strip().lower()
    act_dt = str(prod_meta.get("activation_format") or comp_dt).strip().lower()
    weight_dt = str(prod_meta.get("weight_format") or comp_dt).strip().lower()
    comp_canon = _MMRS_CANONICAL_DTYPES.get(comp_dt)
    out_canon = _MMRS_CANONICAL_DTYPES.get(out_dt)
    act_canon = _MMRS_CANONICAL_DTYPES.get(act_dt)
    weight_canon = _MMRS_CANONICAL_DTYPES.get(weight_dt)
    req_canon = (
        _MMRS_CANONICAL_DTYPES.get(str(dtype).strip().lower())
        if dtype else comp_canon
    )
    if (
        comp_canon is None
        or out_canon != comp_canon
        or act_canon != comp_canon
        or weight_canon != comp_canon
        or req_canon != comp_canon
    ):
        return None
    if not mmrs_eligible(
        tp=plan.tp, dtype=comp_canon, has_bias=has_bias, k_dim=k_dim,
        enable_mmrs_fusion=True,
    ):
        return None
    if machine is None or not getattr(machine, "dies", ()) or not getattr(machine, "links", ()):
        return None
    die = machine.dies[0]
    die_rates = dict(getattr(die, "rates", ()) or ())
    if any(float(die_rates.get(role, 0.0) or 0.0) <= 0.0 for role in _MMRS_REQUIRED_RATES):
        return None
    arch = getattr(die, "compute_arch", None)
    if arch is None or not arch.declared:
        return None
    if arch.has_native_modes and not arch.supports_native("matrix", comp_canon):
        return None
    arch_preset = str(getattr(arch, "preset", "") or "").strip().lower()
    if arch_preset == _ARCH22_PRESET:
        transport_binding = {"caller_abi": "hccl_client", "server_mode": "aiv"}
    elif arch_preset == _ARCH35_PRESET:
        transport_binding = {"caller_abi": "ccu"}
    else:
        return None
    ranks_ok, explicit_ranks = _resolve_mmrs_ranks(
        machine, plan, tail, tail_meta,
        replica=replica, physical_rank=physical_rank, group_ranks=group_ranks,
    )
    if not ranks_ok:
        return None
    mc2_contract: dict[str, Any] = {
        "abi": "matmul_reduce_scatter",
        "world_size": int(plan.tp),
        "m": m_dim,
        "k": k_dim,
        "n": n_dim,
        "compute_dtype": comp_canon,
        "output_dtype": comp_canon,
        "operand_dtype": comp_canon,
        "internal_chunks": _MMRS_INTERNAL_CHUNKS,
        "independent_buffers": True,
        "independent_handles": True,
        "transport_binding": transport_binding,
    }
    if explicit_ranks is not None:
        mc2_contract["ranks"] = explicit_ranks
    rep_meta = dict(replacement.metadata)
    rep_meta.update({
        "mmrs_fused": True,
        "mc2_contract": mc2_contract,
        "abi_symbol": "aclnnMatmulReduceScatterV2",
        "aclnn_symbol": "aclnnMatmulReduceScatterV2",
        "native_kinds": ("MatmulReduceScatter",),
        "fused_producer_op_id": producer_op.op_id,
        "original_gemm_recipe": producer_op.recipe,
        "compute_format": comp_canon,
        "output_format": comp_canon,
        "activation_format": comp_canon,
        "weight_format": comp_canon,
    })
    if explicit_ranks is not None:
        rep_meta["group_ranks"] = explicit_ranks
    return replace(
        replacement,
        family="matmul_reduce_scatter",
        recipe="mc2_matmul_reduce_scatter",
        deps=producer_op.deps,
        semantic_shape=producer_op.semantic_shape,
        local_shape=(m_dim, n_dim, k_dim),
        padded_shape=producer_op.padded_shape,
        launch_shape=producer_op.launch_shape,
        flops=producer_op.flops,
        matrix_flops=producer_op.matrix_flops,
        hbm_bytes=producer_op.hbm_bytes + replacement.hbm_bytes,
        mte2_bytes=producer_op.mte2_bytes + replacement.mte2_bytes,
        mte3_bytes=producer_op.mte3_bytes + replacement.mte3_bytes,
        dtype=comp_canon,
        stream=tail.stream,
        metadata=tuple(rep_meta.items()),
    )


def close_output(ops, last, *, plan: SequenceParallelPlan, width: int, dtype: str,
                 op_id: str | None = None, row_multiplier: int = 1,
                 channel_multiplier: int = 1,
                 input_rows: int | None = None,
                 machine: Any | None = None,
                 replica: int | None = None,
                 physical_rank: int | None = None,
                 group_ranks: tuple[int, ...] | None = None) -> tuple[tuple[OpSpec, ...], str]:
    """Close a TP-sharded attention/MLP output into SP-local rows.

    Replaces an exact trailing TP ``allreduce`` (sole producer) with
    ``reduce_scatter``, or appends a local row-shard view/right-pad when the
    tail explicitly declares ``tp_output_placement="replicated"`` (e.g. grouped
    MLA output projection when ``o_groups`` is not divisible by TP).
    """
    ops = tuple(ops)
    if not ops or ops[-1].op_id != last:
        raise ModelingError("sequence_parallel_close_output_requires_exact_tail")
    tail = ops[-1]
    metadata = dict(tail.metadata)
    is_tp_allreduce = tail.family == "allreduce" and metadata.get("group_axis") == "tp"
    if not is_tp_allreduce:
        if metadata.get("tp_output_placement") == "replicated":
            return _close_replicated_output(
                ops, tail, plan=plan, width=width, dtype=dtype, op_id=op_id,
                row_multiplier=row_multiplier, channel_multiplier=channel_multiplier,
                input_rows=input_rows,
            )
        raise ModelingError("sequence_parallel_close_output_requires_tp_allreduce_tail")
    if len(tail.deps) != 1:
        raise ModelingError("sequence_parallel_close_output_requires_sole_producer")
    replacement_id = op_id or tail.op_id + ".reduce_scatter"
    if any(op.op_id == replacement_id for op in ops[:-1]):
        raise ModelingError(f"sequence_parallel_close_output_duplicate_id:{replacement_id}")
    prefix, dependencies = ops[:-1], tail.deps
    pad = None
    if input_rows is not None:
        _, global_rows, physical_width = _shape(
            plan, width, row_multiplier=row_multiplier,
            channel_multiplier=channel_multiplier)
        source_rows = input_rows * row_multiplier
        if source_rows < 0 or source_rows > global_rows:
            raise ModelingError("sequence_parallel_reduce_scatter_rows_outside_allocation")
        pad = emit_reduce_scatter_pad(
            plan=plan, op_id=replacement_id + ".pad", deps=dependencies,
            stage=tail.stage, layer=tail.layer, requests=tail.requests,
            phase=tail.phase, width=physical_width, dtype=dtype,
            source_rows=source_rows, target_rows=global_rows, stream=tail.stream,
        )
        if pad is not None:
            prefix += (pad,)
            dependencies = (pad.op_id,)
    replacement = reduce_scatter(
        plan=plan, op_id=replacement_id, deps=dependencies, stage=tail.stage,
        layer=tail.layer, requests=tail.requests, phase=tail.phase,
        width=width, dtype=dtype, row_multiplier=row_multiplier,
        channel_multiplier=channel_multiplier,
    )
    if plan.mmrs_fused:
        producer_idx = next(
            (i for i, op in enumerate(prefix) if op.op_id == tail.deps[0]),
            None,
        )
        producer_op = prefix[producer_idx] if producer_idx is not None else None
        has_other_consumer = (
            producer_op is not None
            and any(
                producer_op.op_id in op.deps
                for i, op in enumerate(prefix)
                if i != producer_idx
            )
        )
        fused = (
            _build_executable_mmrs(
                producer_op, replacement, tail, plan=plan, dtype=dtype,
                tail_meta=metadata, machine=machine,
                replica=replica, physical_rank=physical_rank, group_ranks=group_ranks,
            )
            if (pad is None and producer_op is not None and not has_other_consumer)
            else None
        )
        if fused is not None and producer_idx is not None:
            prefix = prefix[:producer_idx] + prefix[producer_idx + 1:]
            return prefix + (fused,), replacement_id
        rep_meta = dict(replacement.metadata)
        rep_meta["mmrs_fused"] = False
        replacement = replace(replacement, metadata=tuple(rep_meta.items()))
    return prefix + (replace(replacement, stream=tail.stream),), replacement_id
