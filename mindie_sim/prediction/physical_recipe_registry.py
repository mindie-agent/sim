"""Physical mechanism binding registry for the 238-pair operator census.

[Role] Resolve one operator family/recipe pair to one of the five declared
  physical mechanism classes, or to an explicit control/transfer mechanism.
[Invariants] Selection never returns an aggregate envelope. Unknown mechanisms
  raise the canonical ``unmodeled_operator_mechanism`` error. Exact source
  planners and caller recipe/family bindings are resolved before this registry
  by ``RecipeBook``.
[Timeouts & Locks] Constant-time family lookups and bounded recipe prefix
  checks; no I/O or locks.
[Side-Effect Policy] Pure immutable binding records.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from mindie_sim.schema.common import ModelingError
from .attention_pipeline import TiledFlashAndPagedAttentionRecipe
from .linear_attention_pipeline import ChunkedLinearAttentionRecipe
from .program import KernelProgram, Stage
from .sparse_indexer_pipeline import TiledSparseIndexerRecipe
from .topology_collective import TopologyCollectiveRecipe
from .vector_pipeline import AIVTiledVectorAndGatherRecipe



@dataclass(frozen=True)
class MechanismBinding:
    phase: str
    family: str
    recipe: str
    mechanism: str
    variant: str


@dataclass(frozen=True)
class ControlSyncRecipe:
    """Declared event/barrier/state-publication mechanism with no payload."""

    mechanism: str = "control_sync"

    def lower(self, context):
        # A declared host/event service is the control mechanism itself and is
        # paid exactly once.  Any *payload* resource (HBM, MTE2/3, vector,
        # Cube, link) means the family was mis-bound to a control recipe and
        # must raise instead of executing an empty program.
        declared = {**dict(context.native_work or {}), **dict(context.work or {})}
        control = {}
        payload = {}
        for name, value in declared.items():
            amount = float(value)
            if amount <= 0.0:
                continue
            key = str(name)
            if key.startswith("host") or key.startswith("scheduler@"):
                control[key] = amount
            else:
                payload[key] = amount
        if payload:
            family = str(getattr(context.op, "family", ""))
            recipe = str(getattr(context.op, "recipe", ""))
            work = sorted((name, value) for name, value in payload.items())
            raise ModelingError(
                f"unmodeled_operator_mechanism:{family}:{recipe}:"
                + ",".join(f"{name}={value:.6g}" for name, value in work))
        # A declared host resource pays the control's own event latency once.
        # A caller that declares no host resource keeps the empty program; the
        # bookkeeping then belongs to that caller's submission boundary.
        # The same host service can be published under a logical role and its
        # physical name; the control pays that service once, not per alias.
        latency = float(max(control.values())) if control else 0.0
        return KernelProgram((Stage("arrive", keep_empty=True),
                              Stage("publish", after=("arrive",), latency=latency,
                                    keep_empty=latency <= 0.0)),
                             minimum_ms=context.minimum_ms)


ATTENTION = TiledFlashAndPagedAttentionRecipe()
INDEXER = TiledSparseIndexerRecipe()
LINEAR = ChunkedLinearAttentionRecipe()
VECTOR = AIVTiledVectorAndGatherRecipe()
COLLECTIVE = TopologyCollectiveRecipe()
CONTROL = ControlSyncRecipe()

ATTENTION_FAMILIES = frozenset({
    "attention", "mla", "compressed_attention", "dcp_logsumexp_attention_update",
})
INDEXER_FAMILIES = frozenset({"index_pool", "indexer"})
INDEXER_TOPK_RECIPES = frozenset({
    "index_pool_topk", "index_topk", "index_pool_compress",
    "index_pool_scores", "index_pool_weighted_query", "index_scores",
})
LINEAR_FAMILIES = frozenset({
    "recurrent_local", "recurrent_output", "triangular_solve",
})
COLLECTIVE_FAMILIES = frozenset({
    "allgather", "allreduce", "alltoall", "alltoallv", "dcp_allgather",
    "reduce_scatter", "tp_allreduce", "engram_broadcast", "engram_ids",
})
VECTOR_FAMILIES = frozenset({
    "activation", "add", "add_rmsnorm", "add_rmsnorm_quant", "attention_merge", "attention_residual", "cache_store", "cast", "conv1d", "copy", "elementwise",
    "embedding", "engram_apply", "engram_gather", "engram_query", "fill", "geglu", "gdn_gating", "hbm_load", "histogram", "kv_cache_write", "kv_cast",
    "layernorm", "mhc", "multiply", "native_fusion", "pad", "quantize", "relu", "rmsnorm", "rmsnorm_quant", "rotary", "routing", "sigmoid", "silu", "state_gather", "state_transfer", "test",
    "sampling", "acceptance", "situ", "softmax", "swiglu", "tensor_copy",
    "tensor_fill", "tensor_materialization", "tensor_select", "tensor_view",
    "state_scatter", "topk", "vector", "zero_fill",
})
CONTROL_FAMILIES = frozenset({
    "barrier", "event", "hccl_boundary", "token_commit",
    "state_publication", "state_publish",
})
TRANSFER_FAMILIES = frozenset({"cache_transfer", "kv_transfer", "offload", "pp_p2p", "transfer"})
PROJECTION_FAMILIES = frozenset({
    "gemm", "qkv_projection", "output_projection", "gate_up_projection",
    "down_projection", "projection", "router", "index_projection", "latent_projection",
    "lm_head", "ple_projection", "grouped_gemm",
})
LATENT_ATTENTION_RE = re.compile(r"^latent_attention(?:\d+)?$")
CHUNK_RECIPE_RE = re.compile(
    r"^(?:chunk_products|wy_transform|triangular_solve|recurrent_output)\d+$")


def _mechanism(family: str, recipe: str):
    if family in ATTENTION_FAMILIES or LATENT_ATTENTION_RE.match(recipe):
        return ATTENTION, "class_a", "tiled_flash_paged"
    if family in INDEXER_FAMILIES or recipe in INDEXER_TOPK_RECIPES:
        return INDEXER, "class_b", "tiled_sparse_indexer"
    if family in LINEAR_FAMILIES or CHUNK_RECIPE_RE.match(recipe):
        return LINEAR, "class_c", "chunked_linear_attention"
    if family in COLLECTIVE_FAMILIES:
        return COLLECTIVE, "class_e", "topology_collective"
    if family in CONTROL_FAMILIES:
        return CONTROL, "control", "declared_control_sync"
    if family in VECTOR_FAMILIES:
        return VECTOR, "class_d", VECTOR.submode_from_family(family, recipe)
    if family in PROJECTION_FAMILIES:
        return None, "matrix", "registered_matrix_program"
    if family in TRANSFER_FAMILIES:
        from .program import TransferRecipe
        return TransferRecipe(), "transfer", "declared_transfer"
    return None, "", ""


def resolve_physical_recipe(op, mechanisms=None):
    """Return the registered physical recipe, or raise the canonical error."""
    del mechanisms
    family = str(getattr(op, "family", ""))
    recipe = str(getattr(op, "recipe", ""))
    selected, mechanism, variant = _mechanism(family, recipe)
    if selected is None:
        raise ModelingError(
            f"unmodeled_operator_mechanism:{op.family}:{op.recipe}")
    return selected


def binding_for(op) -> MechanismBinding:
    family = str(getattr(op, "family", ""))
    recipe = str(getattr(op, "recipe", ""))
    selected, mechanism, variant = _mechanism(family, recipe)
    if selected is None:
        raise ModelingError(
            f"unmodeled_operator_mechanism:{op.family}:{op.recipe}")
    return MechanismBinding(str(getattr(op, "phase", "")), family, recipe,
                            mechanism, variant)


def physical_family_bindings() -> dict[str, object]:
    """Default family recipes installed ahead of caller overrides."""
    bindings = {family: ATTENTION for family in ATTENTION_FAMILIES}
    bindings.update({family: INDEXER for family in INDEXER_FAMILIES})
    bindings.update({family: LINEAR for family in LINEAR_FAMILIES})
    bindings.update({family: COLLECTIVE for family in COLLECTIVE_FAMILIES})
    bindings.update({family: VECTOR for family in VECTOR_FAMILIES})
    bindings.update({family: CONTROL for family in CONTROL_FAMILIES})
    return bindings


def physical_recipe_bindings() -> dict[str, object]:
    """Default exact recipes for topk/indexer names that are family-ambiguous."""
    return {recipe: INDEXER for recipe in INDEXER_TOPK_RECIPES}
