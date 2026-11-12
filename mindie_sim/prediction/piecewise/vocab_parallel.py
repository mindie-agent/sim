"""Vocab-parallel embedding and LM-head collectives.

[Role]
  Encode the source admission for the one TP allreduce after a sharded
  embedding lookup, and for the LM-head logits collective that follows the
  local matmul. Builders call these predicates. They do not price a hop.

[Invariants]
  * ``AscendVocabParallelEmbedding._forward_standard`` all-reduces only when
    ``tp_size > 1`` and ``output_reduce_group.world_size > 1``. The group is
    the TP group. Replicated embeddings, TP 1, a missing group, and
    fine-grained embedding token exchange do not emit this allreduce.
    FlashComm does not gate it. The payload is the lookup result, one row
    per real query token, in the activation dtype.
  * ``LogitsProcessor._gather_logits`` runs only when the head ``tp_size > 1``.
    ``use_all_gather`` (platform default, and the enrolled Ascend graph) is an
    all-gather of the matmul result. An explicit gather declaration is
    gather-to-sampler. ``skip_gather`` and ``logits_as_input`` emit nothing.
    The bytes are the producing LM-head shard in its result dtype, not the
    sampler's fp32 policy and not the physical prompt length.
  * Vocabulary padding stays the 64-aligned ``pad_vocab_size`` split. The
    HCCL tensor keeps the padded shard. The embedding weight is that local
    partition for a sharded mode and the padded full table when the mode is
    replicated or the shard world is 1. A lookup reads the live token rows.
    It does not walk the table.

[Timeouts & Locks] Pure. No I/O.
[Side-Effect Policy] Returns frozen values only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.common import dtype_bytes


def _as_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _sources(model: Any, features: Any) -> tuple[dict[str, Any], ...]:
    """Serving metadata wins, then the model record, then catalog facts."""
    return (
        _as_map(getattr(features, "metadata", ())),
        _as_map(getattr(model, "metadata", ())),
        _as_map(getattr(model, "catalog_facts", ())),
    )


def _declared(model: Any, features: Any, *keys: str) -> Any:
    for source in _sources(model, features):
        for key in keys:
            if key in source and source[key] not in (None, ""):
                return source[key]
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mode(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def group_world(model: Any, features: Any) -> int:
    """Configured TP-group size. An explicit rank list shorter than 2 is absent."""
    ranks = _declared(model, features, "tp_group_ranks", "tp_ranks")
    if isinstance(ranks, (list, tuple)):
        return len(ranks)
    world = _declared(model, features, "tp_group_world")
    if world is not None:
        return max(0, int(world))
    return max(1, int(getattr(features, "tp", 1) or 1))


def vocab_padding(model: Any, features: Any) -> int:
    declared = _declared(model, features, "vocab_padding", "vocab_padding_size")
    if declared is not None and int(declared) > 0:
        return int(declared)
    raw = getattr(features, "vocab_padding", 64) or 64
    return max(1, int(raw))


@dataclass(frozen=True)
class VocabLayout:
    """Padded vocabulary split. ``group_world`` is the collective, not the shard."""

    tp: int
    group_world: int
    padding: int
    padded: int
    local: int
    full: int


def vocab_layout(model: Any, features: Any) -> VocabLayout:
    """``pad_vocab_size`` then one local shard. TP 1 keeps the padded width."""
    padding = vocab_padding(model, features)
    vocab = int(getattr(model, "vocab_size", 0) or 0)
    padded = ((vocab + padding - 1) // padding) * padding
    tp = max(1, int(getattr(features, "tp", 1) or 1))
    local = (padded + tp - 1) // tp if tp > 1 else padded
    full = local * tp if tp > 1 else padded
    return VocabLayout(tp, group_world(model, features), padding, padded, local, full)


def _embedding_mode(model: Any, features: Any) -> str:
    mode = _mode(_declared(model, features, "embedding_parallel"))
    if mode:
        return mode
    if _truthy(_declared(model, features, "embedding_disable_tp")):
        return "replicated"
    etp = _declared(model, features, "embedding_tensor_parallel_size")
    if etp is not None and int(etp) > 1:
        return "fine_grained"
    pcp = int(getattr(features, "pcp", 1) or 1)
    if pcp > 1 and _truthy(_declared(
            model, features, "pcp_embedding_lmhead_weight_sharding")):
        return "pcp"
    return "standard"


@dataclass(frozen=True)
class EmbeddingTable:
    """Local embedding weight. ``rows`` is the resident vocabulary partition.

    ``masked`` is the out-of-shard zero applied to the live ids. It is not a
    collective and it does not scan the partition.
    """

    mode: str
    rows: int
    cols: int
    masked: bool
    shard_world: int


def _partition_rows(padded: int, world: int) -> int:
    world = max(1, int(world))
    if world <= 1:
        return int(padded)
    return (int(padded) + world - 1) // world


def embedding_table(model: Any, features: Any) -> EmbeddingTable:
    """Weight the lookup actually holds.

    Standard mode uses the TP vocab partition, including when the TP group is
    missing and the allreduce is omitted. Fine-grained embedding TP shards by
    its own world, not by the model TP width. PCP weight sharding uses the
    TP x PCP grid. Replicated, disabled, and any other mode keep the padded
    full table. TP 1 is that full table with no mask.
    """
    layout = vocab_layout(model, features)
    hidden = int(getattr(model, "hidden_size", 0) or 0)
    mode = _embedding_mode(model, features)
    if mode in {"replicated", "none", "disabled"}:
        return EmbeddingTable(mode, layout.padded, hidden, False, 1)
    if mode in {"fine_grained", "finegrained", "token_exchange"}:
        declared = _declared(model, features, "embedding_tensor_parallel_size")
        world = max(1, int(declared or 1))
        return EmbeddingTable(
            mode, _partition_rows(layout.padded, world), hidden, world > 1, world)
    if mode == "pcp":
        pcp = max(1, int(getattr(features, "pcp", 1) or 1))
        world = max(1, layout.tp) * pcp
        return EmbeddingTable(
            mode, _partition_rows(layout.padded, world), hidden, world > 1, world)
    if mode == "standard":
        return EmbeddingTable(
            mode, layout.local, hidden, layout.tp > 1, max(1, layout.tp))
    return EmbeddingTable(mode, layout.padded, hidden, False, 1)


def embedding_allreduce_world(model: Any, features: Any) -> int:
    """TP world of the post-lookup allreduce, or 0 when the source skips it.

    Fine-grained embedding TP exchanges token rows on its own group. That is
    not this allreduce. PCP weight sharding still finishes with the TP
    allreduce of this rank's token rows; the PCP reduce-scatter is a different
    edge and is not emitted here.
    """
    mode = _embedding_mode(model, features)
    if mode in {"replicated", "none", "disabled", "fine_grained", "finegrained",
                "token_exchange"}:
        return 0
    if mode not in {"standard", "pcp"}:
        return 0
    world = group_world(model, features)
    return world if world > 1 else 0


def lm_head_result(model: Any, features: Any) -> tuple[str, float]:
    """Dtype of the LM-head matmul result, before the sampler cast."""
    quant = getattr(features, "quant", None)
    declared = ""
    if quant is not None and hasattr(quant, "output_format_for"):
        declared = str(quant.output_format_for("lm_head") or "")
    fmt = declared or str(getattr(features, "activation_dtype", "") or "bf16")
    return fmt, float(dtype_bytes(fmt))


def logits_collective_kind(model: Any, features: Any) -> str | None:
    """``all_gather``, ``gather``, or ``None``.

    The enrolled graph all-gathers the local BF16 shard. Gather-to-sampler is
    only the explicit ``logits_collective=gather`` or ``use_all_gather`` false
    declaration. Fine-grained LM-head TP is a different group and does not
    emit this edge.
    """
    if _truthy(_declared(model, features, "skip_gather", "logits_as_input")):
        return None
    if _truthy(_declared(model, features, "lmhead_disable_tp")):
        return None
    mode = _mode(_declared(model, features, "logits_collective"))
    if mode in {"skip", "none", "replicated"}:
        return None
    ltp = _declared(model, features, "lmhead_tensor_parallel_size")
    fine = mode in {"all_to_all", "fine_grained", "finegrained"} or (
        ltp is not None and int(ltp) > 1 and mode not in {"all_gather", "gather"})
    if fine:
        return None
    if group_world(model, features) <= 1:
        return None
    if mode in {"gather", "gather_to_sampler"}:
        return "gather"
    if mode in {"all_gather", "allgather"}:
        return "all_gather"
    use = _declared(model, features, "use_all_gather")
    if use is not None and not _truthy(use):
        return "gather"
    return "all_gather"
