"""Raw serving cases shared by the memory, KV-cache and offload runtime tests.

[Role] Single definition of the compact raw ``Case`` mappings and the linear
  lowering used by ``runtime/test_memory_ledger_and_capacity.py``,
  ``runtime/test_kv_cache_datapath.py`` and ``runtime/test_offload_tiers.py``
  so no runtime module imports another test module.
[Invariants] Every builder returns a fresh mapping / immutable ``Case``; the
  linear lowering charges exactly ``1e6`` vector ops per scheduled token, so
  one token costs one millisecond on the declared ``1e9`` vector rate.
[Timeouts & Locks] Pure in-memory construction; no I/O, sleeps, or locks.
[Side-Effect Policy] Side-effect-free; zero global mutation.
"""
from __future__ import annotations

from dataclasses import replace

from _support import axes
from mindie_sim import Case
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec

#: Packaged sparse-KV-offload capable SKU (HiSparse / MemFabric D2H path).
SPARSE_OFFLOAD_SKU = axes.GEN5_DT_SKU


def case(*, dp=1, tp=1, pp=1, ep=1, prompt=4, output=3, requests=None) -> dict:
    """Two-layer declared-GPU raw case with a 1e9 vector rate and 1 KB/ms fabric."""
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


def budget_memory(budgets, *, policy: MemoryPolicy | None = None, **case_kw) -> Memory:
    """``Memory`` over ``case(**case_kw)`` whose die ``i`` holds exactly ``budgets[i]``
    bytes after its stage's fixed weights (``kv_block_size=1``, no derived workspace).

    A scalar budget applies to every die.  One KV row of the default two-layer
    model is ``2 layers * 2 kv_heads * 8 dim * 2 (K,V) * 2 B = 128 B``.
    """
    raw = case(**case_kw)
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    weights = Memory(parsed).weights
    if isinstance(budgets, (int, float)):
        budgets = (budgets,) * len(parsed.machine.dies)
    dies = tuple(
        replace(die, memory_bytes=int(weights[index % len(weights)] + budget))
        for index, (die, budget) in enumerate(zip(parsed.machine.dies, budgets)))
    return Memory(replace(parsed, machine=replace(parsed.machine, dies=dies)),
                  policy or MemoryPolicy(derive_workspace=False))


def linear_work(plan, model, runtime, machine):
    """One vector op of ``tokens * 1e6`` ops over every scheduled (and proposed) token."""
    tokens = sum(view.q_len + view.draft_proposal_tokens for view in plan.requests)
    shape = (tokens,)
    op = OpSpec("work", "vector", "body", 0, 0, tuple(v.request_id for v in plan.requests),
                shape, shape, shape, shape, flops=tokens * 1e6,
                vector_ops=tokens * 1e6, dtype="bf16")
    return IterationWork((op,), plan, 1, tokens, tokens, None, "body", "test_linear")


def compressed_case() -> Case:
    """Two MLA layers with a window-compressed (ratio 4) long cache owned by layer 0."""
    parsed = Case.from_mapping(case())
    owner = replace(
        parsed.model.layer_plans[0], attention="mla", cache_kind="window_compressed", window=5,
        compress_ratio=4, kv_lora_rank=8, head_dim=8, compressor_width=1,
        compressor_state_rows=2, owns_long_kv=True, kv_owner_layer_id=0,
        q_lora_rank=8, qk_nope_head_dim=8, mla_variant="latent_mqa",
    )
    return replace(parsed, model=replace(parsed.model, layer_plans=(
        owner, replace(owner, index=1, owns_long_kv=False))))


def draft_case() -> Case:
    """``case(prompt=4, output=6)`` with a one-layer MTP draft proposing 2, accepting 1."""
    parsed = Case.from_mapping(case(prompt=4, output=6))
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16,
                      vocab_size=128, target_shared_embedding=True, target_shared_lm_head=True)
    draft = replace(draft, layer_plans=native_draft_layer_plans(parsed.model, draft))
    return replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=draft)),
                   runtime=replace(parsed.runtime, metadata=(
                       ("draft_mode", "mtp"), ("draft_proposal_tokens", 2),
                       ("draft_accept_tokens", 1))))


def _sparse_mla_layer(index: int, *, owner: bool) -> dict:
    return {
        "index": index, "attention": "mla", "ffn": "dense", "architecture": "mla",
        "num_heads": 16, "num_kv_heads": 16, "head_dim": 64, "intermediate_size": 2048,
        "q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "mla_variant": "latent_mla",
        "index_heads": 32, "index_dim": 128, "index_topk": 2048,
        "indexer": owner, "owns_index_k": owner, "owns_topk": owner,
        "index_query_per_layer": owner, "kv_owner_layer_id": index,
        "index_k_owner_layer_id": 0, "topk_owner_layer_id": 0,
    }


def sparse_mla_case(*, context_tokens: int = 8192, offloading: str | dict | None = None,
                    tp: int = 1) -> dict:
    """Two-layer sparse-MLA (indexer top-2048) raw case on the sparse-offload SKU."""
    lowerings = {} if offloading is None else {"sparse_kv_offload": offloading}
    return {
        "machine": {"backend": "ascend", "sku": SPARSE_OFFLOAD_SKU, "num_devices": tp},
        "model": {
            "name": "glm5-sparse-mla-mini", "layers": 2, "hidden_size": 1024,
            "num_heads": 16, "num_kv_heads": 16, "head_dim": 64,
            "intermediate_size": 2048, "vocab_size": 4096, "attention": "mla",
            "mla": {"q_lora_rank": 512, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
                    "qk_rope_head_dim": 64, "v_head_dim": 128},
            "layer_plans": [_sparse_mla_layer(0, owner=True), _sparse_mla_layer(1, owner=False)],
        },
        "workload": {"requests": [{"prompt_tokens": context_tokens, "output_tokens": 2}]},
        "runtime": {
            "tp": tp, "max_model_len": 32768, "max_num_seqs": 16, "kv_block_size": 128,
            "kernels": {"lowerings": lowerings} if lowerings else {},
        },
    }


def index_pool_case() -> dict:
    """Four-layer indexed-attention packed-int8 case from the matrix-extension suite."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    raw = json.loads((
        root / "examples/cases/high-concurrency-20261030/matrix-extension/"
        "a3_glm53flash_tp16_aclgraph_c32.json"
    ).read_text(encoding="utf-8"))
    raw["model"]["layers"] = 4
    raw["runtime"]["kernels"]["programs"]["indexed_attention"] = {
        "kind": "indexed_attention_packed_int8",
        "scratch_policy": "offchip",
    }
    return raw


def index_pool_layer(raw: dict, batch: int = 1, query: int = 6, context: int = 131201) -> list:
    """Decode body layer-3 ops for ``index_pool_case``."""
    from mindie_sim import target_graphs
    return [
        op
        for op in target_graphs(
            raw, scope="body", phase="decode", tokens=query, context=context, sequences=batch
        )[0]
        if op.layer == 3
    ]
