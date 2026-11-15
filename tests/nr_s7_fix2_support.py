"""Shared fixtures for the NR-S7 FIX2 regression witnesses.

[Role]
  Build the catalog cases, option records, linear layers and fully resolved
  cache signatures the FIX2 regression modules reuse.

[Invariants]
  Every helper builds immutable declared records only; no ambient host value
  enters a case or an option.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device, subprocess or network access.

[Side-Effect Policy]
  Builds fresh records; creates no files.
"""
from __future__ import annotations

import dataclasses as dc
from functools import lru_cache

from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.dspark_kv_physics import KvSpecSignature
from mindie_sim.dspark_prefix_cache import (
    DSparkPrefixPage, ModelIdentity, dspark_prefix_keys,
    dspark_prefix_page_bytes)
from mindie_sim.hybrid_kv_state_options import HybridKvStateOptions
from mindie_sim.linear_state_physics import LinearLayer
from mindie_sim.prediction.structure import parse_case
from mindie_sim.recoverssm_config import RecoverSsmConfig


@lru_cache(maxsize=None)
def _cached_case(key, tp, pp, pcp, dcp, max_model_len, prompt, sku, topology):
    runtime = {"tp": tp, "dp": 1, "ep": 1, "pp": pp, "pcp": pcp, "cp": dcp,
               "max_model_len": max_model_len, "max_num_seqs": 8,
               "max_num_batched_tokens": 16384, "kv_block_size": 128}
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": 64,
                    "ranks_per_node": 16, "topology_preset": topology},
        "model": {"catalog_key": key},
        "runtime": runtime,
        "workload": {"requests": [{"request_id": "r",
                                   "prompt_tokens": prompt,
                                   "output_tokens": 16}]}})


def _case(key="glm-5.3-flash", *, tp=8, pp=1, pcp=1, dcp=1,
          max_model_len=4096, memory=None, prompt=1024,
          sku="Ascend910_9362", topology="a3_16p"):
    if memory is None:
        return _cached_case(key, tp, pp, pcp, dcp, max_model_len, prompt, sku, topology)
    runtime = {"tp": tp, "dp": 1, "ep": 1, "pp": pp, "pcp": pcp, "cp": dcp,
               "max_model_len": max_model_len, "max_num_seqs": 8,
               "max_num_batched_tokens": 16384, "kv_block_size": 128,
               "memory": memory}
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": 64,
                    "ranks_per_node": 16, "topology_preset": topology},
        "model": {"catalog_key": key},
        "runtime": runtime,
        "workload": {"requests": [{"request_id": "r",
                                   "prompt_tokens": prompt,
                                   "output_tokens": 16}]}})


def _glm(**overrides):
    values = dict(use_replayssm=True, num_spec=4, mamba_cache_mode="none",
                  enable_prefix_caching=False, batch=8, sequence_tokens=2048,
                  accept_rate=0.8, kernel_launch_ms=0.02,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


def _kimi(**overrides):
    values = dict(use_replayssm=False, num_spec=3, mamba_cache_mode="align",
                  enable_prefix_caching=True, batch=2, sequence_tokens=4096,
                  prefix_hit_tokens=1024, kernel_launch_ms=0.02,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


def _ds41(**overrides):
    values = dict(use_replayssm=False, num_spec=3, mamba_cache_mode="align",
                  enable_prefix_caching=True, batch=2, sequence_tokens=4096,
                  prefix_hit_tokens=1024, swa_bounded_replay=True,
                  kernel_launch_ms=0.02, dspark_active=True,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


def _layer(**overrides):
    values = dict(name="layer0", kind="kda", index=0, num_heads=64,
                  key_dim=128, value_dim=128, conv_kernel=4, conv_width=24576,
                  activation_dtype="bf16", recurrent_elements=64 * 128 * 128,
                  conv_history_elements=24576 * 3)
    values.update(overrides)
    return LinearLayer(**values)


def _config(**overrides):
    from mindie_sim.recoverssm_config import RecoverSsmConfig
    values = dict(use_replayssm=True, speculative_k=4,
                  enable_prefix_caching=False, mamba_cache_mode="none",
                  tp=8, model_type="glm5_next", kv_block_size=128)
    values.update(overrides)
    return RecoverSsmConfig(**values)


def _kda_layers(case):
    return [plan for plan in case.model.layer_plans
            if plan.attention in {"kda", "gdn"}]



def _target(index):
    return f"language_model.model.layers.{index}.cache"


def _draft(index):
    return f"model.layers.{index}.cache"


def _spec(name, kind="mamba", *, scope="target", page=147_456,
          noncausal=False, heads=64, kv_heads=8, head_dim=128,
          shapes=((8, 128, 128),), dtypes=("fp32",)):
    attention = kind != "mamba"
    return KvSpecSignature(
        name, kind, 128, page, scope, "kda",
        noncausal_query_block=noncausal,
        state_shapes=shapes if not attention else (),
        state_dtypes=dtypes if not attention else (),
        speculative_slots=4 if not attention else 0,
        num_heads=heads if attention else 0,
        num_kv_heads=kv_heads if attention else 0,
        head_dim=head_dim if attention else 0,
        head_dtype="bf16")


def _base_specs():
    return [_spec(_target(0), "full_attention"),
            _spec(_target(1), "full_attention"),
            _spec(_draft(0), "full_attention", scope="draft"),
            _spec(_target(10)), _spec(_target(11)), _spec(_target(12))]


def _dcp(**overrides):
    values = dict(draft_block_size=5, context_tokens=4096, dcp=2, pcp=1,
                  tp=8, num_q_heads=64, q_nope_dim=512, q_rope_dim=64,
                  kv_heads=16, head_dim=64, latent_width=512,
                  num_draft_layers=5, batch=1, swa_window=128,
                  collective_bytes_per_s=1.0e12, hbm_bytes_per_s=1.0e12,
                  compute_flops_per_s=4.0e14, index_entries_per_us=1000.0)
    values.update(overrides)
    return values


def _identity(name="m", dtype="bf16"):
    return ModelIdentity(name, dtype, "bf16")


def _geometry():
    return dspark_prefix_page_bytes(
        pages=[DSparkPrefixPage("d", 0, 1, 128, 128, 1, 128, 1)],
        num_blocks=1, block_size=128, hash_block_size=128, group_ids=(0,),
        block_sizes=(128,), draft_group_ids=(0,),
        require_hash_equals_block=True)


def _keys():
    return dspark_prefix_keys(
        target=_identity(), draft=_identity(), aux_layers=(1,), tp_size=1,
        pp_size=1, block_size=128, prefix_match_unit=128, draft_width=5,
        dcp_size=2, include_dcp_in_key=True)


__all__ = ["_case", "_config", "_dcp", "_draft", "_ds41", "_geometry",
           "_glm", "_identity", "_kda_layers", "_keys", "_kimi", "_layer",
           "_spec", "_target"]
