"""Dual-plane frontier alignment: structural PCP/DCP and MTP composition.

[Role] Verify that mathematically valid hybrid prefill/decode parallelism is
  not blocked by framework merge status, and that PLE host offload and MTP
  auxiliary structure are exposed as first-class composition primitives.
[Invariants] Assertions inspect declared operators and runtime fields; they do
  not pin a captured duration or a framework commit identity.
[Timeouts & Locks] Pure in-process composition over four-layer catalogs.
[Side-Effect Policy] Read-only; no files, devices or network access.
"""
from __future__ import annotations

import pytest

from mindie_sim import target_graphs
from mindie_sim.compose import compose_case


def _recipes(case, *, phase: str, tokens: int, context: int):
    graph = target_graphs(case, phase=phase, tokens=tokens, context=context,
                          sequences=1)[0]
    return graph, {op.recipe for op in graph}


def test_qwen38_flash_next_pcp_prefill_with_ple_host_offload():
    for pcp in (2, 4):
        case = compose_case(
            "qwen3.8-flash-next", hardware="a5", tp=4, ep=4, pcp=pcp,
            layers=4, engram_cpu_offload=True,
        )
        assert int(case["runtime"]["parallel"]["pcp"]) == pcp
        assert case["runtime"]["engram_cpu_offload"] is True
        rates = case["machine"]["device"]["rates"]
        assert rates["host_memory"] > 0.0
        assert rates["host_vector"] > 0.0
        assert rates["h2d"] > 0.0
        _graph, recipes = _recipes(
            case, phase="prefill", tokens=64, context=128)
        assert "pcp_ple_conv_state_step" in recipes
        assert "pcp_index_gather" in recipes


def test_kimi_k31_pcp_prefill_and_dcp_decode():
    prefill = compose_case(
        "kimi-k3.1", hardware="a5", tp=4, ep=4, pcp=4, layers=4)
    assert int(prefill["runtime"]["parallel"]["pcp"]) == 4
    _graph, recipes = _recipes(
        prefill, phase="prefill", tokens=16, context=128)
    assert "pcp_index_gather" in recipes
    assert "pcp_conv_state_step" in recipes

    decode = compose_case(
        "kimi-k3.1", hardware="a5", tp=4, ep=4, dcp=4, layers=4)
    assert int(decode["runtime"]["parallel"]["cp"]) == 4
    _graph, recipes = _recipes(
        decode, phase="decode", tokens=1, context=1024)
    assert {"dcp_lse_merge", "catalog_dcp_lse_merge"} & recipes


@pytest.mark.parametrize("model", ["deepseek-v3.2", "glm-5.3-flash"])
def test_pooled_indexer_and_hybrid_kda_decode_context_parallelism(model):
    case = compose_case(
        model, hardware="a5", tp=4, ep=4, dcp=2, layers=4)
    assert int(case["runtime"]["parallel"]["cp"]) == 2
    _graph, recipes = _recipes(
        case, phase="decode", tokens=1, context=1024)
    assert "catalog_dcp_lse_merge" in recipes


def test_deepseek_v32_on_demand_nextn_mla_auxiliary():
    case = compose_case(
        "deepseek-v3.2", hardware="a5", tp=4, ep=4, layers=4,
        mtp_proposal_tokens=3,
    )
    auxiliary = case["model"]["auxiliary"]
    assert auxiliary is not None
    mtp = auxiliary["mtp"]
    assert mtp["kind"] == "mtp"
    assert int(mtp["layers"]) == 1
    assert mtp["attention_kind"] == "mla"
    assert case["runtime"]["metadata"]["draft_mode"] == "mtp"
    assert int(case["runtime"]["metadata"]["draft_proposal_tokens"]) == 3
