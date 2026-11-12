"""Source-selected arch35 MC2 protocol contract in the ordinary MoE graph.

[Role]
  Pin the producer declaration boundary for bf16 dispatch_quant=none
  MoeDistributeDispatchV3/CombineV3: optional finite peer window, explicit
  override preservation, and rejection of incompatible caller/quant modes.

[Invariants]
  * A full-mesh protocol without max_inflight_per_rank is valid; the physical
    bound resolves to the actual remote peer count, never a guessed setting.
  * An explicit positive window is preserved exactly.
  * arch22 caller requires per-token INT8 dispatch; unknown kinds/keys and
    invalid windows are rejected before any graph lowering.
  * EP1 cannot use the full-mesh peer protocol.

[Side-Effect Policy] Immutable declarations and pure validation only.
"""
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.direct_peer import DirectPeerCollective
from mindie_sim.prediction.mc2_graph import NETWORK, RECIPE, lower, recipes, select
from mindie_sim.schema.common import ModelingError

NORMAL = {"kind": "arch35_fullmesh", "dispatch_quant": "none"}


def test_normal_bf16_fullmesh_accepts_absent_window_and_derives_topology_bound():
    selected = select({"native_moe_protocol": dict(NORMAL)})
    assert selected["kind"] == "arch35_fullmesh"
    assert selected["dispatch_quant"] == "none"
    assert "max_inflight_per_rank" not in selected
    network = recipes({"native_moe_protocol": dict(NORMAL)})[NETWORK]
    assert isinstance(network, DirectPeerCollective)
    assert network.max_inflight_per_rank is None
    assert network.plan_options == {"max_inflight_per_rank": None}
    assert RECIPE in recipes({"native_moe_protocol": dict(NORMAL)})


def test_explicit_peer_window_is_preserved_exactly():
    selected = select({"native_moe_protocol": {**NORMAL, "max_inflight_per_rank": 4}})
    assert selected["max_inflight_per_rank"] == 4
    network = recipes({"native_moe_protocol": {**NORMAL,
                                             "max_inflight_per_rank": 4}})[NETWORK]
    assert network.max_inflight_per_rank == 4
    assert network.plan_options == {"max_inflight_per_rank": 4}


@pytest.mark.parametrize("raw", (
    {"kind": "arch22_v2_caller", "dispatch_quant": "none"},
    {"kind": "unknown"},
    {"kind": "arch35_fullmesh", "dispatch_quant": "int4"},
    {"kind": "arch35_fullmesh", "dispatch_quant": "none", "extra": 1},
    {"kind": "arch35_fullmesh", "dispatch_quant": "none", "max_inflight_per_rank": 0},
    {"kind": "arch35_fullmesh", "dispatch_quant": "none", "max_inflight_per_rank": True},
    {"kind": "arch35_fullmesh", "dispatch_quant": "none", "max_inflight_per_rank": -2},
))
def test_other_protocol_modes_are_rejected(raw):
    with pytest.raises(ModelingError):
        select({"native_moe_protocol": raw})


def test_arch22_per_token_int8_caller_remains_supported():
    selected = select({"native_moe_protocol": {
        "kind": "arch22_v2_caller", "dispatch_quant": "per_token_int8",
        "max_inflight_per_rank": 2, "global_bs": 8,
    }})
    assert selected["kind"] == "arch22_v2_caller"
    assert selected["dispatch_quant"] == "per_token_int8"


def test_fullmesh_ep1_is_ineligible_before_graph_lowering():
    runtime = SimpleNamespace(kernels={"native_moe_protocol": dict(NORMAL)},
                              moe_recipe="alltoall")
    with pytest.raises(ModelingError):
        lower([], runtime=runtime, prefix="s0.l0", tokens=1, top_k=8,
              width=2048, experts=128, ep=1, source_partitions=1,
              received_assignments=8, dtype="bf16")
