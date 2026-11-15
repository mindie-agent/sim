"""Canonical runtime/parallel feature normalization and CLI preservation."""
from __future__ import annotations

import argparse
import json

import pytest

from mindie_sim.issue34_cli import add_issue34_arguments, issue34_overrides
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.runtime_features import (
    normalize_case_runtime, normalize_parallel_features, normalize_runtime_features,
    split_routing_features, validate_runtime_features,
)


def test_parallel_alias_is_canonical_and_conflicts_fail():
    spec = ParallelSpec.from_mapping({
        "kda_proj_tp": 4, "o_proj_tp": 2, "lm_head_tp": 8,
        "redundant_experts_per_rank": 1})
    assert spec.linear_proj_tp == 4 and spec.o_proj_tp == 2
    encoded = spec.to_dict()
    assert "kda_proj_tp" not in encoded and encoded["linear_proj_tp"] == 4
    assert ParallelSpec.from_mapping(encoded) == spec
    with pytest.raises(ModelingError, match="runtime_feature_conflict:linear_proj_tp"):
        ParallelSpec.from_mapping({"kda_proj_tp": 2, "linear_proj_tp": 3})
    for bad in (True, -1, 1.5, "4", None):
        with pytest.raises(ModelingError):
            normalize_parallel_features({"o_proj_tp": bad})


def test_modes_are_normalized_and_validated():
    result = normalize_runtime_features({
        "collective_startup_mode": "direct",
        "prefix_cache_mode": "SHARED",
        "prefix_read_mode": "batch_shared",
        "homogeneous_graph_reuse": "auto",
    })
    assert result["collective_startup_mode"] == "concurrent"
    assert result["prefix_cache_mode"] == "shared"
    with pytest.raises(ModelingError, match="runtime_feature_unsupported:collective_startup_mode"):
        normalize_runtime_features({"collective_startup_mode": "broadcast"})
    with pytest.raises(ModelingError, match="runtime_recover_ssm_invalid"):
        normalize_runtime_features({"recover_ssm": "true"})


def test_routing_slot_alias_is_lifted_without_mutating_input():
    raw = {
        "recover_ssm": True,
        "parallel": {"tp": 2},
        "routing": {"policy": "balanced", "redundant_experts_per_rank": 2,
                    "expert_slot_policy": "explicit"},
    }
    normalized = normalize_runtime_features(raw)
    assert normalized["parallel"]["redundant_experts_per_rank"] == 2
    assert "redundant_experts_per_rank" not in normalized["routing"]
    assert raw["routing"]["redundant_experts_per_rank"] == 2
    logical, slots = split_routing_features(raw["routing"])
    assert logical == {"policy": "balanced"}
    assert slots["expert_slot_policy"] == "explicit"
    with pytest.raises(ModelingError, match="runtime_feature_conflict"):
        normalize_runtime_features({
            "parallel": {"redundant_experts_per_rank": 1},
            "routing": {"redundant_experts_per_rank": 2}})


def test_top_level_software_alias_accepts_only_recover_ssm():
    merged = normalize_case_runtime({"software": {"recover_ssm": True}})
    assert merged["runtime"]["recover_ssm"] is True
    with pytest.raises(ModelingError, match="case_software_alias_unknown"):
        normalize_case_runtime({"software": {"max_num_seqs": 4}})
    with pytest.raises(ModelingError, match="runtime_feature_conflict:recover_ssm"):
        normalize_case_runtime({"runtime": {"recover_ssm": False},
                                "software": {"recover_ssm": True}})


def test_software_roundtrip_preserves_canonical_fields_and_world_size():
    software = Software.from_mapping({
        "tp": 2, "collective_startup_mode": "log2",
        "prefix_cache_mode": "shared", "prefix_read_mode": "per_sequence",
        "homogeneous_graph_reuse": "require",
        "parallel": {"o_proj_tp": 2, "linear_proj_tp": 4, "lm_head_tp": 2},
    })
    validate_runtime_features(software)
    roundtripped = Software.from_mapping(software.to_dict())
    assert roundtripped == software
    assert roundtripped.parallel.world_size == 2


def test_cli_omitted_values_do_not_overwrite_case_declarations():
    parser = argparse.ArgumentParser()
    add_issue34_arguments(parser, command="compose")
    omitted = parser.parse_args([])
    assert issue34_overrides(omitted) == {}
    supplied = parser.parse_args(["--o-proj-tp", "2", "--linear-proj-tp", "4",
                                  "--recover-ssm", "--memory-gib", "1"])
    values = issue34_overrides(supplied)
    assert values["o_proj_tp"] == 2 and values["linear_proj_tp"] == 4
    assert values["recover_ssm"] is True
    assert values["memory_bytes"] == 2 ** 30
    with pytest.raises(ModelingError, match="runtime_feature_conflict:linear_proj_tp"):
        issue34_overrides(parser.parse_args([
            "--linear-proj-tp", "2", "--kda-proj-tp", "3"]))
