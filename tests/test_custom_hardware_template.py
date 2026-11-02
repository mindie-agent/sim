"""Tests for neutral custom hardware templates and private SKU composition.

[Role]
  Exercise the five exported structural archetypes, partial-spec machine
  materialization, registry/file/environment workflows, and end-to-end custom
  case prediction through the shared engine.

[Invariants]
  * Every exported template carries complete field documentation, and every
    builder result parses through the neutral ``Hardware`` schema.
  * Cube/MTE/link derivations preserve physical direction under key-spec
    overrides.
  * Registry names and aliases resolve case-insensitively, while environment
    discovery never overwrites an explicitly registered in-memory SKU.
  * Composed custom cases run through ``predict`` and
    ``analyze_network_efficiency`` without a vendor profile entry.

[Timeouts & Locks]
  Tier 0 in-memory and ``tmp_path`` JSON tests; no device access or locks.

[Side-Effect Policy]
  Writes only pytest temporary files and resets the process-local registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mindie_sim import predict
from mindie_sim.custom_hardware import (
    HARDWARE_ARCHETYPE_TEMPLATES,
    build_custom_hardware_machine,
    clear_custom_hardware_skus,
    compose_custom_case,
    export_hardware_sku_template,
    list_custom_hardware_skus,
    load_custom_hardware_skus_from_file,
    parse_hardware,
    register_custom_hardware_sku,
    resolve_custom_hardware_sku,
    unregister_custom_hardware_sku,
)
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


MODEL = "deepseek-v3.2-prune-3l"
MOCK_MINIMAL: dict[str, Any] = {
    "sku": "MOCK_CONFIDENTIAL_NPU",
    "hbm_gb": 192,
    "hbm_bw_gbps": 6000,
    "cube_fp16_tflops": 1000,
}
REQUIRED_DOC_KEYS = {
    "hbm_gb", "hbm_bw_gbps", "cube_fp16_tflops", "cube_fp8_tflops",
    "cube_int4_tflops", "intra_bw_4p_gbps", "intra_bw_8p_gbps",
    "intra_node_bw_gbps", "superpod_size", "superpod_bw_gbps",
    "inter_node_bw_gbps", "inter_pod_bw_gbps", "intra_node_bw_util",
    "inter_node_bw_util", "intra_node_latency_us", "inter_node_latency_us",
}
ARCHETYPES = tuple(HARDWARE_ARCHETYPE_TEMPLATES)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MINDIE_SIM_CUSTOM_SKUS", raising=False)
    clear_custom_hardware_skus()
    yield
    clear_custom_hardware_skus()


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_export_hardware_templates_for_all_archetypes(archetype: str) -> None:
    exported = export_hardware_sku_template(archetype)
    expected = HARDWARE_ARCHETYPE_TEMPLATES[archetype]
    assert exported["sku"] == "CUSTOM_SKU_TEMPLATE"
    assert exported["archetype"] == archetype
    for key, value in expected.items():
        assert exported[key] == value
    docs = exported["_field_docs"]
    assert REQUIRED_DOC_KEYS <= set(docs)
    assert set(expected) <= set(docs)
    assert all(isinstance(text, str) and text for text in docs.values())
    original_doc = docs["hbm_gb"]
    hint = exported["_agent_usage_hint"]
    assert "register_custom_hardware_sku" in hint
    assert "compose_custom_case" in hint
    assert "MINDIE_SIM_CUSTOM_SKUS" in hint

    exported["hbm_gb"] = -1.0
    docs["hbm_gb"] = "mutated"
    fresh = export_hardware_sku_template(archetype)
    assert fresh["hbm_gb"] == expected["hbm_gb"]
    assert fresh["_field_docs"]["hbm_gb"] == original_doc

    compact = export_hardware_sku_template(archetype, include_schema_docs=False)
    assert "_field_docs" not in compact
    assert "_agent_usage_hint" not in compact
    assert compact["hbm_gb"] == expected["hbm_gb"]


def test_export_rejects_unknown_archetype() -> None:
    with pytest.raises(ModelingError, match="unknown_hardware_archetype"):
        export_hardware_sku_template("no_such_archetype")


def test_minimal_spec_materializes_parseable_hardware() -> None:
    machine = build_custom_hardware_machine(MOCK_MINIMAL, num_devices=8)
    hardware = parse_hardware(machine)
    assert hardware.sku == "MOCK_CONFIDENTIAL_NPU"
    assert hardware.arch_family == "Custom_Ascend_UB"
    assert hardware.hardware_key == "MOCK_CONFIDENTIAL_NPU"
    assert len(hardware.dies) == 8
    assert len(machine["dies"]) == 8
    assert machine["num_devices"] == 8
    assert machine["ranks_per_node"] == 64
    assert machine["dies_per_package"] == 2
    assert [item["rank"] for item in machine["dies"]] == list(range(8))
    assert [item["package"] for item in machine["dies"]] == [0, 0, 1, 1, 2, 2, 3, 3]

    die = hardware.dies[0]
    assert die.memory_bytes == 192_000_000_000
    assert die.peak("bf16") == pytest.approx(1000e12)
    assert die.peak("fp16") == pytest.approx(1000e12)
    assert die.peak("fp8") == pytest.approx(2000e12)
    assert die.peak("fp8_e4m3") == pytest.approx(2000e12)
    assert die.peak("int8") == pytest.approx(2000e12)
    rates = dict(die.rates)
    assert rates["hbm"] == pytest.approx(6000e9)
    assert die.compute_units == 32
    assert dict(die.buffers)["ub"] == 256 * 1024
    assert dict(die.buffers)["l2"] == 192 * 1024**2
    assert dict(die.resource_units)["vector"] == 64

    channels = {link.channel for link in hardware.links}
    assert {"intra_package", "intra_node", "superpod", "inter_node"} <= channels
    assert "inter_pod" not in channels
    assert all(link.capacity_bytes_per_ms > 0 for link in hardware.links)
    assert any(link.channel == "intra_package" and len(link.members) == 2
               for link in hardware.links)
    assert any(link.channel == "superpod" and len(link.members) == 8
               for link in hardware.links)

    high_hbm = build_custom_hardware_machine(
        {**MOCK_MINIMAL, "hbm_bw_gbps": 6000}, num_devices=1,
    )["device"]["rates"]
    low_hbm = build_custom_hardware_machine(
        {**MOCK_MINIMAL, "hbm_bw_gbps": 3000}, num_devices=1,
    )["device"]["rates"]
    for role in ("mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe"):
        assert high_hbm[role] == pytest.approx(2.0 * low_hbm[role])


def test_derived_rates_and_explicit_overrides() -> None:
    spec = {
        **MOCK_MINIMAL,
        "cube_fp8_tflops": 700,
        "cube_int4_tflops": 400,
        "mte1_bw_gbps": 1234,
        "mte2_bw_gbps": 4321,
        "mte3_bw_gbps": 999,
        "fixpipe_bw_gbps": 888,
        "_field_docs": {"ignored": "yes"},
        "_agent_usage_hint": "ignored",
    }
    machine = build_custom_hardware_machine(spec, num_devices=1)
    die = parse_hardware(machine).dies[0]
    assert die.peak("fp8") == pytest.approx(700e12)
    assert die.peak("int4") == pytest.approx(400e12)
    rates = dict(die.rates)
    assert rates["mte1"] == pytest.approx(1234e9)
    assert rates["mte2"] == pytest.approx(4321e9)
    assert rates["mte2_aic"] == pytest.approx(4321e9)
    assert rates["mte2_aiv"] == pytest.approx(4321e9)
    assert rates["mte3"] == pytest.approx(999e9)
    assert rates["fixpipe"] == pytest.approx(888e9)

    fp4 = build_custom_hardware_machine(
        {**MOCK_MINIMAL, "fp4_tflops": 250}, num_devices=1,
    )
    assert parse_hardware(fp4).dies[0].peak("int4") == pytest.approx(250e12)


@pytest.mark.parametrize(
    "patch",
    [
        {"hbm_gb": 0},
        {"hbm_bw_gbps": 0},
        {"cube_fp16_tflops": 0},
        {"vector_fp32_tflops": 0},
        {"ranks_per_node": 0},
        {"intra_node_bw_util": 0.0},
        {"inter_node_bw_util": 1.01},
        {"intra_node_bw_gbps": -1.0},
        {"archetype": "no_such_archetype"},
        {"cube_fp8_tflops": -2},
    ],
)
def test_invalid_specs_raise_modeling_error(patch: dict[str, Any]) -> None:
    with pytest.raises(ModelingError):
        build_custom_hardware_machine({**MOCK_MINIMAL, **patch}, num_devices=4)


def test_builder_rejects_invalid_num_devices_and_archetype() -> None:
    for count in (0, -1, True, 1.5):
        with pytest.raises(ModelingError):
            build_custom_hardware_machine(MOCK_MINIMAL, num_devices=count)
    with pytest.raises(ModelingError, match="unknown_hardware_archetype"):
        build_custom_hardware_machine(MOCK_MINIMAL, archetype="no_such_archetype")


def test_link_hierarchy_and_superpod_links() -> None:
    two_node = build_custom_hardware_machine(
        {**MOCK_MINIMAL, "ranks_per_node": 2}, num_devices=8,
    )
    links = parse_hardware(two_node).links
    node_links = [link for link in links if link.channel == "intra_node"]
    inter_links = [link for link in links if link.channel == "inter_node"]
    assert len(node_links) == 4
    assert all(len(link.members) == 2 for link in node_links)
    assert len(inter_links) == 1
    assert len(inter_links[0].members) == 8
    assert node_links[0].capacity_bytes_per_ms == pytest.approx(800 * 0.85 * 1e6)
    assert inter_links[0].capacity_bytes_per_ms == pytest.approx(800 * 0.80 * 1e6)

    clustered = build_custom_hardware_machine(
        {**MOCK_MINIMAL, "ranks_per_node": 2, "superpod_size": 4}, num_devices=8,
    )
    clustered_channels = {link.channel for link in parse_hardware(clustered).links}
    assert {"superpod", "inter_pod"} <= clustered_channels
    assert "inter_node" not in clustered_channels
    assert clustered["superpod_size"] == 4
    assert clustered["max_dies"] == 4
    superpod_members = [link.members for link in parse_hardware(clustered).links
                        if link.channel == "superpod"]
    assert sorted(len(members) for members in superpod_members) == [4, 4]

    server = build_custom_hardware_machine(
        {}, archetype="ascend_dual_die_server", num_devices=16,
    )
    server_channels = {link.channel for link in parse_hardware(server).links}
    assert {"intra_package", "intra_4p", "intra_8p", "intra_node"} <= server_channels
    four_p = [link for link in parse_hardware(server).links
              if link.channel == "intra_4p"]
    assert len(four_p) == 2
    assert all(len(link.members) == 8 for link in four_p)


def test_registry_register_resolve_unregister_clear() -> None:
    profile = register_custom_hardware_sku(
        "MOCK_PRIVATE_NPU",
        {"hbm_gb": 96, "hbm_bw_gbps": 3000, "cube_fp16_tflops": 500},
        archetype="generic_accelerator",
        aliases=("MOCK_PRIVATE_ALIAS",),
    )
    assert profile["sku"] == "MOCK_PRIVATE_NPU"
    assert profile["archetype"] == "generic_accelerator"
    assert profile["hbm_gb"] == 96

    listed = list_custom_hardware_skus()
    assert {"MOCK_PRIVATE_NPU", "mock_private_npu",
            "MOCK_PRIVATE_ALIAS", "mock_private_alias"} <= set(listed)
    assert resolve_custom_hardware_sku("mock_private_npu")["hbm_gb"] == 96
    assert resolve_custom_hardware_sku("MOCK_PRIVATE_ALIAS")["hbm_gb"] == 96
    assert resolve_custom_hardware_sku("missing_sku") is None

    resolved = resolve_custom_hardware_sku("MOCK_PRIVATE_NPU")
    assert resolved is not None
    resolved["hbm_gb"] = 1
    assert resolve_custom_hardware_sku("MOCK_PRIVATE_NPU")["hbm_gb"] == 96

    assert unregister_custom_hardware_sku("mock_private_alias") is True
    assert resolve_custom_hardware_sku("MOCK_PRIVATE_NPU") is None
    assert unregister_custom_hardware_sku("MOCK_PRIVATE_NPU") is False

    register_custom_hardware_sku("MOCK_ONE", MOCK_MINIMAL)
    register_custom_hardware_sku("MOCK_TWO", MOCK_MINIMAL)
    assert len(list_custom_hardware_skus()) == 4
    clear_custom_hardware_skus()
    assert list_custom_hardware_skus() == {}

    with pytest.raises(ModelingError, match="unknown_hardware_archetype"):
        register_custom_hardware_sku("MOCK_BAD", MOCK_MINIMAL, archetype="bogus")
    with pytest.raises(ModelingError):
        register_custom_hardware_sku("MOCK_BAD", {"hbm_gb": 0})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_file_and_environment_sku_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = tmp_path / "direct.json"
    _write_json(direct, {
        "MOCK_FILE_NPU": {
            "hbm_gb": 80, "hbm_bw_gbps": 2500, "cube_fp16_tflops": 600,
        },
    })
    loaded = load_custom_hardware_skus_from_file(direct)
    assert set(loaded) == {"MOCK_FILE_NPU"}
    assert loaded["MOCK_FILE_NPU"]["sku"] == "MOCK_FILE_NPU"
    assert resolve_custom_hardware_sku("mock_file_npu")["hbm_gb"] == 80

    nested = tmp_path / "nested.json"
    _write_json(nested, {"skus": {
        "MOCK_FILE_A": {"hbm_gb": 96, "hbm_bw_gbps": 3000,
                        "cube_fp16_tflops": 700},
        "MOCK_FILE_B": {"hbm_gb": 64, "hbm_bw_gbps": 2000,
                        "cube_fp16_tflops": 400,
                        "aliases": ["MOCK_FILE_B_ALIAS"]},
    }})
    loaded_nested = load_custom_hardware_skus_from_file(nested)
    assert set(loaded_nested) == {"MOCK_FILE_A", "MOCK_FILE_B"}
    assert resolve_custom_hardware_sku("MOCK_FILE_B_ALIAS")["hbm_gb"] == 64

    clear_custom_hardware_skus()
    register_custom_hardware_sku(
        "MOCK_ENV_NPU",
        {"hbm_gb": 12, "hbm_bw_gbps": 500, "cube_fp16_tflops": 100},
    )
    env_file = tmp_path / "private_skus.json"
    _write_json(env_file, {
        "MOCK_ENV_NPU": {"hbm_gb": 99, "hbm_bw_gbps": 900,
                         "cube_fp16_tflops": 900},
        "MOCK_ENV_ONLY": {"hbm_gb": 32, "hbm_bw_gbps": 1000,
                          "cube_fp16_tflops": 200},
    })
    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(env_file))
    assert resolve_custom_hardware_sku("MOCK_ENV_NPU")["hbm_gb"] == 12
    assert resolve_custom_hardware_sku("MOCK_ENV_ONLY")["hbm_gb"] == 32
    assert {"MOCK_ENV_NPU", "MOCK_ENV_ONLY"} <= set(list_custom_hardware_skus())

    clear_custom_hardware_skus()
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(missing))
    assert list_custom_hardware_skus() == {}
    assert resolve_custom_hardware_sku("MOCK_ENV_ONLY") is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelingError):
        load_custom_hardware_skus_from_file(malformed)


def test_compose_custom_case_inline_and_registered() -> None:
    case = compose_custom_case(
        MODEL,
        MOCK_MINIMAL,
        tp=2, dp=2, ep=4,
        prompt_tokens=128, output_tokens=8, num_requests=2,
        max_num_seqs=2, max_num_batched_tokens=512, max_model_len=512,
    )
    assert case["machine"]["sku"] == "MOCK_CONFIDENTIAL_NPU"
    assert case["machine"]["num_devices"] == 4
    assert len(case["workload"]["requests"]) == 2
    assert case["runtime"]["parallel"]["cp"] == 1
    parsed = parse_case(case)
    assert parsed.machine.sku == "MOCK_CONFIDENTIAL_NPU"
    assert len(parsed.machine.dies) == 4

    prediction = predict(
        case, scope="body", phase="decode", tokens=1, context=2048, sequences=1,
    )
    assert prediction["duration_ms"] > 0.0
    efficiency = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=2048, sequences=1,
    )
    assert efficiency["operator_sum_ms"] > 0.0

    register_custom_hardware_sku(
        "MOCK_REGISTERED_NPU",
        {**MOCK_MINIMAL, "sku": "MOCK_REGISTERED_NPU", "hbm_gb": 96},
        aliases=("MOCK_REGISTERED_ALIAS",),
    )
    registered_case = compose_custom_case(
        MODEL, "mock_registered_alias", tp=1, dp=1, ep=1,
        prompt_tokens=64, output_tokens=4, num_requests=1,
        max_num_seqs=1, max_num_batched_tokens=256, max_model_len=256,
    )
    assert registered_case["machine"]["sku"] == "MOCK_REGISTERED_NPU"
    assert registered_case["machine"]["num_devices"] == 1
    assert predict(
        registered_case, scope="body", phase="decode", tokens=1,
        context=1024, sequences=1,
    )["duration_ms"] > 0.0
    assert analyze_network_efficiency(
        registered_case, phase="decode", tokens=1, context=1024, sequences=1,
    )["operator_sum_ms"] > 0.0

    overridden = compose_custom_case(
        MODEL, "MOCK_REGISTERED_NPU", tp=1, dp=1,
        prompt_tokens=64, output_tokens=4, num_requests=1,
        max_num_seqs=1, max_num_batched_tokens=256, max_model_len=256,
        hbm_gb=32,
    )
    assert overridden["machine"]["device"]["memory_bytes"] == 32_000_000_000

    with pytest.raises(ModelingError, match="unknown_custom_hardware_sku"):
        compose_custom_case(MODEL, "MOCK_MISSING_SKU")
    with pytest.raises(ModelingError, match="num_devices"):
        compose_custom_case(MODEL, MOCK_MINIMAL, num_devices=8)


def test_custom_hardware_scaling_direction() -> None:
    compute_base = {**MOCK_MINIMAL, "sku": "MOCK_SCALE_NPU"}
    fast = compose_custom_case(
        MODEL, {**compute_base, "cube_fp16_tflops": 2000},
        tp=1, dp=1, prompt_tokens=1024, output_tokens=16, num_requests=1,
        max_num_seqs=1, max_num_batched_tokens=2048,
    )
    slow = compose_custom_case(
        MODEL, {**compute_base, "cube_fp16_tflops": 250},
        tp=1, dp=1, prompt_tokens=1024, output_tokens=16, num_requests=1,
        max_num_seqs=1, max_num_batched_tokens=2048,
    )
    fast_ms = predict(
        fast, scope="body", phase="prefill", tokens=1024, context=1024,
        sequences=1,
    )["duration_ms"]
    slow_ms = predict(
        slow, scope="body", phase="prefill", tokens=1024, context=1024,
        sequences=1,
    )["duration_ms"]
    assert fast_ms < slow_ms

    network_base = {
        **MOCK_MINIMAL, "sku": "MOCK_NETWORK_NPU", "ranks_per_node": 2,
    }
    high_util = compose_custom_case(
        MODEL, {**network_base, "inter_node_bw_util": 1.0},
        tp=8, dp=1, ep=8, prompt_tokens=1024, output_tokens=16,
        num_requests=1, max_num_seqs=1, max_num_batched_tokens=2048,
    )
    low_util = compose_custom_case(
        MODEL, {**network_base, "inter_node_bw_util": 0.2},
        tp=8, dp=1, ep=8, prompt_tokens=1024, output_tokens=16,
        num_requests=1, max_num_seqs=1, max_num_batched_tokens=2048,
    )
    high_prediction = predict(
        high_util, scope="body", phase="prefill", tokens=1024, context=1024,
        sequences=1,
    )
    low_prediction = predict(
        low_util, scope="body", phase="prefill", tokens=1024, context=1024,
        sequences=1,
    )
    assert low_prediction["duration_ms"] > high_prediction["duration_ms"]

    high_eff = analyze_network_efficiency(
        high_util, phase="prefill", tokens=1024, context=1024, sequences=1,
    )
    low_eff = analyze_network_efficiency(
        low_util, phase="prefill", tokens=1024, context=1024, sequences=1,
    )
    assert low_eff["total_link_bytes"] == high_eff["total_link_bytes"]
    assert low_eff["total_link_ideal_ms"] > high_eff["total_link_ideal_ms"]
    assert low_eff["network_link_util"] > high_eff["network_link_util"]


def test_compose_case_direct_integration_with_custom_sku() -> None:
    from mindie_sim import compose_case, resolve_hardware_sku

    register_custom_hardware_sku(
        "MOCK_DIRECT_COMPOSE_NPU",
        {"hbm_gb": 192.0, "hbm_bw_gbps": 5600.0, "cube_fp16_tflops": 1200.0},
        aliases=("mock-direct-npu",),
    )
    assert resolve_hardware_sku("mock-direct-npu") == "MOCK_DIRECT_COMPOSE_NPU"
    case_from_alias = compose_case(
        MODEL,
        hardware="mock-direct-npu",
        tp=2,
        dp=1,
        prompt_tokens=256,
        output_tokens=16,
        inter_node_bw_util=0.75,
    )
    assert case_from_alias["machine"]["sku"] == "MOCK_DIRECT_COMPOSE_NPU"
    assert predict(case_from_alias, scope="body", phase="decode", tokens=1, context=256, sequences=1)["duration_ms"] > 0

    case_from_inline = compose_case(
        MODEL,
        hardware={
            "sku": "MOCK_INLINE_UNREGISTERED_NPU",
            "archetype": "ascend_dual_die_server",
            "hbm_gb": 128.0,
            "hbm_bw_gbps": 3200.0,
            "cube_fp16_tflops": 800.0,
        },
        tp=2,
        dp=1,
        prompt_tokens=256,
        output_tokens=16,
    )
    assert case_from_inline["machine"]["sku"] == "MOCK_INLINE_UNREGISTERED_NPU"
    assert predict(case_from_inline, scope="body", phase="decode", tokens=1, context=256, sequences=1)["duration_ms"] > 0

    sp_machine = build_custom_hardware_machine(
        {
            "sku": "MOCK_SUPERPOD_64P_TIE",
            "archetype": "ascend_dual_die_server",
            "hbm_gb": 128.0,
            "hbm_bw_gbps": 3200.0,
            "cube_fp16_tflops": 800.0,
            "ranks_per_node": 16,
            "superpod_size": 64,
            "superpod_bw_gbps": 400.0,
            "inter_node_bw_gbps": 50.0,
        },
        num_devices=64,
    )
    names = [link["name"] for link in sp_machine["links"] if len(link["members"]) == 64]
    assert "superpod:0" in names
    assert all(name >= "superpod:0" for name in names)

    pkg_throttled = build_custom_hardware_machine(
        {"intra_package_bw_util": 0.5},
        archetype="ascend_dual_die_server",
        num_devices=4,
    )
    pkg_links = [link for link in parse_hardware(pkg_throttled).links if link.channel == "intra_package"]
    assert pkg_links
    assert pkg_links[0].bw_utilization == pytest.approx(0.5)
    assert pkg_links[0].capacity_bytes_per_ms == pytest.approx(
        pkg_links[0].effective_peak_capacity_bytes_per_ms * 0.5
    )
    with pytest.raises(ModelingError):
        build_custom_hardware_machine(
            {"intra_package_bw_util": 1.5},
            archetype="ascend_dual_die_server",
            num_devices=4,
        )


