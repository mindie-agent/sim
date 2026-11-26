"""Tier-0 usability, topology, collective, quantization and ingress-contract matrix.

[Role] Verify minimal zero-ceremony SDK/CLI calls, single-source hardware rank
  and bandwidth geometry across every registered Ascend alias, collective band
  selection without silent drops, offline checkpoint/quant/mechanism
  preservation, and fail-fast ``ModelingError`` reporting on contradictory user
  inputs.
[Invariants] Pure in-memory Tier-0 verification; zero network, zero subprocess,
  zero host-environment mutation; every test uses public package entrypoints.
[Timeouts & Locks] Sub-second per case; no external I/O or locks.
[Side-Effect Policy] Read-only except pytest ``tmp_path`` for CLI report smoke.
"""
from __future__ import annotations

import math

import pytest

import mindie_sim.api as api
from mindie_sim import resolve_hardware_sku
from mindie_sim.backends.ascend import (
    default_ranks_per_node,
    materialize as materialize_ascend,
    profiles as ascend_profiles,
)
from mindie_sim.catalog.npu_checkpoints import resolve_checkpoint_metadata
from mindie_sim.stacks.vllm_ascend.speculative import build_va_serving_rules
from mindie_sim.cli import main as cli_main
from mindie_sim.compose import compose_case
from mindie_sim.compose.inputs import HARDWARE_ALIASES
from mindie_sim.prediction.collectives.protocol import lower_collective, select_band
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware_links import CollectiveBandSpec
from mindie_sim.sweep.pd import solve_optimal_deployment, sweep_pd_disaggregation_ratios

pytestmark = pytest.mark.tier0

COLLECTIVES = ("allreduce", "allgather", "reduce_scatter", "alltoall", "alltoallv")
PAYLOADS = (4096, 262144, 4194304)
DIE_COUNTS = (8, 16, 32, 64)


def _link_by_channel(hardware: dict, channel: str) -> dict | None:
    for link in hardware.get("links", ()):
        if link.get("channel") == channel:
            return link
    return None


def test_hardware_alias_matrix_and_ranks_per_node_consistency() -> None:
    assert resolve_hardware_sku("910_9362") == "Ascend910_9362"
    assert default_ranks_per_node("Ascend910_9362") == 16
    assert default_ranks_per_node("Ascend910_9382") == 8
    assert default_ranks_per_node("Ascend950DT_9582") == 8

    registered = ascend_profiles()
    for alias in sorted(HARDWARE_ALIASES):
        sku = HARDWARE_ALIASES[alias]
        if not sku.startswith("Ascend") or sku not in registered:
            continue
        assert resolve_hardware_sku(sku) == sku
        expected_rpn = default_ranks_per_node(sku)
        for num_dies in DIE_COUNTS:
            hw = materialize_ascend({"sku": sku, "num_devices": num_dies})
            assert hw["sku"] == sku
            assert hw["physical_ranks_per_node"] == expected_rpn
            assert hw["ranks_per_node"] == min(num_dies, expected_rpn)
            assert hw["num_devices"] == num_dies


def test_superpod_optical_pcie_and_bandwidth_unit_parity() -> None:
    # A5 SuperPoD >64 dies keeps UB_SUPERPOD instead of dropping to RoCE.
    hw_a5_128 = materialize_ascend({
        "sku": "Ascend950DT_9582", "num_devices": 128, "topology_preset": "a5_pod",
    })
    inter_a5 = hw_a5_128["transfer_channels"]["d2d_inter_node"]
    assert inter_a5["protocol"] == "UB_SUPERPOD"
    assert any("ub_superpod" in b.get("protocol", "") for b in inter_a5["protocol_bands"])

    # Bandwidth-only override on A3 SuperPoD preserves optical HCCS protocol.
    hw_a3_opt = materialize_ascend({
        "sku": "Ascend910_9362", "num_devices": 32,
        "topology_preset": "a3_superpod", "inter_node_bw_gbps": 150.0,
    })
    inter_a3 = hw_a3_opt["transfer_channels"]["d2d_inter_node"]
    assert inter_a3["protocol"] == "HCCS_OPTICAL_SUPERPOD"

    # a5_pcie_16p uses PCIe_Gen5 bands with zero ub_mesh bands.
    hw_pcie = materialize_ascend({
        "sku": "Ascend950DT_9582", "num_devices": 16, "topology_preset": "a5_pcie_16p",
    })
    intra_pcie = hw_pcie["transfer_channels"]["d2d_intra_node"]
    assert intra_pcie["protocol"] == "PCIe_Gen5"
    assert all("ub_mesh" not in b.get("protocol", "") for b in intra_pcie["protocol_bands"])

    # inter_node_bw_gbps is canonical GB/s everywhere (no 8x bits-vs-bytes gap).
    hw_mat = materialize_ascend({
        "sku": "Ascend910_9362", "num_devices": 32, "inter_node_bw_gbps": 200.0,
    })
    composed = compose_case(
        "deepseek-v2-lite", hardware="Ascend910_9362", dp=2, tp=16, ep=1,
        parallel="fixed", inter_node_bw_gbps=200.0,
    )
    mat_link = _link_by_channel(hw_mat, "d2d_inter_node")
    comp_link = _link_by_channel(composed["machine"], "d2d_inter_node")
    assert mat_link is not None and comp_link is not None
    assert math.isclose(mat_link["capacity_bytes_per_ms"], 200.0e6, rel_tol=1e-6)
    assert math.isclose(
        comp_link["capacity_bytes_per_ms"], mat_link["capacity_bytes_per_ms"], rel_tol=1e-6,
    )


def test_single_node_pd_handoff_uses_intra_node_link() -> None:
    for sku, dies in (("Ascend910_9362", 8), ("Ascend950DT_9582", 4)):
        p_case = compose_case(
            "deepseek-v2-lite", hardware=sku, dp=1, tp=dies, ep=dies,
            parallel="fixed", role="pd_prefill", prompt_tokens=1024, output_tokens=64,
            ttft_slo_ms=5000.0,
        )
        d_case = compose_case(
            "deepseek-v2-lite", hardware=sku, dp=1, tp=dies, ep=dies,
            parallel="fixed", role="pd_decode", prompt_tokens=1024, output_tokens=64,
        )
        res = sweep_pd_disaggregation_ratios(
            p_case, d_case, total_dies=dies * 2, prompt_tokens=1024, output_tokens=64,
            ttft_sla_ms=5000.0, tpot_sla_ms=100.0,
        )
        assert res["kv_link_source"] == "sku_link:d2d_intra_node"


def test_collective_band_selection_and_wildcard_normalization() -> None:
    wildcard = CollectiveBandSpec.from_mapping({
        "collective": "any", "stack_id": "*", "expansion_mode": "*",
        "topology_scope": "any", "algorithm": "direct_mesh",
    })
    assert wildcard.collective == "all"
    assert wildcard.stack_id == "default"
    assert wildcard.expansion_mode == "*"
    assert wildcard.topology_scope == "any"

    # A5 UB alltoall/alltoallv under None / concurrent / ccu_ms stay 1-round mesh.
    hw_a5 = materialize_ascend({"sku": "Ascend950DT_9582", "num_devices": 8})
    a5_case = compose_case(
        "deepseek-v2-lite", hardware="Ascend950DT_9582", dp=1, tp=8, ep=8, parallel="fixed",
    )
    a5_parsed = parse_case(a5_case)
    a5_topo = Topology(a5_parsed.machine, a5_parsed.runtime)
    a5_link = next(lnk for lnk in a5_parsed.machine.links if lnk.channel == "d2d_intra_node")
    for coll in ("alltoall", "alltoallv"):
        for mode in (None, "concurrent", "ccu_ms"):
            for payload in PAYLOADS:
                band = select_band(
                    coll, tuple(range(8)), a5_topo, payload, expansion_mode=mode,
                )
                assert band is not None
                assert band.algorithm in {"one_shot_mesh", "direct_mesh"}
                assert "ub_mesh" in band.protocol
    assert a5_topo.p18_allreduce_rate_applies("allreduce", (0, 1, 2, 3), scope="node") is False
    assert hw_a5["sku"] == "Ascend950DT_9582"
    assert a5_link.channel == "d2d_intra_node"

    # A3 SuperPoD optical + SIO + HCCS never drop valid collectives under aiv/ccu_sched.
    from mindie_sim.primitives.loader import load_primitives

    a3_case = compose_case(
        "deepseek-v2-lite", hardware="Ascend910_9362", topology_preset="a3_superpod",
        dp=2, tp=16, ep=32, parallel="fixed",
    )
    a3_parsed = parse_case(a3_case)
    a3_ptab = load_primitives("Ascend910_9362")
    a3_topo = Topology(a3_parsed.machine, a3_parsed.runtime.parallel, primitive_table=a3_ptab)
    opt_link = next(lnk for lnk in a3_parsed.machine.links if lnk.channel == "d2d_inter_node")
    assert opt_link.channel == "d2d_inter_node"
    assert a3_topo.p18_allreduce_rate_applies("allreduce", (0, 1), scope="package") is True
    assert a3_topo.p18_allreduce_rate_applies("allreduce", (0, 1, 2, 3), scope="node") is False
    assert a3_topo.p18_allreduce_rate_applies("allreduce", (0, 16), scope="cross_node") is False
    ep32 = tuple(range(32))
    for coll in COLLECTIVES:
        for payload in PAYLOADS:
            extra_kw: dict = {}
            api_payload = float(payload)
            if coll == "alltoallv":
                per_peer = max(1, int(payload) // (len(ep32) - 1))
                peer_mat = tuple(
                    tuple(0 if r == c else per_peer for c in range(len(ep32)))
                    for r in range(len(ep32))
                )
                splits = tuple(sum(row) for row in peer_mat)
                api_payload = float(max(splits))
                extra_kw = {
                    "peer_matrix": peer_mat,
                    "send_splits": splits,
                    "recv_splits": splits,
                }
            for exp_mode in ("aiv", "ccu_sched", None):
                lowered = lower_collective(
                    coll, ep32, a3_topo, api_payload, expansion_mode=exp_mode, **extra_kw,
                )
                assert lowered.band.algorithm is not None
                assert lowered.total_wire_bytes >= 0.0


def test_checkpoint_quant_and_mechanism_flag_preservation() -> None:
    ckpt_a3 = resolve_checkpoint_metadata("kimi-k3.1", hw_family="a3", allow_network=False)
    assert ckpt_a3 is not None and ckpt_a3.quant_spec is not None
    assert ckpt_a3.quant_spec.kv_dtype == "int8"
    assert ckpt_a3.quant_spec.index_dtype == "int8"

    kimi_w4a8_mxfp = compose_case(
        "kimi-k3.1", hardware="Ascend950DT_9582", quant="w4a8_mxfp",
        dp=1, tp=8, ep=8, parallel="fixed",
    )
    q_dict = kimi_w4a8_mxfp["model"]["quant"]
    assert q_dict["weight_format"] == "fp8_e4m3"
    assert q_dict["expert_weight_format"] == "mxfp4"

    dsv4_a5 = compose_case(
        "deepseek-v4-pro-0813", hardware="Ascend950DT_9582",
        dp=1, tp=8, ep=8, parallel="fixed", kv_cache_dtype="fp8_e4m3",
    )
    assert dsv4_a5["model"]["kv_dtype"] == "fp8_e4m3"
    assert dsv4_a5["runtime"]["kv_block_size"] == 32

    explicit_off = compose_case(
        "kimi-k3.1", hardware="Ascend950DT_9582", dp=1, tp=8, ep=8, parallel="fixed",
        kv_cache_dtype="bf16", recover_ssm=False, enable_torch_compile=False,
    )
    assert explicit_off["model"]["kv_dtype"] == "bf16"
    assert explicit_off["runtime"]["recover_ssm"] is False
    assert explicit_off["runtime"]["enable_torch_compile"] is False

    rules = build_va_serving_rules(
        "deepseek-r1", hw_family="a5", target_kv_dtype="fp8_e4m3",
        pd_disaggregated=True, pd_role="pd_prefill", enable_speculative=True,
        num_speculative_tokens=3,
    )
    assert any(flag == "--kv-cache-dtype fp8" for flag in rules.cli_flags)
    assert rules.speculative_config["num_speculative_tokens"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "deepseek-v2-lite", "hardware": "a3", "quant": "mxfp4", "dp": 1, "tp": 8, "parallel": "fixed"},
        {"model": "deepseek-v2-lite", "hardware": "a3", "topology_preset": "a5_pod", "dp": 1, "tp": 8, "parallel": "fixed"},
        {"model": "deepseek-v2-lite", "hardware": "a5", "topology_preset": "a3_superpod", "dp": 1, "tp": 8, "parallel": "fixed"},
        {"model": "kimi-k3.1", "hardware": "a5", "enable_fused_mc2": 3, "dp": 1, "tp": 8, "ep": 8, "parallel": "fixed"},
        {"model": "qwen3.8-flash-next", "hardware": "a3", "enable_fused_mc2": 3, "dp": 1, "tp": 8, "ep": 8, "parallel": "fixed"},
    ],
)
def test_contradictory_compose_inputs_raise_modeling_error(kwargs: dict) -> None:
    with pytest.raises(ModelingError):
        compose_case(**kwargs)


def test_contradictory_solve_inputs_raise_modeling_error() -> None:
    with pytest.raises(ModelingError):
        solve_optimal_deployment(
            "deepseek-v2-lite", "a3", num_devices=8, prompt_tokens=1024, output_tokens=64,
            ttft_sla_ms=5000.0, tpot_sla_ms=50.0, deployment="colocated", quant="mxfp4",
        )
    with pytest.raises(ModelingError):
        solve_optimal_deployment(
            "deepseek-v2-lite", "a5", num_devices=8, prompt_tokens=1024, output_tokens=64,
            ttft_sla_ms=5000.0, tpot_sla_ms=50.0, deployment="colocated", pinned_parallel={"tpp": 1},
        )
    with pytest.raises(ModelingError):
        solve_optimal_deployment(
            "deepseek-v2-lite", "a5", num_devices=8, prompt_tokens=1024, output_tokens=64,
            ttft_sla_ms=5000.0, tpot_sla_ms=50.0, deployment="pd_disaggregated",
            pd_transfer_mode="layerwise_overlap_d2d", pool_ssd_ratio=0.5,
        )


def test_minimal_sdk_cli_and_infeasible_honesty(tmp_path) -> None:
    # Keyword hardware= works cleanly in api.solve_deployment.
    solved = api.solve_deployment(
        "deepseek-v2-lite", hardware="a5", num_devices=2, layers=2,
        prompt_tokens=1024, output_tokens=64, ttft_slo_ms=5000.0, tpot_slo_ms=50.0,
        deployment="colocated",
    )
    assert solved["feasible"] is True
    assert solved["recommended_deployment"]["best_configuration"] is not None

    # TPOT-only compose_case auto-solves without requiring ttft_slo_ms.
    auto_tpot = compose_case(
        "deepseek-v2-lite", hardware="a5", num_devices=2, layers=2,
        prompt_tokens=1024, output_tokens=64, tpot_slo_ms=50.0,
        deployment="colocated",
    )
    assert auto_tpot["machine"]["sku"] == "Ascend950DT_9582"

    # Impossible SLA returns feasible=False and recommended_case=None (no fake 1x1).
    infeasible = solve_optimal_deployment(
        "deepseek-v2-lite", "a5", num_devices=1, layers=2,
        prompt_tokens=1024, output_tokens=64, ttft_sla_ms=0.01, tpot_sla_ms=0.01,
        deployment="colocated",
    )
    assert infeasible["feasible"] is False
    assert infeasible["recommended_case"] is None

    # Minimal CLI report smoke (explicit --deployment-mode, no xlsx/trace to keep Tier-0 fast).
    out_dir = tmp_path / "min_report"
    rc = cli_main([
        "report", "--model", "deepseek-v2-lite", "--hardware", "a5",
        "--deployment-mode", "colocated",
        "--dp", "1", "--tp", "2", "--ep", "2", "--concurrency", "2",
        "--context", "1024", "--output-tokens", "64",
        "--tpot-ms", "50", "--no-xlsx", "--no-trace", "--out-dir", str(out_dir),
    ])
    assert rc == 0
    assert any(p.name.endswith("_agent_brief.json") for p in out_dir.iterdir())


def test_deterministic_ci_sharding_covers_all_items_disjointly() -> None:
    from _support.sharding import partition_shard_items

    items = [f"tests/pkg/test_mod.py::test_{i:02d}" for i in range(19)]
    shards = [partition_shard_items(items, idx, 4)[0] for idx in range(4)]
    assert sorted(x for shard in shards for x in shard) == sorted(items)
    assert sum(len(shard) for shard in shards) == len(items)
    with pytest.raises(ValueError):
        partition_shard_items(items, 4, 4)
    with pytest.raises(ValueError):
        partition_shard_items(items, 0, 0)


def test_v5_five_content_bugs_regression_locks() -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from mindie_sim.export.action_plan import (
        NEW_KERNEL_MIN_SAVED_MS,
        _public_effective_config,
        _qualify_action_groups,
    )
    from mindie_sim.export.bundle import _collapse_micro_split_ops
    from mindie_sim.export.kernel_catalog import resolve_ascend_kernel_identity
    from mindie_sim.export.models import OperatorExportRow
    from mindie_sim.templates.contracts import _format_dag_equation
    from mindie_sim.prediction.results import OpOccurrence
    from mindie_sim.prediction.runtime.exposed_observer import ObservedInterval, _pick_stream0_witness
    from mindie_sim.rewrite.algebra import AlgebraicTrait, OpTraitInfo, _resolve_symbolic_kind

    # Bug 1: Stream 0 takes witness priority over Stream 1 (MSO) even if Stream 1 is in witness_set.
    ivs = [
        ObservedInterval("main0", "main", 0, 0, 10_000, ordinal=1),
        ObservedInterval("aux0", "aux", 1, 0, 10_000, ordinal=0),
    ]
    assert _pick_stream0_witness(ivs, frozenset({"aux0"})).occurrence_id == "main0"

    # Bug 2a/2b: 69 target_tail RecoverSSM rows collapse to 1 and birth KernelBinding overrides generic Add.
    from mindie_sim.prediction.operators.recurrent.recover_ssm_commit import (
        COMMIT_KERNEL_BACKEND,
        KDA_COMMIT_KERNEL_ID,
    )
    from mindie_sim.schema.operators import KernelBinding, OpSpec

    a5_case_obj = parse_case(compose_case(
        "deepseek-v2-lite", hardware="Ascend950DT_9582", dp=1, tp=2, ep=1, layers=1, parallel="fixed",
    ))
    commit_spec = OpSpec(
        op_id="s0.l0.recover_ssm_commit", family="elementwise", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(4, 128), local_shape=(4, 128), padded_shape=(4, 128),
        launch_shape=(4, 128), flops=0.0, vector_ops=512.0, hbm_bytes=2048.0, mte2_bytes=1024.0,
        mte3_bytes=1024.0, deps=("accept",), recipe="recover_ssm_commit", dtype="bf16",
        kernel_binding=KernelBinding(
            kernel_id=KDA_COMMIT_KERNEL_ID, backend=COMMIT_KERNEL_BACKEND, arch_family="A5",
            submodule="RecoverSSM State Commit", macro_module="attention", readiness="shipping",
            recipe_id="recover_ssm_commit",
        ),
    )
    desc = resolve_ascend_kernel_identity(
        "s0.l0.recover_ssm_commit", commit_spec, "elementwise", "recover_ssm_commit",
        (4, 128), "bf16", a5_case_obj, None, (),
        native_binding={"kernels": [{"name": "Add"}]},
    )
    assert desc.ascend_kernel == KDA_COMMIT_KERNEL_ID and desc.ascend_kernel != "Add"
    rows, demands = [], []
    for idx in range(69):
        oid = f"s0.l{idx}.recover_ssm_commit"
        rows.append(OperatorExportRow(
            ordinal=idx, occurrence=OpOccurrence(0, 0, 0, oid), op_id=oid,
            layer=idx, layer_label=f"L{idx}", submodule="Linear Attention",
            family="elementwise", recipe="recover_ssm_commit", implementation="recover_ssm_commit",
            stream="main", stream_index=0, task_type="AI_VECTOR_CORE",
            shape=(4, 128), dtype="bf16", predecessors=(OpOccurrence(0, 0, 0, "accept"),),
            start_us=100.0 + idx * 1.0, end_us=101.0 + idx * 1.0, dur_us=1.0,
            flops=0.0, vector_ops=0.0, hbm_read_bytes=0.0, hbm_write_bytes=0.0, hbm_total_bytes=0.0,
            mte2_bytes=0.0, mte3_bytes=0.0, fixpipe_bytes=0.0, link_bytes=0.0, l2_hit_pct=0.0,
            dominant_bound="memory_bound", l_star_lean_us=1.0, gap_us=0.0, gap_ratio=1.0,
            gap_radar_rank=0, roofline_efficiency=1.0, absorbed_source_ops=(),
            exposed_dur_us=1.0, archetype_id="target_tail",
            logical_role=desc.logical_role, ascend_kernel=desc.ascend_kernel,
        ))
        demands.append((0, 1000, 0, 0, 0))
    c_rows, c_dems = _collapse_micro_split_ops(rows, demands)
    assert len(c_rows) == 1 and c_rows[0].op_id == "s0.recover_ssm_commit"
    assert c_rows[0].repeat_count == 69
    assert math.isclose(c_rows[0].dur_us * c_rows[0].repeat_count, 69.0) and math.isclose(c_rows[0].start_us, 100.0)
    assert "RecoverSSM" in c_rows[0].logical_role and "(×69 layers)" not in c_rows[0].logical_role and len(c_dems) == 1

    # Bug 3: Sheet 1 [A] effective config hides feature_track & raw moe_route dict while exposing clean scalars.
    pub = _public_effective_config(SimpleNamespace(effective_config={
        "solver_feasible": True, "feature_track": "expected_ideal",
        "moe_route": {"enable_fused_mc2": 3, "multistream_overlap_shared_expert": False},
        "recover_ssm": True,
    }))
    assert "feature_track" not in pub and "moe_route" not in pub
    assert pub["enable_fused_mc2"] == 3 and pub["multistream_overlap_shared_expert"] is False

    # Bug 4: Sub-threshold (<NEW_KERNEL_MIN_SAVED_MS = 0.050 ms) new_groups entries are filtered out under comparison mode.
    assert NEW_KERNEL_MIN_SAVED_MS == pytest.approx(0.050)
    q_new, q_keys = _qualify_action_groups(
        {
            "micro_quant": {"gain_ms": 0.000988, "gain_upt": 0.05, "has_split": False, "gap_ms": 0.001},
            "sub_50us_fusion": {"gain_ms": 0.014, "gain_upt": 0.7, "has_split": False, "gap_ms": 0.02},
            "material_op": {"gain_ms": 0.050, "gain_upt": 1.2, "has_split": False, "gap_ms": 0.06},
        },
        is_comparison=True, check_gap_when_single=False, min_saved_ms=NEW_KERNEL_MIN_SAVED_MS,
    )
    assert [e["gain_ms"] for e in q_new] == [0.050] and q_keys == {"material_op"}

    # Bug 5: Symbolic equations for megamoe_arch35 and moe_gating_topk are accurate.
    k_mm = _resolve_symbolic_kind("grouped_moe", "megamoe_arch35", {}, AlgebraicTrait.BILINEAR_MATMUL, "situ")
    k_rt = _resolve_symbolic_kind("moe_gating", "moe_gating_topk", {}, AlgebraicTrait.ROW_REDUCTION, "")
    assert k_mm == "megamoe" and k_rt == "moe_router_topk"
    t_mm = OpTraitInfo(
        op_id="s0.l0.moe", trait=AlgebraicTrait.BILINEAR_MATMUL, symbolic_op_kind=k_mm,
        input_dtype="bf16", output_dtype="bf16", group_size=32, activation_kind="situ",
        activation_params=(), collective_axis="ep", row_working_bytes=0,
    )
    t_rt = replace(t_mm, op_id="s0.l0.router", symbolic_op_kind=k_rt)
    eq_mm = _format_dag_equation(
        "s0.l0.moe", (32, 4096), k_mm, ("h",), "y",
        output_dtype=t_mm.output_dtype, activation_kind=t_mm.activation_kind,
        activation_params=t_mm.activation_params, group_size=t_mm.group_size or 0,
        collective_axis=t_mm.collective_axis, op_meta={"ep": 4},
    )
    eq_rt = _format_dag_equation(
        "s0.l0.router", (32, 256), k_rt, ("h",), "w",
        output_dtype=t_rt.output_dtype,
    )
    assert "AllToAllDispatch" in eq_mm and "AllToAllCombine" in eq_mm and "ActMxQuant" in eq_mm
    assert "TopK(softmax/sigmoid" in eq_rt and "softplus" not in eq_rt


def test_h3_single_spine_ingress_deployment_mode_domain_routing_and_auto_va(tmp_path) -> None:
    import json
    from mindie_sim.cli.parser import build_parser
    from mindie_sim.cli.report import run_report_command
    from mindie_sim.compose import resolve_case_contracts

    parser = build_parser()

    # (a) Missing --deployment-mode on report without script hint raises crisp ModelingError listing all 4 modes.
    missing_args = parser.parse_args([
        "report", "--model", "deepseek-v2-lite", "--hardware", "a5",
        "--dp", "1", "--tp", "2", "--ep", "2", "--concurrency", "2",
        "--context", "1024", "--output-tokens", "64",
        "--no-xlsx", "--no-trace", "--out-dir", str(tmp_path / "missing_dep"),
    ])
    with pytest.raises(ModelingError, match="colocated.*pd_disaggregated.*pd_prefill.*pd_decode"):
        run_report_command(missing_args)

    # Script with explicit kv_role=kv_consumer supplies pd_decode via adapter.launch_codec without --deployment-mode.
    dec_sh = tmp_path / "decode_serve.sh"
    dec_sh.write_text(
        'vllm serve /weights/deepseek-v2-lite --tensor-parallel-size 2 --data-parallel-size 1 '
        '--enable-expert-parallel --kv-transfer-config \'{"kv_connector":"AscendHcclConnector","kv_role":"kv_consumer"}\'\n',
        encoding="utf-8",
    )
    script_rec = tmp_path / "script_pd_decode.json"
    rc_script = cli_main([
        "report", "--from-script", str(dec_sh), "--hardware", "a5",
        "--dp", "1", "--tp", "2", "--ep", "2", "--concurrency", "1",
        "--context", "512", "--output-tokens", "32", "--mode", "simulation",
        "--no-xlsx", "--no-trace", "--out-dir", str(tmp_path / "script_dec_out"),
        "--json", "--receipt", str(script_rec),
    ])
    assert rc_script == 0
    script_payload = json.loads(script_rec.read_text(encoding="utf-8"))
    assert script_payload["deployment_mode"] == "pd_decode"
    assert script_payload["ttft_slo_ms"] is None
    assert script_payload["agent_brief"]["scenario_and_parallelism"]["ttft_slo_ms"] is None

    # (b) pd_decode keeps ttft_slo_ms=None end-to-end (never 5000.0) and rejects explicit --ttft-ms.
    pd_dec_rec = tmp_path / "pd_dec_rec.json"
    rc_dec = cli_main([
        "report", "--model", "deepseek-v2-lite", "--hardware", "a5",
        "--deployment-mode", "pd_decode", "--dp", "1", "--tp", "2", "--ep", "2", "--concurrency", "1",
        "--context", "512", "--output-tokens", "32", "--tpot-ms", "30",
        "--mode", "simulation", "--no-xlsx", "--no-trace",
        "--out-dir", str(tmp_path / "pd_dec_out"), "--json", "--receipt", str(pd_dec_rec),
    ])
    assert rc_dec == 0
    pd_dec_payload = json.loads(pd_dec_rec.read_text(encoding="utf-8"))
    assert pd_dec_payload["ttft_slo_ms"] is None
    assert pd_dec_payload["agent_brief"]["scenario_and_parallelism"]["ttft_slo_ms"] is None
    bad_pd_dec_args = parser.parse_args([
        "report", "--model", "deepseek-v2-lite", "--hardware", "a5",
        "--deployment-mode", "pd_decode", "--ttft-ms", "5000",
        "--dp", "1", "--tp", "2", "--ep", "2", "--concurrency", "1",
        "--no-xlsx", "--no-trace", "--out-dir", str(tmp_path / "bad_pd_dec"),
    ])
    with pytest.raises(ModelingError, match="pd_decode_ttft_unavailable"):
        run_report_command(bad_pd_dec_args)

    # (c) Diffusion/DiT defaults to diffusion-native (zero LLM KV cache) and rejects --stack vllm-ascend.
    dit_case = compose_case("hunyuan-video-dit", hardware="a5", dp=1, tp=1, parallel="fixed")
    assert dit_case["runtime"]["metadata"]["stack_id"] == "diffusion-native"
    assert dit_case["runtime"]["metadata"]["workload_domain"] == "diffusion"
    _sc_d, qp_d, _lp_d = resolve_case_contracts(dit_case)
    assert qp_d is not None and qp_d.kv_cache_abi == () and qp_d.index_cache_abi == () and qp_d.recurrent_state_abi == ()
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        compose_case("hunyuan-video-dit", hardware="a5", dp=1, tp=1, parallel="fixed", runtime_stack="vllm-ascend")
    bad_dit_args = parser.parse_args([
        "report", "--model", "hunyuan-video-dit", "--hardware", "a5",
        "--stack", "vllm-ascend", "--deployment-mode", "colocated",
        "--dp", "1", "--tp", "1", "--concurrency", "1",
        "--no-xlsx", "--no-trace", "--out-dir", str(tmp_path / "bad_dit"),
    ])
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        run_report_command(bad_dit_args)

    # (d) Kimi-K3.1 & Qwen3.8-Flash-Next via compose_case(..., quant="auto_va") carry real CheckpointQuantMap -> RuntimeQuantPlan.
    kimi_raw = compose_case(
        "kimi-k3.1", hardware="950dt-96g-pod", dp=1, tp=8, dcp=8, ep=8, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024, mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        quant="auto_va",
    )
    sc_k, qp_k, lp_k = resolve_case_contracts(kimi_raw)
    assert sc_k is not None and qp_k is not None and lp_k is not None
    assert qp_k.provenance == "verified_checkpoint_descriptor"
    assert sc_k.sla.ttft_ms is None
    assert kimi_raw["runtime"]["recover_ssm"] is False
    kimi_opt_in = compose_case(
        "kimi-k3.1", hardware="950dt-96g-pod", dp=1, tp=8, dcp=8, ep=8, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024, mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        quant="auto_va", recover_ssm=True,
    )
    assert kimi_opt_in["runtime"]["recover_ssm"] is True
    ports_k = dict(qp_k.port_abis)
    assert ports_k[("target", 1, "self_attention_res_proj")].linear_method == "dense"
    assert ports_k[("target", 1, "mlp_res_proj")].linear_method == "dense"
    assert ports_k[("target", 1, "b_proj")].linear_method == "dense"
    assert ports_k[("target", 1, "f_a_proj")].linear_method == "dense"
    assert ports_k[("target", 1, "g_proj")].linear_method == "dense"
    assert ports_k[("target", 4, "mlp.latent_moe")].linear_method == "mx_fp8"
    assert ports_k[("target", 4, "mlp.experts.w13")].linear_method == "mx_fp4"
    assert ports_k[("target", 4, "mlp.experts.w2")].linear_method == "mx_fp4"

    qwen_raw = compose_case(
        "qwen3.8-flash-next", hardware="950dt-96g-pod", dp=1, tp=8, ep=8,
        prompt_tokens=32768, output_tokens=256, mtp_proposal_tokens=3, quant="auto_va",
    )
    sc_q, qp_q, lp_q = resolve_case_contracts(qwen_raw)
    assert sc_q is not None and qp_q is not None and lp_q is not None
    assert qp_q.provenance == "verified_checkpoint_descriptor"
    ports_q = dict(qp_q.port_abis)
    assert ports_q[("target", 0, "linear_attn.in_proj_qkv")].linear_method == "dense"
    assert ports_q[("target", 0, "linear_attn.out_proj")].linear_method == "dense"
    assert ports_q[("target", 1, "ple.key_proj")].linear_method == "dense"
    assert ports_q[("target", 1, "ple.conv1d")].linear_method == "dense"
    assert ports_q[("draft", 0, "self_attn.q_proj")].linear_method == "dense"
    assert ports_q[("draft", 0, "mlp.experts.w13")].linear_method == "dense"
    assert ports_q[("target", 0, "mlp.experts.w13")].linear_method == "mx_fp4"
    assert ports_q[("target", 3, "q_proj")].linear_method == "mx_fp8"


def test_diffusion_domain_detection_uses_token_boundaries_and_causal_arch() -> None:
    """Substrings and causal-LM architectures stay LLM while standalone diffusion tokens route to diffusion."""
    from mindie_sim.backends.ascend import profiles
    from mindie_sim.compose.inputs import HARDWARE_TOPOLOGY_ALIASES, resolve_hardware_sku
    from mindie_sim.compose.runtime import resolve_case_contracts
    from mindie_sim.prediction.structure import parse_case

    # Verify 64 GiB SKU variant capacity and explicit aliases without disturbing 32 GiB default aliases.
    assert profiles()["Ascend910B4-1"]["device"]["memory_bytes"] == 64 * 1024 ** 3
    assert profiles()["Ascend910B4"]["device"]["memory_bytes"] == 32 * 1024 ** 3
    for alias in ("910b4-1", "910b4-64g", "a2-64g", "800i-a2-64g"):
        assert resolve_hardware_sku(alias) == "Ascend910B4-1", alias
    assert HARDWARE_TOPOLOGY_ALIASES["800i-a2-64g"] == ("Ascend910B4-1", "a2_standard")
    for base_alias in ("a2", "910b", "910b4", "800i-a2", "a2-8p"):
        assert resolve_hardware_sku(base_alias) == "Ascend910B4", base_alias

    base_cfg = dict(
        architectures=["LlamaForCausalLM"],
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1024,
        max_position_embeddings=4096,
    )
    for model_type in ("llama", "swan_lm", "swandit_llm", "credit_lm", "wan"):
        case = parse_case(compose_case(
            dict(base_cfg, model_type=model_type),
            hardware="a3", tp=1, prompt_tokens=16, output_tokens=2, quant="bf16",
        ))
        _sc, plan, _lp = resolve_case_contracts(case)
        assert [p.attention for p in case.model.layer_plans] == ["gqa", "gqa"], model_type
        assert plan.stack_id == "vllm-ascend", model_type
        assert len(plan.kv_cache_abi) == 2, model_type

    diff_base = {k: v for k, v in base_cfg.items() if k != "architectures"}
    for diff_type in ("wan", "wan2.1", "hunyuan_video_dit", "STDiT3", "flux", "sd3"):
        diff_case = compose_case(
            dict(diff_base, model_type=diff_type),
            hardware="a5", tp=1, prompt_tokens=16, output_tokens=2, quant="bf16",
        )
        assert diff_case["runtime"]["metadata"]["workload_domain"] == "diffusion", diff_type
        assert diff_case["runtime"]["metadata"]["stack_id"] == "diffusion-native", diff_type


def test_nominal_product_clocks_derived_clocks_sort_scaling_and_abi_shape_equations() -> None:
    """Product nominal clocks (1650/1855 MHz), sort preset clock scaling, and CANN ABI shape equations hold."""
    from mindie_sim.backends.abi_registry import default_abi_registry
    from mindie_sim.backends.ascend import apply_observed_aicore_clock, profiles
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig
    from mindie_sim.schema.hardware import Hardware

    # Nominal 1650 MHz SKUs yield exact 270.336 TFLOPS BF16, unchanged HBM/DRAM ceilings, and 1.0 observed ratio.
    all_profs = profiles()
    for sku, exp_hbm, exp_mem in (
        ("Ascend910B4", 0.8e12, 34359738368),
        ("Ascend910B4-1", 0.8e12, 68719476736),
        ("Ascend910_9362", 1.6e12, 68719476736),
    ):
        prof = all_profs[sku]
        assert prof["source"]["declaration"]["configured_clock_mhz"] == pytest.approx(1650.0)
        assert prof["device"]["compute_peaks"]["bf16"] == pytest.approx(270.336e12)
        assert prof["device"]["rates"]["hbm"] == pytest.approx(exp_hbm)
        assert prof["device"]["memory_bytes"] == exp_mem
        obs = apply_observed_aicore_clock({**prof["device"], "observed_aicore_freq_mhz": 1650.0}, prof)
        assert obs["compute_peaks"]["bf16"] == pytest.approx(prof["device"]["compute_peaks"]["bf16"])
        assert obs["rates"]["mte2_aic"] == pytest.approx(prof["device"]["rates"]["mte2_aic"])

    # Derived commercial SKUs retain nominal 1650 MHz Cube clock with 4.0 TB/s HBM and scaled core counts.
    for sku, aic, mem_gib in (
        ("Ascend950DT_425_96", 28, 96),
        ("Ascend950DT_486_96", 32, 96),
        ("Ascend950DT_547_96", 36, 96),
        ("Ascend950DT_547_144", 36, 144),
    ):
        die = parse_case(compose_case("qwen3-32b", sku, layers=1, tp=1)).machine.dies[0]
        assert dict(die.compute_peaks)["bf16"] == pytest.approx(aic * 1.65e9 * 8192)
        assert die.compute_arch.clock_hz == pytest.approx(1.65e9)
        assert dict(die.rates)["hbm"] == pytest.approx(4.0e12)
        assert die.memory_bytes == mem_gib * 1024 ** 3

    # Restating preset cycles_per_unit without an explicit clock still scales with observed frequency.
    for services in ({}, {"vbitsort32": {"cycles_per_unit": 16}}):
        die = Hardware.from_mapping({
            "backend": "ascend", "sku": "Ascend910_9391", "num_devices": 1,
            "device": {"observed_aicore_freq_mhz": 925,
                       "compute_arch": {"presets": ["ascend_sort32_spec"], "instruction_services": services}},
        }).dies[0]
        svc = next(s for s in die.compute_arch.instruction_services if s.name == "vbitsort32")
        assert svc.per_owner_rate == pytest.approx(die.compute_arch.clock_hz / svc.cycles_per_unit)

    # Physical 4D FRACTAL_NZ weight in QuantMatmulV4, TBMM batch % split, and rank-3 MoeTokenPermute.
    reg = default_abi_registry()
    x1 = OperandPortSig("x1", "int8", "ND", False, (16, 128), "per_token_int8", (), "fp32", (16,))
    x2 = OperandPortSig("x2", "int8", "FRACTAL_NZ", True, (4, 4, 16, 32), "per_channel_int8", (), "fp32", (64,))
    y_out = OperandPortSig("y", "bf16", "ND", False, (16, 64), "dense", (), "none", ())
    q_nz = AbiQueryKey("A3", "Ascend910_9391", "aclnnQuantMatmulV4", "quant_matmul", "target", "none", "shipping",
                       (x1, x2), (y_out,), "local", 1, "local", 1, "none", "none", "none", (), False, False,
                       -1, 0, 0, 0, 0, "none", False, 1, "bf16", False, 1, False, frozenset())
    assert reg.verify(q_nz).status == "admitted_shipping"
    for bad_shape in ((2, 4, 16, 64), (3, 4, 4, 16, 32)):
        bad_x2 = OperandPortSig("x2", "int8", "FRACTAL_NZ", True, bad_shape, "per_channel_int8", (), "fp32", (64,))
        q_bad = AbiQueryKey("A3", "Ascend910_9391", "aclnnQuantMatmulV4", "quant_matmul", "target", "none", "shipping",
                            (x1, bad_x2), (y_out,), "local", 1, "local", 1, "none", "none", "none", (), False, False,
                            -1, 0, 0, 0, 0, "none", False, 1, "bf16", False, 1, False, frozenset())
        assert reg.verify(q_bad).status == "rejected_abi", bad_shape

    def _bf16_port(role: str, shape: tuple[int, ...]) -> OperandPortSig:
        return OperandPortSig(role, "bf16", "ND", False, shape, "dense", (), "none", ())

    for batch, expected in ((4, "admitted_shipping"), (3, "rejected_abi")):
        q_tbmm = AbiQueryKey(
            "A3", "Ascend910_9391", "aclnnTransposeBatchMatMul", "transpose_batch_matmul", "target", "none", "shipping",
            (_bf16_port("x1", (batch, 16, 64)), _bf16_port("x2", (batch, 64, 32))),
            (_bf16_port("y", (2, 16, batch * 32 // 2)),), "local", 1, "local", 1, "none", "none", "none",
            (("batch_split_factor", 2.0),), False, False, -1, 0, 0, 0, 0, "none", False, 1, "bf16", False, 1, False, frozenset(),
        )
        assert reg.verify(q_tbmm).status == expected

    for tail in ((64,), (2, 32)):
        tok_in = OperandPortSig("x", "bf16", "ND", False, (4, *tail), "dense", (), "none", ())
        idx_in = OperandPortSig("idx", "int32", "ND", False, (4, 2), "none", (), "none", ())
        tok_out = OperandPortSig("permute_tokens", "bf16", "ND", False, (8, *tail), "dense", (), "none", ())
        idx_out = OperandPortSig("sorted_indices", "int32", "ND", False, (8,), "none", (), "none", ())
        q_perm = AbiQueryKey(
            "A3", "Ascend910_9391", "aclnnMoeTokenPermute", "moe_token_permute", "target", "none", "shipping",
            (tok_in, idx_in), (tok_out, idx_out), "local", 1, "local", 1, "none", "none", "none", (), False, False,
            -1, 0, 2, 4, 0, "none", False, 1, "bf16", False, 1, False, frozenset(),
        )
        assert reg.verify(q_perm).status == "admitted_shipping"
        q_perm_bad = AbiQueryKey(
            "A3", "Ascend910_9391", "aclnnMoeTokenPermute", "moe_token_permute", "target", "none", "shipping",
            (tok_in, idx_in), (tok_out, idx_out), "local", 1, "local", 1, "none", "none", "none",
            (("num_out_tokens", 3.0),), False, False, -1, 0, 2, 4, 0, "none", False, 1, "bf16", False, 1, False, frozenset(),
        )
        assert reg.verify(q_perm_bad).status == "rejected_abi"


def test_structural_linear_attention_dispatch_projected_draft_width_and_explicit_schedules() -> None:
    """Unseen GDN configs preserve recurrent state, projected_concat draft width matches taps, and schedules parse."""
    import json
    from pathlib import Path
    from mindie_sim.catalog.draft_spec import SpeculativeExecutionContract
    from mindie_sim.catalog.minimax_draft import build_minimax_auxiliary
    from mindie_sim.model_catalog import build_model_spec, load_model
    from mindie_sim.prediction.sharding.ffn_rules import raw_selection

    # Unseen model_type with full GDN geometry preserves recurrent and convolution state.
    gdn_cfg = dict(
        num_hidden_layers=1, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=256, layer_types=["linear_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, torch_dtype="bfloat16",
    )
    for model_type in ("qwen3_5", "unseen_gated_delta"):
        lp = build_model_spec("synthetic-gdn", dict(gdn_cfg, model_type=model_type)).layer_plans[0]
        assert (lp.attention, lp.recurrent_state_elements, lp.conv_state_elements) == ("gdn", 1024, 384)

    # projected_concat draft uses len(target_layer_ids) * hidden_size even when kind="mtp".
    d_cfg = dict(num_mtp_modules=2, dspark_target_layer_ids=[0], hidden_size=64,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256)
    draft = build_minimax_auxiliary(d_cfg, body_layers=1).dspark
    plan = SpeculativeExecutionContract("dspark", "dspark", 2, True).build_prepare_plan(8, draft_spec=draft)
    assert draft.input_width == 64 and plan.exact_gemm_flops == 2 * 8 * 64 * (64 + 2 * (64 + 2 * 32))

    # DeepSeek V4 routed_scaling_factor, explicit mlp_layer_types hash schedule, and HF layer_types/compress_rates.
    cfg_path = Path(__file__).resolve().parents[2] / "mindie_sim/model_configs/deepseek-v4-pro-0813.json"
    v4_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    v4_model = load_model("deepseek-v4-pro")
    learned = next(p for p in v4_model.layer_plans if p.router_mode != "hash")
    ops, *_ = raw_selection(prefix="r", plan=learned, logits_input="logits", weight_dtype="fp32",
                            weight_bytes=4, stage=0, requests=("r",), tokens=1, mode="decode")
    assert learned.routed_scaling_factor == pytest.approx(v4_raw["routed_scaling_factor"])
    assert sum(op.vector_ops for op in ops if op.recipe == "router_routed_scale") == learned.top_k

    cfg_hash = dict(v4_raw)
    cfg_hash.pop("num_hash_layers")
    cfg_hash["mlp_layer_types"] = ["hash_moe" if i in (0, 2, 4) else "moe" for i in range(cfg_hash["num_hidden_layers"])]
    assert [p.index for p in build_model_spec("v4-hash-sched", cfg_hash).layer_plans if p.router_mode == "hash"] == [0, 2, 4]

    cfg_lt = dict(v4_raw)
    names = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    cfg_lt["layer_types"] = [names[r] for r in cfg_lt.pop("compress_ratios")[:cfg_lt["num_hidden_layers"]]]
    cfg_lt["compress_rates"] = {names[4]: 4, names[128]: 128}
    m_lt = build_model_spec("v4-layer-types", cfg_lt)
    assert [p.compress_ratio for p in m_lt.layer_plans] == [p.compress_ratio for p in v4_model.layer_plans]
