"""Piecewise supply uses declared 9579 physics, not 36/72 or HBM-as-MTE2.

[Role] Verify physical supply constants and declared AIC/AIV/MTE2/HCCL pipe rates.
[Invariants] Pure in-memory arithmetic assertions; no empirical timing fallbacks.
[Timeouts & Locks] Tier 0 (<5s), no I/O or locks.
[Side-Effect Policy] Pure read-only in-memory unit tests.
"""
from fractions import Fraction

import pytest

from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.piecewise.families import apply_hardware_service
from mindie_sim.prediction.piecewise.families_moe_collective import (
    DEFAULT_L0C_BYTES, FRACTAL,
)
from mindie_sim.prediction.piecewise.physical import (
    A3_ANALYTICAL_CORES, A3_L0C_BYTES, ARCH3510_AIC, ARCH3510_AIV,
    ARCH3510_CLOCK_HZ, ARCH3510_CUBE_MIN_MS, ARCH3510_EMPTY_AIC_MIN_MS,
    ARCH3510_L0C_BYTES, ARCH3510_VECTOR_MIN_MS, ARCH3510_W8A8_ND_BF16_MIN_MS,
    MMAD_FRACTAL, ROCE_400G_ALPHA_US, ROCE_400G_BI_BYTES_PER_MS,
    ROCE_400G_UNI_BYTES_PER_MS, SOFTWARE_TILE_K, SOFTWARE_TILE_M, SOFTWARE_TILE_N,
    UB_MESH_ALPHA_US, UB_MESH_UNI_BYTES_PER_MS, arch3510_bytes_per_second,
    declared_aic_owners, gbps_to_bytes_per_ms,
)
from mindie_sim.schema.hardware import Hardware


class _Theta:
    def __init__(self, values):
        self.values = values

    def find(self, name, default=None):
        return self.values.get(name, default)


class _Operator:
    def __init__(self, theta):
        self.theta = theta
        self.role = ""


class _Op:
    def __init__(self, supply):
        self.supply = supply
        self.operator = _Operator(_Theta({"weight_format": "int8", "tp": 1}))
        self.family = "gemm"
        self.name = "proj"
        self.kind = "F3"


class _Geometry:
    tokens = 32
    batch = 1
    phase = "prefill"
    s_max = 32


class _Die:
    def __init__(self, pairs):
        self.rates = tuple(pairs)
        self.compute_units = 1

    def peak(self, dtype):
        return 8.0


class _Hardware:
    def __init__(self, die):
        self.dies = (die,)
        self.links = ()


def test_9579_supply_aggregates_equal_mte2_splits_and_rejects_36_72():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 1,
    })
    die = hardware.dies[0]
    supply = PipeSupply.from_hardware(hardware)
    assert die.compute_units == ARCH3510_AIC == 32
    assert die.units("vector") == ARCH3510_AIV == 64
    assert die.compute_units != 36
    assert die.units("vector") != 72
    mte2 = arch3510_bytes_per_second(80, ARCH3510_AIC)
    assert arch3510_bytes_per_second(40, ARCH3510_AIV) == pytest.approx(mte2)
    assert mte2 == pytest.approx(4.224e12)
    assert supply.rate("mte2") == pytest.approx(mte2 * 1e-3)
    assert supply.rate("mte2") != pytest.approx(1.6e12 * 1e-3)
    assert supply.rate("hbm") == pytest.approx(1.6e12 * 1e-3)
    assert supply.rate("mte1") == pytest.approx(
        arch3510_bytes_per_second(256, ARCH3510_AIC) * 1e-3)
    assert supply.rate("mte3") == pytest.approx(
        arch3510_bytes_per_second(128, ARCH3510_AIV) * 1e-3)
    assert supply.rate("fixpipe") == pytest.approx(
        arch3510_bytes_per_second(128, ARCH3510_AIC) * 1e-3)
    assert ARCH3510_CLOCK_HZ == pytest.approx(1.65e9)


def test_9579_links_are_ub_mesh_and_roce_not_hccs():
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579",
        "num_devices": 16, "ranks_per_node": 8,
    })
    links = {link.channel: link for link in hardware.links}
    intra, inter = links["d2d_intra_node"], links["d2d_inter_node"]
    assert intra.capacity_bytes_per_ms == pytest.approx(UB_MESH_UNI_BYTES_PER_MS)
    assert intra.capacity_bytes_per_ms == pytest.approx(gbps_to_bytes_per_ms(392.0))
    assert intra.latency_us == pytest.approx(UB_MESH_ALPHA_US)
    assert inter.capacity_bytes_per_ms == pytest.approx(ROCE_400G_UNI_BYTES_PER_MS)
    assert inter.capacity_bytes_per_ms == pytest.approx(gbps_to_bytes_per_ms(50.0))
    assert inter.latency_us == pytest.approx(ROCE_400G_ALPHA_US)
    assert ROCE_400G_BI_BYTES_PER_MS == pytest.approx(gbps_to_bytes_per_ms(90.0))
    assert inter.capacity_bytes_per_ms != pytest.approx(ROCE_400G_BI_BYTES_PER_MS)
    assert "hccs" not in {link.channel for link in hardware.links}


def test_software_tile_is_not_the_mmad_fractal_and_cores_stay_a3_by_default():
    policy = ImplementationPolicy()
    assert (policy.mt, policy.nt, policy.kt) == (
        SOFTWARE_TILE_M, SOFTWARE_TILE_N, SOFTWARE_TILE_K)
    assert policy.mt == 128 and MMAD_FRACTAL == FRACTAL == 16
    assert policy.cores == A3_ANALYTICAL_CORES == 24
    die = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 1,
    }).dies[0]
    assert declared_aic_owners(die) == 32
    assert ImplementationPolicy(cores=declared_aic_owners(die)).cores == 32
    assert DEFAULT_L0C_BYTES == A3_L0C_BYTES == 128 * 1024
    assert ARCH3510_L0C_BYTES == 256 * 1024
    assert DEFAULT_L0C_BYTES != ARCH3510_L0C_BYTES


def test_missing_invocation_classes_use_declared_minima_not_ratios():
    empty_supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6), ("hbm", 1.0e6), ("mte2", 1.0e6)),
        resource_units=(("compute_units", 32), ("vector", 64)),
    )
    empty_times = [Fraction(1)]
    apply_hardware_service(
        _Op(empty_supply), _Geometry(), {}, [{"cube": Fraction(1)}], empty_times,
        {"cube": Fraction(1)}, {"cube": Fraction(1)}, {"cube": Fraction(1)},
    )
    assert empty_times[-1] == Fraction(1)
    hw_9579 = PipeSupply.from_hardware(Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 1,
    }))
    supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6), ("hbm", 1.0e6), ("mte2", 1.0e6)),
        invocations=hw_9579.invocations,
        resource_units=hw_9579.resource_units,
        flops_by_dtype=hw_9579.flops_by_dtype,
    )
    stage_times = [Fraction(1)]
    apply_hardware_service(
        _Op(supply), _Geometry(), {}, [{"cube": Fraction(1)}], stage_times,
        {"cube": Fraction(1)}, {"cube": Fraction(1)}, {"cube": Fraction(1)},
    )
    ratio_quant = ARCH3510_CUBE_MIN_MS * 1.92
    ratio_empty = ARCH3510_VECTOR_MIN_MS * 0.35
    assert ARCH3510_W8A8_ND_BF16_MIN_MS != pytest.approx(ratio_quant)
    assert ARCH3510_EMPTY_AIC_MIN_MS != pytest.approx(ratio_empty)
    expected = (
        Fraction(1)
        + as_fraction(ARCH3510_W8A8_ND_BF16_MIN_MS)
        + as_fraction(ARCH3510_VECTOR_MIN_MS) * Fraction(3, 2)
    )
    assert stage_times[-1] == expected


def test_undeclared_mte2_does_not_invent_an_arch3510_rate():
    hardware = _Hardware(_Die((("hbm", 1000.0), ("vector", 10.0))))
    supply = PipeSupply.from_hardware(hardware)
    assert supply.rate("mte2") == pytest.approx(1.0)
    assert supply.rate("mte1") == pytest.approx(2.0)
    split = _Hardware(_Die((
        ("hbm", 1.6e12), ("mte1", 8.0e12), ("mte3", 8.0e12), ("fixpipe", 8.0e12),
        ("mte2_aic", 4.0e12), ("mte2_aiv", 2.0e12),
    )))
    slower = PipeSupply.from_hardware(split)
    assert slower.rate("mte2") == pytest.approx(2.0e12 * 1e-3)
    assert slower.rate("mte1") == pytest.approx(8.0e12 * 1e-3)
    assert "mte2" in supply.undeclared_pipes and "mte2" not in slower.undeclared_pipes


def test_single_branch_engine_equivalence_with_hardware_service():
    from mindie_sim.prediction.piecewise import (
        ImplementationPolicy, OperatorPiecewiseCompiler, RequestGeometry,
        WorkloadGeometry, clamp_graph_bucket,
    )
    from mindie_sim.prediction.piecewise.composition import (
        contention_envelope, pipeline_makespan, run_engine_work,
    )
    from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
    from mindie_sim.prediction.subgraphs import FeatureConfig, StateView, instantiate_model
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.model import ModelSpec
    from mindie_sim.schema.subgraphs import StaticSignature

    hw = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 4})
    base_supply = PipeSupply.from_hardware(hw)
    supply = PipeSupply(
        base_supply.rates, base_supply.structure_caps,
        invocations=(("cube", 10.0), ("vector", 0.001312), ("w8a8_nd_bf16", 10.0), ("empty_aic", 0.000439)),
        resource_units=base_supply.resource_units, flops_by_dtype=base_supply.flops_by_dtype,
    )
    model = ModelSpec.from_mapping({
        "name": "toy-idx-moe", "hidden_size": 64, "intermediate_size": 128,
        "layers": 1, "num_heads": 4, "num_kv_heads": 1, "head_dim": 16, "vocab_size": 256,
        "num_experts": 8, "top_k": 2, "moe_intermediate_size": 32, "num_shared_experts": 1,
        "mla": {"q_lora_rank": 16, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8, "index_topk": 8},
        "layer_plans": [{
            "index": 0, "attention": "mla", "ffn": "moe", "num_heads": 4, "num_kv_heads": 1,
            "head_dim": 16, "intermediate_size": 128, "num_experts": 8, "top_k": 2,
            "moe_intermediate_size": 32, "num_shared_experts": 1,
            "indexer": True, "index_heads": 2, "index_dim": 16, "index_topk": 8,
        }],
    })
    features = FeatureConfig.from_model(model, ep=4, dp=4, multistream_attention_indexer=True)
    instantiated = instantiate_model(model, features)
    compiler = OperatorPiecewiseCompiler(supply, ImplementationPolicy())
    f0_or_f4 = instantiated.find("F4", layer=0, role="shared_ffn")
    op = compiler.compile(f0_or_f4).operator("shared_gate_up")
    geom = WorkloadGeometry.homogeneous(4, 1, 16, cohort_batch=16)
    cost = op.evaluate(geom)
    one_branch = contention_envelope({"single": [list(s) for s in cost.stage_work]}, supply)
    assert float(one_branch.makespan_ms) == pytest.approx(float(cost.duration_ms), rel=1e-9)

    # Indexed A1 ordering and causal selection readiness
    kinds = [s.kind for s in instantiated.layers[0].subgraphs]
    assert kinds.index("I0") < kinds.index("I1") < kinds.index("A1")
    a1_comp = compiler.compile(instantiated.find("A1", layer=0, role="attention"))
    with pytest.raises(ModelingError, match="state_port_not_ready"):
        a1_comp.evaluate(geom, StateView())

    # EP cohort_batch conservation between direct F3 and ModelPipeline.forward
    pipe = assemble_model(model, supply=supply, features=features)
    pred = pipe.forward(geom)
    assert pred.active.cohort_batch == 16
    f3_direct = compiler.compile(instantiated.find("F3", layer=0)).evaluate(geom)
    f3_forward = next(sg for sg in pred.layers[0].subgraphs if sg.kind == "F3")
    assert f3_forward.rows[0].cost.get("moe_rows") == f3_direct.operators[0].get("moe_rows") == 8

    # Ragged cohort expansion preserves ragged distribution and rejects invalid contraction
    ragged = WorkloadGeometry((RequestGeometry(1, 15), RequestGeometry(3, 13)), cohort_batch=4)
    expanded = ragged.for_subgraph("F3")
    assert [r.q_len for r in expanded.requests] == [1, 3, 1, 3]
    with pytest.raises(ModelingError, match="workload_ragged_cohort_expansion_ambiguous"):
        ragged.expanded_to(3)

    # clamp_graph_bucket selects valid local capture bucket
    assert clamp_graph_bucket(4, 3, 4) == 4
    assert clamp_graph_bucket(16, 3, 4, (1, 2, 4, 8, 16)) == 4

    # Negative work rejection, StaticSignature deep immutability, O(stages) pipeline_makespan
    with pytest.raises(ModelingError, match="engine_work_negative_or_nonfinite"):
        run_engine_work({"neg": [[("hbm", Fraction(-100))]]}, supply)
    mut_list = [1, 2]
    sig = StaticSignature("test", extras=(("nested", mut_list),))
    mut_list.append(3)
    assert sig.find("nested") == (1, 2) and isinstance(hash(sig), int)
    with pytest.raises(ModelingError, match="mapping_value_invalid"):
        StaticSignature("test", extras=(("bad", {"a": 1}),))
    assert pipeline_makespan((Fraction(1), Fraction(2)), 5000, 2) == Fraction(10001)
