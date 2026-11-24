"""External framework operator libraries: xLLM, MindIE-SD and aclnn calls.
[Role] Prove that third-party operator libraries lower into the same canonical
  occurrence model: xLLM custom math, MindIE-SD diffusion contracts, standalone
  tensor-shaped aclnn declarations, and single-call aclnn lowering.
[Invariants] Every declared producer keeps its analytical byte/FLOP ledger,
  registry capabilities match registered implementations, and each rejected
  declaration keeps its own exact error code.  No measurement or model-name
  table selects a kernel.
[Timeouts & Locks] Bounded in-memory lowering and planning; no devices.
[Side-Effect Policy] Read-only; writes nothing.
"""
from dataclasses import replace
import pytest
from _support.axes import GEN5_DT_SKU as A5_SKU, SOC_910_93, SOC_950
from _support.moe_hc_mc2_fixtures import attnres_case
from mindie_sim import graph_costs, simulate_aclnn_call
from mindie_sim.backends.ascend import materialize
from mindie_sim.catalog.ops import cache_store_op
from mindie_sim.prediction.call_builders import build_aclnn_call, build_call
from mindie_sim.prediction.operators.custom_framework import (
    XLLM_CUSTOM_OPS, XLLM_VENDOR, capabilities as xllm_capabilities,
    fused_region_contract, lower_op as xllm_lower_op, recipes as xllm_recipes)
from mindie_sim.prediction.operators.diffusion import (
    MINDIE_SD_CUSTOM_OPS, capabilities as sd_capabilities,
    compute_mindie_sd_effective_denoising_steps,
    lower_op as sd_lower_op, recipes as sd_recipes)
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.implementations import IMPLEMENTATIONS, lower_operators, select
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.public.environment import ExecutionEnvironment
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
pytestmark = pytest.mark.tier0
BASE_GEOMETRY = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64}, {"ub": 196608}, {"ub": 64})
CORE_NAMES = ("hbm", "mte2", "mte3", "vector", "cube")
REG_NAMES = CORE_NAMES + ("reg_alu_lanes", "reg_issue_slots", "reg_data_writes",
                          "reg_special_lanes", "reg_reduce_lanes", "reg_ub_accesses", "reg_ub_stores")
class _Selection:
    name = "declared_native_fusion"
def _make_op(family, *, recipe="", local_shape=(16, 4096), flops=1_000_000.0,
             mte2_bytes=65536.0, mte3_bytes=32768.0, vector_ops=1024.0, dtype="bf16",
             metadata=()):
    return OpSpec(op_id=f"test.{family}", family=family, phase="decode", stage=0, layer=0,
                  requests=("req-0",), semantic_shape=local_shape, local_shape=local_shape,
                  padded_shape=local_shape, launch_shape=local_shape, flops=flops,
                  hbm_bytes=mte2_bytes + mte3_bytes, mte2_bytes=mte2_bytes,
                  mte3_bytes=mte3_bytes, vector_ops=vector_ops,
                  recipe=recipe or family, dtype=dtype, metadata=metadata)
def _geometry(reg=True):
    if not reg:
        return BASE_GEOMETRY
    return Geometry(64, {name: 64 for name in REG_NAMES + ("mte2_aiv", "mte2_aic")},
                    {"ub": 196608}, {"ub": 64})
def _context(op, names=CORE_NAMES, **changes):
    values = dict(op=op, work={}, resource_names={n: n for n in names},
                  rates={n: 1e9 for n in names}, core_count=1,
                  compute_peaks={"bf16": 1e9, "int8": 1e9})
    values.update(changes)
    return RecipeContext(**values)
def _engine(names=CORE_NAMES, **pools):
    return Engine(tuple(Resource(n, 1e9) for n in names), pools=pools,
                  pool_units={name: 1 for name in pools}, trace_limit=100)
def _fused_op(rows=32, limit=None, dtype="int8", groups=None):
    meta = (("matrix_group_rows", groups or (rows // 2, 0, rows - rows // 2)),
            ("weight_format", "int8"),
            ("weight_scale_format", "fp32"), ("activation_scale_format", "fp32"),
            ("output_format", "int8"), ("output_scale_format", "fp32"))
    if limit is not None:
        meta += (("workspace_limit_rows", limit),)
    return replace(_make_op("grouped_gemm", recipe="moe_grouped_matmul_swiglu_quant",
                            local_shape=(rows, 256, 128), dtype=dtype), metadata=meta)
def _fused_members():
    matrix = replace(_make_op("grouped_gemm", recipe="grouped_gate_up",
                              local_shape=(32, 256, 128), dtype="int8"),
                     metadata=_fused_op().metadata)
    activation = replace(_make_op("swiglu", local_shape=(32, 128)), deps=(matrix.op_id,))
    quant = replace(_make_op("quantize", local_shape=(32, 128)), deps=(activation.op_id,))
    return matrix, activation, quant
def _gdn(rows=4, *, read=(0, -1, 2, 3), write=(0, 1, 2, 3), versions=((1, 1, 1, 1), (1, 1, 1, 1)),
         capacity=None):
    slots = capacity or rows
    meta = (("qkv_shape", (rows, 768)), ("z_shape", (rows, 2, 128)), ("num_k_heads", 2),
            ("conv_state_shape", (slots, 3, 768)), ("ssm_state_shape", (slots, 2, 128, 128)),
            ("conv_weight_shape", (4, 768)), ("b_shape", (rows, 2)), ("a_shape", (rows, 2)),
            ("a_log_shape", (2,)), ("dt_bias_shape", (2,)),
            ("read_state_indices", read), ("write_state_indices", write),
            ("read_versions", versions[0]), ("write_versions", versions[1]),
            ("state_layout", "V,K"))
    if capacity is not None:
        meta += (("state_capacity", capacity),)
    return replace(_make_op("gdn_decode", local_shape=(rows, 768)), metadata=meta)
def _cache_op(shape, slots=None, capacity=None):
    meta = () if slots is None else (("slot_mapping", slots),
                                     ("cache_capacity_slots", capacity or len(slots)))
    return replace(_make_op("kv_cache_write", local_shape=shape, mte3_bytes=0.0), metadata=meta)
def _prepare(book, name, op, geometry=BASE_GEOMETRY):
    prepared, plan = book()[name].prepare(op, geometry)
    return prepared, plan
def _sd_contract(lowered):
    return dict(lowered.metadata)["mindie_sd_contract"]
def _write(contract, name):
    return next(item for item in contract["writes"] if item["name"] == name)
@pytest.mark.parametrize(("family", "shape", "mte3", "kwargs", "recipe", "kind", "returns", "checks"), (
    ("mla_attention", (8, 512), 32768.0,
     {"op_name": "smla_va", "soc_family": SOC_910_93}, "smla_va", "SmlaVa",
     ("attn_out",), (("native_vendor", XLLM_VENDOR),)),
    ("indexed_attention", (16, 128), 4096.0,
     {"op_name": "sparse_flash_attention_lse", "return_lse": True},
     "sparse_flash_attention_lse", "SparseFlashAttentionLse",
     ("softmax_lse", "softmax_max", "softmax_sum"), ()),
    ("beam_select", (4, 1024), 128.0, {"op_name": "onerec_final_beam_select", "beam_width": 64},
     "onerec_final_beam_select", "OnerecFinalBeamSelect",
     ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence"),
     (("beam_width", 64),)),
    ("kv_cache_write", (20, 128), 1024.0,
     {"op_name": "reshape_and_cache_a5", "soc_family": SOC_950, "kv_page_size": 128},
     "reshape_and_cache_a5", "ReshapeAndCacheA5", ("xtensor_kv_page",),
     (("aligned_page_rows", 128),)),
))
def test_custom_math_families_register_and_lower(family, shape, mte3, kwargs, recipe, kind,
                                                 returns, checks):
    assert "xllm_custom_xllm_math" in IMPLEMENTATIONS
    lowered = xllm_lower_op(_make_op(family, local_shape=shape, mte3_bytes=mte3), **kwargs)
    meta = dict(lowered.metadata)
    assert lowered.recipe == recipe and meta["native_kind"] == kind and set(returns) <= set(lowered.returns) and all(meta[key] == value for key, value in checks) and select({family: "xllm_custom_xllm_math"}, family) is not None
def test_custom_math_family_work_conserves_declared_bytes():
    base = _make_op("indexed_attention", local_shape=(16, 128), mte3_bytes=4096.0)
    sfa = xllm_lower_op(base, op_name="sparse_flash_attention_lse", return_lse=True)
    assert sfa.mte3_bytes == base.mte3_bytes + 2 * 16 * 4 and dict(sfa.metadata)["xllm_attention_geometry"]["softmax_stats_bytes"] == 128
    moe = xllm_lower_op(_fused_op(), op_name="moe_grouped_matmul_swiglu_quant")
    assert moe.local_shape == (32, 128) and moe.dtype == "int8" and moe.returns == ("y", "y_scale") and moe.matrix_flops == (("int8", 2 * 32 * 256 * 128),) and moe.mte3_bytes == 32 * 128 + 32 * 4 + 32 * 256 * 4
    assert moe.mte2_bytes == (32 * 128 + 2 * 256 * 128 + 2 * 256 * 4
                              + 32 * 4 + 3 * 8 + 32 * 256 * 4)
    beam = xllm_lower_op(_make_op("beam_select", local_shape=(4, 1024), mte3_bytes=128.0),
                         beam_width=64)
    assert beam.mte3_bytes == 3 * 64 * 4
    gdn = xllm_lower_op(_gdn(), op_name="mega_gdn_decode")
    assert gdn.local_shape == (4, 2, 128) and gdn.returns == ("convOut", "convStateOut", "ssmStateOut", "out") and dict(gdn.metadata)["xllm_mega_gdn_config"]["v_heads"] == 2
def test_lower_operators_applies_xllm_bindings_and_shared_aliases():
    ops = (_make_op("smla_va", local_shape=(8, 512)),
           _make_op("reshape_and_cache_a5", local_shape=(16, 128)))
    lowered = lower_operators(ops, {
        "smla_va": "xllm_smla_va",
        "reshape_and_cache_a5": {"implementation": "xllm_reshape_and_cache_a5",
                                 "options": {"soc_family": SOC_950, "kv_page_size": 128}}})
    assert [dict(op.metadata)["implementation_binding"]["implementation"] for op in lowered] == [
        "xllm_smla_va", "xllm_reshape_and_cache_a5"]
    shared = lower_operators(
        (_make_op("mla_attention", local_shape=(8, 512)), _make_op("elementwise", local_shape=(4, 1024))),
        {"mla_attention": "xllm_sparse_flash_attention_lse",
         "elementwise": {"implementation": "xllm_onerec_final_beam_select",
                         "options": {"beam_width": 32}}})
    assert [op.recipe for op in shared] == [
        "sparse_flash_attention_lse", "onerec_final_beam_select"]
    with pytest.raises(ModelingError, match="xllm_custom_op_soc_unsupported"):
        xllm_lower_op(_make_op("reshape_and_cache_a5"), soc_family="Ascend910B")
@pytest.mark.parametrize(("builder", "kwargs", "match"), (
    (lambda: _make_op("grouped_gemm", local_shape=(32, 8192, 4096)), {}, "requires_explicit_name"),
    (lambda: _make_op("smla_va", dtype="int8"), {"op_name": "smla_va"},
     "input_dtype_unsupported"),
    (lambda: _make_op("beam_select", local_shape=(4, 1024)),
     {"op_name": "onerec_final_beam_select"}, "requires_positive_beam_width"),
    (lambda: _make_op("attention", recipe="flash_attention_decode",
                      local_shape=(8, 16392, 4, 256)), {"op_name": "smla_va"},
     "requires_latent_sparse_or_mla_attention"),
    (lambda: _make_op("recurrent_vector", recipe="recurrent_scan0",
                      local_shape=(8, 48, 128, 128)), {"op_name": "mega_gdn_decode"},
     "gdn_decode_requires_declared_fusion_region"),
    (lambda: _make_op("grouped_gemm", local_shape=(32, 256, 128), dtype="int8"),
     {"op_name": "moe_grouped_matmul_swiglu_quant"}, "requires_declared_fusion_region"),
))
def test_xllm_declarations_keep_exact_rejection_codes(builder, kwargs, match):
    with pytest.raises(ModelingError, match=match):
        xllm_lower_op(builder(), **kwargs)
def test_mindie_sd_lowering_and_denoising_step_cache():
    attn = _make_op("indexed_attention", local_shape=(256, 128), flops=10_000_000.0)
    rain = sd_lower_op(attn, op_name="rainfusion", sparsity_ratio=0.75, block_size=64)
    assert rain.recipe == "mindie_sd_rainfusion" and rain.flops == pytest.approx(2_500_000.0)
    role = (("query_shape", (1, 256, 1, 128)), ("key_shape", (1, 256, 1, 128)),
            ("value_shape", (1, 256, 1, 128)))
    qbsa = sd_lower_op(
        replace(attn, metadata=role + (("k_dtype", "fp8_e4m3"), ("v_dtype", "fp8_e4m3"))),
        op_name="eagle_qbsa", sparsity_ratio=0.5, block_size=64)
    bsa = sd_lower_op(
        replace(attn, metadata=role + (("k_dtype", "bf16"), ("v_dtype", "bf16"))),
        op_name="eagle_bsa", sparsity_ratio=0.5, block_size=64)
    assert qbsa.hbm_bytes < bsa.hbm_bytes
    laurel = sd_lower_op(_make_op("gemm", local_shape=(128, 1024)),
                         op_name="laurel", low_rank=32)
    assert laurel.flops == pytest.approx(4 * 128 * 1024 * 32)
    q46 = sd_lower_op(_make_op("quantize", local_shape=(64, 128)),
                      op_name="quant_four_over_six_a5", soc_family=SOC_950)
    assert "mxfp4_scale" in q46.returns
    sd = lower_operators(
        (_make_op("indexed_attention", local_shape=(128, 128)),
         _make_op("rmsnorm", local_shape=(128, 1024)), _make_op("gemm", local_shape=(128, 1024))),
        {"indexed_attention": {"implementation": "mindie_sd_ada_bsa",
                               "options": {"sparsity_ratio": 0.5, "block_size": 64}},
         "rmsnorm": "mindie_sd_adalayernorm", "gemm": "mindie_sd_eagle_ffn"})
    assert [op.recipe for op in sd] == [
        "mindie_sd_ada_bsa", "mindie_sd_adalayernorm", "mindie_sd_eagle_ffn"]
    implicit = sd_lower_op(
        _make_op("ada_bsa", recipe="indexed_attention", local_shape=(128, 128)),
        sparsity_ratio=0.5, block_size=64)
    assert implicit.recipe == "mindie_sd_ada_bsa"
    steps = compute_mindie_sd_effective_denoising_steps(
        infer_steps=50, warmup_steps=2, cache_interval=2)
    assert (steps["computed_steps"], steps["cached_steps"]) == (26, 24) and steps["speedup_factor"] == pytest.approx(50.0 / 26.0)
    with pytest.raises(ModelingError, match="mindie_sd_custom_op_soc_unsupported"):
        sd_lower_op(_make_op("quantize"), op_name="quant_four_over_six_a5",
                    soc_family="Ascend910B")
def test_capability_registries_match_registered_implementations():
    for operators, module, implementation_name in (
            (XLLM_CUSTOM_OPS, "mindie_sim.prediction.operators.custom_framework", "xllm_custom_xllm_math"),
            (MINDIE_SD_CUSTOM_OPS, "mindie_sim.prediction.operators.diffusion", "mindie_sd_custom_ops")):
        capabilities = xllm_capabilities() if module.endswith("custom_framework") else sd_capabilities()
        assert set(capabilities["operators"]) == set(operators)
        registered = {name for name, item in IMPLEMENTATIONS.items() if item.module == module}
        assert set(capabilities["implementations"]) == registered
        for name, spec in capabilities["operators"].items():
            assert spec["native_kind"] == operators[name]["native_kind"]
            assert set(spec["families"]) == (
                set(spec["selector_families"]) | set(spec["direct_only_families"]))
            for family in spec["selector_families"]:
                assert family in IMPLEMENTATIONS[implementation_name].operators
@pytest.mark.parametrize(("run", "match"), (
    (lambda: sd_lower_op(_make_op("rmsnorm", local_shape=(128, 1024))), "requires_explicit_name"),
    (lambda: sd_lower_op(_make_op("indexed_attention", local_shape=(128, 128),
                                  flops=1_000_000.0), op_name="rainfusion"),
     "requires_declared_sparsity_ratio"),
    (lambda: sd_lower_op(_make_op("indexed_attention", local_shape=(128, 128),
                                  flops=1_000_000.0), op_name="eagle_qbsa",
                         sparsity_ratio=0.5, block_size=64), "requires_declared_cache_dtype"),
    (lambda: sd_lower_op(_make_op("indexed_attention", local_shape=(128, 128), flops=0.0),
                         op_name="rainfusion", sparsity_ratio=0.5, block_size=64),
     "sparse_requires_declared_logical_flops"),
))
def test_mindie_sd_declarations_keep_exact_rejection_codes(run, match):
    with pytest.raises(ModelingError, match=match):
        run()
def test_cache_write_family_conserves_valid_rows_and_pages():
    view = {"key_shape": (8, 1, 256), "value_shape": (8, 1, 256), "key_dtype": "bf16",
            "value_dtype": "bf16", "slot_shape": (8,)}
    declared = cache_store_op(op_id="s0.l3.cache", deps=(), stage=0, layer=3, requests=(),
                              phase="decode", recipe="kv_store", rows=8, row_bytes=1024,
                              dtype="bf16", kind="gqa",
                              metadata=(("cache_tensor_view", view),
                                        ("slot_mapping", tuple(range(8))),
                                        ("cache_capacity_slots", 8)))
    lowered = xllm_lower_op(declared, op_name="reshape_and_cache_a5", soc_family=SOC_950,
                            kv_page_size=128)
    meta = dict(lowered.metadata)
    assert meta["cache_row_bytes"] == 1024 and meta["cache_valid_write_rows"] == 8 and lowered.mte3_bytes == 8 * 1024 and lowered.deps == declared.deps
    vendor = xllm_lower_op(_cache_op((8, 2, 128), tuple(range(8))),
                           op_name="reshape_and_cache_a5", soc_family=SOC_950, kv_page_size=128)
    assert dict(vendor.metadata)["cache_row_basis"] == "vendor_shape_elements_times_dtype" and vendor.mte3_bytes == 8 * 2 * 128 * 2
    page4 = xllm_lower_op(_cache_op((8, 2, 128), tuple(range(8))),
                          op_name="reshape_and_cache_a5", soc_family=SOC_950, kv_page_size=4)
    assert page4.mte3_bytes == vendor.mte3_bytes
    invalid = xllm_lower_op(_cache_op((3, 2, 256), (0, -1, 2), capacity=2),
                            op_name="reshape_and_cache_a5", soc_family=SOC_950, kv_page_size=128)
    assert dict(invalid.metadata)["cache_valid_write_rows"] == 1 and invalid.mte3_bytes == 1 * 2 * 256 * 2
    unknown = xllm_lower_op(_cache_op((3, 2, 128)), op_name="reshape_and_cache_a5",
                            soc_family=SOC_950, kv_page_size=128)
    assert dict(unknown.metadata)["cache_work_basis"] == "deferred_slot_binding" and unknown.mte3_bytes == 0.0 and unknown.mte2_bytes == 3 * 4
def test_cache_plan_orders_index_read_and_masked_write():
    known = xllm_lower_op(_cache_op((3, 2, 256), (0, -1, 2), capacity=2),
                          op_name="reshape_and_cache_a5",
                          soc_family=SOC_950, kv_page_size=128)
    prepared, plan = _prepare(xllm_recipes, "reshape_and_cache_a5_plan", known)
    stages = [stage.name for stage in plan.program(_context(prepared)).stages]
    assert stages == ["slot_external_ready", "slot_index_read", "masked_cache_write",
                      "cache_publish"]
    assert plan.work.resources["mte2"] == 1024 + 12 and plan.work.resources["mte3"] == 1024
    case = attnres_case(8, hidden=128)
    cache = xllm_lower_op(_cache_op((8, 2, 128), tuple(range(8))),
                          op_name="reshape_and_cache_a5", soc_family=SOC_950, kv_page_size=128)
    consumer = replace(_make_op("elementwise", local_shape=(8, 128)), op_id="after_cache",
                       deps=(cache.op_id,))
    result = graph_costs(case, {0: (cache, consumer)}, input_ready_ms=1,
                         external_inputs={"analysis_input": 0})
    timeline = {row["op_id"]: row for row in result["timeline"]}
    assert timeline["after_cache"]["ready_ms"] >= timeline[cache.op_id]["complete_ms"]
    deferred = xllm_lower_op(_cache_op((3, 2, 128)), op_name="reshape_and_cache_a5",
                             soc_family=SOC_950, kv_page_size=128)
    prepared_u, plan_u = _prepare(xllm_recipes, "reshape_and_cache_a5_plan", deferred)
    assert [stage.name for stage in plan_u.program(_context(prepared_u)).stages] == [
        "slot_external_ready", "slot_index_read"]
def test_fused_grouped_region_and_workspace_split():
    matrix, activation, quant = _fused_members()
    direct = xllm_lower_op(replace(matrix, recipe="moe_grouped_matmul_swiglu_quant"),
                           op_name="moe_grouped_matmul_swiglu_quant")
    config = dict(direct.metadata)["xllm_fused_config"]
    assert config["group_rows"] == (16, 0, 16) and config["workspace_bytes"] == 32 * 256 * 4
    assert config["matrix_flops"] == 2 * 32 * 256 * 128
    assert direct.matrix_flops == (("int8", config["matrix_flops"]),) and direct.mte3_bytes == 32 * 128 + 32 * 4 + 32 * 256 * 4
    prepared, plan = _prepare(xllm_recipes, "moe_grouped_matmul_swiglu_quant", direct,
                              _geometry())
    context = _context(prepared, REG_NAMES, rates={n: 1e9 for n in REG_NAMES},
                       pool_names={"gm": "gm"}, pool_units={"gm": 1})
    stages = {stage.name: stage for stage in plan.program(context).stages}
    assert stages["matrix_0"].acquire and stages["workspace_read_0"].free and plan.work.resources["mte3"] == direct.mte3_bytes and plan.work.resources["hbm"] == direct.hbm_bytes
    assert type(RecipeBook.from_mapping({"lowerings": {
        "grouped_gemm": "xllm_moe_grouped_matmul_swiglu_quant"}}).select(direct)).__name__ == "PlannedRecipe"
    granted = replace(direct, metadata=direct.metadata + (("workspace_grant", "caller-held"),))
    _, grant_plan = _prepare(xllm_recipes, "moe_grouped_matmul_swiglu_quant", granted,
                             _geometry())
    assert grant_plan.program(context).stages
    region = fused_region_contract((matrix, activation, quant), {quant.op_id}, _Selection())
    assert region["config"] == config
    with pytest.raises(ModelingError, match="exclusive_member_chain"):
        fused_region_contract((matrix, activation, quant),
                              {quant.op_id, activation.op_id}, _Selection())
    with pytest.raises(ModelingError, match="fp8_requires_owner_recipe"):
        xllm_lower_op(replace(matrix, recipe="moe_grouped_matmul_swiglu_quant",
                              dtype="fp8_e4m3"), op_name="moe_grouped_matmul_swiglu_quant")
    split_op = xllm_lower_op(_fused_op(limit=8), op_name="moe_grouped_matmul_swiglu_quant")
    assert dict(split_op.metadata)["xllm_fused_config"]["workspace_live_bytes"] == 16 * 256 * 4
    split_prep, split_plan = _prepare(xllm_recipes, "moe_grouped_matmul_swiglu_quant", split_op, _geometry())
    split_ctx = replace(context, op=split_prep)
    split_stages = {s.name: s for s in split_plan.program(split_ctx).stages}
    assert split_stages["matrix_0"].acquire[0] is split_stages["matrix_2"].acquire[0] and split_stages["workspace_read_0"].free[0] is split_stages["matrix_0"].acquire[0]
    split_eng = Engine(tuple(Resource(n, 1e9) for n in REG_NAMES), pools={"gm": 2 * 8 * 256 * 4}, pool_units={"gm": 1}, trace_limit=50)
    done = split_plan.program(split_ctx).start(split_eng, "fused", split_eng.submit("ready"))
    split_eng.advance(until=(done,))
    assert (split_eng.peak_allocated["gm"], split_eng.allocated["gm"], [r["task"] for r in split_eng.trace].count("fused:matrix_2")) == (2 * 8 * 256 * 4, 0, 1)
    wide = xllm_lower_op(_fused_op(rows=65537, groups=(65537, 0)),
                         op_name="moe_grouped_matmul_swiglu_quant")
    prepared, plan = _prepare(xllm_recipes, "moe_grouped_matmul_swiglu_quant", wide, _geometry())
    assert plan.metadata["workspace_window_rows"] == (32768, 32768, 1) and plan.metadata["workspace_live_bytes"] == 2 * 32768 * 256 * 4
def test_fused_epilogue_competes_on_shared_vector_resource():
    lowered = xllm_lower_op(_fused_op(), op_name="moe_grouped_matmul_swiglu_quant")
    prepared, plan = _prepare(xllm_recipes, "moe_grouped_matmul_swiglu_quant", lowered,
                              _geometry(reg=False))
    context = _context(prepared, CORE_NAMES, rates={"hbm": 1e9, "mte2": 1e9, "mte3": 1e9,
                                                    "vector": 1.0, "cube": 1e9})
    engine = _engine(CORE_NAMES)
    ready = engine.submit("ready")
    fused = plan.program(context).start(engine, "fused", ready)
    external = engine.submit("external", {"vector": 50.0})
    engine.advance(until=(fused, external), first=False)
    assert engine.serviced["vector"] == plan.work.resources["vector"] + 50.0 and fused.time > external.time
def test_mindie_sd_norm_rope_and_eagle_ffn_contracts():
    base = replace(_make_op("norm_rope_concat", local_shape=(1, 64, 2, 128)),
                   metadata=(("query_shape", (1, 64, 2, 128)), ("key_shape", (1, 64, 2, 128)),
                             ("value_shape", (1, 64, 2, 128)), ("norm_type", 0),
                             ("rope_type", 0), ("concat_order", 0)))
    lowered = sd_lower_op(base, op_name="norm_rope_concat")
    contract = _sd_contract(lowered)
    assert {item["name"] for item in contract["writes"]} == {
        "query_output", "key_output", "value_output"}
    assert lowered.mte3_bytes == 3 * 1 * 64 * 2 * 128 * 2 and lowered.vector_ops == 3 * 1 * 64 * 2 * 128
    enabled = replace(base, metadata=(("query_shape", (1, 64, 2, 128)),
                                      ("key_shape", (1, 64, 2, 128)),
                                      ("value_shape", (1, 64, 2, 128)),
                                      ("encoder_query_shape", (1, 8, 2, 128)),
                                      ("encoder_key_shape", (1, 8, 2, 128)),
                                      ("encoder_value_shape", (1, 8, 2, 128)),
                                      ("norm_type", 2), ("norm_added_type", 2),
                                      ("rope_type", 1), ("rope_len", 8), ("is_training", True)))
    rich = sd_lower_op(enabled, op_name="norm_rope_concat")
    assert _write(_sd_contract(rich), "query_output")["shape"] == (1, 2, 72, 128)
    aliased = replace(base, metadata=dict(base.metadata) and tuple(dict(base.metadata).items())
                      + (("output_aliases", {"value": "value"}),))
    assert _write(_sd_contract(sd_lower_op(aliased, op_name="norm_rope_concat")),
                  "value_output")["materialized"] is False
    ffn = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0, mte2_bytes=0.0,
                           mte3_bytes=0.0),
                  metadata=(("x_shape", (1, 128)), ("weight1_shape", (256, 128)),
                            ("weight2_shape", (128, 256)), ("bias1_shape", (256,)),
                            ("bias2_shape", (128,)), ("bias_dtype", "bf16"),
                            ("activation", "silu")))
    ffn_low = sd_lower_op(ffn, op_name="eagle_ffn")
    ffn_contract = _sd_contract(ffn_low)
    matrices = [item for item in ffn_contract["compute"] if item["kind"] == "matrix"]
    assert len(matrices) == 2 and ffn_low.flops == 131072 and ffn_low.mte3_bytes == 1 * 256 * 2 + 1 * 128 * 2
    prepared, plan = _prepare(sd_recipes, "mindie_sd_eagle_ffn", ffn_low)
    context = _context(prepared, CORE_NAMES, compute_peaks={"bf16": 1e9},
                       pool_names={"gm": "gm"}, pool_units={"gm": 1})
    stages = {stage.name: stage for stage in plan.program(context).stages}
    assert set(stages["compute_linear1"].after) >= {"read_x", "read_weight1"} and "compute_linear1" in stages["compute_activation"].after and "write_hidden_store" in stages["read_hidden"].after and stages["write_hidden_store"].acquire[0].size == 512
def test_mindie_sd_block_contracts_and_shared_recipes():
    adaln = replace(_make_op("rmsnorm", local_shape=(2, 3, 128)),
                    metadata=(("x_shape", (2, 3, 128)), ("scale_shape", (128,)),
                              ("shift_shape", (128,)), ("weight_shape", (128,)),
                              ("bias_shape", (128,)), ("return_stats", True)))
    adaln_low = sd_lower_op(adaln, op_name="adalayernorm")
    assert adaln_low.mte2_bytes == (2 * 3 * 128 * 2 + 4 * 128 * 2) and adaln_low.mte3_bytes == (2 * 3 * 128 * 2 + 2 * 2 * 3 * 1 * 2) and {"y", "mean", "rstd"} <= set(adaln_low.returns)
    laurel = replace(_make_op("gemm", local_shape=(128, 1024), flops=0.0),
                     metadata=(("x_shape", (128, 1024)), ("low_rank", 32),
                               ("weight_a_shape", (1024, 32)), ("weight_b_shape", (32, 1024))))
    laurel_low = sd_lower_op(laurel, op_name="laurel", low_rank=32)
    assert laurel_low.flops == 4 * 128 * 1024 * 32 and "y" in laurel_low.returns
    quant = replace(_make_op("quantize", local_shape=(64, 128)),
                    metadata=(("x_shape", (64, 128)),))
    contract = _sd_contract(sd_lower_op(quant, op_name="quant_four_over_six_a5",
                                        soc_family=SOC_950))
    assert _write(contract, "mxfp4_payload")["bytes"] == 64 * 64 and _write(contract, "mxfp4_scale")["bytes"] == 64 * 4
    for name, lowered in (("adalayernorm", adaln_low), ("laurel", laurel_low)):
        prepared, plan = _prepare(sd_recipes, f"mindie_sd_{name}", lowered)
        assert plan.work.resources["mte3"] == lowered.mte3_bytes
def test_sparse_attention_roles_masks_and_layouts():
    op = replace(_make_op("indexed_attention", local_shape=(1, 32, 4, 64)),
                 metadata=(("query_shape", (1, 32, 4, 64)), ("key_shape", (1, 64, 2, 64)),
                           ("value_shape", (1, 64, 2, 64)),
                           ("block_mask", (1, 0, 1, 0, 1, 1, 0, 1)),
                           ("kv_dtype", "fp8_e4m3")))
    lowered = sd_lower_op(op, op_name="rainfusion", sparsity_ratio=0.5, block_size=16,
                          kv_dtype="fp8_e4m3")
    contract = _sd_contract(lowered)
    reads = {item["name"]: item for item in contract["reads"]}
    assert reads["query"]["bytes"] == 1 * 32 * 4 * 64 * 2 and reads["kv_load"]["bytes"] == 4 * 16 * 2 * (64 + 64) * 1 and contract["attributes"]["active_block_pairs"] == 5 and lowered.flops == 2 * 5 * 16 * 16 * 4 * (64 + 64)
    batch = replace(op, metadata=(("query_shape", (2, 32, 4, 64)),
                                  ("key_shape", (2, 64, 2, 64)), ("value_shape", (2, 64, 2, 64)),
                                  ("block_mask", (1, 0, 1, 0, 1, 1, 0, 1)),
                                  ("q_scale_shape", (2, 32, 4)), ("k_scale_shape", (2, 8, 2)),
                                  ("v_scale_shape", (2, 8, 2)), ("lse_shape", (2, 32, 4))))
    batch_low = sd_lower_op(batch, op_name="rainfusion", sparsity_ratio=0.5, block_size=16)
    assert batch_low.flops == 2 * 1_310_720 and "softmax_lse" in batch_low.returns
    bnsd = replace(_make_op("indexed_attention", local_shape=(1, 4, 128, 128)),
                   metadata=(("layout", "BNSD"), ("query_shape", (1, 4, 128, 128)),
                             ("key_shape", (1, 2, 256, 128)), ("value_shape", (1, 2, 256, 128)),
                             ("block_mask", (1, 1)), ("return_softmax_lse", True)))
    bnsd_low = sd_lower_op(bnsd, op_name="rainfusion", sparsity_ratio=0.0, block_size=128)
    assert bnsd_low.flops == 67_108_864 and _write(_sd_contract(bnsd_low), "softmax_lse")["bytes"] == 2048
    tnd = replace(_make_op("indexed_attention", local_shape=(128, 4, 128)),
                  metadata=(("layout", "TND"), ("query_shape", (128, 4, 128)),
                            ("key_shape", (256, 2, 128)), ("value_shape", (256, 2, 128)),
                            ("query_lengths", (128,)), ("key_lengths", (256,)),
                            ("value_lengths", (256,)), ("block_mask", (1, 1)),
                            ("return_softmax_lse", True)))
    tnd_low = sd_lower_op(tnd, op_name="rainfusion", sparsity_ratio=0.0, block_size=128)
    assert tnd_low.flops == 67_108_864 and dict(tnd_low.metadata)["layout"] == "TND"
    per_head = replace(_make_op("indexed_attention", local_shape=(1, 2, 128, 128)),
                       metadata=(("layout", "BNSD"), ("query_shape", (1, 2, 128, 128)),
                                 ("key_shape", (1, 2, 256, 128)),
                                 ("value_shape", (1, 2, 256, 128)),
                                 ("block_mask", (((1, 0), (0, 1)),)),
                                 ("mask_dtype", "int8")))
    head_low = sd_lower_op(per_head, op_name="rainfusion", sparsity_ratio=0.0, block_size=128)
    assert head_low.flops == 16_777_216
    assert _sd_contract(head_low)["attributes"]["mask_layout"] == "B,H,Qblock,KVblock"
    tail = replace(_make_op("indexed_attention", local_shape=(1, 20, 4, 64)),
                   metadata=(("query_shape", (1, 20, 4, 64)),
                             ("key_shape", (1, 40, 2, 64)), ("value_shape", (1, 40, 2, 64)),
                             ("block_mask", (1, 1, 1, 1, 1, 1))))
    assert sd_lower_op(tail, op_name="rainfusion", sparsity_ratio=0.5,
                       block_size=16).flops == 2 * 800 * 4 * (64 + 64)
def test_smla_ports_and_onerec_result_width():
    smla = replace(_make_op("mla_attention", local_shape=(2, 8, 4, 64), mte2_bytes=0.0,
                            mte3_bytes=0.0),
                   metadata=(("q_shape", (2, 8, 4, 64)), ("ori_kv_shape", (2, 64, 2, 64)),
                             ("cmp_kv_shape", (2, 32, 2, 64)),
                             ("sparse_index_count", 8), ("return_softmax_lse", True)))
    lowered = xllm_lower_op(smla, op_name="smla_va", return_lse=True)
    meta = dict(lowered.metadata)
    assert set(meta["xllm_ports"]["outputs"]) >= {"attn_out", "softmax_lse"}
    assert lowered.mte2_bytes == (2 * 8 * 4 * (64 + 64) * 2 + 2 * 64 * 2 * 64 * 2
                                  + 2 * 32 * 2 * 64 * 2 + 8 * 4)
    assert lowered.mte3_bytes == 2 * 8 * 4 * 64 * 2 + 2 * 8 * 4 * 4
    sparse = replace(_make_op("indexed_attention", local_shape=(1, 32, 4, 64)),
                     metadata=(("q_shape", (1, 32, 4, 64)), ("key_shape", (1, 64, 2, 64)),
                               ("value_shape", (1, 64, 2, 64)),
                               ("sparse_index_count", 16), ("return_lse", True)))
    sparse_low = xllm_lower_op(sparse, op_name="sparse_flash_attention_lse", return_lse=True)
    assert set(dict(sparse_low.metadata)["xllm_ports"]["inputs"]) >= {
        "query", "key", "value", "sparse_indices"}
    beam = xllm_lower_op(replace(_make_op("beam_select", local_shape=(1, 128), mte3_bytes=0.0),
                                 metadata=(("beam_width", 16), ("topk", 128),
                                           ("result_width", 64), ("batch", 1),
                                           ("sequence_capacity", 3), ("current_step", 2))),
                         op_name="onerec_final_beam_select")
    assert beam.mte3_bytes == 3 * 64 * 4 + 1 * 64 * 3 * 4 and dict(beam.metadata)["xllm_beam_selection"]["sequence_shape"] == (1, 64, 3)
def test_mega_gdn_state_recurrence_and_slot_versions():
    lowered = xllm_lower_op(_gdn(rows=1, read=(7,), write=(2,), versions=((1,), (2,)),
                                 capacity=9), op_name="mega_gdn_decode")
    meta = dict(lowered.metadata)["xllm_mega_gdn_config"]
    assert meta["read_state_indices"] == (7,) and meta["conv_state_shape"] == (9, 3, 768)
    prepared, plan = _prepare(xllm_recipes, "mega_gdn_decode", lowered)
    effects = plan.metadata["gdn_state_effects"]
    assert any(item["target"] == "conv_state" and item["slot"] == 7 and item["action"] == "read"
               and item["version"] == 1 for item in effects)
    assert any(item["target"] == "ssm_state" and item["slot"] == 2 and item["action"] == "write"
               and item["version"] == 2 and item["span_bytes"] == 2 * 128 * 128 * 4
               and item["commit_event"] for item in effects)
    assert plan.metadata["two_states"] == ("conv_state_out", "ssm_state_out") and plan.metadata["state_program"] == "gdn_recurrent.program"
    stages = [stage.name for stage in plan.program(_context(prepared, pool_names={"ub": "ub"},
                                                            pool_units={"ub": 64})).stages]
    assert stages == ["conv", "qk_norm_gate", "recurrent_state", "readout_rmsnorm_zgate"]
    negative = xllm_lower_op(_gdn(), op_name="mega_gdn_decode")
    _, negative_plan = _prepare(xllm_recipes, "mega_gdn_decode", negative)
    assert any(item["slot"] == 0 for item in negative_plan.metadata["gdn_state_effects"])
def test_eagle_ffn_engine_capacity_and_event_ordering():
    op = replace(_make_op("ffn", local_shape=(1, 128), flops=0.0, mte2_bytes=0.0,
                          mte3_bytes=0.0),
                 metadata=(("x_shape", (1, 128)), ("weight1_shape", (128, 256)),
                           ("weight2_shape", (256, 128)), ("activation", "silu")))
    lowered = sd_lower_op(op, op_name="eagle_ffn")
    prepared, plan = _prepare(sd_recipes, "mindie_sd_eagle_ffn", lowered)
    context = _context(prepared, CORE_NAMES, pool_names={"gm": "gm"}, pool_units={"gm": 1})
    engine = _engine(CORE_NAMES, gm=511)
    program = plan.program(context)
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        program.start(engine, "ffn", engine.submit("ready"))
    granted = replace(prepared, metadata=tuple(
        (key, value) for key, value in dict(prepared.metadata).items()
        if key != "workspace_grant") + (("workspace_grant", "caller-held"),))
    stages = {stage.name: stage for stage in plan.program(replace(context, op=granted)).stages}
    assert not stages["write_hidden_store"].acquire
    engine = _engine(CORE_NAMES, gm=512)
    done = program.start(engine, "ffn", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.peak_allocated["gm"] == 512 and engine.allocated["gm"] == 0
    engine = _engine(CORE_NAMES, gm=512)
    done = program.start(engine, "ffn", engine.submit("ready"))
    engine.advance(until=(done,))
    rows = {row["task"].removeprefix("ffn:"): row for row in engine.trace}
    assert rows["compute_linear1"]["start_ms"] >= rows["read_weight1"]["end_ms"] and rows["write_hidden_store"]["start_ms"] >= rows["compute_activation"]["end_ms"]
def test_standalone_aclnn_declarations_and_single_call_lowering():
    declarations = (
        {"api": "aclnnMatmul", "tensors": {
            "x": {"shape": [32, 128], "dtype": "bf16"}, "w": {"shape": [128, 32], "dtype": "bf16"},
            "out": {"shape": [32, 32], "dtype": "bf16"}}, "phase": "decode"},
        {"api": "aclnnFlashAttentionScore", "tensors": {
            "query": {"shape": [1, 8, 64], "dtype": "bf16"},
            "key": {"shape": [1, 8, 64], "dtype": "bf16"},
            "value": {"shape": [1, 8, 32], "dtype": "bf16"}},
         "attrs": {"causal": True}, "phase": "prefill"},
        {"api": "aclnnRmsNorm", "tensors": {
            "x": {"shape": [16, 128], "dtype": "bf16"},
            "gamma": {"shape": [128], "dtype": "bf16"},
            "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
        {"api": "aclnnAddRmsNorm", "tensors": {
            "x": {"shape": [16, 128], "dtype": "bf16"},
            "residual": {"shape": [16, 128], "dtype": "bf16"},
            "gamma": {"shape": [128], "dtype": "bf16"},
            "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
        {"api": "aclnnSwiGlu", "tensors": {
            "x": {"shape": [16, 256], "dtype": "bf16"},
            "y": {"shape": [16, 128], "dtype": "bf16"}}, "phase": "decode"},
        {"api": "aclnnDynamicQuant", "tensors": {
            "x": {"shape": [16, 128], "dtype": "bf16"},
            "y": {"shape": [16, 128], "dtype": "int8"}}, "phase": "decode"},
    )
    for declaration in declarations:
        call = build_aclnn_call(declaration)
        assert call.ops and call.ops[0].hbm_bytes > 0.0
    fp32 = build_aclnn_call({"api": "aclnnMatmul", "tensors": {
        "x": {"shape": [32, 128], "dtype": "fp32"}, "w": {"shape": [128, 32], "dtype": "fp32"},
        "out": {"shape": [32, 32], "dtype": "fp32"}}, "phase": "decode"}).ops[0]
    assert fp32.dtype == "fp32" and fp32.hbm_bytes == pytest.approx(36864.0)
    causal = {"api": "aclnnFlashAttentionScore", "tensors": {
        "query": {"shape": [1, 8, 64], "dtype": "bf16"},
        "key": {"shape": [1, 8, 64], "dtype": "bf16"},
        "value": {"shape": [1, 8, 32], "dtype": "bf16"}},
        "attrs": {"causal": True}, "phase": "prefill"}
    dense = build_aclnn_call(causal).ops[0]
    paged = build_aclnn_call({**causal, "block_size": 128}).ops[0]
    assert dense.flops == pytest.approx(6912.0) == paged.flops
    environment = ExecutionEnvironment.from_hardware(
        Hardware.from_mapping(materialize({"sku": A5_SKU})))
    declaration = {"builder": "aclnn", "arguments": {
        "api": "aclnnMatmul", "m": 2048, "n": 4096, "k": 4096, "dtype": "bf16",
        "phase": "prefill"}}
    report = simulate_aclnn_call(environment, declaration)
    assert report["schema"] == "mindie-sim.call-costs/1" and report["replay_scope"] == "none" and report["timeline"][0]["program"] == "SteadyMatmul"
    assert simulate_aclnn_call(environment, declaration)["duration_ms"] == pytest.approx(
        report["duration_ms"])
    assert build_call(declaration) is not None

def test_exact_mindie_sd_recipe_name_and_workspace_pool_fallbacks():
    op = replace(_make_op("rainfusion", recipe="mindie_sd_rainfusion",
                          local_shape=(128, 128), flops=1_000_000.0),
                 metadata=(("sparsity_ratio", 0.5), ("block_size", 64)))
    assert sd_lower_op(op).recipe == "mindie_sd_rainfusion"
    ffn = sd_lower_op(replace(_make_op("ffn", local_shape=(1, 128), flops=0.0, mte2_bytes=0.0, mte3_bytes=0.0),
                              metadata=(("x_shape", (1, 128)), ("weight1_shape", (128, 256)),
                                        ("weight2_shape", (256, 128)), ("activation", "silu"))), op_name="eagle_ffn")
    prepared, plan = _prepare(sd_recipes, "mindie_sd_eagle_ffn", ffn)
    hbm_stages = {s.name: s for s in plan.program(_context(prepared, CORE_NAMES, pool_names={"hbm": "hbm"}, pool_units={"hbm": 1})).stages}
    no_pool_stages = {s.name: s for s in plan.program(_context(prepared, CORE_NAMES, pool_names={}, pool_units={})).stages}
    assert hbm_stages["write_hidden_store"].acquire and not no_pool_stages["write_hidden_store"].acquire
