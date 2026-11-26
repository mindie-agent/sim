"""External framework operator libraries: MindIE-SD and aclnn calls.
[Role] Prove that third-party operator libraries lower into the same canonical
  occurrence model for MindIE-SD diffusion contracts, standalone
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
from _support.axes import GEN5_DT_SKU as A5_SKU, SOC_950
from mindie_sim import simulate_aclnn_call
from mindie_sim.backends.ascend import materialize
from mindie_sim.prediction.call_builders import build_aclnn_call, build_call
from mindie_sim.prediction.operators.diffusion import (
    MINDIE_SD_CUSTOM_OPS, capabilities as sd_capabilities,
    compute_mindie_sd_effective_denoising_steps,
    lower_op as sd_lower_op, recipes as sd_recipes)
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.implementations import IMPLEMENTATIONS, lower_operators
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.public.environment import ExecutionEnvironment
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
pytestmark = pytest.mark.tier0
BASE_GEOMETRY = Geometry(64, {"vector": 64, "mte2": 64, "mte3": 64}, {"ub": 196608}, {"ub": 64})
CORE_NAMES = ("hbm", "mte2", "mte3", "vector", "cube")
def _make_op(family, *, recipe="", local_shape=(16, 4096), flops=1_000_000.0,
             mte2_bytes=65536.0, mte3_bytes=32768.0, vector_ops=1024.0, dtype="bf16",
             metadata=()):
    return OpSpec(op_id=f"test.{family}", family=family, phase="decode", stage=0, layer=0,
                  requests=("req-0",), semantic_shape=local_shape, local_shape=local_shape,
                  padded_shape=local_shape, launch_shape=local_shape, flops=flops,
                  hbm_bytes=mte2_bytes + mte3_bytes, mte2_bytes=mte2_bytes,
                  mte3_bytes=mte3_bytes, vector_ops=vector_ops,
                  recipe=recipe or family, dtype=dtype, metadata=metadata)
def _context(op, names=CORE_NAMES, **changes):
    values = dict(op=op, work={}, resource_names={n: n for n in names},
                  rates={n: 1e9 for n in names}, core_count=1,
                  compute_peaks={"bf16": 1e9, "int8": 1e9})
    values.update(changes)
    return RecipeContext(**values)
def _engine(names=CORE_NAMES, **pools):
    return Engine(tuple(Resource(n, 1e9) for n in names), pools=pools,
                  pool_units={name: 1 for name in pools}, trace_limit=100)
def _prepare(book, name, op, geometry=BASE_GEOMETRY):
    prepared, plan = book()[name].prepare(op, geometry)
    return prepared, plan
def _sd_contract(lowered):
    return dict(lowered.metadata)["mindie_sd_contract"]
def _write(contract, name):
    return next(item for item in contract["writes"] if item["name"] == name)
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
def test_capability_registry_matches_registered_diffusion_implementations():
    capabilities = sd_capabilities()
    assert set(capabilities["operators"]) == set(MINDIE_SD_CUSTOM_OPS)
    registered = {name for name, item in IMPLEMENTATIONS.items()
                  if item.module == "mindie_sim.prediction.operators.diffusion"}
    assert set(capabilities["implementations"]) == registered
    for name, spec in capabilities["operators"].items():
        assert spec["native_kind"] == MINDIE_SD_CUSTOM_OPS[name]["native_kind"]
        selectors = set(spec["selector_families"])
        assert set(spec["families"]) == selectors | set(spec["direct_only_families"])
        assert selectors <= set(IMPLEMENTATIONS["mindie_sd_custom_ops"].operators)
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
    bnsd_lse = _write(_sd_contract(bnsd_low), "softmax_lse")
    assert bnsd_low.flops == 67_108_864 and (bnsd_lse["shape"], bnsd_lse["dtype"], bnsd_lse["bytes"]) == ((1, 4, 128), "bf16", 1024)
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


def test_aclnn_call_builders_honor_batched_seqlens_band_nz_stats_and_quant():
    from mindie_sim.prediction.implementations import produced_values

    # Padded BNSD decode honors actual_seq_lengths_kv rather than full KV tensor length.
    padded_decode = build_aclnn_call({
        "api": "aclnnFusedInferAttentionScore",
        "tensors": {"query": [2, 1, 1, 64], "key": [2, 1, 128, 64], "value": [2, 1, 128, 64]},
        "attrs": {"input_layout": "BNSD", "num_heads": 1, "actual_seq_lengths_kv": [1, 3]},
    }).ops[0]
    assert int(padded_decode.flops) == 2 * (64 + 64) * (1 + 3)

    # Sparse band mode (sparse_mode=4) prices exact pre_tokens/next_tokens window and touched KV rows.
    band_tensors = {k: [1, 1, 128, 64] for k in ("query", "key", "value")}
    band_tensors["atten_mask"] = {"shape": [2048, 2048], "dtype": "bool"}
    band_op = build_aclnn_call({
        "api": "aclnnFusedInferAttentionScore",
        "tensors": band_tensors,
        "attrs": {"input_layout": "BNSD", "num_heads": 1, "sparse_mode": 4,
                  "pre_tokens": 1, "next_tokens": 0},
    }).ops[0]
    assert int(band_op.flops) == (1 + 127 * 2) * 2 * (64 + 64)
    wide_band = build_aclnn_call({
        "api": "aclnnFusedInferAttentionScore",
        "tensors": {"query": [1, 1, 128, 64], "key": [1, 1, 4096, 64], "value": [1, 1, 4096, 64]},
        "attrs": {"input_layout": "BNSD", "num_heads": 1, "sparse_mode": 4,
                  "pre_tokens": 127, "next_tokens": 0},
    }).ops[0]
    assert int(wide_band.mte2_bytes) == (128 + 2 * 255) * 64 * 2

    # FRACTAL_NZ weight layout decodes 4D fractal tiles, propagates layout, and dtype-only out still works.
    nz_op = build_aclnn_call({
        "api": "aclnnMatmulWeightNz",
        "tensors": {
            "x": {"shape": [16, 16], "dtype": "bf16"},
            "weight": {"shape": [2, 1, 16, 16], "dtype": "bf16", "layout": "FRACTAL_NZ"},
            "out": {"shape": [16, 17], "dtype": "bf16"},
        },
    }).ops[0]
    assert nz_op.local_shape == (16, 17, 16)
    assert dict(nz_op.metadata).get("weight_layout") == "FRACTAL_NZ"
    assert int(nz_op.flops) == 2 * 16 * 17 * 16
    assert int(nz_op.mte3_bytes) == 16 * 17 * 2
    dtype_only_out = build_aclnn_call({
        "api": "aclnnMatmul", "m": 16, "n": 16, "k": 16,
        "tensors": {"out": {"dtype": "bf16"}},
    }).ops[0]
    assert dtype_only_out.local_shape == (16, 16, 16) and dtype_only_out.dtype == "bf16"

    # FlashAttentionScore includes and publishes mandatory FP32 softmaxMaxOut and softmaxSumOut.
    fa_op = build_aclnn_call({
        "api": "aclnnFlashAttentionScore",
        "tensors": {"query": [1, 1, 16, 64], "key": [1, 1, 16, 64], "value": [1, 1, 16, 64]},
        "attrs": {"input_layout": "BNSD", "num_heads": 1},
    }).ops[0]
    assert int(fa_op.mte3_bytes) == 16 * 64 * 2 + 2 * 16 * 8 * 4
    assert fa_op.returns == ("softmax_max", "softmax_sum")
    assert [v["output"] for v in produced_values(fa_op)] == ["output", "softmax_max", "softmax_sum"]

    # DynamicQuant accounts for smoothScalesOptional reads/ops, INT4 width, and published row scale.
    dq_unsmoothed = build_aclnn_call({
        "api": "aclnnDynamicQuant",
        "tensors": {"x": {"shape": [1, 128], "dtype": "bf16"}, "y": {"shape": [1, 128], "dtype": "int4"}},
    }).ops[0]
    dq_op = build_aclnn_call({
        "api": "aclnnDynamicQuant",
        "tensors": {
            "x": {"shape": [1, 128], "dtype": "bf16"},
            "smoothScalesOptional": {"shape": [128], "dtype": "bf16"},
            "y": {"shape": [1, 128], "dtype": "int4"},
        },
    }).ops[0]
    assert (int(dq_op.mte2_bytes), int(dq_op.mte3_bytes)) == ((128 + 128) * 2, 128 // 2 + 4)
    assert dq_op.vector_ops - dq_unsmoothed.vector_ops == pytest.approx(128.0)
    assert dq_op.returns == ("activation_scales",)
    assert [(v["output"], v["dtype"]) for v in produced_values(dq_op)] == [("payload", "int4"), ("activation_scales", "fp32")]


def test_feedforward_shared_overlap_tp_pack_situ_quant_and_padding_selection():
    from _support.axes import REPRESENTATIVE_SKUS
    from _support.hardware import sku_case
    from _support.operator_cases import declared_case
    from mindie_sim import graph_costs, target_operators
    from mindie_sim.catalog.quant import quant_preset
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.implementations import produced_values
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import IterationPlan, RequestView

    # Single-device shared-expert overlap keeps producer before join consumer.
    single = declared_case()
    single["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2,
        num_shared_experts=1, moe_intermediate_size=32,
    )
    single["runtime"]["multistream_overlap_shared_expert"] = True
    overlap_ops = target_operators(
        single, scope="body", phase="decode", tokens=1, context=8, resource_fusion=False,
    )
    ids = {op.op_id for op in overlap_ops}
    ext = {dep: 0 for op in overlap_ops for dep in op.deps if dep not in ids}
    order = {op.op_id: i for i, op in enumerate(overlap_ops)}
    assert not [(order[d], order[o.op_id]) for o in overlap_ops for d in o.deps if d in order and order[d] > order[o.op_id]]
    assert graph_costs(single, overlap_ops, external_inputs=ext, isolated=False)["duration_ms"] > 0.0

    # TP route packing bills each physical rank by TP lane even under permuted EP rank_table.
    tp_moe = declared_case(tp=2, ep=2)
    tp_moe["model"].update(layers=1, ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    tp_moe["runtime"]["kernels"] = {
        "moe_token_partition": {"kind": "vllm_ascend_mc2_tp_slice", "padding": "right_to_tp"},
    }
    for rank in (0, 1):
        ops = target_operators(
            tp_moe, scope="model", phase="prefill", tokens=3,
            physical_rank=rank, apply_body_defaults=False,
        )
        pack = next(op for op in ops if op.recipe == "route_pack")
        rows = dict(pack.metadata)["moe_token_partition"]["valid_rows_per_rank"][rank]
        assert (int(pack.vector_ops), int(pack.mte2_bytes)) == (2 * rows, 72 * rows)
    tp_moe_perm = {**tp_moe, "runtime": {**tp_moe["runtime"], "rank_table": [[1, 0]]}}
    perm_ops = target_operators(
        tp_moe_perm, scope="model", phase="prefill", tokens=3,
        physical_rank=0, apply_body_defaults=False,
    )
    perm_pack = next(op for op in perm_ops if op.recipe == "route_pack")
    assert (int(perm_pack.vector_ops), int(perm_pack.mte2_bytes)) == (4, 144)

    # Shipping quantized SiTU binds fused INT8+FP32 only when fused and scale geometry matches.
    situ_case = parse_case(sku_case(REPRESENTATIVE_SKUS["a3"]))
    layer_plan = replace(situ_case.model.layer_plans[0], ffn_activation="situ", intermediate_size=128)
    situ_model = replace(
        situ_case.model, intermediate_size=128, layers=1,
        layer_plans=(layer_plan,), quant=quant_preset("w8a8_int8"),
    )
    situ_case = replace(
        situ_case, model=situ_model,
        runtime=replace(situ_case.runtime, metadata=(("arch_family", "A3"),)),
    )
    situ_ops = target_operators(
        situ_case, phase="decode", tokens=1, context=4, scope="body", apply_body_defaults=False,
    )
    situ_op = next(op for op in situ_ops if op.family == "situ")
    assert tuple(v["dtype"] for v in produced_values(situ_op)) == ("int8", "fp32")
    assert situ_op.mte3_bytes == pytest.approx(132.0)
    assert [op.op_id for op in situ_ops if situ_op.op_id in op.deps] == ["s0.l0.down"]

    eager_situ = replace(
        situ_case,
        runtime=replace(situ_case.runtime, metadata=(("arch_family", "A3"), ("raw_eager", True))),
    )
    eager_ops = target_operators(
        eager_situ, phase="decode", tokens=1, context=4, scope="body", apply_body_defaults=False,
    )
    eager_cast = next(op for op in eager_ops if op.op_id == "s0.l0.situ")
    assert eager_cast.mte3_bytes == pytest.approx(256.0)
    assert dict(eager_cast.metadata).get("activation_abi_shipping") is False
    assert any(op.op_id == "s0.l0.down.quantize" and op.family == "quantize" for op in eager_ops)

    grouped_quant = replace(quant_preset("w8a8_int8"), activation_group_size=128)
    grouped_layer = replace(situ_case.model.layer_plans[0], ffn_activation="situ", intermediate_size=512)
    grouped_case = replace(
        situ_case,
        model=replace(situ_case.model, intermediate_size=512, layers=1, layer_plans=(grouped_layer,), quant=grouped_quant),
    )
    grouped_ops = target_operators(
        grouped_case, phase="decode", tokens=1, context=4, scope="body", apply_body_defaults=False,
    )
    assert any(op.op_id == "s0.l0.down.quantize" and "s0.l0.situ" in op.deps for op in grouped_ops)

    # Padding RequestViews do not inflate valid_tokens during implementation policy selection.
    rule = {"cases": [{"domain": {"valid_tokens": {"max": 1}}, "binding": "vllm_ascend_native_norm"}], "otherwise": "error"}
    sel_raw = declared_case()
    sel_raw["model"]["layers"] = 1
    sel_raw["runtime"].update(lowering_passes=["residual_norm"], kernels={"lowerings": {"add_rmsnorm": rule}})
    sel_case = parse_case(sel_raw)
    active_view = RequestView("a", 1, 32, True, "decode")
    pad_view = RequestView("pad", 1, 0, False, "decode", padding=True)
    for views in ((active_view,), (active_view, pad_view)):
        plan = IterationPlan(views)
        work = Frontend()(plan, sel_case.model, sel_case.runtime, sel_case.machine)
        assert plan.valid_tokens == 1
        assert any(op.family == "add_rmsnorm" for op in work.ops)


def test_padded_eagle_sparse_attention_preserves_physical_output_and_lse_extents():
    bnsd = replace(
        _make_op("eagle_bsa", local_shape=(2, 4, 64, 128)),
        metadata=(("layout", "BNSD"), ("query_shape", (2, 4, 64, 128)),
                  ("key_shape", (2, 4, 64, 128)), ("value_shape", (2, 4, 64, 128)),
                  ("query_lengths", (16, 20)), ("key_lengths", (16, 20)), ("value_lengths", (16, 20)),
                  ("block_mask", (1, 1)), ("return_softmax_lse", True)),
    )
    bnsd_contract = _sd_contract(sd_lower_op(bnsd, soc_family="A3", sparsity_ratio=0.0, block_size=64))
    bnsd_out = _write(bnsd_contract, "attention_out")
    bnsd_lse = _write(bnsd_contract, "softmax_lse")
    assert bnsd_out["shape"] == (2, 4, 64, 128) and int(bnsd_out["bytes"]) == (16 + 20) * 4 * 128 * 2
    assert (bnsd_lse["shape"], bnsd_lse["dtype"], int(bnsd_lse["bytes"])) == ((2, 4, 64, 1), "fp32", (16 + 20) * 4 * 4)

    # Padded BNSD accepts physical (B, H, ceil(S_phys/bq), ceil(S_kv_phys/bkv)) rectangular masks.
    phys_head = ((1, 0), (0, 0), (0, 0), (0, 0))
    padded_mask = replace(
        _make_op("eagle_bsa", local_shape=(1, 2, 256, 64)),
        metadata=(("layout", "BNSD"), ("query_shape", (1, 2, 256, 64)),
                  ("key_shape", (1, 2, 256, 64)), ("value_shape", (1, 2, 256, 64)),
                  ("query_lengths", (64,)), ("key_lengths", (64,)), ("value_lengths", (64,)),
                  ("block_shape", (64, 128)), ("block_mask", ((phys_head, phys_head),)),
                  ("mask_dtype", "int8")),
    )
    padded_low = sd_lower_op(padded_mask, soc_family="A3", sparsity_ratio=0.0, block_size=128)
    assert padded_low.flops == 4 * 2 * 64 * 64 * 64
    assert int(_write(_sd_contract(padded_low), "attention_out")["bytes"]) == 64 * 2 * 64 * 2

    tnd = replace(
        _make_op("eagle_bsa", local_shape=(36, 4, 128)),
        metadata=(("layout", "TND"), ("query_shape", (36, 4, 128)),
                  ("key_shape", (36, 4, 128)), ("value_shape", (36, 4, 128)),
                  ("query_lengths", (16, 20)), ("key_lengths", (16, 20)), ("value_lengths", (16, 20)),
                  ("block_mask", (1, 1)), ("return_softmax_lse", True)),
    )
    tnd_contract = _sd_contract(sd_lower_op(tnd, soc_family="A3", sparsity_ratio=0.0, block_size=64))
    assert _write(tnd_contract, "attention_out")["shape"] == (36, 4, 128)
    assert (_write(tnd_contract, "softmax_lse")["shape"], _write(tnd_contract, "softmax_lse")["dtype"]) == ((36, 4, 1), "fp32")

    bad_tnd = replace(
        _make_op("eagle_bsa", local_shape=(100, 4, 128)),
        metadata=(("layout", "TND"), ("query_shape", (100, 4, 128)),
                  ("key_shape", (100, 4, 128)), ("value_shape", (100, 4, 128)),
                  ("query_lengths", (16, 20)), ("key_lengths", (16, 20)), ("value_lengths", (16, 20)),
                  ("block_mask", (1, 1))),
    )
    with pytest.raises(ModelingError, match="mindie_sd_sparse_query_TND_lengths_invalid"):
        sd_lower_op(bad_tnd, soc_family="A3", sparsity_ratio=0.0, block_size=64)


def test_eagle_qbsa_enforces_pinned_qmode1_dtypes_and_kv_block_alignment():
    q = (1, 2, 128, 64)
    head = ((1,), (1,), (1,), (1,))
    meta = (("layout", "BNSD"), ("query_shape", q), ("key_shape", q), ("value_shape", q),
            ("block_mask", ((((1,),), ((1,),)),)), ("mask_dtype", "int8"))
    for dt in ("bf16", "fp16"):
        op = replace(_make_op("eagle_qbsa", local_shape=q, dtype=dt), metadata=meta)
        with pytest.raises(ModelingError, match="mindie_sd_eagle_qbsa_cache_dtype_unsupported"):
            sd_lower_op(op, soc_family="A5", sparsity_ratio=0.0, block_size=128, kv_dtype=dt)
    for v_dt in ("fp8_e4m3", "int8"):
        qmode1 = replace(
            _make_op("eagle_qbsa", local_shape=q, dtype="int8"),
            metadata=(("layout", "BNSD"), ("query_shape", q), ("key_shape", q), ("value_shape", q),
                      ("q_dtype", "int8"), ("k_dtype", "int8"), ("v_dtype", v_dt),
                      ("block_shape", (32, 128)), ("block_mask", ((head, head),)), ("mask_dtype", "int8")),
        )
        lowered = sd_lower_op(qmode1, soc_family="A5", sparsity_ratio=0.0, block_size=128)
        assert lowered.flops == 4 * 2 * 128 * 128 * 64
    bad_block = replace(
        _make_op("eagle_qbsa", local_shape=q, dtype="int8"),
        metadata=(("layout", "BNSD"), ("query_shape", q), ("key_shape", q), ("value_shape", q),
                  ("q_dtype", "int8"), ("k_dtype", "int8"), ("v_dtype", "fp8_e4m3"),
                  ("block_shape", (32, 64)), ("block_mask", (1,)), ("mask_dtype", "int8")),
    )
    with pytest.raises(ModelingError, match="mindie_sd_sparse_block_shape_invalid"):
        sd_lower_op(bad_block, soc_family="A5", sparsity_ratio=0.0, block_size=64)


def test_sparse_attention_rectangular_block_shape_overrides_scalar_block_size():
    q = (1, 2, 128, 64)
    head = ((1,), (1,), (1,), (1,))
    op = replace(
        _make_op("eagle_bsa", local_shape=q),
        metadata=(("layout", "BNSD"), ("query_shape", q), ("key_shape", q), ("value_shape", q),
                  ("block_shape", (32, 128)), ("block_mask", ((head, head),)), ("mask_dtype", "int8")),
    )
    for scalar_side in (32, 128):
        lowered = sd_lower_op(op, soc_family="A3", sparsity_ratio=0.0, block_size=scalar_side)
        attrs = _sd_contract(lowered)["attributes"]
        assert lowered.flops == 4 * 2 * 128 * 128 * 64
        assert attrs["block_shape"] == (32, 128) and attrs["active_block_pairs"] == 8


def test_rainfusion_tnd_lse_tracks_query_dtype_and_rank_three_shape():
    q = (128, 2, 128)
    op = replace(
        _make_op("rainfusion", local_shape=q, dtype="fp16"),
        metadata=(("layout", "TND"), ("query_shape", q), ("key_shape", q), ("value_shape", q),
                  ("query_lengths", (128,)), ("key_lengths", (128,)), ("value_lengths", (128,)),
                  ("block_mask", (1,))),
    )
    lowered = sd_lower_op(op, soc_family="A3", sparsity_ratio=0.0, block_size=128)
    assert "softmax_lse" in lowered.returns
    lse = _write(_sd_contract(lowered), "softmax_lse")
    assert (lse["shape"], lse["dtype"], int(lse["bytes"])) == ((128, 2, 128), "fp16", 128 * 2 * 128 * 2)


def test_eager_eagle_ffn_swiglu_emits_separate_silu_and_mul_pointwise_ops():
    m, k, h = 64, 128, 256
    op = replace(
        _make_op("eagle_ffn", local_shape=(m, k), flops=0.0, mte2_bytes=0.0, mte3_bytes=0.0),
        metadata=(("weight1_shape", (2 * h, k)), ("weight2_shape", (k, h)), ("activation", "swiglu")),
    )
    ops = sd_lower_op(op, soc_family="A3")
    pointwise = [item for item in ops if item.family != "gemm"]
    assert len(pointwise) == 2
    assert int(sum(item.hbm_bytes for item in pointwise)) == 5 * m * h * 2
    assert int(sum(item.vector_ops for item in pointwise)) == 6 * m * h
    assert pointwise[1].deps == (ops[0].op_id, pointwise[0].op_id)
