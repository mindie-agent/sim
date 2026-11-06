"""Default body pipelines execute physical work and honor explicit bindings."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import predict
from test_prediction_system import case


@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
def test_gqa_body_default_executes_private_pipes_and_preserves_explicit_envelope(sku):
    raw = case(output=1)
    raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    report = predict(raw, scope="body", tokens=4, context=128, sequences=2)
    attention = [row for row in report["timeline"] if row["family"] == "attention"]
    assert attention and all(row["program"] == "StagedGQADecode" for row in attention)
    assert report["execution"]["serviced"]["mte1@r0:s0"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    explicit = deepcopy(raw)
    explicit["runtime"]["kernels"] = {"recipes": {"flash_attention": {"kind": "envelope"}}}
    prior = predict(explicit, scope="body", tokens=4, context=128, sequences=2)
    assert all(row["program"] == "ResourceEnvelope" for row in prior["timeline"]
               if row["family"] == "attention")
    assert prior["execution"]["work"] == report["execution"]["work"]
    explicit["runtime"]["kernels"]["recipes"] = {
        "flash_attention_decode": {"kind": "staged_gqa_decode", "kv_splits": 2}}
    split = predict(explicit, scope="body", tokens=4, context=128, sequences=2)
    assert split["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved"
    assert split["body_forward_ms"] > 0


@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
def test_indexed_mla_defaults_keep_per_sequence_queries_and_native_mte1_work(sku):
    raw = {
        "machine": {"backend": "ascend", "sku": sku, "num_devices": 8},
        "model": {"preset": "glm-5.3", "quantization": "w8a8", "layers": 1},
        "runtime": {"tp": 8},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    }
    single = predict(raw, scope="body", context=2048)
    batch = predict(raw, scope="body", context=2048, sequences=2)
    a = [row for row in single["timeline"] if row["family"] == "attention"]
    b = [row for row in batch["timeline"] if row["family"] == "attention"]
    assert len(a) == len(b) == 1
    assert b[0]["recipe"] == "latent_attention_batch"
    assert batch["geometry"]["valid_query_tokens"] == 2
    assert batch["execution"]["serviced"]["mte1@r0:s0"] > 0
    assert batch["default_mechanisms"]["selections"][0]["selection"] == "default_installed"
    assert batch["execution"]["live_tasks"] == batch["execution"]["live_programs"] == 0


def test_unavailable_pipe_domain_is_visible_and_keeps_analytic_execution():
    raw = case(output=1)
    report = predict(raw, scope="body", context=64)
    assert report["default_mechanisms"]["selections"][0]["reason"] == "independent_copy_pipes_not_declared"
    assert report["body_forward_ms"] > 0


@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
def test_wide_head_defaults_fit_resident_and_overlapping_tile_buffers(dtype, sku):
    raw = case(output=1)
    raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    raw["model"].update(layers=1, hidden_size=2048, head_dim=512,
                        activation_dtype=dtype, kv_dtype=dtype)
    report = predict(raw, scope="body", tokens=4, context=256, sequences=2)
    rows = [r for r in report["timeline"] if r["family"] == "attention"]
    assert rows and all(r["program"] == "StagedGQADecode" for r in rows)
    assert report["default_mechanisms"]["gqa_tiles"]["key_tile"] < 128
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["body_forward_ms"] > 0
    raw["runtime"]["kernels"] = {"recipes": {"flash_attention": {"kind": "envelope"}}}
    control = predict(raw, scope="body", tokens=4, context=256, sequences=2)
    assert control["execution"]["work"] == report["execution"]["work"]


@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
@pytest.mark.parametrize("kv_dtype", ("bf16", "int8"))
def test_rope_free_indexed_body_defaults_execute_declared_cache_representation(sku, kv_dtype):
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/source-families/packed-int8-indexed-c32.json").read_text())
    raw["model"]["layers"] = 4
    raw["model"]["quant"]["kv_dtype"] = kv_dtype
    raw["machine"]["sku"] = sku
    raw["runtime"]["kernels"]["programs"].pop("indexed_attention")
    result = predict(raw, scope="body", tokens=4, context=128, sequences=3)
    kind = "indexed_attention_packed_int8" if kv_dtype == "int8" else "indexed_attention_pipeline"
    assert result["execution"]["serviced"]["mte2_aiv@r0:s0"] > 0
    selected = next(row for row in result["default_mechanisms"]["selections"]
                    if row["family"] == "indexed_mla")
    assert selected["selection"] == "default_installed"
    from mindie_sim.prediction.body_defaults import attention_defaults
    from mindie_sim.prediction.structure import parse_case
    effective, _ = attention_defaults(parse_case(raw), "decode", tokens=4)
    assert effective.runtime.kernels["programs"]["indexed_attention"]["kind"] == kind
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


@pytest.mark.parametrize("model", ("bailing-v3", "deepseek-v2-lite", "longcat-flash-lite", "kimi-k3"))
@pytest.mark.parametrize("sku", ("Ascend910B2", "Ascend910_9362", "Ascend950DT_9582"))
def test_dense_mla_body_default_respects_kernel_head_domain(model, sku):
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
    from mindie_sim.prediction.structure import parse_case

    raw = compose_case(model, hardware=sku, tp=8, dp=1, num_devices=8)
    raw["model"]["layers"] = 4
    parsed = parse_case(raw)
    report = operator_costs(parsed, target_graphs(
        parsed, phase="decode", tokens=1, context=4096, sequences=1, scope="body"))
    assert report["operator_sum_ms"] > 0


@pytest.mark.parametrize("sku,expected_kda", (
    ("Ascend910_9392", "vllm_ascend_recurrent_kda_arch22"),
    ("Ascend950DT_9582", "vllm_ascend_recurrent_kda_arch35"),
))
def test_kimi_body_defaults_install_kda_and_attention_residual(sku, expected_kda):
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.body_defaults import attention_defaults
    from mindie_sim.prediction.structure import parse_case

    raw = compose_case("kimi-k3", hardware=sku, tp=8, dp=1, ep=8, num_devices=8, quant="w4a8", layers=8)
    effective, report = attention_defaults(parse_case(raw), "decode", tokens=1)
    lowerings = effective.runtime.kernels.get("lowerings", {})
    assert lowerings.get("kda") == expected_kda
    assert lowerings.get("attention_residual") == "vllm_ascend_attn_res_prefix"
    by_fam = {row["family"]: row for row in report["selections"]}
    assert by_fam["kda_decode"]["selection"] == "default_installed"
    assert by_fam["attention_residual"]["selection"] == "default_installed"


@pytest.mark.parametrize("recipe", (
    "residual_values_fp32", "attention_residual_scores", "attention_residual_softmax",
    "attention_residual_mix", "attention_residual_output_cast",
))
def test_attention_residual_default_preserves_explicit_member_programs(recipe):
    from mindie_sim import target_operators
    from mindie_sim.compose import compose_case

    raw = compose_case("kimi-k3", hardware="a3", tp=8, ep=8, layers=4)
    raw["runtime"]["kernels"] = {"recipes": {recipe: {"kind": "envelope"}}}
    ops = target_operators(raw, phase="decode", tokens=1, context=4)
    assert any(op.recipe == recipe for op in ops)
    assert not any(op.recipe == "attn_res_fwd" for op in ops)
    decision, = [row for row in ops.default_selection["selections"]
                 if row["family"] == "attention_residual"]
    assert decision["reason"] == "explicit_implementation_preserved"


@pytest.mark.parametrize("sku", ("Ascend910_9392", "Ascend950DT_9582"))
def test_explicit_kda_scan_family_keeps_the_requested_execution(sku):
    from mindie_sim import operator_costs, target_operators
    from mindie_sim.compose import compose_case

    raw = compose_case("kimi-k3", hardware=sku, tp=8, dp=1, ep=1,
                       num_devices=8, quant="w4a8", layers=1)
    raw["runtime"]["kernels"] = {"families": {"recurrent_scan": {"kind": "envelope"}}}
    ops = target_operators(raw, phase="decode", tokens=1, context=128)
    selected = next(row for row in ops.default_selection["selections"]
                    if row["family"] == "kda_decode")
    assert selected["reason"] == "explicit_implementation_preserved"
    scans = [row for row in operator_costs(raw, ops)["operators"]
             if row["operator"]["family"] == "recurrent_scan"]
    assert scans and all(row["program"] == "ResourceEnvelope" for row in scans)
    assert all(row["duration_ms"] > 0 and not row["live_tasks"] for row in scans)
