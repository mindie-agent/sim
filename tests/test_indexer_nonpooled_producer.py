"""Non-pooled indexer query producer follows the declared low-rank input."""
from dataclasses import replace
from pathlib import Path
import json

from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.builders_attention import _build_i0
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, _Ctx
from mindie_sim.prediction.piecewise.compiler import OperatorPiecewiseCompiler, PipeSupply
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.schema.subgraphs import BoundOperator

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/high-concurrency-20261030/glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json"


def _ctx(q_lora_rank, *, rope=64, owns_index_k=True, index_input="hidden",
         head_dim=64, cache=None, owner_layer=-1):
    raw = json.loads(CASE.read_text(encoding="utf-8"))
    case = parse_case(raw)
    holder = case.model.layer_plans[3]
    layer = replace(holder, index=0, indexer=True, indexer_type="full",
                    index_kpool=1, index_kpool_compress=False,
                    index_heads=32, index_dim=128, index_topk=2048,
                    qk_rope_head_dim=rope, q_lora_rank=q_lora_rank,
                    owns_index_k=owns_index_k, index_input=index_input,
                    head_dim=head_dim, index_k_owner_layer_id=owner_layer)
    model = replace(case.model, layer_plans=tuple([layer]))
    features = FeatureConfig.from_runtime(case.runtime, model)
    if cache is not None:
        features = replace(features, metadata=tuple({**dict(features.metadata or ()),
                                                "indexer_cache": cache}.items()))
    return case, _Ctx(model, layer, features, stage=0, layers=1)


def _index_q(case, ctx):
    built = _build_i0(ctx)
    theta = built.signature("index_q_projection")
    return theta, built


def test_nonpooled_index_q_uses_declared_q_lora_rank():
    case, ctx = _ctx(2048)
    theta, built = _index_q(case, ctx)
    assert theta.require("K") == 2048
    assert [op.name for op in built.template.operators].count("index_weights_projection") == 1
    assert built.signature("index_k_projection").require("N") == 128
    assert built.signature("index_weights_projection").require("N") == 32
    assert theta.require("K") != ctx.hidden
    assert theta.find("query_k_source") == "q_lora_rank"
    width = {"fp8_e4m3": 1.0, "int8": 1.0, "bf16": 2.0}.get(
        theta.find("weight_format"), 1.0)
    assert theta.find("weight_payload") == theta.require("N") * 2048 * width


def test_nonpooled_index_q_falls_back_to_hidden_without_lora():
    case, ctx = _ctx(0)
    theta, _ = _index_q(case, ctx)
    assert theta.require("K") == ctx.hidden
    assert theta.find("query_k_source") == "hidden_rank_absent"


def test_q_lora_equal_to_hidden_retains_rank_source():
    _, ctx = _ctx(5120)
    theta, _ = _index_q(*_ctx(5120))
    assert theta.require("K") == 5120
    assert theta.find("query_k_source") == "q_lora_rank"


def test_nonowner_emits_weights_but_not_index_k():
    case, ctx = _ctx(2048, owns_index_k=False)
    built = _build_i0(ctx)
    names = [op.name for op in built.template.operators]
    assert "index_k_projection" not in names
    assert "index_weights_projection" in names


def test_compressed_owner_index_k_uses_latent_head_dim():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed", head_dim=512)
    built = _build_i0(ctx)
    k = built.signature("index_k_projection")
    assert k.require("K") == 512
    assert k.require("K") != ctx.hidden
    assert k.find("index_k_source") == "compressed_latent"


def test_owned_cache_store_uses_declared_mxfp4_row_and_owner():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed",
                     head_dim=512, cache={"row_bytes": 68, "scale_bytes": 4,
                                          "rows": 3, "format": "mxfp4_e2m1"})
    built = _build_i0(ctx)
    store = built.signature("index_cache_store")
    assert store.require("rows") == 3
    assert store.require("row_bytes") == 68
    assert store.find("scale_bytes") == 4
    assert store.find("cache_format") == "mxfp4_e2m1"
    assert store.find("owner_layer") == 0


def test_shared_nonowner_has_no_cache_store():
    case, ctx = _ctx(2048, owns_index_k=False, index_input="compressed",
                     head_dim=512, owner_layer=0)
    built = _build_i0(ctx)
    names = [op.name for op in built.template.operators]
    assert "index_cache_store" not in names


def test_owned_cache_store_zero_valid_rows_is_explicit():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed",
                     head_dim=512, cache={"row_bytes": 68, "scale_bytes": 4,
                                          "rows": 0, "format": "mxfp4_e2m1"})
    built = _build_i0(ctx)
    store = built.signature("index_cache_store")
    assert store.require("rows") == 0
    assert store.require("row_bytes") == 68


def test_owner_layer_zero_is_not_conflated_with_absent():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed",
                     head_dim=512, cache={"row_bytes": 68, "scale_bytes": 4,
                                          "rows": 1}, owner_layer=0)
    built = _build_i0(ctx)
    assert built.signature("index_cache_store").find("owner_layer") == 0


def test_compiled_cache_store_uses_explicit_valid_slots():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed",
                     head_dim=512, cache={"row_bytes": 68, "scale_bytes": 4,
                                          "rows": 3})
    built = _build_i0(ctx)
    template = next(op for op in built.template.operators
                    if op.name == "index_cache_store")
    operator = BoundOperator(template, built.signature("index_cache_store"))
    supply = PipeSupply.from_hardware(case.machine, dtype="int8")
    compiled = OperatorPiecewiseCompiler(supply).compile_operator(operator, kind="I0")
    geometry = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=1, local_batch=1)
    cost = compiled.evaluate(geometry)
    assert cost.pipe_work("hbm") == 3 * 68
    assert cost.pipe_work("mte3") == 3 * 68
    derived = dict(cost.derived)
    assert derived["valid_rows"] == 3
    assert derived["data_bytes"] == 3 * 64
    assert derived["scale_bytes"] == 3 * 4


def test_glm_full_owner_cache_store_is_dynamic_and_positive():
    case, ctx = _ctx(1536, owns_index_k=True, index_input="hidden",
                     head_dim=256, cache=None)
    built = _build_i0(ctx)
    template = next(op for op in built.template.operators
                    if op.name == "index_cache_store")
    operator = BoundOperator(template, built.signature("index_cache_store"))
    supply = PipeSupply.from_hardware(case.machine, dtype="int8")
    compiled = OperatorPiecewiseCompiler(supply).compile_operator(operator, kind="I0")
    cost = compiled.evaluate(WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=1, local_batch=1))
    assert cost.pipe_work("hbm") == 1024 * int(built.signature("index_cache_store").require("row_bytes"))
    assert dict(cost.derived)["valid_rows"] == 1024


def test_packed_cache_without_valid_count_stays_unresolved():
    case, ctx = _ctx(2048, owns_index_k=True, index_input="compressed",
                     head_dim=512, cache={"row_bytes": 68, "scale_bytes": 4})
    built = _build_i0(ctx)
    store = built.signature("index_cache_store")
    assert store.require("rows") == 0
    assert store.find("valid_rows_mode") == "unresolved"
    operator = BoundOperator(
        next(op for op in built.template.operators
             if op.name == "index_cache_store"), store)
    supply = PipeSupply.from_hardware(case.machine, dtype="int8")
    compiled = OperatorPiecewiseCompiler(supply).compile_operator(operator, kind="I0")
    cost = compiled.evaluate(WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=1, local_batch=1))
    assert cost.pipe_work("hbm") == 0
    assert dict(cost.derived)["valid_rows"] == 0
