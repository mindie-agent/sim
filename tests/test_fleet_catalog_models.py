"""Offline fleet-catalog contract for the seven multinode/workstream models.

[Role]
  Prove that the Gemma-4, MiniMax-M3/M3.1, Qwen3.6/Qwen3.8 and
  Step-5-Preview catalog configs are self-contained, roundtrip through the
  public ModelSpec boundary and lower to exact supported subgraphs.

[Invariants]
  * Catalog JSONs contain declared architecture only: no host paths, IPs,
    remote paths, provenance paths or measured timing keys.
  * Every canonical key and prompt/checkpoint alias resolves offline.
  * Layer plans never use ``projection_only`` or ``unmodeled``; Gemma PLE is
    absent without a ``ple_not_instantiated`` hole.
  * Prefill/decode evaluation and serving produce positive body work.

[Timeouts & Locks]
  Tier 0/1 only: local JSON reads and bounded in-memory arithmetic; no NPU,
  network, service, lock or subprocess work. Each model is evaluated once.

[Side-Effect Policy]
  Read-only tests; HOME and MINDIE_* roots point at ``tmp_path``. Filesystem
  reads under ``/Volumes/data`` raise immediately to prove offline execution.
"""
from __future__ import annotations

import builtins
from dataclasses import replace
import ipaddress
import json
import re
from pathlib import Path

import pytest

try:  # Imported before the offline-root fixture scans installed distributions.
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.dist import Distribution as _Distribution
except ModuleNotFoundError:  # pragma: no cover - minimal offline test envs
    _build_py = _Distribution = None

from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.model_catalog import (
    CATALOG, MODEL_ALIASES, build_model_spec, load_model,
)
from mindie_sim.prediction.gated_activation import gated_activation
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.piecewise.compiler import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import (
    FrameworkConfig, ModelPipeline, predict_framework,
)
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, instantiate_model
from mindie_sim.schema.model import ModelSpec

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "mindie_sim" / "catalog" / "configs"
CANONICAL = (
    "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b", "gemma-4-31b",
    "qwen3.6-35b-a3b", "qwen3.8-flash-next", "step-5-preview",
)
ALIASES = (
    "minimax_m3", "MiniMax-M3", "MiniMax-M3-VL", "MiniMax-M3-MXFP8",
    "minimax-m3-mxfp8", "minimax_m3dot1_mxfp8", "minimax_m31_flash",
    "minimax-m31-flash", "minimax_m3_1_flash", "MiniMax-M3.1-Flash",
    "MiniMax-M3.1-Flash-BF16", "Minimax-M3.1-Flash-w8a8-mxfp8",
    "MiniMax-M3.1-Flash-w8a8-mxfp8",
    "MiniMax-M3.1-Flash-W4A4-MXFP4-CeilX-IterSmooth",
    "minimax-m31-flash-mxfp8", "minimax_m3dot1_flash",
    "minimax_m3dot1_flash_mxfp8", "gemma4_26b_a4b", "gemma4-26b-a4b",
    "Gemma-4-26B-A4B", "gemma-4-26B-A4B-it", "gemma-4-26B-A4B-it-w4a4-mxfp4",
    "gemma4-26b-a4b-mxfp4", "gemma4_31b", "gemma4-31b", "Gemma-4-31B",
    "gemma-4-31B-it", "gemma-4-31B-it-w4a4-mxfp4", "gemma4-31b-mxfp4",
    "qwen36_35b_a3b", "qwen3_6_35b_a3b", "qwen36-35b-a3b",
    "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B-w8a8-mxfp8",
    "Qwen3.6-35B-A3B-w8a8c8-mxfp8", "Qwen3.6-35B-A3B-w4a8-mxfp",
    "qwen36-35b-mxfp8", "qwen36-35b-w8a8-mxfp8", "qwen36-35b-w4a8-mxfp",
    "qwen38_flash_next", "qwen3_8_flash_next", "qwen38-flash-next",
    "Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-w8a8-mxfp8",
    "Qwen3.8-Flash-Next-w4a4-mxfp4", "qwen38-flash-next-mxfp8",
    "qwen38-flash-next-mxfp4", "step5_preview", "step5-preview",
    "Step-5-Preview", "Step-5-Preview-BF16", "Step-5-Preview-w8a8-mxfp8",
    "step5-preview-mxfp8", "Qwen3-30B-A3B-MXFP8", "qwen3-30b-a3b-mxfp8",
    "Qwen3.6-27B-w8a8-mxfp8", "qwen36-27b-mxfp8",
    "Qwen3.8-27B-w8a8-mxfp8", "Qwen3.8-27B-w8a8-mxfp8-mtp",
    "qwen38-27b-bf16", "DeepSeek-V4.1-Flash-1017",
    "GLM-5.1-w4a4c8-mxfp4", "GLM-5.2-w4a4c8-mxfp4",
    "GLM-5.3-W8A8C8-mxfp8", "GLM-5.3-w4a4c8-mxfp4",
    "GLM-5.3-Flash-w8a8-mxfp8", "GLM-5.3-Flash-w4a8-mxfp4",
    "Qwen3.5-397B-A17B-w8a8-mxfp8", "Qwen3.5-397B-A17B-w4a4-mxfp4",
)
SUPPLY = PipeSupply((("cube", 1_000_000.0), ("vector", 1_000_000.0),
    ("hbm", 1_000_000.0), ("mte1", 2_000_000.0), ("mte2", 2_000_000.0),
    ("mte3", 2_000_000.0), ("fixpipe", 2_000_000.0), ("link", 1_000_000.0)))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)
FEATURES = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                split_target=1000, max_splits=4)
_TIMING = ("timing", "latency", "throughput", "measured", "benchmark",
           "duration_ms", "tokens_per_second")
_HOST = ("/Volumes/", "/mnt/", "/Users/")
SPARSE_LAYERS = {
    "minimax-m3": 3, "minimax-m3.1-flash": 0,
    "qwen3.8-flash-next": 3, "step-5-preview": 3,
}


def _forbid(path: object) -> None:
    text = str(path)
    if text.startswith(("/Volumes/data", "/private/Volumes/data")):
        raise AssertionError(f"offline_catalog_read_attempted:{text}")


@pytest.fixture(autouse=True)
def _isolated_offline_roots(monkeypatch, tmp_path):
    for name in ("MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME",
                 "MINDIE_DIAGNOSTICS_ROOT", "HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    real_read_text, real_open, real_path_open = Path.read_text, builtins.open, Path.open

    def guarded_read_text(self, *args, **kwargs):
        _forbid(self)
        return real_read_text(self, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        _forbid(file)
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        _forbid(self)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_open)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_catalog_configs_are_clean_source_faithful_architecture():
    assert len(list(CONFIG_DIR.glob("*.json"))) == len(CANONICAL)
    for key in CANONICAL:
        path = CONFIG_DIR / f"{key}.json"
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        assert not any(token in text for token in _HOST), key
        assert "_name_or_path" not in text and "remote_path" not in text, key
        for token in re.findall(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text):
            try:
                ip = ipaddress.ip_address(token)
            except ValueError:
                continue
            raise AssertionError(f"ip_address_in_catalog_config:{key}:{ip}")
        lowered = {name.lower() for name in _walk_keys(config)}
        assert not any(token in name for name in lowered for token in _TIMING), key
        assert config.get("text_config", {}).get("num_hidden_layers"), key


def test_canonical_keys_and_aliases_roundtrip_offline():
    for key in CANONICAL:
        assert key in CATALOG
        model = load_model(key)
        assert model.name == key
        restored = ModelSpec.from_mapping(json.loads(json.dumps(model.to_dict())))
        assert restored == model
    for alias in ALIASES:
        target = MODEL_ALIASES[alias]
        if target in CANONICAL:
            model = load_model(alias)
            assert model.name == target
            assert ModelSpec.from_mapping(json.loads(json.dumps(model.to_dict()))) == model
        else:
            assert load_model(alias).name == target


def test_fleet_layer_plans_use_supported_attention_cores():
    for key in CANONICAL:
        model = load_model(key)
        assert model.layers == model.body_layers == len(model.layer_plans)
        assert all(plan.attention in {"gqa", "gdn", "mla"} for plan in model.layer_plans)
        assert not any(plan.attention in {"projection_only", "unmodeled"}
                       for plan in model.layer_plans)

    gemma26 = load_model("gemma-4-26b-a4b")
    local26 = [p for p in gemma26.layer_plans if p.window]
    global26 = [p for p in gemma26.layer_plans if not p.window]
    assert len(gemma26.layer_plans) == 30
    assert [p.index for p in global26] == [5, 11, 17, 23, 29]
    assert all(p.architecture == "gemma4" and p.qk_norm for p in gemma26.layer_plans)
    assert all((p.num_kv_heads, p.head_dim, p.kv_cache_elements) == (8, 256, 2 * 8 * 256)
               for p in local26)
    assert all((p.num_kv_heads, p.head_dim, p.kv_cache_elements) == (2, 512, 2 * 512)
               for p in global26)
    assert all((p.ffn, p.num_experts, p.top_k, p.moe_intermediate_size,
                p.num_shared_experts, p.shared_intermediate_size, p.ffn_activation)
               == ("moe", 128, 8, 704, 1, 2112, "gelu") for p in gemma26.layer_plans)
    gemma31 = load_model("gemma-4-31b")
    assert len(gemma31.layer_plans) == 60
    assert len([p for p in gemma31.layer_plans if p.window]) == 50
    assert all((p.ffn, p.intermediate_size, p.ffn_activation) == ("dense", 21504, "gelu")
               for p in gemma31.layer_plans)
    for gemma in (gemma26, gemma31):
        facts = dict(gemma.catalog_facts)
        assert facts["gemma4"] == 1 and facts["gemma4_ple"] == 0
        assert facts["gemma4_ple_layers"] == ()
        assert facts["attention_schedule"].count("global") == len(
            [p for p in gemma.layer_plans if not p.window])

    for key in ("minimax-m3", "minimax-m3.1-flash"):
        model = load_model(key)
        plans = model.layer_plans
        assert len(plans) == 60
        assert all((p.num_heads, p.num_kv_heads, p.head_dim) == (64, 4, 128) for p in plans)
        assert [p.ffn for p in plans[:3]] == ["dense"] * 3
        assert all(p.intermediate_size == 12288 for p in plans[:3])
        assert all((p.num_experts, p.top_k, p.moe_intermediate_size,
                    p.num_shared_experts, p.shared_intermediate_size, p.router_mode)
                   == (128, 4, 3072, 1, 3072, "sigmoid") for p in plans[3:])
        active = [p.index for p in plans if p.indexer]
        assert active == (list(range(3, 60)) if key == "minimax-m3" else list(range(60)))
        assert all((p.indexer_type, p.index_heads, p.index_dim, p.index_topk,
                    p.candidate_block_size, p.candidate_topk_blocks)
                   == ("full", 4, 128, 2176, 128, 17) for p in plans if p.indexer)
        metadata = dict(model.metadata)
        assert metadata["model_coverage"] == "partial"
        assert metadata["unmodeled_features"]

    qwen36 = load_model("qwen3.6-35b-a3b")
    assert len(qwen36.layer_plans) == 40
    assert len([p for p in qwen36.layer_plans if p.attention == "gdn"]) == 30
    assert len([p for p in qwen36.layer_plans if p.attention == "gqa"]) == 10
    assert (qwen36.num_experts, qwen36.top_k) == (256, 8)

    qwen38 = load_model("qwen3.8-flash-next")
    plans38 = qwen38.layer_plans
    assert len(plans38) == 48
    assert len([p for p in plans38 if p.attention == "gdn"]) == 36
    assert len([p for p in plans38 if p.attention == "gqa"]) == 12
    assert (qwen38.hidden_size, qwen38.num_experts, qwen38.top_k) == (2560, 512, 10)
    assert all(p.hyper_connections and p.hc_mult == 4 for p in plans38)
    assert all(p.linear_value_heads == 48 for p in plans38 if p.attention == "gdn")
    assert all(p.q_output_gate for p in plans38)
    indexers = [p for p in plans38 if p.indexer]
    assert [p.index for p in indexers] == list(range(3, 48, 4))
    assert all((p.indexer_type, p.index_heads, p.index_dim, p.index_topk,
                p.index_kpool, p.index_kpool_compress)
               == ("full", 4, 128, 2048, 4, True) for p in indexers)
    assert qwen38.engram is not None
    assert [table.layer_id for table in qwen38.engram.tables] == [2]
    assert (qwen38.engram.tables[0].rows, qwen38.engram.tables[0].width) == (20_000_000, 160)
    assert plans38[2].engram

    step = load_model("step-5-preview")
    plans = step.layer_plans
    assert len(plans) == 92
    assert len([p for p in plans if p.window == 512]) == 69
    assert len([p for p in plans if p.window == 0]) == 23
    assert [p.index for p in plans if p.ffn == "dense"] == [0, 1, 2, 91]
    assert all(p.intermediate_size == 13824 for p in plans if p.ffn == "dense")
    assert all((p.num_experts, p.top_k, p.moe_intermediate_size,
                p.num_shared_experts, p.shared_intermediate_size)
               == (352, 8, 1536, 1, 1536) for p in plans if p.ffn == "moe")
    assert all((p.num_heads, p.num_kv_heads, p.head_dim, p.q_output_gate)
               == (64, 4, 192, True) for p in plans)


@pytest.mark.parametrize("key", CANONICAL)
def test_fleet_models_instantiate_compile_prefill_decode_and_serve(key):
    model = load_model(key)
    features = FeatureConfig.from_model(model, **FEATURES)
    instantiated = instantiate_model(model, features)
    assert not [hole for hole in instantiated.coverage.holes
                if hole.reason == "ple_not_instantiated"]
    pipeline = ModelPipeline(model, features, SUPPLY, POLICY)
    assert isinstance(pipeline.compiler, OperatorPiecewiseCompiler)
    compiled = pipeline.compiler.compile(instantiated.layers[0].subgraphs[0])
    assert compiled.evaluate(WorkloadGeometry.homogeneous(2, 3, 4)).total_ms > 0
    prefill = pipeline.forward(WorkloadGeometry.homogeneous(2, 3, 4))
    decode = pipeline.forward(WorkloadGeometry.homogeneous(1, 5, 8))
    assert prefill.total_ms > 0 and decode.total_ms > 0
    result = predict_framework(
        pipeline, [dict(id="fleet", prompt_tokens=3, output_tokens=2)],
        FrameworkConfig(supply=SUPPLY, policy=POLICY, max_batch=1))
    assert result.steps and result.steps[0]["body_ms"] > 0


def _manifest_files(root: Path) -> set[str]:
    """Expand the local ``MANIFEST.in`` include/recursive-include directives."""
    includes: list[str] = []
    recursive: list[tuple[str, tuple[str, ...]]] = []
    for line in (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "include" and len(fields) == 2:
            includes.append(fields[1])
        elif fields[0] == "recursive-include" and len(fields) >= 3:
            recursive.append((fields[1], tuple(fields[2:])))
    found = {name for name in includes if (root / name).is_file()}
    for directory, patterns in recursive:
        for pattern in patterns:
            found.update(path.relative_to(root).as_posix()
                         for path in (root / directory).rglob(pattern) if path.is_file())
    return found


def _catalog_config_names() -> set[str]:
    return {f"mindie_sim/catalog/configs/{path.name}" for path in CONFIG_DIR.glob("*.json")}


def test_catalog_configs_are_declared_package_data():
    import tomllib

    configs = _catalog_config_names()
    assert len(configs) == len(CANONICAL) == 7
    assert configs <= _manifest_files(ROOT)
    patterns = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["package-data"]["mindie_sim"]
    assert "catalog/configs/*.json" in patterns
    package_root = ROOT / "mindie_sim"
    declared = {f"mindie_sim/{path.relative_to(package_root).as_posix()}"
                for pattern in patterns
                for path in package_root.glob(pattern) if path.is_file()}
    assert configs <= declared


def test_setuptools_build_py_discovers_catalog_configs():
    if _build_py is None or _Distribution is None:
        pytest.skip("setuptools unavailable")
    import tomllib

    package_data = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["package-data"]
    distribution = _Distribution()
    distribution.packages = ["mindie_sim"]
    distribution.package_data = package_data
    command = _build_py(distribution)
    command.finalize_options()
    rows = command.get_data_files_without_manifest()
    discovered = {f"mindie_sim/{name}" for _, _, _, names in rows for name in names}
    assert _catalog_config_names() <= discovered


def _fleet_case(key: str, *, tp: int = 1) -> dict:
    model = load_model(key)
    return {
        "machine": {
            "name": "declared-fleet", "architecture": "cuda", "num_devices": tp,
            "device": {"memory_bytes": 1024 ** 4,
                       "compute_peaks": {"bf16": 1e12, "fp32": 1e12},
                       "rates": {"memory": 1e10, "vector": 1e10}, "compute_units": 64},
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1e6,
                       "members": list(range(tp)), "semantics": "shared"}],
        },
        "model": model.to_dict(),
        "workload": {"requests": [{"id": "probe", "prompt_tokens": 8,
                                   "output_tokens": 1}]},
        "runtime": {"tp": tp, "dp": 1, "pp": 1, "ep": 1,
                    "max_num_batched_tokens": 64, "max_num_seqs": 8,
                    "max_model_len": 131072, "enable_chunked_prefill": True,
                    "gpu_memory_utilization": 0.9, "graph_mode": "auto",
                    "graph_buckets": [1, 2, 4, 8], "kv_block_size": 128},
    }


def test_gated_gelu_activation_uses_tanh_approximated_geglu_work():
    plan = load_model("gemma-4-31b").layer_plans[0]
    for activation in ("gelu", "geglu", "gelu_tanh", "gelu_pytorch_tanh"):
        op = gated_activation(
            plan=replace(plan, ffn_activation=activation), op_id="probe",
            deps=("input",), rows=2, width=4, dtype="bf16", stage=0,
            requests=(), phase="decode", recipe="swiglu")
        assert op.family == op.recipe == "geglu"
        assert op.vector_ops == 8 * 2 * 4
        assert op.mte2_bytes == 2 * 8 * 2 and op.mte3_bytes == 8 * 2


def test_fleet_models_lower_gated_activations_in_ordinary_graphs():
    for key in CANONICAL:
        case = _fleet_case(key)
        for phase in ("prefill", "decode"):
            operators = target_operators(case, phase=phase, tokens=1)
            assert operators, (key, phase)
            graphs = target_graphs(case, phase=phase, tokens=1)
            assert graphs and all(items for items in graphs.values()), (key, phase)


def test_sparse_attention_orders_indexer_before_attention_and_bounds_kv():
    for key, layer_index in SPARSE_LAYERS.items():
        model = load_model(key)
        instantiated = instantiate_model(
            model, FeatureConfig.from_model(model, **FEATURES))
        layer = instantiated.layers[layer_index]
        kinds = [subgraph.kind for subgraph in layer.subgraphs]
        assert kinds.index("I0") < kinds.index("A0"), key
        assert kinds.index("I1") < kinds.index("A0"), key
        a0 = layer.find("A0", role="attention")
        assert int(a0.operator("attention").theta.find("index_topk")) > 0, key
        i1 = layer.find("I1", role="index_select")
        compiler = OperatorPiecewiseCompiler(SUPPLY, POLICY)

        def cost_at(instance, name, context):
            evaluation = compiler.compile(instance).evaluate(
                WorkloadGeometry.homogeneous(1, 1, context))
            return next(cost for cost in evaluation.operators if cost.name == name)

        high_kv = cost_at(a0, "attention", 32768)
        low_kv = cost_at(a0, "attention", 16384)
        assert high_kv.get("KV_ROWS") == low_kv.get("KV_ROWS") > 0, key
        assert (high_kv.get("KV_LOGICAL_BYTES") == low_kv.get("KV_LOGICAL_BYTES") > 0), key
        assert (cost_at(i1, "topk", 32768).get("index_rows")
                > cost_at(i1, "topk", 16384).get("index_rows")), key


def test_sparse_target_graphs_bound_attention_traffic_by_index_topk():
    names = ("index_wq", "index_wk", "index_store", "index_scores", "index_topk")
    for key, layer_index in SPARSE_LAYERS.items():
        case = _fleet_case(key)
        body = {context: [op for op in target_operators(
                    case, phase="decode", tokens=1, context=context)
                if op.layer == layer_index]
                for context in (16384, 32768)}
        graph = {context: [op for op in target_graphs(
                    case, phase="decode", tokens=1, context=context)[0]
                if op.layer == layer_index]
                for context in (16384, 32768)}
        for ops in (body, graph):
            high, low = ops[32768], ops[16384]
            attention = next(op for op in high if op.op_id.endswith(".attention"))
            prefix = attention.op_id.rsplit(".", 1)[0]
            order = [op.op_id for op in high]
            for name in names:
                assert f"{prefix}.{name}" in order, (key, name)
                assert order.index(f"{prefix}.{name}") < order.index(attention.op_id), key
            low_attention = next(op for op in low if op.op_id.endswith(".attention"))
            assert attention.mte2_bytes == low_attention.mte2_bytes > 0, key
            assert attention.hbm_bytes == low_attention.hbm_bytes > 0, key


def test_gemma4_k_eq_v_omits_value_projection_and_halves_cache_plane():
    model = load_model("gemma-4-26b-a4b")
    global_plan, local_plan = model.layer_plans[5], model.layer_plans[0]
    assert global_plan.k_eq_v and not local_plan.k_eq_v
    terms = {term.name: term for term in layer_weight_terms(global_plan, model)}
    assert terms["q_proj"].rows == 16 * 512
    assert terms["k_proj"].rows == 2 * 512
    assert "v_proj" not in terms
    local_terms = {term.name: term for term in layer_weight_terms(local_plan, model)}
    assert local_terms["q_proj"].rows == 16 * 256
    assert local_terms["k_proj"].rows == local_terms["v_proj"].rows == 8 * 256

    case = _fleet_case("gemma-4-26b-a4b")

    def qkv_width(ops, layer):
        widths = {op.local_shape[1] for op in ops
                  if op.layer == layer and op.recipe == "qkv_projection"}
        assert len(widths) == 1
        return widths.pop()

    operators = target_operators(case, phase="decode", tokens=1)
    graphs = target_graphs(case, phase="prefill", tokens=1)[0]
    assert qkv_width(operators, 5) == 16 * 512 + 2 * 512 == 9216
    assert qkv_width(graphs, 5) == 9216
    assert qkv_width(operators, 0) == 16 * 256 + 2 * 8 * 256 == 8192
    assert qkv_width(graphs, 0) == 8192
    assert global_plan.cache_row_elements == 1 * 2 * 512


def test_gemma4_inline_config_without_intermediate_size_builds():
    config = json.loads((CONFIG_DIR / "gemma-4-26b-a4b.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    text.pop("intermediate_size", None)
    text["moe_intermediate_size"] = 704
    text["shared_expert_intermediate_size"] = 2112
    model = build_model_spec("gemma-4-26b-a4b-inline", config)
    assert model.intermediate_size == 2112
    assert model.moe_intermediate_size == 704
    assert all(plan.k_eq_v for plan in model.layer_plans if not plan.window)
    assert not any(plan.k_eq_v for plan in model.layer_plans if plan.window)


def test_gemma4_routed_expert_count_without_moe_flag_stays_moe():
    # Legacy HF-shaped Gemma4 callers declare only n_routed_experts.
    from mindie_sim.catalog.specs import build_model_spec
    cfg = {"model_type": "gemma4", "num_hidden_layers": 6, "hidden_size": 2816,
           "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
           "vocab_size": 262144, "moe_intermediate_size": 704,
           "shared_expert_intermediate_size": 2112, "n_routed_experts": 128,
           "num_experts_per_tok": 8, "n_shared_experts": 1,
           "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}
    model = build_model_spec("gemma4-legacy", cfg)
    assert all(plan.ffn == "moe" and plan.num_experts == 128 and plan.top_k == 8
               for plan in model.layer_plans)
    dense = build_model_spec("gemma4-dense", {**cfg, "enable_moe_block": False})
    assert all(plan.ffn == "dense" for plan in dense.layer_plans)


def test_step5_routed_expert_count_without_schedule_is_all_moe():
    from mindie_sim.catalog.specs import build_model_spec
    cfg = {"model_type": "step5", "num_hidden_layers": 4, "hidden_size": 4096,
           "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
           "vocab_size": 151936, "moe_intermediate_size": 1280,
           "shared_expert_intermediate_size": 1280, "n_routed_experts": 256,
           "num_experts_per_tok": 8, "n_shared_experts": 1,
           "layer_types": ["full_attention"] * 4}
    model = build_model_spec("step5-legacy", cfg)
    assert all(plan.ffn == "moe" and plan.num_experts == 256 and plan.top_k == 8
               for plan in model.layer_plans)
