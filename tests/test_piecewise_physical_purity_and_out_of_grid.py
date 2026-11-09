"""Adversarial AST physical purity, time conservation, and out-of-grid generalization suite.

[Role]
  Enforce zero empirical curve-fitting in ``mindie_sim/prediction/piecewise/``
  through static AST inspection, per-layer/subgraph/body time conservation checks,
  operator metadata invariance, and out-of-grid monotonicity/continuity tests.

[Invariants]
  * Zero ``math.cos`` / ``math.sin`` harmonic series calls or imports in ``piecewise/``.
  * Zero model-family fingerprint identifiers or stealth modulo/layer/bucket switches
    anywhere in ``mindie_sim/prediction/piecewise/``.
  * Zero hardcoded sequence-length equality or set-membership branches
    (``q_len == 2048``, ``q_len == 4096``, ``prompt_tokens == ...``,
    ``prompt_tokens in {...}``, ``q_len in {...}``).
  * ``lower_layer_predictions`` strictly conserves total per-layer operator time,
    subgraph attribution sums, and both active and inactive body durations.
  * Compiled operators and ``ModelPipeline.forward`` vary monotonically across
    interleaved grid and off-grid prompt lengths (``768..49152``) and off-grid
    concurrencies (``3, 5, 6, 7, 10, 12, 20, 24``) with no step discontinuities
    at ``1024, 2048, 4096, 8192, 16384``.

[Timeouts & Locks]
  Tier 0 (<5s) pure in-memory AST and arithmetic checks; no I/O or locks.

[Side-Effect Policy]
  Read-only source inspection and in-memory model evaluation; mutates no files.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.compiler import (
    CompiledOperator,
    ImplementationPolicy,
    PipeSupply,
)
from mindie_sim.prediction.piecewise.geometry import AnalysisDomain, RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs import FeatureConfig
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature

PIECEWISE_DIR = Path(__file__).resolve().parents[1] / "mindie_sim" / "prediction" / "piecewise"
_DOMAIN = AnalysisDomain(
    max_tokens=1_048_576, max_q=1_048_576, max_s_kv=1_048_576, max_seq=1_048_576,
    max_moe_rows=8_388_608, max_query_tiles=262_144, max_kv_tiles=262_144,
    max_index_rows=10**10, max_work=10**15,
)
INTERLEAVED_TOKENS: tuple[int, ...] = (
    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
)
INTERLEAVED_CONCURRENCIES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)
GRID_BOUNDARY_POINTS: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)

_FORBIDDEN_TRIG_CALLS = frozenset({"cos", "sin", "acos", "asin", "atan", "atan2", "exp"})
_FORBIDDEN_IDENT_SUBSTRINGS = (
    "is_glm", "is_dsv41", "is_qwen", "is_qwen38", "is_k3", "is_v41_f3",
    "is_k3_f3", "is_glm53", "is_w4_glm", "is_ep1", "_glm_", "_dsv41_",
    "_qwen_", "_k3_", "_harm1", "_harm2", "glm_ov", "k3_map",
    "is_mxfp4_tp8", "is_fp48_f3", "is_fp8_or_nobucket", "is_mxfp4_low_tp",
)
_FORBIDDEN_IDENT_RE = re.compile(
    r"(?:^|_)(?:glm53|glm51|glm|dsv41|dsv|deepseek|v41_f3|qwen38|qwen36|qwen|kimi|k3_f3|k3|llama|harm1|harm2)(?:_|$)",
    re.IGNORECASE,
)
_SEQ_VARS = frozenset({"q_len", "q_len_g", "prompt_tokens", "kv_len", "seq_len", "context", "tokens"})
_GRID_LITERALS = frozenset({512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072})

_LOADED: dict[str, Any] = {}
_OWNERS = (
    ("kv_owner_layer_id", "owns_long_kv"),
    ("index_k_owner_layer_id", "owns_index_k"),
    ("topk_owner_layer_id", "owns_topk"),
)
_SOURCES = ("candidate_source_layer_id", "index_source_layer_id")
_CAPS = {
    "q_lora_rank": 128, "kv_lora_rank": 64, "index_heads": 8,
    "index_dim": 32, "expert_latent_size": 256,
}
_FAMILIES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("deepseek-v4.1-flash", (2, 3)),
    ("glm-5.3", (2, 3)),
    ("kimi-k3", (1, 3)),
    ("qwen3.8-27b", (2, 3)),
)


def _supply(sku: str = "Ascend950PR_9579", **hw: Any) -> PipeSupply:
    raw = {"backend": "ascend", "sku": sku, "num_devices": 16, **hw}
    return PipeSupply.from_hardware(Hardware.from_mapping(raw))


def _compact_model(key: str, indices: tuple[int, int], *, hidden: int = 256, intermediate: int = 128, experts: int = 16):
    cache_key = f"{key}:{indices}:{hidden}:{intermediate}:{experts}"
    if (cached := _LOADED.get(cache_key)) is not None:
        return cached
    source = _LOADED.get(key) or _LOADED.setdefault(key, load_model(key))
    mapped = {old: new for new, old in enumerate(indices)}
    plans = []
    for new, old in enumerate(indices):
        plan = source.layer_plans[old]
        updates: dict[str, Any] = {"index": new, "native_layer_id": new}
        for field, flag in _OWNERS:
            owner = int(getattr(plan, field))
            if owner in mapped:
                updates[field], updates[flag] = mapped[owner], mapped[owner] == new
            elif owner >= 0:
                updates[field], updates[flag] = new, True
        for field in _SOURCES:
            owner = int(getattr(plan, field))
            if owner >= 0:
                updates[field] = mapped.get(owner, new)
        for field, cap in _CAPS.items():
            if int(getattr(plan, field) or 0) > cap:
                updates[field] = cap
        if plan.ffn == "moe":
            updates.update(moe_intermediate_size=intermediate, num_experts=experts, top_k=min(int(plan.top_k or 1), experts, 2))
            if plan.shared_intermediate_size:
                updates["shared_intermediate_size"] = intermediate
        elif plan.intermediate_size:
            updates["intermediate_size"] = intermediate
        plans.append(replace(plan, **updates))
    moe = any(plan.ffn == "moe" for plan in plans)
    compact = replace(
        source, name=f"{key}-compact", hidden_size=hidden, intermediate_size=intermediate,
        num_experts=experts if moe else 0, top_k=min(int(source.top_k or 1), experts, 2) if moe else 0,
        moe_intermediate_size=intermediate if moe else 0,
        layers=len(plans), body_layers=len(plans), vocab_size=256, layer_plans=tuple(plans),
    )
    return _LOADED.setdefault(cache_key, compact)


def _extract_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _extract_int_literals(node: ast.AST) -> set[int]:
    return {
        int(sub.value)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, int) and not isinstance(sub.value, bool)
    }


def test_piecewise_ast_has_no_empirical_harmonics_or_model_fingerprints() -> None:
    py_files = sorted(PIECEWISE_DIR.glob("*.py"))
    assert py_files, f"No Python files found in {PIECEWISE_DIR}"
    violations: list[str] = []

    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trig_aliases: set[str] = set(_FORBIDDEN_TRIG_CALLS)
        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TRIG_CALLS:
                        trig_aliases.add(alias.asname or alias.name)
                        violations.append(f"{path.name}:L{lineno}: forbidden trig import '{alias.name}'")
            if isinstance(node, ast.Call):
                func_name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else "")
                if func_name in trig_aliases:
                    violations.append(f"{path.name}:L{lineno}: forbidden trigonometric call '{func_name}'")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Name, ast.arg)):
                ident = getattr(node, "name", None) or getattr(node, "id", None) or getattr(node, "arg", None) or ""
                ident_low = ident.lower()
                if any(sub in ident_low for sub in _FORBIDDEN_IDENT_SUBSTRINGS) or _FORBIDDEN_IDENT_RE.search(ident_low):
                    violations.append(f"{path.name}:L{lineno}: forbidden model/harmonic identifier '{ident}'")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                if (_extract_names(node.left) & {"residency_bytes", "f3_res"}) and _extract_int_literals(node.right):
                    violations.append(f"{path.name}:L{lineno}: stealth modulo fingerprint '{ast.unparse(node)}'")
            if isinstance(node, ast.Compare):
                comparands = [node.left, *node.comparators]
                for left, op, right in zip(comparands, node.ops, comparands[1:]):
                    names = _extract_names(left) | _extract_names(right)
                    ints = _extract_int_literals(left) | _extract_int_literals(right)
                    if isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) and (names & _SEQ_VARS) and (ints & _GRID_LITERALS):
                        violations.append(f"{path.name}:L{lineno}: hardcoded seq branch '{ast.unparse(node)}'")
                    if (names & {"max_b", "max_bucket"}) and (64 in ints):
                        violations.append(f"{path.name}:L{lineno}: unused max_bucket switch '{ast.unparse(node)}'")

    assert not violations, "Empirical curve-fitting or model-specific branches detected:\n" + "\n".join(violations)


@dataclass(frozen=True)
class _SyntheticCost:
    work: tuple[tuple[str, float], ...]
    times: tuple[tuple[str, float], ...]
    residency_bytes: int = 0
    derived: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class _SyntheticRow:
    kind: str
    name: str
    mechanism: str
    count: int
    total_ms: float
    cost: _SyntheticCost


def test_lower_layer_predictions_conserves_layer_and_body_time() -> None:
    from mindie_sim.prediction.piecewise.lowering import lower_layer_predictions

    def _make_rows(residency: int) -> tuple[_SyntheticRow, ...]:
        cost = _SyntheticCost((("cube", 1.0e9), ("vector", 1.0e8), ("hbm", 1.0e8)), (("cube", 1.0), ("vector", 0.5), ("hbm", 0.8)), residency, (("kv_bf16", 1.0),))
        return (
            _SyntheticRow("N0", "attn_norm", "norm_residual", 60, 0.24, cost),
            _SyntheticRow("A1", "q_down", "gemm", 60, 0.90, cost),
            _SyntheticRow("A1", "q_up", "gemm", 60, 1.20, cost),
            _SyntheticRow("A1", "kv_down", "gemm", 60, 0.60, cost),
            _SyntheticRow("A1", "mla_core", "attention", 60, 4.80, cost),
            _SyntheticRow("A1", "rope_apply", "rope", 60, 0.18, cost),
            _SyntheticRow("K0", "kv_update", "kv_cache", 60, 0.06, cost),
            _SyntheticRow("I0", "index_proj", "indexer", 60, 1.50, cost),
            _SyntheticRow("A3", "gdn_scan", "gdn_kda", 384, 1.80, cost),
            _SyntheticRow("R0", "mhc_mix", "hyperconnection", 60, 0.90, cost),
            _SyntheticRow("R1", "attn_resid", "norm_residual", 60, 0.12, cost),
            _SyntheticRow("F1", "router_topk", "moe_router", 60, 0.54, cost),
            _SyntheticRow("F2", "dispatch_quant", "elementwise_quant", 60, 0.72, cost),
            _SyntheticRow("F2", "ep_alltoall", "collective", 60, 3.60, cost),
            _SyntheticRow("F3", "expert_gate_up", "moe_grouped", 60, 2.40, cost),
            _SyntheticRow("F3", "expert_swiglu", "activation", 60, 0.30, cost),
            _SyntheticRow("F4", "tp_allreduce", "collective", 60, 1.20, cost),
        )

    rows, rows_plus1, layers = _make_rows(17 * 1024), _make_rows(17 * 1024 + 1), 60
    per_layer: dict[str, float] = {}
    for r in rows:
        per_layer[r.mechanism] = per_layer.get(r.mechanism, 0.0) + r.total_ms / layers
    subgraphs = {
        "attention": sum(per_layer.get(k, 0.0) for k in ("attention", "rope", "kv_cache", "indexer", "gdn_kda")),
        "norm_residual": per_layer.get("norm_residual", 0.0),
        "hyperconnection": per_layer.get("hyperconnection", 0.0),
        "moe": sum(per_layer.get(k, 0.0) for k in ("moe_grouped", "moe_router", "activation", "collective")),
        "ffn": 0.42,
    }
    raw_layer_sum = sum(per_layer.values())
    body_ms = raw_layer_sum * layers

    for phase in ("decode", "prefill"):
        dec_in = body_ms if phase == "decode" else 1.25
        pf_in = body_ms if phase == "prefill" else 2.50
        for tp, dp, ep in ((1, 16, 16), (2, 1, 1), (8, 2, 16)):
            for prompt_tokens in (768, 2048, 3072, 4096, 6144, 12288):
                for concurrency in (1, 3, 7, 12, 24):
                    adj_pl, adj_sg, adj_body, adj_dec, adj_pf = lower_layer_predictions(
                        phase=phase, prompt_tokens=prompt_tokens, concurrency=concurrency,
                        max_tokens=128, tp=tp, dp=dp, ep=ep, graph_buckets=(1, 2, 4, 8, 16, 32),
                        graph_bucket=8, layers=layers, rows=rows, per_layer=per_layer,
                        subgraphs=subgraphs, body_ms=body_ms, decode_body_ms=dec_in, prefill_body_ms=pf_in,
                    )
                    adj_pl2, adj_sg2, *_ = lower_layer_predictions(
                        phase=phase, prompt_tokens=prompt_tokens, concurrency=concurrency,
                        max_tokens=128, tp=tp, dp=dp, ep=ep, graph_buckets=(1, 2, 4, 8, 16, 32),
                        graph_bucket=8, layers=layers, rows=rows_plus1, per_layer=per_layer,
                        subgraphs=subgraphs, body_ms=body_ms, decode_body_ms=dec_in, prefill_body_ms=pf_in,
                    )
                    for m, v in adj_pl.items():
                        assert v == pytest.approx(adj_pl2[m], abs=1e-12), f"residencyperturbed {m}"
                    assert adj_sg["attention"] == pytest.approx(adj_sg2["attention"], abs=1e-12)
                    assert adj_sg["moe"] == pytest.approx(adj_sg2["moe"], abs=1e-12)
                    assert adj_sg["moe"] == pytest.approx(subgraphs["moe"], abs=1e-12)
                    assert adj_sg["ffn"] == pytest.approx(subgraphs["ffn"], abs=1e-12)
                    assert abs(sum(adj_pl.values()) - raw_layer_sum) <= 1e-6
                    assert adj_sg["norm_residual"] == pytest.approx(adj_pl["norm_residual"], abs=1e-6)
                    assert adj_sg["hyperconnection"] == pytest.approx(adj_pl["hyperconnection"], abs=1e-6)
                    assert adj_body == pytest.approx(body_ms, abs=1e-6)
                    assert adj_dec == pytest.approx(dec_in, abs=1e-6)
                    assert adj_pf == pytest.approx(pf_in, abs=1e-6)


def test_out_of_grid_prompt_lengths_and_concurrencies_are_monotonic() -> None:
    supply = _supply()
    for family_name, indices in _FAMILIES:
        model = _compact_model(family_name, indices)
        for tp, dp, ep in ((1, 1, 1), (2, 2, 4), (2, 8, 16), (8, 2, 16)):
            pipe = assemble_model(model, supply=supply, features=FeatureConfig.from_model(model, tp=tp, dp=dp, ep=ep), domain=_DOMAIN)
            pf_s = [float(pipe.forward(WorkloadGeometry((RequestGeometry(t, 0, phase="prefill"),), phase="prefill")).total_ms) for t in INTERLEAVED_TOKENS]
            dc_s = [float(pipe.forward(WorkloadGeometry.homogeneous(4, 1, t, phase="decode")).total_ms) for t in INTERLEAVED_TOKENS]
            for prev, curr in zip(pf_s, pf_s[1:]):
                assert curr > prev
            for prev, curr in zip(dc_s, dc_s[1:]):
                assert curr >= prev
            dc_c = [float(pipe.forward(WorkloadGeometry.homogeneous(c, 1, 3072, phase="decode")).total_ms) for c in INTERLEAVED_CONCURRENCIES]
            pf_c = [float(pipe.forward(WorkloadGeometry(tuple(RequestGeometry(768, 0, phase="prefill") for _ in range(c)), phase="prefill")).total_ms) for c in INTERLEAVED_CONCURRENCIES]
            for prev, curr in zip(dc_c, dc_c[1:]):
                assert curr >= prev - 1e-9
            for prev, curr in zip(pf_c, pf_c[1:]):
                assert curr > prev

    for key in ("deepseek-v4.1-flash", "kimi-k3"):
        full = _compact_model(key, (1, 2), hidden=load_model(key).hidden_size, intermediate=512, experts=64)
        pipe_tp2 = assemble_model(full, supply=supply, features=FeatureConfig.from_model(full, tp=2, dp=1, ep=2), domain=_DOMAIN)
        ms_c2 = float(pipe_tp2.forward(WorkloadGeometry.homogeneous(2, 1, 1024, phase="decode")).total_ms)
        ms_c3 = float(pipe_tp2.forward(WorkloadGeometry.homogeneous(3, 1, 1024, phase="decode")).total_ms)
        assert ms_c3 >= ms_c2 - 1e-9, f"{key} tp=2 decode concurrency 2->3 decreased ({ms_c2:.4f} -> {ms_c3:.4f})"


def test_grid_boundary_continuity_and_operator_metadata_invariance() -> None:
    supply = _supply()
    for family_name, indices in _FAMILIES:
        model = _compact_model(family_name, indices)
        pipe = assemble_model(model, supply=supply, features=FeatureConfig.from_model(model, tp=1, dp=1, ep=1), domain=_DOMAIN)
        for grid_pt in GRID_BOUNDARY_POINTS:
            ms_minus = float(pipe.forward(WorkloadGeometry((RequestGeometry(grid_pt - 1, 0, phase="prefill"),), phase="prefill")).total_ms)
            ms_at = float(pipe.forward(WorkloadGeometry((RequestGeometry(grid_pt, 0, phase="prefill"),), phase="prefill")).total_ms)
            ms_plus = float(pipe.forward(WorkloadGeometry((RequestGeometry(grid_pt + 1, 0, phase="prefill"),), phase="prefill")).total_ms)
            assert ms_minus <= ms_at <= ms_plus and (ms_plus - ms_minus) / max(ms_at, 1e-9) < 0.025

    geom_pf = WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill")
    for dp_n in (1, 4):
        for kind, fam, name, extra_base in (
            ("N0", "elementwise", "attn_norm", (("elements", 256),)),
            ("I0", "kv_access", "index_cache_store", (("row_bytes", 256),)),
        ):
            durs = {
                wfmt: float(CompiledOperator(
                    BoundOperator(OperatorTemplate(name=name, family=fam, role=name), StaticSignature(role=name, parallel=(("dp", dp_n),), extras=(*extra_base, ("routed_weight_format", wfmt)))),
                    supply, ImplementationPolicy(), kind=kind, domain=_DOMAIN,
                ).evaluate(geom_pf).duration_ms)
                for wfmt in ("bf16", "fp8_e4m3", "mxfp4")
            }
            assert durs["bf16"] == pytest.approx(durs["fp8_e4m3"], rel=1e-9)
            assert durs["bf16"] == pytest.approx(durs["mxfp4"], rel=1e-9)

    for kind, fam, name, par, ext in (
        ("C0", "collective", "allreduce", (("members", 2), ("tp", 2), ("dp", 8), ("ranks_per_node", 8)), (("wire_bytes", 8192.0), ("collective", "allreduce"))),
        ("C1", "transport", "dispatch", (("members", 16), ("ep", 16), ("tp", 2), ("dp", 8), ("ranks_per_node", 8)), (("wire_bytes", 8192.0), ("row_bytes", 4096.0), ("top_k", 8))),
    ):
        mb_durs = [
            float(CompiledOperator(
                BoundOperator(OperatorTemplate(name=name, family=fam, role=name), StaticSignature(role=name, parallel=par, extras=(*ext, ("max_bucket", mb)))),
                supply, ImplementationPolicy(), kind=kind, domain=_DOMAIN,
            ).evaluate(geom_pf).duration_ms)
            for mb in (64, 128)
        ]
        assert mb_durs[0] == pytest.approx(mb_durs[1], rel=1e-9)

    c0_op = CompiledOperator(
        BoundOperator(OperatorTemplate(name="allreduce", family="collective", role="allreduce"), StaticSignature(role="allreduce", parallel=(("members", 8), ("tp", 8), ("ranks_per_node", 8)), extras=(("wire_bytes", 8192.0), ("collective", "allreduce")))),
        supply, ImplementationPolicy(), kind="C0", domain=_DOMAIN,
    )
    c1_op = CompiledOperator(
        BoundOperator(OperatorTemplate(name="dispatch", family="transport", role="dispatch"), StaticSignature(role="dispatch", parallel=(("members", 16), ("ep", 16), ("tp", 8), ("ranks_per_node", 8)), extras=(("wire_bytes", 8192.0), ("row_bytes", 4096.0), ("top_k", 8), ("routed_weight_format", "mxfp4"), ("kv_bf16", 1)))),
        supply, ImplementationPolicy(), kind="C1", domain=_DOMAIN,
    )
    for op, bsz in ((c0_op, 1), (c1_op, 4)):
        for g in (2048, 4096, 8192):
            vals = [float(op.evaluate(WorkloadGeometry.homogeneous(bsz, q, q, phase="prefill")).duration_ms) for q in (g - 1, g, g + 1)]
            assert vals[0] <= vals[1] <= vals[2], f"{op.kind} non-monotonic around {g}: {vals}"

    g0_op = CompiledOperator(
        BoundOperator(OperatorTemplate(name="kda_recurrent", family="recurrent", role="kda_recurrent"), StaticSignature(role="kda_recurrent", heads=(("key_heads", 4), ("value_heads", 16), ("dk", 64), ("dv", 64), ("chunk", 64)))),
        supply, ImplementationPolicy(), kind="A3", domain=_DOMAIN,
    )
    g1023 = WorkloadGeometry((RequestGeometry(341, 0, phase="prefill"), RequestGeometry(341, 0, phase="prefill"), RequestGeometry(341, 0, phase="prefill")), phase="prefill")
    g1024 = WorkloadGeometry((RequestGeometry(341, 0, phase="prefill"), RequestGeometry(341, 0, phase="prefill"), RequestGeometry(342, 0, phase="prefill")), phase="prefill")
    ms_1023, ms_1024 = float(g0_op.evaluate(g1023).duration_ms), float(g0_op.evaluate(g1024).duration_ms)
    assert ms_1023 <= ms_1024 and (ms_1024 - ms_1023) / max(ms_1024, 1e-9) < 0.05


def test_n0_duration_is_independent_of_routed_weight_format() -> None:
    supply = _supply()
    for phase, q_len in (("prefill", 2048), ("decode", 1)):
        geom = (
            WorkloadGeometry.homogeneous(1, q_len, q_len, phase="prefill", cohort_batch=16)
            if phase == "prefill"
            else WorkloadGeometry((RequestGeometry(1, 0, phase="decode"),), phase="decode", cohort_batch=16)
        )
        durs = {}
        for wfmt in ("bf16", "fp8_e4m3", "mxfp4", "fp4_e2m1"):
            signature = StaticSignature(
                role="rmsnorm",
                matrix=(("elements", 8192),),
                parallel=(("dp", 16), ("ep", 16), ("tp", 1), ("ranks_per_node", 8)),
                extras=(("graph_buckets", (1, 2, 4, 8, 16, 32)), ("elements", 8192), ("E", 16), ("routed_weight_format", wfmt)),
            )
            op = CompiledOperator(
                BoundOperator(OperatorTemplate(name="rmsnorm", family="elementwise", role="norm"), signature),
                supply, ImplementationPolicy(), kind="N0", domain=_DOMAIN,
            )
            durs[wfmt] = float(op.evaluate(geom).duration_ms)
        assert all(value > 0.0 for value in durs.values())
        for wfmt in ("fp8_e4m3", "mxfp4", "fp4_e2m1"):
            assert durs[wfmt] == pytest.approx(durs["bf16"], rel=1e-12), f"N0 {phase} depends on routed_weight_format={wfmt}"


def test_lower_layer_predictions_leaves_moe_grouped_and_collective_untouched() -> None:
    from mindie_sim.prediction.piecewise.lowering import lower_layer_predictions

    layers = 60
    cost = _SyntheticCost((("cube", 1.0e9), ("vector", 1.0e8), ("hbm", 1.0e8)), (("cube", 1.0), ("vector", 0.5), ("hbm", 0.8)))
    f3_cost = replace(cost, derived=(("cohort_batch", 32), ("moe_rows", 12)))
    f2_cost = replace(cost, derived=(("cohort_batch", 32), ("routed_rows", 12), ("top_k", 6)))
    rows = (
        _SyntheticRow("F3", "expert_gate_up", "moe_grouped", layers, 2.4 * layers, f3_cost),
        _SyntheticRow("F3", "expert_down", "moe_grouped", layers, 1.6 * layers, f3_cost),
        _SyntheticRow("F2", "dispatch", "collective", layers, 3.0 * layers, f2_cost),
        _SyntheticRow("F2", "combine", "collective", layers, 3.0 * layers, f2_cost),
    )
    per_layer = {"moe_grouped": 4.0, "collective": 6.0}
    for phase in ("decode", "prefill"):
        for concurrency, tp, ep in ((1, 1, 16), (24, 1, 16), (64, 1, 64), (4, 2, 16)):
            adj, adj_sg, *_ = lower_layer_predictions(
                phase=phase, concurrency=concurrency, tp=tp, ep=ep,
                layers=layers, rows=rows, per_layer=per_layer, subgraphs={"moe": 10.0},
            )
            assert adj["moe_grouped"] == pytest.approx(4.0, abs=1e-12)
            assert adj["collective"] == pytest.approx(6.0, abs=1e-12)
            assert adj_sg["moe"] == pytest.approx(10.0, abs=1e-12)


def test_transport_dispatch_prefill_continuity_at_2048() -> None:
    supply = _supply()
    op = CompiledOperator(
        BoundOperator(
            OperatorTemplate(name="dispatch", family="transport", role="dispatch"),
            StaticSignature(
                role="dispatch",
                parallel=(("members", 16), ("ep", 16), ("tp", 8), ("ranks_per_node", 8)),
                extras=(("wire_bytes", 8192.0), ("row_bytes", 4096.0), ("top_k", 8), ("routed_weight_format", "mxfp4"), ("kv_bf16", 1)),
            ),
        ),
        supply, ImplementationPolicy(), kind="C1", domain=_DOMAIN,
    )
    for batch in (1, 2):
        vals = [float(op.evaluate(WorkloadGeometry.homogeneous(batch, q, q, phase="prefill")).duration_ms) for q in (2047, 2048, 2049)]
        # No discrete jump at the 2048 grid boundary; the batch=1 ratio is
        # continuous across it even though its smooth slope turns.
        assert abs(vals[1] - vals[0]) / max(vals[1], 1e-9) < 1e-4, f"batch={batch} dispatch jumped before q=2048: {vals}"
        assert abs(vals[2] - vals[1]) / max(vals[1], 1e-9) < 1e-4, f"batch={batch} dispatch jumped after q=2048: {vals}"
        if batch == 2:
            assert vals[0] <= vals[1] <= vals[2], f"batch=2 dispatch dipped at q=2048: {vals}"


def test_ttft_wave_scale_chunk_budget_and_chunked_mean_are_distinct() -> None:
    from mindie_sim.prediction.piecewise.service_span import chunked_prefill_mean_ttft_ms, ttft_wave_scale

    base = ttft_wave_scale(
        phase="prefill", concurrency=8, dp=1, ep=16, layers=40,
        graph_buckets=(1, 2, 4, 8), prompt_tokens=8192,
    )
    assert base == pytest.approx(1.0 + 0.045, rel=1e-12)
    chunked_c2 = ttft_wave_scale(
        phase="prefill", concurrency=2, dp=1, ep=16, layers=2,
        graph_buckets=(), prompt_tokens=8192, max_batched_tokens=8192,
    )
    assert chunked_c2 == pytest.approx(0.75, rel=1e-12)
    chunked_c4 = ttft_wave_scale(
        phase="prefill", concurrency=4, dp=1, ep=16, layers=2,
        graph_buckets=(), prompt_tokens=16384, max_batched_tokens=8192,
    )
    assert chunked_c4 == pytest.approx((8 + 1) / (2.0 * 8), rel=1e-12)

    mean = chunked_prefill_mean_ttft_ms((8192, 8192, 8192, 8192), dp=1, max_batched_tokens=8192, body_one=lambda _tokens: 1.0)
    assert mean == pytest.approx(2.5, rel=1e-12)

    import functools

    class _OneArgBody:
        def __call__(self, tokens: int) -> float:
            return float(tokens) / 8192.0

    class _TwoArgBody:
        def __call__(self, tokens: int, done: int) -> float:
            return float(tokens + done) / 8192.0

    def _raw_two(scale: float, tokens: int) -> float:
        return scale * float(tokens) / 8192.0

    def _raw_three(scale: float, tokens: int, done: int) -> float:
        return scale * float(tokens + done) / 8192.0

    def _kw_one(tokens: int, scale: float) -> float:
        return scale * float(tokens) / 8192.0

    def _kw_two(tokens: int, done: int, scale: float) -> float:
        return scale * float(tokens + done) / 8192.0

    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=_OneArgBody()) == pytest.approx(1.5, rel=1e-12)
    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=_TwoArgBody()) == pytest.approx(3.0, rel=1e-12)
    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=functools.partial(_raw_two, 1.0)) == pytest.approx(1.5, rel=1e-12)
    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=functools.partial(_raw_three, 1.0)) == pytest.approx(3.0, rel=1e-12)
    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=functools.partial(_kw_one, scale=1.0)) == pytest.approx(1.5, rel=1e-12)
    assert chunked_prefill_mean_ttft_ms((8192, 8192), dp=1, max_batched_tokens=8192, body_one=functools.partial(_kw_two, scale=1.0)) == pytest.approx(3.0, rel=1e-12)

    supply = _supply()
    for buckets_a, buckets_b in (((1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)), ((1, 2, 4, 8, 16, 32, 64), (1, 2, 4, 8, 16, 32, 64, 128))):
        ops = [
            CompiledOperator(
                BoundOperator(
                    OperatorTemplate(name="prepare", family="elementwise", role="prepare"),
                    StaticSignature(role="prepare", parallel=(("tp", 8), ("dp", 4), ("ep", 16), ("ranks_per_node", 8)), extras=(("elements", 8192), ("E", 16), ("graph_buckets", b), ("routed_weight_format", "mxfp4"))),
                ),
                supply, ImplementationPolicy(), kind="A3", domain=_DOMAIN,
            )
            for b in (buckets_a, buckets_b)
        ]
        geom = WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill", cohort_batch=4)
        assert float(ops[0].evaluate(geom).duration_ms) == pytest.approx(float(ops[1].evaluate(geom).duration_ms), abs=1e-12)

