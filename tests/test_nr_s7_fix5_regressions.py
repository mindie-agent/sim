"""Permanent FIX5 source-wrapper and universal raw-type boundary regressions.

[Role]
  Compare HF wrappers with the pinned source predicate and exercise every
  annotated public argument and dataclass field in the 26 production modules.
[Invariants]
  * Wrong types must raise ModelingError before the implementation runs.
  * Inert, correctly typed sentinels make accidental dereferences observable;
    concrete witnesses and the retained suite separately exercise valid inputs.
  * Recursive sequences test both container and element types, including bool
    versus int and integral float versus int before equality or normalization.
  * The source predicate is a standalone transcription of utils.py:196-219 at
    6e3299ab492ddbadb4c8e3a00b88c982c9a28131; the external probe extracts it too.
[Timeouts & Locks] Bounded CPU checks and source inventory reads; no locks.
[Side-Effect Policy] No persistent writes; scratch belongs to external basetemp.
"""
from __future__ import annotations

import dataclasses as dc
import importlib
import inspect
import itertools
from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import SimpleNamespace as NS, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest

from nr_s7_fix2_support import _case, _config, _geometry, _keys, _layer
from mindie_sim.dspark_prefix_cache import dspark_prefix_plan
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, hardware_supports_fp8_attention, model_is_sparse_sfa,
    model_mlapo_eligible, projection_quant_class)
from mindie_sim.kv_layout_physics import KvNzPreconditions
from mindie_sim.linear_state_physics import linear_state_memory, compare_linear_state_memory
from mindie_sim.linear_state_replay import precopy_cost, verify_commit_cost, recoverssm_crossover
from mindie_sim.linear_state_prefix_replay import prefix_replay_cost, prefix_replay_plan
from mindie_sim.schema.common import ModelingError

CASE = lru_cache(None)(_case)


@dc.dataclass
class SourceConfig:
    hf_text_config: object
    hf_config: object


class CustomConfig:
    def __init__(self, hf_text_config, hf_config):
        self.hf_text_config, self.hf_config = hf_text_config, hf_config


def _source_sparse(model_config):
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_text_config is None:
        return False
    if any(hasattr(getattr(model_config, attr, None), "index_kpool")
           for attr in ("hf_text_config", "hf_config")):
        return False
    return (hasattr(hf_text_config, "index_topk")
            and not hasattr(hf_text_config, "compress_ratios")
            and not hasattr(hf_config, "compress_ratios"))

@lru_cache(None)
def sparse_witnesses():
    rows = []
    for wrapper in (NS, SourceConfig, CustomConfig):
        for flags in itertools.product((False, True), repeat=5):
            text, hf = NS(), NS()
            for present, obj, key, value in zip(flags, (text, text, hf, text, hf),
                    ("index_topk", "index_kpool", "index_kpool", "compress_ratios", "compress_ratios"),
                    (0, 1, None, [], [])):
                if present:
                    setattr(obj, key, value)
            model = wrapper(hf_text_config=text, hf_config=hf)
            expected, actual = _source_sparse(model), model_is_sparse_sfa(model)
            assert actual is expected, (wrapper, flags, expected, actual)
            rows.append({"wrapper": wrapper.__name__, "flags": flags,
                         "reference": expected, "local": actual})
    catalog = []
    base = CASE("deepseek-v3.2").model
    for topk, kpool, compress in itertools.product((0, 8), (False, True), (False, True)):
        facts = {k: v for k, v in base.catalog_facts
                 if k not in {"index_topk", "index_kpool", "compress_ratios"}}
        facts["index_topk"] = topk
        if kpool:
            facts["index_kpool"] = 1
        if compress:
            facts["compress_ratios"] = []
        model = dc.replace(base, mla=dc.replace(base.mla, index_topk=topk),
                           catalog_facts=tuple(facts.items()))
        actual = model_is_sparse_sfa(model)
        assert actual is (not kpool and not compress)
        catalog.append({"topk": topk, "kpool": kpool, "compress": compress, "sparse": actual})
    for topk in (0, 8):
        facts = tuple((k, v) for k, v in base.catalog_facts
                      if k not in {"index_topk", "index_kpool", "compress_ratios"})
        model = dc.replace(base, mla=dc.replace(base.mla, index_topk=topk), catalog_facts=facts)
        assert model_is_sparse_sfa(model) is (topk > 0)
        catalog.append({"fallback_topk": topk, "sparse": topk > 0})
    return {"r5_cases": 64, "custom_cases": 32, "mismatches": 0,
            "rows": rows, "catalog": catalog}


def test_f03_all_wrapper_representations_and_normalized_catalog():
    assert len(sparse_witnesses()["rows"]) == 96


@pytest.mark.parametrize("bad", ["false", 0, 1, None, [], {}])
def test_f14_mlapo_controls_before_all_short_circuits(bad):
    base = CASE().model
    for model, arch, disabled in itertools.product(
            (base, dc.replace(base, mla=None)), ("A3", "A5"), (False, True)):
        for control in ("enable_mlapo", "decode_instance", "is_draft"):
            values = dict(enable_mlapo=not disabled, decode_instance=True, is_draft=disabled)
            values[control] = bad
            with pytest.raises(ModelingError):
                model_mlapo_eligible(model, hardware_arch=arch, **values)


@pytest.mark.parametrize("bad", [[], {}, 123, False, None, "", "   "])
def test_f14_hardware_arch_validated_even_when_disabled_or_draft(bad):
    with pytest.raises(ModelingError):
        hardware_supports_fp8_attention(bad)
    for flags in ({"enable_mlapo": False}, {"is_draft": True}, {}):
        with pytest.raises(ModelingError):
            model_mlapo_eligible(CASE().model, hardware_arch=bad, **flags)


def _prefix(**overrides):
    values = dict(keys=_keys(), page_geometry=_geometry(), aux_layer_ids=(1,),
                  target_hidden_size=8, target_dtype="bf16", tokens=128,
                  prompt_tokens=128, block_hash_count=1)
    values.update(overrides)
    return dspark_prefix_plan(**values)


@pytest.mark.parametrize("field", ["draft_groups", "block_sizes"])
@pytest.mark.parametrize("bad", [None, "0", b"0", {}, 123, [None], [False], [0.0], [128.0]])
def test_f14_prefix_geometry_raw_container_and_elements(field, bad):
    geometry = _geometry()
    geometry[field] = bad
    with pytest.raises(ModelingError):
        _prefix(page_geometry=geometry)


@pytest.mark.parametrize("bad", [None, 1, "x", {}, [None], [False], [1.0]])
def test_f14_linear_public_entrypoints_validate_config_and_layers(bad):
    for align in (False, True):
        config = _config(mamba_cache_mode="align" if align else "none", enable_prefix_caching=align)
        layer = _layer()
        plan = prefix_replay_plan(config, [layer], isl=256, prefix_hit=128)
        calls = (
            lambda c, ls: precopy_cost(c, ls, sequence_start=128, sequence_end=132, batch=1, hbm_bytes_per_s=1e12),
            lambda c, ls: verify_commit_cost(c, ls, batch=1, sequence_tokens=128, hbm_bytes_per_s=1e12),
            lambda c, ls: recoverssm_crossover(c, ls, hbm_bytes_per_s=1e12),
            lambda c, ls: linear_state_memory(c, ls, sequence_tokens=128),
            lambda c, ls: compare_linear_state_memory(c, ls, sequence_tokens=128),
            lambda c, ls: prefix_replay_plan(c, ls, isl=256, prefix_hit=128),
            lambda c, ls: prefix_replay_cost(c, ls, plan=plan, hbm_bytes_per_s=1e12))
        for call in calls:
            with pytest.raises(ModelingError):
                call(config, bad)
            with pytest.raises(ModelingError):
                call(bad, [layer])


def test_f14_nz_disabled_and_aux_preexisting_paths_validate_raw_inputs():
    for bad in ([], {}, 123):
        with pytest.raises(ModelingError):
            KvNzPreconditions(enable_kv_nz=False, is_mla=True, kv_role=bad)
    options = HybridKvStateOptions(dspark_aux_layer_ids=(1,))
    for bad in ([None], [False], [1.0], "1", {}, None):
        with pytest.raises(ModelingError):
            options.with_dspark_aux(bad)
    assert options.with_dspark_aux((2,)) is options


def test_f14_registration_methods_check_all_raw_ids_before_lookup():
    plan = _prefix()
    for bad in (False, 0.0, "0", None):
        with pytest.raises(ModelingError):
            plan.validate_block_table({bad: [0]}, [0])
        with pytest.raises(ModelingError):
            plan.validate_block_table({0: [0, bad]}, [0])
    for bad in (False, 1.0, None, [], {}, "1", 0):
        with pytest.raises(ModelingError):
            plan.registration.pages[0].validate_geometry(bad, 128)
        with pytest.raises(ModelingError):
            plan.registration.pages[0].validate_geometry(1, bad)


def test_f14_quantization_mapping_and_fields_are_not_string_coerced():
    base = CASE().model
    for bad in (None, [], 123, "int8", {"name": []}, {"weight_format": False}):
        facts = {**dict(base.catalog_facts), "quantization": bad}
        with pytest.raises(ModelingError):
            projection_quant_class(dc.replace(base, catalog_facts=tuple(facts.items())))


def _sentinel(annotation):
    """Type-correct inert inputs: an implementation dereference must fail."""
    if annotation is Any or annotation is inspect.Parameter.empty:
        return None
    if annotation is type(None):
        return None
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (UnionType, Union):
        return _sentinel(next(t for t in args if t is not type(None)))
    if annotation in (int, float, bool, str):
        return {int: 1, float: 1.0, bool: False, str: "valid"}[annotation]
    if origin is tuple:
        return tuple(_sentinel(t) for t in (args[:1] if args[-1:] == (Ellipsis,) else args))
    if origin in (Sequence, list):
        return [_sentinel(args[0])] if args else []
    if origin in (dict, Mapping):
        return {}
    if annotation is type:
        return int
    return object.__new__(annotation)


def _invalid_values(annotation):
    """A fixed adversarial matrix, independent of production validation code."""
    if annotation in (Any, inspect.Parameter.empty):
        return []
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (UnionType, Union):
        # Scalar/record optional fields and Case-or-mapping both reject these.
        nonnull = [t for t in args if t is not type(None)]
        if len(nonnull) > 1:
            return [False, 1.0, "bad", []]
        return [v for v in _invalid_values(nonnull[0]) if v is not None]
    if annotation is int:
        return [False, 1.0, "1", None, [], {}]
    if annotation is bool:
        return ["false", 0, 1, None, [], {}]
    if annotation is float:
        return [False, "1", None, [], {}, float("nan"), float("inf")]
    if annotation is str:
        return [[], {}, 123, False, None]
    if origin in (tuple, list, Sequence):
        result = [None, 123, "x", b"x", {}, bytearray(b"x")]
        if args:
            item_type = args[0]
            for bad in _invalid_values(item_type):
                base = _sentinel(annotation)
                if base:
                    row = list(base)
                    row[0] = bad
                    result.append(tuple(row) if origin is tuple else row)
        return result
    if origin in (dict, Mapping):
        return [None, [], "x", 123, {False: None}]
    if annotation is type:
        return [None, 1, "int", [], {}]
    return [None, False, 1.0, "x", [], {}]


@lru_cache(None)
def _inventory():
    from nr_s7_fix3_verify import SCOPED
    rows = []
    for filename in SCOPED:
        if not filename.startswith("mindie_sim/"):
            continue
        module = importlib.import_module(filename[:-3].replace("/", "."))
        seen = set()
        for name, obj in vars(module).items():
            if getattr(obj, "__module__", None) != module.__name__ or id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.isclass(obj) and dc.is_dataclass(obj):
                rows.append((filename, name, obj, inspect.signature(obj), get_type_hints(obj), {}))
                for method_name, raw in vars(obj).items():
                    if method_name.startswith("_"):
                        continue
                    method = getattr(obj, method_name)
                    if isinstance(raw, classmethod):
                        rows.append((filename, name + "." + method_name, method, inspect.signature(method), get_type_hints(method), {}))
                    elif inspect.isfunction(method):
                        rows.append((filename, name + "." + method_name, method, inspect.signature(method), get_type_hints(method), {"self": object.__new__(obj)}))
            elif inspect.isfunction(obj) and (not name.startswith("_") or name in getattr(module, "__all__", ())):
                if name != "typed_inputs":  # decorator configuration, not a modeling input
                    rows.append((filename, name, obj, inspect.signature(obj), get_type_hints(obj), {}))
    return tuple(rows)

@lru_cache(None)
def boundary_matrix():
    """Every declared boundary, including output records and exported helpers."""
    files = {}
    for filename, name, call, signature, hints, receiver in _inventory():
        row = files.setdefault(filename, {"boundaries": 0, "parameters": 0, "checks": 0, "entries": []})
        row["boundaries"] += 1
        kwargs = {p: _sentinel(hints.get(p, Any)) for p, arg in signature.parameters.items()
                  if p not in receiver and arg.kind not in (arg.VAR_POSITIONAL, arg.VAR_KEYWORD)}
        kwargs.update(receiver)
        tested = []
        for parameter, annotation in hints.items():
            if parameter == "return" or parameter not in kwargs:
                continue
            invalid = _invalid_values(annotation)
            if not invalid:
                continue
            row["parameters"] += 1
            tested.append(parameter)
            for bad in invalid:
                try:
                    call(**{**kwargs, parameter: bad})
                except ModelingError as error:
                    message = str(error)
                else:
                    raise AssertionError((filename, name, parameter, bad, "accepted"))
                # Another missing field or an implementation exception cannot
                # masquerade as validation of the parameter under test.
                expected = ("summary_not_text" if name == "summary_token_count"
                            else "aux_layers" if name == "dspark_prefix_plan" and parameter == "aux_layer_ids"
                            else parameter)
                assert expected in message, (filename, name, parameter, bad, message)
                row["checks"] += 1
        row["entries"].append({"name": name, "parameters": tested})
    assert len(files) == 26
    return {"modules": len(files), "checks": sum(r["checks"] for r in files.values()),
            "parameters": sum(r["parameters"] for r in files.values()),
            "boundaries": sum(r["boundaries"] for r in files.values()), "files": files}


def test_f14_systematic_all_public_boundaries_and_dataclasses():
    assert boundary_matrix()["checks"] > 3000


def test_f14_mapping_and_unannotated_payload_paths():
    from mindie_sim.hybrid_kv_state_mcp import hybrid_kv_state_mcp_envelope, envelope_token_upper_bound
    from mindie_sim.parallel_decode_cost import iter_axis_groups, check_route_qualified
    report = {"schema": "mindie-sim.hybrid-kv-state/1", "status": "qualified", "errors": [], "diagnostics": []}
    for bad in (1, False, {}):
        with pytest.raises(ModelingError):
            hybrid_kv_state_mcp_envelope({**report, "errors": bad})
    for bad in ("false", 1, None, [], {}):
        for extra in ({"dspark": {"active": bad}}, {"layout": {"kv_nz": {"mlapo_enabled": bad}}}):
            with pytest.raises(ModelingError):
                hybrid_kv_state_mcp_envelope({**report, **extra})
    for bad in (None, 123, "x", [None], [[False]], [[1.0]], {0: [1]}):
        with pytest.raises(ModelingError):
            iter_axis_groups({"tp_groups": bad})
    for bad in (None, 1, [], {}, "x"):
        with pytest.raises(ModelingError):
            check_route_qualified(bad, {})
    with pytest.raises(ModelingError):
        envelope_token_upper_bound({"bad": object()})
    with pytest.raises(ModelingError):
        prefix_replay_cost(_config(), [_layer()], plan={"replay_tokens": 0}, hbm_bytes_per_s=1e12)


def test_f14_disabled_mode_valid_inputs_still_execute():
    cfg, layer = _config(), _layer()
    assert precopy_cost(cfg, [layer], sequence_start=128, sequence_end=132,
                        batch=1, hbm_bytes_per_s=1e12)["copy_bytes"] == 0
    assert _prefix().draft_groups == (0,)
    assert hardware_supports_fp8_attention("A5") is True
    assert hardware_supports_fp8_attention("A3") is False


def test_f14_structural_any_interfaces_and_forwarding_aliases():
    from mindie_sim import nr_s7_validate as v
    from mindie_sim.dspark_dcp_cost import non_causal_block_drafter_cost
    from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state
    from mindie_sim.hybrid_kv_state_options import _digest, _json_safe
    from mindie_sim.linear_state_physics import linear_layers_from_model
    from mindie_sim.memory_breakdown import memory_ledger, memory_breakdown, apply_state_pool_delta
    from mindie_sim.parallel_decode_cost import linear_state_step_overhead
    from mindie_sim.xpyd_transfer import nz_transfer_end_reformat
    for bad in (None, False, 123, [], {}, "bad"):
        for call in (evaluate_hybrid_kv_state, memory_ledger, memory_breakdown,
                     linear_layers_from_model, apply_state_pool_delta):
            with pytest.raises(ModelingError):
                call(bad)
        with pytest.raises(ModelingError):
            linear_state_step_overhead(bad, [_layer()], batch=1, context=128, hbm_bytes_per_s=1e12)
        with pytest.raises(ModelingError):
            dc.replace(_geometry()["registration"].pages[0], page=bad)
        assert isinstance(v.sanitize_symbol(bad), str)
        assert isinstance(v.sanitize_code(bad), str)
        with pytest.raises(ModelingError):
            linear_layers_from_model(NS(layer_plans=CASE().model.layer_plans,
                                        activation_dtype=bad), tp=8)
    for bad in ([None], [False], [1.0], None, 123, "bad", {}):
        with pytest.raises(ModelingError):
            linear_layers_from_model(NS(layer_plans=bad))
        with pytest.raises(ModelingError):
            dc.replace(_geometry()["registration"], loaded_pages=bad)
        with pytest.raises(ModelingError):
            v.typed_sequence("ids", bad, int)
    for fn in (v.strict_int, v.strict_number, v.positive_rate, v.strict_text,
               v.strict_dtype, v.dtype_elem_bytes, v.strict_bool):
        for bad in ([], {}, "false" if fn != v.strict_text else 1):
            with pytest.raises(ModelingError):
                fn("value", bad)
    for bad in (None, [], {}, False, 123):
        with pytest.raises(ModelingError):
            v.bpe_token_upper_bound(bad)
    for bad in (False, 123, [], {}, "bad"):
        with pytest.raises(ModelingError):
            nz_transfer_end_reformat(pages=0, layers=0, nd_page_bytes=0,
                                     hbm_bytes_per_s=1e12, nz_layout=bad)
    with pytest.raises(ModelingError):
        non_causal_block_drafter_cost(unrecognized=True)
    with pytest.raises(ModelingError):
        _json_safe({"bad": object()})
    with pytest.raises(ModelingError):
        _digest({0: 1, "a": 2})

@lru_cache(None)
def fix6_domain_witnesses():
    """Concrete valid controls and inactive/active domain counterexamples."""
    from mindie_sim.hybrid_kv_state_options import (
        _attention_row, _attention_layer_count, _attention_layers_for_stage, _layer_stage)
    from mindie_sim.dspark_kv_physics import dspark_swa_footprint
    from mindie_sim.kv_pool_geometry import grouped_pool_layout
    from mindie_sim.xpyd_transfer import (
        kv_transfer_contention, expected_decode_cycles, completed_requests_per_s)
    counts = {"nested": 0, "transfer": 0, "swa": 0, "fallback": 0}

    def reject(group, call):
        with pytest.raises(ModelingError):
            call()
        counts[group] += 1

    model = CASE().model
    plan = next(p for p in model.layer_plans if p.attention == "mla")
    malformed = [None, False, 0.0, [], {}]
    malformed += [dc.replace(plan, **{key: value}) for key, value in (
        ("attention", []), ("attention", "unknown"), ("index", -1),
        ("kv_lora_rank", -1), ("qk_rope_head_dim", False),
        ("mla_qkv_a_gate_fused", []), ("head_dim", 10**400))]
    for bad, later in itertools.product(malformed, (False, True)):
        broken = dc.replace(model, layer_plans=(plan, bad) if later else (bad,))
        for arch, flags in itertools.product(("A2", "A5"),
                ({}, {"enable_mlapo": False}, {"is_draft": True}, {"decode_instance": True})):
            reject("nested", lambda: model_mlapo_eligible(broken, hardware_arch=arch, **flags))
        for call in (_attention_row, _attention_layer_count,
                     lambda m: _attention_layers_for_stage(m, 0, (45,))):
            reject("nested", lambda: call(broken))
    for field, value in (("layer_plans", None), ("layer_plans", []),
                         ("mla", False), ("auxiliary", [])):
        broken = dc.replace(model, **{field: value})
        reject("nested", lambda: model_mlapo_eligible(broken, enable_mlapo=False))
    for partition in ((45, -1), (), (False, 45)):
        reject("nested", lambda: _layer_stage(partition, 0))
    assert _attention_layer_count(model) > 0
    assert _attention_layers_for_stage(model, 0, (45,))
    kw = dict(n_prefill=1, n_decode=1, nic_bw_gbps=1.0, hbm_bw_bytes_per_s=1e12,
              p_world=1, d_world=1, decode_step_ms=1.0, osl=1, requests_per_s=1.0)
    invalid = dict(n_prefill=0, n_decode=-1, nic_bw_gbps=-1.0,
        hbm_bw_bytes_per_s=0.0, p_world=0, d_world=0, decode_step_ms=-1.0,
        osl=-1, requests_per_s=-1.0, roce_bw_gbps=-1.0, roce_oversubscription=0.5,
        staging_bytes=-1.0, transfer_latency_ms=-1.0, p_hbm_fraction=-1.0,
        d_hbm_fraction=1.5, utilization_cap=1.0, shared_cut_bytes_per_s=-1.0)
    for size, rate, latency in itertools.product((0.0, 1.0), (0.0, 1.0), (0.0, 0.1)):
        base = {**kw, "requests_per_s": rate, "transfer_latency_ms": latency}
        assert kv_transfer_contention(size, **base)["feasible"]
        for key, bad in invalid.items():
            reject("transfer", lambda: kv_transfer_contention(size, **{**base, key: bad}))
    swa = dict(sequence_tokens=128, swa_window_tokens=128, extra_retained_tokens=0,
               kv_block_size=128, layers=1, kv_heads=1, head_dim=128, tp=1, dtype="bf16")
    for mla in (None, 128):
        assert dspark_swa_footprint(**swa, mla_row_elements=mla).bounded_bytes > 0
        for key in ("sequence_tokens", "swa_window_tokens", "kv_block_size", "layers",
                    "kv_heads", "head_dim", "tp"):
            for bad in (0, -1):
                reject("swa", lambda: dspark_swa_footprint(**{**swa, key: bad}, mla_row_elements=mla))
    for bad in (-0.1, 1.1):
        for draft in (0, 2):
            reject("fallback", lambda: expected_decode_cycles(4, draft, bad))
            reject("fallback", lambda: completed_requests_per_s(1, 1.0, 4, 0.0,
                                                                 speculative_k=draft, accept_rate=bad))
    reject("fallback", lambda: expected_decode_cycles(4, 2, 0.5, -1.0))
    reject("fallback", lambda: completed_requests_per_s(1, 1.0, 4, 0.0, speculative_k=-1))
    for extra in ({"blocks": -1}, {"recurrent_pages": (-1,)}):
        reject("fallback", lambda: grouped_pool_layout(attn_group_size=1, attn_page_bytes=128,
                recurrent_groups=(), recurrent_blocks=(), **extra))
    for bad in ((-1,), (2, 1), (1, 1)):
        reject("fallback", lambda: HybridKvStateOptions(dspark_aux_layer_ids=(1,)).with_dspark_aux(bad))
    return {"checks": sum(counts.values()), "families": counts, "failures": 0}


def test_fix6_nested_records_and_prebranch_domains():
    assert fix6_domain_witnesses()["checks"] >= 400


def _overflow_values(annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (int, float):
        return [10**400, -10**400]
    if origin in (UnionType, Union):
        return [v for a in args for v in _overflow_values(a)]
    if origin in (tuple, list, Sequence) and args:
        result = []
        types = args[:1] if origin is not tuple or args[-1:] == (Ellipsis,) else args
        for index, kind in enumerate(types):
            for bad in _overflow_values(kind):
                row = list(_sentinel(annotation))
                row[index] = bad
                result.append(tuple(row) if origin is tuple else row)
        return result
    if origin in (dict, Mapping) and args:
        return [{_sentinel(args[0]): v} for v in _overflow_values(args[1])]
    return []

@lru_cache(None)
def fix6_overflow_matrix():
    """Both signs of 10**400 at every annotated numeric argument/leaf."""
    import sys
    files = {}
    for filename, name, call, signature, hints, receiver in _inventory():
        row = files.setdefault(filename, {"boundaries": 0, "numeric_boundaries": 0, "checks": 0})
        row["boundaries"] += 1
        kwargs = {p: _sentinel(hints.get(p, Any)) for p, arg in signature.parameters.items()
                  if p not in receiver and arg.kind not in (arg.VAR_POSITIONAL, arg.VAR_KEYWORD)}
        kwargs.update(receiver)
        target = call.__init__ if inspect.isclass(call) else call
        code = getattr(getattr(target, "__wrapped__", None), "__code__", None)
        checked = False
        for parameter, annotation in hints.items():
            if parameter == "return" or parameter not in kwargs:
                continue
            for bad in _overflow_values(annotation):
                entered = []
                def profile(frame, event, arg):
                    if event == "call" and frame.f_code is code:
                        entered.append(True)
                sys.setprofile(profile)
                try:
                    with pytest.raises(ModelingError) as error:
                        call(**{**kwargs, parameter: bad})
                finally:
                    sys.setprofile(None)
                expected = "aux_layers" if name == "dspark_prefix_plan" and parameter == "aux_layer_ids" else parameter
                assert expected in str(error.value), (filename, name, parameter, str(error.value))
                assert not entered, (filename, name, parameter)
                row["checks"] += 1
                checked = True
        row["numeric_boundaries"] += int(checked)
    return {"modules": len(files), "boundaries": sum(r["boundaries"] for r in files.values()),
            "checks": sum(r["checks"] for r in files.values()), "failures": 0, "files": files}


def test_fix6_all_numeric_boundaries_reject_giant_integers_before_body():
    assert fix6_overflow_matrix()["checks"] > 900


def test_fix6_computed_product_overflow_and_untyped_numeric_validators():
    from mindie_sim.nr_s7_validate import strict_number, strict_int, positive_rate
    from mindie_sim.kv_layout_physics import KvNzPlane
    for value in (10**400, -10**400):
        for function in (strict_int, strict_number, positive_rate):
            with pytest.raises(ModelingError):
                function("value", value)
    plane = KvNzPlane("nope", 128, "bf16", 10**200, 1, 10**200)
    for read in (lambda: plane.logical_bytes, lambda: plane.physical_bytes, plane.to_dict):
        with pytest.raises(ModelingError, match="overflow"):
            read()


def test_fix6_nested_case_and_stage_records_before_inactive_paths():
    from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state
    from mindie_sim.memory_breakdown import memory_breakdown, build_memory_ledger
    from mindie_sim.memory_state_delta import apply_state_pool_delta
    from mindie_sim.parallel_decode_cost import DecodeCostModel
    from mindie_sim.xpyd_transfer import logical_transfer_bytes
    case = CASE()
    broken_cases = [dc.replace(case, model=dc.replace(case.model, layer_plans=(None,))),
        dc.replace(case, runtime=dc.replace(case.runtime, parallel=None)),
        dc.replace(case, machine=dc.replace(case.machine, dies=(None,)))]
    for broken in broken_cases:
        for call in (evaluate_hybrid_kv_state, memory_breakdown, DecodeCostModel.from_case,
                     lambda c: logical_transfer_bytes(c, prompt_tokens=1, destination_cached_tokens=1)):
            with pytest.raises(ModelingError):
                call(broken)
    base = memory_breakdown(case, sequences=1, sequence_tokens=128)
    malformed = (dc.replace(base.components[0], per_stage_bytes=(None,)), *base.components[1:])
    for field, bad in (("components", malformed), ("components", ()),
                       ("stage_peak_bytes", ()), ("per_rank_bytes", ((0, 99, 0, 1.0, 1.0),))):
        with pytest.raises(ModelingError):
            apply_state_pool_delta(dc.replace(base, **{field: bad}), delta_bytes=0.0)
    with pytest.raises(ModelingError):
        apply_state_pool_delta(base, delta_bytes=-1.0, per_stage_bytes=(0.0,))
    ledger = build_memory_ledger(case)
    bad_coordinates = tuple((99, *row[1:]) for row in ledger.rank_map)
    for field, bad in (("aux", []), ("workspace", []), ("cache_detail", []),
                       ("stage_capacity", []), ("rank_map", bad_coordinates),
                       ("rank_map", tuple((0,) for _ in ledger.rank_map)),
                       ("aux_rank", tuple(() for _ in ledger.rank_map)),
                       ("capacity_rank", (*ledger.capacity_rank, 0.0))):
        with pytest.raises(ModelingError):
            dc.replace(ledger, **{field: bad})


def test_fix6_parallel_and_memory_override_domains():
    from mindie_sim.parallel_decode_cost import DecodeCostModel, linear_state_step_overhead
    from mindie_sim.memory_breakdown import build_memory_ledger
    model = DecodeCostModel.from_case(CASE())
    for field, bad in (("context", -1), ("global_batch", -1), ("routing_skew", 0.5),
                       ("eplb_redundant_experts", -1), ("host_overlap", 1.5),
                       ("pp_p2p_ms", -1.0), ("endpoints", (-1,))):
        with pytest.raises(ModelingError):
            model.step(1, **{field: bad})
    with pytest.raises(ModelingError):
        model.step(-1, per_replica_batch=1)
    for bad in (-0.1, 1.1):
        with pytest.raises(ModelingError):
            linear_state_step_overhead(_config(), [_layer()], batch=1, context=128,
                                        hbm_bytes_per_s=1e12, overlap_fraction=bad)
    disabled = dc.replace(CASE(), runtime=dc.replace(CASE().runtime, graph_capture_mode="none"))
    slots = (0,) * disabled.runtime.parallel.world_size
    for extra in ({"graph_buckets": (-1,)},
                  {"eplb_redundant_slots_per_rank": slots, "eplb_slots_per_rank": (-1,)},
                  {"eplb_redundant_slots_per_rank": slots, "eplb_slots_per_rank": (*slots, 0)},
                  {"eplb_slots_per_rank": (*slots, 0)},
                  {"eplb_redundant_slots_per_rank": ()},
                  {"enable_eplb": False, "eplb_rebalance_staging_bytes_per_rank": (*slots, 0)}):
        with pytest.raises(ModelingError):
            build_memory_ledger(disabled, **extra)


def test_fix6_hybrid_disabled_helpers_validate_declared_domains():
    from mindie_sim.hybrid_kv_state_blocks import _nz_block, _pool_block, _attention_page_bytes
    from mindie_sim.hybrid_kv_state_dspark import _dspark_block
    case, options = CASE(), HybridKvStateOptions()
    nz = dict(attn_kind="mla", attn_row=576, block_size=128, batch=1,
              tokens=128, sparse=False, mlapo=False, hbm=1e12, launch_ms=0.0,
              staging_bytes=0, errors=[])
    assert _nz_block(case, options, **nz)[1] == [0.0]
    for name, bad in (("attn_row", 0), ("block_size", 0), ("batch", 0), ("tokens", -1),
                      ("hbm", 0.0), ("launch_ms", -1.0), ("staging_bytes", -1)):
        with pytest.raises(ModelingError):
            _nz_block(case, options, **{**nz, name: bad})
    empty = dc.replace(case, model=dc.replace(case.model, auxiliary=None, layer_plans=()))
    pool = dict(tokens=128, batch=1, block_size=128, recoverssm=False,
                sparse=False, errors=[], warnings=[])
    assert _pool_block(empty, options, (), **pool)[0] is None
    draft = dict(active=False, tokens=128, prompt_tokens=128, available_hashes=1,
                 batch=1, block_size=128, kv_elem=2.0, errors=[], warnings=[])
    assert _dspark_block(empty, options, **draft)[0]["present"] is False
    for name in ("tokens", "batch", "block_size"):
        with pytest.raises(ModelingError):
            _pool_block(empty, options, (), **{**pool, name: 0})
    for name in ("tokens", "prompt_tokens", "available_hashes", "batch", "block_size", "kv_elem"):
        with pytest.raises(ModelingError):
            _dspark_block(empty, options, **{**draft, name: -1})
    plan = next(p for p in case.model.layer_plans if p.attention == "mla")
    with pytest.raises(ModelingError):
        _attention_page_bytes(case.model, plan, 128, 2.0, 0)

@lru_cache(None)
def fix7_boundary_witnesses():
    from mindie_sim.parallel_decode_cost import DecodeCostModel
    from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state as evaluate
    from mindie_sim.memory_breakdown import build_memory_ledger, memory_breakdown
    case = CASE()
    counts = dict(device=0, workload=0, modes=0)
    def reject(group, call):
        with pytest.raises(ModelingError):
            call()
        counts[group] += 1
    for collection in ("rates", "compute_peaks"):
        die = case.machine.dies[0]
        for key, _ in getattr(die, collection):
            for bad in (0.0, -1.0):
                if collection == "rates" and key == "link" and bad == 0:
                    continue
                entries = tuple((k, bad if k == key else v) for k, v in getattr(die, collection))
                broken = dc.replace(case, machine=dc.replace(case.machine, dies=(dc.replace(die, **{collection: entries}), *case.machine.dies[1:])))
                reject("device", lambda: DecodeCostModel.from_case(broken))
    # Bypass construction so these assertions still exercise consumer guards.
    empty_workload = dc.replace(case.workload)
    object.__setattr__(empty_workload, "requests", ())
    empty = dc.replace(case, workload=empty_workload)
    for call in (lambda: evaluate(empty), lambda: evaluate(empty, HybridKvStateOptions(sequence_tokens=128)),
                 lambda: build_memory_ledger(empty), lambda: memory_breakdown(empty)):
        reject("workload", call)
    for capture, mode in itertools.product(("none", "full", "piecewise"), ("r7-invalid", "")):
        disabled = dc.replace(case, runtime=dc.replace(case.runtime, graph_capture_mode=capture))
        for call in (build_memory_ledger, memory_breakdown):
            reject("modes", lambda: call(disabled, graph_pool_mode=mode))
    return counts


def test_fix7_record_and_prebranch_domains():
    from nr_s7_fix4_verify import fix7_witnesses
    result = fix7_witnesses()
    assert result["records"]["classes"] == 36
    assert result["boundaries"]["workload"] == 4 and result["boundaries"]["modes"] == 12
