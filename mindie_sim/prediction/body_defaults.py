"""Structural resource-program defaults for device-body probes.

[Role] Select resource pipelines and complete omitted helpers of the same caller.
[Invariants] Explicit software choices win. Selection uses tensor and resource
  geometry, never model labels or measured latency. Unsupported domains retain
  their existing analytic program and a visible selection reason.
[Timeouts & Locks] Bounded model metadata traversal; no I/O or locks.
[Side-Effect Policy] Returns a new immutable case and its selection record.
"""
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import replace
from math import ceil

from mindie_sim.schema.common import dtype_bytes


_GDN_CALLER = "vllm_ascend_gdn_decode"


def _gdn_source_options(binding):
    if binding == _GDN_CALLER:
        return {}
    if isinstance(binding, dict) and binding.get("implementation") == _GDN_CALLER:
        return binding.get("options", {})
    return None


def _inherit_gdn_store(binding, default=None, carried=None):
    """Inherit one helper of the same caller; explicit empty means DMA.

    Policy leaves inherit independently. Recurrence and other implementation
    options remain whole-entry choices, not recursively merged defaults.
    """
    from .implementation_selection import is_policy

    if is_policy(binding):
        prior = carried if is_policy(carried) else {}
        cases = []
        for case in binding["cases"]:
            previous = next((item["binding"] for item in prior.get("cases", ())
                             if item["domain"] == case["domain"]), None)
            cases.append({**case, "binding": _inherit_gdn_store(
                case["binding"], default, previous)})
        result = {**binding, "cases": cases}
        if "otherwise" in binding:
            result["otherwise"] = _inherit_gdn_store(
                binding["otherwise"], default, prior.get("otherwise"))
        return result
    options = _gdn_source_options(binding)
    if options is None or "gating_beta_store" in options:
        return binding
    prior = _gdn_source_options(carried)
    store = prior.get("gating_beta_store", default) if prior is not None else default
    if store is None:
        return binding
    declaration = {"implementation": binding} if isinstance(binding, str) else binding
    return {**declaration, "options": {**options, "gating_beta_store": deepcopy(store)}}


def _gdn_store_default(case, kernels):
    """Select only the declared BF16 store helper, independently of recurrence."""
    layers = [p for p in case.model.layer_plans if p.attention == "gdn"]
    if (not layers or case.model.activation_dtype != "bf16" or case.runtime.parallel.pcp != 1
            or any(p.recurrent_state_dtype != "fp32" for p in layers)):
        return None
    declared = kernels.get("families", {}).get("gdn_gating", {})
    outputs = declared.get("outputs", ()) if declared.get("kind") == "tiled_pointwise" else ()
    if len(outputs) == 2 and "queue" in outputs[1]:
        return dict(outputs[1])
    queues = [dict(dict(d.store_queues).get("scalar_store_issue", ()))
              for d in case.machine.dies]
    if (queues and queues[0].get("element_bytes") == 2
            and all(q == queues[0] for q in queues)
            and all("scalar_store_issue" in dict(d.rates) for d in case.machine.dies)):
        return {"bytes_per_element": 2, "resource": "scalar_store_issue", "from_hardware": True}
    return None


class TargetGraphSet(dict):
    """A graph mapping plus one shared effective software declaration."""

    def __init__(self, values, *, kernels, selection):
        self.effective_kernels = deepcopy(kernels)
        self.default_selection = deepcopy(selection)
        super().__init__((rank, TargetOperators._with_declaration(
            ops, self.effective_kernels, self.default_selection))
            for rank, ops in values.items())


class TargetOperators(tuple):
    """Tuple-compatible operator sequence with the same declaration carrier."""

    def __new__(cls, values, *, kernels, selection):
        return cls._with_declaration(values, deepcopy(kernels), deepcopy(selection))

    @classmethod
    def _with_declaration(cls, values, kernels, selection):
        """Rank extraction shares the parent declaration without per-rank copies."""
        result = super().__new__(cls, values)
        result.effective_kernels = kernels
        result.default_selection = selection
        return result

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return (self._with_declaration(result, self.effective_kernels, self.default_selection)
                if isinstance(key, slice) else result)

    def __getnewargs_ex__(self):
        return (tuple(self),), {"kernels": self.effective_kernels,
                               "selection": self.default_selection}


def effective_body_declaration(case, phase, *, tokens=1, scope="body", enabled=True):
    """Resolve one body declaration before either structure or timing lowers it."""
    if scope != "body" or not enabled:
        return case, {"profile": "explicit_declaration_preserved",
                      "reason": "model_scope" if scope != "body" else "body_defaults_disabled"}
    case, selection = attention_defaults(case, phase, tokens=tokens)
    programs = case.runtime.kernels.setdefault("programs", {})
    explicit = "matmul" in programs
    if not explicit:
        # Compact the regular K recurrence on the same physical Engine.
        # This is the lowest-priority matrix program: named/source recipes
        # and family choices still win through RecipeBook.select.
        # Compression must not enable the opt-in mapping search. Keep the
        # original 128x128 output-owner grid. K64 is a capacity alternative
        # when two FP32 K128 operand buffers do not fit a declared L0 pool.
        programs["matmul"] = {"kind": "steady_matmul",
                              "candidates": [[128, 128, 128], [128, 128, 64]]}
        # Grouped matrices already had a mapped Steady fallback before the
        # body default. Do not narrow their independent expert-bank domain.
        families = case.runtime.kernels.setdefault("families", {})
        grouped_explicit = "grouped_gemm" in families
        if not grouped_explicit:
            families["grouped_gemm"] = {"kind": "steady_matmul"}
        selection["selections"].append({
            "family": "grouped_gemm",
            "selection": "preserved" if grouped_explicit else "default_installed",
            "reason": "explicit_implementation_preserved" if grouped_explicit
                      else "preserve_existing_grouped_mapping_domain",
            "program": deepcopy(families["grouped_gemm"]),
            "priority": "family_default; named_and_source_recipes_override",
        })
    selection["selections"].append({
        "family": "matmul", "selection": "preserved" if explicit else "default_installed",
        "reason": "explicit_implementation_preserved" if explicit
                  else "compact_finite_k_pipeline_preserving_128_output_grid",
        "program": deepcopy(programs["matmul"]),
        "priority": "program_default; named_source_and_family_recipes_override",
        "contention": "coupled_fluid_approximation_with_finite_owner_buffers",
    })
    return case, selection


def consume_target_declaration(case, target):
    """Keep caller hardware/timing; caller kernel entries override carried defaults.

    A complete recipe entry replaces the corresponding entry. The same GDN
    caller retains its carried store helper when that field was omitted.
    Recursively merging an envelope override into a tiled recipe would leave
    obsolete tile fields and change the intended implementation.
    """
    carried = getattr(target, "effective_kernels", None)
    if carried is None and isinstance(target, Mapping):
        declarations = [getattr(items, "effective_kernels", None)
                        for items in target.values()]
        supplied = [value for value in declarations if value is not None]
        if supplied:
            if len(supplied) != len(declarations) or any(value != supplied[0] for value in supplied):
                raise ValueError("body_target_kernel_declarations_differ_between_ranks")
            carried = supplied[0]
    if carried is None:
        return case
    kernels = deepcopy(carried)
    for key, value in case.runtime.kernels.items():
        if key in {"recipes", "programs", "families", "lowerings"}:
            entries = deepcopy(value)
            if key == "lowerings" and "gdn_decode" in entries:
                entries["gdn_decode"] = _inherit_gdn_store(
                    entries["gdn_decode"], carried=kernels.get(key, {}).get("gdn_decode"))
            kernels[key] = {**kernels.get(key, {}), **entries}
        else:
            kernels[key] = deepcopy(value)
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


def _gqa_tiles(case, layers):
    """Select finite tiles whose simultaneously live buffers fit every owner.

    K and V reuse the same tile lifetime after QK, while query/online state
    remain resident. Two window slots may each retain probabilities. This is
    a capacity policy, not a prediction of the vendor's tiling dispatch.
    """
    qb, kb = dtype_bytes(case.model.activation_dtype), dtype_bytes(case.model.quant.kv_dtype)
    for query in (64, 32, 16, 8, 4, 2, 1):
        for key in (128, 64, 32, 16, 8, 4, 2, 1):
            fits = True
            for layer in layers:
                dim = layer.head_dim
                sizes = {
                    "l1": query * dim * qb + 2 * key * dim * kb,
                    "l0a": query * dim * qb + 2 * query * key * qb,
                    "l0b": 2 * key * dim * kb,
                    "l0c": 2 * query * max(key, dim) * 4,
                    "ub": query * (dim * 4 + 8) + 2 * query * key * 4,
                }
                for die in case.machine.dies:
                    capacities = dict(die.buffers)
                    if any(size * ceil(die.compute_units / die.pool_units(pool))
                           > capacities[pool] for pool, size in sizes.items()
                           if pool in capacities):
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                return {"kind": "staged_gqa_decode", "query_tile": query,
                        "key_tile": key, "slots": 2}
    return None


def _gdn_default(case, phase, tokens, kernels):
    layers = [p for p in case.model.layer_plans if p.attention == "gdn"]
    if not layers:
        return None
    parallel = case.runtime.parallel
    recipes = kernels.get("recipes", {})
    # The analytic caller emits one named recurrence per request. Replacing
    # that caller would remove an explicitly configured recurrent_scan0, etc.
    per_request_scan = any(name.startswith("recurrent_scan")
                           and name[len("recurrent_scan"):].isdigit() for name in recipes)
    explicit = ("gdn_decode" in kernels.get("lowerings", {}) or
                bool({"gdn_gating", "short_convolution", "recurrent_vector"}
                     & kernels.get("families", {}).keys()) or
                bool({"delta_gates", "short_convolution", "qk_l2norm", "recurrent_scan"}
                     & recipes.keys()) or per_request_scan or
                "recurrent_vector" in kernels.get("programs", {}))
    pipes = {"hbm", "vector", "mte2_aic", "mte2_aiv", "mte3"}
    reason = ("explicit_implementation_preserved" if explicit else
              "decode_profile_only" if phase != "decode" else
              "partitioned_gdn_caller_domain" if parallel.pcp != 1 else
              "gdn_requires_bf16_inputs_fp32_state" if case.model.activation_dtype != "bf16"
              or any(p.recurrent_state_dtype != "fp32" for p in layers) else
              "gdn_independent_copy_pipes_not_declared" if any(
                  not pipes <= dict(d.rates).keys() for d in case.machine.dies) else
              "gdn_convolution_buffers_outside_declared_capacity" if any(
                  dict(d.buffers).get("ub", 0) < 188416 for d in case.machine.dies) else "")
    if not reason:
        for plan in layers:
            keys = ceil((plan.linear_key_heads or plan.linear_heads) / parallel.tp)
            heads = ceil((plan.linear_value_heads or plan.linear_heads) / parallel.tp)
            if (heads % keys or plan.conv_kernel not in {2, 3, 4}
                    or tokens > 1 and plan.conv_kernel != 4):
                reason = "gdn_head_ratio_or_convolution_outside_source_domain"
                break
    options = {}
    recurrence = "analytical; no reviewed source recurrence for this hardware/domain"
    store = "DMA resource estimate; no compatible physical store queue declared"
    if not reason:
        # arch35's resident-state kernel has a stricter UB/query domain than
        # the shared caller. Keep the caller useful outside that kernel domain.
        from .gdn_recurrent import UB_BYTES
        native = (case.machine.arch_family == "A5" and tokens <= 8
                  and all(dict(d.buffers).get("ub", 0) >= UB_BYTES for d in case.machine.dies)
                  and all(1 <= (p.linear_key_head_dim or p.linear_head_dim) <= 512
                          and 1 <= p.linear_head_dim <= 512
                          and 1 <= ceil((p.linear_value_heads or p.linear_heads) / parallel.tp) <= 256
                          and tokens * ceil(max(p.linear_key_head_dim or p.linear_head_dim,
                                                p.linear_head_dim) / 16) * 16 * 4 <= 16384
                          for p in layers))
        if native:
            options["recurrent_implementation"] = "arch35_fp32"
            recurrence = "arch35_fp32; source-compatible resource program, not captured dispatch"
        kernels.setdefault("lowerings", {})["gdn_decode"] = {
            "implementation": "vllm_ascend_gdn_decode", "options": options}
    lowerings = kernels.get("lowerings", {})
    binding = lowerings.get("gdn_decode")
    resolved = _inherit_gdn_store(
        binding, _gdn_store_default(case, kernels) if phase == "decode" else None)
    helper_default = resolved != binding
    if helper_default:
        lowerings["gdn_decode"] = resolved
        store = "declared BF16 scalar store service and finite queue"
    return {"family": "gdn_decode", "selection": "preserved" if reason else "default_installed",
            "reason": reason or "supported_tensor_and_resource_geometry",
            "caller": "non_interleaved_qkvz_ba",
            "recurrence": "explicit_implementation_preserved" if explicit else recurrence,
            "gating_store": ("explicit_implementation_preserved" if explicit and not helper_default
                             else store),
            **({"helper_defaults": ["gating_beta_store"]} if explicit and helper_default else {})}


def attention_defaults(case, phase, *, tokens=1):
    kernels = deepcopy(case.runtime.kernels)
    records = []
    required = {"mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", "vector", "hbm"}
    pipes = all(required <= dict(d.rates).keys() for d in case.machine.dies)
    directional = all({"mte1_l0a", "mte1_l0b"} <= dict(d.service_modes).keys()
                      for d in case.machine.dies)
    parallel = case.runtime.parallel
    floating = {"bf16", "fp16", "fp32", "bfloat16", "float16", "float32"}
    gqa = [p for p in case.model.layer_plans if p.attention == "gqa"]
    indexed = [p for p in case.model.layer_plans if p.attention == "mla" and p.index_topk]
    recipes, programs = kernels.setdefault("recipes", {}), kernels.setdefault("programs", {})
    lowerings = kernels.get("lowerings", {})
    for family, layers in (("gqa_decode", gqa), ("indexed_mla", indexed)):
        if not layers:
            continue
        explicit = ("attention" in kernels.get("families", {}) or
                    (family == "gqa_decode" and ("gqa_attention" in lowerings or
                     any(k in recipes for k in ("flash_attention", "flash_attention_decode")))) or
                    (family == "indexed_mla" and ("indexed_attention" in lowerings or
                     "indexed_attention" in programs)))
        reason = ("explicit_implementation_preserved" if explicit else
                  "decode_profile_only" if phase != "decode" else
                  "independent_copy_pipes_not_declared" if not pipes else
                  "partitioned_attention_domain" if parallel.cp != 1 or parallel.pcp != 1 else "")
        if not reason and family == "gqa_decode":
            if (case.model.activation_dtype not in floating or case.model.quant.kv_dtype not in floating
                    or any(p.window or p.noncausal_attention for p in layers)):
                reason = "windowed_noncausal_or_quantized_kv_domain"
            else:
                tiles = _gqa_tiles(case, layers)
                if tiles is None:
                    reason = "gqa_resident_buffers_outside_declared_capacity"
                else:
                    recipes["flash_attention_decode"] = tiles
        elif not reason:
            if (case.model.activation_dtype != "bf16" or case.model.quant.kv_dtype not in {"bf16", "int8"}
                    or any(p.cache_kind != "dense" or p.window or p.noncausal_attention
                           or (p.kv_lora_rank, p.qk_rope_head_dim) not in {(512, 64), (512, 0)}
                           or not 1 <= p.num_heads // parallel.tp <= 16 for p in layers)):
                reason = "indexed_source_geometry_outside_reviewed_domain"
            else:
                packed = case.model.quant.kv_dtype == "int8"
                if packed:
                    from .indexed_quant_attention import VECTOR_BUFFER_BYTES
                    if any(d.pool_units("ub") != d.compute_units * 2
                           or dict(d.buffers).get("ub", 0) < VECTOR_BUFFER_BYTES
                           or any(d.units(role) != d.compute_units * ratio for role, ratio in (
                               ("vector", 2), ("mte2_aiv", 2), ("mte3", 2),
                               ("mte2_aic", 1), ("mte1", 1), ("fixpipe", 1)))
                           for d in case.machine.dies):
                        reason = "packed_indexed_scratch_or_paired_pipe_domain"
                if reason:
                    records.append({"family": family, "selection": "preserved", "reason": reason})
                    continue
                programs["indexed_attention"] = {
                    "kind": "indexed_attention_packed_int8" if packed else "indexed_attention_pipeline",
                    "scratch_policy": "offchip",
                    "operand_modes": directional,
                }
                if not packed and any(p.qk_rope_head_dim == 0 for p in layers):
                    programs["indexed_attention"]["rope0_analytic"] = True
        records.append({"family": family, "selection": "preserved" if reason else "default_installed",
                        "reason": reason or "supported_tensor_and_resource_geometry"})
    gdn = _gdn_default(case, phase, tokens, kernels)
    if gdn is not None:
        records.append(gdn)
    model = case.model
    if programs.get("indexed_attention", {}).get("kind") == "indexed_attention_packed_int8":
        from mindie_sim.catalog.cache_format import with_packed_mla_cache
        widths = {p.qk_rope_head_dim for p in indexed}
        if len(widths) == 1:
            model = with_packed_mla_cache(model, next(iter(widths)))
    return replace(case, model=model, runtime=replace(case.runtime, kernels=kernels)), {
        "profile": "decode_resource_pipelines", "selections": records,
        "basis": "analytic resource programs; no measured dispatch or latency fit",
        "gqa_kv_splits": "default 1; explicit flash_attention_decode recipe may override",
        "gqa_tiles": recipes.get("flash_attention_decode"),
        "selection_boundary": "family default; per-operator recipes retain priority",
        "indexed_scratch": "offchip; no inferred cache hit rate",
        "indexed_operand_service": "directional_mte1" if directional else "generic_mte1",
        "sharedkv": "SCFA source currently requires explicit prefill declarations",
    }
