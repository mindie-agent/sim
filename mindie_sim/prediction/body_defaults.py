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

from mindie_sim.schema.common import ModelingError, dtype_bytes

_GDN_CALLER = "vllm_ascend_gdn_decode"
# Dense paged-MLA source domains. A5 keeps source split-core v2 for every
# integral local head count; A2/A3 use the page-tile kernel.
MAX_PAGED_MLA_HEADS = 64
PAGED_MLA_A5_PAGE_TILE = 128
ARCH22_PAGED_MLA_PAGE_TILE = 512
# The arch22 split-core tile rounds local heads to these power-of-two ceilings.
ARCH22_PAGED_MLA_HEAD_CEILINGS = frozenset({8, 16})
# Declared absorbed-MLA cache head geometry (latent, rope) of both families.
PAGED_MLA_HEAD_GEOMETRY = (512, 64)
PAGED_MLA_ARCH_FAMILIES = frozenset({"A2", "A3", "A5"})
# Declared domain of the KDA recurrent source: decode-only, at most this many
# query rows per call, and this much UB for its resident state tiles.
KDA_SOURCE_MAX_QUERY_TOKENS = 8
KDA_SOURCE_MIN_UB_BYTES = 65536
MQSMLA_FAMILY = "mqsmla_decode"
MQSMLA_NATIVE_KIND = "aclnnMixedQuantSparseFlashMla"
LIGHTNING_INDEXER_FAMILY = "lightning_indexer"
LIGHTNING_INDEXER_IMPLEMENTATION = "cann_bf16_lightning_indexer_v2_complete"
BF16_INDEX_CACHE_DTYPES = frozenset({"bf16", "bfloat16"})
INDEX_PROBE_ROWS = 1
ARCH22_FAI_TARGET_CORES = 20
ARCH35_FAI_TARGET_CORES = 32
FAI_MAX_HOST_CORES = 64
FAI_MAX_QUERY_TOKENS = 16
FAI_HOST_KV_TILE = 512
FAI_MAX_QUERY_GROUP_ROWS = 128
GQA_QUERY_TILE_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
GQA_KEY_TILE_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)
GQA_SCORE_ACCUM_BYTES = 4
GQA_ONLINE_STATS_BYTES = 8

_GDN_CONV_MIN_UB_BYTES = 188416
_GDN_CONV_WIDTHS = (2, 3, 4)
_GDN_MULTI_TOKEN_CONV_WIDTH = _GDN_CONV_WIDTHS[-1]
_GDN_RESIDENT_MAX_TOKENS = 8
_GDN_RESIDENT_MAX_HEAD_DIM = 512
_GDN_RESIDENT_MAX_HEADS = 256
_GDN_STATE_TILE_ALIGNMENT = 16
_GDN_STATE_TILE_BYTES = _GDN_STATE_TILE_ALIGNMENT * 4
_GDN_QUERY_TILE_BYTES = 16384
_KDA_SOURCE_MAX_HEADS = 256
_KDA_SOURCE_KEY_DIM = 128
_KDA_SOURCE_WIDE_VALUE_DIM = 256
_ATTNRES_UB_RESERVE_BYTES = 32768
_INDEXED_MAX_HEADS = {"int8": 64, "bf16": 128}
_INDEXED_MLA_CACHE_GEOMETRIES = ((512, 64), (512, 0))


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
    if (not layers or case.model.activation_dtype != "bf16"
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


def _collective_defaults(case):
    """Consume adapter-declared protocols without overriding caller recipes."""
    protocols = dict(case.machine.collective_protocols)
    mode = case.runtime.collective_startup_mode
    kernels = deepcopy(case.runtime.kernels)
    families = kernels.setdefault("families", {})
    for family in ("alltoall", "alltoallv"):
        if protocols.get(family) == "concurrent" and mode in {"topology", "concurrent"}:
            families.setdefault(family, {"kind": "direct_peer_collective"})
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


DEFAULT_SCOPE = "body"


class TargetGraphSet(dict):
    """A graph mapping plus one shared effective software declaration."""

    def __init__(self, values, *, kernels, selection, scope=DEFAULT_SCOPE):
        self.scope = str(scope)
        self.effective_kernels = deepcopy(kernels)
        self.default_selection = deepcopy(selection)
        super().__init__((rank, TargetOperators._with_declaration(
            ops, self.effective_kernels, self.default_selection, self.scope))
            for rank, ops in values.items())


class TargetOperators(tuple):
    """Tuple-compatible operator sequence with the same declaration carrier."""

    def __new__(cls, values, *, kernels, selection, scope=DEFAULT_SCOPE):
        return cls._with_declaration(values, deepcopy(kernels), deepcopy(selection), scope)

    @classmethod
    def _with_declaration(cls, values, kernels, selection, scope=DEFAULT_SCOPE):
        """Rank extraction shares the parent declaration without per-rank copies."""
        result = super().__new__(cls, values)
        result.scope = str(scope)
        result.effective_kernels = kernels
        result.default_selection = selection
        return result

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return (self._with_declaration(result, self.effective_kernels,
                                      self.default_selection, self.scope)
                if isinstance(key, slice) else result)

    def __getnewargs_ex__(self):
        return (tuple(self),), {"kernels": self.effective_kernels,
                               "selection": self.default_selection,
                               "scope": self.scope}


def execution_policy(case) -> dict:
    """Resolved execution policy stored by ``compose_case`` on the declaration."""
    runtime = getattr(case, "runtime", None)
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    policy = metadata.get("execution_policy")
    return dict(policy) if isinstance(policy, Mapping) else {}


def _strip_automatic_lowerings(case, declared):
    """Remove lowerings that a body default installed beyond the caller's set."""
    kernels = dict(case.runtime.kernels)
    lowerings = kernels.get("lowerings")
    if isinstance(lowerings, Mapping):
        kernels["lowerings"] = {key: value for key, value in lowerings.items()
                                if key in declared}
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


def effective_body_declaration(case, phase, *, tokens=1, scope=DEFAULT_SCOPE,
                              enabled=True):
    """Resolve one effective declaration before either structure or timing lowers it.

    Assembly selection (``scope``) and effective implementation selection are
    separate concerns: both body assembly and model assembly consume the same
    resolved service-program and attention defaults. The resolved scope is
    carried on the returned graph set instead of being re-labelled ``body``.
    """
    if not enabled:
        return case, {"profile": "explicit_declaration_preserved",
                      "scope": str(scope), "reason": "body_defaults_disabled"}
    if str(getattr(case.machine, "arch_family", "")) == "A5":
        # The INT8 packed MLA row is resolved before phase-scoped operator
        # selection so prefill, decode and the serving inventory agree.
        from mindie_sim.compose_operator_defaults import (
            resolve_packed_indexed_mla_cache,
        )
        case = replace(
            case, model=resolve_packed_indexed_mla_cache(case.model))
    case = _collective_defaults(case)
    policy = execution_policy(case)
    declared_lowerings = frozenset(dict(case.runtime.kernels.get("lowerings", {}) or {}))
    case, selection = attention_defaults(case, phase, tokens=tokens)
    if not bool(policy.get("automatic_optional_fusions", True)):
        # ``raw_eager`` (and a caller that disabled recommendations) must never
        # re-acquire an optional fused lowering from the structural defaults.
        case = _strip_automatic_lowerings(case, declared_lowerings)
        selection = {**selection,
                     "automatic_optional_fusions": "disabled_by_execution_policy",
                     "declared_lowerings": sorted(declared_lowerings)}
    programs = case.runtime.kernels.setdefault("programs", {})
    explicit = "matmul" in programs
    if not explicit:
        # Compact the regular K recurrence on the same physical Engine.
        # This is the lowest-priority matrix program: named/source recipes
        # and family choices still win through RecipeBook.select.
        # Compression must not enable the opt-in mapping search. Keep the
        # original 128x128 output-owner grid. K64 is a capacity alternative
        # when two FP32 K128 operand buffers do not fit a declared L0 pool.
        programs["matmul"] = _steady_matmul_default(
            candidates=True, a5=case.machine.sku.startswith("Ascend950"))
        # Grouped matrices already had a mapped Steady fallback before the
        # body default. Do not narrow their independent expert-bank domain.
        families = case.runtime.kernels.setdefault("families", {})
        grouped_explicit = "grouped_gemm" in families
        if not grouped_explicit:
            families["grouped_gemm"] = _steady_matmul_default()
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


DEFAULT_MATMUL_CANDIDATES = (
    (256, 256, 64),
    (128, 256, 128),
    (256, 128, 128),
    (128, 256, 64),
    (128, 128, 128),
    (128, 128, 64),
    (64, 256, 128),
    (64, 128, 128),
    (32, 128, 256),
    (32, 128, 128),
    (32, 64, 256),
    (32, 32, 256),
    (16, 128, 256),
    (16, 64, 256),
)
LEGACY_MATMUL_CANDIDATES = ((128, 128, 128), (128, 128, 64))


def _steady_matmul_default(*, candidates=False, a5=False):
    """The compact finite-K matrix program, including symmetric owner cohorts.

    Candidate tiles are declared architecture policy, not a dense-only
    128-output narrowing. The mapping still screens L1/L0A/L0B/L0C capacity.
    """
    program = {"kind": "steady_matmul", "owner_cohorts": True}
    if candidates:
        tiles = DEFAULT_MATMUL_CANDIDATES if a5 else LEGACY_MATMUL_CANDIDATES
        program["candidates"] = [list(item) for item in tiles]
    return program


def service_program_defaults(case):
    """Install the omitted physical programs of one serving step.

    A serving step declares the same operators as the body entry; only the
    request scheduler, cache and delivery boundaries differ.  This supplies the
    same default matrix program/family so the step lowers each declared
    operator through the compact finite-K pipeline instead of the detailed
    per-tile reference.  It never selects attention, scheduling, cache,
    submission or service-regime policy, and an explicit caller declaration is
    preserved verbatim.  The case is returned unchanged (a new runtime dict is
    built) when an install is required, so the caller's input is not mutated.
    """
    case = _collective_defaults(case)
    kernels = case.runtime.kernels
    if "matmul" in (kernels.get("programs") or {}):
        return case
    programs = {**kernels.get("programs", {}),
                "matmul": _steady_matmul_default(
        candidates=True, a5=case.machine.sku.startswith("Ascend950"))}
    families = dict(kernels.get("families", {}))
    families.setdefault("grouped_gemm", _steady_matmul_default())
    return replace(case, runtime=replace(
        case.runtime, kernels={**kernels, "programs": programs,
                               "families": families}))


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
    policy = execution_policy(case)
    automatic = bool(policy.get("automatic_optional_fusions", True))
    declared = frozenset(dict(case.runtime.kernels.get("lowerings", {}) or {}))
    kernels = deepcopy(carried)
    for key, value in case.runtime.kernels.items():
        if key in {"recipes", "programs", "families", "lowerings"}:
            entries = deepcopy(value)
            if key == "lowerings":
                if not automatic:
                    # A raw-eager declaration never re-acquires an optional fused
                    # lowering from a carried target declaration.
                    kernels[key] = {name: item for name, item in
                                    kernels.get(key, {}).items()
                                    if name in declared}
                if "gdn_decode" in entries:
                    entries["gdn_decode"] = _inherit_gdn_store(
                        entries["gdn_decode"], carried=kernels.get(key, {}).get("gdn_decode"))
            kernels[key] = {**kernels.get(key, {}), **entries}
        else:
            kernels[key] = deepcopy(value)
    return replace(case, runtime=replace(case.runtime, kernels=kernels))


def _gqa_tiles(case, layers, tokens=1):
    """Select finite tiles whose simultaneously live buffers fit every owner.

    K and V reuse the same tile lifetime after QK, while query/online state
    remain resident. Two window slots may each retain probabilities. This is
    a capacity policy, not a prediction of the vendor's tiling dispatch.
    """
    qb, kb = dtype_bytes(case.model.activation_dtype), dtype_bytes(case.model.quant.kv_dtype)
    tp = max(case.runtime.parallel.tp, 1)
    cores = min((d.compute_units for d in case.machine.dies), default=0)
    is_a5 = case.machine.arch_family == "A5"
    target_cores = min(
        cores, ARCH35_FAI_TARGET_CORES if is_a5 else ARCH22_FAI_TARGET_CORES
    )
    for query in GQA_QUERY_TILE_CANDIDATES:
        for key in GQA_KEY_TILE_CANDIDATES:
            fits = True
            for layer in layers:
                dim = layer.head_dim
                sizes = {
                    "l1": query * dim * qb + 2 * key * dim * kb,
                    "l0a": query * dim * qb + 2 * query * key * qb,
                    "l0b": 2 * key * dim * kb,
                    "l0c": 2 * query * max(key, dim) * GQA_SCORE_ACCUM_BYTES,
                    "ub": (
                        query * (dim * GQA_SCORE_ACCUM_BYTES + GQA_ONLINE_STATS_BYTES)
                        + 2 * query * key * GQA_SCORE_ACCUM_BYTES
                    ),
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
                result = {"kind": "staged_gqa_decode", "query_tile": query,
                          "key_tile": key, "slots": 2}
                fai_valid = (
                    case.machine.arch_family in {"A2", "A3", "A5"}
                    and 1 <= target_cores <= FAI_MAX_HOST_CORES
                    and 0 < tokens <= FAI_MAX_QUERY_TOKENS
                    and key <= FAI_HOST_KV_TILE and FAI_HOST_KV_TILE % key == 0
                    and all(
                        layer.num_kv_heads >= tp
                        and (layer.num_heads // tp) % (layer.num_kv_heads // tp) == 0
                        and tokens * ((layer.num_heads // tp) // (layer.num_kv_heads // tp))
                            <= min(FAI_MAX_QUERY_GROUP_ROWS, query)
                        and min(max(1, (FAI_MAX_QUERY_GROUP_ROWS // tokens) // 2 * 2),
                                (layer.num_heads // tp) // (layer.num_kv_heads // tp))
                            == (layer.num_heads // tp) // (layer.num_kv_heads // tp)
                        for layer in layers
                    )
                )
                if fai_valid:
                    result["target_cores"] = target_cores
                    result["kv_mapping"] = "fai_host_arch35" if is_a5 else "fai_host"
                return result
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
              "partitioned_gdn_caller_domain" if phase != "decode" and parallel.pcp != 1 else
              "decode_profile_only" if phase != "decode" else
              "gdn_requires_bf16_inputs_fp32_state" if case.model.activation_dtype != "bf16"
              or any(p.recurrent_state_dtype != "fp32" for p in layers) else
              "gdn_independent_copy_pipes_not_declared" if any(
                  not pipes <= dict(d.rates).keys() for d in case.machine.dies) else
              "gdn_convolution_buffers_outside_declared_capacity" if any(
                  dict(d.buffers).get("ub", 0) < _GDN_CONV_MIN_UB_BYTES for d in case.machine.dies) else "")
    if not reason:
        for plan in layers:
            keys = ceil((plan.linear_key_heads or plan.linear_heads) / parallel.tp)
            heads = ceil((plan.linear_value_heads or plan.linear_heads) / parallel.tp)
            if (heads % keys or plan.conv_kernel not in _GDN_CONV_WIDTHS
                    or tokens > 1 and plan.conv_kernel != _GDN_MULTI_TOKEN_CONV_WIDTH):
                reason = "gdn_head_ratio_or_convolution_outside_source_domain"
                break
    options = {}
    recurrence = "analytical; no reviewed source recurrence for this hardware/domain"
    store = "DMA resource estimate; no compatible physical store queue declared"
    if not reason:
        # arch35's resident-state kernel has a stricter UB/query domain than
        # the shared caller. Keep the caller useful outside that kernel domain.
        from .operators.linear_attention.gdn_recurrent import UB_BYTES
        native = (case.machine.arch_family == "A5" and tokens <= _GDN_RESIDENT_MAX_TOKENS
                  and all(dict(d.buffers).get("ub", 0) >= UB_BYTES for d in case.machine.dies)
                  and all(1 <= (p.linear_key_head_dim or p.linear_head_dim) <= _GDN_RESIDENT_MAX_HEAD_DIM
                          and 1 <= p.linear_head_dim <= _GDN_RESIDENT_MAX_HEAD_DIM
                          and 1 <= ceil((p.linear_value_heads or p.linear_heads) / parallel.tp) <= _GDN_RESIDENT_MAX_HEADS
                          and tokens * ceil(max(p.linear_key_head_dim or p.linear_head_dim,
                                                p.linear_head_dim) / _GDN_STATE_TILE_ALIGNMENT)
                          * _GDN_STATE_TILE_BYTES <= _GDN_QUERY_TILE_BYTES
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


def _kda_default(case, phase, tokens, kernels):
    layers = [p for p in case.model.layer_plans if p.attention == "kda"]
    if not layers:
        return None
    parallel = case.runtime.parallel
    recipes = kernels.get("recipes", {})
    per_request_scan = any(name.startswith("recurrent_scan")
                           and name[len("recurrent_scan"):].isdigit() for name in recipes)
    raw_bf16 = (case.model.quant.weight_format == "bf16"
                and "moe_dispatch_quantization" not in dict(case.runtime.metadata))
    # A5 Kimi K3 decode is an actual native KDA source call even when the
    # frontend declaration is raw BF16.  Keep raw BF16 as an explicit-generic
    # guard for non-A5 / foreign KDA domains only.
    native_a5 = case.machine.arch_family == "A5"
    raw_bf16_generic = raw_bf16 and not native_a5
    explicit = ("kda" in kernels.get("lowerings", {}) or
                bool({"short_convolution", "recurrent_vector", "recurrent_scan"}
                     & kernels.get("families", {}).keys()) or
                bool({"delta_gates", "short_convolution", "qk_l2norm", "recurrent_scan"}
                     & recipes.keys()) or per_request_scan or
                "recurrent_vector" in kernels.get("programs", {}) or
                bool(case.runtime.tensor_layouts) or raw_bf16_generic)
    from mindie_sim.catalog.paired_sfa import execution
    recurrent_sharded = execution(case.model).get("recurrent_context_sharded") is not False
    pipes = {"hbm", "vector", "mte2_aiv", "mte3"}
    reason = ("explicit_implementation_preserved" if explicit else
              # The source recurrence is a decode-only, single-context domain.
              "kda_source_requires_decode_only_pcp1" if (
                  parallel.pcp != 1 or parallel.cp != 1 and recurrent_sharded) else
              "partitioned_kda_caller_domain" if phase != "decode" and parallel.pcp != 1 else
              "decode_profile_only" if phase != "decode" or not 1 <= tokens <= KDA_SOURCE_MAX_QUERY_TOKENS else
              "kda_requires_bf16_inputs_and_state" if case.model.activation_dtype != "bf16"
              or any(p.recurrent_state_dtype not in {"bf16", "fp32"} for p in layers) else
              "kda_independent_copy_pipes_not_declared" if any(
                  not pipes <= dict(d.rates).keys()
                  or dict(d.buffers).get("ub", 0) < KDA_SOURCE_MIN_UB_BYTES
                  for d in case.machine.dies) else "")
    if not reason:
        for plan in layers:
            keys = ceil((plan.linear_key_heads or plan.linear_heads) / parallel.tp)
            heads = ceil((plan.linear_value_heads or plan.linear_heads) / parallel.tp)
            dk = plan.linear_key_head_dim or plan.linear_head_dim
            dv = plan.linear_head_dim
            if (not 1 <= keys <= heads <= _KDA_SOURCE_MAX_HEADS or heads % keys != 0
                    or dk != _KDA_SOURCE_KEY_DIM or dv not in {_KDA_SOURCE_KEY_DIM, _KDA_SOURCE_WIDE_VALUE_DIM}):
                reason = "kda_head_geometry_outside_source_domain"
                break
    if not reason:
        impl = ("vllm_ascend_recurrent_kda_arch35" if case.machine.arch_family == "A5"
                else "vllm_ascend_recurrent_kda_arch22")
        options = {}
        if case.machine.arch_family == "A5":
            from mindie_sim.prediction.operators.kda.kda_source import resolve as resolve_kda_source
            bounds = {getattr(p, "kda_gate_lower_bound", None) for p in layers}
            if len(bounds) != 1:
                reason = "kda_gate_lower_bound_not_uniform_across_layers"
            elif phase != "decode":
                reason = "kda_source_decode_only"
            else:
                options = resolve_kda_source(
                    layers[0], arch=case.machine.arch_family, phase=phase,
                    kernels=kernels, activation_dtype=case.model.activation_dtype)
                if options is None:
                    reason = "kda_source_explicit_override_or_gate_invalid"
        if not reason:
            kernels.setdefault("lowerings", {})["kda"] = {
                "implementation": impl, "options": options}
    return {"family": "kda_decode", "selection": "preserved" if reason else "default_installed",
            "reason": reason or "supported_tensor_and_resource_geometry"}


def _attnres_default(case, phase, kernels):
    block = next((p.residual_block_size for p in case.model.layer_plans
                  if p.residual_block_size), 0)
    if block < 1:
        return None
    # Fusing the block would delete these independently declared member
    # programs, so an explicit residual recipe/family keeps the unfused
    # boundary.  Generic matmul/service declarations are not residual choices.
    from mindie_sim.prediction.operators.hyper_connections import family_attnres
    recipes = {"attn_res_fwd", "residual_values_fp32", "rmsnorm",
               "attention_residual_scores", "attention_residual_softmax",
               "attention_residual_mix", "attention_residual_output_cast"}
    families = {"gemm", "rmsnorm", "softmax", "elementwise", "cast"}
    explicit = ("attention_residual" in kernels.get("lowerings", {}) or
                bool(recipes & kernels.get("recipes", {}).keys()) or
                bool(families & kernels.get("families", {}).keys()))
    source_domain = (family_attnres.source_eligible(
        case.model, case.machine.arch_family, phase)
        and family_attnres.source_image_match(case.runtime))
    pipes = {"hbm", "vector", "mte2_aiv", "mte3"}
    reason = ("explicit_implementation_preserved" if explicit else
              "attnres_source_image_or_domain_unsupported" if not source_domain else
              "attnres_independent_copy_pipes_not_declared" if any(
                  not pipes <= dict(d.rates).keys() or dict(d.buffers).get("ub", 0) <= _ATTNRES_UB_RESERVE_BYTES
                  for d in case.machine.dies) else "")
    if not reason:
        kernels.setdefault("lowerings", {})["attention_residual"] = "vllm_ascend_attn_res_prefix"
    return {"family": "attention_residual", "selection": "preserved" if reason else "default_installed",
            "reason": reason or "supported_tensor_and_resource_geometry"}


def attention_defaults(case, phase, *, tokens=1):
    prior_kernels = case.runtime.kernels or {}
    prior_keys = {
        section: set((prior_kernels.get(section) or {}).keys())
        for section in ("recipes", "families", "programs", "lowerings")
    }
    kernels = deepcopy(case.runtime.kernels)
    records = []
    required = {"mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", "vector", "hbm"}
    pipes = all(required <= dict(d.rates).keys() for d in case.machine.dies)
    directional = all({"mte1_l0a", "mte1_l0b"} <= dict(d.service_modes).keys()
                      for d in case.machine.dies)
    parallel = case.runtime.parallel
    floating = {"bf16", "fp16", "fp32", "bfloat16", "float16", "float32"}
    quantized_kv = {"int8", "fp8", "fp8_e4m3", "fp4", "fp4_e2m1"}
    gqa = [p for p in case.model.layer_plans if p.attention == "gqa"]
    indexed = [p for p in case.model.layer_plans if p.attention == "mla" and p.index_topk]
    dense_mla = [p for p in case.model.layer_plans if p.attention == "mla" and not p.index_topk
                 and p.mla_variant == "latent_mla" and p.cache_kind == "dense"]
    recipes, programs = kernels.setdefault("recipes", {}), kernels.setdefault("programs", {})
    lowerings = kernels.setdefault("lowerings", {})
    for family, layers in (("gqa_decode", gqa), ("indexed_mla", indexed)):
        if not layers:
            continue
        explicit = ("attention" in kernels.get("families", {}) or
                    (family == "gqa_decode" and ("gqa_attention" in lowerings or
                     any(k in recipes for k in ("flash_attention", "flash_attention_decode")))) or
                    (family == "indexed_mla" and ("indexed_attention" in lowerings or
                     "indexed_attention" in programs)))
        reason = ("explicit_implementation_preserved" if explicit else
                  "paired_sfa_allocator_contract" if family == "indexed_mla"
                  and dict(case.model.metadata).get("cache_allocator") == "paired_sfa" else
                  "partitioned_attention_domain" if phase != "decode" and parallel.pcp != 1 else
                  "decode_profile_only" if phase != "decode" else
                  "independent_copy_pipes_not_declared" if not pipes else
                  "partitioned_attention_domain" if parallel.cp != 1 else "")
        if not reason and family == "gqa_decode":
            if case.model.activation_dtype not in floating:
                reason = "gqa_activation_precision_outside_fattention_domain"
            elif (case.model.quant.kv_dtype not in floating
                  and case.model.quant.kv_dtype not in quantized_kv):
                reason = "gqa_kv_storage_outside_fattention_domain"
            elif any(p.window or p.index_topk or p.noncausal_attention for p in layers):
                # The declared staged decode source rejects any non-zero
                # window, as well as indexed and non-causal attention, so those
                # layers keep analytical execution with this explicit reason.
                reason = "windowed_noncausal_gqa_domain"
            else:
                tiles = _gqa_tiles(case, layers, tokens)
                if tiles is None:
                    reason = "gqa_resident_buffers_outside_declared_capacity"
                else:
                    recipes["flash_attention_decode"] = tiles
        elif not reason:
            max_heads = _INDEXED_MAX_HEADS.get(case.model.quant.kv_dtype, _INDEXED_MAX_HEADS["bf16"])
            if (case.model.activation_dtype != "bf16" or case.model.quant.kv_dtype not in {"bf16", "int8"}
                    or any(p.cache_kind != "dense" or p.window or p.noncausal_attention
                           or (p.kv_lora_rank, p.qk_rope_head_dim) not in _INDEXED_MLA_CACHE_GEOMETRIES
                           or not 1 <= p.num_heads // parallel.tp <= max_heads for p in layers)):
                reason = "indexed_source_geometry_outside_reviewed_domain"
            else:
                packed = case.model.quant.kv_dtype == "int8"
                if packed:
                    from mindie_sim.prediction.operators.paged_mla.indexed_quant_attention import VECTOR_BUFFER_BYTES
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
    if dense_mla:
        from mindie_sim.prediction.operators.paged_mla import paged_mla, paged_mla_membase
        explicit = ("attention" in kernels.get("families", {}) or "mla_attention" in recipes
                    or any(k in lowerings for k in ("paged_mla_attention", "mla_decode")))
        arch_family = case.machine.arch_family
        is_a5 = arch_family == "A5"
        # A5 keeps the source split-core v2 kernel for every integral local head
        # count. A2/A3 use the page-tile kernel whose split-core domain is the
        # power-of-two ceiling {8, 16}.
        valid_heads = ((lambda h: 1 <= h <= MAX_PAGED_MLA_HEADS)
                       if is_a5 else (lambda h: h > 0 and (1 << (h - 1).bit_length())
                                      in ARCH22_PAGED_MLA_HEAD_CEILINGS))
        page_tile = PAGED_MLA_A5_PAGE_TILE if is_a5 else ARCH22_PAGED_MLA_PAGE_TILE
        req_buffers = paged_mla.BUFFERS if is_a5 else paged_mla_membase.BUFFERS
        paired_pipes = all(
            all(dict(d.buffers).get(pool, 0) >= needed for pool, needed in req_buffers.items())
            and all(d.units(role) == d.compute_units * ratio for role, ratio in (
                ("mte1", 1), ("mte2_aic", 1), ("fixpipe", 1),
                ("vector", 2), ("mte2_aiv", 2), ("mte3", 2)))
            for d in case.machine.dies
        )
        reason = ("explicit_implementation_preserved" if explicit else
                  "partitioned_attention_domain" if phase != "decode" and parallel.pcp != 1 else
                  "decode_profile_only" if phase != "decode" else
                  "independent_copy_pipes_not_declared" if not (pipes and paired_pipes) else
                  "partitioned_attention_domain" if parallel.cp != 1 else
                  "paged_mla_source_geometry_outside_reviewed_domain" if (
                      arch_family not in PAGED_MLA_ARCH_FAMILIES
                      or tokens != 1 or case.model.activation_dtype != "bf16"
                      or case.model.quant.kv_dtype != "bf16"
                      or case.runtime.kv_block_size < 1 or page_tile % case.runtime.kv_block_size != 0
                      or any(p.window or p.noncausal_attention
                             or (p.kv_lora_rank, p.qk_rope_head_dim) != PAGED_MLA_HEAD_GEOMETRY
                             or not valid_heads(p.num_heads // max(parallel.tp, 1))
                             for p in dense_mla)) else "")
        if not reason:
            domain = {"phases": ["decode"],
                      "max_query_tokens": {"min": 1, "max": 1},
                      "sequences": {"min": 1}}
            if is_a5:
                binding = {"implementation": "ascend_paged_mla_decode_arch35",
                           "options": {"dispatch": "source_split_core_v2"}}
            else:
                # The arch22 split-core tile needs a declared page-aligned
                # context; a shorter context stays analytical instead of
                # becoming a planner exception.
                domain["min_context_tokens"] = {"min": ARCH22_PAGED_MLA_PAGE_TILE}
                binding = {"implementation": "ascend_paged_mla_decode_arch22"}
            lowerings["paged_mla_attention"] = {
                "cases": [{"domain": domain, "binding": binding}],
                "otherwise": "analytical",
            }
        records.append({"family": "paged_mla_decode",
                        "selection": "preserved" if reason else "default_installed",
                        "reason": reason or "supported_tensor_and_resource_geometry"})
    gdn = _gdn_default(case, phase, tokens, kernels)
    if gdn is not None:
        records.append(gdn)
    kda = _kda_default(case, phase, tokens, kernels)
    if kda is not None:
        records.append(kda)
    attnres = _attnres_default(case, phase, kernels)
    if attnres is not None:
        records.append(attnres)
    from mindie_sim.compose_operator_defaults import (
        lightning_indexer_default, mqsmla_default,
    )
    mixed_quant = mqsmla_default(case, kernels)
    if mixed_quant is not None:
        records.append(mixed_quant)
    lightning = lightning_indexer_default(case, kernels)
    if lightning is not None:
        records.append(lightning)
    model = case.model
    if programs.get("indexed_attention", {}).get("kind") == "indexed_attention_packed_int8":
        from mindie_sim.catalog.cache_format import with_packed_mla_cache
        widths = {p.qk_rope_head_dim for p in indexed}
        if len(widths) == 1:
            model = with_packed_mla_cache(model, next(iter(widths)))
    from .implementation_selection import (
        record_binding_origin,
        records_summary,
        with_automatic_lowering,
    )
    metadata = dict(case.runtime.metadata)
    for section, existing_names in prior_keys.items():
        current_names = set((kernels.get(section) or {}).keys())
        for added in sorted(current_names - existing_names):
            metadata = record_binding_origin(metadata, section, added, "body_default")
    if mixed_quant is not None and mixed_quant.get("installed"):
        metadata = with_automatic_lowering(metadata, "mqsmla_attention")
    return replace(case, model=model, runtime=replace(
        case.runtime, kernels=kernels, metadata=tuple(metadata.items()))), {
        "profile": "decode_resource_pipelines", "selections": records,
        "basis": "analytic resource programs; no measured dispatch or latency fit",
        "gqa_kv_splits": "default 1; explicit flash_attention_decode recipe may override",
        "gqa_tiles": recipes.get("flash_attention_decode"),
        "selection_boundary": "family default; per-operator recipes retain priority",
        "indexed_scratch": "offchip; no inferred cache hit rate",
        "indexed_operand_service": "directional_mte1" if directional else "generic_mte1",
        "sharedkv": "SCFA source currently requires explicit prefill declarations",
        "operator_selection_summary": records_summary(
            arch_family=case.machine.arch_family, phase=phase, records=records),
    }
