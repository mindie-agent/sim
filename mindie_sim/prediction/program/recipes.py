"""RecipeBook dispatch, decode contracts, and standard host/recurrent recipes.

[Role] Provide ``RecipeBook`` selection/preparation/expansion, ``GENERIC_MATRIX_PROGRAM``,
  ``HostStaging``, ``RecurrentScan``, and the staged decode physical contract builder.
[Invariants] Explicit caller recipes and programs take strict precedence over
  automatic physical defaults; ``select`` and ``expand`` caches are bounded.
[Timeouts & Locks] Selection and expansion caches are bounded by
  ``_SELECT_CACHE_LIMIT`` and ``_EXPAND_CACHE_LIMIT``; no I/O or locks.
[Side-Effect Policy] Pure recipe lookup and declaration assembly.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from .binding import Recipe, RecipeContext, TransferRecipe
from .stages import KernelProgram, Stage

GENERIC_MATRIX_PROGRAM = "matmul"
_SELECT_CACHE_LIMIT = 4096
_EXPAND_CACHE_LIMIT = 16384


class RecipeBook:
    """Explicit implementation bindings; adding recipes does not edit the loop."""

    def __init__(
        self,
        recipes: Mapping[str, Recipe] | None = None,
        *,
        families: Mapping[str, Recipe] | None = None,
        default: Recipe | None = None,
        programs: Mapping[str, Recipe] | None = None,
        family_registry: Any = None,
    ):
        from mindie_sim.prediction.operators.attention.batch_prefix_attention import RECIPE as _batch_prefix_attn
        from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import RECIPE as _batch_prefix_idx
        from mindie_sim.prediction.operators.mla.generic_latent_attention import build as _generic_latent_build
        from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
        from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
        from ..mechanism import PlannedRecipe as _PlannedRecipe
        from ..operators.registry import FamilyDispatcher, builtin_family_registry
        from ..physical_recipe_registry import physical_family_bindings, physical_recipe_bindings
        from mindie_sim.prediction.operators.recurrent.state_program import RecurrentVector

        generic_grouped = (programs or {}).get("matmul", MappedMatmul())
        self._generic_grouped = generic_grouped
        balanced_matrix = (families or {}).get("grouped_gemm", generic_grouped)
        self._family_registry = (
            builtin_family_registry() if family_registry is None else family_registry
        )
        self._family_dispatcher = FamilyDispatcher(self._family_registry)
        self._governed_recipes: dict[str, Any] = {}
        self._explicit_recipes = frozenset((recipes or {}).keys()) | {
            "batch_prefix_attention", "batch_prefix_indexer",
        }
        self._caller_recipes = frozenset((recipes or {}).keys())
        self._caller_programs = frozenset((programs or {}).keys())
        self._explicit_families = frozenset((families or {}).keys())
        self._family_kinds: dict[str, Any] = {}
        self.recipes = {
            **physical_recipe_bindings(),
            "balanced_grouped_gate_up": balanced_matrix,
            "balanced_grouped_down": balanced_matrix,
            "batch_prefix_attention": _batch_prefix_attn,
            "batch_prefix_indexer": _batch_prefix_idx,
            **(recipes or {}),
        }
        self.families = {
            **physical_family_bindings(),
            "recurrent_scan": RecurrentScan(),
            "recurrent_vector": RecurrentVector(),
            "engram_rh2d": HostStaging(),
            "grouped_gemm": (programs or {}).get("matmul", SteadyMatmul()),
            **(families or {}),
        }
        self.programs = {
            "matmul": SteadyMatmul(),
            "dense_latent_attention": _PlannedRecipe(_generic_latent_build),
            **(programs or {}),
        }
        self.default = default
        self._select_id_cache: dict[int, tuple[object, Recipe]] = {}
        self._expand_id_cache: dict[int, tuple[object, tuple]] = {}
        self._prev_expand: tuple | None = None

    def _cache_select(self, op: object, recipe: Recipe) -> Recipe:
        if len(self._select_id_cache) >= _SELECT_CACHE_LIMIT:
            self._select_id_cache.clear()
        self._select_id_cache[id(op)] = (op, recipe)
        return recipe

    def _governed_recipe(self, spec: Any, op):
        cached = self._governed_recipes.get(spec.name)
        if cached is not None:
            return cached
        from ..operators.registry import make_governed_recipe

        planned = make_governed_recipe(spec, self, self._family_dispatcher)
        self._governed_recipes[spec.name] = planned
        return planned

    def select(self, op: object) -> Recipe:
        hit = self._select_id_cache.get(id(op))
        if hit is not None and hit[0] is op:
            return hit[1]
        from ..mechanism import PlannedRecipe
        if op.plan_required:
            from ..operators.linear_attention.gdn_decode import RECIPE_IDS as GDN_DECODE_RECIPES

            recipe = self.recipes.get(op.recipe)
            if recipe is None and op.recipe in GDN_DECODE_RECIPES:
                raise ModelingError(f"source_plan_binding_required:{op.recipe}")
            if op.recipe and op.recipe not in self._caller_recipes and not (
                op.source_boundary == "call" and isinstance(recipe, PlannedRecipe)
                and recipe.kernels is not None
            ):
                governed = self._family_registry.match(op)
                if governed is not None:
                    recipe = self._governed_recipe(governed, op)
            if not isinstance(recipe, PlannedRecipe):
                raise ModelingError(f"source_plan_binding_required:{op.recipe}")
            if op.source_boundary == "call" and recipe.kernels is None:
                raise ModelingError(f"source_kernel_expansion_required:{op.recipe}")
            return self._cache_select(op, recipe)
        meta = dict(op.metadata) if op.metadata else {}
        view = dict(meta.get("attention_tensor_view") or {})
        recipe_name = str(getattr(op, "recipe", ""))
        family_name = str(getattr(op, "family", ""))
        legacy_recover_commit = (
            recipe_name == "recover_ssm_commit" and family_name == "elementwise"
            and not any(k in meta for k in ("lengths", "sequence_lengths", "sequence_tokens"))
        )
        governed = None if legacy_recover_commit else self._family_registry.match(op)
        exact_governed = governed if recipe_name else None
        explicit = (
            exact_governed is not None or recipe_name in self._explicit_recipes
            or family_name in self._explicit_families
        )
        kind = meta.get("program_kind")
        if (
            not explicit and kind == "dense_latent_attention_generic"
            and view.get("kind") == "dense_paged_latent"
            and "latent_attention_schedule" in meta
        ):
            recipe = self.programs.get("dense_latent_attention")
            if isinstance(recipe, PlannedRecipe):
                return self._cache_select(op, recipe)
        generic_grouped = {"balanced_grouped_gate_up", "balanced_grouped_down"}
        if recipe_name in self._caller_recipes:
            recipe = self.recipes[recipe_name]
        elif exact_governed is not None:
            recipe = self._governed_recipe(exact_governed, op)
        elif kind in self._caller_programs and (
            kind == family_name or kind != GENERIC_MATRIX_PROGRAM
        ):
            recipe = self.programs[kind]
        elif family_name in self._explicit_families and family_name != "grouped_gemm":
            recipe = self.families[op.family]
        elif family_name in {"gate_up_projection", "down_projection"}:
            from mindie_sim.prediction.operators.mx_quant_matrix.shared_w4a8 import (
                config_from_op as _shared_config,
                matches_source_interface as _shared_match,
                recipe as _shared_recipe,
            )
            if _shared_match(op):
                _shared_config(op)
                recipe = _shared_recipe()
            else:
                recipe = self.recipes.get(
                    recipe_name, self.families.get(op.family, self.programs.get(kind))
                )
        elif (
            family_name == "grouped_gemm"
            and "grouped_gemm" in self._explicit_families
            and recipe_name in generic_grouped
        ):
            recipe = self.families["grouped_gemm"]
            if self._family_kinds.get("grouped_gemm") == "ascend_mxa8w4_nznk":
                from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_grouped import (
                    config_from_op as _mxfp4_config, matches_source_interface,
                )
                if not matches_source_interface(op):
                    recipe = self._generic_grouped
                else:
                    _mxfp4_config(op)
        else:
            if governed is not None and (
                family_name == "rotary" or family_name not in self.families
            ):
                recipe = self._governed_recipe(governed, op)
            else:
                fam_spec = self._family_registry.spec_for_family(family_name)
                if (
                    recipe_name and fam_spec is not None
                    and fam_spec.name == "rotary"
                    and recipe_name not in self.recipes
                    and recipe_name not in fam_spec.delegated_legacy_recipe_ids
                ):
                    raise ModelingError(
                        f"unregistered_operator_recipe:{family_name}:{recipe_name}"
                    )
                recipe = self.recipes.get(
                    recipe_name, self.families.get(op.family, self.programs.get(kind))
                )
        if recipe is None:
            from ..physical_recipe_registry import PROJECTION_FAMILIES, resolve_physical_recipe
            if family_name in PROJECTION_FAMILIES:
                recipe = self.programs.get("matmul")
            elif self.default is not None:
                recipe = self.default
            else:
                recipe = resolve_physical_recipe(op)
        if "matrix_group_rows" in meta:
            from mindie_sim.prediction.operators.grouped_moe.grouped_rows import adapt
            recipe = adapt(recipe, op)
        else:
            from mindie_sim.prediction.operators.grouped_moe.expected_grouped import adapt
            recipe = adapt(recipe, op)
        return self._cache_select(op, recipe)

    def prepare(self, op, geometry):
        recipe = self.select(op)
        if op.source_boundary == "call":
            raise ModelingError(f"source_call_requires_kernel_expansion:{op.recipe}")
        prepare_fn = getattr(recipe, "prepare", None)
        return prepare_fn(op, geometry) if callable(prepare_fn) else (op, recipe)

    def _expand_one(self, op, expand_call, planned_cls) -> tuple:
        cache = self._expand_id_cache
        hit = cache.get(id(op))
        if hit is not None and hit[0] is op:
            return hit[1]
        recipe = self.select(op)
        items = (
            expand_call(op, recipe.kernels)
            if (isinstance(recipe, planned_cls) and recipe.kernels
                and op.source_boundary != "kernel")
            else (op,)
        )
        if len(cache) >= _EXPAND_CACHE_LIMIT:
            cache.clear()
        cache[id(op)] = (op, items)
        return items

    def expand(self, ops):
        from ..boundaries import expand_call
        from ..mechanism import PlannedRecipe
        is_tup = ops.__class__ is tuple
        if is_tup and self._prev_expand is not None:
            p_ops, p_one_to_one, p_res = self._prev_expand
            if ops is p_ops:
                return p_res
            if p_one_to_one and len(ops) == len(p_ops):
                result = None
                still_1to1 = True
                for idx, (op, pop) in enumerate(zip(ops, p_ops)):
                    if op is pop:
                        if result is not None:
                            result.append(p_res[idx])
                        continue
                    if result is None:
                        result = list(p_res[:idx])
                    items = self._expand_one(op, expand_call, PlannedRecipe)
                    if len(items) != 1:
                        still_1to1 = False
                    result.extend(items)
                if result is None:
                    self._prev_expand = (ops, True, p_res)
                    return p_res
                out = tuple(result)
                self._prev_expand = (ops, still_1to1, out)
                return out
        result_list = []
        one_to_one = True
        for op in ops:
            items = self._expand_one(op, expand_call, PlannedRecipe)
            if len(items) != 1:
                one_to_one = False
            result_list.extend(items)
        out = tuple(result_list)
        if is_tup:
            self._prev_expand = (ops, one_to_one, out)
        return out

    @classmethod
    def from_mapping(cls, raw, *, family_registry: Any = None):
        from mindie_sim.prediction.operators.attention.attention_decode import StagedGQADecode
        from mindie_sim.prediction.operators.attention.attention_prefill import StagedGQAPrefill
        from mindie_sim.prediction.operators.attention.batch_prefix_attention import BatchPrefixAttention
        from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import BatchPrefixIndexer
        from mindie_sim.prediction.operators.grouped_moe.bf16_grouped import recipe as bf16_grouped
        from ..collectives import PairwiseCollective, RingCollective
        from mindie_sim.prediction.operators.collectives.direct_peer import DirectPeerCollective
        from mindie_sim.prediction.operators.kda.family_kda_recurrent import recipes as kda_recurrent_recipes
        from mindie_sim.prediction.operators.collectives.hccl_graph import recipes as hccl_recipes
        from ..implementations import recipes as implementation_recipes
        from mindie_sim.prediction.operators.paged_mla.indexed_attention import recipe as indexed_attention
        from mindie_sim.prediction.operators.paged_mla.indexed_quant_attention import recipe as indexed_quant_attention
        from mindie_sim.prediction.operators.mla.latent_dcp import recipes as latent_dcp_recipes
        from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
        from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
        from mindie_sim.prediction.operators.matrix.matrix_phase import PhaseMatmul
        from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
        from mindie_sim.prediction.operators.mc2.mc2_graph import recipes as native_moe_recipes
        from ..mechanism import OperatorPlan, PlannedRecipe
        from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_grouped import recipe as mxa8w4_nznk
        from mindie_sim.prediction.operators.mx_quant_matrix.mxfp8_grouped import recipe as mxfp8_grouped
        from ..operators import grouped_moe, hyper_connections, linear_attention, mc2
        from ..operators.linear_attention.common import CHUNK32_KERNELS
        from ..pointwise import TiledPointwise
        from ..services import ResourceService
        from mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload import recipes as sparse_kv_offload_recipes
        from mindie_sim.prediction.operators.recurrent.state_program import RecurrentVector
        from mindie_sim.prediction.operators.vector.tensor_copy import recipe as aiv_tensor_copy

        from mindie_sim.prediction.operators.attention.attention_decode import staged_gqa_decode
        from mindie_sim.prediction.operators.attention.attention_prefill import staged_gqa_prefill

        factories = {
            "tiled_matmul": TiledMatmul, "hierarchical_matmul": HierarchicalMatmul,
            "mapped_matmul": MappedMatmul, "phase_matmul": PhaseMatmul,
            "steady_matmul": SteadyMatmul,
            "indexed_attention_pipeline": indexed_attention,
            "indexed_attention_packed_int8": indexed_quant_attention,
            "staged_gqa_decode": staged_gqa_decode,
            "staged_gqa_prefill": staged_gqa_prefill,
            "batch_prefix_attention": BatchPrefixAttention,
            "batch_prefix_indexer": BatchPrefixIndexer,
            "transfer": TransferRecipe, "recurrent_scan": RecurrentScan,
            "recurrent_vector": RecurrentVector, "tiled_pointwise": TiledPointwise,
            "aiv_tensor_copy": aiv_tensor_copy, "ascend_mxa8w4_nznk": mxa8w4_nznk,
            "ascend_bf16_grouped": bf16_grouped, "ascend_mxfp8_grouped_nz": mxfp8_grouped,
            "resource_service": ResourceService, "ring_collective": RingCollective,
            "pairwise_collective": PairwiseCollective,
            "direct_peer_collective": DirectPeerCollective,
        }

        def build(values):
            return {
                name: factories[value["kind"]](**{k: v for k, v in value.items() if k != "kind"})
                for name, value in values.items()
            }

        declared_recipes = build(raw.get("recipes", {}))
        explicit = {
            **native_moe_recipes(raw), **hccl_recipes(raw),
            **sparse_kv_offload_recipes(), **latent_dcp_recipes(),
            **implementation_recipes(raw.get("lowerings", {})),
            **kda_recurrent_recipes(),
            "attn_res_fwd": hyper_connections.GovernedAttnResRecipe(hyper_connections.build),
            **declared_recipes,
        }
        book = cls(
            explicit, families=build(raw.get("families", {})),
            programs=build(raw.get("programs", {})), family_registry=family_registry,
        )
        # Keep the public recipe table and selection on the same owner. These
        # IDs also arrive through the physical defaults and protocol inventory.
        governed_builders = {
            "megamoe_arch35": grouped_moe.build,
            "arch22_v2_caller_local": mc2.build,
            **dict.fromkeys(CHUNK32_KERNELS, linear_attention.build),
        }
        for recipe_id in governed_builders.keys() & (book.recipes.keys() - declared_recipes.keys()):
            book.recipes[recipe_id] = PlannedRecipe(governed_builders[recipe_id])
        book._caller_recipes = frozenset(declared_recipes)
        book._family_kinds = {
            name: value.get("kind")
            for name, value in (raw.get("families") or {}).items()
            if isinstance(value, Mapping)
        }
        return book


class HostStaging:
    """CPU lookup, host-vector decode and H2D transfer over qualified work keys."""

    def lower(self, context: RecipeContext):
        names, work = context.resource_names, dict(context.work)
        roles = ("host_memory", "host_vector", "h2d")
        if any(role not in names for role in roles):
            raise ModelingError("host_staging_requires_host_memory_host_vector_h2d_rates")
        if any(names[role] not in work for role in roles):
            raise ModelingError("host_staging_requires_bound_host_and_h2d_work")
        memory = {names["host_memory"]: work.pop(names["host_memory"])}
        compute = {names["host_vector"]: work.pop(names["host_vector"])}
        return KernelProgram((
            Stage("lookup", memory), Stage("decode", compute, after=("lookup",)),
            Stage("transfer", work, after=("decode",)),
        ), context.minimum_ms)


class RecurrentScan:
    """C1 -> vector delta -> C2 -> state update for each dependent chunk."""

    def lower(self, context: RecipeContext):
        from mindie_sim.prediction.operators.recurrent.recurrent_template import lower_recurrent_scan
        return lower_recurrent_scan(context)
