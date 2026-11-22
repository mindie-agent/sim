"""Operator ownership, execution physics and runtime structure governance.

[Role] Check static boundaries, live recipe bindings and physical invariants
  for AG006–AG008.
[Invariants] Cache comparisons keep the access constant. Query comparisons keep
  the context and geometry constant and verify causal QK FLOP conservation.
[Timeouts & Locks] Bounded AST scans, registry lookups and local planning.
[Side-Effect Policy] Uses private simulation state and restores the thread's
  profiler after observing dispatch. Reads no measurements.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from inspect import unwrap
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindie_sim.prediction.runtime.memory_physics import MemoryAccessPattern

_STRICT_OWNED_RECIPE_MODULES = frozenset({
    "mindie_sim.prediction.operators.sparse_indexer.bf16_lightning",
    "mindie_sim.prediction.operators.linear_attention.gdn_decode",
})
_OWNER_MODULE_PREFIXES = ("mindie_sim.prediction.operators.", "mindie_sim.prediction.sharding.")
_NON_DEVICE_INFRA_RECIPES = frozenset({
    ("mindie_sim.prediction.physical_recipe_registry", "ControlSyncRecipe"),
    ("mindie_sim.prediction.program.binding", "TransferRecipe"),
    ("mindie_sim.prediction.program.recipes", "HostStaging"),
    ("mindie_sim.prediction.program.recipes", "RecurrentScan"),
    # Transport-only programs and a single-resource service adapter.
    ("mindie_sim.prediction.collectives.ring_pairwise", "PairwiseCollective"),
    ("mindie_sim.prediction.collectives.ring_pairwise", "RingCollective"),
    ("mindie_sim.prediction.service_regime_domains.services", "ResourceService"),
})
_RECIPE_BOOK_PATH = PurePosixPath("mindie_sim/prediction/program/recipes.py")
_PHYSICAL_REGISTRY_PATH = PurePosixPath("mindie_sim/prediction/physical_recipe_registry.py")
_IMPLEMENTATION_PATH = PurePosixPath("mindie_sim/prediction/program/implementation_registry.py")
_SHIPPED_RECIPE_CASE = PurePosixPath("examples/cases/qwen38-a5.json")
_GATE_PATH = PurePosixPath("scripts/check_arch_governance.py")


@dataclass(frozen=True, slots=True)
class Violation:
    """Single architecture governance finding."""

    rule: str
    path: PurePosixPath
    line: int
    fingerprint: str
    message: str


DOCSTRING_HEADER_MAX_LINE = 30
GOVERNED_PATH_DEPTH = 5
FAMILY_PKG_DEPTH = 4
OPERATORS_PARTS = ("mindie_sim", "prediction", "operators")
HEURISTIC_IDENTIFIER_RE = re.compile(
    r"(?:^|_)(?:fudge|discount|overlap_decay|heuristic_factor|latency_multiplier)(?:_|$)",
    re.IGNORECASE,
)
CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Lock B downward ratchet on flat top-level ``mindie_sim/prediction/*.py`` files
#: (+25% headroom over post-Wave-7 count of 38; any uncontrolled sprawl fails AG008).
MAX_FLAT_PREDICTION_MODULES = 48
#: Bounded package directory fan-out ceiling for governed ``operators/<family>/``
#: packages (+25% evolution headroom over 20; unlimited new operator families).
MAX_OPERATOR_PACKAGE_FILES = 25
#: Top-level ``mindie_sim/*.py`` ceiling (+25% evolution headroom over post-Wave-7 count of 20).
MAX_TOPLEVEL_MODULES = 25
#: Bounded ``mindie_sim/onboarding/**/*.py`` footprint (+25% headroom over 6 / 4,127).
MAX_ONBOARDING_FILES = 8
MAX_ONBOARDING_LOC = 5200
#: Cohesion-First readability ceilings (watched monotonically by Hub Lock-6 ratchet).
COHESIVE_SOFT_MAX_LINES = 1800
MAX_FUNCTION_LINES = 150
MAX_NESTING_DEPTH = 6

PARALLEL_CONCEPT_CLASS_RE = re.compile(
    r"^(?:Ragged\w*Batch|SequenceBatch|StepBatch|RequestBatch"
    r"|L2CacheView|MemoryCacheView|AdmissionCacheView"
    r"|RateOverlay|PrimitiveOverlay)$"
)
BANNED_BYPASS_KWARGS = frozenset({"ragged_batch", "l2_view", "admission_view"})
OPERATOR_BANNED_BYPASS_KWARGS = frozenset({"memory_view"})
SHARD_FILENAME_RE = re.compile(
    r"(?:_helper|_helpers|_part\d*|_parts|_shard|_mixin|_extra"
    r"|_legacy|_old|_backup|_fallback|_copy\d*)$"
)
_DOMAIN_COPY_OPERATORS = frozenset({
    "mindie_sim/prediction/operators/causal_conv/conv_state_copy.py",
    "mindie_sim/prediction/operators/vector/tensor_copy.py",
})
_LEGACY_SHARD_FILES = frozenset()
_LEGACY_PRIVATE_SHARD_FILES = frozenset({
    "mindie_sim/optimization/_dual_obligations.py",
    "mindie_sim/prediction/operators/attention/_cp_tiling.py",
    "mindie_sim/prediction/operators/attention/_fia.py",
    "mindie_sim/prediction/operators/linear_attention/_gdn_decode_programs.py",
    "mindie_sim/prediction/operators/linear_attention/_gdn_native.py",
    "mindie_sim/prediction/operators/mla_prolog/_lowering.py",
    "mindie_sim/prediction/operators/paged_mla/_lowering.py",
    "mindie_sim/prediction/operators/sparse_indexer/_bf16_access.py",
    "mindie_sim/prediction/operators/sparse_indexer/_bf16_program.py",
    "mindie_sim/prediction/operators/sparse_indexer/_lowering.py",
    "mindie_sim/prediction/runtime/_engine_hbm_bus.py",
    "mindie_sim/prediction/runtime/_periodic_stream.py",
    "mindie_sim/primitives/_lowering_comms.py",
    "mindie_sim/rewrite/_oracle_prefetch.py",
    "mindie_sim/templates/_catlass_periodic.py",
    "mindie_sim/templates/_catlass_streaming.py",
})
_AG009_ROOTS = frozenset({"prediction", "primitives", "backends", "catalog", "templates"})
_AG009_PARAM_RE = re.compile(
    r"^(?:(?:num|n|max|min|total|active|used|available|die|device|sku|chip|source|default"
    r"|fallback|nominal|core|vector|cube|aic|aiv|host|node|peer|d2d|h2d|d2h|l2|ub|hbm|dram)_)?(?:"
    r"cores|aic|aic_cores|aic_count|aiv|aiv_cores|aiv_count|core_num|core_count"
    r"|clock_hz|core_clock_hz|freq|freq_mhz|freq_ghz|l2_bytes|l2_capacity_bytes"
    r"|ub_bytes|ub_capacity_bytes|hbm_bw|hbm_bw_gbps|link_bw|link_bps"
    r"|h2d_bw|h2d_bps|dma_bps|host_dma_bps)$",
    re.IGNORECASE,
)
_AG009_CONST_RE = re.compile(
    r"^_?(?:[A-Z0-9]+_)?(?:DEFAULT|FALLBACK|NOMINAL|REFERENCE)_"
    r"(?:(?:A\d+|AIC|AIV|CORE|VECTOR|SWIGLU|MX|ROUTER|GATING|HOST|FIA|SOURCE)_)*"
    r"(?:CORES?|AIC(?:_CORES|_COUNT)?|AIV(?:_CORES|_COUNT|_OWNERS|_TILES)?|CORE_NUM"
    r"|CLOCK_HZ|FREQ(?:_MHZ|_GHZ)?|L2_BYTES|L2_CAPACITY_BYTES|UB_BYTES|UB_CAPACITY_BYTES"
    r"|HBM_BW\w*|LINK_BW\w*|LINK_BPS|H2D_BW\w*|HOST_DMA_BPS|HOST_RC_BPS|DMA_BPS)$"
)
_AG009_OR_VAR_RE = re.compile(
    r"(?:^|_)(?:cores|aic|aiv|clock_hz|freq|l2_bytes|l2_capacity_bytes|ub_bytes|ub_capacity_bytes"
    r"|hbm_bw|link_bw|link_uni|h2d_bw|h2d_solo|dma_bps)(?:$|_)",
    re.IGNORECASE,
)
_AG009_FALLBACK_FNS = frozenset({
    "default_hierarchy", "_default_hierarchy", "default_geometry", "_default_geometry",
})
_AG010_L1_PKGS = frozenset({"backends", "primitives", "catalog", "templates"})
_AG010_L2_SUBS = frozenset({"program", "runtime", "execution", "collectives", "capacity"})
_AG010_L2_SHARDING = frozenset({"local_tp", "routing", "sequence_parallel"})
_AG010_L3_SUBS = frozenset({"operators", "sharding"})
_AG010_L4_PKGS = frozenset({"optimization", "rewrite", "observations", "onboarding", "export"})
_AG010_LAYER_NAMES = ("L0:schema", "L1:hardware", "L2:runtime", "L3:operators+sharding", "L4:product")
#: Ratcheted legacy upward edges (40 edges / 27 files; zero L3->L4, zero primitives/loader->L4).
_LEGACY_UPWARD_IMPORTS: Mapping[str, frozenset[str]] = {
    "mindie_sim/catalog/qwen4_exp.py": frozenset({"mindie_sim.prediction.program"}),
    "mindie_sim/catalog/speculative_state.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/capacity/__init__.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/collectives/plans.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/collectives/protocol.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/execution/cohort_launch.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/execution/cohort_native.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/execution/context.py": frozenset({"mindie_sim.prediction.operators"}),
    "mindie_sim/prediction/execution/runner.py": frozenset({"mindie_sim.observations"}),
    "mindie_sim/prediction/program/recipes.py": frozenset({
        "mindie_sim.prediction.operators", "mindie_sim.prediction.sharding",
    }),
    "mindie_sim/primitives/_lowering_comms.py": frozenset({
        "mindie_sim.prediction.collectives", "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/primitives/lowering.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/primitives/schema.py": frozenset({
        "mindie_sim.prediction.collectives", "mindie_sim.prediction.execution",
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/schema/datapath.py": frozenset({"mindie_sim.backends"}),
    "mindie_sim/schema/hardware.py": frozenset({"mindie_sim.backends"}),
    "mindie_sim/schema/model.py": frozenset({"mindie_sim.catalog"}),
    "mindie_sim/schema/parallel.py": frozenset({"mindie_sim.catalog"}),
    "mindie_sim/schema/serialization.py": frozenset({"mindie_sim.catalog"}),
    "mindie_sim/templates/_catlass_periodic.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/_catlass_streaming.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/catlass_lower.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/contracts.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/dsl_catalog.py": frozenset({"mindie_sim.prediction.runtime"}),
    "mindie_sim/templates/dsl_lower.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/fuse_region.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/fuse_region_live.py": frozenset({
        "mindie_sim.prediction.program", "mindie_sim.prediction.runtime",
    }),
    "mindie_sim/templates/registry.py": frozenset({"mindie_sim.prediction.operators"}),
}
_AG011_DISPATCH_ROOTS = frozenset({
    "mindie_sim.prediction.spine",
    "mindie_sim.prediction.frontend",
    "mindie_sim.prediction.frontend.body",
    "mindie_sim.prediction.frontend.iteration",
    "mindie_sim.prediction.program.recipes",
    "mindie_sim.prediction.physical_recipe_registry",
    "mindie_sim.prediction.program.implementation_registry",
    "mindie_sim.prediction.body_defaults",
    "mindie_sim.prediction.call_builders",
    "mindie_sim.prediction.operators.registry",
    "mindie_sim.prediction.operators",
    "mindie_sim.prediction.runtime.engine",
    "mindie_sim.compose.operator_defaults",
    "mindie_sim.stack_registry",
    "mindie_sim.templates.registry",
})
#: Ratcheted legacy standalone operator contract modules (budget=3; zero new orphans allowed).
_LEGACY_UNREACHABLE_OPERATOR_MODULES = frozenset({
    "mindie_sim/prediction/operators/collectives/hccl_alltoallv.py",
    "mindie_sim/prediction/operators/matrix/steady_pipeline.py",
    "mindie_sim/prediction/operators/vector/dynamic_quant_schedule.py",
})
_FLAT_PLAN_INFRA_FILES = frozenset({
    "mindie_sim/prediction/graph_analysis.py",
    "mindie_sim/prediction/mechanism.py",
    "mindie_sim/prediction/native_binding.py",
    "mindie_sim/prediction/physical_recipe_base.py",
    "mindie_sim/prediction/results.py",
    "mindie_sim/prediction/spine.py",
})
_FLAT_PREDICTION_DEPTH = 3


def _is_governed_family_path(rel: PurePosixPath) -> tuple[bool, str | None, str]:
    parts = rel.parts
    if len(parts) >= GOVERNED_PATH_DEPTH and parts[: len(OPERATORS_PARTS)] == OPERATORS_PARTS:
        return True, parts[len(OPERATORS_PARTS)], rel.name
    return False, None, rel.name


def _check_ag007(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    if not _is_governed_family_path(rel)[0]:
        return []
    violations = []
    seen = set()
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if field not in {"id", "name", "asname", "attr", "arg", "rest", "value"}:
                continue
            if not isinstance(value, str) or not value.isidentifier():
                continue
            key = (node.lineno, value)
            spelling = CAMEL_CASE_BOUNDARY_RE.sub("_", value)
            if key not in seen and HEURISTIC_IDENTIFIER_RE.search(spelling):
                seen.add(key)
                violations.append(Violation(
                    "AG007", rel, node.lineno, f"AG007:{rel.as_posix()}:{value}",
                    f"ad hoc heuristic identifier {value!r} in an operator family",
                ))
    return violations


def _check_ag006_flat_prediction(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    """Forbid OperatorPlan/kernel PlannedRecipe builders in flat mindie_sim/prediction/*.py."""
    pos = rel.as_posix()
    if (
        len(rel.parts) != _FLAT_PREDICTION_DEPTH
        or rel.parts[:2] != ("mindie_sim", "prediction")
        or pos in _FLAT_PLAN_INFRA_FILES
    ):
        return []
    violations: list[Violation] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name in {"recipes", "build"}:
            violations.append(Violation(
                "AG006", rel, stmt.lineno, f"AG006:flat_builder:{pos}:{stmt.name}",
                f"flat prediction module must not define kernel builder {stmt.name}(); "
                "move into mindie_sim/prediction/operators/<family>/",
            ))
        elif isinstance(stmt, ast.ClassDef):
            for base in stmt.bases:
                bname = base.id if isinstance(base, ast.Name) else (
                    base.attr if isinstance(base, ast.Attribute) else ""
                )
                if bname in {"Recipe", "PlannedRecipe"}:
                    violations.append(Violation(
                        "AG006", rel, stmt.lineno, f"AG006:flat_recipe:{pos}:{stmt.name}",
                        f"flat prediction module must not subclass {bname} ({stmt.name}); "
                        "move into mindie_sim/prediction/operators/<family>/",
                    ))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if fname in {"OperatorPlan", "PlannedRecipe"}:
                violations.append(Violation(
                    "AG006", rel, getattr(node, "lineno", 1),
                    f"AG006:flat_plan_call:{pos}:{fname}",
                    f"flat prediction module must not construct {fname}(...); "
                    "move into mindie_sim/prediction/operators/<family>/",
                ))
                break
    return violations


_NON_DECLARATIVE_EXPR = (
    ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.NamedExpr, ast.IfExp, ast.Await, ast.Yield, ast.YieldFrom,
)
_NESTING_STMTS = (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.Match)


def _declarative_extra_lines(node: ast.AST) -> int:
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)) and node.lineno and node.end_lineno:
        span = node.end_lineno - node.lineno + 1
        if span > 1 and not any(isinstance(sub, _NON_DECLARATIVE_EXPR) for sub in ast.walk(node)):
            return span - 1
    return sum(
        _declarative_extra_lines(child) for child in ast.iter_child_nodes(node)
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _max_nesting_depth(node: ast.AST, depth: int = 0) -> int:
    if isinstance(node, ast.If):
        best = depth + 1
        for stmt in node.body:
            best = max(best, _max_nesting_depth(stmt, depth + 1))
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            best = max(best, _max_nesting_depth(node.orelse[0], depth))
        else:
            for stmt in node.orelse:
                best = max(best, _max_nesting_depth(stmt, depth + 1))
        return best
    if isinstance(node, _NESTING_STMTS):
        next_depth = depth + 1
        return max((next_depth, *(
            _max_nesting_depth(child, next_depth) for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )))
    return max((depth, *(
        _max_nesting_depth(child, depth) for child in ast.iter_child_nodes(node)
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )))


def _check_cohesive_readability(rel: PurePosixPath, text: str, tree: ast.Module) -> list[Violation]:
    """Enforce [Role] header and function readability budgets on cohesive >1,800L modules."""
    line_count = len(text.splitlines())
    if line_count <= COHESIVE_SOFT_MAX_LINES:
        return []
    pos = rel.as_posix()
    violations: list[Violation] = []
    if "[Role]" not in "\n".join(text.splitlines()[:DOCSTRING_HEADER_MAX_LINE]):
        violations.append(Violation(
            "AG008", rel, 1, f"AG008:cohesive_role:{pos}",
            f"cohesive module ({line_count}L > {COHESIVE_SOFT_MAX_LINES}L) requires top-{DOCSTRING_HEADER_MAX_LINE} [Role] header",
        ))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = node.end_lineno or node.lineno
            eff = max(1, (end_line - node.lineno + 1) - _declarative_extra_lines(node))
            depth = max((0, *(_max_nesting_depth(stmt, 0) for stmt in node.body)))
            if eff > MAX_FUNCTION_LINES:
                violations.append(Violation(
                    "AG008", rel, node.lineno, f"AG008:fn_lines:{pos}:{node.name}",
                    f"function {node.name} has {eff} non-table lines (limit {MAX_FUNCTION_LINES})",
                ))
            if depth > MAX_NESTING_DEPTH:
                violations.append(Violation(
                    "AG008", rel, node.lineno, f"AG008:fn_depth:{pos}:{node.name}",
                    f"function {node.name} has nesting depth {depth} (limit {MAX_NESTING_DEPTH})",
                ))
    return violations


def _check_ag008_module(
    rel: PurePosixPath,
    text: str,
    tree: ast.Module,
    concept_owners: Mapping[str, tuple[str, ...]],
) -> list[Violation]:
    """Enforce canonical concept singletons, bypass-kwarg bans, and anti-shard governance."""
    violations = _check_cohesive_readability(rel, text, tree)
    pos = rel.as_posix()
    in_ops = rel.parts[: len(OPERATORS_PARTS)] == OPERATORS_PARTS

    if SHARD_FILENAME_RE.search(rel.stem) and pos not in _LEGACY_SHARD_FILES and pos not in _DOMAIN_COPY_OPERATORS:
        violations.append(Violation(
            "AG008", rel, 1, f"AG008:shard_file:{pos}",
            f"artificial shard filename {rel.name!r} is banned; keep cohesive domain logic together",
        ))
    if (
        rel.name.startswith("_")
        and rel.name not in {"__init__.py", "__main__.py"}
        and pos not in _LEGACY_PRIVATE_SHARD_FILES
    ):
        violations.append(Violation(
            "AG008", rel, 1, f"AG008:private_shard_file:{pos}",
            f"private overflow shard {rel.name!r} is banned; keep cohesive domain logic in its owning module",
        ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            owners = concept_owners.get(node.name)
            if owners is not None and pos not in owners:
                violations.append(Violation(
                    "AG008", rel, node.lineno, f"AG008:duplicate_concept:{pos}:{node.name}",
                    f"canonical concept {node.name!r} belongs exclusively to {owners}",
                ))
        if isinstance(node, ast.ClassDef):
            if PARALLEL_CONCEPT_CLASS_RE.match(node.name):
                violations.append(Violation(
                    "AG008", rel, node.lineno, f"AG008:parallel_concept:{pos}:{node.name}",
                    f"parallel batch/cache/overlay class {node.name!r} is banned; reuse canonical schema owners",
                ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for arg in all_args:
                if arg.arg in BANNED_BYPASS_KWARGS:
                    violations.append(Violation(
                        "AG008", rel, node.lineno, f"AG008:bypass_arg:{pos}:{node.name}:{arg.arg}",
                        f"ad-hoc bypass parameter {arg.arg!r} is banned; pass canonical context state",
                    ))
                elif in_ops and arg.arg in OPERATOR_BANNED_BYPASS_KWARGS:
                    violations.append(Violation(
                        "AG008", rel, node.lineno, f"AG008:op_bypass_arg:{pos}:{node.name}:{arg.arg}",
                        f"operator parameter {arg.arg!r} bypasses OperatorPlanningContext.admission_memory_view",
                    ))
    return violations


def _check_ag008_repo_bounds(
    rels: Sequence[PurePosixPath], *, texts: Mapping[PurePosixPath, str] | None = None
) -> list[Violation]:
    """Enforce directory module ceilings (+25% headroom) and flat prediction ratchet."""
    violations: list[Violation] = []
    toplevel = [rel for rel in rels if len(rel.parts) == 2 and rel.parts[0] == "mindie_sim"]
    if len(toplevel) > MAX_TOPLEVEL_MODULES:
        violations.append(Violation(
            "AG008", PurePosixPath("mindie_sim"), 1, "AG008:toplevel_module_ceiling",
            f"top-level mindie_sim/*.py count {len(toplevel)} exceeds ceiling {MAX_TOPLEVEL_MODULES}",
        ))
    flat_pred = [
        rel for rel in rels
        if len(rel.parts) == _FLAT_PREDICTION_DEPTH
        and rel.parts[:2] == ("mindie_sim", "prediction")
    ]
    if len(flat_pred) > MAX_FLAT_PREDICTION_MODULES:
        violations.append(Violation(
            "AG008", PurePosixPath("mindie_sim/prediction"), 1,
            "AG008:flat_prediction_ratchet",
            f"flat mindie_sim/prediction/*.py count {len(flat_pred)} exceeds ratchet ceiling "
            f"{MAX_FLAT_PREDICTION_MODULES}; consolidate operator modules into prediction/operators/<family>/",
        ))
    onboarding = [rel for rel in rels if len(rel.parts) >= 3 and rel.parts[:2] == ("mindie_sim", "onboarding")]
    if len(onboarding) > MAX_ONBOARDING_FILES:
        violations.append(Violation(
            "AG008", PurePosixPath("mindie_sim/onboarding"), 1, "AG008:onboarding_file_ceiling",
            f"onboarding module count {len(onboarding)} exceeds ceiling {MAX_ONBOARDING_FILES}",
        ))
    if texts and onboarding:
        onboarding_loc = sum(len(texts.get(rel, "").splitlines()) for rel in onboarding)
        if onboarding_loc > MAX_ONBOARDING_LOC:
            violations.append(Violation(
                "AG008", PurePosixPath("mindie_sim/onboarding"), 1, "AG008:onboarding_loc_ceiling",
                f"onboarding total LOC {onboarding_loc} exceeds ceiling {MAX_ONBOARDING_LOC}",
            ))
    by_pkg: dict[str, int] = {}
    for rel in rels:
        in_pkg, fam, _fname = _is_governed_family_path(rel)
        if in_pkg and fam is not None and len(rel.parts) == GOVERNED_PATH_DEPTH:
            by_pkg[fam] = by_pkg.get(fam, 0) + 1
    for fam, count in sorted(by_pkg.items()):
        if count > MAX_OPERATOR_PACKAGE_FILES:
            violations.append(Violation(
                "AG008", PurePosixPath(*OPERATORS_PARTS, fam), 1,
                f"AG008:package_fanout:{fam}",
                f"operator package {fam!r} has {count} peer .py files exceeding ceiling "
                f"{MAX_OPERATOR_PACKAGE_FILES}",
            ))
    return violations


def _is_numeric_default_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_numeric_default_expr(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.LShift)
    ):
        return _is_numeric_default_expr(node.left) and _is_numeric_default_expr(node.right)
    return False


def _is_forbidden_phys_default(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return False
    if _is_numeric_default_expr(node):
        return True
    return isinstance(node, ast.Name) and node.id.isupper()


def _check_ag009_silent_physical_defaults(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    """Reject numeric physical hardware defaults, fallback constants, and fallback helpers (AG009)."""
    if (
        len(rel.parts) < 3
        or rel.parts[0] != "mindie_sim"
        or rel.parts[1] not in _AG009_ROOTS
        or rel.name.startswith("custom_hardware")
        or rel.name == "hardware_derivation.py"
    ):
        return []
    pos = rel.as_posix()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _AG009_FALLBACK_FNS:
                violations.append(Violation(
                    "AG009", rel, node.lineno, f"AG009:fallback_fn:{pos}:{node.name}",
                    f"silent hardware fallback function {node.name!r} is banned; require explicit machine/die profile",
                ))
            pos_args = (*node.args.posonlyargs, *node.args.args)
            for arg, default in zip(pos_args[len(pos_args) - len(node.args.defaults):], node.args.defaults):
                if _AG009_PARAM_RE.match(arg.arg) and _is_forbidden_phys_default(default):
                    violations.append(Violation(
                        "AG009", rel, node.lineno, f"AG009:param_default:{pos}:{node.name}:{arg.arg}",
                        f"silent physical default {arg.arg}={ast.unparse(default)} in {node.name}(); "
                        "require explicit hardware parameter or None",
                    ))
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None and _AG009_PARAM_RE.match(arg.arg) and _is_forbidden_phys_default(default):
                    violations.append(Violation(
                        "AG009", rel, node.lineno, f"AG009:param_default:{pos}:{node.name}:{arg.arg}",
                        f"silent physical default {arg.arg}={ast.unparse(default)} in {node.name}(); "
                        "require explicit hardware parameter or None",
                    ))
        elif isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if fn in _AG009_FALLBACK_FNS:
                violations.append(Violation(
                    "AG009", rel, getattr(node, "lineno", 1), f"AG009:fallback_call:{pos}:{fn}",
                    f"call to silent hardware fallback {fn}() is banned; pass explicit hardware state",
                ))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and _AG009_CONST_RE.match(target.id)
                    and _is_numeric_default_expr(node.value)
                ):
                    violations.append(Violation(
                        "AG009", rel, node.lineno, f"AG009:fallback_const:{pos}:{target.id}",
                        f"silent physical fallback constant {target.id} is banned; read from active die profile",
                    ))
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and len(node.values) >= 2:
            rhs = node.values[-1]
            if not _is_forbidden_phys_default(rhs):
                continue
            if isinstance(rhs, ast.Constant) and rhs.value in (0, 0.0):
                continue
            lhs_names = (
                sub.id if isinstance(sub, ast.Name) else sub.attr
                for expr in node.values[:-1]
                for sub in ast.walk(expr)
                if isinstance(sub, (ast.Name, ast.Attribute))
            )
            if any(_AG009_OR_VAR_RE.search(name) for name in lhs_names):
                violations.append(Violation(
                    "AG009", rel, getattr(node, "lineno", 1),
                    f"AG009:or_fallback:{pos}:{getattr(node, 'lineno', 1)}",
                    f"numeric 'or' fallback {ast.unparse(node)!r} on physical hardware quantity is banned",
                ))
    return violations


def _ag010_layer(parts: tuple[str, ...]) -> int | None:
    if len(parts) < 2 or parts[0] != "mindie_sim":
        return None
    top = parts[1]
    if top == "schema":
        return 0
    if top in _AG010_L1_PKGS:
        return 1
    if top == "prediction" and len(parts) >= 3:
        if parts[2] in _AG010_L2_SUBS:
            return 2
        if parts[2] == "sharding" and len(parts) >= 4 and parts[3] in _AG010_L2_SHARDING:
            return 2
        if parts[2] in _AG010_L3_SUBS:
            return 3
    if top in _AG010_L4_PKGS or top.startswith(("compose", "sweep")):
        return 4
    return None


def _ag010_bucket(parts: tuple[str, ...]) -> str:
    if len(parts) >= 3 and parts[1] == "prediction":
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def _resolve_relative_import(rel: PurePosixPath, module: str | None, level: int) -> tuple[str, ...]:
    if level == 0:
        return tuple(module.split(".")) if module else ()
    pkg = tuple(rel.with_suffix("").parts[:-1])
    up = level - 1
    base = pkg[: len(pkg) - up] if up <= len(pkg) else ()
    return base + (tuple(module.split(".")) if module else ())


def _iter_runtime_import_nodes(stmts: Sequence[ast.stmt]):
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            test = stmt.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc:
                yield from _iter_runtime_import_nodes(stmt.orelse)
                continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            yield stmt
            continue
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            val = getattr(stmt, field, None)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, ast.stmt):
                        yield from _iter_runtime_import_nodes((item,))
                    elif hasattr(item, "body") and isinstance(item.body, list):
                        yield from _iter_runtime_import_nodes(item.body)


def _check_ag010_layer_dag(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    """Enforce unidirectional L0->L1->L2->L3->L4 layer import DAG (AG010)."""
    src_layer = _ag010_layer(rel.with_suffix("").parts)
    if src_layer is None:
        return []
    pos = rel.as_posix()
    allowed = _LEGACY_UPWARD_IMPORTS.get(pos, frozenset())
    seen: set[str] = set()
    violations: list[Violation] = []
    for node in _iter_runtime_import_nodes(tree.body):
        targets: list[tuple[str, ...]] = []
        if isinstance(node, ast.Import):
            targets.extend(tuple(alias.name.split(".")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative_import(rel, node.module, node.level)
            base_layer = _ag010_layer(base) if base else None
            if base_layer is not None and base_layer > src_layer:
                targets.append(base)
            elif base:
                targets.extend((*base, alias.name) for alias in node.names if alias.name != "*")
        for target in targets:
            dst_layer = _ag010_layer(target)
            if dst_layer is None or dst_layer <= src_layer:
                continue
            bucket = _ag010_bucket(target)
            if bucket in allowed or bucket in seen:
                continue
            seen.add(bucket)
            violations.append(Violation(
                "AG010", rel, getattr(node, "lineno", 1),
                f"AG010:upward_import:{pos}:{bucket}",
                f"upward layer import {_AG010_LAYER_NAMES[src_layer]} -> "
                f"{_AG010_LAYER_NAMES[dst_layer]} ({bucket}) violates unidirectional DAG",
            ))
    return violations


def _module_name_of_rel(rel: PurePosixPath) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _check_ag011_operator_reachability(
    parsed_files: Sequence[tuple[PurePosixPath, ast.Module]],
) -> list[Violation]:
    """Verify every operators/<family> module is reachable from governed dispatch roots (AG011)."""
    mod_to_rel: dict[str, PurePosixPath] = {}
    mod_to_tree: dict[str, ast.Module] = {}
    for rel, tree in parsed_files:
        mod = _module_name_of_rel(rel)
        mod_to_rel[mod] = rel
        mod_to_tree[mod] = tree
    present_roots = [root for root in _AG011_DISPATCH_ROOTS if root in mod_to_tree]
    if not present_roots:
        return []
    edges: dict[str, set[str]] = {}
    for mod, tree in mod_to_tree.items():
        rel = mod_to_rel[mod]
        deps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    for idx in range(1, len(parts) + 1):
                        cand = ".".join(parts[:idx])
                        if cand in mod_to_tree:
                            deps.add(cand)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_relative_import(rel, node.module, node.level)
                if base:
                    for idx in range(1, len(base) + 1):
                        cand = ".".join(base[:idx])
                        if cand in mod_to_tree:
                            deps.add(cand)
                    for alias in node.names:
                        if alias.name != "*":
                            cand = ".".join((*base, alias.name))
                            if cand in mod_to_tree:
                                deps.add(cand)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("mindie_sim.prediction.operators.") and node.value in mod_to_tree:
                    deps.add(node.value)
        edges[mod] = deps
    visited = set(present_roots)
    stack = list(present_roots)
    while stack:
        cur = stack.pop()
        for nxt in edges.get(cur, ()):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    violations: list[Violation] = []
    for mod, rel in sorted(mod_to_rel.items()):
        in_pkg, _fam, _fname = _is_governed_family_path(rel)
        if not in_pkg or mod in visited or rel.as_posix() in _LEGACY_UNREACHABLE_OPERATOR_MODULES:
            continue
        violations.append(Violation(
            "AG011", rel, 1, f"AG011:unreachable_operator:{rel.as_posix()}",
            f"operator module {rel.as_posix()!r} is unreachable from governed IMPLEMENTATIONS/Engine dispatch graph",
        ))
    return violations


_AG013_MODEL_BRANCH_ROOTS = frozenset({
    "structure", "solve", "sweep", "rewrite", "optimization", "export",
})
_AG013_SUFFIX_BAN_ROOTS = frozenset({
    "structure", "solve", "sweep", "rewrite", "optimization",
})
_AG013_MODEL_EXEMPT = frozenset({
    # CLI launch-script string adapter mapping --served-model-name -> catalog ID.
    "mindie_sim/export/script_audit.py",
})
_AG013_MODEL_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:"
    r"kimi(?:[-_]?k\d+(?:\.\d+)?)?|k3\.1|qwen\w*|deepseek\w*|glm[-_]?\d\w*|minimax\w*"
    r"|llama(?:[-_]?\d\w*)?|gemma[-_]?\d\w*|step[-_]?5\w*"
    r")(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_AG013_EXACT_MODEL_LOWER = frozenset({"k3", "glm", "gemma"})
_AG013_BANNED_SUFFIX_TABLES = frozenset({
    "_KNOWN_FUSION_SOURCES",
    "_GDN_PROLOGUE_SUFFIXES",
    "_KDA_PROLOGUE_SUFFIXES",
})
_AG013_SUBMODULE_SUFFIX_RE = re.compile(r"^(?:gdn|kda)$", re.IGNORECASE)


def _is_ag013_model_literal(value: str) -> bool:
    return bool(_AG013_MODEL_TOKEN_RE.search(value)) or value.strip() in _AG013_EXACT_MODEL_LOWER


def _ag013_test_exprs(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            yield node.test


def _check_ag013_anti_hardcoding(rel: PurePosixPath, tree: ast.Module) -> list[Violation]:
    """Forbid model-name branches and static op-suffix tables in general solver/rewrite/export layers (AG013)."""
    if len(rel.parts) < 2 or rel.parts[0] != "mindie_sim":
        return []
    top = rel.parts[1][:-3] if rel.parts[1].endswith(".py") else rel.parts[1]
    if top not in _AG013_MODEL_BRANCH_ROOTS:
        return []
    pos = rel.as_posix()
    violations: list[Violation] = []
    seen: set[tuple[int, str]] = set()

    if pos not in _AG013_MODEL_EXEMPT:
        for test in _ag013_test_exprs(tree):
            for sub in ast.walk(test):
                if isinstance(sub, ast.Compare):
                    for node in (sub.left, *sub.comparators):
                        if (
                            isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                            and _is_ag013_model_literal(node.value)
                        ):
                            key = (getattr(sub, "lineno", 1), node.value.lower())
                            if key not in seen:
                                seen.add(key)
                                violations.append(Violation(
                                    "AG013", rel, key[0],
                                    f"AG013:model_branch:{pos}:{key[1]}",
                                    f"hardcoded model-family branch literal {node.value!r} is forbidden in general {top} logic",
                                ))
                elif (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in {"startswith", "endswith"}
                ):
                    for arg in sub.args:
                        if (
                            isinstance(arg, ast.Constant)
                            and isinstance(arg.value, str)
                            and _is_ag013_model_literal(arg.value)
                        ):
                            key = (getattr(sub, "lineno", 1), arg.value.lower())
                            if key not in seen:
                                seen.add(key)
                                violations.append(Violation(
                                    "AG013", rel, key[0],
                                    f"AG013:model_branch:{pos}:{key[1]}",
                                    f"hardcoded model-family branch literal {arg.value!r} is forbidden in general {top} logic",
                                ))

    if top in _AG013_SUFFIX_BAN_ROOTS:
        for stmt in tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else (stmt.target,)
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in _AG013_BANNED_SUFFIX_TABLES:
                        violations.append(Violation(
                            "AG013", rel, getattr(stmt, "lineno", 1),
                            f"AG013:suffix_table:{pos}:{target.id}",
                            f"static op-suffix fusion table {target.id!r} is forbidden; use typed algebraic traits",
                        ))
        if top in {"solve", "sweep", "rewrite", "optimization"}:
            for test in _ag013_test_exprs(tree):
                for sub in ast.walk(test):
                    if isinstance(sub, ast.Compare) and any(
                        isinstance(op, (ast.In, ast.NotIn)) for op in sub.ops
                    ):
                        if (
                            isinstance(sub.left, ast.Constant)
                            and isinstance(sub.left.value, str)
                            and _AG013_SUBMODULE_SUFFIX_RE.match(sub.left.value)
                        ):
                            key = (getattr(sub, "lineno", 1), f"suf:{sub.left.value.lower()}")
                            if key not in seen:
                                seen.add(key)
                                violations.append(Violation(
                                    "AG013", rel, key[0],
                                    f"AG013:suffix_branch:{pos}:{sub.left.value.lower()}",
                                    f"ad-hoc op-suffix substring check {sub.left.value!r} is forbidden in {top}; use typed traits",
                                ))
    return violations


def _binding_module(recipe) -> str:
    def module_of(builder):
        # functools.wraps copies __module__ even when the wrapper runs elsewhere.
        return getattr(builder, "__globals__", {}).get(
            "__name__", getattr(builder, "__module__", type(builder).__module__),
        )

    def outside_owner(builder):
        return not isinstance(builder, partial) and not _is_governed_module(module_of(builder))

    builder = unwrap(getattr(recipe, "build", getattr(recipe, "lower", recipe)), stop=outside_owner)
    while isinstance(builder, partial):
        builder = unwrap(builder.func, stop=outside_owner)
    return module_of(builder)


def _is_governed_module(module: str) -> bool:
    for prefix in _OWNER_MODULE_PREFIXES:
        if module.startswith(prefix):
            if prefix != _OWNER_MODULE_PREFIXES[0]:
                return True
            name = module.removeprefix(prefix)
            # Registry-generated recipes dispatch through the family resolver audited
            # below. Other flat modules directly under operators/ have no family.
            return "." in name or name == "registry" or hasattr(sys.modules.get(module), "__path__")
    return False


def _is_shared_transport_lowering(binding, module: str) -> bool:
    from mindie_sim.prediction.collectives import PairwiseCollective

    cls = binding if isinstance(binding, type) else type(binding)
    return (
        module == PairwiseCollective.__module__ and _is_governed_module(cls.__module__)
        and getattr(cls, "lower", None) is PairwiseCollective.lower
    )


def verify_ag006_ownership(root: Path | None = None) -> tuple[Violation, ...]:
    """Check 100% of active recipe bindings, IMPLEMENTATIONS, RecipeBook, and live dispatch."""
    from mindie_sim.prediction.program.implementation_registry import IMPLEMENTATIONS
    from mindie_sim.prediction.operators import mc2
    from mindie_sim.prediction.operators.linear_attention import gdn_decode
    from mindie_sim.prediction.operators.registry import builtin_family_registry
    from mindie_sim.prediction.physical_recipe_registry import (
        physical_family_bindings,
        physical_recipe_bindings,
    )
    from mindie_sim.prediction.program import RecipeBook

    specs = builtin_family_registry().specs()
    owned = {recipe_id for spec in specs for recipe_id in spec.owned_recipe_ids}
    book = RecipeBook.from_mapping({})
    violations = []

    def check(binding, path, fingerprint, description, *, allow_infra: bool = False):
        module = _binding_module(binding)
        cls = binding if isinstance(binding, type) else type(binding)
        infra = (cls.__module__, cls.__name__) in _NON_DEVICE_INFRA_RECIPES
        if allow_infra and infra and module == cls.__module__:
            return
        if not _is_governed_module(module) and not _is_shared_transport_lowering(binding, module):
            violations.append(Violation(
                "AG006", path, 1, f"AG006:{fingerprint}",
                f"{description} uses builder outside governed packages: {module}",
            ))
        expand = getattr(binding, "kernels", None)
        if callable(expand):
            check(expand, path, f"{fingerprint}:kernels", f"{description} kernel expander")

    def check_book(candidate, path, label, *, skip_recipes=()):
        for table in ("recipes", "families", "programs"):
            for name, binding in sorted(getattr(candidate, table).items()):
                if table == "recipes" and name in skip_recipes:
                    continue
                check(binding, path, f"{label}:{table}:{name}",
                      f"{label} {table} entry {name!r}", allow_infra=True)
        if candidate.default is not None:
            check(candidate.default, path, f"{label}:default", f"{label} default", allow_infra=True)

    checked_book_recipes: set[str] = set()
    for spec in specs:
        owner = import_module(_OWNER_MODULE_PREFIXES[0] + spec.name)
        path = PurePosixPath(*owner.__name__.split("."), "__init__.py")
        check(spec.resolve, path, f"resolver:{spec.name}", f"{spec.name} resolver")
        if hasattr(owner, "build"):
            check(owner.build, path, f"builder:{spec.name}", f"{spec.name} builder")
        for table_name in ("_STRATEGIES", "STRATEGIES", "_BUILDERS"):
            for key, binding in sorted(getattr(owner, table_name, {}).items()):
                check(binding, path, f"dispatcher:{spec.name}:{table_name}:{key}",
                      f"{spec.name} dispatcher entry {key!r}")
        for recipe_id in sorted(spec.owned_recipe_ids & book.recipes.keys()):
            checked_book_recipes.add(recipe_id)
            check(book.recipes[recipe_id], _RECIPE_BOOK_PATH, f"recipe:{recipe_id}",
                  f"{spec.name} recipe {recipe_id!r}")

    path = PurePosixPath(*gdn_decode.__name__.split(".")).with_suffix(".py")
    for recipe_id, builder in sorted(gdn_decode._BUILDERS.items()):
        check(builder, path, f"dispatcher:gdn_decode:{recipe_id}",
              f"GDN decode dispatcher entry {recipe_id!r}")

    check_book(book, _RECIPE_BOOK_PATH, "mapping_book", skip_recipes=checked_book_recipes)
    check_book(RecipeBook(), _RECIPE_BOOK_PATH, "default_book")
    for name, factory in sorted(RecipeBook._mapping_factories().items()):
        check(factory, _RECIPE_BOOK_PATH, f"factory:{name}",
              f"RecipeBook factory {name!r}", allow_infra=True)
    for fam_id, binding in sorted(physical_family_bindings().items()):
        check(binding, _PHYSICAL_REGISTRY_PATH, f"physical_family:{fam_id}",
              f"physical_recipe_registry family {fam_id!r}", allow_infra=True)
    for rec_id, binding in sorted(physical_recipe_bindings().items()):
        check(binding, _PHYSICAL_REGISTRY_PATH, f"physical_recipe:{rec_id}",
              f"physical_recipe_registry recipe {rec_id!r}", allow_infra=True)

    # Full-table Lock A: enforce 100% of IMPLEMENTATIONS (zero selector loophole!).
    checked_modules: dict[str, dict] = {}
    for name, implementation in sorted(IMPLEMENTATIONS.items()):
        module = implementation.module
        if not module.startswith(_OWNER_MODULE_PREFIXES):
            violations.append(Violation(
                "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}",
                f"implementation {name!r} points outside governed packages: {module}",
            ))
            continue
        if module not in checked_modules:
            try:
                owner = import_module(module)
                if not _is_governed_module(module):
                    violations.append(Violation(
                        "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}",
                        f"implementation {name!r} has no governed family: {module}",
                    ))
                    continue
                checked_modules[module] = owner.recipes()
            except (ImportError, AttributeError) as exc:
                violations.append(Violation(
                    "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}:recipes",
                    f"implementation {name!r} cannot expose its recipes: {exc}",
                ))
                continue
        recipes = checked_modules[module]
        strict_owned = module in _STRICT_OWNED_RECIPE_MODULES
        for recipe_id, recipe in sorted(recipes.items()):
            fingerprint = f"implementation:{name}:{recipe_id}"
            description = f"implementation {name!r} recipe {recipe_id!r}"
            if strict_owned and recipe_id not in owned:
                violations.append(Violation(
                    "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:{fingerprint}",
                    f"migrated implementation {name!r} exposes unowned recipe {recipe_id!r}",
                ))
            check(recipe, _IMPLEMENTATION_PATH, f"{fingerprint}:builder", description, allow_infra=True)
    if not violations:
        # These registries only publish transport recipes for declared protocols.
        for kind in sorted(mc2._NATIVE_PROTOCOLS):
            protocol = {"kind": kind, "dispatch_quant": "per_token_int8", "max_inflight_per_rank": 1}
            candidate = RecipeBook.from_mapping({
                "hccl": {"allgather": {}},
                "native_moe_protocol": protocol,
            })
            check_book(candidate, _RECIPE_BOOK_PATH, f"protocol_book:{kind}")
        violations.extend(_verify_live_dispatch(book))
        _check_shipped_case(root or Path(__file__).resolve().parents[1], check, check_book)
    return tuple(violations)


def _check_shipped_case(root: Path, check, check_book) -> None:
    from mindie_sim import target_graphs
    from mindie_sim.prediction.program import RecipeBook

    raw = json.loads((root / _SHIPPED_RECIPE_CASE).read_text(encoding="utf-8"))
    for phase in ("prefill", "decode"):
        graphs = target_graphs(
            raw, scope="model", phase=phase, tokens=1, context=1024, sequences=1,
        )
        for replica, graph in graphs.items():
            book = RecipeBook.from_mapping(graph.effective_kernels)
            label = f"case:{phase}:{replica}"
            check_book(book, _SHIPPED_RECIPE_CASE, label)
            call_sites = {id(op): op for op in (*graph, *book.expand(graph))}
            for op in call_sites.values():
                check(book.select(op), _SHIPPED_RECIPE_CASE, f"{label}:{op.op_id}",
                      f"{phase} selection {op.op_id!r}", allow_infra=True)


def _dispatched_builder_modules(book, op, geometry, dispatcher) -> tuple[str, ...]:
    """Observe the callbacks that return the prepared plan to the dispatcher."""
    previous = sys.getprofile()
    returns = []

    def observe(frame, event, result):
        caller = frame.f_back
        if event == "return" and caller is not None and caller.f_code is dispatcher.__code__:
            returns.append((result, frame.f_globals["__name__"]))

    try:
        sys.setprofile(observe)
        _, plan = book.prepare(op, geometry)
    finally:
        if previous is None or callable(previous):
            sys.setprofile(previous)
        else:
            previous.enable()
    return tuple(module for result, module in returns if result is plan)


def _gdn_decode_probe():
    from mindie_sim import compose_case, target_graphs
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.structure import parse_case

    case = compose_case("qwen3.5-35b-a3b", "a5", layers=1, concurrency=1,
                        prompt_tokens=1, output_tokens=1)
    graph = target_graphs(case, scope="body", phase="decode", tokens=1, context=1024, sequences=1)[0]
    geometry = Geometry.from_devices(parse_case(case).machine.dies)
    return RecipeBook.from_mapping(graph.effective_kernels), graph, geometry


def _verify_live_dispatch(default_book) -> tuple[Violation, ...]:
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators import sparse_indexer
    from mindie_sim.prediction.operators.linear_attention import gdn_decode, gdn_recurrent

    gdn_book, graph, gdn_geometry = _gdn_decode_probe()
    gdn_ops = {op.recipe: op for op in graph}
    indexer_op = sparse_indexer.bf16_lightning.make_op(query_rows=1, context=1024, phase="decode")
    probes = [
        (recipe_id, gdn_book, gdn_ops.get(recipe_id), gdn_geometry, gdn_decode.build)
        for recipe_id in (gdn_decode.GATING, gdn_decode.CONV, gdn_recurrent.IMPLEMENTATION)
    ]
    probes.append((indexer_op.recipe, default_book, indexer_op,
                   Geometry(1, {"vector": 2}), sparse_indexer.build))
    violations = []
    for recipe_id, book, op, geometry, dispatcher in probes:
        modules = () if op is None else _dispatched_builder_modules(book, op, geometry, dispatcher)
        if not modules or any(not _is_governed_module(module) for module in modules):
            violations.append(Violation(
                "AG006", _RECIPE_BOOK_PATH, 1, f"AG006:dispatch:{recipe_id}",
                f"live recipe {recipe_id!r} must dispatch to a governed builder; observed {modules}",
            ))
    return tuple(violations)


def _cache_ledgers(pattern: MemoryAccessPattern | None):
    from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView, bind_memory_access
    from mindie_sim.prediction.runtime.memory_physics import (
        DEFAULT_SECTOR_BYTES, AddressRun, AddressRunPlan, HbmBusSpec, MemoryAccessPattern, ReuseKey,
    )

    if pattern is None:
        tensor = "ag007_cache_probe"
        logical = DEFAULT_SECTOR_BYTES - 1
        pattern = MemoryAccessPattern(
            tensor, 0, 0, 0, "read", "mte2_aic", logical,
            AddressRunPlan((AddressRun(0, 1, logical, logical),)),
            ReuseKey(0, tensor, 0),
        )
    capacity = 2 * pattern.runs.aligned_copy_bytes()
    view = AdmissionMemoryView({pattern.die: capacity}, granule_bytes=DEFAULT_SECTOR_BYTES)
    bus = HbmBusSpec(pattern.die, 1.0, 1.0, 0.0)

    def bind(*, commit):
        return bind_memory_access(
            pattern, view, hbm_bus=bus, copy_hit_bytes_per_ms=1.0,
            copy_miss_bytes_per_ms=1.0, l2_bytes_per_ms=1.0,
            pipe_row_id=pattern.pipe, l2_row_id="l2", commit=commit,
        ).ledger

    return bind(commit=True), bind(commit=False)


def _indexer_query_probe() -> Callable[[int, int], float]:
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators.sparse_indexer.bf16_lightning import make_op
    from mindie_sim.prediction.program import RecipeBook

    book = RecipeBook.from_mapping({})
    geometry = Geometry(1, {"vector": 2})

    def physical_flops(query_tokens: int, context_tokens: int) -> float:
        op = make_op(query_rows=query_tokens, context=context_tokens, phase="decode")
        _, plan = book.prepare(op, geometry)
        return sum(plan.work.matrix.values())

    return physical_flops


def _expected_indexer_qk_flops(query_tokens: int, context_tokens: int) -> int:
    from mindie_sim.prediction.operators.sparse_indexer import bf16_lightning

    # Each query sees complete compressed key groups through its own token.
    pairs = sum((context_tokens + row + 1) // bf16_lightning._DEFAULT_COMPRESSION
                for row in range(query_tokens))
    return 2 * pairs * bf16_lightning._DEFAULT_QUERY_HEADS * bf16_lightning._DEFAULT_HEAD_DIM


def verify_ag007_discrimination(
    query_work: Callable[[int, int], float] | None = None,
    *,
    context_tokens: int = 1024,
    multi_query_tokens: int = 2,
    pattern: MemoryAccessPattern | None = None,
) -> tuple[str, ...]:
    """Return failed physical checks, verifying exact BF16 QK FLOPs by default.

    A custom query probe takes (query_tokens, context_tokens) and returns a
    physical work counter expected to increase with query tokens. A custom
    pattern can exercise a specific cacheable read through the same binder.
    """
    if context_tokens < 0 or multi_query_tokens <= 1:
        raise ValueError("AG007 requires a nonnegative context and a multi-query probe")
    if pattern is not None and pattern.direction != "read":
        raise ValueError("AG007 requires a read access pattern")

    failures = []
    cold, warm = _cache_ledgers(pattern)
    if (
        (cold.logical_tensor_bytes, cold.aligned_transfer_bytes)
        != (warm.logical_tensor_bytes, warm.aligned_transfer_bytes)
    ):
        failures.append("cold and warm L2 reads changed logical or aligned bytes")
    if cold.dram_bus_transaction_bytes <= warm.dram_bus_transaction_bytes:
        failures.append("cold and warm L2 reads must use less DRAM when warm")

    probe = _indexer_query_probe() if query_work is None else query_work
    single = probe(1, context_tokens)
    multiple = probe(multi_query_tokens, context_tokens)
    if not (isfinite(single) and isfinite(multiple) and 0 <= single < multiple):
        failures.append("same-context multi-query plan must increase physical work")
    if query_work is None:
        for query_tokens, actual in ((1, single), (multi_query_tokens, multiple)):
            expected = _expected_indexer_qk_flops(query_tokens, context_tokens)
            if actual != expected:
                failures.append(
                    f"BF16 indexer QK FLOPs for q={query_tokens}, context={context_tokens} "
                    f"must be {expected}, got {actual:g}"
                )
    return tuple(failures)


def verify_ag011_engine_reachability() -> tuple[Violation, ...]:
    """Verify governed recipes execute through runtime.Engine and registries stay wired."""
    from mindie_sim.prediction.program.implementation_registry import IMPLEMENTATIONS
    from mindie_sim.prediction.operators.registry import builtin_family_registry
    from mindie_sim.prediction.runtime.engine import Engine
    from mindie_sim.prediction.runtime.engine_types import Resource

    violations: list[Violation] = []
    for name, impl in sorted(IMPLEMENTATIONS.items()):
        try:
            import_module(impl.module)
        except Exception as exc:
            violations.append(Violation(
                "AG011", _IMPLEMENTATION_PATH, 1, f"AG011:impl_import:{name}",
                f"registered implementation {name!r} ({impl.module}) failed import: {exc}",
            ))
    for spec in builtin_family_registry().specs():
        mod_name = _OWNER_MODULE_PREFIXES[0] + spec.name
        try:
            import_module(mod_name)
        except Exception as exc:
            violations.append(Violation(
                "AG011", _GATE_PATH, 1, f"AG011:family_import:{spec.name}",
                f"registered family {spec.name!r} failed import: {exc}",
            ))
    engine = Engine([Resource("vector", 1.0, units=1)])
    engine.submit("ag011_probe", work={"vector": 1.0})
    engine.advance()
    elapsed = engine.now
    if not (isfinite(elapsed) and elapsed > 0.0):
        violations.append(Violation(
            "AG011", PurePosixPath("mindie_sim/prediction/runtime/engine.py"), 1,
            "AG011:engine_smoke_failed",
            f"runtime.Engine reachability smoke returned invalid duration {elapsed!r}",
        ))
    return tuple(violations)


def scan_runtime(root: Path) -> tuple[Violation, ...]:
    """Load the requested checkout ahead of installed packages for strict checks."""
    sys.path.insert(0, str(root))
    try:
        sim = import_module("mindie_sim")
        if Path(sim.__file__).resolve().parent != (root / "mindie_sim").resolve():
            raise RuntimeError("strict architecture governance requires the target checkout's runtime")
        ownership = verify_ag006_ownership(root)
        discrimination = verify_ag007_discrimination()
        reachability = verify_ag011_engine_reachability()
    finally:
        sys.path.remove(str(root))
    return ownership + tuple(
        Violation("AG007", _GATE_PATH, 1, f"AG007:{failure}", failure)
        for failure in discrimination
    ) + reachability
