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
_OWNER_MODULE_PREFIXES = ("mindie_sim.prediction.operators.",)
_NON_DEVICE_INFRA_RECIPES = frozenset({
    ("mindie_sim.prediction.physical_recipe_registry", "ControlSyncRecipe"),
    ("mindie_sim.prediction.program.binding", "TransferRecipe"),
    ("mindie_sim.prediction.program.recipes", "HostStaging"),
    ("mindie_sim.prediction.program.recipes", "RecurrentScan"),
    # Transport-only programs and a single-resource service adapter.
    ("mindie_sim.prediction.collectives.ring_pairwise", "PairwiseCollective"),
    ("mindie_sim.prediction.collectives.ring_pairwise", "RingCollective"),
    ("mindie_sim.prediction.services", "ResourceService"),
})
_RECIPE_BOOK_PATH = PurePosixPath("mindie_sim/prediction/program/recipes.py")
_PHYSICAL_REGISTRY_PATH = PurePosixPath("mindie_sim/prediction/physical_recipe_registry.py")
_IMPLEMENTATION_PATH = PurePosixPath("mindie_sim/prediction/implementation_registry.py")
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
#: (any new flat file under mindie_sim/prediction/ fails AG008).
MAX_FLAT_PREDICTION_MODULES = 119
#: Bounded package directory fan-out ceiling for governed ``operators/<family>/`` packages.
MAX_OPERATOR_PACKAGE_FILES = 20
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
SHARD_FILENAME_RE = re.compile(r"(?:_helpers|_mixin|_part\d+)$")
_LEGACY_SHARD_FILES = frozenset({
    "mindie_sim/prediction/execution/step_helpers.py",
    "mindie_sim/prediction/runtime/engine_periodic_mixin.py",
})
_FLAT_PLAN_INFRA_FILES = frozenset({
    "mindie_sim/prediction/fusion_bridge.py",
    "mindie_sim/prediction/fusion_recipe.py",
    "mindie_sim/prediction/implementation_registry.py",
    "mindie_sim/prediction/mechanism.py",
    "mindie_sim/prediction/native_binding.py",
    "mindie_sim/prediction/native_coverage.py",
    "mindie_sim/prediction/physical_recipe_base.py",
    "mindie_sim/prediction/results.py",
    "mindie_sim/prediction/spine.py",
    "mindie_sim/prediction/verification.py",
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

    if SHARD_FILENAME_RE.search(rel.stem) and pos not in _LEGACY_SHARD_FILES:
        violations.append(Violation(
            "AG008", rel, 1, f"AG008:shard_file:{pos}",
            f"artificial shard filename {rel.name!r} is banned; keep cohesive domain logic together",
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


def _check_ag008_repo_bounds(rels: Sequence[PurePosixPath]) -> list[Violation]:
    """Enforce downward ratchet on flat prediction/*.py count and operator package fan-out."""
    violations: list[Violation] = []
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
    prefix = _OWNER_MODULE_PREFIXES[0]
    if not module.startswith(prefix):
        return False
    name = module.removeprefix(prefix)
    # Registry-generated recipes dispatch through the family resolver audited
    # below. Other flat modules directly under operators/ have no family.
    return "." in name or name == "registry" or hasattr(sys.modules.get(module), "__path__")


def _is_shared_transport_lowering(binding, module: str) -> bool:
    from mindie_sim.prediction.collectives import PairwiseCollective

    cls = binding if isinstance(binding, type) else type(binding)
    return (
        module == PairwiseCollective.__module__ and _is_governed_module(cls.__module__)
        and getattr(cls, "lower", None) is PairwiseCollective.lower
    )


def verify_ag006_ownership(root: Path | None = None) -> tuple[Violation, ...]:
    """Check 100% of active recipe bindings, IMPLEMENTATIONS, RecipeBook, and live dispatch."""
    from mindie_sim.prediction.implementation_registry import IMPLEMENTATIONS
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


def scan_runtime(root: Path) -> tuple[Violation, ...]:
    """Load the requested checkout ahead of installed packages for strict checks."""
    sys.path.insert(0, str(root))
    try:
        sim = import_module("mindie_sim")
        if Path(sim.__file__).resolve().parent != (root / "mindie_sim").resolve():
            raise RuntimeError("strict architecture governance requires the target checkout's runtime")
        ownership = verify_ag006_ownership(root)
        discrimination = verify_ag007_discrimination()
    finally:
        sys.path.remove(str(root))
    return ownership + tuple(
        Violation("AG007", _GATE_PATH, 1, f"AG007:{failure}", failure)
        for failure in discrimination
    )
