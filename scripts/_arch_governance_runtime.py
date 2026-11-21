"""Live recipe ownership and physical discrimination checks for AG006/AG007.

[Role] Inspect active registry bindings and exercise bounded physical probes.
[Invariants] Cache comparisons keep the access constant. Query comparisons keep
  the context and geometry constant and verify causal QK FLOP conservation.
[Timeouts & Locks] Finite registry lookups and local planning without locks.
[Side-Effect Policy] Uses private simulation state and restores the thread's
  profiler after observing dispatch. Reads no measurements.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from inspect import unwrap
from math import isfinite
from pathlib import Path, PurePosixPath
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindie_sim.prediction.runtime.memory_physics import MemoryAccessPattern

_GOVERNED_FAMILIES = frozenset({
    "sparse_indexer", "linear_attention", "attention", "paged_mla", "mla_prolog",
    "rotary", "grouped_moe", "hyper_connections", "mc2",
})
_GOVERNED_BLOCK_SELECTORS = frozenset({"lightning_indexer", "gdn_decode"})
_OWNER_MODULE_PREFIXES = ("mindie_sim.prediction.operators.", "mindie_sim.templates.")
_RECIPE_BOOK_PATH = PurePosixPath("mindie_sim/prediction/program/recipes.py")
_IMPLEMENTATION_PATH = PurePosixPath("mindie_sim/prediction/implementation_registry.py")
_GATE_PATH = PurePosixPath("scripts/check_arch_governance.py")


@dataclass(frozen=True, slots=True)
class Violation:
    """Single architecture governance finding."""

    rule: str
    path: PurePosixPath
    line: int
    fingerprint: str
    message: str


def _binding_module(recipe) -> str:
    def module_of(builder):
        # functools.wraps copies __module__ even when the wrapper runs elsewhere.
        return getattr(builder, "__globals__", {}).get(
            "__name__", getattr(builder, "__module__", type(builder).__module__),
        )

    def outside_owner(builder):
        return not isinstance(builder, partial) and not module_of(builder).startswith(
            _OWNER_MODULE_PREFIXES
        )

    builder = unwrap(getattr(recipe, "build", getattr(recipe, "lower", recipe)), stop=outside_owner)
    while isinstance(builder, partial):
        builder = unwrap(builder.func, stop=outside_owner)
    return module_of(builder)


def verify_ag006_ownership() -> tuple[Violation, ...]:
    """Check active recipe bindings, dispatcher tables and live kernel selection."""
    from mindie_sim.prediction.implementation_registry import IMPLEMENTATIONS
    from mindie_sim.prediction.operators.registry import builtin_family_registry
    from mindie_sim.prediction.operators.linear_attention import gdn_decode
    from mindie_sim.prediction.program import RecipeBook

    specs = tuple(spec for spec in builtin_family_registry().specs()
                  if spec.name in _GOVERNED_FAMILIES)
    owned = {recipe_id for spec in specs for recipe_id in spec.owned_recipe_ids}
    book = RecipeBook.from_mapping({})
    violations = []

    def check(binding, path, fingerprint, description):
        module = _binding_module(binding)
        if not module.startswith(_OWNER_MODULE_PREFIXES):
            violations.append(Violation(
                "AG006", path, 1, f"AG006:{fingerprint}",
                f"{description} uses builder outside governed packages: {module}",
            ))

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
            check(book.recipes[recipe_id], _RECIPE_BOOK_PATH, f"recipe:{recipe_id}",
                  f"{spec.name} recipe {recipe_id!r}")

    path = PurePosixPath(*gdn_decode.__name__.split(".")).with_suffix(".py")
    for recipe_id, builder in sorted(gdn_decode._BUILDERS.items()):
        check(builder, path, f"dispatcher:gdn_decode:{recipe_id}",
              f"GDN decode dispatcher entry {recipe_id!r}")

    # Operator replacement selectors can share block names while addressing
    # different contracts, such as an imported xLLM custom operation.
    for name, implementation in sorted(IMPLEMENTATIONS.items()):
        if _GOVERNED_BLOCK_SELECTORS.isdisjoint(implementation.blocks):
            continue
        module = implementation.module
        if not module.startswith(_OWNER_MODULE_PREFIXES):
            violations.append(Violation(
                "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}",
                f"migrated implementation {name!r} points outside governed packages: {module}",
            ))
            continue
        try:
            recipes = import_module(module).recipes()
        except (ImportError, AttributeError) as exc:
            violations.append(Violation(
                "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}:recipes",
                f"migrated implementation {name!r} cannot expose its recipes: {exc}",
            ))
            continue
        for recipe_id, recipe in sorted(recipes.items()):
            fingerprint = f"implementation:{name}:{recipe_id}"
            description = f"migrated implementation {name!r} recipe {recipe_id!r}"
            if recipe_id not in owned:
                violations.append(Violation(
                    "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:{fingerprint}",
                    f"migrated implementation {name!r} exposes unowned recipe {recipe_id!r}",
                ))
            check(recipe, _IMPLEMENTATION_PATH, f"{fingerprint}:builder", description)
    return tuple(violations) if violations else _verify_live_dispatch(book)


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
        if not modules or any(not module.startswith(_OWNER_MODULE_PREFIXES) for module in modules):
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
        ownership = verify_ag006_ownership()
        discrimination = verify_ag007_discrimination()
    finally:
        sys.path.remove(str(root))
    return ownership + tuple(
        Violation("AG007", _GATE_PATH, 1, f"AG007:{failure}", failure)
        for failure in discrimination
    )
