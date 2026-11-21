"""Live recipe ownership and physical discrimination checks for AG006/AG007.

[Role] Inspect active registry bindings and exercise bounded physical probes.
[Invariants] Cache comparisons keep the access constant. Query comparisons keep
  the context and geometry constant and vary only the number of query rows.
[Timeouts & Locks] Finite registry lookups and local planning without locks.
[Side-Effect Policy] Uses private simulation state and reads no measurements.
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
    builder = unwrap(getattr(recipe, "build", type(recipe)))
    while isinstance(builder, partial):
        builder = unwrap(builder.func)
    return getattr(builder, "__module__", type(builder).__module__)


def verify_ag006_ownership() -> tuple[Violation, ...]:
    """Check the actual default book and migrated block lowering registry."""
    from mindie_sim.prediction.implementation_registry import IMPLEMENTATIONS
    from mindie_sim.prediction.operators.registry import builtin_family_registry
    from mindie_sim.prediction.program import RecipeBook

    specs = builtin_family_registry().specs()
    owned = {recipe_id for spec in specs for recipe_id in spec.owned_recipe_ids}
    book = RecipeBook.from_mapping({})
    violations = []
    for spec in specs:
        if spec.name not in _GOVERNED_FAMILIES:
            continue
        for recipe_id in sorted(spec.owned_recipe_ids & book.recipes.keys()):
            module = _binding_module(book.recipes[recipe_id])
            if not module.startswith(_OWNER_MODULE_PREFIXES):
                violations.append(Violation(
                    "AG006", _RECIPE_BOOK_PATH, 1, f"AG006:recipe:{recipe_id}",
                    f"{spec.name} recipe {recipe_id!r} uses builder outside governed packages: {module}",
                ))
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
        try:
            recipe_ids = import_module(module).recipes().keys()
        except (ImportError, AttributeError) as exc:
            violations.append(Violation(
                "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}:recipes",
                f"migrated implementation {name!r} cannot expose its recipes: {exc}",
            ))
            continue
        for recipe_id in sorted(recipe_ids - owned):
            violations.append(Violation(
                "AG006", _IMPLEMENTATION_PATH, 1, f"AG006:implementation:{name}:{recipe_id}",
                f"migrated implementation {name!r} exposes unowned recipe {recipe_id!r}",
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


def verify_ag007_discrimination(
    query_work: Callable[[int, int], float] | None = None,
    *,
    context_tokens: int = 1024,
    multi_query_tokens: int = 2,
    pattern: MemoryAccessPattern | None = None,
) -> tuple[str, ...]:
    """Return failed physical checks, using BF16 indexer FLOPs by default.

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
