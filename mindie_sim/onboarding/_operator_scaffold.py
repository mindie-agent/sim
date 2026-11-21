"""Governed operator-family and template scaffolding for Stage 1A onboarding.

[Role] Scaffold clean, architecture-governance-compliant operator family
  packages under ``mindie_sim/prediction/operators/<family>/`` or template
  specs under ``mindie_sim/templates/<family>_spec.py`` with 4-part top-30
  docstrings, AST-grounded ``Upstream-ABI:``, pinned ``SourceRef`` tuples,
  ``SemanticWorkItem`` + ``TensorRangeObligation`` builders, paired
  ``mindie-profiler`` bench builder declaration, and orthogonal Tier-0 test
  skeleton + ``tests/TAXONOMY.toml`` registration snippet.
[Invariants] Rejects hardware/ISA tokens in family stems (AG001), rejects
  generic or duplicate ``Upstream-ABI:`` symbols across leaves, validates
  40-hex commit revisions and 64-hex SHA-256 digests, and never overwrites
  existing files unless ``overwrite=True``.
[Timeouts & Locks] Pure local template generation; no locks or device contexts.
[Side-Effect Policy] Writes only inside caller-specified ``target_root``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import keyword
from pathlib import Path
import re
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

DEFAULT_SCAFFOLD_TILE_DIM = 128

_FORBIDDEN_FAMILY_RE = re.compile(
    r"(?:arch22|arch35|cann|(?:^|_)(?:a2|a3|a5)(?:$|_))",
    re.IGNORECASE,
)
_ABI_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{3,}$")
_ABI_GENERIC = frozenset({
    "v1", "v2", "v3", "v4", "legacy", "internal", "custom",
    "default", "none", "todo", "refactor", "fallback",
})
_HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_family_identifier(family: str) -> None:
    if not family or not family.isidentifier() or keyword.iskeyword(family):
        raise ModelingError(f"operator_family_identifier_invalid:{family}")
    if _FORBIDDEN_FAMILY_RE.search(family):
        raise ModelingError(f"operator_family_forbidden_hardware_token:{family}")


def _validate_abi_symbol(abi: str, label: str) -> None:
    if not abi or not _ABI_SYMBOL_RE.fullmatch(abi) or abi.lower() in _ABI_GENERIC:
        raise ModelingError(f"operator_upstream_abi_invalid:{label}:{abi}")


def _render_source_refs_code(source_refs: Sequence[Mapping[str, Any]]) -> str:
    if not source_refs:
        raise ModelingError("operator_scaffold_requires_source_refs")
    lines = ["SOURCES: tuple[SourceRef, ...] = ("]
    for ref in source_refs:
        repo = str(ref.get("repository") or "").strip()
        rev = str(ref.get("revision") or "").strip()
        rpath = str(ref.get("relative_path") or "").strip()
        sha = str(ref.get("source_sha256") or "").strip()
        if not repo or not _HEX_40_RE.fullmatch(rev) or not rpath or not _HEX_64_RE.fullmatch(sha):
            raise ModelingError(f"operator_scaffold_source_ref_invalid:{repo}:{rpath}")
        lines.extend((
            "    SourceRef(",
            f"        repository={repo!r},",
            f"        revision={rev!r},",
            f"        relative_path={rpath!r},",
            f"        source_sha256={sha.lower()!r},",
            "    ),",
        ))
    lines.append(")")
    return "\n".join(lines)


def _render_leaf(family: str, arch_label: str, arch_id: str, abi: str, sources_block: str) -> str:
    return (
        f'"""{arch_label} ISA lowering strategy for ``{family}``.\n\n'
        f"Upstream-ABI: {abi}\n\n"
        f"[Role] Lower ``{family}`` via ``{abi}`` on ``{arch_id}``.\n"
        "[Invariants] Verified upstream SourceRef provenance and conserved work.\n"
        "[Timeouts & Locks] O(1) planning; no I/O or locks.\n"
        "[Side-Effect Policy] Pure immutable ``OperatorPlan`` output.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n"
        "from mindie_sim.schema.datapath import ArchId, SourceRef\n"
        "from mindie_sim.schema.operators import OpSpec\n"
        "from ...mechanism import OperatorPlan\n"
        "from ..contracts import OperatorPlanningContext\n"
        "from .common import build_plan\n\n"
        f"ARCH_ID = ArchId({arch_id!r})\n"
        f"UPSTREAM_ABI = {abi!r}\n"
        f"{sources_block}\n\n\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class _Strategy:\n"
        "    arch_id: ArchId = ARCH_ID\n"
        "    abi: str = UPSTREAM_ABI\n"
        "    sources: tuple[SourceRef, ...] = SOURCES\n\n"
        "    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:\n"
        "        del op\n"
        "        return ctx.compute_arch.arch_id == self.arch_id\n\n"
        "    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:\n"
        "        return build_plan(op, ctx, self.abi, self.sources)\n\n\n"
        "STRATEGY = _Strategy()\n"
    )


def _render_common(family: str, tile_dim: int, builder_name: str) -> str:
    return (
        f'"""Shared work conservation, semantic obligations, and TilingAxes for ``{family}``.\n\n'
        f"[Role] Declare ``TILING_AXES``, conserved work, ``SemanticWorkItem``, and\n"
        f"  ``TensorRangeObligation`` hooks for ``{family}`` across Ascend ISAs.\n"
        "[Invariants] Pure structural derivation with zero SKU-name or peak sniffing;\n"
        "  scaffold stubs raise ``NotImplementedError`` until real operator physics\n"
        "  and tensor obligations are authored.\n"
        "[Timeouts & Locks] O(1) deterministic planning; no I/O or locks.\n"
        "[Side-Effect Policy] Returns immutable work and plan records.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "from mindie_sim.onboarding.tiling_axes import TilingAxes\n"
        "from mindie_sim.schema.operators import OpSpec\n"
        "from mindie_sim.templates.contracts import SemanticWorkItem, TensorRangeObligation\n"
        "from ...mechanism import OperatorPlan\n"
        "from ...results import FusedPipelineDescriptor, FusedPipelineSpec\n"
        "from ..contracts import OperatorPlanningContext\n\n"
        f"FAMILY = {family!r}\n"
        f"DEFAULT_TILE_DIM = {int(tile_dim)}\n"
        f"PROFILER_BENCH_BUILDER = {builder_name!r}\n"
        "TILING_AXES = TilingAxes(\n"
        "    tile_dims=(DEFAULT_TILE_DIM,),\n"
        '    core_product_axes=("tokens", "batch_x_kv_heads"),\n'
        ")\n\n\n"
        "def build_semantic_work(op: OpSpec, *, die: int = 0) -> tuple[SemanticWorkItem, ...]:\n"
        "    del op, die\n"
        f'    raise NotImplementedError("implement_{family}_semantic_work_from_upstream_abi")\n\n\n'
        "def build_tensor_obligations(\n"
        "    op: OpSpec, *, die: int = 0\n"
        ") -> tuple[tuple[TensorRangeObligation, ...], tuple[TensorRangeObligation, ...]]:\n"
        "    del op, die\n"
        f'    raise NotImplementedError("implement_{family}_tensor_range_obligations")\n\n\n'
        "def build_plan(op: OpSpec, ctx: OperatorPlanningContext, abi: str, sources) -> OperatorPlan:\n"
        "    del op, ctx, abi, sources\n"
        f'    raise NotImplementedError("implement_{family}_operator_plan_lowering")\n\n\n'
        "def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:\n"
        "    spec = plan.pipeline_spec\n"
        "    assert isinstance(spec, FusedPipelineSpec)\n"
        "    return FusedPipelineDescriptor(\n"
        "        owner=ctx.occurrence, family=spec.family, implementation=spec.implementation,\n"
        "        member_op_ids=(op.op_id,), absorbed_source_op_ids=(), phases=spec.phases,\n"
        "        sources=spec.sources, repeats=spec.repeats,\n"
        "    )\n\n\n"
        '__all__ = [\n    "DEFAULT_TILE_DIM", "FAMILY", "PROFILER_BENCH_BUILDER", "TILING_AXES",\n'
        '    "build_plan", "build_semantic_work", "build_tensor_obligations", "describe",\n]\n'
    )


def _render_init(family: str) -> str:
    return (
        f'"""Single public entrypoint for the governed ``{family}`` operator family.\n\n'
        f"[Role] Register ``{family}`` in ``FamilyRegistry`` and dispatch across ISA leaves.\n"
        "[Invariants] Never re-exports internal ISA leaf modules in ``__all__``.\n"
        "[Timeouts & Locks] O(1) dispatch; no I/O or locks.\n"
        "[Side-Effect Policy] Pure immutable registry and plan records.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "from mindie_sim.schema.common import ModelingError\n"
        "from mindie_sim.schema.operators import OpSpec\n"
        "from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator\n"
        "from . import arch22 as _arch22, arch35 as _arch35\n"
        "from .common import FAMILY, PROFILER_BENCH_BUILDER, describe\n\n"
        "_STRATEGIES = {_arch22.ARCH_ID: _arch22.STRATEGY, _arch35.ARCH_ID: _arch35.STRATEGY}\n\n\n"
        "def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:\n"
        "    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)\n"
        "    if strategy is None:\n"
        '        raise ModelingError(f"unsupported_arch:{FAMILY}:{ctx.compute_arch.arch_id}")\n'
        "    plan = strategy.build(op, ctx)\n"
        "    descriptor = describe(op, ctx, plan)\n"
        "    return ResolvedOperator(\n"
        "        occurrence=ctx.occurrence, op=op, recipe=plan,\n"
        "        implementation=strategy.abi, sources=strategy.sources, descriptor=descriptor,\n"
        "    )\n\n\n"
        "def register_family() -> OperatorFamilySpec:\n"
        "    return OperatorFamilySpec(\n"
        "        name=FAMILY, families=frozenset({FAMILY}),\n"
        "        owned_recipe_ids=frozenset({FAMILY}), resolve=resolve_operator_plan,\n"
        "    )\n\n\n"
        '__all__ = ["PROFILER_BENCH_BUILDER", "register_family", "resolve_operator_plan"]\n'
    )


def _render_template_spec(
    family: str, abi: str, builder_name: str, sources_block: str, tile_dim: int
) -> str:
    return (
        f'"""Governed template specification for ``{family}``.\n\n'
        f"Upstream-ABI: {abi}\n\n"
        f"[Role] Declare pinned provenance, ``TILING_AXES``, ``SemanticWorkItem``,\n"
        f"  and ``TensorRangeObligation`` hooks for ``{family}``.\n"
        "[Invariants] Verified SourceRef provenance; scaffold stubs raise\n"
        "  ``NotImplementedError`` until real template lowering is authored.\n"
        "[Timeouts & Locks] Pure deterministic declarations; no locks.\n"
        "[Side-Effect Policy] Zero side effects.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "from mindie_sim.onboarding.tiling_axes import TilingAxes\n"
        "from mindie_sim.schema.datapath import SourceRef\n"
        "from mindie_sim.templates.contracts import SemanticWorkItem, TensorRangeObligation\n\n"
        f"UPSTREAM_ABI = {abi!r}\n"
        f"PROFILER_BENCH_BUILDER = {builder_name!r}\n"
        f"TILING_AXES = TilingAxes(tile_dims=({int(tile_dim)},))\n"
        f"{sources_block}\n\n\n"
        "def build_template_obligations(\n"
        "    problem: object,\n"
        ") -> tuple[tuple[SemanticWorkItem, ...], tuple[TensorRangeObligation, ...]]:\n"
        "    del problem\n"
        f'    raise NotImplementedError("implement_{family}_template_obligations")\n'
    )


def scaffold_operator(
    *,
    family: str,
    upstream_abi_arch22: str,
    upstream_abi_arch35: str,
    source_refs: Sequence[Mapping[str, Any]],
    target_root: str | Path,
    target_kind: Literal["operator_family", "template"] = "operator_family",
    profiler_builder: str = "",
    tile_dim: int = DEFAULT_SCAFFOLD_TILE_DIM,
    overwrite: bool = False,
    scaffold_test: bool = True,
) -> dict[str, Any]:
    """Stage 1A: Scaffold a governed operator package or template skeleton."""
    _validate_family_identifier(family)
    _validate_abi_symbol(upstream_abi_arch22, "arch22")
    _validate_abi_symbol(upstream_abi_arch35, "arch35")
    if target_kind == "operator_family" and upstream_abi_arch22 == upstream_abi_arch35:
        raise ModelingError("operator_scaffold_duplicate_leaf_abi")
    builder_name = (profiler_builder or family).strip()
    root = Path(target_root).resolve()
    sources_block = _render_source_refs_code(source_refs)

    if target_kind == "template":
        rel_target = f"mindie_sim/templates/{family}_spec.py"
        files_to_write: dict[Path, str] = {
            root / rel_target: _render_template_spec(
                family, upstream_abi_arch35, builder_name, sources_block, tile_dim
            ),
        }
    else:
        rel_target = f"mindie_sim/prediction/operators/{family}"
        pkg_dir = root / rel_target
        files_to_write = {
            pkg_dir / "common.py": _render_common(family, tile_dim, builder_name),
            pkg_dir / "arch22.py": _render_leaf(
                family, "Arch22", "npu_arch_2201", upstream_abi_arch22, sources_block
            ),
            pkg_dir / "arch35.py": _render_leaf(
                family, "Arch35", "npu_arch_3510", upstream_abi_arch35, sources_block
            ),
            pkg_dir / "__init__.py": _render_init(family),
        }

    if scaffold_test:
        test_path = root / "tests" / "operators" / f"test_{family}.py"
        files_to_write[test_path] = (
            f'"""Orthogonal unit tests for the governed ``{family}`` operator.\n\n'
            f"[Role] Verify work conservation, semantic obligations, and ABI dispatch for ``{family}``.\n"
            "[Invariants] Pure CPU Tier-0 verification; zero device contexts.\n"
            "[Timeouts & Locks] Fast local tests; no locks.\n"
            "[Side-Effect Policy] Zero filesystem writes.\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "import pytest\n\n"
            "pytestmark = pytest.mark.tier0\n\n\n"
            f"def test_{family}_conservation_and_ag007_contract():\n"
            f'    raise NotImplementedError("implement_{family}_conservation_and_ag007_tests")\n'
        )

    for path in files_to_write:
        if path.exists() and not overwrite:
            raise ModelingError(f"operator_scaffold_target_exists:{path.name}")

    written: list[str] = []
    for path, content in files_to_write.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
        written.append(path.relative_to(root).as_posix())

    taxonomy_snippet = (
        "[[module]]\n"
        f'path = "operators/test_{family}.py"\n'
        'tier = "tier0"\n'
        f'owns = ["{rel_target}"]\n'
    )
    return {
        "stage": "scaffold",
        "ok": True,
        "status": "scaffolded",
        "family": family,
        "target_kind": target_kind,
        "package_rel": rel_target,
        "profiler_builder": builder_name,
        "upstream_abis": {"arch22": upstream_abi_arch22, "arch35": upstream_abi_arch35},
        "files_written": sorted(written),
        "taxonomy_toml_snippet": taxonomy_snippet,
    }


__all__ = ["DEFAULT_SCAFFOLD_TILE_DIM", "scaffold_operator"]
