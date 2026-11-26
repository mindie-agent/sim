#!/usr/bin/env python3
"""Architecture governance gate AG015: UML bijection, reachability & hardware target lock.

[Role]
  Enforce permanent zero-allowlist compliance for:
  - ``AG015-A``: Exact ``Code <-> docs/architecture.md`` ownership manifest and
    canonical Mermaid ``flowchart`` bijection, total disjoint file partition,
    runtime cross-subsystem import edge checks, and bilingual mirror lock.
  - ``AG015-B``: 100% production-root lexical reachability under an explicit
    ``PUBLIC_ENTRYPOINT_CONTRACT`` (anti-barrel & anti-registry-stuffing) with
    witness-verified ``IMPLEMENTATIONS`` keys and governed hooks.
  - ``AG015-C``: Ascend/hardware-neutral ``TARGET_BACKEND`` + pinned ``SourceRef``
    provenance and structural bans against shipping GPU/CUDA profiles or kernels.

[Invariants]
  - Literal-zero ceilings: ``AG015_MAX_UNMAPPED_MODULES = 0``,
    ``AG015_MAX_UNREACHABLE_MODULES = 0``, ``AG015_MAX_UNREACHABLE_HOOKS = 0``,
    and ``AG015_MAX_FOREIGN_HW_MODULES = 0`` (zero allowlists or overrides).
  - ``tests/``, ``tools/``, ``scripts/``, and ``examples/`` never count as
    ``AG015-B`` production library roots.
  - Evaluated unconditionally on the current revision; never subtracted by
    ``--base`` differential filtering or legacy allowlists.

[Timeouts & Locks]
  Bounded local filesystem/AST scans and single-repo ``git ls-files -z`` listing
  (15s timeout, windowless on Windows); no locks or network access.

[Side-Effect Policy]
  Read-only governance scanner; never mutates repository files or git state.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tokenize
import tomllib
from types import MappingProxyType

if __package__:
    from ._arch_governance_runtime import _LEGACY_UPWARD_IMPORTS, Violation
else:
    from _arch_governance_runtime import _LEGACY_UPWARD_IMPORTS, Violation

# Ratchet-locked literal-zero ceilings (defined exactly once at module top-level).
AG015_MAX_UNMAPPED_MODULES = 0
AG015_MAX_UNREACHABLE_MODULES = 0
AG015_MAX_UNREACHABLE_HOOKS = 0
AG015_MAX_FOREIGN_HW_MODULES = 0

RULE_A = "AG015-A"
RULE_B = "AG015-B"
RULE_C = "AG015-C"

_FINGERPRINT_HEX_LEN = 16
_GIT_TIMEOUT_S = 15.0
_MANIFEST_SCHEMA = "mindie-sim.ag015-architecture-manifest/1"
_MANIFEST_START = "<!-- ag015:architecture-manifest:start -->"
_MANIFEST_END = "<!-- ag015:architecture-manifest:end -->"
_ARCH_DOC_REL = PurePosixPath("docs/architecture.md")
_ARCH_EN_DOC_REL = PurePosixPath("docs/architecture.en.md")
_PYPROJECT_REL = PurePosixPath("pyproject.toml")
_INIT_REL = PurePosixPath("mindie_sim/__init__.py")
_IMPL_REGISTRY_REL = PurePosixPath("mindie_sim/prediction/program/implementation_registry.py")

_VALID_LAYERS = frozenset({"L0", "L1", "L2", "L3", "L4", "L5a", "L5b", "DEV"})
_SIM_GOVERNED_SUFFIXES = frozenset({".py", ".c", ".h", ".json"})
_EXCLUDED_ANYWHERE_PARTS = frozenset({"__pycache__", ".git"})
_EXCLUDED_ROOT_DIRS = frozenset({"build", "dist", ".venv", ".pytest_cache"})
_MERMAID_NODE_DECL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:\[|\(|\{)")
_MERMAID_NODE_REF_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)(?:\s*(?:\[.*\]|\(.*\)|\{.*\}))?$")
_MERMAID_SUBGRAPH_RE = re.compile(r"^\s*subgraph\s+([A-Za-z0-9_]+)")
_MERMAID_SOLID_ARROW_SPLIT_RE = re.compile(r"\s*(?:-->|--\s+[^>]+-->)(?:\s*\|[^|]*\|)?\s*")
_MERMAID_DASHED_RE = re.compile(r"-\.|\.->")
_MERMAID_NON_GRAPH_PREFIXES = ("style ", "classDef ", "class ", "linkStyle ", "click ", "direction ", "%%")

_REVIEWED_INIT_ALL = frozenset({
    "observations",
    "CapabilityGap", "UnmodeledCapabilityError",
    "predict", "predict_pd", "graph_costs", "operator_costs",
    "target_graphs", "target_operators", "parse_case", "study",
    "call_costs", "build_call",
    "compose_case", "resolve_hardware_sku", "derive_hardware_sku",
    "EfficiencyCurve", "customize_model_spec",
    "HARDWARE_ARCHETYPE_TEMPLATES", "export_hardware_sku_template",
    "build_custom_hardware_machine", "register_custom_hardware_sku",
    "unregister_custom_hardware_sku", "clear_custom_hardware_skus",
    "list_custom_hardware_skus", "load_custom_hardware_skus_from_file",
    "resolve_custom_hardware_sku", "compose_custom_case",
    "analyze_network_efficiency", "analyze_operator_row",
    "scale_case_hardware", "solve_sla_sizing",
    "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
    "analyze_shared_expert_overlap", "OffloadPolicy",
    "compute_speculative_accept_step",
    "enumerate_parallel_candidates", "sweep_topologies",
    "sweep_topology_grid", "sweep_topologies_iter",
    "sweep_topology_grid_iter", "reconstruct_sweep_summary",
    "solve_hardware_sla_throughput",
    "sweep_pd_disaggregation_ratios", "compute_pareto_frontier",
    "CORE_PROVIDER", "ENTRY_POINT_GROUP", "INTERFACE_VERSION",
    "ExecutionEnvironment", "USER_PROFILE_SCHEMA",
    "Extension", "ExtensionContext", "ExtensionError", "Provider", "UserProfile",
    "FormulaTensorSpec", "FormulaKernelSpec", "FormulaKernelEvaluation",
    "evaluate_formula_kernel", "simulate_formula_kernel",
    "bind_kernel_implementation", "model_target_ceiling",
    "load_extensions", "load_user_profile",
    "simulate_operator", "simulate_native_call", "simulate_aclnn_call",
    "simulate_graph", "simulate_model", "simulate_service",
    "Case", "Software", "Hardware", "Device", "Link",
    "ModelSpec", "LayerPlan", "RequestSpec", "WorkloadSpec", "SloSpec",
    "ParallelSpec", "ModelingError",
    "OptimizationMechanism", "register_optimization_mechanism",
    "list_optimization_mechanisms",
    "RuntimeStackProfile", "StackMechanismBinding",
    "register_runtime_stack", "load_runtime_stack_manifest",
    "compare_feature_tracks", "optimize_ladder",
})
_REVIEWED_LAZY_SUBMODULES = frozenset({"api", "cli", "cohort_accuracy", "structure"})
_REVIEWED_LAZY_ALIASES: Mapping[str, str] = MappingProxyType({
    "cli_export": "mindie_sim.cli.export",
    "cohort_comparison": "mindie_sim.cohort_accuracy",
})

PUBLIC_ENTRYPOINT_CONTRACT: Mapping[str, object] = MappingProxyType({
    "cli_scripts": MappingProxyType({"mindie-sim": ("mindie_sim.cli", "main")}),
    "main_modules": ("mindie_sim.__main__",),
    "init_all": _REVIEWED_INIT_ALL,
    "init_lazy_submodules": _REVIEWED_LAZY_SUBMODULES,
    "init_lazy_aliases": _REVIEWED_LAZY_ALIASES,
    "sdk_root_modules": (
        "mindie_sim.api",
        "mindie_sim.solve",
        "mindie_sim.export",
        "mindie_sim.public",
        "mindie_sim.observations.communication",
    ),
    "python_m_roots": (
        "mindie_sim.observations.scoreboard",
        "mindie_sim.observations.subgraph_parity",
        "mindie_sim.structure",
        "mindie_sim.cohort_accuracy",
    ),
    "profiler_ssot_roots": (
        "mindie_sim.observed_contracts",
        "mindie_sim.observed_contracts.attention",
        "mindie_sim.observed_contracts.fusion",
        "mindie_sim.observed_contracts.matrix",
        "mindie_sim.observed_contracts.moe",
        "mindie_sim.observed_contracts.signatures",
        "mindie_sim.observed_contracts.vocabulary",
        "mindie_sim.structure",
        "mindie_sim.structure_values",
        "mindie_sim.structure_contracts",
        "mindie_sim.structure_implementations",
        "mindie_sim.structure_subgraphs",
        "mindie_sim.subgraph_math",
    ),
})

_BOUNDED_DYNAMIC_IMPORT_SITES = frozenset({
    ("mindie_sim/__init__.py", "__getattr__"),
    ("mindie_sim/onboarding/contracts.py", "family_default_tiling_axes"),
    ("mindie_sim/prediction/program/implementation_registry.py", "implementation"),
    ("mindie_sim/prediction/sharding/__init__.py", "__getattr__"),
    ("mindie_sim/stacks/registry.py", "_instantiate_plugin_from_module"),
})
_GOVERNED_HOOK_NAMES = frozenset({
    "recipes", "lower_op", "lower_attention", "cache_store",
    "validate", "prepare_model", "wrap", "build",
})
_SELECTION_FACTORY_NAMES = frozenset({"select", "prepared_selection"})
_MIN_SELECT_ARGS = 2
_MIN_GETATTR_ARGS = 2
_SOURCE_REF_MAX_DEPTH = 4
_SOURCE_REF_POSITIONAL = ("repository", "revision", "relative_path", "source_sha256")
_ABI_DATA_PARTS_LEN = 3
_PLUGIN_SPEC_MIN_LEN = 3

_ALLOWED_TARGET_BACKENDS = frozenset({"ascend", "hardware_neutral"})
_ALLOWED_EXACT_ASCEND_REPOS = frozenset({
    "Ascend/MindIE-SD",
    "Ascend/op-plugin",
    "cann/ops-nn",
    "cann/ops-transformer",
    "vllm-project/vllm-ascend",
    "vllm-ascend",
    "vllm-ascend-v41-private",
    "tilelang-ascend/kernels",
    "fla-npu",
    "flash-attn-npu",
    "torch-npu",
    "MindIE-SD",
})
_ALLOWED_ORG_PREFIXES = ("Ascend/", "cann/ops-", "tilelang-ascend/")
_FOREIGN_REPO_SUBSTR_RE = re.compile(r"(?:cuda|nvidia|rocm|hip)", re.IGNORECASE)
_REVISION_HEX_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_SOURCE_EXT_RE = re.compile(r"\.(?:cu|cuh|ptx)\b", re.IGNORECASE)
_CUDA_HEADER_RE = re.compile(r"(?:cuda_runtime\.h|cuda_fp16\.h|cublas_v2\.h|nccl\.h)", re.IGNORECASE)
_FOREIGN_GPU_TOKEN_RE = re.compile(
    r"(?:\b(?:compute_capability|sm_?80|sm_?89|sm_?90"
    r"|vllm_cuda_[a-zA-Z0-9_]*|cuda_moe_[a-zA-Z0-9_]*"
    r"|triton_unified_attention_2d|triton_decode_grouped_mla)\b|custom_framework/)",
    re.IGNORECASE,
)
_FOREIGN_BACKEND_FILE_RE = re.compile(r"(?:nvidia|cuda)", re.IGNORECASE)
_FOREIGN_BACKEND_VALUE_RE = re.compile(r"^(?:nvidia|cuda|rocm|hopper|blackwell|ampere|sm_?(?:80|89|90))$", re.IGNORECASE)
_FOREIGN_RECIPE_ID_RE = re.compile(
    r"^(?:vllm_cuda_|cuda_moe_|triton_unified_attention_2d$|triton_decode_grouped_mla$)",
    re.IGNORECASE,
)
_SELECTABLE_OP_HOOKS = frozenset({"lower_op", "lower_attention"})


@dataclass(frozen=True, slots=True)
class _SubsystemSpec:
    subsystem_id: str
    layer: str
    selectors: tuple[str, ...]
    allowed_edges: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParsedArchitecture:
    raw_json: str
    bilingual_mirror_required: bool
    subsystems: tuple[_SubsystemSpec, ...]
    mermaid_nodes: frozenset[str]
    mermaid_edges: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class _ImplementationEntry:
    key: str
    module: str
    operators: tuple[str, ...]
    blocks: tuple[str, ...]
    options: tuple[str, ...]
    lineno: int


@dataclass(frozen=True, slots=True)
class _ModuleIndex:
    mod_name: str
    rel: PurePosixPath
    tree: ast.Module
    is_package_init: bool
    is_pure_barrel: bool
    barrel_map: Mapping[str, tuple[str, str | None]]
    import_bindings: Mapping[str, tuple[str, str | None]]
    module_aliases: Mapping[str, str]
    defined_funcs: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef]
    defined_classes: Mapping[str, ast.ClassDef]
    top_assigns: Mapping[str, list[ast.AST]]
    static_all: frozenset[str] | None
    tc_names: frozenset[str]
    typing_mods: frozenset[str]


def _fp(rule: str, tag: str) -> str:
    return hashlib.sha256(f"{rule}:{tag}".encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX_LEN]


def _make_violation(rule: str, path: PurePosixPath, line: int, tag: str, message: str) -> Violation:
    return Violation(rule, path, line, _fp(rule, tag), message)


def _git_ls_files(root: Path, *paths: str) -> list[str] | None:
    if not ((root / ".git").is_file() or (root / ".git" / "HEAD").is_file()):
        return None
    kwargs: dict[str, object] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *paths],
            capture_output=True, text=True, encoding="utf-8", timeout=_GIT_TIMEOUT_S, check=False, **kwargs,
        )
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard", "--", *paths],
            capture_output=True, text=True, encoding="utf-8", timeout=_GIT_TIMEOUT_S, check=False, **kwargs,
        )
    except TimeoutError:
        raise
    except OSError:
        return None
    if tracked.returncode != 0 or untracked.returncode != 0:
        return None
    raw = (*tracked.stdout.split("\0"), *untracked.stdout.split("\0"))
    return [rel for rel in sorted({entry for entry in raw if entry}) if (root / rel).is_file()]


def _walk_files(root: Path, subdir: str) -> list[str]:
    base = root / subdir
    if not base.is_dir():
        return []
    out: list[str] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_excluded_rel(PurePosixPath(rel.as_posix())):
            continue
        out.append(rel.as_posix())
    return out


def _is_excluded_rel(rel: PurePosixPath) -> bool:
    if not rel.parts:
        return True
    if rel.parts[0] in _EXCLUDED_ROOT_DIRS:
        return True
    return any(part in _EXCLUDED_ANYWHERE_PARTS or part.endswith(".egg-info") for part in rel.parts)


def _enumerate_governed_files(root: Path) -> list[PurePosixPath]:
    git_files = _git_ls_files(root, "mindie_sim", "data/abi", "data/primitives", "tools")
    raw_list = (
        git_files if git_files is not None else (
            _walk_files(root, "mindie_sim")
            + _walk_files(root, "data/abi")
            + _walk_files(root, "data/primitives")
            + _walk_files(root, "tools")
        )
    )
    governed: list[PurePosixPath] = []
    for rel_str in sorted(set(raw_list)):
        rel = PurePosixPath(rel_str)
        if _is_excluded_rel(rel):
            continue
        if rel.parts[0] == "mindie_sim" and rel.suffix in _SIM_GOVERNED_SUFFIXES:
            governed.append(rel)
        elif rel.parts[:2] == ("data", "abi") and len(rel.parts) == _ABI_DATA_PARTS_LEN and rel.suffix == ".json":
            governed.append(rel)
        elif rel.parts[:2] == ("data", "primitives"):
            governed.append(rel)
        elif rel.parts[0] == "tools" and rel.suffix == ".py":
            governed.append(rel)
    return governed


def _validate_selector(selector: str) -> str | None:
    if not selector or selector.startswith("/") or "\\" in selector:
        return f"invalid selector path syntax {selector!r}"
    parts = selector.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return f"invalid selector segment in {selector!r}"
    if selector in {"**", "mindie_sim/**"}:
        return f"broad wildcard selector {selector!r} is forbidden"
    if "**" in parts and (parts[-1] != "**" or parts.count("**") > 1):
        return f"recursive '**' only allowed as final segment in {selector!r}"
    if any("**" in part and part != "**" for part in parts):
        return f"malformed '**' token inside segment of {selector!r}"
    return None


def _match_selector(rel: str, selector: str) -> bool:
    rel_parts = rel.split("/")
    if selector.endswith("/**"):
        pref_parts = selector[:-3].split("/")
        return len(rel_parts) > len(pref_parts) and all(
            fnmatch.fnmatchcase(rp, sp) for rp, sp in zip(rel_parts, pref_parts)
        )
    if any(ch in selector for ch in ("*", "?", "[")):
        sel_parts = selector.split("/")
        return len(sel_parts) == len(rel_parts) and all(
            fnmatch.fnmatchcase(rp, sp) for rp, sp in zip(rel_parts, sel_parts)
        )
    return rel == selector


def _extract_manifest_and_mermaid(
    doc_rel: PurePosixPath, text: str,
) -> tuple[_ParsedArchitecture | None, list[Violation]]:
    violations: list[Violation] = []
    starts = [m.start() for m in re.finditer(re.escape(_MANIFEST_START), text)]
    ends = [m.start() for m in re.finditer(re.escape(_MANIFEST_END), text)]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"manifest_markers:{doc_rel.as_posix()}",
            f"{doc_rel.as_posix()} must contain exactly one valid ag015:architecture-manifest block",
        ))
        return None, violations

    block_body = text[starts[0] + len(_MANIFEST_START) : ends[0]]
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", block_body, re.DOTALL)
    if json_match is None:
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"manifest_json_fence:{doc_rel.as_posix()}",
            f"{doc_rel.as_posix()} ag015 manifest block must contain a fenced ```json object",
        ))
        return None, violations

    raw_json = json_match.group(1).strip()
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"manifest_json_parse:{doc_rel.as_posix()}",
            f"malformed ag015 manifest JSON in {doc_rel.as_posix()}: {exc}",
        ))
        return None, violations

    subsystems, parse_errs = _parse_manifest_payload(doc_rel, payload)
    violations.extend(parse_errs)

    after_end = text[ends[0] + len(_MANIFEST_END) :]
    mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", after_end, re.DOTALL)
    if mermaid_match is None or "flowchart" not in mermaid_match.group(1):
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"mermaid_block_missing:{doc_rel.as_posix()}",
            f"{doc_rel.as_posix()} must place a canonical Mermaid flowchart immediately after the manifest",
        ))
        return None, violations

    nodes, edges, m_errs = _parse_mermaid_flowchart(doc_rel, mermaid_match.group(1))
    violations.extend(m_errs)
    if subsystems is None:
        return None, violations
    return _ParsedArchitecture(
        raw_json=raw_json,
        bilingual_mirror_required=bool(payload.get("bilingual_mirror_required", False)),
        subsystems=subsystems,
        mermaid_nodes=nodes,
        mermaid_edges=edges,
    ), violations


def _parse_manifest_payload(
    doc_rel: PurePosixPath, payload: object,
) -> tuple[tuple[_SubsystemSpec, ...] | None, list[Violation]]:
    violations: list[Violation] = []
    if not isinstance(payload, dict) or payload.get("schema") != _MANIFEST_SCHEMA:
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"manifest_schema:{doc_rel.as_posix()}",
            f"manifest schema must equal {_MANIFEST_SCHEMA!r}",
        ))
        return None, violations
    raw_subs = payload.get("subsystems")
    if not isinstance(raw_subs, list) or not raw_subs:
        violations.append(_make_violation(
            RULE_A, doc_rel, 1, f"manifest_subsystems:{doc_rel.as_posix()}",
            "manifest must declare a non-empty 'subsystems' list",
        ))
        return None, violations

    seen_ids: set[str] = set()
    seen_selectors: set[str] = set()
    parsed: list[_SubsystemSpec] = []
    for item in raw_subs:
        if not isinstance(item, dict):
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, "manifest_item_type", "each subsystem entry must be a JSON object",
            ))
            continue
        sub_id = item.get("id")
        layer = item.get("layer")
        sels = item.get("selectors")
        edges = item.get("allowed_edges")
        if not isinstance(sub_id, str) or not sub_id or sub_id in seen_ids:
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, f"subsystem_id:{sub_id}",
                f"duplicate or invalid subsystem id {sub_id!r}",
            ))
            continue
        seen_ids.add(sub_id)
        if not isinstance(layer, str) or layer not in _VALID_LAYERS:
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, f"subsystem_layer:{sub_id}",
                f"subsystem {sub_id!r} has invalid layer {layer!r}",
            ))
        if not isinstance(sels, list) or not sels or not all(isinstance(s, str) for s in sels):
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, f"subsystem_selectors:{sub_id}",
                f"subsystem {sub_id!r} must declare a non-empty string list of selectors",
            ))
            sels = []
        for sel in sels:
            err = _validate_selector(sel)
            if err is not None:
                violations.append(_make_violation(RULE_A, doc_rel, 1, f"bad_selector:{sub_id}:{sel}", err))
            if sel in seen_selectors:
                violations.append(_make_violation(
                    RULE_A, doc_rel, 1, f"dup_selector:{sel}",
                    f"duplicate selector {sel!r} in architecture manifest",
                ))
            seen_selectors.add(sel)
        if not isinstance(edges, list) or not all(isinstance(e, str) for e in edges):
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, f"subsystem_edges:{sub_id}",
                f"subsystem {sub_id!r} allowed_edges must be a list of subsystem ids",
            ))
            edges = []
        parsed.append(_SubsystemSpec(sub_id, str(layer), tuple(sels), tuple(edges)))

    all_ids = {spec.subsystem_id for spec in parsed}
    for spec in parsed:
        seen_e: set[str] = set()
        for dst in spec.allowed_edges:
            if dst not in all_ids or dst == spec.subsystem_id or dst in seen_e:
                violations.append(_make_violation(
                    RULE_A, doc_rel, 1, f"bad_edge:{spec.subsystem_id}:{dst}",
                    f"invalid/duplicate/self allowed_edge {spec.subsystem_id!r} -> {dst!r}",
                ))
            seen_e.add(dst)
    return tuple(parsed), violations


def _parse_mermaid_chain(
    line: str,
) -> tuple[list[str], list[tuple[str, str]]] | None:
    segments = _MERMAID_SOLID_ARROW_SPLIT_RE.split(line)
    if len(segments) < _MIN_SELECT_ARGS:
        return None
    chain_ids: list[str] = []
    inline_nodes: list[str] = []
    for seg in segments:
        seg_clean = seg.strip()
        match = _MERMAID_NODE_REF_RE.match(seg_clean)
        if match is None:
            return None
        nid = match.group(1)
        chain_ids.append(nid)
        if len(seg_clean) > len(nid):
            inline_nodes.append(nid)
    return inline_nodes, [(chain_ids[i], chain_ids[i + 1]) for i in range(len(chain_ids) - 1)]


def _parse_mermaid_flowchart(
    doc_rel: PurePosixPath, block: str,
) -> tuple[frozenset[str], frozenset[tuple[str, str]], list[Violation]]:
    nodes: set[str] = set()
    retired_nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    violations: list[Violation] = []
    subgraph_stack: list[bool] = []

    for lineno, raw_line in enumerate(block.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("flowchart ", "graph ")) or line.startswith(_MERMAID_NON_GRAPH_PREFIXES):
            continue
        sub_match = _MERMAID_SUBGRAPH_RE.match(line)
        if sub_match is not None:
            subgraph_stack.append("RETIRED" in sub_match.group(1).upper())
            continue
        if line == "end":
            if subgraph_stack:
                subgraph_stack.pop()
            continue
        in_retired = any(subgraph_stack)
        if _MERMAID_DASHED_RE.search(line):
            continue
        if "-->" in line:
            parsed_chain = _parse_mermaid_chain(line)
            if parsed_chain is None:
                violations.append(_make_violation(
                    RULE_A, doc_rel, lineno, f"unparsed_mermaid_edge:{lineno}",
                    f"unsupported Mermaid solid edge syntax at flowchart line {lineno}: {line!r}",
                ))
                continue
            inline_nodes, chain_edges = parsed_chain
            if in_retired:
                for src, dst in chain_edges:
                    violations.append(_make_violation(
                        RULE_A, doc_rel, lineno, f"retired_solid_edge:{src}:{dst}",
                        f"solid edge {src} --> {dst} forbidden inside RETIRED subgraph",
                    ))
            else:
                nodes.update(inline_nodes)
                edges.update(chain_edges)
                for src, dst in chain_edges:
                    nodes.add(src)
                    nodes.add(dst)
            continue
        node_match = _MERMAID_NODE_DECL_RE.match(line)
        if node_match is not None:
            (retired_nodes if in_retired else nodes).add(node_match.group(1))
        else:
            violations.append(_make_violation(
                RULE_A, doc_rel, lineno, f"unparsed_mermaid_line:{lineno}",
                f"unrecognized Mermaid flowchart statement at line {lineno}: {line!r}",
            ))

    for src, dst in sorted(edges):
        if src in retired_nodes or dst in retired_nodes:
            violations.append(_make_violation(
                RULE_A, doc_rel, 1, f"solid_edge_touches_retired:{src}:{dst}",
                f"solid Mermaid edge {src} --> {dst} must not touch RETIRED node",
            ))
    return frozenset(nodes), frozenset(edges), violations


def _typing_check_aliases(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    tc_names: set[str] = {"TYPE_CHECKING"}
    typing_mods: set[str] = {"typing"}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "typing":
                    typing_mods.add(alias.asname or "typing")
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module == "typing":
            for alias in stmt.names:
                if alias.name == "TYPE_CHECKING":
                    tc_names.add(alias.asname or "TYPE_CHECKING")
    return frozenset(tc_names), frozenset(typing_mods)


def _is_type_checking_if(
    node: ast.If, tc_names: frozenset[str] = frozenset({"TYPE_CHECKING"}),
    typing_mods: frozenset[str] = frozenset({"typing"}),
) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id in tc_names:
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return not isinstance(test.value, ast.Name) or test.value.id in typing_mods
    return isinstance(test, ast.Constant) and test.value is False


def _resolve_relative_module(rel: PurePosixPath, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = list(rel.with_suffix("").parts)
    pkg = parts[:-1]
    up = level - 1
    base = pkg[: len(pkg) - up] if up <= len(pkg) else []
    tail = module.split(".") if module else []
    return ".".join([*base, *tail])


def _resolve_package_relative_str(rel: PurePosixPath, mod_str: str) -> str:
    if not mod_str.startswith("."):
        return mod_str
    dots = len(mod_str) - len(mod_str.lstrip("."))
    tail = mod_str[dots:] or None
    return _resolve_relative_module(rel, tail, dots)


def _walk_nested_stmt_bodies(stmt: ast.stmt, callback) -> None:
    for field in ("body", "orelse", "finalbody"):
        sub = getattr(stmt, field, None)
        if isinstance(sub, list):
            callback(sub)
    for handler in getattr(stmt, "handlers", ()) or ():
        callback(handler.body)
    for case in getattr(stmt, "cases", ()) or ():
        callback(case.body)


def _extract_runtime_imports(
    rel: PurePosixPath,
    tree: ast.Module,
    known_modules: Mapping[str, PurePosixPath],
    init_reexports: Mapping[str, str],
    text: str | None = None,
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    tc_names, typing_mods = _typing_check_aliases(tree)
    importlib_mods: set[str] = {"importlib"}
    import_fns: set[str] = {"__import__"}
    scan_dyn_exprs = text is None or ("import_module" in text) or ("__import__" in text)

    def scan_exprs(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call) or not sub.args:
                continue
            fn = sub.func
            is_imp = (
                (isinstance(fn, ast.Name) and fn.id in import_fns)
                or (isinstance(fn, ast.Attribute) and fn.attr == "import_module"
                    and isinstance(fn.value, ast.Name) and fn.value.id in importlib_mods)
            )
            if is_imp and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
                target = _resolve_package_relative_str(rel, sub.args[0].value)
                if target:
                    found.append((getattr(sub, "lineno", 1), target))

    def walk_stmts(stmts: Sequence[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, tc_names, typing_mods):
                walk_stmts(stmt.orelse)
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    found.append((stmt.lineno, alias.name))
                    if alias.name == "importlib":
                        importlib_mods.add(alias.asname or "importlib")
            elif isinstance(stmt, ast.ImportFrom):
                base = _resolve_relative_module(rel, stmt.module, stmt.level)
                if base == "importlib":
                    for alias in stmt.names:
                        if alias.name == "import_module":
                            import_fns.add(alias.asname or "import_module")
                if base == "mindie_sim":
                    for alias in stmt.names:
                        if alias.name in init_reexports:
                            found.append((stmt.lineno, init_reexports[alias.name]))
                elif base:
                    found.append((stmt.lineno, base))
                for alias in stmt.names:
                    if alias.name != "*" and base:
                        candidate = f"{base}.{alias.name}"
                        if candidate in known_modules:
                            found.append((stmt.lineno, candidate))
            elif scan_dyn_exprs:
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.expr):
                        scan_exprs(child)
            _walk_nested_stmt_bodies(stmt, walk_stmts)

    walk_stmts(tree.body)
    return found


def _is_legacy_upward_edge(src_rel: str, dst_mod: str) -> bool:
    allowed = _LEGACY_UPWARD_IMPORTS.get(src_rel, frozenset())
    return any(dst_mod == pref or dst_mod.startswith(pref + ".") for pref in allowed)


def _init_reexport_modules(
    parsed_py: Mapping[PurePosixPath, ast.Module], known_modules: Mapping[str, PurePosixPath],
) -> dict[str, str]:
    tree = parsed_py.get(_INIT_REL)
    if tree is None:
        return {}
    out: dict[str, str] = {}
    for sym in _REVIEWED_LAZY_SUBMODULES:
        out[sym] = f"mindie_sim.{sym}"
    out.update(_REVIEWED_LAZY_ALIASES)
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            base = _resolve_relative_module(_INIT_REL, stmt.module, stmt.level)
            if not base:
                continue
            for alias in stmt.names:
                local = alias.asname or alias.name
                sub_cand = f"{base}.{alias.name}"
                out[local] = sub_cand if sub_cand in known_modules else base
    return out


def _check_ag015_a(
    root: Path,
    governed: Sequence[PurePosixPath],
    parsed_py: Mapping[PurePosixPath, ast.Module],
    texts: Mapping[PurePosixPath, str],
    all_mod_to_rel: Mapping[str, PurePosixPath],
) -> list[Violation]:
    doc_path = root / _ARCH_DOC_REL
    if not doc_path.is_file():
        return [_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, "missing_architecture_doc",
            "docs/architecture.md is required for AG015-A ownership verification",
        )]
    arch, violations = _extract_manifest_and_mermaid(_ARCH_DOC_REL, doc_path.read_text(encoding="utf-8"))
    if arch is None:
        return violations

    manifest_ids = frozenset(spec.subsystem_id for spec in arch.subsystems)
    manifest_edges = frozenset(
        (spec.subsystem_id, dst) for spec in arch.subsystems for dst in spec.allowed_edges
    )
    for missing in sorted(manifest_ids - arch.mermaid_nodes):
        violations.append(_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, f"mermaid_missing_node:{missing}",
            f"manifest subsystem {missing!r} missing from canonical Mermaid flowchart",
        ))
    for extra in sorted(arch.mermaid_nodes - manifest_ids):
        violations.append(_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, f"mermaid_extra_node:{extra}",
            f"Mermaid solid node {extra!r} is undeclared in ag015 manifest",
        ))
    for missing_edge in sorted(manifest_edges - arch.mermaid_edges):
        violations.append(_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, f"mermaid_missing_edge:{missing_edge[0]}:{missing_edge[1]}",
            f"manifest allowed_edge {missing_edge[0]} -> {missing_edge[1]} missing from Mermaid flowchart",
        ))
    for extra_edge in sorted(arch.mermaid_edges - manifest_edges):
        violations.append(_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, f"mermaid_extra_edge:{extra_edge[0]}:{extra_edge[1]}",
            f"Mermaid solid edge {extra_edge[0]} --> {extra_edge[1]} missing from manifest allowed_edges",
        ))

    violations.extend(_check_bilingual_mirror(root, arch))
    file_owner, part_errs = _check_file_partition(arch.subsystems, governed)
    violations.extend(part_errs)
    allowed_map = {spec.subsystem_id: frozenset(spec.allowed_edges) | {spec.subsystem_id} for spec in arch.subsystems}
    init_reexports = _init_reexport_modules(parsed_py, all_mod_to_rel)
    violations.extend(
        _check_cross_subsystem_edges(parsed_py, texts, all_mod_to_rel, file_owner, allowed_map, init_reexports)
    )
    return violations


def _check_bilingual_mirror(root: Path, primary: _ParsedArchitecture) -> list[Violation]:
    en_path = root / _ARCH_EN_DOC_REL
    if not en_path.is_file():
        if primary.bilingual_mirror_required:
            return [_make_violation(
                RULE_A, _ARCH_EN_DOC_REL, 1, "missing_bilingual_mirror",
                "bilingual_mirror_required is true but docs/architecture.en.md is missing",
            )]
        return []
    mirror, errors = _extract_manifest_and_mermaid(_ARCH_EN_DOC_REL, en_path.read_text(encoding="utf-8"))
    if mirror is None:
        return errors
    if mirror.raw_json != primary.raw_json:
        errors.append(_make_violation(
            RULE_A, _ARCH_EN_DOC_REL, 1, "bilingual_manifest_drift",
            "docs/architecture.en.md ag015 manifest JSON must match docs/architecture.md byte-for-byte",
        ))
    if mirror.mermaid_nodes != primary.mermaid_nodes or mirror.mermaid_edges != primary.mermaid_edges:
        errors.append(_make_violation(
            RULE_A, _ARCH_EN_DOC_REL, 1, "bilingual_mermaid_drift",
            "docs/architecture.en.md Mermaid solid nodes/edges must match docs/architecture.md exactly",
        ))
    return errors


def _check_file_partition(
    subsystems: Sequence[_SubsystemSpec], governed: Sequence[PurePosixPath],
) -> tuple[dict[PurePosixPath, str], list[Violation]]:
    selector_hits: dict[str, int] = {sel: 0 for spec in subsystems for sel in spec.selectors}
    file_owner: dict[PurePosixPath, str] = {}
    violations: list[Violation] = []
    unmapped_count = 0

    for rel in governed:
        rel_str = rel.as_posix()
        matches: list[tuple[str, str]] = []
        for spec in subsystems:
            for sel in spec.selectors:
                if _match_selector(rel_str, sel):
                    selector_hits[sel] += 1
                    matches.append((spec.subsystem_id, sel))
        if not matches:
            unmapped_count += 1
            violations.append(_make_violation(
                RULE_A, rel, 1, f"unmapped_file:{rel_str}",
                f"governed file {rel_str!r} is unmapped in docs/architecture.md manifest",
            ))
        elif len(matches) > 1:
            owners = ", ".join(f"{sid}({sel})" for sid, sel in matches)
            violations.append(_make_violation(
                RULE_A, rel, 1, f"overlapping_selectors:{rel_str}",
                f"governed file {rel_str!r} matches multiple selectors: {owners}",
            ))
        else:
            file_owner[rel] = matches[0][0]

    for sel, count in sorted(selector_hits.items()):
        if count == 0:
            violations.append(_make_violation(
                RULE_A, _ARCH_DOC_REL, 1, f"phantom_selector:{sel}",
                f"manifest selector {sel!r} matches zero files on disk (phantom architecture doc)",
            ))
    if unmapped_count > AG015_MAX_UNMAPPED_MODULES and not violations:
        violations.append(_make_violation(
            RULE_A, _ARCH_DOC_REL, 1, "unmapped_budget_exceeded",
            f"unmapped module count {unmapped_count} exceeds ceiling {AG015_MAX_UNMAPPED_MODULES}",
        ))
    return file_owner, violations


def _check_cross_subsystem_edges(
    parsed_py: Mapping[PurePosixPath, ast.Module],
    texts: Mapping[PurePosixPath, str],
    mod_to_rel: Mapping[str, PurePosixPath],
    file_owner: Mapping[PurePosixPath, str],
    allowed_map: Mapping[str, frozenset[str]],
    init_reexports: Mapping[str, str],
) -> list[Violation]:
    violations: list[Violation] = []
    seen: set[tuple[str, str, str]] = set()
    for rel, tree in sorted(parsed_py.items()):
        src_sub = file_owner.get(rel)
        if src_sub is None:
            continue
        src_str = rel.as_posix()
        allowed_dsts = allowed_map.get(src_sub, frozenset({src_sub}))
        for lineno, dst_mod in _extract_runtime_imports(rel, tree, mod_to_rel, init_reexports, texts.get(rel)):
            if dst_mod == "mindie_sim":
                continue
            dst_rel = mod_to_rel.get(dst_mod)
            if dst_rel is None:
                continue
            dst_sub = file_owner.get(dst_rel)
            if dst_sub is None or dst_sub in allowed_dsts or _is_legacy_upward_edge(src_str, dst_mod):
                continue
            key = (src_str, dst_mod, dst_sub)
            if key not in seen:
                seen.add(key)
                violations.append(_make_violation(
                    RULE_A, rel, lineno, f"forbidden_edge:{src_str}:{dst_mod}",
                    f"runtime import {src_sub} -> {dst_sub} ({dst_mod}) not in docs/architecture.md allowed_edges",
                ))
    return violations


def _module_name_for_rel(rel: PurePosixPath) -> str:
    if rel.name == "__init__.py":
        return ".".join(rel.parts[:-1])
    return ".".join(rel.with_suffix("").parts)


def _extract_string_collection(node: ast.AST | None) -> set[str] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return {e.value for e in node.elts}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"frozenset", "set", "tuple"}:
        if len(node.args) == 1:
            return _extract_string_collection(node.args[0])
    return None


def _extract_string_dict(node: ast.AST | None) -> dict[str, str] | None:
    if isinstance(node, ast.Dict) and all(
        isinstance(k, ast.Constant) and isinstance(k.value, str)
        and isinstance(v, ast.Constant) and isinstance(v.value, str)
        for k, v in zip(node.keys, node.values)
    ):
        return {k.value: v.value for k, v in zip(node.keys, node.values)}
    return None


def _top_level_assignments(tree: ast.Module, name: str) -> list[tuple[int, ast.AST]]:
    found: list[tuple[int, ast.AST]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
                found.append((stmt.lineno, stmt.value))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            if stmt.value is not None:
                found.append((stmt.lineno, stmt.value))
    return found


def _is_pure_barrel_init(tree: ast.Module, tc_names: frozenset[str], typing_mods: frozenset[str]) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, tc_names, typing_mods):
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id == "__all__" or isinstance(stmt.value, ast.Name):
                continue
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
            continue
        return False
    return True


def _build_module_index(
    mod_name: str, rel: PurePosixPath, tree: ast.Module, known_modules: Mapping[str, PurePosixPath],
) -> _ModuleIndex:
    is_pkg = rel.name == "__init__.py"
    tc_names, typing_mods = _typing_check_aliases(tree)
    barrel_map: dict[str, tuple[str, str | None]] = {}
    import_bindings: dict[str, tuple[str, str | None]] = {}
    module_aliases: dict[str, str] = {}
    defined_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    defined_classes: dict[str, ast.ClassDef] = {}
    top_assigns: dict[str, list[ast.AST]] = defaultdict(list)
    static_all: frozenset[str] | None = None

    for stmt in tree.body:
        if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, tc_names, typing_mods):
            continue
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                module_aliases[local] = alias.name
                import_bindings[local] = (alias.name, None)
                if is_pkg:
                    barrel_map[local] = (alias.name, None)
        elif isinstance(stmt, ast.ImportFrom):
            base = _resolve_relative_module(rel, stmt.module, stmt.level)
            if not base:
                continue
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                sub_cand = f"{base}.{alias.name}"
                if sub_cand in known_modules:
                    binding = (sub_cand, None)
                    module_aliases[local] = sub_cand
                else:
                    binding = (base, alias.name)
                import_bindings[local] = binding
                if is_pkg:
                    barrel_map[local] = binding
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_funcs[stmt.name] = stmt
        elif isinstance(stmt, ast.ClassDef):
            defined_classes[stmt.name] = stmt
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__":
                        parsed_all = _extract_string_collection(stmt.value)
                        static_all = frozenset(parsed_all) if parsed_all is not None else None
                    else:
                        top_assigns[target.id].append(stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            if stmt.target.id == "__all__":
                parsed_all = _extract_string_collection(stmt.value)
                static_all = frozenset(parsed_all) if parsed_all is not None else None
            else:
                top_assigns[stmt.target.id].append(stmt.value)

    return _ModuleIndex(
        mod_name=mod_name, rel=rel, tree=tree, is_package_init=is_pkg,
        is_pure_barrel=is_pkg and _is_pure_barrel_init(tree, tc_names, typing_mods),
        barrel_map=barrel_map, import_bindings=import_bindings, module_aliases=module_aliases,
        defined_funcs=defined_funcs, defined_classes=defined_classes,
        top_assigns=dict(top_assigns), static_all=static_all,
        tc_names=tc_names, typing_mods=typing_mods,
    )


def _symbol_exists_in_index(idx: _ModuleIndex, sym: str) -> bool:
    return (
        sym in idx.defined_funcs
        or sym in idx.defined_classes
        or sym in idx.top_assigns
        or sym in idx.import_bindings
    )


def _public_symbols_of(idx: _ModuleIndex) -> list[str]:
    if idx.static_all is not None:
        return sorted(idx.static_all)
    names = {*idx.defined_funcs, *idx.defined_classes, *idx.barrel_map}
    return sorted(name for name in names if not name.startswith("_"))


def _check_public_entrypoint_contract(
    root: Path, indices: Mapping[str, _ModuleIndex],
) -> tuple[list[tuple[str, str | None]], list[Violation]]:
    violations: list[Violation] = []
    seeds: list[tuple[str, str | None]] = []

    pyproject_path = root / _PYPROJECT_REL
    expected_scripts = dict(PUBLIC_ENTRYPOINT_CONTRACT["cli_scripts"])  # type: ignore[arg-type]
    if not pyproject_path.is_file():
        violations.append(_make_violation(
            RULE_B, _PYPROJECT_REL, 1, "missing_pyproject",
            "pyproject.toml is required to verify PUBLIC_ENTRYPOINT_CONTRACT [project.scripts]",
        ))
    else:
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            violations.append(_make_violation(
                RULE_B, _PYPROJECT_REL, 1, "invalid_pyproject", f"failed to parse pyproject.toml: {exc}",
            ))
            data = {}
        scripts = (data.get("project") or {}).get("scripts") or {}
        for name, target in sorted(scripts.items()):
            exp = expected_scripts.get(name)
            expected_str = f"{exp[0]}:{exp[1]}" if exp is not None else None
            if target != expected_str:
                violations.append(_make_violation(
                    RULE_B, _PYPROJECT_REL, 1, f"unreviewed_script:{name}",
                    f"pyproject.toml script {name!r} -> {target!r} does not match PUBLIC_ENTRYPOINT_CONTRACT",
                ))
        for name, (mod, sym) in sorted(expected_scripts.items()):
            if name not in scripts:
                violations.append(_make_violation(
                    RULE_B, _PYPROJECT_REL, 1, f"missing_script:{name}",
                    f"PUBLIC_ENTRYPOINT_CONTRACT script {name!r} missing from pyproject.toml",
                ))
            target_idx = indices.get(mod)
            if target_idx is None or not _symbol_exists_in_index(target_idx, sym):
                violations.append(_make_violation(
                    RULE_B, _PYPROJECT_REL, 1, f"missing_script_symbol:{mod}:{sym}",
                    f"PUBLIC_ENTRYPOINT_CONTRACT script {name!r} target {mod}:{sym} does not exist",
                ))
            else:
                seeds.append((mod, sym))

    init_idx = indices.get("mindie_sim")
    if init_idx is None:
        violations.append(_make_violation(
            RULE_B, _INIT_REL, 1, "missing_package_init", "mindie_sim/__init__.py missing",
        ))
    else:
        violations.extend(_verify_init_contract(init_idx))
        for sym in sorted(_REVIEWED_INIT_ALL):
            seeds.append(("mindie_sim", sym))
        for sub in sorted(_REVIEWED_LAZY_SUBMODULES):
            seeds.append((f"mindie_sim.{sub}", None))
        for target_mod in sorted(_REVIEWED_LAZY_ALIASES.values()):
            seeds.append((target_mod, None))

    for key in ("main_modules", "sdk_root_modules", "python_m_roots", "profiler_ssot_roots"):
        for mod in PUBLIC_ENTRYPOINT_CONTRACT[key]:  # type: ignore[index]
            seeds.append((mod, None))
            if mod in indices:
                for sym in _public_symbols_of(indices[mod]):
                    seeds.append((mod, sym))
                if mod == "mindie_sim.observations.communication" and "__getattr__" in indices[mod].defined_funcs:
                    seeds.append((mod, "__getattr__"))
    return seeds, violations


def _verify_init_contract(init_idx: _ModuleIndex) -> list[Violation]:
    violations: list[Violation] = []
    all_assigns = _top_level_assignments(init_idx.tree, "__all__")
    sub_assigns = _top_level_assignments(init_idx.tree, "_LAZY_SUBMODULES")
    alias_assigns = _top_level_assignments(init_idx.tree, "_LAZY_ALIASES")

    actual_all = _extract_string_collection(all_assigns[-1][1]) if all_assigns else None
    actual_lazy_subs = _extract_string_collection(sub_assigns[-1][1]) if sub_assigns else None
    actual_lazy_aliases = _extract_string_dict(alias_assigns[-1][1]) if alias_assigns else None

    if actual_all is None:
        violations.append(_make_violation(
            RULE_B, _INIT_REL, 1, "init_all_opaque",
            "mindie_sim/__init__.py __all__ must be a static string literal collection",
        ))
    else:
        for extra in sorted(actual_all - _REVIEWED_INIT_ALL):
            violations.append(_make_violation(
                RULE_B, _INIT_REL, 1, f"unreviewed_init_all:{extra}",
                f"unreviewed symbol {extra!r} in mindie_sim/__init__.py:__all__ is forbidden by PUBLIC_ENTRYPOINT_CONTRACT",
            ))
    if actual_lazy_subs is None:
        violations.append(_make_violation(
            RULE_B, _INIT_REL, 1, "init_lazy_subs_opaque",
            "mindie_sim/__init__.py _LAZY_SUBMODULES must be a static string collection",
        ))
    else:
        for extra in sorted(actual_lazy_subs - _REVIEWED_LAZY_SUBMODULES):
            violations.append(_make_violation(
                RULE_B, _INIT_REL, 1, f"unreviewed_lazy_sub:{extra}",
                f"unreviewed lazy submodule {extra!r} in mindie_sim/__init__.py",
            ))
    if actual_lazy_aliases is None:
        violations.append(_make_violation(
            RULE_B, _INIT_REL, 1, "init_lazy_aliases_opaque",
            "mindie_sim/__init__.py _LAZY_ALIASES must be a static string dict",
        ))
    else:
        for key, val in sorted(actual_lazy_aliases.items()):
            if _REVIEWED_LAZY_ALIASES.get(key) != val:
                violations.append(_make_violation(
                    RULE_B, _INIT_REL, 1, f"unreviewed_lazy_alias:{key}",
                    f"unreviewed lazy alias {key!r} -> {val!r} in mindie_sim/__init__.py",
                ))
    return violations


def _extract_kw_str_tuple(call: ast.Call, kw_name: str) -> tuple[str, ...]:
    for kw in call.keywords:
        if kw.arg == kw_name and isinstance(kw.value, (ast.Tuple, ast.List)):
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in kw.value.elts):
                return tuple(e.value for e in kw.value.elts)
    return ()


def _extract_impl_module_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    for kw in call.keywords:
        if kw.arg == "module" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _parse_implementations_registry(
    indices: Mapping[str, _ModuleIndex],
) -> tuple[dict[str, _ImplementationEntry], list[Violation]]:
    idx = indices.get("mindie_sim.prediction.program.implementation_registry")
    if idx is None:
        return {}, [_make_violation(
            RULE_B, _IMPL_REGISTRY_REL, 1, "missing_impl_registry", "implementation_registry.py missing",
        )]
    entries: dict[str, _ImplementationEntry] = {}
    violations: list[Violation] = []
    assigns = _top_level_assignments(idx.tree, "IMPLEMENTATIONS")
    if not assigns:
        return {}, [_make_violation(
            RULE_B, _IMPL_REGISTRY_REL, 1, "impl_dict_missing", "IMPLEMENTATIONS dict missing",
        )]
    for lineno, value in assigns:
        if not isinstance(value, ast.Dict):
            violations.append(_make_violation(
                RULE_B, _IMPL_REGISTRY_REL, lineno, "impl_dict_opaque",
                "IMPLEMENTATIONS must be a static dict literal",
            ))
            continue
        for k_node, v_node in zip(value.keys, value.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str) and isinstance(v_node, ast.Call)):
                violations.append(_make_violation(
                    RULE_B, _IMPL_REGISTRY_REL, getattr(v_node, "lineno", lineno),
                    "impl_entry_opaque", "IMPLEMENTATIONS entries must be static Implementation(...) calls",
                ))
                continue
            mod = _extract_impl_module_arg(v_node)
            if mod is None:
                violations.append(_make_violation(
                    RULE_B, _IMPL_REGISTRY_REL, k_node.lineno, f"impl_module_opaque:{k_node.value}",
                    f"IMPLEMENTATIONS[{k_node.value!r}] must supply a static string module path",
                ))
                continue
            entries[k_node.value] = _ImplementationEntry(
                key=k_node.value,
                module=mod,
                operators=_extract_kw_str_tuple(v_node, "operators"),
                blocks=_extract_kw_str_tuple(v_node, "blocks"),
                options=_extract_kw_str_tuple(v_node, "options"),
                lineno=k_node.lineno,
            )
    return entries, violations


def _parse_builtin_stack_plugins(indices: Mapping[str, _ModuleIndex]) -> list[tuple[str, str]]:
    idx = indices.get("mindie_sim.stacks.registry")
    if idx is None:
        return []
    plugins: list[tuple[str, str]] = []
    for _, val in _top_level_assignments(idx.tree, "_BUILTIN_PLUGIN_SPECS"):
        if not isinstance(val, (ast.Tuple, ast.List)):
            continue
        for elt in val.elts:
            if isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) >= _PLUGIN_SPEC_MIN_LEN:
                mod_n, cls_n = elt.elts[1], elt.elts[2]
                if (isinstance(mod_n, ast.Constant) and isinstance(mod_n.value, str)
                        and isinstance(cls_n, ast.Constant) and isinstance(cls_n.value, str)):
                    plugins.append((mod_n.value, cls_n.value))
    return plugins


def _check_open_ended_dynamic_imports(
    indices: Mapping[str, _ModuleIndex],
    texts: Mapping[PurePosixPath, str],
) -> list[Violation]:
    violations: list[Violation] = []
    for idx in sorted(indices.values(), key=lambda item: item.rel.as_posix()):
        text = texts.get(idx.rel)
        if (
            text is not None
            and "import_module" not in text
            and "__import__" not in text
            and "getattr" not in text
        ):
            continue
        rel_str = idx.rel.as_posix()
        violations.extend(_scan_dynamic_calls_in_scope(
            idx.rel, rel_str, idx.tree.body, "<module>",
            set(idx.module_aliases), {"importlib"}, {"__import__"},
        ))
    return violations


def _scan_dynamic_calls_in_scope(
    rel: PurePosixPath,
    rel_str: str,
    stmts: Sequence[ast.stmt],
    enclosing_fn: str,
    mod_aliases: set[str],
    importlib_aliases: set[str],
    import_fn_aliases: set[str],
) -> list[Violation]:
    local_mods = set(mod_aliases)
    local_il = set(importlib_aliases)
    local_fns = set(import_fn_aliases)
    violations: list[Violation] = []

    def check_expr(expr: ast.AST) -> None:
        for node in ast.walk(expr):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_dyn_imp = (
                (isinstance(func, ast.Name) and func.id in local_fns)
                or (isinstance(func, ast.Attribute) and func.attr == "import_module"
                    and isinstance(func.value, ast.Name) and func.value.id in local_il)
            )
            if is_dyn_imp:
                if (rel_str, enclosing_fn) in _BOUNDED_DYNAMIC_IMPORT_SITES:
                    continue
                is_literal = bool(
                    node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                )
                if not is_literal:
                    violations.append(_make_violation(
                        RULE_B, rel, getattr(node, "lineno", 1),
                        f"open_dynamic_import:{rel_str}:{getattr(node, 'lineno', 1)}",
                        f"open-ended dynamic import_module() with non-literal target is forbidden in {rel_str}",
                    ))
            elif (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= _MIN_GETATTR_ARGS
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in local_mods
                and not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
            ):
                violations.append(_make_violation(
                    RULE_B, rel, getattr(node, "lineno", 1),
                    f"open_dynamic_getattr:{rel_str}:{getattr(node, 'lineno', 1)}",
                    f"open-ended getattr({node.args[0].id}, <dynamic>) on imported module is forbidden in {rel_str}",
                ))

    for stmt in stmts:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                local_mods.add(local)
                if alias.name == "importlib":
                    local_il.add(local)
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level == 0 and stmt.module == "importlib":
                for alias in stmt.names:
                    if alias.name == "import_module":
                        local_fns.add(alias.asname or "import_module")
            for alias in stmt.names:
                if alias.name != "*" and stmt.module and stmt.module.startswith("mindie_sim"):
                    local_mods.add(alias.asname or alias.name)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_scan_dynamic_calls_in_scope(
                rel, rel_str, stmt.body, stmt.name, local_mods, local_il, local_fns,
            ))
        elif isinstance(stmt, ast.ClassDef):
            violations.extend(_scan_dynamic_calls_in_scope(
                rel, rel_str, stmt.body, enclosing_fn, local_mods, local_il, local_fns,
            ))
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    check_expr(child)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list):
                    violations.extend(_scan_dynamic_calls_in_scope(
                        rel, rel_str, sub, enclosing_fn, local_mods, local_il, local_fns,
                    ))
            for handler in getattr(stmt, "handlers", ()) or ():
                violations.extend(_scan_dynamic_calls_in_scope(
                    rel, rel_str, handler.body, enclosing_fn, local_mods, local_il, local_fns,
                ))
            for case in getattr(stmt, "cases", ()) or ():
                violations.extend(_scan_dynamic_calls_in_scope(
                    rel, rel_str, case.body, enclosing_fn, local_mods, local_il, local_fns,
                ))
    return violations


def _extract_selector_calls(expr: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            fn = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if fn in _SELECTION_FACTORY_NAMES and len(node.args) >= _MIN_SELECT_ARGS:
                arg1 = node.args[1]
                if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
                    found.add(arg1.value)
    return found


def _reached_ast_roots(idx: _ModuleIndex, reached_syms: set[str]) -> list[ast.AST]:
    roots: list[ast.AST] = []
    for stmt in idx.tree.body:
        if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, idx.tc_names, idx.typing_mods):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name in reached_syms:
                roots.append(stmt)
        elif isinstance(stmt, ast.ClassDef):
            if stmt.name in reached_syms:
                roots.append(stmt)
        else:
            roots.append(stmt)
    return roots


def _collect_selector_hooks_in_reached(
    idx: _ModuleIndex, reached_syms: set[str],
) -> dict[str, set[str]]:
    """Map selector -> governed hook attrs invoked via ``<sel_var>.module.<attr>`` in reached code."""
    roots = _reached_ast_roots(idx, reached_syms)
    var_to_sels: dict[str, set[str]] = defaultdict(set)
    fn_return_sels: dict[str, list[set[str]]] = {}

    for root_node in roots:
        for node in ast.walk(root_node):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                sels = _extract_selector_calls(node.value)
                if sels and isinstance(node.targets[0], ast.Name):
                    var_to_sels[node.targets[0].id].update(sels)

    for root_node in roots:
        if isinstance(root_node, ast.FunctionDef):
            for sub in ast.walk(root_node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                    tuple_sels = [
                        set(var_to_sels.get(elt.id, ())) if isinstance(elt, ast.Name) else _extract_selector_calls(elt)
                        for elt in sub.value.elts
                    ]
                    if any(tuple_sels):
                        fn_return_sels[root_node.name] = tuple_sels

    for root_node in roots:
        for node in ast.walk(root_node):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Call):
                    callee = node.value.func.id if isinstance(node.value.func, ast.Name) else ""
                    ret_sels = fn_return_sels.get(callee)
                    if ret_sels and len(ret_sels) == len(target.elts):
                        for elt, sels in zip(target.elts, ret_sels):
                            if isinstance(elt, ast.Name) and sels:
                                var_to_sels[elt.id].update(sels)
                elif isinstance(target, ast.Name):
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.Name) and sub.id in var_to_sels:
                            var_to_sels[target.id].update(var_to_sels[sub.id])

    sel_to_hooks: dict[str, set[str]] = defaultdict(set)
    for root_node in roots:
        for node in ast.walk(root_node):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "module"
                and isinstance(node.value.value, ast.Name)
            ):
                for sel in var_to_sels.get(node.value.value.id, ()):
                    sel_to_hooks[sel].add(node.attr)
    return dict(sel_to_hooks)


def _string_constants_in_nodes(nodes: Sequence[ast.AST]) -> set[str]:
    doc_ids: set[int] = set()
    out: set[str] = set()
    for root_node in nodes:
        for node in ast.walk(root_node):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    doc_ids.add(id(node.body[0].value))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
                out.add(node.value)
    return out


def _stack_manifest_recipe_witnesses(root: Path, governed: Sequence[PurePosixPath]) -> dict[str, set[str]]:
    """Extract only explicit ``supported_recipe_ids`` / ``bindings`` from ``stacks/*/manifest.json``."""
    by_pkg: dict[str, set[str]] = defaultdict(set)
    for rel in governed:
        if rel.parts[:2] != ("mindie_sim", "stacks") or rel.name != "manifest.json":
            continue
        pkg = ".".join(rel.parts[:-1])
        try:
            payload = json.loads((root / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for rid in payload.get("supported_recipe_ids") or ():
            if isinstance(rid, str):
                by_pkg[pkg].add(rid)
        for binding in payload.get("bindings") or ():
            if isinstance(binding, dict):
                impl_id = binding.get("implementation")
                if isinstance(impl_id, str):
                    by_pkg[pkg].add(impl_id)
    return dict(by_pkg)


class _ReachabilitySolver:
    """Two-level ``(ModuleInit(M), SymbolExec(M, sym))`` anti-barrel reachability solver."""

    def __init__(
        self,
        indices: Mapping[str, _ModuleIndex],
        known_modules: Mapping[str, PurePosixPath],
        impl_entries: Mapping[str, _ImplementationEntry],
        stack_plugins: Sequence[tuple[str, str]],
        stack_manifest_ids: Mapping[str, set[str]],
        texts: Mapping[PurePosixPath, str],
    ) -> None:
        self.indices = indices
        self.known_modules = known_modules
        self.impl_entries = impl_entries
        self.stack_plugins = tuple(stack_plugins)
        self.stack_manifest_ids = stack_manifest_ids
        self.texts = texts
        self.reached_modules: set[str] = set()
        self.reached_symbols: set[tuple[str, str]] = set()
        self.reached_syms_by_mod: dict[str, set[str]] = defaultdict(set)
        self.witnessed_impl_keys: set[str] = set()
        self._init_done: set[str] = set()
        self._sym_done: set[tuple[str, str]] = set()
        self._queue_mods: list[str] = []
        self._queue_syms: list[tuple[str, str]] = []
        self._selector_hooks: dict[str, set[str]] = defaultdict(set)
        self._scanned_hook_counts: dict[str, int] = {}
        self._scanned_str_syms: dict[str, set[str]] = defaultdict(set)
        self._scanned_str_init: set[str] = set()
        self._reached_strings: dict[str, set[str]] = defaultdict(set)
        self._witness_index: dict[str, set[str]] = defaultdict(set)
        self._impl_key_set = frozenset(impl_entries)

    def mark_module(self, mod: str) -> None:
        if mod not in self.indices:
            return
        if mod not in self.reached_modules:
            self.reached_modules.add(mod)
            self._queue_mods.append(mod)
        parts = mod.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent in self.indices and parent not in self.reached_modules:
                self.reached_modules.add(parent)
                self._queue_mods.append(parent)

    def mark_symbol(self, mod: str, sym: str) -> None:
        if mod not in self.indices:
            return
        self.mark_module(mod)
        pair = (mod, sym)
        if pair not in self.reached_symbols:
            self.reached_symbols.add(pair)
            self.reached_syms_by_mod[mod].add(sym)
            self._queue_syms.append(pair)

    def solve(self, seeds: Sequence[tuple[str, str | None]]) -> None:
        for mod, sym in seeds:
            if sym is None:
                self.mark_module(mod)
            else:
                self.mark_symbol(mod, sym)
        changed = True
        while changed:
            changed = False
            while self._queue_mods or self._queue_syms:
                changed = True
                while self._queue_mods:
                    self._process_module_init(self._queue_mods.pop())
                while self._queue_syms:
                    mod, sym = self._queue_syms.pop()
                    self._process_symbol(mod, sym)
            if self._propagate_dynamic_registries():
                changed = True

    def _process_module_init(self, mod: str) -> None:
        if mod in self._init_done:
            return
        self._init_done.add(mod)
        idx = self.indices[mod]
        if mod == "mindie_sim" or idx.is_pure_barrel:
            return
        for stmt in idx.tree.body:
            if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, idx.tc_names, idx.typing_mods):
                self._scan_block(idx, stmt.orelse, dict(idx.import_bindings), dict(idx.module_aliases))
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in stmt.decorator_list:
                    self._scan_expr(idx, dec, idx.import_bindings, idx.module_aliases)
                continue
            if isinstance(stmt, ast.ClassDef):
                for dec in stmt.decorator_list:
                    self._scan_expr(idx, dec, idx.import_bindings, idx.module_aliases)
                for base in stmt.bases:
                    self._scan_expr(idx, base, idx.import_bindings, idx.module_aliases)
                continue
            self._scan_stmt(idx, stmt, dict(idx.import_bindings), dict(idx.module_aliases))

    def _process_symbol(self, mod: str, sym: str) -> None:
        pair = (mod, sym)
        if pair in self._sym_done:
            return
        self._sym_done.add(pair)
        idx = self.indices[mod]
        if sym in idx.barrel_map:
            tmod, tsym = idx.barrel_map[sym]
            self.mark_module(tmod) if tsym is None else self.mark_symbol(tmod, tsym)
        elif sym in idx.import_bindings:
            tmod, tsym = idx.import_bindings[sym]
            self.mark_module(tmod) if tsym is None else self.mark_symbol(tmod, tsym)
        if sym in idx.defined_funcs:
            fn = idx.defined_funcs[sym]
            self._scan_block(idx, fn.body, dict(idx.import_bindings), dict(idx.module_aliases))
        if sym in idx.defined_classes:
            cls = idx.defined_classes[sym]
            for base in cls.bases:
                self._scan_expr(idx, base, idx.import_bindings, idx.module_aliases)
            self._scan_block(idx, cls.body, dict(idx.import_bindings), dict(idx.module_aliases))
        for rhs in idx.top_assigns.get(sym, ()):
            self._scan_expr(idx, rhs, idx.import_bindings, idx.module_aliases)

    def _scan_block(
        self, idx: _ModuleIndex, stmts: Sequence[ast.stmt],
        bindings: dict[str, tuple[str, str | None]], aliases: dict[str, str],
    ) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, idx.tc_names, idx.typing_mods):
                self._scan_block(idx, stmt.orelse, bindings, aliases)
                continue
            self._scan_stmt(idx, stmt, bindings, aliases)

    def _scan_stmt(
        self, idx: _ModuleIndex, stmt: ast.stmt,
        bindings: dict[str, tuple[str, str | None]], aliases: dict[str, str],
    ) -> None:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                aliases[local] = alias.name
                bindings[local] = (alias.name, None)
                self.mark_module(alias.name)
            return
        if isinstance(stmt, ast.ImportFrom):
            base = _resolve_relative_module(idx.rel, stmt.module, stmt.level)
            if not base:
                return
            for alias in stmt.names:
                if alias.name == "*":
                    self._mark_star_import(base)
                    continue
                local = alias.asname or alias.name
                sub_cand = f"{base}.{alias.name}"
                if sub_cand in self.known_modules:
                    aliases[local] = sub_cand
                    bindings[local] = (sub_cand, None)
                    self.mark_module(sub_cand)
                else:
                    bindings[local] = (base, alias.name)
                    self.mark_symbol(base, alias.name)
            return
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "__all__":
                return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in stmt.decorator_list:
                self._scan_expr(idx, dec, bindings, aliases)
            self._scan_block(idx, stmt.body, dict(bindings), dict(aliases))
            return
        if isinstance(stmt, ast.ClassDef):
            for base in stmt.bases:
                self._scan_expr(idx, base, bindings, aliases)
            self._scan_block(idx, stmt.body, dict(bindings), dict(aliases))
            return
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                self._scan_stmt(idx, child, bindings, aliases)
            elif isinstance(child, ast.expr):
                self._scan_expr(idx, child, bindings, aliases)
            else:
                self._scan_misc(idx, child, bindings, aliases)

    def _scan_misc(
        self, idx: _ModuleIndex, node: ast.AST,
        bindings: dict[str, tuple[str, str | None]], aliases: dict[str, str],
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._scan_stmt(idx, child, bindings, aliases)
            elif isinstance(child, ast.expr):
                self._scan_expr(idx, child, bindings, aliases)
            else:
                self._scan_misc(idx, child, bindings, aliases)

    def _scan_expr(
        self, idx: _ModuleIndex, expr: ast.expr,
        bindings: Mapping[str, tuple[str, str | None]], aliases: Mapping[str, str],
    ) -> None:
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                fn = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                if fn == "import_module" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                        self.mark_module(_resolve_package_relative_str(idx.rel, arg0.value))
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                target_mod = aliases.get(node.value.id)
                if target_mod is not None:
                    sub_cand = f"{target_mod}.{node.attr}"
                    self.mark_module(sub_cand) if sub_cand in self.known_modules else self.mark_symbol(target_mod, node.attr)
            elif isinstance(node, ast.Name):
                name = node.id
                if name in idx.defined_funcs or name in idx.defined_classes or name in idx.top_assigns:
                    self.mark_symbol(idx.mod_name, name)
                elif name in bindings:
                    self.mark_symbol(idx.mod_name, name)
                    tmod, tsym = bindings[name]
                    self.mark_module(tmod) if tsym is None else self.mark_symbol(tmod, tsym)

    def _mark_star_import(self, target_mod: str) -> None:
        target = self.indices.get(target_mod)
        if target is None or target.is_pure_barrel:
            return
        self.mark_module(target_mod)
        for sym in (*target.defined_funcs, *target.defined_classes, *target.top_assigns):
            self.mark_symbol(target_mod, sym)

    def _propagate_dynamic_registries(self) -> bool:
        before = (len(self.reached_modules), len(self.reached_symbols), len(self.witnessed_impl_keys))
        if "mindie_sim.stacks.registry" in self.reached_modules:
            for mod, cls in self.stack_plugins:
                self.mark_symbol(mod, cls)
        if "mindie_sim.prediction.sharding" in self.reached_modules:
            sharding_idx = self.indices.get("mindie_sim.prediction.sharding")
            if sharding_idx is not None:
                for val in sharding_idx.top_assigns.get("_SUBMODULES", ()):
                    for sub in _extract_string_collection(val) or ():
                        self.mark_module(f"mindie_sim.prediction.sharding.{sub}")

        for mod in list(self.reached_modules):
            self._index_reached_strings(mod)
            sym_count = len(self.reached_syms_by_mod.get(mod, ()))
            if self._scanned_hook_counts.get(mod) != sym_count:
                self._scanned_hook_counts[mod] = sym_count
                if ".module." in self.texts.get(self.indices[mod].rel, ""):
                    for sel, hooks in _collect_selector_hooks_in_reached(
                        self.indices[mod], self.reached_syms_by_mod.get(mod, set()),
                    ).items():
                        self._selector_hooks[sel].update(hooks)

        for key, entry in self.impl_entries.items():
            if key not in self.witnessed_impl_keys and self._has_rooted_witness(entry):
                self.witnessed_impl_keys.add(key)
            if key in self.witnessed_impl_keys:
                self._seed_witnessed_impl_hooks(entry)

        after = (len(self.reached_modules), len(self.reached_symbols), len(self.witnessed_impl_keys))
        return after != before

    def _index_reached_strings(self, mod: str) -> None:
        idx = self.indices[mod]
        new_nodes: list[ast.AST] = []
        if mod not in self._scanned_str_init:
            self._scanned_str_init.add(mod)
            for stmt in idx.tree.body:
                if isinstance(stmt, ast.If) and _is_type_checking_if(stmt, idx.tc_names, idx.typing_mods):
                    continue
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    new_nodes.append(stmt)
        done_syms = self._scanned_str_syms[mod]
        for sym in self.reached_syms_by_mod.get(mod, ()):
            if sym in done_syms:
                continue
            done_syms.add(sym)
            if sym in idx.defined_funcs:
                new_nodes.append(idx.defined_funcs[sym])
            if sym in idx.defined_classes:
                new_nodes.append(idx.defined_classes[sym])
        if not new_nodes:
            return
        found = _string_constants_in_nodes(new_nodes)
        self._reached_strings[mod].update(found)
        for token in found & self._impl_key_set:
            self._witness_index[token].add(mod)

    def _has_rooted_witness(self, entry: _ImplementationEntry) -> bool:
        excluded = {"mindie_sim.prediction.program.implementation_registry", entry.module}
        if any(mod not in excluded for mod in self._witness_index.get(entry.key, ())):
            return True
        reached_pkgs = {m.rsplit(".", 1)[0] for m in self.reached_modules} | self.reached_modules
        for pkg, strings in self.stack_manifest_ids.items():
            if pkg in reached_pkgs and entry.key in strings:
                return True
        return False

    def _seed_witnessed_impl_hooks(self, entry: _ImplementationEntry) -> None:
        self.mark_module(entry.module)
        self.mark_symbol(entry.module, "recipes")
        if entry.operators:
            self.mark_symbol(entry.module, "lower_op")
        for sel in (*entry.blocks, *entry.operators):
            for hook in self._selector_hooks.get(sel, ()):
                self.mark_symbol(entry.module, hook)


def _check_ag015_b(
    root: Path,
    governed: Sequence[PurePosixPath],
    parsed_py: Mapping[PurePosixPath, ast.Module],
    texts: Mapping[PurePosixPath, str],
    sim_mod_to_rel: Mapping[str, PurePosixPath],
    collision_violations: Sequence[Violation],
) -> tuple[dict[str, _ImplementationEntry], list[Violation]]:
    indices = {
        mod: _build_module_index(mod, rel, parsed_py[rel], sim_mod_to_rel)
        for mod, rel in sorted(sim_mod_to_rel.items())
    }
    seeds, violations = _check_public_entrypoint_contract(root, indices)
    violations.extend(collision_violations)
    violations.extend(_check_open_ended_dynamic_imports(indices, texts))
    impl_entries, impl_errs = _parse_implementations_registry(indices)
    violations.extend(impl_errs)
    stack_plugins = _parse_builtin_stack_plugins(indices)
    manifest_witnesses = _stack_manifest_recipe_witnesses(root, governed)

    solver = _ReachabilitySolver(indices, sim_mod_to_rel, impl_entries, stack_plugins, manifest_witnesses, texts)
    solver.solve(seeds)

    unreachable_mods = sorted(set(sim_mod_to_rel) - solver.reached_modules)
    for mod in unreachable_mods:
        rel = sim_mod_to_rel[mod]
        violations.append(_make_violation(
            RULE_B, rel, 1, f"unreachable_module:{rel.as_posix()}",
            f"production module {mod!r} is unreachable from PUBLIC_ENTRYPOINT_CONTRACT",
        ))

    unreachable_hooks = 0
    for key, entry in sorted(impl_entries.items()):
        if not (entry.blocks or entry.operators):
            unreachable_hooks += 1
            violations.append(_make_violation(
                RULE_B, _IMPL_REGISTRY_REL, entry.lineno, f"unsatisfiable_impl:{key}",
                f"IMPLEMENTATIONS[{key!r}] declares neither blocks nor operators",
            ))
        if entry.module not in indices:
            unreachable_hooks += 1
            violations.append(_make_violation(
                RULE_B, _IMPL_REGISTRY_REL, entry.lineno, f"missing_impl_module:{key}",
                f"IMPLEMENTATIONS[{key!r}] points to missing module {entry.module!r}",
            ))
            continue
        if key not in solver.witnessed_impl_keys:
            unreachable_hooks += 1
            violations.append(_make_violation(
                RULE_B, _IMPL_REGISTRY_REL, entry.lineno, f"unwitnessed_impl:{key}",
                f"IMPLEMENTATIONS[{key!r}] has no rooted production selector witness",
            ))
            continue
        target_idx = indices[entry.module]
        if not _symbol_exists_in_index(target_idx, "recipes"):
            unreachable_hooks += 1
            violations.append(_make_violation(
                RULE_B, target_idx.rel, 1, f"missing_recipes_hook:{target_idx.rel.as_posix()}",
                f"IMPLEMENTATIONS target {entry.module!r} is missing mandatory recipes() hook",
            ))
        if entry.operators and not _symbol_exists_in_index(target_idx, "lower_op"):
            unreachable_hooks += 1
            violations.append(_make_violation(
                RULE_B, target_idx.rel, 1, f"missing_lower_op_hook:{target_idx.rel.as_posix()}",
                f"operator IMPLEMENTATIONS target {entry.module!r} is missing mandatory lower_op() hook",
            ))

    for target_mod in sorted({entry.module for entry in impl_entries.values() if entry.module in indices}):
        idx = indices[target_mod]
        defined_hooks = sorted(
            name for name in _GOVERNED_HOOK_NAMES
            if name in idx.defined_funcs or name in idx.import_bindings or name in idx.top_assigns
        )
        for hook in defined_hooks:
            if (target_mod, hook) not in solver.reached_symbols:
                unreachable_hooks += 1
                lineno = idx.defined_funcs[hook].lineno if hook in idx.defined_funcs else 1
                violations.append(_make_violation(
                    RULE_B, idx.rel, lineno, f"unreachable_hook:{idx.rel.as_posix()}:{hook}",
                    f"governed hook {hook!r} in {idx.rel.as_posix()} is unreachable from production roots",
                ))

    if len(unreachable_mods) > AG015_MAX_UNREACHABLE_MODULES and not violations:
        violations.append(_make_violation(
            RULE_B, _INIT_REL, 1, "unreachable_mod_budget",
            f"unreachable modules {len(unreachable_mods)} exceeds {AG015_MAX_UNREACHABLE_MODULES}",
        ))
    if unreachable_hooks > AG015_MAX_UNREACHABLE_HOOKS and not violations:
        violations.append(_make_violation(
            RULE_B, _IMPL_REGISTRY_REL, 1, "unreachable_hook_budget",
            f"unreachable hooks {unreachable_hooks} exceeds {AG015_MAX_UNREACHABLE_HOOKS}",
        ))
    return impl_entries, violations


def _eval_static_expr(node: ast.AST, env: Mapping[str, object]) -> tuple[bool, object]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Name):
        return (True, env[node.id]) if node.id in env else (False, None)
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_eval_static_expr(e, env) for e in node.elts]
        if not all(ok for ok, _ in vals):
            return False, None
        seq = [v for _, v in vals]
        return True, tuple(seq) if isinstance(node, ast.Tuple) else seq
    if isinstance(node, ast.Dict):
        out: dict[object, object] = {}
        for k_node, v_node in zip(node.keys, node.values):
            if k_node is None:
                vok, vval = _eval_static_expr(v_node, env)
                if not (vok and isinstance(vval, Mapping)):
                    return False, None
                out.update(vval)
                continue
            kok, kval = _eval_static_expr(k_node, env)
            vok, vval = _eval_static_expr(v_node, env)
            if not (kok and vok):
                return False, None
            out[kval] = vval
        return True, out
    if isinstance(node, ast.Subscript):
        bok, bval = _eval_static_expr(node.value, env)
        sok, sval = _eval_static_expr(node.slice, env)
        return (True, bval[sval]) if (bok and sok and isinstance(bval, Mapping) and sval in bval) else (False, None)
    if isinstance(node, ast.Attribute):
        bok, bval = _eval_static_expr(node.value, env)
        return (True, bval[node.attr]) if (bok and isinstance(bval, Mapping) and node.attr in bval) else (False, None)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lok, lval = _eval_static_expr(node.left, env)
        rok, rval = _eval_static_expr(node.right, env)
        return (True, lval + rval) if (lok and rok and isinstance(lval, str) and isinstance(rval, str)) else (False, None)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for val_node in node.values:
            ok, piece = _eval_static_expr(val_node.value if isinstance(val_node, ast.FormattedValue) else val_node, env)
            if not ok or not isinstance(piece, (str, int)):
                return False, None
            parts.append(str(piece))
        return True, "".join(parts)
    if isinstance(node, ast.IfExp):
        tok, tval = _eval_static_expr(node.test, env)
        return _eval_static_expr(node.body if tval else node.orelse, env) if tok else (False, None)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        lok, lval = _eval_static_expr(node.left, env)
        rok, rval = _eval_static_expr(node.comparators[0], env)
        if lok and rok:
            op = node.ops[0]
            if isinstance(op, ast.In) and isinstance(rval, (str, tuple, list, dict)):
                return True, lval in rval
            if isinstance(op, ast.Eq):
                return True, lval == rval
            if isinstance(op, ast.NotEq):
                return True, lval != rval
    return False, None


def _build_module_static_env(
    rel: PurePosixPath,
    parsed_py: Mapping[PurePosixPath, ast.Module],
    mod_to_rel: Mapping[str, PurePosixPath],
    env_cache: dict[PurePosixPath, dict[str, object]],
    depth: int = 0,
) -> dict[str, object]:
    if rel in env_cache:
        return env_cache[rel]
    tree = parsed_py.get(rel)
    if tree is None or depth > _SOURCE_REF_MAX_DEPTH:
        return {}
    env: dict[str, object] = {}
    env_cache[rel] = env
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and depth < _SOURCE_REF_MAX_DEPTH:
            base = _resolve_relative_module(rel, stmt.module, stmt.level)
            if not base:
                continue
            for alias in stmt.names:
                local = alias.asname or alias.name
                sub_cand = f"{base}.{alias.name}"
                if sub_cand in mod_to_rel:
                    env[local] = _build_module_static_env(
                        mod_to_rel[sub_cand], parsed_py, mod_to_rel, env_cache, depth + 1,
                    )
                elif base in mod_to_rel:
                    src_env = _build_module_static_env(
                        mod_to_rel[base], parsed_py, mod_to_rel, env_cache, depth + 1,
                    )
                    if alias.name in src_env:
                        env[local] = src_env[alias.name]
        elif isinstance(stmt, ast.Import) and depth < _SOURCE_REF_MAX_DEPTH:
            for alias in stmt.names:
                if alias.name in mod_to_rel:
                    env[alias.asname or alias.name.split(".", 1)[0]] = _build_module_static_env(
                        mod_to_rel[alias.name], parsed_py, mod_to_rel, env_cache, depth + 1,
                    )
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            ok, val = _eval_static_expr(stmt.value, env)
            if ok:
                env[stmt.targets[0].id] = val
            else:
                env.pop(stmt.targets[0].id, None)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            ok, val = _eval_static_expr(stmt.value, env)
            if ok:
                env[stmt.target.id] = val
            else:
                env.pop(stmt.target.id, None)
    return env


def _eval_source_ref_call_fields(call: ast.Call, env: Mapping[str, object]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for idx, arg in enumerate(call.args[: len(_SOURCE_REF_POSITIONAL)]):
        ok, val = _eval_static_expr(arg, env)
        if ok and isinstance(val, str):
            fields[_SOURCE_REF_POSITIONAL[idx]] = val
    for kw in call.keywords:
        if kw.arg:
            ok, val = _eval_static_expr(kw.value, env)
            if ok and isinstance(val, str):
                fields[kw.arg] = val
    return fields


def _expand_source_ref_generator(
    rel: PurePosixPath, gen: ast.GeneratorExp, env: Mapping[str, object],
) -> tuple[list[tuple[PurePosixPath, int, dict[str, str]]], bool]:
    if not isinstance(gen.elt, ast.Call) or len(gen.generators) != 1:
        return [], True
    fn = gen.elt.func.id if isinstance(gen.elt.func, ast.Name) else (
        gen.elt.func.attr if isinstance(gen.elt.func, ast.Attribute) else ""
    )
    if fn != "SourceRef":
        return [], True
    comp = gen.generators[0]
    if not (
        isinstance(comp.target, ast.Tuple)
        and len(comp.target.elts) == _MIN_SELECT_ARGS
        and all(isinstance(e, ast.Name) for e in comp.target.elts)
        and isinstance(comp.iter, ast.Call)
        and isinstance(comp.iter.func, ast.Attribute)
        and comp.iter.func.attr == "items"
    ):
        return [], True
    ok, mapping = _eval_static_expr(comp.iter.func.value, env)
    if not ok or not isinstance(mapping, Mapping) or not mapping:
        return [], True
    k_var, v_var = comp.target.elts[0].id, comp.target.elts[1].id  # type: ignore[union-attr]
    out: list[tuple[PurePosixPath, int, dict[str, str]]] = []
    for k_val, v_val in mapping.items():
        local_env = {**env, k_var: k_val, v_var: v_val}
        out.append((rel, getattr(gen.elt, "lineno", 1), _eval_source_ref_call_fields(gen.elt, local_env)))
    return out, False


def _resolve_imported_symbol(
    rel: PurePosixPath, tree: ast.Module, sym: str, mod_to_rel: Mapping[str, PurePosixPath],
) -> tuple[PurePosixPath, str] | None:
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            base = _resolve_relative_module(rel, stmt.module, stmt.level)
            if base and base in mod_to_rel:
                for alias in stmt.names:
                    if (alias.asname or alias.name) == sym:
                        return mod_to_rel[base], alias.name
    return None


def _resolve_imported_module(
    rel: PurePosixPath, tree: ast.Module, alias_name: str, mod_to_rel: Mapping[str, PurePosixPath],
) -> PurePosixPath | None:
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if (alias.asname or alias.name.split(".", 1)[0]) == alias_name and alias.name in mod_to_rel:
                    return mod_to_rel[alias.name]
        elif isinstance(stmt, ast.ImportFrom):
            base = _resolve_relative_module(rel, stmt.module, stmt.level)
            if base:
                for alias in stmt.names:
                    if (alias.asname or alias.name) == alias_name:
                        cand = f"{base}.{alias.name}"
                        if cand in mod_to_rel:
                            return mod_to_rel[cand]
    return None


def _extract_source_ref_entries(
    rel: PurePosixPath,
    node: ast.AST,
    parsed_py: Mapping[PurePosixPath, ast.Module],
    mod_to_rel: Mapping[str, PurePosixPath],
    env_cache: dict[PurePosixPath, dict[str, object]],
    depth: int = 0,
) -> tuple[list[tuple[PurePosixPath, int, dict[str, str]]], bool]:
    if depth > _SOURCE_REF_MAX_DEPTH:
        return [], True
    env = _build_module_static_env(rel, parsed_py, mod_to_rel, env_cache)
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if fn == "SourceRef":
            return [(rel, getattr(node, "lineno", 1), _eval_source_ref_call_fields(node, env))], False
        if fn in {"tuple", "list"} and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return _expand_source_ref_generator(rel, node.args[0], env)
        return [], True
    if isinstance(node, ast.Starred):
        return _extract_source_ref_entries(rel, node.value, parsed_py, mod_to_rel, env_cache, depth + 1)
    if isinstance(node, (ast.Tuple, ast.List)):
        entries: list[tuple[PurePosixPath, int, dict[str, str]]] = []
        opaque = False
        for elt in node.elts:
            sub_entries, sub_opaque = _extract_source_ref_entries(rel, elt, parsed_py, mod_to_rel, env_cache, depth + 1)
            entries.extend(sub_entries)
            opaque = opaque or sub_opaque
        return entries, opaque
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, lo = _extract_source_ref_entries(rel, node.left, parsed_py, mod_to_rel, env_cache, depth + 1)
        right, ro = _extract_source_ref_entries(rel, node.right, parsed_py, mod_to_rel, env_cache, depth + 1)
        return left + right, lo or ro
    tree = parsed_py.get(rel)
    if tree is None:
        return [], True
    if isinstance(node, ast.Name):
        local = _top_level_assignments(tree, node.id)
        if local:
            return _extract_source_ref_entries(rel, local[-1][1], parsed_py, mod_to_rel, env_cache, depth + 1)
        target = _resolve_imported_symbol(rel, tree, node.id, mod_to_rel)
        if target is not None and target[0] in parsed_py:
            found = _top_level_assignments(parsed_py[target[0]], target[1])
            if found:
                return _extract_source_ref_entries(target[0], found[-1][1], parsed_py, mod_to_rel, env_cache, depth + 1)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        target_rel = _resolve_imported_module(rel, tree, node.value.id, mod_to_rel)
        if target_rel is not None and target_rel in parsed_py:
            found = _top_level_assignments(parsed_py[target_rel], node.attr)
            if found:
                return _extract_source_ref_entries(target_rel, found[-1][1], parsed_py, mod_to_rel, env_cache, depth + 1)
    return [], True


def _is_allowed_repo(repo: str, target_backend: str | None) -> bool:
    if target_backend == "hardware_neutral" and repo == "analytical":
        return True
    if repo in _ALLOWED_EXACT_ASCEND_REPOS:
        return True
    if _FOREIGN_REPO_SUBSTR_RE.search(repo):
        return False
    return (
        repo.startswith(_ALLOWED_ORG_PREFIXES)
        and repo.count("/") == 1
        and not any(ch.isspace() for ch in repo)
        and ".." not in repo
    )


def _is_valid_source_rel_path(rpath: str) -> bool:
    if not rpath or rpath.startswith("/") or "\\" in rpath:
        return False
    if any(part in {"", ".", ".."} for part in rpath.split("/")):
        return False
    return _CUDA_SOURCE_EXT_RE.search(rpath) is None


def _validate_source_ref_fields(
    rel: PurePosixPath, lineno: int, fields: Mapping[str, str], target_backend: str | None,
) -> list[Violation]:
    violations: list[Violation] = []
    repo = fields.get("repository", "")
    rev = fields.get("revision", "")
    rpath = fields.get("relative_path", "")
    sha = fields.get("source_sha256", "")

    if not repo or not _is_allowed_repo(repo, target_backend):
        violations.append(_make_violation(
            RULE_C, rel, lineno, f"foreign_source_repo:{rel.as_posix()}:{repo}",
            f"SourceRef repository {repo!r} is not an allowed Ascend/open-NPU upstream",
        ))
    is_analytical = target_backend == "hardware_neutral" and repo == "analytical"
    if not is_analytical and not _REVISION_HEX_RE.match(rev):
        violations.append(_make_violation(
            RULE_C, rel, lineno, f"unpinned_source_rev:{rel.as_posix()}:{rev}",
            f"SourceRef revision {rev!r} must be a pinned 40-char commit hash",
        ))
    if not _is_valid_source_rel_path(rpath):
        violations.append(_make_violation(
            RULE_C, rel, lineno, f"bad_source_path:{rel.as_posix()}:{rpath}",
            f"SourceRef relative_path {rpath!r} is invalid, escapes repo root, or cites foreign CUDA/PTX source",
        ))
    if not is_analytical and not _SHA256_HEX_RE.match(sha):
        violations.append(_make_violation(
            RULE_C, rel, lineno, f"bad_source_sha:{rel.as_posix()}:{sha}",
            f"SourceRef source_sha256 {sha!r} must be a 64-char lowercase hex digest",
        ))
    return violations


def _scan_py_tokens_for_foreign_hw(rel: PurePosixPath, text: str) -> list[Violation]:
    if _CUDA_SOURCE_EXT_RE.search(text) is None and _FOREIGN_GPU_TOKEN_RE.search(text) is None:
        return []
    violations: list[Violation] = []
    seen: set[tuple[int, str]] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type not in {tokenize.NAME, tokenize.STRING}:
                continue
            lineno = tok.start[0]
            cu_match = _CUDA_SOURCE_EXT_RE.search(tok.string)
            if cu_match is not None and (lineno, "cu") not in seen:
                seen.add((lineno, "cu"))
                violations.append(_make_violation(
                    RULE_C, rel, lineno, f"cuda_citation:{rel.as_posix()}:{lineno}",
                    f"foreign CUDA/PTX source citation ({cu_match.group(0)}) forbidden in {rel.as_posix()}",
                ))
            gpu_match = _FOREIGN_GPU_TOKEN_RE.search(tok.string)
            if gpu_match is not None and (lineno, gpu_match.group(0).lower()) not in seen:
                seen.add((lineno, gpu_match.group(0).lower()))
                violations.append(_make_violation(
                    RULE_C, rel, lineno, f"foreign_gpu_token:{rel.as_posix()}:{gpu_match.group(0).lower()}",
                    f"foreign GPU/CUDA identifier or shim ({gpu_match.group(0)!r}) forbidden in {rel.as_posix()}",
                ))
    except (tokenize.TokenError, IndentationError):
        pass
    return violations


def _scan_backend_json_for_foreign_hw(root: Path, rel: PurePosixPath) -> list[Violation]:
    try:
        payload = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    violations: list[Violation] = []

    def walk_json(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and _FOREIGN_GPU_TOKEN_RE.search(k):
                    violations.append(_make_violation(
                        RULE_C, rel, 1, f"foreign_backend_key:{rel.as_posix()}:{k.lower()}",
                        f"foreign GPU/CUDA key {k!r} forbidden in backend profile {rel.as_posix()}",
                    ))
                walk_json(v)
        elif isinstance(obj, list):
            for item in obj:
                walk_json(item)
        elif isinstance(obj, str):
            if _FOREIGN_BACKEND_VALUE_RE.match(obj.strip()) or _FOREIGN_GPU_TOKEN_RE.search(obj):
                violations.append(_make_violation(
                    RULE_C, rel, 1, f"foreign_backend_val:{rel.as_posix()}:{obj.lower()}",
                    f"foreign GPU/CUDA backend value {obj!r} forbidden in {rel.as_posix()}",
                ))

    walk_json(payload)
    return violations


def _planned_recipe_builder_rels(
    impl_target_rels: set[PurePosixPath],
    parsed_py: Mapping[PurePosixPath, ast.Module],
    mod_to_rel: Mapping[str, PurePosixPath],
) -> set[PurePosixPath]:
    builders: set[PurePosixPath] = set()
    for rel in impl_target_rels:
        tree = parsed_py.get(rel)
        if tree is None:
            continue
        mod_aliases: dict[str, PurePosixPath] = {}
        sym_owners: dict[str, PurePosixPath] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    if a.name in mod_to_rel:
                        mod_aliases[a.asname or a.name.split(".", 1)[0]] = mod_to_rel[a.name]
            elif isinstance(stmt, ast.ImportFrom):
                base = _resolve_relative_module(rel, stmt.module, stmt.level)
                if not base:
                    continue
                for a in stmt.names:
                    local = a.asname or a.name
                    sub_cand = f"{base}.{a.name}"
                    if sub_cand in mod_to_rel:
                        mod_aliases[local] = mod_to_rel[sub_cand]
                    elif base in mod_to_rel:
                        sym_owners[local] = mod_to_rel[base]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                if fn == "PlannedRecipe" and node.args:
                    arg0 = node.args[0]
                    target_rel = None
                    if isinstance(arg0, ast.Attribute) and isinstance(arg0.value, ast.Name):
                        target_rel = mod_aliases.get(arg0.value.id)
                    elif isinstance(arg0, ast.Name):
                        target_rel = sym_owners.get(arg0.id)
                    if target_rel is not None and target_rel.parts[:3] == ("mindie_sim", "prediction", "operators"):
                        builders.add(target_rel)
    return builders


def _check_ag015_c(
    root: Path,
    governed: Sequence[PurePosixPath],
    parsed_py: Mapping[PurePosixPath, ast.Module],
    texts: Mapping[PurePosixPath, str],
    mod_to_rel: Mapping[str, PurePosixPath],
    impl_entries: Mapping[str, _ImplementationEntry],
) -> list[Violation]:
    violations: list[Violation] = []
    foreign_hw_count = 0

    for rel in governed:
        if rel.parts[:2] == ("mindie_sim", "backends"):
            if _FOREIGN_BACKEND_FILE_RE.search(rel.name):
                foreign_hw_count += 1
                violations.append(_make_violation(
                    RULE_C, rel, 1, f"foreign_backend_file:{rel.as_posix()}",
                    f"foreign NVIDIA/CUDA backend file {rel.as_posix()!r} is permanently banned",
                ))
            elif rel.suffix == ".json":
                violations.extend(_scan_backend_json_for_foreign_hw(root, rel))
        if rel.parts[:2] == ("mindie_sim", "stacks") and rel.name == "manifest.json":
            violations.extend(_check_stack_manifest_foreign_ids(root, rel))
        if rel.parts[:3] == ("mindie_sim", "prediction", "operators") and rel.suffix in {".c", ".h"}:
            native_text = (root / rel).read_text(encoding="utf-8", errors="replace")
            if (_CUDA_HEADER_RE.search(native_text) or _CUDA_SOURCE_EXT_RE.search(native_text)
                    or _FOREIGN_GPU_TOKEN_RE.search(native_text)):
                foreign_hw_count += 1
                violations.append(_make_violation(
                    RULE_C, rel, 1, f"foreign_native_kernel:{rel.as_posix()}",
                    f"foreign CUDA/GPU kernel header or token forbidden in {rel.as_posix()}",
                ))

    impl_target_rels = {mod_to_rel[entry.module] for entry in impl_entries.values() if entry.module in mod_to_rel}
    builder_rels = _planned_recipe_builder_rels(impl_target_rels, parsed_py, mod_to_rel)

    for rel, tree in sorted(parsed_py.items()):
        is_operator = rel.parts[:3] == ("mindie_sim", "prediction", "operators")
        if not (is_operator or rel == _IMPL_REGISTRY_REL or rel in impl_target_rels):
            continue
        tok_errs = _scan_py_tokens_for_foreign_hw(rel, texts.get(rel, ""))
        foreign_hw_count += len(tok_errs)
        violations.extend(tok_errs)

        tb_assigns = _top_level_assignments(tree, "TARGET_BACKEND")
        tb_val: str | None = None
        for tb_line, tb_node in tb_assigns:
            val = tb_node.value if isinstance(tb_node, ast.Constant) and isinstance(tb_node.value, str) else None
            if val not in _ALLOWED_TARGET_BACKENDS:
                foreign_hw_count += 1
                violations.append(_make_violation(
                    RULE_C, rel, tb_line, f"invalid_target_backend:{rel.as_posix()}:{tb_line}",
                    f"TARGET_BACKEND must be in {sorted(_ALLOWED_TARGET_BACKENDS)!r} (got {val!r})",
                ))
            tb_val = val

        defines_selectable_hook = is_operator and any(
            isinstance(stmt, ast.FunctionDef) and stmt.name in _SELECTABLE_OP_HOOKS for stmt in tree.body
        )
        if rel in impl_target_rels or rel in builder_rels or defines_selectable_hook:
            if not tb_assigns or tb_val not in _ALLOWED_TARGET_BACKENDS:
                foreign_hw_count += 1
                violations.append(_make_violation(
                    RULE_C, rel, 1, f"missing_target_backend:{rel.as_posix()}",
                    f"selectable operator {rel.as_posix()!r} must declare TARGET_BACKEND in {sorted(_ALLOWED_TARGET_BACKENDS)!r}",
                ))
            if rel in impl_target_rels:
                violations.extend(_check_impl_source_provenance(rel, tree, tb_val, parsed_py, mod_to_rel))

    if foreign_hw_count > AG015_MAX_FOREIGN_HW_MODULES and not violations:
        violations.append(_make_violation(
            RULE_C, _IMPL_REGISTRY_REL, 1, "foreign_hw_budget",
            f"foreign hardware modules {foreign_hw_count} exceeds {AG015_MAX_FOREIGN_HW_MODULES}",
        ))
    return violations


def _check_impl_source_provenance(
    rel: PurePosixPath,
    tree: ast.Module,
    target_backend: str | None,
    parsed_py: Mapping[PurePosixPath, ast.Module],
    mod_to_rel: Mapping[str, PurePosixPath],
) -> list[Violation]:
    all_src_assigns = (
        _top_level_assignments(tree, "SOURCES")
        + _top_level_assignments(tree, "SOURCE_REFS")
        + _top_level_assignments(tree, "SOURCE_REF")
    )
    if not all_src_assigns:
        return [_make_violation(
            RULE_C, rel, 1, f"missing_source_ref:{rel.as_posix()}",
            f"IMPLEMENTATIONS target {rel.as_posix()!r} must declare pinned SOURCES / SOURCE_REFS / SOURCE_REF",
        )]
    env_cache: dict[PurePosixPath, dict[str, object]] = {}
    violations: list[Violation] = []
    for lineno, node in sorted(all_src_assigns, key=lambda item: item[0]):
        entries, has_opaque = _extract_source_ref_entries(rel, node, parsed_py, mod_to_rel, env_cache)
        if has_opaque:
            violations.append(_make_violation(
                RULE_C, rel, lineno, f"opaque_source_ref:{rel.as_posix()}:{lineno}",
                f"IMPLEMENTATIONS target {rel.as_posix()!r} contains unresolvable/opaque SourceRef expression",
            ))
        if not entries:
            violations.append(_make_violation(
                RULE_C, rel, lineno, f"empty_source_ref:{rel.as_posix()}:{lineno}",
                f"IMPLEMENTATIONS target {rel.as_posix()!r} must resolve to >= 1 pinned SourceRef",
            ))
            continue
        for def_rel, ref_line, fields in entries:
            violations.extend(_validate_source_ref_fields(def_rel, ref_line, fields, target_backend))
    return violations


def _check_stack_manifest_foreign_ids(root: Path, rel: PurePosixPath) -> list[Violation]:
    try:
        payload = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    violations: list[Violation] = []
    for recipe_id in payload.get("supported_recipe_ids") or ():
        if isinstance(recipe_id, str) and _FOREIGN_RECIPE_ID_RE.match(recipe_id):
            violations.append(_make_violation(
                RULE_C, rel, 1, f"foreign_manifest_recipe:{rel.as_posix()}:{recipe_id}",
                f"foreign GPU/CUDA recipe id {recipe_id!r} forbidden in {rel.as_posix()}",
            ))
    return violations


def check_ag015(
    root: Path,
    /,
    *,
    revision: str | None = None,
    require_manifest: bool = True,
    preparsed_py: Mapping[PurePosixPath, ast.Module] | None = None,
    preloaded_texts: Mapping[PurePosixPath, str] | None = None,
) -> tuple[Violation, ...]:
    """Run AG015-A (UML bijection), AG015-B (reachability), and AG015-C (hardware target lock)."""
    if revision is not None:
        return ()
    resolved = root.resolve()
    governed = _enumerate_governed_files(resolved)
    parsed_py: dict[PurePosixPath, ast.Module] = dict(preparsed_py or {})
    texts: dict[PurePosixPath, str] = dict(preloaded_texts or {})
    all_mod_to_rel: dict[str, PurePosixPath] = {}
    sim_mod_to_rel: dict[str, PurePosixPath] = {}
    collision_violations: list[Violation] = []
    violations: list[Violation] = []

    for rel in governed:
        if rel.suffix != ".py":
            continue
        if rel not in parsed_py or rel not in texts:
            try:
                text = texts[rel] if rel in texts else (resolved / rel).read_text(encoding="utf-8")
                tree = parsed_py[rel] if rel in parsed_py else ast.parse(text, filename=rel.as_posix())
            except (OSError, SyntaxError):
                continue
            texts[rel] = text
            parsed_py[rel] = tree
        mod_name = _module_name_for_rel(rel)
        if mod_name in all_mod_to_rel and rel.parts[0] == "mindie_sim":
            other = all_mod_to_rel[mod_name]
            shadowed = other if rel.name == "__init__.py" else rel
            collision_violations.append(_make_violation(
                RULE_B, shadowed, 1, f"module_package_collision:{shadowed.as_posix()}",
                f"shadowed module {shadowed.as_posix()!r} collides with package {mod_name!r}",
            ))
        all_mod_to_rel[mod_name] = rel
        if rel.parts[0] == "mindie_sim":
            sim_mod_to_rel[mod_name] = rel

    has_repo_contract = (
        require_manifest
        or (resolved / _ARCH_DOC_REL).is_file()
        or (resolved / _PYPROJECT_REL).is_file()
        or (resolved / ".git").exists()
    )

    impl_entries: dict[str, _ImplementationEntry] = {}
    if has_repo_contract:
        violations.extend(_check_ag015_a(resolved, governed, parsed_py, texts, all_mod_to_rel))
        impl_entries, b_errs = _check_ag015_b(
            resolved, governed, parsed_py, texts, sim_mod_to_rel, collision_violations,
        )
        violations.extend(b_errs)
    elif _IMPL_REGISTRY_REL in parsed_py:
        idx_map = {
            m: _build_module_index(m, r, parsed_py[r], sim_mod_to_rel)
            for m, r in sim_mod_to_rel.items()
        }
        impl_entries, _ = _parse_implementations_registry(idx_map)

    violations.extend(_check_ag015_c(resolved, governed, parsed_py, texts, sim_mod_to_rel, impl_entries))
    violations.sort(key=lambda v: (v.path.as_posix(), v.line, v.rule, v.message))
    return tuple(violations)
