"""AG015 zero-allowlist UML bijection, reachability & hardware target lock tests.

[Role] Verify permanent ``AG015-A``, ``AG015-B``, and ``AG015-C`` governance
  invariants: literal-zero ceilings, zero banned allowlist identifiers, exact
  ``Code <-> docs/architecture.md`` manifest/Mermaid bijection, production-root
  anti-barrel & anti-registry-stuffing reachability, Ascend/neutral hardware
  provenance enforcement, unconditional non-differential CLI enforcement, and
  ``ci_scope.py`` / ``ci.yml`` architecture-contract wiring.
[Invariants] Pure Tier-0 in-memory/``tmp_path`` behavioral tests with zero
  subprocesses, network I/O, or repository mutations.
[Timeouts & Locks] Bounded local AST and synthetic fixture checks (< 5s).
[Side-Effect Policy] Writes synthetic fixtures exclusively under ``tmp_path``.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re

import pytest

import scripts._arch_governance_ag015 as ag015
import scripts.check_arch_governance as arch_gov

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
_BANNED_AG015_NAME_RE = re.compile(r"^_?AG015_.*(?:ALLOWLIST|EXEMPT|IGNORE|SKIP)$")
_VALID_REV = "a" * 40
_VALID_SHA = "b" * 64


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_ci_scope():
    scope_path = REPO_ROOT / ".github" / "scripts" / "ci_scope.py"
    spec = importlib.util.spec_from_file_location("sim_ci_scope_test_mod", scope_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_minimal_valid_repo(root: Path) -> None:
    """Create a minimal self-contained repo that passes AG015-A, AG015-B, and AG015-C."""
    _write(
        root / "pyproject.toml",
        '[project]\nname = "mindie-sim"\n\n[project.scripts]\nmindie-sim = "mindie_sim.cli:main"\n',
    )
    init_all = sorted(ag015.PUBLIC_ENTRYPOINT_CONTRACT["init_all"])
    init_lines = [
        "from . import api as _api_ref",
        "__all__ = " + repr(init_all),
        '_LAZY_SUBMODULES = frozenset({"api", "cli", "cohort_accuracy", "structure"})',
        '_LAZY_ALIASES = {"cli_export": "mindie_sim.cli.export", "cohort_comparison": "mindie_sim.cohort_accuracy"}',
    ]
    _write(root / "mindie_sim" / "__init__.py", "\n".join(init_lines) + "\n")
    _write(root / "mindie_sim" / "__main__.py", "from mindie_sim.cli import main\n")
    _write(
        root / "mindie_sim" / "cli" / "__init__.py",
        "from mindie_sim.prediction.program import implementation_registry as reg\n"
        "def main():\n"
        '    return reg.select({"demo_block": "ascend_demo_kernel"}, "demo_block")\n',
    )
    _write(root / "mindie_sim" / "cli" / "export.py", "def run(): return 0\n")
    _write(root / "mindie_sim" / "api.py", "from mindie_sim.cli import export as _e\n")
    _write(root / "mindie_sim" / "solve.py", "SOLVE_OK = True\n")
    _write(root / "mindie_sim" / "export.py", "EXPORT_OK = True\n")
    _write(root / "mindie_sim" / "public.py", "PUBLIC_OK = True\n")
    _write(root / "mindie_sim" / "cohort_accuracy.py", "COHORT_OK = True\n")
    _write(root / "mindie_sim" / "structure.py", "STRUCT_OK = True\n")
    _write(root / "mindie_sim" / "structure_values.py", "SV_OK = True\n")
    _write(root / "mindie_sim" / "structure_contracts.py", "SC_OK = True\n")
    _write(root / "mindie_sim" / "structure_implementations.py", "SI_OK = True\n")
    _write(root / "mindie_sim" / "structure_subgraphs.py", "SS_OK = True\n")
    _write(root / "mindie_sim" / "subgraph_math.py", "SM_OK = True\n")
    for sub in ("__init__", "attention", "fusion", "matrix", "moe", "signatures", "vocabulary"):
        _write(root / "mindie_sim" / "observed_contracts" / f"{sub}.py", "OK = True\n")
    for sub in ("__init__", "scoreboard", "subgraph_parity"):
        _write(root / "mindie_sim" / "observations" / f"{sub}.py", "OK = True\n")
    _write(root / "mindie_sim" / "observations" / "communication" / "__init__.py", "OK = True\n")
    _write(root / "mindie_sim" / "prediction" / "__init__.py", "OK = True\n")
    _write(root / "mindie_sim" / "prediction" / "program" / "__init__.py", "OK = True\n")
    _write(
        root / "mindie_sim" / "prediction" / "program" / "implementation_registry.py",
        "class Implementation:\n"
        "    def __init__(self, module, operators=(), blocks=(), options=()):\n"
        "        self.module = module\n"
        "IMPLEMENTATIONS = {\n"
        '    "ascend_demo_kernel": Implementation(\n'
        '        "mindie_sim.prediction.operators.demo.kernel", blocks=("demo_block",)\n'
        "    ),\n"
        "}\n"
        "def select(bindings, selector):\n"
        "    return bindings.get(selector)\n",
    )
    _write(root / "mindie_sim" / "prediction" / "operators" / "__init__.py", "OK = True\n")
    _write(root / "mindie_sim" / "prediction" / "operators" / "demo" / "__init__.py", "OK = True\n")
    _write(
        root / "mindie_sim" / "prediction" / "operators" / "demo" / "kernel.py",
        "# sm90 is unsupported; Ascend Triton kernel only.\n"
        'TARGET_BACKEND = "ascend"\n'
        "class SourceRef:\n"
        "    def __init__(self, repository, revision, relative_path, source_sha256):\n"
        "        pass\n"
        "SOURCES = (\n"
        f'    SourceRef("vllm-project/vllm-ascend", "{_VALID_REV}", "vllm_ascend/ops/demo.py", "{_VALID_SHA}"),\n'
        ")\n"
        "def recipes():\n"
        "    return {}\n",
    )
    _write(root / "tools" / "probe.py", "from mindie_sim.api import _e\n")

    manifest = {
        "schema": "mindie-sim.ag015-architecture-manifest/1",
        "bilingual_mirror_required": False,
        "subsystems": [
            {
                "id": "L0_CONTRACTS",
                "layer": "L0",
                "selectors": ["mindie_sim/observed_contracts/**"],
                "allowed_edges": [],
            },
            {
                "id": "L2_PROGRAM",
                "layer": "L2",
                "selectors": [
                    "mindie_sim/prediction/__init__.py",
                    "mindie_sim/prediction/program/**",
                ],
                "allowed_edges": ["L0_CONTRACTS"],
            },
            {
                "id": "L3_OPERATORS",
                "layer": "L3",
                "selectors": ["mindie_sim/prediction/operators/**"],
                "allowed_edges": ["L0_CONTRACTS", "L2_PROGRAM"],
            },
            {
                "id": "L5_ENTRY",
                "layer": "L5b",
                "selectors": [
                    "mindie_sim/__init__.py",
                    "mindie_sim/__main__.py",
                    "mindie_sim/cli/**",
                    "mindie_sim/api.py",
                    "mindie_sim/solve.py",
                    "mindie_sim/export.py",
                    "mindie_sim/public.py",
                    "mindie_sim/cohort_accuracy.py",
                    "mindie_sim/structure.py",
                    "mindie_sim/structure_values.py",
                    "mindie_sim/structure_contracts.py",
                    "mindie_sim/structure_implementations.py",
                    "mindie_sim/structure_subgraphs.py",
                    "mindie_sim/subgraph_math.py",
                    "mindie_sim/observations/**",
                ],
                "allowed_edges": ["L0_CONTRACTS", "L2_PROGRAM", "L3_OPERATORS"],
            },
            {
                "id": "DEV_TOOLS",
                "layer": "DEV",
                "selectors": ["tools/**"],
                "allowed_edges": ["L0_CONTRACTS", "L2_PROGRAM", "L3_OPERATORS", "L5_ENTRY"],
            },
        ],
    }
    mermaid = "\n".join([
        "flowchart TD",
        '    L0_CONTRACTS["L0_CONTRACTS"]',
        '    L2_PROGRAM["L2_PROGRAM"]',
        '    L3_OPERATORS["L3_OPERATORS"]',
        '    L5_ENTRY["L5_ENTRY"]',
        '    DEV_TOOLS["DEV_TOOLS"]',
        "    L2_PROGRAM --> L0_CONTRACTS",
        "    L3_OPERATORS --> L0_CONTRACTS",
        "    L3_OPERATORS --> L2_PROGRAM",
        "    L5_ENTRY --> L0_CONTRACTS",
        "    L5_ENTRY --> L2_PROGRAM",
        "    L5_ENTRY --> L3_OPERATORS",
        "    DEV_TOOLS --> L0_CONTRACTS",
        "    DEV_TOOLS --> L2_PROGRAM",
        "    DEV_TOOLS --> L3_OPERATORS",
        "    DEV_TOOLS --> L5_ENTRY",
    ])
    doc = (
        "# Architecture\n\n"
        "<!-- ag015:architecture-manifest:start -->\n"
        "```json\n"
        + json.dumps(manifest, indent=2)
        + "\n```\n"
        "<!-- ag015:architecture-manifest:end -->\n\n"
        "```mermaid\n"
        + mermaid
        + "\n```\n"
    )
    _write(root / "docs" / "architecture.md", doc)


def test_ag015_real_repo_passes_and_ratchet_zero_ceilings_locked() -> None:
    expected_ceilings = (
        "AG015_MAX_UNMAPPED_MODULES",
        "AG015_MAX_UNREACHABLE_MODULES",
        "AG015_MAX_UNREACHABLE_HOOKS",
        "AG015_MAX_FOREIGN_HW_MODULES",
    )
    ag015_src = (REPO_ROOT / "scripts" / "_arch_governance_ag015.py").read_text(encoding="utf-8")
    ag015_tree = ast.parse(ag015_src)
    counts = {name: 0 for name in expected_ceilings}
    for node in ast.walk(ag015_tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in counts:
                    assert isinstance(node.value, ast.Constant) and type(node.value.value) is int and node.value.value == 0
                    counts[t.id] += 1
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in counts:
            assert isinstance(node.value, ast.Constant) and type(node.value.value) is int and node.value.value == 0
            counts[node.target.id] += 1
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assert node.target.id not in counts
    assert all(count == 1 for count in counts.values())

    for script_rel in (
        "scripts/_arch_governance_ag015.py",
        "scripts/check_arch_governance.py",
        "scripts/verify_m1.py",
    ):
        tree = ast.parse((REPO_ROOT / script_rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for attr in ("id", "name", "attr", "arg", "asname"):
                val = getattr(node, attr, None)
                if isinstance(val, str):
                    assert not _BANNED_AG015_NAME_RE.match(val), f"banned AG015 identifier {val!r} in {script_rel}"

    assert ag015.check_ag015(REPO_ROOT) == ()


def test_ag015_minimal_fixture_passes_and_bare_tree_fails_closed(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _build_minimal_valid_repo(proj)
    assert ag015.check_ag015(proj) == ()

    bare = tmp_path / "bare"
    _write(bare / "mindie_sim" / "lone.py", "X = 1\n")
    bare_findings = ag015.check_ag015(bare)
    assert any(v.rule == "AG015-A" and "docs/architecture.md" in v.message for v in bare_findings)


def test_ag015_a_catches_unmapped_phantom_overlap_mermaid_edge_and_bilingual_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Unmapped python, json, and nested build/ data file
    proj = tmp_path / "unmapped"
    _build_minimal_valid_repo(proj)
    _write(proj / "mindie_sim" / "unmapped_mod.py", "X = 1\n")
    _write(proj / "data" / "abi" / "orphan.json", "{}\n")
    _write(proj / "data" / "primitives" / "build" / "table.json", "{}\n")
    rc = arch_gov.main(["--root", str(proj)])
    out = capsys.readouterr().out
    assert rc == arch_gov.EXIT_VIOLATIONS
    assert "AG015-A" in out and "unmapped_mod.py" in out and "orphan.json" in out and "build/table.json" in out

    # 2. Phantom selector, overlapping selectors, & Mermaid edge mismatch
    proj2 = tmp_path / "phantom_mermaid"
    _build_minimal_valid_repo(proj2)
    doc_path = proj2 / "docs" / "architecture.md"
    doc_txt = doc_path.read_text(encoding="utf-8")
    doc_txt = doc_txt.replace('"tools/**"', '"tools/**", "mindie_sim/nonexistent_phantom.py", "mindie_sim/api.py"')
    doc_txt = doc_txt.replace("    L2_PROGRAM --> L0_CONTRACTS\n", "")
    _write(doc_path, doc_txt)
    findings = ag015.check_ag015(proj2)
    msgs = [v.message for v in findings if v.rule == "AG015-A"]
    assert any("phantom" in m for m in msgs)
    assert any("matches multiple selectors" in m and "mindie_sim/api.py" in m for m in msgs)
    assert any("L2_PROGRAM -> L0_CONTRACTS" in m for m in msgs)

    # 3. Forbidden cross-subsystem import edges (direct, import_module, from mindie_sim, tools) & aliased TYPE_CHECKING
    proj3 = tmp_path / "bad_edge"
    _build_minimal_valid_repo(proj3)
    _write(
        proj3 / "mindie_sim" / "observed_contracts" / "attention.py",
        "from typing import TYPE_CHECKING as TC\n"
        "import importlib\n"
        "if TC:\n"
        "    from mindie_sim.prediction.operators.demo import kernel\n"
        "from mindie_sim import api\n"
        "import tools.probe\n"
        'def go(): return importlib.import_module("mindie_sim.cli")\n',
    )
    findings3 = [v for v in ag015.check_ag015(proj3) if v.rule == "AG015-A"]
    assert not any("demo.kernel" in v.message for v in findings3)
    assert any("L0_CONTRACTS -> L5_ENTRY" in v.message and "mindie_sim.api" in v.message for v in findings3)
    assert any("L0_CONTRACTS -> L5_ENTRY" in v.message and "mindie_sim.cli" in v.message for v in findings3)
    assert any("L0_CONTRACTS -> DEV_TOOLS" in v.message and "tools.probe" in v.message for v in findings3)

    # 4. Chained Mermaid edge in bilingual mirror & wildcard recursive selector matching
    proj4 = tmp_path / "bilingual"
    _build_minimal_valid_repo(proj4)
    en_doc = (proj4 / "docs" / "architecture.md").read_text(encoding="utf-8").replace(
        "DEV_TOOLS --> L5_ENTRY", "DEV_TOOLS --> L5_ENTRY\n    L0_CONTRACTS --> Ghost --> DEV_TOOLS",
    )
    _write(proj4 / "docs" / "architecture.en.md", en_doc)
    findings4 = ag015.check_ag015(proj4)
    assert any(v.rule == "AG015-A" and "architecture.en.md" in v.message for v in findings4)
    assert ag015._match_selector("mindie_sim/prediction/operators/demo/kernel.py", "mindie_sim/prediction/op*/**")


def test_ag015_b_catches_unreviewed_all_barrel_unwitnessed_impl_dead_hook_and_collisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Unreviewed __all__, pure barrel hiding zombie, unread JSON cannot forge witness, tools/ cannot rescue
    proj = tmp_path / "barrel_and_all"
    _build_minimal_valid_repo(proj)
    init_path = proj / "mindie_sim" / "__init__.py"
    bad_all = [*sorted(ag015.PUBLIC_ENTRYPOINT_CONTRACT["init_all"]), "sneaky_zombie"]
    init_path.write_text(
        re.sub(r"^__all__ = .*$", f"__all__ = {bad_all!r}", init_path.read_text(encoding="utf-8"), flags=re.M),
        encoding="utf-8",
    )
    _write(proj / "mindie_sim" / "prediction" / "operators" / "demo" / "zombie.py", "Z = 1\n")
    _write(
        proj / "mindie_sim" / "prediction" / "operators" / "demo" / "__init__.py",
        "from .zombie import Z\n__all__ = ['Z']\n",
    )
    _write(proj / "mindie_sim" / "cli" / "unused.json", '{"note": "ghost_key"}\n')
    _write(proj / "tools" / "probe.py", "from mindie_sim.prediction.operators.demo.zombie import Z\n")
    rc = arch_gov.main(["--root", str(proj)])
    out = capsys.readouterr().out
    assert rc == arch_gov.EXIT_VIOLATIONS
    assert "AG015-B" in out
    assert "sneaky_zombie" in out
    assert "mindie_sim.prediction.operators.demo.zombie" in out

    # 2. Unwitnessed alias, dead hook not rescued by uncalled fn, missing lower_op, aliased import_module, collision
    proj2 = tmp_path / "impl_and_hook"
    _build_minimal_valid_repo(proj2)
    reg_path = proj2 / "mindie_sim" / "prediction" / "program" / "implementation_registry.py"
    reg_path.write_text(
        "class Implementation:\n"
        "    def __init__(self, module, operators=(), blocks=(), options=()):\n"
        "        self.module = module\n"
        "IMPLEMENTATIONS: dict[str, Implementation] = {\n"
        '    "ascend_demo_kernel": Implementation(\n'
        '        module="mindie_sim.prediction.operators.demo.kernel", operators=("demo_op",)\n'
        "    ),\n"
        '    "ghost_key": Implementation(\n'
        '        module="mindie_sim.prediction.operators.demo.kernel", blocks=("demo_block",)\n'
        "    ),\n"
        "}\n"
        "def select(bindings, selector):\n"
        "    return bindings.get(selector)\n",
        encoding="utf-8",
    )
    _write(proj2 / "mindie_sim" / "cli" / "unused.json", '{"note": "ghost_key"}\n')
    _write(proj2 / "mindie_sim" / "cli" / "export" / "__init__.py", "OK = True\n")
    _write(
        proj2 / "mindie_sim" / "cli" / "__init__.py",
        "from mindie_sim.prediction.program import implementation_registry as reg\n"
        "def main():\n"
        '    return reg.select({"demo_op": "ascend_demo_kernel"}, "demo_op")\n'
        "def _never_called():\n"
        '    s = reg.select({"demo_op": "ascend_demo_kernel"}, "demo_op")\n'
        "    return s.module.wrap(1)\n",
    )
    kern_path = proj2 / "mindie_sim" / "prediction" / "operators" / "demo" / "kernel.py"
    kern_path.write_text(
        kern_path.read_text(encoding="utf-8")
        + "from importlib import import_module as dyn_load\n"
        + "def wrap(x): return x\n"
        + "def load_dyn(name): return dyn_load(name)\n",
        encoding="utf-8",
    )
    findings2 = ag015.check_ag015(proj2)
    b_msgs = [v.message for v in findings2 if v.rule == "AG015-B"]
    assert any("ghost_key" in m and "no rooted production selector witness" in m for m in b_msgs)
    assert any("missing mandatory lower_op()" in m for m in b_msgs)
    assert any("governed hook 'wrap'" in m for m in b_msgs)
    assert any("open-ended dynamic import_module" in m for m in b_msgs)
    assert any("collides with package" in m for m in b_msgs)


def test_ag015_c_catches_foreign_hw_reassignments_bad_repos_profiles_and_allows_ascend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proj = tmp_path / "foreign_hw"
    _build_minimal_valid_repo(proj)
    _write(proj / "mindie_sim" / "backends" / "nvidia.py", "PROFILE = {}\n")
    _write(
        proj / "mindie_sim" / "backends" / "profiles" / "hopper.json",
        '{"backend": "cuda", "compute_capability": "sm90"}\n',
    )
    _write(
        proj / "mindie_sim" / "prediction" / "operators" / "vector" / "fast_norm.py",
        'SRC = "csrc/kernels/rmsnorm_kernel.cu"\n'
        "def lower_op(op):\n"
        "    return op\n",
    )
    _write(
        proj / "mindie_sim" / "prediction" / "operators" / "demo" / "kernel.py",
        'TARGET_BACKEND = "ascend"\n'
        'TARGET_BACKEND = "cuda"\n'
        "class SourceRef:\n"
        "    def __init__(self, repository, revision, relative_path, source_sha256):\n"
        "        pass\n"
        "def extra(): return None\n"
        "SOURCES = (\n"
        f'    SourceRef("vllm-project/vllm-ascend-cuda", "{_VALID_REV}", "../external/kernel.cpp", "{_VALID_SHA}"),\n'
        "    extra(),\n"
        ")\n"
        "def recipes():\n"
        "    return {}\n",
    )
    rc = arch_gov.main(["--root", str(proj)])
    out = capsys.readouterr().out
    assert rc == arch_gov.EXIT_VIOLATIONS
    c_lines = [line for line in out.splitlines() if "AG015-C" in line]
    assert any("backends/nvidia.py" in line for line in c_lines)
    assert any("hopper.json" in line for line in c_lines)
    assert any("fast_norm.py" in line and ".cu" in line for line in c_lines)
    assert any("kernel.py" in line and "TARGET_BACKEND" in line for line in c_lines)
    assert any("vllm-project/vllm-ascend-cuda" in line for line in c_lines)
    assert any("../external/kernel.cpp" in line for line in c_lines)
    assert any("unresolvable/opaque SourceRef" in line for line in c_lines)


def test_ag015_unconditional_non_subtraction_and_ci_scope_contract() -> None:
    v_a = arch_gov.Violation("AG015-A", PurePosixPath("docs/architecture.md"), 1, "fp_a", "unmapped")
    v_b = arch_gov.Violation("AG015-B", PurePosixPath("mindie_sim/foo.py"), 1, "fp_b", "unreachable")
    v_c = arch_gov.Violation("AG015-C", PurePosixPath("mindie_sim/backends/nvidia.py"), 1, "fp_c", "foreign")
    findings = (v_a, v_b, v_c)

    assert arch_gov.filter_allowlisted(findings) == findings
    assert arch_gov.new_violations(findings, findings) == findings

    ci_scope = _load_ci_scope()
    assert ci_scope.is_pure_doc_path("docs/architecture.md") is False
    assert ci_scope.is_pure_doc_path("docs/architecture.en.md") is False
    assert ci_scope.is_pure_doc_path("CONTRIBUTING.md") is True
    assert ci_scope.is_pure_doc_path("docs/modeling-fidelity.md") is True
    ci_scope_src = (REPO_ROOT / ".github" / "scripts" / "ci_scope.py").read_text(encoding="utf-8")
    assert '"--no-renames"' in ci_scope_src

    ci_yml = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    quality_section = ci_yml.split("\n  test:\n", 1)[0]
    assert "python3 -B scripts/check_arch_governance.py --root . --strict" in quality_section
