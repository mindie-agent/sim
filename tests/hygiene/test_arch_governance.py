"""Architecture governance gate (AG001..AG011) verification.

[Role] Verify ``scripts/check_arch_governance.py`` catches flat ISA/CANN module
  sprawl, direct/dynamic/re-exported leaf access, ``service_modes`` and
  ``compute_peaks`` capability sniffing, missing ISA leaf provenance, silent
  physical hardware defaults, upward layer DAG imports, artificial shard sprawl,
  and unreachable operator modules while exempting model catalog JSON configs,
  legal ``service_rate`` lookups, and ``None`` optional overrides.
[Invariants] Real repository has zero non-allowlisted violations and zero
  violations under ``mindie_sim/prediction/operators/<family>/**``. Multiset
  differential subtraction never waives governed operator packages.
[Timeouts & Locks] Local Tier-0 AST scans and bounded physical probes.
[Side-Effect Policy] Writes synthetic fixtures only inside pytest ``tmp_path``.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath

import pytest

from _support.linear_cases import gdn_decode_source_case
from mindie_sim.prediction.program import implementation_registry
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import builtin_family_registry
from mindie_sim.prediction.operators.linear_attention import gdn_decode, gdn_recurrent
from mindie_sim.prediction.operators.sparse_indexer import bf16_lightning
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.common import ModelingError
from scripts import _arch_governance_runtime as runtime_checks
import scripts.check_arch_governance as arch_gov

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEXER_HEADS = 4
INDEXER_HEAD_DIM = 128
INDEXER_COMPRESSION = 4


def test_real_repository_passes_arch_governance_gate():
    real_all = arch_gov.scan_repository(REPO_ROOT)
    assert arch_gov.filter_allowlisted(real_all) == ()
    assert not any(
        len(v.path.parts) == 5 and v.path.parts[:3] == ("mindie_sim", "prediction", "operators")
        for v in real_all
    )


def test_arch_governance_rules_ag001_through_ag005_and_differential_mode(tmp_path):
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "models" / "qwen3.5-35b-a3b.json").write_text("{}", encoding="utf-8")

    pkg = tmp_path / "mindie_sim" / "prediction"
    fam = pkg / "operators" / "demo"
    fam.mkdir(parents=True)
    (pkg / "bad_arch22.py").write_text("X = 1\n", encoding="utf-8")
    (fam / "__init__.py").write_text(
        "from . import arch22\n__all__ = ['arch22']\n", encoding="utf-8"
    )
    (fam / "arch22.py").write_text('"""Missing abi and sources."""\nVAL = 1\n', encoding="utf-8")
    (pkg / "bad_caller.py").write_text(
        "import importlib\n"
        "from mindie_sim.prediction.operators.demo import arch22\n"
        "from mindie_sim.prediction.operators import demo as d\n"
        "def f(ctx):\n"
        "    _ = d.arch22\n"
        "    _ = getattr(d, 'arch35')\n"
        "    importlib.import_module('mindie_sim.prediction.operators.demo.arch22')\n"
        "    if 'mte2_aic_dram' in ctx.service_modes:\n"
        "        return 1\n"
        "    if ctx.service_modes.get('mte2_aiv_dram'):\n"
        "        return 2\n"
        "    if 'fp8' in ctx.compute_peaks:\n"
        "        return 3\n"
        "    if ctx.peak('bf16') > 100.0:\n"
        "        return 4\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (pkg / "good_caller.py").write_text(
        "def ok(ctx):\n"
        "    return ctx.service_rate('mte2_aiv_dram')\n",
        encoding="utf-8",
    )

    found = arch_gov.scan_repository(tmp_path)
    rules_by_file: dict[str, set[str]] = {}
    for item in found:
        rules_by_file.setdefault(item.path.as_posix(), set()).add(item.rule)

    assert "configs/models/qwen3.5-35b-a3b.json" not in rules_by_file
    assert "mindie_sim/prediction/good_caller.py" not in rules_by_file
    assert "AG001" in rules_by_file["mindie_sim/prediction/bad_arch22.py"]
    assert "AG002" in rules_by_file["mindie_sim/prediction/operators/demo/__init__.py"]
    assert "AG005" in rules_by_file["mindie_sim/prediction/operators/demo/arch22.py"]
    assert {"AG002", "AG003", "AG004"} <= rules_by_file["mindie_sim/prediction/bad_caller.py"]

    delta = arch_gov.new_violations(found, found)
    assert len(delta) > 0
    assert all(
        v.path.as_posix().startswith("mindie_sim/prediction/operators/demo/")
        for v in delta
    )


def test_arch_governance_tightened_loopholes(tmp_path):
    pkg = tmp_path / "mindie_sim" / "prediction"
    fam = pkg / "operators" / "demo"
    fam.mkdir(parents=True)
    # Owning family __init__.py static absolute import is allowed; aliased __all__ re-export is rejected.
    (fam / "__init__.py").write_text(
        "import mindie_sim.prediction.operators.demo.arch22 as engine\n"
        "__all__ = ['engine']\n",
        encoding="utf-8",
    )
    # AG005 loopholes: comment (not docstring) Upstream-ABI, bare AnnAssign STRATEGY, 7-char SHA.
    (fam / "arch22.py").write_text(
        "# Upstream-ABI: fake_comment_abi\n"
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY: object\n"
        "SOURCES = (\n"
        "    SourceRef('cann/ops-transformer', '2d3f4f6', 'a/b.cpp', '0' * 64),\n"
        ")\n",
        encoding="utf-8",
    )
    # Keyword family directory must be rejected as an unmanaged ISA leaf (AG001).
    kw_fam = pkg / "operators" / "class"
    kw_fam.mkdir(parents=True)
    (kw_fam / "arch22.py").write_text("X = 1\n", encoding="utf-8")

    # AG003/AG004 subscript & compute_peaks map truthiness sniffing.
    (pkg / "subscript_sniff.py").write_text(
        "def g(ctx):\n"
        "    if 'mte2_aiv_dram' in ctx['service_modes']:\n"
        "        return 1\n"
        "    if ctx.compute_peaks:\n"
        "        return 2\n"
        "    if 'fp8' in ctx['compute_peaks']:\n"
        "        return 3\n"
        "    return 0\n",
        encoding="utf-8",
    )

    found = arch_gov.scan_repository(tmp_path)
    by_file: dict[str, list[arch_gov.Violation]] = {}
    for item in found:
        by_file.setdefault(item.path.as_posix(), []).append(item)

    init_msgs = [v.message for v in by_file.get("mindie_sim/prediction/operators/demo/__init__.py", ())]
    assert any("must not re-export leaf 'engine'" in m for m in init_msgs)
    assert not any("forbidden leaf access" in m for m in init_msgs)

    leaf_msgs = [v.message for v in by_file.get("mindie_sim/prediction/operators/demo/arch22.py", ())]
    assert any("module docstring" in m for m in leaf_msgs)
    assert any("STRATEGY" in m for m in leaf_msgs)
    assert any("malformed SourceRef" in m or "SourceRef must statically" in m for m in leaf_msgs)

    kw_rules = {v.rule for v in by_file.get("mindie_sim/prediction/operators/class/arch22.py", ())}
    assert "AG001" in kw_rules

    sniff_rules = {v.rule for v in by_file.get("mindie_sim/prediction/subscript_sniff.py", ())}
    assert {"AG003", "AG004"} <= sniff_rules

    # Valid leaf scanned with --refs-root pointing at a directory lacking the repo must emit AG005.
    valid_sha = "a" * 64
    valid_rev = "b" * 40
    (fam / "arch22.py").write_text(
        '"""Valid leaf.\n\nUpstream-ABI: aclnnDemo\n"""\n'
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY = object()\n"
        f"SOURCES = (SourceRef('cann/missing-repo', '{valid_rev}', 'src/a.cpp', '{valid_sha}'),)\n",
        encoding="utf-8",
    )
    missing_refs = tmp_path / "empty_refs"
    missing_refs.mkdir()
    refs_found = arch_gov.scan_repository(tmp_path, refs_root=missing_refs)
    assert any(
        v.rule == "AG005" and "refs repo directory not found" in v.message
        for v in refs_found
    )


def test_arch_governance_repo_map_overrides_refs_root_and_validates_sha(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    repo_dir = tmp_path / "custom-sibling"
    repo_dir.mkdir(parents=True)
    payload = b"def kernel():\n    return 42\n"
    rev = "c" * 40
    sha = hashlib.sha256(payload).hexdigest()
    seen_dirs: list[str] = []

    def fake_run(cmd, **_kwargs):
        seen_dirs.append(cmd[2])
        if cmd[2] != str(repo_dir.resolve()):
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        if cmd[3:5] == ["cat-file", "-t"]:
            target = cmd[5]
            if target == rev:
                return SimpleNamespace(returncode=0, stdout="commit\n", stderr="")
            if target == "e" * 40:
                return SimpleNamespace(returncode=0, stdout="tree\n", stderr="")
            if target == f"{rev}:pkg/kernel.py":
                return SimpleNamespace(returncode=0, stdout="blob\n", stderr="")
            if target == f"{rev}:pkg":
                return SimpleNamespace(returncode=0, stdout="tree\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        if cmd[3:5] == ["cat-file", "blob"] and cmd[5] == f"{rev}:pkg/kernel.py":
            return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"missing")

    monkeypatch.setattr(arch_gov.subprocess, "run", fake_run)

    proj = tmp_path / "proj"
    fam = proj / "mindie_sim" / "prediction" / "operators" / "demo"
    fam.mkdir(parents=True)
    (fam / "arch35.py").write_text(
        '"""Demo leaf.\n\nUpstream-ABI: aclnnDemo\n"""\n'
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY = object()\n"
        f"SOURCES = (SourceRef('org/custom-sibling', '{rev}', 'pkg/kernel.py', '{sha}'),)\n",
        encoding="utf-8",
    )
    empty_refs = tmp_path / "empty_refs"
    empty_refs.mkdir()

    assert arch_gov.scan_repository(
        proj, refs_root=empty_refs, repo_map={"custom-sibling": repo_dir}
    ) == ()
    assert str(repo_dir.resolve()) in seen_dirs
    assert arch_gov.main([
        "--root", str(proj),
        "--refs-root", str(empty_refs),
        "--repo-map", f"org/custom-sibling={repo_dir}",
    ]) == arch_gov.EXIT_OK
    assert arch_gov.main(["--root", str(proj), "--repo-map", "broken_entry"]) == arch_gov.EXIT_USAGE

    bad_sha = "d" * 64
    (fam / "arch35.py").write_text(
        '"""Demo leaf.\n\nUpstream-ABI: aclnnDemo\n"""\n'
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY = object()\n"
        f"SOURCES = (SourceRef('org/custom-sibling', '{rev}', 'pkg/kernel.py', '{bad_sha}'),)\n",
        encoding="utf-8",
    )
    mismatch = arch_gov.scan_repository(
        proj, refs_root=empty_refs, repo_map={"custom-sibling": repo_dir}
    )
    assert any(v.rule == "AG005" and "SHA-256 mismatch" in v.message for v in mismatch)

    tree_rev = "e" * 40
    (fam / "arch35.py").write_text(
        '"""Demo leaf.\n\nUpstream-ABI: aclnnDemo\n"""\n'
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY = object()\n"
        f"SOURCES = (SourceRef('org/custom-sibling', '{tree_rev}', 'pkg/kernel.py', '{sha}'),"
        f" SourceRef('org/custom-sibling', '{rev}', 'pkg', '{sha}'))\n",
        encoding="utf-8",
    )
    bad_types = arch_gov.scan_repository(
        proj, refs_root=empty_refs, repo_map={"custom-sibling": repo_dir}
    )
    assert any(v.rule == "AG005" and "not a git commit" in v.message for v in bad_types)
    assert any(v.rule == "AG005" and "not a git blob" in v.message for v in bad_types)


def test_governance_checks_live_recipe_and_implementation_ownership(monkeypatch):
    assert runtime_checks.verify_ag006_ownership() == ()
    book = RecipeBook.from_mapping({})
    book.recipes[bf16_lightning.RECIPE] = PlannedRecipe(lambda op, geometry: None)
    monkeypatch.setattr(RecipeBook, "from_mapping", classmethod(lambda cls, raw: book))

    name = bf16_lightning.IMPLEMENTATION
    entry = implementation_registry.IMPLEMENTATIONS[name]
    monkeypatch.setitem(implementation_registry.IMPLEMENTATIONS, name, replace(
        entry, module="mindie_sim.prediction.family_causal_conv",
    ))
    decode_recipes = gdn_decode.recipes()
    monkeypatch.setattr(gdn_decode, "recipes", lambda: {
        **decode_recipes, "unowned_decode_recipe": decode_recipes[gdn_decode.CONV],
    })

    findings = runtime_checks.verify_ag006_ownership()
    assert {item.fingerprint for item in findings} == {
        f"AG006:recipe:{bf16_lightning.RECIPE}",
        f"AG006:implementation:{name}",
        "AG006:implementation:vllm_ascend_gdn_decode:unowned_decode_recipe",
    }


def test_composed_decode_dispatches_recurrence_inside_linear_attention():
    book, graph, geometry = runtime_checks._gdn_decode_probe()
    op, = (op for op in graph if op.recipe == gdn_recurrent.IMPLEMENTATION)
    modules = runtime_checks._dispatched_builder_modules(book, op, geometry, gdn_decode.build)
    assert len(modules) == 1
    assert modules[0].startswith("mindie_sim.prediction.operators.linear_attention.")


def test_governance_rejects_external_gdn_dispatch_callback(monkeypatch):
    book, graph, geometry = runtime_checks._gdn_decode_probe()
    op, = (op for op in graph if op.recipe == gdn_decode.GATING)
    original = gdn_decode._BUILDERS[op.recipe]
    calls = []

    @wraps(original)
    def external_builder(op, geometry):
        calls.append(op.op_id)
        return original(op, geometry)

    monkeypatch.setitem(gdn_decode._BUILDERS, op.recipe, external_builder)
    book.prepare(op, geometry)
    assert calls == [op.op_id]
    findings = runtime_checks.verify_ag006_ownership()
    assert any(
        item.rule == "AG006" and item.fingerprint == f"AG006:dispatcher:gdn_decode:{op.recipe}"
        and __name__ in item.message
        for item in findings
    )


def test_governance_rejects_external_implementation_recipe_builder(monkeypatch):
    def external_builder(op, geometry):
        return bf16_lightning._build(op, geometry)

    monkeypatch.setattr(bf16_lightning, "recipes", lambda: {
        bf16_lightning.RECIPE: PlannedRecipe(external_builder),
    })
    findings = runtime_checks.verify_ag006_ownership()
    assert any(
        item.rule == "AG006" and item.fingerprint ==
        f"AG006:implementation:{bf16_lightning.IMPLEMENTATION}:{bf16_lightning.RECIPE}:builder"
        for item in findings
    )


def test_governance_observes_indexer_builder_used_by_dispatch(monkeypatch):
    published = bf16_lightning.recipes()
    original = bf16_lightning._build

    def external_builder(op, geometry):
        return original(op, geometry)

    monkeypatch.setattr(bf16_lightning, "recipes", lambda: published)
    monkeypatch.setattr(bf16_lightning, "_build", external_builder)
    findings = runtime_checks.verify_ag006_ownership()
    assert any(
        item.rule == "AG006" and item.fingerprint == f"AG006:dispatch:{bf16_lightning.RECIPE}"
        and __name__ in item.message
        for item in findings
    )


def test_strict_gate_checks_runtime_with_and_without_differential_base(monkeypatch):
    monkeypatch.setattr(arch_gov, "scan_repository", lambda *args, **kwargs: ())
    monkeypatch.setattr(arch_gov, "scan_runtime", lambda root: (
        arch_gov.Violation("AG006", PurePosixPath("recipes.py"), 1, "AG006:recipe:test", "invalid owner"),
    ))
    assert arch_gov.main([]) == arch_gov.EXIT_OK
    assert arch_gov.main(["--strict"]) == arch_gov.EXIT_VIOLATIONS
    assert arch_gov.main(["--strict", "--base", "HEAD"]) == arch_gov.EXIT_VIOLATIONS


def test_governance_rejects_heuristic_identifiers_in_nested_operator_modules(tmp_path):
    operators = tmp_path / "mindie_sim" / "prediction" / "operators"
    family = operators / "demo"
    nested = family / "planner"
    nested.mkdir(parents=True)
    (operators / "shared.py").write_text("overlap_decay = 1\n", encoding="utf-8")
    (family / "valid.py").write_text(
        "# overlap_decay is prohibited\n"
        "def plan(state_decay):\n"
        "    return state_decay\n",
        encoding="utf-8",
    )
    invalid = nested / "invalid.py"
    invalid.write_text(
        "CACHED_REFILL_DISCOUNT = 1\n"
        "def plan(ctx, overlap_decay):\n"
        "    return ctx.key64_turnaround_fudge * ctx.heuristicFactor\n",
        encoding="utf-8",
    )
    findings = arch_gov.scan_repository(tmp_path)
    assert {item.message.split("'")[1] for item in findings} == {
        "CACHED_REFILL_DISCOUNT", "overlap_decay", "key64_turnaround_fudge", "heuristicFactor",
    }
    assert all(item.rule == "AG007" and item.path == invalid.relative_to(tmp_path) for item in findings)
    assert arch_gov.new_violations(findings, findings) == findings


def test_discrimination_checks_fixed_context_and_real_cache_byte_ledgers(monkeypatch):
    assert arch_gov.verify_ag007_discrimination() == ()
    assert arch_gov.verify_ag007_discrimination(context_tokens=1022, multi_query_tokens=3) == ()

    cold, warm = runtime_checks._cache_ledgers(None)
    assert cold.logical_tensor_bytes == warm.logical_tensor_bytes < warm.aligned_transfer_bytes
    assert cold.aligned_transfer_bytes == warm.aligned_transfer_bytes
    assert cold.dram_bus_transaction_bytes > warm.dram_bus_transaction_bytes

    context = 1024
    query_sizes = []

    def constant_work(query_tokens, context_tokens):
        assert context_tokens == context
        query_sizes.append(query_tokens)
        return context_tokens

    monkeypatch.setattr(runtime_checks, "_cache_ledgers", lambda pattern: (cold, cold))
    assert arch_gov.verify_ag007_discrimination(constant_work, context_tokens=context) == (
        "cold and warm L2 reads must use less DRAM when warm",
        "same-context multi-query plan must increase physical work",
    )
    assert query_sizes == [1, 2]
    monkeypatch.setattr(runtime_checks, "_cache_ledgers", lambda pattern: (
        cold, replace(warm, logical_tensor_bytes=0),
    ))
    assert arch_gov.verify_ag007_discrimination() == (
        "cold and warm L2 reads changed logical or aligned bytes",
    )


def test_discrimination_rejects_uniform_indexer_qk_flop_discount(monkeypatch):
    original = bf16_lightning.make_op

    def discounted_op(**kwargs):
        op = original(**kwargs)
        flops = 0.5 * op.flops
        return replace(op, flops=flops, matrix_flops=(("bf16", flops),))

    monkeypatch.setattr(bf16_lightning, "make_op", discounted_op)
    assert arch_gov.verify_ag007_discrimination() == (
        "BF16 indexer QK FLOPs for q=1, context=1024 must be 262144, got 131072",
        "BF16 indexer QK FLOPs for q=2, context=1024 must be 524288, got 262144",
    )
    findings = runtime_checks.scan_runtime(REPO_ROOT)
    assert [item.rule for item in findings] == ["AG007", "AG007"]


def test_lightning_defaults_execute_through_the_sparse_indexer_owner():
    query_tokens, context_tokens = 2, 1024
    op = bf16_lightning.make_op(
        query_rows=query_tokens, context=context_tokens, heads=INDEXER_HEADS,
        head_dim=INDEXER_HEAD_DIM, compression=INDEXER_COMPRESSION, phase="decode",
    )
    assert builtin_family_registry().lookup(op).name == "sparse_indexer"
    prepared, plan = RecipeBook.from_mapping({}).prepare(op, Geometry(1, {"vector": 2}))
    pairs = sum((context_tokens + row + 1) // INDEXER_COMPRESSION for row in range(query_tokens))
    assert prepared.flops == plan.work.matrix["bf16"] == 2 * pairs * INDEXER_HEADS * INDEXER_HEAD_DIM


@pytest.mark.parametrize("recipe_id", (gdn_decode.CONV, gdn_decode.GATING))
def test_decode_kernel_requires_explicit_binding_then_uses_the_linear_attention_owner(recipe_id):
    parsed, _, ops = gdn_decode_source_case((1,))
    op = next(op for op in ops if op.recipe == recipe_id)
    assert op.plan_required
    assert builtin_family_registry().lookup(op).name == "linear_attention"

    unbound = RecipeBook.from_mapping({})
    with pytest.raises(ModelingError, match=f"^source_plan_binding_required:{recipe_id}$"):
        unbound.select(op)

    geometry = Geometry.from_devices(parsed.machine.dies)
    _, plan = RecipeBook.from_mapping(parsed.runtime.kernels).prepare(op, geometry)
    expected = gdn_decode.build(op, geometry)
    assert plan.work == expected.work
    assert plan.metadata == expected.metadata
    assert plan.work.resources["hbm"] > 0


def test_ag006_and_ag008_block_flat_builders_parallel_concepts_shards_and_bypass_kwargs(tmp_path):
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    ops = pred / "operators" / "demo"
    ops.mkdir(parents=True)
    (pred / "__init__.py").write_text("", encoding="utf-8")
    (pred / "shadow_op.py").write_text(
        "class RaggedSequenceBatch:\n    pass\n"
        "def build(op, geom, ragged_batch=None):\n"
        "    return OperatorPlan(op, geom)\n",
        encoding="utf-8",
    )
    (pred / "demo_helpers.py").write_text("class RequestView:\n    pass\n", encoding="utf-8")
    (ops / "__init__.py").write_text("", encoding="utf-8")
    (ops / "core.py").write_text("def plan(op, *, memory_view=None):\n    return op\n", encoding="utf-8")

    findings = arch_gov.scan_repository(proj)
    by_rule = {(item.rule, item.fingerprint) for item in findings}
    assert any(rule == "AG006" and "flat_builder" in fp for rule, fp in by_rule)
    assert any(rule == "AG006" and "flat_plan_call" in fp for rule, fp in by_rule)
    assert any(rule == "AG008" and "parallel_concept" in fp for rule, fp in by_rule)
    assert any(rule == "AG008" and "duplicate_concept" in fp for rule, fp in by_rule)
    assert any(rule == "AG008" and "shard_file" in fp for rule, fp in by_rule)
    assert any(rule == "AG008" and "bypass_arg" in fp for rule, fp in by_rule)
    assert any(rule == "AG008" and "op_bypass_arg" in fp for rule, fp in by_rule)


def test_ag008_flat_prediction_ratchet_and_operator_package_fanout(tmp_path):
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    ops = pred / "operators" / "bloated"
    ops.mkdir(parents=True)
    for idx in range(arch_gov.MAX_FLAT_PREDICTION_MODULES + 1):
        (pred / f"mod_{idx}.py").write_text("", encoding="utf-8")
    for idx in range(arch_gov.MAX_OPERATOR_PACKAGE_FILES + 1):
        (ops / f"op_{idx}.py").write_text("", encoding="utf-8")
    findings = arch_gov.scan_repository(proj)
    fps = {item.fingerprint for item in findings if item.rule == "AG008"}
    assert "AG008:flat_prediction_ratchet" in fps
    assert "AG008:package_fanout:bloated" in fps


def test_ag009_blocks_silent_physical_defaults_and_allows_none_overrides_and_custom_hw(tmp_path):
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    pred.mkdir(parents=True)
    (pred / "bad_defaults.py").write_text(
        "DEFAULT_AIC_CORES = 32\n"
        "_DEFAULT_VECTOR_CLOCK_HZ = 1e9\n"
        "def default_hierarchy():\n    return None\n"
        "def f(cores: int = 32, *, clock_hz: float = 1e9):\n"
        "    h = default_hierarchy()\n"
        "    eff = cores or 32\n"
        "    return (h, eff, clock_hz)\n",
        encoding="utf-8",
    )
    (pred / "clean_overrides.py").write_text(
        "def g(cores: int | None = None, *, clock_hz: float | None = None, aiv: bool = False):\n"
        "    return (cores or 0, clock_hz or 0.0, aiv)\n",
        encoding="utf-8",
    )
    (proj / "mindie_sim" / "custom_hardware_custom.py").write_text(
        "DEFAULT_AIC_CORES = 32\ndef derive(aic_count: int = 36):\n    return aic_count\n",
        encoding="utf-8",
    )
    (proj / "mindie_sim" / "hardware_derivation.py").write_text(
        "def derive(clock_hz: float = 1.65e9):\n    return clock_hz\n",
        encoding="utf-8",
    )
    findings = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG009"]
    files = {v.path.as_posix() for v in findings}
    fps = {v.fingerprint for v in findings}
    assert files == {"mindie_sim/prediction/bad_defaults.py"}
    assert any("fallback_const" in fp and "DEFAULT_AIC_CORES" in fp for fp in fps)
    assert any("fallback_const" in fp and "_DEFAULT_VECTOR_CLOCK_HZ" in fp for fp in fps)
    assert any("fallback_fn" in fp and "default_hierarchy" in fp for fp in fps)
    assert any("fallback_call" in fp and "default_hierarchy" in fp for fp in fps)
    assert any("param_default" in fp and ":cores" in fp for fp in fps)
    assert any("param_default" in fp and ":clock_hz" in fp for fp in fps)
    assert any("or_fallback" in fp for fp in fps)


def test_ag010_enforces_unidirectional_layer_dag_and_ignores_type_checking(tmp_path):
    proj = tmp_path / "repo"
    for rel, code in (
        ("mindie_sim/schema/bad_l0.py", "from mindie_sim.backends import ascend\n"),
        ("mindie_sim/primitives/bad_l1.py", "from mindie_sim.prediction.runtime import engine\n"),
        ("mindie_sim/prediction/runtime/bad_l2.py", "from mindie_sim.prediction.operators import attention\n"),
        ("mindie_sim/prediction/operators/demo/bad_l3.py", "from mindie_sim import compose\n"),
        (
            "mindie_sim/prediction/runtime/clean_l2.py",
            "from typing import TYPE_CHECKING\n"
            "from mindie_sim.schema import hardware\n"
            "if TYPE_CHECKING:\n    from mindie_sim import compose\n",
        ),
        ("mindie_sim/compose.py", "from mindie_sim.prediction.operators import demo\n"),
    ):
        target = proj / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
    ag010 = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG010"]
    assert {v.path.as_posix() for v in ag010} == {
        "mindie_sim/schema/bad_l0.py",
        "mindie_sim/primitives/bad_l1.py",
        "mindie_sim/prediction/runtime/bad_l2.py",
        "mindie_sim/prediction/operators/demo/bad_l3.py",
    }


def test_ag008_blocks_expanded_shards_private_overflow_singletons_and_allows_arch40(tmp_path):
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    ops = pred / "operators" / "demo"
    onb = proj / "mindie_sim" / "onboarding"
    ops.mkdir(parents=True)
    onb.mkdir(parents=True)
    valid_rev, valid_sha = "a" * 40, "b" * 64
    (ops / "arch40.py").write_text(
        '"""Future ISA leaf.\n\nUpstream-ABI: aclnnDemoV40\n"""\n'
        "from mindie_sim.schema.datapath import SourceRef\n"
        "STRATEGY = object()\n"
        f"SOURCES = (SourceRef('cann/ops', '{valid_rev}', 'src/v40.cpp', '{valid_sha}'),)\n",
        encoding="utf-8",
    )
    (pred / "bad_arch40.py").write_text("X = 1\n", encoding="utf-8")
    (pred / "leaf_sneak.py").write_text("from mindie_sim.prediction.operators.demo import arch40\n", encoding="utf-8")
    for shard in ("x_helper.py", "x_part.py", "x_shard.py", "x_extra.py", "x_copy.py", "_overflow.py"):
        (pred / shard).write_text("X = 1\n", encoding="utf-8")
    (pred / "dup_singletons.py").write_text("class Engine:\n    pass\nclass ParallelSpec:\n    pass\n", encoding="utf-8")
    for idx in range(arch_gov.MAX_ONBOARDING_FILES + 1):
        (onb / f"step_{idx}.py").write_text("X = 1\n", encoding="utf-8")
    findings = arch_gov.scan_repository(proj)
    assert not any(v.path.as_posix().endswith("operators/demo/arch40.py") for v in findings)
    fps = {v.fingerprint for v in findings}
    assert any(v.rule == "AG001" and v.path.as_posix().endswith("bad_arch40.py") for v in findings)
    assert any(v.rule == "AG002" and v.path.as_posix().endswith("leaf_sneak.py") for v in findings)
    assert "AG008:shard_file:mindie_sim/prediction/x_shard.py" in fps
    assert "AG008:shard_file:mindie_sim/prediction/x_copy.py" in fps
    assert "AG008:private_shard_file:mindie_sim/prediction/_overflow.py" in fps
    assert "AG008:duplicate_concept:mindie_sim/prediction/dup_singletons.py:Engine" in fps
    assert "AG008:duplicate_concept:mindie_sim/prediction/dup_singletons.py:ParallelSpec" in fps
    assert "AG008:onboarding_file_ceiling" in fps


def test_ag011_catches_unreachable_operator_modules_and_verifies_live_engine_reachability(tmp_path):
    proj = tmp_path / "repo"
    ops = proj / "mindie_sim" / "prediction" / "operators"
    fam = ops / "demo"
    fam.mkdir(parents=True)
    (ops / "registry.py").write_text("from mindie_sim.prediction.operators.demo import wired\n", encoding="utf-8")
    (fam / "__init__.py").write_text("", encoding="utf-8")
    (fam / "wired.py").write_text("VALUE = 1\n", encoding="utf-8")
    (fam / "orphan.py").write_text("VALUE = 2\n", encoding="utf-8")
    ag011 = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG011"]
    assert [v.path.as_posix() for v in ag011] == ["mindie_sim/prediction/operators/demo/orphan.py"]
    assert runtime_checks.verify_ag011_engine_reachability() == ()


def test_fail_closed_physical_defaults_across_geometry_memory_binding_loader_and_sealer(monkeypatch):
    from types import SimpleNamespace
    from mindie_sim.prediction import memory_traffic
    from mindie_sim.prediction.operators.causal_conv import native_causal_conv_spec as cc_spec
    from mindie_sim.prediction.operators.mx_quant_matrix import mxfp4_geometry
    from mindie_sim.prediction.program import RecipeContext
    from mindie_sim.primitives import loader as prim_loader, sealer as prim_sealer
    from mindie_sim.schema.datapath import datapath_for_die
    from mindie_sim.schema.hardware import Device

    cfg = SimpleNamespace(rows=(128,), k=1024, n=4096)
    with pytest.raises(ModelingError, match="mx_quant_requires_positive_cores"):
        mxfp4_geometry.columns(cfg, 0)
    assert mxfp4_geometry.owner_panel_bytes(cfg, 28) != mxfp4_geometry.owner_panel_bytes(cfg, 32)
    assert mxfp4_geometry.owner_panel_bytes(cfg, 32) != mxfp4_geometry.owner_panel_bytes(cfg, 36)

    op = cc_spec.make_op({"lengths": [8], "dim": 128, "width": 4, "dtype": "bf16"})
    assert dict(op.metadata)["host_tiling"] is None
    with pytest.raises(ModelingError):
        cc_spec.host_tiling(op, Geometry(1, {"vector": 4}))

    with pytest.raises(ModelingError, match="machine_memory_hierarchy_missing"):
        memory_traffic.require_machine_memory_hierarchy(None)

    ctx = RecipeContext(
        op=op, work={}, resource_names={"vector": "r0:vector", "vec_issue": "r0:vec_issue"},
        rates={"r0:vector": 1e9, "r0:vec_issue": 1e9}, core_count=2, resource_units={"vector": 2},
    )
    with pytest.raises(ModelingError, match="bind_vector_clock_hz_missing"):
        ctx.bind_vector(())

    a5_table = prim_loader.load_primitives("Ascend950DT_9582")
    stripped_prims = {
        **a5_table.primitives,
        "P12_intracore_flag": replace(
            a5_table.primitives["P12_intracore_flag"], status="unmeasured", metrics={}
        ),
    }
    stripped_table = replace(a5_table, primitives=stripped_prims)
    no_clock_dp = datapath_for_die(Device(
        rank=0, memory_bytes=1024**3, compute_peaks=(("bf16", 1e12),),
        rates=(("hbm", 1e9), ("vector", 1e9)), buffers=(), compute_units=4,
    ))
    with pytest.raises(ModelingError, match="dispatch_envelope_clock_hz_missing"):
        prim_loader.resolve_dispatch_envelope(stripped_table, no_clock_dp)

    base_spec = prim_sealer.load_sku_hardware_spec("Ascend950DT_9582")
    undeclared_spec = {
        **base_spec,
        "link_bytes_per_s": None, "host_h2d_bytes_per_s": None,
        "host_d2h_bytes_per_s": None, "host_rc_h2d_bytes_per_s": None,
        "host_rc_d2h_bytes_per_s": None,
    }
    monkeypatch.setattr(prim_sealer, "load_sku_hardware_spec", lambda _sku: undeclared_spec)
    sealed = prim_sealer.build_primitive_table_payload(
        sku="Ascend950DT_9582", cann_version="9.3.0",
        endpoint_alias="ep-test", campaign_id="camp-test", raw_measurements=None,
    )
    assert sealed["provenance"]["calibrated"] is False
    assert sealed["primitives"]["P20_host_dma_sync"]["status"] == "unmeasured"
    for key in ("host_h2d_bytes_per_s", "host_d2h_bytes_per_s",
                "host_rc_h2d_bytes_per_s", "host_rc_d2h_bytes_per_s"):
        assert key not in sealed["primitives"]["P20_host_dma_sync"]["metrics"]
    assert sealed["primitives"]["P18_hccl_collective_band"]["status"] == "unmeasured"
    assert "link_bytes_per_s" not in sealed["primitives"]["P18_hccl_collective_band"]["metrics"]
    assert "peer_bytes_per_s" not in sealed["primitives"]["P19_direct_peer_window"]["metrics"]


def test_ag013_export_receipts_blocks_canned_prose_paths_targets_and_hybrid_labels(tmp_path):
    proj = tmp_path / "repo"
    exp = proj / "mindie_sim" / "export"
    exp.mkdir(parents=True)

    # Failing fixtures covering all 4 new AG013 export receipt rules (including split strings & renamed tables).
    (exp / "bad_prose.py").write_text(
        'from types import MappingProxyType\n'
        '_CUSTOM_KERNEL_HINTS = {"matmul": "Tune Cube tiling and double buffer"}\n'
        '_RENAMED_MAP = MappingProxyType({"rmsnorm": "Use fused vector kernel"})\n'
        'def build_card(op, raw_map, source_ref, pct):\n'
        '    t1 = "目标 7" + "8% HBM L*"\n'
        '    t2 = f"L* 达成率 >= {pct}%"\n'
        '    t3 = "≥ 75% Cube"\n'
        '    p1 = "vllm_ascend/ops/linear.py"\n'
        '    p2 = source_ref or "vllm/ops/fused.py"\n'
        '    p3 = raw_map.get("source_ref", "cann/ops/norm.py")\n'
        '    g1 = "避免 ACLGraph 图断裂"\n'
        '    g2 = "AIV/CCU small packet"\n'
        '    g3 = "CCU / AIV mode"\n'
        '    k = op.ascend_kernel.lower()\n'
        '    kernel_action = _RENAMED_MAP.get(k, "Fallback prescriptive tuning prose")\n'
        '    action_note = ""\n'
        '    if k.endswith("norm"):\n'
        '        action_note = "Fuse vector epilogue"\n'
        '    framework_action = raw_map.get("fw") or "Register custom op in framework"\n'
        '    return (t1, t2, t3, p1, p2, p3, g1, g2, g3, kernel_action, action_note, framework_action)\n',
        encoding="utf-8",
    )

    # Clean passing fixture: factual measurements, verified SourceRef, token-alias table, arch-isolated labels.
    valid_rev, valid_sha = "a" * 40, "b" * 64
    (exp / "clean_receipts.py").write_text(
        'from mindie_sim.schema.datapath import SourceRef\n'
        '_TOKEN_ALIASES = {"gemm": ("matmul", "mm")}\n'
        f'REF = SourceRef(repository="vllm-ascend", revision="{valid_rev}", '
        f'relative_path="vllm_ascend/ops/linear.py", source_sha256="{valid_sha}")\n'
        'def render_receipt(lstar_us: float, raw_note: str):\n'
        '    fmt = "0.0%"\n'
        '    meas = f"L*={lstar_us:.2f} us"\n'
        '    a5_lbl, a3_lbl = "A5 CCU", "A3 AIV"\n'
        '    kernel_action = raw_note or ""\n'
        '    return (fmt, meas, a5_lbl, a3_lbl, kernel_action, REF)\n',
        encoding="utf-8",
    )

    findings = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG013"]
    assert {v.path.as_posix() for v in findings} == {"mindie_sim/export/bad_prose.py"}
    fps = {v.fingerprint for v in findings}
    assert any("export_percent_target" in fp for fp in fps)
    assert any("export_source_path_guess" in fp for fp in fps)
    assert any("export_guidance_dispatch" in fp for fp in fps)
    assert any("export_unbound_guidance" in fp for fp in fps)
    assert any("aiv_ccu_hybrid" in fp for fp in fps)


def test_ag014_default_on_strict_catches_model_branches_and_allows_non_model_locals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    pred.mkdir(parents=True)

    # Positive cases: real model-name tokens and model identity comparisons MUST fail by default.
    (pred / "bad_model_branches.py").write_text(
        'def check_models(model, model_family: str, spec: "ModelSpec", raw: dict, tok: str) -> int:\n'
        '    if model.name == "aurora-dense-gqa":\n'
        '        return 1\n'
        '    if model_family == "zephyr-moe":\n'
        '        return 2\n'
        '    if spec.name == "nova-hybrid":\n'
        '        return 3\n'
        '    if raw["model"]["name"] == "unseen-v1":\n'
        '        return 4\n'
        '    if tok == "deepseek" or tok.startswith("qwen") or tok in {"gemma4", "minimax_m3", "step5"}:\n'
        '        return 5\n'
        '    if model.name == "deepseek_mtp":\n'
        '        return 6\n'
        '    return 0\n',
        encoding="utf-8",
    )

    # Negative cases: non-model local variables, hardware/op names, and upstream wire protocol enums MUST pass.
    (pred / "clean_non_model_locals.py").write_text(
        'WIRE_METHODS = frozenset({"deepseek_mtp", "qwen3_5_mtp", "qwen3_next_mtp"})\n'
        'def classify(event: dict, item: dict, profile: dict, op, spec, term, method: str) -> bool:\n'
        '    if event.get("name") == "Ascend Hardware" or event["name"] == "process_name":\n'
        '        return True\n'
        '    if item["name"] in {"weight", "q_scale", "k_scale", "v_scale"}:\n'
        '        return True\n'
        '    if profile["architecture"] in {"A2", "A3", "A5"}:\n'
        '        return True\n'
        '    family_name = str(getattr(op, "family", ""))\n'
        '    if family_name == "elementwise" or spec.name == "attention":\n'
        '        return True\n'
        '    if str(getattr(term, "name", "")).startswith("kda_"):\n'
        '        return True\n'
        '    return method in WIRE_METHODS\n',
        encoding="utf-8",
    )

    # Default scan_repository() enforces AG014 unconditionally (include_ag014=True by default).
    findings = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG014"]
    assert {v.path.as_posix() for v in findings} == {"mindie_sim/prediction/bad_model_branches.py"}
    fps = {v.fingerprint for v in findings}
    for expected_tok in (
        "aurora-dense-gqa", "zephyr-moe", "nova-hybrid", "unseen-v1",
        "deepseek", "qwen", "gemma4", "minimax_m3", "step5", "deepseek_mtp",
    ):
        assert any(expected_tok in fp for fp in fps), f"missing AG014 catch for {expected_tok!r}: {fps}"

    # Default CLI invocation (without --strict-ag014) hard-fails with 0 staged warnings.
    rc_default = arch_gov.main(["--root", str(proj)])
    out_default = capsys.readouterr().out
    assert rc_default == arch_gov.EXIT_VIOLATIONS
    assert "AG014" in out_default
    assert "staged warnings" not in out_default

    # Compatibility --strict-ag014 flag remains accepted and yields the exact same hard failure.
    rc_compat = arch_gov.main(["--root", str(proj), "--strict-ag014"])
    out_compat = capsys.readouterr().out
    assert rc_compat == arch_gov.EXIT_VIOLATIONS
    assert "staged warnings" not in out_compat

    # Removing the offending file leaves zero violations on the clean non-model fixture.
    (pred / "bad_model_branches.py").unlink()
    assert arch_gov.main(["--root", str(proj)]) == arch_gov.EXIT_OK
    assert "arch governance violations: 0" in capsys.readouterr().out
    assert arch_gov.main(["--root", str(proj), "--strict-ag014"]) == arch_gov.EXIT_OK
    assert "arch governance violations: 0" in capsys.readouterr().out

    # Real repository passes default-on AG014 with zero violations.
    assert arch_gov.scan_ag014(REPO_ROOT) == ()


def test_ag014_hardened_ast_edges_base_non_waiver_and_manifest_top_level(
    tmp_path: Path,
) -> None:
    import json

    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    pred.mkdir(parents=True)

    (pred / "tricky_evasions.py").write_text(
        'import re\n'
        'from mindie_sim.schema.model import ModelSpec as MS\n'
        'T = {"qwen": 1}\n'
        'D = T\n'
        'R_KNOWN = re.compile("qwen")\n'
        'R_NEW = re.compile("aurora")\n'
        'def check_all(cfg: dict, s: MS, model) -> int:\n'
        '    if cfg["architectures"][0] == "FalconForCausalLM":\n'
        '        return 1\n'
        '    match cfg["architectures"][0]:\n'
        '        case "OrcaForCausalLM":\n'
        '            return 2\n'
        '    if getattr(s, "name") == "deepseek_mtp":\n'
        '        return 3\n'
        '    if D.get(model.name, 0) or D[model.name]:\n'
        '        return 4\n'
        '    if R_KNOWN.search(model.name) or R_NEW.search(model.name):\n'
        '        return 5\n'
        '    if model.name.casefold().startswith("zephyr"):\n'
        '        return 6\n'
        '    return 0\n',
        encoding="utf-8",
    )
    (pred / "clean_edges.py").write_text(
        'from pathlib import Path\n'
        '_SEED = {"qwen": 1}\n'
        'def check_path(parent: "ModelSpec", p: Path) -> bool:\n'
        '    reg = dict(_SEED)\n'
        '    reg["other"] = 2\n'
        '    return parent.layers > 0 and p.parent.name == "configs" and bool(reg)\n',
        encoding="utf-8",
    )
    manifest = proj / "mindie_sim" / "stacks" / "vllm_ascend" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "supported_models": ["qwen3-32b"],
                "unmerged_pr_models": ["step-5-preview"],
                "bindings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    findings = arch_gov.scan_ag014(proj)
    by_file: dict[str, set[str]] = {}
    for item in findings:
        by_file.setdefault(item.path.as_posix(), set()).add(item.fingerprint)

    assert "mindie_sim/prediction/clean_edges.py" not in by_file
    py_fps = by_file["mindie_sim/prediction/tricky_evasions.py"]
    for expected in (
        "falconforcausallm", "orcaforcausallm", "deepseek_mtp",
        "d:qwen", "qwen", "aurora", "zephyr",
    ):
        assert any(expected in fp for fp in py_fps), f"missing {expected!r}: {py_fps}"

    manifest_fps = by_file["mindie_sim/stacks/vllm_ascend/manifest.json"]
    assert manifest_fps == {
        "AG014:manifest_top_models:supported_models",
        "AG014:manifest_top_models:unmerged_pr_models",
    }

    # Differential --base subtraction must NEVER waive AG014 violations.
    assert arch_gov.new_violations(findings, findings) == findings

