"""Architecture governance gate (AG001..AG007) verification.

[Role] Verify ``scripts/check_arch_governance.py`` catches flat ISA/CANN module
  sprawl, direct/dynamic/re-exported leaf access, ``service_modes`` and
  ``compute_peaks`` capability sniffing, and missing ISA leaf provenance while
  exempting model catalog JSON configs and legal ``service_rate`` lookups.
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
from mindie_sim.prediction import implementation_registry
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
