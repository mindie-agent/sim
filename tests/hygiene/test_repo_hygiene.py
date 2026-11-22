"""Repository hygiene: orthogonality gate, CI scope, temp ownership, host portability.

[Role] Prove that ``scripts/check_test_orthogonality.py`` (TO001..TO013) flags
  each banned pattern and accepts a clean tree, that ``tests/TAXONOMY.toml``
  itself only names product-orthogonal modules, that ``.github/scripts/
  ci_scope.py`` fails closed for non-documentation changes, that test temp
  roots stay owned until removed, and that CLI I/O stays UTF-8/LF portable.
[Invariants] Every gate rule has one failing synthetic tree; a clean tree has
  zero findings; legacy root tests may shrink or swap imports but never grow;
  docs-only PR scope is true only for pure documentation paths on PR events.
[Timeouts & Locks] Tier 1: local ``git`` in temporary repositories (15 s cap).
[Side-Effect Policy] Writes only under ``tmp_path`` / registered temp roots.
"""
from __future__ import annotations

import argparse

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

import conftest
from _support import builders, captures, engine_oracles, tempdirs
from mindie_sim import __main__ as cli_main
from mindie_sim import custom_hardware, model_catalog

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_TIMEOUT_S = 15


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script("check_test_orthogonality", REPO_ROOT / "scripts" / "check_test_orthogonality.py")
ci_scope = _load_script("ci_scope", REPO_ROOT / ".github" / "scripts" / "ci_scope.py")

MANIFEST = """schema_version = 1
domains = ["schema", "runtime"]
support_dirs = ["_support", "_oracles", "fixtures"]
max_lines = 600
max_modules = 4

[[module]]
path = "schema/test_profiles.py"
tier = "tier0"

[[module]]
path = "runtime/test_engine_core.py"
tier = "tier0"
reserved = true
"""
CLEAN = '"""Clean module."""\nimport pytest\n\npytestmark = pytest.mark.tier0\n\n\ndef test_law():\n    assert 1 + 1 == 2\n'


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT_S, check=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return done.stdout.strip()


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in {"tests/TAXONOMY.toml": MANIFEST, "scripts/.keep": "", **files}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _codes(root: Path, *, base: str | None = None) -> set[str]:
    checker = gate.Gate(root, gate.load_manifest(root / "tests"))
    checker.run(base, base is None)
    return {f.code for f in checker.findings if f.severity == "fail"}


def _body(code: str) -> str:
    return CLEAN.replace("def test_law():\n    assert 1 + 1 == 2\n", code)


VIOLATIONS = [
    pytest.param({"tests/schema/test_profiles_issue34.py": CLEAN}, "TO001", id="process-token"),
    pytest.param({"tests/schema/test_profiles_a5.py": CLEAN}, "TO001", id="sku-token"),
    pytest.param({"tests/test_profiles.py": CLEAN}, "TO002", id="root-level"),
    pytest.param({"tests/unknown/test_profiles.py": CLEAN}, "TO002", id="unknown-domain"),
    pytest.param({"tests/schema/test_unlisted.py": CLEAN}, "TO003", id="unlisted"),
    pytest.param({}, "TO003", id="listed-missing"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN + "\n" * 600}, "TO004", id="oversize"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN.replace("pytestmark = pytest.mark.tier0\n", "")},
                 "TO005", id="no-tier"),
    pytest.param({"tests/schema/test_profiles.py": "import subprocess\n" + CLEAN}, "TO005", id="tier0-subprocess"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN.replace("tier0", "tier1")}, "TO005", id="tier-mismatch"),
    pytest.param({"tests/schema/test_profiles.py": "from test_other import case\n" + CLEAN}, "TO006",
                 id="cross-test-import"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN, "tests/_oracles/meta.py": "import mindie_sim\n"},
                 "TO006", id="oracle-depends-on-product"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    import os\n"
                                                          "    os.environ['HOME'] = 'x'\n")}, "TO007",
                 id="environ-write"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    import tempfile\n"
                                                          "    tempfile.mkdtemp()\n")}, "TO007", id="mkdtemp"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    pytest.skip('mount')\n")},
                 "TO008", id="silent-skip"),
    pytest.param({"tests/schema/test_profiles.py": _body("@pytest.mark.xfail(reason='flaky')\n"
                                                          "def test_law():\n    pass\n")}, "TO008",
                 id="loose-xfail"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_r10_fix_ports():\n    pass\n")}, "TO009",
                 id="round-named-test"),
    pytest.param({"tests/schema/test_profiles.py": _body(
        "from pathlib import Path\nROOT = Path(__file__).parents[2]\n\n\ndef test_law():\n"
        "    (ROOT / 'README.md').read_text()\n")}, "TO010", id="docs-grep"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    import inspect\n"
                                                          "    inspect.signature(print)\n")}, "TO011",
                 id="signature-point-test"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN, "tests/artifacts/receipt.txt": "ok\n"}, "TO013",
                 id="stray-artifact"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN, "tests/fixtures/test_unlisted.py": CLEAN},
                 "TO002", id="test-inside-fixtures"),
    pytest.param({"tests/schema/test_profiles.py": "from tests.runtime import test_engine_core\n" + CLEAN},
                 "TO006", id="from-package-import-test"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    from os import environ\n"
                                                          "    environ['HOME'] = 'x'\n")}, "TO007",
                 id="aliased-environ-write"),
    pytest.param({"tests/schema/test_profiles.py": _body("def test_law():\n    from inspect import signature\n"
                                                          "    signature(print)\n")}, "TO011",
                 id="aliased-signature"),
    pytest.param({"tests/schema/test_profiles.py": CLEAN,
                  "tests/_support/unused.py": "# mention _support.unused in comment only\nVALUE = 1\n"},
                 "TO013", id="comment-only-support-orphan"),
]


@pytest.mark.parametrize(("files", "code"), VIOLATIONS)
def test_gate_flags_each_banned_pattern(tmp_path, files, code):
    assert code in _codes(_tree(tmp_path, files))


def test_gate_accepts_a_clean_tree_with_used_support_and_platform_skipif(tmp_path):
    module = _body("import sys\nfrom _support.axes import SKUS\n\n\n"
                   "@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX only')\n"
                   "@pytest.mark.parametrize('sku', SKUS)\ndef test_law(sku, monkeypatch, tmp_path):\n"
                   "    monkeypatch.setenv('HOME', str(tmp_path))\n    assert sku\n")
    root = _tree(tmp_path, {"tests/schema/test_profiles.py": module, "tests/_support/axes.py": "SKUS = ()\n",
                            "tests/fixtures/README.md": "data\n", "tests/conftest.py": "import pytest\n"})
    assert _codes(root) == set()


def test_gate_ratchet_lets_legacy_root_tests_shrink_or_swap_imports_but_never_grow(tmp_path):
    legacy = "from test_helpers import case\n\n\ndef test_old():\n    assert case\n"
    root = _tree(tmp_path, {"tests/test_cache_math.py": legacy, "tests/test_helpers.py": "case = 1\n"})
    _git(root, "init", "-q")
    _git(root, "-c", "user.email=ci@example.com", "-c", "user.name=CI", "commit", "-qm", "base", "--allow-empty")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=ci@example.com", "-c", "user.name=CI", "commit", "-qm", "legacy")
    base = _git(root, "rev-parse", "HEAD")
    (root / "tests/test_cache_math.py").write_text(legacy.replace("test_helpers", "_support.cases"), encoding="utf-8")
    assert _codes(root, base=base) == set()
    (root / "tests/test_cache_math.py").write_text(legacy + "\n\ndef test_new():\n    pass\n", encoding="utf-8")
    assert "TO002" in _codes(root, base=base)
    (root / "tests/test_added.py").write_text(CLEAN, encoding="utf-8")
    assert "TO002" in _codes(root, base=base)


def test_gate_ratchet_catches_deleted_manifest_module_and_bad_manifests_exit_2(tmp_path):
    root = _tree(tmp_path, {"tests/schema/test_profiles.py": CLEAN})
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=ci@example.com", "-c", "user.name=CI", "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "tests/schema/test_profiles.py").unlink()
    assert "TO003" in _codes(root, base=base)
    for bad in ("schema_version = 2\n", MANIFEST.replace("max_modules = 4", f"max_modules = {gate.MODULE_CEILING + 1}"),
                MANIFEST.replace("max_lines = 600", 'max_lines = "bad"')):
        (root / "tests/TAXONOMY.toml").write_text(bad, encoding="utf-8")
        assert gate.main(["--root", str(root), "--strict"]) == 2


def test_manifest_names_only_orthogonal_modules_within_the_ceiling():
    manifest = gate.load_manifest(REPO_ROOT / "tests")
    data = tomllib.loads((REPO_ROOT / "tests" / "TAXONOMY.toml").read_text(encoding="utf-8"))
    assert len(manifest["modules"]) == len(data["module"]) <= manifest["max_modules"]
    for rel, entry in manifest["modules"].items():
        domain, _, name = rel.partition("/")
        assert domain in manifest["domains"] and name.startswith("test_") and "/" not in name, rel
        assert gate.name_tokens(Path(name).stem) == [], rel
        assert int(entry.get("max_lines", manifest["max_lines"])) < gate.HARD_MAX_LINES, rel


def _commit_repo(root: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=ci@example.com", "-c", "user.name=CI", "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize(("path", "docs"), [
    ("CONTRIBUTING.md", True), ("AGENTS.md", True), ("docs/contributing/guide.md", True),
    (".github/pull_request_template.md", True), ("docs/releases/sim-0.9.19.md", False),
    ("data/observations/sources/README.md", False), ("tests/fixtures/README.md", False),
    ("docs/evidence.json", False), (".github/workflows/ci.yml", False),
])
def test_ci_scope_pure_documentation_paths(path, docs):
    assert ci_scope.is_pure_doc_path(path) is docs


def test_ci_scope_docs_only_for_pull_requests_and_fails_closed_otherwise(tmp_path):
    _git(tmp_path, "init", "-q")
    base = _commit_repo(tmp_path, {"README.md": "# Demo\n", "src.py": "VALUE = 1\n"}, "init")
    head = _commit_repo(tmp_path, {"CONTRIBUTING.md": "# Guide\n", ".github/pull_request_template.md": "## S\n"},
                        "docs")
    docs_only, paths, _ = ci_scope.classify_pr_scope(repo_root=tmp_path, event_name="pull_request",
                                                     base_sha=base, head_sha=head)
    assert docs_only and sorted(paths) == [".github/pull_request_template.md", "CONTRIBUTING.md"]
    assert not ci_scope.classify_pr_scope(repo_root=tmp_path, event_name="push", base_sha=base, head_sha=head)[0]
    for touched in ("src.py", ".github/workflows/ci.yml", "docs/releases/v1.md"):
        changed = _commit_repo(tmp_path, {touched: f"{touched}\n"}, touched)
        assert not ci_scope.classify_pr_scope(repo_root=tmp_path, event_name="pull_request",
                                              base_sha=head, head_sha=changed)[0], touched
        head = changed


@pytest.mark.parametrize(("quality", "docs_only", "test", "wheel", "ok"), [
    ("success", "true", "skipped", "skipped", True),
    ("success", "false", "skipped", "success", False),
    ("failure", "true", "skipped", "skipped", False),
])
def test_ci_scope_needs_payload_only_allows_skips_for_successful_docs_only(quality, docs_only, test, wheel, ok):
    payload = json.dumps({"quality": {"result": quality, "outputs": {"docs_only": docs_only}},
                          "test": {"result": test, "outputs": {}}, "wheel": {"result": wheel, "outputs": {}}})
    assert ci_scope.verify_needs_payload(payload, required_jobs=["quality", "test", "wheel"])[0] is ok


@pytest.mark.skipif(os.name == "nt", reason="POSIX path simulates a missing temp parent")
def test_short_root_fixture_creates_a_missing_platform_temp_and_unwinds(monkeypatch):
    owned = tempdirs.tempdir("mp-", directory="/tmp")
    missing = owned / "sub"
    assert not missing.exists()
    assert len(os.fspath(missing).encode("utf-8")) <= conftest.SHORT_PARENT_MAX_BYTES
    monkeypatch.setattr(conftest.tempfile, "gettempdir", lambda: str(missing))
    monkeypatch.delenv("MINDIE_REPRO_SHORT_ROOT", raising=False)
    patch = pytest.MonkeyPatch()
    try:
        fixture = conftest.isolated_short_root.__wrapped__(patch)
        root = next(fixture)
        assert missing.is_dir() and root.parent == missing
        assert os.environ["MINDIE_REPRO_SHORT_ROOT"] == str(root)
        patch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root / "override"))
        fixture.close()
        assert not root.exists()
        patch.undo()
        assert os.environ.get("MINDIE_REPRO_SHORT_ROOT") is None
    finally:
        patch.undo()
        shutil.rmtree(owned, ignore_errors=True)


@pytest.mark.skipif(os.name == "nt", reason="short POSIX fallback is not used on Windows")
def test_short_root_parent_never_uses_a_long_platform_temp(monkeypatch, tmp_path):
    long_parent = tmp_path / ("long-" + "x" * 40)
    monkeypatch.setattr(conftest.tempfile, "gettempdir", lambda: str(long_parent))
    parent = conftest._short_root_parent()
    assert long_parent.is_dir() and parent != long_parent
    assert len(os.fspath(parent).encode("utf-8")) <= conftest.SHORT_PARENT_MAX_BYTES


def test_temp_roots_stay_owned_until_removal_succeeds(monkeypatch):
    root = tempdirs.tempdir("retained-")
    monkeypatch.setattr(tempdirs, "_ROOTS", [root])
    real_rmtree = tempdirs.shutil.rmtree
    monkeypatch.setattr(tempdirs.shutil, "rmtree", lambda *_a, **_k: None)
    tempdirs.cleanup()
    assert tempdirs._ROOTS == [root]
    monkeypatch.setattr(tempdirs.shutil, "rmtree", real_rmtree)
    tempdirs.cleanup()
    assert tempdirs._ROOTS == [] and not root.exists()


@pytest.fixture(scope="module")
def _module_fixture_env_snapshot():
    return {
        "home": os.environ["HOME"],
        "appdata": os.environ["APPDATA"],
        "tempdir": conftest.tempfile.gettempdir(),
        "session": str(Path(os.environ["MINDIE_CACHE_HOME"]).parent),
    }


def test_isolated_runtime_and_module_scope_redirect_host_paths_and_tempdir(
    _worker_runtime_root, _module_fixture_env_snapshot, monkeypatch
):
    for name, sub in conftest.HOST_PATH_VARS.items():
        assert os.environ[name] == str(_worker_runtime_root / sub), name
    session = Path(os.environ["MINDIE_CACHE_HOME"]).parent
    assert os.environ["TMPDIR"] == str(session / "tmp")
    assert Path(conftest.tempfile.gettempdir()) == session / "tmp"
    assert os.environ["XDG_CACHE_HOME"] == str(session / "xdg-cache")
    assert _module_fixture_env_snapshot["home"].startswith(_module_fixture_env_snapshot["session"])
    assert _module_fixture_env_snapshot["appdata"].startswith(_module_fixture_env_snapshot["session"])
    assert _module_fixture_env_snapshot["tempdir"] == str(session / "tmp")

    class _FakeItem:
        def get_closest_marker(self, _name):
            return None

        def iter_markers(self, name):
            assert name == "requires_capture_mount"
            return [
                pytest.mark.requires_capture_mount("pyproject.toml").mark,
                pytest.mark.requires_capture_mount("tests/fixtures/__absent_parent_mount__").mark,
            ]

    monkeypatch.setenv(conftest.REQUIRE_ALL_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="__absent_parent_mount__"):
        conftest.pytest_runtest_setup(_FakeItem())


class _Cp936Stdout:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, text: str) -> int:
        self.buf.extend(text.encode("cp936"))
        return len(text)

    def flush(self) -> None:
        pass


def test_cli_json_io_is_utf8_lf_and_survives_a_cp936_console(monkeypatch, tmp_path):
    payload = {"note": "latency \u03bcs \u26a1 \u2014 UTF-8", "value": 42}
    src = tmp_path / "case_utf8.json"
    src.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    assert cli_main.read_json(src) == payload
    out_file = tmp_path / "out.json"
    cli_main.emit(payload, out_file)
    raw = out_file.read_bytes()
    assert b"\r\n" not in raw and json.loads(raw.decode("utf-8")) == payload
    console = _Cp936Stdout()
    monkeypatch.setattr(sys, "stdout", console)
    cli_main.emit(payload, None)
    assert json.loads(console.buf.decode("cp936")) == payload


def test_timeout_errors_propagate_through_catalog_and_custom_hardware_io(monkeypatch, tmp_path):
    sku_file = tmp_path / "skus.json"
    sku_file.write_text("{}", encoding="utf-8")

    def raise_timeout(*_a, **_kw):
        raise TimeoutError("io_deadline")

    monkeypatch.setattr(Path, "open", raise_timeout)
    with pytest.raises(TimeoutError):
        custom_hardware.load_custom_hardware_skus_from_file(sku_file)
    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(sku_file))
    with pytest.raises(TimeoutError):
        custom_hardware.list_custom_hardware_skus()
    monkeypatch.setattr(Path, "read_text", raise_timeout)
    model_catalog._load_builtin_model.cache_clear()
    with pytest.raises(TimeoutError):
        model_catalog.load_model(next(iter(model_catalog.ARCHITECTURE_PRESET_CATALOG)))


def test_support_oracles_builders_and_captures_contracts():
    assert engine_oracles.ulp_distance(+0.0, -0.0) == 0
    assert engine_oracles.ulp_distance(1.0, float.fromhex("0x1.0000000000001p+0")) == 1
    assert engine_oracles.roofline_lower_bound({"cube": 20.0, "hbm": 12.0}, {"cube": 10.0, "hbm": 4.0}) == 3.0
    assert engine_oracles.steady_two_stage_makespan(4, 2.0, 5.0) == 2.0 + 3 * 5.0 + 5.0
    assert engine_oracles.ring_collective_volume_per_rank(800, 8, two_phase=True) == 1400.0
    assert engine_oracles.serial_sum([0.1] * 10) == 1.0
    compact = builders.dense_case(layers=2, dp=2, tp=2)
    assert len(compact.machine.dies) == 4 and len(compact.model.layer_plans) == 2
    manifest = captures.observation_protocol(
        "activation-call", ("activation_a.json",), "activation_producer.py.txt"
    )
    assert len(manifest["captures"][0]["sha256"]) == 64


def test_hygiene_rejects_producer_shape_filters(tmp_path: Path) -> None:
    verifier = _load_script("verify_m1", REPO_ROOT / "scripts" / "verify_m1.py")
    directory = tmp_path / "data" / "verification"
    directory.mkdir(parents=True)
    producer = directory / "sample.producer.py"
    args = argparse.Namespace(sim=tmp_path, profiler=tmp_path)
    producer.write_text("for r in rows:\n    if r['status'] != 'ok' or measured_ns <= 0: continue\n", encoding="utf-8")
    assert verifier.hygiene(args)["status"] == "pass"
    producer.write_text(
        "MIN_MIB = +64\ndef collect():\n    min_payload_bytes: int = 4096\n    for row in rows:\n"
        "        if m < 1024: continue\n        if max(n, k) < 14336 and valid: continue\n"
        "        if payload_mib < MIN_MIB: continue\n        if row.get('M', 0) < 512: continue\n"
        "        if n < 32:\n            for item in row:\n                if not item: continue\n"
        "        if sh.get('swa', 0) != 0: continue\n"
        "        if sh[0] >= 512: add(row)\n"
        "        if 'sinkhorn_T960' not in name: add(row)\n"
        "        if b >= STEADY_COLL_MIN_BYTES: out.append(row)\n"
        "    return [row for row in rows if m >= 256]\n", encoding="utf-8",
    )
    result = verifier.hygiene(args)
    assert result["status"] == "fail"
    expected = {f"{line}:shape_size_filter" for line in (5, 6, 7, 8, 12, 13, 14, 15, 16)} | {
        "1:shape_size_filter_constant:MIN_MIB", "3:shape_size_filter_constant:min_payload_bytes"}
    assert set(result["producer_shape_filters"]) == {f"{producer.relative_to(tmp_path)}:{item}" for item in expected}


@pytest.mark.parametrize("mark", [
    "pytest.mark.xfail",
    "pytest.mark.xfail(strict=False, raises=TimingDrift, reason='profile.json:2')",
    "pytest.mark.xfail(strict=True, reason='profile.json:2')",
    "pytest.mark.xfail(strict=True, raises=AssertionError, reason='profile.json:2')",
    "pytest.mark.xfail(strict=True, raises=TimingDrift, reason='profile.json:1')",
    "pytest.mark.xfail(strict=True, raises=TimingDrift, reason='missing.py:2')",
])
def test_xfail_checker_rejects_invalid_exact_parameter_marks_in_legacy_root(tmp_path, monkeypatch, mark):
    path = "tests/test_profiles.py"
    module = f"""import pytest
class TimingDrift(AssertionError):
    pass
narrow = pytest.mark.xfail(strict=True, raises=TimingDrift, reason="profile.json:2")
cases = [
    pytest.param(1, marks=narrow, id="valid"),
    pytest.param(2, marks=[pytest.mark.tier1, {mark}], id="invalid"),
]
@pytest.mark.parametrize("value", cases)
def test_law(value):
    assert value
"""
    root = _tree(tmp_path, {path: module, "profile.json": '{\n  "rate": 1000\n}\n'})
    monkeypatch.setattr(gate, "diff_lines", lambda *_: ([], []))
    checker = gate.Gate(root, gate.load_manifest(root / "tests"))
    checker.check_file(path, "M", "reference")
    only, = checker.findings
    assert only.code == "TO008" and f"{path}::test_law[invalid]" in only.message


@pytest.mark.parametrize("mark", (
    "marks.skip",
    "marks.skipif",
    "marks.skip(reason='unavailable')",
    "pytest.mark.skipif(True, reason='heldout')",
))
@pytest.mark.parametrize("folder", ("tests", "tests/schema"))
def test_gate_rejects_silent_parameter_marks_at_any_test_depth(tmp_path, monkeypatch, mark, folder):
    path = f"{folder}/test_profiles.py"
    module = "from pytest import mark as marks\n" + _body(
        f"@pytest.mark.parametrize('value', [pytest.param(1, marks={mark})])\n"
        "def test_law(value):\n    assert value\n")
    root = _tree(tmp_path, {path: module})
    monkeypatch.setattr(gate, "diff_lines", lambda *_: ([], []))
    checker = gate.Gate(root, gate.load_manifest(root / "tests"))
    checker.check_file(path, "M", "reference")
    finding, = checker.findings
    assert (finding.path, finding.code) == (path, "TO008")
    assert mark in module.splitlines()[finding.line - 1]
