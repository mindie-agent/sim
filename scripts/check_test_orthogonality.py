#!/usr/bin/env python3
"""Product-orthogonal test-suite gate for ``tests/`` (rules TO001..TO013).

[Role] Keep the test tree organised by product layer and mechanism: one
  ``tests/<domain>/test_<mechanism>.py`` per manifest entry, no process /
  ticket / round / SKU / model provenance in names, explicit tiers, no
  cross-test imports, no host pollution, no silent skips.
[Invariants] Read-only. Findings are never downgraded to success: exit 0 only
  when no ``fail`` finding exists, 1 on findings, 2 on usage/manifest errors.
  ``--base REF`` (ratchet) checks files added/renamed/modified versus REF;
  ``--strict`` checks the whole tree and requires every non-reserved manifest
  module on disk. A modified legacy root ``tests/test_*.py`` may only delete
  lines or retarget imports to ``_support``/``_oracles``.
[Timeouts & Locks] Local ``git diff``/``ls-files`` calls bounded by 60 s each;
  no network, no locks.
[Side-Effect Policy] Writes only to stdout.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

GIT_TIMEOUT_S = 60
HARD_MAX_LINES = 800
LITERAL_ENUM_MIN = 8
PROCESS = re.compile(
    r"^(issues?\d*|gh\d+|pr\d+|open|r\d+|round\d+|pass\d+|fix\d+|finding\d+|iter\d+"
    r"|wave\d+[a-z]?|track|stage\d+|slice|rc|nr|s\d+|t\d{2}|m\d|v\d{2,}[a-z0-9]*|cann\d+|\d+"
    r"|closures?|audit|review|regressions?|adversarial|gatekeeper|delta|final|macro|foundation"
    r"|unified|probe\d+|scratch|tmp|legacy|old|backup|fallback|shadow|misc|bugs?|fixes)$")
SKU = re.compile(r"^(a2|a3|a5|910b\d*|910c|950\w*|arch22|arch35|isa22|isa35|dav\d+|g[235])$")
MODEL = re.compile(r"^(qwen\d*|kimi|glm\d*|deepseek|dsv\d+|v4flash|v41|k31|llama\d*|gemma\d*"
                   r"|minimax|mixtral)$")
PROCESS_FUNC = re.compile(r"^(test_|Test)(?:(?:r|round|pass|fix|issue|wave|w|p|t|nr|s|f)\d+_|new_p\d)"
                          r"|_(?:r|f|pass|finding|round|fix)\d+_", re.I)
SHADOW_STEM = re.compile(r"_(legacy|old|backup|copy|fallback)$")
SKU_LITERAL = re.compile(r"^(Ascend\d+\w*|910[BC]\w*|950\w*|arch22|arch35)$")
ABS_HOME = re.compile(r"^(/(Users|home|root)/|[A-Za-z]:\\Users\\)")
DOC_LITERAL = re.compile(r"(?i)(\.md|\.rst)$|^(README|CHANGELOG|CONTRIBUTING)\b")
XFAIL_REASON = re.compile(r"://|\w\.\w+:\d+")
TIERS = ("tier0", "tier1", "live_npu")
TIER0_BANNED_IMPORTS = ("subprocess", "multiprocessing", "socket", "venv")
TIER0_BANNED_CALLS = {("time", "sleep"), ("os", "fork"), ("os", "system")}
ENV_MUTATORS = {"setdefault", "update", "pop", "clear", "popitem"}
TEMP_CALLS = {"mkdtemp", "mkstemp", "TemporaryDirectory"}
FIELD_ENUM = {"__dataclass_fields__", "__annotations__", "get_type_hints"}
ALLOWED_DATA_SUFFIXES = {".py", ".c", ".h"}
NEW_TEST_DEF = re.compile(r"^\s*((async\s+)?def\s+test_|class\s+Test)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    message: str
    severity: str = "fail"


class ManifestError(Exception):
    """The taxonomy manifest is missing or malformed (exit code 2)."""


def _pos_int(raw: object, field: str) -> int:
    if isinstance(raw, bool):
        raise ManifestError(f"{field} must be an integer, got {raw!r}")
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ManifestError(f"{field} must be >= 1, got {value}")
    return value


def load_manifest(tests: Path) -> dict:
    path = tests / "TAXONOMY.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise ManifestError(f"{path}: schema_version must be 1, got {data.get('schema_version')!r}")
    max_lines = _pos_int(data.get("max_lines", 600), "max_lines")
    max_modules = _pos_int(data.get("max_modules", 96), "max_modules")
    if max_lines >= HARD_MAX_LINES:
        raise ManifestError(f"max_lines ({max_lines}) must be < {HARD_MAX_LINES}")
    if max_modules > 96:
        raise ManifestError(f"max_modules ({max_modules}) must be <= 96")
    modules = {}
    for entry in data.get("module", []):
        if not isinstance(entry, dict):
            raise ManifestError(f"module entry must be a table: {entry!r}")
        rel = entry.get("path")
        if not rel or entry.get("tier") not in TIERS:
            raise ManifestError(f"module entry needs path and tier in {TIERS}: {entry}")
        if "max_lines" in entry:
            entry_cap = _pos_int(entry["max_lines"], f"{rel}.max_lines")
            if entry_cap >= HARD_MAX_LINES:
                raise ManifestError(f"{rel}.max_lines ({entry_cap}) must be < {HARD_MAX_LINES}")
        if rel in modules:
            raise ManifestError(f"duplicate manifest module {rel}")
        modules[rel] = entry
    domains = tuple(data.get("domains", ()))
    if not domains:
        raise ManifestError("manifest declares no domains")
    return {"domains": domains, "modules": modules,
            "support_dirs": tuple(data.get("support_dirs", ())),
            "max_lines": max_lines,
            "max_modules": max_modules}


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT_S, check=False,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode:
        raise ManifestError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def changed_paths(root: Path, base: str) -> dict[str, str]:
    """Map repo-relative tests/ paths to A/M/R status versus ``base`` (incl. untracked)."""
    status: dict[str, str] = {}
    for line in git(root, "diff", "--name-status", "-M", base, "--", "tests").splitlines():
        parts = line.split("\t")
        code = parts[0][:1]
        if code == "D":
            continue
        status[parts[-1]] = code
    for rel in git(root, "ls-files", "--others", "--exclude-standard", "--", "tests").splitlines():
        status.setdefault(rel, "A")
    return status


def diff_lines(root: Path, base: str, rel: str) -> tuple[list[str], list[str]]:
    """Added and removed source lines of ``rel`` versus ``base``."""
    diff = git(root, "diff", "-U0", base, "--", rel).splitlines()
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    return added, removed


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def marks(node: ast.AST) -> set[str]:
    """``pytest.mark.<name>`` names referenced anywhere inside ``node``."""
    found = set()
    for sub in ast.walk(node):
        name = dotted(sub) if isinstance(sub, ast.Attribute) else ""
        if name.startswith("pytest.mark."):
            found.add(name.split(".")[2])
    return found


def name_tokens(stem: str) -> list[tuple[str, str]]:
    out = []
    for token in stem.removeprefix("test_").split("_"):
        for label, rx in (("process", PROCESS), ("sku/isa", SKU), ("model", MODEL)):
            if token and rx.match(token):
                out.append((label, token))
                break
    return out


class ModuleCheck(ast.NodeVisitor):
    """AST rules TO005..TO012 for one test or support module."""

    def __init__(self, rel: str, tier: str | None, *, is_test: bool, allow_fields: bool,
                 is_tempdirs: bool, is_oracle: bool, is_conftest: bool):
        self.rel, self.tier, self.is_test = rel, tier, is_test
        self.allow_fields, self.is_tempdirs = allow_fields, is_tempdirs
        self.is_oracle, self.is_conftest = is_oracle, is_conftest
        self.findings: list[Finding] = []
        self.scope_tier1 = 0
        self.repo_names: set[str] = set()
        self.aliases: dict[str, str] = {}

    def resolve(self, node: ast.AST) -> str:
        raw = dotted(node)
        head, sep, tail = raw.partition(".")
        mapped = self.aliases.get(head)
        return f"{mapped}{sep}{tail}" if mapped else raw

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and ("__file__" in ast.unparse(stmt.value) or any(
                    isinstance(n, ast.Name) and n.id in self.repo_names for n in ast.walk(stmt.value))):
                self.repo_names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        left = node
        while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
            left = left.left
        if self.is_test and isinstance(node.op, ast.Div) and isinstance(left, ast.Name) and \
                left.id in self.repo_names and any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                                                   and DOC_LITERAL.search(c.value) for c in ast.walk(node)):
            self.add(node, "TO010", "test reads repository documentation: tests verify code, not prose")
        self.generic_visit(node)

    def add(self, node: ast.AST, code: str, message: str, severity: str = "fail") -> None:
        self.findings.append(Finding(self.rel, getattr(node, "lineno", 1), code, message, severity))

    def tier0_scope(self) -> bool:
        return self.tier == "tier0" and not self.scope_tier1

    def _scoped(self, node: ast.AST) -> None:
        tier1 = any("tier1" in marks(d) for d in getattr(node, "decorator_list", ()))
        self.scope_tier1 += tier1
        if self.is_test and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.name.startswith(("test", "Test"))) and PROCESS_FUNC.search(node.name):
                self.add(node, "TO009", f"process-named test '{node.name}': name the guarded product law")
            for deco in node.decorator_list:
                self._decorator(deco)
        self.generic_visit(node)
        self.scope_tier1 -= tier1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _decorator(self, deco: ast.AST) -> None:
        if not isinstance(deco, ast.Call):
            return
        name = dotted(deco.func)
        kw = {k.arg: k.value for k in deco.keywords}
        if name == "pytest.mark.xfail":
            strict = kw.get("strict")
            reason = kw.get("reason")
            if not (isinstance(strict, ast.Constant) and strict.value is True and isinstance(reason, ast.Constant)
                    and XFAIL_REASON.search(str(reason.value))):
                self.add(deco, "TO008", "xfail needs strict=True and a reason citing upstream file:line or URL")
        elif name == "pytest.mark.skipif":
            cond = deco.args[0] if deco.args else kw.get("condition")
            text = ast.unparse(cond) if cond is not None else ""
            if not re.search(r"sys\.platform|os\.name|platform\.system", text):
                self.add(deco, "TO008", "skipif only on platform conditions; use requires_* markers otherwise")
        elif name == "pytest.mark.parametrize":
            for sub in ast.walk(deco):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and SKU_LITERAL.match(sub.value):
                    self.add(sub, "TO012", f"SKU literal '{sub.value}' in parametrize: use _support.axes",
                             "warn")
                    break

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imported(node, alias.name)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self._imported(node, module)
        for alias in node.names:
            target = f"{module}.{alias.name}".lstrip(".") if module else alias.name
            if alias.name.startswith("test_") or node.level and not module or alias.name == "ProcessPoolExecutor":
                self._imported(node, target)
            if not node.level and module:
                self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def _imported(self, node: ast.AST, module: str) -> None:
        if any(part.startswith("test_") for part in module.split(".")):
            self.add(node, "TO006", f"cross-test import '{module}': move helpers to conftest/_support")
        if self.is_oracle and module.split(".")[0] == "mindie_sim":
            self.add(node, "TO006", "_oracles must stay independent of mindie_sim")
        if self.tier0_scope() and self.is_test and any(
                module == banned or module.startswith(banned + ".") for banned in TIER0_BANNED_IMPORTS) \
                or self.tier0_scope() and self.is_test and module.endswith("ProcessPoolExecutor"):
            self.add(node, "TO005", f"tier0 module imports '{module}': mark tier1 or move to a session fixture")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._env_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._env_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._env_target(target)
        self.generic_visit(node)

    def _env_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Subscript) and self.resolve(target.value) == "os.environ":
            self.add(target, "TO007", "direct os.environ write: use monkeypatch / MonkeyPatch.context()")

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve(node.func)
        head, _, attr = name.rpartition(".")
        if head == "os.environ" and attr in ENV_MUTATORS or name in {"os.putenv", "os.unsetenv"}:
            self.add(node, "TO007", f"{name}(): use monkeypatch")
        if attr in TEMP_CALLS and head.endswith("tempfile") and not self.is_tempdirs:
            self.add(node, "TO007", f"{name}(): use tmp_path or _support.tempdirs")
        if name in {"Path.home", "pathlib.Path.home", "os.path.expanduser"} and not self.is_tempdirs:
            self.add(node, "TO007", f"{name}(): never resolve the real user home in tests")
        if name in {"pytest.skip", "pytest.importorskip", "pytest.xfail"} and not self.is_conftest:
            self.add(node, "TO008", f"silent {name}(): use requires_profiler / requires_capture_mount markers")
        if name in {"importlib.import_module", "pytest.importorskip", "__import__"} and node.args and \
                isinstance(node.args[0], ast.Constant) and str(node.args[0].value).split(".")[-1].startswith("test_"):
            self.add(node, "TO006", f"dynamic import of test module '{node.args[0].value}'")
        if self.tier0_scope() and self.is_test and tuple(name.split(".")[-2:]) in TIER0_BANNED_CALLS:
            self.add(node, "TO005", f"tier0 module calls {name}()")
        if self.is_test and name in {"open", "io.open", "Path", "pathlib.Path"} and any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and DOC_LITERAL.search(a.value)
                for a in node.args):
            self.add(node, "TO010", "test opens a documentation file: tests verify code, not prose")
        if name == "inspect.signature" and not self.allow_fields:
            self.add(node, "TO011", "inspect.signature point test: assert behaviour instead")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FIELD_ENUM and not self.allow_fields:
            self.add(node, "TO011", f"{node.attr} enumeration is reserved for the profiler SSOT module")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FIELD_ENUM and not self.allow_fields:
            self.add(node, "TO011", f"{node.id} enumeration is reserved for the profiler SSOT module")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and self.is_test:
            if ABS_HOME.match(node.value):
                self.add(node, "TO007", f"host-absolute path literal '{node.value[:40]}'")

    def visit_Compare(self, node: ast.Compare) -> None:
        sides = [node.left, *node.comparators]
        if all(isinstance(s, (ast.List, ast.Set, ast.Tuple, ast.Dict)) for s in sides):
            strings = sum(isinstance(e, ast.Constant) and isinstance(e.value, str)
                          for s in sides for e in ast.walk(s))
            if strings >= LITERAL_ENUM_MIN:
                self.add(node, "TO011", "literal-container equality: static enumeration test", "warn")
        self.generic_visit(node)


def module_tier(tree: ast.Module) -> tuple[set[str], int]:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            return marks(node) & set(TIERS), node.lineno
    return set(), 1


class Gate:
    def __init__(self, root: Path, manifest: dict):
        self.root, self.tests, self.m = root, root / "tests", manifest
        self.findings: list[Finding] = []

    def add(self, rel: str, line: int, code: str, message: str, severity: str = "fail") -> None:
        self.findings.append(Finding(rel, line, code, message, severity))

    def classify(self, rel: str) -> str:
        """Return 'test', 'support', 'oracle', 'conftest', 'data', 'manifest' or 'other'."""
        parts = Path(rel).parts[1:]
        if "__pycache__" in parts:
            return "data"
        if rel == "tests/TAXONOMY.toml":
            return "manifest"
        if parts and parts[-1] == "conftest.py":
            return "conftest"
        if parts and parts[-1].startswith("test_") and parts[-1].endswith(".py"):
            return "test"
        if parts and parts[0] == "fixtures":
            return "data"
        if parts and parts[0] == "_oracles":
            return "oracle"
        if parts and parts[0] == "_support":
            return "support"
        return "other"

    def check_file(self, rel: str, status: str, base: str | None) -> None:
        kind = self.classify(rel)
        path = self.root / rel
        parts = Path(rel).parts[1:]
        if kind in {"data", "manifest"}:
            return
        if kind == "other":
            native = len(parts) == 3 and parts[:2] == ("runtime", "native") and path.suffix in {".c", ".h"}
            if not (native or (parts and parts[-1] == "__init__.py") or path.suffix not in ALLOWED_DATA_SUFFIXES
                    and parts and parts[0] in self.m["support_dirs"]):
                self.add(rel, 1, "TO013", "unexpected file under tests/ (allowed: domain tests, conftest, "
                         "_support, _oracles, fixtures, runtime/native)")
            return
        if SHADOW_STEM.search(path.stem):
            self.add(rel, 1, "TO013", "shadow/legacy stem is banned")
        legacy_root = kind == "test" and len(parts) == 1
        if legacy_root:
            self.legacy_root(rel, status, base)
            return
        if kind in {"test", "support"}:
            for label, token in name_tokens(path.stem):
                self.add(rel, 1, "TO001", f"banned {label} token '{token}' in file name")
        if kind == "test":
            ok = len(parts) == 2 and parts[0] in self.m["domains"] and re.fullmatch(r"test_[a-z0-9_]+\.py", parts[1])
            if not ok:
                self.add(rel, 1, "TO002", f"test modules live at tests/<domain>/test_*.py with domain in "
                         f"{self.m['domains']}")
        if kind == "conftest" and len(parts) > 2:
            self.add(rel, 1, "TO002", "conftest.py only at tests/ or tests/<domain>/")
        self.check_source(rel, kind)

    def legacy_root(self, rel: str, status: str, base: str | None) -> None:
        if status in {"A", "R"} or base is None:
            self.add(rel, 1, "TO002", "root-level tests/test_*.py: move into tests/<domain>/ per TAXONOMY.toml")
            return
        added, removed = diff_lines(self.root, base, rel)
        new_tests = [line for line in added if NEW_TEST_DEF.match(line)]
        if len(added) > len(removed) or new_tests:
            self.add(rel, 1, "TO002", f"legacy root module grew (+{len(added)}/-{len(removed)} lines, "
                     f"{len(new_tests)} new tests): consolidate into its domain module instead")

    def check_source(self, rel: str, kind: str) -> None:
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n") + (not text.endswith("\n"))
        entry = self.m["modules"].get(str(Path(rel).relative_to("tests")).replace("\\", "/"))
        limit = int(entry.get("max_lines", self.m["max_lines"])) if entry else self.m["max_lines"]
        if lines >= HARD_MAX_LINES or kind in {"test", "support"} and lines > limit:
            cap = HARD_MAX_LINES - 1 if lines >= HARD_MAX_LINES else limit
            self.add(rel, lines, "TO004", f"{lines} lines exceeds the {cap}-line cap")
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            self.add(rel, exc.lineno or 1, "TO005", f"syntax error: {exc.msg}")
            return
        tier = None
        if kind == "test":
            if entry is None:
                self.add(rel, 1, "TO003", "test module missing from tests/TAXONOMY.toml")
            found, line = module_tier(tree)
            if len(found) != 1:
                self.add(rel, line, "TO005", f"module-level pytestmark must carry exactly one of {TIERS}")
            else:
                tier = found.pop()
                if entry and entry["tier"] != tier:
                    self.add(rel, line, "TO005", f"pytestmark {tier} != manifest tier {entry['tier']}")
        check = ModuleCheck(rel, tier, is_test=kind == "test",
                            allow_fields=bool(entry and entry.get("allow_field_enumeration")),
                            is_tempdirs=rel == "tests/_support/tempdirs.py", is_oracle=kind == "oracle",
                            is_conftest=kind == "conftest")
        check.visit(tree)
        self.findings.extend(check.findings)

    def _support_imports(self, path: Path, *, in_support: bool) -> set[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError:
            return set()
        refs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[:1] == ["_support"] and len(parts) >= 2:
                        refs.add(parts[1])
                    elif parts[:2] == ["tests", "_support"] and len(parts) >= 3:
                        refs.add(parts[2])
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                parts = mod.split(".") if mod else []
                if in_support and node.level == 1:
                    refs.update((parts[0],) if parts else (a.name for a in node.names))
                elif parts[:1] == ["_support"]:
                    refs.update((parts[1],) if len(parts) >= 2 else (a.name for a in node.names))
                elif parts[:2] == ["tests", "_support"]:
                    refs.update((parts[2],) if len(parts) >= 3 else (a.name for a in node.names))
        return refs

    def check_tree(self, strict: bool, base: str | None) -> None:
        on_disk = {str(p.relative_to(self.tests)).replace("\\", "/")
                   for p in self.tests.glob("*/test_*.py") if p.parent.name in self.m["domains"]}
        if len(on_disk) > self.m["max_modules"]:
            self.add("tests/TAXONOMY.toml", 1, "TO003", f"{len(on_disk)} modules exceed max_modules")
        listed = sum(1 for e in self.m["modules"].values())
        if listed > self.m["max_modules"]:
            self.add("tests/TAXONOMY.toml", 1, "TO003", f"{listed} manifest modules exceed max_modules")
        if strict:
            for rel, entry in self.m["modules"].items():
                if rel not in on_disk and not entry.get("reserved"):
                    self.add(f"tests/{rel}", 1, "TO003", "manifest module missing on disk (mark reserved or remove)")
        elif base is not None:
            base_files = set(git(self.root, "ls-tree", "-r", "--name-only", base, "--", "tests").splitlines())
            for rel, entry in self.m["modules"].items():
                if f"tests/{rel}" in base_files and rel not in on_disk and not entry.get("reserved"):
                    self.add(f"tests/{rel}", 1, "TO003", "established manifest module deleted on disk")
        support_dir = self.tests / "_support"
        support_edges = {p.stem: self._support_imports(p, in_support=True) - {p.stem}
                         for p in support_dir.glob("*.py") if p.name != "__init__.py"}
        used: set[str] = set()
        roots = [p for p in self.tests.rglob("*.py") if "__pycache__" not in p.parts and support_dir not in p.parents]
        roots.extend((self.root / "scripts").glob("*.py"))
        stack = list({ref for p in roots for ref in self._support_imports(p, in_support=False)})
        while stack:
            stem = stack.pop()
            if stem not in used:
                used.add(stem)
                stack.extend(support_edges.get(stem, ()))
        for stem in sorted(support_edges):
            if stem not in used:
                self.add(f"tests/_support/{stem}.py", 1, "TO013", "orphan _support module (no AST importer)")

    def run(self, base: str | None, strict: bool) -> None:
        if strict:
            targets = {str(p.relative_to(self.root)).replace("\\", "/"): "A" for p in self.tests.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts}
            base_ref = None
        else:
            targets = changed_paths(self.root, base)
            base_ref = base
        for rel, status in sorted(targets.items()):
            if (self.root / rel).is_file():
                self.check_file(rel, status, base_ref)
        self.check_tree(strict, base_ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="ratchet: check files changed versus this git ref")
    mode.add_argument("--strict", action="store_true", help="check the whole tree")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        gate = Gate(root, load_manifest(root / "tests"))
        gate.run(args.base, args.strict)
    except ManifestError as exc:
        print(f"check_test_orthogonality: {exc}", file=sys.stderr)
        return 2
    findings = sorted(set(gate.findings), key=lambda f: (f.path, f.line, f.code))
    failures = [f for f in findings if f.severity == "fail"]
    modules = sum(1 for p in (root / "tests").glob("*/test_*.py") if p.parent.name in gate.m["domains"])
    mode_label = "--strict" if args.strict else f"--base {args.base}"
    if args.format == "json":
        print(json.dumps({"findings": [asdict(f) for f in findings], "mode": mode_label,
                          "modules": modules, "max_modules": gate.m["max_modules"]}, indent=1))
    else:
        for f in findings:
            print(f"{f.path}:{f.line}: {f.code}{'' if f.severity == 'fail' else ' (warn)'} {f.message}")
        counts: dict[str, int] = {}
        for f in failures:
            counts[f.code] = counts.get(f.code, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        print(f"summary: {len(failures)} findings ({summary}); warnings={len(findings) - len(failures)}; "
              f"mode={mode_label}; modules={modules}/{gate.m['max_modules']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
