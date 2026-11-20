"""Verify source provenance for builtin K2/K3 template catalogs.

[Role] Audit Python source declarations, runtime inventory, and pinned blobs.
[Invariants] Exact SHA-256, repository-relative paths, and inclusive symbol ranges.
[Timeouts & Locks] Bounded local Git reads. No locks or network.
[Side-Effect Policy] Reads source and Git objects without modifying repositories.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

_SOURCE_FIELDS = ("repository", "revision", "relative_path", "source_sha256")
_TEMPLATE_FIELDS = (*_SOURCE_FIELDS, "symbol", "line_range")
_CONSTRUCTORS = {"SourceRef": _SOURCE_FIELDS, "TemplateSourceRef": _TEMPLATE_FIELDS}
_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_GIT_TIMEOUT_SECONDS = 20
_TEMPLATE_DIRECTORY = Path("mindie_sim/templates")


@dataclass(frozen=True)
class Violation:
    location: str
    reason: str


def _constant(node, names):
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"nonliteral_reference_field:{node.id}")
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_constant(item, names) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constant(node.left, names) + _constant(node.right, names)
    return ast.literal_eval(node)


def _constants(tree):
    names = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    names[target.id] = _constant(statement.value, names)
                except (TypeError, ValueError):
                    names.pop(target.id, None)
    return names


def _projection_calls(tree, relative):
    # The validated TemplateSourceRef fields are copied verbatim into SourceRef.
    # This projection has no independent provenance record.
    if relative != _TEMPLATE_DIRECTORY / "contracts.py":
        return set()
    projected = set()
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "TemplateSourceRef":
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "to_source_ref":
                continue
            for statement in method.body:
                call = statement.value if isinstance(statement, ast.Return) else None
                if (
                    isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "SourceRef" and not call.keywords
                    and len(call.args) == len(_SOURCE_FIELDS)
                    and all(isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                            and arg.value.id == "self" and arg.attr == key
                            for arg, key in zip(call.args, _SOURCE_FIELDS))
                ):
                    projected.add(call)
    return projected


def _ast_references(root):
    violations, refs = [], []
    for file in sorted((root / _TEMPLATE_DIRECTORY).rglob("*.py")):
        rel = file.relative_to(root)
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(rel))
        except TimeoutError:
            raise
        except (SyntaxError, OSError, UnicodeError) as exc:
            violations.append(Violation(str(rel), f"source_scan_failed:{exc}"))
            continue
        names = _constants(tree)
        aliases = dict(_CONSTRUCTORS)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases.update({name.asname or name.name: _CONSTRUCTORS[name.name]
                                for name in node.names if name.name in _CONSTRUCTORS})
        projected = _projection_calls(tree, rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or node in projected:
                continue
            if isinstance(node.func, ast.Name):
                fields = aliases.get(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                fields = _CONSTRUCTORS.get(node.func.attr)
            else:
                fields = None
            if fields is None:
                continue
            location = f"{rel.as_posix()}:{node.lineno}"
            try:
                if len(node.args) > len(fields):
                    raise ValueError("too_many_reference_fields")
                values = dict(zip(fields, (_constant(arg, names) for arg in node.args)))
                for keyword in node.keywords:
                    if keyword.arg not in fields or keyword.arg in values:
                        raise ValueError("unknown_or_duplicate_reference_field")
                    values[keyword.arg] = _constant(keyword.value, names)
                if set(values) != set(fields):
                    raise ValueError("missing_reference_fields")
            except (ValueError, TypeError) as exc:
                violations.append(Violation(location, f"invalid_static_source_ref:{exc}"))
                continue
            refs.append((location, values))
    return refs, violations


def _runtime_references(root):
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        registry = importlib.import_module("mindie_sim.templates.registry")
        if Path(registry.__file__).resolve().parent != root / _TEMPLATE_DIRECTORY:
            raise ValueError("runtime_catalog_outside_requested_root")
        return [("runtime_catalog", ref.to_dict()) for ref in registry.all_template_source_refs()]
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode_policy


def _valid_fields(ref):
    for key in _SOURCE_FIELDS:
        if not isinstance(ref.get(key), str) or not ref[key].strip():
            return f"missing_{key}"
    if not _REVISION.fullmatch(ref["revision"]):
        return "invalid_revision"
    path = ref["relative_path"]
    if set(path.split("/")) & {".", "..", ""} or any(char in path for char in ("\\", ":", "\x00")):
        return "invalid_relative_path"
    if not _SHA256.fullmatch(ref["source_sha256"]):
        return "invalid_source_sha256"
    if set(ref).intersection({"symbol", "line_range"}):
        if not isinstance(ref.get("symbol"), str) or not ref["symbol"].strip():
            return "invalid_symbol"
        lines = ref.get("line_range")
        if (
            not isinstance(lines, (tuple, list)) or len(lines) != 2
            or any(type(value) is not int for value in lines) or not 1 <= lines[0] <= lines[1]
        ):
            return "invalid_line_range"
    return None


def _load_blob(repo_dir, revision, path):
    if repo_dir is None or not repo_dir.is_dir():
        return "source_repository_missing", None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{revision}:{path}"],
            capture_output=True, timeout=_GIT_TIMEOUT_SECONDS, check=False,
        )
    except TimeoutError:
        raise
    except (OSError, subprocess.TimeoutExpired):
        return "source_git_read_failed", None
    if result.returncode:
        return "source_blob_missing", None
    blob = result.stdout
    lines = blob.split(b"\n") if blob else []
    if blob.endswith(b"\n"):
        lines.pop()
    return hashlib.sha256(blob).hexdigest(), lines


def _identity(ref):
    return tuple((key, tuple(value) if key == "line_range" else value)
                 for key, value in sorted(ref.items()))


def verify(root, *, runtime=True, refs_root=None, repo_map=None):
    refs, violations = _ast_references(root)
    if runtime:
        try:
            dynamic = _runtime_references(root)
        except Exception as exc:
            violations.append(Violation("runtime_catalog", f"runtime_catalog_failed:{exc}"))
        else:
            expected = {_identity(ref) for _, ref in refs if _valid_fields(ref) is None and "symbol" in ref}
            observed = {_identity(ref) for _, ref in dynamic if _valid_fields(ref) is None}
            if expected != observed:
                violations.append(Violation("runtime_catalog", "source_ref_inventory_differs_from_ast"))
            refs.extend(dynamic)
    if not refs:
        violations.append(Violation(str(_TEMPLATE_DIRECTORY), "no_source_refs_found"))
    verified, blobs = set(), {}
    repo_map = repo_map or {}
    for location, ref in refs:
        invalid = _valid_fields(ref)
        if invalid is not None:
            violations.append(Violation(location, invalid))
            continue
        identity = _identity(ref)
        if identity in verified:
            continue
        verified.add(identity)
        if refs_root is None and not repo_map:
            continue
        repository, revision, path, sha = (ref[key] for key in _SOURCE_FIELDS)
        basename = repository.split("/")[-1]
        directory = repo_map.get(repository, repo_map.get(basename))
        if directory is None and refs_root is not None:
            if basename in {"", ".", ".."} or "\\" in basename:
                violations.append(Violation(location, f"invalid_repository_basename:{repository}"))
                continue
            directory = refs_root / basename
        key = directory, revision, path
        if key not in blobs:
            blobs[key] = _load_blob(*key)
        digest, lines = blobs[key]
        if lines is None:
            violations.append(Violation(location, f"{digest}:{repository}:{path}"))
            continue
        if digest != sha.lower():
            violations.append(Violation(location, f"source_sha256_mismatch:{repository}:{path}"))
        if "symbol" in ref:
            start, end = ref["line_range"]
            symbol = ref["symbol"]
            if end > len(lines):
                violations.append(Violation(location, f"source_line_range_exceeds_blob:{path}:{end}"))
            elif not any(symbol.encode("utf-8") in line for line in lines[start - 1:end]):
                violations.append(Violation(location, f"symbol_missing_from_range:{path}:{symbol}"))
    return len(verified), violations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--refs-root", type=Path)
    parser.add_argument("--repo-map", metavar="NAME=PATH", action="append", default=[])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--runtime", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not (root / _TEMPLATE_DIRECTORY).is_dir():
        parser.error("root must contain mindie_sim/templates")
    repo_map = {}
    for entry in args.repo_map:
        name, _, path = entry.partition("=")
        if not name.strip() or not path.strip() or name in repo_map:
            parser.error("each --repo-map must be a unique NAME=PATH")
        repo_map[name] = Path(path)
    checked, violations = verify(root, runtime=args.runtime, refs_root=args.refs_root, repo_map=repo_map)
    if args.format == "json":
        print(json.dumps({"checked": checked, "violations": [asdict(item) for item in violations]}, indent=2))
    else:
        for item in violations:
            print(f"{item.location}: {item.reason}")
        print(f"template provenance violations: {len(violations)} ({checked} references checked)")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
