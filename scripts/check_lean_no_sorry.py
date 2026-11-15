#!/usr/bin/env python3
"""Static Lean 4 proof-hygiene gate for ``formal/``.

[Role]
  Scan every ``.lean`` file under ``formal/``, strip nested ``/- ... -/`` block
  comments, ``--`` line comments, and string literals, and reject any occurrence
  of ``sorry``, ``sorryAx``, ``admit``, or custom ``axiom`` declarations.

[Invariants]
  * Pure stdlib scanner; zero network, subprocess, or Lean binary dependency.
  * Unclosed block comments fail closed rather than masking trailing text.

[Timeouts & Locks]
  Bounded in-memory scan over ``formal/**/*.lean`` (< 50 ms); no locks.

[Side-Effect Policy]
  Read-only inspection of ``formal/``.
"""
from __future__ import annotations

import pathlib
import re
import sys

FORBIDDEN_RE = re.compile(r"\b(?:sorry|sorryAx|admit|axiom)\b")


def strip_lean_comments_and_strings(src: str) -> str:
    out: list[str] = []
    idx, n, depth = 0, len(src), 0
    while idx < n:
        if depth == 0 and src[idx] == '"':
            idx += 1
            while idx < n and src[idx] != '"':
                idx += 2 if src[idx] == "\\" else 1
            idx += 1
            out.append('""')
        elif src[idx : idx + 2] == "/-":
            depth += 1
            idx += 2
        elif depth > 0 and src[idx : idx + 2] == "-/":
            depth -= 1
            idx += 2
        elif depth == 0 and src[idx : idx + 2] == "--":
            while idx < n and src[idx] != "\n":
                idx += 1
        else:
            if depth == 0:
                out.append(src[idx])
            elif src[idx] == "\n":
                out.append("\n")
            idx += 1
    if depth != 0:
        raise ValueError("unclosed Lean block comment")
    return "".join(out)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1] / "formal"
    lean_files = sorted(root.rglob("*.lean"))
    if not lean_files:
        print("no Lean files found under formal/", file=sys.stderr)
        return 1
    errors: list[str] = []
    for path in lean_files:
        cleaned = strip_lean_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in FORBIDDEN_RE.finditer(cleaned):
            line = cleaned.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: forbidden token '{match.group(0)}'")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(lean_files)} Lean files: 0 sorry, 0 admit, 0 custom axiom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
