"""Round-15 operational prologues and word-boundary comment discovery.

[Role] Exercise decoder/wrapper units and all three public selection paths for
  set prologues and comments in JSON, YAML flow, block and mixed documents.
[Invariants] Admission is unchanged; only statement-level supported set options
  are skipped. Comments start at lexical word boundaries and retain newlines.
  Literal hashes, explicit forwarding, overlay identity and depth stay intact.
[Timeouts & Locks] Bounded in-memory fixtures; no locks or external execution.
[Side-Effect Policy] No filesystem, network or device writes; explicit env only.
"""
from __future__ import annotations

import json
import shlex
from collections.abc import Mapping

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_docker import _STATEMENT_END, discovery_wrapper
from mindie_sim.launch_env import _EnvOverlay, normalize_env
from mindie_sim.launch_parser import LaunchSyntaxError, parse_shell
from mindie_sim.launch_taint import (
    _decoded_shell_words, _shell_taint, _tolerant_words, prescan_document_secrets,
)
from mindie_sim.launch_yaml_raw import raw_document_taint
from mindie_sim.launch_yaml_taint import _event_taint

try:
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SECRET = "R15_REGRESSION_a630ef"
OLD = "R15_OLD_VALUE"
SCRIPT = 'vllm serve /m --api"-key" "$VALUE"'
BASE = dict(machine="a3", model="qwen3-30b-a3b",
            framework_version="vllm-ascend@0.30.0",
            workload=dict(num_requests=2, prompt_tokens=128, output_tokens=8,
                          prefix_tokens=0))
SETS = ("set -e", "set -u", "set -eu", "set -o pipefail", "set -e -u",
        "set -euo pipefail", "set -e -o pipefail", 'set "-o pipefail"',
        'set "-eu pipefail"', "set -o", "set pipefail", "set -euo")
PREFIXES = ("export VALUE=" + SECRET + " A", "export docker")
WRAPPERS = ("docker run -e VALUE image sh -c ",
            "docker run --rm --env=VALUE image /bin/bash -c ",
            "sh -c ", "bash -c ")
LAYOUTS = ("json", "flow", "block", "mixed")
TAILS = ("", " | true", "; true")


def _env(prefix):
    return {"VALUE": OLD if "VALUE=" in prefix else SECRET,
            "A": "x", "docker": "x", "SEED": SECRET}


def _set_lead(prologue, position, prefix):
    statements = (prologue, prefix) if position == "before" else (prefix, prologue)
    return "\n".join(statements) + "\n"


def test_r15_operational_set_reaches_wrapper():
    for prologue in SETS:
        for position in ("before", "after"):
            for prefix in PREFIXES:
                env = normalize_env(_env(prefix))
                for wrapper in WRAPPERS:
                    command = _set_lead(prologue, position, prefix) + wrapper + shlex.quote(SCRIPT)
                    for suffix in TAILS:
                        words = _decoded_shell_words(command + suffix, env)
                        script, effective = discovery_wrapper(words, env)
                        assert script == SCRIPT and effective["VALUE"] == (SECRET, False)
                        found = set()
                        _shell_taint(command + suffix, found, env)
                        assert SECRET in found and OLD not in found


def _document(command, env, layout, secret):
    first = {"name": "a", "env": env, "command": command}
    second = {"name": "b", "command": ["vllm", "serve", "/m",
                                       "--served-model-name", secret]}
    if layout in {"json", "flow"}:
        return json.dumps({"containers": [first, second]})
    lines = ["containers:", "- name: a"]
    if layout == "mixed":
        lines += ["  env: " + json.dumps(env), "  command: " + json.dumps(command)]
    else:
        lines += ["  env:"] + [f"    {key}: {json.dumps(value)}"
                               for key, value in env.items()]
        lines += ["  command: |-"] + ["    " + line for line in command.split("\n")]
    lines += ["- name: b", "  command: " + json.dumps(second["command"])]
    return "\n".join(lines) + "\n"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _assert_public(command, env, layout, *, rejected=False, secret=SECRET):
    text = _document(command, env, layout, secret)
    fmt = "json" if layout == "json" else "yaml"
    assert secret in prescan_document_secrets(text, fmt)
    if fmt == "yaml":
        raw = set()
        raw_document_taint(text, raw, _shell_taint)
        assert secret in raw
        if HAS_YAML:
            events = set()
            _event_taint(text, events, _shell_taint)
            assert secret in events
    for selection in ("a", "b", secret):
        report = parse_launch(text, format=fmt, container=selection,
                              source=secret, **BASE)
        assert secret not in launch_markdown(report)
        js = canonical_json(report)
        if not (secret.isascii() and "\\" not in secret and '"' not in secret and secret not in js):
            assert not any(secret in s for s in _strings(json.loads(js)))
        assert report.source.name == "[REDACTED]"
        codes = [item.code for item in report.diagnostics]
        if fmt == "yaml" and not HAS_YAML:
            assert report.status == "failed" and codes == ["yaml_parser_unavailable"]
        elif selection == secret:
            assert report.status == "failed" and codes == ["container_not_found"]
        elif selection == "a" and rejected:
            assert report.status == "failed" and codes == ["unsupported_shell_syntax"]
        else:
            assert report.status == "completed" and not codes
            name = "--served-model-name" if selection == "b" else "--api-key"
            settings = (*report.applied, *report.neutral_operational,
                        *report.unmodeled_or_unsupported)
            assert next(item for item in settings if item.name == name).redacted


@pytest.mark.parametrize("layout", LAYOUTS)
def test_r15_set_public_reports(layout):
    for prologue in SETS:
        for position in ("before", "after"):
            for prefix in PREFIXES:
                env = _env(prefix)
                for wrapper in WRAPPERS[:2]:
                    command = _set_lead(prologue, position, prefix) + wrapper + shlex.quote(SCRIPT)
                    _assert_public(command, env, layout)


def test_r15_repeated_set_assignment_export_precedence():
    command = ("set -e\nVALUE=" + OLD + "\nexport VALUE A\nset -u\n"
               "VALUE=\"$SEED\"\nexport docker\nset -o pipefail\n"
               "env A=1 " + WRAPPERS[0] + shlex.quote(SCRIPT))
    for layout in LAYOUTS:
        _assert_public(command, _env("VALUE="), layout)


@pytest.mark.parametrize("prefix", [
    "set -x\n", "set\n", "A=1 set -e\n", "env set -e\n",
    "set -e ", "set -e -x\n", "true\n", "set -e\ntrue\n",
])
def test_r15_unsupported_prologues_never_authorize_wrapper(prefix):
    command = prefix + WRAPPERS[0] + shlex.quote(SCRIPT)
    env = normalize_env({"VALUE": SECRET})
    for decode in (_decoded_shell_words, _tolerant_words):
        assert discovery_wrapper(decode(command, env), env) is None
    with pytest.raises(LaunchSyntaxError) as raised:
        parse_shell(command, env={"VALUE": SECRET})
    assert raised.value.code == "unsupported_shell_syntax"


def test_r15_set_argv_and_end_of_input_have_no_following_statement():
    tail = ["docker", "run", "-e", "VALUE", "image", "sh", "-c", SCRIPT]
    env = normalize_env({"VALUE": SECRET})
    for prefix in (["set", "-e"], ["A=1", "set", "-e"], ["env", "set", "-e"]):
        assert discovery_wrapper(prefix + tail, env) is None
        for literal in (";", "\n"):
            assert discovery_wrapper(prefix + [literal] + tail, env) is None
    for prologue in SETS:
        for ending in ("", "\n"):
            assert discovery_wrapper(_decoded_shell_words(prologue + ending), env) is None


@pytest.mark.parametrize("suffix", TAILS)
def test_r15_env_options_and_wrapper_arguments_stay_statement_local(suffix):
    env = normalize_env({"VALUE": SECRET})
    tail = WRAPPERS[0] + shlex.quote(SCRIPT) + suffix
    for prefix in ("env A=1\n", "env\n", "set -e\nenv A=1\n"):
        assert discovery_wrapper(_decoded_shell_words(prefix + tail, env), env) == (
            SCRIPT, {"VALUE": (SECRET, False)})
    for prefix in ("env -i\n", "env -u A\n", "set -e\ndocker run -e\n",
                   "set -e\nsh -c\n", "docker run --rm\n"):
        assert discovery_wrapper(_decoded_shell_words(prefix + tail, env), env) is None
    wrapped = discovery_wrapper(_decoded_shell_words("set -e\nenv A=1 " + tail, env), env)
    assert wrapped == (SCRIPT, {"VALUE": (SECRET, False)})


def test_r15_comment_public_reports():
    for location in ("leading", "between", "inline"):
        for prefix in PREFIXES:
            env = _env(prefix)
            lead = {"leading": "# ordinary comment\n" + prefix + "\n",
                    "between": prefix + "\n# ordinary comment\n",
                    "inline": prefix + " # ordinary comment\n"}[location]
            for suffix in TAILS:
                for wrapper in WRAPPERS[:2]:
                    command = lead + wrapper + shlex.quote(SCRIPT) + suffix
                    for layout in LAYOUTS:
                        _assert_public(command, env, layout, rejected=bool(suffix))


HASH_VALUES = (("'# " + SECRET + "'", "# " + SECRET),
               ('"# ' + SECRET + '"', "# " + SECRET),
               ("\\#" + SECRET, "#" + SECRET),
               (SECRET + "a#b", SECRET + "a#b"),
               ('""#' + SECRET, "#" + SECRET),
               ("''#" + SECRET, "#" + SECRET))


def test_r15_literal_hash_public_reports():
    for spelling, secret in HASH_VALUES:
        for suffix in TAILS:
            command = ("export VALUE=" + spelling + " A\n" + WRAPPERS[0]
                       + shlex.quote(SCRIPT) + suffix)
            for layout in LAYOUTS:
                _assert_public(command, {"A": "x"}, layout, secret=secret, rejected=bool(suffix))


def test_r15_comment_keeps_terminating_statement_boundary():
    env = normalize_env({"VALUE": SECRET})
    for separator in ("\n", "\r", "\r\n"):
        for boundary in ("", " ", "\t", *"|&;<>()`!{}", "\\\n"):
            words = _tolerant_words(boundary + "# ignored ; ' \" $VALUE" + separator
                                    + "sh -c 'body'", env)
            assert words[-3:] == ("sh", "-c", "body")
            assert all(word is _STATEMENT_END for word in words[:-3])
            assert len(words[:-3]) == 1 + (boundary in ";&|(){}" and bool(boundary))
            assert discovery_wrapper(words)[0] == "body"


def test_r15_hash_after_started_word_is_literal():
    env = normalize_env({"EMPTY": ""})
    for spelling, expected in (
        ("'# comment'", "# comment"), ('"# comment"', "# comment"),
        ('""#tag', "#tag"), ("''#tag", "#tag"), ("\\#tag", "#tag"),
        ("foo#bar", "foo#bar"), ("VALUE=a#b", "VALUE=a#b"),
        ("$EMPTY#tag", "#tag"), ("${EMPTY}#tag", "#tag"),
        ('"$EMPTY"#tag', "#tag"), ("\\ #tag", " #tag"),
        ("word\\\n#tag", "word#tag"), ("$MISSING#tag", "$MISSING#tag"),
    ):
        assert _tolerant_words(spelling + " | true", env) == (expected, _STATEMENT_END, "true")


def test_r15_comment_eof_empty_quotes_and_reexport():
    from mindie_sim.launch_env import _tolerant_words as moved_decoder
    assert moved_decoder is _tolerant_words
    for text in ("# eof", " # eof", "\\\n# eof"):
        assert _tolerant_words(text, {}) == ()
    assert _tolerant_words("export A # eof", {}) == ("export", "A")
    assert _tolerant_words('""#tag # eof', {}) == ("#tag",)
    assert _tolerant_words("''#tag # eof", {}) == ("#tag",)


class _NoEnumeration(Mapping):
    def __getitem__(self, name):
        return {"VALUE": (OLD, False), "A": ("x", False)}[name]

    def __iter__(self):
        raise AssertionError("base environment was enumerated")

    def __len__(self):
        return 2


def test_r15_set_comment_overlay_identity_and_explicit_forwarding():
    base = _EnvOverlay(_NoEnumeration())
    lead = "set -e\nexport VALUE=" + SECRET + " A # note\nset -u\n"
    for decode in (_decoded_shell_words, _tolerant_words):
        script, effective = discovery_wrapper(decode(lead + "sh -c '$VALUE'", base), base)
        assert script == SECRET and isinstance(effective, _EnvOverlay)
        assert effective._base is base and base["VALUE"] == (OLD, False)
        for option in ("-e VALUE", "--env VALUE", "--env=VALUE"):
            words = decode(lead + "docker run " + option + " image sh -c '$VALUE'", base)
            assert discovery_wrapper(words, base) == (SECRET, {"VALUE": (SECRET, False)})
        words = decode(lead + "docker run image sh -c '$VALUE'", base)
        assert discovery_wrapper(words, base) == ("$VALUE", {})


@pytest.mark.parametrize("depth", range(1, 7))
def test_r15_set_and_comments_keep_wrapper_depth_three(depth):
    command = SCRIPT
    for _ in range(depth):
        command = "set -e\n# note\nsh -c " + shlex.quote(command)
    found = set()
    _shell_taint(command, found, normalize_env({"VALUE": SECRET}))
    assert (SECRET in found) == (depth <= 3)


def test_r15_prologues_and_comments_never_read_host_environment(monkeypatch):
    import os

    class NoHost(Mapping):
        def __getitem__(self, key):
            raise AssertionError("host env read")

        def __iter__(self):
            raise AssertionError("host env enumeration")

        def __len__(self):
            raise AssertionError("host env length")

    lead = "set -e\nexport VALUE=" + SECRET + " A # note\nset -u\n"
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoHost())
        for layout in LAYOUTS:
            for suffix in TAILS:
                _assert_public(lead + WRAPPERS[0] + shlex.quote(SCRIPT) + suffix,
                               {"A": "x"}, layout, rejected=bool(suffix))


R24_VALUE = "-R24_payload*7bd129"
R24_DECOY = "public_decoy_R24"
R24_ENV = {"VALUE": R24_VALUE, "VA": R24_DECOY, "EMPTY": ""}
R24_REFS = ("$VA{gap}LUE", "${{VA{gap}LUE}}", "${gap}{{VALUE}}",
            "${gap}VALUE", "${{VALUE{gap}}}", "${{{gap}VALUE}}")
R24_LAYOUTS = ("json", "mapping", "flow", "block")
R24_BINDINGS = ("separate", "attached", "SECRET_KEY", "SECRET_TOKEN")
R24_SHAPES = ("argv", "command_args", "entrypoint_cmd",
              "string_command_args", "string_entrypoint_cmd")


def _r24_fields(operand, binding, shape):
    head = ["vllm", "serve", "/m"]
    if binding == "separate":
        head += ["--api-key"]
        tail = [operand]
    else:
        tail = [("--api-key" if binding == "attached" else binding) + "=" + operand]
    if shape == "argv":
        return {"command": head + tail}
    left, right = ("entrypoint", "cmd") if "entrypoint" in shape else ("command", "args")
    return {left: " ".join(head) if shape.startswith("string_") else head, right: tail}


def _r24_document(fields, expected, layout):
    first = {"name": "a", **fields, "env": R24_ENV}
    second = {"name": "b", "command": ["vllm", "serve", "/m",
                                       "--served-model-name=" + expected]}
    if layout in ("json", "flow"):
        return json.dumps({"containers": [first, second]})
    lines = ["containers:", "  - name: a"]
    for key, words in fields.items():
        if isinstance(words, list):
            lines.append("    " + key + ":")
            for atom in words:
                if layout == "block":
                    lines += ["      - |-"] + ["        " + s for s in atom.split("\n")]
                else:
                    lines.append("      - " + json.dumps(atom))
        elif layout == "block":
            lines += ["    " + key + ": |-"] + ["      " + s for s in words.split("\n")]
        else:
            lines.append("    " + key + ": " + json.dumps(words))
    lines += ["    env: " + json.dumps(R24_ENV), "  - name: b",
              "    command: " + json.dumps(second["command"])]
    return "\n".join(lines) + "\n"


def _r24_inventory(text, layout):
    fmt = "json" if layout == "json" else "yaml"
    inventories = [set(prescan_document_secrets(text, fmt))]
    if fmt == "yaml":
        raw = set()
        raw_document_taint(text, raw, _shell_taint)
        inventories.append(raw)
        if HAS_YAML:
            events = set()
            _event_taint(text, events, _shell_taint)
            inventories.append(events)
    return inventories


def _r24_assert_public(fields, expected, layout):
    text = _r24_document(fields, expected, layout)
    for found in _r24_inventory(text, layout):
        assert expected in found and R24_VALUE in found
        assert not any(R24_DECOY in item for item in found)
    for source in ("ordinary.json", expected):
        for selection in ("a", "b", expected):
            report = parse_launch(text, format="json" if layout == "json" else "yaml",
                                  container=selection, source=source, **BASE)
            assert R24_VALUE not in launch_markdown(report)
            js = canonical_json(report)
            if R24_VALUE in js:
                assert not any(R24_VALUE in s for s in _strings(json.loads(js)))
            if source == expected:
                assert report.source.name == "[REDACTED]"
            if selection == "b" and (layout == "json" or HAS_YAML):
                assert report.status == "completed" and not report.diagnostics


@pytest.mark.parametrize("layout", R24_LAYOUTS)
def test_r24_continued_reference_public(layout):
    from mindie_sim.launch_parser import parse_argv
    for ref in R24_REFS[:3]:
        for quote in ("", '"'):
            operand = "lead " + quote + ref.format(gap="\\\n") + quote + " tail"
            for binding in R24_BINDINGS:
                literal = _r24_fields(operand, binding, "argv")["command"]
                assert parse_argv(literal).commands[0].argv == tuple(literal)
                for shape in R24_SHAPES:
                    fields = _r24_fields(operand, binding, shape)
                    _r24_assert_public(fields, "lead " + R24_VALUE + " tail", layout)


def test_r24_reference_identity_and_suppression():
    from mindie_sim.launch_taint import _static_argv_operand
    from mindie_sim.launch_parser import parse_argv
    for newline in ("\n", "\r", "\r\n"):
        for quote in ("", '"'):
            for ref in R24_REFS:
                reference = ref.format(gap="\\" + newline)
                operand = "lead " + quote + reference + quote + " tail"
                found = set()
                assert _static_argv_operand(operand, R24_ENV, found) == "lead " + R24_VALUE + " tail"
                assert found == {R24_VALUE}
                literal = ["vllm", "serve", "/m", "--api-key", operand]
                assert parse_argv(literal).commands[0].argv == tuple(literal)
                for suppressed in ("'" + reference + "'", quote + "\\" + reference + quote):
                    found = set()
                    assert _static_argv_operand(suppressed, R24_ENV, found) is not None
                    assert not found
                    for suffix in ("", " | true"):
                        found = set()
                        _shell_taint("vllm serve /m --api-key " + suppressed + suffix, found, R24_ENV)
                        assert R24_VALUE not in found and R24_DECOY not in found
                command = 'vllm serve /m --api-key "' + reference + '"'
                assert parse_shell(command, env=R24_ENV).commands[0].argv[-1] == R24_VALUE
                with pytest.raises(LaunchSyntaxError, match="word_splitting_or_globbing_required"):
                    parse_shell("vllm serve /m --api-key " + reference, env=R24_ENV)


def test_r24_shell_reference_public():
    for ref in R24_REFS[:3]:
        for quote in ("", '"'):
            for layout in R24_LAYOUTS:
                for newline in ("\n", "\r", "\r\n"):
                    if layout == "block" and newline != "\n":
                        continue
                    command = "vllm serve /m --api-key " + quote + ref.format(gap="\\" + newline) + quote
                    for suffix in ("", " | true"):
                        _r24_assert_public({"command": command + suffix}, R24_VALUE, layout)


def test_r24_shell_reference_exclusions():
    for layout in R24_LAYOUTS:
        for tail in (
            ">--api-key {ref}", "--api-key >{ref} ordinary", "--api-key; {ref}",
            "--api-key\n{ref}", "--api-key | {ref}", "--api-key & {ref}",
            "--api-key --served-model-name", "--api-key -tp", "--api-key -x", "--api-key --",
        ):
            for ref in R24_REFS[:3]:
                command = "vllm serve /m " + tail.format(ref=ref.format(gap="\\\n"))
                found = set()
                _shell_taint(command, found, R24_ENV)
                inventories = [found, *_r24_inventory(_r24_document(
                    {"command": command}, R24_VALUE, layout), layout)]
                assert all(R24_VALUE not in found and R24_DECOY not in found for found in inventories)


def test_r24_invalid_reference_contributors_are_atomic():
    from mindie_sim.launch_taint import _static_argv_operand
    for ref in ("${}", "${1}", "${A:-x}", "${VALUE", "$UNKNOWN",
                "$(cmd)", "$1", "$?", "$!", "$#", "$$", "$@", "$*", "$-"):
        for newline in ("\n", "\r", "\r\n"):
            continued = "$\\" + newline + ref[1:]
            for operand in ("$VALUE" + continued, continued + "$VALUE"):
                found = {"existing"}
                assert _static_argv_operand(operand, R24_ENV, found) is None
                assert found == {"existing"}
            with pytest.raises(LaunchSyntaxError) as raised:
                parse_shell("vllm serve /m --api-key " + continued, env=R24_ENV)
            if ref.startswith("${") and ref != "${VALUE":
                assert raised.value.message == "parameter_expansion_not_supported"


def test_r24_lookup_only_nonrecursive_and_no_host(monkeypatch):
    import os
    from mindie_sim import launch_env as le
    from mindie_sim.launch_taint import _static_argv_operand
    value = "literal $OTHER ; 'quoted' \\ \n"
    lookups = []

    class LookupOnly(le.EnvMap):
        def __getitem__(self, key):
            lookups.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            return self[key] if key in self else default

        def __iter__(self):
            raise AssertionError("base enumeration")

        def keys(self):
            raise AssertionError("base keys")

        def items(self):
            raise AssertionError("base items")

    def no_host(*args, **kwargs):
        raise AssertionError("host lookup")

    class NoHost(Mapping):
        __getitem__ = __iter__ = __len__ = no_host

    base = LookupOnly(VALUE=(value, False), VA=(R24_DECOY, False))
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoHost())
        patch.setattr(os, "getenv", no_host)
        for env in (base, le._EnvOverlay(base)):
            for ref in R24_REFS:
                operand = ref.format(gap="\\\n")
                found = set()
                assert _static_argv_operand(operand, env, found) == value
                assert found == {value}
                for decode in (_decoded_shell_words, _tolerant_words):
                    assert decode('"' + operand + '"', env) == (value,)
    assert lookups and set(lookups) == {"VALUE"}


def test_r24_reference_continuations_commit_every_cursor(monkeypatch):
    from mindie_sim import launch_env as le
    from mindie_sim.launch_taint import _static_argv_operand
    from mindie_sim.launch_parser import scan
    original = le.continuation_length
    norm_env = normalize_env(R24_ENV)
    for n in (3, 257, 4096, 16384):
        gap = "\\\n\\\r\\\r\n" * n
        for quote in ("", '"'):
            cases = [(ref.format(gap=gap), R24_VALUE) for ref in R24_REFS]
            cases += [("$VA" + gap + "LUE" + gap, R24_VALUE),
                      ("$" + gap, "$"), ("$" + gap + ":", "$:"),
                      ("$" + gap + "\\x", "$x" if not quote else "$\\x")]
            for body, expected in cases:
                text = quote + body + quote
                for decode in (lambda s: _static_argv_operand(s, R24_ENV),
                               lambda s: _tolerant_words(s, norm_env)[0], scan):
                    calls = []

                    def counted(value, index):
                        calls.append(index)
                        return original(value, index)

                    with monkeypatch.context() as patch:
                        patch.setattr(le, "continuation_length", counted)
                        actual = decode(text)
                    if decode is not scan:
                        assert actual == expected
                    assert len(calls) <= len(text) + 4
                    assert len(calls) == len(set(calls))
