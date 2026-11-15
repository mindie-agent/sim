"""Round-14 export and Round-22/23 static credential discovery regressions.

[Role] Close R14-P1-01, R22-P1-01 and R23-P1-01 through public reports
  in JSON, YAML flow, block and mixed layouts in every import mode.
[Invariants] Export operands include assignments and every valid identifier;
  only identity-only statement boundaries end an export. Literal delimiters
  stay arguments, forwarding is explicit, and local overlays never copy bases.
[Timeouts & Locks] Bounded in-memory fixtures; no locks or external execution.
[Side-Effect Policy] No filesystem, network or device writes; in-memory fixtures.
"""
from __future__ import annotations

import json
import shlex
from collections.abc import Mapping

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_docker import _STATEMENT_END, discovery_wrapper
from mindie_sim.launch_env import ShellText, _EnvOverlay, normalize_env
from mindie_sim.launch_taint import (
    _apply_words, _decoded_shell_words, _shell_taint, _tolerant_words,
    prescan_document_secrets,
)
from mindie_sim.launch_yaml_raw import raw_document_taint
from mindie_sim.launch_yaml_taint import _event_taint

try:
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SECRET = "R14_REGRESSION_52f9a8"
OLD = "R14_OLD_VALUE"
SCRIPT = 'vllm serve /m --api"-key" "$VALUE"'
BASE = dict(machine="a3", model="qwen3-30b-a3b",
            framework_version="vllm-ascend@0.30.0",
            workload=dict(num_requests=2, prompt_tokens=128, output_tokens=8,
                          prefix_tokens=0))
PREFIXES = ("export VALUE=" + SECRET + " A", "export A=x VALUE",
            "export A=x B VALUE", "export docker", "export sh", "export bash",
            "export env", "export export")
WRAPPERS = ("docker run -e VALUE image sh -c ",
            "docker run --rm --env=VALUE image /bin/bash -c ",
            "sh -c ", "bash -c ")


def _env(prefix):
    return {"VALUE": OLD if "VALUE=" in prefix else SECRET, "A": "x", "B": "y",
            **{name: "x" for name in ("docker", "sh", "bash", "env", "export")}}


def test_r14_complete_export_statement_reaches_wrapper():
    for prefix in PREFIXES:
        env = normalize_env(_env(prefix))
        for wrapper in WRAPPERS:
            for separator in ("\n", ";"):
                text = prefix + separator + wrapper + shlex.quote(SCRIPT)
                for surface in (text, text + " | true"):
                    words = _decoded_shell_words(surface, env)
                    assert any(word is _STATEMENT_END for word in words)
                    assert not isinstance(_STATEMENT_END, str)
                    script, effective = discovery_wrapper(words, env)
                    assert script == SCRIPT
                    assert effective["VALUE"][0] == SECRET
                    found = set()
                    _shell_taint(surface, found, env)
                    assert SECRET in found
                    assert OLD not in found


def _document(prefix, separator, layout):
    command = prefix + separator + WRAPPERS[0] + shlex.quote(SCRIPT)
    env = _env(prefix)
    first = {"name": "a", "env": env, "command": command}
    second = {"name": "b", "command": ["vllm", "serve", "/m",
                                       "--served-model-name", SECRET]}
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


def _assert_safe(report):
    assert not any(SECRET in text
                   for text in _strings(json.loads(canonical_json(report))))
    assert SECRET not in launch_markdown(report)
    assert SECRET not in report.source.name


def test_r14_export_inventory_protects_public_sibling_and_missing():
    for prefix in PREFIXES:
        for layout in ("json", "flow", "block", "mixed"):
            for separator in ("\n", ";"):
                text = _document(prefix, separator, layout)
                fmt = "json" if layout == "json" else "yaml"
                assert SECRET in prescan_document_secrets(text, fmt)
                if fmt == "yaml":
                    found = set()
                    raw_document_taint(text, found, _shell_taint)
                    assert SECRET in found
                    if HAS_YAML:
                        events = set()
                        _event_taint(text, events, _shell_taint)
                        assert SECRET in events
                for selection in ("a", "b", SECRET):
                    report = parse_launch(text, format=fmt, container=selection,
                                          source=SECRET, **BASE)
                    _assert_safe(report)
                    codes = {item.code for item in report.diagnostics}
                    if fmt == "yaml" and not HAS_YAML:
                        assert report.status == "failed"
                        assert "yaml_parser_unavailable" in codes
                    elif selection == SECRET:
                        assert report.status == "failed"
                        assert "container_not_found" in codes
                    elif selection == "a" and separator == ";":
                        assert report.status == "failed"
                        assert "unsupported_shell_syntax" in codes
                    else:
                        assert report.status == "completed"
                        name = "--served-model-name" if selection == "b" else "--api-key"
                        settings = (*report.applied, *report.neutral_operational,
                                    *report.unmodeled_or_unsupported)
                        assert next(item for item in settings if item.name == name).redacted


def test_r14_leading_statements_accumulate_local_overrides():
    for separator in ("\n", ";"):
        for prefix in (
            "VALUE=" + SECRET + "{s}export VALUE A",
            "export VALUE=" + OLD + " A{s}VALUE=" + SECRET,
            "export docker sh bash env export{s}export VALUE=" + SECRET + " A",
        ):
            command = prefix.format(s=separator) + separator + "env B=y "
            command += WRAPPERS[0] + shlex.quote(SCRIPT)
            env = normalize_env(_env("VALUE="))
            wrapped = discovery_wrapper(_decoded_shell_words(command, env), env)
            assert wrapped[0] == SCRIPT
            assert wrapped[1] == {"VALUE": (SECRET, False)}
            assert env["VALUE"][0] == OLD


def test_r14_wrapper_arguments_never_cross_statement_boundary():
    env = normalize_env({"VALUE": SECRET})
    for separator in ("\n", ";"):
        for left, right in (
            ("docker", "run -e VALUE image sh -c "),
            ("docker run -e", "VALUE image sh -c "),
            ("docker run --name", "name image sh -c "),
            ("docker run --rm", "image sh -c "),
            ("docker run image", "sh -c "),
            ("docker run image sh", "-c "),
            ("sh", "-c "), ("sh -c", ""),
            ("export -p", WRAPPERS[0]),
        ):
            for suffix in ("", " | true"):
                words = _decoded_shell_words(left + separator + right
                                             + shlex.quote(SCRIPT) + suffix, env)
                assert discovery_wrapper(words, env) is None


@pytest.mark.parametrize("literal", [";", "\n"])
@pytest.mark.parametrize("tolerant", [False, True])
def test_r14_quoted_delimiters_are_values_not_boundaries(literal, tolerant):
    env = normalize_env({"A": "x", "VALUE": SECRET})
    decode = _tolerant_words if tolerant else _decoded_shell_words
    words = decode("export A " + shlex.quote(literal) + "\n"
                   + WRAPPERS[0] + shlex.quote(SCRIPT), env)
    assert words[2] == literal and isinstance(words[2], str)
    assert sum(word is _STATEMENT_END for word in words) == 1
    assert discovery_wrapper(words, env) is None
    for argv in (["sh", "-c", literal], ["docker", "run", "image", "sh", "-c",
                                         literal]):
        assert discovery_wrapper(argv, env)[0] == literal
    found = set()
    _apply_words(["--api-key", literal, _STATEMENT_END], found)
    assert found == {literal}
    found.clear()
    _apply_words(["--api-key", _STATEMENT_END, SECRET], found)
    assert not found


@pytest.mark.parametrize("tolerant", [False, True])
def test_r14_escaped_semicolon_and_line_continuation_keep_export_open(tolerant):
    decode = _tolerant_words if tolerant else _decoded_shell_words
    words = decode("export VALUE=x\\\n A \\;\nsh -c 'body'", normalize_env({}))
    assert words[:4] == ("export", "VALUE=x", "A", ";")
    assert words[4] is _STATEMENT_END
    assert discovery_wrapper(words) is None


def test_r14_argv_export_has_no_following_statement_and_env_still_forwards():
    env = normalize_env({"VALUE": SECRET})
    tail = ["docker", "run", "-e", "VALUE", "image", "sh", "-c", SCRIPT]
    for prefix in (["export", "VALUE=" + SECRET, "A"], ["export", "docker"],
                   ["export", "sh"], ["export", "env"], ["export", "export"]):
        assert discovery_wrapper(prefix + tail, env) is None
        for literal in (";", "\n"):
            assert discovery_wrapper(prefix + [literal] + tail, env) is None
    wrapped = discovery_wrapper(["env", "A=1", *tail], env)
    assert wrapped == (SCRIPT, {"VALUE": (SECRET, False)})
    assert discovery_wrapper(["env", "VALUE=" + SECRET, "docker", "run",
                              "image", "sh", "-c", SCRIPT], env)[1] == {}


class _NoEnumeration(Mapping):
    def __getitem__(self, key):
        return {"A": ("x", False), "VALUE": (OLD, False)}[key]

    def __iter__(self):
        raise AssertionError("prefix discovery enumerated the base")

    def __len__(self):
        return 2


def test_r14_export_overlay_keeps_base_identity_and_local_precedence():
    base = _EnvOverlay(_NoEnumeration())
    text = "export VALUE=" + SECRET + " A\nsh -c " + shlex.quote(SCRIPT)
    for decode in (_decoded_shell_words, _tolerant_words):
        script, effective = discovery_wrapper(decode(text, base), base)
        assert script == SCRIPT
        assert isinstance(effective, _EnvOverlay) and effective._base is base
        assert effective["VALUE"] == (SECRET, False)
        assert base["VALUE"] == (OLD, False)


# R22: discovery retains known literals even when strict expansion rejects them.
R22_VALUES = ("-R22 space_e731ba", "-R22\ttab_e731ba", "-R22*glob_e731ba",
              "-R22?glob_e731ba", "-R22[ab]glob_e731ba")
R22_OPERANDS = ("$VALUE", "${VALUE}", '"$VALUE"', "--api-key=$VALUE")
R22_LAYOUTS = ("json", "mapping", "flow", "literal")
R22_SHAPES = ("shell", "argv", "command_args", "entrypoint_cmd",
              "string_command_args", "string_entrypoint_cmd")


class _R22NoHost(Mapping):
    def __getitem__(self, key):
        raise AssertionError("host environment read")

    def __iter__(self):
        raise AssertionError("host environment enumeration")

    def __len__(self):
        raise AssertionError("host environment length")


def _r22_fields(operand, shape):
    head = ["vllm", "serve", "/m"]
    if not operand.startswith("--api-key="):
        head.append("--api-key")
    if shape == "shell":
        return {"command": " ".join([*head, operand])}
    if shape == "argv":
        return {"command": [*head, operand]}
    left, right = ("entrypoint", "cmd") if "entrypoint" in shape else ("command", "args")
    return {left: " ".join(head) if shape.startswith("string_") else head,
            right: [operand]}


def _r22_document(fields, value, layout, env):
    first = dict(name="a", **fields, env=env)
    sibling = dict(name="b", command=["vllm", "serve", "/m",
                                      "--served-model-name=" + value])
    if layout in ("json", "flow"):
        return json.dumps({"containers": [first, sibling]})
    lines = ["containers:", "- name: a"]
    for key, command in fields.items():
        if layout == "mapping":
            if isinstance(command, list):
                lines += ["  " + key + ":"] + ["  - " + json.dumps(s) for s in command]
            else:
                lines += ["  " + key + ": " + json.dumps(command)]
        elif isinstance(command, list):
            lines += ["  " + key + ":"]
            for word in command:
                lines += ["  - |-"] + ["    " + s for s in word.split("\n")]
        else:
            lines += ["  " + key + ": |-"] + ["    " + s for s in command.split("\n")]
    lines += ["  env:"] + ["    " + k + ": " + json.dumps(v) for k, v in env.items()]
    return "\n".join(lines + ["- name: b", "  command: " + json.dumps(sibling["command"])]) + "\n"


def _r22_public(fields, value, layout, monkeypatch, declaring_code=None, env=None):
    import os
    env = {"VALUE": value} if env is None else env
    text = _r22_document(fields, value, layout, env)
    fmt = "json" if layout == "json" else "yaml"
    reports, raw, event = [], set(), set()
    with monkeypatch.context() as isolated:
        isolated.setattr(os, "environ", _R22NoHost())
        inventory = prescan_document_secrets(text, fmt)
        if fmt == "yaml":
            raw_document_taint(text, raw, _shell_taint)
            _event_taint(text, event, _shell_taint)
        for source in ("ordinary.json", value):
            for selection in ("a", "b", value):
                report = parse_launch(text, format=fmt, container=selection,
                                      source=source, **BASE)
                reports.append((source, selection, report, canonical_json(report), launch_markdown(report)))
    assert value in inventory
    if fmt == "yaml":
        assert value in raw and (value in event) is HAS_YAML
    for source, selection, report, js, md in reports:
        assert report.source.name == (source if source == "ordinary.json" else "[REDACTED]")
        assert value not in md
        if not (value.isascii() and "\\" not in value and '"' not in value and value not in js):
            assert not any(value in s for s in _strings(json.loads(js)))
        expected = ("yaml_parser_unavailable" if fmt == "yaml" and not HAS_YAML
                    else "container_not_found" if selection == value
                    else declaring_code if selection == "a" else None)
        assert [d.code for d in report.diagnostics] == ([expected] if expected else [])
        assert report.status == ("partial" if expected == "launch_partial_coverage"
                                 else "failed" if expected else "completed")
        if report.status in ("completed", "partial"):
            settings = (*report.applied, *report.neutral_operational,
                        *report.unmodeled_or_unsupported)
            name = "VALUE" if selection == "a" else "--served-model-name"
            setting = next(s for s in settings if s.name == name.replace(value, "[REDACTED]"))
            assert setting.redacted and setting.value == "[REDACTED]"


@pytest.mark.parametrize("layout", R22_LAYOUTS)
def test_r22_known_expansion_public_reports(layout, monkeypatch):
    for value in R22_VALUES:
        for operand in R22_OPERANDS:
            for shape in R22_SHAPES:
                code = ("structured_command_required" if "entrypoint" in shape else
                        ("launch_partial_coverage" if operand == '"$VALUE"' else
                         "unsupported_shell_syntax") if shape == "shell" else None)
                fields = _r22_fields(operand, shape)
                _r22_public(fields, value, layout, monkeypatch, code)


def test_r22_argv_discovery_keeps_literal_and_static_candidates():
    from mindie_sim.launch_parser import parse_argv, parse_shell, LaunchSyntaxError
    for value in (*R22_VALUES, "--served-model-name", "-x"):
        for operand in R22_OPERANDS:
            argv = _r22_fields(operand, "argv")["command"]
            found = set()
            _shell_taint(tuple(argv), found, {"VALUE": value})
            assert value in found and operand.removeprefix("--api-key=") in found
            assert parse_argv(argv).commands[0].argv == tuple(argv)
            shell = " ".join(argv)
            if value in R22_VALUES and operand != '"$VALUE"':
                with pytest.raises(LaunchSyntaxError, match="word_splitting_or_globbing_required"):
                    parse_shell(shell, env={"VALUE": value})
            else:
                assert parse_shell(shell, env={"VALUE": value}).commands[0].argv[-1].endswith(value)


def test_r22_expanded_option_names_remain_credentials(monkeypatch):
    for value in ("--served-model-name", "-x"):
        for operand in R22_OPERANDS:
            for shape in ("shell", "argv"):
                code = ("launch_partial_coverage" if shape == "shell" and value == "-x"
                        and not operand.startswith("--api-key=") else None)
                fields = _r22_fields(operand, shape)
                for layout in R22_LAYOUTS:
                    _r22_public(fields, value, layout, monkeypatch, code)


def test_r22_composite_and_prologue_static_values(monkeypatch):
    for value in R22_VALUES:
        env = {"VALUE": value, "HEAD": value[:4], "TAIL": value[4:]}
        for origin in ("composite_env", "prologue", "inline"):
            operand = '"${HEAD}"${TAIL}' if origin == "composite_env" else "${ALIAS}"
            prefix = {"composite_env": "", "prologue": "export ALIAS=" + shlex.quote(value) + "\n",
                      "inline": "ALIAS=" + shlex.quote(value) + " "}[origin]
            for attached in (False, True):
                tail = "--api-key" + ("=" if attached else " ") + operand
                fields = {"command": prefix + "vllm serve /m " + tail}
                found = set()
                _shell_taint(fields["command"], found, env)
                assert value in found
                for layout in R22_LAYOUTS:
                    _r22_public(fields, value, layout, monkeypatch, "unsupported_shell_syntax", env)


def test_r22_resolution_does_not_bypass_pair_exclusions():
    for value in R22_VALUES:
        for tail in (">--api-key $VALUE", "--api-key >$VALUE ordinary",
                     "--api-key; $VALUE", "--api-key\n$VALUE",
                     "--api-key --served-model-name", "--api-key -x"):
            found = set()
            _shell_taint("vllm serve /m " + tail, found, {"VALUE": value})
            assert value not in found and "--served-model-name" not in found and "-x" not in found


@pytest.mark.parametrize("tail", ("${MISSING}", "$EMPTY"))
def test_r22_fallback_requires_every_static_segment(tail):
    words = _decoded_shell_words("vllm --api-key $VALUE" + tail,
                                 {"VALUE": R22_VALUES[2], "EMPTY": ""})
    assert words[-1] == "$VALUE" + tail


@pytest.mark.parametrize("attached", (False, True))
def test_r22_static_values_are_not_recursively_expanded(attached):
    literal = "-R22*$OTHER"
    operand = ("--api-key=" if attached else "--api-key ") + "$VALUE"
    env = {"VALUE": literal, "OTHER": "unrelated"}
    for surface in ("vllm " + operand, (ShellText("vllm " + operand),),
                    ("vllm", *operand.split())):
        found = set()
        _shell_taint(surface, found, env)
        assert literal in found and "-R22*unrelated" not in found


R23_VALUE = "-R23*payload_9ac841"
R23_SHAPES = tuple(shape for shape in R22_SHAPES if shape != "shell")
R23_BINDINGS = ("separate", "attached", "SECRET_TOKEN", "SECRET_KEY")
R23_OPERANDS = (
    ("prefix $VALUE suffix", "prefix {} suffix"),
    ("prefix ${VALUE} suffix", "prefix {} suffix"),
    ('prefix "$VALUE" suffix', "prefix {} suffix"),
    ("prefix;\t$VALUE > suffix", "prefix;\t{} > suffix"),
    ("prefix | $VALUE & < suffix", "prefix | {} & < suffix"),
    ("prefix\n${VALUE}\nsuffix", "prefix\n{}\nsuffix"),
)


def _r23_fields(operand, shape, binding):
    head = ["vllm", "serve", "/m"]
    if binding == "separate":
        head.append("--api-key")
    else:
        operand = ("--api-key" if binding == "attached" else binding) + "=" + operand
    if shape == "argv":
        return {"command": [*head, operand]}
    left, right = ("entrypoint", "cmd") if "entrypoint" in shape else ("command", "args")
    return {left: " ".join(head) if shape.startswith("string_") else head,
            right: [operand]}


@pytest.mark.parametrize("layout", R22_LAYOUTS)
def test_r23_composite_argv_public_reports(layout, monkeypatch):
    import os
    from mindie_sim.launch_parser import parse_argv
    for operand, template in R23_OPERANDS:
        expected = template.format(R23_VALUE)
        for binding in R23_BINDINGS:
            literal = _r23_fields(operand, "argv", binding)["command"]
            assert parse_argv(literal).commands[0].argv == tuple(literal)
            for shape in R23_SHAPES:
                fields = _r23_fields(operand, shape, binding)
                for layout in (layout,):
                    text = _r22_document(fields, expected, layout, {"VALUE": R23_VALUE})
                    fmt = "json" if layout == "json" else "yaml"
                    raw, events, reports = set(), set(), []
                    with monkeypatch.context() as isolated:
                        isolated.setattr(os, "environ", _R22NoHost())
                        inventory = prescan_document_secrets(text, fmt)
                        if fmt == "yaml":
                            raw_document_taint(text, raw, _shell_taint)
                            _event_taint(text, events, _shell_taint)
                        for source in ("ordinary.json", expected):
                            for selection in ("a", "b", expected):
                                report = parse_launch(text, format=fmt, container=selection,
                                                      source=source, **BASE)
                                reports.append((selection, report, canonical_json(report),
                                                launch_markdown(report)))
                    assert expected in inventory and R23_VALUE in inventory
                    if fmt == "yaml":
                        assert expected in raw and (expected in events) is HAS_YAML
                    for selection, report, js, md in reports:
                        assert R23_VALUE not in md
                        if R23_VALUE in js:
                            assert not any(R23_VALUE in s for s in _strings(json.loads(js)))
                        if fmt == "yaml" and not HAS_YAML:
                            assert [d.code for d in report.diagnostics] == ["yaml_parser_unavailable"]
                        elif selection == expected:
                            assert [d.code for d in report.diagnostics] == ["container_not_found"]
                        elif selection == "b":
                            assert report.status == "completed" and not report.diagnostics
                            settings = (*report.applied, *report.neutral_operational,
                                        *report.unmodeled_or_unsupported)
                            item = next(s for s in settings if s.name == "--served-model-name")
                            assert item.redacted and item.value == "[REDACTED]"


def test_r23_single_quoted_argv_never_taints_unexpanded_env():
    expected = "prefix " + R23_VALUE + " suffix"
    for binding in R23_BINDINGS:
        for shape in R23_SHAPES:
            fields = _r23_fields("'prefix $VALUE suffix'", shape, binding)
            for layout in R22_LAYOUTS:
                text = _r22_document(fields, expected, layout, {"VALUE": R23_VALUE})
                fmt = "json" if layout == "json" else "yaml"
                inventory = prescan_document_secrets(text, fmt)
                assert expected not in inventory and R23_VALUE not in inventory
                assert "prefix $VALUE suffix" in inventory
                if fmt == "yaml":
                    for walker in (raw_document_taint, _event_taint):
                        found = set()
                        walker(text, found, _shell_taint)
                        assert expected not in found and R23_VALUE not in found
                report = parse_launch(text, format=fmt, container="b", source=expected, **BASE)
                assert R23_VALUE in launch_markdown(report)
                assert any(R23_VALUE in s for s in _strings(json.loads(canonical_json(report))))


def test_r23_static_operand_quotes_escapes_unknown_and_empty():
    from mindie_sim.launch_taint import _static_argv_operand
    for operand, expected in (
        ("$EMPTY", None), ('"$EMPTY"', ""), ("${EMPTY}", None),
        ("prefix $EMPTY suffix", "prefix  suffix"),
        ('prefix "$EMPTY" suffix', "prefix  suffix"),
        ("$VALUE$EMPTY", R23_VALUE), ("$EMPTY$VALUE", R23_VALUE),
        ("prefix $MISSING suffix", None), ("$VALUE${MISSING}", None),
        ("${MISSING}$VALUE", None), ("$VALUE'broken", None),
        ("$VALUE\\", None), ("${VALUE", None), ("${VALUE:-fallback}", None),
        ("prefix \\$VALUE suffix", "prefix $VALUE suffix"),
        ("prefix '$VALUE' suffix", "prefix $VALUE suffix"),
        ("prefix '$MISSING' $VALUE", "prefix $MISSING " + R23_VALUE),
        ('prefix "\\$VALUE" $VALUE', "prefix $VALUE " + R23_VALUE),
        ('prefix "\\q\\\"\\\\\\`$VALUE" suffix', 'prefix \\q"\\`' + R23_VALUE + ' suffix'),
        ("prefix\\ $VALUE\\;suffix", "prefix " + R23_VALUE + ";suffix"),
        ("prefix\\\n$VALUE suffix", "prefix" + R23_VALUE + " suffix"),
        ('"prefix\\\r\n$VALUE suffix"', "prefix" + R23_VALUE + " suffix"),
        ("'prefix\\\n$VALUE' suffix", "prefix\\\n$VALUE suffix"),
        ("prefix $$ $VALUE", None), ("prefix $(cmd) $VALUE", None),
    ):
        found = set()
        assert _static_argv_operand(operand, {"VALUE": R23_VALUE, "EMPTY": ""}, found) == expected
        if expected is None or R23_VALUE not in expected:
            assert not found
        else:
            assert found == {R23_VALUE}
        raw = set()
        _apply_words(["--api-key", operand], raw, env={"VALUE": R23_VALUE, "EMPTY": ""})
        assert operand in raw and "" not in raw
        if expected:
            assert expected in raw


@pytest.mark.parametrize("binding", R23_BINDINGS)
def test_r23_structured_expansion_is_nonrecursive_and_does_not_enumerate(binding):
    literal = "payload $OTHER ; 'quoted' \\ \n"

    class LookupOnly(Mapping):
        def __getitem__(self, name):
            return {"VALUE": (literal, False), "OTHER": ("unrelated", False)}[name]

        def __iter__(self):
            raise AssertionError("enumerated normalized base")

        def __len__(self):
            return 2

    found = set()
    argv = _r23_fields("prefix $VALUE suffix", "argv", binding)["command"]
    _shell_taint(tuple(argv), found, _EnvOverlay(LookupOnly()))
    assert "prefix " + literal + " suffix" in found and literal in found
    assert not any("unrelated" in s for s in found)


def test_r23_structured_operand_continuations_are_linear(monkeypatch):
    from mindie_sim import launch_env
    from mindie_sim.launch_taint import _static_argv_operand
    original = launch_env.continuation_length
    for n in (3, 257, 4096, 16384):
        gap = "\\\n\\\r\\\r\n" * n
        for quote in ("", '"', "'"):
            text = quote + "prefix; " + gap + "$VALUE > " + gap + "suffix" + quote
            calls = []

            def counted(value, index):
                calls.append(index)
                return original(value, index)

            with monkeypatch.context() as mp:
                mp.setattr(launch_env, "continuation_length", counted)
                decoded = _static_argv_operand(text, {"VALUE": R23_VALUE})
            assert decoded == (text[1:-1] if quote == "'" else "prefix; " + R23_VALUE + " > suffix")
            assert len(calls) <= len(text) + 4 and len(calls) == len(set(calls))
