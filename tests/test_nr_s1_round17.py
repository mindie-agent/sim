"""Round-17 boundaries/timeouts and Round-21 credential operand preservation.

[Role] Protect public launch reports across rejected shell operators and prove
  every scoped broad handler propagates cancellation by TimeoutError.
[Invariants] Admission stays strict; discovery uses lexical statement boundaries,
  not POSIX option execution or subshell semantics. Literal operators stay data.
[Timeouts & Locks] Finite synthetic cases and AST walks; no locks or subprocesses.
[Side-Effect Policy] Read-only source inspection and restored in-memory patches.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
import itertools
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shlex
import sys

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_docker import (
    _STATEMENT_END, discovery_wrapper, iter_discovery_wrappers,
)
from mindie_sim.launch_env import normalize_env
from mindie_sim.launch_parser import LaunchSyntaxError, parse_shell
from mindie_sim.launch_taint import (
    _apply_words, _decoded_shell_words, _shell_taint, _tolerant_words,
    prescan_document_secrets,
)
from mindie_sim import launch_yaml_raw as raw, launch_yaml_taint as event
from mindie_sim.prediction import sol_invocation as inv, sol_provenance as prov
from mindie_sim.prediction import sol_work as work

try:
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SECRET = "R17_REGRESSION_c2936d"
DECOY = "R17_LOCAL_DECOY_e194ab"
ENV = {"VALUE": SECRET, "SEED": SECRET, "ENV_CMD": "env"}
SCRIPT = 'vllm serve /second --api"-key" "$VALUE"'
SEPARATORS = ("\n", ";", "&", "&&", "||", "|")
PREFIXES = ("VALUE=" + DECOY + " ", "env VALUE=" + DECOY + " ")
FIRSTS = ("vllm serve /first", "docker run image sh -c 'vllm serve /first'")
LASTS = (SCRIPT, "docker run -e VALUE image sh -c " + shlex.quote(SCRIPT))
LAYOUTS = ("json", "flow", "block", "mixed")
BASE = dict(machine="a3", model="qwen3-30b-a3b",
            framework_version="vllm-ascend@0.30.0",
            workload=dict(num_requests=2, prompt_tokens=128, output_tokens=8,
                          prefix_tokens=0))


def _document(command, layout):
    sibling = ["vllm", "serve", "/m", "--served-model-name", SECRET]
    obj = {"containers": [{"name": "a", "env": ENV, "command": command},
                          {"name": "b", "command": sibling}]}
    if layout in ("json", "flow"):
        return json.dumps(obj)
    lines = ["containers:", "- name: a"]
    if layout == "mixed":
        lines += ["  env: " + json.dumps(ENV), "  command: " + json.dumps(command)]
    else:
        lines += ["  env:"] + [f"    {k}: {json.dumps(v)}" for k, v in ENV.items()]
        lines += ["  command: |-"] + ["    " + s for s in command.split("\n")]
    return "\n".join(lines + ["- name: b", "  command: " + json.dumps(sibling)]) + "\n"


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


def _public(command, layout, rejection):
    text = _document(command, layout)
    fmt = "json" if layout == "json" else "yaml"
    found = prescan_document_secrets(text, fmt)
    assert SECRET in found and DECOY not in found
    if fmt == "yaml":
        raw_found, event_found = set(), set()
        raw.raw_document_taint(text, raw_found, _shell_taint)
        event._event_taint(text, event_found, _shell_taint)
        assert SECRET in raw_found and DECOY not in raw_found
        assert (SECRET in event_found) if HAS_YAML else not event_found
        assert DECOY not in event_found
    for selection in ("a", "b", SECRET):
        report = parse_launch(text, format=fmt, container=selection, source=SECRET,
                              command_index=1 if selection == "a" else None, **BASE)
        assert SECRET not in launch_markdown(report)
        assert not any(SECRET in s for s in _strings(json.loads(canonical_json(report))))
        assert report.source.name == "[REDACTED]"
        expected = ("yaml_parser_unavailable" if fmt == "yaml" and not HAS_YAML
                    else "container_not_found" if selection == SECRET
                    else rejection if selection == "a" else None)
        assert [d.code for d in report.diagnostics] == ([expected] if expected else [])
        assert report.status == ("failed" if expected else "completed")
        if selection == "b" and not expected:
            settings = (*report.applied, *report.neutral_operational,
                        *report.unmodeled_or_unsupported)
            assert next(s for s in settings if s.name == "--served-model-name").redacted


def test_r17_operator_public_reports():
    for separator, prefix, first, last, layout in itertools.product(
        SEPARATORS, PREFIXES, FIRSTS, LASTS, LAYOUTS
    ):
        lead = ('set -e\nenv A=1\nexport VALUE="$SEED"\n' if prefix == PREFIXES[0]
                else 'export VALUE="$SEED"\nenv\nset -u\n')
        command = lead + prefix + first + separator + last
        rejection = "ambiguous_launch" if separator == "\n" else "unsupported_shell_syntax"
        _public(command, layout, rejection)
        env = normalize_env(ENV)
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, env)
            assert sum(w is _STATEMENT_END for w in words) == 4
            wrappers = list(iter_discovery_wrappers(words, env))
            assert len(wrappers) == (first == FIRSTS[1]) + (last == LASTS[1])
            if last == LASTS[1]:
                assert wrappers[-1] == (SCRIPT, {"VALUE": (SECRET, False)})
        assert env == normalize_env(ENV)


def test_r17_unsupported_env_options_stop_without_spilling():
    for option, order, separator in itertools.product(
        ["-u VALUE", "--unset VALUE", "--unset=VALUE", "-i", "--ignore-environment"],
        ["before", "after"], SEPARATORS
    ):
        inline = ("env " + option + " VALUE=" + DECOY if order == "before"
                  else "env VALUE=" + DECOY + " " + option)
        command = "export VALUE=" + SECRET + "\n" + inline + " vllm serve /first" + separator + SCRIPT
        with pytest.raises(LaunchSyntaxError) as raised:
            parse_shell(command, command_index=1)
        assert raised.value.code == "unsupported_shell_syntax"
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, normalize_env(ENV))
            assert discovery_wrapper(words) is None
            assert list(iter_discovery_wrappers(words)) == []
            assert words[-1] == SECRET
            wrapped = decode(command[:-len(SCRIPT)] + LASTS[1], normalize_env(ENV))
            assert discovery_wrapper(wrapped) is None
            assert list(iter_discovery_wrappers(wrapped)) == []
        found = set()
        _shell_taint(command, found)
        assert SECRET in found and DECOY not in found
        _public(command, "json", "ambiguous_launch" if separator == "\n"
                else "unsupported_shell_syntax")


def test_r17_quoted_and_escaped_operators_stay_literal():
    for operator, quote in itertools.product(
        ["&", "&&", "||", "|", "(", ")", "{", "}"], ["single", "double", "escape"]
    ):
        spelled = ("'" + operator + "'" if quote == "single" else
                   '"' + operator + '"' if quote == "double" else
                   "".join("\\" + c for c in operator))
        first = PREFIXES[1] + 'vllm serve /first --served-model-name ' + spelled
        parse_shell(first, env=ENV)
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(first + ' --api"-key" "$VALUE";' + SCRIPT, normalize_env(ENV))
            assert operator in words and sum(w is _STATEMENT_END for w in words) == 1
            assert words[words.index(_STATEMENT_END) - 1] == DECOY
            assert words[-1] == SECRET


def test_r17_redirection_does_not_end_the_command():
    for redirect in ("2>&1", "0<&1", "&>discard", "&>>discard",
                     "2>discard", "2>>discard", "2>|discard"):
        for separator in SEPARATORS:
            first = PREFIXES[0] + 'vllm serve /first ' + redirect + ' --api"-key" "$VALUE"'
            for decode in (_decoded_shell_words, _tolerant_words):
                words = decode(first + separator + LASTS[1], normalize_env(ENV))
                assert sum(w is _STATEMENT_END for w in words) == 1
                assert words[words.index(_STATEMENT_END) - 1] == DECOY
                assert list(iter_discovery_wrappers(words, normalize_env(ENV)))[-1] == (
                    SCRIPT, {"VALUE": (SECRET, False)})
            with pytest.raises(LaunchSyntaxError) as raised:
                parse_shell(first, env=ENV)
            assert raised.value.code == "unsupported_shell_syntax"


def test_r17_literal_redirect_character_cannot_hide_background_boundary():
    for before in ('">"', r'\>', '"<"', r'\<'):
        command = PREFIXES[0] + "vllm serve /first " + before + "&" + SCRIPT
        words = _tolerant_words(command, normalize_env(ENV))
        assert sum(w is _STATEMENT_END for w in words) == 1 and words[-1] == SECRET


def test_r17_separator_is_one_boundary():
    for separator in ("\r", "\r\n", *SEPARATORS):
        words = _tolerant_words(PREFIXES[1] + FIRSTS[0] + separator + SCRIPT,
                                normalize_env(ENV))
        assert sum(w is _STATEMENT_END for w in words) == 1 and words[-1] == SECRET


def test_r17_group_delimiters_do_not_transfer_executable_overlays():
    for (opening, closing), separator, last in itertools.product(
        [("(", ")"), ("{", "}")], SEPARATORS, LASTS
    ):
        for command in (opening + PREFIXES[1] + FIRSTS[0] + closing + separator + last,
                        PREFIXES[1] + FIRSTS[0] + separator + opening + last + closing):
            env = normalize_env(ENV)
            found = set()
            _shell_taint(command, found, env)
            assert SECRET in found and DECOY not in found
            words = _tolerant_words(command, env)
            if last == LASTS[1]:
                assert list(iter_discovery_wrappers(words, env))[-1][1]["VALUE"] == (SECRET, False)
            with pytest.raises(LaunchSyntaxError) as raised:
                parse_shell(command, env=ENV)
            assert raised.value.code == "unsupported_shell_syntax"


FAULTS = ("raw", "loader", "events", "cleanup", "catalog", "template_work",
          "pipeline_work", "scratch_work", "template_path", "pipeline_path",
          "provenance", "lowering", "projection")


def _fault_call(label, fail, patch):
    """Reach each distinct broad-handler body, including finally cleanup."""
    from mindie_sim.prediction import response_functions, execution
    from mindie_sim.prediction.program import PipelineProgram, PipelineTemplateProgram
    template = SimpleNamespace(setup=(), drain=(), tiles=1, reference_stages=fail)
    pipeline = PipelineProgram(1, 1, fail, homogeneous=True)
    if label == "raw":
        patch.setattr(raw, "_scan_raw", fail)
        return lambda: raw.raw_document_taint("API_KEY: value", set())
    if label == "loader":
        patch.setitem(sys.modules, "yaml", SimpleNamespace(SafeLoader=fail))
        return lambda: event._loader_spelling("123")
    if label in ("events", "cleanup"):
        fake = SimpleNamespace(SafeLoader=object, parse=fail)
        for name in ("MappingStartEvent", "MappingEndEvent", "SequenceStartEvent",
                     "SequenceEndEvent", "ScalarEvent"):
            setattr(fake, name, type(name, (), {}))
        if label == "cleanup":
            scalar = fake.ScalarEvent()
            scalar.value, scalar.style = "command", '"'
            fake.parse = lambda *a, **kw: iter([fake.MappingStartEvent(), scalar])
        patch.setitem(sys.modules, "yaml", fake)
        return lambda: event._event_taint("synthetic", set(), fail)
    if label == "catalog":
        return lambda: inv.template_catalog_selected(template)
    if label == "template_work":
        return lambda: inv.program_work(PipelineTemplateProgram(template))
    if label == "pipeline_work":
        return lambda: inv.program_work(pipeline)
    if label == "scratch_work":
        patch.setattr(inv, "_scratch_work", fail)
        return lambda: inv.program_work(SimpleNamespace(start=fail), engine=object())
    if label == "template_path":
        return lambda: inv.program_phase_path(PipelineTemplateProgram(template), {})
    if label == "pipeline_path":
        return lambda: inv.program_phase_path(pipeline, {})
    if label == "provenance":
        patch.setattr(response_functions, "provenance_roles", fail)
        return lambda: prov.affirmative_physical_provenance({"basis": "architectural"})
    from mindie_sim import compose_case
    from mindie_sim.schema.case import Case
    from mindie_sim.schema.operators import OpSpec
    case = Case.from_mapping(compose_case("qwen3-30b-a3b", hardware="a3", layers=1,
                                          graph_capture_mode="none"))
    op = OpSpec(op_id="x", family="matmul", phase="decode", stage=0, layer=0,
                requests=("r0",), semantic_shape=(1, 1), local_shape=(1, 1),
                padded_shape=(1, 1), launch_shape=(1, 1), hbm_bytes=1, dtype="bf16")
    if label == "lowering":
        patch.setattr(execution.Execution, "prepare", fail)
    else:
        assert label == "projection"
        patch.setattr(inv, "program_work", fail)
    return lambda: work.project_ops(case, (op,))


def test_r17_all_broad_handlers_propagate_only_timeouts(monkeypatch):
    for label, error in itertools.product(FAULTS, [TimeoutError, ValueError]):
        with monkeypatch.context() as mp:
            injected, calls = error("synthetic R17 fault"), []

            def fail(*args, **kwargs):
                calls.append(True)
                raise injected

            call = _fault_call(label, fail, mp)
            if error is TimeoutError:
                with pytest.raises(TimeoutError) as raised:
                    call()
                assert raised.value is injected
            else:
                result = call()
                if label in ("raw", "loader", "events", "cleanup"):
                    assert result is None
                elif label == "catalog":
                    assert result is False
                elif label == "provenance":
                    assert result is True
                elif label in ("lowering", "projection"):
                    assert result.coverage == "conditional_subset" and result.holes
                else:
                    assert result[1]
            assert calls == [True]


def scoped_paths():
    root = Path(__file__).resolve().parents[1]
    return sorted({root / "mindie_sim/feature_reports.py",
                   *root.glob("mindie_sim/launch_*.py"),
                   root / "mindie_sim/prediction/invocation_binding.py",
                   *root.glob("mindie_sim/prediction/sol*.py"),
                   *root.glob("tests/test_launch_and_sol*.py"),
                   *root.glob("tests/test_nr_s1_round*.py")})


def test_r17_scoped_structure_and_immediate_timeout_guards():
    paths, broad = scoped_paths(), 0
    assert len(paths) == 41
    markers = ("[Role]", "[Invariants]", "[Timeouts & Locks]", "[Side-Effect Policy]")
    for path in paths:
        data = path.read_bytes()
        assert b"\r" not in data and len(data.splitlines()) < 680, path
        tree = ast.parse(data)
        doc = tree.body[0]
        assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
        for marker in markers:
            assert marker in ast.get_docstring(tree)
            assert any(marker in line for line in data.decode().splitlines()[:30])
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Try, ast.TryStar)):
                continue
            for index, handler in enumerate(node.handlers):
                names = ({n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
                         | {n.attr for n in ast.walk(handler.type) if isinstance(n, ast.Attribute)}
                         if handler.type else {"BaseException"})
                if not names & {"Exception", "BaseException"}:
                    continue
                broad += 1
                assert index > 0, (path, handler.lineno)
                guard = node.handlers[index - 1]
                assert isinstance(guard.type, ast.Name) and guard.type.id == "TimeoutError"
                assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
                assert guard.body[0].exc is None and guard.body[0].cause is None
    assert broad == 13


R21_SECRET = "-R21_CREDENTIAL_49bd7e"
R21_VALUES = (
    ('$VALUE', R21_SECRET), ('"$VALUE"', R21_SECRET),
    ('"${VALUE}"', R21_SECRET), ("'{value}'", R21_SECRET),
    ('"{value}"', R21_SECRET), ('-R21_"CREDENTIAL"_49bd7e', R21_SECRET),
    (r"\{value}", R21_SECRET), ("{value}", -4917),
    ('"$VALUE"', -4917), ('$VALUE', "--served-model-name"),
    ('"$VALUE"', "--served-model-name"), ('"${VALUE}"', "--served-model-name"),
)


class _R21NoHost(Mapping):
    def __getitem__(self, key):
        raise AssertionError("host environment read")

    def __iter__(self):
        raise AssertionError("host environment enumeration")

    def __len__(self):
        raise AssertionError("host environment length")


def _r21_document(command, value, layout):
    # Late env, ordinary key VALUE, and inline sibling option match R21 exactly.
    sibling = ["vllm", "serve", "/m", "--served-model-name=" + str(value)]
    env = {"VALUE": value}
    obj = {"containers": [{"name": "a", "command": command, "env": env},
                          {"name": "b", "command": sibling}]}
    if layout in ("json", "flow"):
        return json.dumps(obj)
    lines = ["containers:", "- name: a"]
    if layout == "mixed":
        lines.append("  command: " + json.dumps(command))
    elif isinstance(command, list):
        lines += ["  command:"]
        for word in command:
            lines += ["  - |-", "    " + word]
    else:
        lines += ["  command: |-"] + ["    " + s for s in command.split("\n")]
    return "\n".join(lines + ["  env: " + json.dumps(env), "- name: b",
                              "  command: " + json.dumps(sibling)]) + "\n"


def _r21_public(command, value, layout, redirect, monkeypatch, secret=True):
    needle, fmt = str(value), "json" if layout == "json" else "yaml"
    text = _r21_document(command, value, layout)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", _R21NoHost())
        found = prescan_document_secrets(text, fmt)
        assert (needle in found) is secret
        if fmt == "yaml":
            raw_found, event_found = set(), set()
            raw.raw_document_taint(text, raw_found, _shell_taint)
            event._event_taint(text, event_found, _shell_taint)
            assert (needle in raw_found) is secret
            assert (needle in event_found) is (secret and HAS_YAML)
        for source in ("ordinary.json", needle):
            for selection in ("a", "b", needle):
                report = parse_launch(text, format=fmt, container=selection,
                                      source=source, **BASE)
                js, md = canonical_json(report), launch_markdown(report)
                assert report.source.name == (
                    "[REDACTED]" if secret and source == needle else source)
                if secret:
                    assert needle not in md
                    if not (needle.isascii() and "\\" not in needle and '"' not in needle and needle not in js):
                        assert not any(needle in s for s in _strings(json.loads(js)))
                elif source == needle:
                    assert needle in md and needle in tuple(_strings(json.loads(js)))
                expected = ("yaml_parser_unavailable" if fmt == "yaml" and not HAS_YAML
                            else "container_not_found" if selection == needle
                            else "unsupported_shell_syntax" if selection == "a"
                            and redirect else
                            {"--": "launch_partial_coverage", "-x": "launch_partial_coverage",
                             "-tp": "non_integer_parallel_axis"}.get(needle)
                            if selection == "a" and not secret else None)
                assert [d.code for d in report.diagnostics] == ([expected] if expected else [])
                assert report.status == ("partial" if expected == "launch_partial_coverage"
                                         else "failed" if expected else "completed")
                if not expected:
                    settings = (*report.applied, *report.neutral_operational,
                                *report.unmodeled_or_unsupported)
                    name = "VALUE" if selection == "a" else "--served-model-name"
                    if secret:
                        name = name.replace(needle, "[REDACTED]")
                    setting = next(s for s in settings if s.name == name)
                    assert setting.redacted is secret
                    assert setting.value == ("[REDACTED]" if secret else needle)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_r21_leading_dash_public_reports(layout, monkeypatch):
    for flag, (spelling, value), context in itertools.product(
        ("--api-key", '--pass"word"', '"--access-key"'), R21_VALUES,
        ("plain", "before_redirect", "after_redirect")
    ):
        operand = spelling.replace("{value}", str(value))
        tail = {"plain": flag + " " + operand,
                "before_redirect": flag + " 17>sink " + operand,
                "after_redirect": "2>sink " + flag + " " + operand}[context]
        command = "vllm serve /m " + tail
        for decode in (_decoded_shell_words, _tolerant_words):
            found = set()
            _apply_words(decode(command, normalize_env({"VALUE": value})), found)
            assert found == {str(value)}
        _r21_public(command, value, layout, context != "plain", monkeypatch)


def test_r21_structured_argv_public_reports(monkeypatch):
    for (operand, value), layout in itertools.product((
        ("$VALUE", R21_SECRET), ("${VALUE}", R21_SECRET),
        ("$VALUE", "--served-model-name"), ("$VALUE", -4917),
        (R21_SECRET, R21_SECRET), ("-4917", -4917),
    ), LAYOUTS):
        command = ["vllm", "serve", "/m", "--api-key", operand]
        _r21_public(command, value, layout, False, monkeypatch)


def test_r21_literal_options_are_not_credential_operands(monkeypatch):
    for option, layout, structured in itertools.product(
        ("--served-model-name", "--served-model-name=ordinary", "--", "-tp", "-x"),
        LAYOUTS, (False, True)
    ):
        argv = ["vllm", "serve", "/m", "--api-key", option, "ordinary"]
        command = argv if structured else " ".join(argv)
        _r21_public(command, option, layout, False, monkeypatch, secret=False)
        found = set()
        _shell_taint(command, found)
        assert option not in found and "ordinary" not in found


def test_r21_dash_values_never_pair_across_statement_boundaries():
    for separator in (*SEPARATORS, "\r", "\r\n", "(", ")", "{", "}"):
        command = 'vllm --api-key 2>sink' + separator + '"$VALUE"'
        for decode in (_decoded_shell_words, _tolerant_words):
            found = set()
            _apply_words(decode(command, normalize_env({"VALUE": R21_SECRET})), found,
                         raw_values=True)
            assert R21_SECRET not in found and "sink" not in found


def test_r21_continuations_do_not_quote_literal_options():
    for brk in ("\n", "\r", "\r\n"):
        option = ("-\\" + brk + "-served-model-name")
        for gap in (" ", " 17>sink "):
            command = "vllm --api-key" + gap + option + " ordinary"
            for decode in (_decoded_shell_words, _tolerant_words):
                found = set()
                _apply_words(decode(command, {}), found, raw_values=True)
                assert not found


def test_yaml_block_shell_continuations_preserve_content_spaces():
    from mindie_sim.schema.case import Case
    from mindie_sim.feature_reports import validate_launch_report, to_dict
    for spaces, binding, header, tail in itertools.product(
        [1, 2, 4], ["separate", "attached", "assignment"],
        ["|-", "|", "|+", ">-", ">", ">+"], ["", " | true", ' "unclosed']
    ):
        expr = 'pre" mid\\\n' + " " * spaces + 'dle"suffix'
        full, benign = "pre mid" + " " * spaces + "dlesuffix", "pre middlesuffix"
        command = ("API_KEY=" + expr + " vllm serve /m" if binding == "assignment"
                   else "vllm serve /m --api-key" + (" " if binding == "separate" else "=") + expr)
        sibling = ["vllm", "serve", "/m", "--served-model-name", benign, "--revision", full]
        text = ("containers:\n- name: a\n  command: " + header + "\n    " +
                (command + tail).replace("\n", "\n    ") +
                "\n- name: b\n  command: " + json.dumps(sibling) + "\n")
        inventories = [set(prescan_document_secrets(text, "yaml"))]
        for scan in (raw.raw_document_taint, event._event_taint):
            found = set()
            scan(text, found, _shell_taint)
            if scan is raw.raw_document_taint or HAS_YAML:
                inventories.append(found)
        for found in inventories:
            assert full in found and benign not in found
        for selection in ("b", "missing"):
            report = parse_launch(text, format="yaml", container=selection, source=benign, **BASE)
            assert report.source.name == benign
            assert full not in launch_markdown(report)
            assert not any(full in s for s in _strings(json.loads(canonical_json(report))))
            if HAS_YAML and selection == "b":
                assert report.status == "completed"
                Case.from_mapping(report.case)
                validate_launch_report(report)
                validate_launch_report(to_dict(report))
                settings = {s.name: s for s in report.neutral_operational}
                assert settings["--served-model-name"].value == benign
                assert settings["--served-model-name"].redacted is False
                assert settings["--revision"].value == "[REDACTED]"
                assert settings["--revision"].redacted is True
            else:
                assert report.status == "failed"
                code = "container_not_found" if HAS_YAML else "yaml_parser_unavailable"
                assert [d.code for d in report.diagnostics] == [code]


def test_secret_yaml_blocks_retain_exact_and_conservative_values():
    from mindie_sim.launch_yaml_blocks import block_scalar
    for header, spaces in itertools.product(["|-", "|", "|+", ">-", ">", ">+"], [1, 2, 4]):
        text = "api_key: " + header + "\n  pre\n  " + " " * spaces + "middle\n  suffix\n\n"
        value, end, lines = block_scalar(text, text.index(header), 0)
        expected = "pre\n" + " " * spaces + "middle\nsuffix"
        expected += "\n\n" if "+" in header else "" if "-" in header else "\n"
        assert value == expected and end == len(text)
        assert lines == ["pre", " " * spaces + "middle", "suffix", ""]
        found = set()
        raw.raw_document_taint(text, found, _shell_taint)
        assert {expected, "pre", "middle", "suffix", "pre middle suffix"} <= found
        if HAS_YAML:
            assert _yaml.safe_load(text)["api_key"] == expected


def test_folded_yaml_physical_lines_retain_comment_and_continuation_discovery():
    for header, tail in itertools.product([">-", ">", ">+"], ["", ' "unclosed']):
        command = "vllm serve /m # comment\n--api-key 'block_command_credential'" + tail
        text = "command: " + header + "\n  " + command.replace("\n", "\n  ") + "\n"
        found = set()
        raw.raw_document_taint(text, found, _shell_taint)
        assert "block_command_credential" in found
