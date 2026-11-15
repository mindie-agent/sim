"""Round-18/19/20 lexical identity, redirection roles and public taint scope.

[Role] Protect credential inventories and public reports on R18/R19/R20 corpora.
[Invariants] Discovery preserves lexical scope; strict admission is unchanged.
  Empty executables stop wrapper discovery. Quoted delimiters remain data.
[Timeouts & Locks] Finite synthetic cases and counted linear work; no locks.
[Side-Effect Policy] Restored in-memory patches; no host environment reads,
  subprocesses, network calls or file writes.
"""
from __future__ import annotations

from collections.abc import Mapping
import itertools
import json
import os
import shlex

import pytest

from mindie_sim import launch_env, launch_yaml_raw as raw, launch_yaml_taint as event
from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_docker import (
    _STATEMENT_END, _prefix_scope, discovery_wrapper, iter_discovery_wrappers,
)
from mindie_sim.launch_env import EnvMap, _ShellWord, _tolerant_words, normalize_env
from mindie_sim.launch_parser import LaunchSyntaxError, parse_shell
from mindie_sim.launch_taint import (
    _apply_words, _decoded_shell_words, _shell_taint, prescan_document_secrets,
)

try:
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SECRET, DECOY = "R18_REGRESSION_77cc23", "R18_LOCAL_DECOY_b2c58f"
LAST = 'vllm serve /second --api"-key" "$VALUE"'
WRAPPED = "docker run -e VALUE image sh -c " + shlex.quote(LAST)
SEPARATORS = ("\n", ";", "&", "&&", "||", "|")
BREAKS = ("\n", "\r", "\r\n")
LAYOUTS = ("json", "flow", "block", "mixed")
REDIRECTS = (("2>", "&1"), ("0<", "&1"), ("&", ">discard"),
             ("&", ">>discard"), ("2>", "|discard"))
BASE = dict(machine="a3", model="qwen3-30b-a3b",
            framework_version="vllm-ascend@0.30.0",
            workload=dict(num_requests=2, prompt_tokens=128, output_tokens=8,
                          prefix_tokens=0))


class NoHost(Mapping):
    def __getitem__(self, key):
        raise AssertionError("host environment read")

    def __iter__(self):
        raise AssertionError("host environment enumeration")

    def __len__(self):
        raise AssertionError("host environment length")


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


def _document(command, env, layout):
    sibling = ["vllm", "serve", "/m", "--served-model-name", SECRET]
    obj = {"containers": [{"name": "a", "env": env, "command": command},
                          {"name": "b", "command": sibling}]}
    if layout in ("json", "flow"):
        return json.dumps(obj)
    lines = ["containers:", "- name: a", "  env: " + json.dumps(env)]
    if layout == "mixed":
        lines.append("  command: " + json.dumps(command))
    else:
        normalized = command.replace("\r\n", "\n").replace("\r", "\n")
        lines += ["  command: |-"] + ["    " + s for s in normalized.split("\n")]
    return "\n".join(lines + ["- name: b", "  command: " + json.dumps(sibling)]) + "\n"


def _public(command, env, layout, rejection, boundaries, monkeypatch):
    text = _document(command, env, layout)
    fmt = "json" if layout == "json" else "yaml"
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoHost())
        found = prescan_document_secrets(text, fmt)
        assert SECRET in found and DECOY not in found
        if fmt == "yaml":
            raw_found, event_found = set(), set()
            raw.raw_document_taint(text, raw_found, _shell_taint)
            event._event_taint(text, event_found, _shell_taint)
            assert SECRET in raw_found and DECOY not in raw_found
            assert (SECRET in event_found) if HAS_YAML else not event_found
            assert DECOY not in event_found
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, normalize_env(env))
            assert words[-1] == SECRET
            assert sum(w is _STATEMENT_END for w in words) == boundaries
        with pytest.raises(LaunchSyntaxError) as raised:
            parse_shell(command, env=env, command_index=1)
        assert raised.value.code == "unsupported_shell_syntax"
        for selection in ("a", "b", SECRET):
            report = parse_launch(text, format=fmt, container=selection,
                                  source=SECRET, **BASE)
            assert SECRET not in launch_markdown(report)
            js = canonical_json(report)
            if SECRET in js:
                assert not any(SECRET in s for s in _strings(json.loads(js)))
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


def test_r18_redirection_public_reports(monkeypatch):
    for (left, right), continuation, prefix, layout in itertools.product(
        REDIRECTS, ("", *("\\" + b for b in BREAKS)), ("", "env "), LAYOUTS
    ):
        command = (prefix + "VALUE=" + SECRET + " vllm serve /first " + left
                   + continuation + right + ' --api"-key" "$VALUE"')
        _public(command, {"VALUE": DECOY}, layout, "unsupported_shell_syntax", 0,
                monkeypatch)


def test_r18_empty_command_public_reports(monkeypatch):
    for empty, separator, prefix, layout in itertools.product(
        ('""', "''", "''\"\"", '"$EMPTY"'), SEPARATORS, ("", "env "), LAYOUTS
    ):
        command = prefix + "VALUE=" + DECOY + " " + empty + separator + LAST
        rejection = "ambiguous_launch" if separator == "\n" else "unsupported_shell_syntax"
        _public(command, {"VALUE": SECRET, "EMPTY": ""}, layout, rejection, 1, monkeypatch)


def test_r18_operator_gaps_and_trailing_continuations():
    for operator, brk, repeated in itertools.product(
        ("<&", ">&", "&>", "&>>", "<<", ">>", "<>", ">|", "&&", "||"),
        BREAKS, (False, True)
    ):
        gap = "\\" + brk
        if repeated:
            gap += "\\\r\n\\\n\\\r"
        env = normalize_env({"VALUE": SECRET})
        logical = operator in ("&&", "||")
        tail = LAST if logical else 'discard --api"-key" "$VALUE"'
        lead = "env VALUE=" + DECOY + " vllm serve /first "
        for mask in range(1, 1 << len(operator)):
            spelled = "".join(c + (gap if mask & (1 << i) else "")
                              for i, c in enumerate(operator))
            command = lead + spelled + tail + ";" + WRAPPED
            for decode in (_decoded_shell_words, _tolerant_words):
                words = decode(command, env)
                assert words == decode(lead + operator + tail + ";" + WRAPPED, env)
                assert sum(w is _STATEMENT_END for w in words) == 1 + logical
                boundary = max(i for i, w in enumerate(words) if w is _STATEMENT_END)
                assert words[boundary - 1] == (SECRET if logical else DECOY)
                assert list(iter_discovery_wrappers(words, env))[-1] == (
                    LAST, {"VALUE": (SECRET, False)})
            with pytest.raises(LaunchSyntaxError) as raised:
                parse_shell(command, env={"VALUE": SECRET})
            assert raised.value.code == "unsupported_shell_syntax"


def test_r18_empty_executable_stops_wrappers_without_committing():
    for empty, separator, prefix in itertools.product(
        ('""', "''", "''\"\"", '"$EMPTY"', "$EMPTY",
         '"\\\n"', '"\\\r"', '"\\\r\n"', '""#foo', '""=bar'),
        SEPARATORS, ("", "env ")
    ):
        env = normalize_env({"VALUE": SECRET, "EMPTY": ""})
        lead = prefix + "VALUE=" + DECOY + " " + empty
        expected = "#foo" if empty == '""#foo' else "=bar" if empty == '""=bar' else ""
        for decode in (_decoded_shell_words, _tolerant_words):
            first = decode(lead, env)
            index, effective = _prefix_scope(first, env)
            word = empty if decode is _decoded_shell_words and empty == "$EMPTY" else expected
            assert first[index:] == (word,)
            assert effective["VALUE"] == (DECOY, False)
            words = decode(lead + separator + LAST, env)
            assert words[-1] == SECRET
            wrapped = decode(lead + separator + WRAPPED, env)
            assert discovery_wrapper(wrapped, env) is None
            assert list(iter_discovery_wrappers(wrapped, env)) == []
        found = set()
        _shell_taint(lead + separator + LAST, found, env)
        assert SECRET in found and DECOY not in found
        assert env == normalize_env({"VALUE": SECRET, "EMPTY": ""})
        with pytest.raises(LaunchSyntaxError) as raised:
            parse_shell(lead + separator + LAST, env={"VALUE": SECRET, "EMPTY": ""},
                        command_index=1)
        assert raised.value.code == "unsupported_shell_syntax"


def test_r18_export_empty_argument_discards_local_scope_and_stops_wrapper():
    for empty, separator in itertools.product(
        ('""', "''", "''\"\"", '"\\\n"', '"\\\r"', '"\\\r\n"', '"$EMPTY"', '$EMPTY'),
        SEPARATORS
    ):
        env = normalize_env({"VALUE": SECRET, "EMPTY": ""})
        lead = 'export VALUE=' + DECOY + ' ' + empty
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(lead + separator + LAST, env)
            assert words[-1] == SECRET
            assert list(iter_discovery_wrappers(decode(lead + separator + WRAPPED, env), env)) == []
        with pytest.raises(LaunchSyntaxError) as raised:
            parse_shell(lead + separator + LAST, env={"VALUE": SECRET, "EMPTY": ""}, command_index=1)
        assert raised.value.code == "unsupported_shell_syntax"
        found = set()
        _shell_taint(lead + separator + LAST, found, env)
        assert SECRET in found and DECOY not in found


def test_r18_empty_assignment_value_remains_a_local_assignment():
    for separator in SEPARATORS:
        env = normalize_env({"FOO": SECRET})
        command = 'FOO="" vllm serve /first "$FOO"' + separator + 'vllm serve /m "$FOO"'
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, env)
            assert words[0] == "FOO=" and words[words.index(_STATEMENT_END) - 1] == ""
            assert words[-1] == SECRET


def test_r18_word_existence_and_comment_boundaries():
    for text, expected in (
        ("", ()), (" \t ", ()), ("# comment", ()), ("\\\n\\\r\\\r\n", ()),
        (" \\\n # comment", ()), ("<> > < >> <<", ()), ("&>\\\n>", ()),
        ("&&", (_STATEMENT_END,)), ("|\\\n|", (_STATEMENT_END,)),
        ('""', ("",)), ("''\"\"", ("",)), ('""#foo', ("#foo",)),
        ('""=bar', ("=bar",)), ('"" # ignored', ("",)),
        ('""\\\n#foo', ("#foo",)), ('"$EMPTY"', ("",)),
        ('"\\\n"', ("",)), ('"\\\r"', ("",)), ('"\\\r\n"', ("",)),
        ('""\n', ("", _STATEMENT_END)),
    ):
        assert _tolerant_words(text, normalize_env({"EMPTY": ""})) == expected


def test_r18_quoted_or_escaped_continued_operators_stay_data():
    for operator, style in itertools.product(
        ("<&", ">&", "&>", "&>>", "&&", "||"), ("single", "double", "escape")
    ):
        literal = ("'" + operator + "'" if style == "single" else
                   '"' + operator + '"' if style == "double" else
                   "\\\n".join("\\" + c for c in operator))
        command = 'VALUE=' + DECOY + ' vllm serve /m ' + literal + ' "$VALUE";' + LAST
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, normalize_env({"VALUE": SECRET}))
            assert operator in words and sum(w is _STATEMENT_END for w in words) == 1
            assert words[words.index(_STATEMENT_END) - 1] == DECOY and words[-1] == SECRET


def test_r18_continuation_runs_have_linear_work(monkeypatch):
    class LookupOnly(EnvMap):
        def __iter__(self):
            raise AssertionError("base environment enumeration")

        def items(self):
            raise AssertionError("base environment copy")

    for size, operator in itertools.product((8, 128, 2048), ("&>>", ">&", "||", "&x", ">")):
        original, calls, consumed = launch_env.continuation_length, [], []

        def counted(text, index):
            calls.append(index)
            result = original(text, index)
            if result:
                consumed.append(index)
            return result

        gap = "\\\n\\\r\\\r\n" * size
        text = 'VALUE=' + SECRET + ' vllm serve /m ' + gap.join(operator) + gap
        with monkeypatch.context() as patch:
            patch.setattr(launch_env, "continuation_length", counted)
            _tolerant_words(text, LookupOnly(VALUE=(DECOY, False)))
        assert len(consumed) == 3 * size * len(operator)
        assert len(consumed) == len(set(consumed))
        assert len(calls) <= len(text) + 4


def test_r19_redirect_only_prologue_public_reports(monkeypatch):
    for (left, right), gap, separator, layout in itertools.product(
        ((">", "discard"), ("2>", "&1"), ("&>", ">discard"), ("0<", "&1"), ("2>", "|discard")),
        ("", *("\\" + b for b in BREAKS)), SEPARATORS, LAYOUTS
    ):
        command = "VALUE=" + SECRET + " " + left + gap + right + separator + LAST
        _public(command, {"VALUE": DECOY}, layout, "unsupported_shell_syntax", 1, monkeypatch)


R19_REDIRECTS = ("<discard", ">discard", "0<&1", "2>&1", "&>discard", "&>>discard",
                 "<<discard", "<<-discard", "<<<discard", ">>discard", "<>discard",
                 "2>|discard", "2>&-", '>"discard"', "> discard", ">\\\ndiscard",
                 "2\\\n>&1", "0\\\r<&1", "2\\\r\n>|discard", "&\\\n>\\\r>discard")


def test_r19_redirect_roles_preserve_direct_and_wrapper_scope():
    for redirect, separator in itertools.product(R19_REDIRECTS, SEPARATORS):
        env = normalize_env({"VALUE": DECOY})
        for decode in (_decoded_shell_words, _tolerant_words):
            lead = "VALUE=" + SECRET + " " + redirect
            first = decode(lead, env)
            assert all(isinstance(w, _ShellWord) and isinstance(w, str) for w in first)
            assert first[0].assignment_name == "VALUE" and not first[0].is_redirect
            assert all(w.is_redirect for w in first[1:])
            index, effective = _prefix_scope(first, env)
            assert index == len(first) and effective["VALUE"] == (SECRET, False)
            assert decode(lead + separator + LAST, env)[-1] == SECRET
            wrappers = list(iter_discovery_wrappers(decode(lead + separator + WRAPPED, env), env))
            assert wrappers == [(LAST, {"VALUE": (SECRET, False)})]
            assert discovery_wrapper(decode(lead + separator + WRAPPED, env), env) == wrappers[0]
            local = "VALUE=" + SECRET + " vllm serve /first " + redirect
            assert decode(local + separator + LAST, env)[-1] == DECOY
            assert list(iter_discovery_wrappers(decode(local + separator + WRAPPED, env), env)) == [
                (LAST, {"VALUE": (DECOY, False)})]


def test_r19_only_adjacent_unquoted_ascii_digits_are_io_numbers():
    for fd in ('"2"', r"\2", "2 ", "2\\\n ", "٢", "2x", "''2"):
        env = normalize_env({"VALUE": SECRET})
        lead = "VALUE=" + DECOY + " " + fd + ">discard"
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(lead + ";" + LAST, env)
            assert not words[1].is_redirect and words[2].is_redirect
            assert words[-1] == SECRET
            assert not list(iter_discovery_wrappers(decode(lead + ";" + WRAPPED, env), env))


def test_r19_missing_redirect_target_resets_at_every_boundary():
    for separator in (*SEPARATORS, "\r", "\r\n", "(", ")", "{", "}"):
        env = normalize_env({"VALUE": SECRET})
        lead = "VALUE=" + DECOY + " > " + separator
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(lead + LAST, env)
            assert words[words.index(_STATEMENT_END) + 1] == "vllm"
            assert not any(getattr(w, "is_redirect", False) for w in words)
            assert words[-1] == DECOY


def test_r19_redirects_do_not_break_credential_flag_value_pairing():
    for redirect, placement in itertools.product(R19_REDIRECTS, ("before", "between", "after")):
        tail = {'before': redirect + ' --api"-key" "$VALUE"',
                'between': '--api"-key" ' + redirect + ' "$VALUE"',
                'after': '--api"-key" "$VALUE" ' + redirect}[placement]
        for decode in (_decoded_shell_words, _tolerant_words):
            found = set()
            _apply_words(decode("vllm serve /m " + tail, normalize_env({"VALUE": SECRET})), found)
            assert found == {SECRET}


def test_r19_redirect_targets_still_contribute_credential_taint():
    for target in ("TOKEN=" + SECRET, "--api-key=" + SECRET):
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode("vllm >" + target, normalize_env({}))
            assert words[-1].is_redirect
            found = set()
            _apply_words(words, found)
            assert found == {SECRET}
        words = _decoded_shell_words('>VALUE=' + DECOY + ';' + LAST, {"VALUE": SECRET})
        assert words[0].is_redirect and words[-1] == SECRET


def test_r19_nonassignment_executables_stop_wrappers():
    for lead_tmpl, separator, wrapper in itertools.product(
        ('"VALUE"=', '""VALUE=', "''VALUE=", '"VALUE={}"', r"VAL\UE=", r"VALUE\="),
        SEPARATORS, (WRAPPED, "sh -c " + shlex.quote(LAST))
    ):
        lead = lead_tmpl.format(DECOY) if "{}" in lead_tmpl else lead_tmpl + DECOY
        env = normalize_env({"VALUE": SECRET})
        for decode in (_decoded_shell_words, _tolerant_words):
            first = decode(lead, env)
            assert first == ("VALUE=" + DECOY,)
            assert first[0].assignment_name is None and not first[0].unquoted_ident
            index, effective = _prefix_scope(first, env)
            assert index == 0 and effective is env
            words = decode(lead + separator + wrapper, env)
            assert discovery_wrapper(words, env) is None
            assert not list(iter_discovery_wrappers(words, env))
            assert decode(lead + separator + LAST, env)[-1] == SECRET
        with pytest.raises(LaunchSyntaxError):
            parse_shell(lead + separator + wrapper, env={"VALUE": SECRET}, command_index=1)


def test_r19_assignment_metadata_requires_a_lexical_ascii_name():
    for prefix, name in (("VALUE=", "VALUE"), ("_V9=", "_V9"),
                         ("VA\\\nLUE=", "VALUE"), ("9VALUE=", None), ("VÁLUE=", None),
                         ('""VALUE=', None), ("$NAME=", None), (r"VALUE\=", None)):
        for decode in (_decoded_shell_words, _tolerant_words):
            word = decode(prefix + '""', normalize_env({"NAME": "VALUE"}))[0]
            assert isinstance(word, str) and word.assignment_name == name


def test_r19_decoded_prefix_keywords_preserve_compatibility():
    for keyword, style, separator in itertools.product(
        ("env", "export", "set"), ('"{}"', '""{}', '\\{}', '${}', '"${}"', 'mixed'), SEPARATORS
    ):
        variable = "ENV_CMD" if keyword == "env" else keyword.upper()
        values = {"VALUE": SECRET, variable: keyword}
        env = normalize_env(values)
        head = (keyword[0] + '"' + keyword[1] + '"' + keyword[2:] if style == "mixed"
                else style.format(variable if "$" in style else keyword))
        lead = head + (" -e" if keyword == "set" else " VALUE=" + DECOY)
        expected = SECRET if keyword == "set" else DECOY
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(lead + separator + WRAPPED, env)
            assert words[0] == keyword and not words[0].unquoted_ident
            index, effective = _prefix_scope(words, env)
            assert words[index] == "docker" and effective["VALUE"][0] == expected
            for wrapper in (WRAPPED, "sh -c " + shlex.quote(LAST)):
                wrapped = decode(lead + separator + wrapper, env)
                first = discovery_wrapper(wrapped, env)
                assert first[0] == LAST and first[1]["VALUE"][0] == expected
                assert list(iter_discovery_wrappers(wrapped, env)) == [first]
            assert decode(lead + separator + LAST, env)[-1] == expected
        if separator == "\n":
            assert parse_shell(lead + separator + LAST, env=values).commands[0].argv[-1] == expected
        else:
            with pytest.raises(LaunchSyntaxError):
                parse_shell(lead + separator + LAST, env=values)


def test_r19_invalid_export_arguments_never_commit():
    for argument, separator in itertools.product(
        ('"OTHER"', r"OT\HER", "$OTHER", "-x", "bad-name", "9NAME",
         '"sh" -c ' + shlex.quote(LAST), "sh -c " + shlex.quote(LAST)),
        SEPARATORS
    ):
        env = normalize_env({"VALUE": SECRET, "OTHER": "OTHER"})
        lead = "export VALUE=" + DECOY + " " + argument
        for decode in (_decoded_shell_words, _tolerant_words):
            assert decode(lead + separator + LAST, env)[-1] == SECRET
            words = decode(lead + separator + WRAPPED, env)
            index, effective = _prefix_scope(words, env)
            assert index == 0 and effective is env
            assert discovery_wrapper(words, env) is None
            assert not list(iter_discovery_wrappers(words, env))


def test_r19_valid_exports_and_structured_argv_keep_their_scope():
    for value, separator in itertools.product((SECRET, '""'), SEPARATORS):
        env = normalize_env({"VALUE": DECOY})
        expected = "" if value == '""' else value
        for decode in (_decoded_shell_words, _tolerant_words):
            lead = "export VALUE=" + value + " OTHER"
            assert decode(lead + separator + LAST, env)[-1] == expected
            assert list(iter_discovery_wrappers(decode(lead + separator + WRAPPED, env), env)) == [
                (LAST, {"VALUE": (expected, False)})]
        argv = ("VALUE=" + expected, "docker", "run", "-e", "VALUE", "image", "sh", "-c", LAST)
        assert discovery_wrapper(argv, env) == (LAST, {"VALUE": (expected, False)})


def test_r19_redirect_flag_operands_cannot_start_flag_value_pairs():
    for redirect in (">", "0<", "2>", "&>>"):
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode("vllm " + redirect + "--api-key " + DECOY, normalize_env({}))
            assert any(w == "--api-key" and w.is_redirect for w in words)
            found = set()
            _apply_words(words, found)
            assert not found


def test_r19_io_number_and_target_continuations_are_consumed_once(monkeypatch):
    class LookupOnly(EnvMap):
        def __iter__(self):
            raise AssertionError("base enumeration")
        def items(self):
            raise AssertionError("base copy")

    for size in (3, 37, 1021):
        calls, consumed, original = [], [], launch_env.continuation_length
        def counted(text, index):
            calls.append(index)
            length = original(text, index)
            if length:
                consumed.append(index)
            return length
        gap = "\\\n\\\r\\\r\n" * size
        command = "VALUE=" + SECRET + " 2" + gap + ">" + gap + "&" + gap + "1;" + LAST
        with monkeypatch.context() as patch:
            patch.setattr(launch_env, "continuation_length", counted)
            words = _tolerant_words(command, LookupOnly(VALUE=(DECOY, False)))
        assert words[1:3] == ("2", "1") and all(w.is_redirect for w in words[1:3])
        assert words[-1] == SECRET
        assert len(consumed) == len(set(consumed)) == 9 * size
        assert len(calls) <= len(command) + 4


R20_OPERATORS = (">", ">>", "<", "<<", "<<<", "<>", ">&", "<&", ">|",
                 "&>", "&>>", "0<", "2>", "17>&", "2>|", "<<-")
R20_STYLES = ("adjacent", "spaced", "continued", "double", "single", "mixed")


def _r20_public(command, layout, secret, monkeypatch, malformed=False, text=None):
    # SECRET is deliberately also the ordinary sibling value in negative cases.
    # Assert the full report, not just the last decoded extraction pass.
    text = _document(command, {}, layout) if text is None else text
    fmt = "json" if layout == "json" else "yaml"
    if malformed:
        text = text[:-2] if layout in ("json", "flow") else text + "bad: [\n"
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoHost())
        assert (SECRET in prescan_document_secrets(command)) is secret
        assert (SECRET in prescan_document_secrets(text, fmt)) is secret
        if fmt == "yaml":
            raw_found, event_found = set(), set()
            raw.raw_document_taint(text, raw_found, _shell_taint)
            event._event_taint(text, event_found, _shell_taint)
            assert (SECRET in raw_found) is secret
            assert (SECRET in event_found) is (secret and HAS_YAML)
        for selection in ("a", "b", SECRET):
            report = parse_launch(text, format=fmt, container=selection,
                                  source=SECRET, **BASE)
            md, js = launch_markdown(report), canonical_json(report)
            assert report.source.name == ("[REDACTED]" if secret else SECRET)
            if secret:
                assert SECRET not in md
                if SECRET in js:
                    assert not any(SECRET in s for s in _strings(json.loads(js)))
            else:
                assert SECRET in md and (SECRET in js) and SECRET in tuple(_strings(json.loads(js)))
            unavailable = fmt == "yaml" and not HAS_YAML
            if unavailable:
                assert [d.code for d in report.diagnostics] == ["yaml_parser_unavailable"]
            elif malformed:
                assert report.status == "failed"
            elif selection == SECRET:
                assert [d.code for d in report.diagnostics] == ["container_not_found"]
            elif selection == "b":
                assert report.status == "completed" and not report.diagnostics
                settings = (*report.applied, *report.neutral_operational,
                            *report.unmodeled_or_unsupported)
                served = next(s for s in settings if s.name == "--served-model-name")
                assert served.redacted is secret
                assert served.value == ("[REDACTED]" if secret else SECRET)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_r20_redirect_targets_never_taint_public_siblings(layout, monkeypatch):
    for operator, style, flag in itertools.product(
        R20_OPERATORS, R20_STYLES, ("--api-key", "--password", "--access-key")
    ):
        target = {"adjacent": flag, "spaced": " \t" + flag,
                  "continued": "\\\n" + flag, "double": '"' + flag + '"',
                  "single": "'" + flag + "'", "mixed": flag[:3] + '"' + flag[3:] + '"'}[style]
        op = "\\\n".join(operator) if style == "continued" else operator
        _r20_public("vllm serve /m " + op + target + " " + SECRET,
                    layout, False, monkeypatch)


def test_r20_redirects_preserve_public_credentials(monkeypatch):
    for operator, control, layout in itertools.product(
        R20_OPERATORS, ("inline_flag", "inline_assignment", "before", "after"), LAYOUTS
    ):
        tail = {"inline_flag": operator + "--api-key=" + SECRET,
                "inline_assignment": operator + "OPENAI_API_KEY=" + SECRET,
                "before": "--api-key " + operator + '"--password" ' + SECRET,
                "after": operator + "sink --api-key " + SECRET}[control]
        _r20_public("vllm serve /m " + tail, layout, True, monkeypatch)


def test_r20_malformed_shell_and_documents_preserve_credentials(monkeypatch):
    for tail, malformed, layout in itertools.product(
        (' "unclosed', " 'unclosed", " && true", " | true",
         " $(true)", " `true`", " ; unsupported(", " >out"),
        (False, True), LAYOUTS
    ):
        _r20_public("vllm --api-key " + SECRET + tail, layout, True, monkeypatch, malformed)


def test_r20_quoted_wrapper_scripts_keep_redirect_roles(monkeypatch):
    for wrapper, secret, layout in itertools.product(
        ("sh -c ", "docker run image sh -c "), (False, True), LAYOUTS
    ):
        script = "vllm " + ("--api-key >out " if secret else '>"--api-key" ') + SECRET
        _r20_public(wrapper + shlex.quote(script), layout, secret, monkeypatch)


def test_r20_separate_flags_keep_raw_and_decoded_value_spellings():
    for tail in ("", ' "unclosed', " >sink"):
        command = "vllm --api-key R20_sec\\ret" + tail
        assert {r"R20_sec\ret", "R20_secret"} <= set(prescan_document_secrets(command))


def test_r20_unclosed_quote_recovery_preserves_redirect_exclusion(monkeypatch):
    for lead, secret, layout in itertools.product(
        ('"unclosed ', "'unclosed ", "'unclosed \"nested "), (False, True), LAYOUTS
    ):
        script = lead + ("--api-key >out " if secret else ">--api-key ") + SECRET
        _r20_public(script, layout, secret, monkeypatch)


def test_r20_raw_dash_values_remain_protected_across_redirects():
    for operator in R20_OPERATORS:
        command = "vllm --api-key " + operator + "sink -R20_secret"
        assert "-R20_secret" in prescan_document_secrets(command)


def test_r20_raw_recovery_keeps_linear_continuation_work(monkeypatch):
    for size, lead, secret in itertools.product(
        (1, 29, 997), ('"unclosed ', "'unclosed ", "'unclosed \"nested "), (False, True)
    ):
        original, calls = launch_env.continuation_length, []
        def counted(text, index):
            calls.append(index)
            return original(text, index)
        gap = "\\\n\\\r\\\r\n" * size
        target = 'sink ' if secret else '--api-key '
        text = (lead + ('--api-key ' if secret else '') + gap + '17>' + gap
                + '&' + gap + target + gap + SECRET)
        with monkeypatch.context() as patch:
            patch.setattr(launch_env, "continuation_length", counted)
            patch.setattr(os, "environ", NoHost())
            assert (SECRET in prescan_document_secrets(text)) is secret
        assert len(calls) <= len(text) + 4


def test_r20_block_command_chomping_keeps_literal_args_and_redirect_roles(monkeypatch):
    for (left, right), header, secret in itertools.product(
        (("command", "args"), ("entrypoint", "cmd"), ("command", "cmd"), ("entrypoint", "args")),
        ("|", "|-", "|+", ">", ">-", ">+", "|2", "|2-", "|2+", ">2", ">2-", ">2+"),
        (False, True)
    ):
        command = "vllm serve /m " + ("--api-key" if secret else ">--api-key")
        text = (f"containers:\n- name: a\n  {left}: {header}\n    {command}\n"
                f"  {right}: [{SECRET}]\n- name: b\n"
                f"  command: [vllm, serve, /m, --served-model-name, {SECRET}]\n")
        _r20_public(command + " " + SECRET, "block", secret, monkeypatch, text=text)
