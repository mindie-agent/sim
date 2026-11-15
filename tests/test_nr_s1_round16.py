"""Round-16 scope and FIX25 delimited credentials / Case schema integrity.

[Role] Cover the R16 public disclosure and executable-statement scope boundary.
[Invariants] Prologues persist; executable inline values and Docker options do
  not. First-wrapper discovery stays bounded, and admission stays unchanged.
[Timeouts & Locks] Finite in-memory matrices; no locks or external execution.
[Side-Effect Policy] Synthetic credentials, explicit env; CLI fixtures use tmp_path.
"""
from __future__ import annotations

from collections.abc import Mapping
import itertools
import json
import shlex

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_docker import discovery_wrapper, iter_discovery_wrappers
from mindie_sim.launch_env import _EnvOverlay, normalize_env
from mindie_sim.launch_parser import parse_shell
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

SECRET = "R16_REGRESSION_73bd92"
DECOY = "R16_DECOY_c84019"
SCRIPT = 'vllm serve /m --api"-key" "$VALUE"'
BASE = dict(machine="a3", model="qwen3-30b-a3b",
            framework_version="vllm-ascend@0.30.0",
            workload=dict(num_requests=2, prompt_tokens=128, output_tokens=8,
                          prefix_tokens=0))
PROLOGUES = ("env A=1", "env", "B=2 env A=1", "env A=1 # comment",
             '"$ENV_CMD" A=1')
EXPORTS = ("export VALUE=" + SECRET + " A", "export docker")
WRAPPERS = ("docker run -e VALUE image sh -c ",
            "docker run --rm --env=VALUE image /bin/bash -c ")
LAYOUTS = ("json", "flow", "block", "mixed")
TAILS = ("", " | true", "; true")


def _env(export=""):
    return {"VALUE": DECOY if "VALUE=" in export else SECRET,
            "A": "x", "docker": "x", "ENV_CMD": "env", "SCRIPT": SCRIPT}


def _lead(prologue, position, export):
    lines = (prologue, export) if position == "before" else (export, prologue)
    return "# leading\nset -e\n" + "\n# between\n".join(lines) + "\nset -u\n"


def _document(command, env, layout):
    first = {"name": "a", "env": env, "command": command}
    second = {"name": "b", "command": ["vllm", "serve", "/m",
                                       "--served-model-name", SECRET]}
    if layout in {"json", "flow"}:
        return json.dumps({"containers": [first, second]})
    lines = ["containers:", "- name: a"]
    if layout == "mixed":
        lines += ["  env: " + json.dumps(env), "  command: " + json.dumps(command)]
    else:
        lines += ["  env:"] + [f"    {k}: {json.dumps(v)}" for k, v in env.items()]
        lines += ["  command: |-"] + ["    " + s for s in command.split("\n")]
    return "\n".join(lines + ["- name: b", "  command: " +
                             json.dumps(second["command"])]) + "\n"


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


def _safe(report):
    assert SECRET not in launch_markdown(report)
    assert not any(SECRET in s for s in _strings(json.loads(canonical_json(report))))
    assert report.source.name == "[REDACTED]"


def _public(command, env, layout, *, rejection=None, command_index=None,
            no_decoy=False):
    text = _document(command, env, layout)
    fmt = "json" if layout == "json" else "yaml"
    found = prescan_document_secrets(text, fmt)
    assert SECRET in found
    if no_decoy:
        assert DECOY not in found
    if fmt == "yaml":
        raw, events = set(), set()
        raw_document_taint(text, raw, _shell_taint)
        _event_taint(text, events, _shell_taint)
        assert SECRET in raw
        if HAS_YAML:
            assert SECRET in events
        else:
            assert not events
        if no_decoy:
            assert DECOY not in raw and DECOY not in events
    for selection in ("a", "b", SECRET):
        report = parse_launch(text, format=fmt, container=selection, source=SECRET,
                              command_index=command_index if selection == "a" else None,
                              **BASE)
        _safe(report)
        expected = ("yaml_parser_unavailable" if fmt == "yaml" and not HAS_YAML
                    else "container_not_found" if selection == SECRET
                    else rejection if selection == "a" else None)
        codes = [d.code for d in report.diagnostics]
        assert codes == ([expected] if expected else [])
        assert report.status == ("failed" if expected else "completed")
        if not expected:
            name = "--api-key" if selection == "a" else "--served-model-name"
            settings = (*report.applied, *report.neutral_operational,
                        *report.unmodeled_or_unsupported)
            assert next(s for s in settings if s.name == name).redacted


def test_r16_env_prologue_first_wrapper_contract():
    for prologue, position, export, wrapper in itertools.product(
        PROLOGUES, ["before", "after"], EXPORTS, (*WRAPPERS, "sh -c ", "bash -c ")
    ):
        env = normalize_env(_env(export))
        command = _lead(prologue, position, export) + wrapper + shlex.quote(SCRIPT)
        for tail in TAILS:
            for decode in (_decoded_shell_words, _tolerant_words):
                script, effective = discovery_wrapper(decode(command + tail, env), env)
                assert script == SCRIPT and effective["VALUE"] == (SECRET, False)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_r16_env_prologue_public_reports(layout):
    for prologue, position, export, wrapper, tail in itertools.product(
        PROLOGUES, ["before", "after"], EXPORTS, WRAPPERS, TAILS
    ):
        command = _lead(prologue, position, export) + wrapper + shlex.quote(SCRIPT) + tail
        _public(command, _env(export), layout,
                rejection="unsupported_shell_syntax" if tail else None)


FIRST_COMMANDS = ("vllm serve /first", "python3 -m vllm.entrypoints.openai.api_server",
                  "docker run image vllm serve /first",
                  "docker run image sh -c 'vllm serve /first'")


def _multi(first, prefix, style):
    script = {"literal": shlex.quote(SCRIPT), "reference": "'$SCRIPT'",
              "outer": json.dumps(SCRIPT)}[style]
    forwarding = "-e VALUE -e SCRIPT" if style == "reference" else "-e VALUE"
    return (prefix + "VALUE=" + DECOY + " " + first + "\n# boundary\nenv A=1\n"
            "docker run " + forwarding + " image sh -c " + script)


def test_r16_multi_command_public_reports():
    for first, prefix, style, layout, command_index in itertools.product(
        FIRST_COMMANDS, ["", "env "], ["literal", "reference", "outer"], LAYOUTS, [None, 1]
    ):
        command = _multi(first, prefix, style)
        # Structured command strings currently require an unambiguous command even
        # when the public caller supplies command_index. Preserve that admission
        # diagnostic while independently protecting sibling/failure reports.
        _public(command, _env(), layout, rejection="ambiguous_launch",
                command_index=command_index, no_decoy=True)


def test_r16_multi_command_walker_discards_inline_values():
    for first, prefix, style in itertools.product(
        (*FIRST_COMMANDS, "python -m vllm.entrypoints.cli.main"),
        ["", "env "], ["literal", "reference", "outer"]
    ):
        command, env = _multi(first, prefix, style), normalize_env(_env())
        for tail in TAILS:
            for decode in (_decoded_shell_words, _tolerant_words):
                words = decode(command + tail, env)
                first_wrapper = discovery_wrapper(words, env)
                if first == FIRST_COMMANDS[-1]:
                    assert first_wrapper == ("vllm serve /first", {})
                else:
                    assert first_wrapper is None
                wrappers = list(iter_discovery_wrappers(words, env))
                assert wrappers[-1][1]["VALUE"] == (SECRET, False)
                found = set()
                _shell_taint(command + tail, found, env)
                assert SECRET in found and DECOY not in found
        assert env["VALUE"] == (SECRET, False)


def test_r16_selected_second_shell_launch_redacts_without_local_leak():
    for prefix in ("", "env "):
        command = ("export VALUE=" + SECRET + "\n" + prefix + "VALUE=" + DECOY +
                   " vllm serve /first\ndocker run -e VALUE image sh -c " +
                   shlex.quote("vllm serve /inner-first\n" + SCRIPT))
        parsed = parse_shell(command, command_index=1)
        assert dict(parsed.commands[1].env)["VALUE"] == SECRET
        for index, code in ((1, None), (None, "ambiguous_launch")):
            report = parse_launch(command, source=SECRET, command_index=index, **BASE)
            _safe(report)
            assert [d.code for d in report.diagnostics] == ([code] if code else [])
            assert report.status == ("failed" if code else "completed")


NEGATIVES = ("env -i\n", "env -u A\n", "env env ", "A=1 export B\n",
             "env export B\n", "true\n", "set -x\n", "set -e -x\n",
             "docker run -e\n", "docker run --name\n", "docker run --rm\n",
             "sh -c\n", "vllm other /first\n", "mindie serve /first\n",
             "sglang serve /first\n", "python -m other\n",
             "docker run image true\n", "docker run\n")


def test_r16_unsupported_statement_never_advances():
    for prefix in NEGATIVES:
        command = prefix + WRAPPERS[0] + shlex.quote(SCRIPT)
        env = normalize_env(_env())
        for decode in (_decoded_shell_words, _tolerant_words):
            words = decode(command, env)
            assert discovery_wrapper(words, env) is None
            assert list(iter_discovery_wrappers(words, env)) == []
            words = decode("vllm serve /first\n" + command, env)
            assert list(iter_discovery_wrappers(words, env)) == []
        found = set()
        _shell_taint(command, found, env)
        assert SECRET not in found


def test_r16_docker_option_and_wrapper_argument_split_never_borrows():
    env = normalize_env(_env())
    for text in ("docker run -e\nVALUE image sh -c " + shlex.quote(SCRIPT),
                 "docker run image sh -c\n" + shlex.quote(SCRIPT)):
        for decode in (_decoded_shell_words, _tolerant_words):
            assert discovery_wrapper(decode(text, env), env) is None
            assert list(iter_discovery_wrappers(decode(text, env), env)) == []


def test_r16_prologue_overlay_compatibility_and_parser_export_visibility():
    for prefix, visibility in itertools.product(
        ["VALUE=", "env VALUE="], ["absent", "base", "before", "after"]
    ):
        env = {"VALUE": DECOY} if visibility == "base" else {}
        command = ("export VALUE=" + DECOY + "\n" if visibility == "before" else "")
        command += prefix + SECRET + "\n"
        if visibility == "after":
            command += "export VALUE\n"
        parsed = parse_shell(command + "docker run -e VALUE image vllm serve /m", env=env)
        assert ("VALUE" in dict(parsed.commands[0].env)) == (visibility != "absent")
        for decode in (_decoded_shell_words, _tolerant_words):
            normalized = normalize_env(env)
            words = decode(command + "sh -c '$VALUE'", normalized)
            script, effective = discovery_wrapper(words, normalized)
            assert script == SECRET and effective["VALUE"] == (SECRET, False)
            assert isinstance(effective, _EnvOverlay) and effective._base is normalized


def test_r16_decoder_persists_only_prefix_statements():
    for decode in (_decoded_shell_words, _tolerant_words):
        env = normalize_env({"VALUE": SECRET})
        command = ("env VALUE=" + DECOY + " vllm serve /first\n"
                   'vllm serve /second --api-key "$VALUE"')
        assert decode(command, env)[-1] == SECRET
        command = ('vllm serve /first --served-model-name VALUE=' + DECOY +
                   '\nvllm serve /second --api-key "$VALUE"')
        assert decode(command, env)[-1] == SECRET
        command = "env VALUE=" + DECOY + '\nvllm serve /m --api-key "$VALUE"'
        assert decode(command, env)[-1] == DECOY


class _NoEnumeration(Mapping):
    def __getitem__(self, key):
        return {"VALUE": (SECRET, False), "ENV_CMD": ("env", False)}[key]

    def __iter__(self):
        raise AssertionError("base enumeration")

    def __len__(self):
        raise AssertionError("base length query")


def test_r16_scope_walk_never_enumerates_base_or_builds_overlay_chains():
    base = _EnvOverlay(_NoEnumeration())
    assert "FOO" not in base and "VALUE" in base
    command = "env A=1\n" * 1500 + _multi(FIRST_COMMANDS[0], "env ", "literal")
    for decode in (_decoded_shell_words, _tolerant_words):
        words = decode(command, base)
        assert list(iter_discovery_wrappers(words, base)) == [
            (SCRIPT, {"VALUE": (SECRET, False)})]
    assert base["VALUE"] == (SECRET, False) and "A" not in base


def test_r16_later_prologue_and_docker_values_have_independent_scope():
    command = ("env VALUE=" + DECOY + " vllm serve /first\n"
               "env VALUE=" + SECRET + "\nexport VALUE\n" + WRAPPERS[0] +
               shlex.quote(SCRIPT) + "\ndocker run -e VALUE=" + DECOY +
               " image sh -c 'vllm serve /next'\n" + WRAPPERS[0] + shlex.quote(SCRIPT))
    for decode in (_decoded_shell_words, _tolerant_words):
        wrappers = list(iter_discovery_wrappers(decode(command, normalize_env({})), {}))
        assert [env["VALUE"][0] for _, env in wrappers] == [SECRET, DECOY, SECRET]
        found = set()
        _shell_taint(command, found)
        assert SECRET in found and DECOY not in found


def test_r16_standalone_env_and_multiple_launches_never_read_host(monkeypatch):
    import os

    class NoHost(_NoEnumeration):
        def __getitem__(self, key):
            raise AssertionError("host lookup")

    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoHost())
        for layout in LAYOUTS:
            _public(_lead(PROLOGUES[-1], "after", EXPORTS[0]) + WRAPPERS[0] +
                    shlex.quote(SCRIPT), _env(EXPORTS[0]), layout)
            _public(_multi(FIRST_COMMANDS[0], "env ", "outer"), _env(), layout,
                    rejection="ambiguous_launch", no_decoy=True)


# FIX25: delimited credentials and the serialized Case have separate boundaries.
R25_LAYOUTS = ('json', 'mapping', 'flow', 'block')
R25_SECRET = 'R25_REGRESSION_39bc71'


def _r25_document(operand, binding, layout, sibling='pre', shape='argv'):
    head = ['vllm', 'serve', '/m']
    tail = ['--api-key', operand] if binding == 'separate' else ['--api-key=' + operand]
    fields = ({'command': head + tail} if shape == 'argv' else
              {'command': head + tail[:-1], 'args': tail[-1:]} if shape == 'split' else
              {'entrypoint': ' '.join(head + tail[:-1]), 'cmd': tail[-1:]})
    other = ['vllm', 'serve', '/m', '--served-model-name=' + sibling]
    env = {'VALUE': R25_SECRET, 'VA': 'R25_DECOY'}
    if layout in ('json', 'flow'):
        return json.dumps({'containers': [dict(name='a', env=env, **fields),
                                         dict(name='b', command=other)]})
    lines = ['containers:', '- name: a']
    for key, value in fields.items():
        if isinstance(value, list):
            lines.append('  ' + key + ':')
            for atom in value:
                if layout == 'block':
                    lines += ['  - |-'] + ['    ' + s for s in atom.split('\n')]
                else:
                    lines.append('  - ' + json.dumps(atom))
        else:
            lines.append('  ' + key + ': ' + json.dumps(value))
    return '\n'.join(lines + ['  env: ' + json.dumps(env), '- name: b',
                             '  command: ' + json.dumps(other)]) + '\n'


def _r25_operand(word, position, variant):
    middle = {'continued': '$VA\\\nLUE', 'contiguous': '$VALUE',
              'literal': 'ordinary_fragment'}[variant]
    parts = ['start', middle, 'end']
    parts.insert(position, word)
    return ' '.join(parts)


def _r25_case(report):
    from mindie_sim.feature_reports import validate_launch_report
    from mindie_sim.schema.case import Case
    assert report.status == 'completed' and not report.diagnostics
    assert validate_launch_report(report) == ()
    assert validate_launch_report(json.loads(canonical_json(report))) == ()
    Case.from_mapping(report.case)
    assert 'compress_ratio' in report.case['model']['layer_plans'][0]
    arch = report.case['machine']['devices'][0]['compute_arch']
    assert {'preset', 'representations'} <= arch.keys()
    assert 'head_dim' in report.case['model']
    assert 'prefix_tokens' in report.case['workload']['requests'][0]


def test_r25_delimited_operand_preserves_benign_words():
    for word, position, variant, binding, layout in itertools.product(
        ('pre', 'head', 'prefix'), (0, 1, 3), ('continued', 'contiguous', 'literal'),
        ('separate', 'attached'), R25_LAYOUTS
    ):
        operand = _r25_operand(word, position, variant)
        text = _r25_document(operand, binding, layout, word)
        fmt = 'json' if layout == 'json' else 'yaml'
        inventories = [set(prescan_document_secrets(text, fmt))]
        if fmt == 'yaml':
            raw, events = set(), set()
            raw_document_taint(text, raw, _shell_taint)
            _event_taint(text, events, _shell_taint)
            inventories += [raw] + ([events] if HAS_YAML else [])
        expected = operand.replace('$VA\\\nLUE', R25_SECRET).replace('$VALUE', R25_SECRET)
        for found in inventories:
            assert {operand, expected} <= found
            assert not found.intersection((word, 'start', 'end', 'ordinary_fragment', 'R25_DECOY'))
            assert (R25_SECRET in found) == (variant != 'literal')
        for selection in ('a', 'b'):
            report = parse_launch(text, format=fmt, container=selection, **BASE)
            if fmt == 'yaml' and not HAS_YAML:
                assert report.status == 'failed' and report.case is None
                assert [d.code for d in report.diagnostics] == ['yaml_parser_unavailable']
                continue
            _r25_case(report)
            rows = (*report.applied, *report.neutral_operational, *report.unmodeled_or_unsupported)
            if selection == 'b':
                served = next(s for s in rows if s.name == '--served-model-name')
                assert served.value == word and not served.redacted
            else:
                assert next(s for s in rows if s.name == '--api-key').value == '[REDACTED]'
            if variant != 'literal':
                assert R25_SECRET not in canonical_json(report) + launch_markdown(report)


def test_r25_real_schema_substring_credentials_keep_case_keys():
    for secret, binding, layout in itertools.product(
        ('pre', 'head', 'prefix', 'model'), ('separate', 'attached'), R25_LAYOUTS
    ):
        text = _r25_document(secret, binding, layout, secret)
        fmt = 'json' if layout == 'json' else 'yaml'
        assert secret in prescan_document_secrets(text, fmt)
        for selection in ('a', 'b'):
            report = parse_launch(text, format=fmt, container=selection, source=secret, **BASE)
            assert report.source.name == '[REDACTED]'
            if fmt == 'yaml' and not HAS_YAML:
                assert report.status == 'failed' and report.case is None
                continue
            _r25_case(report)
            name = '--api-key' if selection == 'a' else '--served-model-name'
            rows = (*report.applied, *report.neutral_operational, *report.unmodeled_or_unsupported)
            setting = next(s for s in rows if s.name == name.replace(secret, '[REDACTED]'))
            assert setting.value == '[REDACTED]' and setting.redacted


def test_r25_mixed_command_fields_keep_operand_boundaries():
    for shape, binding, layout in itertools.product(
        ('split', 'string'), ('separate', 'attached'), R25_LAYOUTS
    ):
        operand = 'pre $VA\\\nLUE head suffix'
        text = _r25_document(operand, binding, layout, 'head', shape)
        found = prescan_document_secrets(text, 'json' if layout == 'json' else 'yaml')
        assert {operand, 'pre ' + R25_SECRET + ' head suffix', R25_SECRET} <= set(found)
        assert not set(found).intersection(('pre', 'head', 'suffix'))


def test_r25_only_wrapper_payloads_have_shell_word_boundaries():
    for wrapper in ('sh', 'bash', 'eval'):
        script = 'vllm serve /m --api-key actual_secret ordinary_suffix'
        argv = [wrapper, '-c', script] if wrapper != 'eval' else [wrapper, script]
        found = set()
        _shell_taint(argv, found)
        assert 'actual_secret' in found and 'ordinary_suffix' not in found
        found.clear()
        _shell_taint(['vllm serve /m --api-key hidden_word', '--api-key',
                      'pre literal suffix'], found)
        assert found == {'pre literal suffix'}


def test_r25_cli_case_consumers_agree(tmp_path, capsys):
    from mindie_sim.feature_cli import main
    source, work, ops = (tmp_path / name for name in ('launch.txt', 'work.json', 'ops.json'))
    work.write_text(json.dumps(BASE['workload']))
    ops.write_text(json.dumps([dict(op_id='matrix', family='matmul', flops=2e9,
                                    hbm_bytes=1048576)]))
    for operand, binding, layout in itertools.product(
        ('pre $VA\\\nLUE suffix', 'head $VALUE prefix',
         'prefix ordinary_fragment head', 'pre', 'head', 'model'),
        ('separate', 'attached'), R25_LAYOUTS
    ):
        source.write_text(_r25_document(operand, binding, layout))
        fmt = 'json' if layout == 'json' else 'yaml'
        for action in ('inspect-launch', 'sol', 'feature-deltas'):
            for render in ('json', 'markdown'):
                argv = [action, str(source), '--input-format', fmt, '--container', 'a',
                        '--machine', BASE['machine'], '--model', BASE['model'],
                        '--framework-version', BASE['framework_version'],
                        '--workload', str(work), '--format', render]
                if action != 'inspect-launch':
                    argv += ['--scope', 'body', '--ops-json', str(ops), '--context', '128',
                             '--sequences', '2']
                if action == 'feature-deltas':
                    argv += ['--features', '']
                assert main(argv) == (2 if fmt == 'yaml' and not HAS_YAML else 0)
                out, err = capsys.readouterr()
                assert out and not err
                if '$' in operand:
                    assert R25_SECRET not in out
                if render == 'json':
                    data = json.loads(out)
                    assert data['status'] == ('failed' if fmt == 'yaml' and not HAS_YAML
                                              else 'completed')
                    if action == 'inspect-launch' and data['case']:
                        from mindie_sim.schema.case import Case
                        Case.from_mapping(data['case'])


def test_r25_case_metadata_keys_are_still_untrusted():
    from mindie_sim.compose import compose_case
    base = compose_case(model=BASE['model'], hardware=BASE['machine'])
    base['runtime']['metadata'] = {'pre': {'head': 'pre'}, 'secret_key': 'head'}
    report = parse_launch(['vllm', 'serve', '/m', '--api-key', 'pre', '--password', 'head'],
                          format='argv', base_case=base, workload=BASE['workload'])
    _r25_case(report)
    metadata = report.case['runtime']['metadata']
    assert metadata['[REDACTED]'] == {'[REDACTED]': '[REDACTED]'}
    assert metadata['secret_key'] == '[REDACTED]'
    assert not any(secret in text for text in _strings(metadata) for secret in ('pre', 'head'))


def test_r25_validator_rejects_corrupt_completed_cases():
    from copy import deepcopy
    from dataclasses import replace
    from mindie_sim.feature_reports import ReportValidationError, validate_launch_report
    for as_mapping in (False, True):
        report = parse_launch(['vllm', 'serve', '/m'], format='argv', **BASE)
        case = deepcopy(report.case)
        layer = case['model']['layer_plans'][0]
        layer['com[REDACTED]ss_ratio'] = layer.pop('compress_ratio')
        corrupt = replace(report, case=case)
        if as_mapping:
            corrupt = json.loads(canonical_json(corrupt))
        with pytest.raises(ReportValidationError, match='L02:case_round_trip_invalid'):
            validate_launch_report(corrupt)


def test_r25_parse_never_completes_with_corrupted_scrubbed_case(monkeypatch):
    from mindie_sim import launch_actions
    original = launch_actions._scrub_case
    def corrupt(value, secrets):
        result = original(value, secrets)
        result['model']['layer_plans'][0]['bad_field'] = 1
        return result
    monkeypatch.setattr(launch_actions, '_scrub_case', corrupt)
    report = parse_launch(['vllm', 'serve', '/m', '--api-key', 'pre'], format='argv', **BASE)
    assert report.status == 'failed' and report.case is None
    assert any('case_round_trip_invalid' in d.message for d in report.diagnostics)


COMPOSITE_WORDS = (
    "'pre'\" literal suffix\"", '"pre"\' literal suffix\'',
    'pre" literal suffix"', r'pre\ literal\ suffix',
    "'pre'\\\n\" literal suffix\"", "'pre literal suffix'",
    "'pre'' literal suffix'", '"pre"" literal suffix"',
    "'pre'\\ literal\\ suffix", '"pre literal "suffix',
)


def test_composite_shell_credentials_preserve_sibling():
    for expression, binding, layout in itertools.product(
        COMPOSITE_WORDS, ('separate', 'attached', 'assignment'),
        ('json', 'flow', 'mapping', 'block', 'mixed')
    ):
        full = 'pre literal suffix'
        command = ('API_KEY=' + expression + ' vllm serve /m' if binding == 'assignment'
                   else 'vllm serve /m --api-key' +
                   (' ' if binding == 'separate' else '=') + expression)
        other = ['vllm', 'serve', '/m', '--served-model-name', 'pre']
        if layout in ('json', 'flow'):
            text = json.dumps({'containers': [dict(name='a', command=command),
                                             dict(name='b', command=other)]})
        else:
            field = ('  command: |-\n' + '\n'.join('    ' + s for s in command.split('\n'))
                     if layout == 'block' else '  command: ' + json.dumps(command))
            sibling = ('  command:\n' + '\n'.join('  - ' + json.dumps(s) for s in other)
                       if layout == 'mapping' else '  command: ' + json.dumps(other))
            text = 'containers:\n- name: a\n' + field + '\n- name: b\n' + sibling + '\n'
        fmt = 'json' if layout == 'json' else 'yaml'
        inventories = [set(prescan_document_secrets(text, fmt))]
        if fmt == 'yaml':
            raw, events = set(), set()
            raw_document_taint(text, raw, _shell_taint)
            _event_taint(text, events, _shell_taint)
            inventories += [raw] + ([events] if HAS_YAML else [])
        for found in inventories:
            assert full in found and found <= {full, expression}
        for selection in ('a', 'b'):
            report = parse_launch(text, format=fmt, container=selection, source=full, **BASE)
            assert report.source.name == '[REDACTED]'
            assert not any(full in s for s in _strings(json.loads(canonical_json(report))))
            assert full not in launch_markdown(report)
            if fmt == 'yaml' and not HAS_YAML:
                assert report.status == 'failed' and report.case is None
                assert [d.code for d in report.diagnostics] == ['yaml_parser_unavailable']
                continue
            _r25_case(report)
            name = ('--served-model-name' if selection == 'b' else
                    'API_KEY' if binding == 'assignment' else '--api-key')
            rows = (*report.applied, *report.neutral_operational, *report.unmodeled_or_unsupported)
            setting = next(s for s in rows if s.name == name)
            assert setting.value == ('pre' if selection == 'b' else '[REDACTED]')
            assert setting.redacted is (selection == 'a')
        if binding == 'assignment':
            script = command + ' --served-model-name pre'
            assert 'pre' not in parse_shell(script).secret_values
            report = parse_launch(script, **BASE)
            _r25_case(report)
            served = next(s for s in report.neutral_operational if s.name == '--served-model-name')
            assert served.value == 'pre' and not served.redacted


def test_composite_raw_recovery_and_json_strings():
    for binding, tail in itertools.product(
        ('--api-key ', '--api-key=', 'API_KEY='), ('', ' | true', ' "unclosed')
    ):
        for reference in ('${VALUE}', '$\\\n{VALUE}', '${VA\\\r\nLUE}'):
            script = 'vllm serve /m ' + binding + 'pre' + reference + 'suffix' + tail
            found = set()
            _shell_taint(script, found, {'VALUE': 'COMPLETE'})
            assert 'preCOMPLETEsuffix' in found
            assert found <= {'preCOMPLETEsuffix', 'pre' + reference + 'suffix'}
        for expression, full in (("'pre'bare", 'prebare'), ('pre\\ suffix', 'pre suffix'),
                                 ('pre,tail:part[0]', 'pre,tail:part[0]'),
                                 *((f'pre{space}bare', f'pre{space}bare')
                                   for space in ('\v', '\f', '\u00a0', '\u2003')),
                                 *(("'pre'\\" + nl + '" suffix"', 'pre suffix')
                                   for nl in ('\n', '\r', '\r\n'))):
            script = 'vllm serve /m ' + binding + expression + tail
            if binding == 'API_KEY=':
                from mindie_sim.launch_env import prescan_secret_values
                assert set(prescan_secret_values(script)) <= {expression, full}
            for text, fmt in ((script, 'shell'), (json.dumps({'command': script}), 'json')):
                found = set(prescan_document_secrets(text, fmt))
                assert full in found and found <= {expression, full}
        secret = 'COMPOSITE_JSON_SECRET'
        for script in ("vllm --additional-config '{\"api_key\":\"" + secret + "\"}'",
                       'vllm ' + binding + '"' + secret,
                       '"unclosed ' + binding + secret,
                       'vllm ' + binding + r'\"' + secret + r'\"'):
            surface = script if script.startswith('"unclosed') else script + tail
            assert any(secret in value for value in prescan_document_secrets(surface))
