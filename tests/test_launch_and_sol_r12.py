"""Round-12 regressions for `astra_review_r12.md` (P1-01c and P2-19).

[Role] Turn every Round-12 counterexample into a public launcher,
  credential-inventory and bounded-work assertion in all three import modes
  (default, PyYAML 6.0.3 and blocked PyYAML), without changing parse admission
  or mutating ``sys.path``.
[Invariants] A container's completed literal ``env`` resolves every buffered
  command/args surface, including standalone arrays and a late ``env``, in
  JSON, YAML events and raw flow; nested flow mappings keep their own scope.
  Supported Docker/``sh -c`` wrappers forward only ``-e/--env`` values.  The
  tolerant tokenizer keeps static assignments and strips all three escaped
  line continuations.  Truncated frames attach bottom-up before composition,
  and no unrelated scalar re-normalizes a whole environment.
[Timeouts & Locks] In-memory checks take no locks; external CLI checks have a
  30-second timeout and preserve the process's fixed optional-loader mode.
[Side-Effect Policy] CLI fixtures use only pytest's supplied temporary path;
  other fixtures are in-memory. No network, device or host-environment reads.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_env import EnvMap
from mindie_sim.launch_taint import (
    _shell_taint, prescan_document_secrets,
)
from mindie_sim.launch_yaml_raw import raw_document_taint
from mindie_sim.launch_yaml_taint import _event_taint

try:  # Availability is fixed at collection time for this process.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
SECRET = "R12_SECRET_5d8e2f"
TRUNCATED = "R12_TRUNCATED_46bde8"
SCRIPT = 'vllm serve /m --api"-key" "$VALUE"'
REFLECTOR = {"name": "b", "command": ["vllm", "serve", "/m",
                                     "--served-model-name", SECRET]}
PAIRS = (("command", "args"), ("entrypoint", "cmd"),
         ("command", "cmd"), ("entrypoint", "args"))
TAILS = ("| true", "&& true", "; true", "> /tmp/out", "$(true)",
         "`true`", '"unclosed')
CONTINUATIONS = ("\\\n", "\\\r\n", "\\\r")
VARIANTS = {
    "scalar": {"command": SCRIPT},
    "args_wrapper": {"args": ["sh", "-c", SCRIPT]},
    "argv_split": {"command": ["sh", "-c"], "args": [SCRIPT]},
    "docker_argv": {"command": ["docker", "run", "-e", "VALUE", "image",
                                "sh", "-c", SCRIPT]},
    "docker_scalar": {"command": "docker run -e VALUE image sh -c '"
                               + SCRIPT + "'"},
}


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION,
                                 **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _leaks(report, needle):
    return needle in canonical_json(report) or needle in launch_markdown(report)


def _containers(first, fmt):
    text = json.dumps({"containers": [first, REFLECTOR]})
    return text, fmt


def _assert_reflector_is_safe(text, fmt):
    report = _parse(text, format=fmt, container="b")
    if fmt == "yaml" and not HAS_YAML:
        assert report.status == "failed"
    else:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)


def test_r12_scoped_env_reaches_every_argv_surface():
    for variant in sorted(VARIANTS):
        for order in ("before", "after"):
            for fmt in ("json", "yaml"):
                first = {"name": "a"}
                if order == "before":
                    first["env"] = {"VALUE": SECRET}
                first.update(VARIANTS[variant])
                if order == "after":
                    first["env"] = {"VALUE": SECRET}
                text, fmt_used = _containers(first, fmt)
                assert SECRET in prescan_document_secrets(text, fmt_used)
                _assert_reflector_is_safe(text, fmt_used)


def _block_document(field, env_position):
    lines = ["containers:", "- name: a"]
    if env_position == "before":
        lines += ["  env:", "    VALUE: " + SECRET]
    if field == "scalar":
        lines.append("  command: '" + SCRIPT + "'")
    elif field == "command_seq":
        lines += ["  command:", "  - vllm", "  - serve", "  - /m",
                  "  - --api-key", "  - '\"$VALUE\"'"]
    elif field == "args_seq":
        lines += ["  args:", "  - sh", "  - -c", "  - '" + SCRIPT + "'"]
    elif field == "docker_seq":
        lines += ["  command:", "  - docker", "  - run", "  - -e", "  - VALUE",
                  "  - image", "  - sh", "  - -c", "  - '" + SCRIPT + "'"]
    if env_position == "after":
        lines += ["  env:", "    VALUE: " + SECRET]
    lines += ["- name: b", "  command: [vllm, serve, /m, "
              "--served-model-name, " + SECRET + "]"]
    return "\n".join(lines) + "\n"


def test_r12_raw_yaml_standalone_fields_use_late_scoped_env():
    for field in ("scalar", "command_seq", "args_seq", "docker_seq"):
        for order in ("before", "after"):
            text = _block_document(field, order)
            assert SECRET in prescan_document_secrets(text, "yaml")
            found = set()
            raw_document_taint(text, found, _shell_taint)
            assert SECRET in found
            if HAS_YAML:
                events = set()
                _event_taint(text, events, _shell_taint)
                assert SECRET in events
            _assert_reflector_is_safe(text, "yaml")


def test_r12_nested_flow_frames_keep_their_own_environment():
    for fmt in ("json", "yaml"):
        text = json.dumps({"containers": [
            {"name": "a", "env": {"VALUE": "OUTER_888"},
             "command": 'vllm serve /m --api-key "outer-$VALUE"'},
            {"name": "b", "command": 'vllm serve /m --api-key "own-$VALUE"',
             "meta": {"env": {"VALUE": "INNER_777"},
                      "command": 'vllm serve /m --api-key "inner-$VALUE"'}},
        ]})
        found = prescan_document_secrets(text, fmt)
        for expected in ("outer-OUTER_888", "inner-INNER_777"):
            assert expected in found
        assert "inner-OUTER_888" not in found
        assert "outer-INNER_777" not in found


def test_r12_docker_forwarding_is_explicit_and_literal():
    forbidden = json.dumps({"containers": [
        {"name": "a", "env": {"VALUE": SECRET},
         "command": "docker run -e OTHER image sh -c "
                    "'vllm serve /m --api-key \"$VALUE\"'"},
    ]})
    found = prescan_document_secrets(forbidden, "json")
    assert SECRET not in found
    assert "$VALUE" in found
    inline = json.dumps({"containers": [
        {"name": "a", "command": "docker run -e VALUE=INLINE_999 image "
                                "sh -c 'vllm serve /m --api-key \"$VALUE\"'"},
    ]})
    assert "INLINE_999" in prescan_document_secrets(inline, "json")
    assert SECRET not in prescan_document_secrets(inline, "json")


def test_r12_bare_shell_wrapper_uses_the_container_environment():
    for shell in ("sh", "bash", "/bin/sh", "/bin/bash"):
        for order in ("before", "after"):
            first = {"name": "a"}
            if order == "before":
                first["env"] = {"VALUE": SECRET}
            first["command"] = [shell, "-c",
                                'vllm serve /m --api-key "$VALUE"']
            if order == "after":
                first["env"] = {"VALUE": SECRET}
            text, fmt = _containers(first, "json")
            assert SECRET in prescan_document_secrets(text, fmt)
            _assert_reflector_is_safe(text, fmt)


def test_r12_tolerant_tokenizer_keeps_inline_assignments():
    for prefix in ("VALUE=" + SECRET + " ", "export VALUE=" + SECRET + "\n"):
        for tail in TAILS:
            command = prefix + SCRIPT + " " + tail
            text, fmt = _containers({"name": "a", "command": command}, "json")
            assert SECRET in prescan_document_secrets(text, fmt)
            _assert_reflector_is_safe(text, fmt)
            selected = _parse(text, format=fmt, container="a")
            assert selected.status == "failed"
            assert any(item.code == "unsupported_shell_syntax"
                       for item in selected.diagnostics)


def test_r12_tolerant_tokenizer_strips_continuations():
    for continuation in CONTINUATIONS:
        for quoted in (False, True):
            split = SECRET[:-2] + continuation + SECRET[-2:]
            value = '"' + split + '"' if quoted else split
            command = 'vllm serve /m --api"-key" ' + value + " | true"
            text, fmt = _containers({"name": "a", "command": command}, "json")
            found = prescan_document_secrets(text, fmt)
            assert SECRET in found
            assert SECRET[:-2] + "\n" + SECRET[-2:] not in found
            _assert_reflector_is_safe(text, fmt)


def test_r12_tolerant_tokenizer_records_between_commands():
    command = "true; VALUE=" + SECRET + ' vllm serve /m --api-key "$VALUE"'
    text, fmt = _containers({"name": "a", "command": command}, "json")
    assert SECRET in prescan_document_secrets(text, fmt)
    _assert_reflector_is_safe(text, fmt)


def test_r12_truncated_frames_attach_bottom_up():
    for pair in PAIRS:
        for cut in (0, 1, 2):
            for fmt in ("json", "yaml"):
                left, right = pair
                text = json.dumps({left: ["vllm", "serve", "/m", "--api-key"],
                                   right: [TRUNCATED]})
                if cut:
                    text = text[:-cut]
                assert TRUNCATED in prescan_document_secrets(text, fmt)
                raw = set()
                raw_document_taint(text, raw, _shell_taint)
                assert TRUNCATED in raw
                if fmt == "yaml" and HAS_YAML:
                    events = set()
                    _event_taint(text, events, _shell_taint)
                    assert TRUNCATED in events
                report = _parse(text, format=fmt, source=TRUNCATED)
                if cut:
                    assert report.status == "failed"
                assert TRUNCATED not in canonical_json(report)
                assert TRUNCATED not in launch_markdown(report)


def test_r12_truncated_nested_flow_attaches_every_open_frame():
    text = ('{"outer": {"command": ["vllm", "serve", "/m", "--api-key"], '
            '"args": ["' + TRUNCATED + '"')
    assert TRUNCATED in prescan_document_secrets(text, "yaml")
    raw = set()
    raw_document_taint(text, raw, _shell_taint)
    assert TRUNCATED in raw
    report = _parse(text, format="yaml", source=TRUNCATED)
    assert report.status == "failed"
    assert TRUNCATED not in canonical_json(report)
    assert TRUNCATED not in launch_markdown(report)


def _env_document(n):
    return {"env": {f"V{index}": "x" for index in range(n)},
            "command": ["vllm", "serve", "/m"],
            **{f"note{index}": "x" for index in range(n)}}


def _block_env_document(n):
    lines = ["env:"] + [f"  V{index}: x" for index in range(n)]
    lines.append("command: [vllm, serve, /m]")
    lines += [f"note{index}: x" for index in range(n)]
    return "\n".join(lines) + "\n"


def _run_mode(mode, n):
    if mode == "json":
        return prescan_document_secrets(json.dumps(_env_document(n)), "json")
    if mode == "event":
        _event_taint(json.dumps(_env_document(n)), set(), _shell_taint)
        return None
    if mode == "raw_flow":
        raw_document_taint(json.dumps(_env_document(n)), set(), _shell_taint)
        return None
    raw_document_taint(_block_env_document(n), set(), _shell_taint)
    return None


@pytest.mark.parametrize("mode", ["json", "event", "raw_flow", "raw_block"])
def test_r12_environment_normalization_is_linear(monkeypatch, mode):
    if mode == "event" and not HAS_YAML:
        return
    import mindie_sim.launch_taint as taint

    original = taint._decoded_shell_words
    counts = {}
    try:
        for n in (64, 128, 256, 512, 1024):
            counter = {"calls": 0, "env_entries": 0}

            def counted(value, env=None):
                counter["calls"] += 1
                counter["env_entries"] += len(env or {})
                return original(value, env)

            monkeypatch.setattr(taint, "_decoded_shell_words", counted)
            _run_mode(mode, n)
            counts[n] = counter
    finally:
        monkeypatch.setattr(taint, "_decoded_shell_words", original)
    # Unrelated scalars never hand the whole environment to the decoder.
    assert all(item["env_entries"] == 0 for item in counts.values()), mode
    for small, large in ((64, 128), (128, 256), (256, 512), (512, 1024)):
        assert counts[large]["calls"] <= 2.2 * counts[small]["calls"] + 8, mode


def test_r12_env_is_normalized_once_per_completed_frame(monkeypatch):
    import mindie_sim.launch_taint as taint

    original = taint.normalize_env
    calls = []

    def counted(env):
        if env and not isinstance(env, EnvMap):
            calls.append(len(env))
        return original(env)

    monkeypatch.setattr(taint, "normalize_env", counted)
    n = 512
    document = _env_document(n)
    document["command"] = ["vllm", "serve", "/m", "--api-key", "$V0"]
    assert "x" in prescan_document_secrets(json.dumps(document), "json")
    assert len(calls) == 1
    assert calls[0] == n


NUMERIC_MARKER = "SYNTHETIC_PLACEHOLDER_TOKEN_R29"
NUMERIC_BENIGN = "SYNTHETIC_CONFIG_VALUE"
SEXAGESIMAL_CASES = [
    (f"{head}_{groups}", head + ":00" * groups + ".0",
     "-0.0" if head == "-0" else "0.0", None)
    for head in ("0", "-0") for groups in (173, 174, 175, 500, 601)
] + [
    (f"tail_{groups}", "0" + ":00" * groups + ":01:30.5", "90.5", None)
    for groups in (174, 175, 200, 501)
] + [
    (f"overflow_{head}_{groups}", head + ":00" * groups + ".0",
     None, "non_finite_json_number")
    for head in ("1", "-1") for groups in (174, 175, 500)
] + [
    ("finite_two_173", "2" + ":00" * 173 + ".0", str(2.0 * 60 ** 173), None),
    ("multiply_overflow", "5" + ":00" * 173 + ".0", None, "non_finite_json_number"),
    ("sum_overflow", "4" + ":59" * 173 + ".0", None, "non_finite_json_number"),
    ("finite_173", "1" + ":00" * 173 + ".0", str(float(60 ** 173)), None),
    ("negative_173", "-1" + ":00" * 173 + ".0", str(-float(60 ** 173)), None),
    ("short_zero", "0.0", "0.0", None),
    ("integer_zero", "0" + ":00" * 700, None, None),
] + [
    (f"integer_{sign}_{groups}", sign + "1" + ":00" * groups,
     str((-1 if sign else 1) * 60 ** groups),
     "integer_too_large" if (60 ** groups).bit_length() > 4096 else None)
    for sign in ("", "-") for groups in (600, 693, 694, 700)
]


def _numeric_document(case, quoted, field, location):
    _label, raw, normalized, _error = case
    rendered = json.dumps(raw) if quoted else raw
    reflected = raw if quoted or normalized is None else normalized
    command = json.dumps(["vllm", "serve", "/m", "--served-model-name", NUMERIC_BENIGN,
                          "--revision", NUMERIC_MARKER, "--tokenizer=" + reflected])
    text = ("containers:\n- name: a\n  env:\n    " + field + ": " + rendered
            + "\n    API_KEY: " + NUMERIC_MARKER + "\n    SOURCE: " + NUMERIC_BENIGN
            + "\n  command: " + (command if location == "selected" else "[vllm, serve, /m]")
            + "\n- name: b\n  command: " + command + "\n")
    return text, reflected, "a" if location == "selected" else "b"


def _numeric_report_checks(report, success, code, field, reflected, source):
    from mindie_sim.feature_reports import to_dict, validate_launch_report
    from mindie_sim.schema.case import Case

    validate_launch_report(report)
    validate_launch_report(to_dict(report))
    assert report.status == ("completed" if success else "failed")
    secret_source = source == NUMERIC_MARKER or field == "TOKEN" and source == reflected
    assert report.source.name == ("[REDACTED]" if secret_source else source)
    for body in (canonical_json(report), launch_markdown(report)):
        assert NUMERIC_MARKER not in body
    if success:
        assert report.case is not None
        for name, value, redacted in (
            ("--served-model-name", NUMERIC_BENIGN, False),
            ("--revision", "[REDACTED]", True),
            ("--tokenizer", "[REDACTED]" if field == "TOKEN" else reflected,
             field == "TOKEN"),
        ):
            setting = _setting(report, name)
            assert (setting.value, setting.redacted) == (value, redacted)
    else:
        assert [item.code for item in report.diagnostics] == [code]


@pytest.mark.parametrize("quoted", (False, True))
@pytest.mark.parametrize("field", ("ORDINARY", "TOKEN"))
def test_yaml_numeric_public_admission_and_inventory(quoted, field):
    from mindie_sim.launch_documents import _yaml_load, DocumentInputError
    from mindie_sim.launch_yaml_numbers import numeric_spelling

    for case in SEXAGESIMAL_CASES:
        _label, raw, normalized, error = case
        assert numeric_spelling(raw) == normalized
        for location in ("selected", "sibling"):
            text, reflected, selected = _numeric_document(case, quoted, field, location)
            found = prescan_document_secrets(text, "yaml")
            assert NUMERIC_MARKER in found and NUMERIC_BENIGN not in found
            assert (reflected in found) == (field == "TOKEN")
            raw_found, event_found = set(), set()
            raw_document_taint(text, raw_found, _shell_taint)
            _event_taint(text, event_found, _shell_taint)
            for inventory in (raw_found, event_found) if HAS_YAML else (raw_found,):
                assert NUMERIC_MARKER in inventory and NUMERIC_BENIGN not in inventory
                assert (reflected in inventory) == (field == "TOKEN")
            success = HAS_YAML and (quoted or error is None)
            code = error if HAS_YAML else "yaml_parser_unavailable"
            if success:
                value = _yaml_load(text)["containers"][0]["env"][field]
                assert str(value) == reflected
                if isinstance(value, float):
                    assert value.hex() == float(reflected).hex()
            else:
                with pytest.raises(DocumentInputError) as caught:
                    _yaml_load(text)
                assert caught.value.code == code
                assert set(found) <= set(caught.value.secrets)
                assert NUMERIC_MARKER not in caught.value.message
            for source in (NUMERIC_MARKER, NUMERIC_BENIGN, reflected):
                report = _parse(text, format="yaml", container=selected, source=source)
                _numeric_report_checks(report, success, code, field, reflected, source)


@pytest.mark.parametrize("field", ("ORDINARY", "TOKEN"))
def test_yaml_numeric_external_cli(field, tmp_path, capsys):
    from pathlib import Path
    import subprocess
    import sys
    from mindie_sim.cli import main as cli_main
    from mindie_sim.feature_reports import validate_launch_report
    from mindie_sim.schema.case import Case

    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD))
    path = tmp_path / (NUMERIC_MARKER + ".yaml")

    def check_cli_output(rc, stdout, stderr, case, quoted, field, reflected):
        success = HAS_YAML and (quoted or case[3] is None)
        assert rc == (0 if success else 2), stderr
        assert stdout and not stderr
        assert NUMERIC_MARKER not in stdout + stderr
        if stdout.lstrip().startswith("{"):
            report = json.loads(stdout)
            validate_launch_report(report)
            assert report["status"] == ("completed" if success else "failed")
            assert report["source"]["name"] == str(path).replace(NUMERIC_MARKER, "[REDACTED]")
            if success:
                assert report["case"] is not None
                settings = {item["name"]: item for item in report["neutral_operational"]}
                for name, value, redacted in (
                    ("--served-model-name", NUMERIC_BENIGN, False),
                    ("--revision", "[REDACTED]", True),
                    ("--tokenizer", "[REDACTED]" if field == "TOKEN" else reflected,
                     field == "TOKEN"),
                ):
                    assert (settings[name]["value"], settings[name]["redacted"]) == (value, redacted)
            else:
                code = case[3] if HAS_YAML else "yaml_parser_unavailable"
                assert [item["code"] for item in report["diagnostics"]] == [code]

    for case in SEXAGESIMAL_CASES:
        for quoted in (False, True):
            for location in ("selected", "sibling"):
                text, reflected, selected = _numeric_document(case, quoted, field, location)
                path.write_text(text)
                for renderer in ("json", "markdown"):
                    argv = ["inspect-launch", str(path), "--input-format", "yaml",
                            "--container", selected, "--machine", "a3",
                            "--model", "qwen3-30b-a3b", "--framework-version", VERSION,
                            "--workload", str(workload), "--format", renderer]
                    rc = cli_main(argv)
                    captured = capsys.readouterr()
                    check_cli_output(rc, captured.out, captured.err, case, quoted, field, reflected)

    ok_case = SEXAGESIMAL_CASES[0]
    err_case = next(c for c in SEXAGESIMAL_CASES if c[3] is not None)
    smoke = (ok_case, False, "TOKEN", "selected") if field == "TOKEN" else (err_case, False, "ORDINARY", "sibling")
    for case, quoted, fld, location in (smoke,):
        text, reflected, selected = _numeric_document(case, quoted, fld, location)
        path.write_text(text)
        for renderer in ("json", "markdown"):
            command = [sys.executable, "-B", "-m", "mindie_sim.cli", "inspect-launch", str(path),
                       "--input-format", "yaml", "--container", selected, "--machine", "a3",
                       "--model", "qwen3-30b-a3b", "--framework-version", VERSION,
                       "--workload", str(workload), "--format", renderer]
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                     capture_output=True, text=True, timeout=30)
            check_cli_output(process.returncode, process.stdout, process.stderr,
                             case, quoted, fld, reflected)


def test_yaml_float_constructor_matches_safe_loader_bits():
    import math
    from mindie_sim.launch_documents import _yaml_load
    from mindie_sim.launch_yaml_numbers import numeric_spelling

    samples = ["0.0", "-0.0", "+0.0", "1_234.5_6", ".5", "1.0e-300", "1.0e+308"]
    samples += [sign + str(head) + ":" + ":".join(["01"] * groups) + ":30.125"
                for sign in ("", "-", "+") for head in (0, 1, 59)
                for groups in (0, 1, 2, 10, 170)]
    # Empty intermediate groups are not YAML numeric scalars.
    samples = [sample.replace("::", ":") for sample in samples]
    for raw in samples:
        if HAS_YAML:
            expected = _yaml.safe_load(raw)
            assert math.isfinite(expected)
            assert _yaml_load(raw).hex() == expected.hex()
            assert numeric_spelling(raw) == str(expected)
        else:
            assert numeric_spelling(raw) is not None


def test_yaml_constructor_overflow_retains_prescan(monkeypatch):
    import mindie_sim.launch_documents as documents

    if not HAS_YAML:
        return
    for stage in ("load", "digit", "finite"):
        with monkeypatch.context() as patch:
            fault = OverflowError("SYNTHETIC_CONFIG_VALUE")

            def fail(*args, **kwargs):
                raise fault

            if stage == "load":
                patch.setattr(_yaml, "load", fail)
            elif stage == "digit":
                patch.setattr(documents, "_sexagesimal", fail)
            else:
                patch.setattr(documents, "_check_value", fail)
            with pytest.raises(documents.DocumentInputError) as caught:
                documents._yaml_load("ordinary: 1:30.5", prescanned=(NUMERIC_MARKER,))
            assert caught.value.code == "non_finite_json_number"
            assert caught.value.message == "non_finite_number"
            assert caught.value.secrets == (NUMERIC_MARKER,)


def test_yaml_constructor_timeout_identity(monkeypatch):
    import mindie_sim.launch_documents as documents
    import mindie_sim.launch_yaml_numbers as numbers

    class DerivedTimeout(TimeoutError):
        pass

    class NumericTimeout(TimeoutError, OverflowError):
        pass

    for stage in ("load", "digit", "finite", "spelling", "multiply", "public"):
        for derived in ("base", "derived", "numeric"):
            with monkeypatch.context() as patch:
                kind = {"base": TimeoutError, "derived": DerivedTimeout, "numeric": NumericTimeout}[derived]
                fault = kind("SYNTHETIC_CONFIG_VALUE")
                visits = []

                def fail(*args, **kwargs):
                    visits.append(stage)
                    raise fault

                if stage == "multiply":
                    class Digit(float):
                        __mul__ = fail
                    call = lambda: numbers._sexagesimal("1:30.5", Digit)
                elif stage == "spelling":
                    patch.setattr(numbers, "_sexagesimal", fail)
                    call = lambda: numbers.numeric_spelling("1:30.5")
                elif not HAS_YAML:
                    continue
                else:
                    if stage in ("load", "public"):
                        patch.setattr(_yaml, "load", fail)
                    else:
                        patch.setattr(documents, "_sexagesimal" if stage == "digit" else "_check_value", fail)
                    call = lambda: documents._yaml_load("ordinary: 1:30.5", prescanned=(NUMERIC_MARKER,))
                    if stage == "public":
                        call = lambda: _parse("ordinary: 1:30.5\nTOKEN: " + NUMERIC_MARKER, format="yaml")
                with pytest.raises(TimeoutError) as caught:
                    call()
                assert caught.value is fault and visits == [stage]


def test_yaml_numeric_source_binding_in_unselected_container():
    for case in SEXAGESIMAL_CASES[:14]:
        for quoted in (False, True):
            raw = json.dumps(case[1]) if quoted else case[1]
            reflected = case[1] if quoted else case[2]
            text = ("containers:\n- name: a\n  env:\n    SOURCE: " + raw
                    + "\n    TOKEN: " + NUMERIC_MARKER
                    + "\n  command: 'vllm serve /m --api-key \"$SOURCE\"'\n- name: b\n  command: "
                    + json.dumps(["vllm", "serve", "/m", "--served-model-name", NUMERIC_BENIGN,
                                  "--revision", NUMERIC_MARKER, "--tokenizer=" + reflected]) + "\n")
            inventory = prescan_document_secrets(text, "yaml")
            assert reflected in inventory and NUMERIC_MARKER in inventory
            assert NUMERIC_BENIGN not in inventory
            for source in (reflected, NUMERIC_BENIGN):
                report = _parse(text, format="yaml", container="b", source=source)
                _numeric_report_checks(report, HAS_YAML, "yaml_parser_unavailable",
                                       "TOKEN", reflected, source)
