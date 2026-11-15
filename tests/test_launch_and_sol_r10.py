"""Round-10 regressions for `astra_review_r10.md` (P1-01/P2-23).

[Role] Turn every Round-10 counterexample into a public launcher,
  credential-inventory, exact-interval and provenance-preservation assertion
  in all three import modes, without changing parse admission.
[Invariants] Composed `command`/`args` pairs are discovered across JSON and
  YAML containers before selection; the launch shell tokenizer supplies the
  decoded spelling of quoted flags and escaped assignment values; YAML plain
  numeric scalars use the loader's 1.1 resolution; raw quoted/block scalars
  match PyYAML's break, escaping and indentation semantics; a consumed shell
  escape and an assignment composition close at the escaped character's end;
  a bare `export NAME` keeps an existing value-origin map while a genuinely
  external expansion still fails closed.
[Timeouts & Locks] Bounded checks; external CLI calls have a 30-second limit.
[Side-Effect Policy] Synthetic fixtures use pytest temporary paths; no network.
"""
from __future__ import annotations

import itertools
import json

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_taint import prescan_document_secrets

try:  # Availability is fixed at collection time for the whole process.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
SECRET = "R10_SECRET_7f3d9b"
SCRIPT = ('vllm serve /m --compilation-config '
          "'{\"cudagraph_mode\":\"FULL\"}'")


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _leaks(report, needle):
    return needle in canonical_json(report) or needle in launch_markdown(report)


def _yaml_unavailable(report):
    return any(item.code == "yaml_parser_unavailable"
               for item in report.diagnostics)


def _docker_env(document):
    text = json.dumps(document, indent=2)
    report = _parse(text, format="json")
    assert report.status == "completed"
    return text, _setting(report, "cudagraph_mode").span


# ---------------------------------------------------------------- P1-01
def test_p1_01_split_flag_across_command_and_args_is_discovered():
    for format in ("json", "yaml"):
        first = {"name": "a", "command": ["vllm", "serve", "/m", "--api-key"],
                 "args": [SECRET]}
        second = {"name": "b", "command": ["vllm", "serve", "/m",
                                           "--served-model-name", SECRET]}
        if format == "json":
            text = json.dumps({"containers": [first, second]})
        else:
            text = ("containers:\n"
                    "- name: a\n"
                    "  command: [vllm, serve, /m, --api-key]\n"
                    "  args: [" + SECRET + "]\n"
                    "- name: b\n"
                    "  command: [vllm, serve, /m, --served-model-name, \""
                    + SECRET + "\"]\n")
        assert SECRET in prescan_document_secrets(text, format)
        report = _parse(text, format=format, container="b")
        if format == "yaml" and not HAS_YAML:
            assert report.status == "failed" and _yaml_unavailable(report)
            assert not _leaks(report, SECRET)
            continue
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
        assert not _leaks(report, SECRET)
        missing = _parse(text, format=format, container=SECRET)
        assert missing.status == "failed"
        assert not _leaks(missing, SECRET)


def test_p1_01_yaml_block_sequence_command_and_args_are_composed():
    text = ("containers:\n"
            "- name: a\n"
            "  command:\n"
            "  - vllm\n"
            "  - serve\n"
            "  - /m\n"
            "  - --api-key\n"
            "  args:\n"
            "  - " + SECRET + "\n"
            "- name: b\n"
            "  command: [vllm, serve, /m, --served-model-name, \""
            + SECRET + "\"]\n")
    assert SECRET in prescan_document_secrets(text, "yaml")
    report = _parse(text, format="yaml", container="b")
    if HAS_YAML:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    else:
        assert report.status == "failed" and _yaml_unavailable(report)
    assert not _leaks(report, SECRET)


def test_p1_01_argv_composition_preserves_element_spaces():
    secret = "R10 SECRET with spaces"
    first = {"name": "a", "command": ["vllm", "serve", "/m", "--api-key"],
             "args": [secret]}
    second = {"name": "b", "command": ["vllm", "serve", "/m",
                                       "--served-model-name", secret]}
    text = json.dumps({"containers": [first, second]})
    found = prescan_document_secrets(text, "json")
    assert secret in found
    report = _parse(text, format="json", container="b")
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, secret)
    string_command = json.dumps({"command": "vllm serve /m --api-key",
                                 "args": [secret]})
    assert secret in prescan_document_secrets(string_command, "json")


def test_p1_01_command_string_plus_args_is_composed():
    text = json.dumps({"command": "vllm serve /m --api-key",
                       "args": [SECRET]}, indent=2)
    assert SECRET in prescan_document_secrets(text, "json")
    report = _parse(text, format="json")
    assert report.status == "completed"
    assert not _leaks(report, SECRET)


def test_p1_01_shell_quoting_yields_the_decoded_secret():
    for command, format in itertools.product([
        'vllm serve /m --api"-key" ' + SECRET,
        "vllm serve /m '--api-key'=" + SECRET,
        "vllm serve /m --api-key='" + SECRET + "'",
        "export \"API_KEY\"=" + SECRET + " vllm serve /m",
        "API_KEY=R10_SECRET_7f3d\\9b vllm serve /m",
    ], ["json", "yaml"]):
        first = {"name": "a", "command": command}
        second = {"name": "b", "command": ["vllm", "serve", "/m",
                                           "--served-model-name", SECRET]}
        if format == "json":
            text = json.dumps({"containers": [first, second]})
        else:
            text = ("containers:\n"
                    "- name: a\n"
                    "  command: " + json.dumps(command) + "\n"
                    "- name: b\n"
                    "  command: [vllm, serve, /m, --served-model-name, \""
                    + SECRET + "\"]\n")
        assert SECRET in prescan_document_secrets(text, format)
        report = _parse(text, format=format, container="b")
        if format == "yaml" and not HAS_YAML:
            assert report.status == "failed" and _yaml_unavailable(report)
        else:
            assert report.status == "completed"
            assert _setting(report, "--served-model-name").redacted is True
        assert not _leaks(report, SECRET)


NUMERIC_CASES = (
    ("012345670123", "1402433619"),
    ("0b10101010101010101010", "699050"),
    ("0x123abc", "1194684"),
    ("12:34:56", "45296"),
    ("1:20.5", "80.5"),
    ("1.5e+3", "1500.0"),
)


def test_p1_01_yaml_numeric_env_uses_loader_spelling():
    for raw, expected in NUMERIC_CASES:
        if HAS_YAML:
            assert str(_yaml.safe_load("x: " + raw)["x"]) == expected
        text = ("containers:\n"
                "- name: a\n"
                "  env: {API_KEY: " + raw + "}\n"
                "  command: [vllm, serve, /m]\n"
                "- name: b\n"
                "  command: [vllm, serve, /m, --served-model-name, \""
                + expected + "\"]\n")
        assert expected in prescan_document_secrets(text, "yaml")
        report = _parse(text, format="yaml", container="b")
        if HAS_YAML:
            assert report.status == "completed"
            assert _setting(report, "--served-model-name").redacted is True
        else:
            assert report.status == "failed" and _yaml_unavailable(report)
        assert not _leaks(report, expected)


def test_p1_01_yaml_float_without_sign_exponent_stays_a_string():
    assert str(_yaml.safe_load("x: 1.5e3")["x"]) == "1.5e3" if HAS_YAML else True
    text = ("containers:\n"
            "- name: a\n"
            "  env: {API_KEY: 1.5e3}\n"
            "  command: [vllm, serve, /m]\n"
            "- name: b\n"
            "  command: [vllm, serve, /m, --served-model-name, \"1.5e3\"]\n")
    found = prescan_document_secrets(text, "yaml")
    assert "1500.0" not in found
    report = _parse(text, format="yaml", container="b")
    if HAS_YAML:
        assert report.status == "completed"
        # The raw plain scalar is still a credential value, so it is redacted;
        # only the non-existent loader spelling 1500.0 must stay absent.
        assert _setting(report, "--served-model-name").redacted is True
    else:
        assert report.status == "failed" and _yaml_unavailable(report)


RAW_CASES = {
    "escaped_blank": ('credential: "R10_A\\\n\n  R10_B"\n',
                      "R10_A\nR10_B"),
    "unicode_single": ("credential: 'R10_A\x85  R10_B'\n",
                       "R10_A R10_B"),
    "unicode_double": ('credential: "R10_A\u2028  R10_B"\n',
                       "R10_A\u2028R10_B"),
    "crlf_literal": ("credential: |-\r\n  R10_A\r\n  R10_B\r\n",
                     "R10_A\nR10_B"),
    "space_line_literal": ("credential: |+\n  R10_A\n    \n  R10_B\n",
                           "R10_A\n  \nR10_B\n"),
}


def test_p1_01_quoted_yaml_number_keeps_its_string_spelling():
    for quote in ('"', "'"):
        text = ("credential: " + quote + "012345670123" + quote
                + "\ncommand: [vllm, serve, /m]\n")
        found = prescan_document_secrets(text, "yaml")
        assert "012345670123" in found
        assert "1402433619" not in found


def test_p1_01_raw_yaml_scalar_matches_the_loader():
    for label in tuple(RAW_CASES):
        fragment, expected = RAW_CASES[label]
        text = fragment + "command: [vllm, serve, /m]\n"
        assert expected in prescan_document_secrets(text, "yaml")
        report = _parse(text, format="yaml", source=expected)
        if HAS_YAML:
            assert report.status == "completed"
            assert not any(item.code == "source_coordinates_unavailable"
                           for item in report.diagnostics)
        else:
            assert report.status == "failed" and _yaml_unavailable(report)
        assert not _leaks(report, expected)


def test_p1_01_blocked_source_identity_is_redacted():
    text = 'credential: "R10_A\\\n\n  R10_B"\ncommand: [vllm, serve, /m]\n'
    report = _parse(text, format="yaml", source="R10_A\nR10_B")
    if HAS_YAML:
        assert report.status == "completed"
    else:
        assert report.status == "failed" and _yaml_unavailable(report)
    assert not _leaks(report, "R10_A\nR10_B")


def test_p1_01_yaml_flow_command_args_blocked_inventory():
    text = ("containers:\n"
            " - name: a\n"
            "   command: [vllm, serve, /m, --api-key]\n"
            "   args: [" + SECRET + "]\n"
            " - name: b\n"
            "   command: [vllm, serve, /m]\n")
    assert SECRET in prescan_document_secrets(text, "yaml")
    if HAS_YAML:
        return
    report = _parse(text, format="yaml", container="b", source=SECRET)
    assert report.status == "failed" and _yaml_unavailable(report)
    assert not _leaks(report, SECRET)


# ---------------------------------------------------------------- P2-23
def test_p2_23_shell_escape_interval_covers_the_escaped_character():
    report = _parse('vllm serve /m --compilation-config '
                    '"{\\"cudagraph_mode\\":\\"FULL\\"}"', format="shell")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (1, 38)
    assert (span.line_end, span.column_end) == (1, 56)


def test_p2_23_unquoted_shell_escape_interval_is_identical():
    report = _parse(r'vllm serve /m --compilation-config '
                    r'\{\"cudagraph_mode\":\"FULL\"\}', format="shell")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (1, 38)
    assert (span.line_end, span.column_end) == (1, 56)


def test_p2_23_assignment_composition_keeps_both_escape_endpoints():
    command = ('export SCRIPT="' + SCRIPT.replace('"', '\\"')
               + '"\ndocker run -e SCRIPT image sh -c "$SCRIPT"')
    text, span = _docker_env({"command": command})
    assert (span.line_start, span.column_start) == (2, 69)
    assert (span.line_end, span.column_end) == (2, 90)
    assert "SCRIPT" in text.splitlines()[span.line_start - 1]


def test_p2_23_bare_export_preserves_existing_provenance():
    for layout in ("document", "local", "export"):
        inline = SCRIPT.replace('"', '\\"')
        command = ('docker run -e SCRIPT image sh -c "$SCRIPT"')
        if layout == "document":
            document = {"env": {"SCRIPT": SCRIPT},
                        "command": "export SCRIPT\n" + command}
        elif layout == "local":
            document = {"command": 'SCRIPT="' + inline
                        + '"\nexport SCRIPT\n' + command}
        else:
            document = {"command": 'export SCRIPT="' + inline
                        + '"\nexport SCRIPT\n' + command}
        text, span = _docker_env(document)
        line = text.splitlines()[span.line_start - 1]
        assert "SCRIPT" in line
        if layout != "document":
            shell = _parse(document["command"], format="shell")
            assert shell.status == "completed"


def test_p2_23_external_bare_export_still_fails_closed():
    text = json.dumps({
        "env": {"DECOY": SCRIPT},
        "command": ('export EXTERNAL_SCRIPT\n'
                    'docker run -e EXTERNAL_SCRIPT image '
                    'sh -c "$EXTERNAL_SCRIPT"'),
    }, indent=2)
    report = _parse(text, format="json", env={"EXTERNAL_SCRIPT": SCRIPT})
    assert report.status == "failed"
    assert any(item.code == "unmapped_shell_expansion_provenance"
               for item in report.diagnostics)


NUMERIC_PLACEHOLDER = "SYNTHETIC_PLACEHOLDER_TOKEN_R28"


def _numeric_operand_document(digits, fmt, attached=False):
    operand = '{"api_key":' + digits + ',"token":"' + NUMERIC_PLACEHOLDER + '"}'
    option = (["--additional-config=" + operand] if attached else
              ["--additional-config", operand])
    commands = [["vllm", "serve", "/m", *option],
                ["vllm", "serve", "/m", "--served-model-name", "pre",
                 "--revision", NUMERIC_PLACEHOLDER]]
    if fmt == "json":
        return json.dumps({"containers": [dict(name=name, command=command)
                           for name, command in zip(("a", "b"), commands)]})
    return "containers:\n" + "".join(
        f"- name: {name}\n  command: {json.dumps(command)}\n"
        for name, command in zip(("a", "b"), commands))


def _assert_numeric_sibling(report, available, source):
    from mindie_sim.feature_reports import to_dict, validate_launch_report
    from mindie_sim.schema.case import Case

    assert report.source.name == source
    assert not _leaks(report, NUMERIC_PLACEHOLDER)
    if not available:
        assert report.status == "failed" and _yaml_unavailable(report)
        return
    assert report.status == "completed"
    Case.from_mapping(report.case)
    validate_launch_report(report)
    validate_launch_report(to_dict(report))
    benign = _setting(report, "--served-model-name")
    repeated = _setting(report, "--revision")
    assert benign.value == "pre" and benign.redacted is False
    assert repeated.value == "[REDACTED]" and repeated.redacted is True


def test_finite_integer_options_preserve_sibling_and_source():
    for power, sign, fmt, attached in itertools.product(
        [309, 400, 1000, 1250, 4300, 5000], ["", "-"], ["json", "yaml"], [False, True]
    ):
        text = _numeric_operand_document(sign + "1" + "0" * power, fmt, attached)
        found = prescan_document_secrets(text, fmt)
        assert NUMERIC_PLACEHOLDER in found and "pre" not in found
        if power <= 1000:
            assert sign + "1" + "0" * power in found
        for source in ("ordinary", NUMERIC_PLACEHOLDER):
            report = _parse(text, format=fmt, container="b", source=source)
            expected = "ordinary" if source == "ordinary" else "[REDACTED]"
            _assert_numeric_sibling(report, fmt == "json" or HAS_YAML, expected)


def test_supported_integer_spelling_never_uses_float_finiteness(monkeypatch):
    import mindie_sim.launch_taint as taint

    original = taint.math.isfinite

    def finite(value):
        assert isinstance(value, float)
        return original(value)

    for power, sign in itertools.product([309, 400, 1000], [1, -1]):
        with monkeypatch.context() as mp:
            mp.setattr(taint.math, "isfinite", finite)
            value = sign * 10 ** power
            found = taint.prescan_payload_secret_values(
                {"api_key": value, "token": NUMERIC_PLACEHOLDER, "secret": [True, 1.5]})
            assert found == tuple(sorted((str(value), NUMERIC_PLACEHOLDER, "true", "1.5")))
            assert taint._number_spelling(str(value)) == str(value)


def test_oversized_yaml_integer_keeps_later_placeholders():
    from mindie_sim.launch_taint import _shell_taint, prescan_payload_secret_values
    from mindie_sim.launch_yaml_raw import raw_document_taint
    from mindie_sim.launch_yaml_taint import _event_taint

    for digits in ("0x1" + "0" * 1100, "-0x1" + "0" * 4000,
                   "1" + "0" * 5000, "-1" + "0" * 5000):
        text = ("env:\n  API_KEY: " + digits + "\n  TOKEN: " + NUMERIC_PLACEHOLDER
                + "\ncommand: [vllm, serve, /m]\n")
        raw, event = set(), set()
        raw_document_taint(text, raw, _shell_taint)
        _event_taint(text, event, _shell_taint)
        assert NUMERIC_PLACEHOLDER in raw
        assert NUMERIC_PLACEHOLDER in prescan_document_secrets(text, "yaml")
        if HAS_YAML:
            assert NUMERIC_PLACEHOLDER in event
            if "0x" in digits:
                loaded = _yaml.safe_load(text)
                assert loaded["env"]["API_KEY"].bit_length() > 4096
                assert NUMERIC_PLACEHOLDER in prescan_payload_secret_values(loaded)
        report = _parse(text, format="yaml", source=NUMERIC_PLACEHOLDER)
        assert report.status == "failed" and report.source.name == "[REDACTED]"
        assert not _leaks(report, NUMERIC_PLACEHOLDER)
        codes = {item.code for item in report.diagnostics}
        assert ("integer_too_large" if HAS_YAML else "yaml_parser_unavailable") in codes


def test_typed_integer_bounds_keep_exact_supported_spelling():
    from mindie_sim.launch_taint import prescan_payload_secret_values

    for bits, sign in itertools.product([4096, 4097, 16001], [1, -1]):
        value = sign * ((1 << bits) - 1)
        found = prescan_payload_secret_values({"api_key": value, "token": NUMERIC_PLACEHOLDER})
        assert NUMERIC_PLACEHOLDER in found
        expected = tuple(sorted((str(value), NUMERIC_PLACEHOLDER))) if bits == 4096 else (
            NUMERIC_PLACEHOLDER,)
        assert found == expected


def test_integer_stringification_failure_keeps_later_sibling():
    from mindie_sim.launch_taint import prescan_payload_secret_values

    class UnspellableInteger(int):
        def __str__(self):
            raise ValueError("synthetic scalar conversion limit")

    assert prescan_payload_secret_values({"api_key": UnspellableInteger(1),
                                         "token": NUMERIC_PLACEHOLDER}) == (NUMERIC_PLACEHOLDER,)


def test_advisory_typed_failure_preserves_lexical_inventory(monkeypatch):
    import mindie_sim.launch_taint as taint

    raw = json.dumps({"token": NUMERIC_PLACEHOLDER, "ordinary": "pre"})
    original = taint.json.loads

    for error, stage in itertools.product([RecursionError, ValueError, OverflowError],
                                          ["load", "walk"]):
        def fail(payload, _err=error, _stage=stage):
            if _stage == "walk" or payload == raw:
                raise _err("synthetic typed traversal limit")
            return original(payload)

        with monkeypatch.context() as mp:
            if stage == "walk":
                mp.setattr(taint, "prescan_payload_secret_values", fail)
            else:
                mp.setattr(taint.json, "loads", fail)
            assert taint.prescan_cli_secret_values("--additional-config", raw) == (
                NUMERIC_PLACEHOLDER,)
            assert set(taint.prescan_cli_secret_values("--api-key", raw)) == {
                raw, NUMERIC_PLACEHOLDER}


def test_partial_typed_discovery_retains_completed_values(monkeypatch):
    import mindie_sim.launch_taint as taint

    original = taint.is_secret_name
    for error in (RecursionError, ValueError, OverflowError):
        def name(value, _err=error):
            if value == "unreadable":
                raise _err("synthetic typed traversal limit")
            return original(value)

        with monkeypatch.context() as mp:
            mp.setattr(taint, "is_secret_name", name)
            assert taint.prescan_payload_secret_values(
                {"token": NUMERIC_PLACEHOLDER, "unreadable": None}) == (NUMERIC_PLACEHOLDER,)


def test_numeric_discovery_propagates_timeout_identity(monkeypatch):
    import mindie_sim.launch_taint as taint

    class DerivedTimeout(TimeoutError):
        pass

    for entry, derived in itertools.product(["payload", "cli", "number"], [False, True]):
        fault = (DerivedTimeout if derived else TimeoutError)("synthetic timeout")

        def fail(*args, _fault=fault):
            raise _fault

        with monkeypatch.context() as mp:
            if entry == "payload":
                mp.setattr(taint, "is_secret_name", fail)
                call = lambda: taint.prescan_payload_secret_values(
                    {"token": NUMERIC_PLACEHOLDER})
            elif entry == "cli":
                mp.setattr(taint, "prescan_payload_secret_values", fail)
                call = lambda: taint.prescan_cli_secret_values("-c", '{"token":"synthetic"}')
            else:
                mp.setattr(taint.math, "isfinite", fail)
                call = lambda: taint.prescan_payload_secret_values({"api_key": 1.5})
            with pytest.raises(TimeoutError) as caught:
                call()
            assert caught.value is fault


def test_external_inspection_of_finite_integer_options(tmp_path, capsys):
    from pathlib import Path
    import subprocess
    import sys
    from mindie_sim.cli import main as cli_main
    from mindie_sim.schema.case import Case

    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD))
    for power, sign, fmt, renderer in itertools.product(
        [309, 400, 1000], ["", "-"], ["json", "yaml"], ["json", "markdown"]
    ):
        path = tmp_path / (NUMERIC_PLACEHOLDER + "." + fmt)
        path.write_text(_numeric_operand_document(sign + "1" + "0" * power, fmt))
        cli_args = ["inspect-launch", str(path), "--input-format", fmt, "--container", "b",
                    "--machine", "a3", "--model", "qwen3-30b-a3b",
                    "--framework-version", VERSION, "--workload", str(workload),
                    "--format", renderer]
        if (power, sign, fmt, renderer) in ((309, "", "json", "json"),
                                            (1000, "-", "yaml", "markdown")):
            process = subprocess.run([sys.executable, "-B", "-m", "mindie_sim.cli", *cli_args],
                                     cwd=Path(__file__).resolve().parents[1],
                                     capture_output=True, text=True, timeout=30)
            rc, stdout, stderr = process.returncode, process.stdout, process.stderr
        else:
            rc = cli_main(cli_args)
            stdout, stderr = capsys.readouterr()
        available = fmt == "json" or HAS_YAML
        assert rc == (0 if available else 2), stderr
        assert not stderr and NUMERIC_PLACEHOLDER not in stdout
        if renderer == "json":
            report = json.loads(stdout)
            assert report["status"] == ("completed" if available else "failed")
            if available:
                Case.from_mapping(report["case"])
                settings = {item["name"]: item for item in report["neutral_operational"]}
                assert settings["--served-model-name"]["value"] == "pre"
                assert settings["--served-model-name"]["redacted"] is False
                assert settings["--revision"]["value"] == "[REDACTED]"
                assert settings["--revision"]["redacted"] is True
            else:
                assert "yaml_parser_unavailable" in {item["code"] for item in report["diagnostics"]}


def test_yaml_numeric_conversion_limit_keeps_later_placeholder():
    from mindie_sim.launch_yaml_numbers import numeric_spelling

    value = "1" + ":00" * 500 + ".0"
    assert numeric_spelling(value) is None
    text = "API_KEY: " + value + "\nTOKEN: " + NUMERIC_PLACEHOLDER + "\n"
    assert NUMERIC_PLACEHOLDER in prescan_document_secrets(text, "yaml")
