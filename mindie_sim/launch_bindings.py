"""Version-pinned vLLM/vLLM-Ascend launch bindings and coverage classification.

[Role]
  Translate literal launch declarations into existing Case fields and
  lowering/metadata consumers.  Every CLI/env occurrence and JSON child lands
  in exactly one of `applied`, `neutral_operational` or
  `unmodeled_or_unsupported`; nothing here executes a deployment.

[Invariants]
  * Adapter semantics are pinned to an explicitly supported release.  Unknown,
    absent or future versions keep version-sensitive settings unresolved.
  * TP/DP/PP and DCP/PCP land on `ParallelSpec`; explicit `/runtime/parallel/*`
    simulator overrides are applied before final `EP = dp*tp*pcp` derivation
    and before feature predicates, and EP is never a boolean 2.
  * Environment occurrences are taken only from the selected command's
    exported/inline shell scope.  Explicit `env=` is outer host scope; a Docker
    container starts empty and receives only forwarded `-e`/`--env-file`.
  * Boolean actions have explicit domains; a shadowed JSON parent shadows every
    child.  Supported patches apply independently of unrelated unsupported
    declarations; only `strict=True` rejects the whole command.
  * Secret provenance travels from a scoped assignment, explicit `env=`, Docker
    forward, override path or diagnostic word through variable expansion.
    Every bucket, nested payload, override, diagnostic and the final Case is
    scrubbed; a credential value never reaches serialization.
  * `applied` requires a real Case consumer; unmodeled is never operational.

[Timeouts & Locks]
  Bounded in-memory parsing; no locks.
[Side-Effect Policy]
  No I/O, subprocesses, host env lookup, network or eval.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import wraps
import hashlib
import json

from . import feature_reports as _reports
from .compose import compose_case
from .feature_reports import (
    Diagnostic, LaunchParseReport, Patch, RequiredInput, Setting, SourceIdentity,
    Span, validate_launch_report as _validate_launch_schema,
)
from .launch_documents import DocumentInputError, parse_document
from .launch_parser import LaunchSyntaxError, SourceSpan, is_secret_name
from .launch_spans import prescan_cli_secret_values
from .launch_actions import (
    Adapter, BindingFailure, Draft, Occurrence, UnsupportedModeling,
    _DELETE, _VALID_BLOBS, _VERSION_SENSITIVE_ENV, _axis, _base_bundle,
    _cached_mutated_blob, _draft, _occurrences, _scrub, _scrub_report,
    _scrub_text, _set_path, _shadow, adapter_for,
)
from .launch_overrides import _bool_value, _eager_conflict, _resolve
from .schema.case import Case
from .schema.common import ModelingError
from .schema.model import WorkloadSpec


@wraps(_validate_launch_schema)
def validate_launch_report(report):
    """Extend the frozen report contract with Case re-materialization."""
    errors = _validate_launch_schema(report)
    status = report.get("status") if isinstance(report, dict) else report.status
    case = report.get("case") if isinstance(report, dict) else report.case
    if status == "completed" and case is not None:
        known = False
        if isinstance(case, dict):
            try:
                known = json.dumps(case, allow_nan=False) in _VALID_BLOBS
            except (TypeError, ValueError):
                known = False
        if not known:
            try:
                Case.from_mapping(case)
            except (ValueError, TypeError, KeyError, AttributeError):
                raise _reports.ReportValidationError(("L02:case_round_trip_invalid",)) from None
    return errors


# Package initialization imports this adapter before exposing report APIs.
# Register the extension here while keeping feature_reports.py byte-identical.
_reports.validate_launch_report = validate_launch_report


def _parallel_overrides(overrides):
    """Explicit simulator axes that precede EP derivation and predicates."""
    axes = {}
    items = overrides.items() if isinstance(overrides, dict) else (overrides or ())
    for path, value in items:
        parts = [part for part in str(path).strip("/").split("/") if part]
        if len(parts) == 3 and parts[:2] == ["runtime", "parallel"]:
            name = parts[2]
            if name in {"tp", "dp", "pp", "cp", "dcp", "pcp", "ep"}:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number >= 1:
                    axes[name] = number
    return axes


def _apply(base, drafts):
    for draft in drafts:
        if not draft.effective:
            continue
        for path, value in draft.patches:
            _set_path(base, path, value)


def _report(status, *, source, source_kind, source_text, input_format, framework_version,
            adapter, strict, drafts=(), required=(), overrides=(), assumptions=(),
            case=None, coverage, diagnostics=(), command_index=0, secrets=()):
    settings = {name: [] for name in ("applied", "neutral_operational",
                                      "unmodeled_or_unsupported")}
    for draft in drafts:
        span = draft.span if isinstance(draft.span, SourceSpan) else SourceSpan(
            source, 1, 1, 1, 1)
        failed_path = status == "failed"
        settings[draft.bucket].append(Setting(
            draft.uid, draft.parent_uid, draft.kind, draft.name, draft.pointer,
            "[REDACTED]" if failed_path else _scrub(draft.value, secrets),
            draft.redacted or failed_path,
            Span(source, span.line_start, span.column_start, span.line_end, span.column_end),
            draft.effective, draft.shadowed_by, draft.effect, draft.targets,
            draft.rule_id, _scrub_text(draft.reason, secrets)))
    identity = SourceIdentity(source, None,
                              hashlib.sha256((source_text or "").encode()).hexdigest(),
                              source_kind)
    report = LaunchParseReport(
        "launch-parse/1", status, identity, input_format, command_index,
        framework_version, adapter.id if adapter else None, strict,
        tuple(draft.uid for draft in drafts), tuple(settings["applied"]),
        tuple(settings["neutral_operational"]),
        tuple(settings["unmodeled_or_unsupported"]), tuple(required),
        tuple(overrides), tuple(assumptions), case, coverage, tuple(diagnostics))
    report = _scrub_report(report, secrets)
    validate_launch_report(report)
    return report


def _required(name, reason, metrics=("ttft", "tpot", "throughput")):
    return RequiredInput(name, reason, tuple(metrics))


def _missing_inputs(machine, model, workload, base_case):
    return [] if base_case is not None else [
        _required(name, "Launch text does not declare this input.",
                  ("structure",) if name in {"machine", "model"} else ("ttft", "tpot", "throughput"))
        for name, value in (("machine", machine), ("model", model),
                            ("workload", workload)) if value is None]


def _merge_required(required, names):
    """Append independent names once, preserving the already-known order."""
    seen = {item.name for item in required}
    for name in names or ():
        if name and name not in seen:
            required.append(_required(
                name,
                "Container environment contents are required; no host lookup.",
                ("structure",)))
            seen.add(name)
    return required


def parse_launch(text_or_argv, *, format="shell", command_index=None,
                 framework_version=None, env=None, machine=None, model=None,
                 workload=None, speculative=None, base_case=None, overrides=None,
                 strict=False, container=None, env_files=None, source=None):
    """Parse one launch declaration into a validated Case and coverage report."""
    text = text_or_argv if isinstance(text_or_argv, str) else None
    source_name = source or ("<argv>" if format == "argv" else "<launch>")
    source_kind = "argv" if format == "argv" else "user"
    if source and source not in {"<launch>", "<argv>", "-"}:
        source_kind = "file"
    elif source == "-":
        source_kind = "stdin"
    source_text = text if isinstance(text, str) else json.dumps(text_or_argv or ())
    adapter = adapter_for(framework_version)
    explicit_env = {str(name): str(value) for name, value in dict(env or {}).items()}
    secrets = {value for name, value in explicit_env.items() if is_secret_name(name)}
    drafts, diagnostics = [], []
    required = _missing_inputs(machine, model, workload, base_case)
    try:
        document = parse_document(text_or_argv, format=format, source=source_name,
                                  env=env, container=container,
                                  command_index=command_index, env_files=env_files)
        if command_index not in (None, document.command_index):
            raise BindingFailure("command_index_out_of_range")
        occurrences = _occurrences(document)
        # A literal credential argument taints the whole report, not only its
        # own row: ``--api-key VALUE`` must also protect an unrelated row that
        # repeats VALUE.  JSON-flag credentials are scanned at the same time.
        for item in occurrences:
            if item.kind == "cli":
                secrets.update(prescan_cli_secret_values(item.name, item.raw))
        # Accumulate Docker/shell/machine/model/workload requirements before any
        # scalar/EP prepass can fail: a known ``-e MISSING`` must survive an
        # independent malformed boolean in the same command.
        missing_env = tuple(document.missing_env) + tuple(document.missing_env_files)
        for name in missing_env:
            required.append(_required(
                f"env:{name}",
                "Container environment contents are required; no host lookup.",
                ("structure",)))
        # ``env=`` is the outer host environment.  A Docker run starts with an
        # empty container environment, so a host value becomes an effective
        # setting only when explicitly forwarded by ``-e`` or ``--env-file``.
        container_isolated = document.image is not None
        forwarded = set(document.forwarded_env)
        container_env_names = {name for name, _value in document.env}
        for index, (name, value) in enumerate(sorted(explicit_env.items())):
            if container_isolated:
                if name not in forwarded:
                    continue
                if name in container_env_names:
                    # The forwarded ``-e``/``--env-file`` record already owns
                    # the container occurrence; do not duplicate the setting.
                    continue
            span = SourceSpan(source_name, 1, 1, 1, 1)
            occurrences += (Occurrence(
                f"x{index + 1}", "env", name, value, span, 0, span,
                is_secret_name(name), None, value),)
        for record in (*document.env_occurrences, *document.outer_env_occurrences):
            if record.secret:
                secrets.add(record.value)
        secrets.update(document.secret_values)
        secrets |= {value for name, value in document.env if is_secret_name(name)}
        secrets |= {item.raw for item in occurrences
                    if item.secret and isinstance(item.raw, str)}
        axes, axis_failures = _axis(occurrences)
        ep_occurrence = next((item for item in reversed(occurrences)
                              if item.name == "--enable-expert-parallel"), None)
        if base_case is not None:
            inherited = dict((base_case.to_dict() if hasattr(base_case, "to_dict")
                              else base_case).get("runtime", {}).get("parallel", {}))
        else:
            inherited = {"tp": 1, "dp": 1, "pp": 1, "cp": 1, "pcp": 1, "ep": 1}
        override_axes = _parallel_overrides(overrides)
        merged = {**inherited, **axes, **override_axes}
        if ep_occurrence is not None:
            if _bool_value(ep_occurrence, True):
                if merged.get("pp", 1) > 1:
                    merged["ep"] = 1
                else:
                    merged["ep"] = (merged.get("dp", 1) * merged.get("tp", 1)
                                    * merged.get("pcp", 1))
                    if int(merged.get("pcp", 1)) > 1 and not merged.get("rank_table"):
                        merged["rank_table"] = [list(range(int(merged["ep"])))]
            else:
                merged["ep"] = 1
        elif "ep" not in merged:
            merged["ep"] = 1
        diagnostics.extend(Diagnostic(message, "error",
                                      "Malformed parallel axis was not applied.",
                                      item.uid, ()) for item, message in axis_failures)
        malformed = [item.uid for item, _ in axis_failures]
        for item, _ in axis_failures:
            drafts.append(_draft(item, "unmodeled", "malformed_declaration",
                                 reason="Malformed parallel axis was not applied.",
                                 value="[REDACTED]", redacted=True, secrets=secrets))
        base = parsed = base_error = cached_blob = None
        if base_case is not None or (machine is not None and model is not None):
            try:
                base, parsed, cached_blob = _base_bundle(
                    machine, model, workload, merged, base_case)
            except (ModelingError, ValueError) as exc:
                message = _scrub_text(str(exc).splitlines()[0], secrets)[:200]
                base_error = Diagnostic(
                    f"invalid_base_case:{getattr(exc, 'args', ('',))[0]}",
                    "error", message, None, ())
                diagnostics.append(base_error)
                base = parsed = cached_blob = None
            if parsed is not None and (document.device_visibility_declared
                                       or document.unidentified_devices):
                span = document.command_span or SourceSpan(source_name, 1, 1, 1, 1)
                required_local = int(merged.get("tp", 1)) * int(merged.get("pp", 1)) \
                    * int(merged.get("pcp", 1))
                if len(document.visible_devices) < required_local:
                    occurrences = occurrences + (Occurrence(
                        "vcount", "visibility", "visible_device_count",
                        f"required {required_local} local NPU(s), visible "
                        f"{len(document.visible_devices)}", span, 1, span),)
                occurrences = occurrences + tuple(
                    Occurrence(f"u{index + 1}", "visibility",
                               "visible_device_unidentified", device, span, 1, span)
                    for index, device in enumerate(document.unidentified_devices))
                ranks = {die.rank for die in parsed.machine.dies}
                unverified = [device for device in document.visible_devices
                              if int(device) not in ranks]
                if unverified:
                    occurrences = occurrences + tuple(
                        Occurrence(f"v{index + 1}", "visibility", "visible_device",
                                   device, span, 1) for index, device in enumerate(unverified))
        context = {"adapter": adapter, "base": base, "parsed": parsed,
                   "axes": merged, "speculative": speculative}
        version_sensitive = any(
            item.name in {"--compilation-config", "--additional-config"} or
            (item.kind == "env" and item.name in _VERSION_SENSITIVE_ENV)
            for item in occurrences)
        if version_sensitive and adapter is None:
            required.append(_required(
                "framework_version",
                "Supported exact release is required for version-sensitive declarations.",
                ("graph", "mlapo")))
        for item in occurrences:
            if item.uid in set(malformed):
                continue
            try:
                resolved, needed = _resolve(item, context, secrets)
            except UnsupportedModeling as exc:
                drafts.append(_draft(item, "unmodeled", exc.rule, reason=exc.reason,
                                     secrets=secrets))
                continue
            except (BindingFailure, ModelingError, ValueError) as exc:
                code = getattr(exc, "code", None) or "invalid_declaration"
                message = _scrub_text(str(exc).splitlines()[0], secrets)[:200]
                diagnostics.append(Diagnostic(code, "error", message, item.uid, ()))
                malformed.append(item.uid)
                drafts.append(_draft(item, "unmodeled", "malformed_declaration",
                                     reason="Malformed declaration was not applied.",
                                     value="[REDACTED]", redacted=True,
                                     secrets=secrets))
                continue
            drafts.extend(resolved)
            required.extend(needed)
        drafts = _shadow(drafts)
        drafts, eager_shadowed = _eager_conflict(drafts)
        if eager_shadowed:
            diagnostics.append(Diagnostic(
                "enforce_eager_overrides_compilation", "warning",
                "enforce_eager forces eager execution; the compilation capture "
                "declaration and its children are shadowed.", eager_shadowed[0], ()))
        unsupported = any(draft.bucket == "unmodeled_or_unsupported" for draft in drafts)
        if base_error is not None:
            status = "needs_inputs" if required else "failed"
            if status == "needs_inputs":
                diagnostics.append(Diagnostic(
                    "launch_required_inputs", "warning",
                    "No Case or latency was fabricated; malformed declarations "
                    "remain diagnosed.", None, ()))
            return _report(status, source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required,
                           coverage="unresolved", diagnostics=diagnostics,
                           command_index=document.command_index, secrets=secrets)
        if malformed:
            status = "needs_inputs" if missing_env else "failed"
            if status == "needs_inputs":
                diagnostics.append(Diagnostic(
                    "launch_required_inputs", "warning",
                    "No Case or latency was fabricated; malformed declarations "
                    "remain diagnosed.", None, ()))
            return _report(status, source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required,
                           coverage="unresolved", diagnostics=diagnostics,
                           command_index=document.command_index, secrets=secrets)
        if missing_env:
            return _report("needs_inputs", source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required,
                           coverage="unresolved",
                           diagnostics=(Diagnostic("launch_required_inputs", "warning",
                                                   "No Case or latency was fabricated.",
                                                   None, ()),),
                           command_index=document.command_index, secrets=secrets)
        if strict and (unsupported or required):
            return _report("failed", source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=True, drafts=drafts, required=required,
                           coverage="unresolved",
                           diagnostics=(Diagnostic(
                               "strict_rejected_unsupported", "error",
                               "Strict mode rejects unsupported or unresolved declarations.",
                               None, ()),),
                           command_index=document.command_index, secrets=secrets)
        if required:
            return _report("needs_inputs", source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required,
                           coverage="unresolved",
                           diagnostics=(Diagnostic("launch_required_inputs", "warning",
                                                   "No Case or latency was fabricated.", None, ()),),
                           command_index=document.command_index, secrets=secrets)
        if base is None:
            return _report("needs_inputs", source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=document.input_format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required or (
                               _required("machine", "Machine is required.", ("structure",)),
                               _required("model", "Model is required.", ("structure",))),
                           coverage="unresolved",
                           diagnostics=(Diagnostic("launch_required_inputs", "warning",
                                                   "No Case or latency was fabricated.", None, ()),),
                           command_index=document.command_index, secrets=secrets)
        items = list(overrides.items() if isinstance(overrides, dict) else (overrides or ()))
        has_mutations = (any(d.effective and d.patches for d in drafts)
                         or speculative is not None or bool(items))
        if cached_blob is not None and not has_mutations:
            override_records = []
            canonical = json.loads(cached_blob)
        elif cached_blob is not None and speculative is None and not items:
            override_records = []
            patch_rows = [(path, val is _DELETE, None if val is _DELETE else val)
                          for d in drafts if d.effective for path, val in d.patches]
            try:
                patches_json = json.dumps(patch_rows, allow_nan=False)
            except (TypeError, ValueError):
                patches_json = None
            if patches_json is not None:
                canonical = json.loads(_cached_mutated_blob(cached_blob, patches_json))
            else:
                base = json.loads(cached_blob)
                _apply(base, drafts)
                canonical = Case.from_mapping(base).to_dict()
        else:
            if cached_blob is not None:
                base = json.loads(cached_blob)
            _apply(base, drafts)
            if speculative is not None:
                from .schema.speculative import SpeculativeSpec
                spec = (speculative if isinstance(speculative, SpeculativeSpec)
                        else SpeculativeSpec.from_mapping(dict(speculative)))
                _set_path(base, "/runtime/metadata/speculative", asdict(spec))
                _set_path(base, "/runtime/mtp_acceptance_length",
                          spec.expected_committed_tokens())
            override_records = []
            for path, value in items:
                secret_path = any(is_secret_name(part)
                                  for part in str(path).strip("/").split("/"))
                if secret_path and isinstance(value, str):
                    secrets.add(value)
            for path, value in items:
                secret_path = any(is_secret_name(part)
                                  for part in str(path).strip("/").split("/"))
                before = _set_path(base, path, value)
                if secret_path:
                    redacted_before = redacted_after = "[REDACTED]"
                else:
                    redacted_before, redacted_after = _scrub(before, secrets), _scrub(value, secrets)
                override_records.append(Patch(path, redacted_before, redacted_after,
                                              "Explicit simulator override."))
            canonical = Case.from_mapping(base).to_dict()
        assumptions = (["container_image_tag_is_provenance_not_a_verified_runtime_version"]
                       if document.image else []) + (
            ["visible_devices_establish_visibility_only"]
            if document.visible_devices else [])
        return _report("partial" if unsupported else "completed",
                       source=source_name, source_kind=source_kind,
                       source_text=source_text, input_format=document.input_format,
                       framework_version=framework_version, adapter=adapter, strict=strict,
                       drafts=drafts, required=required, overrides=override_records,
                       assumptions=assumptions, case=canonical,
                       command_index=document.command_index,
                       coverage="conditional_modeled_subset" if unsupported
                       else "exact_declared_inputs",
                       diagnostics=tuple(diagnostics) + (
                           (Diagnostic("launch_partial_coverage", "warning",
                                       "Supported subset only; see unsupported entries.",
                                       None, ()),) if unsupported else ()),
                       secrets=secrets)
    except DocumentInputError as exc:
        secrets |= set(getattr(exc, "secrets", ()))
        _merge_required(required, getattr(exc, "required_inputs", ()))
        if exc.code in {"needs_env_value", "needs_env_file"}:
            name = exc.name or ("env_file" if exc.code == "needs_env_file" else "env")
            if name.startswith("-"):
                name = "env_file" if exc.code == "needs_env_file" else "env"
            _merge_required(required, (f"env:{name}",))
            return _report("needs_inputs", source=source_name, source_kind=source_kind,
                           source_text=source_text, input_format=format,
                           framework_version=framework_version, adapter=adapter,
                           strict=strict, drafts=drafts, required=required,
                           coverage="unresolved",
                           diagnostics=(Diagnostic(exc.code, "warning",
                                                   "No Case or latency was fabricated.",
                                                   None, ()),),
                           secrets=secrets)
        return _report("failed", source=source_name, source_kind=source_kind,
                       source_text=source_text, input_format=format,
                       framework_version=framework_version, adapter=adapter, strict=strict,
                       drafts=drafts, required=required, coverage="unresolved",
                       diagnostics=(Diagnostic(exc.code, "error",
                                               _scrub_text(exc.message, secrets), None, ()),),
                       secrets=secrets)
    except (LaunchSyntaxError, BindingFailure, ModelingError, ValueError) as exc:
        secrets |= set(getattr(exc, "secrets", ()))
        _merge_required(required, getattr(exc, "required_inputs", ()))
        message = getattr(exc, "message", None) or str(exc)
        return _report("failed", source=source_name, source_kind=source_kind,
                       source_text=source_text, input_format=format,
                       framework_version=framework_version, adapter=adapter, strict=strict,
                       drafts=drafts, required=required, coverage="unresolved",
                       diagnostics=(Diagnostic(getattr(exc, "code", "invalid_launch"),
                                               "error",
                                               _scrub_text(message, secrets)[:300],
                                               None, ()),),
                       secrets=secrets)


__all__ = ["Adapter", "BindingFailure", "adapter_for", "parse_launch"]
