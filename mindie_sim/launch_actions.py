"""Literal launch action tables, occurrence records and redaction.

[Role]
  Own the version-pinned action vocabulary, occurrence/draft records, exact
  source-span attribution and provenance-carrying secret scrubbing used by
  `launch_bindings`.

[Invariants]
  * Action kinds and value domains are explicit; aliases and negation pairs
    share one semantic key.
  * Occurrence IDs are source-order identities, never renumbered by bucket.
  * Secret provenance travels from a scoped assignment through expansion;
    every user value/key, id, pointer and diagnostic is scrubbed; fixed Case
    dataclass field names remain schema, even for short credential values.
  * JSON leaf spans are derived from the original word offset, not fabricated
    line/column indices.

[Timeouts & Locks]
  Bounded in-memory table lookup and string scans; no locks.

[Side-Effect Policy]
  No I/O, environment lookup, subprocess or eval.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from copy import deepcopy
from functools import lru_cache
import json
import re

from .compose import compose_case
from .launch_documents import DocumentInputError, finite_json_loads
from .launch_json_paths import json_member_paths, json_pointer_join
from .launch_parser import SourceSpan, Word, is_secret_name
from .schema.case import Case, Software
from .schema.model import WorkloadSpec
from .schema.parallel import ParallelSpec
from .catalog.types import QuantSpec

#: Report vocabulary fields are contract enums, not user payload.  Scrubbing
#: a very short credential out of them would invalidate the report itself.
_PROTECTED_REPORT_FIELDS = frozenset({
    "schema_version", "status", "input_format", "coverage", "effect", "kind",
    "severity", "source_kind", "statistic", "scope", "adapter_id",
})

_DELETE = object()
_VALID_BLOBS: set[str] = set()
_CLEAN_BLOBS: dict[str, tuple[str, str]] = {}

_TOKEN = re.compile(r"(?i)(\"?(?:api[_-]?key|token|secret|password|credential|authorization|bearer)\"?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)")
_NUMBER = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

_SCALAR = {
    "--tensor-parallel-size": ("/runtime/parallel/tp", int), "-tp": ("/runtime/parallel/tp", int),
    "--data-parallel-size": ("/runtime/parallel/dp", int), "-dp": ("/runtime/parallel/dp", int),
    "--pipeline-parallel-size": ("/runtime/parallel/pp", int), "-pp": ("/runtime/parallel/pp", int),
    "--decode-context-parallel-size": ("/runtime/parallel/cp", int),
    "--context-parallel-size": ("/runtime/parallel/cp", int),
    "--prefill-context-parallel-size": ("/runtime/parallel/pcp", int),
    "--max-model-len": ("/runtime/max_model_len", int),
    "--max-num-batched-tokens": ("/runtime/max_num_batched_tokens", int),
    "--max-num-seqs": ("/runtime/max_num_seqs", int),
    "--block-size": ("/runtime/kv_block_size", int),
    "--gpu-memory-utilization": ("/runtime/gpu_memory_utilization", float),
    "--seed": ("/runtime/metadata/random_seed", int),
}
_AXIS = {"--tensor-parallel-size": "tp", "-tp": "tp",
         "--data-parallel-size": "dp", "-dp": "dp",
         "--pipeline-parallel-size": "pp", "-pp": "pp",
         "--decode-context-parallel-size": "cp", "--context-parallel-size": "cp",
         "--prefill-context-parallel-size": "pcp"}
_BOOL = {
    "--enable-chunked-prefill": ("chunked_prefill", True, "/runtime/enable_chunked_prefill"),
    "--no-enable-chunked-prefill": ("chunked_prefill", False, "/runtime/enable_chunked_prefill"),
    "--enable-prefix-caching": ("prefix_caching", True, None),
    "--no-enable-prefix-caching": ("prefix_caching", False, None),
    "--async-scheduling": ("async_scheduling", True, "/runtime/async_scheduling"),
    "--enable-expert-parallel": ("expert_parallel", True, None),
}
_OPERATIONAL = {
    "--host", "--port", "--served-model-name", "--trust-remote-code", "--model",
    "--disable-log-requests", "--disable-log-stats", "--disable-fastapi-docs",
    "--enforce-eager", "--tokenizer", "--tokenizer-mode", "--revision",
}
_JSON_FLAGS = {"--compilation-config", "--additional-config",
               "--speculative-config", "--kv-transfer-config"}
_CAPTURE_DEFAULTS = {
    "vllm-ascend/0.18-static": (1, 2, 4, 8, 16, 32, 64, 128, 256),
    "vllm-ascend/0.30-static": (1, 2, 4, 8, 16, 32, 64, 128, 256),
}
_VERSION_SENSITIVE_ENV = frozenset({"VLLM_ASCEND_ENABLE_MLAPO",
                                    "VLLM_ASCEND_ENABLE_FLASHCOMM1"})
_METHODS = {"deepseek_mtp": "mtp", "mtp": "mtp", "eagle": "eagle",
            "eagle3": "eagle3", "ngram": "ngram", "dspark": "dspark"}
_ALL2ALL = {"mc2": ("mc2", False), "deepep": ("alltoall", False),
            "naive": ("alltoall", False), "alltoall": ("alltoall", False),
            "deepep_high_throughput": ("alltoall", True)}


class BindingFailure(ValueError):
    """Malformed or contradictory declaration; becomes a `failed` report."""


class UnsupportedModeling(Exception):
    def __init__(self, rule, reason):
        self.rule, self.reason = rule, reason
        super().__init__(reason)


@dataclass(frozen=True)
class Adapter:
    id: str
    version: str
    mlaapo_env: bool
    mlaapo_config: bool
    capture_defaults: tuple[int, ...]


@dataclass(frozen=True)
class Occurrence:
    uid: str
    kind: str
    name: str
    raw: object
    span: SourceSpan
    priority: int
    value_span: SourceSpan | None = None
    secret: bool = False
    assignment: EnvAssignment | None = None
    raw_text: str | None = None
    json_escaped: bool = False
    #: Decoded ``raw_text`` -> exact outer source interval map.
    char_coords: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class Draft:
    uid: str
    key: str
    kind: str
    name: str
    pointer: str | None
    value: object
    redacted: bool
    effect: str
    targets: tuple[str, ...]
    rule_id: str | None
    reason: str
    bucket: str
    span: SourceSpan
    patches: tuple[tuple[str, object], ...] = ()
    parent_uid: str | None = None
    priority: int = 0
    effective: bool = True
    shadowed_by: str | None = None


def adapter_for(version):
    """Pin semantics to an explicitly supported release; future versions fail."""
    if not version:
        return None
    text = str(version).lower()
    match = re.fullmatch(r"\s*(?:vllm-ascend@|vllm@)?v?(\d+)\.(\d+)(?:\.(\d+))?\s*", text)
    if not match:
        return None
    key = (int(match.group(1)), int(match.group(2)))
    if key == (0, 18):
        return Adapter("vllm-ascend/0.18-static", "0.18", True, False,
                       _CAPTURE_DEFAULTS["vllm-ascend/0.18-static"])
    if key == (0, 30):
        return Adapter("vllm-ascend/0.30-static", "0.30", False, True,
                       _CAPTURE_DEFAULTS["vllm-ascend/0.30-static"])
    return None


def _ordered_secrets(secrets) -> tuple[str, ...]:
    if not secrets:
        return ()
    return tuple(sorted({str(item) for item in secrets if item}, key=len, reverse=True))


def _scrub_text_ordered(text, ordered: tuple[str, ...]) -> str:
    value = str(text)
    for secret in ordered:
        if secret in value:
            value = value.replace(secret, "[REDACTED]")
    if ":" not in value and "=" not in value:
        return value
    return _TOKEN.sub(lambda match: match.group(1) + "[REDACTED]", value)


def _scrub_text(text, secrets=()):
    return _scrub_text_ordered(text, _ordered_secrets(secrets))


def _scrub_ordered(value, ordered: tuple[str, ...], exact: frozenset[str]):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            # The key itself is a diagnostic surface: a secret-named key may
            # embed the credential value (``R4_CANARY_api_key``), so scrub the
            # key text before deciding that its value is credential material.
            scrubbed_key = _scrub_text_ordered(str(key), ordered)
            if is_secret_name(key):
                result[scrubbed_key] = "[REDACTED]"
            else:
                result[scrubbed_key] = _scrub_ordered(item, ordered, exact)
        return result
    if isinstance(value, (list, tuple)):
        return [_scrub_ordered(item, ordered, exact) for item in value]
    if isinstance(value, str):
        if value in exact:
            return "[REDACTED]"
        return _scrub_text_ordered(value, ordered)
    return value


def _scrub(value, secrets=()):
    ordered = _ordered_secrets(secrets)
    return _scrub_ordered(value, ordered, frozenset(ordered))


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_strings(item)


def _scrub_case_uncached(value, secrets=()):
    case = Case.from_mapping(value)

    def walk(data, record):
        if isinstance(data, dict) and is_dataclass(record):
            aliases = {"devices": "dies", "architecture": "arch_family"}
            # These validated schema enums describe policy, not credential
            # payload (e.g. a YAML secret `off` colliding with a runtime mode).
            enums = {"int8_weight_metadata", "runtime_int8_weight_metadata"}
            modes = {"collective_startup_mode", "matrix_compute_policy",
                     "prefix_cache_mode", "prefix_read_mode",
                     "homogeneous_graph_reuse", "moe_recipe", "graph_mode",
                     "graph_capture_mode", "final_norm_scope", "cp_plan",
                     "pcp_lane_policy"}
            return {key: (item if (isinstance(record, QuantSpec) and key in enums
                                  and item in {"", "scale_only", "scale_offset"})
                          or (isinstance(record, (Software, ParallelSpec))
                              and key in modes)
                          else walk(item, getattr(record, aliases.get(key, key), None)))
                    for key, item in data.items()}
        if isinstance(data, (list, tuple)) and isinstance(record, (list, tuple)):
            return [walk(item, record[index] if index < len(record) else None)
                    for index, item in enumerate(data)]
        return _scrub(data, secrets)

    return walk(case.to_dict(), case)


def _scrub_case(value, secrets=()):
    """Scrub a canonical Case using its typed records as the schema boundary.

    Dataclass projections own their keys. Free-form mappings (metadata,
    kernels, catalog facts, etc.) still scrub both keys and values. Hardware's
    serialized devices/architecture names are aliases of dies/arch_family.
    """
    if isinstance(value, dict):
        try:
            blob = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            blob = None
        entry = _CLEAN_BLOBS.get(blob) if blob is not None else None
        if entry is not None:
            haystack, scrubbed_blob = entry
            if not any(str(s) in haystack for s in secrets if s):
                return json.loads(scrubbed_blob)
    return _scrub_case_uncached(value, secrets)


def _scrub_report(report, secrets=()):
    """Recursively scrub every report string, dictionary key and payload.

    Secret values can arrive in a shell-local assignment name, a JSON key, an
    occurrence id, a pointer or free-form Case metadata. Fixed record keys
    remain schema; the constructed report is projected before validation.
    """
    ordered = _ordered_secrets(secrets)
    if not ordered:
        return report

    def walk(value):
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{
                field.name: (getattr(value, field.name)
                             if field.name in _PROTECTED_REPORT_FIELDS
                             else _scrub_case(value.case, ordered)
                             if field.name == "case" and value.case is not None
                             else walk(getattr(value, field.name)))
                for field in fields(value)})
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                # Keys (setting ids, child names and JSON pointers derived
                # from them) are payload, not labels: redact the credential
                # substring even when the key also names a secret setting.
                scrubbed_key = _scrub_text_ordered(str(key), ordered)
                if is_secret_name(key):
                    result[scrubbed_key] = "[REDACTED]"
                else:
                    result[scrubbed_key] = walk(item)
            return result
        if isinstance(value, (list, tuple)):
            return type(value)(walk(item) for item in value)
        if isinstance(value, str):
            return _scrub_text_ordered(value, ordered)
        return value

    return walk(report)


def _display(occurrence, secrets):
    raw = occurrence.raw
    if isinstance(raw, str) and raw[:1] in "{[":
        try:
            raw = finite_json_loads(raw, occurrence.span.source)
        except DocumentInputError:
            pass
    return _scrub(raw, secrets)


def _value_span(word: Word, name: str):
    """Span of an attached ``--flag=value`` value inside the raw token.

    The first value character is taken from the word's exact ``char_coords``
    entry after ``=``; the closing boundary is the word's composed end.  A
    word without a source map falls back to the whole token rather than
    inventing a column.
    """
    equals = (word.text or "").find("=")
    if equals >= 0 and word.char_coords and equals + 1 < len(word.char_coords):
        start = word.char_coords[equals + 1]
        coords = word.char_coords
        last = coords[-2] if len(coords) >= 2 else coords[-1]
        return SourceSpan(word.span.source, start[0], start[1],
                          last[3], last[4])
    return word.span


def _attached_raw(word: Word):
    """Raw value slice after the first ``=``; ``None`` when not attached."""
    raw = word.raw_text or ""
    equals = raw.find("=")
    return raw[equals + 1:] if equals >= 0 else None


def _looks_flag(text):
    return isinstance(text, str) and text.startswith("-") and not _NUMBER.fullmatch(text)


def _prefix(argv):
    if argv[:1] == ["vllm"]:
        return 2 if len(argv) > 2 and argv[2].startswith("-") else 3
    if argv[:1] in (["python"], ["python3"]):
        return 3
    return 0


def _occurrences(document):
    result = []
    for index, record in enumerate(document.env_occurrences):
        result.append(Occurrence(f"e{index + 1}", "env", record.name, record.value,
                                 record.span, 1, record.span,
                                 record.secret, record))
    argv, words = list(document.argv), list(document.words)
    aliases = {"-tp": "--tensor-parallel-size", "-dp": "--data-parallel-size",
               "-pp": "--pipeline-parallel-size"}
    index = _prefix(argv)
    while index < len(argv):
        token = argv[index]
        word = words[index] if index < len(words) else Word(
            token, document.command_span or SourceSpan(document.source, 1, 1, 1, 1),
            True, 0, False, (), token)
        original, equals, inline = token.partition("=")
        name = aliases.get(original, original)
        value_word, raw_text = word, None
        json_escaped = False
        char_coords = ()
        def attached_coords():
            split = word.text.find("=")
            return (word.char_coords[split + 1:]
                    if split >= 0 and word.char_coords else ())
        if name in _BOOL:
            if equals:
                raw = inline
                value_span = _value_span(word, original)
                raw_text = _attached_raw(word)
                json_escaped = word.json_escaped
                char_coords = attached_coords()
            else:
                raw, value_span = _BOOL[name][1], word.span
        elif name in _JSON_FLAGS:
            if equals:
                raw, value_span = inline, _value_span(word, original)
                raw_text = _attached_raw(word)
                json_escaped = word.json_escaped
                char_coords = attached_coords()
            elif index + 1 < len(argv):
                index += 1
                value_word = words[index]
                raw, value_span = argv[index], value_word.span
                raw_text = value_word.raw_text
                json_escaped = value_word.json_escaped
                char_coords = value_word.char_coords
            else:
                raw, value_span = "", word.span
        elif equals:
            raw, value_span = inline, _value_span(word, original)
            raw_text = _attached_raw(word)
            json_escaped = word.json_escaped
            char_coords = attached_coords()
        elif index + 1 < len(argv) and not _looks_flag(argv[index + 1]):
            index += 1
            value_word = words[index]
            raw, value_span = argv[index], value_word.span
            raw_text = value_word.raw_text
            json_escaped = value_word.json_escaped
            char_coords = value_word.char_coords
        else:
            raw, value_span = True, word.span
        result.append(Occurrence(f"o{index + 1}", "cli", name, raw, word.span, 2,
                                 value_span, value_word.secret, None, raw_text,
                                 json_escaped, char_coords))
        index += 1
    for index, wrapper in enumerate(document.operational):
        result.append(Occurrence(f"w{index + 1}", "wrapper",
                                 wrapper.text.split("=", 1)[0], wrapper.text,
                                 wrapper.span, 1, wrapper.span))
    return tuple(result)


def _axis(occurrences):
    axes, failures = {}, []
    for item in occurrences:
        if item.kind != "cli" or item.name not in _AXIS:
            continue
        if item.raw is True or item.raw == "":
            failures.append((item, "parallel_axis_value_required"))
            continue
        try:
            value = int(item.raw)
        except (TypeError, ValueError):
            failures.append((item, "non_integer_parallel_axis"))
            continue
        if value < 1:
            failures.append((item, "parallel_axis_must_be_positive"))
            continue
        axes[_AXIS[item.name]] = value
    return axes, failures


def _leaf_span(item, key, json_path, paths):
    """Exact leaf span bound to the member at this leaf's own JSON path."""
    decoded = item.raw if isinstance(item.raw, str) else str(item.raw)
    coords = getattr(item, "char_coords", ())
    if not coords or len(coords) != len(decoded) + 1:
        raise BindingFailure("source_coordinates_unavailable")
    records = paths.get(json_path)
    if not records:
        raise BindingFailure("json_leaf_provenance_unavailable")
    key_start, key_end, _value_start, _value_end = records[0]
    start = coords[key_start]
    if key_end > key_start:
        # The half-open interval closes at the *last key character's* end.
        # coords[key_end] is the next decoded token's start, which may sit
        # after a removed YAML line continuation or an adjacent shell quote.
        last = coords[key_end - 1]
        end_line, end_column = last[3], last[4]
    else:
        boundary = coords[key_end]
        end_line, end_column = boundary[0], boundary[1]
    return SourceSpan(item.span.source, start[0], start[1],
                      end_line, end_column)


def _child(item, key, value, pointer, target, patches, secrets, paths=None,
           json_path=None):
    redacted = is_secret_name(key)
    display = "[REDACTED]" if redacted else _scrub(value, secrets)
    if paths is None:
        decoded = item.raw if isinstance(item.raw, str) else str(item.raw)
        paths = json_member_paths(decoded)
    return Draft(f"{item.uid}/c:{key}", f"{item.uid}:{pointer}/{key}", "json_leaf", key,
                 f"{pointer}/{key}", display, redacted,
                 "modeled" if target else "unmodeled", (target,) if target else (),
                 "json_leaf", "Existing consumer." if target else
                 "No physical consumer in this scope.",
                 "applied" if target else "unmodeled_or_unsupported",
                 _leaf_span(item, key, json_path
                            or json_pointer_join("", key), paths),
                 tuple(patches), item.uid, item.priority)


def _draft(item, bucket, rule, *, patches=(), targets=(), reason="", value=None,
           redacted=False, effect=None, pointer=None, secrets=(), key=None):
    raw = item.raw if value is None else value
    if isinstance(raw, str) and value is None:
        raw = _display(item, secrets)
    elif value is None:
        raw = _scrub(item.raw, secrets)
    redacted = redacted or item.secret or is_secret_name(item.name)
    if isinstance(raw, str) and raw == "[REDACTED]":
        redacted = True
    if redacted:
        raw = "[REDACTED]"
    return Draft(item.uid, key or item.name, item.kind, item.name, pointer, raw, redacted,
                 effect or ("modeled" if bucket == "applied" else
                            "unmodeled" if bucket == "unmodeled" else "operational"),
                 tuple(targets or (path for path, _ in patches)), rule,
                 reason or ("Existing selected-Case consumer." if bucket == "applied"
                            else "No physical consumer in this scope."),
                 {"operational": "neutral_operational", "unmodeled": "unmodeled_or_unsupported"}
                 .get(bucket, bucket), item.span, tuple(patches),
                 None, item.priority)


def _shadow(drafts):
    by_uid = {draft.uid: draft for draft in drafts}
    winners = {}
    for index, draft in enumerate(drafts):
        if draft.parent_uid is not None:
            continue
        current = winners.get(draft.key)
        if current is None or (draft.priority, index) >= (
                drafts[current].priority, current):
            winners[draft.key] = index
    parents = {}
    for index, draft in enumerate(drafts):
        if draft.parent_uid is not None:
            continue
        winner = winners.get(draft.key)
        if winner is not None and winner != index:
            parents[draft.uid] = replace(
                draft, effect="shadowed", effective=False,
                shadowed_by=drafts[winner].uid, patches=())
        else:
            parents[draft.uid] = draft
    result = []
    for draft in drafts:
        if draft.parent_uid is not None:
            parent = parents.get(draft.parent_uid)
            if parent is not None and not parent.effective:
                result.append(replace(
                    draft, effect="shadowed", effective=False,
                    shadowed_by=parent.shadowed_by or parent.uid, patches=()))
                continue
            result.append(draft)
            continue
        result.append(parents[draft.uid])
    return result

def _register_clean_case(canonical: dict) -> str:
    blob = json.dumps(canonical, allow_nan=False)
    _VALID_BLOBS.add(blob)
    scrubbed = _scrub_case_uncached(canonical, ())
    scrubbed_blob = json.dumps(scrubbed, allow_nan=False)
    try:
        Case.from_mapping(scrubbed)
        _VALID_BLOBS.add(scrubbed_blob)
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    haystack = "\x00".join((*_iter_strings(canonical), *_iter_strings(scrubbed)))
    _CLEAN_BLOBS[blob] = (haystack, scrubbed_blob)
    return blob


@lru_cache(maxsize=None)
def _cached_mutated_blob(base_blob: str, patches_json: str) -> str:
    base = json.loads(base_blob)
    for path, is_del, value in json.loads(patches_json):
        _set_path(base, path, _DELETE if is_del else value)
    return _register_clean_case(Case.from_mapping(base).to_dict())


@lru_cache(maxsize=None)
def _cached_composed_bundle(machine: str, model: str, tp: int, dp: int, ep: int,
                            pp: int, pcp: int, cp: int, workload_json: str | None):
    wl = None if workload_json is None else json.loads(workload_json)
    axes = {"tp": tp, "dp": dp, "ep": ep, "pp": pp, "pcp": pcp, "cp": cp}
    base = _base(machine, model, wl, axes, None)
    parsed = Case.from_mapping(base)
    blob = _register_clean_case(parsed.to_dict())
    return base, parsed, blob


def _base_bundle(machine, model, workload, axes, base_case):
    if base_case is None and isinstance(machine, str) and isinstance(model, str):
        wl_raw = workload.to_dict() if hasattr(workload, "to_dict") else workload
        try:
            wl_json = (None if wl_raw is None
                       else json.dumps(dict(wl_raw), sort_keys=True, allow_nan=False))
            cacheable = True
        except (TypeError, ValueError):
            wl_json, cacheable = None, False
        if cacheable:
            return _cached_composed_bundle(
                machine, model, int(axes.get("tp", 1)), int(axes.get("dp", 1)),
                int(axes.get("ep", 1)), int(axes.get("pp", 1)),
                int(axes.get("pcp", 1)), int(axes.get("cp", 1)), wl_json)
    base = _base(machine, model, workload, axes, base_case)
    return base, Case.from_mapping(base), None


def _base(machine, model, workload, axes, base_case):
    if base_case is not None:
        base = deepcopy(base_case.to_dict() if hasattr(base_case, "to_dict") else dict(base_case))
        for key, value in axes.items():
            _set_path(base, f"/runtime/parallel/{key}", value)
            if key in {"cp", "dcp"}:
                _set_path(base, "/runtime/parallel/cp", value)
                _set_path(base, "/runtime/parallel/dcp", value)
        if workload is not None:
            raw = workload.to_dict() if hasattr(workload, "to_dict") else dict(workload)
            base["workload"] = (raw if "requests" in raw
                                else WorkloadSpec.from_mapping(raw).to_dict())
            WorkloadSpec.from_mapping(base["workload"])
        return base
    base = compose_case(model=model, hardware=machine, tp=axes.get("tp", 1),
                        dp=axes.get("dp", 1), ep=axes.get("ep", 1),
                        pp=axes.get("pp", 1), pcp=axes.get("pcp", 1),
                        dcp=axes.get("cp", 1), graph_capture_mode="none")
    if workload is not None:
        raw = workload.to_dict() if hasattr(workload, "to_dict") else dict(workload)
        base["workload"] = raw if "requests" in raw else WorkloadSpec.from_mapping(raw).to_dict()
        WorkloadSpec.from_mapping(base["workload"])
    return base


def _set_path(target, path, value):
    parts = [part for part in path.strip("/").split("/") if part]
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    before = cursor.get(parts[-1])
    if value is _DELETE:
        cursor.pop(parts[-1], None)
    else:
        cursor[parts[-1]] = deepcopy(value)
    return None if before is None else before
