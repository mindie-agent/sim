"""Analysis request geometry and replica identity rebinding.

[Role] Normalize explicit verification rows and rebind immutable analysis records.
[Invariants] Tokens are physical target rows; request identities retain subsets.
  A caller probing an arbitrary row count keeps the non-speculative control.
[Timeouts & Locks] Bounded by one active program; no locks.
[Side-Effect Policy] Pure in-memory transformations.
"""
from dataclasses import fields, is_dataclass, replace
import re

from mindie_sim.schema.workload import RequestView

NO_PROPOSALS = 0
DECODE_BASE_ROWS = 1
FIRST_PARTICIPANT = 0
UNIFORM_COUNT_CARDINALITY = 1


def analysis_view(case, request_id, tokens, context, scope, phase, **fields):
    """One analysis view whose target rows match the declared speculative width.

    The frontend performs the single ``q_len + draft_proposal_tokens``
    expansion, so the stored view keeps one base row and the configured
    proposal count.  A probe that does not match the declared verification
    width remains an ordinary non-speculative row count.
    """
    proposal = NO_PROPOSALS
    if scope == "model" and phase == "decode" and tokens > DECODE_BASE_ROWS:
        from .efficiency import _resolve_speculative_tokens
        configured, _ = _resolve_speculative_tokens(
            case, output_budget=DECODE_BASE_ROWS, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
        if configured > DECODE_BASE_ROWS and tokens == configured:
            proposal = configured - DECODE_BASE_ROWS
    return RequestView(
        request_id, tokens - proposal, context, scope == "model", phase,
        draft_proposal_tokens=proposal, **fields)


def rebind_analysis_requests(value, source_views, target_views):
    """Rename every complete program record from one replica to another.

    The immutable records carry nested request subsets, draft-row maps, ports
    and typed metadata, so a string rewrite is bounded by the exact declared
    request identities of both replicas.
    """
    binding = {source.request_id: target.request_id
               for source, target in zip(source_views, target_views, strict=True)}
    if not binding:
        return value
    pattern = re.compile(
        r"(?<![\w@])(?:" + "|".join(
            re.escape(key) for key in sorted(binding, key=len, reverse=True))
        + r")(?![\w@])")
    memo = {}

    def visit(item):
        key = id(item)
        if key not in memo:
            memo[key] = transform(item)
        return memo[key]

    def transform(item):
        if isinstance(item, str):
            return pattern.sub(lambda match: binding[match.group()], item)
        if is_dataclass(item) and not isinstance(item, type):
            updates = {}
            for field in fields(item):
                if field.init:
                    before = getattr(item, field.name)
                    after = visit(before)
                    if after is not before:
                        updates[field.name] = after
            return replace(item, **updates) if updates else item
        if isinstance(item, tuple):
            result = tuple(visit(part) for part in item)
            return item if all(a is b for a, b in zip(item, result)) else result
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            return {visit(key): visit(part) for key, part in item.items()}
        return item

    return visit(value)
