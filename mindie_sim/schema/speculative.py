"""Acceptance distributions, independent of execution cost.

[Role] Convert prefix marginals or conditional probabilities to accepted counts.
[Invariants] Commit = accepted prefix + one verified bonus token. No step-time
  constants or output-rate formulas participate in this declaration.
[Timeouts & Locks] O(proposal length), no I/O.
[Side-Effect Policy] Immutable records.
"""
from dataclasses import dataclass, fields
import math

from .common import ModelingError


@dataclass(frozen=True)
class SpeculativeSpec:
    draft_len: int
    acceptance: tuple[float, ...] = ()
    acceptance_mode: str = "marginal"
    accepted_distribution: tuple[float, ...] | None = None
    source: str = ""
    label: str = ""

    def __post_init__(self):
        if self.draft_len < 0 or self.acceptance_mode not in {"marginal", "conditional"}:
            raise ModelingError("speculative_proposal_or_acceptance_mode_invalid")
        if self.acceptance and self.accepted_distribution is not None:
            raise ModelingError("speculative_acceptance_ambiguous")
        if self.accepted_distribution is not None:
            values = self.accepted_distribution
            if len(values) != self.draft_len + 1 or abs(sum(values) - 1) > 1e-8:
                raise ModelingError("speculative_distribution_must_cover_all_accepted_counts")
        else:
            values = self.acceptance
            if len(values) != self.draft_len:
                raise ModelingError("speculative_acceptance_requires_each_proposal_position")
            if self.acceptance_mode == "marginal" and any(a < b for a, b in zip(values, values[1:])):
                raise ModelingError("prefix_acceptance_must_be_nonincreasing")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ModelingError("acceptance_probability_invalid")

    @classmethod
    def from_mapping(cls, raw):
        unknown = raw.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ModelingError("speculative_unknown_fields:" + ",".join(sorted(unknown)))
        data = dict(raw)
        for key in ("acceptance", "accepted_distribution"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def acceptance_marginals(self):
        if self.accepted_distribution is not None:
            return tuple(sum(self.accepted_distribution[index + 1:]) for index in range(self.draft_len))
        if self.acceptance_mode == "marginal":
            return self.acceptance
        result, survival = [], 1
        for probability in self.acceptance:
            survival *= probability
            result.append(survival)
        return tuple(result)

    def acceptance_distribution(self):
        if self.accepted_distribution is not None:
            return self.accepted_distribution
        survival = (1.0, *self.acceptance_marginals(), 0.0)
        return tuple(a - b for a, b in zip(survival, survival[1:]))

    def expected_committed_tokens(self):
        return 1 + sum(self.acceptance_marginals())
