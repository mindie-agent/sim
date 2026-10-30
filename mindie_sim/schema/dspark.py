"""Block drafter acceptance and confidence policy.

[Role] Choose the verified proposal prefix from declared confidence scores.
[Invariants] Structural draft work is computed by the frontend. This policy
  contains probabilities only, never measured proposal/verify durations.
[Timeouts & Locks] O(proposal length), no I/O.
[Side-Effect Policy] Immutable records.
"""
from dataclasses import dataclass, fields

from .common import ModelingError
from .speculative import SpeculativeSpec


@dataclass(frozen=True)
class DsparkSpec:
    draft_len: int
    acceptance: tuple[float, ...]
    acceptance_mode: str = "marginal"
    confidence: tuple[float, ...] = ()
    confidence_threshold: float = 0
    source: str = ""
    label: str = ""

    @classmethod
    def from_mapping(cls, raw):
        unknown = raw.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ModelingError("dspark_unknown_fields:" + ",".join(sorted(unknown)))
        data = dict(raw)
        data["acceptance"] = tuple(data["acceptance"])
        data["confidence"] = tuple(data.get("confidence", ()))
        result = cls(**data)
        result.step_spec()
        return result

    def step_spec(self):
        declared = SpeculativeSpec(self.draft_len, self.acceptance, self.acceptance_mode)
        if self.confidence and len(self.confidence) != self.draft_len:
            raise ModelingError("confidence_requires_each_proposal_position")
        if any(not 0 <= value <= 1 for value in (*self.confidence, self.confidence_threshold)):
            raise ModelingError("confidence_probability_invalid")
        emitted = self.draft_len
        for index, score in enumerate(self.confidence):
            if score < self.confidence_threshold:
                emitted = index
                break
        return SpeculativeSpec(emitted, declared.acceptance_marginals()[:emitted],
                               source=self.source, label=self.label)
