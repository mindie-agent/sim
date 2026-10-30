"""Select declared draft geometry.

[Role] Separate optional draft structure from target model structure.
[Invariants] Selection never depends on a model name or measured duration.
[Timeouts & Locks] Pure computation.
[Side-Effect Policy] No I/O.
"""
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.catalog.types import DraftSpec
from dataclasses import replace


def draft_quant(model, draft):
    """Use the draft's declared representation in both work and residency."""
    quant = model.quant
    if not draft.weight_category:
        return quant
    fmt = quant.format_for(draft.weight_category)
    return replace(quant, weight_format=fmt, expert_weight_format=fmt,
                   activation_format="bf16" if fmt == "bf16" else quant.activation_format)


def optional_draft_spec(model: ModelSpec, kind: str) -> DraftSpec:
    aux = model.auxiliary
    if aux is None:
        raise ModelingError("catalog_draft_missing")
    dspark_mode = kind in {"dspark", "verify", "draft_verify"}
    draft = aux.dspark if dspark_mode else aux.mtp
    if draft is None:
        if dspark_mode and aux.mtp is not None and aux.mtp.kind == "mtp_dspark":
            draft = aux.mtp
        elif not dspark_mode and aux.dspark is not None:
            raise ModelingError("catalog_draft_mode_unavailable:mtp")
        elif dspark_mode:
            raise ModelingError("catalog_draft_mode_unavailable:dspark")
    if draft is None:
        raise ModelingError("catalog_draft_mode_missing")
    draft.validate()
    return draft
