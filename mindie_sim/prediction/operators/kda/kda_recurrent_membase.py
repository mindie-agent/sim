"""Source-selected KDA decode with per-sequence beta reuse.

[Role] Build RecurrentKda work for the retained memory-base header.
[Invariants] Each owner scans all sequences and loads beta once per owned
  sequence. The state update materializes delta in UB before MulAddDst.
[Timeouts & Locks] Pure finite source plans; no I/O, locks or device work.
[Side-Effect Policy] Returns immutable plans/operators on the shared Engine.
"""
from dataclasses import replace

from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.kda import kda_recurrent

IMPLEMENTATION = RECIPE = "vllm_ascend_recurrent_kda_arch22"
SOURCE = {
    **kda_recurrent.SOURCE,
    "kernel": {
        "path": "csrc/attention/recurrent_kda/op_kernel/recurrent_kda.h",
        "sha256": "043a02e8cbbdffd0f6794793e0837e2bed75f6fc2a764ccfe026201943a949de",
        "lines": ((148, 195), (233, 273), (382, 597), (600, 725), (815, 882)),
    },
    "compile_branch": "non_310_header_with_RKDA_ENABLE_ADD_FOLD_REDUCE=1",
}
EXECUTION = kda_recurrent.SourceExecution(
    beta_per_head=False, delta_broadcast=True, validate_all_sequences=True)


def build(op, geometry, *, compact=True):
    return kda_recurrent.build(op, geometry, compact=compact, execution=EXECUTION)


def bind_source(op):
    """Apply the source identity and minimum-owner membase floor work to a shared recurrent op."""
    raw_meta = dict(op.metadata)
    metadata = {**raw_meta, "implementation": IMPLEMENTATION, "source": SOURCE,
                "source_binding": "explicit_arch22_kernel_selection_not_binary_equivalence"}
    bound = replace(op, recipe=RECIPE, metadata=tuple(sorted(metadata.items())))
    params = raw_meta.get("kda_recurrent_params")
    if params is None:
        return bound
    floor_geo = Geometry(
        1, {"vector": 1, "mte2": 1, "mte3": 1},
        dict(kda_recurrent.frontend_geometry().buffers),
    )
    summary = kda_recurrent.summarize_work(params, floor_geo, execution=EXECUTION)
    bound = summary.apply(bound)
    physical_read = float(sum(
        amount for role, amount in summary.resources.items()
        if role in {"mte2", "mte2_aiv"}
    ))
    return replace(bound, mte2_bytes=physical_read)


def make_op(params, **kwargs):
    return bind_source(kda_recurrent.make_op(params, **kwargs))
