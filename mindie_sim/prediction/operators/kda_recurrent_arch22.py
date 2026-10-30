"""Source-selected generic AscendC KDA decode, including its beta reuse.

[Role] Select the non-arch35 header dispatched for the retained arch22 build.
[Invariants] Reuse shared state/queue lifetimes, never arch35 timing. Every
  owner scans all sequences; beta is loaded once per owned sequence, and
  the state update materializes the delta broadcast in UB before MulAddDst.
[Timeouts & Locks] Pure finite source plans; no I/O, locks or device work.
[Side-Effect Policy] Returns immutable plans/operators on the shared Engine.
"""
from dataclasses import replace

from . import kda_recurrent

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


def make_op(params, **kwargs):
    op = kda_recurrent.make_op(params, **kwargs)
    metadata = {**dict(op.metadata), "implementation": IMPLEMENTATION, "source": SOURCE,
                "source_binding": "explicit_arch22_kernel_selection_not_binary_equivalence"}
    return replace(op, recipe=RECIPE, metadata=tuple(sorted(metadata.items())))
