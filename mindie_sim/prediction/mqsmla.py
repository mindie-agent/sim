"""Mixed-quant sparse MLA with resident dequantized KV and paired AIV owners.

[Role] Price the retained whole-row, scalar-address source branch.
[Invariants] One query's original and compressed lists share online state.
  A padded tile repeats valid KV reads; QK computes 128 rows and PV rounds its
  valid prefix to 16. UB/L1 exchanges never become invented HBM workspaces.
[Timeouts & Locks] O(active query rows) metadata; two live phase tiles per owner.
[Side-Effect Policy] Pure plans and the common causal Engine, no observations.
"""
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

SOURCE = {
    "repository": "cannbot-dsl",
    "revision": "4519e5372a8b9306f7d823688c0cb8141ee34fb1",
    "kernel": {
        "path": "samples/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
        "sha256": "40b5b14ced15bb7c3d3687ab7387d508a19065eb5353dfb0675348e0de22fa3f",
    },
    "metadata": {
        "path": "samples/mixed_quant_sparse_flash_mla_metadata/mixed_quant_sparse_flash_mla_metadata.py",
        "sha256": "a62295be0fdf6acc7c0c606e60424e3fbc03310bee4e1effa2a384ccd6415b65",
    },
    "capture_binary_identity": "not_established",
}

# Source Channel depths. BF16 output aliases the FP32 accumulator. The P NZ
# buffers include one 16-element pad per N1 face; all scalar state is FP32.
BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144,
           "ub": 2 * 32 * 128 * 4 + 2 * 32 * 512 * 4
                 + 2 * (32 * 128 + 8 * 16) * 2
                 + 4 * 16 * 544 + 2 * 32 * 17 * 16 * 2 + 10 * 32 * 4}


def query_rows(shape):
    """Global query order preserves each sequence's causal context."""
    window, ratio, topk = (shape[key] for key in ("window", "ratio", "topk"))
    for query, context in shape["sequences"]:
        for offset in range(query):
            position = context + offset + 1
            original = min(position, window) if window else 0
            compressed = position // ratio if ratio else 0
            yield original, min(compressed, topk) if topk else compressed


def owner_ranges(rows, cores):
    """Exact metadata SUPPORT_FD=False contiguous q/r row ownership."""
    q, r = divmod(rows, cores)
    start = 0
    result = []
    for core in range(cores):
        count = q + (core < r)
        if count:
            result.append((core, start, start + count))
        start += count
    return tuple(result)


@lru_cache(maxsize=512)
def tile_phases(branch, valid, first, last):
    """Source phase payloads, not measured durations or utilization factors."""
    if not valid:
        return (
            ("sinks", Work(resources={"hbm": 256, "mte2_aiv": 256})),
            ("empty", Work(resources={"vector": 64 * 512})),
            ("output", Work(resources={"hbm": 64 * 512 * 2, "mte3": 64 * 512 * 2})),
        )
    physical = 128
    row_bytes = 544 if branch == "original" else 320
    # Four 16-row subtiles per AIV; duplicate tail indices still issue copies.
    gather = physical * row_bytes
    index = physical * 8  # INT32 sparse index and INT32 page-table entry.
    padded_pv = (valid + 15) // 16 * 16
    q = 64 * 512 * 2 if first else 0
    output = 64 * 512 * 2 if last else 0
    return (
        ("query", Work(resources={"hbm": q, "mte2_aic": q})),
        ("sinks", Work(resources={"hbm": 256 if first else 0,
                                  "mte2_aiv": 256 if first else 0})),
        ("gather", Work(resources={"hbm": gather + index, "mte2_aiv": gather})),
        ("dequant", Work(resources={"vector": physical * 512 * (6 if branch == "original" else 4)})),
        ("kv_l1", Work(resources={"mte3": physical * 512 * 2})),
        ("qk_operands", Work(resources={"mte1": (64 + physical) * 512 * 2})),
        ("qk", Work(matrix={"bf16": 2 * 64 * physical * 512})),
        ("scores", Work(resources={"fixpipe": 64 * physical * 4})),
        ("softmax", Work(resources={"vector": 8 * 64 * valid + 26 * 64})),
        ("probability_l1", Work(resources={"mte3": 64 * physical * 2})),
        ("pv_operands", Work(resources={"mte1": (64 * physical + physical * 512) * 2})),
        ("pv", Work(matrix={"bf16": 2 * 64 * padded_pv * 512})),
        ("values", Work(resources={"fixpipe": 64 * 512 * 4})),
        ("accumulate", Work(resources={"vector": 64 * 512 * ((1 if first else 3) + (2 if last else 0))})),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    )


@dataclass(frozen=True)
class OwnerTiles:
    """Index finite source tiles using O(query rows), never O(key tiles)."""
    rows: tuple
    ends: tuple

    @classmethod
    def create(cls, rows):
        ends, total = [], 0
        for original, compressed in rows:
            total += max(1, (original + 127) // 128 + (compressed + 127) // 128)
            ends.append(total)
        return cls(tuple(rows), tuple(ends))

    def key(self, index):
        row = bisect_right(self.ends, index)
        local = index - (self.ends[row - 1] if row else 0)
        original, compressed = self.rows[row]
        nt = (original + 127) // 128
        count = self.ends[row] - (self.ends[row - 1] if row else 0)
        if local < nt:
            return "original", min(128, original - local * 128), local == 0, local + 1 == count
        return "compressed", max(0, min(128, compressed - (local - nt) * 128)), local == 0, local + 1 == count

    @property
    def count(self):
        return self.ends[-1]


def build(op, geometry):
    shape = dict(op.metadata)["mqsmla_geometry"]
    if (shape["heads"] != 64 or shape["width"] != 512
            or shape["metadata_support_fd"] or shape["addressing"] != "scalar"):
        raise ModelingError("mqsmla_source_geometry_or_branch_unsupported")
    rows = tuple(query_rows(shape))
    cores = geometry.compute_units
    if not rows or not 1 <= cores <= 36:
        raise ModelingError("mqsmla_requires_positive_rows_and_at_most36_aic")
    # Retained wrapper SELECTIVE_FD_72 supersedes SUPPORT_FD=False for this
    # one geometry; do not silently price its split reduction as whole rows.
    if cores == 32 and len(rows) == 72 and all(row == (128, 512) for row in rows):
        raise ModelingError("mqsmla_selective_72_row_fd_branch_not_modeled")
    ownership = owner_ranges(len(rows), cores)
    streams = tuple((owner, OwnerTiles.create(rows[start:end]))
                    for owner, start, end in ownership)
    for pool, amount in BUFFERS.items():
        units = geometry.resource_units.get("vector", cores) if pool == "ub" else cores
        if (geometry.buffers.get(pool, 0) < amount
                or geometry.buffer_units.get(pool, 0) != units):
            raise ModelingError("mqsmla_source_buffer_capacity_or_ownership:" + pool)
    for role in ("vector", "mte2_aiv", "mte3"):
        if geometry.resource_units.get(role) != 2 * cores:
            raise ModelingError("mqsmla_requires_paired_aiv:" + role)
    for role in ("mte1", "mte2_aic", "fixpipe"):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError("mqsmla_requires_private_aic:" + role)
    total = Work()
    for _, stream in streams:
        for index in range(stream.count):
            for _, work in tile_phases(*stream.key(index)):
                total += work

    def program(context):
        def one(owner, stream):
            def stages(index):
                key = stream.key(index)
                result = []
                previous = None
                for name, work in tile_phases(*key):
                    if name in {"query", "sinks", "gather"}:
                        after = ()
                    elif name == "qk_operands":
                        after = ("kv_l1", "query")
                    elif name == "softmax":
                        after = ("scores", "sinks")
                    else:
                        after = (previous,) if previous else ()
                    # Coarse phases preserve finite Channel reuse. Two Q
                    # halves are replaced only after the previous row's QK;
                    # QK/PV each hold at most one of the two shared L0C slots.
                    prior = {
                        "query": ("output",) if key[2] else (),
                        "sinks": ("output",) if key[2] else (),
                        "qk_operands": ("qk",), "qk": ("scores",),
                        "pv_operands": ("pv",), "pv": ("values",),
                        "accumulate": ("output",) if key[2] else ("accumulate",),
                    }.get(name, ())
                    result.append(Stage(
                        name, context.bind(work), after=after, previous=prior,
                        placement=context.owned(owner, cores, set(work.resources) - {"hbm"}
                                                | ({"cube"} if work.matrix else set()))))
                    previous = name
                return tuple(result)

            allocations = tuple(context.allocate(
                f"mqsmla{owner}_{pool}", pool, amount,
                (2 * owner, 2 * owner + 1) if pool == "ub" else (owner,))
                for pool, amount in BUFFERS.items())
            return KernelProgram((
                Stage("reserve", acquire=allocations),
                # Two complete phase tiles are a conservative source subset:
                # P and alpha have two slots. The third source KV slot can
                # prefetch more aggressively, which this compact DAG omits.
                Stage("tiles", after=("reserve",), program=PipelineProgram(stream.count, 2, stages)),
                Stage("release", after=("tiles",), free=allocations),
            ))
        return ParallelProgram(tuple(one(owner, stream) for owner, stream in streams))

    return OperatorPlan(total, program, {
        "source_execution_domain": "mixed", "source_program": SOURCE,
        "mqsmla_owner_ranges": ownership, "mqsmla_buffers_per_owner": BUFFERS,
        "mqsmla_schedule": "whole_query_rows;metadata_SUPPORT_FD_false;two_live_phase_tiles",
        "mqsmla_approximation": "coarse_source_phase_DAG;register_operations_not_instruction_latency",
        "unmodeled_source_features": ["aicpu_metadata_program", "scalar_page_lookup_response",
                                      "vector_address_prepass", "flash_decode_partial_reduction",
                                      "third_KV_slot_prefetch_overlap",
                                      "cross_query_sinks_prefetch_overlap",
                                      "source_compiler_instruction_scheduling"],
    })
