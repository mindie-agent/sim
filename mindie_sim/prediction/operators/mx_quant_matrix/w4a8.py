"""A8W4 MSD grouped gate/up + SwiGLU + dynamic INT8 output program.

[Role]
  Execute the source-verified Key2 A8W4 MSD grouped kernel as explicit pre,
  INT4 matrix, and post phases on the common Engine.

[Invariants]
  * Inputs are INT8 activations, packed INT4 per-channel weights, uint64
    encoded column scales, float32 assist/token scales and explicit integer
    expert rows. No probability histogram, bias, clip, group quant or
    transposed weights are inferred.
  * The physical matrix rate is the case-declared ``int4`` peak. There is no
    INT8/FP16 fallback; a missing peak raises ``compute_precision_rate_required``.
  * Pre writes the high/low packed INT4 scratch (M*K bytes); the matrix reads
    it and the post phase reads the two FP16 outputs. Intermediate activation
    and quantization values stay local to the post body.
  * Physical work is the explicit ledger below, never a measured duration or a
    magic per-kernel constant.
  * Column ``SetQuantVector`` scale application is charged once (2*M*N). Its
    source Fixpipe placement is a declared vector-equivalent interpretation,
    not a claim about dedicated Fixpipe issue rate.

[Timeouts & Locks]
  O(experts + owners) retained records; no I/O, captures or locks.

[Side-Effect Policy]
  Returns immutable Work/Program records; no device contexts or observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, check_live_buffers
from .w4a8_matrix import matrix_program, matrix_work
FP32_BYTES = 4
from .w4a8_schedule import (
    MSD_FLOPS_PER_K_STEP, MSD_WORKSPACE_BYTES_PER_OUTPUT, NZ_N_ALIGNMENT,
    POST_DEQUANT_OPS_PER_OUTPUT, PRE_OPS_PER_INPUT, QUANT_OPS_PER_OUTPUT,
    SWIGLU_OPS_PER_OUTPUT, align_up, gate_schedule, total_work,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/quantization/methods/w4a8/w4a8.py",
        source_sha256="51682287230ab8b4e618a8dd7763a22250d8f416730c12a1d7f1f79aa2c7f89e",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h",
        source_sha256="08e07b078ca73626df3a135bff230c9b7b61108b235e0f98f22420f4c94f9733",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_base_tiling.cpp",
        source_sha256="8e2d570efd12ddda8360fcd515c81fd50444571d47e2ed3200c10f72f7aaef97",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h",
        source_sha256="b87eda2542700e1800412f9775360552b5c6ae6e38e2afcbd77da677d60f8332",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h",
        source_sha256="1881d03e65258e846fe9bbfb736f4cbbcd770b84a6730319dd0c9d00abdc4087",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h",
        source_sha256="8cf63fb3e34955a2ed28d4f9f56c2030e74744cbe5a63aaeb5d130eb0940758f",
    ),
)

GATE_NZ_K_ALIGNMENT = 16
GATE_MAX_N = 10_240
GATE_K_EXCLUSIVE_LIMIT = 20_000
UB_ALIGN_BYTES = 32
PRE_UB_MASK_BYTES = 256
PRE_UB_FP16_BYTES_PER_K = 2
PRE_UB_FP32_BYTES_PER_K = 4
POST_UB_MM_OUT_BYTES_PER_ELEM = 8
POST_UB_ASSIST_BYTES_PER_COL = 4
POST_UB_REDUCE_BYTES_PER_COL = 2
POST_UB_REDUCE_BLOCK_BYTES = 64
POST_UB_ROW_SCALE_ALIGN = 8
POST_UB_ROW_WEIGHT_HALF_BYTES = 17
POST_UB_STATIC_COL_BYTES = POST_UB_ASSIST_BYTES_PER_COL + POST_UB_REDUCE_BYTES_PER_COL


RECIPE = "ascend_w4a8_msd"
ROLE = "grouped_gate_up_activation_quant"
PROFILE = "ascend_w4a8_msd"
SUPPORTED_EXPERT_ROW_BASES = {
    "declared_integer_counts", "forced_balanced_integer_counts",
    "source_force_eplb_masked_integer_counts",
}

SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
    "files": (
        {"path": "vllm_ascend/quantization/methods/w4a8/w4a8.py",
         "sha256": "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b",
         "lines": "495-507"},
        {"path": "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/"
                 "grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h",
         "sha256": "b87eda2542700e1800412f9775360552b5c6ae6e38e2afcbd77da677d60f8332",
         "lines": "145-218"},
        {"path": "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/"
                 "grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h",
         "sha256": "1881d03e65258e846fe9bbfb736f4cbbcd770b84a6730319dd0c9d00abdc4087",
         "lines": "180-285"},
        {"path": "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/"
                 "grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h",
         "sha256": "8cf63fb3e34955a2ed28d4f9f56c2030e74744cbe5a63aaeb5d130eb0940758f",
         "lines": "100-250"},
        {"path": "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/"
                 "grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h",
         "sha256": "e7398287dcc06dd2a70d7c658cd4b3839367f32f34a9c6a5a6f68a8bf0077ae9",
         "lines": "90-130"},
    ),
    "branch": "A8W4_MSD Key2 (INT8 x INT4, double-row cube, FP16 workspace)",
}


@dataclass(frozen=True)
class Config:
    m: int
    n: int
    k: int
    expert_rows: tuple[int, ...]
    routing_basis: str = "declared_integer_counts"

    @property
    def active(self):
        return sum(rows > 0 for rows in self.expert_rows)

    @property
    def half(self):
        return self.n // 2

    def to_dict(self):
        return {
            "m": self.m, "n": self.n, "k": self.k,
            "expert_rows": list(self.expert_rows), "active_groups": self.active,
            "expert_scan_count": len(self.expert_rows),
            "zero_expert_indices": [index for index, rows in enumerate(self.expert_rows)
                                    if rows == 0],
            "zero_bank_offsets_int4_elements": [index * self.n * self.k
                                                for index, rows in enumerate(self.expert_rows)
                                                if rows == 0],
            "routing_basis": self.routing_basis,
            "weight_format": "int4", "activation_format": "int8",
            "output_format": "int8", "swiglu": True, "swiglu_limit": 0.0,
            "column_scale": "uint64_bitpattern_fp32_8B_per_channel",
            "assist_scale": "float32_4B_per_channel",
            "token_scale": "float32_4B_per_row",
            "source": SOURCE,
        }


def _require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ModelingError(f"w4a8_msd_{name}_must_be_integer_at_least_{minimum}")
    return value


def _validate_gate_dimensions(n: int, k: int) -> None:
    """Enforce pinned Key2 A8W4 host alignment and N/K ceilings (gmm_dsq_base.h:481)."""
    if n % 2:
        raise ModelingError("w4a8_msd_swiglu_width_must_be_even")
    if n % NZ_N_ALIGNMENT or k % GATE_NZ_K_ALIGNMENT:
        raise ModelingError("w4a8_msd_nz_requires_n_multiple64_k_multiple16")
    if n > GATE_MAX_N:
        raise ModelingError(f"w4a8_msd_n_exceeds_host_limit:{n}>{GATE_MAX_N}")
    if k >= GATE_K_EXCLUSIVE_LIMIT:
        raise ModelingError(f"w4a8_msd_k_reaches_host_limit:{k}>={GATE_K_EXCLUSIVE_LIMIT}")


def _gate_post_ub_for_rows(n: int, rows: int) -> int:
    """Simultaneous post-phase InitBuffer UB bytes for ``rows`` rows (post.h:303-313)."""
    return (
        POST_UB_MM_OUT_BYTES_PER_ELEM * rows * n
        + (rows * n) // 2
        + align_up(rows, POST_UB_ROW_SCALE_ALIGN) * FP32_BYTES
        + POST_UB_ASSIST_BYTES_PER_COL * n
        + POST_UB_REDUCE_BYTES_PER_COL * n
        + POST_UB_REDUCE_BLOCK_BYTES
    )


def _gate_host_max_rows(n: int, ub_capacity: int) -> int:
    """Host ``CalMaxRowInUbA8W4`` row selection (base_tiling.cpp:56-91)."""
    avail = ub_capacity - POST_UB_REDUCE_BLOCK_BYTES - POST_UB_STATIC_COL_BYTES * n
    if avail <= 0:
        return 0
    est = (2 * avail) // (POST_UB_ROW_WEIGHT_HALF_BYTES * n)
    while est > 0 and _gate_post_ub_for_rows(n, est) > ub_capacity:
        est -= 1
    return est


def gate_phase_ub_bytes(config: Config, *, ub_capacity: int | None = None) -> tuple[int, int]:
    """Per-AIV simultaneous aligned InitBuffer UB bytes for pre and host-tiled post."""
    pre_ub = (
        align_up(config.k, UB_ALIGN_BYTES)
        + 2 * align_up(config.k // 2, UB_ALIGN_BYTES)
        + align_up(PRE_UB_FP16_BYTES_PER_K * config.k, UB_ALIGN_BYTES)
        + align_up(PRE_UB_FP32_BYTES_PER_K * config.k, UB_ALIGN_BYTES)
        + PRE_UB_MASK_BYTES
    )
    host_rows = _gate_host_max_rows(config.n, ub_capacity) if ub_capacity is not None else 1
    post_ub = _gate_post_ub_for_rows(config.n, max(1, host_rows))
    return pre_ub, post_ub


def _geometry_int4_subtile(geometry):
    datapath = getattr(geometry, "datapath", None)
    waves = getattr(datapath, "subtile_wave", ()) if datapath is not None else ()
    for wave in waves:
        if getattr(wave, "native_mode", "") == "int4":
            return wave
    return None


def config_from_op(op):
    """Validate and return the only supported source domain."""
    raw = dict(op.metadata).get("w4a8_msd")
    if raw is None:
        raise ModelingError("w4a8_msd_requires_explicit_program_configuration")
    if not isinstance(raw, dict):
        raise ModelingError("w4a8_msd_configuration_must_be_mapping")
    m = _require_int(raw.get("m"), "m", 1)
    n = _require_int(raw.get("n"), "n", 2)
    k = _require_int(raw.get("k"), "k", 1)
    rows = raw.get("expert_rows")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)) or not rows:
        raise ModelingError("w4a8_msd_expert_rows_must_be_nonempty_sequence")
    rows = tuple(_require_int(value, "expert_rows") for value in rows)
    if sum(rows) != m:
        raise ModelingError("w4a8_msd_expert_rows_must_conserve_matrix_rows")
    basis = raw.get("routing_basis")
    if basis not in SUPPORTED_EXPERT_ROW_BASES:
        raise ModelingError("w4a8_msd_rejects_probability_moment_routing")
    _validate_gate_dimensions(n, k)
    for key, expected in (("weight_format", "int4"),
                          ("activation_format", "int8"),
                          ("output_format", "int8")):
        if raw.get(key) != expected:
            raise ModelingError(f"w4a8_msd_representation_mismatch:{key}")
    if raw.get("swiglu") is not True or raw.get("swiglu_limit") not in (0, 0.0):
        raise ModelingError("w4a8_msd_requires_unclipped_swiglu_domain")
    return Config(m, n, k, rows, basis)


def config_from_group(declaration, members):
    """Build the config from the selected complete group's real operands."""
    role = declaration.get("role")
    if role != ROLE:
        raise ModelingError(f"w4a8_msd_role_mismatch:{role}")
    view = declaration.get("tensor_view", {})
    if (view.get("input_dtype"), view.get("weight_dtype"), view.get("output_dtype")) != (
            "int8", "int4", "int8"):
        raise ModelingError("w4a8_msd_group_representation_mismatch")
    gates = [op for op in members if op.family == "grouped_gemm"
             and op.recipe in {"grouped_gate_up", "grouped_expert_gemm", "balanced_grouped_gate_up"}]
    activations = [op for op in members if op.family == "swiglu"]
    quantizers = [op for op in members if op.family == "quantize"]
    if len(gates) != 1 or len(activations) != 1 or len(quantizers) != 1:
        raise ModelingError("w4a8_msd_requires_gate_swiglu_quant_complete_group")
    gate, activation, quant = gates[0], activations[0], quantizers[0]
    metadata = dict(gate.metadata)
    quant_metadata = dict(quant.metadata)
    if metadata.get("weight_format") not in {"int4", "w4a8"}:
        raise ModelingError("w4a8_msd_requires_packed_int4_weights")
    if metadata.get("activation_format") != "int8":
        raise ModelingError("w4a8_msd_requires_int8_activations")
    if metadata.get("weight_representation") not in (None, "int4_packed_per_channel"):
        raise ModelingError("w4a8_msd_requires_perchannel_packed_int4_weights")
    if dict(quant.metadata).get("output_format", dict(quant.metadata).get("output_dtype")) != "int8":
        raise ModelingError("w4a8_msd_requires_dynamic_int8_output")
    if metadata.get("bias") or metadata.get("transpose_weight"):
        raise ModelingError("w4a8_msd_bias_or_transposed_weight_not_supported")
    realization = metadata.get("routing_realization")
    if not isinstance(realization, dict):
        raise ModelingError("w4a8_msd_requires_explicit_integer_expert_rows")
    if realization.get("basis") not in SUPPORTED_EXPERT_ROW_BASES:
        raise ModelingError("w4a8_msd_rejects_probability_moment_routing")
    rows = realization.get("local_expert_rows", realization.get("expert_rows"))
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        raise ModelingError("w4a8_msd_requires_explicit_integer_expert_rows")
    rows = tuple(_require_int(value, "expert_rows") for value in rows)
    m, n, k = (_require_int(value, "matrix_axis", 1)
               for value in gate.local_shape[-3:])
    if m != sum(rows):
        raise ModelingError("w4a8_msd_group_rows_do_not_match_explicit_histogram")
    config = Config(m, n, k, tuple(rows), realization["basis"])
    _validate_gate_dimensions(n, k)
    if (metadata.get("activation_scale_format") != "fp32"
            or metadata.get("activation_scale_bytes") != FP32_BYTES * m
            or metadata.get("activation_group_size", 0) < k):
        raise ModelingError("w4a8_msd_requires_one_fp32_input_scale_per_row")
    if (tuple(activation.local_shape) != (m, n // 2)
            or tuple(quant.local_shape) != (m, n // 2)):
        raise ModelingError("w4a8_msd_activation_quant_geometry_mismatch")
    if (quant_metadata.get("activation_scale_format") != "fp32"
            or quant_metadata.get("activation_scale_bytes") != FP32_BYTES * m
            or quant_metadata.get("activation_group_size", 0) < n // 2
            or quant.mte3_bytes != m * (n // 2) + FP32_BYTES * m):
        raise ModelingError("w4a8_msd_requires_one_fp32_output_scale_per_row")
    # A complete group is exactly these three source phases; additional
    # members would leave undeclared work outside the explicit ledger.
    if len(members) != 3:
        raise ModelingError("w4a8_msd_group_requires_exact_source_phases")
    return config


def _role_work(work, name, resource_roles):
    if resource_roles is None:
        return work
    resource_work = dict(work.resources)
    amount = resource_work.pop("mte2", 0)
    if amount:
        role = read_role(resource_roles, aiv=None if name == "matrix" else 0)
        resource_work[role] = amount
    return Work(work.matrix, resource_work)


def _schedule_phases(schedule, resource_roles=None):
    groups = {
        "pre": schedule.pre, "matrix": schedule.matrix,
        "post_dequant": schedule.post_dequant,
        "swiglu": schedule.swiglu, "quantize": schedule.quantize,
    }
    return {name: _role_work(total_work(group), name, resource_roles)
            for name, group in groups.items()}


def _phase_work(config, *, fixpipe=False, resource_roles=None):
    """Global source ledger; ownership-dependent duplicates resolve at prepare."""
    schedule = gate_schedule(
        config.m, config.n, config.k, config.expert_rows,
        aic_owners=1, aiv_owners=2, fixpipe=fixpipe)
    return _schedule_phases(schedule, resource_roles)


def _geometry_owners(geometry):
    aic = geometry.resource_units.get("cube", geometry.compute_units)
    aiv = geometry.resource_units.get("vector", geometry.compute_units)
    return aic, aiv


def work_from_config(config, *, fixpipe=False, resource_roles=None, geometry=None):
    if geometry is None:
        phases = _phase_work(config, fixpipe=fixpipe, resource_roles=resource_roles)
    else:
        aic, aiv = _geometry_owners(geometry)
        schedule = gate_schedule(
            config.m, config.n, config.k, config.expert_rows,
            aic_owners=aic, aiv_owners=aiv, fixpipe=fixpipe,
            subtile_wave=_geometry_int4_subtile(geometry))
        phases = _schedule_phases(schedule, resource_roles)
    total = Work()
    for work in phases.values():
        total += work
    return total


def ledger(config):
    """Reference arithmetic; final ownership-dependent work binds at prepare."""
    m, n, k = config.m, config.n, config.k
    h, active = config.half, config.active
    reference = _phase_work(config)
    return {
        "reference_inventory": {"aic": 1, "aiv": 2,
                                "scope": "unbound_reference; rebind_actual_device_at_prepare"},
        "pre": {"read_bytes": m * k, "write_bytes": m * k,
                "vector_ops": PRE_OPS_PER_INPUT * m * k,
                "expression": "high cast+mul+cast (3MK) + halfword-and (MK/2) "
                              "+ cast+add+cast (3MK)"},
        "matrix": {"flops_int4": reference["matrix"].matrix["int4"],
                   "logical_flops_int4": MSD_FLOPS_PER_K_STEP * m * n * k,
                   "read_bytes": reference["matrix"].resources["mte2"],
                   "write_bytes": MSD_WORKSPACE_BYTES_PER_OUTPUT * m * n,
                   "publication_role": "fixpipe; additional service only when declared",
                   "expression": "double-M INT4 product 4MNK; every N tile reads "
                                 "its A slice; each M tile reads its weight panel"},
        "post_dequant": {"read_bytes": reference["post_dequant"].resources["mte2"],
                         "unique_payload_read_bytes": (MSD_WORKSPACE_BYTES_PER_OUTPUT * m * n
                                                       + FP32_BYTES * n * active + FP32_BYTES * m),
                         "assist_bytes_per_owner_segment": FP32_BYTES * n,
                         "owner_segments": "one assist load per contiguous AIV "
                                           "row chunk and expert segment; resolved "
                                           "from declared ownership at prepare",
                         "vector_ops": POST_DEQUANT_OPS_PER_OUTPUT * m * n,
                         "expression": "2MN cast + MN high*16 + MN low add "
                                       "+ MN assist + MN token scale"},
        "swiglu": {"outputs": m * h, "vector_ops": SWIGLU_OPS_PER_OUTPUT * m * h,
                   "expression": "exp+add+reciprocal+gate_mul+up_mul per output"},
        "quantize": {"write_bytes": m * h + FP32_BYTES * m, "vector_ops": QUANT_OPS_PER_OUTPUT * m * h + m,
                     "expression": "Abs MH + max M(H-1) + 2M scale/reciprocal "
                                   "+ MH mul + 2MH casts"},
        "experts": {"scan_count": len(config.expert_rows),
                    "active_groups": active,
                    "zero_bank_offsets_int4_elements": [
                        index * n * k for index, rows in enumerate(config.expert_rows)
                        if rows == 0],
                    "expression": "all source group entries are scanned; zero "
                                  "groups keep bank offsets but issue no tile"},
    }


def _base_program(config, *, fixpipe=False, resource_roles=None):
    def program(context):
        raise ModelingError("w4a8_msd_execution_requires_device_geometry")
    return program


def _owner_program(schedule, *, pre_ub_bytes: int = 0, post_ub_bytes: int = 0):
    order = (("pre", schedule.pre, schedule.aiv_owners, ()),
             ("matrix", schedule.matrix, schedule.aic_owners, "pre"),
             ("post_dequant", schedule.post_dequant, schedule.aiv_owners, "matrix"),
             ("swiglu", schedule.swiglu, schedule.aiv_owners, "post_dequant"),
             ("quantize", schedule.quantize, schedule.aiv_owners, "swiglu"))

    def program(context):
        stages, barrier = [], {}
        has_ub = "ub" in context.pool_names
        post_allocs = {}
        for name, group, owners_count, after in order:
            predecessors = barrier.get(after, ()) if after else ()
            if name == "matrix":
                stages.append(Stage("matrix", after=predecessors,
                                    program=matrix_program(context, schedule)))
                barrier[name] = ("matrix",)
                continue
            names = []
            seen_owners = set()
            for record in group:
                seen_owners.add(record.owner)
                work = _role_work(record.work, name, context.resource_units)
                bound = context.for_phase(
                    work, kernel=f"grouped_matmul_swiglu_quant_v2_a8w4_msd_{name}",
                    dtype="int4" if name == "matrix" else None)
                # HBM stays one shared bandwidth domain.  Cube, vector,
                # read/store pipes and a declared Fixpipe follow their owner.
                roles = set(work.resources) - {"hbm"}
                if work.matrix:
                    roles.add("cube")
                acq, rel = (), ()
                if has_ub and name == "pre" and pre_ub_bytes > 0:
                    alloc = context.allocate(f"pre_ub:{record.owner}", "ub", pre_ub_bytes, (record.owner,))
                    acq, rel = (alloc,), (alloc,)
                elif has_ub and name == "post_dequant" and post_ub_bytes > 0:
                    alloc = context.allocate(f"post_ub:{record.owner}", "ub", post_ub_bytes, (record.owner,))
                    post_allocs[record.owner] = alloc
                    acq = (alloc,)
                elif has_ub and name == "quantize" and record.owner in post_allocs:
                    rel = (post_allocs[record.owner],)
                stage_name = f"{name}:{record.owner}"
                stages.append(Stage(
                    stage_name, work=bound.work, after=predecessors,
                    acquire=acq, free=rel,
                    placement=context.owned(record.owner, owners_count, roles)))
                names.append(stage_name)
            if has_ub and name == "pre" and pre_ub_bytes > 0:
                for idle_owner in range(schedule.aiv_owners):
                    if idle_owner in seen_owners:
                        continue
                    alloc = context.allocate(f"pre_ub:{idle_owner}", "ub", pre_ub_bytes, (idle_owner,))
                    idle_name = f"pre_ub_init:{idle_owner}"
                    stages.append(Stage(idle_name, after=predecessors, acquire=(alloc,), free=(alloc,)))
                    names.append(idle_name)
            elif has_ub and name == "post_dequant" and post_ub_bytes > 0:
                for idle_owner in range(schedule.aiv_owners):
                    if idle_owner in seen_owners:
                        continue
                    alloc = context.allocate(f"post_ub:{idle_owner}", "ub", post_ub_bytes, (idle_owner,))
                    idle_name = f"post_ub_init:{idle_owner}"
                    stages.append(Stage(idle_name, after=predecessors, acquire=(alloc,), free=(alloc,)))
                    names.append(idle_name)
            barrier[name] = tuple(names)
        return KernelProgram(tuple(stages))

    return program


def build(op, geometry=None):
    config = config_from_op(op)
    if geometry is None:
        return OperatorPlan(work_from_config(config), _base_program(config))
    ub_cap = int(geometry.buffers["ub"]) if "ub" in geometry.buffers else None
    pre_ub_bytes, post_ub_bytes = gate_phase_ub_bytes(config, ub_capacity=ub_cap)
    if "ub" in geometry.buffers:
        check_live_buffers(
            (
                LiveBufferSpec(
                    "w4a8_gate_pre_ub", "ub", pre_ub_bytes, 0, 1,
                    owner_kind="aiv", alignment_bytes=UB_ALIGN_BYTES,
                ),
                LiveBufferSpec(
                    "w4a8_gate_post_ub", "ub", post_ub_bytes, 1, 2,
                    owner_kind="aiv", alignment_bytes=UB_ALIGN_BYTES,
                ),
            ),
            geometry,
            family=RECIPE,
            kernel_abi="aclnnGroupedMatmulSwigluQuantV2",
            source_guard="w4a8_msd_ub_capacity",
        )
    fixpipe = "fixpipe" in geometry.resource_units
    aic, aiv = _geometry_owners(geometry)
    schedule = gate_schedule(
        config.m, config.n, config.k, config.expert_rows,
        aic_owners=aic, aiv_owners=aiv, fixpipe=fixpipe,
        subtile_wave=_geometry_int4_subtile(geometry))
    total = Work()
    for name, work in _schedule_phases(schedule, geometry.resource_units).items():
        total += (_role_work(matrix_work(schedule, geometry.resource_units), name,
                             geometry.resource_units) if name == "matrix" else work)
    return OperatorPlan(
        total,
        _owner_program(schedule, pre_ub_bytes=pre_ub_bytes, post_ub_bytes=post_ub_bytes),
    )


def describe(config):
    return {
        "profile": PROFILE, "config": config.to_dict(), "ledger": ledger(config),
        "limitations": [
            "explicit_coarse_phase_order_does_not_claim_native_workspace_pipeline_overlap",
            "workspace_split_chunk_boundaries_not_reconstructed",
            "matrix_publication_HBM_retained; Fixpipe_service_requires_declared_rate",
            "post_assist_loads_duplicate_across_owner_group_segments",
            "per_owner_phases_use_global_coarse_phase_barriers_not_chunk_overlap",
            "matrix_uses_shared_steady_K_pipeline_and_declared_inner_mapping",
            "scalar_group_scan_and_cross_core_flag_latency_not_charged",
            "capture_binary_identity_not_established",
        ],
    }
