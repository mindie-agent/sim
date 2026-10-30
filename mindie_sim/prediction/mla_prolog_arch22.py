"""Explicit arch22 MLA Prolog full-int8, BF16-KV source phases.

[Role] Execute the inspected arch22 full-quant/no-KV-quant branch on the common
  resource engine. The selected source is a 128-token chunk loop with four
  matrix products, RMS/dequant/rope vector consumers and real cache writes.
[Invariants] Selection is explicit and rejects other quant, cache, split-M or
  dequant-option branches. Matrix work keeps the int8/bf16 precision split.
  INT32 GM results are written once and read once by their actual dequant
  consumer; matrix-epilogue work is not applied again. A global workspace is
  ordinary GM traffic, not an L2 hit. Logical FLOPs and padded native work are
  reported separately. No duration or measured total selects a phase.
[Timeouts & Locks] Finite arithmetic over at most 128-token chunks; no I/O.
[Side-Effect Policy] Returns plans/metadata only; no device or file access.
"""
from dataclasses import dataclass
from mindie_sim.schema.common import ModelingError

from .mechanism import OperatorPlan, Work
from .program import KernelProgram, PipelineProgram, Stage


PROFILE = "mla_prolog_arch22_fullquant_a3925fd4"
CHUNK = 128
INT8 = 1
BF16 = 2
FP32 = 4

SOURCE = {
    "repository": "ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "branch": "arch22_full_quant_kv_no_quant",
    "program": {
        "path": "attention/mla_prolog/op_kernel/arch22/"
                "kernel_mla_prolog_split_n_arch22.h",
        "sha256": "ea32ad7c8e4af21684281e18f5b0bc16e9195550f3133ce820582e96dc772998",
        "lines": "628-675,716-775,788-815,828-861,867-890,1073-1090,"
                 "1983-2070,2096-2143,1305-1400,1666-1775",
    },
    "tiling": {
        "path": "attention/mla_prolog/op_host/mla_prolog_tiling.cpp",
        "sha256": "08a5a781a7fea8cdbba4739558de15ffd27777256bc94cfa3dd46d051c5a1bd2",
        "lines": "393-495,496-527,655-702",
    },
    "v3_registration": {
        "path": "attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp",
        "sha256": "66305b042eed4a53200570d9107ed2f6ad91951515f1b031767c9c0140a00011",
        "lines": "19-164,171-174",
    },
    "v3_dispatch": {
        "path": "attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp",
        "sha256": "da68a7a1536da460e8bacff60f970db637f56a1b28618040ef216dc6e52b3684",
        "lines": "103-112",
    },
    "services": [
        {
            "path": "attention/mla_prolog/op_kernel/arch22/service_rms_norm_arch22.h",
            "sha256": "5766c65f4c87b9ac6ba2c88be2e1794401bcd74d99dece7cd762b1d3983a76c6",
            "lines": "31-127",
        },
        {
            "path": "attention/mla_prolog/op_kernel/arch22/service_dequant_arch22.h",
            "sha256": "aa17aa13c99d8fb67e34a16173b76ac715a627fce3845555f57eee14ae498a51",
            "lines": "107-156,193-194",
        },
        {
            "path": "attention/mla_prolog/op_kernel/arch22/"
                    "service_rotary_position_embedding_arch22.h",
            "sha256": "2c25e5585764ca08b55d0e0623654a02143415b66cbdb18cd31a250a4d9558b5",
            "lines": "53-130,145-218",
        },
    ],
    "evidence": {
        "semantic_boundary": "explicit_source_profile_not_capture_binary_identity",
        "dispatch_basis": "declared_dtype_attributes_and_geometry",
        "measured_duration_used": False,
        "historical_binary_or_tiling_asserted": False,
    },
}

LIMITATIONS = (
    "source_arch22_profile_does_not_establish_a_retained_binary_or_tiling",
    "other_quant_cache_split_m_and_group_compute_branches_are_rejected",
    "int8_matrix_operands_use_logical_operand_payload_once_per_product",
    "l1_tile_reloads_and_fixpipe_instruction_issue_are_not_instruction_reconstructed",
    "workspace_streams_use_the_declared_default_gm_hbm_rate_without_l2_residency",
    "ub_l1_and_l0_capacity_is_not_reserved_because_the_loaded_phase_footprint_is_unavailable",
    "vector_ops_are_declared_elementwise_work_not_a_compiler_instruction_schedule",
    "within_chunk_cube_vector_overlap_is_preserved_chunks_are_serialized",
    "per_head_qcqr_dequant_streaming_is_coarsened_to_chunk_phase_barriers",
    "sin_cos_tail_duplicate_reads_follow_the_source_ceil_token_gather",
)

# Supported host-checked source domains from mla_prolog_tiling_check.cpp.
_HIDDEN = frozenset((1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192))
_HCQ = frozenset((1536, 2048))
_D = frozenset((128, 192))
_HCKV, _ROPE = 512, 64


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align(value, multiple):
    return _ceil_div(value, multiple) * multiple


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _view(op):
    metadata = dict(op.metadata)
    if metadata.get("fusion_role") != "mla_prolog":
        return metadata, None
    view = metadata.get("fusion_tensor_view")
    if not isinstance(view, dict):
        raise ModelingError("mla_prolog_arch22_requires_declared_mla_prolog_group")
    return metadata, view


def validate_declared_view(view):
    """Validate dtype/branch facts that do not depend on device geometry."""
    weights = _pair(view.get("projection_weight_dtypes"))
    if (view.get("token_dtype") != "int8" or weights != ("int8", "int8", "int8")
            or view.get("absorbed_weight_dtype") != "bf16"
            or view.get("norm_dtype") != "bf16"
            or view.get("cache_dtype") != "bf16"
            or view.get("query_norm_dtype") != "int8"
            or not isinstance(view.get("query_norm_output"), bool)):
        raise ModelingError(
            "mla_prolog_arch22_requires_int8_token_projection_and_bf16_uk_norm_cache")
    hidden = int(view.get("hidden", 0))
    heads = int(view.get("heads", 0))
    query_rank = int(view.get("query_rank", 0))
    if (hidden not in _HIDDEN or query_rank not in _HCQ
            or int(view.get("latent_rank", 0)) != _HCKV
            or int(view.get("rope_width", 0)) != _ROPE
            or int(view.get("nope_width", 0)) not in _D
            or not 1 <= heads <= 128):
        raise ModelingError("mla_prolog_arch22_geometry_outside_source_domain")
    # enableDequantOpt is selected by the int8 UQ/QR weight and N >= 8.
    if heads < 8:
        raise ModelingError(
            "mla_prolog_arch22_profile_covers_only_enableDequantOpt_n_ge_8")
    return {
        "hidden": hidden, "heads": heads, "query_rank": query_rank,
        "latent_rank": _HCKV, "nope": int(view["nope_width"]),
        "rope": _ROPE, "tokens": int(view.get("tokens", 0)),
    }


@dataclass(frozen=True)
class SourceGeometry:
    tokens: int
    hidden: int
    hcq: int
    hckv: int
    rope: int
    heads: int
    nope: int
    aic: int
    aiv: int
    ratio: int
    cq_single_n: int
    cq_blocks: int
    ckv_single_n: int
    ckv_blocks: int
    qcqr_single_n: int
    qcqr_blocks: int
    qn_single_heads: int
    qn_blocks: int

    @property
    def chunks(self):
        return _ceil_div(self.tokens, CHUNK)

    @property
    def chunk_sizes(self):
        return (self.tokens_at(index) for index in range(self.chunks))

    def tokens_at(self, index):
        return min(CHUNK, self.tokens - index * CHUNK)

    @property
    def chunk_runs(self):
        first = min(CHUNK, self.tokens)
        runs = [(first, 1, True)]
        full, tail = divmod(self.tokens - first, CHUNK)
        if full:
            runs.append((CHUNK, full, False))
        if tail:
            runs.append((tail, 1, False))
        return tuple(runs)


def geometry(view, device):
    """Reproduce FillMatmul1..4Tiling and the vector-block ownership."""
    sizes = validate_declared_view(view)
    tokens = sizes["tokens"]
    if tokens < 1:
        raise ModelingError("mla_prolog_arch22_requires_positive_tokens")
    aic = int(device.compute_units)
    aiv = int(device.resource_units.get("vector", aic))
    # GenTilingKey rejects every quant mode in CV1:1, even though the
    # templated device code also contains CV1:1 branches.
    if aic < 1 or aiv != aic * 2:
        raise ModelingError("mla_prolog_arch22_full_quant_requires_aiv_aic_ratio_2")
    # FillMatmul1Tiling: aligned32, minimum 64.
    cq_single = max(_align(_ceil_div(sizes["query_rank"], 32 * aic) * 32, 1), 64)
    cq_blocks = _ceil_div(sizes["query_rank"], cq_single)
    # FillMatmul2Tiling: the inspected source uses fixed N=64 with >=9 AIC.
    if aic >= 9:
        ckv_single = 64
        if (sizes["latent_rank"] + sizes["rope"]) % ckv_single:
            raise ModelingError(
                "mla_prolog_arch22_ckv_kr_n64_requires_divisible_head_width")
        ckv_blocks = (sizes["latent_rank"] + sizes["rope"]) // ckv_single
    else:
        ckv_single = _align(
            _ceil_div(sizes["latent_rank"] + sizes["rope"], 32 * aic) * 32, 1)
        ckv_blocks = _ceil_div(sizes["latent_rank"] + sizes["rope"], ckv_single)
    # FillMatmul3Tiling: enableDequantOpt aligns Qc+Qr per core.
    ori_m = sizes["heads"] * (sizes["nope"] + sizes["rope"])
    align = sizes["nope"] + sizes["rope"]
    qcqr_single = _ceil_div(ori_m, align * aic) * align
    qcqr_blocks = _ceil_div(ori_m, qcqr_single)
    # FillMatmul4Tiling: N/M axis over AIC.
    qn_single = _ceil_div(sizes["heads"], aic)
    qn_blocks = _ceil_div(sizes["heads"], qn_single)
    return SourceGeometry(
        tokens, sizes["hidden"], sizes["query_rank"], sizes["latent_rank"],
        sizes["rope"], sizes["heads"], sizes["nope"], aic, aiv, aiv // aic,
        cq_single, cq_blocks, ckv_single, ckv_blocks,
        qcqr_single, qcqr_blocks, qn_single, qn_blocks)


def _vector_tokens(g, tokens):
    """ComputeAivOffset's front/back token split over the 128-step AIV window."""
    fixed = min(CHUNK, g.aiv)
    front = tokens % fixed
    back = tokens // fixed
    return tuple(back + 1 if index < front else back for index in range(fixed))


def owners(g, tokens):
    """Exact active owners for every source stage of one chunk."""
    vector = tuple(index for index, count in enumerate(_vector_tokens(g, tokens)) if count)
    qc_vector = tuple(range(min(g.ratio * g.qcqr_blocks, g.aiv)))
    qn_vector = tuple(range(min(g.ratio * g.qn_blocks, g.aiv)))
    return {
        "cq": tuple(range(g.cq_blocks)),
        "ckvkr": tuple(range(g.ckv_blocks)),
        "qcqr": tuple(range(g.qcqr_blocks)),
        "qn": tuple(range(g.qn_blocks)),
        "rms": vector,
        "dequant_qc": qc_vector,
        "rope_qr": qc_vector,
        "all_vector": tuple(range(g.aiv)),
    }


def _work(*, read=0, write=0, vector=0, precision=None, flops=0):
    matrix = {precision: float(flops)} if precision and flops else {}
    resources = {key: float(value) for key, value in
                 (("hbm", read + write), ("vector", vector)) if value}
    return Work(matrix, resources)


def _phase_io(sizes, tokens, query_norm_output, g=None, initial=False):
    h, hcq = sizes["hidden"], sizes["query_rank"]
    hckv, rope = sizes["latent_rank"], sizes["rope"]
    heads, nope = sizes["heads"], sizes["nope"]
    width = nope + rope
    dq_scale = tokens * FP32
    query_norm_scale = FP32 if query_norm_output else 0
    aiv = g.aiv if g else 1
    qc_cores = g.qcqr_blocks if g else 1
    # The pinned caller supplies no smooth_scales_cq. Global gamma/channel
    # vectors are copied once by every AIV, not once per token chunk.
    params = (hcq * (FP32 + BF16) + hckv * BF16
              + (hckv + rope) * FP32) * aiv
    rows = [
        ("cq",
         tokens * h * INT8 + hcq * h * INT8,
         tokens * hcq * FP32, 0, "int8", 2 * tokens * hcq * h, "cube", "cq", ()),
        ("ckvkr",
         tokens * h * INT8 + (hckv + rope) * h * INT8,
         tokens * (hckv + rope) * FP32, 0, "int8",
         2 * tokens * (hckv + rope) * h, "cube", "ckvkr", ("cq",)),
        ("sincos_qr",
         _ceil_div(tokens, 2) * 2 * qc_cores * rope * BF16 * 2,
         0, 0, None, 0, "vector", "dequant_qc", ("params",) if initial else ()),
        ("rms_cq",
         tokens * hcq * FP32 + dq_scale,
         tokens * hcq * INT8 + tokens * 32 + tokens * query_norm_scale,
         tokens * hcq * 10 + tokens * 4, None, 0, "vector", "rms", ("cq", "sincos_qr")),
        ("scale_gather", tokens * 32 * aiv, 0, 0, None, 0,
         "vector", "all_vector", ("rms_cq",)),
        ("rms_ckv",
         tokens * hckv * FP32 + dq_scale + tokens * rope * BF16 * 2 + tokens * 8,
         tokens * hckv * BF16, tokens * hckv * 8 + tokens * 4,
         None, 0, "vector", "rms", ("ckvkr", "scale_gather")),
        ("rope_kr",
         tokens * rope * FP32 + tokens * 8,
         tokens * rope * BF16, tokens * rope * 6, None, 0,
         "vector", "rms", ("rms_ckv",)),
        ("qcqr",
         tokens * hcq * INT8 + heads * width * hcq * INT8,
         tokens * heads * width * FP32, 0, "int8",
         2 * tokens * heads * width * hcq, "cube", "qcqr", ("ckvkr", "scale_gather")),
        ("dequant_qc",
         tokens * heads * nope * FP32 + heads * nope * FP32,
         tokens * heads * nope * BF16, tokens * heads * nope * 3,
         None, 0, "vector", "dequant_qc", ("qcqr", "rope_kr")),
        ("rope_qr",
         tokens * heads * rope * FP32 + heads * rope * FP32 * 2,
         tokens * heads * rope * BF16, tokens * heads * rope * 6,
         None, 0, "vector", "rope_qr", ("dequant_qc",)),
        ("qn",
         tokens * heads * nope * BF16 + heads * nope * hckv * BF16,
         tokens * heads * hckv * BF16, 0, "bf16",
         2 * tokens * heads * nope * hckv, "cube", "qn", ("dequant_qc",)),
    ]
    if initial:
        rows.insert(0, ("params", params, 0, 0, None, 0, "vector", "all_vector", ()))
    return rows


def phase_io(view, tokens, *, g=None, initial=False):
    """Explicit read/write/vector/matrix work for one chunk of ``tokens``."""
    sizes = validate_declared_view(view)
    if tokens < 1 or tokens > CHUNK:
        raise ModelingError("mla_prolog_arch22_chunk_size_invalid")
    return {
        name: {"read": read, "write": write, "vector": vector,
               "matrix": {precision: float(flops)} if precision else {},
               "role": role, "key": key, "after": list(after)}
        for name, read, write, vector, precision, flops, role, key, after
        in _phase_io(sizes, tokens, view.get("query_norm_output"), g, initial)
    }


def phase_work(view, g, tokens, *, initial=False):
    """One chunk's declared source Work, including every GM scratch edge."""
    if not (g.hidden and g.hcq):
        raise ModelingError("mla_prolog_arch22_geometry_incomplete")
    return tuple(
        (name, _work(read=read, write=write, vector=vector,
                     precision=precision, flops=flops), role, key, after)
        for name, read, write, vector, precision, flops, role, key, after
        in _phase_io(
            {"hidden": g.hidden, "query_rank": g.hcq, "latent_rank": g.hckv,
             "rope": g.rope, "heads": g.heads, "nope": g.nope},
            tokens, view.get("query_norm_output"), g, initial)
    )


def plan_work(view, g):
    total = Work()
    for tokens, count, initial in g.chunk_runs:
        for _, work, _, _, _ in phase_work(view, g, tokens, initial=initial):
            total += work.scaled(count)
    return total


def source_chunk_io(view, tokens=CHUNK, *, g=None, initial=False):
    """Exact payload and matrix work for one real source chunk."""
    return phase_io(view, tokens, g=g, initial=initial)


def source_chunk_work(view, tokens=CHUNK, *, g=None, initial=False):
    """Exact Work for one real source chunk, including fixed weight reads."""
    return {
        name: _work(read=item["read"], write=item["write"],
                    vector=item["vector"],
                    precision=next(iter(item["matrix"]), None),
                    flops=next(iter(item["matrix"].values()), 0))
        for name, item in source_chunk_io(view, tokens, g=g, initial=initial).items()
    }


def _owner_rules(g):
    return {
        "cq": {"aic": list(range(g.cq_blocks)),
               "single_n": g.cq_single_n,
               "rule": "max(align32(ceil(Hcq/(32*AIC))*32),64)"},
        "ckvkr": {"aic": list(range(g.ckv_blocks)), "single_n": g.ckv_single_n,
                  "rule": "AIC>=9: N=64, blocks=(512+64)/64"},
        "qcqr": {"aic": list(range(g.qcqr_blocks)), "single_n": g.qcqr_single_n,
                 "rule": "align whole (D+Dr) over AIC"},
        "qn": {"aic": list(range(g.qn_blocks)), "single_heads": g.qn_single_heads,
               "rule": "ceil(N/AIC) heads per active AIC"},
        "vector": {"aiv": g.aiv, "rule": "min(128,AIV) front/back token split"},
        "qc_dequant_vector": {"aiv": list(range(min(g.ratio * g.qcqr_blocks, g.aiv))),
                              "rule": "one AIV pair per QcQr AIC"},
        "qn_vector": {"aiv": list(range(min(g.ratio * g.qn_blocks, g.aiv))),
                      "rule": "one AIV pair per Qn AIC"},
    }


def describe(view, g):
    """A geometry-specific ledger for tests and diagnostics."""
    per_chunk = source_chunk_work(view, g.tokens_at(0), g=g, initial=True)
    total = plan_work(view, g)
    phases = []
    for tokens, count, initial in g.chunk_runs:
        for name, work, role, keys, after in phase_work(view, g, tokens, initial=initial):
            phases.append({
                "chunk_tokens": tokens, "chunk_count": count, "initial": initial,
                "name": name, "role": role,
                "after": list(after),
                "matrix": dict(work.matrix),
                "hbm": work.resources.get("hbm", 0),
                "vector": work.resources.get("vector", 0),
            })
    return {
        "profile": PROFILE, "chunks": g.chunks,
        "chunk_runs": [{"tokens": t, "count": n, "initial": first}
                       for t, n, first in g.chunk_runs],
        "owners": _owner_rules(g),
        "first_chunk": {name: {"matrix": dict(work.matrix),
                             "hbm": work.resources.get("hbm", 0),
                             "vector": work.resources.get("vector", 0)}
                      for name, work in per_chunk.items()},
        "total": {"matrix": dict(total.matrix),
                  "hbm": total.resources.get("hbm", 0),
                  "vector": total.resources.get("vector", 0)},
        "phases": phases, "limitations": list(LIMITATIONS),
    }


def _owner_weights(g, tokens, name, key):
    """Preserve source N/head/token tails instead of averaging over owners."""
    if key == "all_vector":
        return tuple((index, 1) for index in range(g.aiv))
    if key == "rms":
        return tuple((index, n) for index, n in enumerate(_vector_tokens(g, tokens)) if n)
    if key in {"cq", "ckvkr", "qcqr", "qn"}:
        total, width = {
            "cq": (g.hcq, g.cq_single_n),
            "ckvkr": (g.hckv + g.rope, g.ckv_single_n),
            "qcqr": (g.heads * (g.nope + g.rope), g.qcqr_single_n),
            "qn": (g.heads, g.qn_single_heads),
        }[key]
        return tuple((index, min(width, total - index * width))
                     for index in range(_ceil_div(total, width)))
    heads_per_core = g.qcqr_single_n // (g.nope + g.rope)
    if name == "sincos_qr":
        return tuple((owner, 1) for owner in range(g.qcqr_blocks * 2))
    rows = []
    for core in range(g.qcqr_blocks):
        heads = min(heads_per_core, g.heads - core * heads_per_core)
        for sub in range(2):
            # Qc splits its columns; Qr splits tokens (odd AIV gets tail).
            count = heads * ((tokens + sub) // 2 if name == "rope_qr" else 1)
            if count:
                rows.append((core * 2 + sub, count))
    return tuple(rows)


def selected_plan(op, device):
    """Return the source plan only for the explicitly selected profile."""
    metadata, view = _view(op)
    boundary = metadata.get("fusion_boundary", {})
    if view is None or boundary.get("traffic_profile") != PROFILE:
        return None
    g = geometry(view, device)

    def program(context):
        cache = {}

        def tile(index):
            tokens, initial = g.tokens_at(index), index == 0
            key = tokens, initial
            if key not in cache:
                stages = []
                for name, work, role, owner_key, after in phase_work(
                        view, g, tokens, initial=initial):
                    weights = _owner_weights(g, tokens, name, owner_key)
                    denominator = sum(weight for _, weight in weights)
                    children = tuple(
                        Stage(str(owner), context.bind(work.scaled(weight / denominator)),
                              placement=context.placement(role, (owner,)))
                        for owner, weight in weights)
                    stages.append(Stage(name, after=after, program=KernelProgram(children)))
                cache[key] = tuple(stages)
            return cache[key]

        return PipelineProgram(g.chunks, 1, tile)

    return OperatorPlan(plan_work(view, g), program)


def boundary_metadata(declaration, members, outputs):
    """Source ledger retained on the selected native-fusion boundary."""
    view = declaration["tensor_view"]
    sizes = validate_declared_view(view)
    tokens = sizes["tokens"]
    full, tail = divmod(tokens, CHUNK)
    chunk_runs = [(CHUNK, full)] if full else []
    if tail:
        chunk_runs.append((tail, 1))
    total = Work()
    for chunk_tokens, count in chunk_runs:
        for work in source_chunk_work(view, chunk_tokens).values():
            total += work.scaled(count)
    return {
        "profile": PROFILE,
        "source": SOURCE,
        "selected_domain": {
            "token": "int8", "projection_weights": "int8", "absorbed_weight": "bf16",
            "norm_gamma": "bf16", "kv_cache": "bf16", "query": "bf16",
            "query_rope": "bf16", "query_norm": "int8", "query_norm_scale": "fp32",
            "query_norm_output": bool(view.get("query_norm_output")),
            "cache_kind": "dense_bf16", "split_m": False, "group_compute": False,
            "enable_dequant_opt": True, "heads": sizes["heads"],
            "hidden": sizes["hidden"], "query_rank": sizes["query_rank"],
            "latent_rank": sizes["latent_rank"], "nope": sizes["nope"],
            "rope": sizes["rope"], "chunk": CHUNK,
        },
        "branch": {
            "op": "MlaPrologV3", "revision": SOURCE["revision"],
            "weight_quant_mode": "FULL_QUANT", "kv_cache_quant_mode": "NO_QUANT",
            "arch": "arch22", "dequant_opt": "n>=8_int8_uq_qr",
            "selection": "explicit_dtype_and_geometry",
            "duration_used": False,
        },
        "program": {
            "kind": "bounded_lazy_pipeline",
            "chunks": _ceil_div(tokens, CHUNK),
            "chunk_runs": [{"tokens": size, "count": count}
                           for size, count in chunk_runs],
            "slots": 1,
            "work_binding": "unbound_geometry_payload; replaced_on_recipe_prepare",
            "work": {"matrix": dict(total.matrix),
                     "hbm": total.resources.get("hbm", 0),
                     "vector": total.resources.get("vector", 0)},
            "per_128_token_chunk": {
                name: {"matrix": dict(work.matrix),
                       "hbm": work.resources.get("hbm", 0),
                       "vector": work.resources.get("vector", 0)}
                for name, work in source_chunk_work(view, CHUNK).items()},
            "fusion_semantic_ids": [member.op_id for member in members],
            "fusion_semantic_recipes": [
                {"op_id": member.op_id, "recipe": member.recipe,
                 "family": member.family, "local_shape": list(member.local_shape)}
                for member in members],
            "fusion_external_outputs": list(outputs),
            "dependencies": {
                "ckvkr": ["cq"], "rms_cq": ["cq", "sincos_qr"],
                "scale_gather": ["rms_cq"],
                "rms_ckv": ["ckvkr", "scale_gather"], "rope_kr": ["rms_ckv"],
                "qcqr": ["ckvkr", "scale_gather"],
                "dequant_qc": ["qcqr", "rope_kr"],
                "rope_qr": ["dequant_qc"], "qn": ["dequant_qc"],
            },
            "owners": {
                "cq": "max(align32(ceil(Hcq/(32*AIC))*32),64)",
                "ckvkr": "AIC>=9: N=64; blocks=(Hckv+Dr)/64",
                "qcqr": "ceil(N*(D+Dr)/((D+Dr)*AIC))*(D+Dr)",
                "qn": "ceil(N/AIC) heads per active AIC",
                "vector": "min(128,AIV) front/back token split",
                "qc_dequant": "min(2*AIC,AIV) one AIV pair per QcQr AIC",
            },
            "matrix_precision": "int8_cq_ckvkr_qcqr; bf16_qn",
            "int32_scratch": {
                "cq_write": sizes["query_rank"] * FP32,
                "ckvkr_write": (sizes["latent_rank"] + sizes["rope"]) * FP32,
                "qcqr_write": sizes["heads"] * (sizes["nope"] + sizes["rope"]) * FP32,
                "consumer_reads_once": True,
            },
            "matrix_epilogue_policy": "not_applied; dequant counted in RMS/Qc phases once",
            "dequant_traffic_policy": "int32_result_read_once_at_actual_consumer",
            "launch_count": 1,
        },
        "timing_basis": "explicit_source_phase_dag_not_sum_of_kernel_durations",
        "fidelity": {
            "kind": "source_phase_pipeline_dag",
            "compute": "exact_int8_and_bf16_2MNK",
            "workspace": "declared_gm_bytes_at_default_hbm_rate",
            "within_chunk_overlap": True,
            "inter_chunk_overlap": "serialized_by_workspace_slots",
            "padded_native_work": "not_introduced_by_this_model",
            "logical_flops_distinct_from_bytes": True,
        },
        "limitations": list(LIMITATIONS),
    }
