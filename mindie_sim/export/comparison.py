"""Two-level semantic bipartite alignment, L1/L2 breakdown, Sheet 4 cards & Agent Brief.

[Role] Pure functional export comparison & rendering logic over ``ExportBundle``:
  (1) Two-Level Semantic-Atom Provenance Bipartite Alignment (``align_before_after_operators``)
      handling ``1->1`` (including ``a -> d = f_new + a``), ``N->1`` fusions (contiguous
      and non-adjacent), staggered ``1->0`` / ``0->1`` strategy switches, and ``M->N``
      subgraph re-association blocks without any static fusion-source tables;
  (2) Two-level L1/L2 macro-module breakdown conserving BOTH ``Step_ms`` and
      ``us / (tok*die)``;
  (3) 100%-conserved Sheet 4 ``Pending_Kernel_Details`` cards ranked by
      ``Δ(us / tok*die)`` with composed ``KernelSynthesisSpec`` formulas & SRAM specs;
  (4) Pure rendering 3-track Ascend guidance, co-domain parallelism display, and
      turnkey ``< 8 KB`` ``agent_brief.json`` + Markdown executive digest.
[Invariants] Zero solver mutations, zero ``_KNOWN_FUSION_SOURCES`` or model-name
  branches (AG013), zero ``[待 Agent 补充]`` blanks, zero ``"4D"`` jargon, strict
  zero-saving status invariant, and exact dual conservation across fixed-batch
  and variable-``(Parallel, B)`` Iso-SLA comparisons.
[Timeouts & Locks] O(Unique Archetype Ops) in-memory passes; atomic ``os.replace``
  only inside ``export_agent_brief``; no locks or device contexts.
[Side-Effect Policy] Pure in-memory except ``export_agent_brief`` which writes
  only its caller-requested JSON path.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from .kernel_catalog import classify_macro_module, extract_semantic_atoms, strip_op_prefix
from .models import (
    AGENT_BRIEF_SCHEMA,
    BINDING_STATE_BOUND,
    BINDING_STATE_UNBOUND_FORMULA,
    COLLECTIVE_FAMILIES,
    DEFAULT_TOP_OPTIMIZE_KERNELS,
    GIB_BYTES,
    KERNEL_STATUS_AT_FLOOR,
    L1_MOD_RESIDUAL_MEM,
    MACRO_L1_HIERARCHY,
    MACRO_MOD_DRAFT_LAYERS,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_VERIFY_HEAD,
    MAX_SOURCE_REF_SUMMARY_LEN,
    MIN_DENOM_EPS,
    MODE_MODELING,
    OVERLAP_EXPOSED_EPS_US,
    PENDING_KIND_TO_IMPLEMENT,
    PENDING_KIND_TO_OPTIMIZE,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    TRACK_KERNEL,
    US_PER_MS,
    ActionRoadmapRow,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    NewKernelBacklogItem,
    OperatorExportRow,
    PendingKernelItem,
    format_binding_badge,
)

#: Epsilon (ms) for treating an exposed Before->After delta as non-zero.
CONSERVATION_EPS_MS = 1e-6
#: Epsilon (us) for zero-saving Hard Status Invariant.
ZERO_SAVING_EPS_US = 1e-6
#: Maximum number of compact kernel tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_KERNEL_TASKS = 5
#: Maximum number of framework/config tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_FRAMEWORK_TASKS = 3
#: Maximum number of synthesis specs & audit triggers embedded in ``agent_brief.json``.
BRIEF_MAX_TRIGGERS = 4
BRIEF_MAX_SPECS = 4
BRIEF_MAX_CAPTURE_SIZES = 8
SLICE_MAX_EXPLICIT_LAYERS = 4
#: Maximum character length for compact string fields inside ``agent_brief.json``.
BRIEF_TEXT_LIMIT = 72
#: Minimum single-op gap (us) to call out gap-vs-L* in Sheet 2 notes.
ACTION_GAP_CALLOUT_US = 1.0
#: Byte budget for ``agent_brief.json`` (< 8 KB) and pretty-print threshold.
AGENT_BRIEF_MAX_BYTES = 8192
AGENT_BRIEF_PRETTY_LIMIT_BYTES = 6400
#: Minimum TP and HBM headroom (GiB) for parallelism co-design callouts.
CODESIGN_MIN_TP = 4
CODESIGN_MIN_HEADROOM_GIB = 4.0
#: Minimum display saving (ms) and roadmap row caps for Sheet 1 Block 4.
ROADMAP_MIN_SAVE_MS = 0.05
MAX_ROADMAP_FW_ITEMS = 3
MAX_ROADMAP_KERNEL_ITEMS = 5
MAX_FW_DEDUP_KEY_LEN = 40
#: Card slug character cap.
CARD_SLUG_MAX_LEN = 44
#: Minimum cluster members for M->N / N->1 classification.
MIN_GROUP_SIZE = 2

_SCOPE_PREFIXES = ("attention.", "ffn.", "shared.", "final.")
_OPTIMIZE_DESIGN_HINTS: dict[str, str] = {
    "link_bound": "切换超节点 AIV UB-Direct / HCCS One-Shot 集合通信原语与 MC2 通算融合，消除 Ring-HCCL 启动开销",
    "memory_bound": "紧凑动态 M-Tiling / L1-L2 权重驻留 / 算子融合降访存与量化压缩 (MXFP8+MXFP4)，逼近 HBM 带宽极限 L*",
    "cube_bound": "AIC 核内 M/N/K 分块与 NZ 布局对齐 / L0C Ping-Pong 双缓冲，提升 Cube AIC 利用率",
    "vector_bound": "AIV 向量算子融合与 UB 双缓冲流水编排，减少中间 HBM 回写与同步开销",
    "launch_bound": "开启 ACLGraph FULL_DECODE_ONLY 全图捕获与算子合并，消除 Host 发射气泡",
}
_DEV_BADGE_TOKENS = ("[需新开发", "[需扩展", "[待新开发", "[需算子深度调优")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def format_parallelism_codomain_summary(bundle: ExportBundle) -> tuple[str, str]:
    """Render §3 coherent DCP<->TP co-domain banner and per-module parallel breakdown."""
    mem = bundle.memory_ledger
    sp_zh = "开启" if bundle.sequence_parallel else "关闭"
    sp_en = "ON" if bundle.sequence_parallel else "OFF"
    if mem.dcp > 1 and mem.dcp == mem.tp:
        short = f"DP{mem.dp} · TP{mem.tp} (共域 DCP{mem.dcp}) · EP{mem.ep} · SP({sp_zh})"
        dcp_tag = f"DCP={mem.dcp} (与 TP 共域)"
        mla_dom = f"MLA: TP{mem.tp} + 共域 KV-DCP{mem.dcp}"
    elif mem.dcp > 1:
        short = f"DP{mem.dp} · TP{mem.tp} · DCP{mem.dcp} · EP{mem.ep} · SP({sp_zh})"
        dcp_tag = f"DCP={mem.dcp}"
        mla_dom = f"MLA: TP{mem.tp} + KV-DCP{mem.dcp}"
    else:
        short = f"DP{mem.dp} · TP{mem.tp} · EP{mem.ep} · SP({sp_zh})"
        dcp_tag = f"DCP={mem.dcp}"
        mla_dom = f"MLA/Attn: Local-TP{mem.local_tp}"
    lin_dom = f"Linear/Dense: TP{mem.tp}" + (" + SP" if bundle.sequence_parallel else "")
    moe_dom = f"Routed MoE: EP{mem.ep} | Shared MoE: TP{mem.shared_tp}"
    detail = (
        f"DP={mem.dp}, TP={mem.tp}, EP={mem.ep}, PP={mem.pp}, {dcp_tag}, "
        f"PCP={mem.pcp}, Local-TP={mem.local_tp}, SP={sp_en} | "
        f"{mla_dom} | {lin_dom} | {moe_dom}"
    )
    return short, detail


def build_op_action_guidance(
    op: OperatorExportRow,
    *,
    is_a5_ub: bool,
    phase: str = "decode",
    hardware_sku: str = "",
) -> tuple[str, str, str, str]:
    """Pure renderer returning concrete 3-track ``(config, kernel, framework, badge)``."""
    suf = strip_op_prefix(op.op_id).lower()
    fam = op.family.lower()
    kern_l = op.ascend_kernel.lower()
    is_decode = phase.lower() == "decode"
    sku_low = hardware_sku.lower()
    is_a2 = (not is_a5_ub) and any(tok in sku_low for tok in ("910b", "a2"))
    nz_flag = '"enable_weight_nz_layout": true, ' if not is_a5_ub else ""
    kv_flag = "--kv-cache-dtype fp8_e4m3" if is_a5_ub else "--kv-cache-dtype int8"
    prefill_caveat = (
        " (注意: Prefill 双流 micro-batch 若出现 AIV 争抢，优先切 A5 CCU 硬件引擎)"
        if (not is_decode and is_a5_ub) else ""
    )

    if op.kernel_synthesis_spec is not None:
        spec = op.kernel_synthesis_spec
        formula_blk = spec.format_formula_block()
        src = op.source_ref or "vllm-ascend/vllm_ascend/ops/"
        return (
            "配合 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' (静态预分配保 ACLGraph 零图断裂)",
            f"[{spec.abi_tier}] {spec.kernel_name}: {formula_blk}",
            f"[框架接线] {src} 替换原 {len(spec.constituent_op_ids)} 算子子图为单 CustomOp 并消除中间拷贝",
            f"[{spec.status_tag}]",
        )
    if fam == "mhc" or ".hc_" in f".{suf}":
        return (
            "开启 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'; 无现成 CANN mHC 开关，需新融合算子",
            "[需新写算子] AscendC/Triton 融合 FusedMHCGatedResidualMix + FusedMHCCombine，将 [B,4*D] 驻留 UB/L1 消除多次 HBM 往返",
            "[框架改造] vllm_ascend/ops/mhc.py 替换多步 Eager 链路为 2 个 CustomOp，消除 .item()/动态分配",
            "[算子新写 + 框架 Pass 替换]",
        )
    if fam == "attn_res_prefix" or "attn_res" in suf:
        return (
            "开启 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' 消除小算子 Host 发射气泡",
            "[需新写算子] Triton-Ascend/AscendC 实现 BlockAttnResFusedPrefix，单次 UB FP32 流水完成 Norm+Score+Softmax+加权残差聚合",
            "[框架改造] vllm_ascend/ops/attn_res.py 注册静态 block-history 指针表并接入 ACLGraph 捕获",
            "[算子新写 + 框架图捕获适配]",
        )
    if "mlaprolog" in kern_l or suf.startswith("mla_prolog"):
        return (
            "export VLLM_ASCEND_ENABLE_MLAPO=1 && --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'",
            "[需算子调优] 优化 aclnnMlaPrologV3 在 head pad 与 MXFP8/BF16 下 L1 权重驻留与 Cube/AIV 流水配比（目标 >=78% L*）",
            "[框架改造] vllm_ascend/attention/mla_v1.py 打开 Target 与 Draft MLA 全量化模式 MLAPO 路由并预排布 NZ 权重",
            "[配置开启 MLAPO + 算子调优]",
        )
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        ccu_txt = "硬件 CCU 调度 (CCU_SCHED/CCU_MS) + " if is_a5_ub else ""
        return (
            f'export HCCL_OP_EXPANSION_MODE="AIV" VLLM_ASCEND_ENABLE_FUSED_MC2=1 && '
            f'--additional-config \'{{"enable_fused_mc2": 2, "enable_flashcomm1": true}}\'{prefill_caveat}',
            f"[需算子调优] MC2 通算融合切 {ccu_txt}AIV 展开并优化小 Batch M-tiling 掩盖通信延迟（目标 >=78% L*）",
            "[框架改造] vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py 融合线性层与 SP Gather/ReduceScatter",
            "[配置开 MC2/CCU-AIV + 框架通算融合]",
        )
    if op.task_type == "HCCL" or fam in COLLECTIVE_FAMILIES:
        if is_a5_ub:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512 + A5 硬件 CCU UB-Direct'
            comm_kern = "[需算子调优] 超节点 <=8MiB 小包切 AIV UB-Direct / CCU One-Shot 模板，消除 Host/AI_CPU 启动开销（目标 >=75% Link L*）"
        elif is_a2:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_BUFFSIZE=512 (A2 机内 HCCS / 跨机 RoCE：优先机内 TP<=8 + 跨机 DP/EP)'
            comm_kern = "[需算子调优] A2 机内切 AIV HCCS One-Shot 降小包启动延迟，跨机 RoCE 用 DP/EP 替代跨机 TP AllReduce"
        else:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512 (A3 双 Die 超节点 HCCS)'
            comm_kern = "[需算子调优] A3 双 Die 超节点启用 <=8MiB 小包 AIV HCCS One-Shot 模板（目标 >=75% Link L*）"
        sp_tip = "；小 Batch Decode 可设 additional_config.enable_sp=false 消除负收益 SP" if "sp_" in suf else ""
        return (
            f"{comm_env}{sp_tip}{prefill_caveat}",
            comm_kern,
            "[框架改造] vllm_ascend/attention/context_parallel/mla_cp.py 将通信下发至独立流并用 npu.Event 异步重叠",
            "[通信切 CCU/AIV + 框架异步掩盖]" if is_a5_ub else "[通信切 AIV HCCS + 框架异步掩盖]",
        )
    if "lightning" in kern_l or fam == "indexer":
        return (
            f"export VLLM_ASCEND_ENABLE_MLAPO=1 && {kv_flag} (开启量化 KV/Indexer K-Cache 降访存)",
            "[需算子调优] 优化 aclnnQuantLightningIndexer 向量化 Index-K 预取与 UB 双缓冲（目标 >=78% HBM L*）",
            "[框架改造] vllm_ascend/attention/sfa_v1.py 保持静态 TopK 缓冲并异步流水编排 Indexer 与 DCP AllGather",
            "[算子调优 + 框架流水编排]",
        )
    if fam == "grouped_gemm" or "megamoe" in op.recipe.lower() or "dispatch_ffn_combine" in kern_l:
        quant_gmm = (
            "MXFP4 group=32 GroupedMatmul / MegaMoE 紧凑动态 M-tiling + 激活量化融合"
            if is_a5_ub else
            "W4A8 INT4 FRACTAL_NZ aclnnGroupedMatmulV4 / A3 MegaMoE + aclnnSwiGluQuant 紧凑动态 M-tiling"
        )
        return (
            f'export VLLM_ASCEND_ENABLE_FUSED_MC2=1 HCCL_BUFFSIZE_EP=2048 && '
            f'--additional-config \'{{{nz_flag}"enable_fused_mc2": 3, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true}}\'',
            f"[需算子调优] {quant_gmm}，消除稀疏专家 M-padding 浪费（目标 >=85% L*）",
            "[框架改造] vllm_ascend/ops/fused_moe/moe_comm_method.py 接线 MegaMoE + MSO Stream 1 共享专家异步掩盖",
            "[配置开 MegaMoE/MSO + 算子 Tiling 调优]",
        )
    if "recover_ssm" in suf or "recurrent" in suf or fam in {"recurrent_scan", "recurrent_vector"}:
        return (
            "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' (保证投机校验循环无 Host 同步气泡)",
            "[需算子调优] 优化 Persistent Recurrent 核内 FP32 状态驻留与按接收槽位增量写回（目标 >=78% HBM L*）",
            "[框架改造] vllm_ascend/worker/model_runner_v1.py 投机接收路径原地更新状态槽，避免整批回滚拷贝",
            "[算子调优 + 框架原地状态提交]",
        )
    if op.is_overlapped:
        return (
            '--additional-config \'{"multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true}\'',
            "[现成可用] 副流算子已被主计算流 100% 掩盖，无需额外算子改动",
            "[框架配置] 维持独立副流 npu.Event 同步，避免插入 host synchronize 破坏掩盖",
            "[已有特效: 副流 100% 掩盖]",
        )
    return (
        "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' (配合 msmodelslim QuaRot 生产量化)",
        f"[需算子调优] 针对 {op.bound_display or op.dominant_bound} 优化分块与双缓冲（目标 >=78% L*）",
        "[框架改造] 保持静态 shape 与预分配 workspace，避免 ACLGraph 图断裂与冗余 TensorMove",
        "[算子调优 + ACLGraph 保图]",
    )


def build_parallelism_codesign_note(
    before_bundle: ExportBundle, after_bundle: ExportBundle,
) -> str:
    """Emit concise Parallelism & Macro-Wall Co-Design callout."""
    mem = after_bundle.memory_ledger
    tp = int(mem.tp)
    headroom_gib = min((st.headroom_bytes for st in mem.stage_budgets), default=0.0) / float(GIB_BYTES)
    wall = after_bundle.macro_wall_trigger
    wall_prefix = ""
    if (
        wall is not None
        and wall.wall_kind in {"HBM_CAPACITY_WALL", "LATENCY_WALL", "LATENCY_COMPUTE_WALL"}
        and wall.recommended_actions
    ):
        wall_prefix = f"[{wall.wall_kind}] {wall.recommended_actions[0]}；"
    if tp < CODESIGN_MIN_TP:
        return wall_prefix.rstrip("；")
    comm_ms = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in before_bundle.operators
        if (op.task_type == "HCCL" or op.family in COLLECTIVE_FAMILIES)
        and "dispatch" not in op.op_id.lower() and "combine" not in op.op_id.lower()
    )
    after_comm_ms = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in after_bundle.operators
        if (op.task_type == "HCCL" or op.family in COLLECTIVE_FAMILIES)
        and "dispatch" not in op.op_id.lower() and "combine" not in op.op_id.lower()
    )
    if comm_ms < ROADMAP_MIN_SAVE_MS and not wall_prefix:
        return ""
    fit_hint = (
        f"单卡 HBM 余量 {headroom_gib:.1f} GiB 充裕，建议评估 Attention Local-TP=1/2 (Attn-DP) "
        "或小 Batch Decode 关闭 SP，直接从架构侧消除 TP/SP 集合通信与 AllGather 启动开销"
        if headroom_gib >= CODESIGN_MIN_HEADROOM_GIB else
        f"单卡 HBM 余量 {headroom_gib:.1f} GiB，优先采用 QuaRot 量化/KV Offload 释放 HBM 后再评估 Attn-DP 减通信"
    )
    return (
        f"{wall_prefix}[并行与通信协同建议] TP={tp} 非 EP 暴露通信 Before={comm_ms:.2f}ms -> "
        f"After={after_comm_ms:.2f}ms；{fit_hint}。"
    )


def build_actionable_roadmap(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    pending_items: Sequence[PendingKernelItem],
) -> tuple[ActionRoadmapRow, ...]:
    """Synthesize concrete 3-track Sheet 1 Block 4 roadmap directly from Before->After delta."""
    is_a5 = any(tok in after_bundle.hardware_sku.lower() for tok in ("950", "a5", "3510"))
    mem = after_bundle.memory_ledger
    rows: list[ActionRoadmapRow] = []

    comm_before = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in before_bundle.operators
        if op.task_type == "HCCL" or op.family in {"all_gather_matmul", "matmul_reduce_scatter"}
    )
    comm_after = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in after_bundle.operators
        if op.task_type == "HCCL" or op.family in {"all_gather_matmul", "matmul_reduce_scatter"}
    )
    comm_saved = max(0.0, comm_before - comm_after)
    if comm_saved >= ROADMAP_MIN_SAVE_MS or comm_before >= ROADMAP_MIN_SAVE_MS:
        ccu_env = (
            'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 '
            'HCCL_BUFFSIZE=512 (950DT Arch35 切硬件 CCU 调度 UB-Direct)'
            if is_a5 else
            'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512'
        )
        rows.append(ActionRoadmapRow(
            track=TRACK_CONFIG_ENV,
            track_label="[A] 已有特效与通信展开配置",
            title="集合通信展开切 AIV/CCU + MC2/FlashComm1 通算融合",
            expected_gain_ms=comm_saved,
            action_detail=(
                f"{ccu_env}; export VLLM_ASCEND_ENABLE_FUSED_MC2=1; "
                '--additional-config \'{"enable_fused_mc2": 2, "enable_flashcomm1": true}\''
            ),
            cross_ref="Sheet 2 通信/MC2 行 & Sheet 4",
        ))

    has_mla = any("mla" in op.submodule.lower() or "mlaprolog" in op.ascend_kernel.lower() for op in after_bundle.operators)
    has_mso = any(op.stream_index == 1 for op in after_bundle.operators)
    switch_parts = ['--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'']
    if has_mla:
        switch_parts.insert(0, "export VLLM_ASCEND_ENABLE_MLAPO=1")
    if has_mso:
        switch_parts.append('--additional-config \'{"multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true}\'')
    mso_hidden = sum((op.dur_us - op.exposed_dur_us) * op.repeat_count / US_PER_MS for op in after_bundle.operators if op.stream_index == 1)
    rows.append(ActionRoadmapRow(
        track=TRACK_CONFIG_ENV,
        track_label="[A] 已有特效与通信展开配置",
        title="开启 ACLGraph 全图捕获 + MLAPO / MSO 共享专家异步掩盖",
        expected_gain_ms=mso_hidden,
        action_detail="; ".join(switch_parts) + (f" (MSO 掩盖 {mso_hidden:.2f}ms 共享专家耗时)" if mso_hidden > 0.0 else ""),
        cross_ref="Sheet 1 Block 3 MSO 掩盖列",
    ))

    ckpt_ref = (
        f" | 推荐权重: {after_bundle.recommended_checkpoint['repo_id']}"
        if isinstance(after_bundle.recommended_checkpoint, Mapping) and after_bundle.recommended_checkpoint.get("repo_id")
        else ""
    )
    quant_tip = (
        f"当前权重={mem.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB/die (原生 BF16={mem.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB, "
        f"压缩 {mem.weight_compression_ratio:.2f}x); A5 生产建议 msmodelslim + QuaRot MXFP8+MXFP4 (--quantization ascend --kv-cache-dtype fp8_e4m3){ckpt_ref}"
        if is_a5 else
        f"当前权重={mem.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB/die (原生 BF16={mem.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB, "
        f"压缩 {mem.weight_compression_ratio:.2f}x); A2/A3 生产建议 msmodelslim W4A8_DYNAMIC/W8A8_DYNAMIC + C8 KV "
        f'--quantization ascend --kv-cache-dtype int8 --additional-config \'{{"enable_weight_nz_layout": true}}\'{ckpt_ref}'
    )
    rows.append(ActionRoadmapRow(
        track=TRACK_CONFIG_ENV,
        track_label="[A] 已有特效与通信展开配置",
        title="生产量化方案 (msmodelslim QuaRot + KV Cache 量化)",
        expected_gain_ms=0.0,
        action_detail=quant_tip,
        cross_ref="Sheet 3 Memory_Quant_Calculator",
    ))

    codesign = build_parallelism_codesign_note(before_bundle, after_bundle)
    if codesign:
        rows.append(ActionRoadmapRow(
            track=TRACK_CONFIG_ENV,
            track_label="[A] 已有特效与通信展开配置",
            title="并行策略与空间-时间协同建议 (Local-TP / Wall Trade-off)",
            expected_gain_ms=0.0,
            action_detail=codesign,
            cross_ref="Sheet 1 Block 1 并行配置",
        ))

    fw_seen: set[str] = set()
    for item in pending_items:
        if not item.framework_action:
            continue
        fw_key = item.framework_action.split("；", 1)[0][:MAX_FW_DEDUP_KEY_LEN]
        if fw_key in fw_seen:
            continue
        fw_seen.add(fw_key)
        rows.append(ActionRoadmapRow(
            track=TRACK_FRAMEWORK,
            track_label="[B] 推理框架层改造项 (vllm-ascend)",
            title=f"接线 {item.target_kernel.split('[', 1)[0].strip()} ({item.submodule})",
            expected_gain_ms=item.expected_gain_ms,
            action_detail=item.framework_action,
            cross_ref=f"Sheet 4 Card #{item.priority}",
        ))
        if len(fw_seen) >= MAX_ROADMAP_FW_ITEMS:
            break

    for item in pending_items[:MAX_ROADMAP_KERNEL_ITEMS]:
        rows.append(ActionRoadmapRow(
            track=TRACK_KERNEL,
            track_label="[C] 算子新开发与深度调优项",
            title=f"{item.target_kernel} ({item.before_model_ms:.2f} -> {item.after_model_ms:.2f} ms)",
            expected_gain_ms=item.expected_gain_ms,
            action_detail=item.kernel_action or item.default_design_slot,
            cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
        ))
    return tuple(rows)


def _scope_of_suffix(suf: str) -> str:
    for prefix in _SCOPE_PREFIXES:
        if suf.startswith(prefix):
            return prefix
    return ""


def _op_exact_key(op: OperatorExportRow) -> str:
    return strip_op_prefix(op.op_id).lower()


def _op_atoms(op: OperatorExportRow) -> frozenset[str]:
    if op.semantic_atoms:
        return frozenset(op.semantic_atoms)
    return frozenset(extract_semantic_atoms(op.op_id, op.family, op.recipe, op.absorbed_source_ops))


def _scope_compatible(a_scope: str, b_scope: str) -> bool:
    if not a_scope or not b_scope:
        return True
    return a_scope == b_scope


def _enforce_zero_saving_status_invariant(
    b_ops: Sequence[OperatorExportRow],
    a_op: OperatorExportRow,
    *,
    before_ntok: float = 1.0,
    after_ntok: float = 1.0,
    before_rep: int = 1,
    after_rep: int = 1,
) -> OperatorExportRow:
    """Hard Status Invariant: strip dev/tuning badges when neither raw nor amortized cost improves."""
    if not b_ops:
        return a_op
    b_w = max(1, int(before_rep))
    a_w = max(1, int(after_rep))
    b_exp = sum(0.0 if b.is_overlapped else (b.exposed_dur_us * b_w) for b in b_ops)
    a_exp = 0.0 if a_op.is_overlapped else (a_op.exposed_dur_us * a_w)
    b_upt = b_exp / max(float(before_ntok), MIN_DENOM_EPS)
    a_upt = a_exp / max(float(after_ntok), MIN_DENOM_EPS)
    if a_exp >= b_exp - ZERO_SAVING_EPS_US and a_upt >= b_upt - ZERO_SAVING_EPS_US:
        status_str = str(a_op.kernel_status or "")
        if a_op.needs_new_kernel or any(tok in status_str for tok in _DEV_BADGE_TOKENS):
            return replace(
                a_op,
                kernel_status=KERNEL_STATUS_AT_FLOOR,
                needs_new_kernel=False,
                opt_pillar="",
            )
    return a_op


def _detect_added_glue_note(b_op: OperatorExportRow, a_op: OperatorExportRow) -> str:
    """Detect newly introduced glue fused directly into ``d = f_new + a`` (Case 1)."""
    b_ex = _op_exact_key(b_op)
    a_fam = a_op.family.lower()
    b_fam = b_op.family.lower()
    if a_fam == "all_gather_matmul" and b_fam != "all_gather_matmul":
        return "+新增 AllGather(SP) 融合"
    if a_fam == "matmul_reduce_scatter" and b_fam != "matmul_reduce_scatter":
        return "+新增 ReduceScatter(SP) 融合"
    extra = [
        strip_op_prefix(s) for s in a_op.absorbed_source_ops
        if strip_op_prefix(s).lower() != b_ex and not extract_semantic_atoms(s, "", "")
    ]
    if extra:
        return f"+新增 {'/'.join(extra[:2])} 融合"
    return ""


def _build_action_note(
    relation: str,
    before_ops: Sequence[OperatorExportRow],
    after_ops: Sequence[OperatorExportRow],
    *,
    added_glue_note: str = "",
    non_adjacent_anchor: str = "",
) -> str:
    if relation == "fused_into_anchor":
        return f"[↑ 已并入 {non_adjacent_anchor}, After=0 (零重复计数)]"
    if relation == "N->1" and after_ops:
        a = after_ops[0]
        detail = a.kernel_action or a.framework_action or a.kernel_status
        return f"N->1 融合 ({len(before_ops)} ops -> {a.ascend_kernel}) | {detail}"
    if "->" in relation and relation not in {"1->1", "N->1"}:
        return f"[{relation} 子图等价重组块] 保持簇级耗时守恒 (Before {len(before_ops)} ops <-> After {len(after_ops)} ops)"
    if relation == "eliminated":
        b0 = before_ops[0] if before_ops else None
        is_comm = bool(b0 and (b0.task_type == "HCCL" or b0.family in COLLECTIVE_FAMILIES))
        return "— [随策略切换移除, 0.000 ms]" if is_comm else "优化后被融合/消除 (0 暴露耗时)"
    if relation == "added" and after_ops:
        a = after_ops[0]
        is_comm = a.task_type == "HCCL" or a.family in COLLECTIVE_FAMILIES
        prefix = "— [随新策略引入, 0.000 ms] -> " if is_comm else "优化后新增算子: "
        return f"{prefix}{a.logical_role} | {a.kernel_action or a.kernel_status}"
    if after_ops:
        a = after_ops[0]
        glue_tag = f" ({added_glue_note})" if added_glue_note else ""
        if a.is_overlapped:
            masker = a.masked_by or "主计算流"
            return f"副流被 {masker} 100% 掩盖 (0 暴露，不计入总耗时)"
        if a.opt_pillar == "pillar3_comm":
            return f"配置+通信升级{glue_tag}: {a.config_howto or a.kernel_status}"
        if a.opt_pillar == "pillar2_roofline":
            att_lbl = f" (目标 {a.opt_attainment_pct:.0f}% L*)" if a.opt_attainment_pct else ""
            return f"算子调优{glue_tag}{att_lbl}: {a.kernel_action or a.kernel_status}"
        if a.needs_new_kernel and a.kernel_action:
            return f"{a.kernel_status}{glue_tag} | {a.kernel_action}"
        if before_ops:
            b = before_ops[0]
            if b.ascend_kernel != a.ascend_kernel:
                return f"算子升级{glue_tag}: {b.ascend_kernel} -> {a.ascend_kernel} | {a.config_howto or a.kernel_status}"
            if a.gap_us > ACTION_GAP_CALLOUT_US and a.l_star_lean_us > 0.0:
                return f"Gap vs L*{glue_tag} ({a.l_star_lean_us:.1f}us vs {a.dur_us:.1f}us) | {a.kernel_action or a.bound_display}"
        return f"{a.bound_display}{glue_tag} | {a.kernel_status}"
    return ""


def _build_bipartite_edges(
    b_list: Sequence[OperatorExportRow], a_list: Sequence[OperatorExportRow],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build Level-2 semantic-atom & explicit provenance bipartite adjacency."""
    b_atoms = [_op_atoms(b) for b in b_list]
    a_atoms = [_op_atoms(a) for a in a_list]
    b_keys = [_op_exact_key(b) for b in b_list]
    a_absorbed = [
        {strip_op_prefix(x).lower() for x in a.absorbed_source_ops if x}
        for a in a_list
    ]
    b_adj: dict[int, set[int]] = defaultdict(set)
    a_adj: dict[int, set[int]] = defaultdict(set)

    # Rules 1 & 2: Shared semantic atoms or explicit absorbed_source_ops provenance.
    for b_idx, b_op in enumerate(b_list):
        b_scope = _scope_of_suffix(b_keys[b_idx])
        for a_idx, a_op in enumerate(a_list):
            a_scope = _scope_of_suffix(_op_exact_key(a_op))
            if not _scope_compatible(a_scope, b_scope):
                continue
            if (b_atoms[b_idx] and (b_atoms[b_idx] & a_atoms[a_idx])) or (b_keys[b_idx] in a_absorbed[a_idx]):
                b_adj[b_idx].add(a_idx)
                a_adj[a_idx].add(b_idx)

    # Rule 3: Unmatched pure glue (atoms == empty) matches ONLY exact same glue identity + comm domain.
    used_a_glue: set[int] = set()
    for b_idx, b_op in enumerate(b_list):
        if b_idx in b_adj or b_atoms[b_idx]:
            continue
        b_ex = b_keys[b_idx]
        for a_idx, a_op in enumerate(a_list):
            if a_idx in a_adj or a_idx in used_a_glue or a_atoms[a_idx]:
                continue
            if _op_exact_key(a_op) != b_ex:
                continue
            if b_op.comm_domain_tag and a_op.comm_domain_tag and b_op.comm_domain_tag != a_op.comm_domain_tag:
                continue
            b_adj[b_idx].add(a_idx)
            a_adj[a_idx].add(b_idx)
            used_a_glue.add(a_idx)
            break
    return b_adj, a_adj


def _extract_bipartite_components(
    n_b: int, n_a: int, b_adj: Mapping[int, set[int]], a_adj: Mapping[int, set[int]],
) -> list[tuple[list[int], list[int]]]:
    """Decompose bipartite graph into connected components ``(B_indices, A_indices)``."""
    vis_b: set[int] = set()
    vis_a: set[int] = set()
    comps: list[tuple[list[int], list[int]]] = []
    for b_start in range(n_b):
        if b_start in vis_b:
            continue
        qb, qa = [b_start], []
        comp_b: list[int] = []
        comp_a: list[int] = []
        vis_b.add(b_start)
        while qb or qa:
            while qb:
                bi = qb.pop()
                comp_b.append(bi)
                for ai in b_adj.get(bi, ()):
                    if ai not in vis_a:
                        vis_a.add(ai)
                        qa.append(ai)
            while qa:
                ai = qa.pop()
                comp_a.append(ai)
                for bi in a_adj.get(ai, ()):
                    if bi not in vis_b:
                        vis_b.add(bi)
                        qb.append(bi)
        comps.append((sorted(comp_b), sorted(comp_a)))
    for a_start in range(n_a):
        if a_start not in vis_a:
            vis_a.add(a_start)
            comps.append(([], [a_start]))
    return comps


def _align_archetype_bipartite(
    aid: str,
    title: str,
    rep: int,
    b_list: Sequence[OperatorExportRow],
    a_list: Sequence[OperatorExportRow],
    *,
    before_ntok: float = 1.0,
    after_ntok: float = 1.0,
    before_rep: int | None = None,
    after_rep: int | None = None,
) -> list[AlignedComparisonRow]:
    """Align one archetype via Two-Level Semantic-Atom Bipartite Connected Components."""
    b_w = max(1, int(before_rep)) if before_rep is not None else max(1, int(rep))
    a_w = max(1, int(after_rep)) if after_rep is not None else max(1, int(rep))
    row_rep = a_w if a_list else b_w
    b_adj, a_adj = _build_bipartite_edges(b_list, a_list)
    comps = _extract_bipartite_components(len(b_list), len(a_list), b_adj, a_adj)

    # Track topological anchor of each A node to place staggered 0->1 added rows cleanly.
    anchor_for_a: dict[int, float] = {}
    for b_idxs, a_idxs in comps:
        if b_idxs and a_idxs:
            for ai in a_idxs:
                anchor_for_a[ai] = float(b_idxs[0])

    # Precompute interpolated anchors for 0->1 added ops immediately AFTER preceding matched A op.
    added_anchor: dict[int, float] = {}
    last_anchor = -1.0
    for a_idx in range(len(a_list)):
        if a_idx in anchor_for_a:
            last_anchor = anchor_for_a[a_idx]
        else:
            added_anchor[a_idx] = (last_anchor + 0.5) if last_anchor >= 0.0 else float(len(b_list) + a_idx)

    entries: list[tuple[float, int, int, AlignedComparisonRow]] = []
    for b_idxs, a_idxs in comps:
        nb, na = len(b_idxs), len(a_idxs)
        if nb == 1 and na == 1:
            b_op = b_list[b_idxs[0]]
            a_op = _enforce_zero_saving_status_invariant(
                (b_op,), a_list[a_idxs[0]], before_ntok=before_ntok, after_ntok=after_ntok,
                before_rep=b_w, after_rep=a_w,
            )
            glue = _detect_added_glue_note(b_op, a_op)
            note = _build_action_note("1->1", (b_op,), (a_op,), added_glue_note=glue)
            entries.append((
                float(b_idxs[0]), 0, a_idxs[0],
                AlignedComparisonRow(
                    aid, title, row_rep, a_op.submodule, (b_op,), (a_op,), "1->1", note,
                    added_glue_note=glue, before_repeat_count=b_w, after_repeat_count=a_w,
                ),
            ))
        elif nb >= MIN_GROUP_SIZE and na == 1:
            a_op = _enforce_zero_saving_status_invariant(
                [b_list[i] for i in b_idxs], a_list[a_idxs[0]], before_ntok=before_ntok, after_ntok=after_ntok,
                before_rep=b_w, after_rep=a_w,
            )
            is_contiguous = (b_idxs[-1] - b_idxs[0] + 1) == nb
            if is_contiguous:
                b_tuple = tuple(b_list[i] for i in b_idxs)
                note = _build_action_note("N->1", b_tuple, (a_op,))
                entries.append((
                    float(b_idxs[0]), 0, a_idxs[0],
                    AlignedComparisonRow(
                        aid, title, row_rep, a_op.submodule, b_tuple, (a_op,), "N->1", note,
                        before_repeat_count=b_w, after_repeat_count=a_w,
                    ),
                ))
            else:
                b_anchor = b_list[b_idxs[0]]
                note = _build_action_note("N->1", tuple(b_list[i] for i in b_idxs), (a_op,))
                entries.append((
                    float(b_idxs[0]), 0, a_idxs[0],
                    AlignedComparisonRow(
                        aid, title, row_rep, a_op.submodule, (b_anchor,), (a_op,), "N->1", note,
                        before_repeat_count=b_w, after_repeat_count=a_w,
                    ),
                ))
                for sec_i in b_idxs[1:]:
                    b_sec = b_list[sec_i]
                    sec_note = _build_action_note("fused_into_anchor", (b_sec,), (), non_adjacent_anchor=a_op.ascend_kernel)
                    entries.append((
                        float(sec_i), 0, -1,
                        AlignedComparisonRow(
                            aid, title, b_w, b_sec.submodule, (b_sec,), (), "fused_into_anchor",
                            sec_note, non_adjacent_anchor_kernel=a_op.ascend_kernel,
                            before_repeat_count=b_w, after_repeat_count=0,
                        ),
                    ))
        elif nb == 1 and na == 0:
            b_op = b_list[b_idxs[0]]
            note = _build_action_note("eliminated", (b_op,), ())
            entries.append((
                float(b_idxs[0]), 1, -1,
                AlignedComparisonRow(
                    aid, title, b_w, b_op.submodule, (b_op,), (), "eliminated", note,
                    before_repeat_count=b_w, after_repeat_count=0,
                ),
            ))
        elif nb == 0 and na == 1:
            a_op = a_list[a_idxs[0]]
            note = _build_action_note("added", (), (a_op,))
            entries.append((
                added_anchor.get(a_idxs[0], float(len(b_list) + a_idxs[0])), 2, a_idxs[0],
                AlignedComparisonRow(
                    aid, title, a_w, a_op.submodule, (), (a_op,), "added", note,
                    before_repeat_count=0, after_repeat_count=a_w,
                ),
            ))
        else:
            # Case 4: M -> N (or 1 -> N) Cross-Boundary Re-association Block.
            b_tuple = tuple(b_list[i] for i in b_idxs)
            raw_a = tuple(a_list[j] for j in a_idxs)
            b_cluster_exp = sum(0.0 if b.is_overlapped else (b.exposed_dur_us * b_w) for b in b_tuple)
            a_cluster_exp = sum(0.0 if a.is_overlapped else (a.exposed_dur_us * a_w) for a in raw_a)
            b_cluster_upt = b_cluster_exp / max(float(before_ntok), MIN_DENOM_EPS)
            a_cluster_upt = a_cluster_exp / max(float(after_ntok), MIN_DENOM_EPS)
            if b_tuple and a_cluster_exp >= b_cluster_exp - ZERO_SAVING_EPS_US and a_cluster_upt >= b_cluster_upt - ZERO_SAVING_EPS_US:
                a_tuple = tuple(
                    replace(a, kernel_status=KERNEL_STATUS_AT_FLOOR, needs_new_kernel=False, opt_pillar="")
                    if (a.needs_new_kernel or any(tok in str(a.kernel_status or "") for tok in _DEV_BADGE_TOKENS))
                    else a
                    for a in raw_a
                )
            else:
                a_tuple = raw_a
            rel = f"{nb}->{na}"
            note = _build_action_note(rel, b_tuple, a_tuple)
            anchor = float(b_idxs[0]) if b_idxs else float(len(b_list) + a_idxs[0])
            entries.append((
                anchor, 0, a_idxs[0] if a_idxs else -1,
                AlignedComparisonRow(
                    aid, title, row_rep, (a_tuple[0] if a_tuple else b_tuple[0]).submodule,
                    b_tuple, a_tuple, rel, note,
                    before_repeat_count=b_w, after_repeat_count=a_w,
                ),
            ))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return [row for _, _, _, row in entries]


def _format_slice_layer_range(indices: Sequence[int]) -> str:
    ordered = sorted(set(int(x) for x in indices))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f"L{ordered[0]}"
    if len(ordered) <= SLICE_MAX_EXPLICIT_LAYERS:
        return ", ".join(f"L{x}" for x in ordered)
    return f"L{ordered[0]}..L{ordered[-1]}"


def _index_target_layer_archetypes(
    bundle: ExportBundle,
    ops_by_arch: Mapping[str, Sequence[OperatorExportRow]],
) -> tuple[dict[int, tuple[str, str, tuple[OperatorExportRow, ...]]], set[str]]:
    """Map physical target layer indices to ``(template_key, title, ops)`` without inflating PP weights."""
    layer_map: dict[int, tuple[str, str, tuple[OperatorExportRow, ...]]] = {}
    handled_aids: set[str] = set()
    for arch in bundle.archetypes:
        if arch.scope != "target" or not arch.layer_indices:
            continue
        ops = tuple(ops_by_arch.get(arch.archetype_id, ()))
        if not ops:
            continue
        handled_aids.add(arch.archetype_id)
        distinct_layers = {int(o.layer) for o in ops if o.layer is not None}
        is_dedup = (
            all(int(o.repeat_count) == len(arch.layer_indices) for o in ops)
            and len(distinct_layers) <= 1
        )
        if is_dedup:
            for phys_idx in arch.layer_indices:
                layer_map[int(phys_idx)] = (arch.archetype_id, arch.title, ops)
            continue
        by_layer: dict[int, list[OperatorExportRow]] = defaultdict(list)
        for op in ops:
            if op.layer is not None:
                by_layer[int(op.layer)].append(op)
        for phys_idx in arch.layer_indices:
            l_ops = tuple(by_layer.get(int(phys_idx), ()))
            if l_ops:
                layer_map[int(phys_idx)] = (f"{arch.archetype_id}@L{phys_idx}", arch.title, l_ops)
    return layer_map, handled_aids


def _build_target_layer_slices(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    b_layer_map: Mapping[int, tuple[str, str, tuple[OperatorExportRow, ...]]],
    a_layer_map: Mapping[int, tuple[str, str, tuple[OperatorExportRow, ...]]],
) -> list[tuple[str, str, int, int, tuple[OperatorExportRow, ...], tuple[OperatorExportRow, ...]]]:
    """Intersect Before/After physical target layer-index sets for exact PP1<->PP2 conservation."""
    b_arch_meta = {a.archetype_id: a for a in before_bundle.archetypes}
    a_arch_meta = {a.archetype_id: a for a in after_bundle.archetypes}
    grouped: dict[tuple[str | None, str | None], list[int]] = {}
    for phys_idx in sorted(set(b_layer_map) | set(a_layer_map)):
        b_key = b_layer_map[phys_idx][0] if phys_idx in b_layer_map else None
        a_key = a_layer_map[phys_idx][0] if phys_idx in a_layer_map else None
        grouped.setdefault((b_key, a_key), []).append(phys_idx)

    slices: list[tuple[str, str, int, int, tuple[OperatorExportRow, ...], tuple[OperatorExportRow, ...]]] = []
    used_ids: set[str] = set()
    for (b_key, a_key), idxs in grouped.items():
        rep = len(idxs)
        b_ops = b_layer_map[idxs[0]][2] if b_key is not None else ()
        a_ops = a_layer_map[idxs[0]][2] if a_key is not None else ()
        b_rep = rep if b_ops else 0
        a_rep = rep if a_ops else 0
        same_unsplit = (
            b_key is not None
            and b_key == a_key
            and b_key in b_arch_meta
            and b_key in a_arch_meta
            and len(idxs) == len(b_arch_meta[b_key].layer_indices) == len(a_arch_meta[b_key].layer_indices)
        )
        if same_unsplit:
            slice_aid = b_key
            slice_title = a_arch_meta[b_key].title
        else:
            base_aid = a_key or b_key or "target_slice"
            slice_aid = base_aid if base_aid not in used_ids else f"{b_key or 'none'}__{a_key or 'none'}"
            raw_title = (a_layer_map[idxs[0]][1] if a_key is not None else b_layer_map[idxs[0]][1])
            clean_title = re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", raw_title).strip()
            rng_txt = _format_slice_layer_range(idxs)
            slice_title = f"{clean_title} (×{rep}, {rng_txt})" if rng_txt else f"{clean_title} (×{rep})"
        used_ids.add(slice_aid)
        slices.append((slice_aid, slice_title, b_rep, a_rep, b_ops, a_ops))
    return slices


def align_before_after_operators(
    before_bundle: ExportBundle, after_bundle: ExportBundle,
) -> tuple[AlignedComparisonRow, ...]:
    """Align Before vs After via physical layer-index set alignment + semantic bipartite matching."""
    before_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    after_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    for op in before_bundle.operators:
        before_by_arch[op.archetype_id or op.layer_label].append(op)
    for op in after_bundle.operators:
        after_by_arch[op.archetype_id or op.layer_label].append(op)

    b_layer_map, b_handled = _index_target_layer_archetypes(before_bundle, before_by_arch)
    a_layer_map, a_handled = _index_target_layer_archetypes(after_bundle, after_by_arch)
    handled_target_aids = b_handled | a_handled
    target_slices = (
        _build_target_layer_slices(before_bundle, after_bundle, b_layer_map, a_layer_map)
        if (b_layer_map or a_layer_map) else []
    )

    pre_slices: list[tuple[str, str, int, int, tuple[OperatorExportRow, ...], tuple[OperatorExportRow, ...]]] = []
    post_slices: list[tuple[str, str, int, int, tuple[OperatorExportRow, ...], tuple[OperatorExportRow, ...]]] = []
    seen_arch: set[str] = set()
    seen_target = False
    for src in (before_bundle, after_bundle):
        for op in src.operators:
            aid = op.archetype_id or op.layer_label
            if aid in handled_target_aids:
                seen_target = True
                continue
            if aid in seen_arch:
                continue
            seen_arch.add(aid)
            b_ops = tuple(before_by_arch.get(aid, ()))
            a_ops = tuple(after_by_arch.get(aid, ()))
            b_rep = max(1, int(b_ops[0].repeat_count)) if b_ops else 0
            a_rep = max(1, int(a_ops[0].repeat_count)) if a_ops else 0
            entry = (aid, op.archetype_title or op.layer_label, b_rep, a_rep, b_ops, a_ops)
            if not seen_target and target_slices:
                pre_slices.append(entry)
            else:
                post_slices.append(entry)

    b_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(after_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    aligned: list[AlignedComparisonRow] = []
    for aid, title, b_rep, a_rep, b_ops, a_ops in (*pre_slices, *target_slices, *post_slices):
        aligned.extend(_align_archetype_bipartite(
            aid, title, a_rep or b_rep or 1, b_ops, a_ops,
            before_ntok=b_ntok, after_ntok=a_ntok,
            before_rep=b_rep or 1, after_rep=a_rep or 1,
        ))
    return tuple(aligned)


def _accumulate_macro_stats(bundle: ExportBundle) -> dict[str, dict[str, Any]]:
    arch_by_id = {a.archetype_id: a for a in bundle.archetypes}
    stats: dict[str, dict[str, Any]] = {}
    for op in bundle.operators:
        mod = classify_macro_module(op)
        bucket = stats.setdefault(mod, {
            "exposed_ms": 0.0, "raw_ms": 0.0, "lstar_ms": 0.0,
            "bound_ms": defaultdict(float), "masker_ms": defaultdict(float),
            "target_layer_indices": set(), "draft_layers": 0, "has_head_or_embed": False,
        })
        rep = max(1, int(op.repeat_count))
        exp_ms = 0.0 if op.is_overlapped else (op.exposed_dur_us * rep / US_PER_MS)
        raw_ms = op.dur_us * rep / US_PER_MS
        masked_pos_ms = max(0.0, raw_ms - exp_ms)
        lstar_ms = 0.0 if op.is_overlapped else (op.l_star_lean_us * rep / US_PER_MS)
        bucket["exposed_ms"] += exp_ms
        bucket["raw_ms"] += raw_ms
        bucket["lstar_ms"] += lstar_ms
        if exp_ms > 0.0:
            bucket["bound_ms"][op.bound_display or op.dominant_bound] += exp_ms
        if masked_pos_ms > CONSERVATION_EPS_MS:
            bucket["masker_ms"][op.masked_by or "主计算流并发算子"] += masked_pos_ms
        arch = arch_by_id.get(op.archetype_id)
        if op.model_scope == "draft":
            if arch is not None and arch.layer_indices:
                bucket["draft_layers"] = max(bucket["draft_layers"], len(arch.layer_indices))
            else:
                bucket["has_head_or_embed"] = True
        elif arch is not None and arch.layer_indices:
            bucket["target_layer_indices"].update(int(x) for x in arch.layer_indices)
        else:
            bucket["has_head_or_embed"] = True
    return stats


def _format_macro_layer_label(
    mod: str, target_layer_indices: set[int], draft_layers: int, has_head: bool, bundle: ExportBundle,
) -> tuple[str, int]:
    t_layers = len(target_layer_indices)
    if mod == MACRO_MOD_DRAFT_LAYERS:
        d_cnt = draft_layers or bundle.total_draft_layers
        return (f"Draft {d_cnt}L", d_cnt) if d_cnt > 0 else ("Draft ×1", 1)
    if mod == MACRO_MOD_VERIFY_HEAD:
        return "Embed + Head (×1)", 1
    if t_layers > 0:
        return f"{t_layers}L", t_layers
    return ("×1", 1) if has_head else ("0L", 0)


def _build_masking_note(
    mod: str, raw_aft: float, aft_ms: float, raw_bef: float, bef_ms: float,
    a_maskers: Mapping[str, float], b_maskers: Mapping[str, float], dom_bound: str,
) -> str:
    aft_masked = max(0.0, raw_aft - aft_ms)
    bef_masked = max(0.0, raw_bef - bef_ms)
    if aft_masked > CONSERVATION_EPS_MS:
        top_m = max(a_maskers.items(), key=lambda kv: kv[1])[0] if a_maskers else "主计算流"
        if aft_ms <= OVERLAP_EXPOSED_EPS_US:
            if mod == MACRO_MOD_MOE_SHARED:
                return f"MSO 副流被 {top_m} 100% 掩盖 (-{aft_masked:.2f}ms，不计入总耗时)"
            return f"被 {top_m} 100% 掩盖 (-{aft_masked:.2f}ms，不计入总耗时)"
        pct = 100.0 * aft_masked / max(raw_aft, CONSERVATION_EPS_MS)
        return f"{dom_bound} | 被 {top_m} 掩盖 {pct:.0f}% (-{aft_masked:.2f}ms)"
    if bef_masked > CONSERVATION_EPS_MS:
        top_m = max(b_maskers.items(), key=lambda kv: kv[1])[0] if b_maskers else "主计算流"
        return f"{dom_bound} | 优化前被 {top_m} 掩盖 -{bef_masked:.2f}ms"
    return dom_bound


def build_macro_module_breakdown(
    before_bundle: ExportBundle, after_bundle: ExportBundle | None = None,
) -> tuple[MacroModuleRow, ...]:
    """Aggregate L1/L2 Raw, Masked, Exposed ms and ``us / (tok*die)`` with 100% conservation."""
    bef_stats = _accumulate_macro_stats(before_bundle)
    aft_bundle = after_bundle or before_bundle
    aft_stats = bef_stats if after_bundle is None else _accumulate_macro_stats(after_bundle)

    b_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(aft_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    tot_bef = sum(v["exposed_ms"] for v in bef_stats.values())
    tot_aft = sum(v["exposed_ms"] for v in aft_stats.values())
    l1_by_mod = {mod: l1 for l1, mods in MACRO_L1_HIERARCHY for mod in mods}
    ordered_mods = [mod for _, mods in MACRO_L1_HIERARCHY for mod in mods]
    for extra in (*bef_stats.keys(), *aft_stats.keys()):
        if extra not in ordered_mods:
            ordered_mods.append(extra)

    rows: list[MacroModuleRow] = []
    for mod in ordered_mods:
        b_info = bef_stats.get(mod)
        a_info = aft_stats.get(mod)
        if b_info is None and a_info is None:
            continue
        bef_ms = float(b_info["exposed_ms"]) if b_info else 0.0
        aft_ms = float(a_info["exposed_ms"]) if a_info else 0.0
        raw_bef = float(b_info["raw_ms"]) if b_info else 0.0
        raw_aft = float(a_info["raw_ms"]) if a_info else 0.0
        lstar_ms = float(a_info["lstar_ms"]) if a_info else 0.0
        if raw_bef <= 0.0 and raw_aft <= 0.0:
            continue
        merged_indices: set[int] = set(b_info["target_layer_indices"]) if b_info else set()
        if a_info:
            merged_indices.update(a_info["target_layer_indices"])
        d_layers = max(int(b_info["draft_layers"]) if b_info else 0, int(a_info["draft_layers"]) if a_info else 0)
        has_head = bool((b_info and b_info["has_head_or_embed"]) or (a_info and a_info["has_head_or_embed"]))
        layer_lbl, layer_cnt = _format_macro_layer_label(mod, merged_indices, d_layers, has_head, aft_bundle)
        bound_weights = a_info["bound_ms"] if (a_info and a_info["bound_ms"]) else (b_info["bound_ms"] if b_info else {})
        if mod == MACRO_MOD_MOE_SHARED and aft_ms <= OVERLAP_EXPOSED_EPS_US and raw_aft > 0.0:
            dom_bound = f"MSO Stream 1 100% 掩盖 (-{raw_aft:.3f}ms)"
        elif bound_weights:
            dom_bound = max(bound_weights.items(), key=lambda kv: kv[1])[0]
        else:
            dom_bound = "已融合 / 100% 掩盖"
        mask_note = _build_masking_note(
            mod, raw_aft, aft_ms, raw_bef, bef_ms,
            a_info["masker_ms"] if a_info else {}, b_info["masker_ms"] if b_info else {}, dom_bound,
        )
        b_upt = bef_ms * US_PER_MS / b_ntok
        a_upt = aft_ms * US_PER_MS / a_ntok
        rows.append(MacroModuleRow(
            module_name=mod, layer_calls_label=layer_lbl, layer_count=layer_cnt,
            before_ms=bef_ms, before_share_pct=(100.0 * bef_ms / tot_bef) if tot_bef > 0.0 else 0.0,
            after_ms=aft_ms, after_share_pct=(100.0 * aft_ms / tot_aft) if tot_aft > 0.0 else 0.0,
            saved_ms=bef_ms - aft_ms, lstar_ms=lstar_ms, dominant_bound=dom_bound,
            raw_before_ms=raw_bef, raw_after_ms=raw_aft,
            l1_group=l1_by_mod.get(mod, L1_MOD_RESIDUAL_MEM),
            masked_before_ms=bef_ms - raw_bef, masked_after_ms=aft_ms - raw_aft,
            masking_note=mask_note,
            before_us_per_tok_die=b_upt, after_us_per_tok_die=a_upt,
            saved_us_per_tok_die=b_upt - a_upt,
        ))
    return tuple(rows)


def _short_arch_name(title: str) -> str:
    return re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", title.split("—")[0].strip())


def _make_card_slug(raw: str, used: set[str]) -> str:
    base = _SLUG_RE.sub("_", raw.lower()).strip("_") or "kernel_card"
    slug = base[:CARD_SLUG_MAX_LEN].rstrip("_")
    cand = slug
    idx = 2
    while cand in used:
        cand = f"{slug}_{idx}"
        idx += 1
    used.add(cand)
    return cand


def _classify_comparison_card_key(
    row: AlignedComparisonRow,
    saved_ms: float,
    saved_upt: float,
    *,
    is_comparison: bool,
    include_to_optimize: bool,
) -> tuple[str, str, OperatorExportRow | None] | None:
    if row.after_ops:
        a_op = row.after_ops[0]
        is_new = bool(
            a_op.needs_new_kernel
            or a_op.binding_state == BINDING_STATE_UNBOUND_FORMULA
        )
        if is_comparison:
            has_nonzero_delta = (
                abs(saved_ms) > CONSERVATION_EPS_MS or abs(saved_upt) > CONSERVATION_EPS_MS
            )
            has_positive_saving = (
                saved_ms > CONSERVATION_EPS_MS or saved_upt > CONSERVATION_EPS_MS
            )
            if not has_nonzero_delta:
                return None
            if is_new and has_positive_saving:
                return f"new::{a_op.ascend_kernel}", PENDING_KIND_TO_IMPLEMENT, a_op
            if not include_to_optimize:
                return None
            if a_op.is_overlapped and has_positive_saving:
                return "opt::mso_stream_overlap_masking", PENDING_KIND_TO_OPTIMIZE, a_op
            return f"opt::{a_op.ascend_kernel}", PENDING_KIND_TO_OPTIMIZE, a_op

        a_rep = row.effective_after_repeat
        gap_ms = sum(
            0.0 if a.is_overlapped else max(0.0, (a.exposed_dur_us - a.l_star_lean_us) * a_rep / US_PER_MS)
            for a in row.after_ops
        )
        if gap_ms <= CONSERVATION_EPS_MS:
            return None
        if is_new:
            return f"new::{a_op.ascend_kernel}", PENDING_KIND_TO_IMPLEMENT, a_op
        if include_to_optimize:
            return f"opt::{a_op.ascend_kernel}", PENDING_KIND_TO_OPTIMIZE, a_op
        return None

    if is_comparison and include_to_optimize and row.before_ops and (
        abs(saved_ms) > CONSERVATION_EPS_MS or abs(saved_upt) > CONSERVATION_EPS_MS
    ):
        b0 = row.before_ops[0]
        return f"opt::eliminated::{b0.submodule}", PENDING_KIND_TO_OPTIMIZE, None
    return None


def _accumulate_pending_group(
    grouped: dict[str, dict[str, Any]],
    *,
    key: str,
    category: str,
    row: AlignedComparisonRow,
    row_index: int,
    a_op: OperatorExportRow | None,
    after_bundle: ExportBundle,
    before_ntok: float,
    after_ntok: float,
    is_comparison: bool,
) -> None:
    b_rep = row.effective_before_repeat
    a_rep = row.effective_after_repeat
    rep = a_rep or b_rep or max(1, int(row.repeat_count))
    before_ms = sum(0.0 if b.is_overlapped else (b.exposed_dur_us * b_rep / US_PER_MS) for b in row.before_ops)
    after_ms = sum(0.0 if a.is_overlapped else (a.exposed_dur_us * a_rep / US_PER_MS) for a in row.after_ops)
    lstar_ms = sum(0.0 if a.is_overlapped else (a.l_star_lean_us * a_rep / US_PER_MS) for a in row.after_ops)
    before_lstar_ms = sum(0.0 if b.is_overlapped else (b.l_star_lean_us * b_rep / US_PER_MS) for b in row.before_ops)
    gap_ms = max(0.0, after_ms - lstar_ms)
    gain_ms = (before_ms - after_ms) if is_comparison else gap_ms
    b_upt = before_ms * US_PER_MS / before_ntok
    a_upt = after_ms * US_PER_MS / after_ntok
    gain_upt = (b_upt - a_upt) if is_comparison else (gap_ms * US_PER_MS / after_ntok)

    ref_op = a_op or row.before_ops[0]
    bef_bounds = [b.bound_display or b.dominant_bound for b in row.before_ops if b.bound_display or b.dominant_bound]
    bef_bound_lbl = " / ".join(dict.fromkeys(bef_bounds)) or (ref_op.bound_display or ref_op.dominant_bound)
    is_mso_card = key == "opt::mso_stream_overlap_masking"
    is_elim_card = key.startswith("opt::eliminated::")
    card_kernel = (
        "MSO 副流异步掩盖 (Stream 1 Overlap)" if is_mso_card
        else (f"策略切换/融合消除算子 ({ref_op.submodule})" if is_elim_card else ref_op.ascend_kernel)
    )
    card_status = (
        "已有特效配置: MSO + MC2 加速后共享专家 100% 掩盖" if is_mso_card
        else ("优化后消除 (0 暴露)" if is_elim_card else ref_op.kernel_status)
    )
    kspec = ref_op.kernel_synthesis_spec
    math_formula = " ; ".join(kspec.symbolic_equations) if kspec is not None else ""
    sram_sum = kspec.format_formula_block() if kspec is not None else ""

    entry = grouped.get(key)
    if entry is None:
        entry = {
            "kernel_key": key, "category": category, "target_kernel": card_kernel,
            "logical_role": ref_op.logical_role or ref_op.op_id, "roles": [],
            "submodule": ref_op.submodule, "backend": ref_op.kernel_backend or "CANN / AscendC",
            "status": card_status, "replaces": [], "sources": [], "archetypes": [],
            "calls": 0, "before_ms": 0.0, "after_ms": 0.0, "lstar_ms": 0.0,
            "before_lstar_ms": 0.0, "gap_ms": 0.0, "gain_ms": 0.0,
            "before_upt": 0.0, "after_upt": 0.0, "gain_upt": 0.0,
            "before_bound": bef_bound_lbl,
            "after_bound": "100% 掩盖 / 消除" if (is_mso_card or is_elim_card) else (ref_op.bound_display or ref_op.dominant_bound),
            "dom_key": ref_op.dominant_bound, "contract": ref_op.tensor_contract,
            "single_lstar_us": ref_op.l_star_lean_us, "single_sim_us": ref_op.dur_us,
            "die_summary": after_bundle.hardware_die_summary, "spec_note": ref_op.new_kernel_spec,
            "config_howto": ref_op.config_howto, "kernel_action": ref_op.kernel_action,
            "framework_action": ref_op.framework_action,
            "opt_pillar": ref_op.opt_pillar or ("pillar1_fusion" if category == PENDING_KIND_TO_IMPLEMENT else "pillar2_roofline"),
            "synthesis_spec": kspec, "math_formula": math_formula, "sram_residency_summary": sram_sum,
            "binding_state": (
                BINDING_STATE_UNBOUND_FORMULA
                if (
                    ref_op.binding_state == BINDING_STATE_UNBOUND_FORMULA
                    or any(op.binding_state == BINDING_STATE_UNBOUND_FORMULA for op in row.after_ops)
                )
                else (ref_op.binding_state or BINDING_STATE_BOUND)
            ),
            "source_row_indices": [],
        }
        grouped[key] = entry
    entry["source_row_indices"].append(int(row_index))
    if any(op.binding_state == BINDING_STATE_UNBOUND_FORMULA for op in row.after_ops):
        entry["binding_state"] = BINDING_STATE_UNBOUND_FORMULA
    for b_op in row.before_ops:
        entry["replaces"].append(b_op.ascend_kernel)
        if b_op.source_ref:
            entry["sources"].append(b_op.source_ref)
    if a_op is not None and a_op.source_ref:
        entry["sources"].append(a_op.source_ref)
    role_lbl = ref_op.logical_role or ref_op.op_id
    if role_lbl not in entry["roles"]:
        entry["roles"].append(role_lbl)
    entry["archetypes"].append(f"{_short_arch_name(row.archetype_title)} (×{rep})")
    entry["calls"] += rep
    entry["before_ms"] += before_ms
    entry["after_ms"] += after_ms
    entry["lstar_ms"] += lstar_ms
    entry["before_lstar_ms"] += before_lstar_ms
    entry["gap_ms"] += gap_ms
    entry["gain_ms"] += gain_ms
    entry["before_upt"] += b_upt
    entry["after_upt"] += a_upt
    entry["gain_upt"] += gain_upt
    if ref_op.dur_us > entry["single_sim_us"]:
        entry["single_sim_us"] = ref_op.dur_us
        entry["single_lstar_us"] = ref_op.l_star_lean_us
        entry["contract"] = ref_op.tensor_contract
    if kspec is not None and entry["synthesis_spec"] is None:
        entry["synthesis_spec"] = kspec
        entry["math_formula"] = math_formula
        entry["sram_residency_summary"] = sram_sum


def _materialize_pending_item(prio: int, entry: Mapping[str, Any], used_slugs: set[str]) -> PendingKernelItem:
    is_new = entry["category"] == PENDING_KIND_TO_IMPLEMENT
    replaces_txt = (
        " + ".join(dict.fromkeys(entry["replaces"])) or "New target kernel"
        if is_new else f"{entry['submodule']} -> {', '.join(entry['roles'][:3])}"
    )
    sources_txt = "; ".join(dict.fromkeys(entry["sources"]))[:MAX_SOURCE_REF_SUMMARY_LEN] or "-"
    arch_txt = "; ".join(dict.fromkeys(entry["archetypes"]))
    bef_ms = float(entry["before_ms"])
    aft_ms = float(entry["after_ms"])
    ls_ms = float(entry["lstar_ms"])
    b_ls = float(entry["before_lstar_ms"]) or ls_ms
    bef_att = min(100.0, 100.0 * b_ls / bef_ms) if bef_ms > 0.0 and b_ls > 0.0 else 0.0
    aft_att = min(100.0, 100.0 * ls_ms / aft_ms) if aft_ms > 0.0 and ls_ms > 0.0 else (100.0 if aft_ms <= 0.0 else 0.0)
    microarch = (
        f"Before: {entry['before_bound']} (L* 达成率 {bef_att:.1f}%) -> "
        f"Target: {entry['after_bound']} (目标 {aft_att:.1f}% L*) | "
        f"Single L*={entry['single_lstar_us']:.2f}us vs Target={entry['single_sim_us']:.2f}us | {entry['die_summary']}"
    )
    hint = _OPTIMIZE_DESIGN_HINTS.get(str(entry["dom_key"]), "Tiling / UB 双缓冲 / 访存优化")
    config_howto = str(entry["config_howto"] or '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'')
    kernel_action = str(entry["kernel_action"] or (entry["spec_note"] or hint))
    framework_action = str(entry["framework_action"] or "vllm-ascend 保持静态 shape 预分配 workspace 并接线 ACLGraph 捕获")
    accept_slot = (
        f"精度误差 <= 1e-3 (BF16/MXFP4/FP8)，端到端实测耗时 <= {aft_ms:.3f}ms "
        f"(L* 达成率 >= {max(65.0, min(90.0, aft_att)):.0f}%，Δ={entry['gain_ms']:.3f}ms / Δ摊薄={entry['gain_upt']:.2f} us/tok·die)"
        if aft_ms > 0.0 else
        f"确认 100% 掩盖或融合消除，实际暴露耗时 = 0.000ms (Δ={entry['gain_ms']:.3f}ms / Δ摊薄={entry['gain_upt']:.2f} us/tok·die)"
    )
    disp_kernel = f"{entry['target_kernel']} [{entry['backend']}]"
    card_id = _make_card_slug(str(entry["target_kernel"]), used_slugs)
    return PendingKernelItem(
        priority=prio, kernel_key=str(entry["kernel_key"]), target_kernel=disp_kernel,
        suggested_backend=str(entry["backend"]), status=str(entry["status"]),
        replaces_chain=replaces_txt, source_refs=sources_txt, archetype_summary=arch_txt,
        total_calls_per_step=int(entry["calls"]), expected_gain_ms=float(entry["gain_ms"]),
        tensor_contract=str(entry["contract"]), microarch_target=microarch,
        default_design_slot=kernel_action, default_acceptance_slot=accept_slot,
        category=str(entry["category"]),
        logical_role=", ".join(entry["roles"][:2]) or str(entry["logical_role"]),
        submodule=str(entry["submodule"]), before_bound=str(entry["before_bound"]),
        after_bound=str(entry["after_bound"]), before_model_ms=bef_ms, after_model_ms=aft_ms,
        lstar_model_ms=ls_ms, gap_vs_lstar_ms=float(entry["gap_ms"]), card_id=card_id,
        config_howto=config_howto, kernel_action=kernel_action, framework_action=framework_action,
        before_attainment_pct=round(bef_att, 1), target_attainment_pct=round(aft_att, 1),
        opt_pillar=str(entry["opt_pillar"]),
        before_us_per_tok_die=float(entry["before_upt"]),
        after_us_per_tok_die=float(entry["after_upt"]),
        saved_us_per_tok_die=float(entry["gain_upt"]),
        synthesis_spec=entry["synthesis_spec"],
        math_formula=str(entry["math_formula"] or ""),
        sram_residency_summary=str(entry["sram_residency_summary"] or ""),
        binding_state=str(entry.get("binding_state") or BINDING_STATE_BOUND),
        source_row_indices=tuple(int(i) for i in entry.get("source_row_indices", ())),
    )


def build_pending_kernel_details(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
    *,
    before_bundle: ExportBundle | None = None,
    top_k_optimize: int = DEFAULT_TOP_OPTIMIZE_KERNELS,
    include_to_optimize: bool = True,
) -> tuple[PendingKernelItem, ...]:
    """Build 100%-conserved Sheet 4 cards ranked by ``Δ(us / tok*die)``."""
    b_ntok = max(float((before_bundle or after_bundle).n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(after_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    is_comparison = before_bundle is not None
    new_groups: dict[str, dict[str, Any]] = {}
    opt_groups: dict[str, dict[str, Any]] = {}
    for row_idx, row in enumerate(aligned_rows):
        b_rep = row.effective_before_repeat
        a_rep = row.effective_after_repeat
        b_ms = sum(0.0 if b.is_overlapped else (b.exposed_dur_us * b_rep / US_PER_MS) for b in row.before_ops)
        a_ms = sum(0.0 if a.is_overlapped else (a.exposed_dur_us * a_rep / US_PER_MS) for a in row.after_ops)
        saved_ms = b_ms - a_ms
        saved_upt = (b_ms * US_PER_MS / b_ntok) - (a_ms * US_PER_MS / a_ntok)
        classified = _classify_comparison_card_key(
            row, saved_ms, saved_upt, is_comparison=is_comparison, include_to_optimize=include_to_optimize,
        )
        if classified is None:
            continue
        key, cat, a_op = classified
        target_map = new_groups if cat == PENDING_KIND_TO_IMPLEMENT else opt_groups
        _accumulate_pending_group(
            target_map, key=key, category=cat, row=row, row_index=row_idx, a_op=a_op,
            after_bundle=after_bundle, before_ntok=b_ntok, after_ntok=a_ntok,
            is_comparison=is_comparison,
        )

    ordered_new = sorted(new_groups.values(), key=lambda x: (-x["gain_upt"], -x["gain_ms"], -x["calls"], x["target_kernel"]))
    if is_comparison:
        ordered_opt = sorted(opt_groups.values(), key=lambda x: (-x["gain_upt"], -x["gain_ms"], -x["calls"], x["kernel_key"]))
        selected = [*ordered_new, *ordered_opt]
    else:
        ordered_opt = sorted(opt_groups.values(), key=lambda x: (-x["gap_ms"], -x["calls"], x["kernel_key"]))
        selected = [*ordered_new, *ordered_opt[: max(0, int(top_k_optimize))]]

    used_slugs: set[str] = set()
    return tuple(_materialize_pending_item(prio, entry, used_slugs) for prio, entry in enumerate(selected, start=1))


def build_new_kernel_backlog(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
    *,
    before_bundle: ExportBundle | None = None,
) -> tuple[NewKernelBacklogItem, ...]:
    """Aggregate To-Implement kernel items into Sheet 4 backlog."""
    return build_pending_kernel_details(
        aligned_rows, after_bundle, before_bundle=before_bundle, include_to_optimize=False,
    )


def _trim_text(text: str, limit: int = BRIEF_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else (clean[: limit - 3] + "...")


def _bundle_kpi_dict(bundle: ExportBundle) -> dict[str, Any]:
    step_ms = bundle.step_span_us / US_PER_MS
    is_decode = bundle.phase == "decode"
    acc = max(1.0, bundle.avg_accepted_tokens)
    handoff_ms = float(getattr(bundle, "handoff_tpot_ms", 0.0) or 0.0) if is_decode else 0.0
    tpot_ms = ((step_ms / acc) + handoff_ms) if is_decode else step_ms
    sla = bundle.tpot_slo_ms if is_decode else bundle.ttft_slo_ms
    l_star_ms = bundle.step_l_star_lean_us / US_PER_MS
    return {
        "analysis_mode": bundle.analysis_mode,
        "binding_state": bundle.binding_state,
        "has_unbound_formula_ops": bool(bundle.has_unbound_formula_ops),
        "step_ms": float(step_ms),
        "handoff_tpot_ms": float(handoff_ms),
        "effective_tpot_ms": float(tpot_ms),
        "sla_target_ms": float(sla),
        "sla_verdict": "PASS" if tpot_ms <= sla else "FAIL",
        "sla_regime": bundle.sla_regime,
        "sla_slack_ms": float(sla - tpot_ms),
        "n_tok_per_die": float(bundle.n_tok_per_die),
        "us_per_tok_die": float(bundle.us_per_tok_die),
        "single_die_tps": float(bundle.throughput_per_die_tokens_per_s),
        "cluster_tps": float(bundle.throughput_tokens_per_s),
        "l_star_ms": float(l_star_ms),
        "l_star_attainment_pct": round((100.0 * l_star_ms / step_ms) if step_ms > 0.0 else 0.0, 1),
        "cube_util_pct": round(bundle.network_cube_mfu, 1),
        "vector_util_pct": round(bundle.network_vec_mfu, 1),
        "hbm_mbu_pct": round(bundle.network_mbu, 1),
        "hccl_link_util_pct": round(bundle.network_link_util, 1),
    }


def _brief_utf8_len(payload: Mapping[str, Any]) -> int:
    return len((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def _truncate_utf8_bytes(text: str, max_bytes: int) -> str:
    """Truncate ``text`` strictly below or equal to ``max_bytes`` UTF-8 bytes."""
    raw = str(text)
    if max_bytes <= 0:
        return ""
    encoded = raw.encode("utf-8")
    if len(encoded) <= max_bytes:
        return raw
    ellipsis = "..."
    ell_b = len(ellipsis)
    if max_bytes <= ell_b:
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
    body = encoded[: max_bytes - ell_b].decode("utf-8", errors="ignore")
    return body + ellipsis


def _shrink_value_for_budget(val: Any, excess_bytes: int) -> Any:
    """Shrink one oversized value by at least ``excess_bytes`` UTF-8 bytes."""
    if isinstance(val, list):
        return val[: len(val) // 2] if len(val) > 1 else []
    if isinstance(val, Mapping):
        if len(val) > 1:
            items = list(val.items())
            return dict(items[: len(items) // 2])
        if len(val) == 1:
            k_only, v_only = next(iter(val.items()))
            shrunk = _shrink_value_for_budget(v_only, excess_bytes)
            return {} if shrunk == v_only else {k_only: shrunk}
        return {}
    if isinstance(val, str):
        cur_bytes = len(val.encode("utf-8"))
        target_bytes = max(0, min(cur_bytes // 2, cur_bytes - max(1, excess_bytes)))
        return _truncate_utf8_bytes(val, target_bytes)
    return None


def _trim_keys_to_budget(trimmed: dict[str, Any], protected: frozenset[str]) -> None:
    while _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        excess = _brief_utf8_len(trimmed) - AGENT_BRIEF_MAX_BYTES + 1
        candidates = [
            (len(json.dumps(val, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), key)
            for key, val in trimmed.items()
            if key not in protected and val not in (None, "", [], {})
        ]
        if not candidates:
            break
        _, largest_key = max(candidates)
        cur_val = trimmed[largest_key]
        shrunk = _shrink_value_for_budget(cur_val, excess)
        if shrunk is None or shrunk == cur_val:
            trimmed.pop(largest_key, None)
        else:
            trimmed[largest_key] = shrunk


def _compact_agent_brief_to_budget(brief: dict[str, Any]) -> dict[str, Any]:
    """Guarantee ``agent_brief.json`` stays strictly below ``AGENT_BRIEF_MAX_BYTES`` (8 KB)."""
    if _brief_utf8_len(brief) < AGENT_BRIEF_MAX_BYTES:
        return brief
    trimmed = dict(brief)
    for key in ("kernel_synthesis_specs", "semantic_audit_triggers"):
        val = trimmed.get(key)
        if isinstance(val, list) and len(val) > 2 and _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
            trimmed[key] = val[:2]
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("va_serving_rules"), Mapping):
        va_copy = dict(trimmed["va_serving_rules"])
        va_copy.pop("hw_isa_guidance", None)
        if isinstance(va_copy.get("capture_sizes"), list):
            va_copy["capture_sizes"] = va_copy["capture_sizes"][:BRIEF_MAX_CAPTURE_SIZES]
        trimmed["va_serving_rules"] = va_copy
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        trimmed["module_hierarchy"] = [
            {k: v for k, v in grp.items() if k != "children"}
            for grp in trimmed.get("module_hierarchy", ())
        ]
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        for key in ("kernel_synthesis_specs", "semantic_audit_triggers"):
            val = trimmed.get(key)
            if isinstance(val, list) and len(val) > 1:
                trimmed[key] = val[:1]
        wall = trimmed.get("macro_wall_trigger")
        if isinstance(wall, Mapping) and isinstance(wall.get("recommended_actions"), list):
            trimmed["macro_wall_trigger"] = {**wall, "recommended_actions": wall["recommended_actions"][:1]}
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("scenario_and_parallelism"), Mapping):
        sc = dict(trimmed["scenario_and_parallelism"])
        if sc.get("module_parallel_domains"):
            sc["module_parallel_domains"] = _trim_text(str(sc["module_parallel_domains"]), BRIEF_TEXT_LIMIT)
        sc.pop("recommended_checkpoint", None)
        trimmed["scenario_and_parallelism"] = sc
    for drop_key in (
        "va_serving_rules", "kernel_synthesis_specs", "semantic_audit_triggers",
        "macro_wall_trigger", "generated_files",
    ):
        if _brief_utf8_len(trimmed) < AGENT_BRIEF_MAX_BYTES:
            break
        if drop_key in trimmed:
            trimmed[drop_key] = [] if isinstance(trimmed[drop_key], list) else ({} if isinstance(trimmed[drop_key], Mapping) else None)
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES and isinstance(trimmed.get("action_plan_3_tracks"), Mapping):
        plan = dict(trimmed["action_plan_3_tracks"])
        for list_key in ("kernel_tasks", "framework_tasks", "config_and_env"):
            items = plan.get(list_key)
            if isinstance(items, list) and len(items) > 2:
                plan[list_key] = items[:2]
        plan["parallelism_codesign_note"] = _trim_text(str(plan.get("parallelism_codesign_note") or ""), BRIEF_TEXT_LIMIT)
        trimmed["action_plan_3_tracks"] = plan
    _trim_keys_to_budget(trimmed, frozenset({"schema", "summary"}))
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        _trim_keys_to_budget(trimmed, frozenset({"schema"}))
    if _brief_utf8_len(trimmed) >= AGENT_BRIEF_MAX_BYTES:
        _trim_keys_to_budget(trimmed, frozenset())
    return trimmed


def build_agent_brief(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact (< 8 KB) deterministic JSON brief for Agent workflows."""
    after = after_bundle or before_bundle
    macro_rows = build_macro_module_breakdown(before_bundle, after_bundle)
    aligned = (
        align_before_after_operators(before_bundle, after)
        if after_bundle is not None
        else tuple(
            AlignedComparisonRow(op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count, op.submodule, (op,), (op,), "1->1", "")
            for op in before_bundle.operators
        )
    )
    pending = build_pending_kernel_details(
        aligned, after, before_bundle=(before_bundle if after_bundle is not None else None),
    )
    roadmap = build_actionable_roadmap(before_bundle, after, pending)
    codesign = build_parallelism_codesign_note(before_bundle, after)
    par_short, par_detail = format_parallelism_codomain_summary(after)
    mem = after.memory_ledger

    l1_groups: list[dict[str, Any]] = []
    by_l1: dict[str, list[MacroModuleRow]] = defaultdict(list)
    for row in macro_rows:
        by_l1[row.l1_group].append(row)
    tot_aft = max(sum(r.after_ms for r in macro_rows), CONSERVATION_EPS_MS)
    for l1_name, _ in MACRO_L1_HIERARCHY:
        children = by_l1.get(l1_name)
        if not children:
            continue
        l1_bef = sum(c.before_ms for c in children)
        l1_aft = sum(c.after_ms for c in children)
        l1_groups.append({
            "l1": l1_name.split(" (", 1)[0],
            "before_ms": round(l1_bef, 3), "after_ms": round(l1_aft, 3),
            "share_pct": round(100.0 * l1_aft / tot_aft, 1),
            "children": [
                {
                    "l2": c.module_name.split(" (", 1)[0],
                    "exposed_ms": round(c.after_ms, 3), "before_ms": round(c.before_ms, 3),
                    "saved_us_per_tok_die": round(c.saved_us_per_tok_die, 2),
                }
                for c in children
            ],
        })

    cfg_items = [
        {"title": _trim_text(r.title, 44), "saved_ms": round(r.expected_gain_ms, 3), "cmd": _trim_text(r.action_detail, 92)}
        for r in roadmap if r.track == TRACK_CONFIG_ENV
    ][:BRIEF_MAX_FRAMEWORK_TASKS]
    fw_items = [
        {"title": _trim_text(r.title, 44), "saved_ms": round(r.expected_gain_ms, 3), "work": _trim_text(r.action_detail, 72)}
        for r in roadmap if r.track == TRACK_FRAMEWORK
    ][:BRIEF_MAX_FRAMEWORK_TASKS]
    kern_items = [
        {
            "card_id": item.card_id, "prio": f"P{item.priority}",
            "cat": "new" if item.category == PENDING_KIND_TO_IMPLEMENT else "opt",
            "kernel": _trim_text(item.target_kernel, 40),
            "saved_ms": round(item.expected_gain_ms, 3),
            "saved_us_per_tok_die": round(item.saved_us_per_tok_die, 2),
            "before_ms": round(item.before_model_ms, 3),
            "after_ms": round(item.after_model_ms, 3),
            "binding": item.binding_state,
            "summary": _trim_text(item.kernel_action or item.default_design_slot, 60),
        }
        for item in sorted(pending, key=lambda x: (-x.saved_us_per_tok_die, -x.expected_gain_ms))[:BRIEF_MAX_KERNEL_TASKS]
    ]
    b_kpi = _bundle_kpi_dict(before_bundle)
    a_kpi = _bundle_kpi_dict(after)
    speedup = round(b_kpi["step_ms"] / a_kpi["step_ms"], 2) if a_kpi["step_ms"] > 0.0 else 1.0

    synth_brief = [
        {
            "kernel": sp.kernel_name, "tier": sp.abi_tier,
            "sram_ok": sp.sram_verified,
            "ub_kb": f"{sp.ub_peak_bytes / 1024:.1f}/{sp.ub_budget_bytes / 1024:.1f}",
            "math": _trim_text(" ; ".join(sp.symbolic_equations), 88),
        }
        for sp in list(after.kernel_synthesis_specs)[:BRIEF_MAX_SPECS]
    ]
    trig_brief = [
        {
            "id": tr.trigger_id, "kind": tr.kind,
            "ops": list(tr.blocked_ops[:4]),
            "gain_us_tok_die": round(tr.potential_gain_us_per_tok_die, 2),
            "reason": _trim_text(tr.blocking_reason, 72),
            "refs": list(tr.source_refs[:2]),
        }
        for tr in sorted(after.semantic_audit_triggers, key=lambda t: -t.potential_gain_us_per_tok_die)[:BRIEF_MAX_TRIGGERS]
    ]
    eff_total_dies = (
        int(after.deployment_total_dies)
        if int(getattr(after, "deployment_total_dies", 0) or 0) > 0
        else mem.total_dies
    )
    eff_cluster_conc = (
        int(after.cluster_concurrency)
        if int(getattr(after, "cluster_concurrency", 0) or 0) > 0
        else (mem.concurrency * mem.dp)
    )
    scen_block: dict[str, Any] = {
        "model": after.model_name, "hardware_sku": after.hardware_sku,
        "topology_preset": after.topology_preset, "phase": after.phase,
        "deployment_mode": after.deployment_mode, "total_dies": eff_total_dies,
        "dp": mem.dp, "tp": mem.tp, "ep": mem.ep, "pp": mem.pp,
        "dcp": mem.dcp, "pcp": mem.pcp, "local_tp": mem.local_tp,
        "sequence_parallel": after.sequence_parallel,
        "parallelism_co_domain": par_short,
        "module_parallel_domains": par_detail,
        "prompt_tokens": after.prompt_tokens, "output_tokens": after.output_tokens,
        "concurrency_per_dp": mem.concurrency, "cluster_concurrency": eff_cluster_conc,
        "max_hbm_concurrency_per_dp": mem.max_hbm_concurrency_per_dp,
        "spec_method": after.spec_method, "spec_proposals": after.spec_proposals,
        "avg_accepted_tokens": after.avg_accepted_tokens,
        "weight_quant": mem.weight_quant_display or mem.weight_quant,
        "cache_dtype": mem.cache_dtype,
        "native_bf16_weight_gib_per_die": round(mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "quantized_weight_gib_per_die": round(mem.weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
        "weight_compression_ratio": round(mem.weight_compression_ratio, 2),
        "fits_hbm": mem.fits_hbm,
        "recommended_checkpoint": after.recommended_checkpoint,
    }
    if after.pd_pool_summary:
        scen_block["pd_pool"] = dict(after.pd_pool_summary)
    brief: dict[str, Any] = {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": scen_block,
        "kpis_before_vs_after": {
            "before": b_kpi, "after": a_kpi,
            "saved_step_ms": float(b_kpi["step_ms"] - a_kpi["step_ms"]),
            "saved_us_per_tok_die": float(b_kpi["us_per_tok_die"] - a_kpi["us_per_tok_die"]),
            "speedup": speedup,
            "sheet4_cards_total_gain_ms": float(sum(x.expected_gain_ms for x in pending)),
            "sheet4_cards_total_gain_us_per_tok_die": float(sum(x.saved_us_per_tok_die for x in pending)),
        },
        "module_hierarchy": l1_groups,
        "action_plan_3_tracks": {
            "config_and_env": cfg_items, "framework_tasks": fw_items,
            "kernel_tasks": kern_items, "parallelism_codesign_note": _trim_text(codesign, 140),
        },
        "kernel_synthesis_specs": synth_brief,
        "semantic_audit_triggers": trig_brief,
        "macro_wall_trigger": after.macro_wall_trigger.to_dict() if after.macro_wall_trigger is not None else None,
        "va_serving_rules": after.va_serving_rules_summary,
        "generated_files": dict(generated_files or {}),
    }
    if script_audit is not None:
        raw_issues = script_audit.get("issues_found") or ()
        brief["script_audit"] = {
            "source_script": script_audit.get("source_script", ""),
            "issues_count": len(raw_issues),
            "top_issues": [
                {"id": i.get("issue_id", ""), "title": _trim_text(str(i.get("title", "")), 40)}
                for i in list(raw_issues)[:BRIEF_MAX_FRAMEWORK_TASKS]
            ],
            "two_step_gain": script_audit.get("two_step_gain", {}),
            "optimized_script_path": script_audit.get("optimized_script_path", ""),
        }
    return _compact_agent_brief_to_budget(brief)


def export_agent_brief(
    first: Mapping[str, Any] | ExportBundle,
    second: str | Path | ExportBundle | None,
    third: str | Path | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
) -> Path | dict[str, Any]:
    """Atomically persist compact (< 8 KB) ``agent_brief.json``.

    Supports both ``export_agent_brief(brief_dict, path) -> Path`` and
    ``export_agent_brief(before_bundle, after_bundle, path, ...) -> dict``.
    """
    return_dict = False
    if isinstance(first, ExportBundle):
        after_b = second if isinstance(second, ExportBundle) else None
        if third is None:
            raise ValueError("export_agent_brief_requires_output_path")
        brief = build_agent_brief(
            first, after_b, generated_files=generated_files, script_audit=script_audit,
        )
        target = Path(third)
        return_dict = True
    else:
        if not isinstance(second, (str, Path)):
            raise ValueError("export_agent_brief_requires_output_path")
        brief = _compact_agent_brief_to_budget(dict(first))
        target = Path(second)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded_len = len(payload.encode("utf-8"))
    if encoded_len <= AGENT_BRIEF_PRETTY_LIMIT_BYTES:
        pretty = json.dumps(brief, ensure_ascii=False, indent=1) + "\n"
        if len(pretty.encode("utf-8")) < AGENT_BRIEF_MAX_BYTES:
            payload = pretty
            encoded_len = len(payload.encode("utf-8"))
    if encoded_len >= AGENT_BRIEF_MAX_BYTES:
        raise ModelingError(f"agent_brief_exceeds_max_bytes:{encoded_len}>={AGENT_BRIEF_MAX_BYTES}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except TimeoutError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return brief if return_dict else target


def format_executive_digest_markdown(brief: Mapping[str, Any]) -> str:
    """Format a crisp ~25-line executive Markdown digest for stdout or chat."""
    sc = brief.get("scenario_and_parallelism") or {}
    kp = brief.get("kpis_before_vs_after") or {}
    bef = kp.get("before") or {}
    aft = kp.get("after") or {}
    plan = brief.get("action_plan_3_tracks") or {}
    is_modeling = str(aft.get("analysis_mode") or "") == MODE_MODELING or bool(aft.get("has_unbound_formula_ops"))
    aft_col = "After [卡SLA目标建模终态]" if is_modeling else "After [仿真对比]"
    bind_badge = format_binding_badge(str(aft.get("binding_state") or BINDING_STATE_BOUND))
    lines = [
        f"## MindIE-Sim Executive Digest: {sc.get('model')} @ {sc.get('hardware_sku')} {bind_badge}",
        (
            f"- **Deploy**: {sc.get('total_dies')} Dies ({sc.get('parallelism_co_domain') or ''}), "
            f"{sc.get('deployment_mode')}, Prompt={sc.get('prompt_tokens')}, "
            f"Concurrency={sc.get('concurrency_per_dp')}/DP (Max HBM={sc.get('max_hbm_concurrency_per_dp')})"
        ),
        (
            f"- **Weights/Die**: {sc.get('quantized_weight_gib_per_die')} GiB "
            f"(原生 BF16: {sc.get('native_bf16_weight_gib_per_die')} GiB, 压缩 {sc.get('weight_compression_ratio')}x, "
            f"Quant={sc.get('weight_quant')}, KV={sc.get('cache_dtype')})"
        ),
        "",
        f"| KPI | Before [当前栈仿真基线] | {aft_col} | Delta / Verdict |",
        "| --- | ---: | ---: | --- |",
        f"| Step Latency (ms) | {bef.get('step_ms')} | **{aft.get('step_ms')}** | -{kp.get('saved_step_ms')} ms (**{kp.get('speedup')}x**) |",
        f"| Effective TPOT (ms) | {bef.get('effective_tpot_ms')} ({bef.get('sla_verdict')}) | **{aft.get('effective_tpot_ms')} ({aft.get('sla_verdict')})** | SLA <= {aft.get('sla_target_ms')} ms ({aft.get('sla_regime')}) |",
        f"| Amortized Cost (us/tok·die) | {bef.get('us_per_tok_die')} | **{aft.get('us_per_tok_die')}** | -{kp.get('saved_us_per_tok_die')} us/tok·die |",
        f"| Single-Die / Cluster TPS | {bef.get('single_die_tps')} / {bef.get('cluster_tps')} | **{aft.get('single_die_tps')} / {aft.get('cluster_tps')}** | tok/s |",
        f"| Physical Limit Attainment (L*/Step) | {bef.get('l_star_attainment_pct')}% | **{aft.get('l_star_attainment_pct')}%** | L*={aft.get('l_star_ms')} ms |",
        "",
        "### 3-Track Action Plan (`Config & Env` / `Framework` / `Kernel`)",
    ]
    wall = brief.get("macro_wall_trigger")
    if isinstance(wall, Mapping) and wall.get("wall_kind") not in {None, "", "BALANCED"}:
        acts = wall.get("recommended_actions") or ()
        act_str = f": {acts[0]}" if acts else ""
        lines.append(f"- **Macro Wall (`{wall.get('wall_kind')}`)**{act_str}")
    for cfg in (plan.get("config_and_env") or ())[:2]:
        lines.append(f"- **[Config/Env]** {cfg.get('title')}: `{cfg.get('cmd') or cfg.get('command_or_flag')}`")
    for fw in (plan.get("framework_tasks") or ())[:2]:
        lines.append(f"- **[Framework]** {fw.get('title')} (-{fw.get('saved_ms')} ms): {fw.get('work')}")
    for kt in (plan.get("kernel_tasks") or ())[:3]:
        lines.append(
            f"- **[Kernel `{kt.get('card_id')}`]** {kt.get('kernel')} "
            f"({kt.get('before_ms')}->{kt.get('after_ms')} ms, -{kt.get('saved_us_per_tok_die')} us/tok·die): {kt.get('summary')}"
        )
    if plan.get("parallelism_codesign_note"):
        lines.append(f"- **Co-Design**: {plan['parallelism_codesign_note']}")
    if brief.get("generated_files"):
        files_str = ", ".join(f"{k}=`{v}`" for k, v in brief["generated_files"].items())
        lines.append(f"- **Artifacts**: {files_str}")
    return "\n".join(lines) + "\n"
