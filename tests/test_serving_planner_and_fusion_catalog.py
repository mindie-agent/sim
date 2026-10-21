"""Tier 0 unit tests for Layer 4 serving planner, Layer 3 canonical fusion patterns,
and extended LLM/VLM/DiT model architecture profiles.

[Role]
  Verify pure analytical HBM KV/state capacity planning, P/D and EPD disaggregation
  ratio optimization, Phase 0 canonical Ascend fusion pattern evaluation/scanning,
  and extended architecture presets (MLA, GDN/KDA hybrid, cross-layer KV owner
  deduplication, Latent MoE, DiT USP + TeaCache).

[Invariants]
  * Pure Tier 0 in-memory execution (< 5s total, zero subprocesses, zero sleep).
  * Every test verifies quantitative mathematical invariants and boundary conditions
    rather than surface types or prose strings.
  * Recurrent/conv state (GDN/KDA) scales with ``max_num_seqs`` and NEVER with ``seq_len``.
  * Cross-layer KV sharing allocates physical blocks ONLY for declared owner layers.

[Timeouts & Locks]
  In-memory arithmetic only; no I/O locks or subprocesses.

[Side-Effect Policy]
  Read-only access to bundled JSON model presets; no file mutations.
"""
from __future__ import annotations

import pytest

from mindie_sim import (
    A3InputError,
    ARCHITECTURE_PRESET_CATALOG,
    CANONICAL_FUSION_PATTERNS,
    CATALOG,
    ServingPlannerError,
    SubgraphError,
    build_architecture_profile,
    evaluate_canonical_fusion_opportunity,
    list_architecture_profiles,
    load_architecture_profile,
    net_report,
    optimize_pd_disaggregation_ratio,
    plan_hbm_kv_capacity,
    scan_fusible_subgraph_patterns,
)


class TestHbmKvCapacityPlanner:
    """Verify HBM weight, paged KV block, EPLB redundant experts, recurrent state, and OOM bounds."""

    def test_mla_vs_gqa_kv_footprint_and_eplb_redundant_experts(self) -> None:
        # MLA: (kv_lora_rank=512 + qk_rope_head_dim=64) * 1 byte (fp8) = 576 B/layer/token
        # Across 61 layers = 35,136 B/token.
        mla_plan = plan_hbm_kv_capacity(
            {
                "num_layers": 61,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "kv_dtype": "fp8",
                "hbm_gib": 64.0,
                "gpu_memory_utilization": 0.90,
                "weight_bytes_per_rank": 32 * 1024**3,
                "single_expert_weight_bytes": 128 * 1024**2,
                "eplb_redundant_experts": 16,  # global across ep=8 -> 2 redundant experts/rank
                "workspace_mib": 1024.0,
                "aclgraph_bucket_count": 8,
                "aclgraph_per_bucket_mib": 32.0,
                "hccl_buffsize_mib": 200.0,
                "hccl_comm_groups": 2,
                "max_num_seqs": 128,
                "seq_len": 4096,
                "kv_block_size": 128,
                "parallel": {"dp": 1, "tp": 8, "ep": 8, "pp": 1, "dcp": 1},
            }
        )
        assert mla_plan["status"] == "fits"
        assert mla_plan["oom_reasons"] == []
        assert mla_plan["weight_budget"]["eplb_redundant_experts_per_rank"] == 2
        assert mla_plan["weight_budget"]["eplb_weight_bytes_per_rank"] == pytest.approx(2 * 128 * 1024**2)
        assert mla_plan["runtime_overhead_budget"]["aclgraph_pool_bytes_per_rank"] == pytest.approx(256 * 1024**2)
        assert mla_plan["runtime_overhead_budget"]["hccl_buffer_bytes_per_rank"] == pytest.approx(400 * 1024**2)
        assert mla_plan["kv_owner_deduplication"]["kv_bytes_per_owner_layer_per_token"] == pytest.approx(576.0)
        assert mla_plan["kv_owner_deduplication"]["kv_bytes_per_token_per_rank"] == pytest.approx(61 * 576.0)
        assert mla_plan["allocatable_kv_blocks"] > 0

        # Equivalent GQA 61-layer 8-KV-head dim-128 BF16 at TP=4:
        # (8 / 4) * 128 * 2 (K+V) * 2 B = 1024 B/layer/token -> fewer allocatable KV blocks than MLA FP8
        gqa_plan = plan_hbm_kv_capacity(
            {
                "num_layers": 61,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "kv_dtype": "bf16",
                "hbm_gib": 64.0,
                "gpu_memory_utilization": 0.90,
                "weight_bytes_per_rank": 32 * 1024**3,
                "workspace_mib": 1024.0,
                "hccl_buffsize_mib": 200.0,
                "hccl_comm_groups": 2,
                "max_num_seqs": 128,
                "seq_len": 4096,
                "kv_block_size": 128,
                "parallel": {"dp": 1, "tp": 4, "ep": 1, "pp": 1, "dcp": 1},
            }
        )
        assert gqa_plan["kv_owner_deduplication"]["kv_bytes_per_owner_layer_per_token"] == pytest.approx(1024.0)
        assert mla_plan["allocatable_kv_blocks"] > gqa_plan["allocatable_kv_blocks"]

    def test_recurrent_state_is_strictly_o1_in_seq_len_and_scales_with_max_num_seqs(self) -> None:
        # Qwen3.5-MoE has 36 GDN layers + 12 GQA layers (48 layers total)
        short_seq = plan_hbm_kv_capacity(
            {
                "model_preset": "qwen3.5-moe",
                "hbm_gib": 64.0,
                "tp_size": 4,
                "ep_size": 8,
                "max_num_seqs": 64,
                "seq_len": 2048,
            }
        )
        long_seq = plan_hbm_kv_capacity(
            {
                "model_preset": "qwen3.5-moe",
                "hbm_gib": 64.0,
                "tp_size": 4,
                "ep_size": 8,
                "max_num_seqs": 64,
                "seq_len": 65536,
            }
        )
        double_seqs = plan_hbm_kv_capacity(
            {
                "model_preset": "qwen3.5-moe",
                "hbm_gib": 64.0,
                "tp_size": 4,
                "ep_size": 8,
                "max_num_seqs": 128,
                "seq_len": 2048,
            }
        )
        assert short_seq["recurrent_state_budget"]["state_scales_with_seq_len"] is False
        assert short_seq["recurrent_state_budget"]["recurrent_state_bytes_per_rank"] > 0
        # Recurrent state bytes MUST NOT change when seq_len increases 32x (2048 -> 65536)
        assert (
            short_seq["recurrent_state_budget"]["recurrent_state_bytes_per_rank"]
            == pytest.approx(long_seq["recurrent_state_budget"]["recurrent_state_bytes_per_rank"])
        )
        assert short_seq["allocatable_kv_blocks"] == long_seq["allocatable_kv_blocks"]
        # Doubling max_num_seqs exactly doubles recurrent_state_bytes_per_rank
        assert (
            double_seqs["recurrent_state_budget"]["recurrent_state_bytes_per_rank"]
            == pytest.approx(2.0 * short_seq["recurrent_state_budget"]["recurrent_state_bytes_per_rank"])
        )

    def test_cross_layer_kv_owner_deduplication_and_oom_reporting(self) -> None:
        # MiniMax-M3 has 60 layers but only 30 physical KV owner layers (2:1 cross-layer KV sharing)
        dedup_plan = plan_hbm_kv_capacity(
            {
                "model_preset": "minimax-m3",
                "hbm_gib": 64.0,
                "tp_size": 4,
                "ep_size": 8,
                "max_num_seqs": 64,
                "seq_len": 8192,
            }
        )
        kv_dedup = dedup_plan["kv_owner_deduplication"]
        assert kv_dedup["num_owner_layers"] == 30
        assert kv_dedup["num_consumer_shared_layers"] == 30
        assert kv_dedup["kv_bytes_per_token_per_rank"] == pytest.approx(
            0.5 * kv_dedup["unshared_kv_bytes_per_token_per_rank"]
        )
        assert kv_dedup["deduplicated_kv_bytes_saved_per_token"] == pytest.approx(
            kv_dedup["kv_bytes_per_token_per_rank"]
        )

        # OOM when static weights + runtime exceed usable HBM
        oom_plan = plan_hbm_kv_capacity(
            {
                "model_preset": "minimax-m3",
                "hbm_gib": 64.0,
                "gpu_memory_utilization": 0.90,
                "weight_bytes_per_rank": 60 * 1024**3,
                "tp_size": 4,
                "ep_size": 8,
            }
        )
        assert oom_plan["status"] == "oom"
        assert "non_kv_deductions_exceed_usable_hbm" in oom_plan["oom_reasons"]
        assert oom_plan["allocatable_kv_blocks"] == 0
        assert oom_plan["available_kv_hbm_bytes"] < 0


class TestPdDisaggregationOptimizer:
    """Verify P/D and EPD instance ratio optimization, KV transfer modes, and SLO gates."""

    def test_pd_disaggregation_transfer_modes_and_colocated_vs_epd_comparison(self) -> None:
        base_pd = {
            "prefill_step_ms": 180.0,
            "decode_step_ms": 14.0,
            "prefill_concurrency": 2,
            "decode_concurrency": 64,
            "prefill_devices_per_instance": 8,
            "decode_devices_per_instance": 8,
            "prompt_len": 4096,
            "output_len": 512,
            "kv_bytes_per_request": 256 * 1024 * 1024,  # 256 MiB KV cache
            "transfer_bw_gbps": 25.0,  # 25 GB/s RDMA
            "ttft_slo_ms": 350.0,
            "tpot_slo_ms": 17.0,
            "target_qps": 20.0,
        }
        rdma = optimize_pd_disaggregation_ratio({**base_pd, "transfer_mode": "direct_rdma"})
        mooncake = optimize_pd_disaggregation_ratio(
            {**base_pd, "transfer_mode": "mooncake_layerwise", "layerwise_overlap_fraction": 0.85}
        )
        host_staged = optimize_pd_disaggregation_ratio(
            {**base_pd, "transfer_mode": "host_staged", "host_staging_bw_gbps": 12.0}
        )

        assert rdma["status"] == "modeled"
        assert rdma["slo_compliant"] is True
        # Layerwise overlap hides most of the KV transfer latency inside prefill step time
        assert mooncake["kv_transfer"]["exposed_kv_transfer_ms"] < rdma["kv_transfer"]["exposed_kv_transfer_ms"]
        assert mooncake["effective_ttft_ms"] < rdma["effective_ttft_ms"]
        # Host-staged incurs D2H + H2D staging penalty
        assert host_staged["kv_transfer"]["kv_transfer_ms"] > rdma["kv_transfer"]["kv_transfer_ms"]

        # Colocated mode suffers prefill interference and fails the tight 18ms TPOT SLO
        assert rdma["mode_comparison"]["colocated"]["slo_compliant"] is False
        assert rdma["mode_comparison"]["pd_disaggregated"]["slo_compliant"] is True
        assert rdma["mode_comparison"]["recommended_mode"] == "pd_disaggregated"

        # Multimodal EPD disaggregation with Vision Encoder + EC cache
        epd = optimize_pd_disaggregation_ratio(
            {
                **base_pd,
                "encoder_step_ms": 90.0,
                "encoder_concurrency": 4,
                "encoder_devices_per_instance": 2,
                "multimodal_request_fraction": 0.8,
                "ec_cache_hit_rate": 0.5,
                "ec_transfer_ms": 4.0,
            }
        )
        assert epd["encoder_instances"] >= 1
        assert epd["mode_comparison"]["epd_disaggregated"]["encoder_instances"] >= 1
        # Offloading Encoder from Prefill lowers P makespan and improves EPD TTFT vs inline PD
        assert epd["mode_comparison"]["epd_disaggregated"]["ttft_ms"] < epd["mode_comparison"]["pd_disaggregated"]["ttft_ms"]

    def test_pd_disaggregation_slo_and_asymmetric_tp_violations(self) -> None:
        violated = optimize_pd_disaggregation_ratio(
            {
                "prefill_step_ms": 400.0,
                "decode_step_ms": 28.0,
                "tp_p": 4,
                "tp_d": 8,  # Invalid: tp_p < tp_d is rejected by vLLM-Ascend asymmetric TP rule
                "ttft_slo_ms": 250.0,
                "tpot_slo_ms": 20.0,
            }
        )
        assert violated["status"] == "modeled"
        assert violated["slo_compliant"] is False
        assert violated["tp_topology_valid"] is False
        assert set(violated["slo_violations"]) == {
            "ttft_exceeds_slo",
            "tpot_exceeds_slo",
            "unsupported_asymmetric_tp_ratio",
        }

    def test_net_report_and_decompose_step_integrate_serving_planners(self) -> None:
        report = net_report(
            {
                "parallel": {"dp": 2, "tp": 4, "ep": 8, "pp": 1, "dcp": 1},
                "step_components_ms": {
                    "compute": 16.0,
                    "attention": 12.0,
                    "moe": 24.0,
                    "communication": 2.0,
                    "overhead": 0.5,
                },
                "hbm_capacity_plan": {
                    "model_preset": "longcat-flash-lite",
                    "hbm_gib": 64.0,
                    "max_num_seqs": 64,
                    "seq_len": 4096,
                },
                "pd_disaggregation": {
                    "prefill_step_ms": 120.0,
                    "decode_step_ms": 11.5,
                    "output_len": 512,
                    "decode_concurrency": 64,
                    "kv_bytes_per_request": 72 * 1024 * 1024,
                    "ttft_slo_ms": 300.0,
                    "tpot_slo_ms": 15.0,
                },
            }
        )
        step = report["step_decomposition"]
        assert step["status"] == "modeled"
        assert step["hbm_capacity_plan"]["status"] == "fits"
        assert step["hbm_capacity_plan"]["allocatable_kv_blocks"] > 0
        assert step["pd_disaggregation"]["status"] == "modeled"
        assert step["pd_disaggregation"]["slo_compliant"] is True


class TestCanonicalAscendFusionPatterns:
    """Verify Phase 0 canonical Ascend fusible subgraph patterns and trace scanner."""

    @pytest.mark.parametrize("pattern_name", tuple(CANONICAL_FUSION_PATTERNS.keys()))
    def test_all_seven_canonical_patterns_save_gm_traffic_and_launch_overhead(self, pattern_name: str) -> None:
        result = evaluate_canonical_fusion_opportunity(
            pattern_name,
            {
                "num_tokens": 2048,
                "hidden_size": 4096,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "top_k": 8,
                "moe_intermediate_size": 2048,
                "dtype": "bf16",
                "op_durations_us": [25.0, 18.0, 14.0],
            },
            arch_family="A3",
        )
        assert result["pattern_name"] == pattern_name
        assert result["status"] == "modeled"
        assert result["ub_l1_feasible"] is True
        assert result["eliminated_gm_roundtrip_bytes"] > 0
        assert result["eliminated_gm_read_bytes"] > 0
        assert result["eliminated_gm_write_bytes"] > 0
        assert result["kernel_launches_before"] == 3
        assert result["kernel_launches_after"] == 1
        assert result["kernel_launch_reduction"] == 2
        assert result["latency_savings_us_interval"] == pytest.approx([0.0, 16.0])

    def test_ub_overflow_flags_infeasible_when_auto_tile_disabled_on_large_tokens(self) -> None:
        result = evaluate_canonical_fusion_opportunity(
            "mla_prologue_epilogue",
            {
                "num_tokens": 4096,
                "hidden_size": 7168,
                "auto_tile": False,  # Forces full 4096 tokens into UB -> overflows 192 KiB UB
                "op_durations_us": [40.0, 20.0, 20.0],
            },
            arch_family="A3",
        )
        assert result["ub_l1_feasible"] is False
        assert result["status"] == "infeasible"
        assert "live_buffer_limit" in result["feasibility"]["failures"]
        assert result["latency_savings_us_interval"] is None
        assert result["unmodeled_reason"] == "infeasible_fusion"

    def test_scan_fusible_subgraph_patterns_detects_canonical_sequences(self) -> None:
        operator_sequence = [
            {"op": "aten.add", "duration_us": 12.0},
            {"op": "npu_rms_norm", "duration_us": 18.0},
            {"op": "npu_dynamic_quant", "duration_us": 10.0},
            {"op": "MatMul", "duration_us": 50.0},
            {"op": "moe_gating_top_k", "duration_us": 15.0},
            {"op": "log2phy", "duration_us": 8.0},
            {"op": "moe_distribute_dispatch", "duration_us": 30.0},
            {"op": "npu_grouped_matmul", "duration_us": 80.0},
            {"op": "npu_swiglu", "duration_us": 22.0},
            {"op": "dynamic_quant", "duration_us": 14.0},
        ]
        scan = scan_fusible_subgraph_patterns(
            operator_sequence,
            {"num_tokens": 4096, "hidden_size": 7168, "top_k": 8, "moe_intermediate_size": 2048},
            arch_family="A3",
        )
        assert scan["status"] == "modeled"
        assert scan["match_count"] == 3
        matched_names = [m["pattern_name"] for m in scan["matches"]]
        assert matched_names == [
            "add_rmsnorm_dynamic_quant",
            "moe_gating_topk_dispatch",
            "gmm_swiglu_dynamic_quant",
        ]
        assert scan["total_kernel_launch_reduction"] == 6
        assert scan["total_eliminated_gm_bytes"] == pytest.approx(
            sum(m["eliminated_gm_roundtrip_bytes"] for m in scan["matches"])
        )
        assert scan["total_latency_savings_us_interval"][1] > 0.0


class TestExtendedModelArchitectureCatalog:
    """Verify the 5 new LLM/VLM/DiT presets and backward compatibility with se-tools CATALOG."""

    def test_se_tools_catalog_retains_nine_models_unchanged(self) -> None:
        assert len(CATALOG) == 9
        assert len(ARCHITECTURE_PRESET_CATALOG) == 5
        preset_rows = list_architecture_profiles()
        assert len(preset_rows) == 5

    def test_five_extended_architecture_presets_and_validation_invariants(self) -> None:
        # 1. Qwen3.5-MoE: 3:1 GDN hybrid (48 GDN + 16 GQA = 64 layers)
        qwen35 = load_architecture_profile("qwen3.5-moe")
        assert qwen35.attention_type == "hybrid_gdn_gqa"
        assert qwen35.recurrent_variant == "gdn"
        assert len(qwen35.recurrent_layer_indices) == 48
        assert len(qwen35.physical_kv_owner_layers) == 16
        assert qwen35.total_state_bytes_per_seq(tp_size=4) > 0
        # GDN TP validation: 16 k_heads is not divisible by tp=6
        with pytest.raises(A3InputError, match="gdn_tp_inexact_k_heads"):
            qwen35.validate_tp(6)

        # 2. Bailing-V3: KDA + MLA hybrid (48 KDA + 16 MLA = 64 layers) + Latent MoE (1024 / 4096 = 0.25)
        bailing = load_architecture_profile("bailing-v3")
        assert bailing.recurrent_variant == "kda"
        assert len(bailing.recurrent_layer_indices) == 48
        assert len(bailing.physical_kv_owner_layers) == 16
        assert bailing.kv_lora_rank == 512
        assert bailing.qk_rope_head_dim == 64
        assert bailing.latent_moe_hidden_size == 1024
        assert bailing.moe_dispatch_token_width == 1024
        assert bailing.moe_dispatch_compression_ratio == pytest.approx(0.25)

        # 3. MiniMax-M3: Cross-layer KV sharing (30 owner layers + 30 consumer layers = 60 layers) + sparse indexer
        minimax = load_architecture_profile("minimax-m3")
        assert minimax.num_layers == 60
        assert len(minimax.physical_kv_owner_layers) == 30
        assert len(minimax.consumer_shared_kv_layers) == 30
        assert len(minimax.kv_consumer_to_owner) == 30
        assert minimax.index_head_dim == 128

        # 4. LongCat-Flash-Lite: MLA + N-gram + zero-compute identity experts
        longcat = load_architecture_profile("longcat-flash-lite")
        assert longcat.attention_type == "mla"
        assert longcat.num_layers == 14
        assert longcat.zero_expert_num == 128

        # 5. HunyuanVideo-DiT: 3D visual patches + TeaCache + USP (Ulysses x Ring)
        dit = load_architecture_profile("hunyuan-video-dit")
        assert dit.visual_tokens == 33 * 45 * 80
        assert dit.text_tokens == 256
        assert dit.total_dit_tokens == 33 * 45 * 80 + 256
        assert dit.dit_cache_method == "TeaCache"
        assert dit.ulysses_degree == 8
        assert dit.ring_degree == 1

        # DiT mutual exclusion invariant: TeaCache + Sparse Attention must fail
        with pytest.raises(A3InputError, match="dit_cache_and_sparse_attention_mutually_exclusive"):
            build_architecture_profile(
                {
                    "name": "invalid-dit",
                    "num_layers": 20,
                    "hidden_size": 3072,
                    "num_attention_heads": 24,
                    "head_dim": 128,
                    "dit_config": {
                        "video_height": 720,
                        "video_width": 1280,
                        "video_frames": 129,
                        "dit_cache_method": "TeaCache",
                        "sparse_attention_method": "topk",
                    },
                }
            )
