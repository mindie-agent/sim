"""Tier-0 runtime-agnostic stack registry, domain routing, and extensibility witness tests.

[Role] Verify dynamic zero-core-edit ``RuntimeStackAdapter`` registration
  (``"dummy-engine"`` witness through ``resolve_stack_for_scenario``,
  ``validate_scenario_contract``, ``LoweringRequest``, and ``PredictionSpine``),
  custom JSON manifest loading, domain-aware routing for LLM/VLM vs
  Diffusion/DiT, strict rejection of LLM stacks on Diffusion, built-in
  ``ideal-target`` / ``raw-eager`` / ``diffusion-native`` capability contracts,
  and ``AG010`` zero-SCC / layer-boundary hygiene under ``mindie_sim/stacks/``.
[Invariants] Pure in-memory Tier-0 verification; zero subprocesses, network, or
  host-environment mutations; all temp manifests stay inside ``tmp_path``;
  global registry mutations are always cleaned up in ``finally``.
[Timeouts & Locks] Sub-second in-memory execution; no external locks.
[Side-Effect Policy] Read-only over repository assets; restores global stack
  registry state before every test exits.
"""
from __future__ import annotations

from collections.abc import Mapping
import ast
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from _support.axes import GEN5_DT_SKU
from mindie_sim.compose import compose_case
from mindie_sim.prediction.operators.diffusion import MINDIE_SD_CUSTOM_OPS
from mindie_sim.prediction.results import LoweringRequest, ReplayOptions, StepRequest
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.schema.case import Case
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointProvenance,
    CheckpointQuantMap,
    LayerQuantException,
    TensorPresenceDomain,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.scenario import (
    ConcurrencyLimits,
    PoolSpec,
    SLAContract,
    ScenarioContract,
    validate_scenario_contract,
)
from mindie_sim.schema.stack import (
    CaptureBucketRule,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    ResolvedStackOptions,
    RuntimeQuantPlan,
    RuntimeStackAdapter,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    WaitForEvent,
)
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.stacks import (
    STACK_MANIFEST_SCHEMA,
    DiffusionNativeStackAdapter,
    DiffusionWorkloadConfig,
    IdealTargetStackAdapter,
    ManifestStackAdapter,
    RawEagerStackAdapter,
    RuntimeStackRegistry,
    get_stack,
    list_stacks,
    load_manifest_stack,
    normalize_hardware_backend,
    register_stack,
    resolve_stack_for_scenario,
    unregister_stack,
)
import scripts.check_arch_governance as arch_gov

pytestmark = pytest.mark.tier0


def _minimal_case() -> Case:
    raw = compose_case(
        "qwen3-32b",
        hardware=GEN5_DT_SKU,
        tp=1,
        layers=2,
        prompt_tokens=16,
        output_tokens=2,
        concurrency=1,
    )
    return parse_case(raw)


def _sample_llm_checkpoint() -> CheckpointQuantMap:
    return CheckpointQuantMap(
        provenance=CheckpointProvenance(
            checkpoint_id="org/mxfp8-layer-exception-ckpt",
            source_format="modelslim_description",
            source_revision="a" * 40,
            source_sha256="b" * 64,
        ),
        native_float_dtype="bf16",
        model_quant_type="W8A8_MXFP8",
        rules=(
            TensorQuantRule(
                pattern="model.layers.*.self_attn.o_proj",
                raw_tag="FLOAT",
                presence=TensorPresenceDomain(scope="target", layer_indices=(0, 1)),
            ),
            TensorQuantRule(
                pattern="model.layers.*.mlp.down_proj",
                raw_tag="W8A8_MXFP8",
                presence=TensorPresenceDomain(scope="target"),
            ),
        ),
        layer_exceptions=(
            LayerQuantException(
                pattern="model.layers.*.self_attn.o_proj",
                scope="target",
                layer_indices=(1,),
                raw_tag="W4A8_MXFP",
            ),
        ),
        ignored_modules=("model.layers.0.mlp.down_proj",),
        cache_rule=CacheQuantRule(
            kv_tag="C8",
            kv_scale_dtype="fp32",
            provenance="checkpoint",
        ),
    )


def _sample_diffusion_checkpoint() -> CheckpointQuantMap:
    return CheckpointQuantMap(
        provenance=CheckpointProvenance(
            checkpoint_id="org/dit-mxfp4-daos",
            source_format="modelslim_description",
            source_revision="c" * 40,
            source_sha256="d" * 64,
        ),
        native_float_dtype="bf16",
        model_quant_type="W4A4F4-DAOS",
        rules=(
            TensorQuantRule(
                pattern="transformer_blocks.*.attn.qkv_proj",
                raw_tag="W4A4F4-DAOS",
                presence=TensorPresenceDomain(scope="target"),
            ),
            TensorQuantRule(
                pattern="transformer_blocks.*.adaln.modulation",
                raw_tag="FLOAT",
                presence=TensorPresenceDomain(scope="target"),
            ),
        ),
        cache_rule=CacheQuantRule(kv_tag="C8", provenance="checkpoint"),
    )


class _DummyThirdPartyAdapter:
    """In-test third-party RuntimeStackAdapter proving zero-Core-edit extensibility."""

    def __init__(
        self,
        stack_id: str = "dummy-engine",
        *,
        supported_domains: tuple[str, ...] = ("llm", "vlm"),
        supported_work_units: tuple[str, ...] = ("prefill_decode_step", "vision_encode"),
        supported_hardware_backends: tuple[str, ...] = ("ascend", "nvidia"),
    ) -> None:
        self._cap = StackCapabilitySpec(
            stack_id=stack_id,
            manifest_revision="0.1.0-witness",
            supported_domains=supported_domains,  # type: ignore[arg-type]
            supported_work_units=supported_work_units,  # type: ignore[arg-type]
            supported_hardware_backends=supported_hardware_backends,
            supported_phases=("prefill", "decode", "mixed"),
            supported_checkpoint_formats=("modelslim_description", "unquantized_native"),
            supported_tracks=("eager", "shipping"),
            supported_recipe_ids=frozenset({"dummy_shipping_gemm"}),
            has_launch_codec=False,
        )

    def capabilities(self) -> StackCapabilitySpec:
        return self._cap

    def default_checkpoint_for(
        self, canonical_model: str, hardware: Hardware
    ) -> str | None:
        del canonical_model, hardware
        return None

    def resolve_options(
        self,
        model: ModelSpec,
        hardware: Hardware,
        parallel: ParallelSpec,
        *,
        role: str,
        track: StackTrack,
        requested_options: Mapping[str, Any],
        field_origins: Mapping[str, str],
    ) -> ResolvedStackOptions:
        del model, hardware, parallel, field_origins
        cap_rule = CaptureBucketRule(mode="full_decode_only", full_keys=(1, 2, 4, 8))
        return ResolvedStackOptions(
            stack_id=self._cap.stack_id,
            track=track,
            requested_mechanisms=tuple(sorted(requested_options.items())),
            effective_mechanisms=(("role", role), ("witness_engine", True)),
            dispositions=tuple((k, "accepted", "dummy_engine") for k in sorted(requested_options)),
            capture_rule=cap_rule,
            graph_capture_mode="full_decode_only",
            graph_capture_keys=(1, 2, 4, 8),
        )

    def resolve_quant_instantiation(
        self,
        model: ModelSpec,
        checkpoint: CheckpointQuantMap,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
        *,
        role: str,
        phase: str,
    ) -> RuntimeQuantPlan:
        del model, hardware, parallel, options, role, phase
        return RuntimeQuantPlan(
            checkpoint_digest=checkpoint.digest(),
            stack_id=self._cap.stack_id,
            provenance=checkpoint.provenance.origin,
        )

    def select_lowering_plan(
        self,
        model: ModelSpec,
        quant_plan: RuntimeQuantPlan,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
    ) -> StackLoweringPlan:
        del model, quant_plan, hardware, parallel
        return StackLoweringPlan(
            stack_id=self._cap.stack_id,
            track=options.track,
            lowering_passes=("activation_quantization",),
            allowed_rewrite_classes=frozenset({"shipping"}),
            allowed_rewrite_tracks=("config",),
            scheduling=SchedulingContract(
                capture_bucket_rule=options.capture_rule,
                stars_submit_policy=(StarsSubmitPolicy(replay_mode="full_graph"),),
            ),
        )

    def plan_iteration(
        self,
        model: ModelSpec,
        state: IterationSchedulerState,
        options: ResolvedStackOptions,
        *,
        phase: str,
    ) -> IterationDecision:
        del model, options
        if not state.ready_request_ids:
            if state.in_flight_completions:
                return WaitForEvent(reason="completion", until_ms=state.in_flight_completions[0][1])
            return SchedulerDone(completed_requests=len(state.request_progress))
        views = tuple(
            RequestView(request_id=rid, q_len=1, context=16, samples=(phase == "decode"), phase=phase)
            for rid in state.ready_request_ids[: state.max_num_seqs]
        )
        return IterationPlan(requests=views, all_request_ids=state.ready_request_ids)

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


def test_builtin_lazy_discovery_and_domain_defaults() -> None:
    installed = list_stacks()
    assert {"ideal-target", "raw-eager", "diffusion-native"} <= set(installed)
    for stack_id in ("ideal-target", "raw-eager", "diffusion-native"):
        adapter = get_stack(stack_id)
        assert isinstance(adapter, RuntimeStackAdapter)
        assert adapter.capabilities().stack_id == stack_id
        assert adapter.launch_codec is None

    assert normalize_hardware_backend("A5") == "ascend"
    assert normalize_hardware_backend("Ascend910_93") == "ascend"
    assert normalize_hardware_backend("dav_3510") == "ascend"
    assert normalize_hardware_backend("H100") == "nvidia"
    assert normalize_hardware_backend("custom_fpga2") == "custom_fpga2"

    # When an LLM-only vllm-ascend plugin is registered, LLM/VLM default to it while
    # Diffusion still defaults strictly to diffusion-native.
    isolated = RuntimeStackRegistry(auto_discover=False)
    isolated.register(_DummyThirdPartyAdapter("vllm-ascend", supported_hardware_backends=("ascend",)))
    isolated.register(IdealTargetStackAdapter())
    isolated.register(RawEagerStackAdapter())
    isolated.register(DiffusionNativeStackAdapter())

    llm_adapter = resolve_stack_for_scenario(
        None, "llm", "prefill_decode_step", "A5", registry=isolated,
    )
    vlm_adapter = resolve_stack_for_scenario(
        None, "vlm", "vision_encode", "Ascend910_93", registry=isolated,
    )
    dit_adapter = resolve_stack_for_scenario(
        None, "diffusion", "denoise_timestep", "A5", registry=isolated,
    )
    nvidia_llm = resolve_stack_for_scenario(
        None, "llm", "prefill_decode_step", "nvidia", registry=isolated,
    )
    assert llm_adapter.capabilities().stack_id == "vllm-ascend"
    assert vlm_adapter.capabilities().stack_id == "vllm-ascend"
    assert dit_adapter.capabilities().stack_id == "diffusion-native"
    assert nvidia_llm.capabilities().stack_id == "ideal-target"


def test_third_party_adapter_zero_core_edit_extensibility_witness() -> None:
    dummy = _DummyThirdPartyAdapter("dummy-engine")
    assert isinstance(dummy, RuntimeStackAdapter)

    with pytest.raises(ModelingError, match="stack_adapter_protocol_invalid"):
        register_stack(object())  # type: ignore[arg-type]

    register_stack(dummy)
    try:
        with pytest.raises(ModelingError, match="stack_already_registered"):
            register_stack(dummy)

        resolved = resolve_stack_for_scenario(
            "dummy-engine", "llm", "prefill_decode_step", "A5",
        )
        assert resolved is dummy

        case = _minimal_case()
        ckpt = _sample_llm_checkpoint()
        parallel = case.runtime.parallel
        opts = resolved.resolve_options(
            case.model, case.machine, parallel,
            role="colocated", track="shipping",
            requested_options={"batch_policy": "continuous"},
            field_origins={"batch_policy": "api_argument"},
        )
        qplan = resolved.resolve_quant_instantiation(
            case.model, ckpt, case.machine, parallel, opts,
            role="colocated", phase="decode",
        )
        lplan = resolved.select_lowering_plan(
            case.model, qplan, case.machine, parallel, opts,
        )

        # Live one-step iteration decisions: IterationPlan / WaitForEvent / SchedulerDone.
        active = resolved.plan_iteration(
            case.model,
            IterationSchedulerState(now_ms=0.0, ready_request_ids=("req-0",), max_num_seqs=4, max_num_batched_tokens=64),
            opts,
            phase="decode",
        )
        waiting = resolved.plan_iteration(
            case.model,
            IterationSchedulerState(now_ms=1.0, in_flight_completions=(("req-0", 5.5),)),
            opts,
            phase="decode",
        )
        done = resolved.plan_iteration(
            case.model,
            IterationSchedulerState(now_ms=6.0, request_progress=(("req-0", 16, 0),)),
            opts,
            phase="decode",
        )
        assert isinstance(active, IterationPlan) and active.valid_tokens == 1
        assert isinstance(waiting, WaitForEvent) and waiting.until_ms == 5.5
        assert isinstance(done, SchedulerDone) and done.completed_requests == 1

        table = load_primitives(GEN5_DT_SKU)
        contract = ScenarioContract(
            deployment_mode="colocated",
            deployment_mode_origin="api_argument",
            workload_domain="llm",
            work_unit_kind="prefill_decode_step",
            optimization_mode="simulation",
            execution_stage="before",
            model=case.model,
            checkpoint=ckpt,
            stack_id=resolved.capabilities().stack_id,
            stack_track="shipping",
            stack_options=opts,
            quant_plan=qplan,
            lowering_plan=lplan,
            hardware=case.machine,
            primitive_table_digest=table.envelope_sha256,
            sla=SLAContract(ttft_ms=None, tpot_ms=25.0),
            limits=ConcurrencyLimits(workload_requested_concurrency=4),
            workload=case.workload,
            num_devices=1,
            pools=(PoolSpec("main", "colocated", num_devices=1, device_ids=(0,)),),
        )
        validate_scenario_contract(contract)

        step_req = StepRequest(phase="decode", tokens=1, context=16, sequences=1, scope="body")
        lowering_req = LoweringRequest(step=step_req, scenario_contract=contract)
        spine = PredictionSpine()
        prepared = spine.lower(case, lowering_req)
        ev = spine.run_step(prepared, ReplayOptions(), primitive_table=table)
        assert prepared.scenario_contract is contract
        assert ev.stack_id == "dummy-engine"
        assert ev.scenario_digest == contract.digest()
        assert ev.span_ms > 0.0
    finally:
        unregister_stack("dummy-engine")

    assert "dummy-engine" not in list_stacks()


def test_custom_manifest_json_loading_and_capability_guards(tmp_path: Path) -> None:
    manifest_payload = {
        "schema": STACK_MANIFEST_SCHEMA,
        "stack_id": "custom-ascend-runtime",
        "manifest_revision": "2.3.1",
        "description": "Custom third-party runtime defined purely via JSON manifest.",
        "supported_domains": ["llm"],
        "supported_work_units": ["prefill_decode_step"],
        "supported_hardware_backends": ["ascend"],
        "supported_phases": ["prefill", "decode"],
        "supported_checkpoint_formats": ["modelslim_description", "unquantized_native"],
        "supported_tracks": ["eager", "shipping"],
        "capture_mode": "full_decode_only",
        "allowed_rewrite_tracks": ["config"],
        "allowed_rewrite_classes": ["shipping"],
        "supported_recipe_ids": ["custom_flash_decode"],
        "lowering_passes": ["activation_quantization"],
        "default_checkpoints": {"qwen3-32b": "org/qwen3-32b-custom"},
    }
    manifest_file = tmp_path / "custom_manifest.json"
    manifest_file.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    loaded = load_manifest_stack(manifest_file)
    assert isinstance(loaded, ManifestStackAdapter)
    assert isinstance(loaded, RuntimeStackAdapter)
    assert loaded.capabilities().stack_id == "custom-ascend-runtime"
    assert loaded.launch_codec is None

    # Resolve directly via --stack /path/to/custom_manifest.json string path.
    resolved = resolve_stack_for_scenario(
        str(manifest_file), "llm", "prefill_decode_step", "A5",
    )
    assert resolved.capabilities().stack_id == "custom-ascend-runtime"

    case = _minimal_case()
    ckpt = _sample_llm_checkpoint()
    assert resolved.default_checkpoint_for("qwen3-32b", case.machine) == "org/qwen3-32b-custom"
    opts = resolved.resolve_options(
        case.model, case.machine, case.runtime.parallel,
        role="colocated", track="shipping", requested_options={}, field_origins={},
    )
    qplan = resolved.resolve_quant_instantiation(
        case.model, ckpt, case.machine, case.runtime.parallel, opts,
        role="colocated", phase="decode",
    )
    lplan = resolved.select_lowering_plan(
        case.model, qplan, case.machine, case.runtime.parallel, opts,
    )
    contract = ScenarioContract(
        deployment_mode="colocated",
        deployment_mode_origin="cli_flag",
        workload_domain="llm",
        work_unit_kind="prefill_decode_step",
        optimization_mode="simulation",
        execution_stage="before",
        model=case.model,
        checkpoint=ckpt,
        stack_id=resolved.capabilities().stack_id,
        stack_track="shipping",
        stack_options=opts,
        quant_plan=qplan,
        lowering_plan=lplan,
        hardware=case.machine,
        primitive_table_digest="",
        workload=case.workload,
    )
    validate_scenario_contract(contract)
    req = LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=16, sequences=1), scenario_contract=contract)
    assert req.scenario_contract.stack_id == "custom-ascend-runtime"

    # Capability & malformed manifest guards.
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        resolve_stack_for_scenario(str(manifest_file), "diffusion", "denoise_timestep", "A5")
    with pytest.raises(ModelingError, match="stack_hardware_unsupported"):
        resolve_stack_for_scenario(str(manifest_file), "llm", "prefill_decode_step", "nvidia")
    with pytest.raises(ModelingError, match="stack_manifest_not_found"):
        resolve_stack_for_scenario(str(tmp_path / "missing.json"), "llm", "prefill_decode_step", "A5")

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema": "wrong/0", "stack_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ModelingError, match="stack_manifest_schema_invalid"):
        load_manifest_stack(bad_schema)


def test_diffusion_routing_guard_and_native_adapter_contract() -> None:
    # Default routing for Diffusion resolves strictly to diffusion-native.
    dit_adapter = resolve_stack_for_scenario(None, "diffusion", "denoise_timestep", "A5")
    assert isinstance(dit_adapter, DiffusionNativeStackAdapter)
    assert dit_adapter.capabilities().supported_domains == ("diffusion",)
    assert dit_adapter.work_unit == "denoise_timestep"

    # Isolated registry with ONLY an LLM stack NEVER falls back for diffusion.
    llm_only_reg = RuntimeStackRegistry(auto_discover=False)
    llm_only_reg.register(_DummyThirdPartyAdapter("vllm-ascend"))
    with pytest.raises(ModelingError, match="no_compatible_stack_for_domain"):
        resolve_stack_for_scenario(None, "diffusion", "denoise_timestep", "A5", registry=llm_only_reg)
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        resolve_stack_for_scenario("vllm-ascend", "diffusion", "denoise_timestep", "A5", registry=llm_only_reg)

    # Forcing diffusion-native onto LLM raises crisp stack_domain_mismatch.
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        resolve_stack_for_scenario("diffusion-native", "llm", "prefill_decode_step", "A5")
    with pytest.raises(ModelingError, match="workload_domain_work_unit_mismatch"):
        resolve_stack_for_scenario("diffusion-native", "diffusion", "prefill_decode_step", "A5")

    # Verify diffusion-native declares timesteps, patches, CFG branches, optional encoder/VAE
    # phases, binds prediction/operators/diffusion recipes, and fabricates ZERO LLM KV-Cache.
    case = _minimal_case()
    dit_ckpt = _sample_diffusion_checkpoint()
    opts = dit_adapter.resolve_options(
        case.model,
        case.machine,
        case.runtime.parallel,
        role="colocated",
        track="shipping",
        requested_options={
            "denoise_timesteps": 50,
            "step_cache_warmup": 2,
            "step_cache_interval": 2,
            "latent_patches": 2048,
            "cfg_branches": 2,
            "include_text_encode": True,
            "include_vae_decode": True,
            "sparse_attention_op": "rainfusion",
            "sparsity_ratio": 0.5,
            "block_size": 128,
            "kv_cache_dtype": "int8",
        },
        field_origins={},
    )
    eff = dict(opts.effective_mechanisms)
    assert eff["work_unit"] == "denoise_timestep"
    assert eff["denoise_timesteps"] == 50
    assert eff["effective_denoise_timesteps"] == 26
    assert eff["latent_patches"] == 2048
    assert eff["cfg_branches"] == 2
    assert eff["tokens_per_denoise_step"] == 4096
    assert eff["active_phases"] == ("text_encode", "denoise_step", "vae_decode")
    assert ("kv_cache_dtype", "rejected", "diffusion_forbids_llm_kv_cache_and_speculation") in opts.dispositions

    qplan = dit_adapter.resolve_quant_instantiation(
        case.model, dit_ckpt, case.machine, case.runtime.parallel, opts,
        role="colocated", phase="denoise_step",
    )
    assert qplan.kv_cache_abi == ()
    assert qplan.index_cache_abi == ()
    assert qplan.recurrent_state_abi == ()
    port_map = dict(qplan.port_abis)
    assert port_map[("target", 0, "attn.qkv_proj")].linear_method == "mx_fp4"
    assert port_map[("target", 0, "adaln.modulation")].linear_method == "dense"

    lplan = dit_adapter.select_lowering_plan(
        case.model, qplan, case.machine, case.runtime.parallel, opts,
    )
    bound_recipes = {recipe_id for _, recipe_id, _ in lplan.recipe_bindings}
    known_diffusion = {f"mindie_sd_{name}" for name in MINDIE_SD_CUSTOM_OPS}
    assert bound_recipes <= known_diffusion
    assert {
        "mindie_sd_adalayernorm",
        "mindie_sd_norm_rope_concat",
        "mindie_sd_eagle_ffn",
        "mindie_sd_rainfusion",
        "mindie_sd_quant_four_over_six_a5",
    } <= bound_recipes

    # Contract validation passes cleanly for diffusion + denoise_timestep (zero KV-Cache).
    dit_contract = ScenarioContract(
        deployment_mode="colocated",
        deployment_mode_origin="api_argument",
        workload_domain="diffusion",
        work_unit_kind="denoise_timestep",
        optimization_mode="simulation",
        execution_stage="before",
        model=case.model,
        checkpoint=dit_ckpt,
        stack_id="diffusion-native",
        stack_track="shipping",
        stack_options=opts,
        quant_plan=qplan,
        lowering_plan=lplan,
        hardware=case.machine,
        primitive_table_digest="",
    )
    validate_scenario_contract(dit_contract)

    # plan_iteration emits denoise_step with latent_patches * cfg_branches and context == 0.
    step_plan = dit_adapter.plan_iteration(
        case.model,
        IterationSchedulerState(now_ms=0.0, ready_request_ids=("img-0",), max_num_seqs=2, max_num_batched_tokens=8192),
        opts,
        phase="denoise_step",
    )
    assert isinstance(step_plan, IterationPlan)
    assert step_plan.requests[0].q_len == 4096
    assert step_plan.requests[0].context == 0
    assert step_plan.requests[0].execution_mode == "denoise_timestep"

    # LLM phases are explicitly rejected on diffusion-native.
    with pytest.raises(ModelingError, match="stack_phase_unsupported"):
        dit_adapter.resolve_quant_instantiation(
            case.model, dit_ckpt, case.machine, case.runtime.parallel, opts,
            role="colocated", phase="decode",
        )
    cfg = DiffusionWorkloadConfig.from_options({"denoise_timesteps": 20, "latent_patches": 1024, "cfg_branches": 1})
    assert cfg.tokens_per_denoise_step == 1024 and cfg.active_phases == ("denoise_step",)


def test_ideal_target_and_raw_eager_capability_and_precision_contracts() -> None:
    case = _minimal_case()
    ckpt = _sample_llm_checkpoint()
    table = load_primitives(GEN5_DT_SKU)

    # 1) ideal-target: supports llm/vlm/diffusion, enables full Pillar A/B/C exploration,
    #    while strictly honoring CheckpointQuantMap precision boundaries.
    ideal = resolve_stack_for_scenario("ideal-target", "llm", "prefill_decode_step", "A5")
    assert isinstance(ideal, IdealTargetStackAdapter)
    assert ideal.capabilities().supported_domains == ("llm", "vlm", "diffusion")

    ideal_opts = ideal.resolve_options(
        case.model, case.machine, case.runtime.parallel,
        role="colocated", track="target",
        requested_options={"optimization_mode": "modeling"},
        field_origins={},
    )
    ideal_qplan = ideal.resolve_quant_instantiation(
        case.model, ckpt, case.machine, case.runtime.parallel, ideal_opts,
        role="colocated", phase="decode",
    )
    ideal_ports = dict(ideal_qplan.port_abis)
    # Exact precedence: layer 1 exception W4A8_MXFP -> ignored layer 0 down_proj FLOAT -> rule FLOAT / W8A8_MXFP8.
    assert ideal_ports[("target", 0, "self_attn.o_proj")].linear_method == "dense"
    assert ideal_ports[("target", 1, "self_attn.o_proj")].linear_method == "mx_fp4"
    assert ideal_ports[("target", 0, "mlp.down_proj")].linear_method == "dense"
    assert ideal_ports[("target", 1, "mlp.down_proj")].linear_method == "mx_fp8"
    assert ideal_qplan.prepared_weights == ()
    assert len(ideal_qplan.kv_cache_abi) == case.model.layers

    ideal_lplan = ideal.select_lowering_plan(
        case.model, ideal_qplan, case.machine, case.runtime.parallel, ideal_opts,
    )
    assert ideal_lplan.allowed_rewrite_tracks == ("config", "framework", "kernel_new", "kernel_tune")
    assert ideal_lplan.allowed_rewrite_classes == frozenset({
        "shipping", "proposed_pr", "synthesized_target", "tiler_extension",
    })
    ideal_contract = ScenarioContract(
        deployment_mode="colocated",
        deployment_mode_origin="api_argument",
        workload_domain="llm",
        work_unit_kind="prefill_decode_step",
        optimization_mode="modeling",
        execution_stage="after",
        model=case.model,
        checkpoint=ckpt,
        stack_id="ideal-target",
        stack_track="target",
        stack_options=ideal_opts,
        quant_plan=ideal_qplan,
        lowering_plan=ideal_lplan,
        hardware=case.machine,
        primitive_table_digest=table.envelope_sha256,
        workload=case.workload,
    )
    validate_scenario_contract(ideal_contract)

    # Unknown non-empty weight tags fail loudly (never silently degraded to FLOAT).
    bad_ckpt = CheckpointQuantMap(
        provenance=CheckpointProvenance(checkpoint_id="org/bad-tag", source_format="modelslim_description"),
        native_float_dtype="bf16",
        rules=(TensorQuantRule(pattern="model.layers.*.mlp.down_proj", raw_tag="MYSTERY_QUANT_99"),),
    )
    with pytest.raises(ModelingError, match="unknown_checkpoint_quant_tag"):
        ideal.resolve_quant_instantiation(
            case.model, bad_ckpt, case.machine, case.runtime.parallel, ideal_opts,
            role="colocated", phase="decode",
        )

    # 2) raw-eager: capture_mode="eager", allowed_rewrite_tracks=(), zero fusions.
    eager = resolve_stack_for_scenario("raw-eager", "llm", "prefill_decode_step", "A5", track="eager")
    assert isinstance(eager, RawEagerStackAdapter)
    assert eager.capture_mode == "eager"
    assert eager.capabilities().supported_tracks == ("eager",)

    with pytest.raises(ModelingError, match="stack_track_unsupported"):
        resolve_stack_for_scenario("raw-eager", "llm", "prefill_decode_step", "A5", track="target")

    eager_opts = eager.resolve_options(
        case.model, case.machine, case.runtime.parallel,
        role="colocated", track="eager",
        requested_options={"capture_mode": "full", "enable_fusion": True},
        field_origins={},
    )
    assert dict(eager_opts.effective_mechanisms)["capture_mode"] == "eager"
    assert eager_opts.graph_capture_mode == "none"
    assert eager_opts.capture_rule is not None
    assert eager_opts.capture_rule.select_bucket(phase="decode", active_tokens=4) == (None, "none")
    assert ("capture_mode", "rejected", "raw_eager_disables_graph_capture_and_fusions") in eager_opts.dispositions

    eager_qplan = eager.resolve_quant_instantiation(
        case.model, ckpt, case.machine, case.runtime.parallel, eager_opts,
        role="colocated", phase="decode",
    )
    eager_lplan = eager.select_lowering_plan(
        case.model, eager_qplan, case.machine, case.runtime.parallel, eager_opts,
    )
    assert eager_lplan.allowed_rewrite_tracks == ()
    assert eager_lplan.lowering_passes == ()
    assert eager_lplan.recipe_bindings == ()

    eager_contract = ScenarioContract(
        deployment_mode="colocated",
        deployment_mode_origin="api_argument",
        workload_domain="llm",
        work_unit_kind="prefill_decode_step",
        optimization_mode="simulation",
        execution_stage="before",
        model=case.model,
        checkpoint=ckpt,
        stack_id="raw-eager",
        stack_track="eager",
        stack_options=eager_opts,
        quant_plan=eager_qplan,
        lowering_plan=eager_lplan,
        hardware=case.machine,
        primitive_table_digest=table.envelope_sha256,
        workload=case.workload,
    )
    validate_scenario_contract(eager_contract)
    spine = PredictionSpine()
    step_req = StepRequest(phase="decode", tokens=1, context=16, sequences=1, scope="body")
    ev_eager = spine.run_step(
        spine.lower(case, LoweringRequest(step=step_req, scenario_contract=eager_contract)),
        ReplayOptions(),
        primitive_table=table,
    )
    assert ev_eager.stack_id == "raw-eager" and ev_eager.span_ms > 0.0


def test_stacks_package_obeys_ag010_and_zero_scc_cycles() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stacks_dir = repo_root / "mindie_sim" / "stacks"
    parsed: list[tuple[PurePosixPath, ast.Module]] = []
    for py_file in sorted(stacks_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        rel = PurePosixPath(py_file.relative_to(repo_root).as_posix())
        parsed.append((rel, ast.parse(py_file.read_text(encoding="utf-8"), filename=rel.as_posix())))
    assert parsed, "expected mindie_sim/stacks Python modules on disk"
    ag010 = [
        violation
        for rel, tree in parsed
        for violation in arch_gov._check_ag010_layer_dag(rel, tree)
    ]
    ag010.extend(arch_gov._check_ag010_package_sccs(parsed))
    assert ag010 == []


def test_wildcard_checkpoint_rules_resolve_across_eager_and_ideal_stacks_and_legacy_modules_deleted() -> None:
    """Wildcard HF rules resolve cleanly across raw-eager & ideal-target; legacy stack shims are gone."""
    from dataclasses import replace
    import importlib.util
    from mindie_sim.catalog.checkpoint_reader import read_hf_quantization_config
    from mindie_sim.model_catalog import load_model

    v41_model = load_model("deepseek-v4.1-flash")
    composite_cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "modules_to_not_convert": ["lm_head"],
        },
    }
    hf_map = read_hf_quantization_config(composite_cfg, checkpoint_id="fixture/v41-wildcard")
    # Also verify explicit composite FP8_DYNAMIC+MXFP4_EXPERTS wildcard rule.
    composite_map = CheckpointQuantMap(
        provenance=replace(hf_map.provenance, checkpoint_id="fixture/v41-composite"),
        model_quant_type="FP8_DYNAMIC+MXFP4_EXPERTS",
        rules=(
            TensorQuantRule(pattern="*.experts.*", raw_tag="MXFP4", presence=TensorPresenceDomain(scope="target")),
            TensorQuantRule(pattern="*", raw_tag="FP8_DYNAMIC", presence=TensorPresenceDomain(scope="target")),
        ),
        cache_rule=CacheQuantRule(kv_tag="FP8_DYNAMIC"),
    )
    case = _minimal_case()
    for stack_id, track in (("raw-eager", "eager"), ("ideal-target", "target")):
        adapter = get_stack(stack_id)
        opts = adapter.resolve_options(
            v41_model, case.machine, case.runtime.parallel,
            role="colocated", track=track, requested_options={}, field_origins={},
        )
        for ckpt in (hf_map, composite_map):
            qplan = adapter.resolve_quant_instantiation(
                v41_model, ckpt, case.machine, case.runtime.parallel, opts,
                role="colocated", phase="decode",
            )
            assert qplan.plan_digest and qplan.port_abis

    repo_root = Path(__file__).resolve().parents[2]
    for retired in ("stack_registry", "stack_tracks", "runtime_profiles"):
        assert not (repo_root / "mindie_sim" / f"{retired}.py").exists()
        assert importlib.util.find_spec(f"mindie_sim.{retired}") is None
