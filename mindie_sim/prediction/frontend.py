"""Compatibility facade for structural model and iteration lowering.

[Role] Assemble body, iteration, and homogeneous layer lowering methods.
[Invariants] The original module remains the lookup point for existing exports and overrides.
[Timeouts & Locks] Imports and construction require no devices or locks.
[Side-Effect Policy] Declares the frontend and initializes per-instance state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.engram import lower_engram
from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, lower_ple
from mindie_sim.catalog.media import lower_media
from mindie_sim.catalog.ops import barrier_op, comm_op, gemm_op, vector_op
from mindie_sim.catalog.context_parallel import pcp_rank_chunk_ranges, split_pcp_pairs
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.subgraphs import (
    EventPortSpec,
    PortSet,
    StatePortSpec,
    TensorPortSpec,
)
from mindie_sim.schema.workload import selected_mode
from .attention import ATTENTION_BUILDERS, AttentionPolicy
from .blocks import RESIDUAL_BUILDERS, lowrank_hyper_final_mixer, norm
from .feedforward import FFN_BUILDERS, RoutingPolicy
from .frontend_features import FeatureConfig
from .graph import GraphPolicy, annotate_geometry
from .routing_realizations import RoutingRealization, select_realization
from .structure import normalize_model, stage_map
from . import family_attnres, sequence_parallel
from .implementation_selection import (
    resolve as resolve_implementations,
    LoweringRegion,
    lower_regions,
    selection_delivery,
    annotate as annotate_selections,
)
from .recover_ssm_commit import bind_recover_ssm_commits, is_recover_ssm_intent
from .recover_ssm_work import recover_ssm_enabled
from .recurrent_kda_helpers import strip_excluded_native_kda
from .frontend_body import _FrontendBody, _scope, _strip_recover_intents
from .frontend_iteration import (
    ANALYTICAL_SELECTIONS,
    SERVICE_FAMILIES,
    _FrontendIteration,
    _TRIGGERS,
    _media_counts,
    _reject_unmodeled_dbo,
    _selection_matches_any,
    _without_service_fragments,
    mark_analytical_runtime,
)
from .frontend_ports import ModelProgram, _declared_format, _ports_for_work, program_attached_to
from .frontend_repeat import (
    _FrontendRepeat,
    _apply_meta_plan,
    _compile_meta_plan,
    _plan_uses_prev_id,
    _retarget_id_only,
)


# Keep legacy imports available while lowering code uses natural literals.
globals().update(ZERO=0, ONE=1, ZERO_FLOAT=0.0)

_ENV_MODEL = 0
_ENV_RUNTIME = 1
_ENV_MACHINE = 2
_ENV_REPLICA = 3
_ENV_FEATURES = 4
FINAL_LAYER = -1


class Frontend(_FrontendBody, _FrontendIteration, _FrontendRepeat):
    def __init__(self, *, attention=None, ffn=None, residuals=None, graph=None,
                 attention_policy=None, routing=None, features=None,
                 resource_fusion=True):
        self.features = features
        self.resource_fusion = bool(resource_fusion)
        self.attention = {**ATTENTION_BUILDERS, **(attention or {})}
        self.ffn = {**FFN_BUILDERS, **(ffn or {})}
        self.residuals = {**RESIDUAL_BUILDERS, **(residuals or {})}
        self.graph = graph or GraphPolicy()
        self.attention_policy = attention_policy or AttentionPolicy()
        self.routing = routing or RoutingPolicy()
        self._prev_work_ops: tuple | None = None
        self._prev_arch_state: tuple | None = None
        self._prev_call_env: tuple | None = None
        self._prev_homo_env: tuple | None = None
