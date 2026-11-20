"""Prefill/decode pools on the same causal engine.

[Role] Schedule prefill, destination admission, cache handoff and decode.
[Invariants] Destination KV is reserved before transfer. The source retains
  state until the transfer completes. TTFT and token intervals include actual
  arrival, queue and handoff times. Transfer bytes are physical destination
  shards, independent of overlap. No post-hoc latency addition.
[Timeouts & Locks] Finite requests and active transfers only; no external work.
[Side-Effect Policy] Mutates only per-run state.
"""
import math
from collections import defaultdict
from collections.abc import Mapping

from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.coverage import combine_coverage
from mindie_sim.schema.common import ModelingError

from ..attention import AttentionPolicy
from ..capacity import Memory
from ..cost import CostModel
from ..dispatch import cohorts, submit_batch
from ..engine import Engine
from ..execution import Execution
from ..frontend import Frontend
from ..metrics import serving_boundaries, summarize
from ..pd_connector import PDConnector, PDHandoff, PDTransfer
from ..pd_ranges import address, owned_runs, transfers_for_runs
from ..scheduler import Scheduler
from ..structure import parse_case
from .handoff import (
    HEAD_COMPONENTS,
)
from .handoff import (
    _component_versions as _component_versions,
)
from .handoff import (
    _consumer_release_done as _consumer_release_done,
)
from .handoff import (
    _head_token_span as _head_token_span,
)
from .handoff import (
    _local_head_spans as _local_head_spans,
)
from .handoff import (
    _pd_handoff_binding as _pd_handoff_binding,
)
from .handoff import (
    _pending_release_targets as _pending_release_targets,
)
from .handoff import (
    _published_for as _published_for,
)
from .handoff import (
    _register_consumer_release as _register_consumer_release,
)
from .handoff import (
    _token_span as _token_span,
)
from .handoff import (
    _uncovered_intervals as _uncovered_intervals,
)
from .serving import (
    DEFAULT_PD_EVENT_LIMIT,
    DEFAULT_PD_TRACE_LIMIT,
    predict_pd,
)
from .serving import (
    _endpoint_execution as _endpoint_execution,
)
from .serving import (
    _memory_binding_report as _memory_binding_report,
)

__all__ = [
    "AttentionPolicy",
    "CostModel",
    "DEFAULT_PD_EVENT_LIMIT",
    "DEFAULT_PD_TRACE_LIMIT",
    "Engine",
    "Execution",
    "Frontend",
    "HEAD_COMPONENTS",
    "Mapping",
    "Memory",
    "ModelingError",
    "PDConnector",
    "PDHandoff",
    "PDTransfer",
    "Scheduler",
    "address",
    "cohorts",
    "combine_coverage",
    "defaultdict",
    "layer_cache_components",
    "math",
    "owned_runs",
    "parse_case",
    "predict_pd",
    "serving_boundaries",
    "submit_batch",
    "summarize",
    "transfers_for_runs",
]
