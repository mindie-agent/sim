"""Piecewise structural compiler and contention-aware composition.

[Role] Expose expression, guard, compiler, composition and diagnostic APIs.
[Invariants] Importing builds no model, reads no measurement and starts no work.
[Timeouts & Locks] No import-time I/O, waits or locks.
[Side-Effect Policy] No global model/device mutation.
"""
from .expressions import (
    Add, CeilDiv, Const, Div, Expr, FloorDiv, Interval, Max, Min, Mul, Neg, Sub,
    Sum, Var, as_fraction, evaluate, simplify, substitute,
)
from .guards import (
    CoreWaveGuard, Crossover, GraphBucketGuard, GuardCell, PartitionAudit,
    PipeDemand, PipeDominanceGuard, PipeEnvelope, ServiceMode, SplitKGuard,
    SplitKPolicy, StepGuard, StructuralPartition,
)
from .compiler import (
    AnalysisDomain, CompiledOperator, CompiledSubgraph, ImplementationPolicy,
    NativeProgram, NativeSchedule, NativeTask, OperatorCost,
    OperatorPiecewiseCompiler, PipeSupply, QuantLedger,
    RequestGeometry, SegmentAxis, SegmentFormula, SegmentTable, WorkloadGeometry,
    quant_ledger,
)
from .geometry import clamp_graph_bucket, ep_batch, local_batch
from .lowering import lower_layer_predictions
from .service_span import (
    attribute_rope,
    chunked_prefill_mean_ttft_ms,
    service_ttft_overhead_ms,
    ttft_wave_scale,
)
from .composition import (
    ClosedForm, EngineResult, ForkJoinResult, FusionResult, OperatorSegment,
    SerialResult, SubgraphCostFunction, compose_sequence,
    contention_envelope, lower_fusion, operator_segment, pipeline_makespan,
    run_engine, run_engine_work,
)
from .diagnostics import DiagnosticReport, boundary_distance, coverage_summary, explain

__all__ = [
    "Add", "AnalysisDomain", "CeilDiv", "CompiledOperator", "CompiledSubgraph",
    "Const", "CoreWaveGuard", "Crossover", "DiagnosticReport", "Div", "Expr",
    "FloorDiv", "ForkJoinResult", "FusionResult", "GraphBucketGuard", "GuardCell",
    "ImplementationPolicy", "Interval", "Max", "Min", "Mul", "NativeProgram",
    "NativeSchedule", "NativeTask", "Neg", "OperatorCost",
    "OperatorPiecewiseCompiler",
    "OperatorSegment", "PartitionAudit", "PipeDemand",
    "PipeDominanceGuard", "PipeEnvelope", "PipeSupply", "QuantLedger",
    "RequestGeometry",
    "SegmentAxis", "SegmentFormula", "SegmentTable", "SerialResult",
    "ServiceMode", "SplitKGuard", "SplitKPolicy", "StepGuard",
    "StructuralPartition", "Sub", "SubgraphCostFunction", "Sum", "Var",
    "WorkloadGeometry", "as_fraction", "attribute_rope", "boundary_distance",
    "chunked_prefill_mean_ttft_ms",
    "clamp_graph_bucket", "compose_sequence", "coverage_summary", "ep_batch",
    "evaluate", "explain", "local_batch", "lower_fusion",
    "lower_layer_predictions", "operator_segment", "contention_envelope",
    "pipeline_makespan", "quant_ledger", "run_engine", "run_engine_work",
    "service_ttft_overhead_ms", "simplify", "substitute", "ttft_wave_scale",
    "ClosedForm", "EngineResult",
]
