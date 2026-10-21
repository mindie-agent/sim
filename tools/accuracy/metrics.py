"""Pure scoring, identity and work-model functions for the A2/A3 accuracy round.

[Role] One dependency-free module that both the assembler and the scorer import, so a
  metric or an identity key has exactly one definition in the whole round.
[Invariants]
  * ``geometry_key`` reproduces the split declarations byte for byte (payload, key
    order and float coercion included) — verified against all 389 declared points.
  * The scored target is ``median of per-trial medians``; the pooled-event median is a
    diagnostic only. Repeats, trials and events never increase independent N.
  * A metric never silently drops an uncovered condition: the coverage denominator is
    every declared condition and ``wape_penalized`` charges an abstention 100 % error.
  * Bootstrap resamples *conditions* (the cluster), B=10000, seed=20261021 by default,
    percentile interval; a cell with n < ``MIN_CI_N`` reports ``insufficient_n`` and no
    interval instead of a fabricated range.
  * No SKU constant lives here: hardware keys, peaks, work models and band edges are
    arguments supplied by a declaration.
[Timeouts & Locks] No I/O, no clock, no randomness beyond the declared seed.
[Side-Effect Policy] Pure functions; never reads a file, never mutates an argument.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

# ------------------------------------------------------------------ schemas ---
OBSERVATION_SCHEMA = "mindie-sim.observations/1"
PREDICTIONS_SCHEMA = "mindie-sim.frozen-predictions/1"
FREEZE_SCHEMA = "mindie-sim.freeze-receipt/1"
SCORE_SCHEMA = "mindie-sim.score/1"
ACCEPTANCE_SCHEMA = "mindie-sim-accuracy-acceptance/2"
PROTOCOL_ACCEPTANCE_SCHEMA = "mindie-sim-accuracy-acceptance/1"

# The exact top-level keys of one observation, in the declared order. ``runs`` and
# ``diagnostics`` are the two additive extensions this round mandates (merged runs and
# the pooled-median / alias / origin diagnostics); the assembler asserts the set.
OBSERVATION_KEYS = (
    "condition_key", "hardware_key", "die", "arch_family", "family", "recipe",
    "protocol_signature", "run_id", "split", "shape", "elements", "flops",
    "hbm_bytes", "duration_ms", "raw_samples_ms", "source_class", "timing_scope",
    "runs", "diagnostics",
)
SPLITS = ("train", "heldout", "excluded")
SOURCE_CLASSES = ("measured", "observed", "derived", "synthetic")

# Protocol fields that identify a *timing protocol* for the prediction/outcome join.
# ``warmup_scope`` and per-call amortisation are excluded here on purpose: no round-1
# measurement attests them, and including them would turn an attestation gap into 100 %
# non-coverage. They stay in ``diagnostics`` and join under ``--strict-protocol``.
CORE_PROTOCOL_FIELDS = (
    "timer", "timing_scope", "warmup", "iterations", "trials", "preallocated",
    "cache_policy", "measurement_scope", "attention_scope",
)
STRICT_PROTOCOL_FIELDS = CORE_PROTOCOL_FIELDS + ("warmup_scope", "amortization")
UNKNOWN = "unknown"

# ------------------------------------------------------------------- basics ---
DEFAULT_BOOTSTRAP_B = 10_000
DEFAULT_SEED = 20_261_021
MIN_CELL_N = 5          # a band/family gate needs >= 5 conditions
MIN_CI_N = 5            # below this a CI is not reported


def canonical_json(value: Any) -> str:
    """Stable JSON used for every hash in the round (sorted, compact, no NaN)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_text(canonical_json(value))


def truncate_hash(digest: str, length: int = 24) -> str:
    return digest[:length]


# ----------------------------------------------------------------- identity ---
def geometry_key(hardware_key: str, family: str, shape: Mapping[str, Any],
                 recipe: Any = None, execution_kind: Any = None,
                 worker_recipe: Any = None, precision: Any = None,
                 layout: str = "contiguous", quantization: str = "none",
                 features: Any = None) -> str:
    """Bare physical geometry identity — identical to the split declarations.

    Protocol and provenance are excluded by construction (leakage rule L1/L6): the
    declared points call this with every optional argument left at ``None``.
    """
    payload = {
        "hardware_key": hardware_key, "family": family, "recipe": recipe,
        "execution_kind": execution_kind, "worker_recipe": worker_recipe,
        "precision": precision, "layout": layout, "quantization": quantization,
        "features": features or {},
        "shape": {str(k): float(v) for k, v in shape.items()},
    }
    return "geo:" + truncate_hash(sha256_value(payload))


def _itemsize(dtype: Any, explicit: Any = None) -> float:
    if explicit:
        return float(explicit)
    name = str(dtype or "").lower().replace("torch.", "")
    if name in {"bfloat16", "bf16", "float16", "fp16", "half", "int16"}:
        return 2.0
    if name in {"float32", "fp32", "float", "int32", "uint32"}:
        return 4.0
    if name in {"float64", "fp64", "int64", "uint64"}:
        return 8.0
    return 1.0


def _identity_number(value: Any) -> str:
    """Exact, stable text for an identity key: integers stay integers.

    ``:g`` formatting is deliberately avoided — six significant digits would make
    1048576 and 1048575 collide on a byte-copy payload axis.
    """
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


# Declared shape-key aliases: the round-1 query manifests name the attention axes
# ``q``/``kv`` and the RoPE axes ``tokens``/``width``, while the measuring artifacts
# name the same axes ``q_tokens``/``kv_tokens`` and ``seq``/``dim``. One physical
# condition must produce one identity string, so the aliases are folded here, once.
SHAPE_KEY_ALIASES = {
    "q_tokens": "q", "kv_tokens": "kv", "seq": "tokens", "dim": "width",
    "seq_len": "tokens", "hidden": "width",
}


def normalize_shape_keys(shape: Mapping[str, Any]) -> dict[str, Any]:
    return {SHAPE_KEY_ALIASES.get(str(key), str(key)): value for key, value in shape.items()}


def shape_identity(family: str, shape: Mapping[str, Any], dtype: Any = None,
                   itemsize: Any = None) -> str:
    """Family-aware shape equality used for split matching and alias detection.

    A byte copy is one contiguous payload: ``rows*row_bytes`` (native uint8 schema) and
    ``elements`` (generic campaign schema, bf16 elements) are the same physical work, so
    both normalise to total payload bytes (leakage rule L7, review-evidence copy_identity).
    """
    values = normalize_shape_keys(shape)
    if family in {"copy", "copy_bytes"} or family.startswith("graph_copy"):
        if "rows" in values and "row_bytes" in values:
            payload = float(values["rows"]) * float(values["row_bytes"])
        elif "bytes" in values:
            payload = float(values["bytes"])
        elif "elements" in values:
            payload = float(values["elements"]) * _itemsize(dtype, itemsize)
        else:
            raise KeyError(f"copy_shape_invalid:{sorted(values)}")
        return f"copy_bytes:{_identity_number(payload)}"
    if len(values) == 1 and "elements" in values:
        return f"{family}:elements={_identity_number(values['elements'])}"
    parts = ",".join(f"{k}={_identity_number(v)}" for k, v in sorted(values.items()))
    return f"{family}:{parts}"


def condition_key(hardware_key: str, die: str, family: str, shape: Mapping[str, Any],
                  recipe: Any, protocol_signature_value: str,
                  precision: Any = None, layout: Any = None,
                  quantization: Any = None, features: Any = None,
                  execution_kind: Any = None, worker_recipe: Any = None,
                  amortization: Any = None) -> str:
    """Row identity: two measurements merge only when all of this is identical.

    ``die`` and ``protocol_signature`` are part of the key, so measurements on different
    dies or under different protocols are separate rows and can never be averaged into
    one observation (requirement: never merge different protocols or dies).
    """
    payload = {
        "hardware_key": hardware_key, "die": die, "family": family,
        "shape": {str(k): float(v) for k, v in shape.items()},
        "recipe": recipe, "execution_kind": execution_kind,
        "worker_recipe": worker_recipe, "precision": precision, "layout": layout,
        "quantization": quantization, "features": features or {},
        "protocol_signature": protocol_signature_value,
        "amortization": amortization,
    }
    return "cond:" + truncate_hash(sha256_value(payload))


def alias_group_key(hardware_key: str, die: str, family: str,
                    protocol_signature_value: str, payload: Any) -> str:
    """Equal-payload alias group: different factorisations of one physical payload."""
    payload_doc = {"hardware_key": hardware_key, "die": die, "family": family,
                   "protocol_signature": protocol_signature_value, "payload": payload}
    return "alias:" + truncate_hash(sha256_value(payload_doc), 16)


# ----------------------------------------------------------------- protocol ---
def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _normalize_protocol_value(name: str, value: Any) -> Any:
    if value is None or value == UNKNOWN:
        return UNKNOWN
    if name == "trials":
        try:
            return int(value)
        except (TypeError, ValueError):
            return UNKNOWN
    if name in {"warmup", "iterations", "amortization"}:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return UNKNOWN
        return int(number) if number.is_integer() else number
    if name == "preallocated":
        return bool(value)
    return str(value)


def protocol_core(protocol: Mapping[str, Any], *,
                  defaults: Mapping[str, Any] | None = None,
                  record_level: Mapping[str, Any] | None = None,
                  strict: bool = False) -> dict[str, Any]:
    """Canonicalise a protocol block into the declared timing-identity fields.

    Returns ``{field: value}`` where every absent field is the literal ``"unknown"``.
    Absence never wildcards a known field: two conditions whose protocol differs only by
    one side being silent produce different signatures and are never merged.

    ``record_level`` supplies fields the artifact attests outside its protocol block
    (e.g. nine-model records ``timing_scope`` as a sibling of ``protocol``);
    ``defaults`` is the source-declared protocol with its evidence citation recorded by
    the assembler. Neither is a wildcard — a default is a declared value.
    """
    source = dict(protocol or {})
    lifted = dict(record_level or {})
    fallback = dict(defaults or {})
    fields = STRICT_PROTOCOL_FIELDS if strict else CORE_PROTOCOL_FIELDS

    def pick(name: str) -> Any:
        value = _first(source, name)
        if value is None:
            value = _first(source, {"warmup": "warmup_iterations",
                                    "iterations": "samples_per_trial"}.get(name, name))
        if value is None:
            value = _first(lifted, name)
        if value is None:
            value = _first(fallback, name)
        return _normalize_protocol_value(name, value)

    core = {}
    for name in fields:
        core[name] = pick(name)
    if "iterations" in core and core["iterations"] == UNKNOWN:
        # A per-sample amortised block is declared under several names; try them all.
        for alias in ("samples_per_condition", "samples_per_case", "samples_per_trial",
                      "iterations"):
            value = _first(source, alias) or _first(fallback, alias)
            if value is not None:
                core["iterations"] = _normalize_protocol_value("iterations", value)
                break
    if "trials" in core and core["trials"] == UNKNOWN:
        value = _first(lifted, "trials") or _first(fallback, "trials")
        if value is not None:
            core["trials"] = int(value)
    if strict:
        core.setdefault("amortization", _normalize_protocol_value(
            "amortization", _first(source, "repeats_per_sample") or _first(fallback, "amortization")))
    return core


_PROTOCOL_ALIASES = {"warmup": ("warmup", "warmup_iterations"),
                     "iterations": ("iterations", "samples_per_trial",
                                    "samples_per_condition", "samples_per_case")}


def protocol_origins(protocol: Mapping[str, Any], *,
                     record_level: Mapping[str, Any] | None = None,
                     defaults: Mapping[str, Any] | None = None,
                     strict: bool = False) -> dict[str, str]:
    """Where each protocol field came from: attested / record / default / unknown."""
    fields = STRICT_PROTOCOL_FIELDS if strict else CORE_PROTOCOL_FIELDS
    origins: dict[str, str] = {}
    for name in fields:
        names = _PROTOCOL_ALIASES.get(name, (name,))

        def present(mapping: Mapping[str, Any] | None) -> bool:
            return bool(mapping) and any(mapping.get(key) is not None for key in names)

        if present(protocol):
            origins[name] = "attested"
        elif present(record_level):
            origins[name] = "record_level"
        elif present(defaults):
            origins[name] = "declared_default"
        else:
            origins[name] = "unknown"
    return origins


def protocol_signature(core: Mapping[str, Any], *, strict: bool = False) -> str:
    fields = STRICT_PROTOCOL_FIELDS if strict else CORE_PROTOCOL_FIELDS
    payload = {name: core.get(name, UNKNOWN) for name in fields}
    return "proto:" + truncate_hash(sha256_value(payload), 16)


def protocol_matches(left: Mapping[str, Any], right: Mapping[str, Any], *,
                     strict: bool = False) -> bool:
    fields = STRICT_PROTOCOL_FIELDS if strict else CORE_PROTOCOL_FIELDS
    return all(_normalize_protocol_value(n, left.get(n)) ==
               _normalize_protocol_value(n, right.get(n)) for n in fields)


# --------------------------------------------------------------- work model ---
class WorkModelError(ValueError):
    """A declared work model could not be evaluated for a shape."""


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
                  ast.UnaryOp, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load,
                  ast.Pow)


def _compile_expression(expression: str) -> Callable[[Mapping[str, float]], float]:
    tree = ast.parse(expression.strip(), mode="eval")

    def evaluate(node: ast.AST, names: Mapping[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body, names)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise WorkModelError(f"unknown_shape_variable:{node.id}")
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand, names)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left, names)
            right = evaluate(node.right, names)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right if right else 0.0
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise WorkModelError(f"unsupported_expression:{expression!r}")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise WorkModelError(f"unsupported_expression:{expression!r}")

    return lambda names: evaluate(tree, names)


class WorkModel:
    """Evaluates the *declared* work model of each family (no SKU constant in code)."""

    def __init__(self, regime_rule: Mapping[str, Any]):
        self.constants = dict(regime_rule.get("constants") or {})
        raw = dict(regime_rule.get("work_models") or {})
        self._exact: dict[str, dict[str, Callable[[Mapping[str, float]], float]]] = {}
        self._prefix: list[tuple[str, dict[str, Callable[[Mapping[str, float]], float]]]] = []
        for family, spec in raw.items():
            formulas: dict[str, Callable[[Mapping[str, float]], float]] = {}
            for term in str(spec).split(";"):
                term = term.strip()
                if not term:
                    continue
                if "=" not in term:
                    raise WorkModelError(f"work_model_term_without_equals:{family}:{term}")
                name, expression = term.split("=", 1)
                formulas[name.strip()] = _compile_expression(expression)
            if family.endswith("*"):
                self._prefix.append((family[:-1], formulas))
            else:
                self._exact[family] = formulas
        self._prefix.sort(key=lambda item: len(item[0]), reverse=True)

    def formulas(self, family: str) -> dict[str, Callable[[Mapping[str, float]], float]]:
        if family in self._exact:
            return self._exact[family]
        for prefix, formulas in self._prefix:
            if family.startswith(prefix):
                return formulas
        raise WorkModelError(f"no_work_model:{family}")

    def demand(self, family: str, shape: Mapping[str, Any]) -> dict[str, float]:
        names = {str(k): float(v) for k, v in shape.items()}
        try:
            return {term: evaluate(names) for term, evaluate in self.formulas(family).items()}
        except KeyError as exc:  # shape keys are floats; a missing key is a data defect
            raise WorkModelError(f"missing_shape_variable:{family}:{exc}") from exc

    def lower_bound_ms(self, family: str, shape: Mapping[str, Any]) -> float:
        demand = self.demand(family, shape)
        peak = self.constants.get("peak_bf16_ops_per_s")
        vector = self.constants.get("vector_ops_per_s")
        hbm = self.constants.get("hbm_bytes_per_s")
        terms = []
        if demand.get("flops") and peak:
            terms.append(demand["flops"] / peak)
        if demand.get("ops") and vector:
            terms.append(demand["ops"] / vector)
        if demand.get("bytes") and hbm:
            terms.append(demand["bytes"] / hbm)
        if not terms:
            raise WorkModelError(f"no_declared_capacity:{family}")
        return 1000.0 * max(terms)


def band_edges(bands: Mapping[str, str]) -> tuple[float | None, float | None]:
    """Extract (launch_edge_ms, large_edge_ms) from the declared band rule strings.

    The declaration carries the bands as data (``"launch": "analytic_lower_bound_ms
    < 0.05"``); reading the numbers keeps the thresholds out of the code. Returns
    ``(None, None)`` when the rule is not in the declared shape, which makes every
    condition ``unbanded`` rather than silently mis-banded.
    """
    def first_number(text: Any) -> float | None:
        if not isinstance(text, str):
            return None
        found = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        return float(found[0]) if found else None

    return first_number(bands.get("launch")), first_number(bands.get("large"))


def band_of(lower_bound_ms: float | None, edges: tuple[float | None, float | None]) -> str | None:
    launch_edge, large_edge = edges
    if lower_bound_ms is None or launch_edge is None or large_edge is None:
        return None
    if lower_bound_ms < launch_edge:
        return "launch"
    if lower_bound_ms >= large_edge:
        return "large"
    return "mid"


# --------------------------------------------------------- family identity ---
# One physical shape can carry different family labels in different campaigns: round 1
# calls the native uint8 byte copy ``copy_bytes`` and the generic bf16 copy ``copy``; the
# native masked-softmax conditions were declared under the softmax family while the
# artifact labels them ``masked_softmax``; the grouped-matmul conditions were declared as
# ``moe_gate_up``/``moe_down`` while the measuring recipe is one list API. Split matching
# therefore tries the family class plus its declared aliases and lets the shape decide.
FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "grouped_matmul": ("moe_gate_up", "moe_down"),
    "moe_gate_up": ("grouped_matmul",),
    "moe_down": ("grouped_matmul",),
    "masked_softmax": ("attention_softmax",),
}


def family_class(family: Any) -> str:
    text = str(family or "")
    if text in {"copy", "copy_bytes", "graph_copy"}:
        return "copy"
    return text


def identity_candidates(family: str, shape: Mapping[str, Any],
                        dtype: Any = None) -> list[str]:
    """Every family-aware identity string a condition may be declared under."""
    candidates: list[str] = []
    family_group = family_class(family)
    for candidate_family in (family_group, *FAMILY_ALIASES.get(family_group, ())):
        try:
            identity = shape_identity(candidate_family, shape, dtype=dtype)
        except KeyError:
            continue
        if identity not in candidates:
            candidates.append(identity)
    return candidates


# --------------------------------------------------------------- estimator ---
def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def trial_medians(trials: Sequence[Sequence[float]]) -> list[float]:
    return [statistics.median([float(x) for x in trial]) for trial in trials if trial]


def scored_target_ms(trials: Sequence[Sequence[float]]) -> tuple[float, list[float]]:
    """The scored target: median of per-trial medians (metrics.md §1)."""
    medians = trial_medians(trials)
    if not medians:
        raise ValueError("condition_has_no_samples")
    return statistics.median(medians), medians


def pooled_median_ms(trials: Sequence[Sequence[float]]) -> float | None:
    pooled = [float(x) for trial in trials for x in trial]
    return statistics.median(pooled) if pooled else None


def ape_pct(prediction_ms: float, observed_ms: float) -> float:
    return 100.0 * abs(prediction_ms - observed_ms) / observed_ms


def signed_error_pct(prediction_ms: float, observed_ms: float) -> float:
    return 100.0 * (prediction_ms - observed_ms) / observed_ms


def p90_nearest_rank(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def wape_pct(pairs: Iterable[tuple[float, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for prediction, observed in pairs:
        numerator += abs(prediction - observed)
        denominator += observed
    return 100.0 * numerator / denominator if denominator > 0 else None


def wape_penalized_pct(numeric: Sequence[tuple[float, float]],
                       uncovered_observed: Sequence[float]) -> float | None:
    """Abstention costs exactly what a 100 %-wrong prediction costs (metrics.md §7)."""
    numerator = sum(abs(p - o) for p, o in numeric)
    denominator = sum(o for _, o in numeric) + sum(uncovered_observed)
    return 100.0 * (numerator + sum(uncovered_observed)) / denominator if denominator > 0 else None


def mae_ms(pairs: Iterable[tuple[float, float]]) -> float | None:
    values = [abs(p - o) for p, o in pairs]
    return sum(values) / len(values) if values else None


def median_signed_bias_pct(pairs: Sequence[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return statistics.median([signed_error_pct(p, o) for p, o in pairs])


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranked = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2.0 + 1.0
            for position in range(index, end + 1):
                ranked[order[position]] = average
            index = end + 1
        return ranked

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def interval_containment(intervals: Sequence[tuple[float, float, float]]) -> dict[str, Any]:
    """``[(lower, upper, observed)]`` -> inclusive containment rate (metrics.md §6)."""
    if not intervals:
        return {"reported_intervals": 0, "contained": 0, "containment_rate": None}
    contained = sum(1 for low, high, observed in intervals if low <= observed <= high)
    return {"reported_intervals": len(intervals), "contained": contained,
            "containment_rate": contained / len(intervals)}


# --------------------------------------------------------------- bootstrap ---
def _percentile_interval(replicates: Sequence[float]) -> list[float] | None:
    if len(replicates) < 2:
        return None
    ordered = sorted(replicates)
    size = len(ordered)
    low = ordered[max(0, int(math.floor(0.025 * size)))]
    high = ordered[min(size - 1, int(math.ceil(0.975 * size)) - 1)]
    return [low, high]


def bootstrap_wape_ci(errors: Sequence[float], observed: Sequence[float], *,
                      b: int = DEFAULT_BOOTSTRAP_B,
                      seed: int = DEFAULT_SEED) -> list[float] | None:
    """Cluster bootstrap over conditions: the condition is the resampling unit.

    Returns ``None`` when n < ``MIN_CI_N`` (no fabricated precision claim) or when the
    resampled observed mass is degenerate.
    """
    size = len(errors)
    if size < MIN_CI_N or size != len(observed) or b <= 0:
        return None
    generator = random.Random(seed)
    randrange = generator.randrange
    replicates: list[float] = []
    for _ in range(b):
        numerator = 0.0
        denominator = 0.0
        for _ in range(size):
            index = randrange(size)
            numerator += errors[index]
            denominator += observed[index]
        if denominator > 0:
            replicates.append(100.0 * numerator / denominator)
    return _percentile_interval(replicates)


def bootstrap_median_ci(values: Sequence[float], *, b: int = DEFAULT_BOOTSTRAP_B,
                        seed: int = DEFAULT_SEED) -> list[float] | None:
    size = len(values)
    if size < MIN_CI_N or b <= 0:
        return None
    generator = random.Random(seed + 1)
    randrange = generator.randrange
    replicates = []
    for _ in range(b):
        sample = [values[randrange(size)] for _ in range(size)]
        replicates.append(statistics.median(sample))
    return _percentile_interval(replicates)


def paired_bootstrap_delta(model_errors: Sequence[float], baseline_errors: Sequence[float],
                           observed: Sequence[float], *, b: int = DEFAULT_BOOTSTRAP_B,
                           seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Paired ΔWAPE = WAPE(model) − WAPE(baseline) on the *same* resampled conditions."""
    size = len(observed)
    if size != len(model_errors) or size != len(baseline_errors):
        raise ValueError("paired_bootstrap_length_mismatch")
    observed_mass = sum(observed)
    if observed_mass <= 0:
        raise ValueError("paired_bootstrap_zero_observed_mass")
    point = 100.0 * (sum(model_errors) - sum(baseline_errors)) / observed_mass
    if size < MIN_CI_N or b <= 0:
        return {"delta_wape_pct": point, "ci95": None, "insufficient_n": size < MIN_CI_N}
    generator = random.Random(seed + 2)
    randrange = generator.randrange
    replicates = []
    for _ in range(b):
        model_total = 0.0
        baseline_total = 0.0
        denominator = 0.0
        for _ in range(size):
            index = randrange(size)
            model_total += model_errors[index]
            baseline_total += baseline_errors[index]
            denominator += observed[index]
        if denominator > 0:
            replicates.append(100.0 * (model_total - baseline_total) / denominator)
    return {"delta_wape_pct": point, "ci95": _percentile_interval(replicates),
            "insufficient_n": False}


# ----------------------------------------------------------------- summary ---
WEIGHTING = {"wape": "duration_weighted_over_covered_conditions",
             "ape": "count_weighted_over_conditions"}


def condition_row(condition_id: str, prediction_ms: float | None, observed_ms: float,
                  *, region: str | None = None, status: str | None = None,
                  lower_bound_ms: float | None = None,
                  upper_bound_ms: float | None = None) -> dict[str, Any]:
    numeric = prediction_ms is not None
    row = {
        "condition_id": condition_id,
        "prediction_ms": prediction_ms,
        "observed_ms": observed_ms,
        "numeric": numeric,
        "region": region,
        "status": status,
        "ape_pct": ape_pct(prediction_ms, observed_ms) if numeric else None,
        "signed_error_pct": signed_error_pct(prediction_ms, observed_ms) if numeric else None,
    }
    if lower_bound_ms is not None and upper_bound_ms is not None:
        row["reported_interval_ms"] = [lower_bound_ms, upper_bound_ms]
        row["observed_inside_reported_interval"] = (
            lower_bound_ms <= observed_ms <= upper_bound_ms)
    return row


def summarize(rows: Sequence[Mapping[str, Any]], *, n_declared: int | None = None,
              b: int = DEFAULT_BOOTSTRAP_B, seed: int = DEFAULT_SEED,
              subset_of: str | None = None) -> dict[str, Any]:
    """One metric cell. ``rows`` must contain every declared condition of the cell.

    Coverage denominator: ``n_declared`` when given (every declared condition, including
    ones with no measurement at all), else ``len(rows)``. Unsupported / unresolved rows
    count in the denominator and are charged 100 % error by ``wape_penalized_pct``.
    """
    declared = len(rows) if n_declared is None else max(n_declared, len(rows))
    covered = [r for r in rows if r.get("numeric") and r.get("prediction_ms") is not None]
    uncovered_observed = [float(r["observed_ms"]) for r in rows
                          if not (r.get("numeric") and r.get("prediction_ms") is not None)]
    pairs = [(float(r["prediction_ms"]), float(r["observed_ms"])) for r in covered]
    apes = [float(r["ape_pct"]) for r in covered]
    errors = [abs(p - o) for p, o in pairs]
    observed_values = [o for _, o in pairs]
    in_envelope = [(float(r["prediction_ms"]), float(r["observed_ms"])) for r in covered
                   if region_class(r.get("region")) == "in_envelope"]
    extrapolated = [(float(r["prediction_ms"]), float(r["observed_ms"])) for r in covered
                    if region_class(r.get("region")) == "extrapolated"]
    intervals = [(row["reported_interval_ms"][0], row["reported_interval_ms"][1],
                  float(row["observed_ms"])) for row in rows
                 if "reported_interval_ms" in row]
    summary = {
        "n": len(rows),
        "n_declared": declared,
        "numeric": len(covered),
        "coverage": (len(covered) / declared) if declared else 0.0,
        "coverage_denominator": "all_declared_conditions_including_unsupported_and_transition_unresolved",
        "weighting": dict(WEIGHTING),
        "wape_pct": wape_pct(pairs),
        "wape_ci95": bootstrap_wape_ci(errors, observed_values, b=b, seed=seed),
        "wape_in_envelope_pct": wape_pct(in_envelope) if in_envelope else None,
        "wape_extrapolated_pct": wape_pct(extrapolated) if extrapolated else None,
        "wape_penalized_pct": wape_penalized_pct(pairs, uncovered_observed),
        "median_ape_pct": median(apes),
        "median_ape_ci95": bootstrap_median_ci(apes, b=b, seed=seed) if apes else None,
        "p90_ape_pct": p90_nearest_rank(apes),
        "max_ape_pct": max(apes) if apes else None,
        "mae_ms": mae_ms(pairs),
        "median_signed_bias_pct": median_signed_bias_pct(pairs),
        "spearman_rho": spearman_rho([p for p, _ in pairs], [o for _, o in pairs]),
        "insufficient_n": len(covered) < MIN_CELL_N,
        "subset_of": subset_of,
        **interval_containment(intervals),
        "statuses": status_histogram(rows),
    }
    return summary


def region_class(region: Any) -> str | None:
    """Product regions -> the two classes the protocol separates.

    The product reports ``exact``, ``interpolation``, ``interpolation_2d``,
    ``extrapolation`` and per-family variants of those; the protocol only needs
    "in envelope" (exact + any interpolation) versus "extrapolated" (labelled bonus).
    """
    text = str(region or "").lower()
    if not text:
        return None
    if text.startswith("extrapolat"):
        return "extrapolated"
    if text == "exact" or text.startswith("interpolation"):
        return "in_envelope"
    return None


def status_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "unreported"
        counts[str(status)] = counts.get(str(status), 0) + 1
    return dict(sorted(counts.items()))


def outlier_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """APE > 2x the cell median; kept in every metric, listed for diagnosis only."""
    covered = [r for r in rows if r.get("ape_pct") is not None]
    if len(covered) < 2:
        return []
    center = statistics.median([float(r["ape_pct"]) for r in covered])
    outliers = []
    for row in covered:
        if float(row["ape_pct"]) <= 2.0 * center:
            continue
        item = {"condition_id": row["condition_id"], "ape_pct": float(row["ape_pct"]),
                "signed_error_pct": float(row["signed_error_pct"])}
        if row.get("region") is not None:
            item["prediction_region"] = str(row["region"])
        outliers.append(item)
    return outliers
