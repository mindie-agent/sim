"""DYNAMIC_BUFFER4 source dependencies for the MXA8W4 producer/consumer.

[Role] One bounded dependency emitter serves actual Engine and nominal cycles.
[Invariants] UB read, widened output, scale transpose, L1 and L0 slots retire
  on their source pipe events. MTE3 waits never block independent MTE2 or V.
  The nominal cycle has no resource arbitration; actual contention stays in Engine.
[Timeouts & Locks] Finite ring state; at most 4096 steps to find a periodic state.
[Side-Effect Policy] Pure plans and local Engine tasks; no device or measured fit.
"""
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError
from .matrix_cycle import MatrixCycle


@dataclass(frozen=True)
class Slots:
    input: int = 4
    widened: int = 4
    scale: int = 2
    l1: int = 2
    l0: int = 2

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in self.values()):
            raise ModelingError('mxa8w4_pipeline_requires_positive_integer_slots')

    def values(self):
        return self.input, self.widened, self.scale, self.l1, self.l0


@dataclass(frozen=True)
class Panel:
    activation: object | None
    activation_scale: object | None
    vectors: tuple[tuple, tuple]
    scales: tuple[bool, bool]
    micros: tuple[tuple, ...]

    def mapped(self, fn):
        return Panel(fn(self.activation) if self.activation is not None else None,
                     fn(self.activation_scale) if self.activation_scale is not None else None,
                     tuple(tuple(fn(v) for v in stages) for stages in self.vectors),
                     self.scales, tuple(tuple(fn(v) for v in pair) for pair in self.micros))


class Dependencies:
    def __init__(self, ready, slots):
        self.slots = slots
        self.tails = dict.fromkeys(('a', 'read0', 'read1', 'v0', 'v1', 'pub0', 'pub1', 'operand', 'cube'), ready)
        self.low = [[ready] * slots.input for _ in range(2)]
        self.wide = [[ready] * slots.widened for _ in range(2)]
        self.scale = [[ready] * slots.scale for _ in range(2)]
        self.b = [ready] * slots.l1
        self.a = [ready] * 2
        self.ascale = [ready] * 2
        self.l0 = [ready] * slots.l0
        self.read_idx = [0, 0]
        self.wide_idx = [0, 0]
        self.scale_idx = [0, 0]
        self.b_idx = self.l0_idx = 0
        self.a_idx = self.as_idx = 1
        self.a_ready = self.as_ready = ready

    def advance(self, panel, emit):
        def issue(name, stage, stream, *deps):
            event = emit(name, stage, (*deps, self.tails[stream]))
            self.tails[stream] = event
            return event
        if panel.activation_scale is not None:
            self.as_idx = (self.as_idx + 1) % 2
            self.as_ready = issue('activation_scale', panel.activation_scale, 'a', self.ascale[self.as_idx])
        if panel.activation is not None:
            self.a_idx = (self.a_idx + 1) % 2
            self.a_ready = issue('activation', panel.activation, 'a', self.a[self.a_idx])
        b_free = self.b[self.b_idx]
        publications = []
        for sub, stages in enumerate(panel.vectors):
            sr, sv, sp, wr, wv, wp = stages
            if panel.scales[sub]:
                low, scale = self.read_idx[sub], self.scale_idx[sub]
                read = issue(f'scale_read{sub}', sr, f'read{sub}', self.low[sub][low])
                vector = issue(f'scale_transpose{sub}', sv, f'v{sub}', read, self.scale[sub][scale])
                self.low[sub][low] = vector
                # The CrossCoreWaitFlag is queued on MTE3 before this panel's
                # first publish. It imposes no dependency on read or vector.
                published = issue(f'scale_publish{sub}', sp, f'pub{sub}', vector, b_free)
                self.scale[sub][scale] = published
                self.read_idx[sub] = (low + 1) % self.slots.input
                self.scale_idx[sub] = (scale + 1) % self.slots.scale
            low, wide = self.read_idx[sub], self.wide_idx[sub]
            read = issue(f'packed_read{sub}', wr, f'read{sub}', self.low[sub][low])
            vector = issue(f'widen{sub}', wv, f'v{sub}', read, self.wide[sub][wide])
            self.low[sub][low] = vector
            published = issue(f'publish{sub}', wp, f'pub{sub}', vector, b_free)
            self.wide[sub][wide] = published
            publications.append(published)
            self.read_idx[sub] = (low + 1) % self.slots.input
            self.wide_idx[sub] = (wide + 1) % self.slots.widened
        for index, (operand, cube) in enumerate(panel.micros):
            loaded = issue(f'operand{index}', operand, 'operand', *publications,
                           self.a_ready, self.as_ready, self.l0[self.l0_idx])
            computed = issue(f'cube{index}', cube, 'cube', loaded)
            self.l0[self.l0_idx] = computed
            self.l0_idx = (self.l0_idx + 1) % self.slots.l0
        # Source AIC->AIV uses PIPE_MTE1. Its B slot can be overwritten while
        # the final Cube operation still consumes the independent L0 slot.
        self.b[self.b_idx] = loaded
        self.a[self.a_idx] = loaded
        self.ascale[self.as_idx] = loaded
        self.b_idx = (self.b_idx + 1) % self.slots.l1
        return computed

    def cursor(self):
        return (*self.read_idx, *self.wide_idx, *self.scale_idx,
                self.b_idx, self.l0_idx, self.a_idx, self.as_idx)

    def values(self):
        return (*self.tails.values(), *(v for groups in (self.low, self.wide, self.scale)
                 for group in groups for v in group), *self.b, *self.a, *self.ascale,
                *self.l0, self.a_ready, self.as_ready)

    def normalized(self):
        origin = self.tails['cube']
        return self.cursor(), tuple(None if v == 0 else v - origin for v in self.values())

    def translate(self, delta):
        # Unused input anchors stay at absolute time zero. They may become
        # relevant only in the finite tail and must not move with a cycle.
        shift = lambda value: value + delta if value else 0
        self.tails = {k: shift(v) for k, v in self.tails.items()}
        for name in ('low', 'wide', 'scale'):
            setattr(self, name, [[shift(v) for v in group] for group in getattr(self, name)])
        for name in ('b', 'a', 'ascale', 'l0'):
            setattr(self, name, [shift(v) for v in getattr(self, name)])
        self.a_ready = shift(self.a_ready)
        self.as_ready = shift(self.as_ready)


@dataclass(frozen=True)
class SourcePipeline:
    count: int
    panel: object
    slots: Slots = Slots()
    activation_panels: int = 8
    scale_panels: int = 16

    def start(self, engine, name, ready, *, owner=''):
        complete = engine.future(name + ':complete')
        state = Dependencies(ready, self.slots)
        issued = retired = 0
        # Two UB rings, scale scratch, L1 and L0 bound the useful lookahead.
        # This is an event-generation window, not an extra physical buffer.
        window = max(sum(self.slots.values()) + 4,
                     2 * max(self.activation_panels, self.scale_panels) + 2)
        def issue():
            nonlocal issued
            if issued == self.count:
                return
            index = issued
            issued += 1
            def emit(label, stage, deps):
                return stage.submit(engine, f'{name}:panel{index}:{label}', deps,
                                    lambda x: x, owner)
            finished = state.advance(self.panel(index), emit)
            engine.submit(f'{name}:panel{index}:retire', after=(finished,), owner=owner,
                          on_complete=retire)
        def retire(_):
            nonlocal retired
            retired += 1
            if retired == self.count:
                engine.resolve(complete)
            else:
                issue()
        def begin(_):
            if not self.count:
                engine.resolve(complete)
            for _ in range(min(window, self.count)):
                issue()
        engine.submit(name + ':begin', after=(ready,), on_complete=begin, owner=owner)
        return complete


def pipeline_cycle(pattern, repeats, *, tail=(), slots=Slots(), max_steps=4096):
    """Exact max-plus completion dependencies; no contention or resource fit."""
    if type(repeats) is not int or repeats < 0 or (repeats and not pattern):
        raise ModelingError('mxa8w4_cycle_requires_nonnegative_count_and_pattern')
    values = []
    def collect(stage):
        value = float(stage)
        if value < 0 or not math.isfinite(value):
            raise ModelingError('mxa8w4_cycle_requires_finite_nonnegative_durations')
        values.append(value)
        return value
    pattern = tuple(p.mapped(collect) for p in pattern)
    tail = tuple(p.mapped(collect) for p in tail)
    denominator = max((v.as_integer_ratio()[1] for v in values), default=1)
    def ticks(value):
        n, d = value.as_integer_ratio()
        return n * (denominator // d)
    pattern = tuple(p.mapped(ticks) for p in pattern)
    tail = tuple(p.mapped(ticks) for p in tail)
    state = Dependencies(0, slots)
    emit = lambda _, duration, deps: max(deps, default=0) + duration
    done = stepped = jumped = period = period_time = 0
    prologue_panel = prologue_time = 0
    anchor = (0, state.normalized())
    anchor_panel = anchor_time = distance = 0
    power = 1
    total = repeats * len(pattern)
    while done < total:
        if not period and stepped >= max_steps:
            raise ModelingError(f'mxa8w4_cycle_period_not_found_within_budget:{max_steps}')
        state.advance(pattern[done % len(pattern)], emit)
        done += 1
        stepped += 1
        if period:
            continue
        distance += 1
        normalized = (done % len(pattern), state.normalized())
        if normalized == anchor:
            period = distance
            period_time = state.tails['cube'] - anchor_time
            prologue_panel, prologue_time = anchor_panel, anchor_time
            cycles = (total - done) // period
            jumped = cycles * period
            state.translate(cycles * period_time)
            done += jumped
        elif distance == power:
            anchor, anchor_panel, anchor_time = normalized, done, state.tails['cube']
            power *= 2
            distance = 0
    if not period:
        prologue_panel, prologue_time = done, state.tails['cube']
    for panel in tail:
        state.advance(panel, emit)
        stepped += 1
    return MatrixCycle(state.tails['cube'] / denominator, prologue_time / denominator,
                       prologue_panel, period_time / denominator, period, stepped, jumped)
