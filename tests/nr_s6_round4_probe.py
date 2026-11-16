"""Real two-thread rendezvous broker for Round-4 probe ordering tests.

[Role]
  Execute two emitted probe ranks concurrently against one rendezvous broker
  that pairs every collective by group, kind, dtype, element count and
  reduction operation.  It does not pre-arrange peer decisions.

[Invariants]
  * A data collective and a MIN decision cannot both be accepted at one
    rendezvous position; a mismatch raises on both threads.
  * Rank-local synchronization/allocation faults and fake-clock advances are
    injected at exact call boundaries.
  * Each rank has an independent clock and output path; the broker owns the
    only shared state.

[Timeouts & Locks]
  One condition variable with a bounded rendezvous timeout; threads join with
  a bounded timeout, a timed-out pair is signalled through the abort file and
  joined briefly, and no device or process group is created.

[Side-Effect Policy]
  Owns one tracked process-temporary workspace for JSONL/abort records and
  removes it whenever both workers are quiescent; a still-live worker defers
  deletion to ``temp_cleanup`` at session/atexit instead of racing its handles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import types

from mindie_profiler.fabric_probe import emit_fabric_probe_script

from temp_cleanup import tempdir


class ProtocolMismatch(RuntimeError):
    pass


class PeerAbortedError(RuntimeError):
    pass


class FakeTensor:
    def __init__(self, values=(), shape=None, dtype=None, device=None, item=1):
        self.values = list(values)
        self.shape = tuple(shape if shape is not None else (len(self.values),))
        self.dtype = dtype
        self.device = device
        self._item = item

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def item(self):
        return self._item

    def all(self):
        return self

    def reshape(self, shape):
        return FakeTensor(self.values, shape=shape, dtype=self.dtype,
                          device=self.device, item=self._item)

    def zero_(self):
        return self

    def fill_(self, *_args):
        return self

    def copy_(self, _other):
        return self


class Clock:
    def __init__(self, now=1000.0):
        self.now = float(now)

    def monotonic(self):
        return self.now

    def perf_counter_ns(self):
        return int(self.now * 1e9)

    def advance(self, seconds):
        self.now += float(seconds)


class Broker:
    """Pair collectives from two ranks and reject protocol divergence."""

    def __init__(self, timeout=3.0):
        self.condition = threading.Condition()
        self.timeout = float(timeout)
        self.pending = []
        self.sequence = {0: [], 1: []}
        self.data_counts = {0: 0, 1: 0}
        self.decision_logs = {0: [], 1: []}
        self.mismatches = []
        self.broken = False
        self.aborted = False
        self.abort_record = None
        self.start = threading.Barrier(2)
        self.world_group = None
        self.groups = {}
        self.new_group_counts = {0: 0, 1: 0}
        self.last_send_recv_group = None

    def group_for(self, ranks, rank):
        """Return the same communicator object to both ranks of a new group.

        WORLD is deliberately a distinct object even when its rank set equals a
        tier group: aborting one communicator must not silently abort the other.
        """
        with self.condition:
            count = self.new_group_counts.get(rank, 0)
            self.new_group_counts[rank] = count + 1
            key = (tuple(sorted(int(item) for item in ranks)), count)
            group = self.groups.get(key)
            if group is None:
                label = ",".join(str(item) for item in key[0])
                group = BrokerGroup(self, ranks, name=f"tier:{label}:{count + 1}")
                self.groups[key] = group
            return group

    def send_recv(self, rank, kind, group=None, timeout=0.2):
        """Block the caller on ITS communicator until that group is aborted."""
        target = group if group is not None else self.world_group
        with self.condition:
            self.last_send_recv_group = target
        deadline = time.monotonic() + float(timeout)
        while True:
            if target is not None and target.abort_event.is_set():
                raise PeerAbortedError(
                    f"{kind} aborted on {getattr(target, 'name', 'WORLD')}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{kind} timed out on {getattr(target, 'name', 'WORLD')}")
            if target is not None:
                target.abort_event.wait(min(0.005, remaining))
            else:
                time.sleep(min(0.005, remaining))

    def signal_abort(self, rank, step, error):
        with self.condition:
            self.aborted = True
            self.abort_record = {"rank": int(rank), "step": str(step),
                                 "error": str(error)}
            self.condition.notify_all()

    def abort(self):
        """Standard ProcessGroup.abort(): wake every waiter, no MIN pairing."""
        with self.condition:
            self.aborted = True
            if self.abort_record is None:
                self.abort_record = {"kind": "abort",
                                     "source": "process_group_abort"}
            self.condition.notify_all()

    @staticmethod
    def _descriptor(kind, tensor, group, op):
        dtype = str(getattr(tensor, "dtype", "")).lower()
        operation = str(op or "sum").lower()
        return (kind, tuple(group or ()), dtype, int(tensor.numel()), operation)

    def collective(self, rank, kind, tensor, group=None, op=None):
        entry = {"rank": rank, "kind": kind,
                 "descriptor": self._descriptor(kind, tensor, group, op),
                 "tensor": tensor, "done": False, "error": None}
        with self.condition:
            if self.broken:
                raise ProtocolMismatch("collective issued on a broken broker")
            if self.aborted and kind != "decision":
                raise PeerAbortedError("peer aborted before collective")
            self.pending.append(entry)
            import time
            deadline = time.monotonic() + self.timeout
            while not entry["done"]:
                if len(self.pending) >= 2 and not self.broken:
                    first, second = self.pending
                    self.pending = []
                    if first["descriptor"] != second["descriptor"]:
                        self.broken = True
                        self.mismatches.append({
                            "reason": "protocol_mismatch",
                            "descriptors": [first["descriptor"],
                                            second["descriptor"]]})
                        for item in (first, second):
                            item["error"] = ProtocolMismatch(
                                "protocol mismatch "
                                f"{first['descriptor']} != {second['descriptor']}")
                            item["done"] = True
                    else:
                        desc = first["descriptor"]
                        self.sequence[first["rank"]].append(desc)
                        self.sequence[second["rank"]].append(desc)
                        if desc[0] == "decision":
                            values = [int(first["tensor"]._item),
                                      int(second["tensor"]._item)]
                            combined = min(values)
                            for item in (first, second):
                                item["tensor"]._item = combined
                                self.decision_logs[item["rank"]].append(combined)
                                item["done"] = True
                        else:
                            for item in (first, second):
                                self.data_counts[item["rank"]] += 1
                                item["done"] = True
                    self.condition.notify_all()
                    continue
                if self.aborted and kind != "decision" and not entry["done"]:
                    entry["error"] = PeerAbortedError(
                        "peer aborted during rendezvous")
                    entry["done"] = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.broken = True
                    self.mismatches.append({"reason": "rendezvous_timeout",
                                            "rank": rank, "kind": kind})
                    entry["error"] = ProtocolMismatch("rendezvous timeout")
                    entry["done"] = True
                    self.condition.notify_all()
                    break
                self.condition.wait(remaining)
            if entry["error"] is not None:
                raise entry["error"]


class World:
    def __init__(self, plan):
        self.plan = plan
        self.broker = Broker()
        self.broker.world_group = BrokerGroup(self.broker, [0, 1], name="WORLD")
        self.clocks = {0: Clock(), 1: Clock()}
        self.outputs = {}
        self.errors = {}
        self.sync_calls = {0: 0, 1: 0}
        self.sync_actions = {0: {}, 1: {}}
        self.data_actions = {0: {}, 1: {}}
        self.data_calls = {0: 0, 1: 0}
        self.decision_advance = 0.0
        self.decision_delay = None
        self.clock_advance = 0.0
        self.fail_alloc = {0: False, 1: False}
        self.fail_destroy = {0: False, 1: False}
        self.destroy_delay = {0: 0.0, 1: 0.0}
        self.names = {"rank-0": 0, "rank-1": 1}
        self.barrier = threading.Barrier(2)
        self.abort_path = None
        self.abort_record = None

    def rank(self):
        name = threading.current_thread().name
        return self.names.get(name, 0)

    def dist(self):
        return BrokerDist(self, self.rank())


class BrokerGroup:
    """Standard group object with a private abort event per communicator."""

    def __init__(self, broker, ranks, name=None):
        self.broker = broker
        self.ranks = tuple(sorted(int(item) for item in ranks))
        self.name = name or "group:" + ",".join(str(item) for item in self.ranks)
        self.abort_event = threading.Event()
        self.aborted = False

    def abort(self):
        self.aborted = True
        self.abort_event.set()
        # Legacy global flag for failed-run assertions; blocking send/recv reads
        # only its own group event, so a WORLD abort cannot wake a tier recv.
        self.broker.aborted = True
        with self.broker.condition:
            self.broker.condition.notify_all()

    def __iter__(self):
        return iter(self.ranks)


class BrokerDist:
    class ReduceOp:
        MIN = "min"

    class _GroupNamespace:
        def __init__(self, world):
            self.world = world

        @property
        def WORLD(self):
            return self.world.broker.world_group

    def __init__(self, world, rank):
        self.world = world
        self.rank = rank
        self.group = self._GroupNamespace(world)

    def init_process_group(self, **_kwargs):
        try:
            self.world.barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            raise RuntimeError("rank start barrier broke")

    def new_group(self, ranks):
        return self.world.broker.group_for(ranks, self.rank)

    def destroy_process_group(self):
        delay = self.world.destroy_delay.get(self.rank, 0.0)
        if delay:
            self.world.clocks[self.rank].advance(delay)
        if self.world.fail_destroy.get(self.rank):
            raise RuntimeError("injected destroy failure")
        return None

    def all_reduce(self, tensor, group=None, op=None):
        decision = (str(getattr(tensor, "dtype", "")).lower() == "int32"
                    and tensor.numel() == 1 and str(op or "").lower() == "min")
        if not decision:
            self.world.data_calls[self.rank] += 1
            action = self.world.data_actions[self.rank].get(
                self.world.data_calls[self.rank])
            if action == "raise":
                raise RuntimeError(
                    f"injected data collective failure rank {self.rank}")
        self.world.broker.collective(
            self.rank, "decision" if decision else "data", tensor, group, op)
        if decision:
            self.world.clocks[self.rank].advance(self.world.decision_advance)
            if (self.world.decision_delay
                    and self.world.decision_delay[0] == self.rank
                    and len(self.world.broker.decision_logs[self.rank])
                    == self.world.decision_delay[1]):
                self.world.clocks[self.rank].advance(
                    self.world.decision_delay[2])

    def all_gather_into_tensor(self, *_args, **_kwargs):
        raise ProtocolMismatch("only all_reduce is exercised")

    reduce_scatter_tensor = all_gather_into_tensor
    all_to_all_single = all_gather_into_tensor

    def send(self, _tensor, dst=None, tag=None, group=None, **_kwargs):
        return self.world.broker.send_recv(self.rank, "send", group)

    def recv(self, _tensor, src=None, tag=None, group=None, **_kwargs):
        return self.world.broker.send_recv(self.rank, "recv", group)


class DistFacade:
    ReduceOp = BrokerDist.ReduceOp

    def __init__(self, world):
        self.world = world

    def _dist(self):
        return self.world.dist()

    def __getattr__(self, name):
        return getattr(self._dist(), name)


class NpuFacade:
    def __init__(self, world):
        self.world = world

    def synchronize(self):
        rank = self.world.rank()
        call = self.world.sync_calls[rank] + 1
        self.world.sync_calls[rank] = call
        action = self.world.sync_actions[rank].get(call)
        if action == "raise":
            raise RuntimeError(f"injected synchronize failure rank {rank} call {call}")
        if isinstance(action, tuple) and action[0] == "advance":
            self.world.clocks[rank].advance(action[1])

    def device_count(self):
        return 2

    def set_device(self, _rank):
        return None


def _fake_modules(world):
    torch = types.ModuleType("torch")
    torch.__path__ = []
    for name in ("int8", "int32", "int64", "uint8", "bfloat16", "float16",
                 "float32", "bool", "float8_e4m3fn", "float8_e5m2"):
        setattr(torch, name, name)

    def zeros(*args, **kwargs):
        rank = world.rank()
        if world.fail_alloc[rank]:
            raise RuntimeError("simulated allocation failure")
        shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) \
            else args
        return FakeTensor(shape=shape, dtype=kwargs.get("dtype"),
                          device=kwargs.get("device"))

    torch.zeros = zeros
    torch.tensor = lambda values, dtype=None, device=None: FakeTensor(
        values, shape=(len(values),), dtype=dtype, device=device, item=values[0])
    torch.empty_strided = lambda shape, strides, dtype=None, device=None: FakeTensor(
        shape=shape, dtype=dtype, device=device)
    torch.isfinite = lambda _tensor: FakeTensor(shape=(), item=True)
    torch.npu = NpuFacade(world)
    npu = types.ModuleType("torch_npu")
    npu.__path__ = []

    def _clock_rate():
        if world.clock_advance:
            world.clocks[world.rank()].advance(world.clock_advance)
        return 1.8e9

    npu.npu_get_clock_rate = _clock_rate
    distributed = types.ModuleType("torch.distributed")
    distributed.ReduceOp = BrokerDist.ReduceOp
    facade = DistFacade(world)
    for name in ("init_process_group", "new_group", "destroy_process_group",
                 "all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
                 "all_to_all_single", "send", "recv"):
        def _call(*args, _name=name, **kwargs):
            return getattr(facade, _name)(*args, **kwargs)
        setattr(distributed, name, _call)
    torch.distributed = distributed
    return torch, npu, distributed


def _rank_script(plan, rank):
    script = emit_fabric_probe_script(plan)["script"]
    script = script.replace('os.environ.get("RANK", "0")', str(rank))
    script = script.replace('os.environ.get("WORLD_SIZE", "1")', "2")
    script = script.replace(
        'os.environ.get("LOCAL_RANK", str(rank % max(1, torch.npu.device_count())))',
        "0")
    return script


def run_ranks(plan, *, sync_actions=None, fail_alloc=None, data_actions=None,
              fail_destroy=None, destroy_delay=None, decision_advance=0.0,
              decision_delay=None, clock_advance=0.0, timeout=10.0):
    """Run both ranks concurrently and return world, codes and output records."""
    world = World(plan)
    world.sync_actions.update(sync_actions or {})
    world.fail_alloc.update(fail_alloc or {})
    world.data_actions.update(data_actions or {})
    world.fail_destroy.update(fail_destroy or {})
    world.destroy_delay.update(destroy_delay or {})
    world.decision_advance = float(decision_advance)
    world.decision_delay = decision_delay
    world.clock_advance = float(clock_advance)
    torch, npu, distributed = _fake_modules(world)
    old = {name: sys.modules.get(name) for name in
           ("torch", "torch_npu", "torch.distributed")}
    sys.modules["torch"] = torch
    sys.modules["torch_npu"] = npu
    sys.modules["torch.distributed"] = distributed
    results = {}
    workspace = tempdir("nr4-probe-")
    tmp = Path(workspace)
    threads: list[threading.Thread] = []
    try:
        def runner(rank):
            namespace = {"__name__": f"round4_probe_{rank}"}
            script = _rank_script(plan, rank)
            exec(compile(script, f"<probe-{rank}>", "exec"), namespace)
            clock = world.clocks[rank]
            namespace["time"] = types.SimpleNamespace(
                monotonic=clock.monotonic,
                perf_counter_ns=clock.perf_counter_ns)
            output = os.path.join(tmp, f"rank{rank}.jsonl")
            abort_path = os.path.join(tmp, "abort.json")
            world.abort_path = abort_path
            try:
                code = namespace["main"](["--output", output,
                                          "--abort-file", abort_path])
                results[rank] = (code, [json.loads(line) for line in
                                        Path(f"{output}.rank{rank}.jsonl").read_text().splitlines()
                                        if line.strip()])
            except TimeoutError:
                raise
            except Exception as exc:  # broker mismatch / injected failure
                results[rank] = (None, [])
                world.errors[rank] = repr(exc)

        threads = [threading.Thread(target=runner, args=(rank,), name=f"rank-{rank}")
                   for rank in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout)
        timed_out = [thread for thread in threads if thread.is_alive()]
        if timed_out:
            for thread in timed_out:
                world.errors[thread.name] = "thread join timeout"
            # The emitted script watches this file and aborts the peer wait;
            # give both workers a bounded chance to leave the workspace before
            # any teardown can race their open handles.
            try:
                (tmp / "abort.json").write_text(json.dumps({
                    "kind": "status", "status": "deadline_exceeded",
                    "reason": "thread_join_timeout", "rank": 0}))
            except TimeoutError:
                raise
            except OSError:
                pass
            for thread in timed_out:
                thread.join(1.0)
        abort_file = tmp / "abort.json"
        if abort_file.is_file():
            try:
                world.abort_record = json.loads(abort_file.read_text())
            except TimeoutError:
                raise
            except (OSError, ValueError):
                world.abort_record = None
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if not any(thread.is_alive() for thread in threads):
            shutil.rmtree(workspace, ignore_errors=True)
        # Otherwise temp_cleanup retains ownership and retries at session
        # finish/atexit instead of deleting a workspace under a live thread.
    return world, results
