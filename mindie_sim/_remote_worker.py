"""Bounded CPU worker for simulation batches.

[Role] Decode a batch, load verified CPU inputs and invoke PredictionSpine.
[Invariants] The same JSON kernel serves local and SSH execution. Only the
  outer dispatcher can select a remote endpoint. All results have stable order.
[Timeouts & Locks] Workers share a monotonic deadline. A lease protects the
  staged bundle through initialization, execution and process-pool shutdown.
[Side-Effect Policy] No device or network access. Only the native cache and
  staged lease directory are writable.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, replace
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

from mindie_sim.remote_exec import (
    AUTO_REMOTE_BATCH_THRESHOLD, ENTRYPOINTS, JOB_TIMEOUT, MANIFEST,
    PRIVATE_DIR, PROTOCOL_VERSION, READY, REQUEST_LIMIT, RESULT_LIMIT,
    UUID_TOKEN, WORKER_MEMORY_MIB, RemoteDispatcher, RemoteExecPolicy,
    RemoteExecReceipt, RemoteExecUnavailableError, _active_leases,
    _boot, _bounded_read, _exchange, _pin,
    _canonical, _create_lease, _digest_lock, _lock, _manifest, _native_root,
    _private_dir, _remaining, _safe_path, size_remote_workers, verify_staged_manifest,
)

WINDOWS_POOL_LIMIT = 61  # Windows WaitForMultipleObjects capacity in concurrent.futures.
MIN_SHM_BYTES = 256 * 1024 * 1024
MIN_PYTHON = (3, 11)
DEFAULT_PRIMITIVE_MANIFEST = "data/primitives/manifest.json"
ENV_LOCK = threading.RLock()
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")
_REMOTE_MODE_ALIASES = {
    "off": "local",
    "local": "local",
    "auto": "auto",
    "prefer_remote": "prefer_remote",
    "require": "remote",
    "remote": "remote",
}

@dataclass(frozen=True, slots=True)
class RemoteBatchRequest:
    protocol_version: str
    request_id: str
    deadline_s: float
    entrypoint: str
    base_case: dict
    candidates: list[dict]
    stage_root: str | None = None
    max_workers: int | None = None
    per_worker_mem_mib: int = WORKER_MEMORY_MIB
    primitive_manifest_path: str | None = None

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION or not UUID_TOKEN.fullmatch(self.request_id):
            raise ValueError("invalid worker protocol identity")
        if (self.entrypoint not in ENTRYPOINTS or isinstance(self.deadline_s, bool)
                or not math.isfinite(self.deadline_s) or self.deadline_s <= 0):
            raise ValueError("invalid worker entrypoint or deadline")
        if (not isinstance(self.base_case, dict) or not isinstance(self.candidates, list)
                or not self.candidates or any(not isinstance(row, dict) for row in self.candidates)):
            raise ValueError("invalid worker batch")
        for value in (self.per_worker_mem_mib, self.max_workers if self.max_workers is not None else 1):
            if type(value) is not int or value < 1:
                raise ValueError("invalid worker capacity")

def _linux_capacity():
    """Account for process cgroups and finite ancestor limits, including nested mounts."""
    def read(path, default=""):
        try:
            return Path(path).read_text().strip()
        except (FileNotFoundError, PermissionError):
            return default

    groups = {}
    for line in read("/proc/self/cgroup").splitlines():
        _, controllers, location = line.split(":", 2)
        groups.update((controller, location) for controller in controllers.split(","))
    cpu, memory = [], []
    for line in read("/proc/self/mountinfo").splitlines():
        left, right = line.split(" - ", 1)
        kind, _, options = right.split()
        if kind not in {"cgroup", "cgroup2"}:
            continue
        _, _, _, mount_root, mountpoint, *_ = left.split()
        for controller in ("cpu", "memory"):
            location = groups.get("" if kind == "cgroup2" else controller)
            if location is None or (kind == "cgroup" and controller not in options.split(",")):
                continue
            mount = Path(mountpoint)
            relative = Path(location)
            if relative.is_relative_to(mount_root):
                relative = relative.relative_to(mount_root)
            directory = mount / relative.as_posix().lstrip("/")
            if ".." in relative.parts or not directory.is_dir():
                directory = mount  # The container may see just its cgroup as the mount root.
            for group in (directory, *directory.parents):
                if not group.is_relative_to(mount):
                    break
                if controller == "cpu":
                    if kind == "cgroup2":
                        quota, period = read(group / "cpu.max", "max 1").split()
                    else:
                        quota = read(group / "cpu.cfs_quota_us", "-1")
                        period = read(group / "cpu.cfs_period_us", "1")
                    if quota != "max" and int(quota) > 0:
                        cpu.append(int(quota) / int(period))
                else:
                    limit_file, usage_file = (("memory.max", "memory.current") if kind == "cgroup2"
                                              else ("memory.limit_in_bytes", "memory.usage_in_bytes"))
                    quota = read(group / limit_file, "max")
                    if quota != "max":
                        memory.append(max(0, int(quota) - int(read(group / usage_file, "0"))) / (1024 * 1024))
    available = next(int(line.split()[1]) / 1024 for line in read("/proc/meminfo").splitlines()
                     if line.startswith("MemAvailable:"))
    return dict(affinity_cpus=len(os.sched_getaffinity(0)), cgroup_cpus=min(cpu, default=None),
                loadavg_1m=os.getloadavg()[0], mem_avail_mib=available,
                cgroup_mem_remaining_mib=min(memory, default=None))

_PROBE = """
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
assert sys.platform == 'linux' and sys.version_info >= MIN_PYTHON
try:
    import ijson  # Optional for streaming observation tools; spine uses stdlib JSON.
except ImportError:
    ijson = None
compiler = next(path for name in ('cc','clang','gcc') if (path := shutil.which(name)))
def cc(*args, source=None):
    return subprocess.run([compiler,*args], input=source, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, check=True).stdout
cc('-std=c11','-x','c','-fsyntax-only','-',source=b'_Static_assert(__STDC_VERSION__ >= 201112L,"C11");')
shm_ok = False
try:
    shm = os.statvfs('/dev/shm')
    shm_ok = (not (shm.f_flag & getattr(os, 'ST_NOEXEC', 0))) and shm.f_bavail * shm.f_frsize >= MIN_SHM_BYTES
except OSError:
    shm_ok = False
if not shm_ok:
    fallback = Path.home() / '.cache'
    fallback.mkdir(parents=True, exist_ok=True)
    fb_st = os.statvfs(fallback)
    assert not (fb_st.f_flag & getattr(os, 'ST_NOEXEC', 0)) and fb_st.f_bavail * fb_st.f_frsize >= MIN_SHM_BYTES
print(json.dumps(dict(_linux_capacity(), compiler=hashlib.sha256(
    os.fsencode(os.path.realpath(compiler)) + cc('--version')).hexdigest())))
"""
def _probe_remote(endpoint, policy):
    deadline = time.monotonic() + policy.probe_timeout_s
    pinned = _pin(endpoint, deadline)
    source = (f"MIN_PYTHON={MIN_PYTHON!r}\nMIN_SHM_BYTES={MIN_SHM_BYTES}\n"
              + inspect.getsource(_linux_capacity) + _PROBE)
    return pinned, json.loads(_exchange(pinned, _boot(source), b"", _remaining(deadline)))
def _local_ready():
    from mindie_sim.prediction.runtime.accel_loader import toolchain_or_cache_ready
    return toolchain_or_cache_ready()

@contextmanager
def _worker_environment(stage):
    settings = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if stage:
        settings.update(MINDIE_CACHE_HOME=str(_native_root(stage)), MINDIE_SIM_ACCEL_SO=None)
    with ENV_LOCK:
        previous = {key: os.environ.pop(key, None) for key in settings}
        os.environ.update({key: value for key, value in settings.items() if value is not None})
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
def _prepare_native(stage_root, *, build=False):
    from mindie_sim.prediction.runtime import accel_build, accel_loader

    stage = Path(stage_root)
    root = _native_root(stage)
    with _worker_environment(stage):
        seal = root / MANIFEST
        if seal.exists():
            try:
                _private_dir(root)
                verify_staged_manifest(root)
                status = accel_build.build_accel(check_only=True)
                if status.get("library_ready") and Path(status["cache_path"]).parent == accel_build.cache_root():
                    accel_loader.load_accel_cdll(status)
                    return
            except TimeoutError:
                raise
            except (OSError, RuntimeError, ValueError):
                if not build:
                    raise ValueError("validated native library unavailable") from None
            if not build:
                raise ValueError("validated native library unavailable")
        if not build:
            raise RuntimeError("staged native library unavailable")
        tmp_root = Path(tempfile.mkdtemp(dir=stage.parent, prefix=f".native.{stage.name}.build-"))
        old_root: Path | None = None
        try:
            if os.name == "posix":
                os.chmod(tmp_root, PRIVATE_DIR)
            os.environ["MINDIE_CACHE_HOME"] = str(tmp_root)
            status = accel_build.build_accel()
            accel_loader.load_accel_cdll(status)
            tmp_res = tmp_root.resolve()
            for sidecar in tmp_root.rglob("*.json"):
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and isinstance(meta.get("library"), str):
                    lib = Path(meta["library"]).resolve()
                    if lib.is_relative_to(tmp_res):
                        meta["library"] = str(root / lib.relative_to(tmp_res))
                        sidecar.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
            (tmp_root / MANIFEST).write_bytes(_manifest(tmp_root, sorted(p for p in tmp_root.rglob("*") if p.is_file())))
            if root.exists():
                old_root = Path(tempfile.mkdtemp(dir=stage.parent, prefix=f".native.{stage.name}.old-"))
                root.rename(old_root / "prev")
            tmp_root.rename(root)
            tmp_root = None
            os.environ["MINDIE_CACHE_HOME"] = str(root)
            if old_root is not None:
                shutil.rmtree(old_root, ignore_errors=True)
                old_root = None
        except BaseException:
            if old_root is not None and not root.exists() and (old_root / "prev").exists():
                (old_root / "prev").rename(root)
            if old_root is not None:
                shutil.rmtree(old_root, ignore_errors=True)
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)
            os.environ["MINDIE_CACHE_HOME"] = str(root)
            raise

@contextmanager
def _job_lease(stage, request_id):
    with _lock(_digest_lock(stage)):
        _private_dir(stage)
        if verify_staged_manifest(stage) != stage.name:
            raise ValueError("bundle digest mismatch")
        lease = _create_lease(stage, request_id)
        _active_leases(stage, request_id=request_id)
    try:
        yield
    finally:
        with _lock(_digest_lock(stage)):
            lease.unlink(missing_ok=True)
def _init_worker(stage_root, primitive_manifest_path):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if stage_root:
        stage = Path(stage_root)
        os.environ["MINDIE_CACHE_HOME"] = str(_native_root(stage))
        os.environ.pop("MINDIE_SIM_ACCEL_SO", None)
        _prepare_native(stage)
    from mindie_sim.primitives.loader import load_manifest, load_primitives

    rel = primitive_manifest_path or DEFAULT_PRIMITIVE_MANIFEST
    path = Path(stage_root or Path(__file__).parents[1]) / _safe_path(rel)
    if path.name != "manifest.json":
        raise ValueError("invalid primitive manifest")
    manifest = load_manifest(path.parent)
    for sku in manifest["tables"]:
        load_primitives(sku, primitives_dir=path.parent)
def _thaw(value):
    return tuple(_thaw(item) for item in value) if isinstance(value, list) else value


def _step_request(raw):
    from mindie_sim.prediction.results import StepRequest

    if isinstance(raw.get("sequences"), list):
        raw = raw | {"sequences": tuple(raw["sequences"])}
    return StepRequest(**raw)


def _lowering(raw):
    from mindie_sim.schema.operators import OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from mindie_sim.stack_registry import LoweringRequest, RankOps

    fields = dict(raw)
    if fields.get("step") is not None:
        fields["step"] = _step_request(fields["step"])
    fields["iterations"] = tuple((rank, IterationPlan(**{
        **plan, "requests": tuple(RequestView(**view) for view in plan["requests"]),
    })) for rank, plan in fields.get("iterations", ()))
    fields["declared_ops"] = tuple(
        RankOps(row["replica"], tuple(OpSpec(**{k: _thaw(v) for k, v in op.items()}) for op in row["ops"]))
        for row in fields.get("declared_ops", ())
    )
    return LoweringRequest(**fields)
def _resolve_batch_callable(module_name: str, qualname: str) -> Callable[[Any], Any]:
    """Resolve a module-level callable strictly inside ``mindie_sim.*``."""
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise ValueError("invalid_batch_callable")
    if not module_name.startswith("mindie_sim.") or ".." in module_name:
        raise ValueError("unauthorized_batch_module")
    if not all(_IDENT_RE.fullmatch(part) for part in module_name.split(".")):
        raise ValueError("invalid_batch_module_name")
    if not _IDENT_RE.fullmatch(qualname) or qualname.startswith("__"):
        raise ValueError("invalid_batch_function_name")
    mod = importlib.import_module(module_name)
    target = getattr(mod, qualname, None)
    if not callable(target):
        raise ValueError("batch_target_not_callable")
    return target


def _decode_primitive_table(table):
    if table.get("kind") == "primitive_table":
        from mindie_sim.primitives.schema import parse_primitive_table

        parsed = parse_primitive_table(table["envelope"])
        # The loader attaches warnings outside the digested envelope.
        object.__setattr__(parsed, "warnings", tuple(table.get("warnings", ())))
        return parsed
    from mindie_sim.optimization.contracts import PrimitiveCapacityEntry, PrimitiveCapacityTable

    return PrimitiveCapacityTable(
        sku=table["sku"],
        entries=tuple(PrimitiveCapacityEntry(**entry) for entry in table["entries"]),
    )


def _candidate(entrypoint, base, candidate, deadline):
    from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO, check_active_deadline
    from mindie_sim.prediction.results import ReplayOptions, SlaRequest, SweepRequest, json_payload
    from mindie_sim.prediction.spine import default_spine
    from mindie_sim.schema.case import Case

    _remaining(deadline)
    if entrypoint == "batch_call":
        fn = _resolve_batch_callable(candidate["module"], candidate["qualname"])
        result = fn(candidate["payload"])
        return {"candidate_id": candidate["candidate_id"], "objective_key": candidate["objective_key"],
                "result": json_payload(result)}
    token = ACTIVE_DEADLINE_MONO.set(deadline)
    try:
        case = Case.from_mapping(candidate.get("case", base))
        raw = dict(candidate["request"])
        spine = default_spine()
        replay = ReplayOptions(**(candidate.get("replay", {}) | {"deadline_mono": deadline}))
        if entrypoint == "run_step":
            result = spine.run_step(spine.lower(case, _lowering(raw)), replay)
        elif entrypoint == "evaluate_three_tiers":
            result = spine.evaluate_three_tiers(
                case, _step_request(raw), replay=replay, **candidate.get("options", {}),
            )
        elif entrypoint == "solve_sla":
            result = spine.solve_sla(case, SlaRequest(**(raw | {"step": _step_request(raw["step"])})))
        elif entrypoint == "sweep_topology":
            sla = raw["sla"]
            request = SweepRequest(**(raw | {"sla": SlaRequest(**(sla | {"step": _step_request(sla["step"])}))}))
            decode = candidate.get("decode_case")
            result = spine.sweep_topology(case, request, decode_case=Case.from_mapping(decode) if decode is not None else None)
        else:
            options = dict(candidate.get("options", {}))
            if (table := options.get("primitive_table")) is not None:
                options["primitive_table"] = _decode_primitive_table(table)
            result = spine.optimize_ladder(case, _step_request(raw), replay=replay, **options)
        check_active_deadline()
        return {"candidate_id": candidate["candidate_id"], "objective_key": candidate["objective_key"],
                "result": json_payload(result)}
    finally:
        ACTIVE_DEADLINE_MONO.reset(token)

def _parallel(workers, request, candidates, deadline):
    methods = mp.get_all_start_methods()
    method = "forkserver" if "forkserver" in methods and sys.platform == "linux" else "spawn"
    ctx = mp.get_context(method)
    if method == "forkserver":
        ctx.set_forkserver_preload(["json", "math"])
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                              initargs=(request.stage_root, request.primitive_manifest_path))
    submit = lambda row: pool.submit(_candidate, request.entrypoint, request.base_case, row, deadline)
    rows = iter(candidates)
    try:
        pending = {submit(next(rows)) for _ in range(workers)}
        while pending:
            done, pending = wait(pending, timeout=_remaining(deadline), return_when=FIRST_COMPLETED)
            if not done:
                raise TimeoutError("worker batch deadline expired")
            for future in done:
                yield future.result()
                row = next(rows, None)
                if row is not None:
                    pending.add(submit(row))
    except BaseException:
        for process in (pool._processes or {}).values():  # Python 3.11 has no public hard-stop API.
            with suppress(ProcessLookupError):
                process.kill()
        raise
    finally:
        pool.shutdown(cancel_futures=True)

def execute_worker_request(request_payload: Mapping | bytes, *, max_result_bytes=RESULT_LIMIT) -> dict:
    """Run a batch and return a size-bounded JSON envelope."""
    raw = request_payload if isinstance(request_payload, bytes) else _canonical(request_payload)
    if len(raw) > REQUEST_LIMIT:
        raise ValueError("request byte limit exceeded")
    request = RemoteBatchRequest(**json.loads(raw))
    deadline = time.monotonic() + request.deadline_s
    stage = Path(request.stage_root) if request.stage_root else None
    cpus = os.cpu_count() or 1
    capacity = _linux_capacity() if sys.platform == "linux" else dict(
        affinity_cpus=cpus, cgroup_cpus=None, loadavg_1m=0, mem_avail_mib=WORKER_MEMORY_MIB * cpus,
        cgroup_mem_remaining_mib=None)
    cap, _ = size_remote_workers(**(capacity | {"loadavg_1m": 0}), per_worker_mem_mib=request.per_worker_mem_mib,
                                 max_remote_workers=request.max_workers)
    workers = min(max(1, cap), len(request.candidates))
    if os.name == "nt":
        workers = min(workers, WINDOWS_POOL_LIMIT)
    candidates = [{"candidate_id": str(index), "objective_key": 0, **row}
                  for index, row in enumerate(request.candidates)]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate candidate ID")
    envelope = dict(protocol_version=PROTOCOL_VERSION, request_id=request.request_id, effective_workers=workers,
                    results=[])
    size = len(_canonical(envelope))
    limit = min(max_result_bytes, RESULT_LIMIT)
    with _worker_environment(stage), (_job_lease(stage, request.request_id) if stage else nullcontext()):
        if workers == 1:
            _init_worker(request.stage_root, request.primitive_manifest_path)
            outputs = (_candidate(request.entrypoint, request.base_case, row, deadline) for row in candidates)
        else:
            outputs = _parallel(workers, request, candidates, deadline)
        try:
            for row in outputs:
                size += len(_canonical(row)) + bool(envelope["results"])
                if size > limit:
                    raise ValueError("result byte limit exceeded")
                envelope["results"].append(row)
        finally:
            outputs.close()
    envelope["results"].sort(key=lambda row: (row["objective_key"], row["candidate_id"]))
    return envelope

def _local_pool_initializer() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _run_local_callable_batch(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    max_workers: int | None,
    deadline_s: float,
) -> tuple[list[Any], int]:
    """Execute ``fn`` over ``items`` locally via bounded ProcessPoolExecutor."""
    n_items = len(items)
    if n_items == 0:
        return [], 1
    cpus = os.cpu_count() or 1
    workers = min(n_items, cpus if max_workers is None else max(1, int(max_workers)))
    if os.name == "nt":
        workers = min(workers, WINDOWS_POOL_LIMIT)
    deadline = time.monotonic() + deadline_s
    can_pool = workers > 1 and _callable_is_picklable(fn)
    if not can_pool:
        outputs = []
        for item in items:
            _remaining(deadline)
            outputs.append(fn(item))
        return outputs, 1

    methods = mp.get_all_start_methods()
    method = "forkserver" if "forkserver" in methods and sys.platform == "linux" else "spawn"
    ctx = mp.get_context(method)
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_local_pool_initializer,
    )
    ordered: list[Any] = [None] * n_items
    try:
        future_to_idx = {
            pool.submit(fn, item): idx for idx, item in enumerate(items)
        }
        pending = set(future_to_idx)
        while pending:
            done, pending = wait(
                pending, timeout=_remaining(deadline), return_when=FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("local_batch_deadline_expired")
            for fut in done:
                ordered[future_to_idx[fut]] = fut.result()
        return ordered, workers
    except BaseException:
        for process in (pool._processes or {}).values():
            with suppress(ProcessLookupError):
                process.kill()
        raise
    finally:
        pool.shutdown(cancel_futures=True)


def _callable_is_picklable(fn: Callable[[Any], Any]) -> bool:
    try:
        pickle.dumps(fn)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


def _remote_callable_spec(fn: Callable[[Any], Any]) -> tuple[str, str] | None:
    module_name = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", "")
    if (isinstance(module_name, str) and module_name.startswith("mindie_sim.")
            and isinstance(qualname, str) and _IDENT_RE.fullmatch(qualname)
            and not qualname.startswith("__")):
        return module_name, qualname
    return None


def _normalize_remote_mode(remote: str) -> str:
    key = str(remote).strip().lower()
    if key not in _REMOTE_MODE_ALIASES:
        raise ValueError(f"invalid_remote_mode:{remote}")
    return _REMOTE_MODE_ALIASES[key]


def execute_batch_with_auto_remote(
    tasks_or_fn: Any,
    tasks: Sequence[Any] | None = None,
    *,
    remote: str = "auto",
    entrypoint: str | None = None,
    base_case: Any = None,
    preferred_alias: str | None = None,
    authorized_aliases: Sequence[str] = (),
    max_workers: int | None = None,
    deadline_s: float = JOB_TIMEOUT,
    min_remote_batch: int = AUTO_REMOTE_BATCH_THRESHOLD,
    policy: RemoteExecPolicy | None = None,
    dispatcher: RemoteDispatcher | None = None,
    local_executor: Callable[[Sequence[Any]], list[Any]] | None = None,
    cwd: Path | None = None,
) -> tuple[list[Any], RemoteExecReceipt]:
    """Execute a batch over SSH remote CPU workers when eligible, else locally."""
    if not math.isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError("invalid execution deadline")
    target = _normalize_remote_mode(remote)

    if entrypoint is not None:
        candidates = list(tasks_or_fn if tasks is None else tasks)
        if not candidates:
            return [], RemoteExecReceipt(resolved_target="local", effective_workers=1)
        eff_policy = (
            replace(policy, target=target)
            if policy is not None else RemoteExecPolicy(
                target=target, preferred_alias=preferred_alias,
                authorized_aliases=tuple(authorized_aliases),
                max_remote_workers=max_workers, job_timeout_s=deadline_s,
            )
        )
        disp = dispatcher or RemoteDispatcher(eff_policy, cwd=cwd)
        if target == "auto" and len(candidates) < min_remote_batch and disp._local_ready():
            local_disp = RemoteDispatcher(
                replace(disp.policy, target="local"), local_c11_ready_fn=disp._local_ready,
            )
            env, _ = local_disp.dispatch_batch(
                entrypoint, base_case or {}, candidates, deadline_s=deadline_s,
            )
            return list(env["results"]), RemoteExecReceipt(
                resolved_target="local", effective_workers=env.get("effective_workers", 1),
                accel_mode="c11", fallback_reason="warm_local_c11",
            )
        env, receipt = disp.dispatch_batch(
            entrypoint, base_case or {}, candidates, deadline_s=deadline_s,
        )
        return list(env["results"]), receipt

    if not callable(tasks_or_fn):
        raise TypeError("execute_batch_with_auto_remote requires callable or entrypoint")
    fn: Callable[[Any], Any] = tasks_or_fn
    items = list(() if tasks is None else tasks)
    if not items:
        return [], RemoteExecReceipt(resolved_target="local", effective_workers=1)

    eff_policy = (
        replace(policy, target=target)
        if policy is not None else RemoteExecPolicy(
            target=target, preferred_alias=preferred_alias,
            authorized_aliases=tuple(authorized_aliases),
            max_remote_workers=max_workers, job_timeout_s=deadline_s,
        )
    )
    disp = dispatcher or RemoteDispatcher(eff_policy, cwd=cwd)
    local_c11 = False
    try:
        local_c11 = bool(disp._local_ready())
    except TimeoutError:
        raise
    except (OSError, RuntimeError):
        local_c11 = False

    fallback_reason: str | None = None
    try_remote = (
        target in {"remote", "prefer_remote"}
        or (target == "auto" and (not local_c11 or len(items) >= min_remote_batch))
    )
    if target == "auto" and not try_remote:
        fallback_reason = "warm_local_c11"

    if try_remote:
        spec = _remote_callable_spec(fn)
        if spec is None:
            if target == "remote":
                raise RemoteExecUnavailableError("endpoint_configuration")
            fallback_reason = "endpoint_configuration"
        else:
            mod_name, qual_name = spec
            receipt_probe, endpoint = disp._resolve(len(items))
            if endpoint is not None:
                cands = [
                    {"candidate_id": str(i), "objective_key": i,
                     "module": mod_name, "qualname": qual_name, "payload": item}
                    for i, item in enumerate(items)
                ]
                env, receipt = disp.dispatch_batch(
                    "batch_call", {}, cands, deadline_s=deadline_s,
                )
                if receipt.resolved_target == "remote":
                    ordered = [row["result"] for row in env["results"]]
                    return ordered, receipt
                fallback_reason = receipt.fallback_reason
            else:
                if target == "remote":
                    raise RemoteExecUnavailableError(
                        receipt_probe.fallback_reason or "no_authorized_endpoint"
                    )
                fallback_reason = receipt_probe.fallback_reason

    if local_executor is not None:
        return list(local_executor(items)), RemoteExecReceipt(
            resolved_target="local", effective_workers=1,
            accel_mode="c11" if local_c11 else "python",
            fallback_reason=fallback_reason,
        )
    results, workers = _run_local_callable_batch(
        fn, items, max_workers=max_workers, deadline_s=deadline_s,
    )
    return results, RemoteExecReceipt(
        resolved_target="local", effective_workers=workers,
        accel_mode="c11" if local_c11 else "python",
        fallback_reason=fallback_reason,
    )


def main(*, ready=False, expected_root=None, max_result_bytes=RESULT_LIMIT):
    request_id = None
    try:
        if ready:
            sys.stdout.buffer.write(READY)
            sys.stdout.buffer.flush()
        raw = _bounded_read(sys.stdin.buffer, REQUEST_LIMIT).removesuffix(b"\n")
        payload = json.loads(raw)
        request_id = payload.get("request_id")
        if _canonical(payload) != raw:
            raise ValueError("worker request must use canonical JSON")
        if expected_root is not None and payload.get("stage_root") != expected_root:
            raise ValueError("staged worker root mismatch")
        result = execute_worker_request(payload, max_result_bytes=max_result_bytes)
    except Exception as error:
        if not isinstance(request_id, str) or not UUID_TOKEN.fullmatch(request_id):
            return 1
        result = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id,
                      error="deadline_exceeded" if isinstance(error, TimeoutError) else "worker_failed")
    encoded = _canonical(result)
    if len(encoded) > max_result_bytes:
        return 1
    sys.stdout.buffer.write(encoded)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
