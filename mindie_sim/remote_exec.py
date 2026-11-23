"""Bounded remote multi-core CPU offload executor and worker runtime.

[Role] Stage read-only bundles over SSH to authorized Linux endpoints, run
  parallel worker batches with automatic local fallback, and serve as the
  `python3 -m mindie_sim.remote_exec` worker entrypoint.
[Invariants] Strict wall-clock budgets, explicit uncertain-submission errors,
  zero unauthenticated endpoints or private path leaks.
[Timeouts & Locks] Enforces outer >= inner + cleanup grace across SSH and pools.
[Side-Effect Policy] Writes only under bounded remote/local staging dirs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Literal
import uuid

if TYPE_CHECKING:
    from mindie_sim.prediction.results import CaseInput, ReplayOptions, SlaRequest, StepRequest, SweepRequest
    from mindie_sim.stack_registry import LoweringRequest
ExecutionTarget = Literal["auto", "prefer_remote", "remote", "local"]
PROTOCOL_VERSION = "sim-remote/1"
MIB = 1024 * 1024
REQUEST_LIMIT = 32 * MIB
RESULT_LIMIT = 64 * MIB
CACHE_QUOTA = 256 * MIB
BUNDLE_LIMIT = 128 * MIB
FILE_LIMIT = 48 * MIB
CONFIG_LIMIT = MIB
WORKER_MEMORY_MIB = 384
PROBE_TIMEOUT = 5.0
STAGE_TIMEOUT = 60.0
JOB_TIMEOUT = 300.0
JOB_GRACE = 30.0
PROBE_TTL = 60.0
REMOTE_CANDIDATE_THRESHOLD = 4
AUTO_REMOTE_BATCH_THRESHOLD = 8
LEASE_HANDOFF_TIMEOUT = STAGE_TIMEOUT + JOB_GRACE
PRIVATE_DIR = 0o700
PRIVATE_FILE = 0o600
UNSAFE_ENDPOINT_MODE = 0o022
ENDPOINT_ID_LENGTH = 16
MAX_PORT = 65535
ALIAS_LIMIT = 64
READY = b"sim-remote/1 ready\n"
MANIFEST = "manifest.sha256"
HEX_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
UUID_TOKEN = re.compile(r"[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
ALIAS_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
IPV4 = re.compile(r"(?<!\d)\d{1,3}(?:[._-]\d{1,3}){3}(?!\d)")
LEASE_TOKEN = re.compile(rf"([1-9][0-9]*)-({UUID_TOKEN.pattern})\.lease\Z")
BUNDLE_PATTERNS = (
    "mindie_sim/**/*.py", "mindie_sim/prediction/runtime/_engine_accel*.c",
    "mindie_sim/prediction/runtime/_engine_accel*.h", "mindie_sim/backends/profiles/*.json",
    "mindie_sim/stacks/*/manifest.json", "mindie_sim/model_configs/*.json",
    "mindie_sim/catalog/configs/*.json", "data/primitives/*.json",
)
PUBLIC_REASONS = frozenset({
    "endpoint_configuration", "no_authorized_endpoint", "probe_unavailable",
    "stage_unavailable", "transport_not_sent", "transport_lost", "warm_local_c11",
})
ENTRYPOINTS = frozenset({
    "run_step", "evaluate_three_tiers", "solve_sla", "sweep_topology",
    "optimize_ladder", "batch_call",
})

class RemoteExecUnavailableError(RuntimeError):
    submission_state = "not_sent"
RemoteExecutionUnavailableError = RemoteExecUnavailableError

class RemoteExecUncertainError(RuntimeError):
    submission_state = "uncertain"

    def __init__(self, request_id: str, alias: str | None, detail: str = "transport_lost"):
        self.request_id = request_id if UUID_TOKEN.fullmatch(request_id) else "unknown"
        self.alias = _sanitize_public_alias(alias)
        self.detail = detail if detail in PUBLIC_REASONS else "transport_lost"
        super().__init__(f"{self.detail} (request {self.request_id})")

@dataclass(frozen=True, slots=True)
class RemoteExecPolicy:
    target: ExecutionTarget = "auto"
    preferred_alias: str | None = None
    authorized_aliases: tuple[str, ...] = ()
    local_single_step_if_c11: bool = True
    max_remote_workers: int | None = None
    per_worker_mem_mib: int = WORKER_MEMORY_MIB
    probe_timeout_s: float = PROBE_TIMEOUT
    stage_timeout_s: float = STAGE_TIMEOUT
    job_timeout_s: float = JOB_TIMEOUT
    max_request_bytes: int = REQUEST_LIMIT
    max_result_bytes: int = RESULT_LIMIT
    cache_quota_bytes: int = CACHE_QUOTA
    probe_ttl_s: float = PROBE_TTL
    policy_path: Path | None = None

    def __post_init__(self):
        capacities = (self.per_worker_mem_mib, self.max_request_bytes, self.max_result_bytes,
                      self.cache_quota_bytes, 1 if self.max_remote_workers is None else self.max_remote_workers)
        timeouts = (self.probe_timeout_s, self.stage_timeout_s, self.job_timeout_s, self.probe_ttl_s)
        if (self.target not in {"auto", "prefer_remote", "remote", "local"}
                or any(type(v) is not int or v < 1 for v in capacities)
                or any(not math.isfinite(v) or v <= 0 for v in timeouts)):
            raise ValueError("invalid remote execution policy")
def _sanitize_public_alias(alias: str | None) -> str | None:
    if alias is None:
        return None
    if (isinstance(alias, str) and len(alias) <= ALIAS_LIMIT and ALIAS_TOKEN.fullmatch(alias) and not IPV4.search(alias)
            and alias.casefold() not in {"host", "user", "root", "auth_ref", "identity_file", "home"}):
        return alias
    return "redacted"
_LIMITS = ("affinity", "cgroup_cpu", "load_headroom", "memory", "policy")

@dataclass(frozen=True, slots=True)
class RemoteExecReceipt:
    alias: str | None = None
    endpoint_id: str | None = None
    resolved_target: Literal["local", "remote"] = "local"
    effective_workers: int = 1
    limiting_factor: str | None = None
    bundle_digest: str | None = None
    accel_mode: str = "auto"
    fallback_reason: str | None = None
    request_id: str | None = None
    submission_state: str = "not_sent"

    def __post_init__(self):
        object.__setattr__(self, "alias", _sanitize_public_alias(self.alias))
        for key, pattern in (("endpoint_id", rf"[a-f0-9]{{{ENDPOINT_ID_LENGTH}}}"),
                             ("bundle_digest", HEX_DIGEST.pattern), ("request_id", UUID_TOKEN.pattern)):
            value = getattr(self, key)
            if value is not None and not re.fullmatch(pattern, value):
                object.__setattr__(self, key, None)
        if (self.resolved_target not in {"local", "remote"} or type(self.effective_workers) is not int
                or self.effective_workers < 1 or self.limiting_factor not in (None, *_LIMITS, "candidates")
                or self.accel_mode not in {"auto", "c11", "python"}
                or self.fallback_reason not in (None, *PUBLIC_REASONS)
                or self.submission_state not in {"not_sent", "acknowledged", "uncertain"}):
            raise ValueError("invalid public execution receipt")
    def to_dict(self) -> dict:
        result = {key: getattr(self, key) for key in RemoteExecReceipt.__match_args__}
        assert not {"host", "user", "root", "auth_ref", "identity_file"}.intersection(result)
        assert not any(IPV4.search(v) or "$HOME" in v for v in result.values() if isinstance(v, str))
        return result
def size_remote_workers(affinity_cpus, cgroup_cpus, loadavg_1m, mem_avail_mib,
                        cgroup_mem_remaining_mib, per_worker_mem_mib, max_remote_workers,
                        force_saturated=False) -> tuple[int, str]:
    cpu_limit = affinity_cpus if cgroup_cpus is None else max(1, math.floor(cgroup_cpus))
    cpu_budget = min(affinity_cpus, cpu_limit)
    headroom = max(0, cpu_budget - math.ceil(max(0, loadavg_1m)))
    if force_saturated and not headroom:
        headroom = 1
    available = min(mem_avail_mib, cgroup_mem_remaining_mib) if cgroup_mem_remaining_mib is not None else mem_avail_mib
    caps = (affinity_cpus, cpu_limit, headroom, max(1, int(available // per_worker_mem_mib)),
            max_remote_workers if max_remote_workers is not None else cpu_budget)
    return min(zip(caps, _LIMITS), key=lambda item: item[0])
def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def _bounded_read(stream, limit: int) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("byte limit exceeded")
    return data

@dataclass(frozen=True, slots=True)
class _Endpoint:
    host: str
    port: int
    user: str = "root"
    root: str = "/"
    container: str | None = None
    identity_file: str | None = None
    known_hosts_file: str | None = None
    auth_ref: str | None = None

    @property
    def endpoint_id(self) -> str:
        key = f"{self.user}@{self.host}:{self.port}|root={self.root}"
        if self.container:
            key += f"|container={self.container}"
        return hashlib.sha256(key.encode()).hexdigest()[:ENDPOINT_ID_LENGTH]
def _endpoint(value) -> _Endpoint:
    keys = _Endpoint.__match_args__
    raw = value if isinstance(value, Mapping) else {key: getattr(value, key) for key in keys if hasattr(value, key)}
    defaults = {"user": os.getenv("REMOTE_DEV_DEFAULT_USER", "root"),
                "root": os.getenv("REMOTE_DEV_DEFAULT_ROOT", "/")}
    ep = _Endpoint(**(defaults | {key: raw[key] for key in keys if key in raw}))
    if (not re.fullmatch(r"[A-Za-z0-9_.:%-]+", ep.host) or not ALIAS_TOKEN.fullmatch(ep.user)
            or "|" in ep.root or type(ep.port) is not int or not 0 < ep.port <= MAX_PORT
            or (ep.container is not None and not ALIAS_TOKEN.fullmatch(ep.container))):
        raise ValueError("invalid endpoint")
    return ep
def _user_endpoint_files() -> tuple[Path, ...]:
    paths = [Path.home() / folder / "endpoints.json" for folder in (".mindie", ".mindie-coordinator")]
    cfg = os.environ.get("MINDIE_CONFIG_HOME")
    if cfg:
        candidate = Path(cfg).expanduser() / "endpoints.json"
        if candidate not in paths:
            paths.append(candidate)
    return tuple(paths)
def _read_config(path: Path, *, private=False, exact_mode: int | None = None):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | (getattr(os, "O_NOFOLLOW", 0) if private else 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return {}
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        bad_mode = (stat.S_IMODE(info.st_mode) != exact_mode if exact_mode is not None
                    else bool(info.st_mode & UNSAFE_ENDPOINT_MODE))
        if not stat.S_ISREG(info.st_mode) or (private and os.name == "posix" and
                (info.st_uid != os.getuid() or bad_mode)):
            raise ValueError("unsafe remote configuration")
        result = json.loads(_bounded_read(stream, CONFIG_LIMIT))
    if isinstance(result, dict):
        return result
    raise ValueError("invalid remote configuration")
def _merge_endpoints(paths, *, private=False):
    entries = {}
    for path in paths:
        data = _read_config(path, private=private)
        rows = data.get("endpoints", {k: v for k, v in data.items() if k != "default_alias"})
        entries.update(rows)
        default = data.get("default_alias")
        if isinstance(default, str) and default in entries and "default" not in rows:
            entries["default"] = entries[default]
    return entries
def _load_endpoints(cwd: Path):
    env = (Path(item.strip()).expanduser() for item in
           os.getenv("REMOTE_DEV_ENDPOINTS_FILE", "").split(os.pathsep) if item.strip())
    return _merge_endpoints((*_user_endpoint_files(), *env, cwd / "endpoints.json", cwd / "endpoints.local.json"))
def _authorized_pairs(policy: RemoteExecPolicy) -> set[tuple[str, str]]:
    cfg = os.environ.get("MINDIE_CONFIG_HOME")
    cfg_policy = (Path(cfg).expanduser() / "sim_remote_policy.json") if cfg else None
    default_path = (cfg_policy if (cfg_policy is not None and cfg_policy.exists())
                    else Path.home() / ".mindie/sim_remote_policy.json")
    path = policy.policy_path or default_path
    config = _read_config(Path(path).expanduser(), private=True, exact_mode=PRIVATE_FILE)
    rows = config.get("authorized_endpoints", config.get("authorized_aliases", ()))
    if isinstance(rows, dict):
        rows = ({"alias": alias, "endpoint_id": pin} for alias, pin in rows.items())
    return {(row["alias"], row["endpoint_id"]) for row in rows if isinstance(row, dict)
            and isinstance(row.get("alias"), str) and isinstance(row.get("endpoint_id"), str)}
def _ssh_plan(ep: _Endpoint, timeout: float):
    options = ["-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "LogLevel=ERROR",
               "-o", "ControlPath=none", "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
               "-o", "UpdateHostKeys=no", "-o", f"ConnectTimeout={max(1, math.ceil(timeout))}"]
    try:
        from remote_dev.auth import prepare_auth
        from remote_dev.core.endpoint import Endpoint
        from remote_dev.core.local_process import OwnedProcess
        from remote_dev.core.ssh_transport import _ssh_executable
    except ImportError:
        if ep.auth_ref:
            raise RemoteExecUnavailableError("transport_not_sent") from None
        system_ssh = Path(os.environ["SystemRoot"]) / "System32/OpenSSH/ssh.exe" if os.name == "nt" else Path("/usr/bin/ssh")
        executable, owner, env = str(system_ssh), None, None
    else:
        endpoint = Endpoint(**asdict(ep), ssh_mux=False, runtime_env=False)
        plan = prepare_auth(endpoint, timeout_ms=max(1, int(timeout * 1000)), repair=False)
        options.extend(token for flag, value in zip(plan.options[::2], plan.options[1::2])
                       if not value.casefold().startswith("batchmode=") for token in (flag, value))
        ep = plan.endpoint
        executable, owner, env = _ssh_executable(), OwnedProcess, plan.env
    if ep.identity_file:
        options.extend(["-i", ep.identity_file])
    if ep.known_hosts_file:
        known_hosts = Path.home() / Path(ep.known_hosts_file).expanduser()
        options.extend(["-o", "UserKnownHostsFile=" + json.dumps(str(known_hosts), ensure_ascii=False)])
    return [executable, *options, "-l", ep.user, "-p", str(ep.port), "--", ep.host], owner, env
def _remaining(deadline: float) -> float:
    budget = deadline - time.monotonic()
    if budget <= 0:
        raise TimeoutError("execution deadline expired")
    return budget
def _background(call):
    future = Future()
    def run():
        try:
            future.set_result(call())
        except Exception as error:
            future.set_exception(error)
    threading.Thread(target=run, daemon=True).start()
    return future
def _exchange(ep, command, payload, timeout, limit=CONFIG_LIMIT, *, handshake=False):
    deadline, process, owner, written = time.monotonic() + timeout, None, None, None
    sent = threading.Event()
    try:
        argv, owner_cls, env = _ssh_plan(ep, timeout)
        if ep.container and not HEX_DIGEST.fullmatch(ep.container):
            raise ValueError("container must be pinned")
        tokens = ["docker", "exec", "-i", ep.container, *command] if ep.container else command
        argv.append(shlex.join(tokens))
        pipes = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _remaining(deadline)
        if owner_cls:
            owner = owner_cls(argv, env=env, **pipes)
            process = owner.process
        else:
            flags = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                     else {"start_new_session": True})
            process = subprocess.Popen(argv, **pipes, **flags)
        if handshake:
            ready = _background(lambda: process.stdout.readline(len(READY) + 1))
            if ready.result(_remaining(deadline)) != READY:
                raise RuntimeError("worker not ready")
        def write():
            with process.stdin:
                view = memoryview(payload)
                while view:
                    view = view[os.write(process.stdin.fileno(), view):]
                    sent.set()
        output = _background(lambda: _bounded_read(process.stdout, limit))
        written = _background(write)
        data = output.result(_remaining(deadline))
        written.result(_remaining(deadline))
        if process.wait(timeout=_remaining(deadline)):
            raise RuntimeError("remote command failed")
        return data
    except (TimeoutError, subprocess.TimeoutExpired):
        if handshake and (sent.is_set() or (written is not None and not written.done())):
            error = RemoteExecUnavailableError("transport_not_sent")
            error.submission_state = "uncertain"
            raise error from None
        raise TimeoutError("remote_exec_timeout") from None
    except Exception:
        error = RemoteExecUnavailableError("transport_not_sent")
        if handshake and (sent.is_set() or (written is not None and not written.done())):
            error.submission_state = "uncertain"
        raise error from None
    finally:
        if process is not None:
            with suppress(OSError, RuntimeError, subprocess.TimeoutExpired):
                if owner is not None:
                    owner.stop(timeout=1)
                elif os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
                process.wait(timeout=1)
            process.stdout.close()
def _pin(ep: _Endpoint, deadline: float) -> _Endpoint:
    if not ep.container:
        return ep
    if not HEX_DIGEST.fullmatch(ep.container):
        # Resolve names under strict SSH before calling the substrate helper.
        raw = _exchange(replace(ep, container=None), [
            "docker", "inspect", "--type", "container", "--format", "{{.Id}} {{.State.Running}}", ep.container,
        ], b"", _remaining(deadline))
        container, running = raw.decode().split()
        if not HEX_DIGEST.fullmatch(container) or running != "true":
            raise ValueError("container unavailable")
        ep = replace(ep, container=container)
    try:
        from remote_dev.core.container_endpoint import pin_container_endpoint
    except ImportError:
        return ep
    return pin_container_endpoint(ep, timeout_ms=max(1, int(_remaining(deadline) * 1000)))
def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name or any(c in name for c in "\\\r\n\0:"):
        raise ValueError("invalid bundle member")
    return path
def _regular(path: Path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > FILE_LIMIT:
        raise ValueError("invalid bundle file")
    return info

def collect_bundle_files(repo_root: Path) -> tuple[Path, ...]:
    root = Path(repo_root)
    files = {path for pattern in BUNDLE_PATTERNS for path in root.glob(pattern)}
    # Path.glob skips recursive symlink directories, so audit the source tree.
    for folder, dirs, names in os.walk(root / "mindie_sim"):
        if any((Path(folder) / name).is_symlink() for name in (*dirs, *names) if name != "__pycache__"):
            raise ValueError("linked bundle member")
    for path in files:
        _safe_path(path.relative_to(root).as_posix())
        if any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)):
            raise ValueError("linked bundle directory")
        _regular(path)
    return tuple(sorted(files))
def _manifest(repo_root, paths):
    return "".join(f"{_hash_file(p)}  {_safe_path(p.relative_to(repo_root).as_posix())}\n" for p in paths).encode()
def _hash_file(path):
    _regular(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def compute_bundle_digest(repo_root: Path) -> str:
    return hashlib.sha256(_manifest(Path(repo_root), collect_bundle_files(repo_root))).hexdigest()

def build_bundle_archive(repo_root: Path) -> bytes:
    root = Path(repo_root)
    files = collect_bundle_files(root)
    manifest = _manifest(root, files)
    if sum(_regular(p).st_size for p in files) + len(manifest) > BUNDLE_LIMIT:
        raise ValueError("bundle byte limit exceeded")
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w|") as archive:
        for path in files:
            member = tarfile.TarInfo(path.relative_to(root).as_posix())
            member.size, member.mode = _regular(path).st_size, PRIVATE_FILE
            with path.open("rb") as stream:
                archive.addfile(member, stream)
        member = tarfile.TarInfo(MANIFEST)
        member.size, member.mode = len(manifest), PRIVATE_FILE
        archive.addfile(member, io.BytesIO(manifest))
    return output.getvalue()
def _private_dir(path: Path):
    path.mkdir(mode=PRIVATE_DIR, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (os.name == "posix" and
            (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != PRIVATE_DIR)):
        raise ValueError("unsafe staging directory")
def verify_staged_manifest(stage_dir: Path) -> str:
    root = Path(stage_dir)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("invalid stage root")
    path = root / MANIFEST
    _regular(path)
    expected = path.read_bytes()
    files = []
    for folder, dirs, names in os.walk(root):
        base = Path(folder)
        if base == root and "leases" in dirs:
            dirs.remove("leases")
            _private_dir(root / "leases")
            for lease in (root / "leases").iterdir():
                with suppress(FileNotFoundError):
                    if not LEASE_TOKEN.fullmatch(lease.name) or _regular(lease).st_size:
                        raise ValueError("invalid staged lease")
        if any((base / d).is_symlink() for d in dirs) or (base != root and not (dirs or names)):
            raise ValueError("extra staged directory")
        files.extend(base / name for name in names if base / name != path)
    if not files or _manifest(root, sorted(files)) != expected:
        raise ValueError("staged manifest mismatch")
    return hashlib.sha256(expected).hexdigest()

@contextmanager
def _lock(path: Path):
    import fcntl

    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, PRIVATE_FILE)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("invalid staging lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
def _digest_lock(stage: Path) -> Path:
    return stage.parent / f".lock.{stage.name}"
def _create_lease(stage: Path, request_id: str) -> Path:
    if not UUID_TOKEN.fullmatch(request_id):
        raise ValueError("invalid request ID")
    directory = stage / "leases"
    _private_dir(directory)
    lease = directory / f"{os.getpid()}-{request_id}.lease"
    fd = os.open(lease, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, PRIVATE_FILE)
    os.close(fd)
    return lease
def _active_leases(stage, *, request_id=None):
    active = False
    for lease in (stage / "leases").glob("*.lease"):
        pid, request = LEASE_TOKEN.fullmatch(lease.name).groups()
        try:
            if os.name == "posix":
                os.kill(int(pid), 0)
            else:
                raise ProcessLookupError
        except ProcessLookupError:
            if request == request_id or time.time() - lease.lstat().st_mtime > LEASE_HANDOFF_TIMEOUT:
                lease.unlink(missing_ok=True)
                continue
        except PermissionError:
            pass
        except TimeoutError:
            raise
        except OSError:
            if request == request_id or time.time() - lease.lstat().st_mtime > LEASE_HANDOFF_TIMEOUT:
                lease.unlink(missing_ok=True)
                continue
        active = True
    return active
def _native_root(stage: Path) -> Path:
    return stage.parent / f".native.{stage.name}"
def _prune(root, quota, protected):
    def size(tree):
        return sum(_regular(p).st_size for p in tree.rglob("*") if not p.is_dir()) if tree.is_dir() else 0
    for stale in list(root.iterdir()):
        if stale.is_symlink() or not stale.is_dir():
            continue
        if stale.name.startswith(".incoming-"):
            shutil.rmtree(stale)
        elif stale.name.startswith(".native.") and not (root / stale.name[len(".native."):]).exists():
            shutil.rmtree(stale)
    stages = sorted((p for p in root.iterdir() if HEX_DIGEST.fullmatch(p.name)), key=lambda p: p.lstat().st_mtime)
    sizes = {p: size(p) + size(_native_root(p)) for p in stages}
    total = sum(sizes.values())
    for path in stages:
        if total <= quota:
            return
        if path == protected:
            continue
        with _lock(_digest_lock(path)):
            _private_dir(path)
            if not _active_leases(path):
                shutil.rmtree(path)
                if _native_root(path).exists():
                    shutil.rmtree(_native_root(path))
                total -= sizes[path]
    if total > quota:
        raise ValueError("staging cache quota exhausted")
def _extract(archive_bytes, destination):
    with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes)) as data:
        budget = BUNDLE_LIMIT
        class TarInput:
            @staticmethod
            def read(size):
                nonlocal budget
                chunk = data.read(min(size, budget + 1))
                budget -= len(chunk)
                if budget < 0:
                    raise ValueError("archive expansion limit exceeded")
                return chunk
        with tarfile.open(fileobj=TarInput(), mode="r|") as archive:
            for member in archive:
                rel = _safe_path(member.name)
                if member.type != tarfile.REGTYPE or member.sparse or rel.parts[0] == "leases" or not 0 <= member.size <= FILE_LIMIT:
                    raise ValueError("invalid tar member")
                path = destination / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream, MIB)

def stage_bundle_local_dir(archive_bytes, cache_root, bundle_digest, *,
                           cache_quota_bytes, request_id, build_c11=True, build_timeout_s=STAGE_TIMEOUT) -> Path:
    if (not HEX_DIGEST.fullmatch(bundle_digest) or len(archive_bytes) > BUNDLE_LIMIT
            or not UUID_TOKEN.fullmatch(request_id)):
        raise ValueError("invalid bundle identity")
    root = Path(cache_root).absolute()
    _private_dir(root)
    target = root / bundle_digest
    with _lock(root / ".quota.lock"), _lock(_digest_lock(target)):
        fresh = not target.exists()
        try:
            if fresh:
                with tempfile.TemporaryDirectory(dir=root, prefix=".incoming-") as tmp:
                    tree = Path(tmp)
                    _extract(archive_bytes, tree)
                    if verify_staged_manifest(tree) != bundle_digest or target.is_symlink():
                        raise ValueError("bundle digest mismatch")
                    tree.rename(target)  # The quota and digest locks serialize publication.
            else:
                _private_dir(target)
                if verify_staged_manifest(target) != bundle_digest:
                    raise ValueError("bundle digest mismatch")
            if build_c11:
                subprocess.run([sys.executable, "-I", "-B", "-c",
                                "import sys; sys.path.insert(0,sys.argv[1]); "
                                "from mindie_sim.remote_exec import _prepare_native; "
                                "_prepare_native(sys.argv[1], build=True)", str(target)],
                               timeout=build_timeout_s, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            _prune(root, cache_quota_bytes, target)
            _create_lease(target, request_id)
            os.utime(target)
        except BaseException:
            if fresh and target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
                if _native_root(target).is_dir():
                    shutil.rmtree(_native_root(target))
            raise
    return target
def _boot(code: str, *args: str):
    return ["python3", "-I", "-B", "-u", "-c", code, *args]
_BOOTSTRAP = """
import hashlib,json,sys,types
from pathlib import Path
mode, location, sha, digest, *args = sys.argv[1:]
if mode == 'stage':
    source = sys.stdin.buffer.read(int(location))
else:
    root = Path(location)
    path = root / 'mindie_sim/remote_exec.py'
    with path.open('rb') as stream:
        source = stream.read(int(args[1])+1)
if hashlib.sha256(source).hexdigest() != sha:
    raise SystemExit(1)
module = types.ModuleType('_sim_bootstrap')
sys.modules[module.__name__] = module
exec(compile(source, '<sim_bootstrap>', 'exec'), module.__dict__)
if mode == 'stage':
    shm = Path('/dev/shm')
    use_shm = False
    try:
        st = module.os.statvfs(shm)
        use_shm = (not (st.f_flag & getattr(module.os, 'ST_NOEXEC', 0))) and st.f_bavail * st.f_frsize >= int(args[0])
    except OSError:
        use_shm = False
    cache = (shm / ('mindie-sim-' + str(module.os.getuid()))) if use_shm else (Path.home() / '.cache' / 'mindie-sim')
    tree = module.stage_bundle_local_dir(module._bounded_read(sys.stdin.buffer, module.BUNDLE_LIMIT),
             cache, digest, cache_quota_bytes=int(args[0]), request_id=args[1], build_timeout_s=float(args[2]))
    print(json.dumps(str(tree)))
else:
    if module.verify_staged_manifest(root) != digest:
        raise SystemExit(1)
    sys.path.insert(0, str(root))
    raise SystemExit(main(ready=True, expected_root=str(root), max_result_bytes=int(args[0])))
"""
def _stage_remote(endpoint, request_id, policy):
    source, repo = Path(__file__).read_bytes(), Path(__file__).resolve().parents[1]
    digest = compute_bundle_digest(repo)
    output = _exchange(endpoint, _boot(_BOOTSTRAP, "stage", str(len(source)), hashlib.sha256(source).hexdigest(),
                       digest, str(policy.cache_quota_bytes), request_id, str(policy.stage_timeout_s)),
                       source + build_bundle_archive(repo), policy.stage_timeout_s)
    stage = json.loads(output)
    if isinstance(stage, str) and PurePosixPath(stage).is_absolute() and PurePosixPath(stage).name == digest:
        return stage, digest
    raise ValueError("invalid staged location")
def _exec_remote(endpoint, payload, digest, policy):
    source = Path(__file__).read_bytes()
    limit = min(policy.max_result_bytes, RESULT_LIMIT)
    command = _boot(_BOOTSTRAP, "job", payload["stage_root"], hashlib.sha256(source).hexdigest(),
                    digest, str(limit), str(len(source)))
    output = _exchange(endpoint, command, _canonical(payload), payload["deadline_s"] + JOB_GRACE, limit, handshake=True)
    try:
        result = json.loads(output)
        if (result["protocol_version"] != PROTOCOL_VERSION or result["request_id"] != payload["request_id"]
                or ("error" not in result and (not isinstance(result["results"], list)
                    or not 1 <= result["effective_workers"] <= payload["max_workers"]))):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise RemoteExecUncertainError(payload["request_id"], None) from None
    return result
class RemoteDispatcher:
    """Resolve to a receipt and dispatch to (worker result, receipt).

    Probe returns capacity, optionally with a pinned endpoint. Stage returns
    (stage_path, bundle_digest). Local runners receive the worker payload.
    """

    def __init__(self, policy=RemoteExecPolicy(), *, probe_fn=None, stage_fn=None,
                 exec_fn=None, local_c11_ready_fn=None, endpoints_loader=None, cwd=None):

        self.policy, self.cwd = policy, Path.cwd() if cwd is None else Path(cwd)
        self._probe_fn, self._stage_fn = probe_fn or _probe_remote, stage_fn or _stage_remote
        self._exec_fn, self._local_ready = exec_fn or _exec_remote, local_c11_ready_fn or _local_ready
        self._load_endpoints = endpoints_loader or _load_endpoints
        self._probe_cache = {}

    def _resolve(self, count, *, job_deadline=None):
        policy = self.policy
        if policy.target == "local":
            return RemoteExecReceipt(), None
        if policy.target == "auto" and policy.local_single_step_if_c11 and count < REMOTE_CANDIDATE_THRESHOLD:
            try:
                if self._local_ready():
                    return RemoteExecReceipt(accel_mode="c11", fallback_reason="warm_local_c11"), None
            except TimeoutError:
                raise
            except (RuntimeError, OSError):
                pass
        reason, choices = "no_authorized_endpoint", []
        preferred = policy.preferred_alias
        explicit = {*policy.authorized_aliases, preferred}
        try:
            pairs = _authorized_pairs(policy)
            entries = self._load_endpoints(self.cwd)
            trusted = _merge_endpoints(_user_endpoint_files(), private=True) if pairs else {}
        except TimeoutError:
            raise
        except (RuntimeError, ValueError, TypeError, KeyError, OSError):
            reason, entries = "endpoint_configuration", {}
        for alias in sorted(entries.keys() if preferred is None else entries.keys() & {preferred}):
            if alias not in explicit and not any(name == alias for name, _ in pairs):
                continue
            reason = "probe_unavailable"
            try:
                ep = _endpoint(entries[alias])
                key = alias, ep.endpoint_id
                if alias not in explicit:
                    if key not in pairs:
                        continue
                    # User-owned files supply trust overrides for unattended use.
                    old = _endpoint(trusted[alias]) if alias in trusted else None
                    auth = old if old and old.endpoint_id == ep.endpoint_id else None
                    ep = replace(ep, **{k: getattr(auth, k, None) for k in ("identity_file", "known_hosts_file", "auth_ref")})
                expires, pinned, capacity = self._probe_cache.get((key, ep), (0, None, None))
                if expires <= time.monotonic():
                    probe_pol = policy if job_deadline is None else replace(policy, probe_timeout_s=min(policy.probe_timeout_s, _remaining(job_deadline)))
                    probe = self._probe_fn(ep, probe_pol)
                    pinned, capacity = (ep, probe) if isinstance(probe, Mapping) else probe
                    self._probe_cache[key, ep] = time.monotonic() + policy.probe_ttl_s, pinned, capacity
                workers, factor = size_remote_workers(
                    **{k: v for k, v in capacity.items() if k != "compiler"},
                    per_worker_mem_mib=policy.per_worker_mem_mib, max_remote_workers=policy.max_remote_workers,
                    force_saturated=alias == preferred)
                if workers:
                    alias_cf = alias.casefold()
                    secrets = [v.casefold() for v in (ep.host, ep.user, ep.auth_ref) if isinstance(v, str) and v]
                    public = "redacted" if any(s in alias_cf for s in secrets) else alias
                    choices.append((workers, alias, pinned, RemoteExecReceipt(
                        alias=public, endpoint_id=key[1], resolved_target="remote", effective_workers=min(count, workers),
                        limiting_factor="candidates" if count < workers else factor, accel_mode="c11")))
            except TimeoutError:
                raise
            except (RuntimeError, ValueError, TypeError, KeyError, OSError):
                pass
        if choices:
            _, _, endpoint, receipt = min(choices, key=lambda row: (-row[0], row[1]))
            return receipt, endpoint
        if policy.target == "remote":
            raise RemoteExecUnavailableError(reason)
        return RemoteExecReceipt(fallback_reason=reason), None

    def resolve_target(self, n_candidates: int) -> RemoteExecReceipt: return self._resolve(n_candidates)[0]
    def dispatch_batch(self, entrypoint: str, base_case: CaseInput, candidates: Sequence[Mapping], *,
                       local_runner=None, deadline_s: float | None = None) -> tuple[object, RemoteExecReceipt]:
        from mindie_sim.prediction.results import json_payload
        from mindie_sim.schema.case import Case

        policy = self.policy
        if entrypoint not in ENTRYPOINTS or not candidates: raise ValueError("invalid batch entrypoint or empty candidates")
        budget = policy.job_timeout_s if deadline_s is None else min(deadline_s, policy.job_timeout_s)
        if not math.isfinite(budget) or budget <= 0: raise ValueError("invalid execution deadline")
        job_deadline, request_id = time.monotonic() + budget, uuid.uuid4().hex
        if entrypoint == "optimize_ladder":
            from mindie_sim.primitives.schema import PrimitiveTable
            candidates = list(candidates)
            for index, candidate in enumerate(candidates):
                options = candidate.get("options")
                if isinstance(options, Mapping) and isinstance(table := options.get("primitive_table"), PrimitiveTable):
                    wire = dict(kind="primitive_table", envelope=table.to_envelope_dict(), warnings=table.warnings)
                    candidates[index] = dict(candidate, options=dict(options, primitive_table=wire))
        payload = dict(protocol_version=PROTOCOL_VERSION, request_id=request_id, deadline_s=budget,
                       entrypoint=entrypoint, base_case=json_payload(base_case.to_dict() if isinstance(base_case, Case) else base_case),
                       candidates=json_payload(candidates), stage_root=None, max_workers=policy.max_remote_workers,
                       per_worker_mem_mib=policy.per_worker_mem_mib, primitive_manifest_path=None)
        limit = min(policy.max_request_bytes, REQUEST_LIMIT)
        if len(_canonical(payload)) > limit: raise ValueError("request byte limit exceeded")
        receipt, endpoint = self._resolve(len(candidates), job_deadline=job_deadline)
        receipt, reason = replace(receipt, request_id=request_id), receipt.fallback_reason
        if endpoint is not None:
            try:
                stage_pol = replace(policy, stage_timeout_s=min(policy.stage_timeout_s, _remaining(job_deadline)))
                stage, digest = self._stage_fn(endpoint, request_id, stage_pol)
            except TimeoutError: raise
            except Exception: reason = "stage_unavailable"
            else:
                remote = payload | dict(deadline_s=_remaining(job_deadline), stage_root=stage, max_workers=receipt.effective_workers,
                                        primitive_manifest_path="data/primitives/manifest.json")
                if len(_canonical(remote)) > limit: raise ValueError("request byte limit exceeded")
                try:
                    result = self._exec_fn(endpoint, remote, digest, policy)
                except TimeoutError: raise
                except Exception as error:
                    if getattr(error, "submission_state", None) != "not_sent":
                        raise RemoteExecUncertainError(request_id, receipt.alias) from None
                    reason = "transport_not_sent"
                else:
                    if isinstance(result, dict) and "error" in result:
                        if result["error"] == "deadline_exceeded": raise TimeoutError("deadline_exceeded")
                        raise RuntimeError("worker_failed")
                    count = result.get("effective_workers", receipt.effective_workers)
                    return result, replace(receipt, effective_workers=count, bundle_digest=digest, submission_state="acknowledged")
            if policy.target == "remote": raise RemoteExecUnavailableError(reason)
        local_payload = payload | {"deadline_s": _remaining(job_deadline)}
        result = (local_runner(local_payload) if local_runner else
                  execute_worker_request(local_payload, max_result_bytes=policy.max_result_bytes))
        count = result.get("effective_workers", 1) if isinstance(result, dict) else 1
        return result, RemoteExecReceipt(effective_workers=count, fallback_reason=reason, request_id=request_id,
                                         accel_mode=receipt.accel_mode if endpoint is None else "auto")

    def run_step(self, case: CaseInput, request: LoweringRequest, *, replay: ReplayOptions | None = None, deadline_s: float | None = None):
        return self.dispatch_batch("run_step", case, [{"request": request, "replay": replay or {}}], deadline_s=deadline_s)
    def evaluate_three_tiers(self, case: CaseInput, request: StepRequest, *, replay: ReplayOptions | None = None,
                             deadline_s: float | None = None):
        return self.dispatch_batch("evaluate_three_tiers", case, [{"request": request, "replay": replay or {}}], deadline_s=deadline_s)
    def solve_sla(self, case: CaseInput, request: SlaRequest, *, deadline_s: float | None = None):
        return self.dispatch_batch("solve_sla", case, [{"request": request}], deadline_s=deadline_s)
    def sweep_topology(self, case: CaseInput, request: SweepRequest, *, decode_case: CaseInput | None = None, deadline_s: float | None = None):
        from mindie_sim.schema.case import Case
        return self.dispatch_batch("sweep_topology", case, [{"request": request, "decode_case": decode_case.to_dict() if isinstance(decode_case, Case) else decode_case}], deadline_s=deadline_s)
    def optimize_ladder(self, case: CaseInput, candidates: Sequence[Mapping], *, local_runner=None, deadline_s=None):
        """Dispatch candidates with a StepRequest, optional replay, and ladder options under `options`."""
        return self.dispatch_batch("optimize_ladder", case, candidates, local_runner=local_runner, deadline_s=deadline_s)


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
        wire = json_payload(result)
        if entrypoint == "optimize_ladder":
            # Public plan encoding lets a remote winner feed `finalize_plan` unchanged.
            from mindie_sim.rewrite import encode_rewrite_plan

            for key in ("framework_plan", "dsl_plan"):
                wire[key] = encode_rewrite_plan(getattr(result, key))
            wire["verification"]["passed"] = bool(result.verification.passed)
        return {"candidate_id": candidate["candidate_id"], "objective_key": candidate["objective_key"],
                "result": wire}
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
