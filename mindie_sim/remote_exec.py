"""Authorized remote execution for the CPU simulation worker.

[Role] Resolve execution policy, probe CPU capacity, publish verified bundles,
  and send one bounded request to a pinned endpoint.
[Invariants] Local resolution takes precedence over endpoint I/O. Only
  authorized endpoints receive data, and only unsent work can fall back.
[Timeouts & Locks] Transport uses monotonic deadlines. Cache publication
  takes the quota lock before the digest lock. Workers only take digest locks.
[Side-Effect Policy] No import-time I/O, device access, or auth repair.
  Remote writes are confined to the private simulation cache.
"""
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
    "mindie_sim/stack_sources/*.json", "mindie_sim/model_configs/*.json",
    "mindie_sim/catalog/configs/*.json", "data/primitives/*.json",
)
PUBLIC_REASONS = frozenset({
    "endpoint_configuration", "no_authorized_endpoint", "probe_unavailable",
    "stage_unavailable", "transport_not_sent", "transport_lost", "warm_local_c11",
})
ENTRYPOINTS = frozenset({"run_step", "evaluate_three_tiers", "solve_sla", "sweep_topology", "optimize_ladder"})

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
def _user_endpoint_files() -> tuple[Path, Path]:
    return tuple(Path.home() / folder / "endpoints.json" for folder in (".mindie", ".mindie-coordinator"))
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
    path = policy.policy_path or Path.home() / ".mindie/sim_remote_policy.json"
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
                                "from mindie_sim._remote_worker import _prepare_native; "
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
    from mindie_sim._remote_worker import main
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
        from mindie_sim._remote_worker import _local_ready, _probe_remote

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
        from mindie_sim._remote_worker import execute_worker_request
        from mindie_sim.prediction.results import json_payload
        from mindie_sim.schema.case import Case

        policy = self.policy
        if entrypoint not in ENTRYPOINTS or not candidates: raise ValueError("invalid batch entrypoint or empty candidates")
        budget = policy.job_timeout_s if deadline_s is None else min(deadline_s, policy.job_timeout_s)
        if not math.isfinite(budget) or budget <= 0: raise ValueError("invalid execution deadline")
        job_deadline, request_id = time.monotonic() + budget, uuid.uuid4().hex
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
                             target_eff: float | None = None, deadline_s: float | None = None):
        opts = {} if target_eff is None else {"target_eff": target_eff}
        return self.dispatch_batch("evaluate_three_tiers", case, [{"request": request, "replay": replay or {}, "options": opts}], deadline_s=deadline_s)
    def solve_sla(self, case: CaseInput, request: SlaRequest, *, deadline_s: float | None = None):
        return self.dispatch_batch("solve_sla", case, [{"request": request}], deadline_s=deadline_s)
    def sweep_topology(self, case: CaseInput, request: SweepRequest, *, decode_case: CaseInput | None = None, deadline_s: float | None = None):
        from mindie_sim.schema.case import Case
        return self.dispatch_batch("sweep_topology", case, [{"request": request, "decode_case": decode_case.to_dict() if isinstance(decode_case, Case) else decode_case}], deadline_s=deadline_s)
    def optimize_ladder(self, case: CaseInput, candidates: Sequence[Mapping], *, local_runner=None, deadline_s=None):
        return self.dispatch_batch("optimize_ladder", case, candidates, local_runner=local_runner, deadline_s=deadline_s)
