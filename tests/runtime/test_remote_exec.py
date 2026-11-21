"""Policy, staging, and CPU worker behavior at the remote batch boundary.

[Role] Exercise dispatch, endpoint authorization, capacity, cache, and wire parity.
[Invariants] Only authorized endpoints qualify, only unsent work can retry,
  and identical inputs replay in the same canonical candidate order.
[Timeouts & Locks] Staging uses real cache locks, with no sleeps. The CLI and
  loopback SSH subprocesses have bounded timeouts and run only at Tier 1.
[Side-Effect Policy] Temporary endpoint and cache files stay under tmp_path.
  Transport is mocked throughout. All prediction runs use the CPU engine.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
from unittest.mock import Mock

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import sku_case
from mindie_sim import _remote_worker as worker
from mindie_sim import remote_exec as remote
from mindie_sim.prediction.results import StepRequest
from mindie_sim.stack_registry import LoweringRequest

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIAS = "compute-lab"
ENDPOINT = dict(host="compute.example.invalid", port=22, user="sim-worker", root="/srv/private")
CAPACITY = dict(affinity_cpus=384, cgroup_cpus=None, loadavg_1m=13.25,
                mem_avail_mib=256 * 1024, cgroup_mem_remaining_mib=None)
REQUEST_ID = "a" * 32
BUNDLE_DIGEST = "b" * 64
CACHE_QUOTA_BYTES = 256 * 1024**2
JOB_TIMEOUT_S = 10
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="requires POSIX modes and cache locks")


def _unexpected(*args, **kwargs):
    pytest.fail("unexpected endpoint I/O, remote submission, or local replay")


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_config(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_wire(data))
    path.chmod(mode)


@pytest.fixture
def endpoint_paths(tmp_path, monkeypatch):
    home, cwd, env_file = (tmp_path / name for name in ("home", "repo", "override.json"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("REMOTE_DEV_ENDPOINTS_FILE", str(env_file))
    return home, cwd, env_file


def _dispatcher(cwd, target, *, warm=False, endpoints=None, probe=None, **kwargs):
    return remote.RemoteDispatcher(
        remote.RemoteExecPolicy(target=target, authorized_aliases=(ALIAS,), max_remote_workers=1),
        cwd=cwd,
        local_c11_ready_fn=lambda: warm,
        endpoints_loader=endpoints if endpoints is not None else lambda _: {ALIAS: ENDPOINT},
        probe_fn=probe if probe is not None else lambda *_: CAPACITY,
        **kwargs,
    )


def test_explicit_local_never_reads_endpoint_configuration(endpoint_paths, monkeypatch):
    _, cwd, _ = endpoint_paths
    (cwd / "endpoints.json").parent.mkdir()
    (cwd / "endpoints.json").write_text("invalid JSON", encoding="utf-8")
    monkeypatch.setattr(remote, "_read_config", _unexpected)
    dispatcher = remote.RemoteDispatcher(
        remote.RemoteExecPolicy(target="local"), cwd=cwd, endpoints_loader=_unexpected,
        probe_fn=_unexpected, local_c11_ready_fn=_unexpected,
    )
    receipt = dispatcher.resolve_target(100)
    assert receipt.resolved_target == "local"
    assert receipt.fallback_reason is None


@pytest.mark.parametrize(
    "target,warm,count,expected,reason",
    [
        ("auto", True, 1, "local", "warm_local_c11"),
        ("auto", True, 3, "local", "warm_local_c11"),
        ("auto", True, 4, "remote", None),
        ("auto", True, 5, "remote", None),
        ("auto", False, 1, "remote", None),
        ("prefer_remote", True, 1, "remote", None),
    ],
)
def test_execution_target_resolution(endpoint_paths, target, warm, count, expected, reason):
    _, cwd, _ = endpoint_paths
    load = Mock(return_value={ALIAS: ENDPOINT})
    probe = Mock(return_value=CAPACITY)
    receipt = _dispatcher(cwd, target, warm=warm, endpoints=load, probe=probe).resolve_target(count)
    assert receipt.resolved_target == expected
    assert receipt.fallback_reason == reason
    assert receipt.accel_mode == "c11"
    if expected == "remote":
        load.assert_called_once_with(cwd)
        probe.assert_called_once()
        assert receipt.alias == ALIAS
    else:
        load.assert_not_called()
        probe.assert_not_called()


def test_unavailable_endpoint_respects_execution_target(endpoint_paths):
    _, cwd, _ = endpoint_paths
    options = dict(warm=True, endpoints=lambda _: {}, probe=_unexpected)
    receipt = _dispatcher(cwd, "prefer_remote", **options).resolve_target(1)
    assert receipt.resolved_target == "local"
    assert receipt.fallback_reason == "no_authorized_endpoint"

    assert remote.RemoteExecutionUnavailableError is remote.RemoteExecUnavailableError
    with pytest.raises(remote.RemoteExecutionUnavailableError, match="no_authorized_endpoint") as error:
        _dispatcher(cwd, "remote", **options).resolve_target(1)
    assert error.value.submission_state == "not_sent"


def test_endpoint_file_precedence(endpoint_paths):
    home, cwd, env_file = endpoint_paths
    sources = (
        home / ".mindie/endpoints.json",
        home / ".mindie-coordinator/endpoints.json",
        env_file,
        cwd / "endpoints.json",
        cwd / "endpoints.local.json",
    )
    for precedence, source in enumerate(sources):
        endpoint = ENDPOINT | {"port": ENDPOINT["port"] + precedence}
        _write_config(source, {"endpoints": {ALIAS: endpoint, f"source-{precedence}": ENDPOINT}})
        loaded = remote._load_endpoints(cwd)
        assert loaded[ALIAS] == endpoint
        assert loaded.keys() - {ALIAS} == {f"source-{index}" for index in range(precedence + 1)}
    for precedence, source in reversed(tuple(enumerate(sources))):
        assert remote._load_endpoints(cwd)[ALIAS]["port"] == ENDPOINT["port"] + precedence
        source.unlink()
    assert remote._load_endpoints(cwd) == {}


def _write_pin(home, endpoint, *, mode=0o600):
    _write_config(home / ".mindie/sim_remote_policy.json", {
        "authorized_endpoints": [{"alias": ALIAS, "endpoint_id": endpoint.endpoint_id}],
    }, mode=mode)


def test_unattended_dispatch_refuses_repository_only_alias(endpoint_paths):
    _, cwd, _ = endpoint_paths
    _write_config(cwd / "endpoints.json", {ALIAS: ENDPOINT})
    local = Mock(return_value="computed locally")
    dispatcher = remote.RemoteDispatcher(cwd=cwd, local_c11_ready_fn=lambda: False,
                                         probe_fn=_unexpected, stage_fn=_unexpected, exec_fn=_unexpected)
    result, receipt = dispatcher.dispatch_batch("run_step", {}, [{}], local_runner=local)
    assert result == "computed locally"
    local.assert_called_once()
    assert receipt.resolved_target == "local"
    assert receipt.fallback_reason == "no_authorized_endpoint"


@pytest.mark.parametrize("spoof", [{"host": "other.example.invalid"}, {"port": 2222}], ids=["host", "port"])
def test_unattended_dispatch_requires_exact_endpoint_pin(endpoint_paths, spoof):
    home, cwd, _ = endpoint_paths
    endpoint = remote._endpoint(ENDPOINT)
    _write_pin(home, endpoint)
    _write_config(home / ".mindie/endpoints.json", {ALIAS: ENDPOINT})
    probe = Mock(return_value=CAPACITY)
    dispatcher = remote.RemoteDispatcher(cwd=cwd, local_c11_ready_fn=lambda: False,
                                         probe_fn=probe, stage_fn=_unexpected, exec_fn=_unexpected)
    assert dispatcher.resolve_target(1).endpoint_id == endpoint.endpoint_id
    probe.reset_mock()

    replacement = ENDPOINT | spoof
    assert remote._endpoint(replacement).endpoint_id != endpoint.endpoint_id
    _write_config(cwd / "endpoints.json", {ALIAS: replacement})
    result, receipt = dispatcher.dispatch_batch("run_step", {}, [{}], local_runner=lambda _: "local")
    assert result == "local"
    assert receipt.resolved_target == "local"
    assert receipt.endpoint_id is None
    probe.assert_not_called()


@POSIX_ONLY
@pytest.mark.parametrize("mode", [0o666, 0o620, 0o602, 0o644, 0o600])
def test_unattended_policy_requires_strict_private_mode(endpoint_paths, mode):
    home, cwd, _ = endpoint_paths
    endpoint = remote._endpoint(ENDPOINT)
    _write_config(cwd / "endpoints.json", {ALIAS: ENDPOINT})
    _write_pin(home, endpoint, mode=mode)
    probe = Mock(return_value=CAPACITY)
    dispatcher = remote.RemoteDispatcher(cwd=cwd, local_c11_ready_fn=lambda: False, probe_fn=probe)
    receipt = dispatcher.resolve_target(1)
    if mode == 0o600:
        assert receipt.resolved_target == "remote"
        assert receipt.endpoint_id == endpoint.endpoint_id
        probe.assert_called_once()
    else:
        assert receipt.resolved_target == "local"
        assert receipt.fallback_reason == "endpoint_configuration"
        probe.assert_not_called()


def test_ssh_requires_host_verification_and_pinned_container(monkeypatch):
    monkeypatch.setitem(sys.modules, "remote_dev.auth", None)
    endpoint = remote._endpoint(ENDPOINT | {"container": "mutable-container-name"})
    argv, _, _ = remote._ssh_plan(endpoint, 1)
    host_check = [arg for arg in argv if arg.lower().startswith("stricthostkeychecking=")]
    assert host_check == ["StrictHostKeyChecking=yes"]
    monkeypatch.setattr(remote.subprocess, "Popen", _unexpected)
    with pytest.raises(remote.RemoteExecUnavailableError, match="transport_not_sent") as error:
        remote._exchange(endpoint, ["python3"], b"private batch", timeout=1, handshake=True)
    assert error.value.submission_state == "not_sent"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, (370, "load_headroom")),
        ({"cgroup_cpus": 32.5, "loadavg_1m": 0}, (32, "cgroup_cpu")),
        ({"cgroup_mem_remaining_mib": 7 * 384 - 1}, (6, "memory")),
        ({"loadavg_1m": 384}, (0, "load_headroom")),
        ({"loadavg_1m": 385, "force_saturated": True}, (1, "load_headroom")),
        ({"max_remote_workers": 19}, (19, "policy")),
    ],
)
def test_worker_count_obeys_tightest_container_limit(overrides, expected):
    capacity = CAPACITY | dict(per_worker_mem_mib=384, max_remote_workers=None) | overrides
    assert remote.size_remote_workers(**capacity) == expected


def test_preferred_endpoint_can_use_one_saturated_worker(endpoint_paths):
    _, cwd, _ = endpoint_paths
    options = dict(cwd=cwd, endpoints_loader=lambda _: {ALIAS: ENDPOINT},
                   probe_fn=lambda *_: CAPACITY | {"loadavg_1m": CAPACITY["affinity_cpus"]})
    policy = remote.RemoteExecPolicy(target="prefer_remote", authorized_aliases=(ALIAS,))
    assert remote.RemoteDispatcher(policy, **options).resolve_target(4).resolved_target == "local"
    policy = remote.RemoteExecPolicy(target="prefer_remote", preferred_alias=ALIAS)
    receipt = remote.RemoteDispatcher(policy, **options).resolve_target(4)
    assert receipt.resolved_target == "remote"
    assert (receipt.effective_workers, receipt.limiting_factor) == (1, "load_headroom")


@pytest.fixture
def bundle_source(tmp_path):
    root = tmp_path / "source"
    module = root / "mindie_sim/example.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"VALUE = 1\n")
    _write_config(root / "data/primitives/manifest.json", {"schema_version": "mindie-l0-primitives/1", "tables": {}})
    return root


def _stage(source, cache, request_id=REQUEST_ID, quota=CACHE_QUOTA_BYTES):
    digest = remote.compute_bundle_digest(source)
    archive = remote.build_bundle_archive(source)
    stage = remote.stage_bundle_local_dir(archive, cache, digest, request_id=request_id,
                                          cache_quota_bytes=quota, build_c11=False)
    return stage, digest


@POSIX_ONLY
def test_digest_stage_private_mode_manifest_and_job_lease(bundle_source, tmp_path):
    assert remote.RemoteExecPolicy().cache_quota_bytes == CACHE_QUOTA_BYTES
    cache = tmp_path / "cache"
    stage, digest = _stage(bundle_source, cache)
    assert stage == cache / digest
    assert len(digest) == 64
    assert remote.verify_staged_manifest(stage) == digest
    for directory in (cache, stage, stage / "leases"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    lease = stage / "leases" / f"{os.getpid()}-{REQUEST_ID}.lease"
    assert list(lease.parent.iterdir()) == [lease]
    assert lease.read_bytes() == b""

    staged_module = stage / "mindie_sim/example.py"
    staged_module.write_bytes(staged_module.read_bytes().replace(b"1", b"2", 1))
    with pytest.raises(ValueError, match="^staged manifest mismatch$"):
        remote.verify_staged_manifest(stage)


@POSIX_ONLY
@pytest.mark.parametrize("member_name,symlink,message", [
    ("../escape", False, "invalid bundle member"),
    ("mindie_sim/link.py", True, "invalid tar member"),
])
def test_staging_rejects_archive_members_that_can_escape(tmp_path, member_name, symlink, message):
    payload = b"escaped"
    output = io.BytesIO()
    member = tarfile.TarInfo(member_name)
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
            archive.addfile(member)
        else:
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match=message):
        remote.stage_bundle_local_dir(output.getvalue(), cache, BUNDLE_DIGEST,
                                      request_id=REQUEST_ID, cache_quota_bytes=CACHE_QUOTA_BYTES, build_c11=False)
    assert not (cache / BUNDLE_DIGEST).exists()
    assert not (cache / "escape").exists()


@POSIX_ONLY
def test_cache_byte_quota_protects_active_lease_then_evicts(bundle_source, tmp_path):
    cache = tmp_path / "cache"
    first, first_digest = _stage(bundle_source, cache)
    lease, = (first / "leases").glob("*.lease")
    quota = sum(path.stat().st_size for path in first.rglob("*") if path.is_file())
    (bundle_source / "mindie_sim/example.py").write_bytes(b"VALUE = 2\n")
    next_digest = remote.compute_bundle_digest(bundle_source)
    assert next_digest != first_digest

    with pytest.raises(ValueError, match="staging cache quota exhausted"):
        _stage(bundle_source, cache, request_id="c" * 32, quota=quota)
    assert lease.exists()
    assert remote.verify_staged_manifest(first) == first_digest
    assert not (cache / next_digest).exists()

    lease.unlink()
    abandoned = cache / ".incoming-stale"
    abandoned.mkdir()
    (abandoned / "junk.bin").write_bytes(b"x" * 64)
    second, _ = _stage(bundle_source, cache, request_id="c" * 32, quota=quota)
    assert remote.verify_staged_manifest(second) == next_digest
    assert not first.exists() and not abandoned.exists()


@pytest.mark.parametrize("target", ["auto", "prefer_remote"])
def test_unsent_submission_can_run_locally(endpoint_paths, target):
    _, cwd, _ = endpoint_paths
    exec_fn = Mock(side_effect=remote.RemoteExecUnavailableError("transport_not_sent"))
    local = Mock(return_value="local result")
    stage = Mock(return_value=(f"/stage/{BUNDLE_DIGEST}", BUNDLE_DIGEST))
    dispatcher = _dispatcher(cwd, target, exec_fn=exec_fn, stage_fn=stage)
    result, receipt = dispatcher.dispatch_batch("run_step", {}, [{}], local_runner=local)

    exec_fn.assert_called_once()
    local.assert_called_once()
    remote_payload = exec_fn.call_args.args[1]
    local_payload = local.call_args.args[0]
    assert remote_payload["request_id"] == stage.call_args.args[1] == receipt.request_id
    assert local_payload["request_id"] == receipt.request_id
    assert local_payload["stage_root"] is None
    assert local_payload["candidates"] == remote_payload["candidates"]
    assert result == "local result"
    assert receipt.resolved_target == "local"
    assert receipt.fallback_reason == "transport_not_sent"
    assert receipt.submission_state == "not_sent"


@pytest.mark.parametrize("state", ["uncertain", None], ids=["explicit", "unknown"])
def test_uncertain_submission_never_replays_locally(endpoint_paths, state):
    _, cwd, _ = endpoint_paths
    error = RuntimeError("connection failed during batch")
    if state is not None:
        error.submission_state = state
    exec_fn = Mock(side_effect=error)
    local = Mock(side_effect=_unexpected)
    dispatcher = _dispatcher(cwd, "auto", exec_fn=exec_fn,
                             stage_fn=lambda *_: (f"/stage/{BUNDLE_DIGEST}", BUNDLE_DIGEST))
    with pytest.raises(remote.RemoteExecUncertainError) as caught:
        dispatcher.dispatch_batch("run_step", {}, [{}], local_runner=local)

    exec_fn.assert_called_once()
    local.assert_not_called()
    assert caught.value.request_id == exec_fn.call_args.args[1]["request_id"]
    assert caught.value.submission_state == "uncertain"
    assert caught.value.alias == ALIAS


@pytest.mark.parametrize("alias", [
    "192.0.2.12", "node-192.0.2.12", "node-192-0-2-12", "host", "user", "root", "auth_ref", "identity_file", "$HOME",
])
def test_receipt_only_exports_allowlisted_redacted_fields(alias):
    receipt = remote.RemoteExecReceipt(alias=alias, endpoint_id=ENDPOINT["host"], bundle_digest="$HOME",
                                       request_id=REQUEST_ID, resolved_target="remote", accel_mode="c11")
    assert receipt.to_dict() == {
        "alias": "redacted", "endpoint_id": None, "resolved_target": "remote", "effective_workers": 1,
        "limiting_factor": None, "bundle_digest": None, "accel_mode": "c11", "fallback_reason": None,
        "request_id": REQUEST_ID, "submission_state": "not_sent",
    }


@pytest.mark.parametrize("field,upper", [("host", False), ("host", True), ("user", False), ("auth_ref", False)])
def test_endpoint_alias_cannot_leak_actual_connection_identity(endpoint_paths, field, upper):
    home, cwd, _ = endpoint_paths
    config = ENDPOINT | {"auth_ref": "private-login", "identity_file": str(home / "private-key")}
    if upper:
        config["host"] = "Compute.Example.Invalid"
    alias = f"lab-{config[field].lower()}" if upper else config[field]
    dispatcher = remote.RemoteDispatcher(
        remote.RemoteExecPolicy(target="remote", authorized_aliases=(alias,)), cwd=cwd,
        endpoints_loader=lambda _: {alias: config}, probe_fn=lambda *_: CAPACITY,
    )
    receipt = dispatcher.resolve_target(1).to_dict()
    assert receipt["resolved_target"] == "remote"
    assert receipt["alias"] == "redacted"
    encoded = _wire(receipt).decode().casefold()
    for secret in (*config.values(), str(home), "$home"):
        if isinstance(secret, str):
            assert secret.casefold() not in encoded


@pytest.fixture
def prediction_case(monkeypatch):
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "python")
    return sku_case(REPRESENTATIVE_SKUS["a2"])


@POSIX_ONLY
@pytest.mark.parametrize("entrypoint", ["run_step", "evaluate_three_tiers"])
def test_batch_parity_preserves_order_counts_and_predicted_spans(
    endpoint_paths, bundle_source, prediction_case, tmp_path, monkeypatch, entrypoint,
):
    _, cwd, _ = endpoint_paths
    cache = tmp_path / "cache"
    candidates = []
    contexts = {"zeta": 16, "beta": 32, "alpha": 8}
    for candidate_id, context in contexts.items():
        step = StepRequest(phase="decode", context=context)
        request = LoweringRequest(step=step) if entrypoint == "run_step" else step
        candidates.append(dict(
            candidate_id=candidate_id, objective_key=int(candidate_id == "beta"), request=request,
        ))

    def stage_fn(endpoint, request_id, policy):
        stage, digest = _stage(bundle_source, cache, request_id, policy.cache_quota_bytes)
        return str(stage), digest

    def exec_fn(endpoint, payload, digest, policy):
        request = remote._canonical(payload)
        assert _wire(json.loads(request)) == request
        response = worker.execute_worker_request(request, max_result_bytes=policy.max_result_bytes)
        assert response["request_id"] == payload["request_id"]
        return json.loads(remote._canonical(response))

    # Mock only the C11 loader. The worker still validates the stage and its manifest.
    native = Mock()
    monkeypatch.setattr(worker, "_prepare_native", native)
    local = _dispatcher(cwd, "local", stage_fn=_unexpected, exec_fn=_unexpected)
    routed = _dispatcher(cwd, "remote", stage_fn=stage_fn, exec_fn=exec_fn)
    args = (entrypoint, prediction_case, candidates)
    local_result, local_receipt = local.dispatch_batch(*args, deadline_s=JOB_TIMEOUT_S)
    remote_result, remote_receipt = routed.dispatch_batch(*args, deadline_s=JOB_TIMEOUT_S, local_runner=_unexpected)

    assert local_receipt.resolved_target == "local" and remote_receipt.resolved_target == "remote"
    assert remote_receipt.submission_state == "acknowledged" and remote_receipt.request_id == remote_result["request_id"]
    assert local_result["effective_workers"] == remote_result["effective_workers"] == 1
    stage = cache / remote_receipt.bundle_digest
    native.assert_called_once_with(stage)
    assert not list((stage / "leases").iterdir())
    expected_order = [(0, "alpha"), (0, "zeta"), (1, "beta")]
    for result in (local_result, remote_result):
        assert [(row["objective_key"], row["candidate_id"]) for row in result["results"]] == expected_order
    for direct, routed_row in zip(local_result["results"], remote_result["results"], strict=True):
        left, right = direct["result"], routed_row["result"]
        context = contexts[direct["candidate_id"]]
        pairs = [(left, right)] if entrypoint == "run_step" else [(left[tier], right[tier]) for tier in ("unmerge", "mainline", "should_achieve")]
        for local_step, remote_step in pairs:
            assert local_step["graph_id"] == remote_step["graph_id"] and local_step["request"]["context"] == remote_step["request"]["context"] == context
            assert local_step["native_steps"] == remote_step["native_steps"] == 0
            local_ops, remote_ops = local_step["operators"], remote_step["operators"]
            assert len(local_ops) == len(remote_ops) > 0 and len(local_step["pipelines"]) == len(remote_step["pipelines"])
            assert all([op["ordinal"] for op in ops] == list(range(len(ops))) for ops in (local_ops, remote_ops))
            assert local_step["span_ms"] > 0 and abs(local_step["span_ms"] - remote_step["span_ms"]) <= 1e-12


@pytest.mark.tier1
def test_worker_module_canonical_stdio_and_mock_ssh_round_trip(tmp_path, monkeypatch, prediction_case):
    import subprocess

    (tmp_path / "sitecustomize.py").write_text(
        "import os, sys\n"
        "blocked = {'torch', 'torch_npu', 'acl', 'npu'}\n"
        "class RejectDeviceImport:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in blocked:\n"
        "            os._exit(97)\n"
        "if blocked.intersection(sys.modules):\n"
        "    os._exit(97)\n"
        "sys.meta_path.insert(0, RejectDeviceImport())\n"
        "sys.stderr.write('device import guard active\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.chdir(REPO_ROOT)
    payload = dict(protocol_version=remote.PROTOCOL_VERSION, request_id=REQUEST_ID,
                   deadline_s=JOB_TIMEOUT_S, entrypoint="run_step", base_case=prediction_case.to_dict(),
                   candidates=[{"request": {"step": {"phase": "decode", "context": 16}}}], max_workers=1)
    request = _wire(payload)
    command = [sys.executable, "-B", "-m", "mindie_sim._remote_worker"]
    process = subprocess.run(command, input=request, capture_output=True, timeout=JOB_TIMEOUT_S, check=False)
    assert process.returncode == 0, process.stderr
    assert b"device import guard active\n" in process.stderr
    response = json.loads(process.stdout)
    assert response["request_id"] == REQUEST_ID
    assert response["protocol_version"] == remote.PROTOCOL_VERSION
    assert response["results"][0]["result"]["span_ms"] > 0
    assert process.stdout == _wire(response)

    # Emulate the SSH command runner locally while retaining the real READY and pipe handling.
    fake_ssh = [sys.executable, "-B", "-c", (
        "import os, shlex, sys\n"
        "command = shlex.split(sys.argv[1])\n"
        "os.execv(command[0], command)\n"
    )]
    monkeypatch.setattr(remote, "_ssh_plan", lambda *_: (fake_ssh.copy(), None, None))
    endpoint = remote._endpoint(ENDPOINT)
    ssh_command = [sys.executable, "-B", "-u", "-c", (
        "from mindie_sim._remote_worker import main\n"
        "raise SystemExit(main(ready=True))\n"
    )]
    assert remote._exchange(endpoint, ssh_command, request, JOB_TIMEOUT_S, handshake=True) == process.stdout


def test_timeout_preserved_remaining_deadline_sent_and_c11_rebuilds_on_miss(
    endpoint_paths, bundle_source, tmp_path, monkeypatch,
):
    _, cwd, _ = endpoint_paths
    with pytest.raises(TimeoutError):
        _dispatcher(cwd, "auto", warm=True, probe=Mock(side_effect=TimeoutError("probe"))).resolve_target(4)
    now, seen = [100.0], []
    monkeypatch.setattr(remote.time, "monotonic", lambda: now[0])
    def stage_fn(*_):
        now[0] += 3.5
        return f"/stage/{BUNDLE_DIGEST}", BUNDLE_DIGEST
    def exec_fn(_ep, payload, _digest, _policy):
        seen.append(payload["deadline_s"])
        return {"protocol_version": remote.PROTOCOL_VERSION, "request_id": payload["request_id"], "effective_workers": 1, "results": []}
    _dispatcher(cwd, "remote", stage_fn=stage_fn, exec_fn=exec_fn).dispatch_batch("run_step", {}, [{}], deadline_s=10.0)
    assert seen == [pytest.approx(6.5)]
    stage, _ = _stage(bundle_source, tmp_path / "cache")
    native_root = remote._native_root(stage)
    remote._private_dir(native_root)
    (native_root / remote.MANIFEST).write_bytes(b"stale")
    from mindie_sim.prediction.runtime import accel_build, accel_loader
    rebuilt = []
    def fake_build(*, check_only=False):
        if check_only:
            return {"library_ready": False, "cache_path": str(native_root / "old.so")}
        so = accel_build.cache_root() / "lib.so"
        so.parent.mkdir(parents=True, exist_ok=True); so.write_bytes(b"ok"); rebuilt.append(so)
        return {"library_ready": True, "cache_path": str(so)}
    monkeypatch.setattr(accel_build, "build_accel", fake_build)
    monkeypatch.setattr(accel_loader, "load_accel_cdll", lambda status: status)
    worker._prepare_native(stage, build=True)
    assert len(rebuilt) == 1 and remote.verify_staged_manifest(native_root)


@POSIX_ONLY
@pytest.mark.tier1
def test_multiworker_staged_c11_matches_python_with_positive_native_steps(endpoint_paths, tmp_path, monkeypatch):
    _, cwd, _ = endpoint_paths
    case = sku_case(REPRESENTATIVE_SKUS["a2"])
    cands = [{"candidate_id": f"c{ctx}", "objective_key": 0, "request": LoweringRequest(step=StepRequest(phase="decode", context=ctx))} for ctx in (16, 32)]
    digest = remote.compute_bundle_digest(REPO_ROOT)
    stage = remote.stage_bundle_local_dir(remote.build_bundle_archive(REPO_ROOT), tmp_path / "c11_cache", digest, request_id=REQUEST_ID, cache_quota_bytes=CACHE_QUOTA_BYTES, build_c11=True)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "python")
    py_res, _ = _dispatcher(cwd, "local").dispatch_batch("run_step", case, cands, deadline_s=30)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "c")
    rem_res, rem_rec = remote.RemoteDispatcher(
        remote.RemoteExecPolicy(target="remote", authorized_aliases=(ALIAS,), max_remote_workers=2),
        cwd=cwd, endpoints_loader=lambda _: {ALIAS: ENDPOINT}, probe_fn=lambda *_: CAPACITY, stage_fn=lambda *_: (str(stage), digest),
        exec_fn=lambda _ep, payload, _d, pol: json.loads(remote._canonical(worker.execute_worker_request(remote._canonical(payload), max_result_bytes=pol.max_result_bytes))),
    ).dispatch_batch("run_step", case, cands, deadline_s=30)
    assert rem_rec.effective_workers == rem_res["effective_workers"] == 2
    assert all(abs(p["result"]["span_ms"] - c["result"]["span_ms"]) <= 1e-12 for p, c in zip(py_res["results"], rem_res["results"], strict=True))
    from mindie_sim.prediction.runtime.engine import Engine, Resource
    with worker._worker_environment(stage):
        worker._prepare_native(stage)
        spans, stats = {}, {}
        for backend in ("python", "c"):
            eng = Engine([Resource("r0", 10.0, units=2), Resource("r1", 5.0)], backend=backend)
            eng.advance(until=tuple(eng.submit(f"t{i}", {"r0": 20.0, "r1": 10.0}) for i in range(4)))
            spans[backend], stats[backend] = eng.now, eng.accel_status()
        assert stats["c"]["native_steps"] > 0 and stats["python"]["native_steps"] == 0
        assert Path(stats["c"]["cache_path"]).is_relative_to(remote._native_root(stage)) and abs(spans["c"] - spans["python"]) <= 1e-12
