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

from dataclasses import replace
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
from mindie_sim import api, remote_exec as remote, remote_exec as worker, rewrite
from mindie_sim.optimization.contracts import PrimitiveCapacityEntry, PrimitiveCapacityTable
from mindie_sim.prediction.results import ReplayOptions, StepRequest, json_payload
from mindie_sim.prediction.spine import default_spine
from mindie_sim.primitives.loader import load_primitives
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


def _unexpected(*args, **kwargs): pytest.fail("unexpected endpoint I/O, remote submission, or local replay")


def _wire(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_config(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_wire(data))
    path.chmod(mode)


@pytest.fixture
def endpoint_paths(tmp_path, monkeypatch):
    home, cwd, env_file = (tmp_path / name for name in ("home", "repo", "override.json"))
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("USERPROFILE", str(home)); monkeypatch.setenv("REMOTE_DEV_ENDPOINTS_FILE", str(env_file))
    return home, cwd, env_file


def _dispatcher(cwd, target, *, warm=False, endpoints=None, probe=None, **kwargs):
    return remote.RemoteDispatcher(
        remote.RemoteExecPolicy(target=target, authorized_aliases=(ALIAS,), max_remote_workers=1),
        cwd=cwd, local_c11_ready_fn=lambda: warm,
        endpoints_loader=endpoints if endpoints is not None else lambda _: {ALIAS: ENDPOINT},
        probe_fn=probe if probe is not None else lambda *_: CAPACITY, **kwargs,
    )


@pytest.mark.parametrize("variant", ["capacity_table", "primitive_table", "three_tiers"])
def test_local_dispatch_matches_public_prediction(tmp_path, monkeypatch, prediction_case, variant):
    monkeypatch.setattr(remote, "_read_config", _unexpected)
    monkeypatch.setattr(remote, "_exchange", _unexpected)
    dispatcher = _dispatcher(tmp_path, "local", endpoints=_unexpected, probe=_unexpected, stage_fn=_unexpected, exec_fn=_unexpected)
    step = StepRequest(phase="decode", context=16, sequences=(2,))
    replay = ReplayOptions(include_submission=True)
    entrypoint = "evaluate_three_tiers" if variant == "three_tiers" else "optimize_ladder"
    options = {}
    if variant == "capacity_table":
        die = prediction_case.machine.dies[0]
        rate = dict(die.rates)["hbm"]
        table = PrimitiveCapacityTable(die.sku, (PrimitiveCapacityEntry("hbm_bus_r", rate, rate, "B/s"),))
        options = dict(max_trials=1, stack_id="raw_eager", primitive_table=table, primitives=None)
    elif variant == "primitive_table":
        table = load_primitives("Ascend950DT_9582")
        prediction_case = sku_case(table.sku)
        step = replace(step, phase="prefill", context=0, tokens=1024)
        options = dict(max_trials=0, primitive_table=table)
    expected = json_payload(getattr(api, entrypoint)(prediction_case, step, replay=replay, **options))
    predict = Mock(wraps=getattr(spine := default_spine(), entrypoint))
    monkeypatch.setattr(spine, entrypoint, predict)
    if entrypoint == "optimize_ladder":
        result, receipt = dispatcher.optimize_ladder(prediction_case, [dict(request=step, replay=replay, options=options)])
    else:
        result, receipt = dispatcher.evaluate_three_tiers(prediction_case, step, replay=replay)
    assert receipt.resolved_target == "local" and receipt.fallback_reason is None
    deadline = predict.call_args.kwargs["replay"].deadline_mono
    predict.assert_called_once_with(prediction_case, step, replay=replace(replay, deadline_mono=deadline), **options)
    actual = result["results"][0]["result"]
    if entrypoint == "optimize_ladder":
        for key in ("framework_plan", "dsl_plan"):  # Public plan wire, so a remote winner feeds finalize_plan.
            actual[key] = json_payload(rewrite.decode_rewrite_plan(actual[key]))
        assert actual["verification"].pop("passed") is all(gate["passed"] for gate in actual["verification"]["gates"])
        for row in (actual, expected): del row["dual_subgradient"]["solve_ms"]
    assert actual == expected
    if variant == "primitive_table":
        assert predict.call_args.kwargs["primitive_table"].to_envelope_dict() == table.to_envelope_dict()
        assert all(gate["passed"] for gate in actual["verification"]["gates"])
        assert 0 < actual["ladder"]["l_star_hw_ns"] <= actual["ladder"]["l_star_lean_ns"] <= actual["ladder"]["a_star_dsl_ns"]


@pytest.mark.parametrize("target,warm,count,expected,reason", [
    ("auto", True, 1, "local", "warm_local_c11"),
    ("auto", True, 3, "local", "warm_local_c11"),
    ("auto", True, 4, "remote", None),
    ("auto", True, 5, "remote", None),
    ("auto", False, 1, "remote", None),
    ("prefer_remote", True, 1, "remote", None),
])
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


@pytest.mark.parametrize("spoof", [{"host": "other.example.invalid"}, {"port": 2222}], ids=["host", "port"])
def test_unattended_dispatch_requires_exact_endpoint_pin(endpoint_paths, spoof):
    home, cwd, _ = endpoint_paths
    endpoint = remote._endpoint(ENDPOINT)
    _write_config(cwd / "endpoints.json", {ALIAS: ENDPOINT})
    probe = Mock(return_value=CAPACITY)
    dispatcher = remote.RemoteDispatcher(cwd=cwd, local_c11_ready_fn=lambda: False,
                                         probe_fn=probe, stage_fn=_unexpected, exec_fn=_unexpected)
    receipt = dispatcher.resolve_target(1)
    assert receipt.resolved_target == "local" and receipt.fallback_reason == "no_authorized_endpoint"
    probe.assert_not_called()
    _write_pin(home, endpoint)
    _write_config(home / ".mindie/endpoints.json", {ALIAS: ENDPOINT})
    assert dispatcher.resolve_target(1).endpoint_id == endpoint.endpoint_id
    probe.reset_mock()
    replacement = ENDPOINT | spoof
    assert remote._endpoint(replacement).endpoint_id != endpoint.endpoint_id
    _write_config(cwd / "endpoints.json", {ALIAS: replacement})
    result, receipt = dispatcher.dispatch_batch("run_step", {}, [{}], local_runner=lambda _: "local")
    assert result == "local" and receipt.resolved_target == "local"
    assert receipt.endpoint_id is None
    probe.assert_not_called()


@POSIX_ONLY
@pytest.mark.parametrize("mode", [0o666, 0o620, 0o602, 0o644, 0o700, 0o600])
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
    stage = remote.stage_bundle_local_dir(remote.build_bundle_archive(source), cache, digest, request_id=request_id,
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


@pytest.mark.parametrize("field,embed", [("host", False), ("host", True), ("user", False), ("user", True), ("auth_ref", False)])
def test_endpoint_alias_cannot_leak_actual_connection_identity(endpoint_paths, field, embed):
    home, cwd, _ = endpoint_paths
    config = ENDPOINT | {"auth_ref": "private-login", "identity_file": str(home / "private-key")}
    if embed:
        config[field] = "Compute.Example.Invalid" if field == "host" else "Operator"
    alias = f"lab-{config[field].upper()}" if embed else config[field]
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
        candidates.append(dict(candidate_id=candidate_id, objective_key=int(candidate_id == "beta"), request=request))

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
    assert local_result["results"] == remote_result["results"]
    assert [(row["objective_key"], row["candidate_id"]) for row in local_result["results"]] == expected_order
    for row in local_result["results"]:
        result = row["result"]
        steps = (result,) if entrypoint == "run_step" else (result[tier] for tier in ("unmerge", "mainline", "should_achieve"))
        for step in steps:
            assert step["request"]["context"] == contexts[row["candidate_id"]]
            assert step["native_steps"] == 0 and step["span_ms"] > 0 and step["operators"]
            assert [op["ordinal"] for op in step["operators"]] == list(range(len(step["operators"])))


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
    command = [sys.executable, "-B", "-m", "mindie_sim.remote_exec"]
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
        "from mindie_sim.remote_exec import main\n"
        "raise SystemExit(main(ready=True))\n"
    )]
    assert remote._exchange(endpoint, ssh_command, request, JOB_TIMEOUT_S, handshake=True) == process.stdout


def test_timeout_preserved_remaining_deadline_sent_and_c11_rebuilds_on_miss(endpoint_paths, bundle_source, tmp_path, monkeypatch):
    _, cwd, _ = endpoint_paths
    with pytest.raises(TimeoutError):
        _dispatcher(cwd, "auto", warm=True, probe=Mock(side_effect=TimeoutError("probe"))).resolve_target(4)
    now, caps, seen = [100.0], [], []
    monkeypatch.setattr(remote.time, "monotonic", lambda: now[0])
    def probe_fn(_ep, pol): caps.append(("probe", pol.probe_timeout_s)); now[0] += 0.5; return CAPACITY
    def stage_fn(_ep, _rid, pol): caps.append(("stage", pol.stage_timeout_s)); now[0] += 1.0; return f"/stage/{BUNDLE_DIGEST}", BUNDLE_DIGEST
    def exec_fn(_ep, p, _d, _pol):
        seen.append(p["deadline_s"])
        return {"protocol_version": remote.PROTOCOL_VERSION, "request_id": p["request_id"], "effective_workers": 1, "results": []}
    _dispatcher(cwd, "remote", probe=probe_fn, stage_fn=stage_fn, exec_fn=exec_fn).dispatch_batch("run_step", {}, [{}], deadline_s=4.0)
    assert caps == [("probe", pytest.approx(4.0)), ("stage", pytest.approx(3.5))] and seen == [pytest.approx(2.5)]
    with pytest.raises(TimeoutError, match="deadline_exceeded"):
        _dispatcher(cwd, "remote", stage_fn=stage_fn, exec_fn=lambda _e, p, _d, _pol: {"protocol_version": remote.PROTOCOL_VERSION, "request_id": p["request_id"], "error": "deadline_exceeded"}).dispatch_batch("run_step", {}, [{}], deadline_s=4.0)
    stage, _ = _stage(bundle_source, tmp_path / "cache")
    native_root = remote._native_root(stage)
    remote._private_dir(native_root)
    prev_so = native_root / "prev.so"; prev_so.write_bytes(b"prev")
    (native_root / remote.MANIFEST).write_bytes(remote._manifest(native_root, [prev_so]))
    from mindie_sim.prediction.runtime import accel_build, accel_loader
    rebuilt, fail_next = [], [True]
    def fake_build(*, check_only=False):
        if check_only: return {"library_ready": False, "cache_path": str(native_root / "old.so")}
        if fail_next[0]: raise RuntimeError("compile_failed")
        so = accel_build.cache_root() / "lib.so"; so.parent.mkdir(parents=True, exist_ok=True); so.write_bytes(b"ok"); rebuilt.append(so)
        return {"library_ready": True, "cache_path": str(so)}
    monkeypatch.setattr(accel_build, "build_accel", fake_build)
    monkeypatch.setattr(accel_loader, "load_accel_cdll", lambda status: status)
    with pytest.raises(RuntimeError, match="compile_failed"): worker._prepare_native(stage, build=True)
    assert prev_so.read_bytes() == b"prev" and remote.verify_staged_manifest(native_root)
    fail_next[0] = False; worker._prepare_native(stage, build=True)
    assert len(rebuilt) == 1 and not prev_so.exists() and remote.verify_staged_manifest(native_root)
    (stage / "data/primitives/manifest.json").write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(worker, "_prepare_native", lambda *_: None)
    with worker._worker_environment(stage), pytest.raises(Exception): worker._init_worker(str(stage), None)


@POSIX_ONLY
@pytest.mark.tier1
def test_multiworker_staged_c11_matches_python_with_positive_native_steps(endpoint_paths, tmp_path, monkeypatch):
    from _support.hardware import MODEL, WORKLOAD
    from mindie_sim.schema.case import Case
    from mindie_sim.stack_registry import RankOps
    _, cwd, _ = endpoint_paths
    case = Case.from_mapping({
        "machine": {"backend": "ascend", "sku": REPRESENTATIVE_SKUS["a5"], "num_devices": 2},
        "model": {**MODEL, "layers": 1, "hidden_size": 64, "num_heads": 2, "num_kv_heads": 2, "head_dim": 32,
                  "intermediate_size": 128, "ffn": "moe", "num_experts": 4, "top_k": 2, "moe_intermediate_size": 64},
        "workload": WORKLOAD, "runtime": {"dp": 2, "ep": 2, "tp": 1, "submission": {"scheduler_ms": 0.5}},
    })
    op0 = default_spine().lower(case, LoweringRequest(step=StepRequest(phase="decode", context=16))).graph.executable[0].ops[1]
    cands = [{"candidate_id": f"c{n}", "objective_key": 0, "replay": ReplayOptions(include_submission=True),
              "request": LoweringRequest(declared_ops=(RankOps(0, (replace(op0, op_id=f"g{n}", deps=()),)),
                                                       RankOps(1, (replace(op0, op_id=f"g{n}", deps=()),))))} for n in (1, 2)]
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
    assert rem_rec.effective_workers == rem_res["effective_workers"] == 2 and any(p.suffix in (".so", ".dylib") for p in remote._native_root(stage).rglob("*"))
    assert all(p["result"]["native_steps"] == 0 and c["result"]["native_steps"] > 0 and abs(p["result"]["span_ms"] - c["result"]["span_ms"]) <= 1e-12
               for p, c in zip(py_res["results"], rem_res["results"], strict=True))
