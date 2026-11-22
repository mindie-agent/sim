"""Cold-cache C11 auto-build and transparent remote batch offload contracts.

[Role] Verify default-on cold-cache C11 compilation + graceful degradation when
  no compiler is present, ``execute_batch_with_auto_remote`` routing across
  ``off``/``local``/``auto``/``prefer_remote``/``require`` modes, local
  ``ProcessPoolExecutor`` fallback, ``batch_call`` module-whitelist enforcement,
  and strict ``HOME`` / ``MINDIE_CONFIG_HOME`` isolation.
[Invariants] Tier 1: every test isolates ``HOME``, ``USERPROFILE``,
  ``MINDIE_CONFIG_HOME``, ``MINDIE_CACHE_HOME``, ``MINDIE_DATA_HOME``, and
  ``TMPDIR`` under ``tmp_path``; never reads real user endpoints or leaks host
  identifiers.
[Timeouts & Locks] Bounded local process pool and mock SSH transport (< 15s).
[Side-Effect Policy] Writes exclusively inside ``tmp_path``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from mindie_sim.remote_exec import execute_worker_request
from mindie_sim.prediction.runtime import accel_build, accel_loader
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.remote_exec import (
    AUTO_REMOTE_BATCH_THRESHOLD,
    PROTOCOL_VERSION,
    RemoteDispatcher,
    RemoteExecPolicy,
    RemoteExecUnavailableError,
    _user_endpoint_files,
    execute_batch_with_auto_remote,
)

pytestmark = pytest.mark.tier1


def _isolate_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[Path, Path, Path]:
    home = root / "home"
    cfg = root / "config"
    cache = root / "cache"
    data = root / "data"
    tmp = root / "tmp"
    for directory in (home, cfg, cache, data, tmp):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(cfg))
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(cache))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(data))
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.delenv("MINDIE_SIM_ACCEL", raising=False)
    monkeypatch.delenv("MINDIE_SIM_ACCEL_CACHE", raising=False)
    return home, cfg, cache


def _write_isolated_endpoint(cfg: Path, alias: str = "cpu_box") -> str:
    cfg.mkdir(parents=True, exist_ok=True)
    ep_path = cfg / "endpoints.json"
    ep_payload = {
        alias: {
            "host": "sim-worker.example.invalid",
            "port": 2222,
            "user": "simuser",
            "root": "/srv/sim",
        }
    }
    ep_path.write_text(json.dumps(ep_payload), encoding="utf-8")
    if os.name == "posix":
        ep_path.chmod(0o600)
    from mindie_sim.remote_exec import _endpoint

    ep_id = _endpoint(ep_payload[alias]).endpoint_id
    pol_path = cfg / "sim_remote_policy.json"
    pol_path.write_text(
        json.dumps({"authorized_endpoints": [{"alias": alias, "endpoint_id": ep_id}]}),
        encoding="utf-8",
    )
    if os.name == "posix":
        pol_path.chmod(0o600)
    return ep_id


@pytest.mark.tier1
def test_cold_cache_auto_builds_native_accelerator_and_degrades_without_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Default auto mode compiles into isolated cache without manual build-accel and degrades cleanly."""
    _isolate_env(monkeypatch, tmp_path / "cold_ok")
    accel_build.reset_build_memo()
    accel_loader._LOADED.clear()

    if accel_build.discover_compiler()[0] is not None:
        eng = Engine([Resource("r", 100.0)], backend="auto", accel_eager_attach=True)
        try:
            for idx in range(32):
                eng.submit(f"t{idx}", work={"r": 10.0})
            eng.advance()
            st = eng.accel_status()
            assert st["backend"] == "c_native"
            assert st["native_steps"] > 0
            assert st["cache_path"] and str(tmp_path / "cold_ok" / "cache") in str(st["cache_path"])
        finally:
            if eng._accel is not None and eng._accel.arena is not None:
                eng._accel.arena.close()

    _isolate_env(monkeypatch, tmp_path / "no_cc")
    monkeypatch.setenv("MINDIE_SIM_CC", "nonexistent_cc_binary_xyz")
    accel_build.reset_build_memo()
    accel_loader._LOADED.clear()
    assert not accel_loader.toolchain_or_cache_ready()
    eng_py = Engine([Resource("r", 100.0)], backend="auto", accel_eager_attach=True)
    try:
        for idx in range(32):
            eng_py.submit(f"t{idx}", work={"r": 10.0})
        eng_py.advance()
        st_py = eng_py.accel_status()
        assert st_py["accel_backend"] == "python_fallback"
        assert st_py["native_steps"] == 0
        assert eng_py.serviced["r"] == pytest.approx(320.0)
    finally:
        if eng_py._accel is not None and eng_py._accel.arena is not None:
            eng_py._accel.arena.close()


@pytest.mark.tier1
def test_batch_auto_remote_local_modes_and_parallel_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """remote='off'/'local' execute ordered batches locally via ProcessPoolExecutor or inline closure."""
    _isolate_env(monkeypatch, tmp_path)

    empty_out, empty_receipt = execute_batch_with_auto_remote(
        lambda x: x, [], remote="off",
    )
    assert empty_out == []
    assert empty_receipt.resolved_target == "local"

    # Non-picklable local closure degrades cleanly to in-process execution with ordered outputs.
    closure_out, closure_receipt = execute_batch_with_auto_remote(
        lambda item: item["x"] * 2 + 1,
        [{"x": idx} for idx in range(6)],
        remote="off",
        max_workers=2,
    )
    assert closure_out == [1, 3, 5, 7, 9, 11]
    assert closure_receipt.resolved_target == "local"

    # Whitelisted mindie_sim callable executes cleanly via ProcessPoolExecutor.
    from mindie_sim.remote_exec import _sanitize_public_alias

    aliases = ["alpha_box", "beta_box", "192.0.2.1", "gamma_box"]
    pool_out, pool_receipt = execute_batch_with_auto_remote(
        _sanitize_public_alias,
        aliases,
        remote="local",
        max_workers=2,
    )
    assert pool_out == ["alpha_box", "beta_box", "redacted", "gamma_box"]
    assert pool_receipt.resolved_target == "local"
    assert pool_receipt.effective_workers >= 1


@pytest.mark.tier1
def test_batch_auto_remote_threshold_and_missing_local_compiler_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """remote='auto' offloads when batch >= threshold OR local C11 is unavailable, and honors warm local C11."""
    _home, cfg, _cache = _isolate_env(monkeypatch, tmp_path)
    _write_isolated_endpoint(cfg, alias="cpu_farm")

    remote_calls: list[dict[str, Any]] = []

    def fake_probe(_ep, _pol):
        return {
            "affinity_cpus": 64,
            "cgroup_cpus": None,
            "loadavg_1m": 1.0,
            "mem_avail_mib": 131072,
            "cgroup_mem_remaining_mib": None,
            "compiler": True,
        }

    def fake_stage(_ep, _req_id, _pol):
        return "/tmp/staged/" + ("a" * 64), "a" * 64

    def fake_exec(_ep, payload, _digest, _pol):
        remote_calls.append(dict(payload))
        sim_payload = payload | {"stage_root": None, "primitive_manifest_path": None}
        return execute_worker_request(sim_payload, max_result_bytes=1024 * 1024)

    from mindie_sim.remote_exec import _sanitize_public_alias

    dispatcher_warm = RemoteDispatcher(
        RemoteExecPolicy(target="auto", authorized_aliases=("cpu_farm",)),
        probe_fn=fake_probe,
        stage_fn=fake_stage,
        exec_fn=fake_exec,
        local_c11_ready_fn=lambda: True,
        cwd=tmp_path,
    )
    small_batch = [f"node_{i}" for i in range(AUTO_REMOTE_BATCH_THRESHOLD - 1)]
    out_small, rec_small = execute_batch_with_auto_remote(
        _sanitize_public_alias, small_batch, remote="auto", dispatcher=dispatcher_warm,
    )
    assert out_small == small_batch
    assert rec_small.resolved_target == "local"
    assert rec_small.fallback_reason == "warm_local_c11"
    assert len(remote_calls) == 0

    large_batch = [f"node_{i}" for i in range(AUTO_REMOTE_BATCH_THRESHOLD)]
    out_large, rec_large = execute_batch_with_auto_remote(
        _sanitize_public_alias, large_batch, remote="auto", dispatcher=dispatcher_warm,
    )
    assert out_large == large_batch
    assert rec_large.resolved_target == "remote"
    assert rec_large.alias == "cpu_farm"
    assert len(remote_calls) == 1
    assert remote_calls[0]["entrypoint"] == "batch_call"

    dispatcher_cold = RemoteDispatcher(
        RemoteExecPolicy(target="auto", authorized_aliases=("cpu_farm",)),
        probe_fn=fake_probe,
        stage_fn=fake_stage,
        exec_fn=fake_exec,
        local_c11_ready_fn=lambda: False,
        cwd=tmp_path,
    )
    out_cold, rec_cold = execute_batch_with_auto_remote(
        _sanitize_public_alias, ["a", "b"], remote="auto", dispatcher=dispatcher_cold,
    )
    assert out_cold == ["a", "b"]
    assert rec_cold.resolved_target == "remote"
    assert len(remote_calls) == 2

    # prefer_remote degrades cleanly to local when remote probe fails.
    def failing_probe(_ep, _pol):
        raise OSError("ssh_unreachable")

    dispatcher_down = RemoteDispatcher(
        RemoteExecPolicy(target="prefer_remote", authorized_aliases=("cpu_farm",)),
        probe_fn=failing_probe,
        stage_fn=fake_stage,
        exec_fn=fake_exec,
        local_c11_ready_fn=lambda: True,
        cwd=tmp_path,
    )
    out_down, rec_down = execute_batch_with_auto_remote(
        _sanitize_public_alias, ["x", "y"], remote="prefer_remote", dispatcher=dispatcher_down,
    )
    assert out_down == ["x", "y"]
    assert rec_down.resolved_target == "local"
    assert rec_down.fallback_reason == "probe_unavailable"


@pytest.mark.tier1
def test_batch_require_remote_raises_and_enforces_callable_whitelist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """remote='require' never swallows missing endpoints; batch_call blocks non-mindie_sim modules."""
    _isolate_env(monkeypatch, tmp_path)
    from mindie_sim.remote_exec import _sanitize_public_alias

    with pytest.raises(RemoteExecUnavailableError):
        execute_batch_with_auto_remote(
            _sanitize_public_alias, ["item0"], remote="require", cwd=tmp_path,
        )

    bad_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "a" * 32,
        "deadline_s": 10.0,
        "entrypoint": "batch_call",
        "base_case": {},
        "candidates": [
            {
                "candidate_id": "0",
                "objective_key": 0,
                "module": "os",
                "qualname": "system",
                "payload": "echo bad",
            }
        ],
        "stage_root": None,
        "max_workers": 1,
        "per_worker_mem_mib": 256,
        "primitive_manifest_path": None,
    }
    with pytest.raises(ValueError, match="unauthorized_batch_module"):
        execute_worker_request(bad_payload, max_result_bytes=1024 * 1024)


@pytest.mark.tier1
def test_mindie_config_home_isolation_reads_only_isolated_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """_user_endpoint_files honors MINDIE_CONFIG_HOME and stays completely inside tmp_path."""
    home, cfg, _cache = _isolate_env(monkeypatch, tmp_path)
    _write_isolated_endpoint(cfg, alias="isolated_host")

    files = _user_endpoint_files()
    assert cfg / "endpoints.json" in files
    assert home / ".mindie" / "endpoints.json" in files
    assert all(str(tmp_path) in str(path) for path in files)

    dispatcher = RemoteDispatcher(
        RemoteExecPolicy(target="auto"),
        local_c11_ready_fn=lambda: False,
        probe_fn=lambda _ep, _pol: {
            "affinity_cpus": 16,
            "cgroup_cpus": None,
            "loadavg_1m": 0.0,
            "mem_avail_mib": 32768,
            "cgroup_mem_remaining_mib": None,
            "compiler": True,
        },
        cwd=tmp_path,
    )
    receipt = dispatcher.resolve_target(1)
    assert receipt.resolved_target == "remote"
    assert receipt.alias == "isolated_host"
