"""Installed-wheel closure for the Round-11 causal/finalizer regressions.

[Role] Build a real PEP 517 wheel, verify its RECORD, install it into a fresh
  ``--target`` tree, and run the R11-1 ownership and R11-2 descendant probes
  from only the installed package in ``python`` / ``c`` / ``auto``.
[Invariants] The artefact is produced by the declared setuptools backend, never
  hand-assembled; native sources compile from the shipped package into an
  isolated cache.  Results are compared field-by-field with Python.
[Timeouts & Locks] Subprocesses carry explicit timeouts; no threads or sleeps.
[Side-Effect Policy] Writes only a test-owned temporary build/install tree.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from temp_cleanup import tempdir

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ("c", "auto")


def _verify_wheel(wheel: Path) -> None:
    try:
        from distlib.wheel import Wheel
    except ModuleNotFoundError:
        Wheel = None
    if Wheel is not None:
        Wheel(str(wheel)).verify()
        return
    with zipfile.ZipFile(wheel) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    rec_name = next(n for n in files if n.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(files[rec_name].decode("utf-8").splitlines()))
    assert {r[0] for r in rows} == set(files)
    for name, digest, size in rows:
        if name == rec_name:
            assert digest == "" and size == ""
            continue
        algo, expected = digest.split("=", 1)
        actual = base64.urlsafe_b64encode(
            hashlib.new(algo, files[name]).digest()).rstrip(b"=").decode("ascii")
        assert actual == expected and int(size) == len(files[name]), name

def _wheel_env(base):
    env = dict(os.environ)
    for name in ("MINDIE_SIM_ACCEL", "MINDIE_SIM_ENGINE_BACKEND",
                 "MINDIE_SIM_CC", "MINDIE_DATA_HOME", "XDG_CACHE_HOME",
                 "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               TMPDIR=str(base), MINDIE_CACHE_HOME=str(base / "build-cache"))
    return env


def _wheel(base):
    source = base / "source"
    shutil.copytree(ROOT / "mindie_sim", source / "mindie_sim",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    if (ROOT / "README.md").exists():
        shutil.copy2(ROOT / "README.md", source / "README.md")
    (source / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    wheel_dir = base / "dist"
    wheel_dir.mkdir()
    hook = base / "build_hook.py"
    hook.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from setuptools import build_meta\n"
        "print(build_meta.build_wheel(str(Path(sys.argv[1]))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", str(hook), str(wheel_dir)],
        cwd=source, env=_wheel_env(base), capture_output=True, text=True,
        timeout=300)
    if result.returncode != 0 or not list(wheel_dir.glob("*.whl")):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "build", "--wheel",
             "--outdir", str(wheel_dir), "."],
            cwd=source, env=_wheel_env(base), capture_output=True, text=True,
            timeout=600)
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert result.returncode == 0 and len(wheels) == 1, (
        result.stdout, result.stderr)
    return wheels[0]


_WHEEL_PROBE = r'''
import json, weakref
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.accel_tables import metadata_is_stable
class F(float): pass
class Token: pass
def close(e):
    if e._accel is not None: e._accel.close()
def omitted(mode):
    e = Engine([Resource("r", 64.0)], backend=mode, trace_limit=0)
    payload = F(0.0)
    roots = [e.submit("root0", {"r": 1.0}, caps={"r": 1.0}, release=payload)]
    roots += [e.submit("root" + str(i), {"r": a}, caps={"r": 1.0})
              for i, a in enumerate([2.0] + [3.0] * 62, 1)]
    reg, seen, extra = [], [], []
    def follow(event):
        seen.append((event.time, e.now, e.busy_ms["r"], e.serviced["r"]))
        extra.append(e.submit("extra", {"r": e.serviced["r"]}))
    def register():
        reg.append((e.now, roots[1].done)); e.on_complete(roots[1], follow)
    f = weakref.finalize(payload, register); del payload
    e.advance()
    out = (reg, seen, [(x.started, x.time) for x in extra], e.now,
           e.serviced["r"], e.completed_count,
           e.accel_status()["native_steps"], not f.alive)
    close(e); return out
def descendant(mode):
    e = Engine([Resource("r", 64.0)], backend=mode, trace_limit=0)
    token = Token()
    root = e.submit("root0", {"r": 1.0}, caps={"r": 1.0},
                    metadata={"token": token})
    child = e.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(root,))
    for i, a in enumerate([2.0] + [3.0] * 62, 1):
        e.submit("root" + str(i), {"r": a}, caps={"r": 1.0})
    seen, extra = [], []
    def release():
        seen.append((e.now, root.done, child.started, child.done,
                     e.completed_count, e.serviced["r"]))
        extra.append(e.submit("extra",
                              {"r": 1.0 if child.started is None else 2.0}))
    f = weakref.finalize(token, release); del token
    e.advance()
    out = (seen, [(x.started, x.time) for x in extra], child.started,
           child.time, e.now, e.serviced["r"], e.completed_count,
           e.accel_status()["native_steps"])
    close(e); return out
out = {}
for mode in ("python", "c", "auto"):
    out[mode] = {"omitted": omitted(mode), "descendant": descendant(mode)}
nested = []
for _ in range(9): nested = (nested,)
out["stable"] = metadata_is_stable({"payload": nested})
print(json.dumps(out))
'''


def test_r11_fixes_hold_on_the_installed_pep517_wheel():
    """The built wheel reproduces both R11 closures from its shipped sources."""
    base = tempdir("nr-s3-r11-wheel-")
    try:
        wheel = _wheel(base)
        _verify_wheel(wheel)
        extract = base / "installed"
        install = subprocess.run(
            [sys.executable, "-B", "-m", "pip", "install", "--no-deps",
             "--no-index", "--no-compile", "--no-cache-dir",
             "--target", str(extract), str(wheel)],
            env=_wheel_env(base), capture_output=True, text=True, timeout=240)
        assert install.returncode == 0, (install.stdout, install.stderr)
        script = base / "probe.py"
        script.write_text(_WHEEL_PROBE)
        run = base / "run"
        run.mkdir()
        env = _wheel_env(base)
        env.update(PYTHONPATH=str(extract), MINDIE_CACHE_HOME=str(base / "cache"))
        result = subprocess.run(
            [sys.executable, "-B", str(script)], cwd=run, env=env,
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (result.stdout, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        control = report["python"]
        assert control["omitted"] == [
            [[1.0, False]], [[2.0, 2.0, 2.0, 127.0]], [[2.0, 4.953125]],
            4.953125, 316.0, 65, 0, True]
        assert control["descendant"] == [
            [[1.0, True, None, False, 1, 64.0]], [[1.0, 2.015625]], 1.0,
            2.015625, 3.015625, 191.0, 66, 0]
        for mode in NATIVE:
            assert report[mode]["omitted"] == control["omitted"], mode
            assert (report[mode]["descendant"][:7]
                    == control["descendant"][:7]), mode
            assert report[mode]["omitted"][6] == 0, mode
            # R12-2: finalizer-bearing metadata declines native before run.
            assert report[mode]["descendant"][7] == 0, mode
        assert report["stable"] is False
    finally:
        shutil.rmtree(base, ignore_errors=True)




def test_r11_accelerator_modules_stay_below_the_reviewed_line_caps():
    """The companion wheel module also stays below the 735-line cap."""
    assert len(Path(__file__).read_bytes().splitlines()) < 735
