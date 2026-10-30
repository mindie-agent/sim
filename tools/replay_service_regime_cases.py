"""Replay complete observable-service cases under per-case process budgets.

[Role] Execute declared Body geometries and retain resource/parameter results.
[Invariants] Observed Body spans are comparison-only inputs, never parameters.
  Execution failures remain failures; a source change during a run is visible.
[Timeouts & Locks] Sequential children, default 60 seconds each; no remote I/O.
[Side-Effect Policy] Read local inputs and write a compact requested report.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_case(item):
    from mindie_sim import predict
    from tools.deployment_generalization import package_identity
    started = time.perf_counter()
    identity = package_identity()
    path = Path(item['case'])
    if not path.is_absolute():
        path = ROOT/path
    raw = json.loads(path.read_text())
    geometry = item['geometry']
    result = predict(raw, scope='body', phase='decode', **geometry)
    observed = item.get('observed_body_ms')
    execution = result['execution']
    return {'case': str(path), 'case_sha256': sha256(path.read_bytes()).hexdigest(),
            'entrypoint': 'predict(scope=body, phase=decode)',
            'geometry': geometry, 'package_start': identity, 'package_end': package_identity(),
            'body_forward_ms': result['duration_ms'], 'observed_body_ms': observed,
            'relative_error': result['duration_ms']/observed-1 if observed else None,
            'signed_gap_ms': result['duration_ms']-observed if observed else None,
            'parameters': result['parameters'],
            'execution': {key: execution[key] for key in ('work', 'serviced', 'live_tasks',
                'live_programs', 'live_allocated', 'peak_live_tasks', 'completed_tasks') if key in execution},
            'wall_seconds': time.perf_counter()-started,
            'peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--child', type=int)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.child is not None:
        print(json.dumps(run_case(manifest['cases'][args.child])))
        return
    rows = []
    for index, case in enumerate(manifest['cases']):
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                str(args.manifest.resolve()), '--child', str(index)],
                capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            row = {'status': 'timeout', 'case': case['case'], 'timeout_seconds': args.timeout}
        else:
            row = ({'status': 'completed', **json.loads(run.stdout)} if run.returncode == 0 else
                   {'status': 'failed', 'case': case['case'], 'returncode': run.returncode,
                    'stderr': run.stderr[-4000:]})
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('case', 'status', 'body_forward_ms', 'relative_error', 'wall_seconds') if k in row}), flush=True)
    packet = {'schema': 'mindie-sim.observable-service-body-replay/1',
              'manifest': str(args.manifest.resolve()),
              'manifest_sha256': sha256(args.manifest.read_bytes()).hexdigest(), 'rows': rows}
    if args.output:
        args.output.write_text(json.dumps(packet, indent=2)+'\n')
    if any(row['status'] != 'completed' for row in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
