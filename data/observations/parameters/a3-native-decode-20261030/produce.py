"""Reproduce bounded native operator calibration from independent short contexts.

[Role] Stream native 4K operator observations and evaluate the separate 16K set.
[Invariants] No Body/step/serving duration enters a response. Unknown captured
  tensor fields are paired only by an explicit caller/consumer ABI condition.
  Original experiment roles remain unassigned; these analysis splits are new
  and historically inspected, not pristine holdouts. Domains are not extended
  to long-context observations or unseen row counts.
[Timeouts & Locks] Sequential local CSV reads; at most 4096 timing samples per
  physical group plus scalar counts. Source hashes stream 1 MiB chunks.
[Side-Effect Policy] Read external raw captures; replace two local evidence JSONs.
"""
import csv, json, re, hashlib, statistics
from pathlib import Path
from collections import defaultdict
root = Path(__file__).resolve().parents[4]
base = Path('/Volumes/data/mindie-sim-data/a3/2026-10-23')

def shape(s):
    try:
        return [tuple((int(v) for v in x.split(','))) if x else () for x in s.strip('"').split(';')]
    except ValueError:
        return []
ABI_FIELDS = ('Type', 'Input Shapes', 'Input Data Types', 'Input Formats',
              'Output Shapes', 'Output Data Types', 'Output Formats', 'Block Num', 'Mix Block Num', 'Accelerator Core')

def signature(row):
    return tuple(row.get(k, '') for k in ABI_FIELDS)

records = []
for family, q, width, heads in [('qwen35-gdn', 2, 2048, 8), ('qwen36-rows', 4, 2560, 12)]:
    for path in sorted((base / family / 'prof').glob('p_*_c*/**/kernel_details.csv')):
        scenario = next((p for p in path.parts if re.fullmatch('p_[0-9]+_c[0-9]+', p)))
        ctx = int(scenario.split('_')[1])
        if ctx not in (4096, 16384):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        groups = defaultdict(list)
        counts = defaultdict(int)
        first = {}
        active = None
        pending_by_stream = {}
        with path.open() as f:
            for line, row in enumerate(csv.DictReader(f), 2):
                typ = row['Type']
                ss = shape(row.get('Input Shapes', ''))
                stream = row['Stream ID']
                pending_prefix = pending_by_stream.pop(stream, None)
                if typ == 'CausalConv1d':
                    pending_by_stream.clear()
                    pending_prefix = None
                    active = ss[0][0] if ss and len(ss[0]) == 2 and (ss[0][1] == width) and (ss[0][0] <= q * 64) and shape(row['Output Shapes']) == [ss[0]] else None
                if active is None:
                    continue
                if typ == 'Cumsum':
                    pending_by_stream[stream] = (line, row)
                    continue
                if typ.startswith('GroupedMatmulSwigluQuant') and ss and (ss[-1] == (256,)) and pending_prefix:
                    pl, pr = pending_prefix
                    key = ('count_prefix', 256, signature(pr) + signature(row))
                    counts[key] += 1
                    first.setdefault(key, {'line': pl, 'type': 'Cumsum', 'input': pr.get('Input Shapes'), 'dtype': pr.get('Input Data Types'), 'block': pr.get('Block Num'), 'shape_basis': 'immediate same-stream native GMM count-prefix consumer + source conversion ABI', 'consumer_line': line, 'consumer_input': row['Input Shapes']})
                    if len(groups[key]) < 4096:
                        groups[key].append(float(pr['Duration(us)']))
                    pending_prefix = None
                pending_prefix = None
                op = None
                coord = active
                if typ in ('MatMulV2', 'MatMulV3') and len(ss) == 2 and (len(ss[0]) == len(ss[1]) == 2) and (ss[0][1] == ss[1][1]) and (row['Input Data Types'] == 'DT_BF16;DT_BF16'):
                    op = f'bf16_matmul_{ss[1][0]}_{ss[1][1]}'
                    coord = ss[0][0]
                elif typ == 'CausalConv1d':
                    op = 'conv'
                elif typ == 'RecurrentGatedDeltaRule' and ss and (ss[0] == (active, 4, 128)):
                    op = 'recurrent'
                elif typ == 'fused_gdn_gating_kernel' and (not ss or ss == [(heads,), (active, heads), (active, heads)]):
                    op = 'gating'
                elif typ == '_layer_norm_fwd_1pass_kernel_npu' and (not ss or ss == [(active * heads, 128), (128,), (active * heads, 128)]):
                    op = 'norm_gate'
                elif typ.startswith('GroupedMatmulSwigluQuant') and ss and (ss[0][1:] == (2048,)) and row['Input Data Types'].startswith('INT8;'):
                    op = 'gate_up'
                    coord = ss[0][0]
                elif typ == 'GroupedMatmul' and ss and (ss[0][1:] == (128,)) and row['Input Data Types'].startswith('INT8;'):
                    op = 'expert_down'
                    coord = ss[0][0]
                elif typ == 'ForeachAddList' and len(ss) >= 2 and (ss[0] == ss[1]) and (ss[0][1:] == (2048,)):
                    op = 'expert_add'
                    coord = ss[0][0]
                elif typ in ('MoeInitRoutingV2', 'MoeInitRoutingV3') and ss and (ss[0][1:] == (2048,)) and row['Output Data Types'].startswith('INT8;'):
                    op = 'route_pack'
                    coord = ss[0][0]
                if op is None:
                    continue
                if op in {'gate_up', 'expert_down'} and coord != active * 8:
                    continue
                if (op.startswith('bf16_matmul_') or op in {'route_pack', 'expert_add'}) and coord != active:
                    continue
                key = (op, coord, signature(row))
                counts[key] += 1
                first.setdefault(key, {'line': line, 'type': typ, 'input': row.get('Input Shapes'), 'dtype': row.get('Input Data Types'), 'block': row.get('Block Num')})
                if len(groups[key]) < 4096:
                    groups[key].append(float(row['Duration(us)']))
        for (op, coord, abi), vals in groups.items():
            records.append({'capture': str(path), 'sha256': digest.hexdigest(), 'scenario': scenario, 'split': 'operator_calibration' if ctx == 4096 else 'independent_context_validation', 'op': op, 'width': width, 'heads': heads, 'q': q, 'coordinate': coord, 'median_us': statistics.median(vals), 'min_us': min(vals), 'original_split': 'unassigned', 'retention': 'first_4096_events_per_capture_physical_group', 'samples_retained': len(vals), 'native_abi': dict(zip(ABI_FIELDS, abi[:len(ABI_FIELDS)])), 'consumer_abi': dict(zip(ABI_FIELDS, abi[len(ABI_FIELDS):])), 'total_count': counts[op, coord, abi], 'first': first[op, coord, abi]})
        print(family, scenario, len(groups), flush=True)

def fit(points):
    if len({x for x, y in points}) == 1:
        return (statistics.median((y for x, y in points)), 0.0)
    n = len(points)
    mx = sum((x for x, y in points)) / n
    my = sum((y for x, y in points)) / n
    slope = max(0, sum(((x - mx) * (y - my) for x, y in points)) / sum(((x - mx) ** 2 for x, y in points)))
    fixed = max(0, my - slope * mx)
    if fixed == 0:
        slope = sum((x * y for x, y in points)) / sum((x * x for x, y in points))
    return (fixed, slope)
rules = []
checks = []
for width, heads in [(2048, 8), (2560, 12)]:
    for op in ['conv', 'recurrent', 'gating', 'norm_gate', 'count_prefix', 'gate_up', 'expert_down', 'route_pack', 'expert_add'] + sorted({r['op'] for r in records if r['width'] == width and r['op'].startswith('bf16_matmul_')}):
        rows = [r for r in records if r['width'] == width and r['op'] == op]
        tr = [r for r in rows if r['split'] == 'operator_calibration']
        if not tr:
            continue
        if op in ('count_prefix', 'gate_up', 'expert_down', 'route_pack', 'expert_add') and width != 2048:
            continue
        fixed, slope = fit([(r['coordinate'], r['median_us']) for r in tr])
        lo = min((r['coordinate'] for r in tr))
        hi = max((r['coordinate'] for r in tr))
        recipe = {'conv': 'ascend_causal_conv1d_update', 'recurrent': 'decode_recurrent_scan', 'gating': 'ascend_gdn_decode_gating', 'norm_gate': 'linear_output_norm_gate', 'count_prefix': 'expert_group_list_cumsum', 'gate_up': 'declared_native_fusion', 'expert_down': 'balanced_grouped_down', 'route_pack': 'route_pack', 'expert_add': 'expert_add'}.get(op, 'matmul')
        match = {'hardware': 'Ascend910_9362', 'phase': 'decode', 'recipe': recipe}
        path = 'local_shape.0'
        scale = 1
        if op.startswith('bf16_matmul_'):
            _, _, n, k = op.split('_')
            match.pop('recipe')
            match.update({'family': 'gemm', 'metadata.compute_format': 'bf16', 'local_shape.1': int(n), 'local_shape.2': int(k)})
        elif op == 'conv':
            match.update({'local_shape.1': width, 'metadata.state_history_slots': (2 if width == 2048 else 4) + 2})
        elif op == 'recurrent':
            match.update({'metadata.heads': heads, 'metadata.dk': 128, 'metadata.dv': 128, 'metadata.state_dtype': 'fp32', 'metadata.state_checkpoints': 2 if width == 2048 else 4})
        elif op == 'gating':
            match['local_shape.1'] = heads
        elif op == 'norm_gate':
            match.update({'local_shape.1': 128, 'metadata.gate_layout.storage_shape.1': heads * 128})
            scale = 1 / heads
        elif op == 'gate_up':
            match.update({'metadata.fusion_role': 'grouped_gate_up_activation_quant', 'metadata.fusion_tensor_view.input_width': 2048, 'metadata.fusion_tensor_view.intermediate_width': 128, 'metadata.fusion_tensor_view.local_experts': 256, 'metadata.fusion_tensor_view.weight_dtype': 'int8'})
        elif op == 'expert_down':
            match.update({'local_shape.1': 2048, 'local_shape.2': 128, 'metadata.matrix_weight_bank_groups': 256, 'dtype': 'int8'})
        elif op == 'expert_add':
            match.update({'dtype': 'bf16', 'local_shape.1': 2048, 'metadata.input_tensor_count': 2})
        elif op == 'count_prefix':
            match['local_shape.0'] = 256
        elif op == 'route_pack':
            match.update({'metadata.num_experts': 256, 'metadata.top_k': 8, 'dtype': 'bf16'})
        rule = {'id': f'a3-native-{op}-{width}-short-context', 'match': match, 'coordinates': {'rows': {'path': path, 'scale': scale}}, 'domain': {'rows': {'min': lo, 'max': hi}}, 'response': {'fixed_us': fixed, 'terms': [{'coordinate': 'rows', 'us_per_unit': slope}]}, 'provenance': {'kind': 'operator_response_calibration', 'training_capture_context': 4096, 'validation_capture_context': 16384, 'parameter_boundary': 'inclusive native kernel duration, not a body residual', 'source_rows': [{'capture': r['capture'], 'sha256': r['sha256'], 'line': r['first']['line'], 'median_us': r['median_us'], 'coordinate': r['coordinate']} for r in tr], 'limitations': ['capture binary not reconstructed', 'routing distribution not identified by grouped tensor ABI', 'training and validation contexts separated; historical observations not pristine heldout']}}
        if op == 'norm_gate':
            rule['match']['metadata.caller_implementation'] = 'vllm_ascend_gdn_decode'
            rule['id'] = f'a3-native-norm-gate-{width}'
        rules.append(rule)
        for r in rows:
            pred = fixed + slope * r['coordinate']
            checks.append({'op': op, 'width': width, 'scenario': r['scenario'], 'split': r['split'], 'coordinate': r['coordinate'], 'predicted_us': pred, 'observed_us': r['median_us'], 'relative_error': pred / r['median_us'] - 1, 'absolute_error_us': abs(pred - r['median_us']), 'inside_domain': lo <= r['coordinate'] <= hi})
out = root / 'data/observations/parameters/a3-native-decode-20261030'
(out / 'parameters.json').write_text(json.dumps({'service_regimes': rules}, indent=2) + '\n')
(out / 'operator-evidence.json').write_text(json.dumps({'records': records, 'comparisons': checks}, indent=2) + '\n')
for c in checks:
    if c['split'] == 'independent_context_validation':
        print(c)
