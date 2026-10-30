"""Selected MC2 source-operation counts, attached to existing physical phases.

[Role] Count known UB scalar accesses, prefix arithmetic, SFF and pipe barriers.
[Invariants] Counts denote source calls, not complete compiled ISA. Successful
  flag checks are work; unobserved retries and arrival waits are not invented.
  Owners follow the existing source/conditional row mapping without idle lanes.
[Timeouts & Locks] O(owners * phase steps); no I/O, locks or measurement access.
[Side-Effect Policy] Pure inventories; rates and availability belong to hardware.
"""
import math

from .instructions import PrimitiveInventory

SOURCE_RANGES = {
    'dispatch': {
        'source_id_and_prefix': (652, 804), 'triple_store': (472, 478),
        'count': (897, 923), 'receive_count': (1087, 1122),
        'prefix': (1024, 1083, 1126, 1162), 'ready': (1221, 1246),
    },
    'combine': {'return_metadata': (855, 896), 'frame': (938, 951),
                'successful_mask': (958, 993), 'weight': (1385, 1402)},
}


def phase_events(raw, phase, owner, weight, layout):
    """Per-owner inserts keyed before/after the existing aggregate phase step.

    Pipe barriers inside a whole vector phase are conservatively retired after
    that vector service, not reconstructed as a compiled instruction schedule.
    """
    result = {}

    def add(position, step, **counts):
        key = position + ':' + step
        values = result.setdefault(key, {})
        for role, count in counts.items():
            if count:
                values[role] = values.get(role, 0) + count

    rows = raw['source_assignments'] * weight
    tokens = raw['source_tokens'] * weight
    phase_name = phase.name
    if raw['kind'] == 'dispatch':
        ids, k = raw['source_tokens'] * raw['top_k'], raw['top_k']
        complete = ids == raw['source_assignments'] and float(ids).is_integer()
        if phase_name == 'send_read_quantize' and complete:
            p = len(phase.owner_domain)
            prefix = max(0, (int(ids) - 1 - (k + (owner-k) % p)) // p + 1)
            add('before', 'assignment_prefix_scan', scalar_ub_access=rows)
            add('after', 'assignment_prefix_scan', scalar_ub_access=prefix,
                pipe_barrier=4 * prefix)
        elif phase_name == 'frame_payload':
            add('before', 'service', scalar_ub_access=3 * rows)
        elif phase_name == 'count_compute':
            count = len(phase.owner_domain); index = phase.owner_domain.index(owner)
            n = raw['experts'] // count + int(index < raw['experts'] % count)
            # CalExpertCnt skips its reduction when no source IDs exist.
            reduced = n if ids >= k else 0
            add('after', 'service', scalar_ub_access=n + reduced,
                pipe_barrier=1 + 4 * reduced)
        elif phase_name == 'receive_counts':
            add('before', 'read_status', scalar_ub_access=1)  # gather pattern SetValue
            add('after', 'check_status', scalar_ub_access=1)  # one successful GetValue
            add('after', 'sum_counts', scalar_ub_access=1)  # sumOfRecvCnt
        elif phase_name == 'prefix_counts':
            count = len(phase.owner_domain); index = phase.owner_domain.index(owner)
            n = raw['experts'] // count + int(index < raw['experts'] % count)
            add('before', 'read_sums', scalar_ub_access=1)  # gather pattern
            # GetCumSum success + output store, extra prefix result/pattern for
            # nonzero owners; CalRecv preSum + every count read/prefix store.
            add('after', 'check_and_prefix', scalar_ub_access=3 + 2 * int(index > 0) + 2*n,
                scalar_alu_issue=n, pipe_barrier=1 + int(index > 0))
        elif phase_name == 'prefix_ready':
            add('after', 'check_ready', scalar_ub_access=1)
    else:
        if phase_name == 'return_read_frame':
            add('after', 'read', scalar_ub_access=3 * rows)
            add('after', 'frame', pipe_barrier=rows)
        elif phase_name == 'receive_flags':
            words = math.ceil(raw['top_k'] * layout.blocks * 8 / 64)
            add('after', 'initialize_flags', pipe_barrier=tokens)
            add('after', 'compare_flags', scalar_ub_access=tokens*words,
                scalar_alu_issue=tokens*words)  # one SFF per successful mask word
        elif phase_name == 'weighted_reduce':
            for slot in range(raw['top_k']):
                add('before', f'slot{slot}_read', scalar_ub_access=tokens)
    return {key: PrimitiveInventory(value) for key, value in result.items()}
