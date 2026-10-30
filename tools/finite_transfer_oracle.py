"""Independent arithmetic for a finite transfer credit window.

[Role] Check finite transfers without importing the simulator or device software.
[Invariants] One invocation serializes transaction service; credit is returned
  after service plus response delay. Payload and charged transaction bytes differ.
[Timeouts & Locks] O(transaction count) arithmetic, O(outstanding) live state.
[Side-Effect Policy] Prints a small report; does not run hardware or fit timing.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math


def finite_transfer(payload_bytes, transaction_bytes, bandwidth_bytes_per_ms,
                    outstanding, response_ms, *, rounded_tail=True):
    """Return exact finite completion under the declared single-server contract.

    The oldest outstanding completion releases a credit. This is a recurrence
    over a serial server, not the simulator's event scheduler. A response is a
    delay after serialization; it does not consume bandwidth. There are no other
    invocations or shared-link contenders in this oracle.
    """
    if payload_bytes < 0 or transaction_bytes <= 0 or outstanding <= 0:
        raise ValueError("payload_and_transaction_geometry_invalid")
    if bandwidth_bytes_per_ms <= 0 or response_ms < 0:
        raise ValueError("service_parameters_invalid")
    count = math.ceil(payload_bytes / transaction_bytes)
    returns = deque()
    service_end, complete, charged = 0.0, 0.0, 0
    for index in range(count):
        credit_ready = returns.popleft() if len(returns) == outstanding else 0
        amount = (transaction_bytes if rounded_tail else
                  min(transaction_bytes, payload_bytes - index * transaction_bytes))
        service_end = max(service_end, credit_ready) + amount / bandwidth_bytes_per_ms
        complete = service_end + response_ms
        returns.append(complete)
        charged += amount
    return {
        "payload_bytes": payload_bytes,
        "charged_bytes": charged,
        "transactions": count,
        "duration_ms": complete,
        "serialization_ms": charged / bandwidth_bytes_per_ms,
        "response_ms": response_ms,
        "credit_stall_ms": service_end - charged / bandwidth_bytes_per_ms,
        "outstanding": outstanding,
        "tail": "rounded" if rounded_tail else "exact",
        "boundary": "last_response_after_serial_transaction_service",
    }


def homogeneous_duration(count, service_ms, outstanding, response_ms):
    """Closed form for equal transactions, including fill, credit stalls and drain."""
    if not count:
        return 0.0
    stalls = (count - 1) // outstanding
    bubble = max(0, response_ms - (outstanding - 1) * service_ms)
    return count * service_ms + response_ms + stalls * bubble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--transaction-bytes", type=int, required=True)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, required=True)
    parser.add_argument("--outstanding", type=int, required=True)
    parser.add_argument("--response-ms", type=float, required=True)
    parser.add_argument("--exact-tail", action="store_true")
    args = parser.parse_args()
    report = finite_transfer(
        args.payload_bytes, args.transaction_bytes, args.bandwidth_bytes_per_ms,
        args.outstanding, args.response_ms, rounded_tail=not args.exact_tail,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
