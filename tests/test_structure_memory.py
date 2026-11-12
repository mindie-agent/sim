"""Structure export interns repeated strings without changing the payload.

[Role] Guard the owned structure-export memory contract: a decode report
  repeats a small string vocabulary across thousands of nodes, so duplicate
  string objects dominate retention.
[Invariants] Repeated enum-like strings share one object; the JSON payload,
  node order, graph identity and caller source are unchanged.
[Timeouts & Locks] One small declared body export; no device, I/O or locks.
[Side-Effect Policy] Read-only case metadata; returns a caller-owned report snapshot.
"""
import json
from pathlib import Path

from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure

ROOT = Path(__file__).resolve().parents[1]


def test_exported_structure_interns_repeated_strings_and_keeps_payload():
    raw = json.loads((ROOT / "examples/cases/qwen35-a3-tp4.json").read_text())
    case = parse_case(raw)
    source = {"kind": "case_json", "path": "examples/cases/qwen35-a3-tp4.json"}
    report = export_structure(case, phase="prefill", tokens=8, scope="body",
                              source=source)
    nodes = report["nodes"]
    assert nodes
    # Every declared node and the enclosing execution record must share the
    # same interned phase object; this is the retained-memory contract.
    phases = {id(node["phase"]) for node in nodes}
    assert len(phases) == 1
    assert id(report["execution"]["phase"]) in phases
    # Serialization, order and provenance are byte-for-byte unchanged.
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    decoded = json.loads(encoded)
    assert decoded == report
    assert decoded["graph_id"] == report["graph_id"]
    assert decoded["source"] == source
    assert [node["node_id"] for node in decoded["nodes"]] == [
        node["node_id"] for node in nodes]
    assert [group["subgraph_id"] for group in decoded["subgraphs"]] == [
        group["subgraph_id"] for group in report["subgraphs"]]
