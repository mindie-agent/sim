"""Compiler source for a separate tiled-store primitive.

[Role] Produce a separate output-store experiment with retained compiler output.
[Invariants] No model operations; one seed load/conversion precedes tiled stores.
  A compiled transfer path must be inspected before its timings are reusable.
[Timeouts & Locks] Caller bounds compilation and prechecks device resources.
[Side-Effect Policy] The backend may initialize a device for capability queries.
  No kernel launches; writes compiler products under the requested directory.
"""
import argparse
import json
from pathlib import Path
import resource

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit(do_not_specialize=["heads", "tokens", "iterations"])
def strided_store(output, source, heads, tokens, iterations):
    pid = tl.program_id(0)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 8)
    seed = tl.load(source + rows[:, None] * 8 + cols[None, :])
    values = (seed.to(tl.float32) * 2.0).to(tl.bfloat16)
    for row_index in range(iterations):
        row = (pid * iterations + row_index) * 64 + rows
        for col_index in range(tl.cdiv(heads, 8)):
            col = col_index * 8 + cols
            tl.store(output + row[:, None] * heads + col[None, :], values,
                     (row[:, None] < tokens) & (col[None, :] < heads))


def compile_probe(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compiled = triton.compile(
        ASTSource(strided_store, signature={"output": "*bf16", "source": "*bf16",
                                           "heads": "i32", "tokens": "i32", "iterations": "i32"}),
        target=GPUTarget("npu", "Ascend950DT_9582", 0),
    )
    for kind, content in compiled.asm.items():
        (output / ("store." + kind)).write_bytes(content.encode() if isinstance(content, str) else content)
    report = {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "metadata": compiled.metadata._asdict()}
    (output / "compile.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
    print(json.dumps({"peak_rss_kib": report["peak_rss_kib"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    compile_probe(parser.parse_args().output)
