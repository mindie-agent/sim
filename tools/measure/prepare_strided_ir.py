"""Derive an address-stride experiment from retained compiler input.

[Role] Add an independent destination leading dimension to a store primitive.
[Invariants] Keep logical masks, tile loops, arithmetic and source layout intact.
  Record both IR hashes; this is an experiment, never an original kernel.
[Timeouts & Locks] Small text input only; no device/runtime initialization.
[Side-Effect Policy] Write new compiler input and provenance to a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def prepare(source, output):
    text = source.read_text()
    result = re.sub(r"%arg(\d+)", lambda m: "%arg" + str(
        int(m[1]) + (int(m[1]) >= 7)), text)
    result = result.replace("@strided_store(", "@strided_store_ld(", 1)
    result = result.replace("%arg8: i32", "%arg7: i32, %arg8: i32", 1)
    old = "%14 = arith.muli %12, %13 : index"
    assert result.count(old) == 1
    result = result.replace(old, "%leading_dim = arith.index_cast %arg7 : i32 to index\n"
                            "        %14 = arith.muli %12, %leading_dim : index")
    assert result.count("strides: [%13, 1]") == 1
    result = result.replace("strides: [%13, 1]", "strides: [%leading_dim, 1]")
    output.mkdir()
    (output / "store-ld.bcmlir").write_text(result)
    (output / "provenance.json").write_text(json.dumps({
        "purpose": "isolate destination row stride from logical column count",
        "source": str(source),
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "result_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "changes": ["new leading_dim i32 before grid arguments",
                    "destination base and row stride use leading_dim",
                    "masks and loops still use logical heads"],
        "calibration": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
