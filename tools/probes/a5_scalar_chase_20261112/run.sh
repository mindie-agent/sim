#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${MINDIE_LOGICAL:?}"
./scalar_chase ./scalar_chase.o "${RUN_ID:?}" | tee "$OUT_JSONL"
