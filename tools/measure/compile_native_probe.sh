#!/usr/bin/env bash
# CPU-only build inside the recorded image; no device mounts are required.
set -eu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
probe_dir="${1:?probe directory required}"
probe_ir="${2:?retained compiler IR required}"
probe_output="${3:-$probe_dir/store-kernel}"
ascend_root=/usr/local/Ascend/ascend-toolkit/latest
compiler_dir=/usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend/bishengir/bin
export PATH="$compiler_dir:$PATH"
g++ -std=c++17 -O2 "$probe_dir/native_kernel_probe.cpp" \
  -I"$ascend_root/include" -L"$ascend_root/lib64" -lascendcl \
  -o "$probe_dir/native_kernel_probe"
"$compiler_dir/bishengir-compile" "$probe_ir" \
  --target=Ascend950DT_9582 --disable-ffts --enable-auto-bind-sub-block=false \
  --enable-vf-operand-substitution=true --enable-hfusion-compile=true \
  --enable-triton-kernel-compile=true -o "$probe_output"
