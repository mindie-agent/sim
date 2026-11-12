#!/usr/bin/env bash
set -euo pipefail
CANN=/usr/local/Ascend/cann-9.2.0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bisheng -std=c++17 -xcce -O2 -D__DAV_C310__ --cce-aicore-only \
  -I"$CANN/aarch64-linux/tikcpp" \
  -I"$CANN/aarch64-linux/tikcpp/tikcfw" \
  -I"$CANN/aarch64-linux/tikcpp/tikcfw/impl" \
  -I"$CANN/aarch64-linux/tikcpp/tikcfw/interface" \
  -I"$CANN/aarch64-linux/include" -I"$CANN/include" \
  --cce-aicore-arch=dav-c310-vec -c scalar_chase.cpp -o scalar_chase.o
g++ -O2 -std=c++17 scalar_chase_host.cpp -I"$CANN/include" -L"$CANN/lib64" \
  -lascendcl -Wl,-rpath,"$CANN/lib64" -o scalar_chase
sha256sum scalar_chase.cpp scalar_chase.o scalar_chase_host.cpp scalar_chase
