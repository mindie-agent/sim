"""Bounded driver-only GPU memory microbenchmark.

[Role] Measure independent primitive parameters and held-out sizes.
[Invariants] No model weights, system changes or package installation. Check
  free VRAM before context creation and allocation; verify copied output.
[Timeouts & Locks] Finite launches, bounded CUDA event queries (10s per batch).
[Side-Effect Policy] Owns and releases one context and <=256 MiB device memory.
"""
import ctypes as C
from ctypes.util import find_library
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.free,utilization.gpu,driver_version",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    name, uuid, free, utilization, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if int(free) < 1024 or int(utilization) > 20:
        raise RuntimeError("gpu_preflight_requires_1024_MiB_free_and_utilization_at_most_20_percent")
    lib = find_library("cuda") or find_library("nvcuda") or "nvcuda.dll"
    cuda = C.CDLL(lib)

    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise RuntimeError(f"{name}:{code}")

    device = C.c_int()
    context, module, kernel = C.c_void_p(), C.c_void_p(), C.c_void_p()
    source, destination = C.c_uint64(), C.c_uint64()
    events = [C.c_void_p(), C.c_void_p()]
    call("cuInit", 0)
    call("cuDeviceGet", C.byref(device), 0)
    call("cuCtxCreate_v2", C.byref(context), 0, device)
    samples = []
    try:
        size = 128 * 1024**2
        call("cuMemAlloc_v2", C.byref(source), C.c_size_t(size))
        call("cuMemAlloc_v2", C.byref(destination), C.c_size_t(size))
        call("cuMemsetD32_v2", source, C.c_uint(0x13579BDF), C.c_size_t(size // 4))
        ptx = C.create_string_buffer(Path(__file__).with_name("copy.ptx").read_bytes())
        call("cuModuleLoadData", C.byref(module), ptx)
        call("cuModuleGetFunction", C.byref(kernel), module, b"stream_copy")
        for event in events:
            call("cuEventCreate", C.byref(event), 0)
        counts = (64, 96, 112, 128)
        for trial in range(5):
            for mib in counts[trial % 4:] + counts[:trial % 4]:
                words = C.c_uint(mib * 1024**2 // 4)
                params = (C.c_void_p * 3)(C.addressof(destination), C.addressof(source), C.addressof(words))
                repetitions = 24
                for warmup in range(4):
                    call("cuLaunchKernel", kernel, (words.value + 255) // 256, 1, 1,
                         256, 1, 1, 0, C.c_void_p(), params, C.c_void_p())
                call("cuEventRecord", events[0], C.c_void_p())
                for _ in range(repetitions):
                    call("cuLaunchKernel", kernel, (words.value + 255) // 256, 1, 1,
                         256, 1, 1, 0, C.c_void_p(), params, C.c_void_p())
                call("cuEventRecord", events[1], C.c_void_p())
                deadline = time.monotonic() + 10
                while True:
                    code = cuda.cuEventQuery(events[1])
                    if code == 0:
                        break
                    if code != 600 or time.monotonic() >= deadline:
                        raise RuntimeError(f"cuda_event_not_complete:{code}")
                    time.sleep(.001)
                elapsed = C.c_float()
                call("cuEventElapsedTime", C.byref(elapsed), events[0], events[1])
                for offset in (0, words.value - 1):
                    value = C.c_uint()
                    call("cuMemcpyDtoH_v2", C.byref(value), C.c_uint64(destination.value + offset * 4),
                         C.c_size_t(4))
                    if value.value != 0x13579BDF:
                        raise RuntimeError("copy_result_mismatch")
                samples.append({"trial": trial, "size_bytes": words.value * 4,
                                "work_bytes": words.value * 8, "duration_ms": elapsed.value / repetitions,
                                "split": "heldout" if mib == 112 else "train",
                                "repetitions": repetitions, "correct": True})
        print(json.dumps({"schema": "mindie-sim.cuda-memory-observations/1", "gpu": name,
                          "uuid": uuid, "driver": driver, "preflight": result.stdout.strip(),
                          "kernel": "stream_copy", "boundary": "cuda_event_device",
                          "samples": samples}))
    finally:
        for event in events:
            if event.value:
                call("cuEventDestroy_v2", event)
        if module.value:
            call("cuModuleUnload", module)
        for buffer in (source, destination):
            if buffer.value:
                call("cuMemFree_v2", buffer)
        call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    main()
