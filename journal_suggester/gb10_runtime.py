"""Opt-in GB10 memory limit and CUDA 13 compatibility for pinned Triton 3.4."""
import math
import os
import platform
import runpy
import sys


def configure():
    # Environment survives the native trainer subprocess; a parent-only CUDA
    # setting would not protect its allocations.
    if value := os.environ.get("JOURNAL_CUDA_MEMORY_GIB"):
        limit = float(value)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("JOURNAL_CUDA_MEMORY_GIB must be finite and positive")
        if platform.machine() not in ("aarch64", "arm64"):
            raise RuntimeError("The memory-limited runtime requires GB10")
        import torch
        if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name():
            raise RuntimeError("The memory-limited runtime requires GB10")
        fraction = limit * 2**30 / torch.cuda.get_device_properties(0).total_memory
        if fraction > 0.5:
            raise ValueError("Keep the GB10 CUDA allocation limit at or below half of device memory")
        torch.cuda.set_per_process_memory_fraction(fraction)
    if os.environ.get("JOURNAL_TRITON_CUDA13") != "1":
        return
    import triton
    from triton.backends.nvidia import compiler
    if triton.__version__ != "3.4.0" or compiler.get_ptxas().version != "13.0":
        raise RuntimeError("The compatibility shim requires Triton 3.4.0 and CUDA 13.0 ptxas")
    if getattr(compiler.ptx_get_version, "_journal_cuda13", False):
        return
    original = compiler.ptx_get_version

    def ptx_get_version(cuda_version):
        # Backport the CUDA 13.0 -> PTX 9.0 mapping from Triton release/3.5.x:
        # https://github.com/triton-lang/triton/blob/release/3.5.x/third_party/nvidia/backend/compiler.py
        return 90 if cuda_version == "13.0" else original(cuda_version)

    ptx_get_version._journal_cuda13 = True
    compiler.ptx_get_version = ptx_get_version


if __name__ == "__main__":
    configure()
    module = sys.argv.pop(1)
    sys.argv[0] = module
    runpy.run_module(module, run_name="__main__")
