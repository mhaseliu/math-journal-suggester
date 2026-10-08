"""Explicit Modal B300 execution; never permit laptop or CPU fallback."""
import os
import platform


def require_modal_b300():
    if os.environ.get("JOURNAL_EXECUTION_BACKEND") != "modal-b300":
        raise RuntimeError("Modal B300 execution must be explicitly selected")
    import modal
    # is_local() is True in a Function's child process. Modal's runtime
    # environment survives subprocess launch; never set these markers ourselves.
    remote_child = (os.environ.get("MODAL_IS_REMOTE") == "1"
                    and os.environ.get("MODAL_TASK_ID", "").startswith("ta-"))
    if (modal.is_local() and not remote_child) or platform.machine() != "x86_64":
        raise RuntimeError("Model execution requires a remote Modal B300 container")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "B300" not in torch.cuda.get_device_name():
        raise RuntimeError("Exactly one NVIDIA B300 CUDA device is required")
    if tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) < (13, 1):
        raise RuntimeError("Modal B300 requires a CUDA 13.1+ PyTorch build")
    return torch


def require_training_gpu(min_available_gib=0):
    backend = os.environ.get("JOURNAL_EXECUTION_BACKEND", "gb10")
    if backend == "modal-b300":
        return require_modal_b300()
    if backend == "cuda":
        return require_cuda()
    if backend != "gb10":
        raise RuntimeError(f"Unknown execution backend: {backend}")
    from .gpu import require_gb10
    return require_gb10(min_available_gib)


def require_cuda():
    """Explicit opt-in only; never fall back to CPU or choose a remote host."""
    if os.environ.get("JOURNAL_ALLOW_MODEL_EXECUTION") != "1":
        raise RuntimeError("Set JOURNAL_ALLOW_MODEL_EXECUTION=1 on your GPU machine")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    return torch
