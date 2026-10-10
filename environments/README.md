# GPU stacks

CPU installation does not install PyTorch. Neural commands run only on a deliberately configured GPU host.

**ASUS Ascent GX10 (NVIDIA GB10):** `uv sync --project environments/gb10-speed --frozen` with Python 3.12. The lock includes the pinned Kev Git revision, PyTorch 2.8.0+cu129, Triton 3.4.0 and the ARM causal-conv1d wheel. This configuration is used for website inference.

**B300:** training and test evaluation used `nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63`, Python 3.12, PyTorch 2.12.1+cu132 and Triton 3.7.1. Inside that GPU host's isolated environment:

```bash
python -m pip install torch==2.12.1+cu132 triton==3.7.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r environments/modal-b300/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu132
python -m pip install --no-deps 'kev @ git+https://github.com/jaredpalmer/kev@5e42a7a03f28134853dd3ff77461457e921e5ec1'
python environments/modal-b300/build_conv.py
python -m pip install --no-deps -e .
```

The explicit Kev dependency override is intentional and was numerically checked in the original experiment. No Modal account is required by this portable runner. Avoid substituting kernels or arithmetic settings when comparing results.
