"""CPU image-build step: unchanged causal-conv1d kernels, B300 target only."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

URL = "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/causal_conv1d-1.7.0.tar.gz"
SHA256 = "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b"


def main():
    root = Path(os.environ.get("JOURNAL_BUILD_DIR", "artifacts/causal-conv-build")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.tar.gz"
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("causal-conv1d source checksum mismatch")
    with tarfile.open(archive) as source:
        source.extractall(root, filter="data")
    project = root / "causal_conv1d-1.7.0"
    setup = project / "setup.py"
    text = setup.read_text()
    anchor = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected upstream build script")
    # Only narrow the build target; kernel implementation and arithmetic stay unchanged.
    text = text.replace(anchor, '    if not HIP_BUILD:\n        cc_flag = ["-gencode", "arch=compute_103,code=sm_103"]\n' + anchor)
    setup.write_text(text)
    command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(project)]
    env = {**os.environ, "CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAX_JOBS": "2",
           "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"}
    log_path = root / "build.log"
    with log_path.open("w") as log:
        try:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        except subprocess.TimeoutExpired:
            print("Build exceeded 15 minutes", flush=True)
            raise
    lines = log_path.read_text().splitlines()
    if result.returncode:
        errors = [line for line in lines if any(key in line.lower() for key in ("error:", "fatal", "killed", "failed:"))]
        print("\n".join(errors[-30:] + lines[-80:]), flush=True)
        raise RuntimeError(f"causal-conv1d build failed; full log: {log_path}")
    print("\n".join(lines[-12:]), flush=True)


if __name__ == "__main__":
    main()
