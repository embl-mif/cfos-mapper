"""GPU helpers: choose the GPU, check it can be used, and load the cuDNN libraries safely."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def use_gpu(gpu):
    """Use only this GPU (0, 1, ...). Must be called before torch / keras start using the GPU."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def gpu_count():
    """Number of NVIDIA GPUs in this machine (asks nvidia-smi, so torch is not started). 0 if none."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def check_gpu(gpu):
    """
    After use_gpu: torch does not complain when it cannot use a GPU, it simply runs
    on the CPU (many times slower). So stop if the chosen GPU cannot be used, and
    print a warning if no GPU was chosen and there is none.
    """
    import torch

    if torch.cuda.is_available():
        return
    found = gpu_count()
    if torch.version.cuda is None:
        reason = "this torch has no CUDA support (CPU-only install)"
    elif found == 0:
        reason = "no NVIDIA GPU found (nvidia-smi)"
    elif gpu is not None and str(gpu).isdigit() and int(gpu) >= found:
        reason = f"this machine has GPU 0 to {found - 1}"
    else:
        reason = "the NVIDIA driver may be missing or too old for this torch"
    if gpu is not None:
        raise RuntimeError(f"GPU {gpu} cannot be used: {reason}.")
    print(f"warning: no GPU, running on the CPU (slow): {reason}.", flush=True)


def load_gpu_libraries(attempts=5, pause=10):
    """
    Load the cuDNN libraries (the GPU maths used by torch) before anything else.
    Now and then loading them fails (seen when the machine is very busy), and if
    that happens inside torch the whole program stops. Here a failure is printed
    and simply tried again after a pause. Does nothing outside Linux or without
    the nvidia-cudnn package.
    """
    import ctypes
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia.cudnn")
    except ModuleNotFoundError:
        spec = None
    if spec is None or not sys.platform.startswith("linux"):
        return
    folder = Path(list(spec.submodule_search_locations)[0]) / "lib"
    names = ["graph", "engines_precompiled", "engines_runtime_compiled", "heuristic", "ops", "cnn", "adv"]
    for name in [f"libcudnn_{n}.so.9" for n in names] + ["libcudnn.so.9"]:
        if not (folder / name).exists():
            continue
        for attempt in range(1, attempts + 1):
            try:
                ctypes.CDLL(str(folder / name), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError as error:
                print(f"loading {name} failed (attempt {attempt}/{attempts}): {error}", flush=True)
                if attempt == attempts:
                    raise
                time.sleep(pause)
