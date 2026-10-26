"""GPU helpers: choose the GPU, and load the cuDNN libraries safely."""

import os
import sys
import time
from pathlib import Path


def use_gpu(gpu):
    """Use only this GPU (0, 1, ...). Must be called before torch / keras start using the GPU."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


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
