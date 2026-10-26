"""
The whole analysis of every brain of a project, on one workstation.

For every brain (<brain>.ome.zarr in settings brains_folder):
  1. run-brain   the cells of the whole brain with NETWORK (GPU, a few hours, ~2 x one channel of RAM)
  2. register    brainreg on the autofluorescence (CPU, ~30-60 min)
  3. count       the cells per atlas region
and at the end
  4. summary     all counted brains in one table -> reports/<network>/cells_per_region_all_brains.csv

Steps that are already done are skipped, so the script can be stopped (Ctrl+C) and
started again: it goes on where it stopped. A brain that fails does not stop the
others; the table at the end says what happened to every brain.

Every step runs as its own `cfos-mapper` process (the memory is freed between
brains), with its log in logs/pipeline/<date_time>/<brain>_<step>.log.

The parameters of the steps themselves (channels, candidate detection,
classification, atlas and orientation) are in settings.json; the ones below only
say what to run. Check first with a quick test (PLANES = (900, 1000)) on one brain.

Run it in tmux, so it keeps going when you log out:
    tmux new -s cfos
    python run_all_brains.py
    (Ctrl+B then D to leave it running, `tmux attach -t cfos` to come back)
"""

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cfos_mapper as cf

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

SETTINGS = "/path/to/project/settings.json"  # brains_folder, results_folder and the parameters of every step
NETWORK = "network_20261023"     # a network in the networks folder, or "pretrained"
GPU = 0                    # GPU for run-brain (0, 1, ...)

BRAINS = None              # None = every brain in brains_folder, or a list, e.g. ["O15", "O16"]
EXCLUDE = []               # brains to leave out, e.g. ["O12"]
ORIENTATION = {
    "O11": "spl", 
    # "O12": "",
    # "O13": "",
    # "O14": "",
    "O15": "sar",
    "O16": "sar",
    "O17": "sar",
    "O18": "sar",
    # "O19": "",
    "O20": "sar",
    # "O21": "",
    "O22": "sar",
    }           # brains that lie differently from settings atlas.orientation, e.g. {"O17": "sar"}

STEPS = ["run-brain", "register", "count", "summary"]  # leave out the steps you do not want now
REDO = []                  # steps to run again although they are done, e.g. ["count"] (empty it afterwards)
PLANES = None              # (first, last): only these z-planes, a quick test of the pipeline, e.g. (900, 1000)
REGISTER_ALONGSIDE = True  # register (CPU) the brains while run-brain (GPU) works, instead of one after the other


# ---------------------------------------------------------------------------

settings = cf.load_settings(SETTINGS)
run_name = cf.run_folder("any", NETWORK, settings, PLANES).name  # network_v3, or network_v3_planes900-1000
log_folder = Path(settings["logs_folder"]) / "pipeline" / f"{datetime.now():%Y%m%d_%H%M}"
results = {}  # (brain, step) -> what happened


def say(text):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {text}", flush=True)


def orientation(brain):
    return ORIENTATION.get(brain, settings["atlas"]["orientation"])


def is_done(step, brain):
    run = cf.run_folder(brain, NETWORK, settings, PLANES)
    if step == "run-brain":
        info = run / "run_info.json"
        return info.exists() and "finished" in json.load(open(info))
    if step == "register":  # registered, and with the orientation wanted now
        return cf.registration_ready(brain, settings) and cf.registered_orientation(brain, settings) == orientation(brain)
    if step == "count":  # counted after the last run-brain and the last registration
        summary = run / "regions" / "summary.json"
        inputs = [run / "cells.csv", cf.registration_folder(brain, settings) / "registered_atlas.tiff"]
        return summary.exists() and all(f.exists() and f.stat().st_mtime < summary.stat().st_mtime for f in inputs)
    return False


def run_step(step, brain, *args):
    """
    One step of one brain (brain None: a step for all brains, i.e. summary), unless
    it is done. True if its results are there afterwards.
    """
    if brain and is_done(step, brain) and step not in REDO:
        results[brain, step] = "done before"
        return True
    if step not in STEPS:
        results[brain, step] = "not done (not in STEPS)"
        return False
    log_file = log_folder / (f"{brain}_{step}.log" if brain else f"{step}.log")
    command = [sys.executable, "-m", "cfos_mapper", "--settings", settings["settings_file"],
               step, *([brain] if brain else []), *[str(a) for a in args]]
    say(f"{brain or 'all brains'}: {step} ...   (log: {log_file})")
    t0 = time.time()
    with open(log_file, "w") as f:
        f.write(" ".join(command) + "\n\n")
        f.flush()
        ok = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT).returncode == 0
    results[brain, step] = f"done ({(time.time() - t0) / 60:.0f} min)" if ok else f"FAILED, see {log_file}"
    say(f"{brain or 'all brains'}: {step} {results[brain, step]}")
    return ok


def brains_to_run():
    available = cf.brain_names(settings)
    brains = available if BRAINS is None else list(BRAINS)
    missing = [b for b in brains if b not in available]
    if missing:
        sys.exit(f"not in {settings['brains_folder']}: {missing}")
    brains = [b for b in brains if b not in EXCLUDE]
    if not brains:
        sys.exit(f"no brains to run (brains_folder: {settings['brains_folder']})")
    return brains


def check(brains):
    """Stop before starting hours of work if something is obviously wrong, and say what will run."""
    if "run-brain" in STEPS and NETWORK != "pretrained" and not Path(cf.network_file(NETWORK, settings)).exists():
        sys.exit(f"network not found: {cf.network_file(NETWORK, settings)}")
    for brain in brains:
        try:
            cf.describe_orientation(orientation(brain))
        except ValueError as e:
            sys.exit(f"{brain}: {e}")
    log_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(settings["settings_file"], log_folder / "settings.json")  # the settings of this run, to keep

    say(f"{len(brains)} brains: {', '.join(brains)}")
    print(f"  network        {NETWORK}  (run folder: {run_name}), GPU {GPU}")
    print(f"  steps          {', '.join(STEPS)}" + (f"   (again: {', '.join(REDO)})" if REDO else ""))
    print(f"  channels       cFos '{settings['signal_channel']}', autofluorescence '{settings['background_channel']}'")
    print(f"  atlas          {settings['atlas']['name']}, orientation '{settings['atlas']['orientation']}' "
          f"({cf.describe_orientation(settings['atlas']['orientation'])})")
    for brain, letters in ORIENTATION.items():
        if brain in brains:
            print(f"                 {brain}: '{letters}' ({cf.describe_orientation(letters)})")
    print(f"  results        {settings['results_folder']}")
    print(f"  logs           {log_folder}")
    for brain in brains:
        done = [step for step in ["run-brain", "register", "count"] if is_done(step, brain)]
        print(f"  {brain:<14} done before: {', '.join(done) or '-'}")
    print(flush=True)


def main():
    brains = brains_to_run()
    check(brains)
    run_brain_args = ["--network", NETWORK, "--gpu", GPU, *(["--planes", *PLANES] if PLANES else [])]

    registrations = ThreadPoolExecutor(1)  # one brainreg at a time, next to the GPU work
    registered = {}
    if REGISTER_ALONGSIDE:
        registered = {b: registrations.submit(run_step, "register", b, "--orientation", orientation(b))
                      for b in brains}
    try:
        for brain in brains:
            has_cells = run_step("run-brain", brain, *run_brain_args)
            is_registered = (registered[brain].result() if REGISTER_ALONGSIDE
                             else run_step("register", brain, "--orientation", orientation(brain)))
            if has_cells and is_registered:
                run_step("count", brain, "--network", NETWORK, "--run", run_name)
            else:
                results[brain, "count"] = "not done (needs run-brain and register)"
    except KeyboardInterrupt:
        say("stopped: run the script again to go on where it stopped")
        registrations.shutdown(wait=False, cancel_futures=True)
        raise
    registrations.shutdown()

    # every brain counted with this run, also by earlier runs of the script
    if "summary" in STEPS and any(is_done("count", b) for b in cf.brain_names(settings)):
        run_step("summary", None, "--network", NETWORK, "--run", run_name)

    say("finished")
    for brain in brains:
        print(f"  {brain:<14}" + "   ".join(f"{step}: {results.get((brain, step), '-')}"
                                           for step in ["run-brain", "register", "count"]))
    if (None, "summary") in results:
        print(f"  summary: {results[None, 'summary']} -> "
              f"{Path(settings['reports_folder']) / run_name / 'cells_per_region_all_brains.csv'}")
    return 0 if all(not r.startswith("FAILED") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
