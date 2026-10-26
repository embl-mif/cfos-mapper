"""
Settings: one JSON file per project (where the brains are, where results go,
and the parameters of every step). See templates/settings.json.

The settings file is found in this order:
  1. the path you give (Python: load_settings(path); command line: --settings)
  2. the environment variable CFOS_MAPPER_SETTINGS
  3. settings.json in the current folder
"""

import json
import os
from pathlib import Path

ENVIRONMENT_VARIABLE = "CFOS_MAPPER_SETTINGS"
TEMPLATES = Path(__file__).parent / "templates"

# values used when the settings file does not give them
DEFAULT_ATLAS = {
    "name": "allen_mouse_25um",
    # brainglobe orientation of the images: each letter = where axis 0, 1, 2 starts.
    # Ours: plane 0 = superior (dorsal), y starts anterior, x starts left -> "sal"
    "orientation": "sal",
    "n_free_cpus": 8,
}


def find_settings_file(path=None):
    """The settings file to use (see the order at the top), or an error saying where we looked."""
    candidates = [path, os.environ.get(ENVIRONMENT_VARIABLE), "settings.json"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "No settings file found. Give one with --settings (command line) or load_settings(path) (Python), "
        f"set {ENVIRONMENT_VARIABLE}, or run in a folder with settings.json. "
        "`cfos-mapper init FOLDER` writes a template.")


def load_settings(path=None):
    """
    Read the settings file and fill in the folders that are not given:
      regions_file     regions.json next to the settings file
      networks_folder  <results_folder>/../networks
      reports_folder   <results_folder>/../reports
      logs_folder      <results_folder>/../logs
      atlas            DEFAULT_ATLAS, atlas files in <results_folder>/../atlases
    """
    file = find_settings_file(path)
    with open(file) as f:
        settings = json.load(f)
    settings["settings_file"] = str(file)
    project = Path(settings["results_folder"]).parent
    settings.setdefault("regions_file", str(file.parent / "regions.json"))
    settings.setdefault("networks_folder", str(project / "networks"))
    settings.setdefault("reports_folder", str(project / "reports"))
    settings.setdefault("logs_folder", str(project / "logs"))
    settings["atlas"] = {**DEFAULT_ATLAS, "folder": str(project / "atlases"), **settings.get("atlas", {})}
    return settings


def load_regions(settings):
    """Test regions per brain: {brain: {region: {"z": [start, end], "y": [...], "x": [...]}}}."""
    file = Path(settings["regions_file"])
    if not file.exists():
        return {}
    with open(file) as f:
        regions = json.load(f)
    return {brain: boxes for brain, boxes in regions.items() if not brain.startswith("_")}


def save_regions(regions, settings):
    """Write the test regions (keeps the comments, i.e. keys starting with "_", already in the file)."""
    file = Path(settings["regions_file"])
    comments = {}
    if file.exists():
        with open(file) as f:
            comments = {k: v for k, v in json.load(f).items() if k.startswith("_")}
    with open(file, "w") as f:
        json.dump({**comments, **regions}, f, indent=4)


def write_templates(folder):
    """Write a template settings.json and regions.json into a folder (existing files are kept)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ["settings.json", "regions.json"]:
        target = folder / name
        if not target.exists():
            target.write_text((TEMPLATES / name).read_text())
            written.append(target)
    return written
