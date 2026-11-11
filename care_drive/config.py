"""Resolve user-controlled paths and paper stage recipes."""
import json
import os
from pathlib import Path
import yaml


def load(path):
    path = Path(path).resolve()
    expanded = os.path.expandvars(path.read_text())
    import re
    if re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", expanded):
        raise ValueError("Set the environment variables referenced by this configuration first")
    value = yaml.safe_load(expanded)
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration mapping")
    return value


def recipes(path):
    return json.loads(Path(path).read_text())


def stage_checkpoint(output, name):
    return Path(output) / name / "epoch_005.pt"
