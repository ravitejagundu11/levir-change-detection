"""Shared helpers: seeding, device selection, YAML config loading and project paths.

torch is imported lazily so config and path helpers work without it.
"""

from __future__ import annotations

import copy
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# Default project folder on Google Drive (holds data, checkpoints and results).
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/cs576-change-detection"
# Environment variable that overrides the project root everywhere.
ROOT_ENV_VAR = "CD_PROJECT_ROOT"

# PyYAML follows YAML 1.1, which reads "1e-3" as a string. Configs use that form for
# learning rates, so values matching this pattern are converted to float.
_SCI_NOTATION = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) torch on CPU and all GPUs.

    deterministic=True also makes cuDNN deterministic, which is slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 (signature required by DataLoader)
    """DataLoader worker_init_fn: derive NumPy/Python seeds from torch's per-worker seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------- device
def get_device(prefer: str = "auto"):
    """Return a torch.device. "auto" picks CUDA, then Apple MPS, then CPU."""
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------- config
def _parse_scalar(value: Any) -> Any:
    if isinstance(value, str) and _SCI_NOTATION.match(value.strip()):
        return float(value)
    return value


def _coerce(node: Any) -> Any:
    """Recursively convert scientific-notation strings to floats."""
    if isinstance(node, dict):
        return {k: _coerce(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce(v) for v in node]
    return _parse_scalar(node)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of cfg with dotted "key.sub=value" overrides applied.

    Values are parsed as YAML, so "epochs=5", "lr=1e-4", "aug=false" and
    "name=run1" get int, float, bool and str types. Missing keys are created.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        keys = [k.strip() for k in key.split(".")]
        if not all(keys):
            raise ValueError(f"Invalid override key: {key!r}")
        node = cfg
        for k in keys[:-1]:
            child = node.get(k)
            if child is None:
                child = node[k] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"Cannot set {key!r}: {k!r} is not a mapping")
            node = child
        node[keys[-1]] = _coerce(yaml.safe_load(raw))
    return cfg


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config file into a dict, then apply optional overrides."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(cfg).__name__}")
    cfg = _coerce(cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | os.PathLike) -> Path:
    """Write cfg as YAML (e.g. next to a run's checkpoints). Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ----------------------------------------------------------------------------- paths
def in_colab() -> bool:
    """True when running inside Google Colab."""
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ProjectPaths:
    """Standard locations under the project root (on Drive when in Colab).

    Layout:
        <root>/data/raw          LEVIR-CD as downloaded (train/val/test/{A,B,label})
        <root>/data/tiles        256x256 tiles written by prepare_data
        <root>/data/stats.json   per-channel mean/std from the training set
        <root>/runs/<name>       checkpoints, CSV logs and saved config per run
        <root>/results           metrics JSON and figures
    """

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def tiles(self) -> Path:
        return self.data / "tiles"

    @property
    def stats_file(self) -> Path:
        return self.data / "stats.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def run_dir(self, name: str) -> Path:
        return self.runs / name

    def make_dirs(self) -> "ProjectPaths":
        for p in (self.raw, self.tiles, self.runs, self.results):
            ensure_dir(p)
        return self


def get_paths(root: str | os.PathLike | None = None) -> ProjectPaths:
    """Resolve the project root.

    Order: explicit root argument, then $CD_PROJECT_ROOT, then the Drive default
    when in Colab, otherwise ./workspace in the current directory.
    """
    if root is None:
        root = os.environ.get(ROOT_ENV_VAR)
    if root is None:
        root = DEFAULT_DRIVE_ROOT if in_colab() else Path.cwd() / "workspace"
    return ProjectPaths(Path(root).expanduser())
