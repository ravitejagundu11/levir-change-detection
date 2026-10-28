import random

import numpy as np
import pytest

from cd import utils
from cd.utils import (
    ProjectPaths,
    apply_overrides,
    get_paths,
    load_config,
    save_config,
    set_seed,
)


# ---------------------------------------------------------------------------- config
def _write(tmp_path, text, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_reads_nested_yaml(tmp_path):
    path = _write(tmp_path, "model:\n  name: unet_ef\n  in_channels: 6\ntrain:\n  epochs: 150\n")
    cfg = load_config(path)
    assert cfg == {"model": {"name": "unet_ef", "in_channels": 6}, "train": {"epochs": 150}}


def test_load_config_parses_scientific_notation_as_float(tmp_path):
    path = _write(tmp_path, "lr: 1e-3\nwd: 5.0e-4\nlrs: [1e-2, 1E-4]\nname: e5\n")
    cfg = load_config(path)
    assert cfg["lr"] == pytest.approx(1e-3) and isinstance(cfg["lr"], float)
    assert cfg["wd"] == pytest.approx(5e-4)
    assert cfg["lrs"] == [pytest.approx(1e-2), pytest.approx(1e-4)]
    assert cfg["name"] == "e5"  # ordinary strings are left alone


def test_load_config_empty_file_gives_empty_dict(tmp_path):
    assert load_config(_write(tmp_path, "")) == {}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_load_config_applies_overrides(tmp_path):
    path = _write(tmp_path, "train:\n  epochs: 150\n  lr: 1e-3\n")
    cfg = load_config(path, overrides=["train.epochs=5", "train.lr=1e-4", "name=debug"])
    assert cfg["train"]["epochs"] == 5
    assert cfg["train"]["lr"] == pytest.approx(1e-4)
    assert cfg["name"] == "debug"


def test_apply_overrides_types_new_keys_and_no_mutation():
    base = {"train": {"epochs": 10}}
    out = apply_overrides(base, ["train.amp=false", "data.tile=256", "model.widths=[32, 64]"])
    assert out["train"] == {"epochs": 10, "amp": False}
    assert out["data"] == {"tile": 256}
    assert out["model"]["widths"] == [32, 64]
    assert base == {"train": {"epochs": 10}}  # original untouched


@pytest.mark.parametrize("bad", ["no_equals", "=5", "a..b=1"])
def test_apply_overrides_rejects_malformed(bad):
    with pytest.raises(ValueError):
        apply_overrides({}, [bad])


def test_apply_overrides_rejects_descending_into_scalar():
    with pytest.raises(ValueError):
        apply_overrides({"train": 3}, ["train.epochs=5"])


def test_save_then_load_roundtrip(tmp_path):
    cfg = {"model": {"name": "siam_unet", "fusion": "diff"}, "train": {"lr": 0.001, "epochs": 2}}
    path = save_config(cfg, tmp_path / "run" / "config.yaml")
    assert path.is_file()
    assert load_config(path) == cfg


# --------------------------------------------------------------------------- seeding
def _draw_python_numpy():
    return [random.random() for _ in range(3)], np.random.rand(3)


def test_set_seed_reproducible_python_and_numpy():
    set_seed(123)
    py1, np1 = _draw_python_numpy()
    set_seed(123)
    py2, np2 = _draw_python_numpy()
    assert py1 == py2
    np.testing.assert_array_equal(np1, np2)

    set_seed(124)
    py3, _ = _draw_python_numpy()
    assert py3 != py1


def test_set_seed_reproducible_torch():
    torch = pytest.importorskip("torch")
    set_seed(7)
    a = torch.rand(4)
    set_seed(7)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_get_device_returns_torch_device():
    torch = pytest.importorskip("torch")
    assert isinstance(utils.get_device(), torch.device)
    assert utils.get_device("cpu").type == "cpu"


# ----------------------------------------------------------------------------- paths
def test_get_paths_explicit_root_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(utils.ROOT_ENV_VAR, str(tmp_path / "from_env"))
    paths = get_paths(tmp_path / "explicit")
    assert paths.root == tmp_path / "explicit"


def test_get_paths_uses_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(utils.ROOT_ENV_VAR, str(tmp_path / "from_env"))
    assert get_paths().root == tmp_path / "from_env"


def test_get_paths_falls_back_to_workspace_outside_colab(tmp_path, monkeypatch):
    monkeypatch.delenv(utils.ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(utils, "in_colab", lambda: False)
    monkeypatch.chdir(tmp_path)
    assert get_paths().root == tmp_path / "workspace"


def test_get_paths_uses_drive_default_in_colab(monkeypatch):
    monkeypatch.delenv(utils.ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(utils, "in_colab", lambda: True)
    assert str(get_paths().root) == utils.DEFAULT_DRIVE_ROOT


def test_project_paths_layout_and_make_dirs(tmp_path):
    paths = ProjectPaths(tmp_path).make_dirs()
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.tiles == tmp_path / "data" / "tiles"
    assert paths.stats_file == tmp_path / "data" / "stats.json"
    assert paths.run_dir("unet_ef") == tmp_path / "runs" / "unet_ef"
    for p in (paths.raw, paths.tiles, paths.runs, paths.results):
        assert p.is_dir()
