"""Tests for vl.paths — MACHINE-resolved storage roots."""

import os
from pathlib import Path

import pytest

from vl import paths


def test_machine_required(monkeypatch):
    monkeypatch.delenv("MACHINE", raising=False)
    with pytest.raises(RuntimeError, match="MACHINE env var not set"):
        paths.data_root()


def test_machine_invalid(monkeypatch):
    monkeypatch.setenv("MACHINE", "potato")
    with pytest.raises(ValueError, match="Unknown MACHINE"):
        paths.data_root()


def test_local(monkeypatch):
    monkeypatch.setenv("MACHINE", "local")
    assert paths.data_root() == paths.REPO_ROOT / "data"
    assert paths.checkpoint_root() == paths.REPO_ROOT / "checkpoints"
    assert paths.results_root() == paths.REPO_ROOT / "results"
    assert paths.log_root() == paths.REPO_ROOT / "logs"


def test_bioai(monkeypatch):
    monkeypatch.setenv("MACHINE", "bioai")
    assert paths.data_root() == Path("/data/joonhee/vl/data")
    assert paths.checkpoint_root() == Path("/data/joonhee/vl/checkpoints")
    assert paths.results_root() == Path("/data/joonhee/vl/results")
    assert paths.log_root() == paths.REPO_ROOT / "logs"


def test_ensure_dirs_creates(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINE", "local")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    paths.ensure_dirs()
    for sub in ("data", "checkpoints", "results", "logs"):
        assert (tmp_path / sub).is_dir()
