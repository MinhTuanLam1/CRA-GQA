"""Shared helpers for Kaggle CRA-GQA smoke-test notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def ensure_repo(repo_root: Path | None = None) -> Path:
    def _use(path: Path) -> Path:
        path = path.resolve()
        os.chdir(path)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return path

    if repo_root is not None:
        repo_root = Path(repo_root)
        if (repo_root / "main.py").exists():
            return _use(repo_root)

    cwd = Path.cwd()
    if (cwd / "main.py").exists():
        return _use(cwd)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for child in kaggle_input.iterdir():
            if child.is_dir() and (child / "main.py").exists():
                return _use(child)
            nested = child / "CRA-GQA"
            if nested.is_dir() and (nested / "main.py").exists():
                return _use(nested)

    clone_dir = Path("/kaggle/working/CRA-GQA")
    if not (clone_dir / "main.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/WissingChen/CRA-GQA.git", str(clone_dir)],
            check=True,
        )
    return _use(clone_dir)


def install_smoke_deps() -> None:
    packages = [
        "transformers==4.47.1",
        "h5py==3.12.1",
        "pyyaml==6.0.2",
        "tabulate==0.9.0",
        "pandas==2.2.3",
        "scipy==1.15.0",
        "peft==0.14.0",
        "sentencepiece==0.2.0",
        "torch-geometric==2.6.1",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def patch_roberta_paths(model_name: str = "roberta-base") -> None:
    import torch.nn as nn

    import modules.m_tempclip.language_model as language_model
    from transformers import RobertaConfig, RobertaModel

    def _patched_init(self, tokenizer, lan):
        nn.Module.__init__(self)
        self.tokenizer = tokenizer
        if lan == "RoBERTa":
            config = RobertaConfig.from_pretrained(model_name, output_hidden_states=True)
            self.bert = RobertaModel.from_pretrained(model_name, config=config)
        else:
            raise ValueError(f"Smoke tests only patch RoBERTa, got: {lan}")

    language_model.Bert.__init__ = _patched_init


def load_smoke_config(
    cfg_path: str = "config/CRA/CRA_NextGQA.yml",
    running_name: str = "kaggle_smoke",
    max_feats: int = 8,
    batch_size: int = 2,
    epochs: int = 1,
    record_dir: str = "/kaggle/working/output",
    resume: str | None = None,
) -> dict:
    with open(cfg_path) as fp:
        cfgs = yaml.safe_load(fp)

    cfgs["dataset"]["batch_size"] = batch_size
    cfgs["dataset"]["num_thread_reader"] = 0
    cfgs["dataset"]["max_feats"] = max_feats
    cfgs["optim"]["epochs"] = epochs
    cfgs["optim"]["save_period"] = 1
    cfgs["optim"]["step_size"] = 1
    cfgs["stat"]["record_dir"] = record_dir
    cfgs["stat"]["resume"] = resume
    cfgs["stat"]["monitor"]["early_stop"] = 1
    cfgs["misc"]["cuda"] = "0"
    cfgs["misc"]["running_name"] = running_name
    cfgs["misc"]["seed"] = 42
    cfgs["model"]["lan_weight_path"] = "roberta-base"
    return cfgs


def assert_cuda_ready() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Set Kaggle accelerator to GPU before running this notebook.")
