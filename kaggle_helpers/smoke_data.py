"""Minimal NextGQA-style fixtures for Kaggle smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _sample_rows(n_videos: int = 2, n_q_per_video: int = 4) -> pd.DataFrame:
    rows = []
    for vi in range(n_videos):
        vid = f"vid{vi:03d}"
        for qi in range(n_q_per_video):
            qid = f"q{qi:03d}"
            rows.append(
                {
                    "video_id": vid,
                    "question": f"What is happening in video {vi} question {qi}?",
                    "answer": "A person walks",
                    "qid": qid,
                    "type": "Tem",
                    "a0": "A person walks",
                    "a1": "A car drives",
                    "a2": "A dog runs",
                    "a3": "Nothing happens",
                    "a4": "A bird flies",
                }
            )
    return pd.DataFrame(rows)


def _write_h5(path: Path, video_ids: list[str], n_frames: int = 16, feat_dim: int = 768) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = np.random.randn(len(video_ids), n_frames, feat_dim).astype(np.float32)
    vids = np.array(video_ids, dtype="S")
    with h5py.File(path, "w") as fp:
        fp.create_dataset("vid", data=vids)
        fp.create_dataset("CLIPL_I", data=feats)


def _make_gsub(df: pd.DataFrame, duration: float = 30.0, fps: float = 30.0) -> dict:
    gsub: dict = {}
    for _, row in df.iterrows():
        vid, qid = str(row["video_id"]), str(row["qid"])
        if vid not in gsub:
            gsub[vid] = {"duration": duration, "fps": fps, "location": {}}
        gsub[vid]["location"][qid] = [[5.0, 15.0]]
    return gsub


def _make_frame2time(video_ids: list[str], n_frames: int, duration: float = 30.0) -> dict:
    times = np.linspace(0, duration, n_frames).tolist()
    return {vid: times for vid in video_ids}


def write_causal_features(causal_dir: Path, num_nodes: int = 32, num_clusters: int = 16) -> None:
    causal_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(num_nodes, 768)
    edge_index = torch.tensor(
        [[i, (i + 1) % num_nodes] for i in range(num_nodes)],
        dtype=torch.long,
    ).t().contiguous()
    graph = Data(x=x, edge_index=edge_index)
    torch.save({"k_center": graph}, causal_dir / "qa_graphs.npy")

    clusters = torch.randn(num_clusters, 768)
    torch.save({"k_center": clusters}, causal_dir / "visual_clusters.npy")


def setup_smoke_dataset(root: str | Path = "data", max_feats: int = 8) -> dict[str, Path]:
    """Create tiny train/val/test splits and return key paths."""
    root = Path(root)
    csv_dir = root / "nextgqa"
    feat_dir = root / "nextqa" / "video_feature" / "CLIP_L"
    causal_dir = csv_dir / "causal_feature"

    train_df = _sample_rows(n_videos=2, n_q_per_video=4)
    val_df = _sample_rows(n_videos=1, n_q_per_video=4)
    test_df = _sample_rows(n_videos=1, n_q_per_video=4)

    csv_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(csv_dir / "train.csv", index=False)
    val_df.to_csv(csv_dir / "val.csv", index=False)
    test_df.to_csv(csv_dir / "test.csv", index=False)

    train_vids = sorted(train_df["video_id"].astype(str).unique())
    val_vids = sorted(val_df["video_id"].astype(str).unique())
    test_vids = sorted(test_df["video_id"].astype(str).unique())

    n_frames = max(max_feats, 16)
    _write_h5(feat_dir / "train.h5", train_vids, n_frames=n_frames)
    _write_h5(feat_dir / "val.h5", val_vids, n_frames=n_frames)
    _write_h5(feat_dir / "test.h5", test_vids, n_frames=n_frames)

    for split, df in [("val", val_df), ("test", test_df)]:
        vids = sorted(df["video_id"].astype(str).unique())
        with open(csv_dir / f"gsub_{split}.json", "w") as fp:
            json.dump(_make_gsub(df), fp)
        with open(csv_dir / f"frame2time_{split}.json", "w") as fp:
            json.dump(_make_frame2time(vids, n_frames=n_frames), fp)

    write_causal_features(causal_dir)

    return {
        "csv_dir": csv_dir,
        "feat_dir": feat_dir,
        "causal_dir": causal_dir,
    }
