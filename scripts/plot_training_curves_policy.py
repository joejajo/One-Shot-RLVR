#!/usr/bin/env python3
"""
plot_training_curves.py — One-Shot-RLVR thesis visualization
=============================================================
Generates 5 publication-quality figures from TensorBoard event files
and/or JSONL shard files produced by the training pipeline.

Usage
-----
  # Auto-discover latest TensorBoard run:
  python scripts/plot_training_curves.py

  # Explicit path:
  python scripts/plot_training_curves.py \
      --tb_dir  output/tensorboard/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128_4xa100 \
      --out_dir output/plots

Outputs (saved to --out_dir)
--------
  fig1_train_reward.png
  fig2_val_accuracy.png
  fig3_combined_reward.png
  fig4_response_length.png
  fig5_entropy.png
  all_metrics.csv          <- raw extracted data for further analysis

Requirements
------------
  pip install tensorboard matplotlib pandas scipy numpy
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed on HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Style constants — premium thesis look
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_COLOR  = "#4C9BE8"   # cool blue
VAL_COLOR    = "#E8854C"   # warm orange
ENTR_COLOR   = "#8EC15A"   # sage green
LEN_COLOR    = "#B07BE8"   # soft violet
GRID_ALPHA   = 0.25
SMOOTH_FRAC  = 0.08        # EMA fraction for smoothing (0 = no smoothing)

plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.family":         "sans-serif",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.alpha":          GRID_ALPHA,
    "legend.framealpha":   0.85,
    "lines.linewidth":     1.8,
    "figure.figsize":      (8, 4.5),
})

# ─────────────────────────────────────────────────────────────────────────────
# Metric name candidates (ordered by preference — first match wins)
# These are the actual tag names that ray_trainer.py writes to TensorBoard
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_REWARD_CANDIDATES = [
    "critic/rewards/mean",
    "critic/score/mean",
    "train/reward/mean",
    "reward/mean",
]

VAL_REWARD_CANDIDATES = [
    "val/accuracy/mean",
    "val/test_score/math500",
    "val/test_score/default",
    "val/test_score/unknown",
    "val/reward/mean",
    "validation/reward/mean",
]

RESPONSE_LEN_CANDIDATES = [
    "response_length/mean",
    "val/response_length/mean",
    "rollout/response_length/mean",
    "global_response_length/mean",
]

POLICY_LOSS_CANDIDATES = [
    "actor/policy_loss",
    "actor/pg_loss",
    "policy_loss",
]


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard reader
# ─────────────────────────────────────────────────────────────────────────────

def read_tb_events(tb_dir: str) -> dict:
    """
    Parse all TensorBoard event files under tb_dir and return
    a dict: { tag_name: pd.DataFrame(columns=['step', 'value']) }
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] tensorboard package not found. Install with: pip install tensorboard")
        sys.exit(1)

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        data[tag] = pd.DataFrame({"step": steps, "value": values}).sort_values("step").reset_index(drop=True)

    print(f"[TensorBoard] loaded {len(tags)} scalar tags from {tb_dir}")
    return data


def find_tag(tb_data: dict, candidates: list):
    """Return (tag_name, DataFrame) for the first candidate found, else (None, None)."""
    for c in candidates:
        if c in tb_data:
            return c, tb_data[c]
    # fuzzy: any tag containing a candidate keyword
    for c in candidates:
        keyword = c.split("/")[-1]  # e.g. "mean"
        for k in tb_data:
            if keyword in k and ("reward" in k or "score" in k or "accuracy" in k
                                  or "entropy" in k or "response_length" in k):
                return k, tb_data[k]
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# JSONL reader (fallback / supplement)
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl_shards(jsonl_dir: str, prefix: str) -> pd.DataFrame:
    """
    Load all shard files matching <jsonl_dir>/<prefix>_steps_*.jsonl
    and return a DataFrame with columns: step, score (aggregated mean per step).
    """
    pattern = os.path.join(jsonl_dir, f"{prefix}_steps_*.jsonl")
    files   = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    rows = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Aggregate per step
    agg = df.groupby("step")["score"].mean().reset_index()
    agg.columns = ["step", "value"]
    return agg.sort_values("step").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def ema_smooth(series: pd.Series, frac: float = SMOOTH_FRAC) -> pd.Series:
    """Exponential moving average smoothing."""
    if frac <= 0:
        return series
    alpha = 1.0 - np.exp(-frac * 10)   # map frac -> EMA alpha
    return series.ewm(alpha=alpha, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Individual figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _add_shaded_band(ax, df, color, alpha=0.15):
    """Draw ±std shading if data has enough points."""
    if len(df) < 5:
        return
    window = max(5, len(df) // 20)
    roll   = df["value"].rolling(window=window, center=True)
    mean_  = roll.mean()
    std_   = roll.std()
    ax.fill_between(df["step"], mean_ - std_, mean_ + std_,
                    color=color, alpha=alpha, linewidth=0)


def fig1_train_reward(tb_data, jsonl_dir, out_dir):
    """Graph 1 — Training reward vs steps."""
    tag, df = find_tag(tb_data, TRAIN_REWARD_CANDIDATES)
    if df is None and jsonl_dir:
        df  = read_jsonl_shards(jsonl_dir, "train")
        tag = "train JSONL"

    if df is None or df.empty:
        print("[WARN] No training reward data found — skipping Graph 1")
        return

    fig, ax = plt.subplots()
    raw    = df["value"]
    smooth = ema_smooth(raw)

    _add_shaded_band(ax, df, TRAIN_COLOR)
    ax.plot(df["step"], raw,    color=TRAIN_COLOR, alpha=0.35, linewidth=1, label="_nolegend_")
    ax.plot(df["step"], smooth, color=TRAIN_COLOR, linewidth=2.2, label=f"{tag}")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Training Reward")
    ax.set_title("Graph 1 — Training Reward vs Steps")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.text(0.5, -0.06,
             "Training reward during one-shot GRPO fine-tuning.\n"
             "The model is trained on repeated samples of a single quadratic reasoning problem;\n"
             "reward measures whether the generated solution satisfies the verifier.",
             ha="center", fontsize=8.5, color="#555555", style="italic",
             transform=ax.transAxes)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_train_reward.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def fig2_val_accuracy(tb_data, jsonl_dir, out_dir):
    """Graph 2 — Evaluation accuracy vs steps."""
    # Collect all val/test_score/* tags so we can plot per-source lines too
    val_tags = {k: v for k, v in tb_data.items()
                if k.startswith("val/test_score/") or k == "val/accuracy/mean"}

    tag, df = find_tag(tb_data, VAL_REWARD_CANDIDATES)
    if df is None and jsonl_dir:
        df  = read_jsonl_shards(jsonl_dir, "val")
        tag = "val JSONL"

    if df is None or df.empty:
        print("[WARN] No validation reward data found — skipping Graph 2")
        return

    fig, ax = plt.subplots()

    # Plot aggregate line (bold)
    raw    = df["value"]
    smooth = ema_smooth(raw)
    _add_shaded_band(ax, df, VAL_COLOR)
    ax.plot(df["step"], raw,    color=VAL_COLOR, alpha=0.35, linewidth=1, label="_nolegend_")
    ax.plot(df["step"], smooth, color=VAL_COLOR, linewidth=2.4, label=f"Eval accuracy ({tag})")

    # Plot per-source sub-lines if present
    source_colors = plt.cm.Set2(np.linspace(0, 1, max(1, len(val_tags))))
    for i, (t, sdf) in enumerate(val_tags.items()):
        if t == tag:
            continue   # already plotted as main
        ax.plot(sdf["step"], ema_smooth(sdf["value"]),
                linewidth=1.2, linestyle="--", alpha=0.75,
                color=source_colors[i % len(source_colors)],
                label=t.replace("val/test_score/", "source: "))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Evaluation Accuracy / Reward")
    ax.set_title("Graph 2 — Evaluation Accuracy vs Steps")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    fig.text(0.5, -0.08,
             "Evaluation reward on held-out problems during GRPO training.\n"
             "This curve indicates whether one-shot RL improves general problem-solving\n"
             "or mainly memorizes the single training instance.",
             ha="center", fontsize=8.5, color="#555555", style="italic",
             transform=ax.transAxes)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_val_accuracy.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def fig3_combined(tb_data, jsonl_dir, out_dir):
    """Graph 3 — Training + Evaluation reward on the same axes."""
    tag_tr, df_tr = find_tag(tb_data, TRAIN_REWARD_CANDIDATES)
    tag_va, df_va = find_tag(tb_data, VAL_REWARD_CANDIDATES)

    if df_tr is None and jsonl_dir:
        df_tr  = read_jsonl_shards(jsonl_dir, "train")
        tag_tr = "Train (JSONL)"
    if df_va is None and jsonl_dir:
        df_va  = read_jsonl_shards(jsonl_dir, "val")
        tag_va = "Val (JSONL)"

    if (df_tr is None or df_tr.empty) and (df_va is None or df_va.empty):
        print("[WARN] No data for combined graph — skipping Graph 3")
        return

    fig, ax = plt.subplots()

    if df_tr is not None and not df_tr.empty:
        s = ema_smooth(df_tr["value"])
        ax.plot(df_tr["step"], df_tr["value"], color=TRAIN_COLOR, alpha=0.25, linewidth=1)
        ax.plot(df_tr["step"], s, color=TRAIN_COLOR, linewidth=2.4,
                label=f"Training reward  ({tag_tr})")

    if df_va is not None and not df_va.empty:
        s = ema_smooth(df_va["value"])
        ax.plot(df_va["step"], df_va["value"], color=VAL_COLOR, alpha=0.25, linewidth=1,
                marker="o", markersize=3)
        ax.plot(df_va["step"], s, color=VAL_COLOR, linewidth=2.4,
                label=f"Eval accuracy    ({tag_va})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward / Accuracy")
    ax.set_title("Graph 3 — Training vs Evaluation Reward")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # Annotation box explaining interpretations
    interp = ("↑ train + ↑ eval  → generalization\n"
              "↑ train + → eval  → overfitting\n"
              "↑ train (noisy)   → unstable GRPO")
    ax.text(0.97, 0.05, interp, transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_combined_reward.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def fig4_response_length(tb_data, jsonl_dir, out_dir):
    """Graph 4 — Response length vs steps (train + val if both available)."""
    # Training response length
    tag_tr, df_tr = find_tag(tb_data, RESPONSE_LEN_CANDIDATES[:1])  # response_length/mean
    # Validation response length
    tag_va, df_va = find_tag(tb_data, ["val/response_length/mean"])

    if df_tr is None or df_tr.empty:
        print("[WARN] No response_length/mean data found — skipping Graph 4")
        return

    fig, ax = plt.subplots()

    if df_tr is not None and not df_tr.empty:
        s = ema_smooth(df_tr["value"])
        ax.plot(df_tr["step"], df_tr["value"], color=LEN_COLOR, alpha=0.3, linewidth=1)
        ax.plot(df_tr["step"], s, color=LEN_COLOR, linewidth=2.4,
                label="Train response length")

    if df_va is not None and not df_va.empty:
        s = ema_smooth(df_va["value"])
        ax.plot(df_va["step"], df_va["value"], color=VAL_COLOR, alpha=0.3,
                linewidth=1, marker="o", markersize=3)
        ax.plot(df_va["step"], s, color=VAL_COLOR, linewidth=2.0, linestyle="--",
                label="Val response length")

    # Also check clip ratio
    _, df_clip = find_tag(tb_data, ["response_length/clip_ratio", "val/response_length/clip_ratio"])
    if df_clip is not None and not df_clip.empty:
        ax2 = ax.twinx()
        ax2.plot(df_clip["step"], df_clip["value"], color="#E87070", linewidth=1.4,
                 linestyle=":", alpha=0.7, label="Clip ratio (right)")
        ax2.set_ylabel("Clip Ratio", color="#E87070", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#E87070", labelsize=8)
        ax2.set_ylim(0, 1)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Average Response Length (tokens)")
    ax.set_title("Graph 4 — Response Length vs Steps")
    ax.legend(fontsize=9, loc="upper left")

    fig.text(0.5, -0.07,
             "Average response length during GRPO training.\n"
             "Increasing length suggests the model learns to produce longer reasoning traces.\n"
             "High clip ratio indicates many responses hit the max_response_length cap.",
             ha="center", fontsize=8.5, color="#555555", style="italic",
             transform=ax.transAxes)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_response_length.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def fig5_policy_loss(tb_data, out_dir):
    """Graph 5 — Policy Loss vs steps."""
    tag, df = find_tag(tb_data, POLICY_LOSS_CANDIDATES)

    if df is None or df.empty:
        print("[WARN] No policy loss data found — skipping Graph 5")
        return

    fig, ax = plt.subplots()
    s = ema_smooth(df["value"])
    _add_shaded_band(ax, df, ENTR_COLOR)
    ax.plot(df["step"], df["value"],  color=ENTR_COLOR, alpha=0.3, linewidth=1)
    ax.plot(df["step"], s,            color=ENTR_COLOR, linewidth=2.4, label=f"Policy Loss  ({tag})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Graph 5 — Policy Loss vs Steps")
    ax.legend(fontsize=9)

    fig.text(0.5, -0.07,
             "Policy loss during GRPO training.\n"
             "Negative values indicate improvements in policy performance.",
             ha="center", fontsize=8.5, color="#555555", style="italic",
             transform=ax.transAxes)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_policy_loss.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(tb_data: dict, out_dir: str):
    """Dump all extracted scalar data to a single wide CSV."""
    frames = []
    for tag, df in tb_data.items():
        tmp = df[["step", "value"]].copy()
        tmp.rename(columns={"value": tag}, inplace=True)
        frames.append(tmp.set_index("step"))

    if not frames:
        return

    combined = pd.concat(frames, axis=1).sort_index()
    path = os.path.join(out_dir, "all_metrics.csv")
    combined.to_csv(path)
    print(f"[OK] Saved: {path}  ({combined.shape[0]} steps × {combined.shape[1]} metrics)")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def auto_find_tb_dir(repo_root: str) -> str:
    """Walk output/tensorboard/ and return the newest non-empty event dir."""
    base = os.path.join(repo_root, "output", "tensorboard")
    if not os.path.isdir(base):
        return None
    candidates = []
    for dirpath, _, files in os.walk(base):
        if any(f.startswith("events.out") for f in files):
            mtime = max(os.path.getmtime(os.path.join(dirpath, f))
                        for f in files if f.startswith("events.out"))
            candidates.append((mtime, dirpath))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def auto_find_jsonl_dir(repo_root: str) -> str:
    base = os.path.join(repo_root, "output", "model_outputs")
    return base if os.path.isdir(base) else None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate thesis plots from TensorBoard logs")
    p.add_argument("--tb_dir",   default=None,
                   help="Path to TensorBoard log directory (auto-detected if omitted)")
    p.add_argument("--jsonl_dir", default=None,
                   help="Path to JSONL shard folder (auto-detected if omitted)")
    p.add_argument("--out_dir",  default=None,
                   help="Where to save PNG figures (default: output/plots)")
    p.add_argument("--smooth",   type=float, default=SMOOTH_FRAC,
                   help=f"EMA smoothing fraction 0–1 (default: {SMOOTH_FRAC}). 0 = off")
    p.add_argument("--graphs",   default="1,2,3,4,5",
                   help="Comma-separated list of graphs to generate (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    # Repo root = directory of this script's parent
    repo_root = Path(__file__).resolve().parent.parent
    print(f"[Info] Repo root: {repo_root}")

    # Resolve paths
    tb_dir = args.tb_dir or auto_find_tb_dir(str(repo_root))
    jsonl_dir = args.jsonl_dir or auto_find_jsonl_dir(str(repo_root))
    out_dir = args.out_dir or os.path.join(str(repo_root), "output", "plots")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] TensorBoard dir : {tb_dir}")
    print(f"[Info] JSONL dir       : {jsonl_dir}")
    print(f"[Info] Output dir      : {out_dir}")

    # Update smoothing globally
    global SMOOTH_FRAC
    SMOOTH_FRAC = args.smooth

    # Load TensorBoard data
    tb_data = {}
    if tb_dir and os.path.isdir(tb_dir):
        tb_data = read_tb_events(tb_dir)
        if tb_data:
            print("[TensorBoard] Available tags:")
            for t in sorted(tb_data.keys()):
                print(f"    {t}  ({len(tb_data[t])} points)")
    else:
        print("[WARN] TensorBoard directory not found — will fall back to JSONL only")

    # Select which graphs to produce
    wanted = set(args.graphs.split(","))

    if "1" in wanted:
        fig1_train_reward(tb_data, jsonl_dir, out_dir)
    if "2" in wanted:
        fig2_val_accuracy(tb_data, jsonl_dir, out_dir)
    if "3" in wanted:
        fig3_combined(tb_data, jsonl_dir, out_dir)
    if "4" in wanted:
        fig4_response_length(tb_data, jsonl_dir, out_dir)
    if "5" in wanted:
        fig5_policy_loss(tb_data, out_dir)

    export_csv(tb_data, out_dir)
    print("\n✓ All done. Figures saved to:", out_dir)


if __name__ == "__main__":
    main()
