"""Training loop for the DeBERTa-v3-base cross-encoder baseline.

Custom loop (no HF Trainer) for transparency:
  * AdamW with linear warmup (10%) and linear decay to 0
  * Early stopping on val binary F1 with patience=3
  * Checkpoint best model by val binary F1
  * Per-step CSV log + per-epoch CSV log + sanity-check val F1 PNG

Selects device automatically: MPS on Apple Silicon (with float32 cast and
PYTORCH_ENABLE_MPS_FALLBACK=1 — fixes the earlier-noted "Destination
NDArray and Accumulator NDArray cannot have different datatype" issue),
CUDA if available, CPU as fallback.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baselines.fine_tuned.data import (
    CONFLICT_TYPES,
    IGNORE_INDEX,
    CrossEncoderDataset,
    collate,
)
from src.baselines.fine_tuned.model import CrossEncoderConflictDetector


# Defensive MPS fallback for any unsupported op
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    learning_rate: float = 3e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    max_epochs: int = 10
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    patience: int = 3
    typology_weight: float = 1.0
    binary_class_weights: tuple[float, float] = (1.0, 6.6)
    dropout: float = 0.1
    seed: int = 42
    grad_clip_norm: float = 1.0
    log_every: int = 20  # batches


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_binary_f1: float
    val_binary_f1_per_epoch: list[float] = field(default_factory=list)
    val_typology_macro_f1_per_epoch: list[float] = field(default_factory=list)
    val_typology_acc_per_epoch: list[float] = field(default_factory=list)
    train_loss_per_epoch: list[float] = field(default_factory=list)
    train_binary_f1_per_epoch: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    epochs_run: int = 0
    early_stopped: bool = False
    device: str = ""


def _linear_warmup_then_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    """Standard LR schedule: linear warmup, then linear decay to 0."""
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


@torch.no_grad()
def evaluate(
    model: CrossEncoderConflictDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute binary + typology metrics on a held-out loader."""
    model.eval()
    binary_true, binary_pred = [], []
    typology_true, typology_pred = [], []
    total_loss = 0.0
    n_loss_batches = 0
    for batch in loader:
        inp = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # Pass labels so the model returns loss too
        out = model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            token_type_ids=inp.get("token_type_ids"),
            binary_label=inp["binary_label"],
            typology_label=inp["typology_label"],
        )
        if out.total_loss is not None and torch.isfinite(out.total_loss):
            total_loss += float(out.total_loss.detach().cpu())
            n_loss_batches += 1
        binary_pred.extend(out.binary_logits.argmax(dim=-1).cpu().tolist())
        binary_true.extend(inp["binary_label"].cpu().tolist())
        # Typology: only evaluate where the gold is a conflict
        bt = inp["binary_label"].cpu().tolist()
        tt = inp["typology_label"].cpu().tolist()
        tp = out.typology_logits.argmax(dim=-1).cpu().tolist()
        for i, bin_true in enumerate(bt):
            if bin_true == 1 and tt[i] != IGNORE_INDEX:
                typology_true.append(tt[i])
                typology_pred.append(tp[i])
        del out
        if device.type == "mps":
            torch.mps.empty_cache()
    val_loss = total_loss / n_loss_batches if n_loss_batches else float("nan")

    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(binary_true, binary_pred, average="macro", zero_division=0)
    precision = precision_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    recall = recall_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    accuracy = float(np.mean(np.array(binary_true) == np.array(binary_pred))) if binary_true else 0.0

    typology_macro_f1 = None
    typology_acc = None
    if typology_true:
        labels_present = sorted(set(typology_true) | set(typology_pred))
        typology_macro_f1 = f1_score(
            typology_true, typology_pred, labels=labels_present, average="macro", zero_division=0
        )
        typology_acc = float(np.mean(np.array(typology_true) == np.array(typology_pred)))

    return {
        "binary_f1": float(binary_f1),
        "binary_macro_f1": float(macro_f1),
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "accuracy": float(accuracy),
        "typology_macro_f1": typology_macro_f1,
        "typology_accuracy": typology_acc,
        "val_loss": val_loss,
        "n_eval": len(binary_true),
        "n_typology_eval": len(typology_true),
    }


def _open_csv_writer(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def _plot_val_curves(epoch_rows: list[dict], png_path: Path) -> None:
    if not epoch_rows:
        return
    epochs = [r["epoch"] for r in epoch_rows]
    val_f1 = [r["val_binary_f1"] for r in epoch_rows]
    val_typ = [r.get("val_typology_macro_f1") or 0.0 for r in epoch_rows]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
    ax.plot(epochs, val_f1, marker="o", label="val binary F1")
    ax.plot(epochs, val_typ, marker="s", label="val typology macro-F1", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(png_path.stem)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def train(
    train_ds: CrossEncoderDataset,
    val_ds: CrossEncoderDataset,
    tokenizer,
    config: TrainConfig,
    *,
    output_dir: Path | None = None,
    log_prefix: str | None = None,
    verbose: bool = True,
) -> tuple[CrossEncoderConflictDetector, TrainingResult]:
    """Train and return the BEST checkpoint by val binary F1, plus full curves."""
    set_seed(config.seed)
    device = pick_device()

    # Float32 cast on MPS — fixes the "different datatype" runtime error
    model = CrossEncoderConflictDetector(
        binary_class_weights=config.binary_class_weights,
        typology_weight=config.typology_weight,
        dropout=config.dropout,
    )
    if device.type == "mps":
        model = model.to(torch.float32)
    model = model.to(device)

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    # Optimiser + LR schedule
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(config.warmup_ratio * total_steps)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = config.patience
    result = TrainingResult(best_epoch=-1, best_val_binary_f1=0.0, device=str(device))

    # ---- Per-step & per-epoch CSV log setup ----
    step_writer = None
    epoch_writer = None
    step_fh = epoch_fh = None
    epoch_rows: list[dict] = []
    if output_dir is not None:
        prefix = log_prefix or ""
        if prefix and not prefix.endswith("_"):
            prefix = prefix + "_"
        step_path = output_dir / f"{prefix}step_log.csv"
        epoch_path = output_dir / f"{prefix}epoch_log.csv"
        step_fh, step_writer = _open_csv_writer(step_path, [
            "global_step", "epoch", "epoch_fraction",
            "batch_loss", "batch_binary_loss", "batch_typology_loss",
            "learning_rate", "grad_norm",
        ])
        epoch_fh, epoch_writer = _open_csv_writer(epoch_path, [
            "epoch", "train_loss_mean", "train_binary_f1",
            "val_loss", "val_binary_f1", "val_typology_macro_f1", "val_accuracy",
            "learning_rate", "epoch_seconds", "timestamp",
        ])

    start = time.perf_counter()
    global_step = 0
    steps_per_epoch = max(len(train_loader), 1)
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        train_true, train_pred = [], []
        for batch_i, batch in enumerate(train_loader):
            inp = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                token_type_ids=inp.get("token_type_ids"),
                binary_label=inp["binary_label"],
                typology_label=inp["typology_label"],
            )
            loss = out.total_loss
            batch_binary_loss = float(out.binary_loss.detach().cpu()) if out.binary_loss is not None else float("nan")
            batch_typology_loss = float(out.typology_loss.detach().cpu()) if (
                out.typology_loss is not None and torch.isfinite(out.typology_loss)
            ) else float("nan")
            optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            lr_scale = _linear_warmup_then_decay(global_step, total_steps, warmup_steps)
            current_lr = config.learning_rate * lr_scale
            for g in optimiser.param_groups:
                g["lr"] = current_lr
            optimiser.step()
            global_step += 1

            running_loss += loss.item()
            train_pred.extend(out.binary_logits.argmax(dim=-1).cpu().tolist())
            train_true.extend(inp["binary_label"].cpu().tolist())

            # Per-step log (raw values — smoothing happens at plot time)
            if step_writer is not None:
                step_writer.writerow({
                    "global_step": global_step,
                    "epoch": epoch,
                    "epoch_fraction": round(epoch - 1 + (batch_i + 1) / steps_per_epoch, 4),
                    "batch_loss": loss.item(),
                    "batch_binary_loss": batch_binary_loss,
                    "batch_typology_loss": batch_typology_loss,
                    "learning_rate": current_lr,
                    "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                })

            # Free intermediate tensors before MPS cache cleanup
            del out, loss
            if device.type == "mps":
                # Periodic flush — MPS doesn't free aggressively
                if (batch_i + 1) % 8 == 0:
                    torch.mps.empty_cache()

            if verbose and (batch_i + 1) % config.log_every == 0:
                avg_loss = running_loss / (batch_i + 1)
                print(f"    epoch {epoch} step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}", flush=True)

        train_loss = running_loss / max(len(train_loader), 1)
        train_bin_f1 = f1_score(train_true, train_pred, pos_label=1, zero_division=0)

        # Free training-loop memory before validation
        if device.type == "mps":
            torch.mps.empty_cache()

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["binary_f1"]
        val_typ = val_metrics["typology_macro_f1"]
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["accuracy"]
        epoch_seconds = time.perf_counter() - epoch_start

        result.train_loss_per_epoch.append(float(train_loss))
        result.train_binary_f1_per_epoch.append(float(train_bin_f1))
        result.val_binary_f1_per_epoch.append(float(val_f1))
        result.val_typology_macro_f1_per_epoch.append(
            float(val_typ) if val_typ is not None else 0.0
        )
        result.val_typology_acc_per_epoch.append(
            float(val_metrics["typology_accuracy"]) if val_metrics["typology_accuracy"] is not None else 0.0
        )
        result.epochs_run = epoch

        # Per-epoch CSV row
        if epoch_writer is not None:
            epoch_row = {
                "epoch": epoch,
                "train_loss_mean": float(train_loss),
                "train_binary_f1": float(train_bin_f1),
                "val_loss": float(val_loss) if not math.isnan(val_loss) else None,
                "val_binary_f1": float(val_f1),
                "val_typology_macro_f1": float(val_typ) if val_typ is not None else None,
                "val_accuracy": float(val_acc),
                "learning_rate": current_lr if epoch == 1 else (
                    config.learning_rate * _linear_warmup_then_decay(global_step, total_steps, warmup_steps)
                ),
                "epoch_seconds": round(epoch_seconds, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            epoch_writer.writerow(epoch_row)
            epoch_fh.flush()
            step_fh.flush()
            epoch_rows.append(epoch_row)

        if verbose:
            print(
                f"  epoch {epoch}: train loss={train_loss:.4f}, train F1={train_bin_f1:.3f} | "
                f"val F1={val_f1:.3f}, val typology F1={val_typ}",
                flush=True,
            )

        # Overfitting alarm: train F1 >= 0.95 and val F1 < 0.4 by epoch >= 3
        if epoch >= 3 and train_bin_f1 >= 0.95 and val_f1 < 0.4:
            print(
                f"  WARNING: possible overfitting (train F1={train_bin_f1:.3f}, val F1={val_f1:.3f})",
                flush=True,
            )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                result.early_stopped = True
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.best_epoch = best_epoch
    result.best_val_binary_f1 = float(best_val_f1)
    result.elapsed_seconds = time.perf_counter() - start

    # Close CSVs + generate sanity-check PNG
    if step_fh is not None:
        step_fh.close()
    if epoch_fh is not None:
        epoch_fh.close()
    if output_dir is not None and epoch_rows:
        prefix = log_prefix or ""
        if prefix and not prefix.endswith("_"):
            prefix = prefix + "_"
        png_path = output_dir / f"{prefix}val_f1.png"
        _plot_val_curves(epoch_rows, png_path)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Persist training curves + final result (small file)
        (output_dir / "training_result.json").write_text(json.dumps({
            "config": config.__dict__,
            "best_epoch": result.best_epoch,
            "best_val_binary_f1": result.best_val_binary_f1,
            "epochs_run": result.epochs_run,
            "early_stopped": result.early_stopped,
            "device": result.device,
            "elapsed_seconds": result.elapsed_seconds,
            "train_loss_per_epoch": result.train_loss_per_epoch,
            "train_binary_f1_per_epoch": result.train_binary_f1_per_epoch,
            "val_binary_f1_per_epoch": result.val_binary_f1_per_epoch,
            "val_typology_macro_f1_per_epoch": result.val_typology_macro_f1_per_epoch,
            "val_typology_acc_per_epoch": result.val_typology_acc_per_epoch,
        }, indent=2))

    return model, result


@torch.no_grad()
def predict(
    model: CrossEncoderConflictDetector,
    dataset: CrossEncoderDataset,
    tokenizer,
    *,
    batch_size: int = 32,
) -> list[dict]:
    """Generate predictions in the harness JSONL schema."""
    device = pick_device()
    model = model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )
    out_records = []
    for batch in loader:
        inp = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            token_type_ids=inp.get("token_type_ids"),
        )
        binary_probs = torch.softmax(out.binary_logits, dim=-1).cpu().numpy()
        binary_preds = out.binary_logits.argmax(dim=-1).cpu().tolist()
        typology_preds = out.typology_logits.argmax(dim=-1).cpu().tolist()
        for i, pid in enumerate(batch["pair_id"]):
            cp = bool(binary_preds[i])
            ct = CONFLICT_TYPES[typology_preds[i]] if cp else None
            out_records.append({
                "pair_id": pid,
                "predicted_conflict_present": cp,
                "predicted_conflict_type": ct,
                "confidence_score": float(binary_probs[i, 1]),
            })
    return out_records
