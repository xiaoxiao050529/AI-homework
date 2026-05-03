"""对比固定迭代次数与验证集早停策略的专门实验。"""

import json
import os
import time
from math import cos, pi
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
from torch import nn

from src.sentiment_hw2.data import prepare_data
from src.sentiment_hw2.experiment import build_main_model_specs
from src.sentiment_hw2.models import initialize_model
from src.sentiment_hw2.train import create_dataloaders, evaluate, set_seed


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
SUMMARY_PATH = OUTPUT_DIR / "stopping_strategy_summary.json"

TARGET_MODELS = {"mlp", "cnn"}
FIXED_EPOCH_CANDIDATES = [1, 2, 4, 6, 8, 10]
EARLY_STOPPING_PATIENCES = [1, 2, 3]


def _build_optimizer(model: nn.Module, learning_rate: float, embedding_learning_rate: float, weight_decay: float):
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("embedding."):
            embedding_params.append(param)
        else:
            other_params.append(param)

    optimizer_groups = []
    if embedding_params:
        optimizer_groups.append({"params": embedding_params, "lr": embedding_learning_rate or learning_rate})
    if other_params:
        optimizer_groups.append({"params": other_params, "lr": learning_rate})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)
    return optimizer, bool(embedding_params)


def _warmup_cosine_scale(epoch: int, warmup_epochs: int, total_epochs: int, min_lr_scale: float) -> float:
    warmup_epochs = max(warmup_epochs, 0)
    min_lr_scale = min(max(min_lr_scale, 0.0), 1.0)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return 0.35 + 0.65 * float(epoch) / float(warmup_epochs)

    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress = float(epoch - warmup_epochs) / float(cosine_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine_scale = 0.5 * (1.0 + cos(pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine_scale


def train_full_trajectory(spec, prepared, device: torch.device) -> Dict[str, object]:
    """训练到最大轮数，并记录每一轮在验证/测试集上的完整指标。"""
    config = spec.train_config
    set_seed(config.seed)

    model = spec.builder()
    initialize_model(model)
    model.to(device)

    loaders = create_dataloaders(prepared, config.batch_size)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer, has_embedding = _build_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        embedding_learning_rate=config.embedding_learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=False
        )
    else:
        lr_scheduler = None

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        print("[{}] epoch {}/{}".format(spec.name, epoch, config.epochs), flush=True)
        embedding_trainable = epoch > config.freeze_embedding_epochs
        if hasattr(model, "embedding"):
            for parameter in model.embedding.parameters():
                parameter.requires_grad = embedding_trainable

        if config.scheduler == "warmup_cosine":
            lr_scale = _warmup_cosine_scale(
                epoch=epoch,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epochs,
                min_lr_scale=config.min_learning_rate_scale,
            )
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group["lr"] = base_lr * lr_scale

        model.train()
        running_loss = 0.0
        seen = 0

        for inputs, lengths, labels in loaders["train"]:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = train_criterion(logits, labels)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / float(max(seen, 1))
        validation_metrics = evaluate(model, loaders["validation"], eval_criterion, device)
        test_metrics = evaluate(model, loaders["test"], eval_criterion, device)
        if config.scheduler == "plateau":
            lr_scheduler.step(validation_metrics["f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation_metrics,
                "test": test_metrics,
                "lr": max(group["lr"] for group in optimizer.param_groups),
                "embedding_lr": optimizer.param_groups[0]["lr"] if has_embedding else 0.0,
                "embedding_trainable": embedding_trainable,
                "epoch_seconds": time.time() - epoch_start,
            }
        )

    return {
        "model_name": spec.name,
        "max_epochs": config.epochs,
        "train_config": {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "grad_clip": config.grad_clip,
            "seed": config.seed,
            "embedding_learning_rate": config.embedding_learning_rate,
            "label_smoothing": config.label_smoothing,
            "warmup_epochs": config.warmup_epochs,
            "min_epochs": config.min_epochs,
            "freeze_embedding_epochs": config.freeze_embedding_epochs,
            "scheduler": config.scheduler,
        },
        "history": history,
        "full_train_seconds": time.time() - start_time,
    }


def _is_better(candidate: Dict[str, object], current_best: Dict[str, object]) -> bool:
    if current_best is None:
        return True
    candidate_f1 = candidate["validation"]["f1"]
    current_f1 = current_best["validation"]["f1"]
    if candidate_f1 > current_f1:
        return True
    if abs(candidate_f1 - current_f1) < 1e-8 and candidate["validation"]["accuracy"] > current_best["validation"]["accuracy"]:
        return True
    return False


def summarize_fixed_epochs(history: List[Dict[str, object]], max_epochs: int) -> List[Dict[str, object]]:
    candidates = [epoch for epoch in FIXED_EPOCH_CANDIDATES if epoch <= max_epochs]
    results = []
    for epoch in candidates:
        record = history[epoch - 1]
        results.append(
            {
                "strategy": "fixed_epoch",
                "stop_epoch": epoch,
                "selected_epoch": epoch,
                "validation_accuracy": record["validation"]["accuracy"],
                "validation_f1": record["validation"]["f1"],
                "test_accuracy": record["test"]["accuracy"],
                "test_f1": record["test"]["f1"],
                "train_seconds": round(sum(item["epoch_seconds"] for item in history[:epoch]), 4),
            }
        )
    return results


def summarize_validation_selection(history: List[Dict[str, object]], min_epochs: int, max_epochs: int) -> Dict[str, object]:
    best_record = None
    for record in history:
        if _is_better(record, best_record):
            best_record = record
    return {
        "strategy": "validation_best_checkpoint",
        "stop_epoch": max_epochs,
        "selected_epoch": best_record["epoch"],
        "validation_accuracy": best_record["validation"]["accuracy"],
        "validation_f1": best_record["validation"]["f1"],
        "test_accuracy": best_record["test"]["accuracy"],
        "test_f1": best_record["test"]["f1"],
        "train_seconds": round(sum(item["epoch_seconds"] for item in history), 4),
        "min_epochs": min_epochs,
    }


def summarize_early_stopping(history: List[Dict[str, object]], min_epochs: int) -> List[Dict[str, object]]:
    results = []
    for patience in EARLY_STOPPING_PATIENCES:
        best_record = None
        best_epoch = 0
        epochs_without_improvement = 0
        stop_epoch = len(history)

        for record in history:
            if _is_better(record, best_record):
                best_record = record
                best_epoch = record["epoch"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if record["epoch"] >= min_epochs and epochs_without_improvement >= patience:
                stop_epoch = record["epoch"]
                break

        results.append(
            {
                "strategy": "validation_early_stopping",
                "patience": patience,
                "stop_epoch": stop_epoch,
                "selected_epoch": best_epoch,
                "validation_accuracy": best_record["validation"]["accuracy"],
                "validation_f1": best_record["validation"]["f1"],
                "test_accuracy": best_record["test"]["accuracy"],
                "test_f1": best_record["test"]["f1"],
                "train_seconds": round(sum(item["epoch_seconds"] for item in history[:stop_epoch]), 4),
                "min_epochs": min_epochs,
            }
        )
    return results


def main() -> None:
    set_seed(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(
        train_path=ROOT / "train.txt",
        validation_path=ROOT / "validation.txt",
        test_path=ROOT / "test.txt",
        embedding_path=ROOT / "wiki_word2vec_50.bin",
        cache_path=CACHE_DIR / "prepared_data.pt",
        max_len=80,
        min_freq=1,
        seed=42,
    )

    trajectories = []
    for spec in build_main_model_specs(prepared.embedding_matrix):
        if spec.name not in TARGET_MODELS:
            continue
        trajectories.append(train_full_trajectory(spec, prepared, device))

    summary = {
        "device": str(device),
        "data": prepared.metadata,
        "methodology": {
            "target_models": sorted(TARGET_MODELS),
            "fixed_epoch_candidates": FIXED_EPOCH_CANDIDATES,
            "early_stopping_patiences": EARLY_STOPPING_PATIENCES,
            "note": "测试集指标仅用于事后比较不同停止策略，不参与实际停止决策。",
        },
        "results": [],
    }

    for trajectory in trajectories:
        history = trajectory["history"]
        train_config = trajectory["train_config"]
        result = {
            "model_name": trajectory["model_name"],
            "max_epochs": trajectory["max_epochs"],
            "full_train_seconds": trajectory["full_train_seconds"],
            "train_config": train_config,
            "fixed_epoch_results": summarize_fixed_epochs(history, trajectory["max_epochs"]),
            "validation_best_checkpoint": summarize_validation_selection(
                history=history,
                min_epochs=train_config["min_epochs"],
                max_epochs=trajectory["max_epochs"],
            ),
            "early_stopping_results": summarize_early_stopping(
                history=history,
                min_epochs=train_config["min_epochs"],
            ),
            "history": history,
        }
        summary["results"].append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Saved:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
