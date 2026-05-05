"""训练、评估、早停与指标保存逻辑。"""

import copy
import json
import random
import time
from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PreparedData, TensorSplitDataset
from .models import count_parameters


@dataclass
class TrainConfig:
    """单个模型训练时需要的超参数配置。"""
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    grad_clip: float
    seed: int
    embedding_learning_rate: Optional[float] = None
    label_smoothing: float = 0.0
    warmup_epochs: int = 0
    min_epochs: int = 1
    freeze_embedding_epochs: int = 0
    scheduler: str = "plateau"
    min_learning_rate_scale: float = 0.2


def set_seed(seed: int) -> None:
    """同步固定 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(prepared: PreparedData, batch_size: int) -> Dict[str, DataLoader]:
    """为训练、验证、测试集构建 DataLoader。"""
    return {
        # 只有训练集需要打乱，验证和测试保持稳定顺序即可。
        "train": DataLoader(TensorSplitDataset(prepared.train), batch_size=batch_size, shuffle=True),
        "validation": DataLoader(TensorSplitDataset(prepared.validation), batch_size=batch_size, shuffle=False),
        "test": DataLoader(TensorSplitDataset(prepared.test), batch_size=batch_size, shuffle=False),
    }


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """基于预测标签和真实标签计算分类指标。"""
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    accuracy = float((predictions == labels).mean())
    precision = tp / float(max(tp + fp, 1))
    recall = tp / float(max(tp + fn, 1))
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """在验证集或测试集上关闭梯度进行完整评估。"""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, lengths, labels in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 先拼接所有 batch 的结果，再统一计算整套指标。
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / float(max(total_count, 1))
    return metrics


def train_single_model(
    model_name: str,
    model: nn.Module,
    prepared: PreparedData,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    """训练单个模型，并保存最优 checkpoint 与完整指标。"""
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "{}_best.pt".format(model_name)

    loaders = create_dataloaders(prepared, config.batch_size)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("embedding."):
            embedding_params.append(param)
        else:
            other_params.append(param)

    optimizer_groups = []
    if embedding_params:
        optimizer_groups.append(
            {
                "params": embedding_params,
                "lr": config.embedding_learning_rate or config.learning_rate,
            }
        )
    if other_params:
        optimizer_groups.append({"params": other_params, "lr": config.learning_rate})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)

    if config.scheduler == "plateau":
        # 监控验证集 F1，长时间不提升时自动降低学习率。
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=False
        )
    else:
        lr_scheduler = None

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def warmup_cosine_scale(epoch: int) -> float:
        warmup_epochs = max(config.warmup_epochs, 0)
        min_lr_scale = min(max(config.min_learning_rate_scale, 0.0), 1.0)
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return 0.35 + 0.65 * float(epoch) / float(warmup_epochs)
        cosine_epochs = max(config.epochs - warmup_epochs, 1)
        progress = float(epoch - warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + cos(pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine_scale

    model.to(device)
    # 先拷贝一份初始状态，保证极端情况下也有可恢复参数。
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        embedding_trainable = epoch > config.freeze_embedding_epochs
        if hasattr(model, "embedding"):
            for parameter in model.embedding.parameters():
                parameter.requires_grad = embedding_trainable
        if config.scheduler == "warmup_cosine":
            lr_scale = warmup_cosine_scale(epoch)
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
                # 循环模型更容易出现梯度爆炸，因此支持按配置裁剪。
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / float(max(seen, 1))
        validation_metrics = evaluate(model, loaders["validation"], eval_criterion, device)
        if config.scheduler == "plateau":
            lr_scheduler.step(validation_metrics["f1"])

        epoch_record = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_start_time,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_f1": validation_metrics["f1"],
            "lr": max(group["lr"] for group in optimizer.param_groups),
            "embedding_lr": optimizer.param_groups[0]["lr"] if embedding_params else 0.0,
            "embedding_trainable": embedding_trainable,
        }
        history.append(epoch_record)

        improved = (
            best_metrics is None
            or validation_metrics["f1"] > best_metrics["f1"]
            or (
                # F1 相同时再比较 Accuracy，避免出现“最佳轮次”不稳定。
                abs(validation_metrics["f1"] - best_metrics["f1"]) < 1e-8
                and validation_metrics["accuracy"] > best_metrics["accuracy"]
            )
        )

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        # 连续若干轮无提升就提前停止，减少无效训练和过拟合风险。
        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
            break

    # 最终统一回滚到验证集表现最好的参数，再做验证/测试汇报。
    model.load_state_dict(best_state)
    validation_metrics = evaluate(model, loaders["validation"], eval_criterion, device)
    test_metrics = evaluate(model, loaders["test"], eval_criterion, device)
    duration = time.time() - start_time

    result = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "parameter_count": count_parameters(model),
        "train_seconds": duration,
        "epochs_ran": len(history),
        "seconds_per_epoch": duration / float(max(len(history), 1)),
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }

    # 逐模型保存独立指标文件，便于后续报告脚本或人工检查直接读取。
    with (output_dir / "{}_metrics.json".format(model_name)).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    return result
