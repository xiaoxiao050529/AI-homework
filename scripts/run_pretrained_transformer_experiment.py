"""微调预训练中文 Transformer 做情感分类。"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.sentiment_hw2.data import read_split
from src.sentiment_hw2.paths import OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.sentiment_hw2.train import compute_metrics, set_seed


DEFAULT_MODEL_ID = "uer/chinese_roberta_L-4_H-512"
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_PATIENCE = 2

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained Chinese Transformer for sentiment classification.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--quick", action="store_true", help="只跑 1 个 epoch 做快速检查。")
    return parser.parse_args()


def join_tokens(samples: Sequence[Tuple[int, List[str]]]) -> Tuple[List[str], List[int]]:
    texts = ["".join(tokens) for label, tokens in samples]
    labels = [label for label, _ in samples]
    return texts, labels


def build_or_load_features(
    tokenizer,
    model_id: str,
    max_length: int,
) -> Dict[str, TensorDataset]:
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "hf_{}_{}.pt".format(slugify_model_id(model_id), max_length)
    if cache_path.exists():
        return torch.load(cache_path)

    datasets = {}
    for split_name, split_path in (
        ("train", TRAIN_PATH),
        ("validation", VALIDATION_PATH),
        ("test", TEST_PATH),
    ):
        texts, labels = join_tokens(read_split(split_path))
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        datasets[split_name] = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

    torch.save(datasets, cache_path)
    return datasets


def evaluate_model(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    golds = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            preds.append(logits.argmax(dim=1).cpu().numpy())
            golds.append(labels.cpu().numpy())
    predictions = np.concatenate(preds)
    labels = np.concatenate(golds)
    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / float(max(total_count, 1))
    return metrics


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.patience = 1

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    datasets = build_or_load_features(tokenizer, args.model_id, args.max_length)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2)
    model.to(device)

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    optimizer_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(term in n for term in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(term in n for term in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.learning_rate)

    best_state = None
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / float(max(seen, 1))
        validation_metrics = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1": validation_metrics["f1"],
            }
        )
        print(
            "epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_acc=validation_metrics["accuracy"],
                val_f1=validation_metrics["f1"],
            )
        )

        improved = (
            best_metrics is None
            or validation_metrics["accuracy"] > best_metrics["accuracy"]
            or (
                abs(validation_metrics["accuracy"] - best_metrics["accuracy"]) < 1e-8
                and validation_metrics["f1"] > best_metrics["f1"]
            )
        )
        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = validation_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected for pretrained transformer.")

    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = evaluate_model(model, validation_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    train_seconds = time.time() - start_time

    slug = slugify_model_id(args.model_id)
    result = {
        "model_name": "pretrained_transformer",
        "hf_model_id": args.model_id,
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_seconds": train_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "config": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
    }

    metrics_path = OUTPUT_DIR / "pretrained_transformer_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_dir = OUTPUT_DIR / "hf_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir / slug)
    tokenizer.save_pretrained(checkpoint_dir / slug)

    print(
        "done: test_acc={:.4f} test_f1={:.4f} train_seconds={:.2f}".format(
            test_metrics["accuracy"],
            test_metrics["f1"],
            train_seconds,
        )
    )


if __name__ == "__main__":
    main()
