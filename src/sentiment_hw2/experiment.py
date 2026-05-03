"""实验配置、模型命名与简易推理辅助函数。"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
from torch import nn

from .data import PreparedData
from .models import BiGRUClassifier, BiLSTMClassifier, BiRNNClassifier, MeanPoolMLP, TextCNN
from .train import TrainConfig


MODEL_LABELS = {
    "mlp": "MLP",
    "cnn": "TextCNN",
    "birnn": "BiRNN",
    "bilstm": "BiLSTM",
    "bigru": "BiGRU",
    "cnn_num_filters_64": "TextCNN (filters=64)",
    "birnn_hidden_64": "BiRNN (hidden=64)",
    "bilstm_hidden_64": "BiLSTM (hidden=64)",
    "bigru_hidden_64": "BiGRU (hidden=64)",
}

ROBUSTNESS_SAMPLES = [
    {"label": 1, "text": "剧情 紧凑 节奏 明快 演员 表现 很 出色 这部 电影 值得 推荐"},
    {"label": 0, "text": "故事 拖沓 剪辑 混乱 看 完 只 想 立刻 退场"},
    {"label": 1, "text": "虽然 成本 不高 但是 情感 真挚 细节 打动 人"},
    {"label": 0, "text": "台词 空洞 表演 生硬 配乐 也 非常 吵闹"},
    {"label": 1, "text": "前半段 稍慢 后半段 反转 精彩 整体 超出 预期"},
    {"label": 0, "text": "宣传 很 热闹 实际 内容 乏味 完全 浪费 时间"},
]


@dataclass
class ModelSpec:
    """单个待训练模型的构造器与训练配置。"""

    name: str
    builder: Callable[[], nn.Module]
    train_config: TrainConfig


def model_label(model_name: str) -> str:
    """把内部模型名转换成报告和日志中更易读的显示名。"""
    return MODEL_LABELS.get(model_name, model_name.upper())


def encode_text(text: str, stoi: Dict[str, int], max_len: int) -> Dict[str, torch.Tensor]:
    """把一条空格分词后的文本编码成模型可直接推理的张量。"""
    tokens = text.split()
    token_ids = [stoi.get(token, stoi["<unk>"]) for token in tokens[:max_len]]
    length = len(token_ids)
    padded = token_ids + [stoi["<pad>"]] * max(0, max_len - length)
    return {
        "tokens": tokens,
        "inputs": torch.tensor([padded], dtype=torch.long),
        "lengths": torch.tensor([max(length, 1)], dtype=torch.long),
    }


def run_robustness(
    models: Dict[str, nn.Module],
    prepared: PreparedData,
    device: torch.device,
) -> Dict[str, List[Dict[str, object]]]:
    """对主实验模型做一组固定样本的鲁棒性检查。"""
    max_len = prepared.metadata["max_len"]
    outputs = {}
    for name, model in models.items():
        model.eval()
        results = []
        with torch.no_grad():
            for sample in ROBUSTNESS_SAMPLES:
                encoded = encode_text(sample["text"], prepared.stoi, max_len)
                logits = model(encoded["inputs"].to(device), encoded["lengths"].to(device))
                prediction = int(logits.argmax(dim=1).item())
                results.append(
                    {
                        "text": sample["text"],
                        "gold": sample["label"],
                        "prediction": prediction,
                    }
                )
        outputs[name] = results
    return outputs


def build_main_model_specs(embedding_matrix: torch.Tensor) -> List[ModelSpec]:
    """定义主实验使用的五类模型。"""
    return [
        ModelSpec(
            name="mlp",
            builder=lambda: MeanPoolMLP(embedding_matrix, hidden_dim=128, dropout=0.3),
            train_config=TrainConfig(
                batch_size=128,
                epochs=10,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=3,
                grad_clip=0.0,
                seed=42,
            ),
        ),
        ModelSpec(
            name="cnn",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=128,
                filter_sizes=(3, 4, 5),
                dropout=0.5,
            ),
            train_config=TrainConfig(
                batch_size=128,
                epochs=10,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=3,
                grad_clip=0.0,
                seed=42,
            ),
        ),
        ModelSpec(
            name="birnn",
            builder=lambda: BiRNNClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=14,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=4,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=5,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
        ModelSpec(
            name="bilstm",
            builder=lambda: BiLSTMClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=14,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=4,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=5,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
        ModelSpec(
            name="bigru",
            builder=lambda: BiGRUClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=14,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=4,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=5,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
    ]


def build_ablation_specs(embedding_matrix: torch.Tensor) -> List[ModelSpec]:
    """定义与主实验直接相关的参数对比实验。"""
    return [
        ModelSpec(
            name="cnn_num_filters_64",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=64,
                filter_sizes=(3, 4, 5),
                dropout=0.5,
            ),
            train_config=TrainConfig(
                batch_size=128,
                epochs=8,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=2,
                grad_clip=0.0,
                seed=42,
            ),
        ),
        ModelSpec(
            name="birnn_hidden_64",
            builder=lambda: BiRNNClassifier(
                embedding_matrix,
                hidden_dim=64,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=10,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=3,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=4,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
        ModelSpec(
            name="bilstm_hidden_64",
            builder=lambda: BiLSTMClassifier(
                embedding_matrix,
                hidden_dim=64,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=10,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=3,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=4,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
        ModelSpec(
            name="bigru_hidden_64",
            builder=lambda: BiGRUClassifier(
                embedding_matrix,
                hidden_dim=64,
                num_layers=1,
                dropout=0.3,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=10,
                learning_rate=3e-4,
                weight_decay=3e-4,
                patience=3,
                grad_clip=1.0,
                seed=42,
                embedding_learning_rate=8e-5,
                label_smoothing=0.05,
                warmup_epochs=2,
                min_epochs=4,
                freeze_embedding_epochs=2,
                scheduler="warmup_cosine",
            ),
        ),
    ]
