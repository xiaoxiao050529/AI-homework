"""评测已训练模型在外部小规模数据上的鲁棒性。"""

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.sentiment_hw2.data import PreparedData, SplitData, TensorSplitDataset, prepare_data, read_split
from src.sentiment_hw2.models import (
    BiGRUClassifier,
    BiLSTMClassifier,
    BiRNNClassifier,
    MeanPoolMLP,
    TextCNN,
    TransformerEncoderClassifier,
)
from src.sentiment_hw2.paths import (
    EMBEDDING_PATH,
    EXTERNAL_WEIBO_META_PATH,
    EXTERNAL_WEIBO_TEXT_PATH,
    OUTPUT_CACHE_DIR,
    OUTPUT_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from src.sentiment_hw2.train import evaluate, set_seed

RESULT_PATH = OUTPUT_DIR / "external_robustness_metrics.json"


class LegacyBiGRUClassifier(nn.Module):
    """兼容旧 checkpoint 的双向 GRU 文本分类器。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(), freeze=False, padding_idx=0
        )
        embedding_dim = embedding_matrix.size(1)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))


class LegacyBiRNNClassifier(nn.Module):
    """兼容旧 checkpoint 的双向普通 RNN 文本分类器。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(), freeze=False, padding_idx=0
        )
        embedding_dim = embedding_matrix.size(1)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))


class LegacyBiLSTMClassifier(nn.Module):
    """兼容旧 checkpoint 的双向 LSTM 文本分类器。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(), freeze=False, padding_idx=0
        )
        embedding_dim = embedding_matrix.size(1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))


def encode_samples(
    samples: Sequence[Tuple[int, List[str]]],
    stoi: Dict[str, int],
    max_len: int,
) -> SplitData:
    """把外部样本映射到训练词表并编码成张量。"""
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]
    inputs = np.full((len(samples), max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(len(samples), dtype=np.int64)
    labels = np.zeros(len(samples), dtype=np.int64)
    raw_tokens: List[List[str]] = []

    for idx, (label, tokens) in enumerate(samples):
        clipped = tokens[:max_len]
        token_ids = [stoi.get(token, unk_id) for token in clipped]
        if not token_ids:
            token_ids = [unk_id]
        inputs[idx, : len(token_ids)] = token_ids
        lengths[idx] = len(token_ids)
        labels[idx] = label
        raw_tokens.append(tokens)

    return SplitData(
        inputs=torch.from_numpy(inputs),
        lengths=torch.from_numpy(lengths),
        labels=torch.from_numpy(labels),
        raw_tokens=raw_tokens,
    )


def compute_token_coverage(samples: Sequence[Tuple[int, List[str]]], stoi: Dict[str, int]) -> Dict[str, float]:
    """统计外部样本在训练词表上的覆盖率。"""
    known = 0
    total = 0
    for _, tokens in samples:
        for token in tokens:
            total += 1
            if token in stoi:
                known += 1
    return {
        "known_tokens": known,
        "total_tokens": total,
        "token_coverage": known / float(max(total, 1)),
    }


def build_model(name: str, embedding_matrix: torch.Tensor) -> nn.Module:
    """按主实验配置重建模型结构。"""
    if name == "mlp":
        return MeanPoolMLP(embedding_matrix, hidden_dim=128, dropout=0.3)
    if name == "cnn":
        return TextCNN(embedding_matrix, num_filters=128, filter_sizes=(3, 4, 5), dropout=0.5)
    if name == "birnn":
        return BiRNNClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.5)
    if name == "bilstm":
        return BiLSTMClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.5)
    if name == "bigru":
        return BiGRUClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.5)
    if name == "transformer":
        return TransformerEncoderClassifier(
            embedding_matrix,
            model_dim=128,
            num_heads=4,
            num_layers=1,
            ff_dim=256,
            dropout=0.1,
            max_len=40,
            pooling="cls",
        )
    raise ValueError("Unknown model name: {}".format(name))


def load_model_for_checkpoint(
    model_name: str,
    checkpoint_path,
    embedding_matrix: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    """优先按当前主实验结构加载，必要时兼容旧版循环模型 checkpoint。"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, embedding_matrix)
    try:
        model.load_state_dict(state_dict)
        model.to(device)
        return model
    except RuntimeError:
        if model_name == "birnn":
            model = LegacyBiRNNClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.3)
        elif model_name == "bilstm":
            model = LegacyBiLSTMClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.3)
        elif model_name == "bigru":
            model = LegacyBiGRUClassifier(embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.3)
        else:
            raise
        model.load_state_dict(state_dict)
        model.to(device)
        return model


def load_test_metrics(model_name: str) -> Dict[str, float]:
    """读取主实验测试集指标，便于做外部结果对比。"""
    metrics_path = OUTPUT_DIR / "{}_metrics.json".format(model_name)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload["test"]


def prepare_external_dataset(prepared: PreparedData) -> Tuple[SplitData, Dict[str, float]]:
    """读取并编码外部测试集。"""
    samples = read_split(EXTERNAL_WEIBO_TEXT_PATH)
    coverage = compute_token_coverage(samples, prepared.stoi)
    encoded = encode_samples(samples, prepared.stoi, int(prepared.metadata["max_len"]))
    return encoded, coverage


def main() -> None:
    if not EXTERNAL_WEIBO_TEXT_PATH.exists():
        raise FileNotFoundError("缺少外部测试集，请先运行 `python -m scripts.fetch_external_samples`")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(
        train_path=TRAIN_PATH,
        validation_path=VALIDATION_PATH,
        test_path=TEST_PATH,
        embedding_path=EMBEDDING_PATH,
        cache_path=OUTPUT_CACHE_DIR / "prepared_data.pt",
        max_len=80,
        min_freq=1,
        seed=42,
    )
    external_split, coverage = prepare_external_dataset(prepared)
    loader = DataLoader(TensorSplitDataset(external_split), batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    results = []
    for model_name in ("mlp", "cnn", "birnn", "bilstm", "bigru", "transformer"):
        checkpoint_path = OUTPUT_DIR / "{}_best.pt".format(model_name)
        if not checkpoint_path.exists():
            continue

        try:
            model = load_model_for_checkpoint(model_name, checkpoint_path, prepared.embedding_matrix, device)
        except RuntimeError as exc:
            print("Skip {} due to checkpoint/model mismatch: {}".format(model_name, exc))
            continue

        external_metrics = evaluate(model, loader, criterion, device)
        test_metrics = load_test_metrics(model_name)
        result = {
            "model_name": model_name,
            "test_metrics": test_metrics,
            "external_metrics": external_metrics,
            "accuracy_drop": test_metrics["accuracy"] - external_metrics["accuracy"],
            "f1_drop": test_metrics["f1"] - external_metrics["f1"],
        }
        results.append(result)
        print(
            "{name}: test_acc={ta:.4f}, ext_acc={ea:.4f}, test_f1={tf:.4f}, ext_f1={ef:.4f}".format(
                name=model_name,
                ta=test_metrics["accuracy"],
                ea=external_metrics["accuracy"],
                tf=test_metrics["f1"],
                ef=external_metrics["f1"],
            )
        )

    payload = {
        "external_dataset": json.loads(EXTERNAL_WEIBO_META_PATH.read_text(encoding="utf-8")),
        "external_stats": {
            "sample_size": int(external_split.labels.size(0)),
            "max_len": int(prepared.metadata["max_len"]),
            **coverage,
        },
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved external robustness metrics to {}".format(RESULT_PATH))


if __name__ == "__main__":
    main()
