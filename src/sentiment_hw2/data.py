"""数据读取、词表构建、词向量抽取与张量化封装。"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class SplitData:
    """单个数据集切分对应的张量与原始分词结果。"""
    inputs: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    raw_tokens: List[List[str]]


@dataclass
class PreparedData:
    """训练、验证、测试集和共享词表/词向量的总封装。"""
    train: SplitData
    validation: SplitData
    test: SplitData
    stoi: Dict[str, int]
    itos: List[str]
    embedding_matrix: torch.Tensor
    metadata: Dict[str, object]


def read_split(path: Path) -> List[Tuple[int, List[str]]]:
    """读取形如 `label token1 token2 ...` 的数据文件。"""
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            tokens = parts[1:]
            samples.append((label, tokens))
    return samples


def build_vocab(samples: Sequence[Tuple[int, List[str]]], min_freq: int = 1) -> Tuple[Dict[str, int], List[str], Counter]:
    """只基于训练集构建词表，避免验证集和测试集信息泄漏。"""
    counter = Counter()
    for _, tokens in samples:
        counter.update(tokens)

    itos = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in counter.most_common():
        if freq >= min_freq:
            itos.append(token)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return stoi, itos, counter


def _read_word(handle) -> str:
    """从 word2vec 二进制文件中逐字节读取一个词。"""
    word_bytes = bytearray()
    while True:
        ch = handle.read(1)
        if ch == b" ":
            break
        if ch != b"\n":
            word_bytes.extend(ch)
    return word_bytes.decode("utf-8", errors="ignore")


def load_word2vec_subset(path: Path, wanted_tokens: Iterable[str]) -> Dict[str, np.ndarray]:
    """只提取词表中需要的向量，避免整份大词向量全部加载进内存。"""
    wanted = set(wanted_tokens)
    vectors = {}

    with path.open("rb") as handle:
        vocab_size, dim = map(int, handle.readline().decode("utf-8").strip().split())
        binary_len = np.dtype(np.float32).itemsize * dim

        for _ in range(vocab_size):
            word = _read_word(handle)
            vector = np.frombuffer(handle.read(binary_len), dtype=np.float32).copy()
            if word in wanted:
                vectors[word] = vector
    return vectors


def _encode_samples(
    samples: Sequence[Tuple[int, List[str]]],
    stoi: Dict[str, int],
    max_len: int,
) -> SplitData:
    """把原始 token 序列截断/补齐到固定长度并转成张量。"""
    pad_id = stoi[PAD_TOKEN]
    unk_id = stoi[UNK_TOKEN]
    inputs = np.full((len(samples), max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(len(samples), dtype=np.int64)
    labels = np.zeros(len(samples), dtype=np.int64)
    raw_tokens: List[List[str]] = []

    for idx, (label, tokens) in enumerate(samples):
        clipped = tokens[:max_len]
        token_ids = [stoi.get(token, unk_id) for token in clipped]
        lengths[idx] = len(token_ids)
        labels[idx] = label
        # 先整体填满 PAD，再把真实 token 覆盖到前缀位置。
        if token_ids:
            inputs[idx, : len(token_ids)] = token_ids
        # raw_tokens 保留未截断版本，便于后续需要回溯原始文本时使用。
        raw_tokens.append(tokens)

    return SplitData(
        inputs=torch.from_numpy(inputs),
        lengths=torch.from_numpy(lengths),
        labels=torch.from_numpy(labels),
        raw_tokens=raw_tokens,
    )


def _build_embedding_matrix(
    itos: Sequence[str],
    pretrained: Dict[str, np.ndarray],
    dim: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """构造 embedding 矩阵，并为 OOV 词生成与预训练分布同尺度的随机向量。"""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((len(itos), dim), dtype=np.float32)

    pretrained_values = list(pretrained.values())
    if pretrained_values:
        stacked = np.stack(pretrained_values)
        mean = float(stacked.mean())
        std = float(stacked.std())
    else:
        mean = 0.0
        std = 0.1
    if std == 0.0:
        std = 0.1

    oov_count = 0
    for idx, token in enumerate(itos):
        if token == PAD_TOKEN:
            # PAD 始终置零，方便模型用掩码安全忽略补齐位置。
            matrix[idx] = 0.0
            continue
        vector = pretrained.get(token)
        if vector is None:
            matrix[idx] = rng.normal(loc=mean, scale=std, size=(dim,)).astype(np.float32)
            oov_count += 1
        else:
            matrix[idx] = vector

    stats = {
        "pretrained_vocab_hits": len(pretrained),
        "vocab_size": len(itos),
        "oov_count": oov_count,
        "oov_ratio": oov_count / float(max(len(itos) - 1, 1)),
    }
    return torch.from_numpy(matrix), stats


def prepare_data(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    embedding_path: Path,
    cache_path: Path,
    max_len: int = 80,
    min_freq: int = 1,
    seed: int = 42,
) -> PreparedData:
    """完成数据准备全流程，并把结果缓存到磁盘。"""
    if cache_path.exists():
        cached = torch.load(cache_path)
        if cached["config"] == {"max_len": max_len, "min_freq": min_freq, "seed": seed}:
            # 配置一致时直接复用缓存，保证重复运行结果稳定且更快。
            return cached["prepared"]

    train_samples = read_split(train_path)
    validation_samples = read_split(validation_path)
    test_samples = read_split(test_path)

    stoi, itos, train_counter = build_vocab(train_samples, min_freq=min_freq)
    pretrained = load_word2vec_subset(embedding_path, itos[2:])
    embedding_dim = len(next(iter(pretrained.values()))) if pretrained else 50
    embedding_matrix, coverage_stats = _build_embedding_matrix(itos, pretrained, embedding_dim, seed)

    train_split = _encode_samples(train_samples, stoi, max_len)
    validation_split = _encode_samples(validation_samples, stoi, max_len)
    test_split = _encode_samples(test_samples, stoi, max_len)

    total_train_tokens = sum(train_counter.values())
    covered_train_tokens = sum(freq for token, freq in train_counter.items() if token in pretrained)
    metadata = {
        "max_len": max_len,
        "embedding_dim": embedding_dim,
        "train_size": len(train_samples),
        "validation_size": len(validation_samples),
        "test_size": len(test_samples),
        "train_token_coverage": covered_train_tokens / float(max(total_train_tokens, 1)),
        "vocab_coverage": coverage_stats["pretrained_vocab_hits"] / float(max(len(itos) - 2, 1)),
        "oov_ratio": coverage_stats["oov_ratio"],
        "vocab_size": len(itos),
    }

    prepared = PreparedData(
        train=train_split,
        validation=validation_split,
        test=test_split,
        stoi=stoi,
        itos=itos,
        embedding_matrix=embedding_matrix,
        metadata=metadata,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 连同配置一起保存，下一次可以安全判断缓存是否仍可复用。
    torch.save(
        {
            "config": {"max_len": max_len, "min_freq": min_freq, "seed": seed},
            "prepared": prepared,
        },
        cache_path,
    )
    return prepared


class TensorSplitDataset(torch.utils.data.Dataset):
    """把 SplitData 包装成 PyTorch DataLoader 可迭代的数据集。"""

    def __init__(self, split: SplitData):
        self.inputs = split.inputs
        self.lengths = split.lengths
        self.labels = split.labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        """返回单条样本的输入张量、真实长度和标签。"""
        return self.inputs[idx], self.lengths[idx], self.labels[idx]
