"""训练集文本增强工具，只作用于训练切分。"""

import copy
import random
from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import PAD_TOKEN, UNK_TOKEN, PreparedData, SplitData


TokenTransform = Callable[[List[str], random.Random], List[str]]

_SYNONYM_GROUPS = [
    ("好看", "精彩", "出色", "优秀"),
    ("喜欢", "喜爱", "偏爱"),
    ("感人", "动人", "打动人"),
    ("震撼", "震动", "惊艳"),
    ("有趣", "好玩", "生动"),
    ("糟糕", "差劲", "很差"),
    ("无聊", "乏味", "枯燥"),
    ("失望", "扫兴", "遗憾"),
    ("拖沓", "冗长", "啰嗦"),
    ("混乱", "凌乱", "杂乱"),
    ("尴尬", "生硬", "别扭"),
    ("感动", "打动", "触动"),
]


def build_synonym_map(groups: Iterable[Sequence[str]]) -> Dict[str, List[str]]:
    """把近义词组展开为 token -> 候选替换列表。"""
    synonym_map: Dict[str, List[str]] = {}
    for group in groups:
        unique_tokens = []
        for token in group:
            if token not in unique_tokens:
                unique_tokens.append(token)
        for token in unique_tokens:
            synonym_map[token] = [item for item in unique_tokens if item != token]
    return synonym_map


DEFAULT_SYNONYM_MAP = build_synonym_map(_SYNONYM_GROUPS)


def word_dropout(tokens: List[str], rng: random.Random, p: float = 0.1) -> List[str]:
    """随机把部分词替换成 <unk>，增强模型对缺词和扰动的鲁棒性。"""
    if not tokens:
        return tokens
    augmented = [UNK_TOKEN if rng.random() < p else token for token in tokens]
    if all(token == UNK_TOKEN for token in augmented):
        keep_index = rng.randrange(len(tokens))
        augmented[keep_index] = tokens[keep_index]
    return augmented


def random_deletion(tokens: List[str], rng: random.Random, p: float = 0.1) -> List[str]:
    """随机删除部分 token，但保证句子不为空。"""
    if len(tokens) <= 1:
        return tokens
    kept = [token for token in tokens if rng.random() > p]
    if kept:
        return kept
    return [tokens[rng.randrange(len(tokens))]]


def synonym_replacement(
    tokens: List[str],
    rng: random.Random,
    synonym_map: Optional[Dict[str, List[str]]] = None,
    p: float = 0.15,
    max_replacements: int = 2,
) -> List[str]:
    """用同极性近义词替换少量 token，尽量保持标签语义稳定。"""
    if not tokens:
        return tokens
    synonym_map = DEFAULT_SYNONYM_MAP if synonym_map is None else synonym_map
    candidate_positions = [
        index for index, token in enumerate(tokens) if token in synonym_map and synonym_map[token]
    ]
    if not candidate_positions:
        return tokens

    rng.shuffle(candidate_positions)
    augmented = list(tokens)
    replacements = 0
    for index in candidate_positions:
        if replacements >= max_replacements:
            break
        if rng.random() > p:
            continue
        choices = synonym_map[augmented[index]]
        augmented[index] = choices[rng.randrange(len(choices))]
        replacements += 1
    return augmented


def _encode_augmented_samples(
    samples: Sequence[Tuple[int, List[str]]],
    stoi: Dict[str, int],
    max_len: int,
) -> SplitData:
    """把增强后的训练样本重新编码成张量。"""
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
        if token_ids:
            inputs[idx, : len(token_ids)] = token_ids
        raw_tokens.append(tokens)

    return SplitData(
        inputs=torch.from_numpy(inputs),
        lengths=torch.from_numpy(lengths),
        labels=torch.from_numpy(labels),
        raw_tokens=raw_tokens,
    )


def augment_prepared_training_data(
    prepared: PreparedData,
    transform: TokenTransform,
    seed: int = 42,
    metadata_name: str = "custom",
) -> PreparedData:
    """仅增强训练集，并返回带有新训练张量的 PreparedData 副本。"""
    rng = random.Random(seed)
    train_samples = []
    for label, tokens in zip(prepared.train.labels.tolist(), prepared.train.raw_tokens):
        train_samples.append((int(label), transform(list(tokens), rng)))

    train_split = _encode_augmented_samples(
        train_samples,
        prepared.stoi,
        int(prepared.metadata["max_len"]),
    )

    metadata = copy.deepcopy(prepared.metadata)
    metadata["train_augmentation"] = metadata_name
    return replace(prepared, train=train_split, metadata=metadata)
