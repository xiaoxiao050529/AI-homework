"""按“逐步优化”路线运行 RNN/CNN 情感分类改进实验。"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from src.sentiment_hw2.augment import (
    augment_prepared_training_data,
    random_deletion,
    synonym_replacement,
    word_dropout,
)
from src.sentiment_hw2.data import PreparedData, prepare_data
from src.sentiment_hw2.models import (
    BiLSTMAttentionClassifier,
    BiLSTMClassifier,
    BiRNNClassifier,
    TextCNN,
    initialize_model,
)
from src.sentiment_hw2.paths import EMBEDDING_PATH, OUTPUT_CACHE_DIR, OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.sentiment_hw2.train import TrainConfig, set_seed, train_single_model

SUMMARY_PATH = OUTPUT_DIR / "improvement_experiment_summary.json"


@dataclass
class ImprovementSpec:
    """单个改进实验的定义。"""

    name: str
    display_name: str
    track: str
    stage: int
    description: str
    builder: Callable[[], nn.Module]
    train_config: TrainConfig
    reference_name: Optional[str] = None
    augmentation_name: str = "none"
    prepared_builder: Optional[Callable[[PreparedData], PreparedData]] = None
    reuse_metrics_path: Optional[Path] = None


def recurrent_config(quick: bool = False) -> TrainConfig:
    if quick:
        return TrainConfig(
            batch_size=96,
            epochs=6,
            learning_rate=3e-4,
            weight_decay=3e-4,
            patience=2,
            grad_clip=1.0,
            seed=42,
            embedding_learning_rate=8e-5,
            label_smoothing=0.05,
            warmup_epochs=1,
            min_epochs=3,
            freeze_embedding_epochs=1,
            scheduler="warmup_cosine",
        )
    return TrainConfig(
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
    )


def cnn_config(quick: bool = False) -> TrainConfig:
    if quick:
        return TrainConfig(
            batch_size=192,
            epochs=6,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=2,
            grad_clip=0.0,
            seed=42,
        )
    return TrainConfig(
        batch_size=128,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
        grad_clip=0.0,
        seed=42,
    )


def build_specs(embedding_matrix: torch.Tensor, quick: bool = False) -> List[ImprovementSpec]:
    """定义本次作业建议的逐步改进路线。"""
    recurrent_full = recurrent_config()
    recurrent_run = recurrent_config(quick=quick)
    cnn_full = cnn_config()
    cnn_run = cnn_config(quick=quick)
    return [
        ImprovementSpec(
            name="birnn",
            display_name="BiRNN 基线",
            track="rnn",
            stage=1,
            description="沿用当前仓库中的基础循环结构，作为 RNN 组对比起点。",
            builder=lambda: BiRNNClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.5,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=18,
                learning_rate=1.5e-4,
                weight_decay=7e-4,
                patience=6,
                grad_clip=0.5,
                seed=42,
                embedding_learning_rate=3e-5,
                label_smoothing=0.1,
                warmup_epochs=4,
                min_epochs=7,
                freeze_embedding_epochs=4,
                scheduler="warmup_cosine",
                min_learning_rate_scale=0.15,
            ),
            reuse_metrics_path=OUTPUT_DIR / "birnn_metrics.json",
        ),
        ImprovementSpec(
            name="bilstm",
            display_name="BiLSTM",
            track="rnn",
            stage=2,
            description="将普通循环单元替换为 LSTM，增强长距离依赖建模能力。",
            builder=lambda: BiLSTMClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.5,
            ),
            train_config=TrainConfig(
                batch_size=64,
                epochs=18,
                learning_rate=1.5e-4,
                weight_decay=7e-4,
                patience=6,
                grad_clip=0.5,
                seed=42,
                embedding_learning_rate=3e-5,
                label_smoothing=0.1,
                warmup_epochs=4,
                min_epochs=7,
                freeze_embedding_epochs=4,
                scheduler="warmup_cosine",
                min_learning_rate_scale=0.15,
            ),
            reference_name="birnn",
            reuse_metrics_path=OUTPUT_DIR / "bilstm_metrics.json",
        ),
        ImprovementSpec(
            name="bilstm_attention",
            display_name="BiLSTM + Attention",
            track="rnn",
            stage=3,
            description="在 BiLSTM 全序列输出上引入加性注意力，让分类器聚焦关键情感词。",
            builder=lambda: BiLSTMAttentionClassifier(
                embedding_matrix,
                hidden_dim=128,
                num_layers=1,
                dropout=0.3,
                attention_dim=128,
            ),
            train_config=recurrent_run,
            reference_name="bilstm",
            reuse_metrics_path=OUTPUT_DIR / "bilstm_attention_metrics.json",
        ),
        ImprovementSpec(
            name="cnn_single_kernel",
            display_name="CNN 单卷积核基线",
            track="cnn",
            stage=1,
            description="只用单一 3-gram 卷积核的简单 CNN，作为 TextCNN 升级前基线。",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=128,
                filter_sizes=(3,),
                dropout=0.5,
            ),
            train_config=cnn_run,
            reuse_metrics_path=OUTPUT_DIR / "cnn_single_kernel_metrics.json",
        ),
        ImprovementSpec(
            name="cnn",
            display_name="TextCNN 多卷积核",
            track="cnn",
            stage=2,
            description="使用 3/4/5 多尺度卷积核捕捉不同长度的局部短语模式。",
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
            reference_name="cnn_single_kernel",
            reuse_metrics_path=OUTPUT_DIR / "cnn_metrics.json",
        ),
        ImprovementSpec(
            name="textcnn_word_dropout",
            display_name="TextCNN + Word Dropout",
            track="cnn",
            stage=3,
            description="在训练集上随机把少量词替换为 <unk>，提升模型抗噪声能力。",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=128,
                filter_sizes=(3, 4, 5),
                dropout=0.5,
            ),
            train_config=cnn_run,
            reference_name="cnn",
            augmentation_name="word_dropout",
            prepared_builder=lambda prepared: augment_prepared_training_data(
                prepared,
                lambda tokens, rng: word_dropout(tokens, rng, p=0.1),
                seed=42,
                metadata_name="word_dropout",
            ),
            reuse_metrics_path=OUTPUT_DIR / "textcnn_word_dropout_metrics.json",
        ),
        ImprovementSpec(
            name="textcnn_random_deletion",
            display_name="TextCNN + Random Deletion",
            track="cnn",
            stage=3,
            description="在训练集上随机删除少量词，测试模型对信息缺失的鲁棒性。",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=128,
                filter_sizes=(3, 4, 5),
                dropout=0.5,
            ),
            train_config=cnn_run,
            reference_name="cnn",
            augmentation_name="random_deletion",
            prepared_builder=lambda prepared: augment_prepared_training_data(
                prepared,
                lambda tokens, rng: random_deletion(tokens, rng, p=0.1),
                seed=42,
                metadata_name="random_deletion",
            ),
            reuse_metrics_path=OUTPUT_DIR / "textcnn_random_deletion_metrics.json",
        ),
        ImprovementSpec(
            name="textcnn_synonym_replacement",
            display_name="TextCNN + Synonym Replacement",
            track="cnn",
            stage=3,
            description="用同极性近义词替换少量 token，检验轻量语义改写是否有利。",
            builder=lambda: TextCNN(
                embedding_matrix,
                num_filters=128,
                filter_sizes=(3, 4, 5),
                dropout=0.5,
            ),
            train_config=cnn_run,
            reference_name="cnn",
            augmentation_name="synonym_replacement",
            prepared_builder=lambda prepared: augment_prepared_training_data(
                prepared,
                lambda tokens, rng: synonym_replacement(tokens, rng, p=0.6, max_replacements=2),
                seed=42,
                metadata_name="synonym_replacement",
            ),
            reuse_metrics_path=OUTPUT_DIR / "textcnn_synonym_replacement_metrics.json",
        ),
    ]


def should_reuse_result(spec: ImprovementSpec, force_retrain: bool) -> bool:
    return not force_retrain and spec.reuse_metrics_path is not None and spec.reuse_metrics_path.exists()


def load_reused_result(spec: ImprovementSpec) -> Dict[str, object]:
    result = json.loads(spec.reuse_metrics_path.read_text(encoding="utf-8"))
    result["reused_existing_result"] = True
    return result


def train_spec(
    spec: ImprovementSpec,
    prepared: PreparedData,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(spec.train_config.seed)
    model = spec.builder()
    initialize_model(model)
    train_data = prepared if spec.prepared_builder is None else spec.prepared_builder(prepared)
    result = train_single_model(
        model_name=spec.name,
        model=model,
        prepared=train_data,
        config=spec.train_config,
        device=device,
        output_dir=OUTPUT_DIR,
    )
    result["reused_existing_result"] = False
    return result


def enrich_result(
    spec: ImprovementSpec,
    result: Dict[str, object],
    result_by_name: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    enriched = dict(result)
    enriched["display_name"] = spec.display_name
    enriched["track"] = spec.track
    enriched["stage"] = spec.stage
    enriched["description"] = spec.description
    enriched["augmentation"] = spec.augmentation_name
    enriched["train_config"] = asdict(spec.train_config)
    enriched["reference_name"] = spec.reference_name
    if spec.reference_name is not None and spec.reference_name in result_by_name:
        reference = result_by_name[spec.reference_name]
        enriched["delta_vs_reference"] = {
            "validation_accuracy": enriched["validation"]["accuracy"] - reference["validation"]["accuracy"],
            "validation_f1": enriched["validation"]["f1"] - reference["validation"]["f1"],
            "test_accuracy": enriched["test"]["accuracy"] - reference["test"]["accuracy"],
            "test_f1": enriched["test"]["f1"] - reference["test"]["f1"],
        }
    else:
        enriched["delta_vs_reference"] = None
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行逐步改进的 CNN/RNN 情感分类实验")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="即使已有结果文件，也重新训练所有实验。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用更紧凑的训练预算，适合在 CPU 负载较高时快速得到对比结果。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    set_seed(42)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

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

    specs = build_specs(prepared.embedding_matrix, quick=args.quick)
    collected: List[Dict[str, object]] = []
    result_by_name: Dict[str, Dict[str, object]] = {}

    print("Device:", device)
    print("Running staged improvement experiments...")
    for spec in specs:
        if should_reuse_result(spec, args.force_retrain):
            result = load_reused_result(spec)
            print("Reuse existing result for", spec.name)
        else:
            print("Train", spec.name)
            result = train_spec(spec, prepared, device)

        enriched = enrich_result(spec, result, result_by_name)
        collected.append(enriched)
        result_by_name[spec.name] = enriched
        print(
            "{name}: val_acc={va:.4f}, val_f1={vf:.4f}, test_acc={ta:.4f}, test_f1={tf:.4f}".format(
                name=spec.name,
                va=enriched["validation"]["accuracy"],
                vf=enriched["validation"]["f1"],
                ta=enriched["test"]["accuracy"],
                tf=enriched["test"]["f1"],
            )
        )

    summary = {
        "device": str(device),
        "data": prepared.metadata,
        "experiments": collected,
        "tracks": {
            "rnn": [item for item in collected if item["track"] == "rnn"],
            "cnn": [item for item in collected if item["track"] == "cnn"],
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary saved to", SUMMARY_PATH)


if __name__ == "__main__":
    main()
