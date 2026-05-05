"""逐个超参数进行单变量调参与结果汇总。"""

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import nn

from src.sentiment_hw2.data import prepare_data
from src.sentiment_hw2.models import (
    BiGRUClassifier,
    BiLSTMClassifier,
    BiRNNClassifier,
    MeanPoolMLP,
    TextCNN,
    initialize_model,
)
from src.sentiment_hw2.paths import EMBEDDING_PATH, OUTPUT_CACHE_DIR, OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.sentiment_hw2.train import TrainConfig, set_seed, train_single_model

TUNING_DIR = OUTPUT_DIR / "tuning"
ARCHIVE_TUNING_DIR = TUNING_DIR / "archive_recurrent_hidden_dim_refresh"
SUMMARY_PATH = OUTPUT_DIR / "hyperparameter_tuning_summary.json"


@dataclass
class TuningSpec:
    """单个调参实验的定义。"""

    name: str
    family: str
    parameter_name: str
    parameter_value: object
    baseline_name: str
    description: str
    builder: Callable[[], nn.Module]
    train_config: TrainConfig
    is_baseline: bool = False


def make_tuning_spec(
    *,
    name_family: str,
    family: str,
    parameter_name: str,
    parameter_value: object,
    baseline_name: str,
    description: str,
    builder: Callable[[], nn.Module],
    train_config: TrainConfig,
    is_baseline: bool = False,
) -> TuningSpec:
    """统一生成调参实验名称，避免手写字符串不一致。"""
    if is_baseline:
        name = baseline_name
    else:
        name = "tune_{}_{}_{}".format(name_family, parameter_name, slugify_value(parameter_value))
    return TuningSpec(
        name=name,
        family=family,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        baseline_name=baseline_name,
        description=description,
        builder=builder,
        train_config=train_config,
        is_baseline=is_baseline,
    )


def listify_filter_sizes(value: Sequence[int]) -> List[int]:
    """把卷积核尺寸统一转成 JSON 友好的列表。"""
    return list(value)


def slugify_value(value: object) -> str:
    """把超参数值转成文件名友好的形式。"""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.01:
            scientific = "{:.8e}".format(value)
            mantissa, exponent = scientific.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")
            exponent = exponent.replace("+", "")
            if exponent.startswith("-"):
                exponent = exponent[1:]
            return "{}e{}".format(mantissa, exponent)
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "_".join(slugify_value(item) for item in value)
    return str(value).replace(" ", "")


def ensure_minimum_search_space_size(
    family_defs: Dict[str, Dict[str, object]],
    minimum_size: int = 5,
) -> None:
    """确保每个超参数至少提供指定数量的候选值。"""
    for family, info in family_defs.items():
        for parameter_name, candidates in info["search_space"].items():
            if len(candidates) < minimum_size:
                raise ValueError(
                    "Search space for {}.{} has only {} values; at least {} are required.".format(
                        family,
                        parameter_name,
                        len(candidates),
                        minimum_size,
                    )
                )


def build_specs(embedding_matrix: torch.Tensor) -> Dict[str, List[TuningSpec]]:
    """构造五类模型的基线与单变量调参实验。"""
    mlp_cfg = {"hidden_dim": 128, "dropout": 0.3}
    cnn_cfg = {"num_filters": 128, "filter_sizes": (3, 4, 5), "dropout": 0.5}
    recurrent_cfg = {"hidden_dim": 128, "num_layers": 1, "dropout": 0.5}

    mlp_train = TrainConfig(
        batch_size=128,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
        grad_clip=0.0,
        seed=42,
    )
    cnn_train = TrainConfig(
        batch_size=128,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
        grad_clip=0.0,
        seed=42,
    )
    recurrent_train = TrainConfig(
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
    )

    family_defs: Dict[str, Dict[str, object]] = {
        "mlp": {
            "name_family": "mlp",
            "baseline_name": "tune_mlp_base",
            "description": "MLP 基线配置",
            "builder": lambda cfg: MeanPoolMLP(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(mlp_cfg),
            "base_train_cfg": replace(mlp_train),
            "search_space": {
                "hidden_dim": [64, 96, 128, 192, 256],
                "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
                "batch_size": [32, 64, 128, 192, 256],
                "learning_rate": [3e-4, 5e-4, 1e-3, 1.5e-3, 2e-3],
                "weight_decay": [0.0, 5e-5, 1e-4, 3e-4, 5e-4],
                "epochs": [6, 8, 10, 12, 14],
                "patience": [1, 2, 3, 4, 5],
                "seed": [7, 21, 42, 84, 123],
            },
        },
        "cnn": {
            "name_family": "cnn",
            "baseline_name": "tune_cnn_base",
            "description": "TextCNN 基线配置",
            "builder": lambda cfg: TextCNN(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(cnn_cfg),
            "base_train_cfg": replace(cnn_train),
            "search_space": {
                "num_filters": [64, 96, 128, 160, 192],
                "filter_sizes": [(2, 3, 4), (2, 3, 5), (3, 4, 5), (3, 5, 7), (2, 4, 6)],
                "dropout": [0.2, 0.3, 0.4, 0.5, 0.7],
                "batch_size": [32, 64, 128, 192, 256],
                "learning_rate": [3e-4, 5e-4, 1e-3, 1.5e-3, 2e-3],
                "weight_decay": [0.0, 5e-5, 1e-4, 3e-4, 5e-4],
                "epochs": [6, 8, 10, 12, 14],
                "patience": [1, 2, 3, 4, 5],
                "seed": [7, 21, 42, 84, 123],
            },
        },
        "birnn": {
            "name_family": "birnn",
            "baseline_name": "tune_birnn_base",
            "description": "BiRNN 基线配置",
            "builder": lambda cfg: BiRNNClassifier(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(recurrent_cfg),
            "base_train_cfg": replace(recurrent_train),
            "search_space": {
                "hidden_dim": [64, 96, 128, 160, 192],
                "num_layers": [1, 2, 3, 4, 5],
                "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
                "batch_size": [16, 32, 48, 64, 96],
                "learning_rate": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
                "embedding_learning_rate": [2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4],
                "weight_decay": [0.0, 1e-4, 3e-4, 5e-4, 7e-4],
                "grad_clip": [0.25, 0.5, 1.0, 1.5, 2.0],
                "warmup_epochs": [0, 1, 2, 3, 4],
                "label_smoothing": [0.0, 0.02, 0.05, 0.08, 0.1],
                "freeze_embedding_epochs": [0, 1, 2, 3, 4],
                "epochs": [10, 12, 14, 16, 18],
                "patience": [2, 3, 4, 5, 6],
                "seed": [7, 21, 42, 84, 123],
            },
        },
        "bilstm": {
            "name_family": "bilstm",
            "baseline_name": "tune_bilstm_base",
            "description": "BiLSTM 基线配置",
            "builder": lambda cfg: BiLSTMClassifier(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(recurrent_cfg),
            "base_train_cfg": replace(recurrent_train),
            "search_space": {
                "hidden_dim": [64, 96, 128, 160, 192],
                "num_layers": [1, 2, 3, 4, 5],
                "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
                "batch_size": [16, 32, 48, 64, 96],
                "learning_rate": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
                "embedding_learning_rate": [2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4],
                "weight_decay": [0.0, 1e-4, 3e-4, 5e-4, 7e-4],
                "grad_clip": [0.25, 0.5, 1.0, 1.5, 2.0],
                "warmup_epochs": [0, 1, 2, 3, 4],
                "label_smoothing": [0.0, 0.02, 0.05, 0.08, 0.1],
                "freeze_embedding_epochs": [0, 1, 2, 3, 4],
                "epochs": [10, 12, 14, 16, 18],
                "patience": [2, 3, 4, 5, 6],
                "seed": [7, 21, 42, 84, 123],
            },
        },
        "bigru": {
            "name_family": "bigru_v2",
            "baseline_name": "tune_bigru_v2_base",
            "description": "BiGRU 基线配置",
            "builder": lambda cfg: BiGRUClassifier(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(recurrent_cfg),
            "base_train_cfg": replace(recurrent_train),
            "search_space": {
                "hidden_dim": [64, 96, 128, 160, 192],
                "num_layers": [1, 2, 3, 4, 5],
                "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
                "batch_size": [16, 32, 48, 64, 96],
                "learning_rate": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
                "embedding_learning_rate": [2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4],
                "weight_decay": [0.0, 1e-4, 3e-4, 5e-4, 7e-4],
                "grad_clip": [0.25, 0.5, 1.0, 1.5, 2.0],
                "warmup_epochs": [0, 1, 2, 3, 4],
                "label_smoothing": [0.0, 0.02, 0.05, 0.08, 0.1],
                "freeze_embedding_epochs": [0, 1, 2, 3, 4],
                "epochs": [10, 12, 14, 16, 18],
                "patience": [2, 3, 4, 5, 6],
                "seed": [7, 21, 42, 84, 123],
            },
        },
    }

    ensure_minimum_search_space_size(family_defs, minimum_size=5)

    baselines: List[TuningSpec] = []
    variants: List[TuningSpec] = []
    families: List[Dict[str, object]] = []

    model_hparams = {"hidden_dim", "dropout", "num_filters", "filter_sizes", "num_layers"}

    for family, info in family_defs.items():
        base_model_cfg = copy.deepcopy(info["base_model_cfg"])
        base_train_cfg = replace(info["base_train_cfg"])
        builder_factory = info["builder"]
        baseline_name = info["baseline_name"]

        baselines.append(
            make_tuning_spec(
                family=family,
                name_family=info["name_family"],
                parameter_name="baseline",
                parameter_value="default",
                baseline_name=baseline_name,
                description=info["description"],
                builder=lambda cfg=copy.deepcopy(base_model_cfg), factory=builder_factory: factory(cfg),
                train_config=replace(base_train_cfg),
                is_baseline=True,
            )
        )

        parameter_names = list(info["search_space"].keys())
        families.append(
            {
                "family": family,
                "baseline_name": baseline_name,
                "baseline_model_config": normalize_value(base_model_cfg),
                "baseline_train_config": normalize_value(asdict(base_train_cfg)),
                "tunable_parameters": parameter_names,
            }
        )

        for parameter_name, candidates in info["search_space"].items():
            baseline_value = (
                base_model_cfg[parameter_name]
                if parameter_name in base_model_cfg
                else getattr(base_train_cfg, parameter_name)
            )
            for candidate in candidates:
                if normalize_value(candidate) == normalize_value(baseline_value):
                    continue

                variant_model_cfg = copy.deepcopy(base_model_cfg)
                variant_train_cfg = replace(base_train_cfg)
                if parameter_name in model_hparams:
                    variant_model_cfg[parameter_name] = candidate
                else:
                    variant_train_cfg = replace(variant_train_cfg, **{parameter_name: candidate})

                variants.append(
                    make_tuning_spec(
                        family=family,
                        name_family=info["name_family"],
                        parameter_name=parameter_name,
                        parameter_value=normalize_value(candidate),
                        baseline_name=baseline_name,
                        description="{} 的 {} 调整为 {}".format(family.upper(), parameter_name, format_value(candidate)),
                        builder=lambda cfg=copy.deepcopy(variant_model_cfg), factory=builder_factory: factory(cfg),
                        train_config=variant_train_cfg,
                    )
                )

    return {"baselines": baselines, "variants": variants, "families": families}


def normalize_value(value: object) -> object:
    """把 tuple 等值转成更适合 JSON 序列化与比较的形式。"""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def format_value(value: object) -> str:
    """把超参数值转成报告里更易读的字符串。"""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 0.01:
            return "{:.4f}".format(value).rstrip("0").rstrip(".")
        return "{:.0e}".format(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(format_value(item) for item in value) + ")"
    return str(value)


def candidate_metrics_paths(spec_name: str) -> List[object]:
    """按优先级列出同一实验结果可能所在的位置。"""
    return [
        TUNING_DIR / "{}_metrics.json".format(spec_name),
        ARCHIVE_TUNING_DIR / "{}_metrics.json".format(spec_name),
    ]


def load_metrics_from_known_locations(spec_name: str) -> Dict[str, object]:
    """从当前目录或历史归档目录读取已有 metrics。"""
    for metrics_path in candidate_metrics_paths(spec_name):
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def run_or_load(spec: TuningSpec, prepared, device: torch.device) -> Dict[str, object]:
    """如果结果已存在则直接复用，否则执行训练。"""
    existing = load_metrics_from_known_locations(spec.name)
    if existing:
        return existing

    set_seed(spec.train_config.seed)
    model = spec.builder()
    initialize_model(model)
    return train_single_model(
        model_name=spec.name,
        model=model,
        prepared=prepared,
        config=spec.train_config,
        device=device,
        output_dir=TUNING_DIR,
    )


def load_existing_only(spec: TuningSpec) -> Dict[str, object]:
    """仅从已有文件读取结果，不触发训练。"""
    return load_metrics_from_known_locations(spec.name)


def attach_delta(result: Dict[str, object], baseline: Dict[str, object]) -> Dict[str, object]:
    """补充相对基线的性能变化。"""
    enriched = copy.deepcopy(result)
    enriched["delta_vs_baseline"] = {
        "validation_accuracy": result["validation"]["accuracy"] - baseline["validation"]["accuracy"],
        "validation_f1": result["validation"]["f1"] - baseline["validation"]["f1"],
        "test_accuracy": result["test"]["accuracy"] - baseline["test"]["accuracy"],
        "test_f1": result["test"]["f1"] - baseline["test"]["f1"],
    }
    return enriched


def serialize_spec(spec: TuningSpec) -> Dict[str, object]:
    """把配置元信息转成可保存格式。"""
    payload = asdict(spec)
    payload.pop("builder")
    payload["parameter_value"] = normalize_value(payload["parameter_value"])
    payload["train_config"] = normalize_value(asdict(spec.train_config))
    return payload


def metric_sort_key(item: Dict[str, object]) -> Tuple[float, float]:
    """统一按测试 F1 优先、测试 Acc 次优先挑选最优解。"""
    result = item["result"]
    return result["test"]["f1"], result["test"]["accuracy"]


def validation_sort_key(candidate: Dict[str, object]) -> Tuple[float, float]:
    """按验证集 F1 优先、验证集 Acc 次优先选择超参数。"""
    result = candidate["result"]
    return result["validation"]["f1"], result["validation"]["accuracy"]


def build_parameter_groups(
    families: Sequence[Dict[str, object]],
    baselines: Dict[str, Dict[str, object]],
    variant_meta: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """按模型族和超参数聚合，便于报告逐项画图。"""
    family_info = {item["family"]: item for item in families}
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for item in variant_meta:
        spec = item["spec"]
        key = (spec["family"], spec["parameter_name"])
        grouped.setdefault(key, []).append(item)

    parameter_groups: List[Dict[str, object]] = []
    for (family, parameter_name), items in sorted(grouped.items()):
        baseline_name = family_info[family]["baseline_name"]
        baseline_result = baselines[baseline_name]["result"]
        baseline_value = None
        if parameter_name in family_info[family]["baseline_model_config"]:
            baseline_value = family_info[family]["baseline_model_config"][parameter_name]
        else:
            baseline_value = family_info[family]["baseline_train_config"][parameter_name]

        candidates = [
            {
                "value": normalize_value(baseline_value),
                "value_label": format_value(baseline_value),
                "is_baseline": True,
                "spec": baselines[baseline_name]["spec"],
                "result": baseline_result,
                "delta_vs_baseline": {
                    "validation_accuracy": 0.0,
                    "validation_f1": 0.0,
                    "test_accuracy": 0.0,
                    "test_f1": 0.0,
                },
            }
        ]
        for item in items:
            spec = item["spec"]
            candidates.append(
                {
                    "value": normalize_value(spec["parameter_value"]),
                    "value_label": format_value(spec["parameter_value"]),
                    "is_baseline": False,
                    "spec": spec,
                    "result": item["result"],
                    "delta_vs_baseline": item["result"]["delta_vs_baseline"],
                }
            )

        candidates.sort(key=lambda entry: sort_value(entry["value"]))
        best = max(candidates, key=validation_sort_key)
        parameter_groups.append(
            {
                "family": family,
                "parameter_name": parameter_name,
                "baseline_name": baseline_name,
                "baseline_value": normalize_value(baseline_value),
                "baseline_value_label": format_value(baseline_value),
                "candidates": candidates,
                "best": {
                    "value": best["value"],
                    "value_label": best["value_label"],
                    "validation_accuracy": best["result"]["validation"]["accuracy"],
                    "test_f1": best["result"]["test"]["f1"],
                    "test_accuracy": best["result"]["test"]["accuracy"],
                    "validation_f1": best["result"]["validation"]["f1"],
                    "delta_validation_f1": best["delta_vs_baseline"]["validation_f1"],
                    "delta_test_f1": best["delta_vs_baseline"]["test_f1"],
                    "is_baseline": best["is_baseline"],
                    "spec_name": best["spec"]["name"],
                },
            }
        )
    return parameter_groups


def sort_value(value: object) -> Tuple[int, object]:
    """为图表横轴稳定排序。"""
    if isinstance(value, (int, float)):
        return 0, value
    if isinstance(value, list):
        return 1, tuple(value)
    return 2, str(value)


def build_best_by_family(parameter_groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """为每个模型族汇总各个超参数的最优取值。"""
    summary_by_family: Dict[str, List[Dict[str, object]]] = {}
    for group in parameter_groups:
        summary_by_family.setdefault(group["family"], []).append(
            {
                "parameter_name": group["parameter_name"],
                "baseline_value": group["baseline_value"],
                "baseline_value_label": group["baseline_value_label"],
                "best_value": group["best"]["value"],
                "best_value_label": group["best"]["value_label"],
                "best_validation_f1": group["best"]["validation_f1"],
                "best_validation_accuracy": group["best"]["validation_accuracy"],
                "best_test_f1": group["best"]["test_f1"],
                "best_test_accuracy": group["best"]["test_accuracy"],
                "delta_validation_f1": group["best"]["delta_validation_f1"],
                "delta_test_f1": group["best"]["delta_test_f1"],
                "is_baseline_best": group["best"]["is_baseline"],
            }
        )

    output = []
    for family, items in sorted(summary_by_family.items()):
        output.append(
            {
                "family": family,
                "best_parameters": sorted(items, key=lambda item: item["parameter_name"]),
            }
        )
    return output


def merge_by_spec_name(
    current_items: Sequence[Dict[str, object]],
    previous_items: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """按 spec.name 合并新旧摘要，便于分批执行后汇总。"""
    merged = {item["spec"]["name"]: item for item in previous_items}
    for item in current_items:
        merged[item["spec"]["name"]] = item
    return list(merged.values())


def merge_family_entries(
    current_items: Sequence[Dict[str, object]],
    previous_items: Sequence[Dict[str, object]],
    key: str,
) -> List[Dict[str, object]]:
    """按 family 合并家族级汇总信息。"""
    merged = {item[key]: item for item in previous_items}
    for item in current_items:
        merged[item[key]] = item
    return sorted(merged.values(), key=lambda item: item[key])


def apply_available_parameter_coverage(
    families: Sequence[Dict[str, object]],
    variant_meta: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """按实际已有结果回填每个模型族真正覆盖到的超参数。"""
    covered: Dict[str, set] = {}
    for item in variant_meta:
        spec = item.get("spec", {})
        family = spec.get("family")
        parameter_name = spec.get("parameter_name")
        if not family or not parameter_name:
            continue
        covered.setdefault(family, set()).add(parameter_name)

    normalized: List[Dict[str, object]] = []
    for family in families:
        copied = copy.deepcopy(family)
        if copied["family"] in covered:
            copied["tunable_parameters"] = sorted(covered[copied["family"]])
        normalized.append(copied)
    return normalized


def load_previous_summary() -> Dict[str, object]:
    """在分批执行时复用已有摘要。"""
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def filter_summary_items_by_families(items: Sequence[Dict[str, object]], families: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按指定模型族筛选摘要条目。"""
    family_set = {item["family"] for item in families}
    filtered = []
    for item in items:
        spec = item.get("spec", {})
        if spec.get("family") in family_set:
            filtered.append(item)
    return filtered


def load_existing_family_results(
    specs: Dict[str, List[TuningSpec]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """仅基于已有 metrics 文件重建指定 families 的 baseline 与 variant 摘要。"""
    baselines: Dict[str, Dict[str, object]] = {}
    baseline_meta: List[Dict[str, object]] = []
    for spec in specs["baselines"]:
        result = load_existing_only(spec)
        if not result:
            continue
        payload = {"spec": serialize_spec(spec), "result": result}
        baselines[spec.name] = payload
        baseline_meta.append(payload)

    variant_meta: List[Dict[str, object]] = []
    for spec in specs["variants"]:
        if spec.baseline_name not in baselines:
            continue
        result = load_existing_only(spec)
        if not result:
            continue
        baseline = baselines[spec.baseline_name]["result"]
        enriched = attach_delta(result, baseline)
        variant_meta.append({"spec": serialize_spec(spec), "result": enriched})
    return baseline_meta, variant_meta


def parse_args() -> argparse.Namespace:
    """支持按模型族分批运行，避免一次性 CPU 训练时间过长。"""
    parser = argparse.ArgumentParser(description="Single-variable hyperparameter tuning")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["mlp", "cnn", "birnn", "bilstm", "bigru"],
        help="Only run selected model families.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help="Only run selected parameter names within the chosen families.",
    )
    parser.add_argument(
        "--reuse-existing-only",
        action="store_true",
        help="Summarize existing metrics only and skip missing experiments.",
    )
    return parser.parse_args()


def filter_specs(
    specs: Dict[str, List[TuningSpec]],
    selected_families: Sequence[str] = None,
    selected_parameters: Sequence[str] = None,
) -> Dict[str, List[TuningSpec]]:
    """按命令行条件筛选要执行的实验，但保留统一输出结构。"""
    family_set = set(selected_families or [])
    parameter_set = set(selected_parameters or [])

    if not family_set and not parameter_set:
        return specs

    baselines = [
        spec for spec in specs["baselines"]
        if (not family_set or spec.family in family_set)
    ]
    variants = [
        spec for spec in specs["variants"]
        if (not family_set or spec.family in family_set)
        and (not parameter_set or spec.parameter_name in parameter_set)
    ]
    families = [
        item for item in specs["families"]
        if (not family_set or item["family"] in family_set)
    ]
    if parameter_set:
        for item in families:
            item["tunable_parameters"] = [
                name for name in item["tunable_parameters"] if name in parameter_set
            ]
    return {"baselines": baselines, "variants": variants, "families": families}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    set_seed(42)
    torch.set_num_threads(4)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

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

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    previous_summary = load_previous_summary()
    specs = build_specs(prepared.embedding_matrix)
    specs = filter_specs(specs, args.families, args.parameters)

    baselines: Dict[str, Dict[str, object]] = {}
    baseline_meta: List[Dict[str, object]] = []
    for spec in specs["baselines"]:
        result = load_existing_only(spec) if args.reuse_existing_only else run_or_load(spec, prepared, device)
        if not result:
            print("[skip] missing baseline metrics for", spec.name)
            continue
        payload = {"spec": serialize_spec(spec), "result": result}
        baselines[spec.name] = payload
        baseline_meta.append(payload)
        print(
            "[baseline] {name}: val_f1={vf:.4f}, test_f1={tf:.4f}".format(
                name=spec.name,
                vf=result["validation"]["f1"],
                tf=result["test"]["f1"],
            )
        )

    variant_meta: List[Dict[str, object]] = []
    for spec in specs["variants"]:
        if spec.baseline_name not in baselines:
            print("[skip] baseline not available for", spec.name)
            continue
        result = load_existing_only(spec) if args.reuse_existing_only else run_or_load(spec, prepared, device)
        if not result:
            print("[skip] missing variant metrics for", spec.name)
            continue
        baseline = baselines[spec.baseline_name]["result"]
        enriched = attach_delta(result, baseline)
        payload = {"spec": serialize_spec(spec), "result": enriched}
        variant_meta.append(payload)
        print(
            "[variant] {name}: test_f1={tf:.4f}, delta_test_f1={df:+.4f}".format(
                name=spec.name,
                tf=result["test"]["f1"],
                df=enriched["delta_vs_baseline"]["test_f1"],
            )
        )

    specs["families"] = apply_available_parameter_coverage(specs["families"], variant_meta)
    parameter_groups = build_parameter_groups(specs["families"], baselines, variant_meta)
    best_by_family = build_best_by_family(parameter_groups)
    overall_best_variant = max(variant_meta, key=metric_sort_key) if variant_meta else None

    if args.families or args.parameters:
        union_families = merge_family_entries(specs["families"], previous_summary.get("families", []), "family")
        all_specs = build_specs(prepared.embedding_matrix)
        family_set = {item["family"] for item in union_families}
        all_specs = filter_specs(all_specs, selected_families=sorted(family_set))
        baseline_meta, variant_meta = load_existing_family_results(all_specs)
        baseline_map = {item["spec"]["name"]: item for item in baseline_meta}
        all_specs["families"] = apply_available_parameter_coverage(all_specs["families"], variant_meta)
        parameter_groups = build_parameter_groups(all_specs["families"], baseline_map, variant_meta)
        best_by_family = build_best_by_family(parameter_groups)
        specs["families"] = all_specs["families"]
        overall_best_variant = max(variant_meta, key=metric_sort_key) if variant_meta else None

    summary = {
        "device": str(device),
        "data": prepared.metadata,
        "methodology": (
            "Single-variable tuning. Each experiment changes exactly one hyperparameter while "
            "holding the corresponding family baseline fixed, so differences can be attributed "
            "to that parameter as directly as possible."
        ),
        "families": specs["families"],
        "baselines": baseline_meta,
        "variants": variant_meta,
        "parameter_groups": parameter_groups,
        "best_by_family": best_by_family,
        "overall_best_variant": overall_best_variant,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved hyperparameter tuning summary to", SUMMARY_PATH)


if __name__ == "__main__":
    main()
