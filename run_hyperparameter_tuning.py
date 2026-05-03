"""逐个超参数进行单变量调参与结果汇总。"""

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
from src.sentiment_hw2.train import TrainConfig, set_seed, train_single_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TUNING_DIR = OUTPUT_DIR / "tuning"
CACHE_DIR = OUTPUT_DIR / "cache"
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
            return "{:.0e}".format(value).replace("-", "")
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "_".join(slugify_value(item) for item in value)
    return str(value).replace(" ", "")


def build_specs(embedding_matrix: torch.Tensor) -> Dict[str, List[TuningSpec]]:
    """构造五类模型的基线与单变量调参实验。"""
    mlp_cfg = {"hidden_dim": 128, "dropout": 0.3}
    cnn_cfg = {"num_filters": 128, "filter_sizes": (3, 4, 5), "dropout": 0.5}
    recurrent_cfg = {"hidden_dim": 128, "num_layers": 1, "dropout": 0.3}

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

    family_defs: Dict[str, Dict[str, object]] = {
        "mlp": {
            "name_family": "mlp",
            "baseline_name": "tune_mlp_base",
            "description": "MLP 基线配置",
            "builder": lambda cfg: MeanPoolMLP(embedding_matrix, **cfg),
            "base_model_cfg": copy.deepcopy(mlp_cfg),
            "base_train_cfg": replace(mlp_train),
            "search_space": {
                "hidden_dim": [64, 128, 256],
                "dropout": [0.1, 0.3, 0.5],
                "batch_size": [64, 128, 256],
                "learning_rate": [5e-4, 1e-3, 2e-3],
                "weight_decay": [0.0, 1e-4, 5e-4],
                "epochs": [8, 10, 12],
                "patience": [2, 3, 4],
                "seed": [7, 42, 123],
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
                "num_filters": [64, 128, 192],
                "filter_sizes": [(2, 3, 4), (3, 4, 5), (3, 5, 7)],
                "dropout": [0.3, 0.5, 0.7],
                "batch_size": [64, 128, 256],
                "learning_rate": [5e-4, 1e-3, 2e-3],
                "weight_decay": [0.0, 1e-4, 5e-4],
                "epochs": [8, 10, 12],
                "patience": [2, 3, 4],
                "seed": [7, 42, 123],
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
                "hidden_dim": [64, 128, 192],
                "num_layers": [1, 2],
                "dropout": [0.1, 0.3, 0.5],
                "batch_size": [32, 64, 96],
                "learning_rate": [2e-4, 3e-4, 5e-4],
                "embedding_learning_rate": [5e-5, 8e-5, 1e-4],
                "weight_decay": [1e-4, 3e-4, 5e-4],
                "grad_clip": [0.5, 1.0, 2.0],
                "warmup_epochs": [1, 2, 3],
                "label_smoothing": [0.0, 0.05, 0.1],
                "freeze_embedding_epochs": [0, 2, 4],
                "epochs": [12, 14, 16],
                "patience": [3, 4, 5],
                "seed": [7, 42, 123],
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
                "hidden_dim": [64, 128, 192],
                "num_layers": [1, 2],
                "dropout": [0.1, 0.3, 0.5],
                "batch_size": [32, 64, 96],
                "learning_rate": [2e-4, 3e-4, 5e-4],
                "embedding_learning_rate": [5e-5, 8e-5, 1e-4],
                "weight_decay": [1e-4, 3e-4, 5e-4],
                "grad_clip": [0.5, 1.0, 2.0],
                "warmup_epochs": [1, 2, 3],
                "label_smoothing": [0.0, 0.05, 0.1],
                "freeze_embedding_epochs": [0, 2, 4],
                "epochs": [12, 14, 16],
                "patience": [3, 4, 5],
                "seed": [7, 42, 123],
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
                "hidden_dim": [64, 128, 192],
                "num_layers": [1, 2],
                "dropout": [0.1, 0.3, 0.5],
                "batch_size": [32, 64, 96],
                "learning_rate": [2e-4, 3e-4, 5e-4],
                "embedding_learning_rate": [5e-5, 8e-5, 1e-4],
                "weight_decay": [1e-4, 3e-4, 5e-4],
                "grad_clip": [0.5, 1.0, 2.0],
                "warmup_epochs": [1, 2, 3],
                "label_smoothing": [0.0, 0.05, 0.1],
                "freeze_embedding_epochs": [0, 2, 4],
                "epochs": [12, 14, 16],
                "patience": [3, 4, 5],
                "seed": [7, 42, 123],
            },
        },
    }

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


def run_or_load(spec: TuningSpec, prepared, device: torch.device) -> Dict[str, object]:
    """如果结果已存在则直接复用，否则执行训练。"""
    metrics_path = TUNING_DIR / "{}_metrics.json".format(spec.name)
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

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
    metrics_path = TUNING_DIR / "{}_metrics.json".format(spec.name)
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


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
        best = max(candidates, key=lambda entry: (entry["result"]["test"]["f1"], entry["result"]["test"]["accuracy"]))
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
                    "test_f1": best["result"]["test"]["f1"],
                    "test_accuracy": best["result"]["test"]["accuracy"],
                    "validation_f1": best["result"]["validation"]["f1"],
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
                "best_test_f1": group["best"]["test_f1"],
                "best_test_accuracy": group["best"]["test_accuracy"],
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


def load_previous_summary() -> Dict[str, object]:
    """在分批执行时复用已有摘要的其他模型结果。"""
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def filter_summary_items_by_families(items: Sequence[Dict[str, object]], families: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """在分批运行时，只保留当前目标模型族对应的历史条目。"""
    family_set = {item["family"] for item in families}
    filtered = []
    for item in items:
        spec = item.get("spec", {})
        if spec.get("family") in family_set:
            filtered.append(item)
    return filtered


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
        train_path=ROOT / "train.txt",
        validation_path=ROOT / "validation.txt",
        test_path=ROOT / "test.txt",
        embedding_path=ROOT / "wiki_word2vec_50.bin",
        cache_path=CACHE_DIR / "prepared_data.pt",
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

    parameter_groups = build_parameter_groups(specs["families"], baselines, variant_meta)
    best_by_family = build_best_by_family(parameter_groups)
    overall_best_variant = max(variant_meta, key=metric_sort_key) if variant_meta else None

    if previous_summary:
        previous_baselines = filter_summary_items_by_families(previous_summary.get("baselines", []), specs["families"])
        previous_variants = filter_summary_items_by_families(previous_summary.get("variants", []), specs["families"])
        baseline_meta = merge_by_spec_name(baseline_meta, previous_baselines)
        variant_meta = merge_by_spec_name(variant_meta, previous_variants)
        merged_families = merge_family_entries(specs["families"], previous_summary.get("families", []), "family")

        baseline_map = {item["spec"]["name"]: item for item in baseline_meta}
        parameter_groups = build_parameter_groups(merged_families, baseline_map, variant_meta)
        best_by_family = build_best_by_family(parameter_groups)
        specs["families"] = merged_families
        if variant_meta:
            overall_best_variant = max(variant_meta, key=metric_sort_key)

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
