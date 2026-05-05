"""单独训练 Transformer 模型，并与现有主模型做效率对比。"""

import argparse
import json
import os
from dataclasses import replace

import torch

from src.sentiment_hw2.data import prepare_data
from src.sentiment_hw2.experiment import build_transformer_spec, run_robustness
from src.sentiment_hw2.models import initialize_model
from src.sentiment_hw2.paths import EMBEDDING_PATH, OUTPUT_CACHE_DIR, OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.sentiment_hw2.train import set_seed, train_single_model


SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.json"
COMPARISON_PATH = OUTPUT_DIR / "transformer_efficiency_comparison.json"


def load_existing_metric(model_name: str):
    metrics_path = OUTPUT_DIR / "{}_metrics.json".format(model_name)
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def update_summary(prepared, transformer_result, transformer_robustness):
    """把 Transformer 结果并入现有实验汇总，便于后续统一生成报告。"""
    if SUMMARY_PATH.exists():
        payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "device": "unknown",
            "data": prepared.metadata,
            "results": [],
            "ablations": [],
            "robustness": {},
        }

    result_by_name = {item["model_name"]: item for item in payload.get("results", [])}
    result_by_name["transformer"] = transformer_result
    payload["results"] = list(result_by_name.values())
    payload["data"] = prepared.metadata
    payload.setdefault("robustness", {})
    payload["robustness"]["transformer"] = transformer_robustness
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer and compare efficiency.")
    parser.add_argument("--quick", action="store_true", help="只跑 1 个 epoch 的快速基准，不更新主实验汇总。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
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

    spec = build_transformer_spec(prepared.embedding_matrix)
    model_name = spec.name
    train_config = spec.train_config
    if args.quick:
        model_name = "transformer_quickcheck"
        train_config = replace(spec.train_config, epochs=1, patience=1, min_epochs=1)
    set_seed(train_config.seed)
    model = spec.builder()
    initialize_model(model)
    transformer_result = train_single_model(
        model_name=model_name,
        model=model,
        prepared=prepared,
        config=train_config,
        device=device,
        output_dir=OUTPUT_DIR,
    )

    reloaded = spec.builder()
    reloaded.load_state_dict(torch.load(transformer_result["checkpoint"], map_location=device))
    reloaded.to(device)
    transformer_robustness = run_robustness({"transformer": reloaded}, prepared, device)["transformer"]
    if not args.quick:
        update_summary(prepared, transformer_result, transformer_robustness)

    comparisons = []
    baseline_names = ["mlp", "cnn", "birnn", "bilstm", "bigru"]
    for name in baseline_names:
        baseline = load_existing_metric(name)
        if baseline is None:
            continue
        comparisons.append(
            {
                "reference_model": name,
                "transformer_train_seconds": transformer_result["train_seconds"],
                "reference_train_seconds": baseline["train_seconds"],
                "transformer_vs_reference_speedup": baseline["train_seconds"] / max(transformer_result["train_seconds"], 1e-8),
                "transformer_test_f1": transformer_result["test"]["f1"],
                "reference_test_f1": baseline["test"]["f1"],
                "transformer_f1_delta": transformer_result["test"]["f1"] - baseline["test"]["f1"],
                "transformer_params": transformer_result["parameter_count"],
                "reference_params": baseline["parameter_count"],
            }
        )

    COMPARISON_PATH.write_text(
        json.dumps(
            {
                "transformer": transformer_result,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "transformer: val_f1={:.4f}, test_f1={:.4f}, train_seconds={:.2f}, params={}".format(
            transformer_result["validation"]["f1"],
            transformer_result["test"]["f1"],
            transformer_result["train_seconds"],
            transformer_result["parameter_count"],
        )
    )
    for item in comparisons:
        print(
            "{ref}: speedup={speed:.2f}x, f1_delta={delta:+.4f}".format(
                ref=item["reference_model"],
                speed=item["transformer_vs_reference_speedup"],
                delta=item["transformer_f1_delta"],
            )
        )
    print("saved comparison to {}".format(COMPARISON_PATH))


if __name__ == "__main__":
    main()
