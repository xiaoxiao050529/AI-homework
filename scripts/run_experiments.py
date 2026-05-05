"""运行全部情感分类实验并汇总结果。"""

import json
import os

import torch

from src.sentiment_hw2.data import prepare_data
from src.sentiment_hw2.experiment import build_ablation_specs, build_main_model_specs, run_robustness
from src.sentiment_hw2.models import initialize_model
from src.sentiment_hw2.paths import EMBEDDING_PATH, OUTPUT_CACHE_DIR, OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.sentiment_hw2.train import set_seed, train_single_model


def train_spec(spec, prepared, device):
    """按给定配置训练单个模型并返回汇总结果。"""
    set_seed(spec.train_config.seed)
    model = spec.builder()
    initialize_model(model)
    return train_single_model(
        model_name=spec.name,
        model=model,
        prepared=prepared,
        config=spec.train_config,
        device=device,
        output_dir=OUTPUT_DIR,
    )


def main():
    """串行执行数据准备、模型训练、对比实验和结果汇总。"""
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

    print("Device:", device)
    print("Vocab size:", prepared.metadata["vocab_size"])
    print("Train token coverage by pretrained vectors: {:.2%}".format(prepared.metadata["train_token_coverage"]))

    results = []
    trained_models = {}
    for spec in build_main_model_specs(prepared.embedding_matrix):
        result = train_spec(spec, prepared, device)
        results.append(result)

        # 用最优参数重新构建一份模型，供鲁棒性推理使用，避免依赖 train_single_model 的内部状态。
        model = spec.builder()
        model.load_state_dict(torch.load(result["checkpoint"], map_location=device))
        model.to(device)
        trained_models[spec.name] = model

        print(
            "{name}: val_acc={va:.4f}, val_f1={vf:.4f}, test_acc={ta:.4f}, test_f1={tf:.4f}".format(
                name=spec.name,
                va=result["validation"]["accuracy"],
                vf=result["validation"]["f1"],
                ta=result["test"]["accuracy"],
                tf=result["test"]["f1"],
            )
        )

    ablations = []
    for spec in build_ablation_specs(prepared.embedding_matrix):
        result = train_spec(spec, prepared, device)
        ablations.append(result)
        print(
            "{name}: val_acc={va:.4f}, val_f1={vf:.4f}".format(
                name=spec.name,
                va=result["validation"]["accuracy"],
                vf=result["validation"]["f1"],
            )
        )

    summary = {
        "device": str(device),
        "data": prepared.metadata,
        "results": results,
        "ablations": ablations,
        "robustness": run_robustness(trained_models, prepared, device),
    }
    with (OUTPUT_DIR / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Summary saved to", OUTPUT_DIR / "experiment_summary.json")


if __name__ == "__main__":
    main()
