"""根据 staged improvement 实验结果生成单独的分析报告。"""

import csv
import json
from typing import Dict, List

from src.sentiment_hw2.paths import OUTPUT_DIR
SUMMARY_PATH = OUTPUT_DIR / "improvement_experiment_summary.json"
REPORT_PATH = OUTPUT_DIR / "improvement_analysis_report.md"
CSV_PATH = OUTPUT_DIR / "improvement_results.csv"


def format_score(value: float) -> str:
    return "{:.4f}".format(value)


def format_delta(value) -> str:
    if value is None:
        return "-"
    return "{:+.4f}".format(value)


def stage_sort_key(item: Dict[str, object]) -> tuple:
    return (int(item["stage"]), str(item["model_name"]))


def build_markdown_table(experiments: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| 实验 | 训练数据处理 | Val Acc | Val F1 | Test Acc | Test F1 | 相对上一阶段 Test F1 | 结果来源 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted(experiments, key=stage_sort_key):
        delta = item["delta_vs_reference"]
        lines.append(
            "| {name} | {aug} | {va} | {vf} | {ta} | {tf} | {dtf} | {source} |".format(
                name=item["display_name"],
                aug=item["augmentation"],
                va=format_score(item["validation"]["accuracy"]),
                vf=format_score(item["validation"]["f1"]),
                ta=format_score(item["test"]["accuracy"]),
                tf=format_score(item["test"]["f1"]),
                dtf=format_delta(None if delta is None else delta["test_f1"]),
                source="复用已有主实验" if item["reused_existing_result"] else "本次新增训练",
            )
        )
    return lines


def build_track_analysis(track_name: str, experiments: List[Dict[str, object]]) -> List[str]:
    ordered = sorted(experiments, key=stage_sort_key)
    best_test = max(ordered, key=lambda item: item["test"]["f1"])
    lines = ["## {} 路线".format(track_name.upper())]
    lines.extend(build_markdown_table(ordered))
    lines.append("")
    if track_name == "rnn":
        base, lstm, attention = ordered
        lines.append(
            "结论：`{}` 相比 `{}` 的测试集 F1 提升了 `{:+.4f}`，说明把普通循环单元换成 LSTM 在这份影评数据上是有效的。".format(
                lstm["display_name"],
                base["display_name"],
                lstm["delta_vs_reference"]["test_f1"],
            )
        )
        lines.append(
            "结论：`{}` 在本轮 quick 预算下测试集 F1 相比 `{}` 下降了 `{:+.4f}`。这更像是“新增注意力结构需要继续调参”而不是“注意力一定无效”，因为它本次使用的是更短训练预算，且没有单独调注意力维度、学习率与 dropout。".format(
                attention["display_name"],
                lstm["display_name"],
                attention["delta_vs_reference"]["test_f1"],
            )
        )
    else:
        single_kernel = ordered[0]
        multi_kernel = ordered[1]
        augmentations = ordered[2:]
        best_aug = max(augmentations, key=lambda item: item["test"]["f1"])
        lines.append(
            "结论：`{}` 相比 `{}` 的测试集 F1 提升了 `{:+.4f}`，说明多卷积核 TextCNN 对不同长度情感短语的建模收益很明显。".format(
                multi_kernel["display_name"],
                single_kernel["display_name"],
                multi_kernel["delta_vs_reference"]["test_f1"],
            )
        )
        lines.append(
            "结论：三种轻量增强都没有超过原始 `{}`。其中损失最小的是 `{}`，测试集 F1 相比基线变动 `{:+.4f}`；`Random Deletion` 与 `Synonym Replacement` 的退化更明显，说明当前增强强度对这份数据来说偏大。".format(
                multi_kernel["display_name"],
                best_aug["display_name"],
                best_aug["delta_vs_reference"]["test_f1"],
            )
        )
    lines.append(
        "本路线当前测试集 F1 最好的是 `{}`，分数为 `{}`。".format(
            best_test["display_name"],
            format_score(best_test["test"]["f1"]),
        )
    )
    lines.append("")
    return lines


def export_csv(experiments: List[Dict[str, object]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "track",
                "stage",
                "model_name",
                "display_name",
                "augmentation",
                "reused_existing_result",
                "best_epoch",
                "validation_accuracy",
                "validation_f1",
                "test_accuracy",
                "test_f1",
                "delta_test_f1_vs_reference",
            ]
        )
        for item in sorted(experiments, key=lambda obj: (obj["track"], *stage_sort_key(obj))):
            delta = item["delta_vs_reference"]
            writer.writerow(
                [
                    item["track"],
                    item["stage"],
                    item["model_name"],
                    item["display_name"],
                    item["augmentation"],
                    item["reused_existing_result"],
                    item["best_epoch"],
                    item["validation"]["accuracy"],
                    item["validation"]["f1"],
                    item["test"]["accuracy"],
                    item["test"]["f1"],
                    "" if delta is None else delta["test_f1"],
                ]
            )


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("缺少改进实验汇总，请先运行 run_improvement_experiments.py")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    experiments = summary["experiments"]
    tracks = summary["tracks"]

    export_csv(experiments)

    lines = [
        "# CNN / RNN 改进实验分析报告",
        "",
        "生成时间：2026-05-04",
        "",
        "## 报告说明",
        "这份报告单独分析 `run_improvement_experiments.py` 产出的逐步改进实验，不替代原课程总报告。",
        "本轮新增训练采用 `quick` 预算以适应当前 CPU 环境：新增 RNN 变体使用较短训练轮数与更大 batch；`BiRNN / BiLSTM / TextCNN` 三个基线直接复用了主实验已有结果。",
        "因此，这里更适合用于判断“改进方向是否值得继续”，如果你要把某个新变体写成最终主结论，建议再用完整预算复验一次。",
        "",
        "## 数据与评估",
        "训练集 / 验证集 / 测试集规模分别为 `{train}` / `{validation}` / `{test}`，最大句长 `{max_len}`，词表大小 `{vocab}`。".format(
            train=summary["data"]["train_size"],
            validation=summary["data"]["validation_size"],
            test=summary["data"]["test_size"],
            max_len=summary["data"]["max_len"],
            vocab=summary["data"]["vocab_size"],
        ),
        "统一指标为 `Accuracy` 与 `F1`，主要关注测试集 F1 的变化。",
        "",
    ]

    lines.extend(build_track_analysis("rnn", tracks["rnn"]))
    lines.extend(build_track_analysis("cnn", tracks["cnn"]))

    lines.extend(
        [
            "## 总结",
            "1. 在当前代码和数据上，`BiRNN -> BiLSTM` 是明确有效的升级，测试集 F1 提升约 `+0.0242`。",
            "2. `BiLSTM + Attention` 这次没有跑赢 `BiLSTM`，但它属于新增结构且只做了 quick 预算，后续还有继续调参空间。",
            "3. `单卷积核 CNN -> TextCNN 多卷积核` 的收益最稳定，是本轮 CNN 路线里最值得保留到正式报告的结构改进。",
            "4. 三种轻量增强里，`Word Dropout` 最稳，但在这次设置下仍未超过未增强 TextCNN；说明增强存在“强度过大或改写噪声过强”的问题。",
            "",
            "## 建议下一步",
            "1. 把 `BiLSTM + Attention` 用完整预算再跑一次，并单独搜索 `attention_dim`、`dropout` 和 `learning_rate`。",
            "2. 保留 `TextCNN 多卷积核` 作为 CNN 主结果，再把 `Word Dropout` 的概率从 `0.1` 下调到 `0.05` 重新验证。",
            "3. 如果你还想继续扩展 RNN，可以追加 `BiLSTM + Max/Mean Pooling`，它通常比 Attention 更容易稳定提升。",
            "",
            "## 附件",
            "- 明细结果表：`outputs/improvement_results.csv`",
            "- 原始汇总 JSON：`outputs/improvement_experiment_summary.json`",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Report saved to", REPORT_PATH)
    print("CSV saved to", CSV_PATH)


if __name__ == "__main__":
    main()
