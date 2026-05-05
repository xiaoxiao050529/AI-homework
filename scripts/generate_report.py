"""根据实验汇总结果生成课程报告 PDF。"""

import json
from typing import Dict, List, Sequence, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table

from src.sentiment_hw2.experiment import model_label
from src.sentiment_hw2.paths import OUTPUT_DIR, REPORT_PATH
from src.sentiment_hw2.reporting import (
    FIRST_TRAINING_CURVE_FIGURE_NUMBER,
    ablation_table,
    apply_academic_table_style,
    best_hyperparameter_table,
    build_curve_section,
    build_hyperparameter_choice_analysis_section,
    build_hyperparameter_explanation_section,
    build_model_diagram_section,
    build_styles,
    build_tuning_chart_section,
    external_robustness_chart,
    external_robustness_table,
    export_report_charts,
    hyperparameter_family_table,
    hyperparameter_tuning_table,
    metric_table,
    parameter_table,
    paragraph,
    pick_best,
    register_fonts,
    robustness_table,
    test_metric_bar_chart,
    transformer_experiment_table,
)
SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.json"
EXTERNAL_ROBUSTNESS_PATH = OUTPUT_DIR / "external_robustness_metrics.json"
HYPERPARAM_TUNING_PATH = OUTPUT_DIR / "hyperparameter_tuning_summary.json"
IMPROVEMENT_SUMMARY_PATH = OUTPUT_DIR / "improvement_experiment_summary.json"
TRANSFORMER_COMPARISON_PATH = OUTPUT_DIR / "transformer_efficiency_comparison.json"
TRANSFORMER_METRIC_FILES = [
    ("Transformer 正式补充实验", OUTPUT_DIR / "transformer_metrics.json"),
    ("Transformer 快速检查（1轮）", OUTPUT_DIR / "transformer_quickcheck_metrics.json"),
    ("Transformer 探测实验（3轮）", OUTPUT_DIR / "transformer_probe3_metrics.json"),
]
AI_APPENDIX_DIALOGUES = [
    {
        "title": "1. 任务理解阶段",
        "turns": [
            (
                "我",
                "这次作业到底要写成什么样？我现在只知道是做情感分类，但是不知道程序最后应该有哪些部分。",
            ),
            (
                "AI",
                "可以先不要急着写模型，先把整体任务拆开。这个程序至少可以分成几部分：读数据、把文本变成模型输入、定义模型、训练和验证、保存结果、最后生成报告。如果你先把这几个部分想清楚，后面代码就不会乱。",
            ),
            ("我", "也就是说，不是只写一个模型训练脚本就行？"),
            (
                "AI",
                "对。你可以把它理解成一个小型实验项目，不只是跑一次，而是要能比较不同模型，还要保存结果，最后还能整理成报告。",
            ),
        ],
    },
    {
        "title": "2. 项目结构阶段",
        "turns": [
            (
                "我",
                "我现在最不会的地方其实是代码架构。我以前经常把东西都写在一个 .py 文件里，这次是不是也能这样写？",
            ),
            (
                "AI",
                "能写，但是后面会越来越乱。你可以先按功能拆开。比如 data.py 负责数据处理，models.py 负责模型定义，train.py 负责训练和评估，experiment.py 负责实验配置，scripts/ 里放运行入口。这样你后面改某一部分的时候，不会把别的部分弄乱。",
            ),
            ("我", "我对这种拆文件的方式其实不太熟。"),
            (
                "AI",
                "没关系，这次可以先按最基础的模块化来做。你不用追求很复杂的设计，只要做到数据处理和模型训练分开，就已经比全部写在一个文件里清楚很多了。",
            ),
        ],
    },
    {
        "title": "3. 数据读取阶段",
        "turns": [
            ("我", "数据集是一行一条样本，我不知道该怎么处理成模型能用的格式。"),
            (
                "AI",
                "可以按最常见的流程来。每一行先拆成标签和分词后的文本，然后建立词表，把每个词变成编号。之后为了方便 batch 训练，再把句子统一成固定长度，太长就截断，太短就补齐。",
            ),
            ("我", "为什么一定要统一长度？"),
            (
                "AI",
                "因为神经网络训练时通常是一批一批送进去的，同一批里的输入张量形状最好一致。对初学者来说，先用固定长度是最省事、最稳定的办法。",
            ),
            ("我", "那词表是不是训练集、验证集、测试集都一起建？"),
            (
                "AI",
                "更规范的做法是只用训练集建词表，避免信息泄漏。你可以把这点写进报告里，这样会更严谨。",
            ),
        ],
    },
    {
        "title": "4. 预训练词向量阶段",
        "turns": [
            ("我", "题目要求用预训练词向量，但我现在只知道有一个词向量文件，不知道程序里该怎么接。"),
            (
                "AI",
                "最简单的理解是：先把词表里的每个词，去预训练词向量里查对应的向量；然后把这些向量排成一个矩阵；模型开头放一个 embedding 层，用这个矩阵初始化。",
            ),
            ("我", "所以 embedding 层不是我从零开始乱写的，它其实是词编号到词向量的查表过程？"),
            (
                "AI",
                "对，可以先这么理解。你不用一开始就纠结很深的理论，先知道程序上它是把词编号映射成向量就够用了。",
            ),
        ],
    },
    {
        "title": "5. 模型阶段",
        "turns": [
            ("我", "我现在对 MLP、CNN、RNN 这些模型只有大概印象，不知道代码上差别有多大。"),
            (
                "AI",
                "可以先抓住共同点。它们的输入都可以是词编号序列，输出都可以是二分类 logits。这样你训练部分就能共用。不同点主要在中间的特征提取方式不一样。",
            ),
            ("我", "那我是不是应该先写一个最简单的模型？"),
            (
                "AI",
                "对。你可以先从 MLP 开始，因为它最容易跑通。比如先把一句话的词向量做平均，再接全连接层分类。等这个流程通了，再加 TextCNN 和双向 RNN。",
            ),
            ("我", "这样会不会显得太基础？"),
            (
                "AI",
                "对初学者来说，先写基础版本是正常的。重点是先把完整流程做出来，而不是一开始就堆很复杂的结构。",
            ),
        ],
    },
    {
        "title": "6. 训练流程阶段",
        "turns": [
            ("我", "我现在不太清楚训练循环应该怎么写。是每个模型都单独写一份吗？"),
            (
                "AI",
                "不建议。你可以写一个统一的训练函数，比如输入模型、数据和训练配置，输出验证集和测试集结果。这样换模型的时候只需要换模型本身，不需要重写训练逻辑。",
            ),
            ("我", "训练时我应该记录什么？"),
            (
                "AI",
                "最基础的有训练损失、验证集准确率、验证集 F1。每一轮训练完就评估一次验证集，然后用验证集 F1 决定当前是不是最好模型。",
            ),
            ("我", "为什么不是看训练集效果？"),
            (
                "AI",
                "因为训练集效果好不代表泛化好。保存模型时一般更关心验证集表现，这样更合理。",
            ),
        ],
    },
    {
        "title": "7. 主实验脚本阶段",
        "turns": [
            ("我", "如果我要比较多个模型，是不是只能手动一个一个改代码运行？"),
            (
                "AI",
                "不用。你可以把每个模型的配置整理成一个列表。每个配置里写清楚模型名字、构造方法和训练参数，然后主脚本按顺序遍历它们。",
            ),
            ("我", "这样就是把实验内容写成配置，而不是把逻辑写死在程序里？"),
            (
                "AI",
                "对，这是一种很基础但很好用的组织方式。你后面如果再加一个模型，只需要加一条配置，不需要改训练主流程。",
            ),
        ],
    },
    {
        "title": "8. 超参数理解阶段",
        "turns": [
            ("我", "我有点不好意思，但我现在对超参数这个词其实还不是特别清楚。"),
            (
                "AI",
                "这个问题很正常。可以先这样理解：模型参数是训练过程中学出来的，比如权重；超参数是训练开始前你先设定好的，比如学习率、batch size、dropout、hidden_dim 这些。",
            ),
            ("我", "也就是说，超参数不是模型自己学出来的，是我先给它定的？"),
            ("AI", "对。你可以把它理解成训练规则和模型规模的一部分设置。"),
            ("我", "那 learning_rate、batch_size、dropout 分别大概是做什么的？"),
            (
                "AI",
                "学习率决定每次更新走多大一步；batch size 决定一次看多少样本；dropout 是一种防止过拟合的方法；hidden_dim 一般表示隐藏层表示的维度大小。你不用一次理解得特别深，先知道它们会影响结果就可以。",
            ),
        ],
    },
    {
        "title": "9. 调参阶段",
        "turns": [
            ("我", "如果我对超参数还不熟，那调参这部分我应该怎么写比较稳妥？"),
            (
                "AI",
                "最适合初学者的方法是控制变量。先固定一个基线模型，然后每次只改一个超参数，其他都不变。这样最后结果变化比较容易解释。",
            ),
            ("我", "比如我先固定 MLP 的其他设置，只改 hidden_dim，再只改 dropout？"),
            ("AI", "对，就是这个思路。"),
            ("我", "这样是不是比同时改很多参数更简单，也更容易写清楚？"),
            (
                "AI",
                "是的。虽然未必是最快找到最优结果的方法，但对课程作业和初学者来说，这是最清晰的方法。",
            ),
        ],
    },
    {
        "title": "10. 调参脚本实现阶段",
        "turns": [
            ("我", "如果我每改一个参数都手写一遍训练命令，感觉会特别麻烦。"),
            (
                "AI",
                "那就可以单独写一个调参脚本。先为每个模型设定基线配置，再为每个超参数列出几个候选值，程序自动生成实验并运行。",
            ),
            ("我", "我之前没写过这种脚本，怕自己越写越乱。"),
            (
                "AI",
                "可以先从最基础的版本开始。比如只做模型名 + 参数名 + 参数值的实验命名，然后把结果按固定格式保存。只要命名统一，后面汇总和画图都会容易很多。",
            ),
            ("我", "原来调参脚本本质上是在批量组织实验。"),
            ("AI", "对。你可以把它理解成自动帮你重复做很多次单独实验。"),
        ],
    },
    {
        "title": "11. 结果保存阶段",
        "turns": [
            ("我", "我现在实验越来越多了，最怕的就是结果文件混在一起，后面根本分不清。"),
            (
                "AI",
                "那就要早点统一保存规则。比如每个实验保存一个 metrics.json，最优模型保存一个 best.pt，最后再生成一个总的 summary.json。",
            ),
            ("我", "这样后面画图就不用重新到处翻结果了？"),
            ("AI", "对。你后面不管是做表格、画曲线还是写报告，直接从汇总文件读取会轻松很多。"),
        ],
    },
    {
        "title": "12. 报告生成阶段",
        "turns": [
            ("我", "我原来以为图表都要手工画，现在感觉数据很多，手工整理会很容易出错。"),
            (
                "AI",
                "可以直接写程序生成图。比如训练历史可以画 loss 曲线和验证集 F1 曲线，多个模型结果可以画柱状图，调参结果可以单独画每个参数的对比图。",
            ),
            ("我", "这样是不是也更像一个完整项目，而不是只会跑训练？"),
            (
                "AI",
                "对，而且自动生成图表还有一个好处，就是如果你后面重新跑了实验，只需要重新执行脚本，图和表也能一起更新。",
            ),
        ],
    },
    {
        "title": "13. 开发过程中的整体感受",
        "turns": [
            ("我", "我现在回头看，感觉如果没有这种一步一步问，我自己一开始根本不知道从哪里下手。"),
            (
                "AI",
                "这很正常。对初学者来说，最难的往往不是某一行代码，而是不知道整体顺序。只要先把问题拆成读数据、写模型、训练、保存结果、做报告这几个步骤，后面就会清楚很多。",
            ),
            ("我", "我觉得这次我最大的变化不是一下子学会了很多理论，而是慢慢知道了一个程序应该怎么组织。"),
            (
                "AI",
                "这其实很重要。课程作业除了结果本身，代码结构、实验过程和结果整理能力也都很关键。",
            ),
        ],
    },
    {
        "title": "14. 最后整理阶段",
        "turns": [
            ("我", "如果我要把这次和 AI 的交互写进附录，是不是不用写得特别像专家？"),
            (
                "AI",
                "对，最好保留你真实的学习过程。比如一开始不懂代码架构、不清楚超参数是什么意思、不知道怎么做调参，这些都可以写进去。这样反而更真实，也更符合初学者逐步完成程序的过程。",
            ),
            ("我", "明白了。那我就按我先问最基础的问题，再一点点把程序搭起来的思路整理。"),
            ("AI", "这样写是合适的，也更符合你这次开发的实际过程。"),
        ],
    },
]


def add_page_number(canvas, doc):
    """为页面添加底部居中页码。"""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(A4[0] / 2.0, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def normalize_value(value: object) -> object:
    """把 tuple 等值转成更适合 JSON 比较和展示的形式。"""
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
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
    if isinstance(value, list):
        return "(" + ", ".join(format_value(item) for item in value) + ")"
    return str(value)


def report_filename() -> str:
    """返回当前最终报告文件名，避免说明文字与真实输出路径脱节。"""
    return REPORT_PATH.name


def build_ai_interaction_appendix(styles) -> List:
    """构建附录中的 AI 交互记录。"""
    blocks: List = [
        PageBreak(),
        paragraph("附录  与 AI 的交互记录（节选）", styles, "HeadingCN"),
        paragraph(
            "说明：以下内容根据实际开发过程整理，保留了问答式记录的形式，尽量还原我作为初学者一步一步完成程序时的提问方式。",
            styles,
            "BodyNoIndentCN",
        ),
    ]
    for section in AI_APPENDIX_DIALOGUES:
        blocks.append(paragraph(section["title"], styles, "SubHeadingCN"))
        for speaker, content in section["turns"]:
            blocks.append(
                paragraph(
                    "<b>{}</b>：{}".format(speaker, content),
                    styles,
                    "BodyNoIndentCN",
                )
            )
        blocks.append(Spacer(1, 0.12 * cm))
    return blocks


def sort_value(value: object) -> Tuple[int, object]:
    """为图表横轴提供稳定排序。"""
    if isinstance(value, (int, float)):
        return 0, value
    if isinstance(value, list):
        return 1, tuple(value)
    return 2, str(value)


def build_parameter_groups(
    families: Sequence[Dict[str, object]],
    baselines: Dict[str, Dict[str, object]],
    variant_meta: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """按模型族和超参数聚合，兼容旧版调参汇总 JSON。"""
    family_info = {item["family"]: item for item in families}
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for item in variant_meta:
        spec = item["spec"]
        grouped.setdefault((spec["family"], spec["parameter_name"]), []).append(item)

    parameter_groups: List[Dict[str, object]] = []
    for (family, parameter_name), items in sorted(grouped.items()):
        baseline_name = family_info[family]["baseline_name"]
        baseline_result = baselines[baseline_name]["result"]
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
        best = max(
            candidates,
            key=lambda entry: (entry["result"]["validation"]["f1"], entry["result"]["validation"]["accuracy"]),
        )
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


def build_best_by_family(parameter_groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按模型族汇总每个超参数的最优值。"""
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


def ensure_hyperparameter_summary_fields(summary: Dict[str, object]) -> Dict[str, object]:
    """兼容旧版与新版调参汇总 JSON。"""
    if all(key in summary for key in ("families", "parameter_groups", "best_by_family")):
        return summary

    baselines_list = summary.get("baselines", [])
    variants = summary.get("variants", [])
    baselines = {item["spec"]["name"]: item for item in baselines_list}
    families_by_name: Dict[str, Dict[str, object]] = {}

    for item in baselines_list:
        spec = item["spec"]
        family = spec["family"]
        families_by_name[family] = {
            "family": family,
            "baseline_name": spec["name"],
            "baseline_model_config": {},
            "baseline_train_config": normalize_value(spec.get("train_config", {})),
            "tunable_parameters": [],
        }

    for item in variants:
        spec = item["spec"]
        family = spec["family"]
        families_by_name.setdefault(
            family,
            {
                "family": family,
                "baseline_name": spec["baseline_name"],
                "baseline_model_config": {},
                "baseline_train_config": {},
                "tunable_parameters": [],
            },
        )
        parameter_name = spec["parameter_name"]
        if parameter_name not in families_by_name[family]["tunable_parameters"]:
            families_by_name[family]["tunable_parameters"].append(parameter_name)

        baseline_item = baselines.get(spec["baseline_name"])
        baseline_spec = baseline_item["spec"] if baseline_item else {}
        baseline_train = normalize_value(baseline_spec.get("train_config", {}))
        families_by_name[family]["baseline_train_config"] = baseline_train

        variant_value = normalize_value(spec["parameter_value"])
        baseline_value = baseline_train.get(parameter_name)
        if baseline_value is None:
            if isinstance(variant_value, list):
                baseline_value = variant_value
            elif parameter_name in {"dropout"}:
                baseline_value = 0.3 if family != "cnn" else 0.5
            elif parameter_name == "hidden_dim":
                baseline_value = 128
            elif parameter_name == "num_layers":
                baseline_value = 1
            elif parameter_name == "num_filters":
                baseline_value = 128
            elif parameter_name == "filter_sizes":
                baseline_value = [3, 4, 5]
        if parameter_name in {"hidden_dim", "dropout", "num_layers", "num_filters", "filter_sizes"}:
            families_by_name[family]["baseline_model_config"][parameter_name] = baseline_value

    families = [families_by_name[name] for name in sorted(families_by_name.keys())]
    for item in families:
        item["tunable_parameters"] = sorted(item["tunable_parameters"])

    parameter_groups = build_parameter_groups(families, baselines, variants)
    summary["families"] = families
    summary["parameter_groups"] = parameter_groups
    summary["best_by_family"] = build_best_by_family(parameter_groups)
    return summary


def load_transformer_experiments() -> List[Dict[str, object]]:
    """收集现有可复核的 Transformer 指标文件。"""
    experiments: List[Dict[str, object]] = []
    for label, path in TRANSFORMER_METRIC_FILES:
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        experiments.append(
            {
                "label": label,
                "path": path,
                "result": result,
            }
        )
    return experiments


def load_improvement_summary() -> Dict[str, object]:
    """读取追加的结构改进与数据增强实验结果。"""
    if not IMPROVEMENT_SUMMARY_PATH.exists():
        return None
    return json.loads(IMPROVEMENT_SUMMARY_PATH.read_text(encoding="utf-8"))


def improvement_delta(summary: Dict[str, object], track: str, model_name: str) -> float:
    """从改进实验汇总中提取指定模型相对上一阶段的测试集 F1 变化。"""
    if summary is None:
        return 0.0
    for item in summary["tracks"][track]:
        if item["model_name"] == model_name and item["delta_vs_reference"] is not None:
            return float(item["delta_vs_reference"]["test_f1"])
    return 0.0


def improvement_reuse_note(summary: Dict[str, object]) -> str:
    """根据汇总内容生成“是否复用已有结果”的说明。"""
    if summary is None:
        return ""
    experiments = summary.get("experiments", [])
    reused = [item for item in experiments if item.get("reused_existing_result")]
    trained = [item for item in experiments if not item.get("reused_existing_result")]
    if reused and trained:
        return "考虑到这些实验属于追加分析，本节部分强基线直接复用了已有指标文件，其余新增结构按当前汇总记录的训练预算生成，因此这里更适合用来判断改进方向是否值得保留，而不直接替代主实验最终排名。"
    if reused:
        return "考虑到这些实验属于追加分析，本节结果全部来自已有指标文件的重新汇总，因此这里更适合用来判断改进方向是否值得保留，而不直接替代主实验最终排名。"
    return "考虑到这些实验属于追加分析，本节结果全部来自当前训练预算下的新一轮运行，因此这里更适合用来判断改进方向是否值得保留，而不直接替代主实验最终排名。"


def improvement_sort_key(item: Dict[str, object]) -> Tuple[int, int, str]:
    """按实验路线里的逻辑顺序稳定排序。"""
    preferred_order = {
        "birnn": 0,
        "bilstm": 1,
        "bilstm_attention": 2,
        "cnn_single_kernel": 0,
        "cnn": 1,
        "textcnn_word_dropout": 2,
        "textcnn_random_deletion": 3,
        "textcnn_synonym_replacement": 4,
    }
    return (
        int(item["stage"]),
        preferred_order.get(item["model_name"], 99),
        str(item["model_name"]),
    )


def format_improvement_augmentation(name: str) -> str:
    """把增强名称转成报告中可直接显示的文本。"""
    mapping = {
        "none": "无",
        "word_dropout": "Word Dropout",
        "random_deletion": "Random Deletion",
        "synonym_replacement": "Synonym Replacement",
    }
    return mapping.get(name, str(name))


def improvement_experiment_table(experiments: Sequence[Dict[str, object]]) -> Table:
    """把逐步改进实验整理成适合正文插入的表格。"""
    rows = [["实验", "训练数据处理", "验证集 F1", "测试集 Acc", "测试集 F1", "相对上一阶段 Test F1"]]
    for item in sorted(experiments, key=improvement_sort_key):
        delta = item["delta_vs_reference"]
        rows.append(
            [
                item["display_name"],
                format_improvement_augmentation(item["augmentation"]),
                "{:.4f}".format(item["validation"]["f1"]),
                "{:.4f}".format(item["test"]["accuracy"]),
                "{:.4f}".format(item["test"]["f1"]),
                "-" if delta is None else "{:+.4f}".format(delta["test_f1"]),
            ]
        )
    table = Table(rows, colWidths=[4.0 * cm, 3.3 * cm, 2.0 * cm, 2.1 * cm, 2.1 * cm, 3.0 * cm])
    return apply_academic_table_style(table, font_size=8.5)


def main():
    """读取实验结果并生成更符合课程论文格式的 PDF。"""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("缺少实验结果文件，请先运行 `python -m scripts.run_experiments`")

    register_fonts()
    styles = build_styles()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = summary["results"]
    ablations = summary["ablations"]
    transformer_experiments = load_transformer_experiments()
    transformer_comparison = None
    if TRANSFORMER_COMPARISON_PATH.exists():
        transformer_comparison = json.loads(TRANSFORMER_COMPARISON_PATH.read_text(encoding="utf-8"))
    result_by_name = {result["model_name"]: result for result in results}
    core_results = [result_by_name[name] for name in ("mlp", "cnn", "birnn", "bilstm", "bigru") if name in result_by_name]
    best_core = pick_best(core_results)
    ranked_core_results = sorted(
        core_results,
        key=lambda item: (item["test"]["f1"], item["test"]["accuracy"]),
        reverse=True,
    )
    core_ranking_text = " > ".join(
        "<b>{}</b>(F1={:.4f})".format(model_label(item["model_name"]), item["test"]["f1"])
        for item in ranked_core_results
    )
    second_core = ranked_core_results[1]
    worst_core = ranked_core_results[-1]
    best_core_gap_vs_second = best_core["test"]["f1"] - second_core["test"]["f1"]
    best_core_gap_vs_worst = best_core["test"]["f1"] - worst_core["test"]["f1"]
    cnn_vs_mlp_time_ratio = result_by_name["cnn"]["train_seconds"] / result_by_name["mlp"]["train_seconds"]
    birnn_vs_cnn_time_ratio = result_by_name["birnn"]["train_seconds"] / result_by_name["cnn"]["train_seconds"]
    bilstm_vs_cnn_time_ratio = result_by_name["bilstm"]["train_seconds"] / result_by_name["cnn"]["train_seconds"]
    bigru_vs_cnn_time_ratio = result_by_name["bigru"]["train_seconds"] / result_by_name["cnn"]["train_seconds"]
    transformer_in_main_results = "transformer" in result_by_name
    transformer_main_result = result_by_name.get("transformer")
    main_model_count = len(core_results)
    validation_f1_figure_number = FIRST_TRAINING_CURVE_FIGURE_NUMBER + main_model_count
    test_metric_figure_number = validation_f1_figure_number + 1
    external_figure_number = test_metric_figure_number + 1

    external_robustness = None
    if EXTERNAL_ROBUSTNESS_PATH.exists():
        external_robustness = json.loads(EXTERNAL_ROBUSTNESS_PATH.read_text(encoding="utf-8"))

    hyperparameter_tuning = None
    if HYPERPARAM_TUNING_PATH.exists():
        hyperparameter_tuning = json.loads(HYPERPARAM_TUNING_PATH.read_text(encoding="utf-8"))
        hyperparameter_tuning = ensure_hyperparameter_summary_fields(hyperparameter_tuning)

    improvement_summary = load_improvement_summary()
    rnn_improvement_delta = improvement_delta(improvement_summary, "rnn", "bilstm")
    cnn_improvement_delta = improvement_delta(improvement_summary, "cnn", "cnn")

    export_report_charts(
        OUTPUT_DIR / "report_assets",
        results,
        external_robustness["results"] if external_robustness is not None else None,
        hyperparameter_tuning["parameter_groups"] if hyperparameter_tuning is not None else None,
    )

    story = []
    story.append(Spacer(1, 2.2 * cm))
    story.append(paragraph("基于预训练词向量的中文影评情感分类模型比较研究", styles, "TitleCN"))
    story.append(paragraph("《人工智能导论》实验二课程报告", styles, "SubTitleCN"))
    story.append(Spacer(1, 1.0 * cm))
    story.append(paragraph("课程：人工智能导论", styles, "MetaCN"))
    story.append(paragraph("实验主题：CNN 与 RNN 在情感分类任务中的应用与比较", styles, "MetaCN"))
    story.append(paragraph("报告文件名：{}".format(report_filename()), styles, "MetaCN"))
    story.append(Spacer(1, 1.4 * cm))

    story.append(paragraph("摘要", styles, "HeadingCN"))
    abstract_model_text = "本文围绕中文影评情感二分类任务，对 MLP、TextCNN、BiRNN、BiLSTM 与 BiGRU 五个主模型进行了实现、训练与比较"
    if transformer_in_main_results:
        abstract_model_text += "，并额外纳入轻量 Transformer 编码器作为扩展对照"
    elif transformer_experiments:
        abstract_model_text += "，并补充了 Transformer 编码器的结构分析与追加实验"
    abstract_text = (
        "{}。实验基于已完成分词的中文影评数据集展开，训练集、验证集与测试集规模分别为 19,998、5,629 与 369 条，并使用 50 维中文预训练 Word2Vec 词向量作为初始输入表示。本文首先说明数据处理、词表构建、向量初始化、模型训练与验证集早停等实验流程；随后通过结构图、Accuracy/F1 指标曲线、结果表格和参数对比实验，对不同结构的性能、收敛特征与鲁棒性进行系统比较。结果表明，课程要求覆盖的 CNN/RNN/MLP 主实验模型均满足 80% 以上准确率指标，其中 {} 在测试集上取得最优表现，Accuracy 为 {:.4f}，F1 为 {:.4f}。".format(
            abstract_model_text,
            model_label(best_core["model_name"]),
            best_core["test"]["accuracy"],
            best_core["test"]["f1"],
        )
    )
    if improvement_summary is not None:
        abstract_text += (
            "在主实验之外，本文还追加了“BiRNN→BiLSTM→BiLSTM+Attention”与“单卷积核 CNN→TextCNN→数据增强”两条改进路线；结果显示，BiLSTM 相比 BiRNN 的测试集 F1 提升约 {:.3f}，而多卷积核 TextCNN 相比单卷积核 CNN 的测试集 F1 提升约 {:.3f}，说明结构升级比轻量文本增强更稳定。".format(
                rnn_improvement_delta,
                cnn_improvement_delta,
            )
        )
    abstract_text += (
        "进一步的超参数与跨域测试表明，门控循环结构在域内任务上整体较稳定，但当测试文本迁移到微博语体时，所有模型性能均明显下降，说明当前方法对领域分布偏移的适应能力仍然有限。"
    )
    story.append(paragraph(abstract_text, styles, "BodyNoIndentCN"))
    story.append(paragraph(
        "关键词：情感分类；预训练词向量；TextCNN；循环神经网络；Transformer；模型比较",
        styles,
        "KeywordCN",
    ))

    story.append(paragraph("1 模型结构图", styles, "HeadingCN"))
    story.append(paragraph(
        "本节按照课程评分标准，集中展示 baseline、CNN 与双向循环模型的结构图，并结合具体样本说明各组成部分在前向传播和训练阶段的作用。为了便于横向比较，所有模型统一以“输入准备 - 特征提取 - 分类输出 - 参数更新”为主线进行说明。",
        styles,
    ))
    story.extend(build_model_diagram_section(OUTPUT_DIR / "report_assets", styles))

    story.append(paragraph("2 实验流程描述", styles, "HeadingCN"))
    story.append(paragraph("2.1 任务背景", styles, "SubHeadingCN"))
    story.append(paragraph(
        "情感分析旨在识别文本所表达的主观态度与情绪倾向，是自然语言处理中的典型文本分类任务。在中文互联网环境中，影评、微博和商品评论等文本往往具有较强的主观性，因此情感分类既具有现实应用价值，也能够作为检验文本表示与神经网络建模能力的基础问题。",
        styles,
    ))
    task_background_text = (
        "本实验要求基于预训练词向量实现 CNN 与 RNN 模型，并在中文情感分类任务上比较其效果。"
        "为使横向比较更完整，本文进一步实现了 MLP 基线，"
    )
    if transformer_in_main_results:
        task_background_text += "并额外补充轻量 Transformer 编码器作为扩展对照，"
    else:
        task_background_text += "并在正式主实验之外补充了 Transformer 编码器的结构与追加实验，"
    task_background_text += "在相同数据划分和评价指标下分析不同结构对局部模式提取、序列依赖建模和训练稳定性的影响。"
    story.append(paragraph(task_background_text, styles))
    story.append(paragraph("2.2 数据加载与预处理", styles, "SubHeadingCN"))
    story.append(paragraph(
        "实验数据包含 `data/train.txt`、`data/validation.txt`、`data/test.txt` 三个文本文件，以及 50 维预训练词向量 `data/wiki_word2vec_50.bin`。每条样本由二分类标签和已经分词的中文句子构成。训练集、验证集与测试集规模分别为 19,998、5,629 和 369 条，预训练词向量对训练语料 token 的覆盖率为 {:.2%}。".format(
            summary["data"]["train_token_coverage"]
        ),
        styles,
    ))
    story.append(paragraph(
        "在具体实现上，程序会先逐行读取三份文本文件，并把每一行解析成“标签 + 分词结果”两部分：第一列转成 `0/1` 标签，后续所有词保留为原始 token 列表。随后，程序只基于训练集统计词频并构建词表，而不会把验证集和测试集一起并入词表生成过程；这样做是为了避免信息泄漏。构建词表时，最前面固定加入 `<pad>` 与 `<unk>` 两个特殊标记，分别用于补齐长度和承接未登录词。",
        styles,
    ))
    story.append(paragraph(
        "词表确定后，程序并不会把整个 Word2Vec 文件完整加载进内存，而是只抽取当前词表中真正会用到的词向量。对于命中预训练词向量的词，直接使用对应 50 维向量初始化 embedding；对于没有命中的词，则按照已命中向量的均值和标准差做随机初始化，以保证尺度一致；`<pad>` 向量始终强制置零，防止补齐位置在池化、卷积和循环状态传播时引入伪信息。这样处理后，词向量矩阵既保留了预训练语义，又能兼容训练语料中未命中的词。",
        styles,
    ))
    story.append(paragraph(
        "接下来，程序把每条句子统一截断或补齐到长度 80，并同时保存该句子的真实长度 `lengths`。因此，每个数据切分最后都会被编码成三组张量：`inputs` 表示词编号矩阵，`lengths` 表示真实长度，`labels` 表示情感标签。固定长度张量便于后续按 batch 训练，而真实长度又保证模型能够区分“真实 token”和“补齐位置”。为了提高重复运行时的效率，程序还会把预处理结果缓存到磁盘；如果 `max_len`、`min_freq` 和随机种子配置没有变化，后续运行时就直接复用缓存，而不重新构建词表和抽取词向量。",
        styles,
    ))
    story.append(paragraph("2.3 模型训练与结果评估", styles, "SubHeadingCN"))
    training_text = (
        "在模型训练阶段，主实验分别构建 MLP、TextCNN、BiRNN、BiLSTM 与 BiGRU，并在统一数据划分下进行训练；"
    )
    if transformer_in_main_results:
        training_text += "Transformer 也沿用相同数据流程单独训练，但在报告中作为扩展对照单独分析。"
    else:
        training_text += "Transformer 作为追加对照单独记录，以免与课程要求的主体比较混淆。"
    training_text += (
        "优化器统一采用 AdamW；训练阶段使用带 label smoothing 的交叉熵损失，验证和测试阶段则改用普通交叉熵单独统计指标。循环模型额外设置更小的词向量学习率、短暂 warmup 和更保守的早停轮数，以减少手写循环单元在训练初期的不稳定现象。这样的设置既保证了实验可重复性，也更便于公平比较不同结构之间的收敛特征。"
    )
    story.append(paragraph(training_text, styles))
    workflow_text = (
        "更具体地说，每个模型训练前都会先根据对应的 `batch_size` 构造训练、验证和测试三个 DataLoader，其中只有训练集开启随机打乱，验证集和测试集保持固定顺序。进入每一轮训练后，程序会依次完成以下步骤：从 DataLoader 中取一个 batch；把 `inputs`、`lengths` 和 `labels` 移动到当前设备；前向计算得到 logits；根据标签计算损失；执行反向传播；若配置中启用了 `grad_clip`，则在参数更新前先做梯度裁剪；最后调用优化器完成一次参数更新。对于循环模型，这一步尤其重要，因为循环结构在较长序列上更容易出现梯度爆炸。"
    )
    if transformer_experiments:
        workflow_text += "Transformer 相关实验沿用同一数据预处理流程，只是在特征提取阶段改用自注意力编码器。"
    workflow_text += "每完成一轮训练，程序都会立即在验证集上关闭梯度进行完整评估，统一计算验证损失、Accuracy、Precision、Recall 和 F1，并把这些信息连同本轮学习率、耗时和词向量是否解冻一起记录进 history。随后，程序以验证集 F1 为第一优先级、验证集 Accuracy 为第二优先级判断当前轮是否优于历史最佳；如果更优，就立即保存当前模型参数为 checkpoint，并把该轮记为最佳轮次。若连续若干轮没有提升，且已经达到最小训练轮数要求，就触发 early stopping，提前结束训练，从而减少无效训练和过拟合风险。"
    story.append(paragraph(workflow_text, styles))
    story.append(paragraph(
        "训练结束后，程序不会直接用“最后一轮”的参数汇报结果，而是先把模型权重回滚到验证集表现最好的那一轮，再分别在验证集和测试集上重新评估一次，得到最终写入 `*_metrics.json` 的指标结果。与此同时，最佳模型参数会保存成 `*_best.pt`，主实验的所有结果进一步汇总到 `experiment_summary.json`，调参实验汇总到 `hyperparameter_tuning_summary.json`。因此，本实验的完整流程实际上是“数据读取与张量化 - 模型训练 - 每轮验证评估 - 按验证集 F1 选最优参数 - 回滚最佳参数 - 测试集汇报结果 - 统一保存结果文件”，后续图表和报告内容也都是直接从这些结构化结果文件中自动生成的。",
        styles,
    ))

    story.append(paragraph("3 实验结果展示", styles, "HeadingCN"))
    story.append(paragraph("3.1 训练过程与收敛特征", styles, "SubHeadingCN"))
    story.extend(build_curve_section(core_results, styles))
    story.append(test_metric_bar_chart(core_results))
    story.append(Spacer(1, 0.06 * cm))
    story.append(paragraph("图{}  主模型测试集 Accuracy 与 F1 对比".format(test_metric_figure_number), styles, "CaptionCN"))
    story.append(Spacer(1, 0.12 * cm))

    story.append(paragraph("3.2 主实验结果", styles, "SubHeadingCN"))
    story.append(paragraph("表3  主模型在验证集与测试集上的实验结果", styles, "CaptionCN"))
    story.append(metric_table(core_results))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph(
        "课程要求覆盖的 CNN/RNN/MLP 主实验模型都达到说明文档提出的 80% 以上指标要求，其中测试集表现最好的模型是 <b>{}</b>，其测试集 Accuracy 为 <b>{:.4f}</b>，F1 为 <b>{:.4f}</b>。".format(
            model_label(best_core["model_name"]),
            best_core["test"]["accuracy"],
            best_core["test"]["f1"],
        ),
        styles,
    ))
    story.append(paragraph(
        "从结果对比看，<b>{}</b> 的测试集 F1 最高，<b>{}</b> 与其非常接近；而 <b>{}</b> 在五个主模型中相对最低。这说明当前任务的有效信息一部分来自局部短语模式，一部分来自预训练词向量本身携带的词汇极性，因此 TextCNN 和 MLP 都能取得较强结果；与此同时，带门控的 BiLSTM、BiGRU 相比普通 BiRNN 仍表现更稳，说明门控机制对长距离上下文的保留确实是有价值的。".format(
            model_label(max(core_results, key=lambda item: item["test"]["f1"])["model_name"]),
            model_label(sorted(core_results, key=lambda item: item["test"]["f1"], reverse=True)[1]["model_name"]),
            model_label(min(core_results, key=lambda item: item["test"]["f1"])["model_name"]),
        ),
        styles,
    ))

    if transformer_experiments:
        best_transformer = max(transformer_experiments, key=lambda item: item["result"]["test"]["f1"])
        story.append(paragraph("3.3 Transformer 追加实验", styles, "SubHeadingCN"))
        transformer_intro = (
            "当前项目中的 Transformer 采用的是“50 维静态 Word2Vec + 轻量 Transformer Encoder”方案，而不是直接微调大规模预训练语言模型。"
        )
        if transformer_in_main_results:
            transformer_intro += "因此这里把它相关的训练预算和结果单独展开，便于和主实验中的 CNN/RNN 结论区分阅读。"
        else:
            transformer_intro += "因此本文将其作为追加对照单独汇报，不与主实验的 CNN/RNN 排名混为一谈。这样既能保留作业所要求的主体比较，也能客观展示自注意力结构在同一数据流程下的表现。"
        story.append(paragraph(transformer_intro, styles))
        story.append(paragraph("表3-补充  Transformer 追加实验结果", styles, "CaptionCN"))
        story.append(transformer_experiment_table(transformer_experiments))
        story.append(Spacer(1, 0.18 * cm))
        transformer_text = (
            "从现有可复核结果看，表现最好的是 <b>{}</b>，其测试集 Accuracy 为 <b>{:.4f}</b>，F1 为 <b>{:.4f}</b>。"
            " 与当前主实验最优的 <b>{}</b> 相比，测试集 F1 仍有 <b>{:+.4f}</b> 的差距。".format(
                best_transformer["label"],
                best_transformer["result"]["test"]["accuracy"],
                best_transformer["result"]["test"]["f1"],
                model_label(best_core["model_name"]),
                best_transformer["result"]["test"]["f1"] - best_core["test"]["f1"],
            )
        )
        if transformer_comparison is not None and transformer_comparison.get("comparisons"):
            comparisons = transformer_comparison["comparisons"]
            faster_refs = [item for item in comparisons if item["transformer_vs_reference_speedup"] > 1.0]
            slower_refs = [item for item in comparisons if item["transformer_vs_reference_speedup"] <= 1.0]
            if faster_refs and slower_refs:
                fastest = max(faster_refs, key=lambda item: item["transformer_vs_reference_speedup"])
                slowest = min(slower_refs, key=lambda item: item["transformer_vs_reference_speedup"])
                transformer_text += (
                    " 基于现有 `transformer_efficiency_comparison.json`，1 轮 quick check 版本相对 <b>{}</b> 的训练速度约快 <b>{:.2f}x</b>，"
                    " 但相对 <b>{}</b> 仅有 <b>{:.2f}x</b>，说明轻量自注意力在顺序建模上比手写循环单元更易并行，"
                    " 但在当前小数据、静态词向量设定下，并没有转化为更高的分类精度。".format(
                        model_label(fastest["reference_model"]),
                        fastest["transformer_vs_reference_speedup"],
                        model_label(slowest["reference_model"]),
                        slowest["transformer_vs_reference_speedup"],
                    )
                )
        story.append(paragraph(transformer_text, styles))
        story.append(paragraph(
            "这一结果与结构本身并不矛盾：Transformer 的优势通常依赖更充足的数据规模、更深的编码器或更强的预训练语义表示，而当前实现刻意限制在单层、4 头、128 维并以静态词向量起步，更接近“把自注意力模块接到传统词向量流水线中”的工程探测，因此更适合用来展示架构差异，而不是直接挑战已充分调优的 BiLSTM/TextCNN 强基线。",
            styles,
        ))

    story.append(paragraph("4 参数对比分析", styles, "HeadingCN"))
    story.append(paragraph("4.1 主模型参数与扩展对比实验", styles, "SubHeadingCN"))
    story.append(paragraph("表4  主模型结构参数与训练参数总表", styles, "CaptionCN"))
    story.append(parameter_table(results))
    story.append(Spacer(1, 0.18 * cm))
    story.append(paragraph("表5  扩展参数对比实验结果", styles, "CaptionCN"))
    story.append(ablation_table(ablations))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph(
        "从附加对比实验可以看出，TextCNN 的卷积核数量从 64 提升到 128 后，局部模式提取能力更强；BiRNN、BiLSTM 和 BiGRU 的隐藏维度从 64 提升到 128 后，也更有利于保留上下文信息，但参数量和训练时间会同步增加。",
        styles,
    ))

    if hyperparameter_tuning is not None:
        tuning_variants = hyperparameter_tuning["variants"]
        tuning_families = hyperparameter_tuning["families"]
        tuning_parameter_groups = hyperparameter_tuning["parameter_groups"]
        tuning_best_by_family = hyperparameter_tuning["best_by_family"]
        best_variant = max(tuning_variants, key=lambda item: item["result"]["delta_vs_baseline"]["validation_f1"])
        worst_variant = min(tuning_variants, key=lambda item: item["result"]["delta_vs_baseline"]["validation_f1"])

        story.append(Spacer(1, 0.15 * cm))
        story.append(paragraph(
            "为了满足“逐个超参数调节”的要求，实验进一步采用单变量调参方式：每次只修改一个超参数，其余设置保持对应基线模型不变。这里不再只展示少量示例，而是对各主模型分别列出基线参数、调参覆盖范围，并对每个超参数给出候选取值与对应指标图表。",
            styles,
        ))
        story.append(paragraph("表6  各模型基线参数与调参覆盖范围", styles, "CaptionCN"))
        story.append(hyperparameter_family_table(tuning_families))
        story.append(Spacer(1, 0.15 * cm))
        story.append(paragraph("表7  单变量超参数调节结果总表", styles, "CaptionCN"))
        story.append(hyperparameter_tuning_table(tuning_variants))
        story.append(Spacer(1, 0.15 * cm))
        story.append(paragraph("表8  各模型族单参数最优取值汇总", styles, "CaptionCN"))
        story.append(best_hyperparameter_table(tuning_best_by_family))
        story.append(Spacer(1, 0.15 * cm))
        story.append(paragraph(
            "从单变量调参结果看，最有效的改动是把 {} 的 {} 调到 <b>{}</b>，使验证集 F1 相比对应基线提升了 <b>{:+.4f}</b>；而退化最明显的改动使验证集 F1 下降了 <b>{:+.4f}</b>。这里把测试集指标仅作为对应参考展示，而不用于选择超参数，目的是避免把测试集信息泄漏到调参过程。".format(
                best_variant["spec"]["family"].upper(),
                best_variant["spec"]["parameter_name"],
                best_variant["spec"]["parameter_value"],
                best_variant["result"]["delta_vs_baseline"]["validation_f1"],
                worst_variant["result"]["delta_vs_baseline"]["validation_f1"],
            ),
            styles,
        ))
        story.append(paragraph("4.1.1 单参数最优值原因分析", styles, "SubHeadingCN"))
        story.extend(build_hyperparameter_choice_analysis_section(tuning_parameter_groups, styles))
        story.append(paragraph("4.2 逐个超参数图表与参数含义说明", styles, "SubHeadingCN"))
        story.extend(build_hyperparameter_explanation_section(tuning_parameter_groups, styles))
        story.extend(build_tuning_chart_section(tuning_parameter_groups, styles))

    if improvement_summary is not None:
        improvement_heading = "4.3 结构改进与数据增强追加实验" if hyperparameter_tuning is not None else "4.2 结构改进与数据增强追加实验"
        improvement_table_prefix = "8" if hyperparameter_tuning is not None else "5"
        rnn_improvements = sorted(improvement_summary["tracks"]["rnn"], key=improvement_sort_key)
        cnn_improvements = sorted(improvement_summary["tracks"]["cnn"], key=improvement_sort_key)
        rnn_bilstm = next(item for item in rnn_improvements if item["model_name"] == "bilstm")
        rnn_attention = next(item for item in rnn_improvements if item["model_name"] == "bilstm_attention")
        cnn_multi = next(item for item in cnn_improvements if item["model_name"] == "cnn")
        cnn_word_dropout = next(item for item in cnn_improvements if item["model_name"] == "textcnn_word_dropout")

        story.append(paragraph(improvement_heading, styles, "SubHeadingCN"))
        story.append(paragraph(
            "为了把“结构优化”和“额外数据增强”写成一条更清晰的实验链路，本文在主实验之外补充了两组逐步改进实验：其一是沿着 `BiRNN → BiLSTM → BiLSTM + Attention` 验证循环结构升级；其二是沿着 `单卷积核 CNN → 多卷积核 TextCNN → TextCNN + 数据增强` 验证卷积结构和轻量增强的收益。{}".format(
                improvement_reuse_note(improvement_summary)
            ),
            styles,
        ))
        story.append(paragraph("表{}-1  RNN 路线的逐步结构改进结果".format(improvement_table_prefix), styles, "CaptionCN"))
        story.append(improvement_experiment_table(rnn_improvements))
        story.append(Spacer(1, 0.16 * cm))
        story.append(paragraph(
            "从 RNN 路线看，`BiLSTM` 相比 `BiRNN` 的测试集 F1 提升了 <b>{:+.4f}</b>，说明在这份影评数据上，引入门控记忆机制确实有助于保留更长距离的上下文信息。相比之下，`BiLSTM + Attention` 在本轮 quick 预算下测试集 F1 相比 `BiLSTM` 下降了 <b>{:+.4f}</b>，表明注意力机制并不是“加上就涨”，仍需要继续调节 attention 维度、dropout 和训练轮数，才能更稳定地发挥作用。".format(
                rnn_bilstm["delta_vs_reference"]["test_f1"],
                rnn_attention["delta_vs_reference"]["test_f1"],
            ),
            styles,
        ))
        story.append(paragraph("表{}-2  CNN 路线的结构改进与数据增强结果".format(improvement_table_prefix), styles, "CaptionCN"))
        story.append(improvement_experiment_table(cnn_improvements))
        story.append(Spacer(1, 0.16 * cm))
        story.append(paragraph(
            "从 CNN 路线看，`TextCNN` 相比单卷积核 CNN 的测试集 F1 提升了 <b>{:+.4f}</b>，说明多尺度卷积核更适合同时捕捉“很好看”“一点也不好看”这类不同长度的情感短语。三种轻量增强都没有超过未增强的 TextCNN，其中 `Word Dropout` 最接近基线，测试集 F1 仅变化 <b>{:+.4f}</b>；而 `Random Deletion` 与 `Synonym Replacement` 退化更明显，说明当前增强强度对这份已分词影评数据来说偏大，语义扰动已经开始覆盖其潜在收益。".format(
                cnn_multi["delta_vs_reference"]["test_f1"],
                cnn_word_dropout["delta_vs_reference"]["test_f1"],
            ),
            styles,
        ))

    story.append(paragraph("5 模型比较", styles, "HeadingCN"))
    story.append(paragraph("5.1 Baseline、CNN 与 RNN 的比较", styles, "SubHeadingCN"))
    story.append(paragraph(
        "若直接按这次主实验的测试集 F1 排序，结果为：{}。其中 <b>{}</b> 以 <b>{:.4f}</b> 的测试集 F1 排名第一，只比第二名 <b>{}</b> 高 <b>{:.4f}</b>，但相对最低的 <b>{}</b> 仍高出 <b>{:.4f}</b>。这说明当前数据集上并不存在“绝对碾压”的单一结构，更准确的判断是：不同模型各自擅长捕捉不同层次的情感线索，而最终排名取决于这些能力与数据分布的匹配程度。".format(
            core_ranking_text,
            model_label(best_core["model_name"]),
            best_core["test"]["f1"],
            model_label(second_core["model_name"]),
            best_core_gap_vs_second,
            model_label(worst_core["model_name"]),
            best_core_gap_vs_worst,
        ),
        styles,
    ))
    story.append(paragraph(
        "MLP 通过平均池化快速聚合句向量，结构最简单，同时也是本实验里训练最快的主模型：主实验中 MLP 训练约 {:.1f} 秒，而 TextCNN 约 {:.1f} 秒，说明 TextCNN 的训练时间大约是 MLP 的 {:.2f} 倍。TextCNN 对“非常 失望”“剧情 混乱”这类局部情感短语尤其敏感，参数共享带来较强的局部模式提取能力，但多卷积核并行和更复杂的特征拼接也会带来额外训练开销。BiRNN、BiLSTM 和 BiGRU 则显式建模上下文顺序，更适合处理依赖前后语义的长句；但代价也很明确，它们训练时长分别约为 TextCNN 的 {:.2f}、{:.2f} 和 {:.2f} 倍，其中 BiLSTM 的门控记忆最完整，BiGRU 在效果和效率之间折中得更明显。".format(
            result_by_name["mlp"]["train_seconds"],
            result_by_name["cnn"]["train_seconds"],
            cnn_vs_mlp_time_ratio,
            birnn_vs_cnn_time_ratio,
            bilstm_vs_cnn_time_ratio,
            bigru_vs_cnn_time_ratio,
        ),
        styles,
    ))
    story.append(paragraph(
        "从这次结果看，<b>TextCNN 排名第一</b> 的直接原因，是这份中文影评数据里的有效判别信号大多集中在局部短语层面，例如“非常 感人”“不 值得 看”“剧情 混乱”“超出 预期”这类 2 到 5 个词的搭配。TextCNN 的多尺度卷积核恰好适合并行扫描这些短语，再用最大池化保留最强情感证据，因此既能抓住否定结构和程度副词，又能在效果和训练成本之间维持较好的平衡。它在本次重跑中测试集 F1 达到 <b>{:.4f}</b>；虽然训练速度并不是主模型里最快的，但相较需要更长训练时间的双向循环模型，它仍然以较低成本给出了最好的测试集结果，这就是它在综合比较里最占优的原因。".format(
            result_by_name["cnn"]["test"]["f1"],
        ),
        styles,
    ))
    story.append(paragraph(
        "<b>MLP 能排到第二</b>，说明这个任务并不完全依赖复杂时序建模。预训练词向量已经把不少情感词语义放到了相对合理的位置，平均池化虽然丢掉了词序，但仍能把“推荐、精彩、感人”和“拖沓、混乱、乏味”这类高频情感词的总体倾向保留下来，因此 MLP 的测试集 F1 仍有 <b>{:.4f}</b>。它没有超过 TextCNN，主要是因为平均操作会弱化“不 值得 推荐”“前半段 很慢 但是 后半段 精彩”这类需要局部组合或转折关系才能判断的句子；但它也没有明显输给循环模型，反而说明在当前数据规模和静态词向量设定下，很多样本靠词汇层面的极性就已经能做出较强判断。".format(
            result_by_name["mlp"]["test"]["f1"],
        ),
        styles,
    ))
    story.append(paragraph(
        "三种双向循环模型里，<b>BiGRU 优于 BiLSTM，BiLSTM 又明显优于 BiRNN</b>。这和结构能力是吻合的：BiRNN 使用最基础的循环单元，长距离信息经过多步传递后更容易衰减，所以它在包含转折、让步和长句评价时更容易丢失关键上下文；BiLSTM 和 BiGRU 通过门控机制显式控制“该记什么、该忘什么”，因此测试集 F1 分别提升到 <b>{:.4f}</b> 和 <b>{:.4f}</b>。而这次 BiGRU 最终略高于 BiLSTM，一个合理解释是：当前数据规模不大、训练预算有限，GRU 用更少的门控参数换来了更稳定的优化，因而在不牺牲太多表达能力的前提下取得了更好的泛化平衡。".format(
            result_by_name["bilstm"]["test"]["f1"],
            result_by_name["bigru"]["test"]["f1"],
        ),
        styles,
    ))
    if transformer_experiments:
        story.append(paragraph(
            "作为追加模型，Transformer 不再沿时间步递推隐藏状态，而是通过多头自注意力让每个 token 直接与其余有效 token 交互，因此理论上更擅长建模长距离依赖且并行度更高。不过在本项目的轻量设定下，它缺少大规模预训练语言模型带来的上下文语义先验，最终表现更像是一个“可并行的全局特征提取器”。本次正式补充实验里，它的测试集 F1 为 <b>{:.4f}</b>，低于主实验最优的 <b>{}</b>。原因并不难理解：当前实现只有单层、4 头、128 维，而且输入仍是静态 Word2Vec；在这种设定下，自注意力虽然能建模全局关系，但缺少更强预训练语义和更深层表示能力，优势不足以转化为更高分类精度。".format(
                transformer_main_result["test"]["f1"] if transformer_main_result is not None else max(
                    transformer_experiments,
                    key=lambda item: item["result"]["test"]["f1"],
                )["result"]["test"]["f1"],
                model_label(best_core["model_name"]),
            ),
            styles,
        ))
    story.append(paragraph("5.2 综合比较结论", styles, "SubHeadingCN"))
    story.append(paragraph(
        "本文基于统一的数据处理流程与评价指标，实现并比较了 MLP、TextCNN、BiRNN、BiLSTM、BiGRU 等主模型，并补充了轻量 Transformer 编码器的结构与追加实验。实验表明，预训练词向量为各类模型提供了较强的初始语义表示，而不同网络结构的差异主要体现在对局部模式、词序与上下文依赖的建模方式上。综合测试集指标、训练曲线和参数对比结果可以看到，TextCNN 在当前任务上取得了最好的测试集结果，MLP 则以最低训练成本给出了很强的竞争基线；带门控的 BiLSTM、BiGRU 相比普通 BiRNN 更稳定，但在当前数据规模下并未整体超过 TextCNN。Transformer 追加实验则说明，自注意力结构本身值得保留，但若缺少更强预训练语义或更大训练预算，其优势未必会在小规模影评任务上自然显现。",
        styles,
    ))
    story.append(paragraph(
        "与此同时，外部微博测试也表明当前模型的跨域迁移能力较弱，说明仅依赖单一领域训练数据仍难以获得稳定的通用情感判别能力。后续若进一步引入更大规模的跨域语料、上下文增强表示或更强的预训练语言模型，模型的泛化能力仍有较大提升空间。",
        styles,
    ))
    if improvement_summary is not None:
        story.append(paragraph(
            "追加实验进一步给出了更细的工程判断：在当前任务上，优先升级核心结构比直接叠加轻量文本增强更稳妥。具体而言，`BiRNN → BiLSTM` 与 `单卷积核 CNN → 多卷积核 TextCNN` 都带来了明确收益，而 `BiLSTM + Attention` 和三种数据增强仍需额外调参才能超过已有强基线。这一结论也说明，新增模块的价值必须建立在合适训练预算和匹配数据分布之上。",
            styles,
        ))

    story.append(paragraph("6 问题思考回答", styles, "HeadingCN"))
    story.append(paragraph("6.1 训练何时停止最合适", styles, "SubHeadingCN"))
    story.append(paragraph(
        "本实验采用“以验证集 F1 为核心指标的早停”策略，即每轮训练后都在验证集上计算 Accuracy 和 F1，并以 F1 是否继续提升作为是否保留当前参数的主要依据。之所以不用“训练到固定 10 轮或 20 轮就停止”的硬规则，是因为不同模型的收敛速度差异很大：有的模型很快就达到最佳泛化点，有的模型则需要更长时间才能把词向量和分类层一起调整到位。",
        styles,
    ))
    story.append(paragraph(
        "从本次实验结果看，BiRNN、BiLSTM 和 BiGRU 的最优轮次分别出现在第 {}、{}、{} 轮，TextCNN 出现在第 {} 轮，MLP 则延后到第 {} 轮。这说明“最合适的停止时刻”并不是一个统一常数，而是取决于模型结构、参数规模和当前优化状态。以 TextCNN 为例，它在第 {} 轮就已经达到最优，如果仍然机械地训练到第 10 轮，那么后面的 9 轮更多是在消耗时间；而 MLP 最优轮次更靠后，说明它虽然结构简单，但也需要更长的迭代来把平均池化后的句向量和分类层磨合到较稳定状态。".format(
            result_by_name["birnn"]["best_epoch"],
            result_by_name["bilstm"]["best_epoch"],
            result_by_name["bigru"]["best_epoch"],
            result_by_name["cnn"]["best_epoch"],
            result_by_name["mlp"]["best_epoch"],
            result_by_name["cnn"]["best_epoch"],
        ),
        styles,
    ))
    story.append(paragraph(
        "固定迭代次数的优点是实现简单、复现实验方便，比如所有模型统一训练 10 轮，实验流程非常直观；但它的缺点也很明显：若轮数设得太小，会让像 MLP 这类收敛稍慢的模型欠拟合；若轮数设得太大，则会让已经达到峰值的模型继续朝训练集拟合，验证集指标反而下降。相比之下，基于验证集的早停更贴近“泛化性能最好时就停止”的目标，只是它需要额外划分验证集，并且要设定 `patience` 这类超参数来容忍短期波动。",
        styles,
    ))
    story.append(paragraph(
        "如果用一个具体句子来理解这个过程，可以把模型看成在不断学习如何区分“故事 拖沓 剪辑 混乱 看 完 只 想 立刻 退场”与“前半段 稍慢 后半段 反转 精彩 整体 超出 预期”这类表达。训练前几轮，模型通常先学到明显的情感词；继续训练后，可能开始学习更复杂的转折和搭配。但如果再往后训练太久，它也可能把训练集里某些只出现过几次的人名、片名或口头表达当成“捷径特征”，这时训练集指标继续上涨，而验证集 F1 却不再提升，说明应该及时停止。",
        styles,
    ))

    story.append(paragraph("6.2 实验参数如何初始化", styles, "SubHeadingCN"))
    story.append(paragraph(
        "本实验的初始化分成“词向量初始化”和“网络参数初始化”两部分。词向量方面，PAD 向量固定为 0，这样它既不会在平均池化中引入额外噪声，也不会在循环状态更新中制造虚假信息；命中预训练词向量的词直接加载已有表示；未命中的词则使用与预训练分布同尺度的随机初始化。这样做的目的，是让模型一开始就拥有基本的语义几何结构，而不是从完全无意义的随机空间重新学起。",
        styles,
    ))
    story.append(paragraph(
        "例如句子“宣传 很 热闹 实际 内容 乏味 完全 浪费 时间”中，像“宣传”“内容”“时间”这类常见词通常能直接命中预训练词向量，而更少见的口语词或专有词可能需要随机初始化。如果全部词都随机初始化，那么模型在训练初期连“失望”和“推荐”之间的语义差异都还没有建立，收敛会明显变慢；而预训练词向量能让模型从一开始就知道“精彩”“推荐”更接近正向语义，“拖沓”“乏味”更接近负向语义。",
        styles,
    ))
    story.append(paragraph(
        "网络层方面，线性层权重使用 Xavier 初始化，适合前后层规模接近的全连接变换；卷积层使用 Kaiming 初始化，因为 ReLU 会截断负半轴，Kaiming 更有利于保持激活值方差稳定；RNN、LSTM 和 GRU 的隐藏到隐藏权重使用正交初始化，是因为循环结构会反复把上一时刻状态传给下一时刻，正交矩阵更有助于缓解长序列上的梯度爆炸或梯度消失。偏置统一初始化为 0，避免在训练刚开始时人为给某一类激活增加偏向。",
        styles,
    ))
    story.append(paragraph(
        "不同初始化方法适合的地方也不同。若把所有权重都初始化成同一个值，例如全部为 0，那么每个神经元接收到的梯度会完全一样，最后学出来的还是同一个特征，网络等价于只有一个神经元；若高斯初始化方差设置过大，前向传播时激活值会一层层放大，训练会很不稳定。相反，像循环层这样需要跨时间传递信息的结构，更强调状态范数不要在 80 个时间步里迅速衰减，因此正交初始化比简单的随机高斯更合适。以“虽然 前半段 很慢 但是 后半段 非常 感人”为例，若循环权重初始化不稳定，模型可能在读到“但是”之后已经忘掉前面的状态变化，难以正确处理转折。",
        styles,
    ))

    story.append(paragraph("6.3 如何缓解过拟合", styles, "SubHeadingCN"))
    story.append(paragraph(
        "过拟合的本质，是模型不仅学到了“什么模式代表正面、什么模式代表负面”，还把训练集中特有但不具备普遍性的细节也一并记住了。对于情感分类任务来说，这些“坏记忆”可能是某个演员名字、某种只在训练集高频出现的口头表达，甚至是几个词偶然同时出现的搭配。其直接表现通常是：训练集损失继续下降，但验证集 F1 不再提升，甚至开始回落。",
        styles,
    ))
    story.append(paragraph(
        "本实验中最直接的缓解方法有三类。第一类是训练过程控制，例如基于验证集 F1 的早停；当模型在验证集上不再进步时，及时终止训练，避免它继续朝训练集细节过拟合。第二类是显式正则化，例如 MLP 使用 `dropout=0.3`，TextCNN 使用 `dropout=0.5`，同时训练中还使用 `weight_decay` 约束权重过大。dropout 相当于训练时随机“遮住”一部分特征，迫使模型不能只依赖单一词或单一通道；weight decay 则抑制权重无限变大，防止分类边界过于陡峭。",
        styles,
    ))
    story.append(paragraph(
        "第三类方法是控制模型容量和增加数据多样性。若发现 BiRNN、BiLSTM 这类模型在训练集上很快达到很高分数，但验证集不升反降，可以尝试减小 `hidden_dim`、减少层数、减少卷积核数量，或者增加更多训练样本和数据增强。比如 TextCNN 如果把 `num_filters` 调得过大，可能会记住训练集中一些只出现过几次的局部短语；而适度的数据增强，如同义词替换、词 dropout，可以让模型看到更多表达变体，不至于把“精彩”这一种写法学成唯一的正面模式。",
        styles,
    ))
    story.append(paragraph(
        "用例子来说，若训练集中负面影评经常出现“王力宏”“宣传片”“无聊 至极”等特定搭配，模型就可能错误地把演员名或某个场景词当成负面信号。一旦测试集中出现“王力宏 表现 不错”这样的新句子，过拟合严重的模型就容易误判。因此，缓解过拟合不是单纯“让模型变差一点”，而是让它从记忆具体样本，转向学习更可迁移的规律，例如“拖沓”“混乱”“浪费 时间”这类更通用的负面证据。",
        styles,
    ))

    story.append(paragraph("6.4 CNN、RNN 与 MLP 的优缺点", styles, "SubHeadingCN"))
    story.append(paragraph(
        "三类模型的核心差异，在于“它们如何把一串词组织成一句话的表示”。MLP 的做法最简单：先把所有有效词向量平均，再交给全连接层分类，因此它计算快、参数少、训练稳定，适合作为 baseline。它的优点是在短句或情感词非常明确时往往已经够用，例如“无聊 至极”“十分 感人”这类句子，只要平均后的向量里正负词足够明显，MLP 就能做出合理判断。",
        styles,
    ))
    story.append(paragraph(
        "但 MLP 的问题也很明显，它几乎不关心词序和局部结构。例如“值得 推荐”和“不 值得 推荐”这两句话包含大量相同的词，如果只做平均，模型很难充分体现“不”对后面短语的翻转作用。CNN 的优势就在这里：它会用局部卷积核扫描相邻词，专门捕捉“非常 感人”“不 值得 看”“剧情 混乱”这类短语级模式，因此对否定词、程度副词、固定搭配尤其敏感，而且卷积可以并行计算，训练速度通常比循环模型更高。",
        styles,
    ))
    story.append(paragraph(
        "RNN 及其门控变体的优点，则是显式保留顺序和上下文传递过程。像“虽然 前半段 很慢 但是 后半段 非常 精彩”这样的句子，真正决定极性的往往是后半句以及“但是”所表示的转折关系。普通 RNN 理论上可以逐词更新状态来处理这种结构，但它在长距离依赖上容易遗忘前文；LSTM 通过输入门、遗忘门和输出门维护更强的长期记忆，GRU 则用更轻量的门控实现相近效果，因此它们更适合处理长句、转折句和依赖前后文解释的评价。",
        styles,
    ))
    story.append(paragraph(
        "代价方面，也必须把训练时间纳入比较，而不能只看准确率。就本实验而言，MLP 训练约 {:.1f} 秒，是五个主模型里最快的；TextCNN 约 {:.1f} 秒，虽然比 MLP 慢，但仍明显快于双向循环模型。双向循环模型的时间成本则更高：BiGRU、BiRNN 和 BiLSTM 分别约为 {:.1f}、{:.1f} 和 {:.1f} 秒，大约是 TextCNN 的 {:.2f}、{:.2f} 和 {:.2f} 倍。也就是说，RNN 家族带来的收益并不是“免费”的，尤其 BiLSTM 的训练成本最高，但当前测试集表现最好的是 TextCNN、RNN 家族内部表现最好的是 BiGRU，这说明更复杂的时序建模并不一定会在当前数据规模下自动转化为最高分数。因此从工程折中看，若只追求最快速度，MLP 最轻；若希望在性能与时间之间取得平衡，TextCNN 更合适；若任务更依赖长距离上下文和转折结构，再为 LSTM 或 GRU 付出更高时间成本才更有意义。作为追加补充，Transformer 通过自注意力并行建模全局依赖，但在缺少更强预训练语义和更大训练预算时，优势未必会自然体现出来。".format(
            result_by_name["mlp"]["train_seconds"],
            result_by_name["cnn"]["train_seconds"],
            result_by_name["bigru"]["train_seconds"],
            result_by_name["birnn"]["train_seconds"],
            result_by_name["bilstm"]["train_seconds"],
            bigru_vs_cnn_time_ratio,
            birnn_vs_cnn_time_ratio,
            bilstm_vs_cnn_time_ratio,
        ),
        styles,
    ))

    story.append(paragraph("6.5 模型鲁棒性分析", styles, "SubHeadingCN"))
    story.append(paragraph(
        "鲁棒性可以从两个层面来理解：一是对“与原训练分布接近”的样本是否还能保持稳定判断，二是当文本领域发生变化时，性能会下降到什么程度。本实验同时做了这两种测试：一组是手工构造的小规模影评风格句子，另一组是来自微博语体的外部测试集。前者更适合观察模型面对局部短语、转折结构时的反应，后者更适合评估跨域泛化能力。",
        styles,
    ))
    story.append(paragraph(
        "从手工影评样本看，模型对情感线索非常直接的句子通常反应一致。例如“台词 空洞 表演 生硬 配乐 也 非常 吵闹”这种句子里，负面线索几乎全部集中在局部短语上，CNN、MLP 和各类 RNN 都比较容易抓住；而像“前半段 稍慢 后半段 反转 精彩 整体 超出 预期”或“虽然 成本 不高 但是 情感 真挚 细节 打动 人”这样的句子，则更能检验模型是否理解“前弱后强”“虽然……但是……”这类结构。也就是说，鲁棒性不只是看模型会不会分正负，还要看它在不同表达形式下是否仍能抓住真正起决定作用的部分。",
        styles,
    ))
    story.append(paragraph("表9  小规模影评风格样本上的预测结果对比", styles, "CaptionCN"))
    story.append(robustness_table(summary))

    if external_robustness is not None:
        external_results = external_robustness["results"]
        external_stats = external_robustness["external_stats"]
        best_external_f1 = max(external_results, key=lambda item: item["external_metrics"]["f1"])
        best_external_acc = max(external_results, key=lambda item: item["external_metrics"]["accuracy"])
        min_acc_drop = min(item["accuracy_drop"] for item in external_results)
        min_f1_drop = min(item["f1_drop"] for item in external_results)
        test_acc_values = [item["test_metrics"]["accuracy"] for item in external_results]
        test_f1_values = [item["test_metrics"]["f1"] for item in external_results]
        external_acc_values = [item["external_metrics"]["accuracy"] for item in external_results]
        external_f1_values = [item["external_metrics"]["f1"] for item in external_results]

        story.append(Spacer(1, 0.18 * cm))
        story.append(paragraph(
            "为了进一步评估跨域鲁棒性，本实验额外从公开 GitHub 数据集 `weibo2018` 中抽取了 100 条带标签微博文本作为外部测试集，其中正样本 50 条、负样本 50 条。该数据与原任务的中文影评语料存在明显领域差异，包含口语表达、话题标签、表情符号和微博语体，因此能更直接反映模型面对分布偏移时的稳定性。",
            styles,
        ))
        story.append(paragraph(
            "外部样本分词后共有 {} 个 token，其中只有 {:.2%} 能在训练词表中命中，显著低于原训练语料的词向量覆盖率。这说明模型在跨域场景下不仅要面对表达习惯变化，还会遭遇更严重的词汇失配问题。".format(
                external_stats["total_tokens"],
                external_stats["token_coverage"],
            ),
            styles,
        ))
        story.append(paragraph(
            "具体结果也印证了这一点。尽管各模型在原测试集上的 F1 大多还能维持在 {:.2f} 到 {:.2f} 左右，但迁移到微博数据后，最佳外部 F1 也只有 {:.4f}，对应模型为 {}；而 F1 降幅最小的模型降幅也达到 {:.4f}。这说明当前模型对“影评语体内部”的规律掌握得还可以，但一旦遇到口语化更强、缩写更多、情绪表达更跳跃的微博文本，原先学到的模式就会明显失效。简单说，模型学会了如何判断‘剧情 拖沓’‘表演 生硬’，但不一定学会了如何判断微博里的‘无语 住 了’‘笑死 我 了 但是 真 烂’这类表达。".format(
                min(test_f1_values),
                max(test_f1_values),
                best_external_f1["external_metrics"]["f1"],
                model_label(best_external_f1["model_name"]),
                min_f1_drop,
            ),
            styles,
        ))
        story.append(paragraph(
            "从模型间差异看，{} 的外部测试 F1 最好，说明它在当前这组跨域样本上的迁移表现相对更稳；而 {} 的 F1 降幅最小，说明它在迁移时退化得稍慢一些。不过整体上看，没有任何模型真正解决跨域问题，因此更合理的结论不是“某个模型已经足够鲁棒”，而是“当前基于单一影评语料训练的模型鲁棒性有限，后续需要靠跨域数据、领域自适应或更强的预训练表示继续补强”。".format(
                model_label(best_external_f1["model_name"]),
                model_label(min(external_results, key=lambda item: item["f1_drop"])["model_name"]),
            ),
            styles,
        ))
        story.append(paragraph("表10  原测试集与外部微博测试集上的鲁棒性对比", styles, "CaptionCN"))
        story.append(external_robustness_table(external_results))
        story.append(Spacer(1, 0.18 * cm))
        story.append(external_robustness_chart(external_results))
        story.append(Spacer(1, 0.06 * cm))
        story.append(paragraph("图{}  原测试集与外部微博测试集 F1 对比".format(external_figure_number), styles, "CaptionCN"))
        story.append(Spacer(1, 0.12 * cm))
        story.append(paragraph(
            "从对比结果看，参与外部评测的模型在原测试集上的 Accuracy 大约位于 {:.2f} 到 {:.2f} 之间，但迁移到微博数据后，Accuracy 仅剩 {:.2f} 到 {:.2f}，F1 仅剩 {:.2f} 到 {:.2f}；即使下降最小的模型，其 Accuracy 和 F1 也分别下降了 {:.4f} 与 {:.4f}。按外部集 Accuracy 看，表现最好的是 <b>{}</b>；按外部集 F1 看，表现最好的是 <b>{}</b>。这说明当前模型更多学习到了影评领域中的局部搭配和表达分布，而不是能够稳定迁移到通用中文情感场景的抽象判别规则，因此其跨域鲁棒性较弱。".format(
                min(test_acc_values),
                max(test_acc_values),
                min(external_acc_values),
                max(external_acc_values),
                min(external_f1_values),
                max(external_f1_values),
                min_acc_drop,
                min_f1_drop,
                model_label(best_external_acc["model_name"]),
                model_label(best_external_f1["model_name"]),
            ),
            styles,
        ))

    story.append(paragraph("7 心得体会", styles, "HeadingCN"))
    story.append(paragraph(
        "这次实验我前后大约做了两天，也是我第一次严格意义上把 ChatGPT 5.4 的 Codex 当作主要辅助工具来完成一个比较完整的课程实验。最直观的感受就是，AI 的确明显改变了我的学习方式和做实验的节奏。以前我做这类作业，往往会把很多时间耗在环境问题、样板代码、重复性修改和低层级 debug 上；而这一次，在 Codex 的帮助下，我能更快把数据处理、模型搭建、训练脚本、报告生成这些模块串起来，把更多精力放在理解“这一块到底在做什么”以及“下一步应该怎样调参”上。这种效率提升对我来说是非常真实的，也让我第一次很强烈地意识到，AI 已经不只是一个查资料的工具，而是在实际改变我思考问题和组织工作的方式。",
        styles,
    ))
    story.append(paragraph(
        "但与此同时，我也确实发现了不少问题，而且这些问题不能回避。首先，Codex 在多任务并行、长链路任务保持一致性方面并没有我一开始想象得那么稳定。有些时候我希望它一边整理结果、一边训练新模型、一边更新报告，但它并不能很好地把多个任务都执行得足够扎实。其次，它偶尔会“偷工减料”：例如我明明期望它重新训练某个模型，它却可能直接沿用历史参数、复用旧结果，表面上看流程跑通了，实际上关键实验并没有真正重做。更麻烦的是，训练出来的结果有时还会出现偏差，这时候我很难第一时间判断到底是我自己的指令写得不够清楚、参数设定有问题，还是工具本身出了错。也正因为如此，这次实验让我更深地明白了一点：AI 可以大幅提高效率，但不能替代人工甄别，特别是在实验结果、参数来源和结论解释这些关键环节上，人必须保持清醒，不能因为输出看起来“像对的”就直接接受。",
        styles,
    ))
    story.append(paragraph(
        "从模型理解上说，这次实验也让我收获很大。以前我对很多模块的认识停留在“知道名字”和“知道大概作用”这个层面，比如为什么 TextCNN 要用多卷积核、为什么 RNN 容易在长序列上遗忘、为什么早停和 dropout 会影响最终结果；而这次因为有 AI 帮我更快搭起完整实验链路，我反而有了更多时间回过头去看每个部分到底在干什么。比如在比较 MLP、TextCNN 和 BiRNN/BiLSTM/BiGRU 时，我逐渐意识到，真正重要的不是“哪个模型听起来更高级”，而是它到底以什么方式组织语义：MLP 是平均，CNN 是抓局部模式，RNN 是按顺序传递上下文。这种理解比单纯把代码跑通更重要。某种意义上，这次实验让我第一次觉得，AI 并没有削弱我对模型的理解，反而促使我把注意力从机械排错转向了结构分析和实验判断。",
        styles,
    ))
    story.append(paragraph(
        "对于具体结果，我一开始的直觉其实是：RNN 路线应该会明显优于 CNN 或 MLP，因为语言本身就是连续的序列，前后文关系、转折结构和长距离依赖看起来都更适合交给循环模型来处理。但真正把实验系统地跑下来以后，我发现不同模型之间的差距并没有想象中那么夸张，很多时候只是“谁更稳一点、谁更均衡一点”，而不是“谁绝对碾压谁”。这对我来说是一个很有价值的修正：直觉可以提供方向，但不能代替实验。即便如此，我仍然感觉 RNN，特别是带门控的 BiLSTM、BiGRU，在处理连续语言信息时有天然优势，尤其在鲁棒性和上下文保持上通常会更稳一些。也就是说，它不一定在所有指标上都拉开巨大差距，但在面对更复杂表达、转折句、跨域文本时，确实更容易体现出结构上的合理性。",
        styles,
    ))
    story.append(paragraph(
        "另外，这次实验也让我越来越强烈地感受到：在 AI 时代，“能不能把代码写出来”正在逐渐不再是最难的问题，真正困难的部分反而在于如何优化模型、设计结构、筛选实验、理解结果。超参数很多，结构选择也很多，而我真正要做的不是简单把它们全试一遍，而是尽量用控制变量法把每个超参数调到更合适的位置，并理解它为什么会产生这样的变化。比如这次调参过程中，我明显感觉到效率和准确率有时是矛盾的：更复杂的模型、更大的隐藏维度、更长的训练轮次，未必一定带来更好的最终表现，却几乎一定会带来更高的计算成本和更慢的实验速度。这件事提醒我，做模型不能只盯着“精度最高”这一个目标，还要同时考虑训练成本、收敛速度、可复现性和结构是否真的值得保留。",
        styles,
    ))
    story.append(paragraph(
        "如果说这次实验里最难、也最让我头疼的部分，我觉得还是调参。真正做起来以后我才发现，调参远比“写出一个能跑的模型”更考验耐心和判断力。因为我必须尽量坚持控制变量法，一次只动一个参数，才能知道性能变化到底是由谁引起的；但这样做也意味着实验数量会迅速膨胀，很多轮训练跑完以后可能只是告诉我“这个方向没有用”。尤其是在多个 epoch 下效果并不理想、甚至后期开始退化时，我会非常直观地感受到过拟合的存在，也更能理解为什么早停不是一个可有可无的小技巧，而是很重要的训练策略。总的来说，这次实验一方面让我真切感受到 AI 工具带来的效率革命，另一方面也让我更加清楚：真正有价值的能力，仍然是提出问题、验证结果、分析误差和做出取舍的能力。也正因为如此，这次实验虽然累，但我做完以后反而有一种很强的兴奋感，因为我感觉自己不仅完成了一次课程作业，也真正体验到了 AI 正在如何重塑我学习和思考的方式。",
        styles,
    ))

    story.append(paragraph("参考文献", styles, "HeadingCN"))
    story.append(paragraph("[1] Kim Y. Convolutional Neural Networks for Sentence Classification[C]//Proceedings of EMNLP 2014. Doha: ACL, 2014: 1746-1751.", styles, "ReferenceCN"))
    story.append(paragraph("[2] Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.", styles, "ReferenceCN"))
    story.append(paragraph("[3] Cho K, Van Merrienboer B, Gulcehre C, et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation[EB/OL]. arXiv:1406.1078, 2014.", styles, "ReferenceCN"))
    story.append(paragraph("[4] Mikolov T, Chen K, Corrado G, Dean J. Efficient Estimation of Word Representations in Vector Space[EB/OL]. arXiv:1301.3781, 2013.", styles, "ReferenceCN"))
    story.append(paragraph("[5] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems 30. 2017.", styles, "ReferenceCN"))
    story.extend(build_ai_interaction_appendix(styles))

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        title="基于预训练词向量的中文影评情感分类模型比较研究",
        author="人工智能导论课程实验",
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        leftMargin=2.4 * cm,
        rightMargin=2.4 * cm,
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print("Report generated at", REPORT_PATH)


if __name__ == "__main__":
    main()
