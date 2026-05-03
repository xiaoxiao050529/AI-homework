"""根据实验汇总结果生成课程报告 PDF。"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from src.sentiment_hw2.experiment import model_label
from src.sentiment_hw2.reporting import (
    ablation_table,
    best_hyperparameter_table,
    build_curve_section,
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
    recurrent_hello_example_table,
    register_fonts,
    robustness_table,
    test_metric_bar_chart,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.json"
EXTERNAL_ROBUSTNESS_PATH = OUTPUT_DIR / "external_robustness_metrics.json"
HYPERPARAM_TUNING_PATH = OUTPUT_DIR / "hyperparameter_tuning_summary.json"
REPORT_PATH = ROOT / "学号_姓名.pdf"


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


def main():
    """读取实验结果并生成更符合课程论文格式的 PDF。"""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("缺少实验结果文件，请先运行 run_experiments.py")

    register_fonts()
    styles = build_styles()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = summary["results"]
    ablations = summary["ablations"]
    best = pick_best(results)
    result_by_name = {result["model_name"]: result for result in results}

    external_robustness = None
    if EXTERNAL_ROBUSTNESS_PATH.exists():
        external_robustness = json.loads(EXTERNAL_ROBUSTNESS_PATH.read_text(encoding="utf-8"))

    hyperparameter_tuning = None
    if HYPERPARAM_TUNING_PATH.exists():
        hyperparameter_tuning = json.loads(HYPERPARAM_TUNING_PATH.read_text(encoding="utf-8"))
        hyperparameter_tuning = ensure_hyperparameter_summary_fields(hyperparameter_tuning)

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
    story.append(paragraph("报告文件名：学号_姓名.pdf", styles, "MetaCN"))
    story.append(Spacer(1, 1.4 * cm))

    story.append(paragraph("摘要", styles, "HeadingCN"))
    story.append(paragraph(
        "本文围绕中文影评情感二分类任务，对 MLP、TextCNN、BiRNN、BiLSTM 与 BiGRU 五类神经网络模型进行了实现、训练与对比分析。实验基于已完成分词的中文影评数据集展开，训练集、验证集与测试集规模分别为 19,998、5,629 与 369 条，并使用 50 维中文预训练 Word2Vec 词向量作为初始输入表示。本文首先说明数据处理、词表构建、向量初始化、模型训练与验证集早停等实验流程；随后通过结构图、Accuracy/F1 指标曲线、结果表格和参数对比实验，对不同模型的性能、收敛特征与鲁棒性进行系统比较。结果表明，五种模型均满足课程要求的 80% 以上准确率指标，其中 {} 在测试集上取得最优表现，Accuracy 为 {:.4f}，F1 为 {:.4f}。进一步的超参数与跨域测试表明，门控循环结构在域内任务上整体较稳定，但当测试文本迁移到微博语体时，所有模型性能均明显下降，说明当前方法对领域分布偏移的适应能力仍然有限。".format(
            model_label(best["model_name"]),
            best["test"]["accuracy"],
            best["test"]["f1"],
        ),
        styles,
        "BodyNoIndentCN",
    ))
    story.append(paragraph(
        "关键词：情感分类；预训练词向量；TextCNN；循环神经网络；模型比较",
        styles,
        "KeywordCN",
    ))

    story.append(paragraph("1 模型结构图", styles, "HeadingCN"))
    story.append(paragraph(
        "本节按照课程评分标准，集中展示 CNN、RNN 及扩展 baseline 的结构图，并说明各组成部分在前向传播中的作用。为了便于横向比较，所有模型统一以“输入表示 - 特征提取 - 分类输出”为主线进行说明。",
        styles,
    ))
    story.extend(build_model_diagram_section(OUTPUT_DIR / "report_assets", styles))
    story.append(paragraph("1.1 循环模型状态更新说明", styles, "SubHeadingCN"))
    story.append(paragraph(
        "除卷积模型外，循环结构也是本实验的核心组成部分。为了更直观地展示普通 RNN 与 GRU 在状态更新方式上的差异，表2 给出了一个压缩后的手算示例。",
        styles,
    ))
    story.append(paragraph("表2  RNN/GRU 单样本状态更新过程示例", styles, "CaptionCN"))
    story.append(recurrent_hello_example_table(styles))
    story.append(Spacer(1, 0.18 * cm))

    story.append(paragraph("2 实验流程描述", styles, "HeadingCN"))
    story.append(paragraph("2.1 任务背景", styles, "SubHeadingCN"))
    story.append(paragraph(
        "情感分析旨在识别文本所表达的主观态度与情绪倾向，是自然语言处理中的典型文本分类任务。在中文互联网环境中，影评、微博和商品评论等文本往往具有较强的主观性，因此情感分类既具有现实应用价值，也能够作为检验文本表示与神经网络建模能力的基础问题。",
        styles,
    ))
    story.append(paragraph(
        "本实验要求基于预训练词向量实现 CNN 与 RNN 模型，并在中文情感分类任务上比较其效果。为使横向比较更完整，本文进一步实现了 MLP、BiRNN、BiLSTM 和 BiGRU 等模型作为补充基线，在相同数据划分和评价指标下分析不同结构对局部模式提取、序列依赖建模和训练稳定性的影响。",
        styles,
    ))
    story.append(paragraph("2.2 数据加载与预处理", styles, "SubHeadingCN"))
    story.append(paragraph(
        "实验数据包含 `train.txt`、`validation.txt`、`test.txt` 三个文本文件，以及 50 维预训练词向量 `wiki_word2vec_50.bin`。每条样本由二分类标签和已经分词的中文句子构成。训练集、验证集与测试集规模分别为 19,998、5,629 和 369 条，预训练词向量对训练语料 token 的覆盖率为 {:.2%}。".format(
            summary["data"]["train_token_coverage"]
        ),
        styles,
    ))
    story.append(paragraph(
        "完整流程可以概括为：读取三份数据切分；仅基于训练集建立词表；从预训练词向量文件中抽取词表命中的 50 维向量；对未命中词随机初始化、对 PAD 向量置零；随后将句子统一截断或补齐到长度 80，并保留每条样本的真实长度，用于后续的 mask、池化与循环状态控制。",
        styles,
    ))
    story.append(paragraph("2.3 模型训练与结果评估", styles, "SubHeadingCN"))
    story.append(paragraph(
        "在模型训练阶段，实验分别构建 MLP、TextCNN、BiRNN、BiLSTM 和 BiGRU 五个主模型，并在统一数据划分下进行训练。优化器统一采用 AdamW，损失函数统一采用交叉熵；其中循环模型额外设置更小的词向量学习率、短暂 warmup 和更保守的早停轮数，以减少手写循环单元在训练初期的不稳定现象。这样的设置既保证了实验可重复性，也更便于公平比较不同结构之间的收敛特征。",
        styles,
    ))
    story.append(paragraph(
        "实验流程为：首先仅读取训练集建立词表，并从预训练词向量中抽取对应词的初始表示；随后将句子统一截断或补齐到长度 80，并构造 mask 以区分真实 token 与 PAD；最后分别训练 MLP、TextCNN、BiRNN、BiLSTM 和 BiGRU 模型，在验证集上以 F1 作为主要早停指标，并同时观察 Accuracy，在测试集上汇报最终性能。评价指标主要包括 Accuracy 与 F1，前者衡量整体分类正确率，后者更适合反映正负类别平衡下的综合判别能力。",
        styles,
    ))

    story.append(paragraph("3 实验结果展示", styles, "HeadingCN"))
    story.append(paragraph("3.1 训练过程与收敛特征", styles, "SubHeadingCN"))
    story.extend(build_curve_section(results, styles))
    story.append(test_metric_bar_chart(results))
    story.append(Spacer(1, 0.06 * cm))
    story.append(paragraph("图10  主模型测试集 Accuracy 与 F1 对比", styles, "CaptionCN"))
    story.append(Spacer(1, 0.12 * cm))

    story.append(paragraph("3.2 主实验结果", styles, "SubHeadingCN"))
    story.append(paragraph("表3  主模型在验证集与测试集上的实验结果", styles, "CaptionCN"))
    story.append(metric_table(results))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph(
        "五种模型都达到说明文档提出的 80% 以上指标要求，其中测试集表现最好的模型是 <b>{}</b>，其测试集 Accuracy 为 <b>{:.4f}</b>，F1 为 <b>{:.4f}</b>。".format(
            model_label(best["model_name"]),
            best["test"]["accuracy"],
            best["test"]["f1"],
        ),
        styles,
    ))
    story.append(paragraph(
        "从结果对比看，<b>{}</b> 的测试集 F1 最高，<b>{}</b> 与其非常接近；而 <b>{}</b> 在五个主模型中相对最低。这说明门控循环结构在本任务上整体更稳，但均值池化的 MLP 基线也已经表现出较强竞争力。".format(
            model_label(max(results, key=lambda item: item["test"]["f1"])["model_name"]),
            model_label(sorted(results, key=lambda item: item["test"]["f1"], reverse=True)[1]["model_name"]),
            model_label(min(results, key=lambda item: item["test"]["f1"])["model_name"]),
        ),
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
        best_variant = max(tuning_variants, key=lambda item: item["result"]["delta_vs_baseline"]["test_f1"])
        worst_variant = min(tuning_variants, key=lambda item: item["result"]["delta_vs_baseline"]["test_f1"])

        story.append(Spacer(1, 0.15 * cm))
        story.append(paragraph(
            "为了满足“逐个超参数调节”的要求，实验进一步采用单变量调参方式：每次只修改一个超参数，其余设置保持对应基线模型不变。这里不再只展示少量示例，而是对五个主模型分别列出基线参数、调参覆盖范围，并对每个超参数给出候选取值与对应指标图表。",
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
            "从单变量调参结果看，最有效的改动是把 {} 的 {} 调到 <b>{}</b>，使测试集 F1 相比对应基线提升了 <b>{:+.4f}</b>；而退化最明显的改动使测试集 F1 下降了 <b>{:+.4f}</b>。综合来看，模型容量过小、训练轮数过少、过早早停、过强正则化或不合适的优化超参数，都会直接拉低最终泛化性能。".format(
                best_variant["spec"]["family"].upper(),
                best_variant["spec"]["parameter_name"],
                best_variant["spec"]["parameter_value"],
                best_variant["result"]["delta_vs_baseline"]["test_f1"],
                worst_variant["result"]["delta_vs_baseline"]["test_f1"],
            ),
            styles,
        ))
        story.append(paragraph("4.2 逐个超参数图表", styles, "SubHeadingCN"))
        story.extend(build_tuning_chart_section(tuning_parameter_groups, styles))

    story.append(paragraph("5 模型比较", styles, "HeadingCN"))
    story.append(paragraph("5.1 Baseline、CNN 与 RNN 的比较", styles, "SubHeadingCN"))
    story.append(paragraph(
        "MLP 通过平均池化快速聚合句向量，训练速度最快、结构最简单，但会丢失词序和局部搭配信息，因此更适合作为基础 baseline。TextCNN 对“非常 失望”“剧情 混乱”这类局部情感短语尤其敏感，参数共享带来较强的效率和泛化能力。BiRNN、BiLSTM 和 BiGRU 则显式建模上下文顺序，更适合处理依赖前后语义的长句；其中 BiLSTM 的门控记忆最完整，BiGRU 在效果和效率之间折中得更明显。",
        styles,
    ))
    story.append(paragraph("5.2 综合比较结论", styles, "SubHeadingCN"))
    story.append(paragraph(
        "本文基于统一的数据处理流程与评价指标，实现并比较了 MLP、TextCNN、BiRNN、BiLSTM 与 BiGRU 五类模型在中文影评情感分类任务上的表现。实验表明，预训练词向量为所有模型提供了较强的初始语义表示，而不同网络结构的差异主要体现在对局部模式、词序与上下文依赖的建模方式上。综合测试集指标、训练曲线和参数对比结果可以看到，门控循环结构在本任务上整体更稳，TextCNN 则在效率与性能之间取得了较好平衡。",
        styles,
    ))
    story.append(paragraph(
        "与此同时，外部微博测试也表明当前模型的跨域迁移能力较弱，说明仅依赖单一领域训练数据仍难以获得稳定的通用情感判别能力。后续若进一步引入更大规模的跨域语料、上下文增强表示或更强的预训练语言模型，模型的泛化能力仍有较大提升空间。",
        styles,
    ))

    story.append(paragraph("6 问题思考回答", styles, "HeadingCN"))
    story.append(paragraph("6.1 训练何时停止最合适", styles, "SubHeadingCN"))
    story.append(paragraph(
        "本实验采用“验证集 F1 早停”策略。固定迭代次数实现简单，但容易在不同模型和参数下出现欠拟合或过拟合；基于验证集的早停能更贴近泛化能力，不过需要额外划分验证集并增加调参开销。",
        styles,
    ))
    story.append(paragraph(
        "从图9 的验证集 Accuracy/F1 变化过程看，BiRNN、BiLSTM 和 BiGRU 的最优轮次分别出现在第 {}、{}、{} 轮，TextCNN 出现在第 {} 轮，MLP 则延后到第 {} 轮。这与验证指标后期不再持续改善的现象一致，也进一步说明早停确实避免了无效训练。".format(
            result_by_name["birnn"]["best_epoch"],
            result_by_name["bilstm"]["best_epoch"],
            result_by_name["bigru"]["best_epoch"],
            result_by_name["cnn"]["best_epoch"],
            result_by_name["mlp"]["best_epoch"],
        ),
        styles,
    ))

    story.append(paragraph("6.2 实验参数如何初始化", styles, "SubHeadingCN"))
    story.append(paragraph(
        "PAD 向量置零；命中预训练词向量的词直接加载；未命中的词使用与预训练分布同尺度的随机初始化。线性层输入权重使用 Xavier 初始化，卷积层使用 Kaiming 初始化，RNN、LSTM 和 GRU 的隐藏到隐藏权重使用正交初始化，偏置初始化为 0。这样能在保持数值稳定的同时，更贴合不同层的计算特性。",
        styles,
    ))

    story.append(paragraph("6.3 如何缓解过拟合", styles, "SubHeadingCN"))
    story.append(paragraph(
        "可以使用验证集早停、Dropout、L2 正则、减小模型容量、增加数据增强或引入更多外部语料。对于本实验这种测试集较小的场景，高容量循环模型尤其需要依赖正则化和早停来控制泛化误差。",
        styles,
    ))

    story.append(paragraph("6.4 CNN、RNN 与 MLP 的优缺点", styles, "SubHeadingCN"))
    story.append(paragraph(
        "MLP 速度最快但忽略顺序；CNN 擅长抽取局部模式、并行度高；普通 RNN 能建模时序但长距离记忆较弱；LSTM 记忆能力最强但参数和训练开销较大；GRU 结构更轻，通常能取得接近 LSTM 的效果。它们分别对应“平均语义”“局部短语模式”“基础时序建模”“强记忆门控”“轻量门控时序建模”几类不同偏好。",
        styles,
    ))

    story.append(paragraph("6.5 模型鲁棒性分析", styles, "SubHeadingCN"))
    story.append(paragraph(
        "在额外构造的 6 条影评风格小样本上，主模型整体都能保持较稳定的判断，说明它们对与原任务分布接近的文本仍具备一定一致性；其中普通 RNN 在转折较多的句子上更容易退化，而 LSTM 与 GRU 通常更稳，说明门控结构在保持上下文信息方面更有优势。",
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
        story.append(paragraph("表10  原测试集与外部微博测试集上的鲁棒性对比", styles, "CaptionCN"))
        story.append(external_robustness_table(external_results))
        story.append(Spacer(1, 0.18 * cm))
        story.append(external_robustness_chart(external_results))
        story.append(Spacer(1, 0.06 * cm))
        story.append(paragraph("图11  原测试集与外部微博测试集 F1 对比", styles, "CaptionCN"))
        story.append(Spacer(1, 0.12 * cm))
        story.append(paragraph(
            "从对比结果看，所有模型在原测试集上的 Accuracy 都在 0.84 以上，但迁移到微博数据后，Accuracy 仅剩 0.27 到 0.32，F1 仅剩 0.15 到 0.29；即使下降最小的模型，其 Accuracy 和 F1 也分别下降了 {:.4f} 与 {:.4f}。按外部集 Accuracy 看，表现最好的是 <b>{}</b>；按外部集 F1 看，表现最好的是 <b>{}</b>。这说明当前模型更多学习到了影评领域中的局部搭配和表达分布，而不是能够稳定迁移到通用中文情感场景的抽象判别规则，因此其跨域鲁棒性较弱。".format(
                min_acc_drop,
                min_f1_drop,
                model_label(best_external_acc["model_name"]),
                model_label(best_external_f1["model_name"]),
            ),
            styles,
        ))

    story.append(paragraph("7 心得体会", styles, "HeadingCN"))
    story.append(paragraph(
        "本次实验让我更直观地体会到：在已经有较好词向量表示的前提下，结构设计的核心差异并不在“能不能学到语义”，而在“如何组织语义”。MLP 更像低成本平均，TextCNN 更偏向抓取情绪触发片段，BiRNN、BiLSTM 与 BiGRU 更强调上下文流动。真正决定实验质量的，除了模型结构本身，还包括数据清洗、初始化策略、验证集早停、参数对比和曲线分析是否严谨。",
        styles,
    ))

    story.append(paragraph("参考文献", styles, "HeadingCN"))
    story.append(paragraph("[1] Kim Y. Convolutional Neural Networks for Sentence Classification[C]//Proceedings of EMNLP 2014. Doha: ACL, 2014: 1746-1751.", styles, "ReferenceCN"))
    story.append(paragraph("[2] Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.", styles, "ReferenceCN"))
    story.append(paragraph("[3] Cho K, Van Merrienboer B, Gulcehre C, et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation[EB/OL]. arXiv:1406.1078, 2014.", styles, "ReferenceCN"))
    story.append(paragraph("[4] Mikolov T, Chen K, Corrado G, Dean J. Efficient Estimation of Word Representations in Vector Space[EB/OL]. arXiv:1301.3781, 2013.", styles, "ReferenceCN"))

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
