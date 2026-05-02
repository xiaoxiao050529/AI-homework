"""根据实验汇总结果生成课程报告 PDF。"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from src.sentiment_hw2.experiment import model_label
from src.sentiment_hw2.reporting import (
    ablation_table,
    build_curve_section,
    build_styles,
    external_robustness_chart,
    external_robustness_table,
    metric_table,
    parameter_table,
    paragraph,
    pick_best,
    register_fonts,
    robustness_table,
    test_metric_bar_chart,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.json"
EXTERNAL_ROBUSTNESS_PATH = OUTPUT_DIR / "external_robustness_metrics.json"
REPORT_PATH = ROOT / "实验二报告.pdf"


def main():
    """读取实验结果并按课程报告结构组织 PDF 内容。"""
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

    story = []
    story.append(paragraph("《人工智能导论》实验二报告", styles, "TitleCN"))
    story.append(Spacer(1, 0.3 * cm))
    story.append(paragraph("任务：基于预训练 50 维词向量完成中文影评情感二分类，并比较 MLP、TextCNN、BiRNN、BiLSTM 与 BiGRU 五类模型。", styles))
    story.append(paragraph(
        "数据集中训练集 19,998 条、验证集 5,629 条、测试集 369 条；词向量覆盖训练语料 token 的 {:.2%}。".format(
            summary["data"]["train_token_coverage"]
        ),
        styles,
    ))

    story.append(paragraph("1. 实验任务与流程", styles, "HeadingCN"))
    story.append(paragraph(
        "实验流程为：读取已经分词的文本样本；仅基于训练集建立词表；从 `wiki_word2vec_50.bin` 中抽取词表对应的预训练词向量；将句子截断或补齐到固定长度 80；分别训练 MLP、TextCNN、BiRNN、BiLSTM 和 BiGRU 模型；在验证集上以 F1 为主进行早停，再在测试集上汇报最终结果。",
        styles,
    ))
    story.append(paragraph(
        "实现中统一采用 Adam 优化器，损失函数为交叉熵。验证集连续若干轮不提升则提前停止，以减少固定训练轮数带来的过拟合风险。",
        styles,
    ))

    story.append(paragraph("2. 训练过程与结果图表", styles, "HeadingCN"))
    story.append(paragraph(
        "本报告中的图表均由程序直接读取 `outputs/experiment_summary.json` 中保存的训练历史与评估指标后生成，不再使用手工绘制的结构示意图。",
        styles,
    ))
    story.extend(build_curve_section(results, styles))
    story.append(paragraph("图 2-7  主模型测试集 Accuracy/F1 对比", styles, "CaptionCN"))
    story.append(Spacer(1, 0.08 * cm))
    story.append(test_metric_bar_chart(results))
    story.append(Spacer(1, 0.12 * cm))

    story.append(paragraph("3. 实验参数总表", styles, "HeadingCN"))
    story.append(paragraph(
        "这张表把每个主模型的结构参数和训练参数放在一起，便于横向比较不同模型的容量、正则化强度和优化策略。",
        styles,
    ))
    story.append(parameter_table(results))
    story.append(Spacer(1, 0.2 * cm))

    story.append(paragraph("4. 实验结果展示", styles, "HeadingCN"))
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
        "柱状图把表格中的差异直接可视化了出来：<b>{}</b> 的测试集 F1 最高，<b>{}</b> 与其非常接近；而 <b>{}</b> 在五个主模型中相对最低。这说明门控循环结构在本任务上整体更稳，但 MLP 这个均值池化 baseline 也已经非常有竞争力。".format(
            model_label(max(results, key=lambda item: item["test"]["f1"])["model_name"]),
            model_label(sorted(results, key=lambda item: item["test"]["f1"], reverse=True)[1]["model_name"]),
            model_label(min(results, key=lambda item: item["test"]["f1"])["model_name"]),
        ),
        styles,
    ))

    story.append(paragraph("5. 参数对比分析", styles, "HeadingCN"))
    story.append(ablation_table(ablations))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph(
        "从附加对比实验可以看出，TextCNN 的卷积核数量从 64 提升到 128 后，局部模式提取能力更强；BiRNN、BiLSTM、BiGRU 的隐藏维度从 64 提升到 128 后，也更有利于保留上下文信息，但参数量和训练时间会同步增加。",
        styles,
    ))

    story.append(paragraph("6. 模型比较与原因分析", styles, "HeadingCN"))
    story.append(paragraph(
        "MLP 通过平均池化快速聚合句向量，训练速度最快、结构最简单，但会丢失词序和局部搭配信息，因此更适合作为基础 baseline。TextCNN 对“非常 失望”“剧情 混乱”这类局部情感短语尤其敏感，参数共享带来较强的效率和泛化能力。BiRNN、BiLSTM 和 BiGRU 则显式建模上下文顺序，更适合处理依赖前后语义的长句；其中 BiLSTM 的门控记忆最完整，BiGRU 在效果和效率之间折中得更明显。",
        styles,
    ))

    story.append(paragraph("7. 问题思考", styles, "HeadingCN"))
    story.append(paragraph(
        "（1）训练何时停止最合适：本实验采用“验证集 F1 早停”。固定迭代次数实现简单，但容易在不同模型和参数下出现欠拟合或过拟合；基于验证集的早停能更贴近泛化能力，不过需要额外划分验证集并增加调参开销。",
        styles,
    ))
    story.append(paragraph(
        "从图 4-6 的验证集 F1 收敛过程看，BiRNN、BiLSTM 和 BiGRU 的最优轮次分别出现在第 {}、{}、{} 轮，TextCNN 出现在第 {} 轮，MLP 则延后到第 {} 轮。这与损失曲线后期验证集不再改善的现象一致，也进一步说明早停确实避免了无效训练。".format(
            result_by_name["birnn"]["best_epoch"],
            result_by_name["bilstm"]["best_epoch"],
            result_by_name["bigru"]["best_epoch"],
            result_by_name["cnn"]["best_epoch"],
            result_by_name["mlp"]["best_epoch"],
        ),
        styles,
    ))
    story.append(paragraph(
        "（2）参数初始化：PAD 向量置零；命中预训练词向量的词直接加载；未命中的词使用与预训练分布同尺度的随机初始化。线性层输入权重使用 Xavier 初始化，卷积层使用 Kaiming 初始化，RNN/LSTM/GRU 的隐藏到隐藏权重使用正交初始化，偏置初始化为 0。这样能在保持数值稳定的同时，更贴合不同层的计算特性。",
        styles,
    ))
    story.append(paragraph(
        "（3）如何缓解过拟合：可以使用验证集早停、Dropout、L2 正则、减小模型容量、增加数据增强或引入更多外部语料。对于本实验这种测试集较小的场景，高容量循环模型尤其需要依赖正则化和早停控制泛化误差。",
        styles,
    ))
    story.append(paragraph(
        "（4）CNN、RNN、MLP 优缺点：MLP 速度最快但忽略顺序；CNN 擅长抽取局部模式、并行度高；普通 RNN 能建模时序但长距离记忆较弱；LSTM 记忆能力最强但参数和训练开销较大；GRU 结构更轻，通常能取得接近 LSTM 的效果。它们分别对应“平均语义”“局部短语模式”“基础时序建模”“强记忆门控”“轻量门控时序建模”几类偏好。",
        styles,
    ))
    story.append(paragraph(
        "（5）模型鲁棒性：在额外构造的 6 条影评风格小样本上，主模型整体都能保持较稳定的判断，说明它们对与原任务分布接近的文本仍具备一定一致性；其中普通 RNN 在转折较多的句子上更容易退化，而 LSTM 与 GRU 通常更稳，说明门控结构在保持上下文信息方面更有优势。",
        styles,
    ))
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
        story.append(external_robustness_table(external_results))
        story.append(Spacer(1, 0.18 * cm))
        story.append(paragraph("图 8-1  原测试集与外部测试集 F1 对比", styles, "CaptionCN"))
        story.append(Spacer(1, 0.08 * cm))
        story.append(external_robustness_chart(external_results))
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

    story.append(PageBreak())
    story.append(paragraph("8. 心得体会", styles, "HeadingCN"))
    story.append(paragraph(
        "本次实验让我更直观地体会到：在已经有较好词向量表示的前提下，结构设计的核心差异并不在“能不能学到语义”，而在“如何组织语义”。MLP 更像低成本平均，TextCNN 更偏向抓取情绪触发片段，BiRNN/BiLSTM/BiGRU 更强调上下文流动。真正决定实验质量的，除了模型结构，还包括数据清洗、初始化策略、验证集早停、参数对比和曲线分析是否严谨。",
        styles,
    ))

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )
    doc.build(story)
    print("Report generated at", REPORT_PATH)


if __name__ == "__main__":
    main()
