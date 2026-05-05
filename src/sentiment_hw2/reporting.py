"""课程报告的版式、表格和结构图插入辅助函数。"""

from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPM
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage, KeepTogether, Paragraph, Spacer, Table, TableStyle

from .experiment import model_label
from .report_diagrams import ensure_diagram_assets


GRAPHICS_FONT_CANDIDATES = [
    ("DroidSansFallback", Path("/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf")),
]
BODY_FONT_NAME = "STSong-Light"
GRAPHICS_FONT_NAME = "Helvetica"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOADED_MODEL_DIAGRAMS = {
    "mlp": PROJECT_ROOT / "ChatGPT Image May 4, 2026, 01_25_52 AM (1).png",
    "cnn": PROJECT_ROOT / "ChatGPT Image May 4, 2026, 01_25_52 AM (2).png",
    "rnn_family": PROJECT_ROOT / "ChatGPT Image May 4, 2026, 01_26_27 AM.png",
}


def register_fonts() -> None:
    """注册中文字体，兼顾中文与正文中的英文、数字显示。"""
    global BODY_FONT_NAME, GRAPHICS_FONT_NAME
    registerFont(UnicodeCIDFont("STSong-Light"))
    BODY_FONT_NAME = "STSong-Light"
    GRAPHICS_FONT_NAME = "Helvetica"
    for font_name, font_path in GRAPHICS_FONT_CANDIDATES:
        if font_path.exists():
            registerFont(TTFont(font_name, str(font_path)))
            GRAPHICS_FONT_NAME = font_name
            break


def build_styles():
    """定义整份课程论文报告复用的中文段落样式。"""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCN",
            parent=styles["Title"],
            fontName=BODY_FONT_NAME,
            fontSize=20,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitleCN",
            parent=styles["Title"],
            fontName=BODY_FONT_NAME,
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetaCN",
            parent=styles["BodyText"],
            fontName=BODY_FONT_NAME,
            fontSize=11,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingCN",
            parent=styles["Heading1"],
            fontName=BODY_FONT_NAME,
            fontSize=14,
            leading=22,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeadingCN",
            parent=styles["Heading2"],
            fontName=BODY_FONT_NAME,
            fontSize=12.5,
            leading=20,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCN",
            parent=styles["BodyText"],
            fontName=BODY_FONT_NAME,
            fontSize=12,
            leading=20,
            alignment=TA_JUSTIFY,
            firstLineIndent=24,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyNoIndentCN",
            parent=styles["BodyCN"],
            firstLineIndent=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionCN",
            parent=styles["BodyText"],
            fontName=BODY_FONT_NAME,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KeywordCN",
            parent=styles["BodyNoIndentCN"],
            fontName=BODY_FONT_NAME,
            fontSize=11,
            leading=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCN",
            parent=styles["BodyText"],
            fontName=BODY_FONT_NAME,
            fontSize=9,
            leading=13,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReferenceCN",
            parent=styles["BodyText"],
            fontName=BODY_FONT_NAME,
            fontSize=10.5,
            leading=16,
            spaceAfter=2,
        )
    )
    return styles


def paragraph(text: str, styles, style_name: str = "BodyCN") -> Paragraph:
    """统一处理换行符，减少正文拼接时的重复样板代码。"""
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


MODEL_CONFIGS = {
    "mlp": {
        "core": "hidden_dim=128, dropout=0.3",
        "train": "batch=128, epochs=10, lr=1e-3, wd=1e-4, patience=3, clip=0",
    },
    "cnn": {
        "core": "num_filters=128, filter_sizes=3/4/5, dropout=0.5",
        "train": "batch=128, epochs=10, lr=1e-3, wd=1e-4, patience=3, clip=0",
    },
    "birnn": {
        "core": "hidden_dim=128, num_layers=1, dropout=0.5",
        "train": "batch=64, epochs=18, lr=1.5e-4, emb_lr=3e-5, wd=7e-4, patience=6, clip=0.5, warmup=4, freeze_emb=4",
    },
    "bilstm": {
        "core": "hidden_dim=128, num_layers=1, dropout=0.5",
        "train": "batch=64, epochs=18, lr=1.5e-4, emb_lr=3e-5, wd=7e-4, patience=6, clip=0.5, warmup=4, freeze_emb=4",
    },
    "bigru": {
        "core": "hidden_dim=128, num_layers=1, dropout=0.5",
        "train": "batch=64, epochs=18, lr=1.5e-4, emb_lr=3e-5, wd=7e-4, patience=6, clip=0.5, warmup=4, freeze_emb=4",
    },
    "transformer": {
        "core": "max_len=40, model_dim=128, heads=4, layers=1, ff_dim=256, pooling=cls, dropout=0.1",
        "train": "batch=256, epochs=12, lr=8e-4, emb_lr=8e-5, wd=1e-4, patience=4, clip=1, warmup=2",
    },
}

MAIN_MODEL_ORDER = ["mlp", "cnn", "birnn", "bilstm", "bigru", "transformer"]
MODEL_DIAGRAM_FIGURE_COUNT = 4
FIRST_TRAINING_CURVE_FIGURE_NUMBER = MODEL_DIAGRAM_FIGURE_COUNT + 1
CHART_COLORS = [
    colors.HexColor("#2563EB"),
    colors.HexColor("#DC2626"),
    colors.HexColor("#059669"),
    colors.HexColor("#D97706"),
    colors.HexColor("#7C3AED"),
    colors.HexColor("#0F766E"),
]
CHART_MARKERS = ["FilledCircle", "FilledSquare", "FilledDiamond", "Cross", "Circle"]
HYPERPARAMETER_EXPLANATIONS = {
    "hidden_dim": (
        "影响隐藏表示的容量。在 MLP 中它是中间全连接层的宽度，在 BiRNN/BiGRU 中它是每个方向的隐藏状态维度；本质上决定模型能用多大的特征空间去压缩句子信息。取值偏小时，模型表达能力不足，容易漏掉细粒度情感线索，表现为欠拟合；取值偏大时，参数量、显存和训练时间都会增加，也更容易把训练集噪声记进去，导致过拟合。"
    ),
    "num_layers": (
        "影响循环编码器的深度。增加层数相当于让高层在低层时序表示之上再做一次抽象，理论上能学习更复杂的上下文组合。层数过少时，模型只能进行较浅的序列变换；层数过多时，优化难度明显上升，训练更慢，也更容易出现梯度传播不稳定和收益递减。"
    ),
    "num_filters": (
        "影响 TextCNN 每种卷积核尺寸下可学习的局部模式探测器数量。每个 filter 都可以理解为一种“情感短语模板”；数量偏少时，可覆盖的局部模式有限，容易漏掉不同表达方式；数量偏多时，虽然模式覆盖更广，但参数量会同步增加，带来更高的过拟合风险和更慢的训练。"
    ),
    "filter_sizes": (
        "影响卷积核沿时间轴一次观察多少个相邻 token，也就是 n-gram 感受野大小。较小的窗口更擅长抓“好看”“失望”这类短局部模式；较大的窗口更容易覆盖“并 不 推荐”“一点 也 不 好”这类更长搭配。若窗口整体偏小，模型可能看不到较长的否定或转折结构；若窗口整体偏大，则容易把过多无关上下文卷进去，对短句尤其可能造成特征稀释。"
    ),
    "dropout": (
        "影响训练时随机屏蔽激活单元的比例，其原理是打破特征之间的过强共适应，从而提升泛化能力。dropout 偏小时，正则化不足，模型更容易记住训练集；dropout 偏大时，有效信息会被丢掉太多，优化变难，训练和验证指标都可能一起下降，表现为欠拟合。"
    ),
    "learning_rate": (
        "影响 AdamW 每次参数更新的步长，是最直接决定收敛速度与稳定性的训练超参数。学习率偏小时，每一步移动太保守，有限训练轮数内可能还没走到较优区域；学习率偏大时，参数会在最优点附近来回震荡，严重时甚至直接发散，导致验证集指标不稳定。"
    ),
    "batch_size": (
        "影响每次梯度更新使用多少条样本。batch 较小时，梯度噪声更大，更新更频繁，往往带来一定正则化效果，但训练波动也更明显；batch 较大时，梯度估计更平滑、吞吐更高，但更新次数减少，且有时会削弱泛化能力。过小可能让训练不稳定、耗时增加；过大则可能让模型更容易停在较“平”的次优点，或受显存与内存限制。"
    ),
    "weight_decay": (
        "影响 L2 正则化强度。它通过惩罚过大的权重，抑制模型把少量训练样本拟合得过于极端。weight decay 偏小时，约束不足，模型更容易过拟合；偏大时，权重会被压得过小，模型难以形成足够强的判别边界，表现为欠拟合。"
    ),
    "epochs": (
        "影响完整遍历训练集的最大轮数，也就是给模型多少优化预算。轮数偏小时，模型可能还没学到稳定的判别模式就停止训练；轮数偏大时，虽然在有早停时风险可控，但仍会增加训练时间，并在早停不敏感时放大过拟合的可能。"
    ),
    "patience": (
        "影响早停策略的容忍度，即验证集指标连续多少轮不提升后才停止训练。patience 偏小时，模型可能在正常波动阶段就被提前终止，错过后续继续提升；patience 偏大时，虽然更稳妥，但也会拉长训练时间，甚至让模型在无效轮次中继续朝训练集过拟合。"
    ),
    "seed": (
        "影响随机初始化、batch 打乱顺序以及部分随机训练行为。它不是“越大越好”或“越小越好”的量纲型超参数，而是用来观察实验稳定性的随机起点。不同 seed 可能带来略有差异的最优点；如果只看单个 seed，结果可能带有偶然性，因此更重要的是检查模型对 seed 是否敏感。"
    ),
}


def ordered_main_results(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按主实验统一顺序排列结果，避免图表次序前后不一致。"""
    rank = {name: idx for idx, name in enumerate(MAIN_MODEL_ORDER)}
    return sorted(results, key=lambda item: rank.get(item["model_name"], len(rank)))


def add_legend(
    drawing: Drawing,
    x: float,
    y: float,
    entries: Sequence[tuple],
    font_name: str = None,
    column_maximum: int = None,
    deltax: float = 54,
    deltay: float = 10,
) -> None:
    """为折线图和柱状图添加统一样式的图例。"""
    legend = Legend()
    legend.x = x
    legend.y = y
    legend.dx = 8
    legend.dy = 8
    legend.deltax = deltax
    legend.deltay = deltay
    legend.fontName = GRAPHICS_FONT_NAME if font_name is None else font_name
    legend.fontSize = 8
    legend.columnMaximum = len(entries) if column_maximum is None else column_maximum
    legend.colorNamePairs = list(entries)
    drawing.add(legend)


def apply_academic_table_style(
    table: Table,
    font_size: float = 9,
    header_background=colors.HexColor("#F2F2F2"),
    align: str = "CENTER",
) -> Table:
    """统一应用更接近课程论文风格的黑白表格样式。"""
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), BODY_FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), header_background),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("ALIGN", (0, 0), (-1, -1), align),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def metric_table(results: List[Dict[str, object]]) -> Table:
    """把主实验结果整理成总表。"""
    rows = [["模型", "最佳轮次", "验证集 Acc", "验证集 F1", "测试集 Acc", "测试集 F1", "参数量"]]
    for result in results:
        rows.append(
            [
                model_label(result["model_name"]),
                str(result["best_epoch"]),
                "{:.4f}".format(result["validation"]["accuracy"]),
                "{:.4f}".format(result["validation"]["f1"]),
                "{:.4f}".format(result["test"]["accuracy"]),
                "{:.4f}".format(result["test"]["f1"]),
                "{:,}".format(result["parameter_count"]),
            ]
        )
    table = Table(rows, colWidths=[2.6 * cm, 2.0 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm])
    return apply_academic_table_style(table, font_size=9)


def parameter_table(results: List[Dict[str, object]]) -> Table:
    """把主模型的结构参数和训练参数合并成一张总表。"""
    small_style = ParagraphStyle(
        name="TableSmallCN",
        fontName=BODY_FONT_NAME,
        fontSize=7.5,
        leading=9,
    )
    rows = [["模型", "结构参数", "训练参数", "验证集 F1", "测试集 F1", "参数量"]]
    for result in results:
        cfg = MODEL_CONFIGS[result["model_name"]]
        rows.append(
            [
                model_label(result["model_name"]),
                Paragraph(cfg["core"].replace(", ", "<br/>"), small_style),
                Paragraph(cfg["train"].replace(", ", "<br/>"), small_style),
                "{:.4f}".format(result["validation"]["f1"]),
                "{:.4f}".format(result["test"]["f1"]),
                "{:,}".format(result["parameter_count"]),
            ]
        )
    table = Table(rows, colWidths=[1.8 * cm, 4.6 * cm, 5.8 * cm, 1.6 * cm, 1.6 * cm, 2.0 * cm])
    return apply_academic_table_style(table, font_size=8.0)


def transformer_experiment_table(experiments: Sequence[Dict[str, object]]) -> Table:
    """把 Transformer 追加实验整理成报告表格。"""
    rows = [["实验", "最佳轮次", "词向量", "验证集 Acc", "验证集 F1", "测试集 Acc", "测试集 F1", "参数量", "用时/s"]]
    for item in experiments:
        result = item["result"]
        history = result.get("history", [])
        if history:
            embedding_state = "已解冻" if history[-1].get("embedding_trainable") else "冻结"
        else:
            embedding_state = "未知"
        rows.append(
            [
                item["label"],
                str(result["best_epoch"]),
                embedding_state,
                "{:.4f}".format(result["validation"]["accuracy"]),
                "{:.4f}".format(result["validation"]["f1"]),
                "{:.4f}".format(result["test"]["accuracy"]),
                "{:.4f}".format(result["test"]["f1"]),
                "{:,}".format(result["parameter_count"]),
                "{:.1f}".format(result["train_seconds"]),
            ]
        )
    table = Table(
        rows,
        colWidths=[2.5 * cm, 1.4 * cm, 1.5 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm],
    )
    return apply_academic_table_style(table, font_size=7.6)


def validation_metric_curve_drawing(
    result: Dict[str, object],
    width: float = 15.8 * cm,
    height: float = 6.0 * cm,
    export_mode: bool = False,
) -> Drawing:
    """把单个模型的验证集 Accuracy 和 F1 画成曲线。"""
    history = result["history"]
    accuracy_points = [(record["epoch"], record["validation_accuracy"]) for record in history]
    f1_points = [(record["epoch"], record["validation_f1"]) for record in history]
    epochs = [record["epoch"] for record in history]
    y_min = min(min(value for _, value in accuracy_points), min(value for _, value in f1_points))
    y_max = max(max(value for _, value in accuracy_points), max(value for _, value in f1_points))

    drawing = Drawing(width, height + 14)
    chart = LinePlot()
    chart.x = 42
    chart.y = 18
    chart.width = width - 74
    chart.height = height - 10
    chart.data = [accuracy_points, f1_points]
    chart.joinedLines = 1
    chart.lines[0].strokeColor = colors.HexColor("#2563EB")
    chart.lines[0].strokeWidth = 1.8
    chart.lines[0].symbol = makeMarker("FilledCircle")
    chart.lines[1].strokeColor = colors.HexColor("#DC2626")
    chart.lines[1].strokeWidth = 1.8
    chart.lines[1].symbol = makeMarker("FilledSquare")
    chart.xValueAxis.valueMin = min(epochs)
    chart.xValueAxis.valueMax = max(epochs)
    chart.xValueAxis.valueStep = 1
    chart.yValueAxis.valueMin = max(0.0, round(y_min - 0.03, 2))
    chart.yValueAxis.valueMax = min(1.0, round(y_max + 0.02, 2))
    chart.yValueAxis.valueStep = 0.01
    drawing.add(chart)
    add_legend(
        drawing,
        x=width - 110,
        y=height + 1,
        entries=[
            (colors.HexColor("#2563EB"), "验证集 Accuracy"),
            (colors.HexColor("#DC2626"), "验证集 F1"),
        ],
        font_name=GRAPHICS_FONT_NAME if export_mode else BODY_FONT_NAME,
        column_maximum=2,
        deltax=44,
        deltay=10,
    )
    return drawing


def validation_f1_curve_drawing(
    results: Sequence[Dict[str, object]],
    width: float = 15.8 * cm,
    height: float = 6.0 * cm,
    export_mode: bool = False,
) -> Drawing:
    """把五个主模型的验证集 F1 收敛过程画在同一张图里。"""
    ordered_results = ordered_main_results(results)
    series = [[(record["epoch"], record["validation_f1"]) for record in result["history"]] for result in ordered_results]
    max_epoch = max(point[0] for curve in series for point in curve)
    min_f1 = min(point[1] for curve in series for point in curve)
    max_f1 = max(point[1] for curve in series for point in curve)

    drawing = Drawing(width, height + 14)
    chart = LinePlot()
    chart.x = 42
    chart.y = 18
    chart.width = width - 150
    chart.height = height - 8
    chart.data = series
    chart.joinedLines = 1
    chart.xValueAxis.valueMin = 1
    chart.xValueAxis.valueMax = max_epoch
    chart.xValueAxis.valueStep = 1
    chart.yValueAxis.valueMin = max(0.0, round(min_f1 - 0.02, 2))
    chart.yValueAxis.valueMax = min(1.0, round(max_f1 + 0.01, 2))
    chart.yValueAxis.valueStep = 0.01
    for idx, result in enumerate(ordered_results):
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        chart.lines[idx].strokeColor = color
        chart.lines[idx].strokeWidth = 1.6
        chart.lines[idx].symbol = makeMarker(CHART_MARKERS[idx % len(CHART_MARKERS)])
        chart.lines[idx].symbol.strokeColor = color
        chart.lines[idx].symbol.fillColor = color
    drawing.add(chart)
    add_legend(
        drawing,
        x=width - 102,
        y=height + 1,
        entries=[(CHART_COLORS[idx % len(CHART_COLORS)], model_label(result["model_name"])) for idx, result in enumerate(ordered_results)],
        font_name="Helvetica" if export_mode else BODY_FONT_NAME,
        column_maximum=5,
        deltax=38,
        deltay=11,
    )
    return drawing


def grouped_metric_bar_chart(
    category_names: Sequence[str],
    series: Sequence[Sequence[float]],
    legend_labels: Sequence[str],
    title: str,
    footnote: str,
    width: float = 15.8 * cm,
    height: float = 6.5 * cm,
    value_min: float = 0.0,
    value_max: float = 1.0,
    value_step: float = 0.1,
    export_mode: bool = False,
) -> Drawing:
    """生成两组指标并列柱状图。"""
    drawing = Drawing(width, height + 24)
    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 30
    chart.width = width - 126
    chart.height = height - 18
    chart.data = [list(values) for values in series]
    chart.groupSpacing = 12
    chart.barSpacing = 3
    chart.categoryAxis.categoryNames = list(category_names)
    chart.categoryAxis.labels.fontName = "Helvetica" if export_mode else BODY_FONT_NAME
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.dy = -10
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = value_min
    chart.valueAxis.valueMax = value_max
    chart.valueAxis.valueStep = value_step
    chart.barLabelFormat = "%.3f"
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 6
    chart.barLabels.nudge = 6
    chart.barLabels.fillColor = colors.HexColor("#334155")
    for idx in range(len(series)):
        chart.bars[idx].fillColor = CHART_COLORS[idx % len(CHART_COLORS)]
    drawing.add(chart)
    add_legend(
        drawing,
        x=width - 76,
        y=height - 2,
        entries=[(CHART_COLORS[idx % len(CHART_COLORS)], label) for idx, label in enumerate(legend_labels)],
        font_name="Helvetica" if export_mode else BODY_FONT_NAME,
        column_maximum=2,
        deltax=36,
        deltay=10,
    )
    drawing.add(
        String(
            42,
            4,
            footnote,
            fontName=GRAPHICS_FONT_NAME if export_mode else BODY_FONT_NAME,
            fontSize=7.2,
        )
    )
    return drawing


def test_metric_bar_chart(results: Sequence[Dict[str, object]], export_mode: bool = False) -> Drawing:
    """比较各主模型在测试集上的 Accuracy 与 F1。"""
    ordered_results = ordered_main_results(results)
    return grouped_metric_bar_chart(
        category_names=[model_label(result["model_name"]) for result in ordered_results],
        series=[
            [result["test"]["accuracy"] for result in ordered_results],
            [result["test"]["f1"] for result in ordered_results],
        ],
        legend_labels=["Test Accuracy", "Test F1"],
        title="主模型测试集指标对比",
        footnote="每根柱顶端标出具体数值，便于和表格结论交叉验证。",
        export_mode=export_mode,
    )


def external_robustness_chart(results: Sequence[Dict[str, object]], export_mode: bool = False) -> Drawing:
    """比较各主模型在原测试集与外部测试集上的 F1 落差。"""
    ordered_results = ordered_main_results(results)
    return grouped_metric_bar_chart(
        category_names=[model_label(result["model_name"]) for result in ordered_results],
        series=[
            [result["test_metrics"]["f1"] for result in ordered_results],
            [result["external_metrics"]["f1"] for result in ordered_results],
        ],
        legend_labels=["In-domain Test F1", "External F1"],
        title="原测试集与外部测试集 F1 对比",
        footnote="同一模型两根柱子的高度差越大，说明跨域迁移时的性能退化越明显。",
        export_mode=export_mode,
    )


def robustness_table(summary: Dict[str, object]) -> Table:
    """把附加鲁棒性测试整理成横向对比表。"""
    ordered_models = [name for name in MAIN_MODEL_ORDER if name in summary["robustness"]]
    rows = [["样本", "真实标签", *[model_label(name) for name in ordered_models]]]
    first_model = summary["robustness"][ordered_models[0]]
    for idx in range(len(first_model)):
        rows.append(
            [
                first_model[idx]["text"],
                str(first_model[idx]["gold"]),
                *[str(summary["robustness"][name][idx]["prediction"]) for name in ordered_models],
            ]
        )
    col_widths = [7.3 * cm, 1.5 * cm] + [1.25 * cm for _ in ordered_models]
    table = Table(rows, colWidths=col_widths)
    apply_academic_table_style(table, font_size=8.1, align="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 1), (1, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def external_robustness_table(results: List[Dict[str, object]]) -> Table:
    """把测试集与外部小样本的鲁棒性对比整理成总表。"""
    rows = [["模型", "测试集 Acc", "测试集 F1", "外部集 Acc", "外部集 F1", "Acc 下降", "F1 下降"]]
    for result in results:
        rows.append(
            [
                model_label(result["model_name"]),
                "{:.4f}".format(result["test_metrics"]["accuracy"]),
                "{:.4f}".format(result["test_metrics"]["f1"]),
                "{:.4f}".format(result["external_metrics"]["accuracy"]),
                "{:.4f}".format(result["external_metrics"]["f1"]),
                "{:.4f}".format(result["accuracy_drop"]),
                "{:.4f}".format(result["f1_drop"]),
            ]
        )
    table = Table(rows, colWidths=[2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])
    return apply_academic_table_style(table, font_size=8.5)


def hyperparameter_tuning_table(variants: List[Dict[str, object]]) -> Table:
    """把逐个超参数调节结果整理成总表。"""
    def stringify(value):
        if isinstance(value, list):
            return "(" + ", ".join(str(item) for item in value) + ")"
        return str(value)

    rows = [["超参数", "代表模型", "调整值", "测试集 Acc", "测试集 F1", "相对基线 ΔF1"]]
    for item in variants:
        spec = item["spec"]
        result = item["result"]
        rows.append(
            [
                spec["parameter_name"],
                model_label(spec["family"]),
                stringify(spec["parameter_value"]),
                "{:.4f}".format(result["test"]["accuracy"]),
                "{:.4f}".format(result["test"]["f1"]),
                "{:+.4f}".format(result["delta_vs_baseline"]["test_f1"]),
            ]
        )
    table = Table(rows, colWidths=[2.3 * cm, 2.1 * cm, 2.0 * cm, 2.3 * cm, 2.3 * cm, 2.8 * cm])
    return apply_academic_table_style(table, font_size=8.5)


def hyperparameter_family_table(families: Sequence[Dict[str, object]]) -> Table:
    """汇总每个模型的基线参数与可调超参数。"""
    small_style = ParagraphStyle(
        name="HyperparamFamilySmallCN",
        fontName=BODY_FONT_NAME,
        fontSize=7.4,
        leading=9,
    )
    rows = [["模型", "基线结构参数", "基线训练参数", "调参覆盖的超参数"]]
    for family in families:
        model_cfg = family["baseline_model_config"]
        train_cfg = family["baseline_train_config"]
        rows.append(
            [
                model_label(family["family"]),
                Paragraph(
                    "<br/>".join(
                        "{}={}".format(
                            key,
                            tuple(model_cfg[key]) if isinstance(model_cfg[key], list) else model_cfg[key],
                        )
                        for key in sorted(model_cfg.keys())
                    ),
                    small_style,
                ),
                Paragraph(
                    "<br/>".join(
                        "{}={}".format(
                            key,
                            tuple(train_cfg[key]) if isinstance(train_cfg[key], list) else train_cfg[key],
                        )
                        for key in sorted(train_cfg.keys())
                    ),
                    small_style,
                ),
                Paragraph("<br/>".join(family["tunable_parameters"]), small_style),
            ]
        )
    table = Table(rows, colWidths=[1.8 * cm, 4.0 * cm, 5.8 * cm, 4.4 * cm])
    return apply_academic_table_style(table, font_size=7.8)


def best_hyperparameter_table(best_by_family: Sequence[Dict[str, object]]) -> Table:
    """汇总每个模型族里每个超参数的最优值。"""
    rows = [["模型", "超参数", "基线值", "最优值", "最优 Val F1", "对应 Test F1"]]
    for family_group in best_by_family:
        for item in family_group["best_parameters"]:
            rows.append(
                [
                    model_label(family_group["family"]),
                    item["parameter_name"],
                    str(item["baseline_value_label"]),
                    str(item["best_value_label"]),
                    "{:.4f}".format(item["best_validation_f1"]),
                    "{:.4f}".format(item["best_test_f1"]),
                ]
            )
    table = Table(rows, colWidths=[1.8 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 2.4 * cm, 2.8 * cm])
    return apply_academic_table_style(table, font_size=8.0)


def tuning_parameter_chart(
    parameter_group: Dict[str, object],
    width: float = 15.8 * cm,
    height: float = 6.2 * cm,
    export_mode: bool = False,
) -> Drawing:
    """为单个超参数画出候选值对应的验证集/测试集 F1。"""
    candidates = parameter_group["candidates"]
    category_names = [entry["value_label"] for entry in candidates]
    validation_values = [entry["result"]["validation"]["f1"] for entry in candidates]
    test_values = [entry["result"]["test"]["f1"] for entry in candidates]
    all_values = validation_values + test_values
    best = parameter_group["best"]

    drawing = Drawing(width, height + 24)
    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 30
    chart.width = width - 74
    chart.height = height - 18
    chart.data = [validation_values, test_values]
    chart.groupSpacing = 12
    chart.barSpacing = 3
    chart.categoryAxis.categoryNames = category_names
    chart.categoryAxis.labels.fontName = GRAPHICS_FONT_NAME if export_mode else BODY_FONT_NAME
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.dy = -10
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = max(0.0, round(min(all_values) - 0.03, 2))
    chart.valueAxis.valueMax = min(1.0, round(max(all_values) + 0.03, 2))
    chart.valueAxis.valueStep = 0.02
    chart.barLabelFormat = "%.3f"
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 6
    chart.barLabels.nudge = 6
    chart.barLabels.fillColor = colors.HexColor("#334155")
    chart.bars[0].fillColor = colors.HexColor("#2563EB")
    chart.bars[1].fillColor = colors.HexColor("#DC2626")
    drawing.add(chart)
    add_legend(
        drawing,
        x=width - 94,
        y=height - 2,
        entries=[
            (colors.HexColor("#2563EB"), "Validation F1"),
            (colors.HexColor("#DC2626"), "Test F1"),
        ],
        font_name="Helvetica" if export_mode else BODY_FONT_NAME,
        column_maximum=2,
        deltax=44,
        deltay=10,
    )
    drawing.add(
        String(
            42,
            4,
            "基线值={}；最优值={}；最优 Test F1={:.4f}".format(
                parameter_group["baseline_value_label"],
                best["value_label"],
                best["test_f1"],
            ),
            fontName=GRAPHICS_FONT_NAME if export_mode else BODY_FONT_NAME,
            fontSize=7.2,
        )
    )
    return drawing


def ablation_table(ablations: List[Dict[str, object]]) -> Table:
    """把参数对比实验结果整理成表格。"""
    rows = [["对比实验", "验证集 Acc", "验证集 F1", "测试集 Acc", "测试集 F1"]]
    for result in ablations:
        rows.append(
            [
                model_label(result["model_name"]),
                "{:.4f}".format(result["validation"]["accuracy"]),
                "{:.4f}".format(result["validation"]["f1"]),
                "{:.4f}".format(result["test"]["accuracy"]),
                "{:.4f}".format(result["test"]["f1"]),
            ]
        )
    table = Table(rows, colWidths=[5.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
    return apply_academic_table_style(table, font_size=9)


def build_curve_section(results: List[Dict[str, object]], styles) -> List:
    """生成主模型训练曲线章节。"""
    ordered_results = ordered_main_results(results)
    first_figure_number = FIRST_TRAINING_CURVE_FIGURE_NUMBER
    validation_f1_figure_number = first_figure_number + len(ordered_results)
    blocks: List = [
        paragraph(
            "为展示不同结构在训练阶段的指标变化特征，下面给出了各模型验证集 Accuracy 和 F1 随训练轮次的变化过程。由于课程评价重点是 Accuracy 与 F1，因此这里直接围绕验证指标作图，更便于判断模型何时达到最佳泛化效果。",
            styles,
        )
    ]
    for offset, result in enumerate(ordered_results):
        blocks.append(validation_metric_curve_drawing(result, export_mode=False))
        blocks.append(Spacer(1, 0.06 * cm))
        blocks.append(
            paragraph(
                "图{}  {}验证集 Accuracy 与 F1 变化曲线".format(
                    first_figure_number + offset,
                    model_label(result["model_name"]),
                ),
                styles,
                "CaptionCN",
            )
        )
        blocks.append(Spacer(1, 0.18 * cm))
    blocks.append(validation_f1_curve_drawing(ordered_results, export_mode=False))
    blocks.append(Spacer(1, 0.06 * cm))
    blocks.append(paragraph("图{}  各模型验证集 F1 收敛曲线".format(validation_f1_figure_number), styles, "CaptionCN"))
    blocks.append(Spacer(1, 0.18 * cm))
    return blocks


def build_tuning_chart_section(parameter_groups: Sequence[Dict[str, object]], styles) -> List:
    """生成逐个超参数的图表章节。"""
    blocks: List = [
        paragraph(
            "下面的图表严格采用控制变量法：每次只改变一个超参数，其余结构参数与训练参数保持对应模型基线不变。横轴为候选取值，纵轴为验证集与测试集 F1，便于直接比较不同取值下的效果差异。",
            styles,
        )
    ]
    current_family = None
    for group in parameter_groups:
        if group["family"] != current_family:
            current_family = group["family"]
            blocks.append(paragraph(model_label(current_family), styles, "SubHeadingCN"))
        blocks.append(tuning_parameter_chart(group, export_mode=False))
        blocks.append(Spacer(1, 0.06 * cm))
        blocks.append(
            paragraph(
                "{}  {} 单变量调参结果".format(model_label(group["family"]), group["parameter_name"]),
                styles,
                "CaptionCN",
            )
        )
        blocks.append(Spacer(1, 0.14 * cm))
    return blocks


def build_hyperparameter_explanation_section(parameter_groups: Sequence[Dict[str, object]], styles) -> List:
    """在逐个调参图表前补充超参数含义说明。"""
    ordered_names = [
        "hidden_dim",
        "num_layers",
        "num_filters",
        "filter_sizes",
        "dropout",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "epochs",
        "patience",
        "seed",
    ]
    available_names = {group["parameter_name"] for group in parameter_groups}

    blocks: List = [
        paragraph(
            "为便于解读后续单变量调参图表，下面先说明本实验实际调到的超参数分别控制什么、其背后的训练原理是什么，以及取值偏小或偏大时通常会带来哪些现象。这样在阅读曲线时，就能把“指标变化”与“参数作用机制”对应起来，而不是只停留在经验比较层面。",
            styles,
        )
    ]
    for name in ordered_names:
        if name not in available_names:
            continue
        explanation = HYPERPARAMETER_EXPLANATIONS.get(name)
        if explanation is None:
            continue
        blocks.append(paragraph("<b>{}</b>：{}".format(name, explanation), styles, "BodyNoIndentCN"))

    blocks.append(Spacer(1, 0.12 * cm))
    return blocks


def build_hyperparameter_choice_analysis_section(parameter_groups: Sequence[Dict[str, object]], styles) -> List:
    """分析当前单变量调参中“最优值”为什么会被选中。"""
    group_map = {(group["family"], group["parameter_name"]): group for group in parameter_groups}

    def baseline_validation_f1(group: Dict[str, object]) -> float:
        for candidate in group["candidates"]:
            if candidate["is_baseline"]:
                return candidate["result"]["validation"]["f1"]
        return group["candidates"][0]["result"]["validation"]["f1"]

    blocks: List = [
        paragraph(
            "这里需要特别说明，表8中的“最优值”统一按验证集 F1 选取，而不是按测试集指标倒推。原因是测试集只能用于最终汇报，不能反向参与调参，否则会造成信息泄漏。因此，下文分析的“为什么这个值最好”，本质上是在解释：为什么它在当前单变量实验设置下给出了更高的验证集泛化表现，而不是单纯追求某一次测试集分数的偶然峰值。",
            styles,
        )
    ]

    mlp_hidden = group_map.get(("mlp", "hidden_dim"))
    mlp_batch = group_map.get(("mlp", "batch_size"))
    mlp_dropout = group_map.get(("mlp", "dropout"))
    mlp_lr = group_map.get(("mlp", "learning_rate"))
    mlp_epochs = group_map.get(("mlp", "epochs"))
    mlp_patience = group_map.get(("mlp", "patience"))
    mlp_wd = group_map.get(("mlp", "weight_decay"))
    mlp_seed = group_map.get(("mlp", "seed"))
    if all(item is not None for item in [mlp_hidden, mlp_batch, mlp_dropout, mlp_lr, mlp_epochs, mlp_patience, mlp_wd, mlp_seed]):
        blocks.append(
            paragraph(
                "从 MLP 的结果看，最优 hidden_dim={}、batch_size={}、dropout={}、learning_rate={}、epochs={}、patience={}、weight_decay={}、seed={}。这组结果说明：对“平均池化后再分类”的 MLP 来说，当前 baseline 的正则和步长都略偏保守。hidden_dim 从 {} 提到 {} 后，验证集 F1 从 {:.4f} 升到 {:.4f}，说明平均句向量本身信息量并不算低，增加分类头容量后，模型能把这些语义更充分地投影到判别空间里；batch_size 选到 {}，则说明在当前数据规模和优化器设置下，更平滑的梯度估计反而更有利于稳定收敛。与此同时，dropout 从 {} 降到 {}、weight_decay 从 {} 降到 {} 反而更优，说明这一路模型并不是过强容量导致过拟合，而更像是 baseline 正则稍强，压制了分类层对有效情感特征的利用。learning_rate 选到 {} 而不是 {}，也支持同样判断：更小的步长虽然收敛慢一点，但能让浅层分类器在预训练词向量上更稳定地调整决策边界。epochs={} 和 patience={} 则表明 MLP 不需要特别长的训练预算，继续训练未必能换来泛化提升；seed={} 最优更多说明 MLP 对随机初始化有一定敏感性，因此后续若要给出更稳妥结论，最好再做多 seed 平均，而不能只盯某一次运行。".format(
                    mlp_hidden["best"]["value_label"],
                    mlp_batch["best"]["value_label"],
                    mlp_dropout["best"]["value_label"],
                    mlp_lr["best"]["value_label"],
                    mlp_epochs["best"]["value_label"],
                    mlp_patience["best"]["value_label"],
                    mlp_wd["best"]["value_label"],
                    mlp_seed["best"]["value_label"],
                    mlp_hidden["baseline_value_label"],
                    mlp_hidden["best"]["value_label"],
                    baseline_validation_f1(mlp_hidden),
                    mlp_hidden["best"]["validation_f1"],
                    mlp_batch["best"]["value_label"],
                    mlp_dropout["baseline_value_label"],
                    mlp_dropout["best"]["value_label"],
                    mlp_wd["baseline_value_label"],
                    mlp_wd["best"]["value_label"],
                    mlp_lr["best"]["value_label"],
                    mlp_lr["baseline_value_label"],
                    mlp_epochs["best"]["value_label"],
                    mlp_patience["best"]["value_label"],
                    mlp_seed["best"]["value_label"],
                ),
                styles,
            )
        )

    cnn_filters = group_map.get(("cnn", "filter_sizes"))
    cnn_num_filters = group_map.get(("cnn", "num_filters"))
    cnn_batch = group_map.get(("cnn", "batch_size"))
    cnn_dropout = group_map.get(("cnn", "dropout"))
    cnn_lr = group_map.get(("cnn", "learning_rate"))
    cnn_epochs = group_map.get(("cnn", "epochs"))
    cnn_patience = group_map.get(("cnn", "patience"))
    cnn_wd = group_map.get(("cnn", "weight_decay"))
    cnn_seed = group_map.get(("cnn", "seed"))
    if all(item is not None for item in [cnn_filters, cnn_num_filters, cnn_batch, cnn_dropout, cnn_lr, cnn_epochs, cnn_patience, cnn_wd, cnn_seed]):
        blocks.append(
            paragraph(
                "TextCNN 的最优组合是 filter_sizes={}、num_filters={}、batch_size={}、dropout={}、learning_rate={}、epochs={}、patience={}、weight_decay={}，而 seed 仍以 {} 为最好。这个结果非常有解释价值。首先，filter_sizes 从 baseline 的 {} 调成 {} 后，验证集 F1 提升最明显之一，说明这份影评数据里的关键情感证据更集中在较短局部模式上，例如“很 失望”“不 推荐”“超出 预期”这类 2 到 4 个词的组合比更长的 5-gram 更常承担判别作用。其次，num_filters 从 {} 降到 {} 反而更好，说明 baseline 的卷积通道数略显冗余；在当前数据规模下，过多 filter 并没有持续带来新模式，反而可能把一些噪声局部模式也学进去。batch_size={} 和 learning_rate={} 共同最优，则表明 CNN 需要比 baseline 更稳的梯度更新：batch 太小会让训练波动过大，太大又会削弱泛化，而 {} 在这次实验里恰好落在较合适的中间位置。dropout 从 {} 降到 {}、weight_decay 从 {} 降到 {}、epochs={} 且 patience={} 的组合，则进一步说明 CNN 在这份任务上收敛很快，baseline 更像是“正则偏强、训练略长”，因此适当放松正则、缩短训练、让模型更早在验证集最好处停下来，反而更容易保住局部模式提取带来的优势。seed={} 仍为最优，也意味着 CNN 在不同随机种子下虽有波动，但 baseline 的初始化并不差。".format(
                    cnn_filters["best"]["value_label"],
                    cnn_num_filters["best"]["value_label"],
                    cnn_batch["best"]["value_label"],
                    cnn_dropout["best"]["value_label"],
                    cnn_lr["best"]["value_label"],
                    cnn_epochs["best"]["value_label"],
                    cnn_patience["best"]["value_label"],
                    cnn_wd["best"]["value_label"],
                    cnn_seed["best"]["value_label"],
                    cnn_filters["baseline_value_label"],
                    cnn_filters["best"]["value_label"],
                    cnn_num_filters["baseline_value_label"],
                    cnn_num_filters["best"]["value_label"],
                    cnn_batch["best"]["value_label"],
                    cnn_lr["best"]["value_label"],
                    cnn_batch["best"]["value_label"],
                    cnn_dropout["baseline_value_label"],
                    cnn_dropout["best"]["value_label"],
                    cnn_wd["baseline_value_label"],
                    cnn_wd["best"]["value_label"],
                    cnn_epochs["best"]["value_label"],
                    cnn_patience["best"]["value_label"],
                    cnn_seed["best"]["value_label"],
                ),
                styles,
            )
        )

    birnn_hidden = group_map.get(("birnn", "hidden_dim"))
    birnn_layers = group_map.get(("birnn", "num_layers"))
    if birnn_hidden is not None and birnn_layers is not None:
        blocks.append(
            paragraph(
                "BiRNN 的最优 hidden_dim={}、num_layers={}。其中提升最明显的是 num_layers：从 baseline 的 {} 层增加到 {} 层后，验证集 F1 从 {:.4f} 提升到 {:.4f}，说明对不带门控的普通 RNN 来说，单层表示能力偏弱，增加堆叠深度比单纯扩大状态维度更能补足建模能力。hidden_dim 从 {} 提高到 {} 也有提升，但幅度明显小于层数，而且测试集并没有同步放大收益，这意味着“加宽”虽然增强了容量，却也带来了更高的不稳定性。因此，这一轮单变量调参给出的核心结论不是“BiRNN 越大越好”，而是“普通循环单元先天表达力有限时，适度增加层数比盲目增大隐藏维更有效”。这也从侧面解释了为什么后续 BiLSTM 往往更容易取得稳定优势：与其不断给 vanilla RNN 叠容量，不如直接换成门控更强的单元。".format(
                    birnn_hidden["best"]["value_label"],
                    birnn_layers["best"]["value_label"],
                    birnn_layers["baseline_value_label"],
                    birnn_layers["best"]["value_label"],
                    baseline_validation_f1(birnn_layers),
                    birnn_layers["best"]["validation_f1"],
                    birnn_hidden["baseline_value_label"],
                    birnn_hidden["best"]["value_label"],
                ),
                styles,
            )
        )

    bigru_hidden = group_map.get(("bigru", "hidden_dim"))
    if bigru_hidden is not None:
        blocks.append(
            paragraph(
                "BiGRU 这组结果反而很有代表性：hidden_dim 的最优值就是 baseline 的 {}，两边无论降到 64 还是升到 192，验证集 F1 都下降。这通常说明当前 GRU 的容量已经处在比较合适的位置。减小到 64 时，模型对上下文的压缩能力不够，容易漏掉情感细节，属于欠拟合；增大到 192 时，参数量和优化难度一起上升，但在当前训练预算和数据规模下并没有换来更好的泛化，反而出现收益回落，更接近“容量增加但没有被有效利用”。换句话说，门控机制已经让 GRU 在 128 维时具备较好的表达效率，因此这一轮调参更像是在告诉我们：对于 GRU，这个任务上“保持适中容量”比继续堆宽度更重要。".format(
                    bigru_hidden["best"]["value_label"],
                ),
                styles,
            )
        )

    blocks.append(
        paragraph(
            "综合来看，这些“最优值”并不是彼此独立的随机结果，而是共同反映了当前任务的数据特点和训练条件：影评情感线索以短局部模式为主，因此 CNN 更偏好较短卷积窗口；平均池化后的 MLP 容量不必太小，但正则不宜过重；普通 BiRNN 若不引入门控，就更依赖增加层数来补足表示能力；而 GRU 由于门控效率更高，反而不需要继续盲目扩宽。也要强调，单变量调参得到的是局部最优判断，而不是全局联合最优解，因此这些结论最适合指导下一轮更有针对性的联合调参，而不应被理解为放之四海皆准的固定规则。",
            styles,
        )
    )
    return blocks


def export_report_charts(
    asset_dir: Path,
    results: Sequence[Dict[str, object]],
    external_results: Sequence[Dict[str, object]] = None,
    tuning_parameter_groups: Sequence[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """把程序直接生成的图表额外导出成独立 PNG 文件。"""
    asset_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    for result in ordered_main_results(results):
        path = asset_dir / "{}_validation_metric_curve.png".format(result["model_name"])
        renderPM.drawToFile(validation_metric_curve_drawing(result, export_mode=True), str(path), fmt="PNG")
        exported["{}_validation_metric_curve".format(result["model_name"])] = path

    validation_path = asset_dir / "validation_f1_curve.png"
    renderPM.drawToFile(validation_f1_curve_drawing(results, export_mode=True), str(validation_path), fmt="PNG")
    exported["validation_f1_curve"] = validation_path

    metric_path = asset_dir / "test_metric_bar_chart.png"
    renderPM.drawToFile(test_metric_bar_chart(results, export_mode=True), str(metric_path), fmt="PNG")
    exported["test_metric_bar_chart"] = metric_path

    if external_results:
        external_path = asset_dir / "external_robustness_chart.png"
        renderPM.drawToFile(external_robustness_chart(external_results, export_mode=True), str(external_path), fmt="PNG")
        exported["external_robustness_chart"] = external_path

    if tuning_parameter_groups:
        tuning_dir = asset_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        for group in tuning_parameter_groups:
            path = tuning_dir / "{}_{}.png".format(group["family"], group["parameter_name"])
            renderPM.drawToFile(tuning_parameter_chart(group, export_mode=True), str(path), fmt="PNG")
            exported["{}_{}".format(group["family"], group["parameter_name"])] = path

    return exported


def pick_best(results: List[Dict[str, object]]) -> Dict[str, object]:
    """按测试集 F1 优先、Accuracy 次优先选出表现最好的模型。"""
    ordered = sorted(results, key=lambda item: (item["test"]["f1"], item["test"]["accuracy"]), reverse=True)
    return ordered[0]


def report_image(path: Path, width_cm: float = 16.2) -> RLImage:
    """按固定版心宽度把位图转成 ReportLab Image。"""
    with PILImage.open(path) as image:
        aspect_ratio = image.height / image.width
    width = width_cm * cm
    height = width * aspect_ratio
    return RLImage(str(path), width=width, height=height)


def diagram_block(caption: str, path: Path, explanation: str, styles) -> KeepTogether:
    """把图题、图片和对应文字说明打包成一个版式块。"""
    return KeepTogether(
        [
            report_image(path),
            Spacer(1, 0.08 * cm),
            paragraph(caption, styles, "CaptionCN"),
            Spacer(1, 0.12 * cm),
            paragraph(explanation, styles),
            Spacer(1, 0.18 * cm),
        ]
    )


def resolve_model_diagram_paths(asset_dir: Path) -> Dict[str, Path]:
    """优先使用用户提供的新结构图，缺失时回退到旧的自动绘图资源。"""
    if all(path.exists() for path in UPLOADED_MODEL_DIAGRAMS.values()):
        return dict(UPLOADED_MODEL_DIAGRAMS)

    fallback = ensure_diagram_assets(asset_dir)
    return {
        "mlp": fallback["mlp"],
        "cnn": fallback["cnn"],
        "rnn_family": fallback["bigru"],
    }


def model_component_table(rows, styles) -> Table:
    """把结构图中的编号说明整理成统一样式的表格。"""
    table_rows = [["对应编号", "模块", "功能说明（统一以样本“1 你好”为例）"]]
    for step, module, description in rows:
        table_rows.append(
            [
                paragraph(step, styles, "SmallCN"),
                paragraph(module, styles, "SmallCN"),
                paragraph(description, styles, "SmallCN"),
            ]
        )

    table = Table(table_rows, colWidths=[1.9 * cm, 3.2 * cm, 11.1 * cm])
    apply_academic_table_style(table, font_size=8.2, align="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def mlp_component_table(styles) -> Table:
    """MLP 结构图对应的功能说明。"""
    return model_component_table(
        [
            (
                "1",
                "分词文本样本",
                "原始样本“1 你好”里的 1 是真实标签，表示正类；真正送入模型计算的文本只有“你好”。如果分词结果把“你好”视为一个 token，那么当前样本的有效 token 序列就是 [你好]。",
            ),
            (
                "2-3",
                "ID 序列 + 长度向量",
                "词表先把“你好”映射成 id(你好)，再按最大长度 80 补齐为 [id(你好), 0, 0, ..., 0]；同时记录真实长度 lengths=1。这里的 0 是 PAD，用来补齐 batch 内不同长度的句子。",
            ),
            (
                "4",
                "词嵌入层",
                "Embedding 把离散编号变成 50 维连续向量，于是第 1 个位置得到 e(你好)，后面 79 个 PAD 位置都对应 padding 向量 0，整个张量尺寸变成 [B,80,50]。",
            ),
            (
                "5-8",
                "Mask -> 求和 -> 归一化 -> 平均句向量",
                "mask 只保留真实 token，对本样本就是 [1,0,0,...,0]。masked sum 之后只剩下 e(你好)；再除以真实长度 1，就得到平均池化后的句向量。若句子更长，这一步就是把所有有效词向量做 masked mean pooling。",
            ),
            (
                "9-12",
                "Dropout -> Linear(50,128) -> ReLU -> Dropout",
                "第一层 dropout 只在训练时生效，用来随机屏蔽部分维度，减少过拟合；Linear(50,128) 把 50 维句向量投影到 128 维隐藏空间；ReLU 保留有效正激活；第二个 dropout 继续约束分类头不要过度依赖少数特征。",
            ),
            (
                "13-15",
                "输出层 -> Logits -> Argmax",
                "Linear(128,2) 输出两个类别分数 [z0,z1]。这两个值是 logits，不是概率；如果 z1 > z0，argmax 就会把样本预测成标签 1。若需要概率解释，还要再经过 softmax。",
            ),
            (
                "训练分支",
                "CrossEntropyLoss -> 反向传播 -> AdamW",
                "训练时把 logits 和真实标签 1 一起送入交叉熵损失；若模型已经把“你好”判得更偏向正类，损失就更小。随后反向传播把误差传回 Embedding 和两层全连接层，AdamW 负责更新参数。",
            ),
        ],
        styles,
    )


def textcnn_component_table(styles) -> Table:
    """TextCNN 结构图对应的功能说明。"""
    return model_component_table(
        [
            (
                "1-4",
                "样本解析 -> ID -> 长度 -> Embedding",
                "“1 你好”会先拆成标签 1 和文本“你好”。文本部分映射为 [id(你好), 0, ..., 0]，真实长度为 1，再被 Embedding 映射成 [B,80,50] 的词向量序列；此时只有第一个时间步是真实词向量，其余位置都是 PAD 向量。",
            ),
            (
                "5",
                "Unsqueeze",
                "TextCNN 的卷积层使用 Conv2d，因此需要把嵌入张量从 [B,80,50] 扩成 [B,1,80,50]。新增的这一维可以理解为“单通道文本图像”的通道维。",
            ),
            (
                "6",
                "三路并行卷积模块",
                "三组卷积核尺寸分别是 (3,50)、(4,50)、(5,50)，每组 128 个 filter。它们沿时间轴滑动，一次覆盖 3、4、5 个相邻 token，并把整条 50 维词向量一起看进去。对“你好”这种短句来说，窗口里会同时看到真实词和 PAD，模型仍可学习“句首出现正向词”这一局部模式。",
            ),
            (
                "6A-6C",
                "ReLU + 时间维 Max Pooling",
                "每一路卷积输出都会先过 ReLU，保留有效激活，再在时间维做最大池化。最大池化的含义是：不管某个情感模式出现在句子哪个位置，只保留这一路 filter 在整句上的最强响应值，因此每一路最终压缩成 128 维向量。",
            ),
            (
                "7",
                "Concat",
                "三路池化结果分别是 [B,128]，拼接后得到 [B,384]。这一步把 3-gram、4-gram、5-gram 三种局部情感证据放到同一个句向量里，让分类器同时参考不同长度的短语模式。",
            ),
            (
                "8-10",
                "Dropout -> 全连接分类层 -> Logits",
                "Dropout 在训练时随机丢弃部分卷积特征，防止模型只记住训练集里的固定搭配；Linear(384,2) 再把拼接后的局部模式映射成两个类别分数。输出的 [z0,z1] 仍然是 logits，而不是概率。",
            ),
            (
                "12",
                "最终预测",
                "如果 z1 大于 z0，argmax 预测为标签 1；否则预测为标签 0。对样本“1 你好”来说，理想情况是与“你好”相关的正向卷积核响应更强，使 z1 高于 z0。",
            ),
            (
                "训练分支",
                "损失与参数更新",
                "训练时同样使用 CrossEntropyLoss 比较 logits 与真实标签 1，再通过反向传播同时更新词向量、三路卷积核和输出层参数。也就是说，TextCNN 学到的不是词频统计，而是“哪些局部 n-gram 模式最能区分正负情感”。",
            ),
        ],
        styles,
    )


def recurrent_family_component_table(styles) -> Table:
    """BiRNN / BiLSTM / BiGRU 结构图对应的功能说明。"""
    return model_component_table(
        [
            (
                "1-4",
                "样本解析 -> ID -> 长度 -> Embedding",
                "“1 你好”同样会被处理成 [id(你好), 0, ..., 0]，真实长度 lengths=1，嵌入后得到 [B,80,50]。这一段和 MLP、TextCNN 一样，负责把离散文本变成后续编码器可处理的连续向量序列。",
            ),
            (
                "5",
                "双向序列编码",
                "双向编码器会从左到右和从右到左各跑一遍。对这个样本来说，只有第一个位置是真实 token，所以前向和后向状态都只会在“你好”这一位置更新一次；后面 79 个 PAD 位置会被长度掩码跳过，不再改写隐藏状态。",
            ),
            (
                "6",
                "最终前向/后向隐藏状态",
                "前向隐藏状态表示“从句首读到最后一个有效 token 后”的摘要，后向隐藏状态表示“从句尾反向读到第一个有效 token 后”的摘要。对长度为 1 的样本，两者都围绕“你好”本身形成表征；对长句，两者则分别保留左右文信息。",
            ),
            (
                "7-9",
                "Concat -> Dropout -> Linear(256,2)",
                "模型把前向 128 维和后向 128 维状态拼成 256 维句向量，再经过 dropout 和线性层输出两个类别分数。若与“你好”相关的正向上下文特征更强，最终就会得到 z1 > z0。",
            ),
            (
                "10-11",
                "Logits -> 最终预测",
                "线性层输出的是 logits [z0,z1]，argmax 才是最终类别。训练时仍然通过交叉熵把预测结果拉向真实标签 1。",
            ),
            (
                "模型差异",
                "BiRNN / BiLSTM / BiGRU 的不同点",
                "三者的外层流程完全一致，差别只在第 5 步循环单元内部。BiRNN 使用最基础的 tanh 状态更新，结构最简单但长距离记忆最弱；BiLSTM 额外维护 cell state，并通过输入门、遗忘门、输出门控制信息流，更适合保留长期依赖；BiGRU 用更新门和重置门做较轻量的门控，在参数量和效果之间取得折中。",
            ),
            (
                "训练分支",
                "损失与参数更新",
                "训练阶段把 logits 与标签 1 送入 CrossEntropyLoss，误差会反向传播到双向循环单元和词向量。由于循环网络按时间步更新状态，它比 MLP 和 CNN 更强调“前后文是如何逐步传递过来的”。",
            ),
        ],
        styles,
    )


def cnn_probability_example_table(styles) -> Table:
    """给出与结构图一致的单样本数值化前向传播示例。"""
    cell_style = styles["SmallCN"]
    rows = [
        ["阶段", "按照结构图展开的计算过程"],
        [
            paragraph("1. 输入样本", styles, "SmallCN"),
            paragraph(
                "样本写作“1 你好”时，前面的 1 是真实标签，表示正类；真正输入 TextCNN 的文本只有“你好”。为了演示计算过程，假设句长固定为 5，因此输入序列补齐为 [你好, PAD, PAD, PAD, PAD]。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("2. 词嵌入层", styles, "SmallCN"),
            paragraph(
                "为方便手算，把真实实验中的 50 维词向量缩成 2 维：你好 -> [0.8, 0.1]，PAD -> [0, 0]。于是输入矩阵可以写成 x1=[0.8,0.1]，x2=x3=x4=x5=[0,0]。这一步对应结构图中的 Embedding，把离散词编号映射成连续向量。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("3. 卷积层", styles, "SmallCN"),
            paragraph(
                "假设取一个窗口大小为 3 的卷积核 W=[[0.6,0.2],[0.1,-0.1],[0,0]]，偏置 b=-0.1。第一个滑动窗口是 [x1,x2,x3]=[你好,PAD,PAD]，因此卷积得分 s1=<W,[x1;x2;x3]>+b = 0.6×0.8 + 0.2×0.1 - 0.1 = 0.40。后两个窗口几乎只含 PAD，得分可近似看成 0。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("4. ReLU 激活", styles, "SmallCN"),
            paragraph(
                "卷积输出经过 ReLU：a1=max(0,0.40)=0.40，a2=max(0,0)=0，a3=max(0,0)=0。于是这个卷积核在整句话上的响应序列为 [0.40, 0, 0]。ReLU 的作用是保留有效激活，同时抑制负值噪声。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("5. 最大池化", styles, "SmallCN"),
            paragraph(
                "时间维最大池化得到 m=max(0.40,0,0)=0.40，表示该卷积核在整句话中检测到的最强局部情感证据强度是 0.40。若再假设另一个卷积核经过同样过程得到 0.20，则池化后句子特征可写成 h=[0.40,0.20]。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("6. 全连接层", styles, "SmallCN"),
            paragraph(
                "设输出层对负类和正类的权重分别为 w_neg=[-1.0,0.0]，w_pos=[2.0,0.0]，偏置都为 0。则 z_neg=w_neg·h=-1.0×0.40+0.0×0.20=-0.4；z_pos=w_pos·h=2.0×0.40+0.0×0.20=0.8。这里 z_neg 与 z_pos 是 logits，即未归一化的类别分数。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("7. Softmax 概率", styles, "SmallCN"),
            paragraph(
                "将 logits 变为概率：P(正类|x)=exp(0.8)/(exp(0.8)+exp(-0.4))≈0.7685，P(负类|x)=exp(-0.4)/(exp(0.8)+exp(-0.4))≈0.2315。因此模型会预测该样本为正类，这一步就是结构图中 Output 所对应的概率输出。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("8. 交叉熵损失", styles, "SmallCN"),
            paragraph(
                "由于真实标签是 1，训练时使用交叉熵 L=-log P(正类|x)=-log(0.7685)≈0.2633。若模型给正类的概率越高，这个损失就越小；反向传播会继续调整词向量、卷积核和全连接层参数，让类似正类样本的概率更大。",
                styles,
                "SmallCN",
            ),
        ],
    ]
    table = Table(rows, colWidths=[2.5 * cm, 13.3 * cm])
    apply_academic_table_style(table, font_size=8.3, align="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def recurrent_hello_example_table(styles) -> Table:
    """把“你好”的 RNN/GRU 手算过程整理成表格，便于报告展示。"""
    rows = [
        ["模型/步骤", "手算过程"],
        [
            paragraph("0. 输入设定", styles, "SmallCN"),
            paragraph(
                "句子写作“1 你好”时，前面的 1 是真实标签，真正输入循环模型的文本只有“你好”。为方便手算，把词向量和隐藏状态都压缩成 1 维，设“你”的词向量 x1=0.8，“好”的词向量 x2=0.4，初始隐藏状态 h0=0。真实代码里这些量是向量和矩阵，这里只是把同样的计算逻辑缩成标量。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("1. RNN 读入“你”", styles, "SmallCN"),
            paragraph(
                "普通 RNN 使用 h_t = tanh(Wx x_t + Wh h_(t-1) + b)。若取 Wx=1.0、Wh=0.5、b=0，则 h1 = tanh(1.0 x 0.8 + 0.5 x 0) = tanh(0.8) ≈ 0.664。此时 h1 表示“读完‘你’之后”的状态。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("2. RNN 读入“好”", styles, "SmallCN"),
            paragraph(
                "继续更新得到 h2 = tanh(1.0 x 0.4 + 0.5 x 0.664) = tanh(0.732) ≈ 0.624。因为 h2 同时依赖当前输入 x2 和上一步状态 h1，所以模型得到的是“你好”的整体表征，而不是单独看“好”这个字。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("3. GRU 读入“你”", styles, "SmallCN"),
            paragraph(
                "GRU 在 RNN 基础上增加更新门 z_t 和重置门 r_t。若设 Wz=0.5、Uz=0.5、bz=-0.2，Wr=0.5、Ur=0.5、br=0，Wn=1.0、Un=0.5、bn=0，则 z1=sigmoid(0.2)≈0.550，r1=sigmoid(0.4)≈0.599，n1=tanh(0.8)≈0.664，最终 h1=(1-z1)n1+z1h0≈0.299。也就是说，GRU 第一时刻不会把新信息整块写入，而是先经过门控筛选。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("4. GRU 读入“好”", styles, "SmallCN"),
            paragraph(
                "第二步有 z2=sigmoid(0.1495)≈0.537，r2=sigmoid(0.3495)≈0.586，n2=tanh(0.4 + 0.586 x 0.1495)≈0.452，最终 h2=(1-z2)n2+z2h1≈0.370。这里 z2≈0.537，表示第二步大约保留了 53.7% 的旧状态，只用 46.3% 的比例接收新的候选信息，因此 GRU 比普通 RNN 更擅长保留上下文。",
                styles,
                "SmallCN",
            ),
        ],
        [
            paragraph("5. 与 LSTM 的关系", styles, "SmallCN"),
            paragraph(
                "LSTM 的思想与 GRU 相同，也是通过门控控制信息保留，只是它额外维护独立的记忆单元 c_t，并把“写入、遗忘、输出”拆成更多门，因此长期记忆能力通常更强，但结构也更复杂。",
                styles,
                "SmallCN",
            ),
        ],
    ]
    table = Table(rows, colWidths=[2.8 * cm, 13.0 * cm])
    apply_academic_table_style(table, font_size=8.3, align="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_model_diagram_section(asset_dir: Path, styles) -> List:
    """生成“模型结构图与说明”章节所需的图片与说明块。"""
    diagram_paths = resolve_model_diagram_paths(asset_dir)
    return [
        paragraph(
            "本节直接给出课程作业要求的模型结构流程图，并按“输入准备 -> 特征提取 -> 分类输出 -> 训练更新”的顺序解释每个组成部分。下面统一用样本“1 你好”举例：其中 1 是真实标签，表示正类；真正进入模型计算的文本是“你好”，若它被分词成一个 token，则长度向量为 1，后续不足 80 的位置都用 PAD=0 补齐。",
            styles,
        ),
        paragraph("1.1 MLP 结构说明", styles, "SubHeadingCN"),
        diagram_block(
            "图1  MLP 情感分类模型流程图",
            diagram_paths["mlp"],
            "这张图对应的是“先做 masked mean pooling，再做浅层分类”的 baseline。它的重点不是保留词序，而是把有效 token 的词向量平均成句向量，然后交给两层感知机做二分类，因此结构最简单、训练速度也最快。",
            styles,
        ),
        mlp_component_table(styles),
        Spacer(1, 0.18 * cm),
        paragraph("1.2 TextCNN 结构说明", styles, "SubHeadingCN"),
        diagram_block(
            "图2  TextCNN 情感分类模型流程图",
            diagram_paths["cnn"],
            "TextCNN 的关键在于“多尺度局部模式抽取”。它并行使用 3、4、5 三种卷积核，分别去抓 3-gram、4-gram、5-gram 级别的情感触发短语，再把三路最强响应拼成句向量，因此对“非常 喜欢”“一点 也 不 好”这类局部搭配尤其敏感。",
            styles,
        ),
        paragraph("表1  TextCNN 各部分功能与“1 你好”示例", styles, "CaptionCN"),
        textcnn_component_table(styles),
        Spacer(1, 0.18 * cm),
        paragraph("1.3 BiRNN / BiLSTM / BiGRU 结构说明", styles, "SubHeadingCN"),
        diagram_block(
            "图3  BiRNN / BiLSTM / BiGRU 情感分类模型流程图",
            diagram_paths["rnn_family"],
            "这三种循环模型共用同一套外层流程：先做词嵌入，再由双向编码器读取上下文，最后拼接前向和后向状态做分类。它们真正的差别只在循环单元内部的信息更新方式，因此适合放在一张总图里对照说明。",
            styles,
        ),
        paragraph("表2  双向循环模型各部分功能与“1 你好”示例", styles, "CaptionCN"),
        recurrent_family_component_table(styles),
        Spacer(1, 0.18 * cm),
        paragraph(
            "综合来看，三类结构的主要区别不在“有没有用到词向量”，而在“怎样把词向量组织成句向量”：MLP 做平均，最省计算但不保留词序；TextCNN 抓局部 n-gram，擅长识别短语级情感模式；双向循环模型按时间步传递状态，最能体现上下文顺序与长期依赖。训练阶段它们都统一输出 logits，并通过交叉熵损失、反向传播与 AdamW 优化器完成参数更新。",
            styles,
        ),
    ]
