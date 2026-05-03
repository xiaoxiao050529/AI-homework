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


ACTIVE_FONT_NAME = "STSong-Light"


def register_fonts() -> None:
    """注册中文字体，兼顾中文与正文中的英文、数字显示。"""
    global ACTIVE_FONT_NAME
    registerFont(UnicodeCIDFont("STSong-Light"))
    ACTIVE_FONT_NAME = "STSong-Light"


def build_styles():
    """定义整份课程论文报告复用的中文段落样式。"""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCN",
            parent=styles["Title"],
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
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
            fontName=ACTIVE_FONT_NAME,
            fontSize=11,
            leading=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCN",
            parent=styles["BodyText"],
            fontName=ACTIVE_FONT_NAME,
            fontSize=9,
            leading=13,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReferenceCN",
            parent=styles["BodyText"],
            fontName=ACTIVE_FONT_NAME,
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
        "core": "hidden_dim=128, num_layers=1, dropout=0.3",
        "train": "batch=64, epochs=14, lr=3e-4, emb_lr=8e-5, wd=3e-4, patience=4, clip=1, warmup=2",
    },
    "bilstm": {
        "core": "hidden_dim=128, num_layers=1, dropout=0.3",
        "train": "batch=64, epochs=14, lr=3e-4, emb_lr=8e-5, wd=3e-4, patience=4, clip=1, warmup=2",
    },
    "bigru": {
        "core": "hidden_dim=128, num_layers=1, dropout=0.3",
        "train": "batch=64, epochs=14, lr=3e-4, emb_lr=8e-5, wd=3e-4, patience=4, clip=1, warmup=2",
    },
}

MAIN_MODEL_ORDER = ["mlp", "cnn", "birnn", "bilstm", "bigru"]
CHART_COLORS = [
    colors.HexColor("#2563EB"),
    colors.HexColor("#DC2626"),
    colors.HexColor("#059669"),
    colors.HexColor("#D97706"),
    colors.HexColor("#7C3AED"),
]
CHART_MARKERS = ["FilledCircle", "FilledSquare", "FilledDiamond", "Cross", "Circle"]


def ordered_main_results(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按主实验统一顺序排列结果，避免图表次序前后不一致。"""
    rank = {name: idx for idx, name in enumerate(MAIN_MODEL_ORDER)}
    return sorted(results, key=lambda item: rank.get(item["model_name"], len(rank)))


def add_legend(
    drawing: Drawing,
    x: float,
    y: float,
    entries: Sequence[tuple],
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
    legend.fontName = ACTIVE_FONT_NAME
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
                ("FONTNAME", (0, 0), (-1, -1), ACTIVE_FONT_NAME),
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
        fontName=ACTIVE_FONT_NAME,
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


def loss_curve_drawing(result: Dict[str, object], width: float = 15.8 * cm, height: float = 6.0 * cm) -> Drawing:
    """把单个模型的训练损失和验证损失画成曲线。"""
    history = result["history"]
    train_points = [(record["epoch"], record["train_loss"]) for record in history]
    val_points = [(record["epoch"], record["validation_loss"]) for record in history]
    epochs = [record["epoch"] for record in history]
    y_max = max(max(value for _, value in train_points), max(value for _, value in val_points))

    drawing = Drawing(width, height + 14)
    chart = LinePlot()
    chart.x = 42
    chart.y = 18
    chart.width = width - 74
    chart.height = height - 10
    chart.data = [train_points, val_points]
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
    chart.yValueAxis.valueMin = 0
    chart.yValueAxis.valueMax = y_max * 1.08
    chart.yValueAxis.valueStep = max(0.05, round(y_max / 5.0, 2))
    drawing.add(chart)
    add_legend(
        drawing,
        x=width - 110,
        y=height + 1,
        entries=[
            (colors.HexColor("#2563EB"), "训练集损失"),
            (colors.HexColor("#DC2626"), "验证集损失"),
        ],
        column_maximum=2,
        deltax=44,
        deltay=10,
    )
    return drawing


def validation_f1_curve_drawing(
    results: Sequence[Dict[str, object]],
    width: float = 15.8 * cm,
    height: float = 6.0 * cm,
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
) -> Drawing:
    """生成两组指标并列柱状图。"""
    drawing = Drawing(width, height + 16)
    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 20
    chart.width = width - 126
    chart.height = height - 6
    chart.data = [list(values) for values in series]
    chart.groupSpacing = 12
    chart.barSpacing = 3
    chart.categoryAxis.categoryNames = list(category_names)
    chart.categoryAxis.labels.fontName = ACTIVE_FONT_NAME
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.dy = -8
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
        column_maximum=2,
        deltax=36,
        deltay=10,
    )
    drawing.add(String(42, 4, footnote, fontName=ACTIVE_FONT_NAME, fontSize=7.2))
    return drawing


def test_metric_bar_chart(results: Sequence[Dict[str, object]]) -> Drawing:
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
    )


def external_robustness_chart(results: Sequence[Dict[str, object]]) -> Drawing:
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
    )


def robustness_table(summary: Dict[str, object]) -> Table:
    """把附加鲁棒性测试整理成横向对比表。"""
    ordered_models = [name for name in ["mlp", "cnn", "birnn", "bilstm", "bigru"] if name in summary["robustness"]]
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
    rows = [["超参数", "代表模型", "调整值", "测试集 Acc", "测试集 F1", "相对基线 ΔF1"]]
    for item in variants:
        spec = item["spec"]
        result = item["result"]
        rows.append(
            [
                spec["parameter_name"],
                model_label(spec["family"]),
                spec["parameter_value"],
                "{:.4f}".format(result["test"]["accuracy"]),
                "{:.4f}".format(result["test"]["f1"]),
                "{:+.4f}".format(result["delta_vs_baseline"]["test_f1"]),
            ]
        )
    table = Table(rows, colWidths=[2.3 * cm, 2.1 * cm, 2.0 * cm, 2.3 * cm, 2.3 * cm, 2.8 * cm])
    return apply_academic_table_style(table, font_size=8.5)


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
    figure_numbers = {"mlp": 4, "cnn": 5, "birnn": 6, "bilstm": 7, "bigru": 8}
    ordered_results = ordered_main_results(results)
    blocks: List = [
        paragraph(
            "为展示不同结构在训练阶段的收敛特征，图4至图9给出了各模型训练损失与验证集 F1 的变化过程。整体上，多数模型都在前几轮快速下降，随后逐步进入平台期，这与验证集早停策略的触发时机基本一致。",
            styles,
        )
    ]
    for result in ordered_results:
        blocks.append(loss_curve_drawing(result))
        blocks.append(Spacer(1, 0.06 * cm))
        blocks.append(
            paragraph(
                "图{}  {}训练集与验证集损失曲线".format(
                    figure_numbers[result["model_name"]],
                    model_label(result["model_name"]),
                ),
                styles,
                "CaptionCN",
            )
        )
        blocks.append(Spacer(1, 0.18 * cm))
    blocks.append(validation_f1_curve_drawing(ordered_results))
    blocks.append(Spacer(1, 0.06 * cm))
    blocks.append(paragraph("图9  各模型验证集 F1 收敛曲线", styles, "CaptionCN"))
    blocks.append(Spacer(1, 0.18 * cm))
    return blocks


def export_report_charts(
    asset_dir: Path,
    results: Sequence[Dict[str, object]],
    external_results: Sequence[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """把程序直接生成的图表额外导出成独立 PNG 文件。"""
    asset_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    for result in ordered_main_results(results):
        path = asset_dir / "{}_loss_curve.png".format(result["model_name"])
        renderPM.drawToFile(loss_curve_drawing(result), str(path), fmt="PNG")
        exported["{}_loss_curve".format(result["model_name"])] = path

    validation_path = asset_dir / "validation_f1_curve.png"
    renderPM.drawToFile(validation_f1_curve_drawing(results), str(validation_path), fmt="PNG")
    exported["validation_f1_curve"] = validation_path

    metric_path = asset_dir / "test_metric_bar_chart.png"
    renderPM.drawToFile(test_metric_bar_chart(results), str(metric_path), fmt="PNG")
    exported["test_metric_bar_chart"] = metric_path

    if external_results:
        external_path = asset_dir / "external_robustness_chart.png"
        renderPM.drawToFile(external_robustness_chart(external_results), str(external_path), fmt="PNG")
        exported["external_robustness_chart"] = external_path

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
    diagram_paths = ensure_diagram_assets(asset_dir)
    return [
        paragraph(
            "下图将主模型按结构统一绘制为“输入表示 - 特征提取 - 分类输出”的流程图，便于直接比较它们分别依赖的是平均语义、局部短语模式还是双向上下文信息。",
            styles,
        ),
        diagram_block(
            "图1  MLP 基线模型结构图",
            diagram_paths["mlp"],
            "MLP 基线模型先将长度不超过 80 的句子映射为 80 x 50 的可微调预训练词向量序列，再根据真实长度对非 PAD 位置做 masked mean pooling，压缩成 50 维句向量。分类头对应代码中的 `Dropout(0.3) -> Linear(50,128) -> ReLU -> Dropout(0.3) -> Linear(128,2)`，参数量较小、训练速度快，但会显式丢失词序信息。",
            styles,
        ),
        diagram_block(
            "图2  TextCNN 模型结构图",
            diagram_paths["cnn"],
            "TextCNN 在嵌入层后并行使用 3、4、5 三种卷积核尺寸，每种尺寸配置 128 个卷积核，对应不同长度的局部情感短语窗口。各卷积分支经 ReLU 和时间维最大池化后分别得到 128 维向量，拼接成 384 维句向量，再经过 `Dropout(0.5)` 和全连接层完成二分类，因此它尤其擅长提取“非常 失望”“节奏 混乱”这类局部触发模式。",
            styles,
        ),
        diagram_block(
            "图3  BiGRU 模型结构图",
            diagram_paths["bigru"],
            "BiGRU 先根据真实长度在每个时间步构造掩码，显式跳过补齐位置的状态更新；随后手写双向 GRU 从前向和后向同时编码上下文。分类时仅取最后一层的前向隐藏状态与后向隐藏状态拼接，得到 256 维句向量，并经过 `Dropout(0.3) -> Linear(256,2)` 输出结果。该结构保留了顺序与上下文依赖，但计算成本高于 CNN。",
            styles,
        ),
        paragraph("以 TextCNN 为例的单样本概率计算过程", styles, "SubHeadingCN"),
        paragraph(
            "下面给出一个与结构图顺序完全一致的数值化示例，目的是说明“输入样本 -> 卷积特征 -> logits -> softmax 概率”这一整条链路在概率统计层面究竟是如何计算出来的。这个例子为了便于手算，故意把真实实验中的 50 维词向量、多个卷积核和更高维的全连接层缩减成了极简版本，但计算逻辑与正式模型完全一致。",
            styles,
        ),
        paragraph("表1  TextCNN 单样本前向传播与概率计算示例", styles, "CaptionCN"),
        cnn_probability_example_table(styles),
        Spacer(1, 0.18 * cm),
        paragraph(
            "从这个例子可以看出，TextCNN 并不是直接统计“你好”在正类和负类中出现了多少次，而是通过卷积核先抽取局部模式，再把这些模式经过线性层变成 logits，最后用 softmax 构造条件概率 P(y|x)。训练阶段通过最小化交叉熵，不断调整词向量、卷积核和输出层参数，使正确类别的概率越来越大。",
            styles,
        ),
    ]
