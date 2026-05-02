"""报告中模型结构图的位图绘制逻辑。"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


FONT_CANDIDATES = [
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]


def resolve_diagram_font(size: int):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str) -> None:
    width, height = text_size(draw, text, font)
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)


def draw_box(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    title: str,
    lines: Sequence[str],
    fill: str,
    border: str = "#3C3C3C",
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=28, fill=fill, outline=border, width=4)
    draw.line((x0 + 18, y0 + 58, x1 - 18, y0 + 58), fill=border, width=2)

    title_font = resolve_diagram_font(30)
    body_font = resolve_diagram_font(22)
    draw_centered_text(draw, (x0 + x1) // 2, y0 + 28, title, title_font, "#1F2937")

    current_y = y0 + 84
    for line in lines:
        draw_centered_text(draw, (x0 + x1) // 2, current_y, line, body_font, "#374151")
        current_y += 34


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: str = "#6B7280",
    width: int = 6,
    label: Optional[str] = None,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    head = 18
    if abs(ex - sx) >= abs(ey - sy):
        if ex >= sx:
            arrow = [(ex, ey), (ex - head, ey - 10), (ex - head, ey + 10)]
        else:
            arrow = [(ex, ey), (ex + head, ey - 10), (ex + head, ey + 10)]
    else:
        if ey >= sy:
            arrow = [(ex, ey), (ex - 10, ey - head), (ex + 10, ey - head)]
        else:
            arrow = [(ex, ey), (ex - 10, ey + head), (ex + 10, ey + head)]
    draw.polygon(arrow, fill=color)
    if label:
        font = resolve_diagram_font(20)
        mx = (sx + ex) // 2
        my = (sy + ey) // 2 - 24
        tw, th = text_size(draw, label, font)
        pad = 8
        draw.rounded_rectangle(
            (mx - tw / 2 - pad, my - th / 2 - pad, mx + tw / 2 + pad, my + th / 2 + pad),
            radius=12,
            fill="#FFF9E8",
            outline="#D4B483",
            width=2,
        )
        draw_centered_text(draw, mx, my, label, font, "#8C5A20")


def init_canvas(size: Tuple[int, int], title: str, subtitle: str) -> Tuple[PILImage.Image, ImageDraw.ImageDraw]:
    image = PILImage.new("RGB", size, "#FAF8F3")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, size[0] - 16, size[1] - 16), radius=34, fill="#FFFDF9", outline="#E6DECF", width=3)
    title_font = resolve_diagram_font(40)
    subtitle_font = resolve_diagram_font(24)
    draw_centered_text(draw, size[0] // 2, 54, title, title_font, "#1F2937")
    draw_centered_text(draw, size[0] // 2, 98, subtitle, subtitle_font, "#6B7280")
    return image, draw


def save_mlp_diagram(path: Path) -> None:
    image, draw = init_canvas((1600, 500), "MLP Baseline", "Mean pooling sentence representation + shallow classifier")
    boxes = [
        ((60, 165, 300, 355), "Input", ["Tokenized review", "length <= 80"], "#FDE68A"),
        ((350, 165, 610, 355), "Embedding", ["80 x 50", "pretrained, trainable"], "#C7E9B4"),
        ((660, 165, 970, 355), "Masked Mean Pooling", ["ignore PAD", "80 x 50 -> 50"], "#BFDBFE"),
        ((1020, 145, 1335, 375), "Classifier Block", ["Dropout(0.3)", "Linear 50 -> 128", "ReLU + Dropout", "Linear 128 -> 2"], "#FBCFE8"),
        ((1385, 165, 1540, 355), "Output", ["2 logits", "pos / neg"], "#DDD6FE"),
    ]
    for rect, title, lines, fill in boxes:
        draw_box(draw, rect, title, lines, fill)

    draw_arrow(draw, (300, 260), (350, 260), label="lookup")
    draw_arrow(draw, (610, 260), (660, 260))
    draw_arrow(draw, (970, 260), (1020, 260))
    draw_arrow(draw, (1335, 260), (1385, 260))

    note_font = resolve_diagram_font(21)
    note = "Key idea: average all valid token embeddings, then classify the sentence vector."
    draw_centered_text(draw, 800, 440, note, note_font, "#7C6750")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_cnn_diagram(path: Path) -> None:
    image, draw = init_canvas((1600, 720), "TextCNN", "Parallel convolution kernels capture local n-gram sentiment patterns")
    draw_box(draw, (60, 250, 280, 440), "Input", ["Tokenized review", "length <= 80"], "#FDE68A")
    draw_box(draw, (340, 250, 590, 440), "Embedding", ["80 x 50", "pretrained, trainable"], "#C7E9B4")

    conv_boxes = [
        ((690, 100, 980, 250), "Conv Branch 1", ["kernel = 3", "128 filters", "ReLU"], "#BFDBFE"),
        ((690, 285, 980, 435), "Conv Branch 2", ["kernel = 4", "128 filters", "ReLU"], "#BFDBFE"),
        ((690, 470, 980, 620), "Conv Branch 3", ["kernel = 5", "128 filters", "ReLU"], "#BFDBFE"),
    ]
    pool_boxes = [
        ((1060, 100, 1290, 250), "Max Pool", ["time-wise max", "128-d"], "#FCD5CE"),
        ((1060, 285, 1290, 435), "Max Pool", ["time-wise max", "128-d"], "#FCD5CE"),
        ((1060, 470, 1290, 620), "Max Pool", ["time-wise max", "128-d"], "#FCD5CE"),
    ]
    for rect, title, lines, fill in conv_boxes + pool_boxes:
        draw_box(draw, rect, title, lines, fill)

    draw_box(draw, (1360, 205, 1540, 355), "Concat", ["128 x 3", "384-d"], "#DDD6FE")
    draw_box(draw, (1360, 410, 1540, 560), "Classifier", ["Dropout(0.5)", "Linear 384 -> 2"], "#FBCFE8")

    draw_arrow(draw, (280, 345), (340, 345), label="lookup")
    draw_arrow(draw, (590, 345), (650, 345))
    draw.line((650, 345, 650, 175), fill="#6B7280", width=6)
    draw.line((650, 345, 650, 360), fill="#6B7280", width=6)
    draw.line((650, 345, 650, 545), fill="#6B7280", width=6)
    draw_arrow(draw, (650, 175), (690, 175))
    draw_arrow(draw, (650, 360), (690, 360))
    draw_arrow(draw, (650, 545), (690, 545))

    draw_arrow(draw, (980, 175), (1060, 175))
    draw_arrow(draw, (980, 360), (1060, 360))
    draw_arrow(draw, (980, 545), (1060, 545))

    draw.line((1290, 175, 1320, 175), fill="#6B7280", width=6)
    draw.line((1290, 360, 1320, 360), fill="#6B7280", width=6)
    draw.line((1290, 545, 1320, 545), fill="#6B7280", width=6)
    draw.line((1320, 175, 1320, 545), fill="#6B7280", width=6)
    draw_arrow(draw, (1320, 360), (1360, 280), label="concat")
    draw_arrow(draw, (1450, 355), (1450, 410))

    note_font = resolve_diagram_font(21)
    note = "Key idea: use multiple kernel sizes in parallel, keep the strongest local evidence from each filter."
    draw_centered_text(draw, 800, 670, note, note_font, "#7C6750")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_bigru_diagram(path: Path) -> None:
    image, draw = init_canvas((1600, 560), "BiGRU", "Bidirectional sequence encoder with forward and backward context")
    boxes = [
        ((60, 195, 285, 385), "Input", ["Tokenized review", "length <= 80"], "#FDE68A"),
        ((340, 195, 600, 385), "Embedding", ["80 x 50", "pretrained, trainable"], "#C7E9B4"),
        ((655, 195, 920, 385), "Packed Sequence", ["use true lengths", "skip PAD in GRU"], "#BFDBFE"),
        ((975, 175, 1250, 405), "Bidirectional GRU", ["hidden = 128", "forward + backward", "final states only"], "#FCD5CE"),
        ((1305, 175, 1540, 405), "Classifier", ["concat -> 256-d", "Dropout(0.3)", "Linear 256 -> 2"], "#DDD6FE"),
    ]
    for rect, title, lines, fill in boxes:
        draw_box(draw, rect, title, lines, fill)

    draw_arrow(draw, (285, 290), (340, 290), label="lookup")
    draw_arrow(draw, (600, 290), (655, 290))
    draw_arrow(draw, (920, 290), (975, 290), label="pack")
    draw_arrow(draw, (1250, 290), (1305, 290), label="h_f ⊕ h_b")

    note_font = resolve_diagram_font(21)
    note = "Key idea: encode context from both directions and concatenate the last forward/backward hidden states."
    draw_centered_text(draw, 800, 500, note, note_font, "#7C6750")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def ensure_diagram_assets(asset_dir: Path) -> Dict[str, Path]:
    """确保报告所需的三张结构图位图存在。"""
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mlp": asset_dir / "mlp_architecture.png",
        "cnn": asset_dir / "textcnn_architecture.png",
        "bigru": asset_dir / "bigru_architecture.png",
    }
    save_mlp_diagram(paths["mlp"])
    save_cnn_diagram(paths["cnn"])
    save_bigru_diagram(paths["bigru"])
    return paths
