"""Inspect `.docx` and `.pt` files from the terminal.

Examples:
    python3 inspect_artifact.py 实验二说明文档.docx --output 实验二说明文档.txt
    python3 inspect_artifact.py outputs/mlp_best.pt --output outputs/mlp_best_summary.json
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree as ET

import torch


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_text(path: Path) -> str:
    """Extract readable text from a `.docx` file without extra dependencies."""
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", WORD_NS):
        runs = []
        for node in paragraph.findall(".//w:t", WORD_NS):
            runs.append(node.text or "")
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def summarize_object(obj: Any, depth: int = 0) -> Any:
    """Convert arbitrary loaded checkpoint content into a JSON-friendly summary."""
    if depth > 3:
        return str(type(obj).__name__)
    if isinstance(obj, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(key): summarize_object(value, depth + 1) for key, value in list(obj.items())[:200]}
    if isinstance(obj, (list, tuple)):
        return [summarize_object(item, depth + 1) for item in obj[:50]]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj).__name__)


def summarize_pt(path: Path) -> Dict[str, Any]:
    """Load a PyTorch `.pt` file and export a structural summary."""
    obj = torch.load(path, map_location="cpu")
    summary = {
        "path": str(path),
        "root_type": type(obj).__name__,
        "summary": summarize_object(obj),
    }
    if isinstance(obj, dict):
        tensor_count = 0
        top_level_keys = list(obj.keys())
        for value in obj.values():
            if isinstance(value, torch.Tensor):
                tensor_count += 1
        summary["top_level_keys"] = [str(key) for key in top_level_keys[:200]]
        summary["top_level_tensor_count"] = tensor_count
    return summary


def default_output_path(path: Path) -> Path:
    if path.suffix.lower() == ".docx":
        return path.with_suffix(".txt")
    if path.suffix.lower() == ".pt":
        return path.with_name(path.stem + "_summary.json")
    return path.with_suffix(path.suffix + ".out")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .docx and .pt files")
    parser.add_argument("path", help="Target .docx or .pt file")
    parser.add_argument("--output", help="Optional output path")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    output_path = Path(args.output).resolve() if args.output else default_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = extract_docx_text(path)
        output_path.write_text(text, encoding="utf-8")
        print("Extracted DOCX text to", output_path)
        return
    if suffix == ".pt":
        summary = summarize_pt(path)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Extracted PT summary to", output_path)
        return

    raise ValueError("Only .docx and .pt are supported")


if __name__ == "__main__":
    main()
