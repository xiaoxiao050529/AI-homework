"""抓取并构造外部小规模鲁棒性测试集。"""

import csv
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import jieba


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "external_data"
TEXT_PATH = DATA_DIR / "weibo_sentiment_100.txt"
JSONL_PATH = DATA_DIR / "weibo_sentiment_100.jsonl"
META_PATH = DATA_DIR / "weibo_sentiment_100_meta.json"
SOURCE_REPO_DIR = Path("/tmp/weibo2018")

SOURCE_URL = "https://github.com/Neal-Bailey/weibo2018.git"
SOURCE_NAME = "weibo2018"
PER_LABEL = 50
SEED = 42

jieba.dt.tmp_dir = str(DATA_DIR)


def normalize_text(text: str) -> str:
    """统一清洗空白符，避免分词和保存时出现脏数据。"""
    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> List[str]:
    """使用 jieba 分词，并去掉空白 token。"""
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def ensure_source_repo() -> None:
    """把外部公开数据仓库克隆到本地缓存目录。"""
    if SOURCE_REPO_DIR.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_URL, str(SOURCE_REPO_DIR)],
        check=True,
    )


def download_rows() -> List[Dict[str, str]]:
    """读取本地缓存仓库中的微博情感数据。"""
    ensure_source_repo()
    rows: List[Dict[str, str]] = []
    for split_name in ("train.txt", "test.txt"):
        split_path = SOURCE_REPO_DIR / "weibo2018" / split_name
        with split_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 3:
                    continue
                label_text = row[1].strip()
                review = normalize_text(row[2])
                if label_text not in {"0", "1"} or not review:
                    continue
                rows.append({"label": label_text, "review": review})
    return rows


def build_samples(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """构造平衡的正负样本集合。"""
    grouped = {0: [], 1: []}
    for row in rows:
        label_text = str(row.get("label", "")).strip()
        review = normalize_text(str(row.get("review", "")))
        if label_text not in {"0", "1"} or not review:
            continue

        tokens = tokenize_text(review)
        if not tokens:
            continue

        label = int(label_text)
        grouped[label].append(
            {
                "label": label,
                "text": review,
                "tokens": tokens,
            }
        )

    rng = random.Random(SEED)
    samples = []
    for label in (0, 1):
        if len(grouped[label]) < PER_LABEL:
            raise ValueError("标签 {} 的有效样本不足 {}".format(label, PER_LABEL))
        chosen = rng.sample(grouped[label], PER_LABEL)
        samples.extend(chosen)

    rng.shuffle(samples)
    return samples


def save_samples(samples: List[Dict[str, object]]) -> None:
    """同时保存项目可直接读取的 txt 和便于分析的 jsonl。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with TEXT_PATH.open("w", encoding="utf-8") as text_handle:
        for sample in samples:
            text_handle.write("{} {}\n".format(sample["label"], " ".join(sample["tokens"])))

    with JSONL_PATH.open("w", encoding="utf-8") as jsonl_handle:
        for sample in samples:
            jsonl_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(samples),
        "per_label": PER_LABEL,
        "seed": SEED,
        "domain": "weibo_post",
        "task_note": "cross-domain robustness evaluation for Chinese sentiment classification",
        "files": {
            "txt": str(TEXT_PATH),
            "jsonl": str(JSONL_PATH),
        },
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rows = download_rows()
    samples = build_samples(rows)
    save_samples(samples)
    print("Saved {} samples to {}".format(len(samples), TEXT_PATH))
    print("Metadata saved to {}".format(META_PATH))


if __name__ == "__main__":
    main()
