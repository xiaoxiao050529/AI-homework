"""项目目录与常用文件路径常量。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TOOLS_DIR = PROJECT_ROOT / "tools"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_TMUX_DIR = OUTPUT_DIR / "tmux"
OUTPUT_TMUX_LOG_DIR = OUTPUT_TMUX_DIR / "logs"
OUTPUT_TMUX_META_DIR = OUTPUT_TMUX_DIR / "sessions"


def _prefer_existing(*candidates: Path) -> Path:
    """优先返回已存在的路径；若都不存在，则返回第一个候选。"""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


EXTERNAL_DATA_DIR = _prefer_existing(DATA_DIR / "external", PROJECT_ROOT / "external_data")

TRAIN_PATH = _prefer_existing(DATA_DIR / "train.txt", PROJECT_ROOT / "train.txt")
VALIDATION_PATH = _prefer_existing(DATA_DIR / "validation.txt", PROJECT_ROOT / "validation.txt")
TEST_PATH = _prefer_existing(DATA_DIR / "test.txt", PROJECT_ROOT / "test.txt")
EMBEDDING_PATH = _prefer_existing(DATA_DIR / "wiki_word2vec_50.bin", PROJECT_ROOT / "wiki_word2vec_50.bin")

EXTERNAL_WEIBO_TEXT_PATH = EXTERNAL_DATA_DIR / "weibo_sentiment_100.txt"
EXTERNAL_WEIBO_JSONL_PATH = EXTERNAL_DATA_DIR / "weibo_sentiment_100.jsonl"
EXTERNAL_WEIBO_META_PATH = EXTERNAL_DATA_DIR / "weibo_sentiment_100_meta.json"

REPORT_PATH = PROJECT_ROOT / "学号_姓名.pdf"
