"""在 tmux 会话中后台启动长实验，并把日志和会话信息落盘。"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.sentiment_hw2.paths import OUTPUT_TMUX_LOG_DIR, OUTPUT_TMUX_META_DIR, PROJECT_ROOT


def shell_join(parts: List[str]) -> str:
    """兼容 Python 3.7 的 shell 参数拼接。"""
    return " ".join(shlex.quote(part) for part in parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 tmux 中后台运行命令，适合长时间实验"
    )
    parser.add_argument("session_name", help="tmux 会话名，例如 main_exp 或 improve_quick")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="若会话已存在，则直接返回对应信息而不是报错。",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="要在 tmux 中运行的命令，前面加 -- 以避免参数歧义。",
    )
    return parser


def ensure_tmux_available() -> None:
    if subprocess.run(["tmux", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
        raise RuntimeError("tmux 不可用，请先安装 tmux")


def normalize_command(command: List[str]) -> List[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("缺少要运行的命令，例如 -- .venv/bin/python -m scripts.run_experiments")
    return command


def session_exists(session_name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).returncode == 0


def build_runner_script(command: List[str], log_path: Path) -> str:
    quoted_command = shell_join(command)
    quoted_log_path = shlex.quote(str(log_path))
    return (
        "cd {root} && "
        "mkdir -p {log_dir} {meta_dir} && "
        "echo \"[$(date '+%F %T')] START {cmd}\" | tee -a {log} && "
        "{cmd} 2>&1 | tee -a {log}; "
        "status=${{PIPESTATUS[0]}}; "
        "echo \"[$(date '+%F %T')] EXIT_CODE $status\" | tee -a {log}; "
        "exec bash"
    ).format(
        root=shlex.quote(str(PROJECT_ROOT)),
        log_dir=shlex.quote(str(OUTPUT_TMUX_LOG_DIR)),
        meta_dir=shlex.quote(str(OUTPUT_TMUX_META_DIR)),
        cmd=quoted_command,
        log=quoted_log_path,
    )


def write_metadata(session_name: str, command: List[str], log_path: Path) -> Path:
    OUTPUT_TMUX_META_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTPUT_TMUX_META_DIR / "{}.json".format(session_name)
    payload: Dict[str, object] = {
        "session_name": session_name,
        "command": command,
        "command_display": shell_join(command),
        "log_path": str(log_path),
        "project_root": str(PROJECT_ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid_hint": None,
        "resume_note": "电脑关机后 tmux 会话不会保留；重启后请重新执行同一命令。",
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def create_session(session_name: str, command: List[str], log_path: Path) -> None:
    runner_script = build_runner_script(command, log_path)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", runner_script],
        check=True,
    )


def print_summary(session_name: str, log_path: Path, meta_path: Path) -> None:
    print("tmux session started:", session_name)
    print("attach:", "tmux attach -t {}".format(session_name))
    print("list:", "tmux ls")
    print("log:", log_path)
    print("meta:", meta_path)
    print("tail:", "tail -f {}".format(log_path))
    print("stop:", "tmux kill-session -t {}".format(session_name))
    print("note: tmux 只能防止终端断开，电脑关机后会话不会继续。")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_tmux_available()
    command = normalize_command(args.command)

    if session_exists(args.session_name):
        if not args.allow_existing:
            raise SystemExit("tmux 会话已存在：{}。如需复用，请加 --allow-existing".format(args.session_name))
        meta_path = OUTPUT_TMUX_META_DIR / "{}.json".format(args.session_name)
        print("tmux session already exists:", args.session_name)
        print("attach:", "tmux attach -t {}".format(args.session_name))
        if meta_path.exists():
            print("meta:", meta_path)
        raise SystemExit(0)

    OUTPUT_TMUX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_TMUX_LOG_DIR / "{}_{}.log".format(args.session_name, timestamp)
    meta_path = write_metadata(args.session_name, command, log_path)
    create_session(args.session_name, command, log_path)
    print_summary(args.session_name, log_path, meta_path)


if __name__ == "__main__":
    main()
