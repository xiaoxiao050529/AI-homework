"""并行运行 BiRNN / BiLSTM / BiGRU 的单变量调参，并在结束后统一汇总与生成报告。"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple


FAMILIES = ["birnn", "bilstm", "bigru"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recurrent tuning families in parallel.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke module entry points.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=4,
        help="Thread cap passed to each tuning worker process.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%F %T")


def worker_env(threads_per_worker: int) -> dict:
    env = os.environ.copy()
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = str(threads_per_worker)
    return env


def launch_family(python_bin: str, family: str, threads_per_worker: int) -> Tuple[str, subprocess.Popen]:
    command = [
        python_bin,
        "-m",
        "scripts.run_recurrent_tuning_pipeline",
        "--families",
        family,
    ]
    print("[{}] LAUNCH {} :: {}".format(timestamp(), family, " ".join(command)), flush=True)
    process = subprocess.Popen(command, env=worker_env(threads_per_worker))
    return family, process


def run_command(command: List[str], threads_per_worker: int) -> int:
    print("[{}] RUN {}".format(timestamp(), " ".join(command)), flush=True)
    result = subprocess.run(command, env=worker_env(threads_per_worker))
    print("[{}] EXIT {} :: {}".format(timestamp(), result.returncode, " ".join(command)), flush=True)
    return result.returncode


def main() -> None:
    args = parse_args()
    workers = [launch_family(args.python, family, args.threads_per_worker) for family in FAMILIES]

    failures = []
    for family, process in workers:
        return_code = process.wait()
        print("[{}] WORKER {} EXIT {}".format(timestamp(), family, return_code), flush=True)
        if return_code != 0:
            failures.append((family, return_code))

    if failures:
        print("[{}] FAILURES {}".format(timestamp(), failures), flush=True)
        raise SystemExit(1)

    summary_code = run_command(
        [
            args.python,
            "-m",
            "scripts.run_hyperparameter_tuning",
            "--reuse-existing-only",
            "--families",
            "birnn",
            "bilstm",
            "bigru",
        ],
        args.threads_per_worker,
    )
    if summary_code != 0:
        raise SystemExit(summary_code)

    report_code = run_command(
        [args.python, "-m", "scripts.generate_report"],
        args.threads_per_worker,
    )
    if report_code != 0:
        raise SystemExit(report_code)


if __name__ == "__main__":
    main()
