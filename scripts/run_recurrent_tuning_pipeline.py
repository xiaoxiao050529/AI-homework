"""按“一个参数一个参数”的方式顺序执行三类 RNN 单变量调参。"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List


FULL_PARAMETER_ORDER: Dict[str, List[str]] = {
    "birnn": [
        "hidden_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "embedding_learning_rate",
        "weight_decay",
        "grad_clip",
        "warmup_epochs",
        "label_smoothing",
        "freeze_embedding_epochs",
        "epochs",
        "patience",
        "seed",
    ],
    "bilstm": [
        "hidden_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "embedding_learning_rate",
        "weight_decay",
        "grad_clip",
        "warmup_epochs",
        "label_smoothing",
        "freeze_embedding_epochs",
        "epochs",
        "patience",
        "seed",
    ],
    "bigru": [
        "hidden_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "embedding_learning_rate",
        "weight_decay",
        "grad_clip",
        "warmup_epochs",
        "label_smoothing",
        "freeze_embedding_epochs",
        "epochs",
        "patience",
        "seed",
    ],
}

CORE_PARAMETERS = [
    "hidden_dim",
    "num_layers",
    "dropout",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "patience",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recurrent tuning family by family, parameter by parameter.")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FULL_PARAMETER_ORDER.keys()),
        default=["birnn", "bilstm", "bigru"],
        help="Only run selected recurrent families.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke module entry points.",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Regenerate the final PDF after all tuning commands finish.",
    )
    parser.add_argument(
        "--generate-report-each-step",
        action="store_true",
        help="Refresh the PDF after each finished parameter group so progress is visible immediately.",
    )
    parser.add_argument(
        "--parameter-preset",
        choices=["full", "core"],
        default="full",
        help="Choose whether to run all recurrent tuning parameters or only the eight core parameters.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later parameters even if one command fails.",
    )
    return parser.parse_args()


def parameter_order(preset: str) -> Dict[str, List[str]]:
    """根据预设返回每个家族要跑的参数顺序。"""
    if preset == "core":
        return {family: [name for name in FULL_PARAMETER_ORDER[family] if name in CORE_PARAMETERS] for family in FULL_PARAMETER_ORDER}
    return FULL_PARAMETER_ORDER


def run_command(command: List[str]) -> int:
    started_at = datetime.now().strftime("%F %T")
    print("[{}] RUN {}".format(started_at, " ".join(command)), flush=True)
    start = time.time()
    completed = subprocess.run(command)
    elapsed = time.time() - start
    finished_at = datetime.now().strftime("%F %T")
    print(
        "[{}] EXIT {} ({:.1f}s)".format(
            finished_at,
            completed.returncode,
            elapsed,
        ),
        flush=True,
    )
    return completed.returncode


def main() -> None:
    args = parse_args()
    order = parameter_order(args.parameter_preset)
    total_steps = sum(len(order[family]) for family in args.families)
    step_index = 0

    for family in args.families:
        for parameter in order[family]:
            step_index += 1
            print(
                "[progress] {}/{} family={} parameter={}".format(
                    step_index,
                    total_steps,
                    family,
                    parameter,
                ),
                flush=True,
            )
            command = [
                args.python,
                "-m",
                "scripts.run_hyperparameter_tuning",
                "--families",
                family,
                "--parameters",
                parameter,
            ]
            return_code = run_command(command)
            if return_code != 0 and not args.continue_on_error:
                raise SystemExit(return_code)
            if args.generate_report_each_step and return_code == 0:
                report_code = run_command([args.python, "-m", "scripts.generate_report"])
                if report_code != 0 and not args.continue_on_error:
                    raise SystemExit(report_code)

    if args.generate_report:
        report_code = run_command([args.python, "-m", "scripts.generate_report"])
        if report_code != 0:
            raise SystemExit(report_code)


if __name__ == "__main__":
    main()
