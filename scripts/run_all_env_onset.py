from __future__ import annotations

"""
Run the envelope+onset backward mTRF model for multiple subjects.

This is the run-all file for the new model pipeline:

    preprocessed file:
        data/processed/sub-004_mtrf_env_onset.npz

    model file:
        src/aad_project/model_env_onset.py

It does NOT run preprocessing.
It assumes that the new preprocessing file has already created the combined
envelope+onset .npz files.

Place this file here:

    scripts/run_all_env_onset.py

Example
-------
From the project root:

python scripts/run_all_env_onset.py ^
  --processed-dir data/processed ^
  --results-dir results_env_onset ^
  --subjects 1 2 3 4 ^
  --max-workers 4

On Linux / HPC:

python scripts/run_all_env_onset.py \
  --processed-dir data/processed \
  --results-dir results_env_onset \
  --subjects 1 2 3 4 \
  --max-workers 4
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def subject_id(subject: int | str) -> str:
    """Return subject id in the form sub-004."""
    digits = "".join(ch for ch in str(subject) if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse subject id from {subject!r}")
    return f"sub-{int(digits):03d}"


def expected_input_file(processed_dir: Path, subject: int | str) -> Path:
    """
    The new preprocessing should save one combined file with a unique name,
    so it does not overwrite the baseline envelope-only file.
    """
    sid = subject_id(subject)
    return processed_dir / f"{sid}_mtrf_env_onset.npz"


def run_subject(
    subject: int,
    processed_dir: Path,
    results_dir: Path,
    python_executable: str,
    score_mode: str,
    error: str,
    overwrite: bool,
) -> tuple[int, bool, str]:
    """
    Run model_env_onset for one subject.

    Returns
    -------
    subject, success, message
    """
    sid = subject_id(subject)
    input_path = expected_input_file(processed_dir, subject)

    subject_results_dir = results_dir / sid
    subject_results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = subject_results_dir / f"{sid}_backward_mtrf_env_onset.summary.json"

    if not input_path.exists():
        return (
            subject,
            False,
            f"Missing preprocessing file for {sid}: {input_path}",
        )

    if summary_path.exists() and not overwrite:
        return (
            subject,
            True,
            f"Skipped {sid}: result already exists at {summary_path}",
        )

    cmd = [
        python_executable,
        "-m",
        "src.aad_project.model_env_onset",
        "--input",
        str(input_path),
        "--out-dir",
        str(subject_results_dir),
        "--subject",
        str(subject),
        "--score-mode",
        score_mode,
        "--error",
        error,
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        msg = completed.stdout.strip()
        if completed.stderr.strip():
            msg += "\nSTDERR:\n" + completed.stderr.strip()

        return subject, True, msg

    except subprocess.CalledProcessError as exc:
        msg = (
            f"Command failed for {sid}\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{exc.stdout}\n\n"
            f"STDERR:\n{exc.stderr}"
        )
        return subject, False, msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backward mTRF envelope+onset model for multiple subjects"
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing sub-XXX_mtrf_env_onset.npz files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory where model results should be saved",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject numbers, e.g. --subjects 1 2 3 4",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of subjects to process in parallel",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use. Defaults to current environment.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["mean", "weighted", "env_only"],
        default="mean",
        help=(
            "How model_env_onset combines envelope and onset correlations. "
            "mean is recommended for the first experiment."
        ),
    )
    parser.add_argument(
        "--error",
        choices=["l1", "l2"],
        default="l2",
        help="Eelbrain boosting error metric",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun subjects even if summary result already exists",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Running envelope+onset backward mTRF pipeline")
    print(f"processed_dir = {args.processed_dir}")
    print(f"results_dir   = {args.results_dir}")
    print(f"subjects      = {args.subjects}")
    print(f"max_workers   = {args.max_workers}")
    print(f"python        = {args.python}")
    print(f"score_mode    = {args.score_mode}")
    print(f"error         = {args.error}")

    # First show which files will be used. This helps catch naming mistakes.
    print("\nInput files:")
    for subject in args.subjects:
        path = expected_input_file(args.processed_dir, subject)
        status = "OK" if path.exists() else "MISSING"
        print(f"  {subject_id(subject)}: {path} [{status}]")

    failures: list[tuple[int, str]] = []

    if args.max_workers == 1:
        for subject in args.subjects:
            sub, success, msg = run_subject(
                subject=subject,
                processed_dir=args.processed_dir,
                results_dir=args.results_dir,
                python_executable=args.python,
                score_mode=args.score_mode,
                error=args.error,
                overwrite=args.overwrite,
            )

            print(f"\n========== {subject_id(sub)} ==========")
            print(msg)

            if not success:
                failures.append((sub, msg))

    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_subject,
                    subject,
                    args.processed_dir,
                    args.results_dir,
                    args.python,
                    args.score_mode,
                    args.error,
                    args.overwrite,
                )
                for subject in args.subjects
            ]

            for future in as_completed(futures):
                sub, success, msg = future.result()

                print(f"\n========== {subject_id(sub)} ==========")
                print(msg)

                if not success:
                    failures.append((sub, msg))

    print("\nFinished envelope+onset model run.")
    print(f"Successful subjects: {len(args.subjects) - len(failures)} / {len(args.subjects)}")

    if failures:
        print("\nFailures:")
        for sub, msg in failures:
            first_line = msg.splitlines()[0] if msg else "Unknown error"
            print(f"  {subject_id(sub)}: {first_line}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
