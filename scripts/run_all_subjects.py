from __future__ import annotations

"""
Run preprocessing and/or backward Eelbrain decoding for multiple subjects.

This script parallelizes subject-wise, meaning each subject is processed in a
separate Python process. The actual preprocessing/model scripts still only know
how to process one subject at a time.

Example
-------
python scripts/run_all_subjects.py ^
  --bidsdir data/raw/ds-eeg-snhl ^
  --processed-dir data/processed ^
  --results-dir results_backward ^
  --subjects 1 2 3 4 ^
  --step both ^
  --max-workers 4
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAD pipeline for multiple subjects")

    parser.add_argument("--bidsdir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)

    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject numbers, e.g. --subjects 1 2 3 4",
    )

    parser.add_argument(
        "--step",
        choices=["preprocess", "model", "both"],
        default="both",
        help="Which part of the pipeline to run",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of subjects to run in parallel",
    )

    parser.add_argument(
        "--audio-variant",
        choices=["plain", "woa", "woacontrol"],
        default="plain",
    )

    parser.add_argument(
        "--run-single-talker",
        action="store_true",
        help="Also run single-talker sanity model",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun even if output already exists",
    )

    return parser.parse_args()


def subject_id(subject: int) -> str:
    return f"sub-{subject:03d}"


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()

        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return result.returncode


def run_subject(
    subject: int,
    bidsdir: Path,
    processed_dir: Path,
    results_dir: Path,
    step: str,
    audio_variant: str,
    run_single_talker: bool,
    overwrite: bool,
) -> tuple[int, bool, str]:
    sid = subject_id(subject)

    processed_file = processed_dir / f"{sid}_backward_eelbrain.npz"
    subject_results_dir = results_dir / sid
    summary_file = subject_results_dir / f"{processed_file.stem}_backward_summary.json"

    subject_log_dir = results_dir / "logs" / sid
    subject_log_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    try:
        if step in {"preprocess", "both"}:
            if processed_file.exists() and not overwrite:
                messages.append(f"{sid}: skipping preprocessing, output exists")
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "models.preprocess_backward_eelbrain",
                    "--bidsdir",
                    str(bidsdir),
                    "--subject",
                    str(subject),
                    "--out",
                    str(processed_file),
                    "--audio-variant",
                    audio_variant,
                ]

                rc = run_command(cmd, subject_log_dir / "preprocess.log")
                if rc != 0:
                    return subject, False, f"{sid}: preprocessing failed, see {subject_log_dir / 'preprocess.log'}"

                messages.append(f"{sid}: preprocessing done")

        if step in {"model", "both"}:
            if not processed_file.exists():
                return subject, False, f"{sid}: missing processed file: {processed_file}"

            if summary_file.exists() and not overwrite:
                messages.append(f"{sid}: skipping model, summary exists")
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "models.eelbrain_backward",
                    "--input",
                    str(processed_file),
                    "--outdir",
                    str(subject_results_dir),
                    "--timing",
                ]

                if run_single_talker:
                    cmd.append("--run-single-talker")

                rc = run_command(cmd, subject_log_dir / "model.log")
                if rc != 0:
                    return subject, False, f"{sid}: model failed, see {subject_log_dir / 'model.log'}"

                messages.append(f"{sid}: model done")

        return subject, True, " | ".join(messages)

    except Exception as exc:
        return subject, False, f"{sid}: crashed with error: {exc}"


def main() -> None:
    args = parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Avoid CPU oversubscription when several subject jobs run in parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("Running subjects:", args.subjects)
    print("Step:", args.step)
    print("Max workers:", args.max_workers)
    print()

    failures: list[str] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_subject,
                subject,
                args.bidsdir,
                args.processed_dir,
                args.results_dir,
                args.step,
                args.audio_variant,
                args.run_single_talker,
                args.overwrite,
            ): subject
            for subject in args.subjects
        }

        for future in as_completed(futures):
            subject, ok, message = future.result()
            status = "OK" if ok else "FAILED"
            print(f"[{status}] {message}")

            if not ok:
                failures.append(message)

    print()
    print("Finished.")

    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()