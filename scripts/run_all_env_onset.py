from __future__ import annotations

"""
Run the full envelope+onset backward mTRF pipeline for multiple subjects.

This runner does BOTH steps:

    1. Preprocess raw EEG/audio into:
        data/processed/sub-004_mtrf_env_onset.npz

    2. Train the backward mTRF model:
        EEG -> [envelope, onset]

Important:
    This version matches your current preprocessing script, which expects:
        --bidsdir
        --subject
        --out

    NOT:
        --processed-dir

Place this file here:
    scripts/run_all_env_onset_pipeline.py

Windows example:
    python scripts/run_all_env_onset_pipeline.py ^
      --bidsdir C:/subset/ds-eeg-snhl ^
      --processed-dir data/processed ^
      --results-dir results_env_onset ^
      --subjects 4 ^
      --step both ^
      --max-workers 1
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def subject_id(subject: int | str) -> str:
    digits = "".join(ch for ch in str(subject) if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse subject id from {subject!r}")
    return f"sub-{int(digits):03d}"


def expected_preprocessed_file(processed_dir: Path, subject: int | str) -> Path:
    return processed_dir / f"{subject_id(subject)}_mtrf_env_onset.npz"


def expected_summary_file(results_dir: Path, subject: int | str) -> Path:
    sid = subject_id(subject)
    return results_dir / sid / f"{sid}_backward_mtrf_env_onset.summary.json"


def run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        msg = completed.stdout.strip()
        if completed.stderr.strip():
            msg += "\nSTDERR:\n" + completed.stderr.strip()
        return True, msg
    except subprocess.CalledProcessError as exc:
        return (
            False,
            f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{exc.stdout}\n\nSTDERR:\n{exc.stderr}",
        )


def preprocess_subject(
    subject: int,
    bidsdir: Path,
    processed_dir: Path,
    python_executable: str,
    audio_variant: str,
    n_bands: int,
    audio_work_fs: float,
    overwrite: bool,
) -> tuple[bool, str, Path]:
    sid = subject_id(subject)
    out_path = expected_preprocessed_file(processed_dir, subject)

    if out_path.exists() and not overwrite:
        return True, f"Preprocessing skipped for {sid}: {out_path} already exists", out_path

    # Your preprocessing script expects --out, not --processed-dir
    cmd = [
        python_executable,
        "-m",
        "src.aad_project.preprocess_mtrf_envelope_onsets",
        "--bidsdir",
        str(bidsdir),
        "--subject",
        str(subject),
        "--out",
        str(out_path),
        "--audio-variant",
        audio_variant,
        "--n-bands",
        str(n_bands),
        "--audio-work-fs",
        str(audio_work_fs),
    ]

    ok, msg = run_command(cmd)

    if ok and not out_path.exists():
        return (
            False,
            f"Preprocessing finished for {sid}, but expected output was not found:\n{out_path}\n\nOutput:\n{msg}",
            out_path,
        )

    return ok, msg, out_path


def model_subject(
    subject: int,
    input_path: Path,
    results_dir: Path,
    python_executable: str,
    score_mode: str,
    error: str,
    overwrite: bool,
) -> tuple[bool, str]:
    sid = subject_id(subject)
    subject_results_dir = results_dir / sid
    subject_results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = expected_summary_file(results_dir, subject)

    if summary_path.exists() and not overwrite:
        return True, f"Model skipped for {sid}: {summary_path} already exists"

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

    return run_command(cmd)


def run_subject(
    subject: int,
    bidsdir: Path,
    processed_dir: Path,
    results_dir: Path,
    python_executable: str,
    step: str,
    audio_variant: str,
    n_bands: int,
    audio_work_fs: float,
    score_mode: str,
    error: str,
    overwrite: bool,
) -> tuple[int, bool, str]:
    sid = subject_id(subject)
    messages: list[str] = []
    input_path = expected_preprocessed_file(processed_dir, subject)

    if step in {"preprocess", "both"}:
        ok, msg, input_path = preprocess_subject(
            subject=subject,
            bidsdir=bidsdir,
            processed_dir=processed_dir,
            python_executable=python_executable,
            audio_variant=audio_variant,
            n_bands=n_bands,
            audio_work_fs=audio_work_fs,
            overwrite=overwrite,
        )
        messages.append("PREPROCESS:\n" + msg)
        if not ok:
            return subject, False, "\n\n".join(messages)

    if step in {"model", "both"}:
        if not input_path.exists():
            messages.append(f"MODEL:\nMissing preprocessed input for {sid}: {input_path}")
            return subject, False, "\n\n".join(messages)

        ok, msg = model_subject(
            subject=subject,
            input_path=input_path,
            results_dir=results_dir,
            python_executable=python_executable,
            score_mode=score_mode,
            error=error,
            overwrite=overwrite,
        )
        messages.append("MODEL:\n" + msg)
        if not ok:
            return subject, False, "\n\n".join(messages)

    return subject, True, "\n\n".join(messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full envelope+onset preprocessing and backward mTRF model for multiple subjects"
    )

    parser.add_argument("--bidsdir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--step", choices=["preprocess", "model", "both"], default="both")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--audio-variant", choices=["plain", "woa", "woacontrol"], default="plain")
    parser.add_argument("--n-bands", type=int, default=16)
    parser.add_argument("--audio-work-fs", type=float, default=16000.0)

    parser.add_argument("--score-mode", choices=["mean", "weighted", "env_only"], default="mean")
    parser.add_argument("--error", choices=["l1", "l2"], default="l2")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Running full envelope+onset backward mTRF pipeline")
    print(f"bidsdir       = {args.bidsdir}")
    print(f"processed_dir = {args.processed_dir}")
    print(f"results_dir   = {args.results_dir}")
    print(f"subjects      = {args.subjects}")
    print(f"step          = {args.step}")
    print(f"max_workers   = {args.max_workers}")
    print(f"python        = {args.python}")

    print("\nExpected preprocessed files:")
    for subject in args.subjects:
        path = expected_preprocessed_file(args.processed_dir, subject)
        if path.exists():
            status = "EXISTS"
        elif args.step in {"preprocess", "both"}:
            status = "WILL CREATE"
        else:
            status = "MISSING"
        print(f"  {subject_id(subject)}: {path} [{status}]")

    failures: list[tuple[int, str]] = []

    if args.max_workers == 1:
        for subject in args.subjects:
            sub, success, msg = run_subject(
                subject=subject,
                bidsdir=args.bidsdir,
                processed_dir=args.processed_dir,
                results_dir=args.results_dir,
                python_executable=args.python,
                step=args.step,
                audio_variant=args.audio_variant,
                n_bands=args.n_bands,
                audio_work_fs=args.audio_work_fs,
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
                    args.bidsdir,
                    args.processed_dir,
                    args.results_dir,
                    args.python,
                    args.step,
                    args.audio_variant,
                    args.n_bands,
                    args.audio_work_fs,
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

    print("\nFinished full envelope+onset pipeline.")
    print(f"Successful subjects: {len(args.subjects) - len(failures)} / {len(args.subjects)}")

    if failures:
        print("\nFailures:")
        for sub, msg in failures:
            first_line = msg.splitlines()[0] if msg else "Unknown error"
            print(f"  {subject_id(sub)}: {first_line}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
