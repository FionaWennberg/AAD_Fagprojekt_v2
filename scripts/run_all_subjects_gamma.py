from __future__ import annotations

"""
Run the full gamma/gammatone spectrogram AAD pipeline for all subjects.

Pipeline
--------
1. Make_spectrogram_features.py
   Uses src/aad_project/spectrogram_features.py to create WAV-level gamma features.

2. Preprocess_gamma_eelbrain.py
   Combines EEG data with gamma features and saves subject-level mTRF-ready .npz files.

3. mtrf_gamma_eelbrain.py
   Runs the backward Eelbrain gamma decoder.

Default processed output:
    data/processed/gamma/

Default results output:
    results_gamma/
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gamma spectrogram AAD pipeline for multiple subjects"
    )

    parser.add_argument(
        "--bidsdir",
        type=Path,
        required=True,
        help="Root directory of the BIDS EEG dataset",
    )
    parser.add_argument(
        "--stimdir",
        type=Path,
        required=True,
        help="Directory containing WAV stimulus files",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=Path("data/processed/gamma/features"),
        help="Directory where WAV-level gamma spectrogram features are stored",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/gamma"),
        help="Directory where subject-level gamma .npz files are stored",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_gamma"),
        help="Directory where gamma model results are saved",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject numbers, e.g. --subjects 1 2 3 or --subjects 1 2 ... 44",
    )
    parser.add_argument(
        "--step",
        choices=["features", "preprocess", "model", "both", "all"],
        default="all",
        help=(
            "features = only compute WAV-level gamma features; "
            "preprocess = only create subject-level gamma files; "
            "model = only run decoder; "
            "both = preprocess + model; "
            "all = features + preprocess + model"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of subjects to run in parallel",
    )

    # Actual script names in your project
    parser.add_argument(
        "--make-features-script",
        type=Path,
        default=Path("scripts/make_spectrogram_features.py"),
        help="Script that creates WAV-level gamma spectrogram features",
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=Path("scripts/Preprocess_gamma_eelbrain.py"),
        help="Script that creates subject-level gamma .npz files",
    )
    parser.add_argument(
        "--model-script",
        type=Path,
        default=Path("scripts/mtrf_gamma_eelbrain.py"),
        help="Script that runs the backward gamma Eelbrain decoder",
    )

    # Expected subject-level input name created by preprocessing
    parser.add_argument(
        "--input-template",
        default="sub-{subject:03d}_mtrf_gamma.npz",
        help=(
            "Filename template for subject-level gamma files inside --processed-dir. "
            "Available field: {subject:03d}"
        ),
    )

    # Options passed to preprocessing
    parser.add_argument(
        "--audio-variant",
        default="plain",
        help="Audio variant to pass to preprocessing, e.g. plain, woa, woacontrol",
    )
    parser.add_argument(
        "--run-single-talker",
        action="store_true",
        help="Pass --run-single-talker to preprocessing if supported",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to preprocessing if supported",
    )

    # Options passed to model
    parser.add_argument("--tstart", type=float, default=-0.5)
    parser.add_argument("--tstop", type=float, default=0.0)
    parser.add_argument("--basis", type=float, default=0.05)
    parser.add_argument("--basis-window", default="hamming")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--error", choices=["l1", "l2"], default="l2")
    parser.add_argument("--selective-stopping", type=int, default=1)
    parser.add_argument("--no-scale-data", action="store_true")
    parser.add_argument("--save-resp", action="store_true")
    parser.add_argument("--timing", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def subject_id(subject: int) -> str:
    return f"sub-{subject:03d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("Command:\n")
        log.write(" ".join(cmd))
        log.write("\n\n")
        log.flush()

        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )


def subject_input_path(args: argparse.Namespace, subject: int) -> Path:
    filename = args.input_template.format(subject=subject)
    return args.processed_dir / filename


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------

def run_feature_step(args: argparse.Namespace) -> None:
    """
    Compute WAV-level spectrogram feature files.

    This is run once, not once per subject, because the WAV features can be reused.
    """
    log_path = args.results_dir / "logs" / "make_spectrogram_features.log"

    cmd = [
        sys.executable,
        str(args.make_features_script),
        "--stimdir",
        str(args.stimdir),
        "--outdir",
        str(args.spectrogram_dir),
    ]

    print("Creating WAV-level gamma spectrogram features")
    print("Log:", log_path)

    run_command(cmd, log_path)


def run_preprocess_subject(args: argparse.Namespace, subject: int) -> None:
    sid = subject_id(subject)
    log_path = args.results_dir / "logs" / sid / "preprocess_gamma.log"

    cmd = [
        sys.executable,
        str(args.preprocess_script),
        "--bidsdir",
        str(args.bidsdir),
        "--processed-dir",
        str(args.processed_dir),
        "--spectrogram-dir",
        str(args.spectrogram_dir),
        "--subject",
        str(subject),
        "--audio-variant",
        args.audio_variant,
    ]

    if args.run_single_talker:
        cmd.append("--run-single-talker")

    if args.overwrite:
        cmd.append("--overwrite")

    run_command(cmd, log_path)


def run_model_subject(args: argparse.Namespace, subject: int) -> None:
    sid = subject_id(subject)
    input_path = subject_input_path(args, subject)
    outdir = args.results_dir / sid
    log_path = args.results_dir / "logs" / sid / "mtrf_gamma.log"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing subject-level gamma input file for {sid}: {input_path}\n"
            "Check whether Preprocess_gamma_eelbrain.py creates the same filename. "
            "If not, change --input-template."
        )

    ensure_dir(outdir)

    cmd = [
        sys.executable,
        str(args.model_script),
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
        "--tstart",
        str(args.tstart),
        "--tstop",
        str(args.tstop),
        "--basis",
        str(args.basis),
        "--basis-window",
        str(args.basis_window),
        "--error",
        str(args.error),
        "--selective-stopping",
        str(args.selective_stopping),
    ]

    if args.partitions is not None:
        cmd.extend(["--partitions", str(args.partitions)])

    if args.no_scale_data:
        cmd.append("--no-scale-data")

    if args.save_resp:
        cmd.append("--save-resp")

    if args.timing:
        cmd.append("--timing")

    run_command(cmd, log_path)


def run_subject(args: argparse.Namespace, subject: int) -> tuple[int, str]:
    sid = subject_id(subject)

    try:
        if args.step in {"preprocess", "both", "all"}:
            print(f"[{sid}] Preprocessing gamma data")
            run_preprocess_subject(args, subject)

        if args.step in {"model", "both", "all"}:
            print(f"[{sid}] Running gamma mTRF model")
            run_model_subject(args, subject)

        return subject, "OK"

    except Exception as exc:
        return subject, f"FAILED: {exc}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    ensure_dir(args.spectrogram_dir)
    ensure_dir(args.processed_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.results_dir / "logs")

    print("Gamma spectrogram AAD pipeline")
    print("------------------------------")
    print("Python:", sys.executable)
    print("Working directory:", Path.cwd())
    print("Step:", args.step)
    print("BIDS dir:", args.bidsdir)
    print("Stim dir:", args.stimdir)
    print("Spectrogram feature dir:", args.spectrogram_dir)
    print("Processed gamma dir:", args.processed_dir)
    print("Results dir:", args.results_dir)
    print("Subjects:", args.subjects)
    print("Max workers:", args.max_workers)
    print()

    if args.step in {"features", "all"}:
        run_feature_step(args)

    if args.step == "features":
        print("Done. Feature step only.")
        return

    # Avoid using too many CPU threads inside each worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if args.max_workers <= 1:
        results = [run_subject(args, subject) for subject in args.subjects]
    else:
        results = []

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_subject, args, subject): subject
                for subject in args.subjects
            }

            for future in as_completed(futures):
                subject = futures[future]

                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append((subject, f"FAILED: {exc}"))

    print()
    print("Summary")
    print("-------")

    n_failed = 0

    for subject, status in sorted(results):
        sid = subject_id(subject)
        print(f"{sid}: {status}")

        if status != "OK":
            n_failed += 1

    if n_failed:
        raise SystemExit(
            f"{n_failed} subject(s) failed. "
            f"Check logs in {args.results_dir / 'logs'}"
        )

    print("All requested subjects finished successfully.")


if __name__ == "__main__":
    main()