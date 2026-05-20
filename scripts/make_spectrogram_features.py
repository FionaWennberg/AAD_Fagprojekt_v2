from __future__ import annotations

import argparse
from pathlib import Path

from src.aad_project.spectrogram_features import save_gammatone_features


def find_wav_files(stimdir: Path) -> list[Path]:
    """Recursively find all WAV files."""
    return sorted(stimdir.rglob("*.wav"))


def make_output_path(wav_path: Path, stimdir: Path, outdir: Path) -> Path:
    """
    Mirror folder structure:
    data/raw/.../file.wav
        -> data/processed/.../file_spectrogram.npz
    """
    rel = wav_path.relative_to(stimdir)
    out_path = outdir / rel

    return out_path.with_name(out_path.stem + "_spectrogram.npz")


def process_file(wav_path: Path, stimdir: Path, outdir: Path) -> None:
    out_path = make_output_path(wav_path, stimdir, outdir)

    if out_path.exists():
        print(f"Skipping (exists): {out_path}")
        return

    print(f"Processing: {wav_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_gammatone_features(wav_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gammatone spectrogram features"
    )

    parser.add_argument(
        "--stimdir",
        type=Path,
        required=True,
        help="Directory with WAV stimulus files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for spectrogram features",
    )

    args = parser.parse_args()

    wav_files = find_wav_files(args.stimdir)

    if not wav_files:
        raise RuntimeError(f"No WAV files found in {args.stimdir}")

    print(f"Found {len(wav_files)} WAV files")

    for wav_path in wav_files:
        process_file(wav_path, args.stimdir, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()