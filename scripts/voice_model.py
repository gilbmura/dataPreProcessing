"""Voice verification model (Member 3).

This module is shared by two places, so they always process audio the same way:

* ``notebooks/03_audio.ipynb`` uses it to extract features and train the model.
* The command-line app (Member 2) uses ``verify_voice`` to approve a transaction.

From Python::

    import sys; sys.path.insert(0, "scripts")
    from voice_model import verify_voice
    approved, confidence = verify_voice("data/audio/gilbert/yes_approve.wav", "gilbert")

From a terminal::

    python scripts/voice_model.py data/audio/gilbert/yes_approve.wav gilbert

The terminal command exits with code 0 when approved, 1 when denied and 2 on an error.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "voice_model.pkl"

SAMPLE_RATE = 16_000            # every clip is resampled to this rate
N_MFCC = 20                     # number of MFCC coefficients
TRIM_TOP_DB = 30                # silence below peak - 30 dB is trimmed from both ends
MIN_TRIMMED_SECONDS = 0.2       # never trim a clip shorter than this
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
PHRASES = ("yes_approve", "confirm_transaction")
DEFAULT_THRESHOLD = 0.6         # used only if the model file has no threshold

PITCH_FMIN, PITCH_FMAX = 60, 400  # covers adult speaking voices

# One row of features per clip. MFCCs describe the shape of the vocal tract,
# roll-off and centroid describe how bright the voice is, RMS is energy,
# zero-crossing rate separates voiced from noisy sounds, and pitch (f0) is how
# high the voice is - the cue that changes least between different phrases.
FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + [
        "rolloff_mean",
        "rolloff_std",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "centroid_mean",
        "pitch_median_hz",
        "pitch_std_semitones",
    ]
)


def _librosa():
    """Import librosa lazily so ``--help`` stays fast."""
    import librosa

    return librosa


# --------------------------------------------------------------------------- #
# Loading and file naming
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, sr: int = SAMPLE_RATE, trim: bool = True) -> np.ndarray:
    """Load a clip as mono at ``sr`` Hz, trim leading/trailing silence and peak-normalise.

    Peak normalisation removes differences in microphone volume, so the model
    learns the voice rather than how close someone held the phone.
    """
    librosa = _librosa()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # audioread fallback warnings for mp3/m4a
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"{path} contains no audio")
    if trim:
        trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        if trimmed.size >= MIN_TRIMMED_SECONDS * sr:
            y = trimmed
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.95 * y / peak
    return y.astype(np.float32)


def parse_recording_name(path: str | Path) -> tuple[str, int]:
    """Split a file name such as ``confirm_transaction_2.wav`` into (phrase, take).

    A file with no take number counts as take 1.
    """
    stem = Path(path).stem.strip().lower().replace("-", "_").replace(" ", "_")
    match = re.fullmatch(r"(.*?)(?:_(\d+))?", stem)
    phrase, take = match.group(1), match.group(2)
    return phrase, int(take) if take else 1


def find_recordings(folder: str | Path) -> list[dict]:
    """List every audio file under ``folder``.

    Member recordings live in ``data/audio/<firstname>/``, so the member is
    the name of the folder directly above the file.
    """
    folder = Path(folder)
    rows = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        phrase, take = parse_recording_name(path)
        rows.append(
            {
                "member": path.parent.name.strip().lower(),
                "phrase": phrase,
                "take": take,
                "path": path,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Summarise a clip as one row of numbers (see ``FEATURE_COLUMNS``)."""
    librosa = _librosa()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    pitch_median, pitch_spread = _pitch_stats(y, sr)

    values = (
        list(mfcc.mean(axis=1))
        + list(mfcc.std(axis=1))
        + [
            rolloff.mean(),
            rolloff.std(),
            rms.mean(),
            rms.std(),
            zcr.mean(),
            centroid.mean(),
            pitch_median,
            pitch_spread,
        ]
    )
    return {name: float(value) for name, value in zip(FEATURE_COLUMNS, values)}


def _pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Median pitch (Hz) and pitch variation (semitones) over the loud, voiced frames."""
    librosa = _librosa()
    frame, hop = 1024, 256
    f0 = librosa.yin(y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
                     frame_length=frame, hop_length=hop)
    energy = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    n = min(len(f0), len(energy))
    f0, energy = f0[:n], energy[:n]
    voiced = f0[energy >= 0.25 * energy.max()] if n else f0
    if voiced.size == 0:
        voiced = f0
    median = float(np.median(voiced))
    spread = float(np.std(12 * np.log2(voiced / median))) if median > 0 else 0.0
    return median, spread


def features_from_file(path: str | Path, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Load a clip and return its feature row."""
    return extract_features(load_audio(path, sr=sr), sr=sr)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[Path, dict] = {}


def load_model(model_path: str | Path | None = None) -> dict:
    """Load the saved model bundle (cached after the first call).

    The bundle is a dict with ``model``, ``classes``, ``feature_columns``,
    ``threshold``, ``sample_rate``, ``model_name`` and ``metrics``.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    path = path.resolve()
    if path not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"No voice model at {path}. Run notebooks/03_audio.ipynb to train it."
            )
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


def _probabilities(bundle: dict, audio_path: str | Path) -> np.ndarray:
    sr = bundle.get("sample_rate", SAMPLE_RATE)
    features = features_from_file(audio_path, sr=sr)
    X = pd.DataFrame([features], columns=bundle["feature_columns"])
    return bundle["model"].predict_proba(X)[0]


def identify_speaker(
    audio_path: str | Path, model_path: str | Path | None = None
) -> tuple[str, float]:
    """Return the most likely member and the model's probability for them."""
    bundle = load_model(model_path)
    proba = _probabilities(bundle, audio_path)
    best = int(np.argmax(proba))
    return str(bundle["classes"][best]), float(proba[best])


def verify_voice(
    audio_path: str | Path,
    expected_name: str,
    model_path: str | Path | None = None,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """Check whether ``audio_path`` is the voice of ``expected_name``.

    Returns ``(approved, confidence)``. ``confidence`` is the model's
    probability that the clip belongs to ``expected_name``. The clip is
    approved only if that member is the most likely speaker *and* the
    confidence reaches the threshold chosen in the notebook. An unknown
    name is always denied with confidence 0.0.
    """
    bundle = load_model(model_path)
    classes = [str(c) for c in bundle["classes"]]
    expected = expected_name.strip().lower()
    if expected not in classes:
        return False, 0.0

    proba = _probabilities(bundle, audio_path)
    confidence = float(proba[classes.index(expected)])
    limit = bundle.get("threshold", DEFAULT_THRESHOLD) if threshold is None else threshold
    approved = confidence >= limit and confidence >= float(proba.max())
    return bool(approved), confidence


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a voice recording belongs to a team member."
    )
    parser.add_argument("audio_path", help="recording to check (.wav recommended)")
    parser.add_argument("expected_name", help="member the speaker claims to be, e.g. gilbert")
    parser.add_argument("--model", default=None, help="path to voice_model.pkl")
    parser.add_argument("--threshold", type=float, default=None, help="override the saved threshold")
    args = parser.parse_args(argv)

    try:
        bundle = load_model(args.model)
        approved, confidence = verify_voice(
            args.audio_path, args.expected_name, args.model, args.threshold
        )
        speaker, speaker_conf = identify_speaker(args.audio_path, args.model)
    except Exception as exc:  # report any loading/feature error and exit cleanly
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    limit = bundle.get("threshold", DEFAULT_THRESHOLD) if args.threshold is None else args.threshold
    name = args.expected_name.strip().lower()
    if name not in [str(c) for c in bundle["classes"]]:
        print(f"DENIED: '{name}' is not an enrolled member "
              f"(members: {', '.join(map(str, bundle['classes']))})")
    elif approved:
        print(f"APPROVED: voice matches '{name}' "
              f"(confidence {confidence:.2f}, threshold {limit:.2f})")
    else:
        print(f"DENIED: voice does not match '{name}' "
              f"(confidence {confidence:.2f}, threshold {limit:.2f})")
    print(f"Closest enrolled voice: {speaker} ({speaker_conf:.2f})")
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
