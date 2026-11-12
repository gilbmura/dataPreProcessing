"""
face_model.py  -  Face recognition model (Member 2)

Pipeline: photos -> augmentation -> features -> classifier -> predict_face()

Public function required by the team contract (Section 4.3 of the task sheet):

    predict_face(image_path) -> (name, confidence)
        name is "unknown" when the confidence is below the saved threshold.

Feature backends
    "facenet"  FaceNet embeddings (512 numbers, facenet-pytorch, pretrained on VGGFace2).
               Needs:  pip install facenet-pytorch   (downloads weights on first use)
    "classic"  HOG shape features + HSV colour histogram (no download, no deep model).
               Available for explicit retraining with FACE_BACKEND=classic.
The backend used for training is stored inside models/face_model.pkl, so
predict_face always uses the same backend as the model was trained with.

Run `python scripts/face_model.py train` to rebuild features + model without the notebook.
Random seed 42 is used everywhere.
"""
import os
import sys
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------------- paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMAGES_DIR = os.path.join(ROOT, "data", "images")
AUG_DIR = os.path.join(ROOT, "data", "images_augmented")
UNAUTH_DIR = os.path.join(ROOT, "data", "unauthorized")
FEATURES_PATH = os.path.join(ROOT, "data", "image_features.csv")
MODEL_PATH = os.path.join(ROOT, "models", "face_model.pkl")

EXPRESSIONS = ["neutral", "smiling", "surprised"]
IMG_EXT = (".jpg", ".jpeg", ".png")
DEFAULT_THRESHOLD = 0.60          # used only if no stranger photo is available to tune it


class NoFaceDetected(ValueError):
    """The detector found no face; never classify the full background as a face."""

# ----------------------------------------------------------------------------- images
def load_rgb(path):
    """Open an image, honour the phone's rotation flag (EXIF) and return RGB."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


AUGMENTATIONS = {
    "rot_pos15":   lambda im: im.rotate(15, resample=Image.BICUBIC, fillcolor=(0, 0, 0)),
    "rot_neg15":   lambda im: im.rotate(-15, resample=Image.BICUBIC, fillcolor=(0, 0, 0)),
    "flip":        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "bright_up":   lambda im: ImageEnhance.Brightness(im).enhance(1.4),
    "bright_down": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "grayscale":   lambda im: im.convert("L").convert("RGB"),
}


def list_member_images(images_dir=IMAGES_DIR):
    """Find data/images/<name>/<name>_<expression>.jpg|png and parse the file names."""
    rows = []
    for path in sorted(glob.glob(os.path.join(images_dir, "**", "*"), recursive=True)):
        if not path.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if "_" not in stem:
            warnings.warn(f"Skipping {path}: name should look like firstname_neutral.jpg")
            continue
        name, expression = stem.split("_", 1)
        rows.append({"member": name.title(), "expression": expression.lower(),
                     "source_image": stem.lower(), "path": path})
    return rows


def list_unauthorized_images(unauth_dir=UNAUTH_DIR):
    return [p for p in sorted(glob.glob(os.path.join(unauth_dir, "*"))) if p.lower().endswith(IMG_EXT)]

# ----------------------------------------------------------------------------- features
_FACENET = {}


def _get_facenet():
    if not _FACENET:
        from facenet_pytorch import MTCNN, InceptionResnetV1   # imported lazily (heavy)
        import torch
        torch.manual_seed(SEED)
        detector = MTCNN(image_size=160, margin=20, select_largest=True, device="cpu")
        network = InceptionResnetV1(pretrained="vggface2").eval()
        _FACENET.update(mtcnn=detector, net=network)
    return _FACENET


def facenet_embedding(img):
    import torch
    fn = _get_facenet()
    face = fn["mtcnn"](img)                       # detected + cropped + normalised face, or None
    if face is None:
        raise NoFaceDetected('No face detected; provide a clear, single-face image.')
    with torch.no_grad():
        emb = fn["net"](face.unsqueeze(0))[0].numpy()
    return emb / (np.linalg.norm(emb) + 1e-9)     # L2-normalised embedding


def classic_features(img):
    from skimage.feature import hog
    gray = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hsv = np.asarray(img.resize((64, 64)).convert("HSV"))
    colour = np.concatenate([np.histogram(hsv[..., c], bins=8, range=(0, 256))[0] / (64 * 64) for c in range(3)])
    return np.concatenate([shape, colour]).astype(np.float32)


def default_backend():
    forced = os.environ.get("FACE_BACKEND")
    if forced:
        if forced not in {'facenet', 'classic'}:
            raise ValueError('FACE_BACKEND must be facenet or classic')
        return forced
    # Preserve the backend of an existing trusted project model. Missing packages
    # must stop execution, not silently replace FaceNet with a different model.
    if os.path.isfile(MODEL_PATH):
        return load_bundle()['backend']
    try:
        import facenet_pytorch  # noqa: F401
        return "facenet"
    except ImportError:
        warnings.warn("facenet-pytorch is not installed - falling back to the 'classic' HOG+colour features. "
                      "Run `pip install facenet-pytorch` for FaceNet embeddings.")
        return "classic"


def extract_features(img, backend):
    if backend not in {'facenet', 'classic'}:
        raise ValueError(f'Unsupported face backend: {backend}')
    features = facenet_embedding(img) if backend == 'facenet' else classic_features(img)
    if not np.isfinite(features).all():
        raise ValueError('Non-finite image features')
    return features


def feature_columns(df):
    return [c for c in df.columns if c.startswith("f_")]


def build_feature_table(backend=None, aug_names=None, save=True, verbose=True):
    """Augment every photo, save the augmented images and return/save the feature table."""
    backend = backend or default_backend()
    aug_names = aug_names or list(AUGMENTATIONS)
    items = list_member_images()
    if not items:
        raise RuntimeError(f"No photos found in {IMAGES_DIR}")
    rows = []
    for it in items:
        img = load_rgb(it["path"])
        variants = [("original", img)] + [(a, AUGMENTATIONS[a](img)) for a in aug_names]
        for aug, im in variants:
            if aug != "original":
                out_dir = os.path.join(AUG_DIR, it["member"].lower())
                os.makedirs(out_dir, exist_ok=True)
                im.save(os.path.join(out_dir, f"{it['source_image']}__{aug}.jpg"), quality=95)
            feats = extract_features(im, backend)
            rows.append({"member": it["member"], "expression": it["expression"],
                         "source_image": it["source_image"], "augmentation": aug,
                         **{f"f_{i:03d}": float(v) for i, v in enumerate(feats)}})
        if verbose:
            print(f"  processed {it['source_image']}")
    df = pd.DataFrame(rows)
    if save:
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        df.to_csv(FEATURES_PATH, index=False)
        if verbose:
            print(f"Saved {FEATURES_PATH} {df.shape} (backend: {backend})")
    return df

# ----------------------------------------------------------------------------- models
def make_models():
    return {
        "Logistic Regression": lambda: Pipeline([("scale", StandardScaler()),
                                                 ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))]),
        "Random Forest": lambda: Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
    }


def grouped_cv_proba(X, y, groups, make_model, n_splits=5):
    """Out-of-fold probabilities. Splitting is by ORIGINAL photo (groups), so an augmented
    copy of a photo is never in the training set while its original is in the test set."""
    classes = np.array(sorted(set(y)))
    n_splits = min(n_splits, len(set(groups)))
    oof = np.full((len(y), len(classes)), 1e-3)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):
            oof[te, list(classes).index(c)] = p[:, j]
    return oof / oof.sum(axis=1, keepdims=True), classes


def score_oof(y, oof, classes):
    pred = classes[oof.argmax(axis=1)]
    return {"accuracy": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "log_loss": log_loss(y, oof, labels=list(classes))}


def fit_final(df, model_name):
    cols = feature_columns(df)
    return make_models()[model_name]().fit(df[cols].values, df["member"].values)


def choose_threshold(member_conf, stranger_conf, default=DEFAULT_THRESHOLD, lo=0.50, hi=0.95):
    """Choose a threshold strictly above observed calibration strangers.

    `hi` is retained for API compatibility, but cannot cap the threshold below a
    stranger score. A score of 1.0 produces an explicit reject-all boundary.
    Pass calibration samples only; never final-test samples.
    """
    if stranger_conf is None or len(stranger_conf) == 0:
        warnings.warn('No calibration strangers: default threshold is uncalibrated.')
        return default
    strangers = np.asarray(stranger_conf, dtype=float)
    members = np.asarray(member_conf, dtype=float)
    for values in (strangers, members):
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Confidence scores must be finite one-dimensional values in [0, 1]')
    s_max = float(strangers.max())
    member_ref = float(np.percentile(members, 10)) if len(members) else 0.0
    if s_max < member_ref:
        t = (s_max + member_ref) / 2
    else:
        warnings.warn("The stranger's confidence overlaps the members' - the threshold cannot separate them cleanly.")
        t = np.nextafter(s_max, np.inf)
    return float(max(lo, t, np.nextafter(s_max, np.inf)))


def split_unauthorized_images():
    """Reserve Placide from calibration. His samples are previously inspected regression data."""
    paths = list_unauthorized_images()
    test = [p for p in paths if os.path.basename(p).lower().startswith('placide_')]
    return [p for p in paths if p not in test], test


def save_model(pipeline, backend, threshold, model_name, feature_cols, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"pipeline": pipeline, "classes": [str(c) for c in pipeline.classes_], "backend": backend,
              "threshold": float(threshold), "model_name": model_name, "n_features": len(feature_cols)}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    _BUNDLE.clear()
    return bundle

# ----------------------------------------------------------------------------- prediction
_BUNDLE = {}


def load_bundle(path=MODEL_PATH):
    if path not in _BUNDLE:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Face model not found: {path}. Train it first (notebooks/02_image.ipynb).")
        with open(path, "rb") as f:
            _BUNDLE[path] = pickle.load(f)
    return _BUNDLE[path]


def predict_face(image_path, threshold=None):
    """Return (name, confidence). name == "unknown" if confidence < threshold."""
    b = load_bundle()
    try:
        x = extract_features(load_rgb(image_path), b["backend"]).reshape(1, -1)
    except NoFaceDetected:
        return 'unknown', 0.0
    if x.shape[1] != b['n_features']:
        raise ValueError('Saved model and image feature dimensions do not match')
    proba = b["pipeline"].predict_proba(x)[0]
    if not np.isfinite(proba).all():
        raise ValueError('Non-finite face prediction')
    i = int(np.argmax(proba))
    conf = float(proba[i])
    t = b["threshold"] if threshold is None else threshold
    if not np.isfinite(t) or not 0 <= t <= np.nextafter(1.0, np.inf):
        raise ValueError('Invalid face threshold')
    unique = np.count_nonzero(proba == proba.max()) == 1
    return ("unknown" if conf < t or not unique else b["classes"][i]), conf

# ----------------------------------------------------------------------------- command line
def run_training():
    """Rebuild image_features.csv and models/face_model.pkl from the photos (same steps as the notebook)."""
    backend = default_backend()
    df = build_feature_table(backend=backend)
    cols = feature_columns(df)
    X, y, groups = df[cols].values, df["member"].values, df["source_image"].values
    results = {}
    for name, mk in make_models().items():
        oof, classes = grouped_cv_proba(X, y, groups, mk)
        results[name] = (score_oof(y, oof, classes), oof)
        print(f"{name:20s}", {k: round(v, 3) for k, v in results[name][0].items()})
    best = min(results, key=lambda n: (-results[n][0]['f1_macro'], results[n][0]['log_loss']))
    oof, classes = results[best][1], np.array(sorted(set(y)))
    is_orig = (df["augmentation"] == "original").values
    correct = oof.argmax(axis=1) == np.searchsorted(classes, y)
    member_conf = oof.max(axis=1)[is_orig & correct]
    pipe = fit_final(df, best)
    stranger = []
    calibration_files, test_files = split_unauthorized_images()
    for p in calibration_files:
        try:
            proba = pipe.predict_proba(extract_features(load_rgb(p), backend).reshape(1, -1))[0]
            stranger.append(float(proba.max()))
        except NoFaceDetected:
            stranger.append(0.0)
    threshold = choose_threshold(member_conf, stranger)
    save_model(pipe, backend, threshold, best, cols)
    print(f"Best model: {best} | stranger confidence: {stranger} | threshold: {threshold:.3f}")
    print("Saved", MODEL_PATH)
    for p in test_files:
        print('Previously inspected regression photo:', os.path.basename(p), predict_face(p))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_training()
    elif len(sys.argv) > 2 and sys.argv[1] == "predict":
        print(predict_face(sys.argv[2]))
    else:
        print("Usage:\n  python scripts/face_model.py train\n  python scripts/face_model.py predict <image_path>")
