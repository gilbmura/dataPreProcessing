"""Check installed packages for the saved face backend without downloading weights."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    required = ['numpy', 'pandas', 'PIL', 'sklearn', 'joblib', 'librosa', 'soundfile']
    root = Path(__file__).resolve().parents[1]
    backend = None
    problem = None
    try:
        import pickle
        with (root / 'models/face_model.pkl').open('rb') as stream:
            backend = pickle.load(stream)['backend']  # trusted project artifact only
    except Exception as exc:
        problem = type(exc).__name__
    if backend == 'facenet':
        required.extend(['torch', 'torchvision', 'facenet_pytorch'])
    elif backend == 'classic':
        required.append('skimage')
    present = {name: importlib.util.find_spec(name) is not None for name in required}
    ready = problem is None and backend in ('classic', 'facenet') and all(present.values())
    print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable,
                      'face_backend': backend, 'packages_present': present,
                      'model_load_error': problem, 'package_check_passed': ready,
                      'weights_and_real_inference_checked': False,
                      'next_step': ('Environment ready: run the notebooks and app scenarios (see README).' if ready else
                                    'pip install -r requirements-face.txt, then pip install --no-deps facenet-pytorch==2.6.0')}, indent=2))
    return 0 if ready else 2


if __name__ == '__main__':
    raise SystemExit(main())
