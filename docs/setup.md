# Setup and troubleshooting

Run commands from the repository root. The observed working environment is Windows
with Python 3.13.2 and CPU inference. Other environments have not been verified here.

## Install

```bash
git clone https://github.com/gilbmura/dataPreProcessing.git
cd dataPreProcessing
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For macOS/Linux shells, activation is `source .venv/bin/activate`. If PowerShell
blocks activation, use `.\.venv\Scripts\python.exe` wherever the commands below say
`python`; no execution-policy change is necessary.

```bash
python -m pip install -r requirements-face.txt
python -m pip install --no-deps facenet-pytorch==2.6.0
python scripts/check_environment.py
```

The full application requires all three saved model files in `models/`,
`data/demo_users.csv`, and actual image/audio inputs. Installing only
`requirements.txt` is insufficient for FaceNet image inference.

## Dependency compatibility

The working environment observed on 12 November 2026 contains:

```text
Python          3.13.2
numpy           2.5.4
Pillow          13.0.0
torch           2.15.0
torchvision     0.29.2
scikit-learn    1.9.1
librosa         1.0.0
facenet-pytorch 2.6.0
```

FaceNet 2.6.0 declares older NumPy, Pillow, torch and torchvision requirements.
Installing it with `--no-deps` preserves the working environment but deliberately
bypasses those constraints. **`python -m pip check` reports four conflicts.**
Actual image inference and the supplied-file app scenarios pass locally; this is
not an upstream-supported compatibility guarantee or a clean dependency resolution.
The requirements files are not a complete lockfile, so a later installation can
resolve different versions. Verify a new environment before replacing a working one.

The environment checker tests package availability and loading the saved face
bundle; its success does not establish dependency compatibility or download weights.

## Pretrained weights

The VGGFace2 weights download on the first FaceNet call (approximately 107 MB).
That first run needs an internet connection. To initialize them explicitly:

```bash
python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2').eval(); print('FaceNet weights ready')"
python scripts/app.py --scenario all --user Gilbert
```

## Run the notebooks

Open the notebooks in an editor/Jupyter client using the project environment as
the kernel, or use the included runner from the repository root:

```bash
python scripts/run_notebook.py notebooks/01_tabular.ipynb
python scripts/run_notebook.py notebooks/02_image.ipynb
python scripts/run_notebook.py notebooks/03_audio.ipynb
```

Training replaces generated feature CSVs and saved model bundles. Review those
changes together with notebook outputs. The runner saves partial outputs on failure;
an executed notebook is successful only if all relevant cells finish without errors.

Notebooks may create a local `report/` folder for exported figures and result files.
CSVs directly under `report/` and PNGs under `report/figures/` can be committed;
other report materials are ignored. These exports are not required to run the
application with its saved models. The final PDF report is submitted separately.

Use `02_image.ipynb` to reproduce the current reported face run. The standalone
`face_model.py train` path excludes Placide photos from calibration, whereas the
current notebook uses all eight unauthorized photos. These are different evaluation
protocols and must not be described as interchangeable.

## Common issues

- **Missing package:** confirm the terminal and notebook use the same environment.
- **Weight download fails:** check network access and rerun initialization.
- **Model load fails:** check the saved bundle and matching scikit-learn version;
  do not replace it with a different backend just to suppress the error.
- **Access Denied:** this can be a model decision, not an installation failure.
  Read the [evaluation limitations](evaluation.md) before changing thresholds.
- **Git cannot create `index.lock`:** check for another Git operation or restricted
  write access. A permission error alone does not establish a stale lock; use a
  normal terminal or Git client with repository access.
