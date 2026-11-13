# Multimodal Authentication and Product Recommendation

A data preprocessing coursework project combining face recognition, product
recommendation and voice verification in one command-line application.

The app recognizes a face, prepares a product recommendation, then verifies that
the voice belongs to the same person. It reveals the recommendation only when both
identity checks pass. Failed checks deny access; runtime errors stop the transaction.

**Project scope:** a working educational prototype using saved images, recordings
and customer data. Its small datasets and model limitations require further
validation before use beyond the coursework demonstration.

## Quick start

Clone the repository and follow the [environment setup guide](docs/setup.md).
From the repository root, with the project environment activated:

```bash
python scripts/check_environment.py
python scripts/app.py --scenario all --user Gilbert
```

Run `python scripts/app.py` for the interactive menu. See the
[usage guide](docs/usage.md) for individual scenarios, explicit file paths,
PowerShell examples and exit codes.

## Documentation

- [Setup and troubleshooting](docs/setup.md) — dependencies, pretrained weights and notebooks.
- [Data conventions](docs/data.md) — recording names, image folders and generated artifacts.
- [Running the application](docs/usage.md) — inputs, scenarios, recommendations and exit codes.
- [Evaluation and limitations](docs/evaluation.md) — scores, validation scope and known weaknesses.

## Project structure

```text
dataPreProcessing/
├── data/                   # Original inputs, augmented images and feature CSVs
├── docs/                   # Setup, usage, data and evaluation guides
├── models/                 # Saved face, product and voice model bundles
├── notebooks/
│   ├── 01_tabular.ipynb     # Customer data preparation and product model
│   ├── 02_image.ipynb       # Image processing and face model
│   └── 03_audio.ipynb       # Audio processing and voice model
├── scripts/
│   ├── app.py              # Integrated command-line application
│   ├── face_model.py       # Face feature extraction and prediction
│   ├── product_model.py    # Customer features and recommendation
│   ├── voice_model.py      # Audio features and speaker verification
│   ├── check_environment.py
│   └── run_notebook.py
├── tests/                  # App control flow and model regression checks
├── report/                 # Verification CSVs and exported PNGs in figures/
├── requirements.txt        # Notebook and core model dependencies
└── requirements-face.txt   # Full environment additions for FaceNet
```

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/app.py --scenario all --user Amazing
python scripts/app.py --scenario all --user Kate
python scripts/app.py --scenario all --user Gilbert
```

Recorded checks from **12 November 2026** include 21 passing automated tests and
nine passing application scenarios. These results include training and calibration
inputs; they are not independent accuracy measurements. See
[evaluation and limitations](docs/evaluation.md) for the metrics, evidence and
remaining validation work.

Held-out voice acceptance is only **25%**. The working environment also has declared
FaceNet dependency conflicts, documented in the [setup guide](docs/setup.md).

## Team

- **Amazing — Member 1:** tabular preprocessing, product model, shared setup and documentation, PR integration.
- **Kate — Member 2:** image preprocessing, face model and integrated CLI application.
- **Gilbert — Member 3:** audio preprocessing, voice model and verification.

The final report PDF and contribution sheet are submitted separately, alongside
the required video and repository links. Verification CSVs in `report/` and PNG
plots in `report/figures/` are kept in the repository. Report drafts, Word/PDF
documents, JSON summaries and the personal checklist remain local and are ignored.
