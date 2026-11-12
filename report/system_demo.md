# Current Kate-based system demonstration

## Prerequisite
Install the environment as described in the README (`requirements-face.txt`, then
facenet-pytorch with `--no-deps`) and run `python scripts/check_environment.py`. It
should report `"package_check_passed": true`.

## Commands (repository root, activated environment)

```bash
python scripts/app.py --scenario a --user Gilbert
python scripts/app.py --scenario b --user Gilbert
python scripts/app.py --scenario c --user Gilbert
python scripts/app.py --scenario all --user Gilbert
```

Scenario a must recognize the member, prepare a product without displaying it,
approve the matching voice, then display the predicted product. Scenario b must deny
the unauthorized face before calling either later model. Scenario c must recognize
the valid face and deny the wrong voice without revealing the prepared product.
The menu is available with `python scripts/app.py` without arguments.

An explicit input example:

```bash
python scripts/app.py --face data/images/gilbert/gilbert_neutral.jpeg --voice data/audio/gilbert/yes_approve.wav
```

Exit 0 means approval (or matching outcomes for all scenarios); exit 1 means an
authentication denial/scenario mismatch; exit 2 means a configuration/runtime error.
An import failure must never be counted as a successful unauthorized demonstration.
Kate's app has no --json or --face-box options. It uses actual files, not live capture.

## What is and is not verified
Verified on 12 November 2026 with the re-run models:

| Check | Result |
|---|---|
| `--scenario all` for Gilbert, Kate and Amazing | PASS for all three |
| Each of the 8 stranger photos | denied at the face step |
| Gilbert's face with each of the 6 stranger voice clips | denied at the voice step |
| A member's face with another member's voice | denied (voice confidence 0.03 or lower) |
| Each member's face with their own voice, both phrases | approved, product shown |
| Menu option `m` with a mistyped path | reports the missing file, returns to the menu |
| Unit tests | 21 pass |

Previous `app_demo_results.json` entries refer to the pre-pull face model and are
out of date. The video is not recorded yet. Record the three scenarios with the
stored files (a fresh voice recording is likely to be rejected), explain the model
order and the limited-data and false-rejection risks, and put the video link in the PDF.
