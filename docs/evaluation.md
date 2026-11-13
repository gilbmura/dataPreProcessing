# Evaluation and limitations

Classification metrics below come from the saved notebook outputs.
They describe different evaluation splits, so they are not directly comparable
authentication success rates.

## Reported classifier results

- **Face — Logistic Regression:** accuracy 1.000, macro F1 1.000, log loss 0.002.
  Evaluation groups augmented variants by their original photo. There are only
  nine original member photos. The same folds help select the classifier.
- **Voice — Random Forest:** accuracy 0.944, macro F1 0.943, log loss 0.573.
  Leave-one-recording-out evaluation keeps augmentations with their recording.
  There are six original member clips; model selection uses these same folds.
- **Product — Logistic Regression:** grouped test accuracy 0.333, macro F1 0.332,
  log loss 1.677. The split groups customers. The majority baseline has accuracy
  0.250 and log loss 1.593, so the selected model has worse log loss.

Detailed methods and outputs are in the [tabular notebook](../notebooks/01_tabular.ipynb),
[image notebook](../notebooks/02_image.ipynb), and
[audio notebook](../notebooks/03_audio.ipynb).

## Authentication decisions

The current face threshold is approximately 0.970. All eight stranger photos
participated in calibration; their rejection is not independent stranger validation.
The original held-out member photos score above the threshold in the saved run.
Face detection does not establish liveness or prevent replay.

The saved voice threshold is `0.7200000000000001`; do not round it down to 0.72.
Both original Placide recordings reject, but one of his 12 original/augmented clips
accepts. Held-out genuine acceptance is **25%** (75% false rejection). Successful
checks on the original training clips therefore do not establish reliability on
new recordings. Placide's known failure prompted the policy revision, so his results
are regression evidence, not an untouched final test.

## Runtime checks

On 12 November 2026:

- 21 automated tests passed, covering app control flow, cached classifier inputs
  and voice rejection boundaries. These tests do not all run real face inference.
- All nine real app scenarios passed across Amazing, Kate and Gilbert.
- All 17 original-image checks passed: nine member images recognized and eight
  unauthorized images rejected.
- All 36 original-audio claims passed: six correct member claims approved,
  12 wrong-member claims rejected and 18 unauthorized claims rejected.
- All three notebooks contained executed code cells with no saved errors. The most
  recent verification inspected these outputs; it did not retrain the notebooks.

The [audio claim checks](../report/audio_verification_checks.csv) and exported PNG
figures are retained in the repository. Detailed runtime-check JSON files are kept
with local submission materials. The audio notebook regenerates the claim CSV and
`report/audio_results.json`; JSON summaries remain ignored by Git. The runtime-check
summary above records the observed run and is not automatically updated. Recheck
after changing data, models, dependencies or code.

## Remaining validation

Collect independent member recordings, recording sessions and new unauthorized
people. Separate model selection, threshold calibration and final testing. The
current app is a working demonstration, but the voice rejection rate, small image
dataset and weak product results prevent stronger reliability claims.
