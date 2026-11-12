# Member 2: Kate image pipeline and integrated app

## Source and review status
Kate's commit `9fdcbf7` supplies the active image notebook, face module, 512-feature
FaceNet model, feature CSV, 54 augmented image files and CLI app. The repository's
`notebooks/02_image.ipynb` is Kate's executed notebook from that commit, with its
saved outputs.

## Images and processing
Nine original member photos cover neutral, smiling and surprised expressions for
Amazing, Gilbert and Kate. Six augmentations per photo (rotation +15/-15 degrees,
reflection, brightness increase/decrease and grayscale) produce 63 feature rows.
FaceNet uses MTCNN localization and a pretrained VGGFace2 network to produce
512-dimensional L2-normalized embeddings. Photo sessions and original framing vary;
augmentation does not create independent sessions. The classic HOG/HSV backend is
an explicit alternative requiring retraining, not a substitute for FaceNet inference.

## Classifier evaluation
Kate's saved run reports Logistic Regression accuracy 1.000, macro F1 1.000 and log
loss 0.002; Random Forest accuracy 1.000, macro F1 1.000 and loss 0.197. Local grouped
classifier validation from her supplied cached embeddings reproduced these rounded
scores; see report/kate_cached_feature_review.json. It does NOT validate raw-image
FaceNet extraction. All variants of an original photo stay in one fold. These same
folds choose the model, so this is not an independent final member test.

## Corrections during integration review
Undetected faces are now denied, rather than replacing the face crop with the full
image. Retraining preserves the existing saved backend unless explicitly overridden.
Unknown backend names, non-finite features/probabilities and invalid thresholds fail
clearly. Tied predictions are denied. Calibration thresholds cannot be capped below
an observed stranger score; confidence 1 requires reject-all. Model selection breaks
macro-F1 ties by lower loss. The revised notebook handles the classic backend without
unconditionally loading FaceNet, validates expression coverage and exports figures.

On 12 November 2026 Kate's notebook was re-run unchanged with all eight stranger
photos now in the repository (Frank, Placide and two earlier stranger photos). The
recomputed FaceNet features match Kate's to a cosine similarity of at least 0.99999,
and the classifier metrics are unchanged. The threshold rose from 0.938 to 0.970:
Frank's smiling photo scores 0.942 and would have passed the old threshold. Every
held-out member photo scores at least 0.995. Because all strangers set the threshold,
there is no independent stranger test; a new person is needed for that.

## CLI integration
Kate's menu and scenarios a/b/c/all remain available. Audio paths now match the real
files; identity names are normalized for the voice API. The app uses all three real
models with no filename-based voice fallback. Product predictions remain hidden until
voice verification approves the face-recognized identity. Missing models and runtime
errors stop the transaction with a nonzero exit code. Unit tests exercise control flow
with mocked inference, separately from classifier checks on cached embeddings.

## Limitations
The full system was re-run end to end on 12 November 2026: all three scenarios pass
for every member, and all stranger photos and voice clips are denied (see
`report/system_demo.md`).

There are only three original member photos per person, no liveness or replay check,
and no independent final face test. The audio gate also has a high held-out false
rejection rate. Successful development demos must not be presented as reliable access
control.
