# Member 3: audio processing and verification

Six member recordings cover two phrases for Amazing, Gilbert and Kate. Frank,
Gabriella and Placide supply six unauthorized recordings. Clips are resampled to
16 kHz, trimmed and normalized, then augmented with two pitch shifts, two time
stretches and pink noise. The feature file has 72 rows with 48 numeric features.

Member fitting excludes every unauthorized row. Original recordings and all their
variants stay together in leave-one-recording-out validation. Random Forest has
accuracy 0.944, macro F1 0.943 and log loss 0.573; model selection uses these same
folds. The rows derive from six independent member originals, not 36 sessions.

## Rejection-policy correction
The old balanced-margin policy chose 0.50, admitted calibration strangers and
incorrectly accepted both Placide originals. The new rule chooses the smallest
floating-point threshold strictly greater than the largest Frank/Gabriella score,
using both the selected model's fold fits and its final member-only fit. The saved
threshold is 0.7200000000000001; rounding down changes the boundary condition.
A tied top identity is denied. Empty calibration is an error; calibration confidence
1.0 produces an explicit reject-all threshold instead of a permissive fallback.

Calibration false acceptance is 0.0%, compared
with 37.5% under the old rule. Both Placide original clips are now rejected (0/2
false accepts, previously 2/2). Across his 12 original/augmented versions, 1/12
still passes (8.3%). His samples never enter
fitting, model selection or threshold selection. However, the known failure prompted
this revision, so these are regression results, not a fresh independent test.

## Genuine-user cost and checks
Held-out member acceptance dropped from 80.6% to 25.0%
(9/36); false rejection is 75.0%. This is a substantial
usability limitation. All six original training clips still approve for their own
member. All 36 original-clip claim checks pass: six correct claims approved,
12 incorrect member claims denied and 18 unauthorized claims denied. Training and
calibration examples are not independent evidence of generalization.

This correction addresses the reported original-clip failures but does not establish
reliable speaker verification. Do not hide the remaining augmented false accept or
75% held-out false rejection. More independently recorded member takes, sessions,
and strangers are needed, with separate calibration and a NEW final test person.
Gabriella's confirm recording remains too quiet. The pipeline has no replay or
synthetic-speech protection; classifier confidence is not proof of identity.

## Deliverables and integration
The executed notebook contains waveform/spectrogram interpretation, comparisons,
threshold plots and the current summary. Results are in report/audio_results.json
and report/audio_verification_checks.csv. scripts/voice_model.py exposes
verify_voice(audio_path, expected_name); the saved model is models/voice_model.pkl.
The integrated scripts/app.py passes the recognized face identity to voice verification
and displays the product only on approval. Supplied-file integration checks pass; the
recorded video remains pending. No threshold was adjusted using
Placide's measured confidence values.

After Kate pull 9fdcbf7, the audio model remains current, but prior face/app integration
checks are historical. Revalidate the Kate-based app after installing FaceNet dependencies.
