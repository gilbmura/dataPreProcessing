# Data conventions

## Original inputs

- `data/raw/customer_social_profiles.csv` and `customer_transactions.csv`: source customer data.
- `data/images/<firstname>/`: three member expressions, named
  `<firstname>_neutral.jpeg`, `<firstname>_smiling.jpeg` and
  `<firstname>_surprised.jpeg`. `.jpg` and `.png` are also supported.
- `data/audio/<firstname>/`: `yes_approve.wav` and `confirm_transaction.wav`.
  Additional takes can use `yes_approve_2.wav` and `confirm_transaction_2.wav`.
- `data/unauthorized/`: non-member images and recordings. Include the person's
  name and expression/phrase in each filename, such as `Frank_neutral.jpeg`
  and `Placide_Yes_approve.wav`.
- `data/demo_users.csv`: maps enrolled face identities to customer IDs.

Use lowercase member folder names consistently across face and audio inputs. Some
existing image filenames retain their original capitalization; demo lookup handles it.
Record both phrases in a quiet environment and prefer WAV audio. Extra takes must
be new recordings, not copies of an existing clip.

## Generated artifacts

The notebooks write the merged customer data and image/audio feature CSVs under
`data/`, model bundles under `models/`, and local figures/results under `report/`.
CSVs directly under `report/` and PNGs under `report/figures/` are versioned;
other report materials are ignored. The final report PDF and contribution sheet
are submitted separately. Plots also remain embedded in the saved notebooks.
`data/images_augmented/` contains image variants created during preprocessing.
Keep originals separate from generated variants so grouped evaluation can keep
every original and its augmentations in the same fold.

The assignment includes data and trained models, so those files are intentionally
versioned. Virtual environments, caches and the personal completion checklist are
ignored. Share identifiable images and voices only with the contributors' agreement.

## Calibration and evaluation

Unauthorized examples are excluded from member-classifier fitting. They still
influence approval thresholds and therefore are not automatically independent tests.

- Current image notebook: all eight unauthorized images calibrate the face threshold.
- Audio notebook: Frank and Gabriella calibrate the threshold; Placide is excluded
  from fitting and calibration but is a previously inspected regression person.
- The audio notebook explicitly lists the expected unauthorized people. Adding a
  new person requires reviewing the calibration/test assignment before rerunning.

Freeze the policy before collecting a new final test set. See the
[evaluation guide](evaluation.md) for the current limitations.
