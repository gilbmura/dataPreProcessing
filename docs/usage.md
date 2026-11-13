# Running the application

The application accepts saved face images and voice recordings. Complete the
[setup instructions](setup.md), activate the project environment and run the
commands below from the repository root.

## Processing order

Every transaction follows the same sequence:

1. Recognize the face. An unknown face stops the transaction.
2. Look up the recognized member's customer record and predict a product category.
   Keep the prediction hidden.
3. Verify that the voice matches the face-recognized identity.
4. Display the recommendation only after voice approval.

**Argument order does not change execution order.** Writing `--voice` before
`--face` still runs facial recognition first. These options specify input files;
the application determines when each model runs.

## Interactive menu

```powershell
python scripts/app.py
```

Choose `a`, `b` or `c` for an individual scenario, `d` for all scenarios, `m` to
enter image and audio paths, or `q` to quit. The default demo member is Kate.
Choose another member when starting the menu:

```powershell
python scripts/app.py --user Gilbert
```

## Built-in scenarios

```powershell
python scripts/app.py --scenario a --user Gilbert
python scripts/app.py --scenario b --user Gilbert
python scripts/app.py --scenario c --user Gilbert
```

- **Scenario a:** an enrolled member's face and matching voice. Expected outcome:
  approval followed by the product recommendation.
- **Scenario b:** an unauthorized face. Expected outcome: denial before the product
  and voice models are called.
- **Scenario c:** an enrolled member's face with an unauthorized voice. Expected
  outcome: denial at voice verification, with the recommendation still hidden.

Run all three together:

```powershell
python scripts/app.py --scenario all --user Gilbert
```

Supported demo names are Amazing, Kate and Gilbert. The `--user` option selects
demo inputs; it does not bypass identity verification. The unauthorized scenario
inputs are selected from the available files. Use explicit paths when demonstrating
a particular unauthorized person.

## Specify input files

```powershell
python scripts/app.py --face data/images/gilbert/gilbert_neutral.jpeg --voice data/audio/gilbert/yes_approve.wav
```

The equivalent PowerShell command across multiple lines is:

```powershell
python scripts\app.py `
  --face data\images\gilbert\gilbert_neutral.jpeg `
  --voice data\audio\gilbert\yes_approve.wav
```

The backtick must be the last character on its line, with no trailing spaces.
Enclose paths containing spaces in double quotes. Provide both `--face` and
`--voice`; do not combine explicit paths with `--scenario`.

To test an unauthorized voice against Gilbert's face:

```powershell
python scripts/app.py --face data/images/gilbert/gilbert_neutral.jpeg --voice data/unauthorized/Placide_Yes_approve.wav
```

The app reads existing files; it does not capture webcam images or microphone audio.

## How recommendations are selected

[demo_users.csv](../data/demo_users.csv) maps each enrolled identity to a customer
record: Amazing to A100, Kate to A105 and Gilbert to A180. These are demonstration
assignments, not evidence of the members' actual shopping habits.

The product classifier uses the mapped customer's saved social-profile and purchase
history features to predict Books, Clothing, Electronics, Groceries or Sports. Face
and voice inputs control access; they are not preference features for that model.
The app returns a category rather than an individual item, does not update the model
after each run, and does not execute a purchase or payment.

## Exit codes and errors

- **0:** an individual transaction was approved, or all combined scenarios matched
  their expected outcomes. Normal exit from the interactive menu also returns 0.
- **1:** an individual transaction was denied, or a combined scenario had an
  unexpected outcome.
- **2:** a configuration, file or processing error occurred, or the menu was interrupted.

Invalid command-line arguments also return 2. A missing model or failed dependency
is an execution error, not evidence of successful unauthorized-user rejection.
Use `python scripts/app.py --help` for the available options and the
[setup guide](setup.md) for troubleshooting.

## Demonstration checklist

Show an authorized transaction, an unauthorized-face rejection and a wrong-voice
rejection. Either the combined scenarios or separate explicit-file commands can
demonstrate these outcomes; there is no need to show every invocation style.

Explain when the recommendation is computed and displayed, identify reused
training/calibration inputs, and state the [evaluation limitations](evaluation.md).
Current verification results and their dates are maintained in that guide.
