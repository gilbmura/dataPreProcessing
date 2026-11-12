#!/usr/bin/env python3
"""
app.py  -  Command-line multimodal transaction system (Member 2)

Flow:  face check -> (look up customer, predict product, keep hidden) -> voice check -> show product.
If the face or the voice fails, the app prints "Access Denied" and stops.

Usage
    python scripts/app.py                       interactive menu
    python scripts/app.py --scenario a          (a) authorized transaction
    python scripts/app.py --scenario b          (b) unauthorized face, denied at the face step
    python scripts/app.py --scenario c          (c) authorized face + wrong voice, denied at the voice step
    python scripts/app.py --scenario all        run a, b and c one after another (good for the demo video)
    python scripts/app.py --face PATH --voice PATH     your own files
    --user Kate                                 which team member plays the authorized user (default Kate)

All three real models are required. Missing components stop the transaction; no filename-based approval is used.
"""
import os
import sys
import glob
import argparse

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import face_model                                   # our own model (always required)
from face_model import predict_face

# Import real components lazily so --help and the menu remain available.
def predict_product(customer_id):
    from product_model import predict_product as predict
    return predict(customer_id)


def verify_voice(audio_path, expected_name):
    from voice_model import verify_voice as verify
    return verify(audio_path, expected_name.strip().lower())

# ----------------------------------------------------------------------------- helpers
def load_users():
    users = pd.read_csv(os.path.join(ROOT, "data", "demo_users.csv"), dtype=str)
    if not {'name', 'customer_id'}.issubset(users.columns) or users.empty:
        raise ValueError('User mapping needs name and customer_id columns')
    if users[['name', 'customer_id']].isna().any().any():
        raise ValueError('Empty user mapping values')
    users['name'] = users['name'].str.strip()
    users['customer_id'] = users['customer_id'].str.strip()
    if users['name'].eq('').any() or users['customer_id'].eq('').any() or users['name'].str.lower().duplicated().any():
        raise ValueError('Empty or duplicate identities in user mapping')
    return users


def first_file(folder, extensions):
    files = [p for p in sorted(glob.glob(os.path.join(folder, "*"))) if p.lower().endswith(extensions)]
    return files[0] if files else None


def find_with_ext(base, extensions):
    # Match filename case on all platforms (older Kate files use an uppercase K).
    folder = os.path.dirname(base)
    stem = os.path.basename(base).lower()
    for path in sorted(glob.glob(os.path.join(folder, '*'))):
        if os.path.splitext(os.path.basename(path))[0].lower() == stem and path.lower().endswith(extensions):
            return path
    for ext in extensions:
        if os.path.isfile(base + ext):
            return base + ext
    return base + extensions[0]


def scenario_paths(key, user):
    u = user.lower()
    if u not in {name.lower() for name in load_users()['name']}:
        raise ValueError('Unknown demo member')
    face_ok = find_with_ext(os.path.join(ROOT, "data", "images", u, f"{u}_neutral"), (".jpg", ".jpeg", ".png"))
    voice_ok = find_with_ext(os.path.join(ROOT, "data", "audio", u, "yes_approve"), (".wav",))
    unauth = os.path.join(ROOT, "data", "unauthorized")
    if key == "a":
        return face_ok, voice_ok
    if key == "b":
        return (first_file(unauth, (".jpg", ".jpeg", ".png")) or os.path.join(unauth, "stranger_face.jpg")), voice_ok
    if key == "c":
        wrong = first_file(unauth, (".wav",))
        if wrong is None:                                    # no stranger recording yet: use another member's voice
            others = [m for m in load_users()["name"] if m.lower() != u]
            wrong = find_with_ext(os.path.join(ROOT, "data", "audio", others[0].lower(), "yes_approve"), (".wav",))
        return face_ok, wrong
    raise ValueError(key)


def denied(reason):
    print(f"   {reason}")
    print("\n   Access Denied\n")
    return False


def check_files(face_path, voice_path):
    missing = [p for p in (face_path, voice_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("File not found: " + ", ".join(missing))


def run_transaction(face_path, voice_path, users):
    print("=" * 64)
    print(" SECURE TRANSACTION")
    print("=" * 64)

    # ---- step 1: face ---------------------------------------------------------
    print(f"[1/2] Face check: {face_path}")
    try:
        name, conf = predict_face(face_path)
    except Exception as e:
        raise RuntimeError('Face processing failed; check backend dependencies') from e
    if name is None or name.lower() == "unknown":
        return denied(f"Face not recognized (confidence {conf:.2f} is below the threshold).")
    print(f"   Face recognized. Welcome, {name}!  (confidence {conf:.2f})")

    row = users[users["name"].str.lower() == name.lower()]
    if row.empty:
        return denied(f"No customer record found for {name}.")
    customer_id = row["customer_id"].iloc[0]
    try:
        product = predict_product(customer_id)  # computed now, but NOT shown yet
    except Exception as e:
        raise RuntimeError('Product model failed') from e
    if not isinstance(product, str) or not product.strip():
        raise ValueError('No product prediction for mapped customer')

    # ---- step 2: voice --------------------------------------------------------
    print(f"[2/2] Voice check: {voice_path}")
    try:
        approved, vconf = verify_voice(voice_path, name.lower())
    except Exception as e:
        raise RuntimeError('Voice processing failed') from e
    if not approved:
        return denied(f"Voice not approved for {name} (confidence {vconf:.2f}).")
    print(f"   Voice verified.  (confidence {vconf:.2f})")

    # ---- both checks passed ---------------------------------------------------
    print("\n   Transaction approved.")
    print(f"   Recommended product for {name} (customer {customer_id}): {product}\n")
    return True


SCENARIO_TITLES = {"a": "Scenario (a): authorized face + authorized voice",
                   "b": "Scenario (b): unauthorized face (denied at the face step)",
                   "c": "Scenario (c): authorized face + wrong voice (denied at the voice step)"}


def run_scenario(key, user, users):
    print(f"\n>>> {SCENARIO_TITLES[key]}")
    face, voice = scenario_paths(key, user)
    return run_transaction(face, voice, users)


def interactive(user, users):
    while True:
        print("\nMenu:  [a] authorized transaction   [b] unauthorized face   [c] wrong voice")
        print("       [d] all three scenarios      [m] enter my own file paths   [q] quit")
        choice = input("Choose: ").strip().lower()
        if choice in ("a", "b", "c"):
            run_scenario(choice, user, users)
        elif choice == "d":
            for k in "abc":
                run_scenario(k, user, users)
        elif choice == "m":
            face = input("Face image path: ").strip().strip('"')
            voice = input("Voice sample path: ").strip().strip('"')
            try:
                check_files(face, voice)
                run_transaction(face, voice, users)
            except FileNotFoundError as exc:      # a typo should not end the demo
                print(f"   {exc}")
        elif choice in ("q", "quit", "exit"):
            break
        else:
            print("   Unknown choice. Type a, b, c, d, m or q.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multimodal authentication + product recommendation demo")
    ap.add_argument("--scenario", choices=["a", "b", "c", "all"])
    ap.add_argument("--user", default="Kate", help="team member who plays the authorized user")
    ap.add_argument("--face")
    ap.add_argument("--voice")
    args = ap.parse_args(argv)

    if bool(args.face) != bool(args.voice):
        ap.error('--face and --voice must be supplied together')
    if args.scenario and args.face:
        ap.error('Use either --scenario or explicit --face/--voice paths')
    try:
        users = load_users()
        if args.face and args.voice:
            check_files(args.face, args.voice)
            return 0 if run_transaction(args.face, args.voice, users) else 1
        if args.scenario == 'all':
            results = {k: run_scenario(k, args.user, users) for k in 'abc'}
            # Expected rejections must not be mistaken for application success:
            # at least the authorized transaction must actually work.
            matched = results == {'a': True, 'b': False, 'c': False}
            print('Scenario checks:', 'PASS' if matched else 'FAIL (inspect model/dependency errors above)')
            return 0 if matched else 1
        if args.scenario:
            return 0 if run_scenario(args.scenario, args.user, users) else 1
        interactive(args.user, users)
        return 0
    except (EOFError, KeyboardInterrupt):
        print('Demo stopped.')
        return 2
    except FileNotFoundError as exc:
        print(f'Transaction stopped. {exc}')
        return 2
    except Exception as exc:
        # Only the error type is shown: a message could contain the hidden product.
        print(f'Transaction stopped ({type(exc).__name__}). Check configuration and model dependencies.')
        print('Environment check: python scripts/check_environment.py')
        return 2



if __name__ == "__main__":
    sys.exit(main())
