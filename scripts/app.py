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

Components that are not available yet (product_model.py / voice_model.py) are replaced by clearly
labelled placeholders so the whole flow can still be tested.
"""
import os
import sys
import glob
import pickle
import argparse

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import face_model                                   # our own model (always required)
from face_model import predict_face

PLACEHOLDERS = []

# ----------------------------------------------------------------------------- product model
try:
    from product_model import predict_product       # Member 1's real function
except Exception:                                    # script missing -> use the saved model directly
    _PRODUCT_BUNDLE = {}

    def predict_product(customer_id):
        path = os.path.join(ROOT, "models", "product_model.pkl")
        if not _PRODUCT_BUNDLE:
            with open(path, "rb") as f:
                _PRODUCT_BUNDLE.update(pickle.load(f))
        cid = int(str(customer_id).strip().upper().lstrip("A"))
        feats = _PRODUCT_BUNDLE["customer_features"]
        if cid not in feats.index:
            return None
        return str(_PRODUCT_BUNDLE["pipeline"].predict(feats.loc[[cid], _PRODUCT_BUNDLE["feature_cols"]])[0])

    PLACEHOLDERS.append("product_model.py not found: using models/product_model.pkl directly")

# ----------------------------------------------------------------------------- voice model
try:
    from voice_model import verify_voice             # Member 3's real function
except Exception:
    def verify_voice(audio_path, expected_name):
        """PLACEHOLDER: approves a file only if its name starts with the expected member's name."""
        ok = os.path.basename(audio_path).lower().startswith(expected_name.lower())
        return ok, (0.95 if ok else 0.10)

    PLACEHOLDERS.append("voice_model.py not found: using a PLACEHOLDER voice check (matches the file name)")

# ----------------------------------------------------------------------------- helpers
def load_users():
    return pd.read_csv(os.path.join(ROOT, "data", "demo_users.csv"))


def first_file(folder, extensions):
    files = [p for p in sorted(glob.glob(os.path.join(folder, "*"))) if p.lower().endswith(extensions)]
    return files[0] if files else None


def find_with_ext(base, extensions):
    for ext in extensions:
        if os.path.isfile(base + ext):
            return base + ext
    return base + extensions[0]


def scenario_paths(key, user):
    u = user.lower()
    face_ok = find_with_ext(os.path.join(ROOT, "data", "images", u, f"{u}_neutral"), (".jpg", ".jpeg", ".png"))
    voice_ok = find_with_ext(os.path.join(ROOT, "data", "audio", u, f"{u}_yes_approve"), (".wav",))
    unauth = os.path.join(ROOT, "data", "unauthorized")
    if key == "a":
        return face_ok, voice_ok
    if key == "b":
        return (first_file(unauth, (".jpg", ".jpeg", ".png")) or os.path.join(unauth, "stranger_face.jpg")), voice_ok
    if key == "c":
        wrong = first_file(unauth, (".wav",))
        if wrong is None:                                    # no stranger recording yet: use another member's voice
            others = [m for m in load_users()["name"] if m.lower() != u]
            wrong = find_with_ext(os.path.join(ROOT, "data", "audio", others[0].lower(), f"{others[0].lower()}_yes_approve"), (".wav",))
        return face_ok, wrong
    raise ValueError(key)


def denied(reason):
    print(f"   {reason}")
    print("\n   Access Denied\n")
    return False


def run_transaction(face_path, voice_path, users):
    print("=" * 64)
    print(" SECURE TRANSACTION")
    print("=" * 64)

    # ---- step 1: face ---------------------------------------------------------
    print(f"[1/2] Face check: {face_path}")
    try:
        name, conf = predict_face(face_path)
    except FileNotFoundError as e:
        return denied(str(e))
    if name == "unknown":
        return denied(f"Face not recognized (confidence {conf:.2f} is below the threshold).")
    print(f"   Face recognized. Welcome, {name}!  (confidence {conf:.2f})")

    row = users[users["name"].str.lower() == name.lower()]
    if row.empty:
        return denied(f"No customer record found for {name}.")
    customer_id = row["customer_id"].iloc[0]
    product = predict_product(customer_id)                  # computed now, but NOT shown yet

    # ---- step 2: voice --------------------------------------------------------
    print(f"[2/2] Voice check: {voice_path}")
    try:
        approved, vconf = verify_voice(voice_path, name)
    except FileNotFoundError as e:
        return denied(str(e))
    if not approved:
        return denied(f"Voice not approved for {name} (confidence {vconf:.2f}).")
    print(f"   Voice verified.  (confidence {vconf:.2f})")

    # ---- both checks passed ---------------------------------------------------
    print("\n   Transaction approved.")
    if product is None:
        print(f"   No product prediction is available for customer {customer_id}.\n")
    else:
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
            run_transaction(input("Face image path: ").strip(), input("Voice sample path: ").strip(), users)
        elif choice in ("q", "quit", "exit"):
            break


def main():
    ap = argparse.ArgumentParser(description="Multimodal authentication + product recommendation demo")
    ap.add_argument("--scenario", choices=["a", "b", "c", "all"])
    ap.add_argument("--user", default="Kate", help="team member who plays the authorized user")
    ap.add_argument("--face")
    ap.add_argument("--voice")
    args = ap.parse_args()

    for msg in PLACEHOLDERS:
        print(f"[note] {msg}")
    users = load_users()

    if args.face and args.voice:
        run_transaction(args.face, args.voice, users)
    elif args.scenario == "all":
        for k in "abc":
            run_scenario(k, args.user, users)
    elif args.scenario:
        run_scenario(args.scenario, args.user, users)
    else:
        interactive(args.user, users)


if __name__ == "__main__":
    main()
