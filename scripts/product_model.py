"""
product_model.py  (Member 1)
============================
Product recommendation model: data preparation helpers + predict_product().

Used in two ways:
  1. notebooks/01_tabular.ipynb imports the helper functions below to build
     data/merged_dataset.csv and to train models/product_model.pkl.
  2. Member 2's app.py imports predict_product(customer_id):

        from product_model import predict_product
        product = predict_product("A178")      # or 178, or "178"

predict_product returns the predicted product category as a string, or None
if the customer ID is not in the dataset (never raises for an unknown ID).
"""

import os
import pickle

import numpy as np
import pandas as pd

SEED = 42

# Paths are built from this file's location so they work from any folder.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODEL_PATH = os.path.join(ROOT, "models", "product_model.pkl")
MERGED_PATH = os.path.join(ROOT, "data", "merged_dataset.csv")

# ---------------------------------------------------------------------------
# Column groups (the model only ever sees FEATURE_COLS, never the target)
# ---------------------------------------------------------------------------
TARGET = "product_category"
NUMERIC_COLS = [
    "n_profiles", "avg_engagement", "avg_purchase_interest",
    "share_positive", "share_neutral", "share_negative",
    "n_prior_txns", "total_prior_spend", "avg_prior_amount",
    "avg_prior_rating", "days_since_last_txn",
]
CATEGORICAL_COLS = ["main_platform", "prior_top_category"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_profiles(profiles):
    """Clean the social profile table.
    - 'A178' style IDs -> integer 178 (matches the transactions file)
    - drop exact duplicate rows
    """
    df = profiles.copy()
    df["customer_id"] = df["customer_id_new"].str.replace("A", "", regex=False).astype(int)
    df = df.drop_duplicates()
    return df


def clean_transactions(transactions):
    """Clean the transactions table.
    - rename ID column, parse dates, sort in time order
    - missing customer_rating is LEFT as NaN (it is imputed inside the
      model pipeline so the fill value comes from training data only)
    """
    df = transactions.copy().rename(columns={"customer_id_legacy": "customer_id"})
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df = df.drop_duplicates()
    return df.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_customer_profile(profiles_clean):
    """One row per customer summarising all of their social media profiles.
    A customer can have several profile rows (one per platform/snapshot)."""
    g = profiles_clean.groupby("customer_id")
    out = pd.DataFrame({
        "n_profiles": g.size(),
        "avg_engagement": g["engagement_score"].mean(),
        "avg_purchase_interest": g["purchase_interest_score"].mean(),
        "share_positive": g["review_sentiment"].apply(lambda s: (s == "Positive").mean()),
        "share_neutral": g["review_sentiment"].apply(lambda s: (s == "Neutral").mean()),
        "share_negative": g["review_sentiment"].apply(lambda s: (s == "Negative").mean()),
        # most common platform; ties are broken alphabetically (mode()[0])
        "main_platform": g["social_media_platform"].agg(lambda s: s.mode().iloc[0]),
    })
    return out.reset_index()


def history_features(prior, reference_date):
    """Summarise a customer's PRIOR transactions (a small DataFrame).
    Only transactions that happened before the one being predicted are
    passed in, so the target purchase never leaks into its own features."""
    if len(prior) == 0:
        return {"n_prior_txns": 0, "total_prior_spend": 0.0, "avg_prior_amount": np.nan,
                "avg_prior_rating": np.nan, "days_since_last_txn": np.nan,
                "prior_top_category": "none"}
    return {
        "n_prior_txns": len(prior),
        "total_prior_spend": float(prior["purchase_amount"].sum()),
        "avg_prior_amount": float(prior["purchase_amount"].mean()),
        "avg_prior_rating": float(prior["customer_rating"].mean()),  # NaN if all missing
        "days_since_last_txn": float((reference_date - prior["purchase_date"].max()).days),
        # most frequent earlier category (ties broken alphabetically)
        "prior_top_category": prior[TARGET].mode().iloc[0],
    }


def build_transaction_dataset(transactions_clean, profiles_clean):
    """Merged, feature-engineered table: one row per transaction.

    Inner join on customer_id (only customers that appear in BOTH files).
    The label is the product_category of that transaction. All history
    features use only earlier transactions of the same customer.
    """
    cust = build_customer_profile(profiles_clean)
    merged = transactions_clean.merge(cust, on="customer_id", how="inner", validate="m:1")

    rows = []
    for _, r in merged.iterrows():
        earlier = transactions_clean[
            (transactions_clean["customer_id"] == r["customer_id"])
            & ((transactions_clean["purchase_date"] < r["purchase_date"])
               | ((transactions_clean["purchase_date"] == r["purchase_date"])
                  & (transactions_clean["transaction_id"] < r["transaction_id"])))
        ]
        rows.append(history_features(earlier, r["purchase_date"]))
    hist = pd.DataFrame(rows, index=merged.index)
    return pd.concat([merged, hist], axis=1)


def build_customer_features(transactions_clean, profiles_clean):
    """One row per customer for INFERENCE: history = all their transactions,
    measured up to the day after the last transaction in the dataset."""
    cust = build_customer_profile(profiles_clean)
    ref_date = transactions_clean["purchase_date"].max() + pd.Timedelta(days=1)
    hist = {cid: history_features(grp, ref_date)
            for cid, grp in transactions_clean.groupby("customer_id")}
    hist = pd.DataFrame.from_dict(hist, orient="index")
    hist.index.name = "customer_id"
    out = cust.merge(hist.reset_index(), on="customer_id", how="inner")
    return out.set_index("customer_id")


# ---------------------------------------------------------------------------
# Prediction API used by Member 2's app
# ---------------------------------------------------------------------------
def _normalise_id(customer_id):
    """Accept 'A178', 'a178', '178' or 178 and return the integer 178 (or None)."""
    text = str(customer_id).strip().upper().lstrip("A")
    return int(text) if text.isdigit() else None


_bundle = None  # loaded once, then reused


def predict_product(customer_id):
    """Return the predicted product category (str) for a customer ID,
    or None if the ID is unknown / not usable by the model."""
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)

    cid = _normalise_id(customer_id)
    table = _bundle["customer_features"]
    if cid is None or cid not in table.index:
        return None

    row = table.loc[[cid], _bundle["feature_cols"]]
    return str(_bundle["pipeline"].predict(row)[0])


if __name__ == "__main__":
    # Quick manual check:  python scripts/product_model.py A178
    import sys
    for cid in (sys.argv[1:] or ["A178"]):
        print(cid, "->", predict_product(cid))
