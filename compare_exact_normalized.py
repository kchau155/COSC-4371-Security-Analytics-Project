"""COSC 4371 Project 17 - final checkpoint exact vs normalized comparison.

Read-only on existing split.csv and results/{train,validation,test}.csv.
Trains two TF-IDF + LinearSVC models with the SAME fixed held-out rows.
Writes scores.csv in the professor's FINAL checkpoint format, preserving a
backup of an existing scores.csv, and a sampled error-review worksheet.

Place next to split.csv in the Project folder; run: python compare_exact_normalized.py
"""
from __future__ import annotations

import re
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
MAX_VALIDATION_FPR = 0.05


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Smallest attainable cutoff with validation FPR <= 5%."""
    negative = np.sort(scores[y_true == 0])[::-1]
    if negative.size == 0:
        raise ValueError("Validation set has no negative examples")
    allowed = int(np.floor(MAX_VALIDATION_FPR * negative.size))
    return float(np.nextafter(negative[allowed], np.inf))


def model_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("classifier", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=10_000,
        )),
    ])


def load_file(root: Path, name: str) -> pd.DataFrame:
    path = root / "results" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Preserve your original splits.")
    df = pd.read_csv(path, dtype={"row_id": str, "dataset": str, "split": str})
    needed = {"row_id", "dataset", "split", "text", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path.name} is missing {needed - set(df.columns)}")
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} contains missing text or label")
    if df["row_id"].duplicated().any():
        raise ValueError(f"{name} has repeated row IDs")
    if not df["label"].isin([0, 1]).all() or not (df["split"] == name).all():
        raise ValueError(f"Invalid labels or split names in {name}")
    return df


def create_normalized_train(train, validation, test):
    candidate = train.copy()
    candidate["_key"] = candidate["text"].map(normalize_text)
    heldout_keys = set(validation["text"].map(normalize_text)) | set(test["text"].map(normalize_text))
    overlaps = candidate["_key"].isin(heldout_keys)
    after_heldout = candidate.loc[~overlaps].copy()
    within = int(after_heldout["_key"].duplicated().sum())
    conflicts = (after_heldout.groupby("_key")["label"].nunique() > 1).sum()
    if conflicts:
        raise ValueError(f"{conflicts} normalized groups have conflicting training labels; inspect these")
    clean = after_heldout.drop_duplicates("_key", keep="first").drop(columns="_key")
    print(f"Normalized cleaning: {len(train)} - {int(overlaps.sum())} held-out matches "
          f"- {within} training duplicates = {len(clean)} rows")
    return clean


def main():
    root = Path(__file__).resolve().parent
    results = root / "results"
    if not (root / "split.csv").is_file():
        raise FileNotFoundError(f"Missing {root / 'split.csv'}. Put script in the Project root.")

    train = load_file(root, "train")
    validation = load_file(root, "validation")
    test = load_file(root, "test")
    all_rows = pd.concat([train, validation, test], ignore_index=True)
    split_csv = pd.read_csv(root / "split.csv", dtype={"row_id": str})
    if set(all_rows["row_id"]) != set(split_csv["row_id"]):
        raise ValueError("Fixed split files disagree with split.csv")
    if set(train["dataset"]) != {"safeguard"} or set(validation["dataset"]) != {"safeguard"}:
        raise ValueError("Expected Safe-Guard-only training and validation")
    if set(test["dataset"]) != {"safeguard", "deepset", "gandalf"}:
        raise ValueError("Unexpected test datasets")

    train_normalized = create_normalized_train(train, validation, test)
    print("Fixed test rows by source:", test["dataset"].value_counts().to_dict())
    print("Fixed validation rows:", len(validation))
    val_y = validation["label"].to_numpy(dtype=int)
    records = []

    for value, training_rows in [("exact", train), ("normalized", train_normalized)]:
        model = model_pipeline()
        start = time.perf_counter()
        model.fit(training_rows["text"], training_rows["label"])
        runtime = time.perf_counter() - start
        val_scores = model.decision_function(validation["text"])
        threshold = choose_threshold(val_y, np.asarray(val_scores, dtype=float))
        predictions = (np.asarray(val_scores) >= threshold)
        val_fpr = predictions[val_y == 0].mean()
        assert val_fpr <= MAX_VALIDATION_FPR + 1e-12
        test_scores = np.asarray(model.decision_function(test["text"]), dtype=float)
        frame = test[["row_id", "dataset", "label"]].copy()
        frame = frame.rename(columns={"dataset": "column"})
        frame.insert(1, "value", value)
        frame["score"] = test_scores
        frame["threshold"] = threshold
        records.append(frame)
        print(f"{value}: fitted {len(training_rows)} rows in {runtime:.3f}s; "
              f"threshold={threshold:.17g}, validation FPR={val_fpr:.4%}")
        for ds, part in frame.groupby("column", sort=False):
            y = part["label"].to_numpy(dtype=int)
            p = part["score"].to_numpy() >= threshold
            recall = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
            print(f"  {ds}: {len(part)} test rows, recall={recall:.4f}")

    scores = pd.concat(records, ignore_index=True)
    assert not scores.duplicated(["row_id", "value", "column"]).any()
    for column, g in scores.groupby("column"):
        by_value = [set(part["row_id"]) for _, part in g.groupby("value")]
        if by_value[0] != by_value[1]:
            raise ValueError(f"Different test rows between values for {column}")

    old_scores = root / "scores.csv"
    if old_scores.is_file():
        backup = results / "scores_before_final_comparison.csv"
        if not backup.exists():
            shutil.copy2(old_scores, backup)
            print(f"Backed up existing scores.csv to {backup}")
        else:
            print(f"Existing scores.csv backup retained at {backup}")
    scores.to_csv(old_scores, index=False, float_format="%.17g")
    print(f"Wrote final-checkpoint scores: {old_scores} ({len(scores)} rows)")

    # One fixed error-analysis cell with both classes: normalized / Deepset.
    review = scores.loc[(scores["value"] == "normalized") &
                        (scores["column"] == "deepset")].copy()
    review["predicted"] = (review["score"] >= review["threshold"]).astype(int)
    review = review.loc[review["predicted"] != review["label"]].copy()
    review["error_type"] = np.where(review["label"] == 0, "false positive", "false negative")
    samples = []
    for error_type in ("false positive", "false negative"):
        part = review.loc[review["error_type"] == error_type]
        if len(part):
            samples.append(part.sample(n=min(10, len(part)), random_state=SEED))
    if samples:
        sample = pd.concat(samples, ignore_index=True)
        sample = sample.merge(test[["row_id", "text"]], on="row_id", how="left", validate="one_to_one")
        sample["category"] = ""  # Fill manually AFTER reading each prompt; don't invent categories.
        out = results / "error_sample_for_review.csv"
        sample[["row_id", "value", "column", "category", "error_type", "text", "label",
                "predicted", "score", "threshold"]].to_csv(out, index=False)
        print(f"Wrote {out} ({len(sample)} errors). Review prompts, assign categories, "
              "then save row_id,value,column,category columns as errors.csv in Project root.")
    else:
        print("No Deepset errors found for normalized variant; choose another cell for error analysis.")

    print("Next command (from Project folder):")
    print("  python check_final_checkpoint.py --split split.csv --scores scores.csv "
          "--metric recall --reference exact --direction either")
    print("Add '--errors errors.csv' after manually categorizing error samples.")
    print("IMPORTANT: use '--direction either' only if this matches your prespecified rule.")


if __name__ == "__main__":
    main()
