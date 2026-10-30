from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def locate_project_dir() -> Path:
    """Find the project containing split.csv and all three generated split files.

    This also works when train_baseline.py is in a different Project folder
    from the one where create_splits.py generated its outputs.
    """
    script_dir = Path(__file__).resolve().parent
    names = ("train.csv", "validation.csv", "test.csv")

    def complete(root: Path) -> bool:
        return (root / "split.csv").is_file() and all(
            (root / "results" / name).is_file() for name in names
        )

    # First check the script's own folder and its immediate parent.
    for root in (script_dir, script_dir.parent):
        if complete(root):
            return root

    # Then look in the nearby repositories, including another COSC 4371
    # folder within the same repos directory.
    repos = next(
        (p for p in (script_dir, *script_dir.parents)
         if p.name.lower() == "repos"),
        None,
    )
    if repos is not None:
        candidates = sorted(
            {p.parent for p in repos.rglob("split.csv") if complete(p.parent)},
            key=lambda p: (p / "split.csv").stat().st_mtime,
            reverse=True,
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            options = "\n  ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                "Found multiple sets of split files:\n  " + options +
                "\nSet PROJECT_DIR explicitly to the intended folder."
            )

    raise FileNotFoundError(
        "Cannot find split.csv together with results/train.csv, "
        "results/validation.csv, and results/test.csv. "
        "Run create_splits.py in your data project, or set PROJECT_DIR "
        "explicitly to the folder containing those outputs."
    )


PROJECT_DIR = locate_project_dir()
RESULTS_DIR = PROJECT_DIR / "results"
SEED = 42
MAX_VALIDATION_FPR = 0.05


def load_split(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run create_splits.py first.")
    df = pd.read_csv(path, dtype={"row_id": str, "dataset": str, "split": str})
    required = {"row_id", "dataset", "split", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing columns: {required - set(df.columns)}")
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"Null text or label in {path}")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"Labels must be 0/1 in {path}")
    if (df["split"] != name).any():
        raise ValueError(f"Split field does not match {name} in {path}")
    if df["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row IDs in {path}")
    return df


def select_threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Lowest score cutoff yielding empirical validation FPR <= target.

    With k=floor(target * N_negative), only up to k negatives may be scored
    positive. Return the smallest representable float strictly exceeding the
    (k+1)-th highest negative score. Using nextafter handles tied scores.
    """
    neg_scores = np.sort(scores[y_true == 0])[::-1]
    if len(neg_scores) == 0:
        raise ValueError("Validation needs negative rows to choose an FPR threshold")
    allowed_false_positives = int(np.floor(target * len(neg_scores)))
    if allowed_false_positives >= len(neg_scores):
        raise ValueError("Target FPR must be less than 1")
    cutoff = np.nextafter(neg_scores[allowed_false_positives], np.inf)
    if not np.isfinite(cutoff):
        raise ValueError("Could not find a finite threshold")
    return float(cutoff)


def main() -> None:
    print(f"Using project files from: {PROJECT_DIR}")
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    all_rows = pd.concat([train, validation, test], ignore_index=True)
    official_split = pd.read_csv(PROJECT_DIR / "split.csv", dtype={"row_id": str})
    if set(all_rows["row_id"]) != set(official_split["row_id"]):
        raise ValueError("results split rows do not match split.csv")
    if set(train["dataset"]) != {"safeguard"}:
        raise ValueError("Expected to train on Safe-Guard only")
    if set(validation["dataset"]) != {"safeguard"}:
        raise ValueError("Expected to tune on Safe-Guard validation only")
    if set(test["dataset"]) != {"safeguard", "deepset", "gandalf"}:
        raise ValueError("Expected Safe-Guard, Deepset, and Gandalf test sets")

    model = Pipeline([
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

    print(f"Fitting TF-IDF + LinearSVC on {len(train)} Safe-Guard training rows...")
    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    fit_seconds = time.perf_counter() - start
    print(f"Training time: {fit_seconds:.2f} seconds")

    val_labels = validation["label"].to_numpy(dtype=int)
    val_scores = model.decision_function(validation["text"])
    threshold = select_threshold_at_fpr(val_labels, val_scores, MAX_VALIDATION_FPR)
    val_predictions = (val_scores >= threshold).astype(int)
    actual_fpr = float(val_predictions[val_labels == 0].mean())
    val_recall = float(val_predictions[val_labels == 1].mean())
    assert actual_fpr <= MAX_VALIDATION_FPR + 1e-12
    print(f"Chosen threshold (LinearSVC decision score): {threshold!r}")
    print(f"Validation FPR: {actual_fpr:.4f} (required <= {MAX_VALIDATION_FPR:.2f})")
    print(f"Validation recall: {val_recall:.4f}")

    # Do not tune anything using test labels or scores.
    test_scores = model.decision_function(test["text"])
    scores_csv = test[["row_id", "dataset", "label"]].copy()
    scores_csv = scores_csv.rename(columns={"dataset": "test_set"})
    scores_csv["score"] = np.asarray(test_scores, dtype=float)
    scores_csv.to_csv(PROJECT_DIR / "scores.csv", index=False, float_format="%.17g")
    print(f"Saved {PROJECT_DIR / 'scores.csv'} ({len(scores_csv)} rows)")

    # Figure for checkpoint Section 4; Safe-Guard test has both classes.
    safe_mask = test["dataset"].eq("safeguard").to_numpy()
    safe_y = test.loc[safe_mask, "label"].to_numpy(dtype=int)
    safe_pred = (test_scores[safe_mask] >= threshold).astype(int)
    matrix = confusion_matrix(safe_y, safe_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    display = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(display, ax=ax, shrink=0.78)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=15,
                    color="white" if matrix[r, c] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], labels=["Benign (0)", "Injection (1)"])
    ax.set_yticks([0, 1], labels=["Benign (0)", "Injection (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Safe-Guard test — exact-match-cleaned baseline")
    positive_share = 100 * safe_y.mean()
    fig.text(0.5, 0.02, f"n={len(safe_y):,}; injection prevalence={positive_share:.2f}%",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    image_file = RESULTS_DIR / "safeguard_confusion_matrix.png"
    fig.savefig(image_file, dpi=175)
    plt.close(fig)
    print(f"Saved {image_file}")

    report_lines = [
        "COSC 4371 checkpoint — exact-match-cleaned baseline",
        "Positive class: 1 = injection",
        "Train: Safe-Guard only",
        "Validation: held-out Safe-Guard training partition, stratified seed 42",
        "Tests: fixed Safe-Guard test + all Deepset + all Gandalf rows",
        "Pipeline: TfidfVectorizer(lowercase=True, strip_accents='unicode',",
        "  analyzer='word', ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True)",
        "  then LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=10000)",
        "Score: LinearSVC decision_function (not a probability)",
        f"Validation threshold selection: lowest cutoff giving FPR <= {MAX_VALIDATION_FPR}",
        f"Threshold: {threshold!r}",
        f"Validation FPR: {actual_fpr:.8f}",
        f"Validation recall: {val_recall:.8f}",
        f"Measured fit time (seconds): {fit_seconds:.4f}",
        f"Hardware: {platform.platform()}, CPU={platform.processor()}, logical CPUs={os.cpu_count()}",
        f"Python: {platform.python_version()}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"matplotlib: {plt.matplotlib.__version__}",
        f"Checkpoint checker command (run from Project folder):",
        f"python check_checkpoint.py --split split.csv --scores scores.csv --threshold {threshold!r}",
    ]
    details = RESULTS_DIR / "model_details.txt"
    details.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved {details}")
    print("\nNEXT: execute exactly:")
    print(f"python check_checkpoint.py --split split.csv --scores scores.csv --threshold {threshold!r}")


if __name__ == "__main__":
    main()
