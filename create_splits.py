"""COSC 4371 Project 17: fixed dataset split + exact-match baseline cleaning.

Place this file inside Project/Datasets/, next to Baseline.py, and run:
    python Datasets/create_splits.py

The script reads original Parquet downloads and writes Project/split.csv and
Project/results/{train,validation,test}.csv. It does NOT train the detector.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VALIDATION_FRACTION = 0.20
DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


def read_downloads():
    """Read every dataset partition, remembering its original row identity."""
    pieces = []
    file_log = []

    for path in sorted(DATA_DIR.rglob("*.parquet")):
        folder = path.parent.name.lower()
        if "safe-guard" in folder:
            dataset = "safeguard"
        elif "deepset" in folder:
            dataset = "deepset"
        elif "gandalf" in folder:
            dataset = "gandalf"
        else:
            continue

        original_partition = path.name.split("-", 1)[0]
        if original_partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown publisher partition: {path.name}")

        raw = pd.read_parquet(path)
        if "text" not in raw.columns or raw["text"].isna().any():
            raise ValueError(f"Missing or null text in {path}")

        # Gandalf contains attack prompts and a similarity column, NOT a
        # binary ground-truth label. Treat all as positive for this project.
        if dataset == "gandalf":
            labels = pd.Series(1, index=raw.index, dtype=int)
        else:
            if "label" not in raw.columns:
                raise ValueError(f"Missing label in {path}")
            labels = pd.to_numeric(raw["label"], errors="raise")
            if not labels.isin([0, 1]).all():
                raise ValueError(f"Nonbinary label in {path}")
            labels = labels.astype(int)

        frame = pd.DataFrame({
            "row_id": [f"{dataset}:{path.name}:{i}" for i in range(len(raw))],
            "dataset": dataset,
            "source_partition": original_partition,
            "text": raw["text"].astype(str).to_numpy(),
            "label": labels.to_numpy(),
        })
        pieces.append(frame)
        file_log.append(f"{dataset}\t{original_partition}\t{len(raw)}\t{path.relative_to(DATA_DIR)}")

    if not pieces:
        raise FileNotFoundError(f"No dataset Parquet files found beneath {DATA_DIR}")
    pool = pd.concat(pieces, ignore_index=True)
    if pool["row_id"].duplicated().any():
        raise ValueError("row_id is not unique; inspect the downloaded filenames")

    for name in ("safeguard", "deepset", "gandalf"):
        if name not in set(pool["dataset"]):
            raise FileNotFoundError(f"Missing {name} Parquet files under {DATA_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "files_used.txt").write_text(
        "dataset\toriginal_partition\trows\tfile\n" + "\n".join(file_log) + "\n",
        encoding="utf-8",
    )
    return pool


def main():
    pool = read_downloads()

    # Keep the publisher's Safe-Guard TEST partition unchanged.
    safeguard_test = pool.loc[
        (pool.dataset == "safeguard") & (pool.source_partition == "test")
    ].copy()
    safeguard_train = pool.loc[
        (pool.dataset == "safeguard") & (pool.source_partition == "train")
    ].copy()
    if safeguard_train.empty or safeguard_test.empty:
        raise ValueError("Safe-Guard must have the publisher's train and test files")

    # External datasets are NEVER used for training or threshold tuning.
    # Every publisher partition from Deepset/Gandalf is external test data.
    external_test = pool.loc[pool.dataset.isin(["deepset", "gandalf"])].copy()
    test_df = pd.concat([safeguard_test, external_test], ignore_index=True)
    test_df["split"] = "test"

    # Uniform label policy: leave original Safe-Guard/Deepset labels as given.
    # Remove ALL Safe-Guard training records with internally conflicting labels.
    conflicting = safeguard_train.groupby("text")["label"].nunique()
    conflicting_texts = set(conflicting.index[conflicting > 1])
    without_conflicts = safeguard_train.loc[
        ~safeguard_train.text.isin(conflicting_texts)
    ].copy()
    removed_conflicts = len(safeguard_train) - len(without_conflicts)

    # Protect the fixed tests FIRST (both Safe-Guard and external sets).
    # This also ensures any future validation rows do not overlap tests.
    test_texts = set(test_df["text"])
    without_test_overlap = without_conflicts.loc[
        ~without_conflicts.text.isin(test_texts)
    ].copy()
    removed_test_overlap = len(without_conflicts) - len(without_test_overlap)

    # Exact-match cleaning: retain one row per distinct training-candidate text.
    # These are the only rows that will be divided into train and validation.
    unique_candidates = without_test_overlap.drop_duplicates(
        subset="text", keep="first"
    ).copy()
    removed_train_duplicates = len(without_test_overlap) - len(unique_candidates)

    # Fixed stratified 80/20 split of the distinct Safe-Guard training texts.
    # Because these are unique, identical strings cannot straddle train/val.
    train_df, val_df = train_test_split(
        unique_candidates,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=unique_candidates["label"],
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"

    # Verify NO raw-text leakage into any held-out set.
    train_texts = set(train_df.text)
    val_texts = set(val_df.text)
    overlap_train_test = len(train_df.loc[train_df.text.isin(test_texts)])
    overlap_train_val = len(train_df.loc[train_df.text.isin(val_texts)])
    overlap_val_test = len(val_df.loc[val_df.text.isin(test_texts)])
    assert overlap_train_test == 0
    assert overlap_train_val == 0
    assert overlap_val_test == 0

    # One row per kept observation, with the EXACT column names used by the
    # professor's check_checkpoint.py script.
    kept = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if kept.row_id.duplicated().any():
        raise AssertionError("Duplicate row_id in retained observations")
    kept[["row_id", "dataset", "split", "label"]].to_csv(
        PROJECT_DIR / "split.csv", index=False
    )

    # Text-bearing files for baseline.py to train and score. Keep these out
    # of public repositories if any source license restricts redistribution.
    for name, frame in [
        ("train", train_df), ("validation", val_df), ("test", test_df)
    ]:
        frame[["row_id", "dataset", "split", "label", "text"]].to_csv(
            RESULTS_DIR / f"{name}.csv", index=False
        )

    report = kept.groupby(["dataset", "split"], sort=True)["label"].agg(
        rows="size", positive="sum"
    ).reset_index()
    report["negative"] = report["rows"] - report["positive"]
    report["positive_pct"] = (report["positive"] / report["rows"] * 100).round(2)
    report.to_csv(RESULTS_DIR / "split_report.csv", index=False)

    removal_lines = [
        f"Original Safe-Guard publisher train rows: {len(safeguard_train)}",
        f"Dropped: internally conflicting labels: {removed_conflicts} rows ({len(conflicting_texts)} distinct texts)",
        f"Dropped: exact overlap with any fixed test set: {removed_test_overlap} rows",
        f"Dropped: remaining repeated Safe-Guard training texts: {removed_train_duplicates} rows",
        f"Safe-Guard unique rows assigned to train + validation: {len(unique_candidates)}",
        f"Fixed original Safe-Guard test rows: {len(safeguard_test)}",
        f"Fixed external test rows: {len(external_test)}",
        f"Post-cleaning train rows matching any test: {overlap_train_test}",
        f"Post-cleaning train rows matching validation: {overlap_train_val}",
        f"Post-cleaning validation rows matching test: {overlap_val_test}",
        "Note: this exact-match baseline does not yet eliminate character-Jaccard or semantic near-duplicate boundary overlaps.",
        "Note: original test rows, including any within-test duplicates, were kept unchanged.",
    ]
    (RESULTS_DIR / "cleaning_report.txt").write_text(
        "\n".join(removal_lines) + "\n", encoding="utf-8"
    )

    print("\nFINAL EXACT-MATCH BASELINE SPLITS")
    print(report.to_string(index=False))
    print("\nCLEANING AND LEAKAGE REPORT")
    print("\n".join(removal_lines))
    print(f"\nWrote {PROJECT_DIR / 'split.csv'}")
    print(f"Wrote dataset splits and reports to {RESULTS_DIR}")
    print("Next: train baseline.py to produce scores.csv; then run check_checkpoint.py")


if __name__ == "__main__":
    main()
