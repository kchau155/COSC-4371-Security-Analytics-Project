"""COSC 4371 Project 17: reproducible character 5-gram Jaccard experiment.

Place in the Project folder, alongside split.csv, scores.csv and results/.
Keep original source Parquet downloads under Project/Datasets/.

  python run_jaccard.py --audit         # audit raw datasets, decide train removals
  python run_jaccard.py --train         # train Jaccard-cleaned SVM, add scores.csv rows
  python run_jaccard.py --all           # both stages

Required: pip install pandas numpy scikit-learn pyarrow

The raw-data Jaccard join uses exact verification plus a Jaccard prefix filter
(a complete candidate-generation method for the stated threshold). It does NOT
approximate the thresholded pair counts with MinHash sampling.

No original test or validation record is changed. The unchanged split.csv
records the common starting partitions. Variant-specific removed training row IDs
are recorded in results/jaccard_train_removed.csv.

use the final checkpoint via command 
python check_final_checkpoint.py --split split.csv --scores scores.csv 
--errors errors.csv --metric recall --reference exact --direction either
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
import shutil
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
GRAM_SIZE = 5
SIM_THRESHOLD = 0.85  # Proposed Jaccard threshold; state it explicitly in the report.
MAX_VALIDATION_FPR = 0.05
DATASETS = ("safeguard", "deepset", "gandalf")


def normalize(text: str) -> str:
    """NFKC + casefold + collapsed whitespace, matching the earlier audit."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def shingles(text: str) -> frozenset[str]:
    value = normalize(text)
    if len(value) < GRAM_SIZE:
        return frozenset(("<SHORT>" + value,))
    return frozenset(value[i:i + GRAM_SIZE] for i in range(len(value) - GRAM_SIZE + 1))


def find_similar_pairs(gram_sets: list[frozenset[str]], threshold: float):
    """Exact similarity join via standard inverted prefix filtering.

    Every qualifying Jaccard pair has at least one common prefix token using
    prefix length |S| - ceil(threshold*|S|) + 1 in global frequency order.
    Candidate pairs are checked using the *exact* set Jaccard calculation.
    """
    document_frequency = Counter(gram for grams in gram_sets for gram in grams)
    postings: dict[str, list[int]] = defaultdict(list)
    pairs: list[tuple[int, int, float]] = []
    candidate_count = 0
    for j, grams in enumerate(gram_sets):
        tokens = sorted(grams, key=lambda x: (document_frequency[x], x))
        prefix_length = len(tokens) - ceil(threshold * len(tokens)) + 1
        candidate_rows: set[int] = set()
        for token in tokens[:prefix_length]:
            candidate_rows.update(postings[token])

        for i in sorted(candidate_rows):
            a = gram_sets[i]
            # Necessary length bound: Jaccard <= min(|A|,|B|)/max(|A|,|B|).
            if min(len(a), len(grams)) < threshold * max(len(a), len(grams)) - 1e-10:
                continue
            candidate_count += 1
            overlap = len(a & grams)
            union = len(a) + len(grams) - overlap
            if overlap + 1e-10 >= threshold * union:
                pairs.append((i, j, overlap / union))

        for token in tokens[:prefix_length]:
            postings[token].append(j)
        if (j + 1) % 3000 == 0:
            print(f"  Jaccard checked {j + 1:,}/{len(gram_sets):,} prompts; "
                  f"{len(pairs):,} qualifying pairs so far", flush=True)
    return pairs, candidate_count


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        ai, aj = self.find(i), self.find(j)
        if ai != aj:
            if self.size[ai] < self.size[aj]:
                ai, aj = aj, ai
            self.parent[aj] = ai
            self.size[ai] += self.size[aj]

    def excess_rows(self):
        return sum(self.size[i] - 1 for i in range(len(self.parent))
                   if self.parent[i] == i and self.size[i] > 1)


def identify_source(path: Path) -> str | None:
    folder = path.parent.name.lower()
    if "safe-guard" in folder or "safeguard" in folder:
        return "safeguard"
    if "deepset" in folder or "deep-set" in folder:
        return "deepset"
    if "gandalf" in folder:
        return "gandalf"
    return None


def load_raw_parquet(root: Path) -> tuple[pd.DataFrame, list[str]]:
    base = root / "Datasets"
    if not base.is_dir():
        raise FileNotFoundError(
            f"Missing {base}. Put the original publisher Parquet folders in Project/Datasets/. "
            "The full within/cross-dataset audit must use raw source rows."
        )
    pieces = []
    files_used = []
    for path in sorted(base.rglob("*.parquet")):
        dataset = identify_source(path)
        if dataset is None:
            continue
        frame = pd.read_parquet(path)
        if "text" not in frame or frame["text"].isna().any():
            raise ValueError(f"Source file missing valid text: {path}")
        if dataset == "gandalf":
            # Dataset schema has 'similarity' but no verified binary labels.
            label = np.ones(len(frame), dtype=int)
        else:
            if "label" not in frame:
                raise ValueError(f"Source file missing label: {path}")
            label = pd.to_numeric(frame["label"], errors="raise").astype(int).to_numpy()
            if not np.isin(label, [0, 1]).all():
                raise ValueError(f"Source has nonbinary label: {path}")
        piece = pd.DataFrame({
            "row_id": [f"{dataset}:{path.name}:{i}" for i in range(len(frame))],
            "dataset": dataset,
            "label": label,
            "text": frame["text"].astype(str).tolist(),
        })
        pieces.append(piece)
        files_used.append(f"{dataset}: {path.relative_to(root)} ({len(frame)} rows)")
    if not pieces:
        raise FileNotFoundError("No recognized .parquet files in Project/Datasets/.")
    raw = pd.concat(pieces, ignore_index=True)
    if raw["row_id"].duplicated().any():
        raise ValueError("Source row IDs are not unique; inspect duplicate file paths")
    if set(raw["dataset"]) != set(DATASETS):
        raise ValueError(f"Expected all three raw datasets, found {set(raw['dataset'])}")
    return raw, files_used


def load_split(root: Path, name: str) -> pd.DataFrame:
    path = root / "results" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; do not recreate existing splits")
    df = pd.read_csv(path, dtype={"row_id": str})
    needed = {"row_id", "dataset", "split", "label", "text"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} lacks columns {needed-set(df.columns)}")
    if df["row_id"].duplicated().any() or (df["split"] != name).any():
        raise ValueError(f"Invalid IDs or split column in {path}")
    if df["text"].isna().any():
        raise ValueError(f"Missing text in {path}")
    return df


def load_fixed_partitions(root: Path):
    train = load_split(root, "train")
    validation = load_split(root, "validation")
    test = load_split(root, "test")
    split = pd.read_csv(root / "split.csv", dtype={"row_id": str})
    combined = pd.concat([train, validation, test], ignore_index=True)
    if combined["row_id"].duplicated().any():
        raise ValueError("Repeated row IDs across fixed partitions")
    if len(split) != len(combined) or set(split["row_id"]) != set(combined["row_id"]):
        raise ValueError("results split files disagree with split.csv; do not retrain")
    split_s = split.set_index("row_id")[["dataset", "split", "label"]].sort_index()
    all_s = combined.set_index("row_id")[["dataset", "split", "label"]].sort_index()
    if not split_s.equals(all_s):
        raise ValueError("Fixed split files have labels/dataset/split mismatches")
    if set(train["dataset"]) != {"safeguard"} or set(validation["dataset"]) != {"safeguard"}:
        raise ValueError("Expected Safe-Guard train and validation")
    return train, validation, test


def audit(root: Path):
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    train, validation, test = load_fixed_partitions(root)
    raw, used_files = load_raw_parquet(root)
    raw_ids = set(raw["row_id"])
    original = pd.concat([train, validation, test], ignore_index=True)
    if not set(original["row_id"]).issubset(raw_ids):
        raise ValueError("Raw Parquet file IDs do not match the original fixed split IDs")
    # Validate that source texts did not change after original splitting.
    raw_index = raw.set_index("row_id")
    for part in (train, validation, test):
        corresponding = raw_index.loc[part["row_id"]]
        if not np.array_equal(corresponding["text"].to_numpy(), part["text"].to_numpy()):
            raise ValueError("Text values in raw datasets and fixed splits differ")

    print(f"Raw audit: {len(raw):,} observations", flush=True)
    print(raw["dataset"].value_counts().to_string())
    print(f"Computing character {GRAM_SIZE}-gram sets (NFKC/casefold/whitespace normalization)...")
    t0 = time.perf_counter()
    gram_sets = [shingles(v) for v in raw["text"]]
    pairs, candidates_checked = find_similar_pairs(gram_sets, SIM_THRESHOLD)
    runtime = time.perf_counter() - t0
    print(f"Jaccard audit found {len(pairs):,} pairs after checking "
          f"{candidates_checked:,} candidates in {runtime:.2f}s")

    records = []
    within = {ds: [] for ds in DATASETS}
    cross = defaultdict(list)
    row_ids = raw["row_id"].tolist()
    datasets = raw["dataset"].tolist()
    labels = raw["label"].astype(int).tolist()
    text = raw["text"].tolist()
    for i, j, score in pairs:
        a, b = datasets[i], datasets[j]
        relation = "within" if a == b else "cross"
        item = {
            "row_id_a": row_ids[i], "row_id_b": row_ids[j],
            "dataset_a": a, "dataset_b": b,
            "label_a": labels[i], "label_b": labels[j],
            "similarity": round(score, 8),
            "same_exact_text": text[i] == text[j],
            "same_normalized_text": normalize(text[i]) == normalize(text[j]),
            "label_conflict": labels[i] != labels[j],
            "relation": relation,
        }
        records.append(item)
        if relation == "within":
            within[a].append((i, j))
        else:
            cross[tuple(sorted((a, b)))].append((i, j))

    pair_columns = ["row_id_a", "row_id_b", "dataset_a", "dataset_b", "label_a",
                    "label_b", "similarity", "same_exact_text", "same_normalized_text",
                    "label_conflict", "relation"]
    pair_df = pd.DataFrame(records, columns=pair_columns)
    pair_df.to_csv(result_dir / "jaccard_pairs.csv", index=False)

    within_rows = []
    for ds in DATASETS:
        ds_indices = raw.index[raw["dataset"] == ds].tolist()
        local = {idx: k for k, idx in enumerate(ds_indices)}
        components = DisjointSet(len(ds_indices))
        participants = set()
        for i, j in within[ds]:
            components.join(local[i], local[j])
            participants.update((i, j))
        frame = pair_df[(pair_df["relation"] == "within") & (pair_df["dataset_a"] == ds)]
        within_rows.append({
            "dataset": ds,
            "raw_rows": len(ds_indices),
            "jaccard_pairs": len(within[ds]),
            "pairs_beyond_normalized": int((~frame["same_normalized_text"]).sum()),
            "rows_in_near_duplicate_pairs": len(participants),
            "extra_rows_if_cluster_deduplicated": components.excess_rows(),
            "conflicting_label_pairs": int(frame["label_conflict"].sum()),
        })
    pd.DataFrame(within_rows).to_csv(result_dir / "jaccard_within_counts.csv", index=False)

    cross_rows = []
    for a, b in [("safeguard", "deepset"), ("safeguard", "gandalf"),
                 ("deepset", "gandalf")]:
        group = cross.get(tuple(sorted((a, b))), [])
        subset = pair_df[(pair_df["relation"] == "cross") &
                         (((pair_df["dataset_a"] == a) & (pair_df["dataset_b"] == b)) |
                          ((pair_df["dataset_a"] == b) & (pair_df["dataset_b"] == a)))]
        cross_rows.append({
            "dataset_pair": f"{a} vs {b}",
            "jaccard_pairs": len(group),
            "pairs_beyond_normalized": int((~subset["same_normalized_text"]).sum()),
            "conflicting_label_pairs": int(subset["label_conflict"].sum()),
        })
    pd.DataFrame(cross_rows).to_csv(result_dir / "jaccard_cross_counts.csv", index=False)

    # TRAIN-CLEANING POLICY: remove any train row near a held-out observation;
    # then greedily retain first train row among still-similar training rows.
    train_ids = train["row_id"].tolist()
    heldout = set(validation["row_id"]) | set(test["row_id"])
    training_set = set(train_ids)
    train_order = {rid: i for i, rid in enumerate(train_ids)}
    adjacency = defaultdict(list)
    heldout_matches: dict[str, tuple[str, float]] = {}
    for i, j, similarity in pairs:
        rid1, rid2 = row_ids[i], row_ids[j]
        if rid1 in training_set and rid2 in heldout:
            heldout_matches.setdefault(rid1, (rid2, similarity))
        if rid2 in training_set and rid1 in heldout:
            heldout_matches.setdefault(rid2, (rid1, similarity))
        if rid1 in training_set and rid2 in training_set:
            adjacency[rid1].append((rid2, similarity))
            adjacency[rid2].append((rid1, similarity))

    removed = []
    keep = set()
    for rid in train_ids:
        if rid in heldout_matches:
            example, sim = heldout_matches[rid]
            removed.append({"row_id": rid, "reason": "Jaccard match to fixed held-out data",
                            "example_match_row_id": example, "similarity": round(sim, 8)})
            continue
        matches_kept = [(rid2, sim) for rid2, sim in adjacency.get(rid, []) if rid2 in keep]
        if matches_kept:
            example, sim = max(matches_kept, key=lambda x: x[1])
            removed.append({"row_id": rid, "reason": "Jaccard match to retained training row",
                            "example_match_row_id": example, "similarity": round(sim, 8)})
            continue
        keep.add(rid)

    removed_df = pd.DataFrame(removed, columns=["row_id", "reason", "example_match_row_id", "similarity"])
    removed_df.to_csv(result_dir / "jaccard_train_removed.csv", index=False)
    if len(keep) + len(removed_df) != len(train):
        raise AssertionError("Training accounting mismatch")
    heldout_n = int((removed_df["reason"] == "Jaccard match to fixed held-out data").sum())
    internal_n = len(removed_df) - heldout_n
    report = [
        "COSC 4371 Project 17 - Jaccard similarity audit",
        f"Rule: NFKC + casefold + whitespace normalization, character {GRAM_SIZE}-gram SET Jaccard >= {SIM_THRESHOLD}",
        "Candidate join: exact Jaccard using frequency-ordered prefix filtering (not approximate LSH)",
        "Raw dataset files:", *[f"  {f}" for f in used_files],
        f"Raw rows audited: {len(raw)}", f"Total qualifying pair comparisons: {len(pairs)}",
        f"Candidate pairs verified with exact Jaccard: {candidates_checked}",
        f"Audit preprocessing/matching time: {runtime:.3f} seconds ({runtime / 60:.4f} minutes)",
        f"Original exact-cleaned Safe-Guard train: {len(train)}",
        f"Additional Jaccard removals matching fixed validation/test: {heldout_n}",
        f"Additional Jaccard removals matching retained train: {internal_n}",
        f"Jaccard-cleaned train: {len(keep)}",
        f"Fixed validation: {len(validation)}; fixed combined tests: {len(test)}",
        "Note: 'jaccard_pairs' includes exact and normalized matches; 'pairs_beyond_normalized' is incremental.",
        "Note: Gandalf's source labels are not binary ground truth; all prompts are treated as class 1 by this project.",
    ]
    (result_dir / "jaccard_audit_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report[-12:]))
    print("\nWithin-dataset counts:")
    print(pd.DataFrame(within_rows).to_string(index=False))
    print("\nCross-dataset counts:")
    print(pd.DataFrame(cross_rows).to_string(index=False))
    print("\nSaved Jaccard audit reports to results/; inspect them before running --train.")


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    negative = np.sort(scores[y_true == 0])[::-1]
    if not len(negative):
        raise ValueError("Validation has no negatives")
    allowed = int(np.floor(MAX_VALIDATION_FPR * len(negative)))
    return float(np.nextafter(negative[allowed], np.inf))


def model_pipeline() -> Pipeline:
    # Match compare_exact_normalized.py exactly.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", analyzer="word",
            ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)),
        ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED,
                                 max_iter=10_000)),
    ])


def train_and_score(root: Path):
    result_dir = root / "results"
    removal_path = result_dir / "jaccard_train_removed.csv"
    if not removal_path.exists():
        raise FileNotFoundError("Run 'python run_jaccard.py --audit' first")
    train, validation, test = load_fixed_partitions(root)
    removed = pd.read_csv(removal_path, dtype={"row_id": str})
    removed_ids = set(removed["row_id"])
    if not removed_ids.issubset(set(train["row_id"])):
        raise ValueError("Jaccard removal file includes unknown or non-training IDs")
    clean = train.loc[~train["row_id"].isin(removed_ids)].copy()
    if clean["label"].nunique() != 2:
        raise ValueError("Training data no longer includes both labels")
    print(f"Training Jaccard variant: {len(train):,} - {len(removed):,} = {len(clean):,} rows")

    model = model_pipeline()
    t0 = time.perf_counter()
    model.fit(clean["text"], clean["label"])
    training_seconds = time.perf_counter() - t0
    val_y = validation["label"].to_numpy(dtype=int)
    val_scores = np.asarray(model.decision_function(validation["text"]), dtype=float)
    threshold = choose_threshold(val_y, val_scores)
    validation_fpr = float(np.mean(val_scores[val_y == 0] >= threshold))
    assert validation_fpr <= MAX_VALIDATION_FPR + 1e-12

    test_scores = np.asarray(model.decision_function(test["text"]), dtype=float)
    result = test[["row_id", "dataset", "label"]].rename(columns={"dataset": "column"}).copy()
    result.insert(1, "value", "jaccard")
    result["score"] = test_scores
    result["threshold"] = threshold

    # Do not discard existing exact/normalized results.
    scores_path = root / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError("Expected existing scores.csv with exact/normalized results")
    old = pd.read_csv(scores_path, dtype={"row_id": str})
    required = {"row_id", "value", "column", "label", "score", "threshold"}
    if not required.issubset(old.columns):
        raise ValueError(f"scores.csv has incorrect format: {required - set(old.columns)}")
    if not {"exact", "normalized"}.issubset(set(old["value"])):
        raise ValueError("scores.csv lacks an original exact/normalized comparison")
    if not old["value"].isin(["exact", "normalized", "jaccard"]).all():
        raise ValueError("scores.csv has other values; check before modifying this file")
    old = old[old["value"] != "jaccard"].copy()
    expected = {(r.row_id, r.column) for r in result.itertuples(index=False)}
    for method in ("exact", "normalized"):
        part = old.loc[old["value"] == method]
        if {(r.row_id, r.column) for r in part.itertuples(index=False)} != expected:
            raise ValueError(f"Existing {method} test rows do not match Jaccard test rows")
    combined = pd.concat([old[list(required)], result[list(required)]], ignore_index=True)
    combined = combined[["row_id", "value", "column", "label", "score", "threshold"]]
    if combined.duplicated(["row_id", "value", "column"]).any():
        raise ValueError("Duplicated score rows")

    backup = result_dir / "scores_before_jaccard.csv"
    if not backup.exists():
        shutil.copy2(scores_path, backup)
    combined.to_csv(scores_path, index=False, float_format="%.17g")

    recalls = []
    for ds, frame in result.groupby("column", sort=True):
        y = frame["label"].to_numpy(dtype=int)
        predicted = frame["score"].to_numpy(dtype=float) >= threshold
        positive = y == 1
        recall = float(predicted[positive].mean()) if positive.any() else float("nan")
        recalls.append({"dataset": ds, "test_rows": len(frame),
                        "positives": int(positive.sum()), "recall": recall})
    pd.DataFrame(recalls).to_csv(result_dir / "jaccard_recall.csv", index=False)
    lines = [
        "Jaccard-cleaned baseline model: TF-IDF word (1,2)-grams + LinearSVC C=1 balanced",
        f"Training rows: {len(clean)}", f"Seed: {SEED}",
        f"Measured model fitting time: {training_seconds:.3f} seconds ({training_seconds / 60:.5f} minutes)",
        f"Validation FPR limit: {MAX_VALIDATION_FPR:.2%}",
        f"Threshold: {threshold:.17g}", f"Actual validation FPR: {validation_fpr:.4%}",
        f"Wrote {len(result)} Jaccard test predictions; total scores rows now {len(combined)}",
        "Test set and validation partitions are unchanged; split.csv was not modified.",
    ]
    (result_dir / "jaccard_model_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(pd.DataFrame(recalls).to_string(index=False))
    print("\nCheck with:")
    print("python check_final_checkpoint.py --split split.csv --scores scores.csv "
          "--errors errors.csv --metric recall --reference exact --direction either")
    print("Use '--direction either' only if that is the decision rule stated in your report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Audit raw Parquet data and prepare removals")
    mode.add_argument("--train", action="store_true", help="Train and extend scores.csv from audit")
    mode.add_argument("--all", action="store_true", help="Audit then train")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.audit or args.all:
        audit(root)
    if args.train or args.all:
        train_and_score(root)


if __name__ == "__main__":
    main()
