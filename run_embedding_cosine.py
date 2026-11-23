"""COSC 4371 / Project 17 - fourth duplicate-cleaning condition: embedding cosine.

Install dependencies (inside the VS Code terminal with your project interpreter):
  python -m pip install sentence-transformers torch pandas numpy scikit-learn pyarrow

Place THIS script next to run_jaccard.py and the fixed split.csv/scores.csv files.
Use the same publisher Parquet files in Project/Datasets/ and fixed CSVs in Project/results/.

  python run_embedding_cosine.py --audit
  python run_embedding_cosine.py --train
  # or, after inspecting the settings: python run_embedding_cosine.py --all

DO NOT change split.csv, validation.csv or test.csv. An additional cleaned train
subset is calculated from the *original* exact-cleaned 6,464-row training set,
not from the Jaccard-cleaned subset. The detector remains TF-IDF + LinearSVC.

This computes all pairwise cosine scores (blockwise exact matrix multiplication),
not an approximate nearest-neighbor or sampled audit. A cosine threshold of 0.95
is a PROPOSED strict near-duplicate rule, not a known optimal threshold; document
it as chosen before evaluating test results, and inspect matched pairs manually.
The embedding model is frozen and used ONLY for duplicate finding.

Produces (in results/):
  embedding_pairs.csv, embedding_pairs_review.csv,
  embedding_within_counts.csv, embedding_cross_counts.csv,
  embedding_train_removed.csv, embedding_audit_report.txt,
  embedding_audit_metadata.json, embedding_embeddings.npz (local cache),
  embedding_model_report.txt, embedding_recall.csv,
  scores_before_embedding.csv (backup, only at first training run)
Then appends/replaces the 'embedding' rows in scores.csv. Existing exact,
normalized and jaccard rows remain untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import warnings

import numpy as np
import pandas as pd

# Reuse exactly the same source-row parsing, fixed partitions, model features,
# classifier hyperparameters, random seed and validation thresholding policy.
from run_jaccard import (
    DATASETS, SEED, DisjointSet, choose_threshold, load_fixed_partitions,
    load_raw_parquet, model_pipeline, normalize,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COS_THRESHOLD = 0.95
DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 256
MAX_PAIRS = 2_000_000  # Guard against accidentally filling the disk with a broad threshold.
SCORE_COLUMNS = ["row_id", "value", "column", "label", "score", "threshold"]


def raw_digest(raw: pd.DataFrame) -> str:
    """Bind a cached embedding matrix to the exact source rows and their order."""
    h = hashlib.sha256()
    for rid, value in zip(raw["row_id"], raw["text"]):
        h.update(str(rid).encode("utf-8")); h.update(b"\0")
        h.update(str(value).encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def split_digest(root: Path) -> str:
    """Guard --train against changing the fixed split after --audit."""
    h = hashlib.sha256()
    for rel in ["split.csv", "results/train.csv", "results/validation.csv", "results/test.csv"]:
        h.update(rel.encode("utf-8")); h.update(b"\0")
        path = root / rel
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def get_embeddings(raw: pd.DataFrame, result_dir: Path, *, model_name: str,
                   revision: str | None, batch_size: int, device: str | None):
    """L2-normalized vectors. Cache is used only when data+model identity match."""
    digest = raw_digest(raw)
    cache = result_dir / "embedding_embeddings.npz"
    cache_meta = result_dir / "embedding_cache_metadata.json"
    identity = {"raw_sha256": digest, "model": model_name, "revision": revision,
                "normalize_text": "NFKC+casefold+collapse-whitespace"}
    if cache.exists() and cache_meta.exists():
        old = json.loads(cache_meta.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in identity.items()):
            with np.load(cache, allow_pickle=False) as obj:
                vecs = np.asarray(obj["embeddings"], dtype=np.float32)
            if vecs.shape[0] == len(raw) and np.isfinite(vecs).all():
                norms = np.linalg.norm(vecs, axis=1)
                if np.allclose(norms, 1.0, atol=1e-4):
                    print(f"Loaded cached vectors: {vecs.shape} from {cache.name}", flush=True)
                    return vecs, 0.0, old
            print("Cached embeddings are invalid; re-encoding.", flush=True)

    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
    except ImportError as exc:
        raise RuntimeError(
            "Missing embedding library. Run: python -m pip install sentence-transformers torch"
        ) from exc

    print(f"Loading frozen embedding model: {model_name}", flush=True)
    print("The first run may download model weights from Hugging Face.", flush=True)
    t0 = time.perf_counter()
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if device:
        kwargs["device"] = device
    encoder = SentenceTransformer(model_name, **kwargs)
    texts = [normalize(x) for x in raw["text"]]
    vectors = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    seconds = time.perf_counter() - t0
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(raw):
        raise RuntimeError(f"Unexpected embedding matrix dimensions: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Nonfinite embedding value detected")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-8):
        raise RuntimeError("Near-zero embedding vector found")
    vectors /= norms[:, None]
    info = {**identity, "shape": list(vectors.shape), "encoder_seconds": seconds,
            "max_seq_length": getattr(encoder, "max_seq_length", None),
            "sentence_transformers_version": getattr(sentence_transformers, "__version__", "unknown"),
            "python": platform.python_version(),
            "device": str(getattr(encoder, "device", device)),
            "batch_size": batch_size}
    np.savez_compressed(cache, embeddings=vectors)
    cache_meta.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Encoded {len(vectors):,} prompts in {seconds:.2f} seconds; "
          f"shape={vectors.shape}, device={info['device']}", flush=True)
    return vectors, seconds, info


def cosine_pairs(vectors: np.ndarray, threshold: float, block_size: int,
                 max_pairs: int = MAX_PAIRS):
    """Enumerate every unordered qualifying pair, using O(block_size*n) RAM.

    Multiplies normalized float32 embeddings so inner products are cosine scores.
    No approximation/candidate sampling; diagonal/lower triangle are excluded.
    """
    if not 0 < threshold <= 1.0:
        raise ValueError("Cosine threshold must be in (0,1]")
    if block_size < 1:
        raise ValueError("block-size must be positive")
    n = vectors.shape[0]
    pairs = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        scores = vectors[start:stop] @ vectors.T
        for offset, i in enumerate(range(start, stop)):
            # Consider only j > i. This scans each unordered pair exactly once.
            cols = np.flatnonzero(scores[offset, i + 1:] >= threshold) + (i + 1)
            if len(pairs) + len(cols) > max_pairs:
                raise RuntimeError(
                    f"More than {max_pairs:,} qualifying pairs. Review whether "
                    f"cosine threshold {threshold} is too permissive before proceeding."
                )
            pairs.extend((i, int(j), min(1.0, float(scores[offset, j]))) for j in cols)
        print(f"  Cosine scanned {stop:,}/{n:,} prompts; "
              f"{len(pairs):,} qualifying pairs", flush=True)
    return pairs


def assert_same_source_rows(raw: pd.DataFrame, train: pd.DataFrame,
                            validation: pd.DataFrame, test: pd.DataFrame):
    ids = set(raw["row_id"])
    full = pd.concat([train, validation, test], ignore_index=True)
    if not set(full["row_id"]).issubset(ids):
        raise ValueError("Fixed split row IDs do not match publisher Parquet downloads")
    index = raw.set_index("row_id")
    for frame in (train, validation, test):
        if not np.array_equal(index.loc[frame["row_id"], "text"].to_numpy(),
                              frame["text"].to_numpy()):
            raise ValueError("Raw texts differ from original fixed split texts")


def audit(root: Path, *, model_name: str, revision: str | None,
          threshold: float, block_size: int, batch_size: int, device: str | None):
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    train, validation, test = load_fixed_partitions(root)
    raw, source_files = load_raw_parquet(root)
    assert_same_source_rows(raw, train, validation, test)
    print(f"Raw embedding audit: {len(raw):,} observations", flush=True)
    print(raw["dataset"].value_counts().to_string())
    print(f"Cosine threshold: {threshold:.4f} (proposed, fixed before test evaluation)")
    vectors, encode_seconds, cache_info = get_embeddings(
        raw, result_dir, model_name=model_name, revision=revision,
        batch_size=batch_size, device=device)
    t0 = time.perf_counter()
    pairs = cosine_pairs(vectors, threshold, block_size)
    compare_seconds = time.perf_counter() - t0
    print(f"Embedding cosine found {len(pairs):,} qualifying pairs "
          f"in {compare_seconds:.2f}s, excluding encoding.")

    row_ids = raw["row_id"].tolist()
    datasets = raw["dataset"].tolist()
    labels = raw["label"].astype(int).tolist()
    texts = raw["text"].tolist()
    jaccard_path = result_dir / "jaccard_pairs.csv"
    jaccard_keys = None
    if jaccard_path.is_file():
        jaccard_df = pd.read_csv(jaccard_path, dtype={"row_id_a": str, "row_id_b": str})
        jaccard_keys = {
            tuple(sorted((r.row_id_a, r.row_id_b)))
            for r in jaccard_df.itertuples(index=False)
        }

    records = []
    within = {ds: [] for ds in DATASETS}
    cross = defaultdict(list)
    train_ids = train["row_id"].tolist()
    training_ids = set(train_ids)
    heldout_ids = set(validation["row_id"]) | set(test["row_id"])
    adj = defaultdict(list)
    heldout_matches = {}

    for i, j, score in pairs:
        ai, aj = row_ids[i], row_ids[j]
        ds_a, ds_b = datasets[i], datasets[j]
        key = tuple(sorted((ai, aj)))
        row = {
            "row_id_a": ai, "row_id_b": aj,
            "dataset_a": ds_a, "dataset_b": ds_b,
            "label_a": labels[i], "label_b": labels[j],
            "similarity": round(score, 8),
            "same_exact_text": texts[i] == texts[j],
            "same_normalized_text": normalize(texts[i]) == normalize(texts[j]),
            "beyond_jaccard": None if jaccard_keys is None else key not in jaccard_keys,
            "label_conflict": labels[i] != labels[j],
            "relation": "within" if ds_a == ds_b else "cross",
        }
        records.append(row)
        if ds_a == ds_b:
            within[ds_a].append((i, j))
        else:
            cross[tuple(sorted((ds_a, ds_b)))].append((i, j))

        if ai in training_ids and aj in heldout_ids:
            # When multiple matches exist, retain the highest cosine example.
            if ai not in heldout_matches or score > heldout_matches[ai][1]:
                heldout_matches[ai] = (aj, score)
        if aj in training_ids and ai in heldout_ids:
            if aj not in heldout_matches or score > heldout_matches[aj][1]:
                heldout_matches[aj] = (ai, score)
        if ai in training_ids and aj in training_ids:
            adj[ai].append((aj, score))
            adj[aj].append((ai, score))

    pair_columns = ["row_id_a", "row_id_b", "dataset_a", "dataset_b", "label_a",
                    "label_b", "similarity", "same_exact_text", "same_normalized_text",
                    "beyond_jaccard", "label_conflict", "relation"]
    pair_df = pd.DataFrame(records, columns=pair_columns)
    pair_df.to_csv(result_dir / "embedding_pairs.csv", index=False)

    # Curated local review sample: contradictory labels, cross-dataset overlaps,
    # and distinct-normalized-text pairs are especially important to inspect.
    # This is a REVIEW sample, not a basis for computing duplicate totals.
    review_sections = []
    if not pair_df.empty:
        sorted_pairs = pair_df.sort_values(["similarity", "row_id_a"], ascending=[False, True])
        review_sections = [
            sorted_pairs[sorted_pairs["label_conflict"]].head(12),
            sorted_pairs[(sorted_pairs["relation"] == "cross") &
                         (~sorted_pairs["same_normalized_text"])].head(12),
            sorted_pairs[(sorted_pairs["relation"] == "within") &
                         (~sorted_pairs["same_normalized_text"])].head(12),
        ]
    if review_sections:
        preview = pd.concat(review_sections).drop_duplicates(["row_id_a", "row_id_b"]).head(35).copy()
    else:
        preview = pair_df.head(0).copy()
    text_by_id = dict(zip(row_ids, texts))
    preview["text_a_excerpt"] = [text_by_id[r][:240] for r in preview["row_id_a"]]
    preview["text_b_excerpt"] = [text_by_id[r][:240] for r in preview["row_id_b"]]
    preview.to_csv(result_dir / "embedding_pairs_review.csv", index=False)
    within_rows = []
    for ds in DATASETS:
        ds_indices = raw.index[raw["dataset"] == ds].tolist()
        index_map = {raw_idx: k for k, raw_idx in enumerate(ds_indices)}
        components = DisjointSet(len(ds_indices))
        participants = set()
        for i, j in within[ds]:
            components.join(index_map[i], index_map[j])
            participants.update((i, j))
        group = pair_df[(pair_df["relation"] == "within") &
                        (pair_df["dataset_a"] == ds)]
        within_rows.append({
            "dataset": ds, "raw_rows": len(ds_indices),
            "embedding_pairs": len(within[ds]),
            "pairs_beyond_normalized": int((~group["same_normalized_text"]).sum()),
            "pairs_beyond_jaccard": (
                "not checked" if jaccard_keys is None else int(group["beyond_jaccard"].sum())),
            "rows_in_near_duplicate_pairs": len(participants),
            "extra_rows_if_cluster_deduplicated": components.excess_rows(),
            "conflicting_label_pairs": int(group["label_conflict"].sum()),
        })
    pd.DataFrame(within_rows).to_csv(result_dir / "embedding_within_counts.csv", index=False)

    cross_rows = []
    for a, b in [("safeguard", "deepset"), ("safeguard", "gandalf"),
                 ("deepset", "gandalf")]:
        group = cross.get(tuple(sorted((a, b))), [])
        subset = pair_df[(pair_df["relation"] == "cross") &
                         (((pair_df["dataset_a"] == a) & (pair_df["dataset_b"] == b)) |
                          ((pair_df["dataset_a"] == b) & (pair_df["dataset_b"] == a)))]
        cross_rows.append({
            "dataset_pair": f"{a} vs {b}",
            "embedding_pairs": len(group),
            "pairs_beyond_normalized": int((~subset["same_normalized_text"]).sum()),
            "pairs_beyond_jaccard": (
                "not checked" if jaccard_keys is None else int(subset["beyond_jaccard"].sum())),
            "conflicting_label_pairs": int(subset["label_conflict"].sum()),
        })
    pd.DataFrame(cross_rows).to_csv(result_dir / "embedding_cross_counts.csv", index=False)

    # Cleaning rule is exactly the same as in run_jaccard.py: first remove any
    # train/held-out match; then greedily keep first train row among matches.
    removed = []
    kept = set()
    for rid in train_ids:
        if rid in heldout_matches:
            example, sim = heldout_matches[rid]
            removed.append({"row_id": rid, "reason": "Embedding match to fixed held-out data",
                            "example_match_row_id": example, "similarity": round(sim, 8)})
            continue
        matching_kept = [(r, s) for r, s in adj.get(rid, []) if r in kept]
        if matching_kept:
            example, sim = max(matching_kept, key=lambda x: x[1])
            removed.append({"row_id": rid, "reason": "Embedding match to retained training row",
                            "example_match_row_id": example, "similarity": round(sim, 8)})
            continue
        kept.add(rid)
    removed_df = pd.DataFrame(removed,
                              columns=["row_id", "reason", "example_match_row_id", "similarity"])
    removed_df.to_csv(result_dir / "embedding_train_removed.csv", index=False)
    if len(kept) + len(removed_df) != len(train):
        raise AssertionError("Embedding cleaning row accounting mismatch")
    for a, b, _ in pairs:
        x, y = row_ids[a], row_ids[b]
        if (x in kept and y in heldout_ids) or (y in kept and x in heldout_ids):
            raise AssertionError("Embedding-cleaned train still matches held-out data")
        if x in kept and y in kept:
            raise AssertionError("Embedding-cleaned train still contains a similar pair")

    heldout_n = int((removed_df["reason"] == "Embedding match to fixed held-out data").sum())
    internal_n = len(removed_df) - heldout_n
    metadata = {
        "embedding_model": model_name, "model_revision": revision,
        "cosine_threshold": threshold, "split_sha256": split_digest(root),
        "raw_sha256": raw_digest(raw), "raw_rows": len(raw),
        "total_embedding_pairs": len(pairs), "encoding_seconds": encode_seconds,
        "matching_seconds": compare_seconds, "auditing_seconds": encode_seconds + compare_seconds,
        "training_rows_before": len(train), "heldout_removals": heldout_n,
        "within_train_removals": internal_n, "training_rows_after": len(kept),
        "heldout_validation_rows": len(validation), "fixed_test_rows": len(test),
        "encoder_info": cache_info,
        "initial_encoder_seconds_if_cached": cache_info.get("encoder_seconds"),
    }
    (result_dir / "embedding_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    details = [
        "COSC 4371 - Project 17 - Embedding cosine similarity audit",
        f"Frozen model: {model_name}; revision: {revision or '(unpinned default)'}",
        f"Normalization: NFKC+casefold+collapsed whitespace; cosine >= {threshold:.4f}",
        "All unordered raw pairs compared by blockwise exact cosine (not ANN/MinHash)",
        "Source files:", *[f"  {s}" for s in source_files],
        f"Raw observations audited: {len(raw):,}",
        f"Qualifying embedding pairs across all raw data: {len(pairs):,}",
        f"Encoding time in this run: {encode_seconds:.3f}s ({encode_seconds/60:.4f} min; 0 if cached)",
        f"Initial cached encoding time: {cache_info.get('encoder_seconds')}s (if available)",
        f"Pairwise cosine time: {compare_seconds:.3f}s ({compare_seconds/60:.4f} min)",
        f"Original exact-cleaned training rows: {len(train):,}",
        f"Removed for held-out overlap: {heldout_n:,}",
        f"Removed for within-training matches: {internal_n:,}",
        f"Final embedding-cleaned train: {len(kept):,}",
        f"Validation/test unchanged: {len(validation):,} / {len(test):,}",
        f"Model max_seq_length: {cache_info.get('max_seq_length')} tokens (long prompts may truncate)",
        "Gandalf source has no verified binary labels; project assigns positive class to all its prompts.",
        "IMPORTANT: High embedding cosine does not establish identical intent or compatible labels.",
        "Manually inspect examples in embedding_pairs.csv before interpreting near-duplicates.",
    ]
    (result_dir / "embedding_audit_report.txt").write_text(
        "\n".join(details) + "\n", encoding="utf-8")
    print("\n" + "\n".join(details))
    print("\nWithin-dataset counts:")
    print(pd.DataFrame(within_rows).to_string(index=False))
    print("\nCross-dataset counts:")
    print(pd.DataFrame(cross_rows).to_string(index=False))
    removed_pct = len(removed_df) / len(train)
    if removed_pct >= 0.20:
        warnings.warn(
            f"Embedding rule removes {removed_pct:.1%} of baseline training data. "
            "Inspect similarity examples and decide whether the threshold is defensible "
            "BEFORE using held-out test performance.", stacklevel=2)
    print("\nAudit is complete. Inspect results/embedding_* counts and removals "
          "before running --train.")


def train_and_score(root: Path):
    result_dir = root / "results"
    removal_path = result_dir / "embedding_train_removed.csv"
    metadata_path = result_dir / "embedding_audit_metadata.json"
    if not removal_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run python run_embedding_cosine.py --audit first")
    audit_info = json.loads(metadata_path.read_text(encoding="utf-8"))
    if audit_info.get("split_sha256") != split_digest(root):
        raise ValueError("Fixed split files changed since embedding audit; do NOT train")
    train, validation, test = load_fixed_partitions(root)
    removed = pd.read_csv(removal_path, dtype={"row_id": str})
    if removed["row_id"].duplicated().any():
        raise ValueError("Duplicated row IDs in embedding_train_removed.csv")
    removed_ids = set(removed["row_id"])
    if not removed_ids.issubset(set(train["row_id"])):
        raise ValueError("Removal file has IDs outside original fixed training partition")
    if len(removed) != (audit_info["heldout_removals"] + audit_info["within_train_removals"]):
        raise ValueError("Removal file length differs from audit summary")
    clean = train.loc[~train["row_id"].isin(removed_ids)].copy()
    if len(clean) != audit_info["training_rows_after"]:
        raise ValueError("Cleaned training count disagrees with audit metadata")
    if clean["label"].nunique() != 2:
        raise ValueError("Embedding cleaning left only one class in training")
    print(f"Fitting embedding-cleaned variant: {len(train):,} - {len(removed):,} "
          f"= {len(clean):,} train rows", flush=True)

    model = model_pipeline()  # identical TF-IDF/LinearSVC as other conditions
    t0 = time.perf_counter()
    model.fit(clean["text"], clean["label"])
    fit_sec = time.perf_counter() - t0
    y_val = validation["label"].to_numpy(dtype=int)
    val_scores = np.asarray(model.decision_function(validation["text"]), dtype=float)
    threshold = choose_threshold(y_val, val_scores)
    val_fpr = float(np.mean(val_scores[y_val == 0] >= threshold))
    if val_fpr > 0.05 + 1e-12:
        raise AssertionError("Validation false-positive-rate constraint not met")
    test_scores = np.asarray(model.decision_function(test["text"]), dtype=float)
    new = test[["row_id", "dataset", "label"]].rename(columns={"dataset": "column"}).copy()
    new.insert(1, "value", "embedding")
    new["score"] = test_scores
    new["threshold"] = threshold
    new = new[SCORE_COLUMNS]

    scores_path = root / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError("Existing scores.csv is missing. Do not regenerate the original runs")
    old = pd.read_csv(scores_path, dtype={"row_id": str})
    missing = set(SCORE_COLUMNS) - set(old.columns)
    if missing:
        raise ValueError(f"scores.csv missing columns: {missing}")
    values = set(old["value"])
    if not {"exact", "normalized", "jaccard"}.issubset(values):
        raise ValueError("Expected complete exact/normalized/jaccard scores.csv")
    if not values.issubset({"exact", "normalized", "jaccard", "embedding"}):
        raise ValueError(f"Unexpected cleaning values: {values}")
    old = old.loc[old["value"] != "embedding", SCORE_COLUMNS].copy()
    expected_rows = set(zip(new["row_id"], new["column"], new["label"]))
    for method in ("exact", "normalized", "jaccard"):
        part = old.loc[old["value"] == method]
        test_rows = set(zip(part["row_id"], part["column"], part["label"]))
        if len(part) != len(new) or test_rows != expected_rows:
            raise ValueError(f"Existing {method} scored rows or labels differ from fixed tests")
    combined = pd.concat([old, new], ignore_index=True)
    if combined.duplicated(["row_id", "value", "column"]).any():
        raise ValueError("Duplicate scored rows detected")
    backup = result_dir / "scores_before_embedding.csv"
    if not backup.exists():
        shutil.copy2(scores_path, backup)
    tmp = root / "scores_embedding_tmp.csv"
    try:
        combined.to_csv(tmp, index=False, float_format="%.17g")
        tmp.replace(scores_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    recalls = []
    for dataset, group in new.groupby("column", sort=True):
        y = group["label"].to_numpy(dtype=int)
        pred = group["score"].to_numpy(dtype=float) >= threshold
        pos = y == 1
        recall = float(pred[pos].mean()) if pos.any() else float("nan")
        recalls.append({"dataset": dataset, "test_rows": len(group),
                        "positive_rows": int(pos.sum()), "recall": recall})
    pd.DataFrame(recalls).to_csv(result_dir / "embedding_recall.csv", index=False)
    model_report = [
        "COSC 4371 Project 17 - Embedding cosine-cleaned TF-IDF + Linear SVM",
        f"Cleaning encoder: {audit_info['embedding_model']}",
        f"Cleaning threshold: cosine >= {audit_info['cosine_threshold']:.4f}",
        f"Training observations: {len(clean):,}; random seed {SEED}",
        f"Measured classifier fitting time: {fit_sec:.4f}s ({fit_sec / 60:.5f} min)",
        "Classifier: word TF-IDF (1,2)-grams, max_features=50000, min_df=2, sublinear_tf=True,",
        "           lowercase=True, strip_accents='unicode'; LinearSVC C=1 balanced",
        f"Chosen on fixed validation: threshold {threshold:.17g}",
        f"Validation FPR: {val_fpr:.4%} (require <= 5%)",
        f"Scored {len(new):,} unchanged test rows; scores.csv now {len(combined):,} rows",
        "split.csv, validation.csv, and test.csv are unchanged.",
    ]
    (result_dir / "embedding_model_report.txt").write_text(
        "\n".join(model_report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(model_report))
    print("\n" + pd.DataFrame(recalls).to_string(index=False))
    print("\nRun the supplied final checker with:")
    print("python check_final_checkpoint.py --split split.csv --scores scores.csv "
          "--errors errors.csv --metric recall --reference exact --direction either")
    print("Use direction=either only if disclosed in your report's decision rule.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Audit raw sources and prepare training removals")
    mode.add_argument("--train", action="store_true", help="Train using audited removals and extend scores.csv")
    mode.add_argument("--all", action="store_true", help="Both audit and train")
    parser.add_argument("--cosine-threshold", type=float, default=DEFAULT_COS_THRESHOLD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None,
                        help="Optional pinned model commit/revision for fully repeatable embeddings")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()
    if not 0 < args.cosine_threshold <= 1:
        parser.error("--cosine-threshold must lie between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    root = Path(__file__).resolve().parent
    if args.audit or args.all:
        audit(root, model_name=args.model, revision=args.revision,
              threshold=args.cosine_threshold, block_size=args.block_size,
              batch_size=args.batch_size, device=args.device)
    if args.train or args.all:
        train_and_score(root)


if __name__ == "__main__":
    main()
