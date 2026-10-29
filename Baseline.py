import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import combinations

BASE_DIR = Path(__file__).resolve().parent

# Find all downloaded Parquet files
parquet_files = sorted(BASE_DIR.rglob("*.parquet"))

datasets = {
    "safeguard": [],
    "deepset": [],
    "gandalf": []
}

# Organize files by dataset
for file in parquet_files:
    folder = file.parent.name.lower()

    if "safe-guard" in folder:
        name = "safeguard"
    elif "deepset" in folder:
        name = "deepset"
    elif "gandalf" in folder:
        name = "gandalf"
    else:
        continue

    df = pd.read_parquet(file)
    df["source_file"] = file.name
    datasets[name].append(df)

# Combine partitions belonging to each dataset
for name in datasets:
    if not datasets[name]:
        raise FileNotFoundError(
            f"No Parquet files found for {name}"
        )

    datasets[name] = pd.concat(
        datasets[name],
        ignore_index=True
    )


# 1. Within-dataset duplicate audit
print("\n===== WITHIN-DATASET AUDIT =====")

counters = {}

for name, df in datasets.items():

    print(f"\nDataset: {name}")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("Missing text:", df["text"].isna().sum())

    if "label" in df.columns:
        print("Class counts:")
        print(df["label"].value_counts())

    counts = Counter(df["text"].dropna())
    counters[name] = counts

    duplicate_rows = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    duplicate_pairs = sum(
        count * (count - 1) // 2
        for count in counts.values()
        if count > 1
    )

    print("Duplicate extra rows:", duplicate_rows)
    print("Duplicate pairs:", duplicate_pairs)


# 2. Cross-dataset exact duplicates
print("\n===== CROSS-DATASET AUDIT =====")

for name1, name2 in combinations(datasets, 2):

    counts1 = counters[name1]
    counts2 = counters[name2]

    shared = set(counts1) & set(counts2)

    pairs = sum(
        counts1[text] * counts2[text]
        for text in shared
    )

    rows1 = sum(counts1[text] for text in shared)
    rows2 = sum(counts2[text] for text in shared)

    print(f"\n{name1} vs {name2}")
    print("Distinct shared texts:", len(shared))
    print("Cross-dataset duplicate pairs:", pairs)
    print("Rows involved in first dataset:", rows1)
    print("Rows involved in second dataset:", rows2)


#final verification of duplicates
print("\n===== SPLIT LEAKAGE CHECK =====")

for name, df in datasets.items():

    # Extract train/test/validation from filename
    temp = df.copy()
    temp["partition"] = temp["source_file"].str.split(
        "-", n=1
    ).str[0]

    # Find texts appearing in multiple partitions
    partition_counts = (
        temp.groupby("text")["partition"].nunique()
    )

    overlapping = (partition_counts > 1).sum()

    print(f"\n{name}")
    print("Texts crossing partitions:", overlapping)

    # Check identical texts with conflicting labels
    if "label" in temp.columns:
        label_counts = (
            temp.groupby("text")["label"].nunique()
        )

        conflicts = (label_counts > 1).sum()

        print("Texts with conflicting labels:", conflicts)


# 4. Check labels for cross-dataset matches
print("\n===== CROSS-DATASET LABEL CHECK =====")

safeguard = datasets["safeguard"][
    ["text", "label"]
].drop_duplicates()

deepset = datasets["deepset"][
    ["text", "label"]
].drop_duplicates()

shared = safeguard.merge(
    deepset,
    on="text",
    suffixes=("_safeguard", "_deepset")
)

print("Shared labeled records:", len(shared))

conflicts = shared[
    shared["label_safeguard"]
    != shared["label_deepset"]
]

print("Cross-dataset label conflicts:", len(conflicts))

#5. inspection of conficting labels
print("\n===== SAFEGUARD LABEL CONFLICTS =====")

safe = datasets["safeguard"].copy()
deep = datasets["deepset"].copy()

# Find texts with more than one label
conflict_counts = safe.groupby("text")["label"].nunique()

conflicting_texts = conflict_counts[
    conflict_counts > 1
].index

for i, text in enumerate(conflicting_texts, 1):
    print(f"\nConflict {i}:")
    print("Text:", repr(text[:150]))

    matches = safe[safe["text"] == text]

    print(matches[["label", "source_file"]].to_string(index=False))

#6. cross-dataset label conflict check

print("\n===== CROSS-DATASET LABEL COMPARISON =====")

shared = safe[["text", "label"]].drop_duplicates().merge(
    deep[["text", "label"]].drop_duplicates(),
    on="text",
    suffixes=("_safeguard", "_deepset")
)

for i, row in enumerate(shared.itertuples(), 1):
    print(f"\nShared text {i}:")
    print("Text:", repr(row.text[:150]))
    print("Safe-Guard label:", row.label_safeguard)
    print("Deepset label:", row.label_deepset)

    if row.label_safeguard != row.label_deepset:
        print("STATUS: LABEL CONFLICT")
    else:
        print("STATUS: LABELS AGREE")


import re
import unicodedata

# Normalize capitalization and whitespace
def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\n===== NORMALIZED DUPLICATE AUDIT =====")

normalized_counts = {}

for name, df in datasets.items():
    normalized = df["text"].map(normalize_text)

    exact_count = df.duplicated(subset=["text"]).sum()
    normalized_count = normalized.duplicated().sum()

    normalized_counts[name] = normalized.value_counts()

    print(f"\nDataset: {name}")
    print("Exact duplicate extra rows:", exact_count)
    print("Normalized duplicate extra rows:", normalized_count)
    print("Additional matches:", normalized_count - exact_count)

print("\n===== CROSS-DATASET NORMALIZED MATCHES =====")

from itertools import combinations

for a, b in combinations(datasets.keys(), 2):
    c1 = normalized_counts[a]
    c2 = normalized_counts[b]

    shared = set(c1.index) & set(c2.index)

    pairs = sum(
        int(c1[t]) * int(c2[t])
        for t in shared
    )

    print(f"{a} vs {b}: {pairs} duplicate pairs")