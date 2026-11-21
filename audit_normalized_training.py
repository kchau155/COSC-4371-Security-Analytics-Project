"""Read-only normalized duplicate audit for the fixed midsemester splits.
Place in the project root beside split.csv and results/ and run with Python.
"""
from pathlib import Path
import re
import unicodedata
import pandas as pd


def normalize_text(text):
    value = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', value).strip()


def main():
    candidates = [Path(__file__).resolve().parent, Path.cwd()]
    root = next((p for p in candidates
                 if all((p / 'results' / f'{s}.csv').is_file()
                        for s in ('train', 'validation', 'test'))), None)
    if root is None:
        raise FileNotFoundError('Put this script in your Project folder containing results/train.csv, results/validation.csv and results/test.csv')

    train = pd.read_csv(root / 'results' / 'train.csv')
    val = pd.read_csv(root / 'results' / 'validation.csv')
    test = pd.read_csv(root / 'results' / 'test.csv')
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        if not {'text', 'label'}.issubset(frame.columns):
            raise ValueError(f'{name}.csv must have text and label columns')
        if frame['text'].isna().any():
            raise ValueError(f'{name}.csv contains null text')

    train = train.copy()
    train['_norm'] = train['text'].map(normalize_text)
    val_keys = set(val['text'].map(normalize_text))
    test_keys = set(test['text'].map(normalize_text))
    match_val = train['_norm'].isin(val_keys)
    match_test = train['_norm'].isin(test_keys)
    match_either = match_val | match_test

    candidates = train.loc[~match_either]
    after_clean = candidates.drop_duplicates('_norm', keep='first')
    conflicts = int((train.groupby('_norm')['label'].nunique() > 1).sum())

    print('===== FIXED-SPLIT NORMALIZED DUPLICATE AUDIT =====')
    print(f'Project folder: {root}')
    print(f'Original exact-cleaned training rows: {len(train)}')
    print(f'Additional normalized duplicates inside training: {int(train["_norm"].duplicated().sum())}')
    print(f'Training rows matching validation after normalization: {int(match_val.sum())}')
    print(f'Training rows matching test after normalization: {int(match_test.sum())}')
    print(f'Training rows matching either held-out set: {int(match_either.sum())}')
    print(f'Duplicate rows inside training after removing held-out matches: {int(candidates["_norm"].duplicated().sum())}')
    print(f'Normalized text groups in training with conflicting labels: {conflicts}')
    print(f'Candidate cleaned training rows: {len(after_clean)}')
    print(f'Additional training rows removed: {len(train)-len(after_clean)}')
    print('No files were modified.')
    if conflicts:
        print('WARNING: Resolve normalized-text label conflicts before training.')


if __name__ == '__main__':
    main()
