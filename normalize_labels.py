#!/usr/bin/env python3
"""
Normalize labels in the metadata CSV: if a label contains more than one word,
replace it with its last word. Backs up the original file before overwriting.

Usage:
  python normalize_labels.py [--metadata path/to/gen_metadata.csv]
"""
import argparse
import os
import sys
from datetime import datetime

try:
    import pandas as pd
except Exception as e:
    print("Required package pandas is not available:", e)
    sys.exit(2)


def find_metadata(default_path):
    if os.path.isfile(default_path):
        return default_path
    # try searching upward from current dir
    for root, dirs, files in os.walk('.', topdown=True):
        if 'gen_metadata.csv' in files:
            return os.path.join(root, 'gen_metadata.csv')
    return None


def choose_label_column(df):
    # Prefer existing 'label' column, else try common alternatives
    candidates = [c for c in df.columns if c.lower() in ('label', 'app', 'class', 'labels')]
    if candidates:
        return candidates[0]
    # fallback to first string-like column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def normalize_value(v):
    if not isinstance(v, str):
        return v
    parts = v.strip().split()
    if len(parts) > 1:
        return parts[-1]
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', '-m', default='./data/gen_metadata.csv', help='Path to metadata CSV (default: ./data/gen_metadata.csv)')
    args = p.parse_args()

    path = args.metadata
    if not os.path.isfile(path):
        found = find_metadata(path)
        if found:
            path = found
            print('Found metadata at', path)
        else:
            print('Metadata file not found at', args.metadata)
            sys.exit(1)

    # Try reading CSV with multiple encodings; if that fails, try Excel format
    df = None
    read_errors = []
    try:
        df = pd.read_csv(path)
    except Exception as e1:
        read_errors.append(str(e1))
        try:
            df = pd.read_csv(path, encoding='latin1')
        except Exception as e2:
            read_errors.append(str(e2))
            try:
                # Sometimes the file is actually an Excel file
                df = pd.read_excel(path)
            except Exception as e3:
                read_errors.append(str(e3))
    if df is None:
        print('Failed to read metadata file with utf-8, latin1, and as Excel.')
        print('Errors:')
        for err in read_errors:
            print(' -', err)
        sys.exit(1)
    col = choose_label_column(df)
    if col is None:
        print('Could not determine a label column in', path)
        print('Columns:', df.columns.tolist())
        sys.exit(1)

    print('Using label column:', col)
    before_unique = df[col].nunique(dropna=True)
    # Apply normalization
    changed_mask = df[col].astype(str).str.split().apply(lambda parts: len(parts) > 1)
    # compute sample before/after
    df_before = df.loc[changed_mask, col].astype(str).copy()
    df.loc[changed_mask, col] = df.loc[changed_mask, col].astype(str).apply(normalize_value)
    df_after = df.loc[changed_mask, col].astype(str).copy()

    after_unique = df[col].nunique(dropna=True)
    changed_count = changed_mask.sum()

    # backup
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path + '.bak.' + ts
    os.rename(path, backup)
    df.to_csv(path, index=False)

    print(f'Wrote normalized metadata to: {path}')
    print(f'Backup of original file at: {backup}')
    print(f'Label column: {col}')
    print(f'Entries changed (multi-word -> last word): {changed_count}')
    print(f'Unique labels before: {before_unique}  after: {after_unique}')
    if changed_count>0:
        print('\nSample changes (before -> after):')
        for b,a in zip(df_before.head(10).tolist(), df_after.head(10).tolist()):
            print(f"  {b}  ->  {a}")

if __name__ == '__main__':
    main()
