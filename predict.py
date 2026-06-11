"""
SpliceLoc-ML — Prediction Script
=================================
Loads the trained models and classifies variants into the five-tier ACMG system.

Usage:
    python predict.py --input your_variants.csv --output predictions.csv

Input CSV must contain these columns (produced by VEP + Pangolin annotation):
    MaxEntScan_alt, MaxEntScan_ref, MaxEntScan_diff,
    CADD_PHRED,
    SpliceAI_pred_DS_AG, SpliceAI_pred_DS_AL,
    SpliceAI_pred_DS_DG, SpliceAI_pred_DS_DL,
    max_splice_val, ada_score, rf_score,
    GC_content, PPT_density, Pangolin_score,
    Category  (one of: 'Exonic', 'Near Splice (3-20 bp)', 'Deep Intronic (>20 bp)')
"""

import pickle
import argparse
import pandas as pd
import numpy as np
import os

# ── Feature list ──────────────────────────────────────────────────────────────
FEATURES = [
    'MaxEntScan_alt', 'MaxEntScan_ref', 'MaxEntScan_diff',
    'CADD_PHRED',
    'SpliceAI_pred_DS_AG', 'SpliceAI_pred_DS_AL',
    'SpliceAI_pred_DS_DG', 'SpliceAI_pred_DS_DL',
    'max_splice_val', 'ada_score', 'rf_score',
    'GC_content', 'PPT_density', 'Pangolin_score'
]

CATEGORIES = ['Exonic', 'Near Splice (3-20 bp)', 'Deep Intronic (>20 bp)']


def load_models(model_dir='models'):
    """Load trained models and thresholds from the models/ directory."""
    with open(os.path.join(model_dir, 'rf_models.pkl'), 'rb') as f:
        models = pickle.load(f)
    with open(os.path.join(model_dir, 'rf_cutoffs_binary.pkl'), 'rb') as f:
        binary_cutoffs = pickle.load(f)
    with open(os.path.join(model_dir, 'rf_cutoffs_5tier.pkl'), 'rb') as f:
        tier_cutoffs = pickle.load(f)
    with open(os.path.join(model_dir, 'rf_scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    return models, binary_cutoffs, tier_cutoffs, scalers


def classify_5tier(probability, thresholds):
    """Map a posterior probability to the five-tier ACMG classification."""
    if probability >= thresholds['T_pathogenic']:
        return 'Pathogenic'
    elif probability >= thresholds['T_likely_path']:
        return 'Likely Pathogenic'
    elif probability >= thresholds['T_vus_lower']:
        return 'VUS'
    elif probability >= thresholds['T_benign']:
        return 'Likely Benign'
    else:
        return 'Benign'


def predict(input_csv, output_csv, model_dir='models'):
    """
    Run SpliceLoc-ML on an annotated variant CSV file.

    Parameters
    ----------
    input_csv  : path to input CSV with VEP + Pangolin annotated variants
    output_csv : path to save predictions
    model_dir  : path to folder containing the .pkl model files
    """
    print(f"Loading models from {model_dir}...")
    models, binary_cutoffs, tier_cutoffs, scalers = load_models(model_dir)

    print(f"Reading variants from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Validate required columns
    required_cols = FEATURES + ['Category']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate categories
    invalid = df[~df['Category'].isin(CATEGORIES)]['Category'].unique()
    if len(invalid) > 0:
        raise ValueError(f"Unknown categories found: {invalid}. Must be one of {CATEGORIES}")

    results = []

    for category in CATEGORIES:
        subset = df[df['Category'] == category].copy()
        if subset.empty:
            continue

        print(f"  Predicting {len(subset)} variants in category: {category}")

        model = models[category]
        thresholds = tier_cutoffs[category]

        # Fill missing feature values
        X = subset[FEATURES].copy()
        zero_impute = [
            'SpliceAI_pred_DS_AG', 'SpliceAI_pred_DS_AL',
            'SpliceAI_pred_DS_DG', 'SpliceAI_pred_DS_DL',
            'max_splice_val', 'ada_score', 'rf_score', 'MaxEntScan_diff'
        ]
        median_impute = ['MaxEntScan_ref', 'MaxEntScan_alt', 'CADD_PHRED']

        X[zero_impute] = X[zero_impute].fillna(0)
        X[median_impute] = X[median_impute].fillna(X[median_impute].median())
        X = X.fillna(0)

        # Predict posterior probability
        probabilities = model.predict_proba(X)[:, 1]

        subset['splicing_probability'] = probabilities
        subset['binary_prediction'] = (probabilities >= binary_cutoffs[category]).astype(int)
        subset['acmg_5tier'] = [classify_5tier(p, thresholds) for p in probabilities]
        subset['cutoff_used'] = binary_cutoffs[category]

        results.append(subset)

    if not results:
        print("No variants were classified.")
        return

    output_df = pd.concat(results, ignore_index=True)
    output_df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(output_df)} variants classified.")
    print(f"Results saved to: {output_csv}")

    # Summary
    print("\n── Classification Summary ──")
    print(output_df['acmg_5tier'].value_counts().to_string())
    print("\n── By Category ──")
    print(output_df.groupby(['Category', 'acmg_5tier']).size().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpliceLoc-ML: Splice Variant Classifier')
    parser.add_argument('--input',  required=True, help='Input CSV file with annotated variants')
    parser.add_argument('--output', required=True, help='Output CSV file for predictions')
    parser.add_argument('--models', default='models', help='Path to models directory (default: models/)')
    args = parser.parse_args()

    predict(args.input, args.output, args.models)
