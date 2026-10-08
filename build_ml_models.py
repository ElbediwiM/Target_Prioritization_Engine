#!/usr/bin/env python3
"""
Leak-free ML for target drug-development prediction (command-line version)
==========================================================================

Trains three models on targets_ml.parquet (or .csv) and evaluates them with
repeated stratified k-fold cross-validation:

  1. approved_drug           has_approved_drug  (binary)
  2. high_trial_engagement   num_trials > 1     (binary, ~7% positives)
  3. druggability            log1p(num_drugs)   (regression)

What changed versus the earlier script (which reported AUC 1.000 / R2 0.999):
  * Model 1 no longer sees has_approved_drug / max_dev_phase / has_clinical_drug
    (the label is exactly max_dev_phase == 4).
  * Model 2 no longer sees num_trials (the label is num_trials > 1) and, by
    default, not total_condition_mentions (derived from the same trials).
  * Model 3 no longer sees count_molecular_mechanisms / count_direct_interactions
    (they equal num_drugs in ~95% of rows) and is fit on log1p(num_drugs).
  * A leakage guard aborts if a forbidden column, or any feature with
    |Spearman| > 0.97 against the label, gets into a feature matrix.
  * Scaling happens inside the pipeline, so it is fitted on training folds only.
  * Repeated CV with mean ± sd, plus a dummy baseline for every task.
  * Out-of-fold predictions are saved - use those, not in-sample predictions,
    to rank the genes in the training table.
  * Each model is saved as ONE pipeline file (scaler included).

Usage:
    python build_ml_models.py
    python build_ml_models.py --skip-export --skip-viz
    python build_ml_models.py --predict EGFR
    python build_ml_models.py --score-file new_targets.csv
    python build_ml_models.py --include-trial-derived   # (leaky-ish; for comparison)
    python build_ml_models.py --include-max-phase       # (outcome-adjacent; for comparison)
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed - continuing without it")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'targets_ml.parquet'      # falls back to targets_ml.csv if needed
MODELS_DIR = Path('./models')
RESULTS_DIR = Path('./ml_results')

# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3

# Columns that must NEVER be features for a given task, because they define
# (or are a near-copy of) that task's label. The leakage guard raises if any of
# these end up in the feature matrix.
FORBIDDEN = {
    'approved_drug': {
        'has_approved_drug',   # the label itself
        'max_dev_phase',       # has_approved_drug == (max_dev_phase == 4) in every row
        'has_clinical_drug',   # downstream of the same development-phase field
    },
    'high_trial_engagement': {
        'num_trials', 'log_num_trials',          # label is num_trials > 1
        'has_clinical_trials', 'trials_per_mechanism',
    },
    'druggability': {
        'num_drugs', 'log_num_drugs',            # the target itself
        'count_molecular_mechanisms',            # == num_drugs in ~95% of rows (r = 0.998)
        'count_direct_interactions',             # == num_drugs in ~95% of rows (r = 0.998)
        'mechanisms_per_drug',
    },
}

LEAK_CORR_THRESHOLD = 0.97   # |Spearman| with the label above this => abort


# ============================================================
# DATA
# ============================================================

def load_data(path=None, verbose=True):
    """Load the targets table (parquet, falling back to CSV)."""
    path = Path(path or DATA_FILE)
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except ImportError:
            if verbose:
                print("  (pyarrow missing - falling back to CSV)")
    if df is None:
        csv = path.with_suffix('.csv')
        if not csv.exists():
            raise FileNotFoundError(f"Neither {path} nor {csv} found")
        df = pd.read_csv(csv)

    required = ['num_drugs', 'num_mechanisms', 'num_trials',
                'total_condition_mentions', 'has_approved_drug']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if verbose:
        print(f"✓ Loaded {len(df):,} targets with {df.shape[1]} columns")
    return df


def engineer_features(df):
    """Derived features. None of them uses a label; the leakage guard
    additionally blocks the ones that are unsafe for a particular task."""
    d = df.copy()
    n_drugs = d['num_drugs'].clip(lower=0)
    cond = d['total_condition_mentions'].fillna(0)
    d['log_num_drugs'] = np.log1p(n_drugs)
    d['log_num_trials'] = np.log1p(d['num_trials'].clip(lower=0))
    d['log_conditions'] = np.log1p(cond)
    d['mechanisms_per_drug'] = d['num_mechanisms'] / n_drugs.clip(lower=1)
    return d


def get_task_specs(include_trial_derived=False, include_max_phase=False):
    """Task definitions (label + leak-free feature list).

    include_trial_derived : let the trial-engagement model see
        total_condition_mentions. It is computed from the same trial records
        as the label (every target with >1 trial has >=2 mentions), so it is
        OFF by default.
    include_max_phase : let the druggability model see max_dev_phase. It is
        outcome-adjacent (a drug that reached phase 4 is also a drug), so it
        is OFF by default.
    """
    t2 = ['log_num_drugs', 'num_mechanisms', 'mechanisms_per_drug']
    if include_trial_derived:
        t2.append('log_conditions')
    t3 = ['num_mechanisms', 'log_num_trials', 'log_conditions']
    if include_max_phase:
        t3.append('max_dev_phase')
    return {
        'approved_drug': dict(
            title='Approved drug (phase 4) prediction', kind='clf',
            features=['log_num_drugs', 'num_mechanisms', 'mechanisms_per_drug',
                      'log_num_trials', 'log_conditions'],
            primary='roc_auc'),
        'high_trial_engagement': dict(
            title='High trial engagement (>1 trial)', kind='clf',
            features=t2, primary='average_precision'),
        'druggability': dict(
            title='Druggability: log1p(num_drugs) regression', kind='reg',
            features=t3, primary='r2'),
    }


def get_label(task, df):
    if task == 'approved_drug':
        return df['has_approved_drug'].astype(int)
    if task == 'high_trial_engagement':
        return (df['num_trials'] > 1).astype(int)
    if task == 'druggability':
        return np.log1p(df['num_drugs'])
    raise KeyError(task)


def check_leakage(task, X, y):
    """Abort if a forbidden column, or any near-copy of the label, is in X."""
    bad = FORBIDDEN[task] & set(X.columns)
    if bad:
        raise ValueError(f"[{task}] leaky features present: {sorted(bad)}")
    corr = X.apply(lambda c: c.corr(y, method='spearman')).abs()
    worst = corr.idxmax()
    if corr.max() > LEAK_CORR_THRESHOLD:
        raise ValueError(f"[{task}] feature '{worst}' has |Spearman|={corr.max():.3f} "
                         f"with the label - looks like leakage")
    return corr.sort_values(ascending=False)


# ============================================================
# MODELS
# ============================================================

def make_models(kind, y):
    """Small, regularised models suited to ~900 rows. Scaling lives inside the
    pipeline so it is fitted on training folds only."""
    if kind == 'clf':
        pos = max(int(y.sum()), 1)
        spw = (len(y) - pos) / pos
        models = {
            'Baseline (prior)': DummyClassifier(strategy='prior'),
            'Logistic Regression': LogisticRegression(
                max_iter=2000, class_weight='balanced', C=1.0),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=spw,
                eval_metric='logloss', verbosity=0, random_state=RANDOM_STATE, n_jobs=1)
    else:
        models = {
            'Baseline (mean)': DummyRegressor(strategy='mean'),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3,
                random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=2, verbosity=0,
                random_state=RANDOM_STATE, n_jobs=1)
    return {n: Pipeline([('scale', StandardScaler()), ('model', m)])
            for n, m in models.items()}


CLF_SCORING = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
               'balanced_accuracy': 'balanced_accuracy', 'f1': 'f1'}
REG_SCORING = {'r2': 'r2', 'mae_log': 'neg_mean_absolute_error'}


class TaskResult:
    """Everything produced for one prediction task."""
    def __init__(self, name, spec, X, y):
        self.name, self.spec, self.X, self.y = name, spec, X, y
        self.models = {}          # fitted on all data
        self.cv_table = None
        self.best = None
        self.oof = None           # out-of-fold predictions of the best model
        self.corr = None


def run_task(task, spec, df_feat, df_raw, verbose=True):
    kind = spec['kind']
    X = df_feat[spec['features']].astype(float).fillna(0)
    y = get_label(task, df_raw)
    res = TaskResult(task, spec, X, y)
    res.corr = check_leakage(task, X, y)

    print("\n" + "=" * 70)
    print(spec['title'].upper())
    print("=" * 70)
    print(f"Features ({len(X.columns)}): {', '.join(X.columns)}")
    if kind == 'clf':
        print(f"Positives: {int(y.sum())} / {len(y)} ({y.mean() * 100:.1f}%)")
    else:
        print(f"Target log1p(num_drugs): mean {y.mean():.2f}, sd {y.std():.2f}")
    print(f"Leakage check passed (max |Spearman| with label: {res.corr.max():.2f} "
          f"for '{res.corr.idxmax()}')")

    if kind == 'clf':
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=RANDOM_STATE)
        scoring = CLF_SCORING
    else:
        cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                           random_state=RANDOM_STATE)
        scoring = REG_SCORING

    models = make_models(kind, y)
    rows = []
    print(f"\nCross-validating ({N_SPLITS}-fold x {N_REPEATS} repeats)...")
    for name, pipe in models.items():
        cvr = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {'Model': name}
        for key in scoring:
            vals = cvr[f'test_{key}']
            if key == 'mae_log':
                vals = -vals
            row[key] = vals.mean()
            row[key + '_sd'] = vals.std()
        rows.append(row)
        print(f"  {name:22s} {spec['primary']}: "
              f"{row[spec['primary']]:.3f} ± {row[spec['primary'] + '_sd']:.3f}")

    table = pd.DataFrame(rows).set_index('Model')
    res.cv_table = table

    real = table.drop(index=[i for i in table.index if i.startswith('Baseline')])
    best = real[spec['primary']].idxmax()
    res.best = best

    # Out-of-fold predictions of the best model (every gene scored by a model
    # that never saw it) - these are the ones to use for ranking.
    if kind == 'clf':
        fold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        res.oof = cross_val_predict(clone(models[best]), X, y, cv=fold,
                                    method='predict_proba')[:, 1]
    else:
        fold = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        res.oof = cross_val_predict(clone(models[best]), X, y, cv=fold)

    # Final fits on all data (for scoring genuinely new targets)
    for name, pipe in models.items():
        res.models[name] = clone(pipe).fit(X, y)

    # Report
    print("\n" + "-" * 70)
    show = table.copy()
    cols = ['roc_auc', 'average_precision', 'balanced_accuracy', 'f1'] if kind == 'clf' \
        else ['r2', 'mae_log']
    out = pd.DataFrame({c: [f"{table.loc[m, c]:.3f} ± {table.loc[m, c + '_sd']:.3f}"
                            for m in table.index] for c in cols}, index=table.index)
    print(out.to_string())
    print(f"\n🏆 Best (by {spec['primary']}): {best}")
    if kind == 'clf':
        base = y.mean()
        print(f"   Chance level: AUC 0.50, average precision {base:.3f} (= prevalence)")
    else:
        raw_true = np.expm1(y)
        raw_pred = np.expm1(res.oof).clip(min=0)
        rho = pd.Series(res.oof).corr(pd.Series(y.values), method='spearman')
        print(f"   Out-of-fold Spearman rho = {rho:.3f}; "
              f"MAE on raw drug count = {np.mean(np.abs(raw_true - raw_pred)):.2f} drugs "
              f"(predicting the median would give "
              f"{np.mean(np.abs(raw_true - raw_true.median())):.2f})")
    return res


# ============================================================
# OUTPUTS
# ============================================================

def make_plot(results, show=False):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (task, r) in zip(axes, results.items()):
        prim = r.spec['primary']
        t = r.cv_table
        colors = ['#bbbbbb' if i.startswith('Baseline') else
                  ('#2a9d8f' if i == r.best else '#457b9d') for i in t.index]
        ax.bar(range(len(t)), t[prim], yerr=t[prim + '_sd'], color=colors, capsize=4)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([i.replace(' ', '\n') for i in t.index], fontsize=8)
        ax.set_ylabel(prim.replace('_', ' '))
        ax.set_title(r.spec['title'], fontsize=10)
        if prim == 'roc_auc':
            ax.axhline(0.5, ls='--', c='k', lw=0.8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Leak-free cross-validated performance (mean ± sd; grey = baseline)',
                 fontsize=11)
    fig.tight_layout()
    path = RESULTS_DIR / 'model_comparison.png'
    fig.savefig(path, dpi=150)
    print(f"✓ Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_all(results, df, specs_args):
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    oof = df[[c for c in ['ensembl_id', 'gene_symbol'] if c in df.columns]].copy()
    for task, r in results.items():
        for name, pipe in r.models.items():
            if name.startswith('Baseline'):
                continue
            fn = MODELS_DIR / f"{task}_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(pipe, fn)
        meta = dict(task=task, kind=r.spec['kind'], title=r.spec['title'],
                    features=list(r.X.columns), best_model=r.best,
                    best_file=f"{task}_{r.best.lower().replace(' ', '_')}.joblib",
                    primary_metric=r.spec['primary'],
                    cv_primary=float(r.cv_table.loc[r.best, r.spec['primary']]),
                    cv_primary_sd=float(r.cv_table.loc[r.best, r.spec['primary'] + '_sd']),
                    options=specs_args,
                    label=('log1p(num_drugs)' if task == 'druggability' else
                           'has_approved_drug' if task == 'approved_drug' else 'num_trials > 1'))
        (MODELS_DIR / f"{task}_meta.json").write_text(json.dumps(meta, indent=2))
        r.cv_table.to_csv(RESULTS_DIR / f"cv_{task}.csv")
        col = {'approved_drug': 'oof_approval_prob',
               'high_trial_engagement': 'oof_high_trial_prob',
               'druggability': 'oof_log1p_drugs'}[task]
        oof[col] = r.oof
        if task == 'druggability':
            oof['oof_pred_num_drugs'] = np.expm1(r.oof).clip(min=0)
    oof.to_csv(RESULTS_DIR / 'oof_predictions.csv', index=False)
    print(f"✓ Saved models + metadata to: {MODELS_DIR}/")
    print(f"✓ Saved CV tables and out-of-fold predictions to: {RESULTS_DIR}/")


def run_all(df=None, skip_export=False, skip_viz=False, show=False,
            include_trial_derived=False, include_max_phase=False):
    """Train, cross-validate, and (optionally) save all three models."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "LEAK-FREE ML: TARGET DRUG-DEVELOPMENT PREDICTION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    if df is None:
        df = load_data()
    df_feat = engineer_features(df)
    specs = get_task_specs(include_trial_derived, include_max_phase)

    results = {t: run_task(t, s, df_feat, df) for t, s in specs.items()}

    print("\n" + "=" * 70)
    print("SUMMARY (cross-validated, leak-free)")
    print("=" * 70)
    for t, r in results.items():
        p = r.spec['primary']
        print(f"  {r.spec['title']:46s} {r.best:20s} {p} = "
              f"{r.cv_table.loc[r.best, p]:.3f} ± {r.cv_table.loc[r.best, p + '_sd']:.3f}")

    if not skip_viz:
        make_plot(results, show=show)
    if not skip_export:
        save_all(results, df, dict(include_trial_derived=include_trial_derived,
                                   include_max_phase=include_max_phase))
    return dict(df=df, results=results)


# ============================================================
# LOOKUP / SCORING
# ============================================================

def predict_gene(gene_symbol, out=None):
    """Out-of-fold predictions for a gene that is IN the training table.
    (Each value comes from a model that never saw that gene.)"""
    f = RESULTS_DIR / 'oof_predictions.csv'
    oof = out if out is not None else pd.read_csv(f)
    row = oof[oof['gene_symbol'].str.upper() == gene_symbol.upper()]
    if row.empty:
        print(f"✗ {gene_symbol} not in the training table")
        return None
    print(f"\nOut-of-fold predictions for {gene_symbol}:")
    print(row.drop(columns=['ensembl_id'], errors='ignore').T.to_string(header=False))
    return row


def score_new_targets(df_new, models_dir=None):
    """Score targets that were NOT in training with the saved best models.

    df_new needs columns: num_drugs, num_mechanisms, num_trials,
    total_condition_mentions. Caveat: the models were trained only on targets
    with >=1 drug and >=1 trial, so inputs far outside that range are
    extrapolation.
    """
    mdir = Path(models_dir or MODELS_DIR)
    d = engineer_features(df_new)
    out = df_new[[c for c in ['ensembl_id', 'gene_symbol'] if c in df_new.columns]].copy()
    for task in ['approved_drug', 'high_trial_engagement', 'druggability']:
        meta = json.loads((mdir / f"{task}_meta.json").read_text())
        pipe = joblib.load(mdir / meta['best_file'])
        X = d[meta['features']].astype(float).fillna(0)
        if meta['kind'] == 'clf':
            out[f'{task}_prob'] = pipe.predict_proba(X)[:, 1]
        else:
            out['pred_num_drugs'] = np.expm1(pipe.predict(X)).clip(min=0)
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description='Leak-free ML for target drug-development prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('Usage:')[1])
    ap.add_argument('--data', default=DATA_FILE, help='parquet/csv with the targets table')
    ap.add_argument('--predict', metavar='GENE',
                    help='show out-of-fold predictions for a gene in the table (needs a prior run)')
    ap.add_argument('--score-file', metavar='FILE',
                    help='score NEW targets from a csv/parquet with the saved models')
    ap.add_argument('--skip-export', action='store_true', help='do not save models/results')
    ap.add_argument('--skip-viz', action='store_true', help='do not draw the figure')
    ap.add_argument('--include-trial-derived', action='store_true',
                    help='give the trial-engagement model total_condition_mentions (leaky-ish)')
    ap.add_argument('--include-max-phase', action='store_true',
                    help='give the druggability model max_dev_phase (outcome-adjacent)')
    args = ap.parse_args()

    if args.predict:
        predict_gene(args.predict)
        return
    if args.score_file:
        p = Path(args.score_file)
        new = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
        res = score_new_targets(new)
        out = p.with_name(p.stem + '_scored.csv')
        res.to_csv(out, index=False)
        print(res.head(20).to_string(index=False))
        print(f"✓ Saved: {out}")
        return

    df = load_data(args.data)
    run_all(df, skip_export=args.skip_export, skip_viz=args.skip_viz,
            include_trial_derived=args.include_trial_derived,
            include_max_phase=args.include_max_phase)


if __name__ == '__main__':
    main()
