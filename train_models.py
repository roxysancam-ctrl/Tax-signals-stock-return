"""Step 6: train Logit, Random Forest, AdaBoost and XGBoost, then feature importance.

Input:  data/modeling_table.csv (run build_modeling_table.py first)
Output: results/ (metrics, importance, charts) and data/test_predictions.csv
        (one row per test company-year with each model's probability, for the
        portfolio step)

How it works
- Time-based split, never random: train on fiscal 2012-2020, test on 2021-2024.
- Tuning also respects time: each setting is fit on 2012-2017 and scored on
  2018-2020 (AUC). The best setting is then refit on all of 2012-2020 and
  scored once on 2021-2024.
- Blank ETRs follow Rosana's rule: tree models use etr_tree / cash_etr_tree
  (-1 for loss and tiny-profit years), logit uses etr_logit / cash_etr_logit
  (training median) with the pretax_loss / pretax_tiny flags.
- Other blanks (etr_change, deferred tax ratios, prior-year return, ...) get
  the training-years median plus a 0/1 "<column>_blank" flag, the same for
  every model, so the model can still tell "missing" apart from "typical".
- Classes are imbalanced (40% beat in train), so every model weights the two
  classes equally. That keeps recall on "beat" from collapsing to zero.
- Two feature sets: "all" (tax + abatements + filing + controls) and
  "controls_only". The tax and filing signals only add something if "all"
  beats "controls_only" on the test years.
- Feature importance = permutation importance on the test years: shuffle one
  feature (with its blank flag, or all dummies of a category) and measure how
  much test AUC falls. Built-in tree importance is biased toward continuous
  features, so it isn't used.

Usage
    python train_models.py              # full run, a few minutes
    python train_models.py --quick      # smaller grids, for checking the code
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config

RESULTS_DIR = config.DATA_DIR.parent / "results"
TUNE_FIT_YEARS = range(2012, 2018)    # tuning: fit on 2012-2017
TUNE_CHECK_YEARS = range(2018, 2021)  # tuning: score on 2018-2020
SEED = 42

# Feature groups. "etr" / "cash_etr" are swapped for the tree or logit version per model.
TAX = ["etr", "cash_etr", "pretax_loss", "pretax_tiny", "pretax_missing", "etr_change",
       "cash_to_book_tax", "deferred_share_of_tax", "net_deferred_tax_to_assets",
       "valuation_allowance_to_assets", "unrecognized_tax_benefits_to_assets"]
ABATEMENT = ["abate_any", "abate_value", "abate_to_revenue", "abate_deals"]
FILING = ["nt_10k", "nt_10q_count", "nt_count", "nt_max_days_late", "nt_never_filed",
          "nt_missed_extension", "tenk_filing_lag_days", "filing_lag_days"]
CONTROLS = ["size_quintile", "log_total_assets", "filer_status_est", "ff12_industry", "exchange",
            "hq_region", "fye_december", "dividend_payer", "years_since_first_filing",
            "prior_ret_12m", "roa", "pretax_roa"]
CATEGORICAL = ["ff12_industry", "exchange", "hq_region"]
CATEGORY_OF = {**{f: "Tax" for f in TAX}, **{f: "Abatement" for f in ABATEMENT},
               **{f: "Filing" for f in FILING}, **{f: "Control" for f in CONTROLS}}
FEATURE_SETS = {"all": TAX + ABATEMENT + FILING + CONTROLS, "controls_only": CONTROLS}
# Not used as features: delisted_in_window (only known after the return window),
# from_comparative / filing_row_missing / abate_ny_covered / abate_data_year (data-coverage flags).

MODEL_NAMES = {"logit": "Logit", "rf": "Random Forest", "ada": "AdaBoost", "xgb": "XGBoost"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ features
def design_matrix(df, features, kind, fit_rows, prep=None):
    """Turn raw columns into a numeric matrix.

    kind = "tree" or "logit" (picks the ETR version). Medians and scaling come
    from fit_rows only. Returns X and {feature: [matrix columns]} for grouping.
    If a dict is passed as prep, it is filled with the medians, categories and
    scaler, so the same steps can be applied to new data (see save_models.py).
    """
    medians, categories = {}, {}
    cols, groups = {}, {}
    for f in features:
        if f in CATEGORICAL:
            cats = sorted(df[f].dropna().unique())
            categories[f] = cats
            dummies = [f"{f}={c}" for c in cats]
            for c, name in zip(cats, dummies):
                cols[name] = (df[f] == c).astype(float)
            groups[f] = dummies
            continue
        source = f"{f}_{kind}" if f in ("etr", "cash_etr") else f
        x = df[source].astype(float)
        groups[f] = [f]
        if x[fit_rows].isna().mean() > 0.01:
            cols[f"{f}_blank"] = x.isna().astype(float)
            groups[f].append(f"{f}_blank")
        medians[f] = x[fit_rows].median()
        cols[f] = x.fillna(medians[f])
    X = pd.DataFrame(cols, index=df.index)
    if kind == "logit":
        if "abate_value" in X:
            X["abate_value"] = np.log1p(X["abate_value"])  # dollars: very skewed
        scaler = StandardScaler().fit(X[fit_rows])
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    if prep is not None:
        prep.update(kind=kind, features=features, columns=list(X.columns), medians=medians,
                    categories=categories, scaler=scaler if kind == "logit" else None)
    return X, groups


# ------------------------------------------------------------------ models
def make_model(name, params, y_fit):
    if name == "logit":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=SEED, **params)
    if name == "ada":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=params["max_depth"]),
                                  n_estimators=params["n_estimators"],
                                  learning_rate=params["learning_rate"], random_state=SEED)
    if name == "xgb":
        ratio = (y_fit == 0).sum() / (y_fit == 1).sum()
        return XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ratio,
                             n_jobs=-1, random_state=SEED, **params)
    raise ValueError(name)


def fit(model, name, X, y):
    if name == "ada":  # AdaBoost has no class_weight; weight the rows instead
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def grid(name, quick):
    if name == "logit":
        return [{"C": c} for c in ([0.1] if quick else [0.001, 0.01, 0.1, 1.0])]
    if name == "rf":
        leaves = [50] if quick else [20, 50, 100, 200]
        depths = [8] if quick else [6, 10, None]
        return [{"min_samples_leaf": l, "max_depth": d} for l in leaves for d in depths]
    if name == "ada":
        depths = [1] if quick else [1, 2, 3]
        rates = [0.5] if quick else [0.05, 0.2, 0.5]
        return [{"max_depth": d, "learning_rate": r, "n_estimators": 100 if quick else 300}
                for d in depths for r in rates]
    if name == "xgb":
        depths = [3] if quick else [2, 3, 4, 6]
        weights = [10] if quick else [5, 20, 50]
        trees = [100] if quick else [100, 300, 600]
        return [{"max_depth": d, "min_child_weight": w, "n_estimators": n, "learning_rate": 0.03}
                for d in depths for w in weights for n in trees]
    raise ValueError(name)


def tune(name, X, y, years, quick):
    """Fit each setting on 2012-2017, score AUC on 2018-2020, keep the best."""
    fit_rows, check_rows = years.isin(TUNE_FIT_YEARS), years.isin(TUNE_CHECK_YEARS)
    best, rows = None, []
    for params in grid(name, quick):
        model = fit(make_model(name, params, y[fit_rows]), name, X[fit_rows], y[fit_rows])
        auc = roc_auc_score(y[check_rows], model.predict_proba(X[check_rows])[:, 1])
        rows.append({**params, "valid_auc": auc})
        if best is None or auc > best[1]:
            best = (params, auc)
    return best[0], best[1], rows


# ------------------------------------------------------------------ scoring
def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def top_decile_beat_rate(test, p):
    """Share of each year's top 10% (by predicted probability) that beat the S&P 500."""
    d = test[["fiscal_year", "beat_sp500"]].assign(p=p)
    d["rank"] = d.groupby("fiscal_year")["p"].rank(pct=True)
    return d.loc[d["rank"] > 0.9, "beat_sp500"].mean()


def bootstrap_auc(y, p, p_base=None, n=500):
    """95% range for test AUC (and for AUC minus the baseline's, same resamples)."""
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    aucs, diffs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], p[i]))
        if p_base is not None:
            diffs.append(aucs[-1] - roc_auc_score(y[i], np.asarray(p_base)[i]))
    out = {"auc_low": np.percentile(aucs, 2.5), "auc_high": np.percentile(aucs, 97.5)}
    if diffs:
        out["gain_vs_controls_low"] = np.percentile(diffs, 2.5)
        out["gain_vs_controls_high"] = np.percentile(diffs, 97.5)
    return out


def permutation_importance(model, X, y, groups, repeats):
    """AUC drop when a feature (with its blank flag / dummies) is shuffled on the test rows."""
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    category_groups = {}
    for f, cols in groups.items():
        category_groups.setdefault(CATEGORY_OF[f], []).extend(cols)
    out = []
    for level, items in (("feature", groups), ("category", category_groups)):
        for f, cols in items.items():
            drops = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[cols] = X[cols].values[rng.permutation(len(X))]  # one shuffle for the whole group
                drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
            out.append({"level": level, "feature": f,
                        "category": CATEGORY_OF.get(f, f), "auc_drop": np.mean(drops),
                        "auc_drop_std": np.std(drops)})
    return out


# ------------------------------------------------------------------ charts
def charts(importance, metrics, predictions, best):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping charts")
        return
    colors = {"Tax": "#2a6fdb", "Filing": "#e07b39", "Abatement": "#3a9e5f", "Control": "#9a9a9a"}

    feat = importance[(importance["level"] == "feature") & (importance["model"] == best)]
    feat = feat.sort_values("auc_drop").tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(feat["feature"], feat["auc_drop"], xerr=feat["auc_drop_std"],
            color=feat["category"].map(colors))
    ax.set_xlabel("Drop in test AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance, {MODEL_NAMES[best]} (test years 2021-2024)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=cat) for cat, c in colors.items()], loc="lower right")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    cat = importance[importance["level"] == "category"].pivot(index="feature", columns="model",
                                                               values="auc_drop")
    cat = cat[[m for m in MODEL_NAMES if m in cat]].rename(columns=MODEL_NAMES)
    ax = cat.plot.bar(figsize=(8, 5), rot=0)
    ax.set_ylabel("Drop in test AUC when the group is shuffled")
    ax.set_xlabel("")
    ax.set_title("Which group of signals the models use (test years 2021-2024)")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "importance_by_group.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(predictions["beat_sp500"], predictions[f"prob_{name}"])
        auc = metrics.loc[(metrics["model"] == name) & (metrics["features"] == "all"), "auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{MODEL_NAMES[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Coin flip (AUC 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, test years 2021-2024")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small grids, for testing the code")
    parser.add_argument("--repeats", type=int, default=5, help="shuffles per feature for importance")
    parser.add_argument("--charts-only", action="store_true", help="redraw charts from saved results")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.charts_only:
        metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
        best = metrics[metrics["features"] == "all"].sort_values("auc").iloc[-1]["model"]
        charts(pd.read_csv(RESULTS_DIR / "permutation_importance.csv"), metrics,
               pd.read_csv(config.DATA_DIR / "test_predictions.csv"), best)
        return

    df = pd.read_csv(config.DATA_DIR / "modeling_table.csv")
    train_rows = (df["split"] == "train").to_numpy()
    test_rows = ~train_rows
    y = df["beat_sp500"].to_numpy()
    years = df["fiscal_year"]
    log(f"{train_rows.sum():,} train rows ({y[train_rows].mean():.1%} beat), "
        f"{test_rows.sum():,} test rows ({y[test_rows].mean():.1%} beat)")

    metrics, by_year, tuning, best_params, importance, coefs = [], [], [], {}, [], None
    test = df.loc[test_rows].reset_index(drop=True)
    predictions = test[["cik", "name", "symbol", "fiscal_year", "ret_start", "ret_end",
                        "ret_12m", "sp500_ret_12m", "excess_ret_12m", "beat_sp500"]].copy()
    test_probs = {}

    for feature_set, features in FEATURE_SETS.items():
        for name in MODEL_NAMES:
            kind = "logit" if name == "logit" else "tree"
            X, groups = design_matrix(df, features, kind, train_rows)
            Xtr, Xte = X[train_rows].reset_index(drop=True), X[test_rows].reset_index(drop=True)
            ytr, yte = y[train_rows], y[test_rows]

            start = time.time()
            params, valid_auc, rows = tune(name, Xtr, ytr, years[train_rows].reset_index(drop=True),
                                           args.quick)
            tuning += [{"features": feature_set, "model": name, **r} for r in rows]
            best_params[f"{feature_set}/{name}"] = params
            model = fit(make_model(name, params, ytr), name, Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            test_probs[(feature_set, name)] = p

            row = {"features": feature_set, "model": name, "n_features": X.shape[1],
                   "valid_auc_2018_2020": valid_auc, **scores(yte, p),
                   "top10pct_beat_rate": top_decile_beat_rate(test, p),
                   "test_beat_rate": yte.mean()}
            row.update(bootstrap_auc(yte, p))
            metrics.append(row)
            for yr, part in test.groupby("fiscal_year"):
                by_year.append({"features": feature_set, "model": name, "fiscal_year": yr,
                                "auc": roc_auc_score(part["beat_sp500"], p[part.index])})
            log(f"{feature_set:13s} {MODEL_NAMES[name]:13s} valid AUC {valid_auc:.3f}  "
                f"test AUC {row['auc']:.3f}  ({time.time() - start:.0f}s, {params})")

            if feature_set == "all":
                predictions[f"prob_{name}"] = p
                if name == "logit":
                    coefs = pd.DataFrame({"column": X.columns, "coef_per_sd": model.coef_[0]})
                    coefs["feature"] = coefs["column"].map({c: f for f, cs in groups.items() for c in cs})
                    coefs["category"] = coefs["feature"].map(CATEGORY_OF)
                    coefs = coefs.reindex(coefs["coef_per_sd"].abs().sort_values(ascending=False).index)
                start = time.time()
                for r in permutation_importance(model, Xte, yte, groups, args.repeats):
                    importance.append({"model": name, **r})
                log(f"  permutation importance done ({time.time() - start:.0f}s)")

    # The controls-only models run after "all", so fill in the gain now.
    metrics = pd.DataFrame(metrics)
    for name in MODEL_NAMES:
        p_all = test_probs[("all", name)]
        p_ctl = test_probs[("controls_only", name)]
        extra = bootstrap_auc(y[test_rows], p_all, p_ctl)
        i = metrics.index[(metrics["features"] == "all") & (metrics["model"] == name)][0]
        metrics.loc[i, "auc_gain_vs_controls"] = metrics.loc[i, "auc"] - roc_auc_score(y[test_rows], p_ctl)
        metrics.loc[i, "gain_vs_controls_low"] = extra["gain_vs_controls_low"]
        metrics.loc[i, "gain_vs_controls_high"] = extra["gain_vs_controls_high"]
    metrics.insert(1, "model_name", metrics["model"].map(MODEL_NAMES))

    importance = pd.DataFrame(importance)
    importance.insert(1, "model_name", importance["model"].map(MODEL_NAMES))
    importance = importance.sort_values(["model", "level", "auc_drop"], ascending=[True, False, False])

    metrics.round(4).to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(by_year).round(4).to_csv(RESULTS_DIR / "auc_by_test_year.csv", index=False)
    pd.DataFrame(tuning).round(4).to_csv(RESULTS_DIR / "tuning_2018_2020.csv", index=False)
    importance.round(5).to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    coefs.round(4).to_csv(RESULTS_DIR / "logit_coefficients.csv", index=False)
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    predictions.to_csv(config.DATA_DIR / "test_predictions.csv", index=False)

    best = metrics[metrics["features"] == "all"].sort_values("auc").iloc[-1]["model"]
    charts(importance, metrics, predictions, best)

    show = ["features", "model_name", "valid_auc_2018_2020", "auc", "auc_low", "auc_high",
            "accuracy", "precision", "recall", "top10pct_beat_rate", "auc_gain_vs_controls"]
    log("Test results (2021-2024):\n" + metrics[show].round(3).to_string(index=False))
    log(f"Best model by test AUC: {MODEL_NAMES[best]}. Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
