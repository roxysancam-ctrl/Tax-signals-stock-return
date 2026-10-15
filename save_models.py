"""Save the four trained models (full feature set) as .pkl files.

Refits each model on fiscal 2012-2020 with the settings train_models.py chose
(results/best_params.json), checks the test AUC matches results/model_metrics.csv,
and writes models/<name>.pkl. Run train_models.py first.

Each .pkl is a plain dict, so loading it needs only scikit-learn / xgboost:
    model       the fitted classifier
    kind        "tree" or "logit" (which ETR version and whether to scale)
    features    the raw modeling_table.csv columns it uses
    columns     the matrix columns, in order
    medians     training medians used to fill blanks
    categories  the one-hot categories for industry / exchange / region
    scaler      the StandardScaler (logit only)

Use it on new rows (same columns as modeling_table.csv):
    import joblib, pandas as pd
    from save_models import prepare
    bundle = joblib.load("models/rf.pkl")
    X = prepare(pd.read_csv("data/modeling_table.csv"), bundle)
    prob_beat = bundle["model"].predict_proba(X)[:, 1]

Usage
    python save_models.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from train_models import (CATEGORICAL, FEATURE_SETS, MODEL_NAMES, RESULTS_DIR, design_matrix,
                          fit, log, make_model)

MODELS_DIR = config.DATA_DIR.parent / "models"


def prepare(df, bundle):
    """Apply a saved model's preprocessing to rows shaped like modeling_table.csv."""
    cols = {}
    for f in bundle["features"]:
        if f in CATEGORICAL:
            for c in bundle["categories"][f]:
                cols[f"{f}={c}"] = (df[f] == c).astype(float)
            continue
        source = f"{f}_{bundle['kind']}" if f in ("etr", "cash_etr") else f
        x = df[source].astype(float)
        if f"{f}_blank" in bundle["columns"]:
            cols[f"{f}_blank"] = x.isna().astype(float)
        cols[f] = x.fillna(bundle["medians"][f])
    X = pd.DataFrame(cols, index=df.index)[bundle["columns"]]
    if bundle["kind"] == "logit":
        if "abate_value" in X:
            X["abate_value"] = np.log1p(X["abate_value"])
        X = pd.DataFrame(bundle["scaler"].transform(X), index=X.index, columns=X.columns)
    return X


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(config.DATA_DIR / "modeling_table.csv")
    train_rows = (df["split"] == "train").to_numpy()
    y = df["beat_sp500"].to_numpy()
    best_params = json.loads((RESULTS_DIR / "best_params.json").read_text())
    metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")

    for name in MODEL_NAMES:
        kind = "logit" if name == "logit" else "tree"
        prep = {}
        X, _ = design_matrix(df, FEATURE_SETS["all"], kind, train_rows, prep)
        params = best_params[f"all/{name}"]
        model = fit(make_model(name, params, y[train_rows]), name,
                    X[train_rows].reset_index(drop=True), y[train_rows])
        bundle = {"model": model, "model_name": MODEL_NAMES[name], "params": params,
                  "trained_on": "fiscal 2012-2020", **prep}

        # Check: the saved preprocessing reproduces the matrix, and the test AUC matches.
        test = df[~train_rows]
        auc = roc_auc_score(y[~train_rows], model.predict_proba(prepare(test, bundle))[:, 1])
        expected = metrics.loc[(metrics["features"] == "all") & (metrics["model"] == name), "auc"].iloc[0]
        bundle["test_auc_2021_2024"] = round(auc, 4)
        path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(bundle, path)
        log(f"{MODEL_NAMES[name]:13s} test AUC {auc:.4f} (train_models.py: {expected:.4f}) -> {path}")


if __name__ == "__main__":
    main()
