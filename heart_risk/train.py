"""Train and compare models; track every run in MLflow."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

from heart_risk.data import TARGET, load_clean
from heart_risk.features import build_preprocessor

RANDOM_STATE = 42
EXPERIMENT = "heart-disease-risk"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "logreg_baseline": (LogisticRegression(max_iter=1000), None),
    "logreg_tuned": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]},
    ),
    "rf_baseline": (RandomForestClassifier(random_state=RANDOM_STATE), None),
    "rf_tuned": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 300],
         "clf__max_depth": [None, 4, 8],
         "clf__min_samples_leaf": [1, 3, 5]},
    ),
}


def run_candidate(name, estimator, grid, X_tr, X_te, y_tr, y_te):
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"model_family": type(estimator).__name__,
                         "tuned": str(grid is not None)})
        mlflow.log_params({"cv_folds": 5, "random_state": RANDOM_STATE,
                           "n_train": len(X_tr), "n_test": len(X_te)})

        if grid:
            search = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc",
                                  cv=cv, n_jobs=-1)
            search.fit(X_tr, y_tr)
            model, i = search.best_estimator_, search.best_index_
            cv_scores = {m: (search.cv_results_[f"mean_test_{m}"][i],
                             search.cv_results_[f"std_test_{m}"][i]) for m in SCORING}
            mlflow.log_params(search.best_params_)
            mlflow.log_text(pd.DataFrame(search.cv_results_).to_csv(index=False),
                            "cv_results.csv")
        else:
            res = cross_validate(pipe, X_tr, y_tr, scoring=SCORING, cv=cv, n_jobs=-1)
            cv_scores = {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std())
                         for m in SCORING}
            model = pipe.fit(X_tr, y_tr)
            mlflow.log_params({f"clf__{k}": v for k, v in estimator.get_params().items()})

        for m, (mean, std) in cv_scores.items():
            mlflow.log_metric(f"cv_{m}_mean", mean)
            mlflow.log_metric(f"cv_{m}_std", std)

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test = {"accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred),
                "recall": recall_score(y_te, pred),
                "f1": f1_score(y_te, pred),
                "roc_auc": roc_auc_score(y_te, proba)}
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_te, proba, ax=ax)
        ax.set_title(f"ROC curve - {name}")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax)
        ax.set_title(f"Confusion matrix - {name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        signature = infer_signature(X_tr, model.predict(X_tr))
        try:
            mlflow.sklearn.log_model(model, name="model", signature=signature,
                                     input_example=X_tr.head(3),
                                     serialization_format="cloudpickle")
        except TypeError:  # older MLflow versions
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     signature=signature, input_example=X_tr.head(3),
                                     serialization_format="cloudpickle")

    return {"name": name, "model": model, "cv": cv_scores, "test": test}

def setup_mlflow():
    """Track runs in a local SQLite database; store artifacts in ./mlartifacts."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

def main():
    setup_mlflow()

    df = load_clean()
    X = df.drop(columns=[TARGET]).astype("float64")
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results = [run_candidate(n, est, grid, X_tr, X_te, y_tr, y_te)
               for n, (est, grid) in CANDIDATES.items()]

    table = pd.DataFrame({
        r["name"]: {**{f"cv_{k}": v[0] for k, v in r["cv"].items()},
                    **{f"test_{k}": v for k, v in r["test"].items()}}
        for r in results}).T.round(3)
    print(table.to_string())

    best = max(results, key=lambda r: r["cv"]["roc_auc"][0])
    print(f"\nSelected model: {best['name']} (highest mean CV ROC-AUC)")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best["model"], "models/heart_risk_model.joblib")
    card = {
        "selected_model": best["name"],
        "selection_rule": "highest mean 5-fold CV ROC-AUC on the training split",
        "cv_mean": {k: round(v[0], 4) for k, v in best["cv"].items()},
        "test": {k: round(v, 4) for k, v in best["test"].items()},
        "features": list(X.columns),
        "versions": {"python": sys.version.split()[0],
                     "scikit-learn": sklearn.__version__,
                     "mlflow": mlflow.__version__},
    }
    Path("models/model_card.json").write_text(json.dumps(card, indent=2))

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    keep = [c for c in runs.columns
            if c.startswith(("metrics.", "params.")) or c == "tags.mlflow.runName"]
    Path("report").mkdir(exist_ok=True)
    runs[keep].to_csv("report/experiment_summary.csv", index=False)


if __name__ == "__main__":
    main()