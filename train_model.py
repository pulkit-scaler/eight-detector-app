"""Train the eight detector and write the artifact the app loads.

This is the notebook's Optuna section as a plain script. Run it to regenerate
rf_eight_best.joblib and model_meta.json from scratch:

    python train_model.py
"""
import json
import warnings

import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

RANDOM_STATE = 42
TARGET_DIGIT = 8
N_TRIALS = 30

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

digits = datasets.load_digits(as_frame=True)
X = digits.data
y = (digits.target == TARGET_DIGIT).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)


def objective(trial):
    """One configuration to one cross-validated F1 on the TRAINING set."""
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        class_weight=trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS)

params = dict(study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
model = RandomForestClassifier(**params).fit(X_train, y_train)
pred = model.predict(X_test)

joblib.dump(model, "rf_eight_best.joblib")
meta = {
    "feature_names": X_train.columns.tolist(),
    "target_digit": TARGET_DIGIT,
    "best_params": {k: v for k, v in params.items() if k != "n_jobs"},
    "cv_f1": float(study.best_value),
    "test_f1": float(f1_score(y_test, pred)),
    "test_recall": float(recall_score(y_test, pred)),
    "test_precision": float(precision_score(y_test, pred, zero_division=0)),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"best cross-validated F1 : {study.best_value:.4f}")
print(f"test F1                 : {meta['test_f1']:.4f}")
print(f"test recall             : {meta['test_recall']:.4f}")
print("wrote rf_eight_best.joblib and model_meta.json")
