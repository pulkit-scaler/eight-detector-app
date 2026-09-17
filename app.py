"""Eight detector — a rare-class screening demo (Streamlit + Optuna).

Loads the artifact trained in the notebook and serves predictions from it.
The tuning modes are for teaching; a production app would serve only.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

MODEL_PATH = "rf_eight_best.joblib"
META_PATH = "model_meta.json"
RANDOM_STATE = 42
TARGET_DIGIT = 8

optuna.logging.set_verbosity(optuna.logging.WARNING)
st.set_page_config(page_title="Eight detector", layout="centered")


# --- data and artifacts ----------------------------------------------------
# cache_data: a copy per cache key, which is what you want for frames.
@st.cache_data(show_spinner=False)
def load_split():
    d = datasets.load_digits(as_frame=True)
    X = d.data
    y = (d.target == TARGET_DIGIT).astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


# cache_resource: one shared object, not copied. Correct for a model.
@st.cache_resource(show_spinner=False)
def load_artifact():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return None, None
    with open(META_PATH) as f:
        meta = json.load(f)
    return joblib.load(MODEL_PATH), meta


X_train, X_test, y_train, y_test = load_split()
FEATURES = X_train.columns.tolist()          # the order the model was fitted in
saved_model, saved_meta = load_artifact()


def to_row(values) -> pd.DataFrame:
    """One row, columns in the order the model was fitted in.

    Passing a bare array here instead would let the pixels arrive in whatever
    order the caller happened to produce, and predict wrong without erroring.
    """
    return pd.DataFrame([list(values)], columns=FEATURES)


def as_image(values, scale=24):
    """64 pixel values (0-16) -> a dark-on-light image, upscaled without smoothing.

    np.kron repeats each pixel into a scale x scale block, so the viewer sees the
    64 numbers the model actually got rather than a browser's smoothed guess.
    """
    arr = np.asarray(values, dtype=float).reshape(8, 8)
    img = (255 - arr / 16.0 * 255).astype(np.uint8)
    return np.kron(img, np.ones((scale, scale), dtype=np.uint8))


def objective(trial):
    """Cross-validated F1 on the TRAINING set. Never touches X_test."""
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        class_weight=trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()


def remember(model, source):
    st.session_state.model = model
    st.session_state.model_source = source


def active_model():
    """Whatever this session trained last, else the artifact on disk."""
    if "model" in st.session_state:
        return st.session_state.model, st.session_state.model_source
    if saved_model is not None:
        return saved_model, f"saved artifact ({MODEL_PATH})"
    return None, None


# --- header ----------------------------------------------------------------
st.title("Eight detector")
st.write(
    f"A screening model for a rare class: is this hand-written digit a {TARGET_DIGIT}? "
    f"Only about one image in ten is."
)
mode = st.sidebar.radio("Mode", ["Review cases", "Manual tuning", "Auto-tuning (Optuna)"])

model, source = active_model()
if model is None:
    st.warning(f"No `{MODEL_PATH}` next to this script. Train one in a tuning mode.")
    st.stop()

pred_test = model.predict(X_test)
c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{precision_score(y_test, pred_test, zero_division=0):.3f}")
c2.metric("Recall", f"{recall_score(y_test, pred_test):.3f}")
c3.metric("F1", f"{f1_score(y_test, pred_test):.3f}")
st.caption(f"Model in use: {source}")

# --- review cases ----------------------------------------------------------
if mode == "Review cases":
    st.header("Review individual cases")

    choice = st.radio(
        "Which cases?",
        ["All", "Missed eights (false negatives)", "False alarms (false positives)"],
        horizontal=True,
    )
    truth, guess = y_test.values, pred_test
    if choice.startswith("Missed"):
        pool = np.where((truth == 1) & (guess == 0))[0]
    elif choice.startswith("False alarms"):
        pool = np.where((truth == 0) & (guess == 1))[0]
    else:
        pool = np.arange(len(truth))

    if len(pool) == 0:
        st.success("No cases of that kind in the test set.")
    else:
        k = st.slider("Case", 0, len(pool) - 1, 0) if len(pool) > 1 else 0
        i = int(pool[k])
        row = X_test.iloc[i]
        proba = model.predict_proba(to_row(row))[0][1]

        left, right = st.columns([1, 2])
        left.image(as_image(row), width=180)
        right.write(f"**Ground truth:** {'an eight' if truth[i] == 1 else 'not an eight'}")
        right.write(f"**Model says:** {'an eight' if guess[i] == 1 else 'not an eight'}")
        right.progress(float(proba), text=f"confidence it is an eight: {proba:.1%}")
        if truth[i] != guess[i]:
            right.error("The model got this one wrong.")
        else:
            right.success("The model got this one right.")
        st.caption(f"{len(pool)} case(s) of this kind. Test-set row {i}.")

# --- manual tuning ---------------------------------------------------------
elif mode == "Manual tuning":
    st.header("Manual tuning")
    st.write("Set the hyperparameters directly, to see which knob moves which metric.")
    with st.form("manual"):
        n_estimators = st.slider("n_estimators", 50, 300, 100, step=10)
        max_depth = st.slider("max_depth (0 means None)", 0, 20, 0)
        class_weight = st.selectbox("class_weight", ["None", "balanced", "balanced_subsample"])
        submitted = st.form_submit_button("Train")

    if submitted:
        with st.spinner("Fitting..."):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None if max_depth == 0 else max_depth,
                class_weight=None if class_weight == "None" else class_weight,
                random_state=RANDOM_STATE, n_jobs=-1,
            ).fit(X_train, y_train)
        remember(clf, "manual model (this session)")
        st.success(f"Test F1: {f1_score(y_test, clf.predict(X_test)):.4f}. "
                   "Switch to Review cases to see what changed.")
        st.caption(
            "Every configuration you try here and reject is a look at the test set. "
            "Try enough of them and this number stops being an estimate."
        )

# --- auto-tuning -----------------------------------------------------------
else:
    st.header("Auto-tuning with Optuna")
    n_trials = st.slider("Number of trials", 5, 40, 15)

    if st.button("Run optimization"):
        # In-memory study: a fresh search per click. Persisting to SQLite from a
        # Streamlit script would write a new study on every rerun of the page.
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        bar, label = st.progress(0), st.empty()

        def tick(study, trial):
            done = trial.number + 1
            bar.progress(done / n_trials)
            label.text(f"trial {done}/{n_trials} — best {study.best_value:.4f}")

        with st.spinner("Searching..."):
            study.optimize(objective, n_trials=n_trials, callbacks=[tick])

        params = dict(study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
        best = RandomForestClassifier(**params).fit(X_train, y_train)
        remember(best, "Optuna model (this session)")

        # Two different numbers. Showing only the first is how apps mislead.
        st.success(f"Best cross-validated F1 (what was optimized): {study.best_value:.4f}")
        st.info(f"F1 on the held-out test set (not optimized): "
                f"{f1_score(y_test, best.predict(X_test)):.4f}")
        st.json(study.best_params)

        joblib.dump(best, MODEL_PATH)
        with open(META_PATH, "w") as f:
            json.dump({
                "feature_names": FEATURES,
                "target_digit": TARGET_DIGIT,
                "best_params": {k: v for k, v in params.items() if k != "n_jobs"},
                "cv_f1": float(study.best_value),
                "test_f1": float(f1_score(y_test, best.predict(X_test))),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, f, indent=2)
        load_artifact.clear()          # the file changed; drop the cached object
        st.caption(f"Saved to {MODEL_PATH}.")

        df = study.trials_dataframe(attrs=("number", "value", "params"))
        st.subheader("Top trials")
        st.dataframe(df.sort_values("value", ascending=False).head())
        st.subheader("Optimization history")
        st.plotly_chart(vis.plot_optimization_history(study))
        if n_trials >= 10:
            st.subheader("Parameter importances")
            st.plotly_chart(vis.plot_param_importances(study))

st.markdown("---")
st.caption(
    "Teaching demo. In production, train offline, serve the artifact, and keep the "
    "search in a tracked pipeline rather than in the request path."
)
