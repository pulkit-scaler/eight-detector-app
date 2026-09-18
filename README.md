# Eight detector

Companion code for the Streamlit and Optuna session.

A screening model for a rare class: given an 8x8 hand-written digit from
scikit-learn's `digits` dataset, is it an eight? Only about one image in ten is,
which is what makes it a useful example. The library defaults are wrong for it
in a specific, fixable way.

## Setup

```bash
git clone https://github.com/pulkit-scaler/eight-detector-app.git
cd eight-detector-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. Tested on 3.12 and 3.14.

`scikit-learn` and `optuna` are pinned exactly because they decide the numbers,
and because `rf_eight_best.joblib` was pickled by that scikit-learn. Everything
else has a floor rather than an exact pin, so pip can pick versions that have
wheels for your Python instead of trying to compile numpy from source.

## Run the app

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. The app loads `rf_eight_best.joblib`, which
is committed here, so it works immediately without training anything.

Three modes:

| Mode | What it does |
|---|---|
| Review cases | Page through the test set, including the errors |
| Manual tuning | Set hyperparameters by hand and refit |
| Auto-tuning | Run an Optuna search from the browser |

**Review cases** is the one that matters. Filter to "missed eights" to see the
images the model failed on, which is the question a domain expert actually asks
and which a precision and recall pair cannot answer.

## Retrain

```bash
python train_model.py
```

Runs the same 30-trial Optuna search the notebook runs and overwrites the two
artifact files. The objective cross-validates inside the training set and never
touches the test set.

## The teaching snippets

`snippets/` holds the eight short Streamlit scripts from the session, each
demonstrating one idea. Run any of them the same way:

```bash
streamlit run snippets/01_hello.py
```

| File | Idea |
|---|---|
| `01_hello.py` | The smallest app |
| `02_order.py` | Layout is execution order |
| `03_sidebar.py` | The sidebar as a second channel |
| `04_widgets.py` | Widgets return values |
| `05_button_chart.py` | Buttons are true for one rerun |
| `06_cache_hit.py` | A cache key that stays put |
| `07_cache_miss.py` | A cache key that changes every call |
| `08_session_state.py` | State that survives a rerun |

## Deploying

Fork this repository, then at <https://share.streamlit.io> sign in with GitHub,
choose **New app**, and point it at your fork with `app.py` as the entry point.
The build takes a minute or two.

The pinned versions in `requirements.txt` matter. Unpinned, the build resolves
to whatever is newest on the day it runs, and a model pickled by one
scikit-learn version may not load under another.
