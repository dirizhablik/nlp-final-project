# Intended Sarcasm Detection on iSarcasmEval

NLP final project. We compare three approaches of increasing sophistication on
**SemEval-2022 Task 6 (iSarcasmEval), Subtask A, English**: binary intended-sarcasm
detection (1 = sarcastic, 0 = not). The question is whether stronger models pick up subtle
sarcasm or mostly learn surface cues.

The three approaches:

1. A rule-based cue model (VADER sentiment contrast, intensifiers, punctuation, emoji, …)
2. TF-IDF + Logistic Regression
3. Fine-tuned `roberta-base`

On top of these, we run a computational efficiency analysis, a rephrasing experiment, a comparison
with human annotators, and error analysis.

**Primary metric:** F1 on the sarcastic class (`f1_score(..., average="binary", pos_label=1)`),
as defined by the shared task. We also report precision, recall, accuracy and the confusion matrix.

> **Status:** the scaffold, EDA notebook, data pipeline (`src/data.py`), shared evaluation
> (`src/evaluate.py`), the majority-class and always-sarcastic baselines, the rule-based model
> and TF-IDF + LR are done. Fine-tuned RoBERTa, the experiments and the analysis scripts are still to come, and this
> README is updated as each one lands.

## Setup

Requires **Python 3.12 or newer** (the pinned NumPy 2.5 doesn't support 3.11 or older).
3.12 is recommended because it's what Google Colab runs.

```bash
git clone https://github.com/dirizhablik/nlp-final-project.git && cd nlp-final-project
make setup          # creates .venv with python3.12 and installs requirements-dev.txt
source .venv/bin/activate
```

If your interpreter has a different name, pass it in: `make setup PYTHON=python3.13`.

Without `make`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

On **Google Colab**, use the RoBERTa notebook (added together with the RoBERTa model).
It installs everything except `torch`, which Colab already provides.

## Data

The raw files are committed in `data/raw/`. They come from
[iabufarha/iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) at commit `dfc708b`. They're
redistributed under the dataset's MIT license (see `data/raw/LICENSE`) and were renamed as follows:

| `data/raw/` file | Source | Contents |
|---|---|---|
| `train.En.csv` | `train/train.En.csv` | 3,468 tweets (867 sarcastic) with author rephrasings and category labels |
| `task_A_En_test.csv` | `test/task_A_En_test.csv` | 1,400 test tweets (200 sarcastic) |
| `task_C_En_test.csv` | `test/task_C_En_test.csv` | 200 test (sarcastic, rephrase) pairs |
| `human_task_A_En.csv` | `third-party annotations/english_task_a.csv` | 5-annotator human labels for the Task A test set |
| `human_task_C_En.csv` | `third-party annotations/english_task_c.csv` | Human labels for the Task C test pairs |

Processed splits (`data/processed/`) are committed too, so all three of us use the exact same
split. Model checkpoints are gitignored and must **not** be committed because they're too large.

### Exploratory analysis

`notebooks/01_eda.ipynb` explores the raw data and motivates every cleaning decision. Its
last section is a summary table of findings and decisions. The notebook is committed with
its outputs, so you can read it on GitHub without running it. To re-run it:

```bash
jupyter lab notebooks/01_eda.ipynb      # or: jupyter nbconvert --execute --inplace --to notebook notebooks/01_eda.ipynb
```

The notebook only reads `data/raw/`. The decisions themselves are implemented in `src/data.py`.

### Processed splits

`python -m src.data` builds `data/processed/{train,val,test}.csv` (columns `id, text, label,
rephrase`) and `stats.json`. The result is deterministic, so re-running it produces identical
files. In order, it:

1. drops the one empty tweet;
2. unescapes HTML entities and replaces URLs with `<URL>` and @mentions with `<USER>`, without
   changing case, punctuation or emoji (the options live in `configs/base.yaml` under `cleaning`);
3. drops all 12 rows of the 5 duplicate texts with conflicting labels, and keeps one copy of
   each other duplicate. One pair becomes identical only after URL masking, so 5 copies are
   dropped rather than the 4 the EDA counted;
4. splits train 85/15 into train/val, stratified by label (seed 42), keeping identical texts on
   the same side;
5. checks for leakage, and fails if a val/test text or val rephrasing appears in training data.

| Split | Rows | Sarcastic |
|---|---|---|
| train | 2,932 | 733 (25.0%) |
| val | 518 | 129 (24.9%) |
| test | 1,400 | 200 (14.3%) |

The test set is cleaned the same way, but no rows are removed. `stats.json` records the cleaning
counts, length statistics and a hash of each split's ids, so you can quickly confirm everyone
has the same split.

## Repository layout

| Path | What it is |
|---|---|
| `src/models/base.py` | `SarcasmModel` abstract base class: the interface every model implements |
| `src/models/baseline.py` + `configs/{majority,always_sarcastic}.yaml` | Reference baselines: majority class (always non-sarcastic) and always sarcastic |
| `src/models/__init__.py` | Model registry (`majority`, `always_sarcastic`, `rule`, `tfidf`, `roberta`), lazily imported |
| `src/models/rule_based.py` + `configs/rule.yaml` | Rule-based cue model |
| `src/models/tfidf_lr.py` + `configs/tfidf.yaml` | TF-IDF + Logistic Regression |
| `src/models/roberta.py` + `configs/roberta.yaml` | Fine-tuned roberta-base |
| `src/data.py` | Load, clean, split, stats, leakage checks |
| `src/evaluate.py` | Metrics, timing, results/predictions writer |
| `src/run.py` | Entry point: `python -m src.run --model ... --split ...` |
| `src/utils.py` | Config loading, seeding, logging, device selection |
| `src/experiments/` | `rephrase.py`, `human_comparison.py` |
| `src/analysis/` | `plots.py`, `results_table.py` |
| `configs/base.yaml` | Seed, paths, split size |
| `notebooks/` | `01_eda.ipynb`: exploratory data analysis (read-only on `data/raw/`) |
| `tests/` | pytest suite |
| `data/raw/`, `data/processed/` | Dataset and our fixed split (committed) |
| `results/` | Metrics JSON + prediction CSVs per model/split (committed) |
| `checkpoints/` | Saved models, local only (gitignored) |

### How the pieces fit

Every model subclasses `SarcasmModel` and implements three methods:

| Method | Contract |
|---|---|
| `fit(train, val)` | Train on `train`; use `val` only for tuning (threshold, grid search, epoch) |
| `predict_scores(texts)` | One score per text, **higher = more sarcastic** |
| `size()` | e.g. `{"n_parameters": ...}` or `{"n_features": ...}` for the efficiency comparison |

The base class provides the rest:

| Method | What it does |
|---|---|
| `predict(texts)` | Thresholds the scores at `self.threshold` |
| `report()` | Extra details stored with the results (override to add best params, top features, …) |
| `save(dir)` / `load(dir)` | Write/read `state.json`: config, seed, threshold and every attribute listed in the class's `state_attrs`. Models with weights extend both (see `TfidfLRModel`) |

`src/run.py` works with any registered model:

- `--split val`: builds a fresh model, fits it, saves it to `checkpoints/<model>/`, then evaluates
  it on val.
- `--split test`: loads that saved model and evaluates it on test. It never refits, so the test
  set can't influence tuning.

To add a model: subclass `SarcasmModel`, add one line to the registry, and add `configs/<name>.yaml`.
`tests/test_models_base.py` checks that every registered model follows the contract.

## Reproducing results

Everything below works now except the RoBERTa and results-table commands, which start working
when those land:

```bash
python -m src.data                                # build data/processed/ splits + stats
python -m src.run --model majority --split val
python -m src.run --model majority --split test
python -m src.run --model always_sarcastic --split val
python -m src.run --model always_sarcastic --split test
python -m src.run --model rule   --split val
python -m src.run --model tfidf  --split val
python -m src.run --model roberta --split val
python -m src.run --model <name> --split test     # once per final config
python -m src.analysis.results_table
```

Each run writes two files to `results/<model>/`:
- `<split>.json` holds the metrics (F1 with a bootstrap 95% CI, precision and recall for the
  sarcastic class, accuracy, macro-F1, ROC-AUC, PR-AUC, the confusion matrix as
  `[[TN, FP], [FN, TP]]`), timings and model size.
- `<split>_predictions.csv` holds one row per text: `id, text, label, pred, score`.

Test-set results (official test, 1,400 tweets, 14.3% sarcastic):

| Model | val F1 (sarc.) | test F1 (sarc.) | test F1 95% CI | test precision | test recall | test accuracy | test ROC-AUC |
|---|---|---|---|---|---|---|---|
| Majority class (always non-sarcastic) | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.857 | 0.500 |
| Always sarcastic | 0.399 | 0.250 | [0.222, 0.278] | 0.143 | 1.000 | 0.143 | 0.500 |
| Rule-based cues (weights + threshold tuned on val) | 0.370 | 0.300 | [0.262, 0.338] | 0.191 | 0.695 | 0.537 | 0.654 |
| TF-IDF + LR (C=1, balanced, threshold 0.5) | 0.360 | 0.285 | [0.239, 0.329] | 0.211 | 0.440 | 0.685 | 0.620 |
| Human majority vote (from the EDA notebook) | – | 0.613 | – | – | – | – | – |

Each `results/<model>/<split>.json` also has PR-AUC and its chance level (the share of
sarcastic texts). The F1 CI is a 1,000-resample bootstrap over test texts, seeded from
`configs/base.yaml`. The rule-based and TF-IDF intervals overlap, so neither model is
clearly better yet.

Sarcastic F1 depends on how common sarcasm is in a split. The always-sarcastic baseline gets
2p / (1 + p) for a sarcastic share p: 0.25 on test, 0.40 on val, 0.67 on a 50/50 set. So compare
models on the same split, never val against test. ROC-AUC does not depend on the class mix (0.5
is chance), so it is the fairer number for comparing ranking quality across splits.

We keep the splits as they are. Train and val are stratified at the training file's 25%. The
official test set is not resampled: that keeps it comparable with the shared task and with human
performance. The difference in class mix still matters for tuning. On val, flagging almost
everything already gives F1 0.40, so a threshold tuned for val F1 flags too much on test. Tuning
TF-IDF's threshold on val (it picked 0.40) raised val F1 to 0.443 but lowered test F1 to 0.272.
So TF-IDF keeps a fixed 0.5 threshold, and `class_weight="balanced"` handles the imbalance (see
`configs/tfidf.yaml` to rerun that ablation).

### Experimental protocol

- All tuning (thresholds, hyperparameters, epoch selection) uses the fixed validation split only.
- The test set is evaluated **once per final config**, and nothing is tuned on it.
- All randomness is seeded from `seed` in `configs/base.yaml`.

## Development workflow

- Make one branch per piece of work, e.g. `feature/rule-based`, `feature/tfidf`,
  `feature/evaluate`. Open a PR into `main` and get a teammate to review it.
  Don't push directly to `main`.
- Keep each branch to the files for that piece of work. Changes to common files, especially
  the `SarcasmModel` interface or the data schema, go in their own small PR so everyone can
  rebase onto them quickly.
- Pull `main` often: `git switch main && git pull && git switch - && git rebase main`.
- Before pushing, run `make lint test`. Use `make format` to auto-fix style.
- Hyperparameters go in `configs/*.yaml`, not in code.
- Use `logging` rather than `print`.

## References

- Abu Farha et al. (2022). *SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in
  English and Arabic.* SemEval-2022. https://aclanthology.org/2022.semeval-1.111
- Abu Farha et al. (2022). *Sarcasm Detection is Way Too Easy! An Empirical Comparison of Human
  and Machine Sarcasm Detection.* Findings of EMNLP 2022.
