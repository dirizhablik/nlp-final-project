# Intended Sarcasm Detection on iSarcasmEval

NLP final project. We compare three approaches of increasing sophistication on
**SemEval-2022 Task 6 (iSarcasmEval), Subtask A, English**: binary intended-sarcasm
detection (1 = sarcastic, 0 = not). The question is whether stronger models pick up subtle
sarcasm or mostly learn surface cues.

The three approaches:

1. A rule-based cue model (VADER sentiment contrast, intensifiers, punctuation, …)
2. TF-IDF + Logistic Regression
3. Fine-tuned `roberta-base`

On top of these, we run a computational efficiency analysis, a rephrasing experiment, a comparison
with human annotators, and error analysis.

**Primary metric:** F1 on the sarcastic class (`f1_score(..., average="binary", pos_label=1)`),
as defined by the shared task. We also report precision, recall, accuracy and the confusion matrix.

> **Status:** the repo scaffold and environment setup are done. The data module, models and
> evaluation are being added in follow-up PRs, and this README is updated as each one lands.

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

On **Google Colab**, open the notebook in `notebooks/` (added together with the RoBERTa model).
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

## Repository layout

| Path | What it is |
|---|---|
| `src/models/base.py` | `SarcasmModel` abstract base class: the interface every model implements |
| `src/models/__init__.py` | Model registry (`rule`, `tfidf`, `roberta`), lazily imported |
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
| `tests/` | pytest suite |
| `data/raw/`, `data/processed/` | Dataset and our fixed split (committed) |
| `results/` | Metrics JSON + prediction CSVs per model/split (committed) |
| `checkpoints/` | Saved models, local only (gitignored) |

### How the pieces fit

Every model subclasses `SarcasmModel` and implements five methods:

| Method | Contract |
|---|---|
| `fit(train, val)` | Train on `train`; use `val` only for tuning (threshold, grid search, epoch) |
| `predict_scores(texts)` | One score per text, **higher = more sarcastic** |
| `size()` | e.g. `{"n_parameters": ...}` or `{"n_features": ...}` for the efficiency comparison |
| `save(dir)` / `load(dir)` | Persist and restore the tuned model |

`predict()` (shared) thresholds the scores at `self.threshold`. `src/run.py` works with any
registered model:

- `--split val`: builds a fresh model, fits it, saves it to `checkpoints/<model>/`, then evaluates
  it on val.
- `--split test`: loads that saved model and evaluates it on test. It never refits, so the test
  set can't influence tuning.

To add a model: subclass `SarcasmModel`, add one line to the registry, and add `configs/<name>.yaml`.
`tests/test_models_base.py` checks that every registered model follows the contract.

## Reproducing results

These commands are filled in as each component lands:

```bash
python -m src.data                                # build data/processed/ splits + stats
python -m src.run --model rule   --split val
python -m src.run --model tfidf  --split val
python -m src.run --model roberta --split val
python -m src.run --model <name> --split test     # once per final config
python -m src.analysis.results_table
```

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
