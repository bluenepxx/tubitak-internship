# QMOF Structural Property Prediction

TUBITAK internship project (AI and Sensor Systems unit): predicting three structural
properties of Metal-Organic Frameworks (MOFs) from the QMOF dataset -- framework density,
pore limiting diameter (PLD), and largest cavity diameter (LCD) -- from SELFIES-encoded
node/linker chemistry plus point-group and topology metadata. Each target is modeled
independently, and 30 regression algorithms (linear models, SVMs, tree ensembles, gradient
boosting, and a SELFIES Transformer) are benchmarked under a leakage-safe
train/validation/test protocol.

## Setup

```
pip install -r requirements.txt
```

The pipeline expects a prepared dataset at `data/processed/forward_model_selfies.jsonl`
(with a matching `split_assignments.csv`). Neither the raw QMOF database nor the processed
data is committed to this repo -- rebuild them locally by running notebooks 02-04 in order
against the raw QMOF database.

## Running the benchmark

```
# quick end-to-end sanity check on a small subset
python -m src.run_30_models --mode smoke

# full run, all stages
python -m src.run_30_models --mode full --stages holdout kfold groupkfold learning_curve final_test
```

Stages can also be run individually, e.g. re-running just the final test evaluation once
holdout/kfold results already exist:

```
python -m src.run_30_models --mode full --stages final_test
```

Two post-processing scripts run against the finished `results/tables/`:

```
python scripts/build_final_test_summary.py   # final-test summary + CV-vs-test comparison table
python scripts/shap_analysis.py              # SHAP + permutation feature-group importance
```

## Project structure

```
src/              benchmark pipeline: data loading, preprocessing, model registry, training, evaluation
scripts/          one-off analysis scripts that read results/tables/ (final-test summary, SHAP)
notebooks/        01-04 dataset exploration & preparation, 05 baseline models, 08 results analysis
configs/          experiment settings and per-model hyperparameters (YAML)
data/             raw QMOF database and processed datasets (not tracked, see Setup)
results/tables/   canonical result tables -- the numbers below come from here
results/plots/    results/figures/   report figures
results/models/   trained winning models (not tracked, see Results)
```

## Results

All 30 algorithms were ranked per target by 5-fold CV validation RMSE. The winner for each
target was refit on train+validation and evaluated once on the held-out test set
(`src/train.py::run_final_test`):

| Target  | Model       | Test RMSE | Test R² |
|---------|-------------|-----------|---------|
| density | LightGBM    | 0.2571    | 0.8198  |
| pld     | Extra Trees | 1.3218    | 0.8317  |
| lcd     | Extra Trees | 1.4333    | 0.8672  |

Test RMSE is within ~4-6% of the 5-fold CV estimate for all three targets (inside one CV
standard deviation), so the model-selection protocol isn't leaking into the test evaluation.
SHAP and permutation-based feature-group importance both agree that the linker and node
SELFIES tokens are the two most influential feature groups across all three targets.

Full numbers, methodology notes, and the SHAP/permutation comparison are in
`HANDOFF_UPDATE.md` and `notebooks/08_results_analysis.ipynb`.

Trained model files (`results/models/*.joblib`) aren't committed -- two of the three winners
are ~170MB, over GitHub's non-LFS limit. Regenerate them locally with:

```
python -m src.run_30_models --mode full --stages final_test
```
