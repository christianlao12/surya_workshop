# Type II radio-burst forecasting with Surya

**Question.** From one SDO frame at issue time *t*, what is the probability that a
**Type II radio burst** starts in *(t, t + 24 h]* — and that an **interplanetary-reaching**
one (ending below 1 MHz, i.e. the shock survived into the solar wind) does?

Type II bursts are the radio signature of CME-driven shocks. Interplanetary-reaching ones
accompany most large solar energetic particle (SEP) events, which is the operational
motivation: a 24 h "shock-driving eruption likely" signal is an input to SEP all-clear
forecasting. Forecasts are issued every 6 h (00/06/12/18 UTC).

Built from `downstream_apps/template/` — see its `ADAPTING.md` for the general pattern.

## Quick start (from the repo root)

```bash
# 0. Data files (already committed; rerun only to rebuild them)
python -m downstream_apps.radioburst.prepare_data                  # add --fetch-flares to refresh mx_flares.csv

# 1. Train (set data.max_samples: 10 in the YAML for a sanity run)
CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.radioburst.3_finetune_template_1D --train_baseline
CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.radioburst.3_finetune_template_1D

# 2. Predict on val + test (GPU), then score (CPU)
python -m downstream_apps.radioburst.evaluate predict --checkpoint checkpoints/baseline-....ckpt
python -m downstream_apps.radioburst.evaluate predict --checkpoint checkpoints/surya-....ckpt
python -m downstream_apps.radioburst.evaluate score

# Tests (CPU, seconds)
pytest downstream_apps/radioburst/tests -v
```

Notebooks: `0_explore_data.ipynb` (labels, splits, data budget, how hard the task is) and
`4_results.ipynb` (score table, reliability, May 2024 case study). Neither needs a GPU.

## Layout

| File | Role |
|---|---|
| `prepare_data.py` | Catalog → `data/typeii_events.csv`; full Surya index → `data/splits/{train,val,test}.csv`; HEK → `data/mx_flares.csv` |
| `labels.py` | The one definition of the labels (onset in *(t, t + 24 h]*), shared by dataset, splits and evaluation |
| `configs.py`, `configs/config_script.yaml` | `TypeIIDataConfig` and the run config |
| `datasets/radioburst_dataset.py` | `TypeIIDataset`: one frame per issue time + the two labels |
| `models/simple_baseline.py` | `LinearTypeIIModel`: one linear layer on per-channel mean/std |
| `metrics/radioburst_metrics.py` | `TypeIIMetrics` (per-label BCE) and `ValidationScores` (epoch-level AUC/TSS) |
| `lightning_modules/pl_simple_baseline.py` | `TypeIILightningModule`, shared by both models |
| `3_finetune_template_1D.py` | Training. Surya = stock `HelioSpectformer1D` with `num_outputs=2` |
| `evaluate.py` | `predict` (GPU) and `score` (CPU) |

## Design decisions worth knowing

**Splits.** Test is 2020–2024 (Solar Cycle 25), which Surya's pretraining excluded
(arXiv:2508.14112 §2.1). Val is every 5th solar rotation of 2010–2019; train the rest. A
2-day gap at every boundary stops label windows and near-identical frames from crossing
splits. The repo-wide Surya split files are not used: they split by day-of-year and drop 2012
and 2022.

| Split | Issue times | P(Type II 24 h) | P(IP Type II 24 h) |
|---|---|---|---|
| train | 10,591 | 4.7% | 3.1% |
| val | 2,230 | 5.3% | 2.6% |
| test | 7,132 | 8.8% | 2.8% |

**Imbalance** is handled by keeping `ds_negative_ratio` negatives per positive in train only,
not by `pos_weight`. Every dropped row is negative for both labels, so multiplying the
predicted odds by the kept fraction (recorded in the checkpoint as `negative_keep_fraction`)
restores calibrated probabilities for both outputs exactly. `evaluate.py predict` applies it.

**Validation scores are per epoch**, not per batch: at a ~5% event rate and batch size 2 most
batches have no positive, so per-batch AUC/TSS is meaningless.

**Evaluation** thresholds probabilistic forecasts at the TSS-maximizing value on *val*, then
scores *test*. Intervals come from a bootstrap over **solar rotations**, because events
cluster (187 test onsets fall in only ~43 rotations).

## The number to beat

| Reference forecast (test, 2020–2024) | TSS, Type II | TSS, IP Type II |
|---|---|---|
| Persistence (a Type II in the past 24 h) | 0.13 [0.05, 0.19] | 0.07 [−0.01, 0.15] |
| **M/X flare in the past 24 h** | **0.29 [0.18, 0.39]** | **0.30 [0.12, 0.45]** |

The M/X-flare baseline costs one line. Surya is useful here only if its TSS minus the
baseline's has a paired interval above zero (`dtss_ci_low > 0` in `evaluate.py score`).

## Limitations

- **Effective sample size** is ~43 test rotations with events, not 7,132 rows.
- **Base-rate shift**: cycle 25 has about twice cycle 24's Type II rate, so probabilities
  learned on 2010–2019 run low on the test years. TSS/AUC are unaffected; Brier skill and
  reliability are not.
- **The IP label** depends on the CME speed, which a pre-eruption frame cannot contain.
  Expect it to add little beyond the Type II label; the flare baseline scores the same on both.
- **Label noise**: 34 onsets come from behind-limb flares (`ds_exclude_behind_limb` tests their
  effect); the catalog covers decameter–hectometric Type IIs only.
- **Operational use** would need near-real-time SDO data, calibrated differently from the
  science data Surya was trained on.

## Data sources

- Type II catalog: CDAW Wind/WAVES DH Type II list,
  <https://cdaw.gsfc.nasa.gov/CME_list/radio/waves_type2.html> (copy in `data/cdaw_typeii_catalog.csv`).
- Flares: SSW Latest Events (LMSAL) via the HEK API.
- SDO frames: `s3://nasa-surya-bench` via the Surya full index.
