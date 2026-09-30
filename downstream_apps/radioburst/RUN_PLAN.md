# Run plan: baseline + LoRA on the GPU machine

Trains the linear baseline and the LoRA-fine-tuned Surya model for 24 h Type II
forecasting, then scores both on 2020–2024. Disk: ~1.1 TB of cached SDO frames
(train + val); test frames are streamed and deleted. All commands run from the repo root.

## What is compared

| Name | What it is | Trained? |
|---|---|---|
| `surya` | Surya backbone + LoRA adapters (r=4) + a 2-output head | yes |
| `baseline` | One linear layer on the spatial mean and std of each of the 13 SDO channels. Tests whether Surya adds anything beyond "how bright / magnetically active is the disk" | yes, same frames and labels |
| `mx_flare_24h` | "Yes" if an M/X flare started in the past 24 h. **The number to beat: TSS 0.29 [0.19, 0.39]** | no (built into `evaluate.py`) |
| `persistence` | "Yes" if a Type II started in the past 24 h. TSS 0.13 | no |

## 0. Setup (once)

```bash
cd surya_workshop && git checkout downstream/radiobursts && git pull
conda env create -f environment.yml && conda activate surya_ws
export SURYA_WS_CACHE_DIR=/path/with/1.1TB/free        # add to ~/.bashrc so every shell has it
python -m workshop_infrastructure.benchmark_s3 s3://nasa-surya-bench/2011/01/20110131_0000.nc --anon --quick
```

In `downstream_apps/radioburst/configs/config_script.yaml`:
- copy the recommended `s3_boto3_max_concurrency` / `s3_boto3_part_size_mb` from the benchmark;
- set `training.deterministic: warn` so the two runs are comparable.

Scalers and Surya weights (~1.8 GB) download automatically on the first run.

## 1. Tests (minutes)

```bash
pytest tests downstream_apps/radioburst/tests -v
```

The model/metric/checkpoint tests have not run anywhere yet. **Stop here if anything fails.**

## 2. Smoke test (≈1 h)

```bash
cp downstream_apps/radioburst/configs/config_script.yaml downstream_apps/radioburst/configs/smoke.yaml
```

In `smoke.yaml` set `max_samples: 10`, `job_id: smoke`, `output.ckpt_dir: checkpoints_smoke`. Then:

```bash
C=downstream_apps/radioburst/configs/smoke.yaml
python -m downstream_apps.radioburst.3_finetune_template_1D --config $C --max-epochs 2 --train_baseline
python -m downstream_apps.radioburst.3_finetune_template_1D --config $C --max-epochs 2
python -m downstream_apps.radioburst.evaluate --config $C --pred-dir predictions_smoke predict \
    --checkpoint checkpoints_smoke/baseline-*.ckpt checkpoints_smoke/surya-*.ckpt
```

- **Record seconds per training step and per validation frame** for LoRA — the only way to
  estimate the real run's length.
- With 10 samples there may be no positives, so the per-epoch `val_metric_*` scores can be
  missing. Expected.

## 3. Train

The baseline goes first: it is cheap on the GPU, and its first epoch fills the frame cache
(~1.1 TB) that LoRA then reuses.

```bash
python -m downstream_apps.radioburst.3_finetune_template_1D --train_baseline
python -m downstream_apps.radioburst.3_finetune_template_1D                  # LoRA, r=4
```

- Metrics: `runs/typeii_forecast_{baseline,surya}/version_*/metrics.csv`. `val_loss` should
  fall and `val_metric_type2_roc_auc` should rise above 0.5.
- Best checkpoints: `checkpoints/baseline-…ckpt` and `checkpoints/surya-…ckpt`.

## 4. Predict (GPU; streams ~3.6 TB of test frames, none kept)

```bash
python -m downstream_apps.radioburst.evaluate predict --delete-test-frames \
    --checkpoint checkpoints/baseline-*.ckpt checkpoints/surya-*.ckpt
```

Writes `predictions/{baseline,surya}_{val,test}.csv`. Both models see each frame in one pass.

## 5. Score (CPU, seconds)

```bash
python -m downstream_apps.radioburst.evaluate score
```

Prints the table and saves `predictions/scores.csv`. Then open
`downstream_apps/radioburst/4_results.ipynb` for reliability curves and the May 2024 case study.

## Reading the result (test, `type2` row)

| Outcome | Meaning |
|---|---|
| `surya` **`dtss_ci_low > 0`** | Beats the M/X-flare baseline. Repeat LoRA with `seed: 43` and `44` to confirm. |
| `surya` above `baseline`, but its `dtss` interval spans 0 | Beats simple image statistics, not the flare baseline. Still reportable. |
| `surya` ≈ `baseline` | The backbone adds nothing here. |

Also compare the `type2_ip` row with `type2`. If the two score the same, the model learned
"an eruptive region is on the disk", not "this shock will reach interplanetary space".

## If something goes wrong

- **"s3:// paths but no cache dir"** — `SURYA_WS_CACHE_DIR` is not set in that shell.
- **GPU out of memory (LoRA)** — `--batch-size 1`.
- **Disk filling up** — only train and val frames should stay cached (~1,080 files). If there
  are many more, check that `predict` got `--delete-test-frames`.
- **Duplicate-name error from `predict`** — more than one `surya-*.ckpt` matched. Pass the
  exact files, or add `--name`.
- **LoRA rank** — `model.lora_config.r` (4). Costs no extra storage; with ~250 positives keep
  it small. Try 8 only if training loss stays high.
