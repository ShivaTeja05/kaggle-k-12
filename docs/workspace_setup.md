# Set up a fresh clone

Use an account with access to the private GitHub repository. The competition files and model outputs come from Kaggle under your authorized account.

```bash
git clone https://github.com/ShivaTeja05/kaggle-k-12.git
cd kaggle-k-12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu-lock.txt kaggle==2.2.5
export PYTHONPATH=src
python -m unittest discover -s tests -p test_core.py -v
```

The adaptation and reading-supervision tests also require PyTorch. Install a CPU-compatible `torch==2.8.0` build for your platform, then run `python -m unittest discover -s tests -v`.

## Download the canonical tables

Sign in to Kaggle and enter the competition before downloading. Keep your Kaggle credentials in the CLI's user configuration. The version-2 files under `competition_data/` use the topic names expected by our saved predictions.

```bash
(
  set -e
  slug=misconception-detection-in-k-12-student-math-workings
  mkdir -p data/kaggle-download data/raw
  for file in train.csv test.csv taxonomy_v1.csv; do
    kaggle competitions download "$slug" -f "competition_data/$file" \
      -p data/kaggle-download --unzip
    cp "data/kaggle-download/competition_data/$file" "data/raw/$file"
  done
  shasum -a 256 data/raw/{train,test}.csv
  python -m k12.baseline
)
```

Compare the checksums with [the saved data version](../reports/data_versions.json). The baseline writes its predictions and metrics to `artifacts/baseline/`.

## Restore the saved Qwen runs

These commands download completed notebook outputs. The private notebooks require access from the authorized Kaggle account.

```bash
kaggle kernels output shivateja005/k12-vision-development \
  -p runs/vision-development --file-pattern '^k12/(validation_epoch_3|test)\.npz$'
python -m k12.holdout_calibration

kaggle kernels output shivateja005/k12-qwen35-27b-compact-grammar-gated \
  -p runs/qwen35-27b --file-pattern '^vllm_ranker/validation\.npz$'
python scripts/audit_model_errors.py
```

The calibration command uses the epoch-3 scores with temperature **2.0** and a **25%** question/topic prior to reproduce the **0.59800** public submission recipe. The candidate reranker uses cached rankings from both models and scores **0.65006** on the reused development fold. It has development results only.

To restore the trained LoRA, classifier heads, and processor, download `^k12/epoch-3/` from the same 4B notebook into `runs/vision-development`. The [official base model](https://www.kaggle.com/models/qwen-lm/qwen-3-vl/Transformers/4b-instruct/1) is a separate input. The saved training recipe is in [the private Kaggle notebook](https://www.kaggle.com/code/shivateja005/k12-vision-development) and the editable implementation is in [`src/k12/vision.py`](../src/k12/vision.py).

## Reproduce the CPU photo comparison

With the canonical tables in place, list the 60 required images using the same fold and sample logic as the evaluation. Download those files from the competition with `kaggle competitions download` and use `--unzip -p data/cpu-image-audit` to preserve their paths.

```bash
python - <<'PY'
from k12.core import make_folds, read_csv
train = read_csv('data/raw/train.csv')
sample = train.loc[make_folds(train) == 0].sample(n=60, random_state=2026)
for name in sample.image_file:
    print(f'competition_data/images/train/{name}')
PY
```

Install Tesseract with the English language model. For PP-FormulaNet, use a separate Python 3.12 environment at `artifacts/formula-ocr/.venv` with `numpy==1.26.4`, `paddlepaddle==3.3.1`, `paddleocr==3.7.0`, `tokenizers==0.23.3`, `ftfy==6.3.1` and Pillow. Place `inference.json`, `inference.yml` and `inference.pdiparams` from the [pinned formula checkpoint](https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-S/tree/3d46f557e3a1752f4bf81202395af3b5ecfadfd2) in `artifacts/formula-ocr/model/`. The recognition stage reads the local model directory in offline mode.

```bash
python scripts/evaluate_formula_reader.py prepare
artifacts/formula-ocr/.venv/bin/python scripts/evaluate_formula_reader.py recognize
python scripts/evaluate_formula_reader.py evaluate
```

For the survey follow-up, place the same three files from the [pinned CTC checkpoint](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec/tree/ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57) in `artifacts/formula-ocr/latin-ctc/`. Run the two recognition stages of `scripts/evaluate_htr_hypotheses.py` with the isolated environment and its final scoring stage with the project environment.

```bash
artifacts/formula-ocr/.venv/bin/python scripts/evaluate_htr_hypotheses.py recognize
artifacts/formula-ocr/.venv/bin/python scripts/evaluate_htr_hypotheses.py recognize-ctc
python scripts/evaluate_htr_hypotheses.py evaluate --with-ctc
```

## Continue experiments

- [Model error audit](../reports/model_error_audit.json) has the candidate recall, failure breakdown, and paired reranker scores.
- [Handwriting research notes](htr_survey_takeaways.md) describe the CPU photo pilot and the next evidence-verification experiment. The photo scripts select 60 labeled fold-0 rows with seed 2026 and read the corresponding official JPEGs from `data/cpu-image-audit/competition_data/images/train/`. Keep those files and all OCR transcripts local.
- [Competition rules and research](competition_research.md) covers the offline submission requirement. [AGENTS.md](../AGENTS.md) records the current CPU-only decision, and [the GPU budget policy](../reports/gpu_budget.json) disables further K-12 GPU launches.

Use `git pull --ff-only` to update the source on later visits. Keep credentials, competition tables and images, prediction bundles, checkpoint weights, and generated logs in the ignored local paths.
