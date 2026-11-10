# Incremental CPU verifier results

The best verified public submission remains **0.59800**, up from **0.58216** for the original Qwen image model. This round tests whether a small verifier can improve ranking by examining the student's written steps. The three text models used fitting rows only. Each variant used the image model's eight leading codes and a fixed 35% text weight.

| Variant on the same 301 development rows | MAP@3 | Rank-one accuracy | Root-error MAP@3 | Root-error rows ranked `NONE` first |
|---|---:|---:|---:|---:|
| Calibrated image reference | **0.60742** | **51.16%** | **0.45167** | 51 |
| Literal teacher text | 0.55039 | 41.86% | 0.35167 | 114 |
| Teacher text with equation and operator features | 0.57475 | 45.51% | 0.38500 | 85 |
| Same structured verifier with image `NONE` probability fixed | 0.59247 | 48.17% | 0.43000 | 53 |
| Structured verifier fitted on error rows only with image `NONE` probability fixed | 0.58804 | 47.18% | 0.42833 | 52 |

Structured evidence improved the literal verifier by **0.02436**. Preserving the image model's `NONE` probability recovered another **0.01772** with the same text model. Removing `NONE` rows from verifier training reduced the overall score. The best new full-fold result improved 24 rows and regressed 33. Its paired question-bootstrap interval versus the image reference was **−0.0410 to +0.0099**. The image reference remains the strongest option in this comparison.

On the same **60 labelled photos**, the image reference scored **0.58611**. The structured verifier with fixed `NONE` probability reached **0.60833** with teacher text and **0.58611** with machine OCR. Its small teacher-text gain had a paired interval of **−0.0424 to +0.0860**, and the full-fold result was lower. The numeric parser found at least one checkable equality on eight teacher transcripts and zero English Tesseract outputs. The verifier was trained on teacher text, so that comparison includes a difference between training and inference input quality. The verifier skips lines marked crossed out or unclear and checks only supported numeric expressions.

The next comparison should give the verifier candidate-specific evidence, such as whether a visible transition matches one of two competing error definitions. Start with common error pairs on labelled work and measure root-error MAP@3 and rank-one accuracy. Evaluate stronger formula OCR against the same verifier before using its output in a candidate. Preserve the image model's error-presence estimate while checking code ordering.

The results are exploratory because the image model and calibration were already selected on this development fold. These CPU experiments produced validation measurements. A score improvement needs to reproduce on a frozen evaluation before promotion.

Reproduce with `PYTHONPATH=src .venv/bin/python scripts/evaluate_working_verifier.py` and the 60 private training JPEGs from the existing CPU audit. [Aggregate results](working_verifier_progress.json)
