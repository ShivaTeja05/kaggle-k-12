# Root evidence and multimodel ranking

10 November 2026. The current public third-place score is **0.84150**, versus our verified **0.59800**. The best submission uses one image-and-text Qwen model with a question/topic prior. The prototype below combines the saved rankings of two Qwen models and adds a separate verifier that examines written steps. [Live leaderboard](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/leaderboard)

## What the prototype does

1. Combine the calibrated 4B model's top eight and the 27B model's top three with reciprocal rank fusion. Their union contains the correct code on **91.69%** of the 301 development rows.
2. Build a reference library from **758 annotated root examples and 180 downstream examples** in the 1,199 fitting rows. Each example is a short window around the annotated line. The root library covers 67 codes.
3. For each candidate, compare the working's line windows with reference examples of that code. The prototype uses character similarity on literal text and on operator patterns with numbers replaced by placeholders. The contrast variant reduces support when the same window resembles examples where that code was a downstream finding.
4. Adjust the selected error-code probabilities with a fixed weight of 0.35 while preserving the existing `NONE` probability. Evaluate teacher transcripts and OCR separately. The teacher annotations are used to build the fitting library. The diagnostic teacher-text inputs on validation are the existing transcripts.

## Measured results

| Variant | 301 development rows | Same 60-photo subset |
|---|---:|---:|
| Calibrated 4B reference | 0.60742 | 0.58611 |
| 4B and 27B rank fusion | 0.62569 | 0.64722 |
| Rank fusion plus root examples, teacher text | **0.63732** | 0.65556 |
| Rank fusion plus root and downstream examples, teacher text | 0.62957 | 0.64722 |
| Rank fusion plus root examples, Tesseract text | — | **0.66389** |

The root-example stage improved the full-fold teacher-text result by **0.01163** over rank fusion. Its paired question-bootstrap interval was **−0.00109 to +0.02365**. On the 60-photo OCR comparison it improved by **0.01667**, with an interval of **−0.01667 to +0.05085**. The OCR estimate uses a small subset, so compare it only with the same 60 rows. Both intervals include zero. On the full fold the 4B alone with the new teacher-text verifier scored **0.60908**. In this prototype, the root-only reference library performed better than adding the downstream similarity penalty.

The saved 27B results cover validation only. These CPU measurements reuse the existing development fold and provide a design comparison. Our scored submission remains the 4B plus prior. [Aggregate metrics](root_example_screen.json)

## What would move the design forward

Test an offline formula reader and the same verifier on a larger, fixed set of photos. Compare the image reference, the two-model fusion where cached, and the verifier with the same examples in each arm. Measure symbol accuracy, candidate recall and root-error ranking. A useful second reader must supply evidence the image model misses.

The next candidate verifier can check the mathematical operation itself. For example, if a student writes `1/3 + 1/6 = 2/9`, test whether the candidate predicts adding both numerators and denominators. A verified equation can support that candidate more directly than a similar-looking string. Extend this to a few common error pairs, with an unknown outcome for unreadable work and unsupported expressions.

The public gap is **0.24350**. On the development fold, achieving a score numerically equal to today's third-place mark with the current `NONE` performance would require **0.80396** MAP@3 on the 200 root-error rows, versus **0.45167** today. Improving candidate coverage and getting more correct codes to rank one both matter. A modest reranking gain alone leaves a large gap.

Reproduce on CPU with `PYTHONPATH=src .venv/bin/python scripts/evaluate_root_examples.py`. The script reads the existing private development outputs and the 60 training photos from the earlier OCR audit.
