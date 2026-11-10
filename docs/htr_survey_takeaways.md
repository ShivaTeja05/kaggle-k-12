# Handwriting recognition methods for the K-12 model

The most useful change from the survey is to separate recognizing the ink from reconstructing the order of the working. Our classifier can continue proposing misconception codes while a dedicated reader supplies equations, their locations and uncertainty to the verifier. The paper reviews handwriting transcription, with benchmark comparisons on Latin-script text. Our adaptation has to handle fractions, exponents, diagrams and incorrect mathematics in phone photos. [Garrido-Munoz et al., Handwritten Text Recognition: A Survey, v1](https://arxiv.org/html/2502.08417v1)

| Survey method | How it applies here | Priority |
|---|---|---|
| Line recognition (§III-A) | Give a formula reader a padded region containing a whole expression. Preserve numerator and denominator, superscripts and nearby writing. Use a line recognizer for prose. | First CPU pilot. PP-FormulaNet is an independent formula-recognition candidate. TrOCR Small is a single-line text control, trained on IAM handwriting. [PP-FormulaNet](https://arxiv.org/abs/2503.18382) · [TrOCR model card](https://huggingface.co/microsoft/trocr-small-handwritten) |
| CTC and sequence decoders (§III-A) | CTC can train a lightweight reader from a crop and its transcript without character boxes. A sequence decoder can represent structured LaTeX. The project has transcribed lines and root references, so a line-training recipe also needs verified image regions or another alignment method. | Compare pretrained readers before training a new one. [Survey, line methods](https://arxiv.org/html/2502.08417v1#S3.SS1) |
| Masking, unfolding and structured page reading (§III-B) | Keep region coordinates and reconstruct reading order before connecting two steps. Compare short blocks with a whole-page path for side calculations, long division and dense work. | Use DAN's explicit layout-token idea if region order is still a problem after recognition improves. SPAN is a reference for paragraph unfolding. [DAN](https://arxiv.org/abs/2203.12273) · [SPAN](https://arxiv.org/abs/2102.08742) |
| Beam search (§III-C) | When a reader exposes alternate readings, retain a few for ambiguous digits or operators. Feed their visual confidence into the candidate verifier. A math expression can be syntactically valid and numerically wrong. Keep the student's written mistake available as evidence. | Compare one versus a few hypotheses on uncertain crops. Measure how often the correct reading is retained and whether the final rank improves. [Survey, decoding](https://arxiv.org/html/2502.08417v1#S3.SS3) · [OCR over-correction study](https://arxiv.org/abs/2604.22774) |
| Synthetic pretraining and curriculum (§III-B, §IV-A, §V-C) | Start with single formulas and gradually add multiple lines, photographed backgrounds, corrections and mixed layouts. Include correct and incorrect expressions with faithful transcripts. Test any adaptation on real held-out work. | Later, if a pretrained reader establishes usable evidence and training is affordable. [DAN](https://arxiv.org/abs/2203.12273) · [TrOCR paper](https://arxiv.org/abs/2109.10282) |

## CPU check inspired by the survey's metrics

I reran the existing 60-photo English Tesseract pilot. Both alignment methods use the same math tokens and detected line fragments. The ordered version preserves the sequence of lines. The second uses minimum-cost line assignment, which lets the comparison pair lines in any order.

| Diagnostic | Result |
|---|---:|
| Math token error with ordered line alignment | 96.41% |
| Math token error with best line assignment regardless of order | 94.73% |
| Annotated math tokens recovered regardless of order or grouping | 14.14% |
| Exact matches to the first cited root equation in a single OCR line | 0 of 27 eligible photos |

Allowing a different line order reduces the error by **1.69 percentage points**. The low order-free token recall points to weak symbol recovery in this English Tesseract recipe. Cropping can still affect which symbols the reader sees. The line matching comparison cannot tell whether a missing symbol was lost by line detection, recognition, or token normalization. The cited-equation check uses existing teacher line references for evaluation. An exact match requires the complete normalized token sequence in one OCR line. [Audit metrics](../reports/ocr_reading_order_audit.json)

## Matched comparison design

1. Freeze the 60 labelled photos, the Qwen candidate scores and the verifier. Keep each expression crop large enough to retain vertical structure, with a short-block fallback for unusual layout.
2. Compare Tesseract on the full page versus Tesseract on the proposed regions. Compare a pretrained formula reader on those same regions. These comparisons isolate the region change and the reader change.
3. Report symbol and digit recovery, ordered and unordered line scores, exact recovery of cited equations, crop coverage and CPU time. Test the new transcripts in the unchanged root-code verifier on the same photo IDs.
4. If symbol recovery improves and root ranking stays flat, improve the verifier's ability to distinguish specific wrong operations. If unordered reading improves while ordered reading lags, work on the region order. If both reading and ranking improve, repeat on a larger fixed split before promoting the candidate.

The audit can be reproduced with `PYTHONPATH=src .venv/bin/python scripts/audit_ocr_reading_order.py`. It uses the existing private photo sample and writes aggregate metrics. No GPU time is required.

## What the cited papers add

| Paper | Application to these worksheets |
|---|---|
| Wick et al., survey reference [61] | Their sequence decoder uses CTC prefix scores as an additional visual check against skipped and repeated text. For student math, keep a few possible readings of a crop and check the deciding digits, signs and operators against a separate visual recognizer. A mathematically false equality is valid transcription evidence. [Paper](https://arxiv.org/abs/2110.05909) · [Author implementation](https://github.com/Planet-AI-GmbH/tfaip-hybrid-ctc-s2s) |
| DAN and DANIEL, survey references [68] and [168] | Both integrate handwriting recognition with structure. DANIEL also extracts labeled entities. Adapt the output idea by attaching region IDs and an evidence type to candidate decisions. Show the verifier the crop, the surrounding working and the original question. [DAN](https://arxiv.org/abs/2203.12273) · [DANIEL](https://arxiv.org/abs/2407.09103) |
| Garrido-Munoz and Calvo-Zaragoza, reference [194], and Kohút and Hradiš, reference [195] | The first study found that text-domain differences mattered more than visual differences across its HTR datasets. The second tested simple adaptation with few labeled lines. If a reader is worth adapting, use verified crop/transcript pairs from the fitting set, with representative signs, fractions, powers and Indonesian text. The competition provides cited lines in the text and has no corresponding pixel boxes, so alignment is a separate step. [Generalization study](https://arxiv.org/abs/2411.17332) · [CTC adaptation study](https://arxiv.org/abs/2302.06308) · [Autoregressive follow-up](https://arxiv.org/abs/2503.19546) |
| CoMER and FERMAT, complementary work | CoMER's decoder uses past alignments to track which parts of a 2D formula have been read. FERMAT evaluates detection and localization separately on handwritten solutions. Compare a formula specialist on exact cited equations and use a separate metric for the final misconception ranking. [CoMER](https://arxiv.org/abs/2207.04410) · [Author implementation](https://github.com/Green-Wood/CoMER) · [FERMAT](https://aclanthology.org/2025.acl-long.720/) |

## Offline follow-up on the same 60 photos

The CPU pilot compared the existing PP-FormulaNet readings with an autocontrast view of each of the same 265 crops and with the separate Latin PP-OCRv5 CTC reader. The comparison uses their decoded outputs. Wick et al. integrate CTC confidence inside beam search, a different intervention. All rankings below reuse the frozen Qwen3-VL-4B scores and root-example verifier. The 301-row candidate reranker was evaluated separately. [CTC reader checkpoint](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec) · [Aggregate pilot](../reports/htr_hypothesis_pilot.json)

| Evidence fed to Qwen 4B verifier | MAP@3 | Exact teacher-cited math-token sequence |
|---|---:|---:|
| Original PP-FormulaNet | 0.59722 | 1/27 |
| PP-FormulaNet on autocontrast crops | 0.59167 | 0/27 |
| Latin PP-OCRv5 | 0.59722 | 2/27 |
| Max evidence from original or CTC | 0.59444 | 2/27 available across the two readers |

Only two photos had a complete arithmetic equation on which the original and autocontrast views agreed. The formula reader and CTC reader agreed on zero complete arithmetic equations under the strict parse-and-match rule. This is a coverage result for the current crops and readers. The CTC reader recovered fewer annotated numbers than PP-FormulaNet, with higher number precision. The two readers' token errors and coverage make a strict agreement gate too sparse for these photos.

On the full 301-row development fold, a separate oracle diagnostic used teacher transcripts and teacher-marked root locations on 200 rows. Restricting the verifier to the marked lines and their predecessors lowered Qwen 4B MAP@3 from **0.60908** to **0.60133**. Rows without a cited line retained their full transcript. This supports testing a crop as additional evidence while preserving the full context in the verifier. The oracle uses labels unavailable at inference.

The next matched comparison should give the frozen candidate reranker evidence from a math reader that handles 2D expressions. Keep the original question and full working alongside the relevant crop. For each proposed code, record the visible step and whether the candidate is supported, contradicted or uncertain. Measure both the recovery of the teacher-cited expression and the change in root-error MAP@3. Use a wider candidate generator when the existing shortlist lacks the correct code.
