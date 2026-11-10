"""Screen a frozen Qwen3.5-4B image reviewer on a fixed offline CPU cohort."""
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from evaluate_formula_reader import cohort
from evaluate_root_examples import combine_model_rankings
from evaluate_working_verifier import summarize
from k12.core import aligned_bundle, label_space, read_csv, row_scores, topk


MODEL = Path('artifacts/qwen35-cpu')
SEED = 1039
ROWS = 16


def review(row, photo, candidates, descriptions, timeout):
    by_letter = {chr(65 + i): code for i, code in enumerate(sorted(candidates))}
    choices = '\n'.join(f'{letter}: {descriptions[code]}' for letter, code in by_letter.items())
    prompt = (
        "Diagnose the main misconception from the photographed student work. "
        "Use visible active steps, respect corrected attempts, and accept equivalent methods. "
        "Include no-major-error as a possibility. A final answer alone is weak evidence. "
        "Rank three distinct candidate letters in a JSON array, most likely first.\n"
        f"Question: {row.question_text}\nOfficial answer: {row.correct_answer}\nCandidates:\n{choices}"
    )
    # Use alphabetic aliases so the constrained output has short tokens for every class.
    options = ' | '.join(json.dumps(json.dumps(letter)) for letter in by_letter)
    grammar = f'root ::= "[" choice "," choice "," choice "]"\nchoice ::= {options}'
    command = ['llama-mtmd-cli', '-m', str(MODEL / 'Qwen3.5-4B-Q4_K_M.gguf'),
               '--mmproj', str(MODEL / 'mmproj-F16.gguf'), '--image', str(photo),
               '--offline', '--device', 'none', '-ngl', '0', '--no-op-offload', '--no-mmproj-offload',
               '-t', '8', '-tb', '8', '-c', '4096', '--image-max-tokens', '512', '--no-warmup',
               '--seed', str(SEED), '--temp', '0', '-n', '24', '--grammar', grammar, '-p', prompt]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TimeoutError('The offline image reviewer exceeded its row timeout') from None
    if result.returncode:
        # Keep private prompt and output text out of diagnostics.
        raise RuntimeError(f'Offline llama-mtmd-cli exited with status {result.returncode}')
    matches = re.findall(r'\[\s*"[A-I]"\s*,\s*"[A-I]"\s*,\s*"[A-I]"\s*\]', result.stdout)
    if not matches:
        return []
    return list(dict.fromkeys(by_letter[letter] for letter in json.loads(matches[-1]) if letter in by_letter))


def main():
    _, development, audit = cohort()
    eligible = audit.loc[audit.question_image_file.str.strip().eq('')]
    sample = eligible.sample(n=ROWS, random_state=SEED)
    files = {p.name: p for p in Path('data/cpu-image-audit').rglob('*.jpeg')}
    if not set(sample.image_file) <= files.keys():
        raise ValueError('The fixed development photo cohort is incomplete')
    tax = read_csv('data/raw/taxonomy_v1.csv')
    classes = label_space(tax)
    descriptions = {r.id: f'{r.name}. {r.description[:180]}' for r in tax.itertuples(index=False)}
    descriptions['NONE'] = 'No major misconception. Correct reasoning or a minor slip only.'
    ids = development.trace_id.tolist()
    base = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', ids, classes)[sample.index]
    large = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz', ids, classes)[sample.index]
    shortlist = topk(base, classes, 8)
    scores = np.full_like(base, 1e-8)
    truth = sample.root_code.to_numpy()
    class_index = {code: index for index, code in enumerate(classes)}
    candidates = [list(dict.fromkeys([*row, 'NONE'])) for row in shortlist]
    start = time.monotonic()
    valid_counts = []
    for i, (row, choices) in enumerate(zip(sample.itertuples(index=False), candidates)):
        remaining = 540 - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError('The CPU pilot reached its nine-minute limit')
        codes = review(row, files[row.image_file], choices, descriptions, min(75, remaining))
        valid_counts.append(len(codes))
        filled = list(dict.fromkeys([*codes, *choices]))[:3]
        for rank, code in enumerate(filled, 1):
            scores[i, class_index[code]] = 1 / rank
        print(f'Reviewed {i + 1}/{ROWS} photos in {time.monotonic() - start:.1f}s', flush=True)

    models = {'calibrated_qwen3_vl_4b': base, 'frozen_qwen3_5_27b': large,
              'qwen3_5_4b_shortlist': scores,
              'calibrated_plus_qwen3_5_4b': combine_model_rankings(base, scores, classes)}
    missed = row_scores(truth, topk(base, classes)) == 0
    report = {'rows': len(sample), 'root_error_rows': int(sum(truth != 'NONE')),
              'eligible_rows': len(eligible), 'seed': SEED, 'device': 'cpu',
              'model': 'unsloth/Qwen3.5-4B-GGUF Q4_K_M with F16 projector',
              'revision': 'bd31bd8cdaf6621fe51e7c2823a6103a810e48b0',
              'scope': 'Exploratory development pilot. Random subset of the existing photo audit, '
                       'restricted to rows without question diagrams. No training or threshold tuning. '
                       'Candidate order is alphabetical by code. Invalid or missing ranks use the '
                       'calibrated model order. All inputs and inference stay local.',
              'rows_with_three_distinct_model_codes': valid_counts.count(3),
              'rows_with_no_model_codes': valid_counts.count(0),
              'shortlist_recall': float(np.mean([code in choices for code, choices in zip(truth, candidates)])),
              'wall_seconds': time.monotonic() - start, 'metrics': {}}
    for name, values in models.items():
        hits = row_scores(truth, topk(values, classes)) > 0
        report['metrics'][name] = {**summarize(sample, values, classes, base),
                                  'recovered_baseline_top3_misses': int(sum(missed & hits)),
                                  'lost_baseline_top3_hits': int(sum(~missed & ~hits))}
    path = Path('reports/qwen35_4b_cpu_shortlist.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
