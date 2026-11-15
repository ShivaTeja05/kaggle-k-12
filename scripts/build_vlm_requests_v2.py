"""Build v2 chat requests for the reasoner arms A1 to A4 on development rows (ARCHITECTURE_REFORM_2026.md 4.4.6). CPU.

A1: prompt v2 (line_checks, root-line label, second look), full taxonomy.
A2: A1 + the row's shortlist, descriptions only.
A3: A2 + one worked example per shortlisted code, taken from another question.
A4: A3, but with an example from the row's own question where the fit rows have one.

Same file format as scripts/build_vlm_requests.py (one header line with the schema and classes, then
{"trace_id", "messages"}), same photos and same per-row text; that script is not changed. The examples come from
fit rows only (make_folds != 0), so no development row is ever shown. The rows are the development rows that
fuse_vlm.py sends to the reasoner: top probability below the threshold and a zero-shot answer on file (the shortlist
uses that answer); --include-unanswered adds the sent rows that have no zero-shot answer.

Training photos only. Test rows are refused: their shortlists and cards would need the index rebuilt from all
training rows, and that run needs its own approval.

Usage: python scripts/build_vlm_requests_v2.py --arm all
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_vlm_requests import user_content
from k12.core import label_space, make_folds, read_csv
from k12.exemplars import NONE_DESCRIPTION, NONE_NAME, ExemplarIndex, render_card, same_question
from k12.prompts import answer_schema_v2, key_sentence_v2, shortlist_block, system_prompt_v2

ARMS = {'A1': None, 'A2': 'none', 'A3': 'cross', 'A4': 'same'}  # arm -> card mode (None: no shortlist at all)


def code_texts(taxonomy):
    """{code: name} and {code: description} for the 75 codes and NONE."""
    names = dict(zip(taxonomy.id.astype(str), taxonomy.name.astype(str)), NONE=NONE_NAME)
    descriptions = dict(zip(taxonomy.id.astype(str), taxonomy.description.astype(str)), NONE=NONE_DESCRIPTION)
    return names, descriptions


def cards_for(row, codes, index, names, descriptions, mode, mu=.5):
    """Card text per shortlisted code and where each example came from (None: description only).

    Reads the row's trace_id, question and topic and nothing else; the row's own label is never looked at.
    """
    cards, sources = {}, {}
    for code in codes:
        if mode == 'none':
            entry = None
        elif code == 'NONE':
            entry = index.none_example(row.trace_id, row.question_id, row.topic_name, row.question_text)
        else:
            entry = index.select(code, row.trace_id, row.question_id, row.question_text, mode, mu)
        same = entry is not None and code != 'NONE' and same_question(entry, row.question_id, row.question_text)
        cards[code] = render_card(code, names[code], descriptions[code], entry, same)
        sources[code] = None if entry is None else {'trace_id': entry['trace_id'], 'same_question': same}
    return cards, sources


def row_block(row, codes, index, names, descriptions, mode, mu=.5):
    """The shortlist text placed before the row's photo, and the card sources."""
    cards, sources = cards_for(row, codes, index, names, descriptions, mode, mu)
    return shortlist_block(row.trace_id, codes, cards, {c: n for c, n in names.items() if c not in codes}), sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/bundle/x/competition_data')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--arm', default='all', choices=list(ARMS) + ['all'])
    parser.add_argument('--shortlist', default='data/v10/shortlist_dev.json', help='written by scripts/build_shortlist.py')
    parser.add_argument('--mu', type=float, default=.5, help='penalty on examples whose question resembles the row\'s (4.4.2)')
    parser.add_argument('--include-unanswered', action='store_true', help='also sent rows without a zero-shot answer')
    parser.add_argument('--no-root-line-pattern', action='store_true', help='leave the L<n>/none pattern out of the schema')
    parser.add_argument('--fold', type=int, default=0, help='development fold of make_folds')
    parser.add_argument('--out-dir', default='data/v10')
    args = parser.parse_args()
    if args.split != 'train':
        raise SystemExit('Development (training) photos only: this script does not build test requests')
    data = Path(args.data)
    taxonomy = read_csv(data / 'taxonomy_v1.csv')
    classes = label_space(taxonomy)
    names, descriptions = code_texts(taxonomy)
    train = read_csv(data / 'train.csv')
    folds = make_folds(train)
    dev, fit = train.loc[folds == args.fold].reset_index(drop=True), train.loc[folds != args.fold].reset_index(drop=True)
    shortlist = json.loads(Path(args.shortlist).read_text())
    if set(shortlist) != set(dev.trace_id):
        raise SystemExit('The shortlist file does not cover exactly the development rows')
    keep = [shortlist[t]['sent'] and (args.include_unanswered or shortlist[t]['has_zero_shot']) for t in dev.trace_id]
    rows = dev.loc[keep].reset_index(drop=True)
    index = ExemplarIndex(fit)
    if index.trace_ids() & set(dev.trace_id):
        raise SystemExit('A development row is in the example index')
    system = system_prompt_v2(taxonomy) + key_sentence_v2()
    header = {'header': True, 'schema': answer_schema_v2(classes, not args.no_root_line_pattern), 'classes': classes, 'rows': len(rows)}
    arms = list(ARMS) if args.arm == 'all' else [args.arm]
    suffix = '' if args.mu == .5 else f'_mu{args.mu:g}'
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    targets = {arm: out / f"q38_dev_{arm}{suffix if ARMS[arm] in ('cross', 'same') else ''}_requests.jsonl" for arm in arms}
    handles = {arm: targets[arm].open('w') for arm in arms}
    chars = {arm: [] for arm in arms}
    count = {arm: {'shortlisted_codes': 0, 'error_codes_with_card': 0, 'error_codes_description_only': 0, 'none_with_card': 0,
                   'none_description_only': 0, 'same_question_cards': 0} for arm in arms}
    for arm in arms:
        handles[arm].write(json.dumps(dict(header, arm=arm, prompt='v2', cards=ARMS[arm],
                                           mu=args.mu if ARMS[arm] in ('cross', 'same') else None)) + '\n')
    for row in rows.itertuples(index=False):
        content = user_content(row, data / 'images', 'train')  # the photos are encoded once and shared by the arms
        own = sum(len(part['text']) for part in content if part['type'] == 'text')
        codes = shortlist[row.trace_id]['codes']
        for arm in arms:
            lead = []
            if ARMS[arm] is not None:
                block, sources = row_block(row, codes, index, names, descriptions, ARMS[arm], args.mu)
                lead = [{'type': 'text', 'text': block}]
                c = count[arm]; c['shortlisted_codes'] += len(codes)
                for code, source in sources.items():
                    kind = 'none' if code == 'NONE' else 'error_codes'
                    c[f"{kind}_{'with_card' if source else 'description_only'}"] += 1
                    c['same_question_cards'] += bool(source and source['same_question'])
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': lead + content}]
            handles[arm].write(json.dumps({'trace_id': row.trace_id, 'messages': messages}) + '\n')
            chars[arm].append(len(system) + own + sum(len(part['text']) for part in lead))
    for arm in arms:
        handles[arm].close()
        report = {'arm': arm, 'rows': len(rows), 'out': str(targets[arm]), 'mb': round(targets[arm].stat().st_size / 1e6, 1),
                  'system_chars': len(system),
                  'text_chars': {'min': int(np.min(chars[arm])), 'median': float(np.median(chars[arm])), 'max': int(np.max(chars[arm]))}}
        if ARMS[arm] is not None:
            report.update(count[arm], cards=ARMS[arm])
        print(json.dumps(report))
    print(json.dumps({'dev_rows': len(dev), 'sent': int(sum(shortlist[t]['sent'] for t in dev.trace_id)), 'rows_written': len(rows),
                      'include_unanswered': args.include_unanswered, 'index_error_rows': len(index.entries),
                      'index_correct_rows': len(index.nones), 'index_rows_without_a_root_entry': len(index.skipped),
                      'fit_rows': len(fit), 'mu': args.mu}))


if __name__ == '__main__':
    main()
