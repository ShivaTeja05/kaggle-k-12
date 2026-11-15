"""The four-slice report: overall, no-error, seen-pair and hard MAP@3 with row counts.

A row's slice is fixed by its true label and by what the fit rows hold for its question:
  none       the true label is NONE;
  seen_pair  an error code that the fit rows already show for this question;
  hard       any other error row: a code new for a fit-seen question (hard_new_code),
             or an error on a question the fit rows never saw (hard_unseen_question).
The definition is a copy of scripts/fuse_candidate.py::unseen_pairs (src/k12 must not import from scripts/);
tests/test_slices.py asserts that the copy and the original give identical masks.

Slices need the true label, so they are for scoring only and can never be an input at test time.
"""
import numpy as np

SLICES = ('none', 'seen_pair', 'hard')
HEADER = f"{'system':<46}{'rows':>5}{'all':>8}{'no-error':>15}{'seen-pair':>15}{'hard':>15}"


def slice_masks(fit, y, qid):
    """Boolean row masks for the three slices and the two kinds of hard row.

    fit is the frame of fit rows (question_id, root_code); y and qid are the scored rows' labels and questions.
    none, seen_pair and hard are disjoint and cover every row; hard = hard_new_code | hard_unseen_question.
    """
    codes = {}
    for q, c in zip(fit['question_id'], fit['root_code']):
        codes.setdefault(q, set()).add(c)
    y, qid = np.asarray(y), np.asarray(qid)
    err = np.asarray(y != 'NONE', bool)
    seen_q = np.array([q in codes for q in qid], bool)
    in_q = np.array([t in codes.get(q, ()) for t, q in zip(y, qid)], bool)
    new_code, unseen_q = seen_q & ~in_q & err, ~seen_q & err
    hard = new_code | unseen_q
    return {'none': ~err, 'seen_pair': err & ~hard, 'hard': hard, 'hard_new_code': new_code, 'hard_unseen_question': unseen_q}


def slice_report(scores, masks, rows=None):
    """Mean row score overall and per slice, with row counts; rows (boolean) restricts the report to a subset.

    An empty slice reports None, not a number, so that a missing slice cannot be read as a score.
    """
    scores = np.asarray(scores, float)
    rows = np.ones(len(scores), bool) if rows is None else np.asarray(rows, bool)
    if any(len(masks[k]) != len(scores) for k in SLICES) or len(rows) != len(scores):
        raise ValueError('Scores, masks and rows must have the same length')
    mean = lambda m: float(scores[m].mean()) if m.any() else None
    report = {'rows': int(rows.sum()), 'map3': mean(rows)}
    report.update({k: mean(rows & masks[k]) for k in SLICES})
    report.update({f'n_{k}': int((rows & masks[k]).sum()) for k in SLICES})
    return report


def format_report(name, report):
    """One aligned text line: name, rows, overall, then each slice as score (rows)."""
    cell = lambda v: '   n/a' if v is None else f'{v:.4f}'
    line = f"{str(name)[:45]:<46}{report['rows']:>5}{cell(report['map3']):>8}"
    for k in SLICES:
        line += f"{cell(report[k]):>9} ({report['n_' + k]:>3})"
    return line


def print_reports(title, named_reports, extra=None, file=None):
    """Print a title, the column header once, then one line per (name, report).

    extra maps a name to text appended to its line (for example a bootstrap range).
    """
    print(f'\n{title}', file=file)
    print(HEADER + ('  ' + extra['header'] if extra and 'header' in extra else ''), file=file)
    for name, report in named_reports:
        tail = '  ' + extra[name] if extra and name in extra else ''
        print(format_report(name, report) + tail, file=file)
