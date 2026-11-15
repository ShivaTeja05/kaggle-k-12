"""Charts for the research report, written as plain SVG (no plotting library needed).

Every number here is copied from a measured result recorded in docs/model_audit_plan_2026-11-12.md,
reports/status.json or runs/deep/DEEP_PLAN.md. Run from the repo root:
    python docs/study/report/build_figs.py
"""
from pathlib import Path

OUT = Path(__file__).parent / 'fig'
OUT.mkdir(exist_ok=True)
# Categorical slots in fixed order (validated: adjacent pairs pass colour-blind and normal-vision checks).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2dc', '#ffffff'
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def text(x, y, s, size=11, anchor='middle', fill=INK, weight='normal'):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" {FONT}>{s}</text>'


def top_rounded(x, y, w, h, r=3):
    r = min(r, w / 2, h) if h > 0 else 0
    return (f'M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h:.1f} Z')


def right_rounded(x, y, w, h, r=3):
    r = min(r, h / 2, w) if w > 0 else 0
    return (f'M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'L{x + w:.1f},{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} L{x:.1f},{y + h:.1f} Z')


def frame(w, h, title, body, note=''):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
             f'<rect width="{w}" height="{h}" fill="{SURF}"/>', text(16, 22, title, 13, 'start', INK, 'bold')]
    if note:
        parts.append(text(16, h - 8, note, 9.5, 'start', INK2))
    return '\n'.join(parts + body + ['</svg>'])


def legend(names, x, y):
    out = []
    for i, n in enumerate(names):
        out.append(f'<rect x="{x}" y="{y - 9}" width="11" height="11" rx="2" fill="{C[i]}"/>')
        out.append(text(x + 16, y, n, 10.5, 'start', INK2))
        x += 26 + 6.2 * len(n)
    return out


def grouped(path, title, groups, series, values, ymax=1.0, w=720, h=330, note='', fmt='{:.3f}', ref=None):
    """values[s][g]; a visible value label sits on every bar."""
    L, R, T, B = 46, 14, 62, 58 if note else 44
    pw, ph = w - L - R, h - T - B
    body = legend(series, L, 44) if len(series) > 1 else []
    for k in range(6):
        v = ymax * k / 5
        y = T + ph - ph * v / ymax
        body.append(f'<line x1="{L}" y1="{y:.1f}" x2="{w - R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(L - 6, y + 3.5, f'{v:.1f}', 9.5, 'end', INK2))
    if ref:
        y = T + ph - ph * ref[0] / ymax
        body.append(f'<line x1="{L}" y1="{y:.1f}" x2="{w - R}" y2="{y:.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 3"/>')
        body.append(text(w - R - 2, y - 4, ref[1], 9.5, 'end', INK2))
    gw = pw / len(groups)
    bw = min(46, (gw - 22) / len(series) - 2)
    for g, name in enumerate(groups):
        x0 = L + gw * g + (gw - (bw + 2) * len(series) + 2) / 2
        for s in range(len(series)):
            v = values[s][g]
            if v is None:
                continue
            bh = ph * v / ymax
            x = x0 + (bw + 2) * s
            body.append(f'<path d="{top_rounded(x, T + ph - bh, bw, bh)}" fill="{C[s]}"/>')
            body.append(text(x + bw / 2, T + ph - bh - 4, fmt.format(v), 9 if bw < 34 else 9.5))
        for j, line in enumerate(name.split('\n')):
            body.append(text(L + gw * g + gw / 2, T + ph + 15 + 12 * j, line, 10, 'middle', INK2))
    body.append(f'<line x1="{L}" y1="{T + ph}" x2="{w - R}" y2="{T + ph}" stroke="{INK2}" stroke-width="1"/>')
    path.write_text(frame(w, h, title, body, note))


def hbars(path, title, names, values, xmax=1.0, w=720, note='', mark=None, highlight=()):
    L, R, T, row = 300, 60, 58, 24
    h = T + row * len(names) + (38 if note else 22)
    pw = w - L - R
    body = []
    for k in range(6):
        v = xmax * k / 5
        x = L + pw * v / xmax
        body.append(f'<line x1="{x:.1f}" y1="{T - 4}" x2="{x:.1f}" y2="{T + row * len(names)}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(x, T - 8, f'{v:.1f}', 9.5, 'middle', INK2))
    if mark:
        x = L + pw * mark[0] / xmax
        body.append(f'<line x1="{x:.1f}" y1="{T - 4}" x2="{x:.1f}" y2="{T + row * len(names)}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 3"/>')
        body.append(text(x + 3, T + row * len(names) + 12, mark[1], 9.5, 'start', INK2))
    for i, (n, v) in enumerate(zip(names, values)):
        y = T + row * i + 5
        body.append(text(L - 8, y + 10, n, 10.5, 'end', INK, 'bold' if i in highlight else 'normal'))
        body.append(f'<path d="{right_rounded(L, y, pw * v / xmax, 14)}" fill="{C[1] if i in highlight else C[0]}"/>')
        body.append(text(L + pw * v / xmax + 5, y + 10.5, f'{v:.3f}', 10, 'start'))
    path.write_text(frame(w, h, title, body, note))


def lines(path, title, xs, xlabels, series, values, ymin, ymax, w=720, h=330, note='', refs=(), point_labels=None, fmt='{:.3f}', ticks=5):
    L, R, T = 46, 110, 62
    B = 22 + 11 * max(len(x.split('\n')) for x in xlabels) + (20 if note else 0)
    pw, ph = w - L - R, h - T - B
    X = lambda i: L + 34 + (pw - 68) * (xs[i] - xs[0]) / (xs[-1] - xs[0])
    Y = lambda v: T + ph - ph * (v - ymin) / (ymax - ymin)
    body = legend(series, L, 44) if len(series) > 1 else []
    for k in range(ticks + 1):
        v = ymin + (ymax - ymin) * k / ticks
        body.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(L - 6, Y(v) + 3.5, f'{v:.2f}', 9.5, 'end', INK2))
    for v, label in refs:
        body.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 3"/>')
        body.append(text(L + pw + 5, Y(v) + 3.5, label, 9.5, 'start', INK2))
    for i, lab in enumerate(xlabels):
        for j, line in enumerate(lab.split('\n')):
            body.append(text(X(i), T + ph + 15 + 11 * j, line, 9.5, 'middle', INK2))
    for s, name in enumerate(series):
        pts = ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(values[s]))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{C[s]}" stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(values[s]):
            body.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{C[s]}" stroke="{SURF}" stroke-width="2"/>')
            if point_labels is None or point_labels[s]:
                body.append(text(X(i), Y(v) - 9, fmt.format(v), 9.5))
        if len(series) > 1:
            body.append(text(X(len(xs) - 1) + 9, Y(values[s][-1]) + 3.5, name, 9.5, 'start', INK2))
    body.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="{INK2}" stroke-width="1"/>')
    path.write_text(frame(w, h, title, body, note))


# 1. Public leaderboard score of each submission.
lines(OUT / 'c1_public_timeline.svg', 'Public leaderboard MAP@3 of each submission',
      [0, 1, 2, 3, 4, 5, 6],
      ['8 Nov\n4B image\nclassifier', '8 Nov\n+ prior and\ntemperature', '11 Nov\n8B image +\n4B blend', '11 Nov\nread, then\ndiagnose', '11 Nov\n+ Claude\ndistilled image', '11 Nov\n3-way blend\n(+32B text)', '12 Nov\n+ label\nbalancing'],
      ['Public score'], [[0.58216, 0.598, 0.62383, 0.66183, 0.68283, 0.7085, 0.71716]], 0.50, 0.90, h=360, ticks=4,
      refs=[(0.8465, 'top of board 0.8465'), (0.80, 'target 0.80')],
      note='Top-of-board value was read on 9 Nov. Seven distinct submissions; one accidental duplicate of the sixth is not shown.')

# 2. Where the gap to Claude lives.
grouped(OUT / 'c2_row_types.svg', 'Dev MAP@3 by row type: our blend against Claude',
        ['No error\n(101 rows)', 'Error, code seen before\non this question (94)', 'Error, code new for the\nquestion or new question (106)', 'All rows\n(301)'],
        ['Our blend + balancing', 'Claude Opus with the photo'], [[0.926, 0.856, 0.451, 0.737], [0.934, 0.812, 0.715, 0.819]],
        note='The whole gap sits in the third group. About 0.63 there would give 0.80 overall.')

# 3. Every model on the hard rows.
names = ['Claude Opus, photo', 'Claude Opus, our transcript only', 'Our blend + balancing', 'Qwen3-235B, untrained, transcript', 'Old 8B text student (11.4k Claude rows)',
         'New reason-first 8B student', 'Qwen3-32B, untrained, transcript', 'Text 8B, teacher labels only, 3 epochs', 'Image 8B (deployed member)', 'Text 32B (deployed member)',
         'Qwen3.5-27B, untrained, photo', 'Qwen3.5-9B, untrained, photo']
vals = [0.715, 0.637, 0.451, 0.448, 0.440, 0.417, 0.392, 0.363, 0.362, 0.357, 0.314, 0.165]
hbars(OUT / 'c3_hard_rows_by_model.svg', 'Dev MAP@3 on the 106 hard rows, every model we have run', names, vals,
      mark=(0.63, 'needed for 0.80 overall'), highlight=(0, 1), note='Orange = Claude. No open model, trained or not, gets past about 0.45.')

# 4. Memorisation step curve.
grouped(OUT / 'c4_pair_examples.svg', 'Error rows: score by number of teacher-labelled examples of the exact question-and-code pair',
        ['0 examples\n(106 rows)', '1 example\n(61)', '2 examples\n(15)', '3 or more\n(18)'],
        ['Our blend + balancing', 'Old 8B text student'], [[0.451, 0.798, 0.922, 1.000], [0.440, 0.694, 0.756, 0.917]], ymax=1.1,
        note='One example of the exact pair is worth about +0.35. Examples of the same code on other questions add almost nothing.')

# 5. More Claude-labelled data.
grouped(OUT / 'c5_claude_data_scaling.svg', 'Text 8B student as Claude-labelled training rows grow (1 epoch each)',
        ['0 Claude rows', '2,856 Claude rows', '11,426 Claude rows'],
        ['All dev rows', 'Hard rows', 'Agrees with Claude on hard rows'],
        [[0.553, 0.666, 0.692], [0.297, 0.420, 0.440], [0.264, 0.377, 0.491]],
        note='Fourfold more data moved the hard rows by +0.02. Agreement with Claude keeps rising but is far from the 0.80 needed.')

# 6. Hard rows split by example coverage.
grouped(OUT / 'c6_groups_abc.svg', 'Hard rows split by whether an example of the pair exists anywhere',
        ['A. Claude labelled this pair\non another photo (32 rows)', 'B. Question has pool photos,\nbut not this error (31)', 'C. Question has no\npool photos (43)'],
        ['Old 8B student', 'Our blend', 'Claude, our transcript', 'Claude, photo'],
        [[0.677, 0.204, 0.434], [0.573, 0.290, 0.477], [0.688, 0.505, 0.694], [0.839, 0.586, 0.717]],
        note='In group A our student equals Claude reading the same transcript. Groups B and C are a quarter of all dev rows.')

# 7. Paired catalogue test.
grouped(OUT / 'c7_catalogue_arms.svg', 'Paired training test of the error catalogue (machine transcripts)',
        ['All 301', 'Target rows\nB + C (74)', 'Group A\n(32)', 'Seen pair\n(94)', 'No error\n(101)'],
        ['Arm 1: without catalogue', 'Arm 2: with 1,503 catalogue rows'],
        [[0.666, 0.360, 0.557, 0.681, 0.911], [0.648, 0.437, 0.375, 0.674, 0.866]],
        note='Same recipe and seed. Target rows rose by 0.077; group A fell by 0.182 and no-error rows by 0.045. Pass mark not met.')

# 8. Candidate student against the gates' reference.
grouped(OUT / 'c8_student_vs_others.svg', 'Reason-first student (Stage B) against earlier models, dev MAP@3',
        ['All rows', 'No error', 'Seen pair', 'Hard rows'],
        ['New reason-first student', 'Old 8B student', 'Blend + balancing', 'Claude, photo'],
        [[0.625, 0.822, 0.647, 0.417], [0.692, 0.906, 0.746, 0.440], [0.737, 0.926, 0.856, 0.451], [0.819, 0.934, 0.812, 0.715]],
        note='The student wrote well-formed diagnoses (99.7% parsed) but did not diagnose better than the classifier student.')
print('wrote', len(list(OUT.glob('c*.svg'))), 'charts to', OUT)
