"""Do our two readers keep the student's error as written? The two audits of ARCHITECTURE_REFORM_2026.md 4.1.1 (CPU).

Both audits compare a machine transcript with the teacher's annotated root error (the root entry of `misconceptions`:
step, span, text) on development rows only. Nothing here reads a test row, a photo, a key file or the network.

  transcriber  runs/pod/tx/tx_dev.json, the 200 dev error rows (the only out-of-fold machine transcripts).
  reasoner     the `transcription` field of runs/deep/q38/pod_out/q38_dev_think.jsonl (270 rows; 169 error rows).

Every heuristic and threshold, in the order it is applied:

1. Normalisation (`clean`; `glue` = a cleaned string without whitespace and with ',' between digits written '.', so
   202,5 and 202.5 are the same text). LaTeX-ish wrappers are unwrapped (\\dfrac{a}{b} -> a/b, \\sqrt{a} -> √a,
   \\text{..}, \\left, \\right, $, braces); a run of superscript digits becomes ^digits, ^(n) becomes ^n, ^3√ becomes
   ∛, sqrt becomes √; NFKC; lower case; every dash/minus -> '-', × ✕ * -> 'x', · • ・ -> '.', ÷ -> ':', ⁄ -> '/',
   ∆ -> δ, primes and curly quotes -> ', every arrow (-> => ⇒ ⟹) -> '→'; notes are dropped: [..] always, (..) when it holds only words (at least
   five letters, every word three letters or more, no sin/cos/tan/log/ln) or at least three such words; 'L<n> |'
   prefixes are dropped, also inside a span. Lines without a prefix are kept (the reasoner does not always number
   its lines). A line that is empty after this is ignored. A dangling '=' at the end of a span is dropped.
2. Root line. The teacher line named by `step` that contains the span; for a range (36 of 200 rows) the line of the
   range that contains the span, else the lines of the range joined. Line labels can repeat on a page, so a line
   that contains the span wins over one that only carries the label. `span_literal` says whether the span occurs in
   the teacher's own transcript at all (a few spans are the annotator's description, not transcript text).
3. KEPT, any of three, all against the whole machine page (lines joined):
   exact   glue(span) is a substring of glue(page);
   tokens  the span's content tokens are a contiguous run of the page's content tokens. Tokens are numbers
           (12, 0.85, 10.000), words and single symbols; 'x' between numbers or brackets is the token '×'; the soft
           tokens ( ) . , ; : ' " | are ignored, except that ':' between numbers is division and is kept;
   fuzzy   some window of page tokens (span length ±FUZZY_SLACK=2 tokens; a window of another length must start and
           end on the span's first and last token) has difflib ratio >= FUZZY_T = 0.85 with the span tokens AND
           exactly the same multiset of number tokens and of operator tokens (+ - × : / = ^ < > √ %). So a changed
           number or sign is never "kept"; a misread letter or word inside a longer span can be.
   Brackets can be the error itself (3√7 - √3 against 3(√7 - √3)), so a tokens or fuzzy match is overruled, and the
   row goes to rule 5, when the aligned machine line now ends in the answer key and the teacher's line does not.
   The stricter `kept_in_line` also requires glue(span) inside the machine lines aligned to the root line (4).
   Spans shorter than SHORT_SPAN = 4 characters after glue match almost anywhere; they are counted, and the rate
   is also given without them. A span that is EMPTY after rule 1 (4 of the 200: the annotator's span is only a
   [note] or a words-only bracket) cannot be looked for at all: the row is NOT CHECKABLE and is left out of every
   KEPT and LOST count. (The two older definitions, "light" and "reading-impact", count 3 of them as present; their
   lines are kept as recorded so that 0.555 and 0.605 reproduce.)
4. Alignment. sim = 1 - d / len(root line), d = the smallest edit distance between the glued teacher root line
   and any substring of the glued machine page (so merged or split lines do not matter). The line is ALIGNED
   when sim >= ALIGN_T = 0.50; the aligned machine text is every machine line that holds half of the best substring
   or is half covered by it. Short root lines align by chance; no positional information is used.
5. Fate of a span that is not KEPT (first rule that fires):
   SILENTLY FIXED   (a) page: `final_answer_correct` is false, the teacher's final value is not the answer key and
                    the machine transcript's final value is; or, with the line ALIGNED, (b) a teacher relation
                    (= < >) in the span or root line that we can evaluate is false, and the aligned machine text
                    has a relation sharing one side's value with it that holds; or (c) the aligned machine text's
                    final value equals the answer key while the teacher's root line's final value does not.
   DROPPED          not aligned (4).
   MISREAD          aligned, and the machine's version of a false teacher relation (shares one side's value) is
                    still false; or some other evaluable relation in the aligned machine text is false; or the root
                    line is the teacher's last line, the machine's value there is a number, and it differs from
                    both the answer key and the teacher's value. `same_wrong_equation` counts MISREAD rows whose
                    false relation has both sides equal to the teacher's (the error survived in substance).
   ALTERED-UNKNOWN  aligned, span absent, and no rule can be checked (symbolic or conceptual line, nothing to
                    evaluate). Sub-split by whether a number of the span is missing from the aligned machine text.
6. Evaluable relation. Adjacent sides of = < > in one line; a side is the text after the last ', ' or ';' (left
   side) or before the first (right side); '36 = 1, 2, 3' is a list and is skipped. A side evaluates only if, after
   dropping 'rp', a leading word label ('kpk :') and one trailing unit from a fixed list, it is made of digits,
   + - x : / ^ ( ) √ % and separators: 'x' is multiplication only between digits or brackets, ':' is division, a/b
   of two integers binds first, 'a b/c' with b < c is a mixed number, a spaced dot multiplies, n% is n/100. A number with one separator and exactly
   three digits after it (7.200, 65,536) is read both as thousands and as a decimal, and a side with a single
   top-level '/' is also read as one tall fraction (31x25/40x25): a relation holds if it holds under any shared
   reading and is false only if it fails under all. A lone decimal literal with d decimals allows a difference
   below 10^-d (rounding or truncation). Anything else is not evaluable.
7. Final-answer flips (4.1.1). The final value of a transcript is the text after the last '=' of its last non-empty
   line (the whole line when it has no '='), without trailing ticks, slashes and full stops, one pair of enclosing
   brackets, or label words before a number ('jab. 11 kali' is '11 kali'), compared with the
   answer key the same way (numbers by value under rule 6, otherwise glued strings with units removed).
   t != a* and t = a* come from `final_answer_correct`. strict: key, teacher value and machine value are all
   numbers and the teacher's parsed value agrees with `final_answer_correct`; wide: the teacher's parsed value
   agrees with the label and the machine's last line has a value (string equality counts). Rows outside a tier are
   not parseable. Flip-to-right needs a positive match and is reliable; flip-to-wrong also counts a machine page
   that ends on another line or adds a note, so it is an upper bound, loosest in the wide tier.
8. Line position. first half: root line index <= n/2 of the teacher's n lines; second half: index > n/2 with n >= 2;
   single-line pages are listed on their own.
9. Shares carry binomial standard errors sqrt(p(1-p)/n). The paired comparison uses the checkable error rows both
   systems cover and an exact two-sided McNemar test on the discordant pairs.
10. Hand reading (HAND, HAND_FLIPS). Every lost span and every flagged flip-to-wrong was read once by a reviewer on
   14 November 2026, teacher transcript against machine transcript, without the photos. Verdicts: FORMAT (the same
   content in another rendering: a merged line, a bracketed fraction, a spelling variant, a table written as a note;
   the span was in fact kept), FIXED (the error is corrected), ALTERED (changed and not corrected: the design's
   "misread", including partial loss), UNCLEAR. What the reading found, and why the printed claims are worded as
   they are: rule 5's SILENTLY FIXED and MISREAD agreed with the reading on 22 of 23 rows; DROPPED on 0 of 4 (the
   line was there each time, as a table or long-division note, so DROPPED means only "no line aligned"); and
   ALTERED-UNKNOWN, 59 of the 86 lost rows, hid 5 more fixes and 15 FORMAT rows. So the rule-decided fate shares
   are lower bounds, not a distribution, and KEPT is a lower bound that is looser for the reasoner (10 of its 33
   lost rows were FORMAT, against 6 of 53), which writes in its own notation. The reader knew which system wrote
   each page, and did not re-read the rows the rules call KEPT beyond the 17 found by the tokens or fuzzy test
   (all 17 held up). Rows are looked up by trace_id, so a later change to the rules shows up as disagreement or as
   rows without a verdict, never silently.

Section 6 asks every evaluation for the four-slice MAP@3 report, so the script also prints it for the submitted
system and for the reasoner (rows without a reasoner answer keep the submitted system) on 301 and 270 rows, and
MAP@3 on error rows split by whether the reader kept the span. That part needs the git-ignored dev bundles and is
skipped with a note when they are absent.

Usage: PYTHONPATH=src:scripts python scripts/audit_transcript_faithfulness.py [--examples 12] [--seed 0]
Writes reports/transcript_faithfulness.json.
"""
import argparse
import ast
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from k12.core import label_space, make_folds, normalize, read_csv, row_scores, topk

ROOT = Path(__file__).resolve().parents[1]
TX_DEV, Q38_DEV = 'runs/pod/tx/tx_dev.json', 'runs/deep/q38/pod_out/q38_dev_think.jsonl'
RANKER_DEV = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz'
ALIGN_T, FUZZY_T, FUZZY_SLACK, SHORT_SPAN, RECORDED = .50, .85, 2, 4, .555
FATES = ('SILENTLY FIXED', 'DROPPED', 'MISREAD', 'ALTERED-UNKNOWN')
SUPER = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
SWAPS = ([(c, '-') for c in '−–—‐‑‒﹣－'] + [(c, 'x') for c in '×✕✖*'] + [(c, '.') for c in '·•・⋅'] + [(c, "'") for c in '′’‘`'] +
         [('÷', ':'), ('⁄', '/'), ('∆', 'δ'), ('->', '→'), ('=>', '→'), ('⇒', '→'), ('⟹', '→'), ('⟶', '→')])
SOFT, OPS = set('().,;:\'"|'), set('+-×:/=^<>√%')
UNIT = (r'(?:(?:\s*(?:°\s*c|°|cm|mm|km|kg|mg|ml|liter|jam|menit|detik|hari|tahun|bulan|buku|siswa|orang|ton|hektar|kotak|gelas|kali|buah|meter)'
        r'|\s+[mlg])(?:\s?\^\s?[23]|[23])?)?')  # single-letter units need a space, so 3m and 2a stay variables
NUMBER = r'\d+(?:[.,]\d+)*'
# Rule 10: one reviewer's verdict for every lost span (dev rows only; transcripts side by side, no photos).
HAND = {
    'transcriber': {
        'FIXED': 'f8cb6f0c8b33 6278eab2db52 daff31cd0287 58fac67fa577 415a8b9a1f78 9b5c7bcf0ee6 c015327e1a40 e0cb7149cd89 '
                 'aed30de55caf 021d266fb934 acf506d66c56',
        'FORMAT': 'd3b9553a37ac 1d6013727219 31b770a32395 9654788547a2 0b1111494202 5c00e2d070a4 8b8a60bbcd12',
        'UNCLEAR': 'a9b7b9d40077 c5c1f0e04008',
        'ALTERED': 'ee7781fcae70 cb3ee732db83 73dac44ed0c2 49ff9fd9fd33 6ce3e471f3d8 80133dd138bc a55528697bb7 7a4e941d68a0 '
                   '24c8d5c9a081 14376cf00a8f ae6a00c3db40 1c3a60a09db1 d62b1cd0eecb ea040c36afa8 befb9c3c7a25 8f83d7e7f6b3 '
                   'd903bb0cd5e7 8ca966c1663e 8977c5fe6eae 929166f0728a 8c7b49a5ef8f e39469134ffd 975e119a22b9 2a0ba97dff8e '
                   '9e34364217fb 28a7850d12c7 868ef0f1ce97 42b91d87d851 0b5a6c49d27b 4da5dfaed034 d8c0a6cffe5d 4e720ec73bab '
                   '9bd9f3b0fc19 b5846b60265c'},
    'reasoner': {
        'FIXED': 'a3f474c69429 9b5c7bcf0ee6 929166f0728a 7ea0ea90c0b5 c015327e1a40 31b770a32395',
        'FORMAT': 'd3b9553a37ac 24c8d5c9a081 1d6013727219 f8cb6f0c8b33 f30fff9641a9 ae6a00c3db40 49ff9fd9fd33 1e9349c8a795 '
                  '0b1111494202 8ca966c1663e',
        'UNCLEAR': 'a9b7b9d40077',
        'ALTERED': '547d395b95ea ee7781fcae70 befb9c3c7a25 80133dd138bc a55528697bb7 7a4e941d68a0 1c3a60a09db1 8f83d7e7f6b3 '
                   '021d266fb934 e39469134ffd db13a4797d4f 21537872c60c cc7d33fa2df1 868ef0f1ce97 42b91d87d851 d91d437f0a67'}}
HAND = {s: {t: verdict for verdict, ids in v.items() for t in ids.split()} for s, v in HAND.items()}
VERDICTS = ('FORMAT', 'FIXED', 'ALTERED', 'UNCLEAR')
RULE_TO_HAND = {'SILENTLY FIXED': 'FIXED', 'MISREAD': 'ALTERED', 'DROPPED': 'DROPPED'}  # no row was read as dropped
# Flagged flips to wrong (wide tier) that are real on reading the last lines: three misread values and one page whose
# final line is missing. Every other flagged row ends on another line or note, or (one row) carries '(sum)' before the value.
HAND_FLIPS = {'transcriber': set(), 'reasoner': set('8859cb21788c ffe05e6614d9 e4fd90cab7e7 e917ed9b6ff1'.split())}


# ---------- normalisation (rule 1) ----------

def unwrap_latex(s):
    """LaTeX-ish wrappers to the plain forms the transcripts use (answer keys are the main users)."""
    s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '°', s.replace('{,}', ','))
    for _ in range(3):  # nested fractions
        s = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}',
                   lambda m: ' ' + '/'.join(x if re.fullmatch(r'\w+', x.strip()) else f'({x})' for x in m.groups()), s)
    s = re.sub(r'\\sqrt\s*\[3\]\s*\{([^{}]*)\}', r'∛(\1)', s)
    s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', lambda m: '√' + (m.group(1) if re.fullmatch(r'\w+', m.group(1).strip()) else f'({m.group(1)})'), s)
    s = re.sub(r'\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}', r' \1 ', s)
    for a, b in ((r'\times', '×'), (r'\cdot', '·'), (r'\div', '÷'), (r'\left', ''), (r'\right', ''), (r'\circ', '°'), (r'\pi', 'π'),
                 (r'\le', '≤'), (r'\ge', '≥'), (r'\ ', ' '), (r'\,', ' '), ('$', '')):
        s = s.replace(a, b)
    s = re.sub(r'\^\s*\{([^{}]*)\}', lambda m: '^' + (m.group(1) if re.fullmatch(r'\w+', m.group(1).strip()) else f'({m.group(1)})'), s)
    return s.replace('\\', '').replace('{', '').replace('}', '')


def is_note(inner):
    """A round-bracket group that is an annotation, not mathematics: only words, or at least three real words."""
    words = re.findall(r'[^\W\d_]+', inner)
    real = [w for w in words if len(w) >= 3 and w not in ('sin', 'cos', 'tan', 'log', 'ln')]
    only_words = bool(words) and len(real) == len(words) and re.fullmatch(r'[^\W\d_]+(?: [^\W\d_]+)*', inner.strip()) and sum(map(len, words)) >= 5
    return bool(only_words) or len(real) >= 3


def clean(s):
    """Rule 1 with single spaces kept (the arithmetic checker and the tokeniser need the spaces)."""
    s = unwrap_latex(str(s))
    s = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+', lambda m: '^' + m.group().translate(SUPER), s)
    s = unicodedata.normalize('NFKC', s).lower()
    for a, b in SWAPS:
        s = s.replace(a, b)
    s = re.sub(r'\[[^\]]*\]', ' ', s)
    s = re.sub(r'\(([^()]*)\)', lambda m: ' ' if is_note(m.group(1)) else m.group(), s)
    s = re.sub(r'\^\(\s*([-+]?\w+)\s*\)', r'^\1', s).replace('sqrt', '√').replace('^3√', '∛')
    return ' '.join(s.split())


def glue(cleaned):
    """A cleaned string without whitespace and with ',' between digits written '.', for substring tests and edit distances."""
    return re.sub(r'(?<=\d),(?=\d)', '.', re.sub(r'\s+', '', cleaned))


def page_lines(text):
    """[(label or '', cleaned line)] for every line that still has content after cleaning."""
    out = []
    for raw in str(text).splitlines():
        m = re.match(r'\s*(L\d+)\s*[|:]\s?(.*)', raw)
        label, body = (m.group(1), m.group(2)) if m else ('', raw)
        body = clean(body)
        if body:
            out.append((label, body))
    return out


def clean_span(span):
    """A span cleaned like a page: line prefixes dropped, lines joined, a dangling '=' at the end removed."""
    return re.sub(r'\s*=$', '', ' '.join(b for _, b in page_lines(span)))


def tokens(s):
    """Content tokens of a cleaned string: numbers, words, symbols; soft punctuation dropped; 'x' and ':' between numbers kept as operators."""
    toks = re.findall(NUMBER + r'|[^\W\d_]+|\S', s)
    keep = []
    for i, t in enumerate(toks):
        before = 0 < i and (toks[i - 1][0].isdigit() or toks[i - 1] == ')')
        after = i < len(toks) - 1 and (toks[i + 1][0].isdigit() or toks[i + 1] in '(√')
        if t == 'x' and before and after:
            keep.append('×')
        elif t not in SOFT or (t == ':' and before and (after or toks[i + 1:i + 2] == ['-'])):
            keep.append(t.replace(',', '.') if t[0].isdigit() else t)  # 202,5 and 202.5 are the same token
    return keep


# ---------- matching (rules 2 to 4) ----------

def substring_match(pattern, text):
    """(edit distance, start, end) of the substring of text closest to pattern (semi-global alignment)."""
    if not pattern:
        return 0, 0, 0
    if not text:
        return len(pattern), 0, 0
    P, T = np.frombuffer(pattern.encode('utf-32-le'), np.uint32), np.frombuffer(text.encode('utf-32-le'), np.uint32)

    def last_row(P, T, free_start):
        ar = np.arange(len(T) + 1)
        prev = np.zeros(len(T) + 1, int) if free_start else ar.copy()
        for i, c in enumerate(P, 1):
            cur = np.empty_like(prev); cur[0] = i
            cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (T != c))
            prev = np.minimum.accumulate(cur - ar) + ar  # insertions: cur[j] = min(cur[j], cur[j-1] + 1)
        return prev

    row = last_row(P, T, True)
    end = int(np.argmin(row)); d = int(row[end])
    back = last_row(P[::-1], T[:end][::-1], False)  # anchored at `end`, free length backwards
    return d, end - int(np.argmin(back)), end


def kept(span, lines):
    """Rule 3: '' when the span is lost (or empty, so not checkable), else which test found it. `span` is already cleaned."""
    page, sp = ''.join(glue(b) for _, b in lines), glue(span)
    if not sp:
        return ''  # nothing to look for; audit_row marks the row NOT CHECKABLE
    if sp in page:
        return 'exact'
    st, pt = tokens(span), tokens(' '.join(b for _, b in lines))
    n = len(st)
    if not n:
        return ''
    # whether an 'x' is read as the operator depends on its neighbours ('a x √a' against '(a) x √a'), so the run test
    # compares the letter itself; the fuzzy rule below keeps counting operators strictly
    letter = lambda toks: ['x' if t == '×' else t for t in toks]
    ls, lp = letter(st), letter(pt)
    if any(lp[i:i + n] == ls for i in range(len(lp) - n + 1)):
        return 'tokens'
    nums = Counter(t for t in st if t[0].isdigit()); ops = Counter(t for t in st if t in OPS)
    for size in range(max(1, n - FUZZY_SLACK), n + FUZZY_SLACK + 1):
        for i in range(len(pt) - size + 1):
            w = pt[i:i + size]
            if (size == n or (w[0] == st[0] and w[-1] == st[-1])) and Counter(t for t in w if t[0].isdigit()) == nums \
                    and Counter(t for t in w if t in OPS) == ops and SequenceMatcher(None, st, w, autojunk=False).ratio() >= FUZZY_T:
                return 'fuzzy'
    return ''


def root_of(row):
    """Rule 2: the cleaned span, the root line, its 1-based index, the number of teacher lines, is-last, span-is-literal."""
    root = next(m for m in json.loads(row.misconceptions) if m.get('root'))
    lines = page_lines(row.transcription_steps); span = clean_span(root.get('span', '')); sp = glue(span)
    m = re.match(r'\s*L(\d+)(?:\s*[-–]\s*L?(\d+))?', str(root.get('step', '')))
    lo, hi = (int(m.group(1)), int(m.group(2) or m.group(1))) if m else (0, 0)
    named = [i for i, (label, _) in enumerate(lines) if label and lo <= int(label[1:]) <= hi]
    holds = lambda i: bool(sp) and sp in glue(lines[i][1])
    hit = [i for i in named if holds(i)] or [i for i in range(len(lines)) if holds(i)]
    if hit:
        idx, text = hit[0], lines[hit[0]][1]
    elif named:
        idx, text = named[0], ' '.join(lines[i][1] for i in named)
    else:
        idx, text = 0, ' '.join(b for _, b in lines)
    literal = bool(sp) and sp in ''.join(glue(b) for _, b in lines)
    return span, text, idx + 1, len(lines), bool(lines) and idx == len(lines) - 1, literal


def align(root_line, lines):
    """Rule 4: similarity of the root line to the machine page and the machine lines under the best substring."""
    parts = [glue(b) for _, b in lines]
    g = glue(root_line)
    d, a, z = substring_match(g, ''.join(parts))
    sim = 1 - d / max(1, len(g))
    pos, hit = 0, []
    for i, p in enumerate(parts):  # a line counts when the match covers half of it, or it holds half of the match
        over = min(z, pos + len(p)) - max(a, pos)
        if over > 0 and 2 * over >= min(len(p), z - a):
            hit.append(i)
        pos += len(p)
    return sim, hit


# ---------- arithmetic (rule 6) ----------

def number(tok, thousands):
    """A numeric literal as a Python literal, or None. '7.200' and '7,200' are 7200 when `thousands`, else 7.2."""
    groups, seps = re.split(r'[.,]', tok), re.findall(r'[.,]', tok)
    if not seps:
        return tok.lstrip('0') or '0'  # Python refuses integer literals with leading zeros
    if len(seps) == 1:
        if thousands and re.fullmatch(r'[1-9]\d{0,2}', groups[0]) and len(groups[1]) == 3:
            return groups[0] + groups[1]
        return groups[0] + '.' + groups[1]
    if not re.fullmatch(r'[1-9]\d{0,2}', groups[0]) or any(len(g) != 3 for g in groups[1:-1]) or len(set(seps[:-1])) > 1:
        return None
    if seps[-1] == seps[0] and len(groups[-1]) == 3:
        return ''.join(groups)  # 1.250.000
    return ''.join(groups[:-1]) + '.' + groups[-1]  # 10.000,50 and 180,000,00


def ast_value(node):
    if isinstance(node, ast.Expression):
        return ast_value(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = ast_value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = ast_value(node.left), ast_value(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(b) > 12 or abs(a) > 1e6 or (a < 0 and b != int(b)) or (a == 0 and b < 0):
                raise ValueError('power out of range')
            return a ** b
        return a / b if isinstance(node.op, ast.Div) else {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}[type(node.op)]
    if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'sqrt' and len(node.args) == 1:
        return math.sqrt(ast_value(node.args[0]))
    raise ValueError('not arithmetic')


def value(side, thousands):
    """(number, decimals of a lone decimal literal or None) for a cleaned expression, or None when not evaluable."""
    s = re.sub(r'^rp\.?\s*', '', side.strip().rstrip('.').strip())
    s = re.sub(r'^[a-z]{2,}[a-z .]*:\s*(?=[\d(√-])', '', s)  # a word label such as 'kpk :'
    s = re.sub(r'(?<=[\d)])' + UNIT + r'$', '', s).strip()
    if not s or not re.fullmatch(r'[\d.,+\-x:/^()√% ]+', s) or not re.search(r'\d', s):
        return None
    lone = re.fullmatch(r'-?\s*(' + NUMBER + r')', s)
    s = re.sub(r'(?<=[\d)])(?:\s+\.\s*|\s*\.\s+)(?=[\d(√-])|(?<=\))\.(?=[\d(√])|(?<=\d)\.(?=[(√])', ' x ', s)  # a spaced dot multiplies
    s = re.sub(r'(?<![\d/.,])(\d+)\s+(\d+)/(\d+)(?![\d.,/])',  # mixed number: 2 5/8, never 20 10/962
               lambda m: f'({m.group(1)}+{m.group(2)}/{m.group(3)})' if int(m.group(2)) < int(m.group(3)) else m.group(), s)
    if re.search(r'[\d)]\s+[\d(√]', s):  # two values side by side: not one expression
        return None
    lits = []

    def literal(m):
        lits.append(number(m.group(), thousands))
        return lits[-1] or '?'
    s = re.sub(NUMBER, literal, s.replace(' ', ''))
    if None in lits or re.search(r'[.,](?!\d)|(?<!\d)[.,]', s):
        return None
    s = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', s)
    s = re.sub(r'(?<![\d.)/])(\d+)/(\d+)(?![\d.(/])', r'(\1/\2)', s)  # a/b of two integers binds first
    s = re.sub(r'(?<=[\d)])x(?=[\d(\-√])', '*', s)
    if 'x' in s or '%' in s:
        return None
    s = re.sub(r'√(\d+(?:\.\d+)?)', r'sqrt(\1)', s).replace('√(', 'sqrt(')
    s = re.sub(r'(?<=[\d)])(?=\(|sqrt)', '*', s); s = re.sub(r'(?<=\))(?=\d)', '*', s)
    s = s.replace(':', '/').replace('^', '**')
    try:
        v = ast_value(ast.parse(s, mode='eval'))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError, TypeError, KeyError):
        return None
    if isinstance(v, complex) or not math.isfinite(v):
        return None
    return v, (len(lits[0].split('.')[1]) if lone and '.' in lits[0] else None)


def readings(side):
    """Every (value, decimals, thousands flag) a side can mean: both readings of 7.200, and a lone top-level '/' as one fraction bar."""
    depth, cuts = 0, []
    for i, c in enumerate(side):
        depth += (c == '(') - (c == ')')
        if c == '/' and depth == 0:
            cuts.append(i)
    out = []
    for t in (True, False):
        v = value(side, t)
        if v:
            out.append(v + (t,))
        if len(cuts) == 1:  # teachers write a tall fraction on one line: 31x25/40x25, 8 - 2/25
            a, b = value(side[:cuts[0]], t), value(side[cuts[0] + 1:], t)
            if a and b and b[0]:
                out.append((a[0] / b[0], None, t))
    return out


def same(A, B):
    """Two sides agree under some shared reading (a lone decimal literal allows rounding or truncation to its decimals)."""
    return any(ta == tb and abs(x - y) <= max([.999 * 10 ** -d for d in (dx, dy) if d] + [1e-9 * max(1., abs(x), abs(y))])
               for x, dx, ta in A for y, dy, tb in B)


def relations(line):
    """[(left, op, right, holds, left readings, right readings)] for adjacent sides of = < > in a cleaned line that both evaluate."""
    parts, out = re.split(r'(=|<|>)', line), []
    for k in range(1, len(parts) - 1, 2):
        rest = re.split(r',\s|;', parts[k + 1])
        left, right = re.split(r',\s|;', parts[k - 1])[-1].strip(), rest[0].strip()
        if len(rest) > 1 and re.fullmatch(r'\s*' + NUMBER + r'\s*', rest[1]):
            continue  # '36 = 1, 2, 3, 4, 6' is a list, not an equation
        L, R = readings(left), readings(right)
        if L and R:
            ok = same(L, R) if parts[k] == '=' else not same(L, R) and any(
                ta == tb and (x < y if parts[k] == '<' else x > y) for x, _, ta in L for y, _, tb in R)
            out.append((left, parts[k], right, ok, L, R))
    return out


def final_value(line):
    """Text after the last '=' of a cleaned line (the whole line when there is none), without trailing ticks, slashes and
    stops, one pair of enclosing brackets, or label words before a number ('jab. 11 kali' -> '11 kali')."""
    v = re.sub(r'[\s✓✔/\\|.;:,]+$', '', line.split('=')[-1]).strip()
    if re.fullmatch(r'\([^()]*\)', v):
        v = v[1:-1].strip()
    return re.sub(r'^(?:[a-z]+[.:,]?\s+)+(?=[-(√]*\d)', '', v)


def compare(v, key):
    """'eq', 'ne' (both numbers, different), or 'ne_str' (not both numbers and the glued strings differ)."""
    A, B = [x + (t,) for t in (True, False) for x in [value(v, t)] if x], [x + (t,) for t in (True, False) for x in [value(key, t)] if x]
    if A and B:  # both are numbers: decided by value alone ('13/20' is not '1 3/20')
        return 'eq' if same(A, B) else 'ne'
    strip = lambda s: glue(re.sub(r'^rp\.?', '', re.sub(r'(?<=[\d)])' + UNIT + r'$', '', s.strip().rstrip('.'))))
    return 'eq' if v and key and strip(v) == strip(key) else 'ne_str'


def is_number(s):
    return bool(value(s, True) or value(s, False))


def answer_key(row):
    """The key's final value, cleaned; '' when the row has no key (one dev row reads 'nan')."""
    key = final_value(clean(row.correct_answer))
    return '' if key in ('', 'nan') else key


def last_value(lines):
    return final_value(lines[-1][1]) if lines else ''


# ---------- the audits ----------

def fate_of(span, root_line, is_last, key, page_fixed, lines, sim, hit):
    """Rule 5 for a lost span: (fate, evidence, same_wrong_equation)."""
    if page_fixed:
        return 'SILENTLY FIXED', f'the machine page ends in the answer key {key!r}; the teacher\'s page does not', False
    if sim < ALIGN_T or not hit:
        return 'DROPPED', f'line similarity {sim:.2f} < {ALIGN_T}', False
    aligned = [lines[i][1] for i in hit]
    wrong_t = [e for e in relations(span) + relations(root_line) if not e[3]]  # the span's own relations first
    machine = [e for line in aligned for e in relations(line)]
    show = lambda e: f'{e[0]} {e[1]} {e[2]}'
    for t in wrong_t:  # the machine's version of the same statement: a relation that shares one side's value with it
        twins = [m for m in machine if same(m[4], t[4]) or same(m[5], t[5])]
        still = [m for m in twins if not m[3]]
        if still:
            return 'MISREAD', f'teacher [{show(t)}] is false; machine [{show(still[0])}] is still false', same(still[0][4], t[4]) and same(still[0][5], t[5])
        if twins:
            return 'SILENTLY FIXED', f'teacher [{show(t)}] is false; machine [{show(twins[0])}] holds', False
    t_final, m_final = final_value(root_line), final_value(aligned[-1])
    if key and compare(m_final, key) == 'eq' and compare(t_final, key) != 'eq':
        return 'SILENTLY FIXED', f'machine line ends in {m_final!r}, the answer key; teacher wrote {t_final!r}', False
    wrong_m = [e for e in machine if not e[3]]
    if wrong_m:
        return 'MISREAD', f'machine [{show(wrong_m[0])}] is false (no matching teacher relation)', False
    if is_last and key and is_number(key) and is_number(m_final) and compare(m_final, key) == 'ne' and compare(m_final, t_final) == 'ne':
        return 'MISREAD', f'last line: machine {m_final!r}, teacher {t_final!r}, key {key!r}', False
    return 'ALTERED-UNKNOWN', 'nothing to evaluate and no answer-key match', False


def audit_row(row, text):
    """One error row against one machine transcript: kept, how, alignment, fate and line position."""
    span, root_line, idx, n, is_last, literal = root_of(row)
    lines = page_lines(text); key = answer_key(row)
    how = kept(span, lines); sim, hit = align(root_line, lines)
    sp = glue(span); page = ''.join(glue(b) for _, b in lines)
    aligned = [lines[i][1] for i in hit] if sim >= ALIGN_T else []
    overruled = bool(how in ('tokens', 'fuzzy') and aligned and key and compare(final_value(aligned[-1]), key) == 'eq'
                     and compare(final_value(root_line), key) != 'eq')
    if overruled:
        how = ''
    page_fixed = bool(key and not row.final_answer_correct and compare(last_value(page_lines(row.transcription_steps)), key) != 'eq'
                      and compare(last_value(lines), key) == 'eq')
    rec = {'trace_id': row.trace_id, 'checkable': bool(sp), 'kept': bool(how), 'kept_by': how, 'line_sim': round(sim, 3),
           'span_sim': round(1 - substring_match(sp, page)[0] / max(1, len(sp)), 3),
           'kept_in_line': bool(how) and bool(sp) and sp in ''.join(glue(b) for b in aligned),
           'short_span': len(sp) < SHORT_SPAN, 'span_literal': literal, 'root_index': idx, 'teacher_lines': n, 'machine_lines': len(lines),
           'position': 'single line' if n <= 1 else ('first half' if idx <= n / 2 else 'second half'),
           'fate': 'KEPT', 'evidence': '', 'same_wrong_equation': False, 'number_missing': None, 'bracket_match_overruled': overruled}
    if not sp:
        rec['fate'], rec['evidence'] = 'NOT CHECKABLE', 'the span is empty after normalisation (an annotator note, not transcript text)'
    elif not how:
        rec['fate'], rec['evidence'], rec['same_wrong_equation'] = fate_of(span, root_line, is_last, key, page_fixed, lines, sim, hit)
        if rec['fate'] == 'ALTERED-UNKNOWN':
            have = Counter(t for t in tokens(' '.join(aligned)) if t[0].isdigit())
            rec['number_missing'] = any(have[t] < c for t, c in Counter(t for t in tokens(span) if t[0].isdigit()).items())
    show = {'teacher_root_line': root_line, 'span': span, 'machine_line': ' ⏎ '.join(lines[i][1] for i in hit) if hit else ''}
    return rec, show


def plain_rates(rows, texts):
    """Share of rows whose span is a plain substring of the machine page, under four normalisations (to match old records)."""
    def light(s):  # lower case, dashes unified, [notes] and whitespace dropped: reproduces the recorded 0.555
        s = re.sub(r'\[[^\]]*\]', '', str(s).lower())
        return re.sub(r'\s+', '', re.sub(r'[−–—]', '-', s))

    def reading_impact(s):  # runs/arch/reading-impact/step1_cer.py::norm: reproduces the recorded 0.605
        s = unicodedata.normalize('NFKC', str(s)).lower()
        for a, b in [('−', '-'), ('–', '-'), ('—', '-'), ('×', 'x'), ('·', '.'), ('÷', ':'), ('\\', '')]:
            s = s.replace(a, b)
        return re.sub(r'\s+', '', re.sub(r'\[[^\]]*\]', '', s))
    body = lambda t: ' '.join(re.sub(r'^\s*L\d+\s*\|\s?', '', x) for x in str(t).splitlines())
    hits = {k: 0 for k in ('raw', 'light', 'reading_impact', 'this_script')}
    checkable = empty_light = empty_impact = 0
    for row in rows:
        span = next(m for m in json.loads(row.misconceptions) if m.get('root')).get('span', ''); t = texts[row.trace_id]
        hits['raw'] += span in t
        hits['light'] += light(span) in light(body(t)); empty_light += not light(span)
        hits['reading_impact'] += reading_impact(span) in reading_impact(body(t)); empty_impact += not reading_impact(span)
        sp = glue(clean_span(span)); checkable += bool(sp)
        hits['this_script'] += bool(sp) and sp in ''.join(glue(b) for _, b in page_lines(t))
    out = {k: share(v, checkable if k == 'this_script' else len(rows)) for k, v in hits.items()}
    out['light']['empty_spans_counted_present'] = int(empty_light); out['reading_impact']['empty_spans_counted_present'] = int(empty_impact)
    return out


def share(k, n):
    """A count, its share and the binomial standard error (None for an empty group, so the report stays valid JSON)."""
    p = k / n if n else None
    return {'count': int(k), 'rows': int(n), 'share': p, 'se': math.sqrt(p * (1 - p) / n) if n else None}


def summary(recs):
    """Counts and shares (with binomial standard errors) for one system's audited rows; every share is over the
    checkable rows (rule 3: a span that is empty after normalisation is left out)."""
    total = len(recs); recs = [r for r in recs if r['checkable']]
    n = len(recs); lost = [r for r in recs if not r['kept']]
    part = lambda rows: share(sum(r['kept'] for r in rows), len(rows))
    out = {'error_rows': total, 'checkable_rows': n, 'not_checkable_empty_span': total - n,
           'kept': part(recs), 'kept_by': dict(Counter(r['kept_by'] for r in recs if r['kept'])),
           'kept_exact_only': share(sum(r['kept_by'] == 'exact' for r in recs), n),
           'kept_in_aligned_line': share(sum(r['kept_in_line'] for r in recs), n),
           'kept_without_short_spans': part([r for r in recs if not r['short_span']]),
           'kept_literal_spans_only': part([r for r in recs if r['span_literal']]),
           'bracket_matches_overruled': int(sum(r['bracket_match_overruled'] for r in recs)),
           'lost': len(lost), 'fates_of_lost': {f: share(sum(r['fate'] == f for r in lost), len(lost)) for f in FATES},
           'fates_of_all_error_rows': {f: share(sum(r['fate'] == f for r in lost), n) for f in FATES},
           'misread_same_wrong_equation': int(sum(r['same_wrong_equation'] for r in lost)),
           'altered_unknown_number_missing': int(sum(r['number_missing'] is True for r in lost)),
           'altered_unknown_numbers_intact': int(sum(r['number_missing'] is False for r in lost)), 'by_position': {}}
    for pos in ('first half', 'second half', 'single line'):
        sub = [r for r in recs if r['position'] == pos]; sl = [r for r in sub if not r['kept']]
        out['by_position'][pos] = {'error_rows': len(sub), 'lost': share(len(sl), len(sub)),
                                   'fates_of_lost': {f: share(sum(r['fate'] == f for r in sl), len(sl)) for f in FATES}}
    late = [r for r in recs if r['position'] != 'first half']
    out['by_position']['second half incl. single line'] = {'error_rows': len(late), 'lost': share(sum(not r['kept'] for r in late), len(late))}
    return out


def flips(rows, texts):
    """Rule 7: flip-to-right and flip-to-wrong, strict (all numeric) and wide tiers, with the parser's agreement with the label."""
    tiers = {t: {'wrong_rows': 0, 'to_right': 0, 'right_rows': 0, 'to_wrong': 0} for t in ('strict', 'wide')}
    check = {'label_right': 0, 'label_right_parsed_eq': 0, 'label_wrong': 0, 'label_wrong_parsed_eq': 0}
    total, flipped = 0, {'to_right': [], 'to_wrong': []}
    for row in rows:
        total += 1
        key = answer_key(row); right = bool(row.final_answer_correct)
        t, m = last_value(page_lines(row.transcription_steps)), last_value(page_lines(texts[row.trace_id]))
        ct, cm = compare(t, key), compare(m, key)
        check['label_right' if right else 'label_wrong'] += 1
        check['label_right_parsed_eq' if right else 'label_wrong_parsed_eq'] += ct == 'eq'
        if (ct == 'eq') != right or not m or not key:
            continue
        for tier, ok in (('strict', is_number(key) and is_number(t) and is_number(m)), ('wide', True)):
            if ok:
                tiers[tier]['right_rows' if right else 'wrong_rows'] += 1
                tiers[tier]['to_wrong' if right else 'to_right'] += (cm != 'eq') if right else (cm == 'eq')
        if right != (cm == 'eq'):
            flipped['to_wrong' if right else 'to_right'].append(row.trace_id)
    out = {'rows': total, 'parser_check_on_teacher_transcripts': check, 'flipped_ids_wide': flipped}
    for tier, c in tiers.items():
        out[tier] = {'parseable_rows': c['wrong_rows'] + c['right_rows'],
                     'flip_to_right': share(c['to_right'], c['wrong_rows']), 'flip_to_wrong': share(c['to_wrong'], c['right_rows'])}
    return out


def mcnemar(a, b):
    """Paired counts for two boolean arrays and the exact two-sided p-value on the discordant pairs."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum()); n = only_a + only_b
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.
    return {'rows': int(len(a)), 'both': int((a & b).sum()), 'first_only': only_a, 'second_only': only_b,
            'neither': int((~a & ~b).sum()), 'exact_p': p}


def hand_reading(recs, flips_wide):
    """Rule 10: the reviewer's verdicts set against the rules, per system. Only rows that are lost now are looked up."""
    out = {'reader': 'one reviewer, 14 November 2026, teacher transcript against machine transcript, no photos, not blind to the system',
           'verdicts': {'FORMAT': 'same content in another rendering; the span was in fact kept', 'FIXED': 'the error is corrected',
                        'ALTERED': 'changed and not corrected (the design\'s misread, including partial loss)', 'UNCLEAR': 'cannot tell'},
           'systems': {}}
    for system, rs in recs.items():
        rs = [r for r in rs if r['checkable']]; lost = [r for r in rs if not r['kept']]
        read = [(r, HAND[system][r['trace_id']]) for r in lost if r['trace_id'] in HAND[system]]
        count = Counter(v for _, v in read)
        by_rule = {f: dict(Counter(v for r, v in read if r['fate'] == f)) for f in FATES}
        agree = {f: {'agree': sum(v == RULE_TO_HAND[f] for r, v in read if r['fate'] == f), 'read': sum(r['fate'] == f for r, _ in read)}
                 for f in RULE_TO_HAND}
        n_kept = sum(r['kept'] for r in rs)
        flagged = flips_wide[system]['to_wrong']
        out['systems'][system] = {
            'lost_rows': len(lost), 'lost_rows_read': len(read), 'lost_rows_without_a_verdict': [r['trace_id'] for r in lost if r['trace_id'] not in HAND[system]],
            'verdicts': {v: int(count.get(v, 0)) for v in VERDICTS}, 'verdicts_by_rule_fate': by_rule, 'rule_agrees_with_reading': agree,
            'kept_in_substance': share(n_kept + count.get('FORMAT', 0), len(rs)), 'fixed_of_error_rows': share(count.get('FIXED', 0), len(rs)),
            'altered_of_error_rows': share(count.get('ALTERED', 0), len(rs)),
            'flips_to_wrong_wide': {'flagged': len(flagged), 'real_on_reading': sum(t in HAND_FLIPS[system] for t in flagged),
                                    'flagged_rows_without_a_verdict_count_as_not_real': True}}
    return out


def four_slices(dev, fit, classes, recs):
    """Section 6 report: MAP@3 overall / no-error / seen-pair / hard on 301 and 270 rows, and by span kept or lost."""
    try:
        import fuse_candidate as F
        from k12.core import aligned_bundle
        from score_vlm_dev import shares
        ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
        SUB = F.fuse(F.geo(F.DEV, ids, classes), aligned_bundle(ROOT / RANKER_DEV, ids, classes), .5, .75, F.marginals(fit.root_code, classes), 30)
        S, used = shares(ROOT / Q38_DEV, ids, classes, first_only=True)
    except (OSError, ValueError, ImportError) as e:
        print(f'\nFour-slice MAP@3 report skipped (dev bundles not available: {e})')
        return None
    newc, unq = F.unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard; have = used > 0
    pts = lambda P: row_scores(y, topk(P, classes))
    sub = pts(SUB); alone = np.where(have, pts(normalize(S + 1e-6)), sub)
    part = lambda s, m: {'overall': [float(s[m].mean()), int(m.sum())], 'no_error': [float(s[m & none].mean()), int((m & none).sum())],
                         'seen_pair': [float(s[m & seen].mean()), int((m & seen).sum())], 'hard': [float(s[m & hard].mean()), int((m & hard).sum())]}
    rep = {'submitted_system': {'all_301': part(sub, np.ones(len(y), bool)), 'answered_270': part(sub, have)},
           'reasoner_alone_unanswered_keep_submitted': {'all_301': part(alone, np.ones(len(y), bool)), 'answered_270': part(alone, have)}}
    print('\nFour-slice MAP@3 (section 6); rows in brackets')
    print(f"{'system':<44}{'rows':<14}{'overall':<14}{'no-error':<14}{'seen-pair':<14}{'hard':<14}")
    for name, both in rep.items():
        for rows, p in both.items():
            print(f'{name:<44}{rows:<14}' + ''.join(f'{v:.4f} ({k})'.ljust(14) for v, k in p.values()))
    row = {t: i for i, t in enumerate(ids)}
    rep['by_span'] = {}
    print('\nMAP@3 on error rows, by whether the reader kept the root span (and the kept share by slice)')
    for system, scores in (('transcriber', sub), ('reasoner', alone)):
        rs = [r for r in recs[system] if r['checkable']]
        ix = np.array([row[r['trace_id']] for r in rs]); k = np.array([r['kept'] for r in rs])
        other = {'transcriber': 'submitted system', 'reasoner': 'reasoner alone'}[system]
        d = {'scored': other, 'kept': [float(scores[ix[k]].mean()), int(k.sum())], 'lost': [float(scores[ix[~k]].mean()), int((~k).sum())],
             'kept_share_seen_pair': share(int(k[seen[ix]].sum()), int(seen[ix].sum())), 'kept_share_hard': share(int(k[hard[ix]].sum()), int(hard[ix].sum()))}
        if system == 'reasoner':
            d['submitted_system_same_rows'] = {'kept': float(sub[ix[k]].mean()), 'lost': float(sub[ix[~k]].mean())}
        rep['by_span'][system] = d
        print(f"  {system:<12} {other:<17} span kept {d['kept'][0]:.4f} ({d['kept'][1]})   span lost {d['lost'][0]:.4f} ({d['lost'][1]})   "
              f"kept share: seen-pair {fmt(d['kept_share_seen_pair'])}, hard {fmt(d['kept_share_hard'])}")
    return rep


def fmt(s):
    return f"{s['count']}/{s['rows']} = {s['share']:.3f} ± {s['se']:.3f}" if s['rows'] else '0/0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='reports/transcript_faithfulness.json')
    ap.add_argument('--examples', type=int, default=12); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    texts = {'transcriber': json.loads((ROOT / TX_DEV).read_text()),
             'reasoner': {r['trace_id']: r['answer']['transcription'] for r in  # a row without a transcription is not covered
                          map(json.loads, filter(str.strip, (ROOT / Q38_DEV).read_text().splitlines()))
                          if (r.get('answer') or {}).get('transcription')}}
    recs, shows, rep = {}, {}, {'thresholds': {'ALIGN_T': ALIGN_T, 'FUZZY_T': FUZZY_T, 'FUZZY_SLACK': FUZZY_SLACK, 'SHORT_SPAN': SHORT_SPAN},
                                'dev_rows': len(dev), 'dev_error_rows': int((dev.root_code != 'NONE').sum()), 'systems': {}}
    for system, tx in texts.items():
        rows = [r for r in dev.itertuples(index=False) if r.trace_id in tx]
        errors = [r for r in rows if r.root_code != 'NONE']
        pairs = [audit_row(r, tx[r.trace_id]) for r in errors]
        recs[system] = [p[0] for p in pairs]; shows[system] = {p[0]['trace_id']: p[1] for p in pairs}
        rep['systems'][system] = {'rows_covered': len(rows), **summary(recs[system]), 'plain_span_present': plain_rates(errors, tx),
                                  'final_answer_flips': flips(rows, tx), 'rows': recs[system]}
    by = {s: {r['trace_id']: r for r in recs[s]} for s in recs}
    common = sorted(t for t in set(by['transcriber']) & set(by['reasoner']) if by['transcriber'][t]['checkable'])
    common_rows = [r for r in dev.itertuples(index=False) if r.trace_id in set(common)]
    answered = [r for r in dev.itertuples(index=False) if r.trace_id in texts['reasoner']]
    light = {s: plain_rates(common_rows, tx)['light'] for s, tx in texts.items()}  # the recorded 0.555 definition, same rows
    pair = lambda k: mcnemar([by['transcriber'][t][k] for t in common], [by['reasoner'][t][k] for t in common])
    rep['paired_on_common_error_rows'] = {
        'order': 'first = transcriber, second = reasoner', 'kept': pair('kept'), 'kept_in_aligned_line': pair('kept_in_line'),
        'kept_exact_only': mcnemar([by['transcriber'][t]['kept_by'] == 'exact' for t in common], [by['reasoner'][t]['kept_by'] == 'exact' for t in common]),
        'transcriber_kept': share(sum(by['transcriber'][t]['kept'] for t in common), len(common)),
        'reasoner_kept': share(sum(by['reasoner'][t]['kept'] for t in common), len(common)),
        'transcriber_fates': dict(Counter(by['transcriber'][t]['fate'] for t in common)),
        'reasoner_fates': dict(Counter(by['reasoner'][t]['fate'] for t in common)),
        'light_definition_of_the_recorded_0.555': light,
        'final_answer_flips_on_the_270_rows': {s: {k: v for k, v in flips(answered, tx).items() if k in ('strict', 'wide')} for s, tx in texts.items()}}
    hand = rep['hand_reading'] = hand_reading(recs, {s: rep['systems'][s]['final_answer_flips']['flipped_ids_wide'] for s in texts})
    in_substance = {s: [by[s][t]['kept'] or HAND[s].get(t) == 'FORMAT' for t in common] for s in texts}
    hand['paired_kept_in_substance_on_common_rows'] = {'order': 'first = transcriber, second = reasoner',
                                                       **mcnemar(in_substance['transcriber'], in_substance['reasoner'])}
    q = light['reasoner']
    rep['decision_rule_4_1_1'] = {'recorded_transcriber_rate': RECORDED, 'reasoner_rate_same_definition': q['share'], 'se': q['se'],
                                  'rows': 'the checkable error rows both systems cover; the recorded 0.555 is on all 200 error rows and '
                                          'counts 3 empty spans as present (108/197 = 0.548 without them)',
                                  'transcriber_rate_same_rows': light['transcriber']['share'],
                                  'reasoner_keeps_fewer': bool(q['share'] < RECORDED),
                                  'reading': 'A5 (cropped-image verification) moves ahead of the card arms' if q['share'] < RECORDED
                                  else 'the reasoner keeps the error text at least as often as the recorded transcriber rate, so the rule does not fire'}

    # ---------- print ----------
    print(f"Transcript faithfulness, dev fold: {rep['dev_rows']} rows, {rep['dev_error_rows']} error rows. "
          f"Thresholds: line aligned >= {ALIGN_T}, fuzzy tokens >= {FUZZY_T}.\n")
    names = list(texts)
    line = lambda label, f: print(f'{label:<46}' + ''.join(f'{f(rep["systems"][s]):<34}' for s in names))
    print(f"{'':<46}" + ''.join(f'{s:<34}' for s in names))
    line('rows covered / error rows / checkable', lambda s: f"{s['rows_covered']} / {s['error_rows']} / {s['checkable_rows']}")
    for k, label in (('raw', 'span present, raw substring'), ('light', 'span present, light norm (recorded 0.555)'),
                     ('reading_impact', 'span present, reading-impact norm (0.605)'), ('this_script', 'span present, this script, exact')):
        line(label, lambda s, k=k: fmt(s['plain_span_present'][k]))
    line('  empty spans the light / impact norms count', lambda s: f"{s['plain_span_present']['light']['empty_spans_counted_present']} / "
                                                                   f"{s['plain_span_present']['reading_impact']['empty_spans_counted_present']}")
    print('From here on, rows whose span is empty after normalisation are left out (not checkable).')
    line('KEPT (exact, tokens or fuzzy)', lambda s: fmt(s['kept']))
    line('  found by exact / tokens / fuzzy', lambda s: ' / '.join(str(s['kept_by'].get(k, 0)) for k in ('exact', 'tokens', 'fuzzy')))
    line('  bracket-only matches overruled as fixes', lambda s: str(s['bracket_matches_overruled']))
    line('KEPT and inside the aligned line', lambda s: fmt(s['kept_in_aligned_line']))
    line(f'KEPT, spans of >= {SHORT_SPAN} characters only', lambda s: fmt(s['kept_without_short_spans']))
    line('KEPT, spans that are literal teacher text', lambda s: fmt(s['kept_literal_spans_only']))
    line('LOST', lambda s: str(s['lost']))
    print('Fates by rule 5. These are what the rules could decide, not a distribution: SILENTLY FIXED is a lower bound,\n'
          'DROPPED means only that no line aligned, and ALTERED-UNKNOWN holds more fixes and format-only rows (hand reading below).')
    for f in FATES:
        line(f"  {f + (' = NOT ALIGNED' if f == 'DROPPED' else '')} (share of lost)", lambda s, f=f: fmt(s['fates_of_lost'][f]))
    line('  MISREAD with the same wrong equation', lambda s: str(s['misread_same_wrong_equation']))
    line('  ALTERED-UNKNOWN: a number missing / intact', lambda s: f"{s['altered_unknown_number_missing']} / {s['altered_unknown_numbers_intact']}")
    print('\nBy position of the root line among the teacher\'s lines (lost share; then FIXED / DROPPED / MISREAD / UNKNOWN counts)')
    for pos in ('first half', 'second half', 'single line'):
        line(f'  {pos}', lambda s, pos=pos: fmt(s['by_position'][pos]['lost']) + '  ' + '/'.join(str(s['by_position'][pos]['fates_of_lost'][f]['count']) for f in FATES))
    line('  second half incl. single line', lambda s: fmt(s['by_position']['second half incl. single line']['lost']))
    print('\nFinal-answer flips (rule 7): strict = key, teacher and machine values all numeric; wide = string equality allowed')
    for tier in ('strict', 'wide'):
        line(f'  {tier}: parseable rows', lambda s, t=tier: f"{s['final_answer_flips'][t]['parseable_rows']} of {s['final_answer_flips']['rows']}")
        line(f'  {tier}: flip-to-right (t != a*, m = a*)', lambda s, t=tier: fmt(s['final_answer_flips'][t]['flip_to_right']))
        line(f'  {tier}: flip-to-wrong (t = a*, m != a*)', lambda s, t=tier: fmt(s['final_answer_flips'][t]['flip_to_wrong']))
    c = rep['systems']['transcriber']['final_answer_flips']['parser_check_on_teacher_transcripts']
    print(f"  parser check on teacher transcripts: parsed value equals the key on {c['label_right_parsed_eq']} of {c['label_right']} rows labelled right "
          f"and on {c['label_wrong_parsed_eq']} of {c['label_wrong']} rows labelled wrong (rows where parser and label disagree are not parseable)")
    print('  flip-to-wrong is an upper bound: it also counts a machine page that ends on another line or note. Hand reading of the wide-tier\n'
          '  rows flagged to-wrong: ' + '; '.join(f"{s} {h['flips_to_wrong_wide']['real_on_reading']} of {h['flips_to_wrong_wide']['flagged']} real"
                                                  for s, h in hand['systems'].items()) + ' (a misread final value or a missing final line).')
    f270 = rep['paired_on_common_error_rows']['final_answer_flips_on_the_270_rows']
    print('  same 270 rows, wide tier: ' + '; '.join(f"{s} to-right {fmt(f270[s]['wide']['flip_to_right'])}, to-wrong {fmt(f270[s]['wide']['flip_to_wrong'])}" for s in names))
    p = rep['paired_on_common_error_rows']
    print(f"\nPaired, the {len(common)} checkable error rows both cover")
    print(f"  transcriber KEPT {fmt(p['transcriber_kept'])};  reasoner KEPT {fmt(p['reasoner_kept'])}")
    for k in ('kept', 'kept_exact_only', 'kept_in_aligned_line'):
        m = p[k]
        print(f"  {k}: both {m['both']}, transcriber only {m['first_only']}, reasoner only {m['second_only']}, neither {m['neither']}; exact McNemar p = {m['exact_p']:.3f}")
    print('  fates on these rows: ' + '; '.join(f"{s} " + ', '.join(f"{f} {p[s + '_fates'].get(f, 0)}" for f in FATES) for s in names))
    print(f"  light definition (recorded 0.555): transcriber {fmt(light['transcriber'])};  reasoner {fmt(light['reasoner'])}")
    print(f'\nHand reading of every lost span (rule 10: {hand["reader"]})')
    hline = lambda label, f: print(f'{label:<46}' + ''.join(f'{f(hand["systems"][s]):<34}' for s in names))
    hline('  lost rows read / lost rows', lambda h: f"{h['lost_rows_read']} / {h['lost_rows']}")
    for v, label in (('FORMAT', 'FORMAT: same content, other rendering'), ('FIXED', 'FIXED: the error is corrected'),
                     ('ALTERED', 'ALTERED: changed, not corrected'), ('UNCLEAR', 'UNCLEAR')):
        hline(f'  {label}', lambda h, v=v: str(h['verdicts'][v]))
    hline('  KEPT in substance = (KEPT + FORMAT) / rows', lambda h: fmt(h['kept_in_substance']))
    hline('  FIXED / checkable error rows', lambda h: fmt(h['fixed_of_error_rows']))
    hline('  ALTERED / checkable error rows', lambda h: fmt(h['altered_of_error_rows']))
    for f in RULE_TO_HAND:
        hline(f'  rule {f} agrees with the reading', lambda h, f=f: f"{h['rule_agrees_with_reading'][f]['agree']} of {h['rule_agrees_with_reading'][f]['read']}")
    hline('  UNKNOWN rows: FORMAT/FIXED/ALTERED/UNCLEAR', lambda h: ' / '.join(str(h['verdicts_by_rule_fate']['ALTERED-UNKNOWN'].get(v, 0)) for v in VERDICTS))
    m = hand['paired_kept_in_substance_on_common_rows']
    print(f"  paired, kept in substance: both {m['both']}, transcriber only {m['first_only']}, reasoner only {m['second_only']}, neither {m['neither']}; "
          f"exact McNemar p = {m['exact_p']:.3f}")
    missing = {s: h['lost_rows_without_a_verdict'] for s, h in hand['systems'].items() if h['lost_rows_without_a_verdict']}
    if missing:
        print(f'  lost rows without a verdict (the rules changed since the reading; read them before quoting): {missing}')
    d = rep['decision_rule_4_1_1']
    print(f"\nDecision rule 4.1.1 (light definition, the {len(common)} common checkable rows): reasoner {d['reasoner_rate_same_definition']:.3f} ± {d['se']:.3f}, "
          f"transcriber on the same rows {d['transcriber_rate_same_rows']:.3f}, recorded {RECORDED} on all 200 rows: {d['reading']}.")

    rep['four_slice_map3'] = four_slices(dev, fit, classes, recs)

    rng = np.random.default_rng(a.seed)
    pool = [(s, r) for s in names for r in recs[s] if not r['kept'] and r['checkable']]
    soft = [(s, r) for s in names for r in recs[s] if r['kept_by'] in ('tokens', 'fuzzy')]
    n_soft = min(len(soft), a.examples // 4)
    pick = [pool[i] for i in rng.choice(len(pool), min(len(pool), a.examples - n_soft), replace=False)] + \
           [soft[i] for i in rng.choice(len(soft), n_soft, replace=False)]
    print(f'\n{len(pick)} random rows for a sanity check (seed {a.seed}): lost spans, then spans kept only by the tokens or fuzzy rule')
    for s, r in pick:
        e = shows[s][r['trace_id']]
        print(f"- {r['trace_id']} [{s}] fate {r['fate']}" + (f" by {r['kept_by']}" if r['kept'] else f" (hand reading: {HAND[s].get(r['trace_id'], 'not read')})") +
              f" | line sim {r['line_sim']:.2f}, span sim {r['span_sim']:.2f}, root line {r['root_index']} of {r['teacher_lines']}"
              + (f"\n    why:     {r['evidence']}" if r['evidence'] else '') +
              f"\n    teacher: {e['teacher_root_line'][:160]}\n    span:    {e['span'][:160]}\n    machine: {e['machine_line'][:200] or '(no aligned line)'}")
    out = ROOT / a.report; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + '\n')
    print(f'\nwrote {a.report}')


if __name__ == '__main__':
    main()
