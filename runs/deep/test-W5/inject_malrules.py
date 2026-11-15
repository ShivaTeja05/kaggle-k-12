"""Approach C (W5): inject one executable buggy rule into a correct gold transcript (fit-fold NONE donors only),
propagate the wrong value forward through later numeric '=' chains, and emit synthetic training rows.

Never reads dev (fold 0) transcripts as donors.  Output: runs/deep/test-W5/synthetic.jsonl
Each row: trace_id (syn_<donor>_<k>), donor, question_id, topic_name, question_text, transcription_steps, root_code, root_line,
          rule, line_before, line_after, n_propagated.
"""
import json, re, random
from fractions import Fraction
from collections import Counter
import numpy as np
from common import load

OUT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W5'
M = r'[−\-]'
MUL = r'[×x·*]'
RES = r'([−\-]?\s?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)'
SEG = r'([\d\s+\-−*×x·:/().,^]*\d[\d+\-−*×x·:/().,^]*)'
SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')


def num(tok):
    """Parse a result token (Indonesian decimals/thousands, fractions, unicode minus) to Fraction, else None."""
    t = tok.replace('−', '-').replace(' ', '')
    try:
        if '/' in t:
            a, b = t.split('/')
            return Fraction(num(a)) / Fraction(num(b))
        if re.fullmatch(r'-?\d{1,3}(\.\d{3})+', t):
            return Fraction(t.replace('.', ''))
        return Fraction(t.replace(',', '.'))
    except Exception:
        return None


def fmt(v, like='', frac=False):
    v = Fraction(v)
    neg = '−' if '−' in like else '-'
    s = neg if v < 0 else ''
    v = abs(v)
    if v.denominator == 1:
        return s + str(v.numerator)
    if frac or '/' in like:
        return s + f'{v.numerator}/{v.denominator}'
    d = f'{float(v):.2f}'.rstrip('0').rstrip('.')
    return s + (d.replace('.', ',') if ',' in like else d)


def I(x):
    return int(x)


# Each rule: (code, regex, f(groups)->(correct_value, buggy_text) or None).  The regex's last group is the result token.
def r_exp01(g):
    a, n = I(g[0]), I(g[1] or str(g[2]).translate(SUP))
    if a < 2 or n < 2 or a ** n > 10 ** 7:
        return None
    return Fraction(a ** n), fmt(a * n, g[-1])


def r_exp02(g):
    b, m, n, r = g[0], I(g[1]), I(g[2]), I(g[3])
    if r != m + n or m * n == m + n:
        return None
    return 'EXPONENT', str(m * n)


def r_fra01(g):
    a, b, op, c, d = I(g[0]), I(g[1]), g[2], I(g[3]), I(g[4])
    if b == d or min(b, d) == 0:
        return None
    s = 1 if op == '+' else -1
    if s < 0 and a - c < 0:
        return None
    return Fraction(a, b) + s * Fraction(c, d), f'{a + s * c}/{b + d}'


def r_fra03(g):
    a, b, c, d = map(I, g[:4])
    if 0 in (b, c, d):
        return None
    return Fraction(a, b) / Fraction(c, d), f'{a * c}/{b * d}'


def r_int02(g):
    s1, a, s2, b = g[0], I(g[1]), g[2], I(g[3])
    neg = (bool(s1) != bool(s2))
    corr = Fraction(a * b) * (-1 if neg else 1)
    return corr, fmt(-corr, g[-1] + ('−' if '−' in (s1 or '') + (s2 or '') else ''))


def r_int03(g):
    a, b = I(g[0]), I(g[1])
    return Fraction(a + b), fmt(a - b, g[-1])


def r_int04a(g):   # -a - b = -(a+b)  ->  -(a-b)  ("−40 − 20 = −20")
    a, b = I(g[1]), I(g[3])
    if a == b:
        return None
    return Fraction(-(a + b)), fmt(-(a - b), '−' if '−' in g[0] + g[2] else '-')


def r_int04b(g):   # a - b with a<b: minus sign silently dropped
    a, b = I(g[0]), I(g[1])
    if a >= b:
        return None
    return Fraction(a - b), fmt(b - a, g[-1])


def r_ooo01(g):
    a, op1, b, op2, c = I(g[0]), g[1], I(g[2]), g[3], I(g[4])
    s = 1 if op1 == '+' else -1
    if op2 in ':/':
        if c == 0:
            return None
        corr = a + s * Fraction(b, c); bug = Fraction(a + s * b, c)
    else:
        corr = a + s * Fraction(b * c); bug = Fraction((a + s * b) * c)
    if corr == bug:
        return None
    return corr, fmt(bug, g[-1])


def r_rat02(g):
    p, n = num(g[0]), num(g[1])
    if p is None or n is None:
        return None
    return p * n / 100, fmt(p * n, g[-1])


def r_exp04(g):   # a^(m/n) -> a*m/n  (fractional exponent read as multiplication)
    a, m, n = I(g[0]), I(g[1]), I(g[2])
    if n in (0, 1):
        return None
    v = a ** (m / n)
    if abs(v - round(v)) > 1e-9:
        return None
    return Fraction(round(v)), fmt(Fraction(a * m, n), g[-1])


def r_exp05(g):
    return Fraction(1), '0'


RULES = [
    ('EXP-01', r'(?<![\d^)(])(\d+)\s*(?:\^\s*\(?(\d)\)?|([²³⁴⁵]))\s*=\s*' + SEG, r_exp01),
    ('EXP-02', r'(\d+|[a-z])\s*\^\s*\(?(\d+)\)?\s*' + MUL + r'?\s*\1\s*\^\s*\(?(\d+)\)?\s*=\s*\1\s*\^\s*\(?(\d+)\)?()', r_exp02),
    ('FRA-01', r'(?<![\d)(])(\d+)\s*/\s*(\d+)\s*([+\-−])\s*(\d+)\s*/\s*(\d+)\s*=\s*' + SEG, r_fra01),
    ('FRA-03', r'(?<![\d)(])(\d+)\s*/\s*(\d+)\s*[:÷]\s*(\d+)\s*/\s*(\d+)\s*=\s*' + SEG, r_fra03),
    ('INT-02', r'\(?\s*(' + M + r')?\s*(\d+)\s*\)?\s*' + MUL + r'\s*\(\s*(' + M + r')\s*(\d+)\s*\)\s*=\s*' + SEG, r_int02),
    ('INT-02', r'(?<![\d)])(' + M + r')\s*(\d+)\s*' + MUL + r'\s*()(\d+)\s*=\s*' + SEG, r_int02),
    ('INT-03', r'(?<![\d)])(\d+)\s*' + M + r'\s*\(\s*' + M + r'\s*(\d+)\s*\)\s*=\s*' + SEG, r_int03),
    ('INT-04', r'(?<![\d)])(' + M + r')\s*(\d+)\s*(' + M + r')\s*(\d+)\s*=\s*' + SEG, r_int04a),
    ('INT-04', r'(?<![\d)/^.,−\-])(\d+)\s*' + M + r'\s*(\d+)\s*=\s*' + SEG, r_int04b),
    ('OOO-01', r'(?<![\d)(/^*×x·:.,])(\d+)\s*([+\-−])\s*(\d+)\s*([×x·*:])\s*(\d+)\s*=\s*' + SEG, r_ooo01),
    ('RAT-02', r'(\d+(?:[.,]\d+)?)\s*%\s*(?:' + MUL + r'|dari)\s*(\d+(?:[.,]\d+)?)\s*=\s*' + SEG, r_rat02),
    ('EXP-04', r'(\d+)\s*\^\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*=\s*' + SEG, r_exp04),
    ('EXP-05', r'(\d+|[a-z])\s*\^\s*\(?0\)?\s*=\s*(1)(?![\d,./])', r_exp05),
]


def safe_eval(expr):
    s = expr.replace('−', '-').replace('×', '*').replace('·', '*').replace(':', '/').replace('÷', '/')
    s = re.sub(r'(?<=\d)\s*x\s*(?=\d)', '*', s)
    s = re.sub(r'(\d+),(\d+)', r'\1.\2', s)
    s = re.sub(r'(\d{1,3})\.(\d{3})(?!\d)', r'\1\2', s)
    s = s.replace('^', '**')
    if not re.fullmatch(r'[\d\s+\-*/().]+', s) or not re.search(r'\d\s*[+\-*/]', s) or '***' in s:
        return None
    s = re.sub(r'(\d+(?:\.\d+)?)', r'F("\1")', s)
    try:
        v = eval(s, {'F': Fraction, '__builtins__': {}})
        return Fraction(v) if abs(v) < 10 ** 9 else None
    except Exception:
        return None


EQRHS = re.compile(r'=\s*(' + RES[1:-1] + r')(?![\d/])')
CHAIN = re.compile(r'([\d\s+\-−*×x·:/().,^]*\d[\d\s+\-−*×x·:/().,^]*)=\s*(' + RES[1:-1] + r')(?![\d/])')


def propagate(lines, k, start, mapping):
    """Recompute numeric '=' chains from line k (char offset start) onwards, after substituting changed values.
    Same-line text after the injection and all later lines get the old->new value substitution."""
    n_changed = 0
    for j in range(k, len(lines)):
        pre, body = lines[j].split('|', 1) if '|' in lines[j] else ('', lines[j])
        off = max(0, start - len(pre) - 1) if j == k else 0
        head, tail = body[:off], body[off:]
        for old, new in list(mapping.items()):
            if len(old.lstrip('-−')) >= 2:
                tail = re.sub(r'(?<![\d.,/])' + re.escape(old) + r'(?![\d,/]|\.\d)', new, tail)
        body = head + tail
        pos = 0
        while True:
            m = EQRHS.search(body, pos)
            if not m:
                break
            if m.start(1) < off:
                pos = m.end(1); continue
            lhs = re.split(r'[=|;]|\s{3,}', body[:m.start()])[-1]
            v = safe_eval(lhs.strip())
            old = m.group(1).strip()
            ov = num(old)
            if v is not None and ov is not None and v != ov and abs(float(v - ov)) > 0.005 * max(1, abs(float(ov))):
                new = fmt(v, old, frac=('/' in old) or (v.denominator != 1 and '/' in lhs))
                s0 = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
                if new.replace(' ', '') == lhs.strip().replace(' ', ''):      # re-expression step: drop '= old'
                    body = body[:m.start()].rstrip() + body[s0 + len(old):]
                    pos = m.start(); n_changed += 1
                    continue
                body = body[:s0] + new + body[s0 + len(old):]
                mapping[old] = new; n_changed += 1
                pos = s0
            else:
                pos = m.start(1)
        lines[j] = (pre + '|' if '|' in lines[j] else '') + body
    return n_changed


def inject(transcript, rule, rng):
    code, pat, f = rule
    lines = transcript.split('\n')
    cands = []
    for k, ln in enumerate(lines):
        if '|' not in ln or ln.strip().startswith('['):
            continue
        for m in re.finditer(pat, ln):
            g = m.groups()
            try:
                out = f(g)
            except Exception:
                out = None
            if not out:
                continue
            corr, bug = out
            if corr == 'EXPONENT':
                cands.append((k, m.start(len(g) - 1), m.end(len(g) - 1), str(g[3]), bug, m)); continue
            seg = g[-1]; lead = len(seg) - len(seg.lstrip()); core = seg.strip().rstrip('.,').rstrip()
            rv = num(core)
            if rv is None:
                rv = safe_eval(core)
            if rv is None or rv != corr or num(bug) == rv:
                continue                      # only corrupt a line whose value is verified correct
            st = m.start(len(g)) + lead
            cands.append((k, st, st + len(core), core, bug, m))
    if not cands:
        return None
    k, a, b, old, bug, m = rng.choice(cands)
    if code == 'EXP-02':      # replace exponent of result
        ln = lines[k]; e0, e1 = m.start(4), m.end(4)
        lines[k] = ln[:e0] + bug + ln[e1:]
        before, after = ln, lines[k]; mapping = {}
        nprop = 0
    else:
        ln = lines[k]
        lines[k] = ln[:a] + bug + ln[b:]
        before, after = ln, lines[k]
        mapping = {old.strip(): bug}
        nprop = propagate(lines, k, a + len(bug), mapping)
    txt = re.sub(r'\[[^\]]*correct[^\]]*\]', '', '\n'.join(lines), flags=re.I)   # donor's 'answered correctly' notes would contradict the label
    txt = '\n'.join(l for l in txt.split('\n') if l.strip())
    return dict(transcription_steps=txt, root_line=f'L{k + 1}', line_before=before, line_after=after,
                n_propagated=nprop, rule=code)


if __name__ == '__main__':
    tr, tax, classes, folds, dev, fit = load()
    rng = random.Random(2026)
    donors = fit[fit.root_code == 'NONE']
    rows, per = [], Counter()
    for r in donors.itertuples():
        made = 0
        order = list(range(len(RULES))); rng.shuffle(order)
        used = set()
        for i in order:
            if made >= 3:
                break
            if RULES[i][0] in used:
                continue
            out = inject(r.transcription_steps, RULES[i], rng)
            if out:
                used.add(RULES[i][0]); made += 1
                rows.append(dict(trace_id=f'syn_{r.trace_id}_{made}', donor=r.trace_id, question_id=r.question_id,
                                 topic_name=r.topic_name, question_text=r.question_text, grade=r.grade,
                                 root_code=out['rule'], **{k: v for k, v in out.items() if k != 'rule'}, rule=out['rule']))
                per[out['rule']] += 1
    with open(f'{OUT}/synthetic.jsonl', 'w') as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    st = dict(donors=len(donors), donors_used=len(set(x['donor'] for x in rows)), rows=len(rows), by_code=per,
              questions=len(set(x['question_id'] for x in rows)),
              propagated_rows=sum(x['n_propagated'] > 0 for x in rows))
    json.dump(st, open(f'{OUT}/synthetic_stats.json', 'w'), indent=1)
    print(json.dumps(st, indent=1))
