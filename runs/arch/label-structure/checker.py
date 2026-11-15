"""Crude CPU final-answer checker: does the transcript's tail state the correct_answer?"""
import re
from fractions import Fraction

SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')

def clean_latex(s):
    s = str(s).replace('\r', ' ').replace('$', '')
    for _ in range(3):
        s = re.sub(r'\\d?frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^{}]*)\}', r'root\1(\2)', s)
    s = re.sub(r'\\sqrt\{([^{}]*)\}', r'√(\1)', s)
    s = s.replace('\\cdot', '·').replace('\\times', '×').replace('^{\\circ}', '°').replace('\\circ', '°')
    s = s.replace('\\left', '').replace('\\right', '').replace('\\ldots', '').replace('\\', '')
    s = s.replace('{', '(').replace('}', ')').replace('−', '-').replace('–', '-')
    return s.translate(SUP)

NUM = re.compile(r'-?\d+(?:[.,]\d+)*')

def to_val(tok):
    t = tok
    neg = t.startswith('-'); t = t.lstrip('-')
    # Indonesian: '.' thousands, ',' decimal; tolerate english decimals
    if re.fullmatch(r'\d{1,3}(\.\d{3})+(,\d+)?', t):
        t = t.replace('.', '').replace(',', '.')
    elif ',' in t and '.' not in t:
        t = t.replace(',', '.')
    elif re.fullmatch(r'\d{1,3}(,\d{3})+', t):
        t = t.replace(',', '')
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v

def values(s):
    s = clean_latex(s)
    vals = []
    for m in NUM.finditer(s):
        v = to_val(m.group())
        if v is not None:
            vals.append(round(v, 6)); vals.append(round(abs(v), 6))
    # simple fractions a/b, (a)/(b)
    for a, b in re.findall(r'\(?(-?\d+(?:[.,]\d+)?)\)?\s*/\s*\(?(-?\d+(?:[.,]\d+)?)\)?', s):
        va, vb = to_val(a), to_val(b)
        if va is not None and vb:
            vals.append(round(va / vb, 6))
    # mixed numbers "1 1/2"
    for w, a, b in re.findall(r'(\d+)\s+(\d+)\s*/\s*(\d+)', s):
        if int(b):
            vals.append(round(int(w) + int(a) / int(b), 6))
    return set(vals)

def key_values(answer):
    """Values in the correct answer that a final line must reproduce."""
    s = clean_latex(answer)
    fr = re.findall(r'\(?(-?\d+(?:[.,]\d+)?)\)?\s*/\s*\(?(-?\d+(?:[.,]\d+)?)\)?', s)
    if len(NUM.findall(s)) == 2 and len(fr) == 1:
        a, b = map(to_val, fr[0])
        if a is not None and b:
            return {round(a / b, 6)}, 'frac'
    toks = [to_val(t) for t in NUM.findall(s)]
    toks = [round(abs(t), 6) for t in toks if t is not None]
    if toks:
        return set(toks), 'num'
    return set(), 'text'

def words(s):
    return set(w for w in re.findall(r'[a-zA-Z]+', clean_latex(s).lower()) if len(w) > 1)

def tail_lines(transcript, k):
    lines = [re.sub(r'^L\d+\s*\|\s*', '', l) for l in str(transcript).splitlines() if l.strip()]
    return lines[-k:]

def check(transcript, answer, k=2):
    """Returns (score in {0,0.5,1}, kind). 1 = tail matches answer."""
    kv, kind = key_values(answer)
    tail = ' '.join(tail_lines(transcript, k))
    if kind in ('num', 'frac'):
        tv = values(tail)
        hit = len(kv & tv) / len(kv)
        return hit, kind
    aw = words(answer)
    if not aw:
        return 0.5, 'empty'
    return len(aw & words(tail)) / len(aw), kind
