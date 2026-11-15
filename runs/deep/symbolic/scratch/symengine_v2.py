"""Symbolic step checker for K-12 handwritten-working transcripts (CPU, sympy).

For each transcript line (``Lk | ...``) it
  * normalises Indonesian/handwriting notation (':' and 'x' as operators, decimal commas,
    thousands dots, superscripts, sqrt, %, 2log 8 ...),
  * splits the line into statements and '='-chains,
  * checks every adjacent pair of a chain: numeric equality (rounding-aware), symbolic identity
    (random-point evaluation), identity under variable bindings found earlier, and for one-variable
    equations whether the solution set is the same as the previous equation in that variable
    (seeded from equations found in the question text),
  * continues chains across lines when a line starts with '='.
Outputs per-line verdicts, the first invalid line, parse coverage and a final-answer-correct flag.
"""
import re, math, random, signal
from fractions import Fraction
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application, convert_xor)

TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
LETTERS = {c: sp.Symbol(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDFGHJKLMRTUVWXYZ'}
LOCAL = dict(LETTERS, sqrt=sp.sqrt, log=sp.log, ln=sp.log, cbrt=sp.cbrt, pi=sp.pi,
             sin=sp.sin, cos=sp.cos, tan=sp.tan, ROOT=sp.root)
FUNCS = {'sqrt', 'log', 'ln', 'cbrt', 'pi', 'sin', 'cos', 'tan', 'ROOT'}
SUPMAP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾', '0123456789-+()')
SUPCH = '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾'


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def guarded(fn, *a, t=0.4):
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return fn(*a)
    except Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# ----------------------------------------------------------------- text normalisation
def clean_latex(s):
    s = str(s).replace('\r', ' ').replace('$', ' ')
    s = re.sub(r'(\d)\s*\\[dt]?frac\s*\{(\d+)\}\s*\{(\d+)\}', r'(\1+(\2)/(\3))', s)   # mixed number
    for _ in range(4):
        s = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\\sqrt\s*\[(\d+)\]\s*\{([^{}]*)\}', r'ROOT((\2),\1)', s)
    for _ in range(2):
        s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)
    s = re.sub(r'\^\s*\{\s*\\circ\s*\}|\\circ|\^\\circ', '°', s)
    rep = {'\\cdot': '·', '\\times': '×', '\\div': '÷', '\\left': '', '\\right': '', '\\ldots': ' ',
           '\\dots': ' ', '\\cdots': ' ', '\\pi': 'π', '\\le': '≤', '\\ge': '≥', '\\neq': '≠', '\\%': '%',
           '\\,': ' ', '\\;': ' ', '\\!': '', '\\quad': ' ', '\\text': '', '\\mathrm': '', '\\log': 'log'}
    for k, v in rep.items():
        s = s.replace(k, v)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s).replace('\\', ' ')
    return s.replace('{', '(').replace('}', ')')


DASHES = str.maketrans({'−': '-', '–': '-', '—': '-', '‐': '-', '∙': '·', '•': '·', '⋅': '·', '・': '·',
                        '×': '·', '÷': '/', '∶': ':', '≈': '≈', '‘': "'", '’': "'"})


def has_var_x(question):
    q = clean_latex(question)
    return bool(re.search(r'(\d\s*x\b|x\s*[\^²³=+\-<>)]|\(x|f\s*\(\s*x|\bx\s*=)', q))


def number_variants(s):
    """Return up to two strings with numbers converted to python syntax.
    Indonesian: '.' thousands separator, ',' decimal separator. 'd.ddd' is ambiguous."""
    amb = [False]

    def conv(m, thousands):
        t = m.group(0)
        if re.fullmatch(r'\d{1,3}(\.\d{3})+,\d+', t):
            return t.replace('.', '').replace(',', '.')
        if re.fullmatch(r'\d{1,3}(\.\d{3}){2,}', t):
            return t.replace('.', '')
        if re.fullmatch(r'\d{1,3}\.\d{3}', t):
            amb[0] = True
            return t.replace('.', '') if thousands else t
        if re.fullmatch(r'\d{1,3}(,\d{3}){2,}', t):
            return t.replace(',', '')
        if re.fullmatch(r'\d+,\d+', t):
            return t.replace(',', '.')
        return t

    pat = r'(?<![\d.,])\d+(?:[.,]\d+)+(?![\d])'
    a = re.sub(pat, lambda m: conv(m, True), s)
    if amb[0]:
        b = re.sub(pat, lambda m: conv(m, False), s)
        return [a, b]
    return [a]


WORD = re.compile(r'[A-Za-z]{2,}(?:\s*\^\s*\(?\d\)?)?')


def strip_words(seg):
    """Remove words (units, Indonesian text), keeping math function names."""
    return WORD.sub(lambda m: m.group(0) if m.group(0) in FUNCS else ' ', seg)


def has_upper_symbol(e):
    return any(str(v).isupper() for v in e.free_symbols)


def has_words(seg):
    return any(w not in FUNCS for w in re.findall(r'[A-Za-z]{2,}', seg))


def normalise_line(s, xvar):
    s = re.sub(r'^\s*L\d+\s*\|\s*', '', s)
    s = re.sub(r'\[[^\]]*\]', '   ', s)              # transcriber notes
    s = re.sub(r'\([^()]*[A-Za-z]{3,}[^()]*\)', '   ', s)   # parenthetical comments
    s = re.sub(r'\(\s*[:×x÷]\s*\d+\s*\)', ' ', s)          # (:2) annotations
    s = re.sub(r'(\d+)\s*:\s*(\d+)\s*/\s*(\d+)\s*:\s*(\d+)', r'((\1/\2)/(\3/\4))', s)
    s = clean_latex(s).translate(DASHES)
    s = re.sub(r'(\d+(?:[.,]\d+)?)\s*π', r'(\1*pi)', s)
    s = s.replace('π', ' pi ').replace('∛', 'cbrt').replace('√', 'sqrt').replace('²√', 'sqrt')
    s = re.sub(r'akar\s*', 'sqrt', s)
    # 2log 8, ²log 8, ^2log(8)
    s = re.sub(r'\^?([' + SUPCH + r'\d]+)\s*log\s*\(?\s*(\d+(?:[.,]\d+)?)\s*\)?',
               lambda m: f'log({m.group(2)},{m.group(1).translate(SUPMAP)})', s)
    # superscripts -> ^(...)
    s = re.sub('[' + SUPCH + ']+', lambda m: '^(' + m.group(0).translate(SUPMAP) + ')', s)
    s = re.sub(r'sqrt\s*(\d+(?:[.,]\d+)*)', r'sqrt(\1)', s)
    s = re.sub(r'cbrt\s*(\d+(?:[.,]\d+)*)', r'cbrt(\1)', s)
    s = re.sub(r'(\d+(?:[.,]\d+)?)\s*%', r'(\1/100)', s)
    s = s.replace('°', ' ').replace('·', '*').replace('**', '^')
    # mixed numbers  2 5/8
    s = re.sub(r'(?<![\d.,)/*^])(\d+)\s+(\d+)\s*/\s*(\d+)(?![\d^])', r'(\1+\2/\3)', s)
    # fraction simplification notation  a/b :k  (divide top and bottom) and  : k/k
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*:\s*(\d+)\s*/\s*(\d+)', r'((\1/\2)/(\3/\4))', s)
    s = re.sub(r'(\d+)\s*/\s*(\d+)\s*:\s*(\d+)(?![\d/.,])', r'(\1/\2)', s)
    # trailing operation annotation on an equation:  2x = 12 :2
    if re.search(r'\d\s*[a-z]\b[^=]*=', s):
        s = re.sub(r'(?<=\d)\s+[:÷]\s*\d+\s*$', '', s)
    # ':' between operands is division; other ':' separates label from content
    s = re.sub(r'(?<=[\d\)a-z])\s*:\s*(?=[\d\(])', '/', s)
    # 'x' as multiplication
    if xvar:
        s = re.sub(r'(?<=[\d\)])\s+[xX]\s+(?=[\d\(])', '*', s)
    else:
        s = re.sub(r'(?<=[\d\)])\s*[xX]\s*(?=[\d\(])', '*', s)
    return s


SEP = re.compile(r'→|⇒|=>|->|⟹|↳|∴|;|\||\bjadi\b|\bmaka\b|\bdan\b|\bJadi\b|\bMaka\b|(?<!\d),|,(?=\s)|:')
EQ = re.compile(r'(?<![<>!≠])=(?!=)|≈')


def statements(line, xvar):
    s = normalise_line(line, xvar)
    out = []
    s = re.sub(r'(?:-?\d+(?:\.\d+)?\s*,\s*){2,}-?\d+(?:\.\d+)?\s*=[^=]*', ' ', s)   # '0, 1, 2, 3 = 6' list sums
    for part in re.split(r'\s{3,}', s):
      for part in SEP.split(part):
        part = part.strip()
        if part and re.search(r'[\d a-zA-Z]', part):
            # trailing operation annotation on an equation:  2x = 12 :2
            out.append(part)
    return out


def parse_side(seg):
    t = strip_words(seg).strip()
    t = re.sub(r'^[\s+*/^.]+|[\s+\-*/^.]+$', '', t)   # dangling operators
    if not t or not re.search(r'[\da-zA-Z]', t):
        return None
    if t.count('(') != t.count(')'):
        t = t.replace('(', ' ').replace(')', ' ') if abs(t.count('(') - t.count(')')) else t
    if re.search(r'\^\s*\(?\s*\d{4,}', t):       # absurd exponent
        return None
    return guarded(lambda: parse_expr(t, local_dict=LOCAL, transformations=TRANS, evaluate=True))


def nvalue(e):
    try:
        v = complex(sp.N(e, 15))
    except Exception:
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v


def decimals_in(raw):
    ds = [len(m) for m in re.findall(r'\d[.,](\d+)', raw)]
    return max(ds) if ds else None


PTS = [{}, {}, {}]


def pts_for(syms):
    rng = random.Random(len(syms) * 7 + 3)
    return [{v: sp.Rational(rng.randint(11, 97), rng.randint(5, 13)) for v in syms} for _ in range(3)]


def close(a, b, raw, approx):
    if a is None or b is None:
        return None
    d = abs(a - b); m = max(abs(a), abs(b), 1e-12)
    dec = decimals_in(raw)
    tol = 1e-7 * m + 1e-9
    if dec is not None:
        tol = max(tol, 0.51 * 10 ** (-dec), 0.01 * m)
    if approx:
        tol = max(tol, 0.003 * m)
    return d <= tol


def identity(a, b, raw, approx, binding=None):
    """True if a == b as math (numerically or at random points), False if numerically different,
    None if undecidable."""
    if binding:
        a = guarded(lambda: a.subs(binding)); b = guarded(lambda: b.subs(binding))
        if a is None or b is None:
            return None
    fs = (a.free_symbols | b.free_symbols)
    if any(str(v).isupper() for v in fs):
        return None
    if not fs:
        r = close(nvalue(a), nvalue(b), raw, approx)
        return r
    res = []
    for P in pts_for(sorted(fs, key=str)):
        va = guarded(lambda: nvalue(a.subs(P))); vb = guarded(lambda: nvalue(b.subs(P)))
        r = close(va, vb, raw, approx)
        if r is None:
            return None
        res.append(r)
    return all(res)


def solve1(a, b):
    fs = list((a.free_symbols | b.free_symbols))
    if len(fs) != 1:
        return None, None
    v = fs[0]
    if str(v).isupper():
        return None, None
    sol = guarded(lambda: sp.solve(sp.Eq(a, b), v), t=0.8)
    if sol is None:
        return v, None
    vals = []
    for s in sol:
        z = nvalue(s)
        if z is not None and abs(z.imag) < 1e-9:
            vals.append(round(z.real, 6))
    return v, frozenset(vals)


def runon_ok(a, rb, approx):
    """'a = b - c' where b == a: the student kept writing the next operation (run-on equals)."""
    t = strip_words(rb); depth = 0; tried = 0
    for i, ch in enumerate(t):
        depth += ch == '('; depth -= ch == ')'
        if depth == 0 and ch in '+-*/' and i > 0 and t[:i].strip():
            e = parse_side(t[:i]); tried += 1
            if e is not None and not e.free_symbols and identity(a, e, t[:i], approx):
                return True
            if tried >= 6:
                break
    return False


def is_label(seg_raw, expr, rest_numeric):
    if has_words(seg_raw):
        return True
    t = strip_words(seg_raw).strip()
    if re.fullmatch(r"[A-Za-z]['_]*\d*", t):
        return True
    if re.fullmatch(r'[A-Za-z]\s*\(.*\)', t) and not re.search(r'[+\-*/^]', t.split('(', 1)[0]):
        return True                                   # f(2a-1), P(A), n(S), K(x)
    return False


OPS = [('pow', r'\^|sqrt|cbrt|ROOT'), ('div', r'/'), ('mul', r'\*|(?<=\d)\(|\)\('), ('add', r'\+'),
       ('sub', r'(?<=[\d\)a-z\s])-'), ('log', r'log'), ('pct', r'/100\)'), ('dec', r'\d\.\d')]


def ops_of(s):
    return sorted({k for k, p in OPS if re.search(p, s)})


# ----------------------------------------------------------------- transcript checker
def question_equations(question, xvar):
    q = str(question)
    segs = re.findall(r'\$([^$]+)\$', q) or [q]
    eqs = []
    for seg in segs:
        for st in statements(clean_latex(seg), xvar):
            parts = EQ.split(st)
            if len(parts) != 2:
                continue
            a, b = parse_side(parts[0]), parse_side(parts[1])
            if a is None or b is None:
                continue
            v, sol = solve1(a, b)
            if v is not None and sol:
                eqs.append((v, sol))
    return eqs


def check_transcript(transcript, question='', answer=''):
    xvar = has_var_x(question)
    approx = bool(re.search(r'π|\\pi|3[.,]14|22/7|22 */ *7', str(question) + str(transcript)))
    lines = [l for l in str(transcript).splitlines() if l.strip()]
    sols = {}
    for v, sol in question_equations(question, xvar):
        sols.setdefault(v, sol)
    binding = {}
    prev_last = None
    out = []
    for i, line in enumerate(lines):
        lno = i + 1
        m = re.match(r'\s*L(\d+)\s*\|', line)
        if m:
            lno = int(m.group(1))
        rec = dict(line=lno, n_stmt=0, n_parsed=0, n_checks=0, n_valid=0, n_invalid=0, n_eq=0,
                   eq_changed=0, ops=[], reasons=[], values=[])
        variants = number_variants(line)
        best = None
        for var in variants:                         # ambiguous thousands/decimal: keep best reading
            r = _check_line(var, xvar, approx, dict(sols), dict(binding), prev_last, rec_template=rec)
            if best is None or (r[0]['n_invalid'], -r[0]['n_parsed']) < (best[0]['n_invalid'], -best[0]['n_parsed']):
                best = r
        rec, sols_new, bind_new, last = best
        sols, binding = sols_new, bind_new
        if last is not None:
            prev_last = last
        out.append(rec)
    return out


def _check_line(line, xvar, approx, sols, binding, prev_last, rec_template):
    rec = {k: (list(v) if isinstance(v, list) else v) for k, v in rec_template.items()}
    last = None
    sts = statements(line, xvar)
    rec['n_stmt'] = len(sts)
    for k, st in enumerate(sts):
        parts = [p for p in EQ.split(st)]
        cont = False
        if parts and not parts[0].strip() and k == 0 and prev_last is not None and prev_last[1]:
            parts = parts[1:]; cont = True        # '= ...' continues the previous line's expression
        parts = [p for p in parts if p.strip()]
        if not parts:
            continue
        exprs = [parse_side(p) for p in parts]
        rec['n_parsed'] += sum(e is not None for e in exprs)
        rec['ops'] = sorted(set(rec['ops']) | set(ops_of(strip_words(st))))
        chain = list(zip(parts, exprs))
        if cont:
            chain = [('<prev>', prev_last[0])] + chain
        # leading label (Luas = ..., f(2a-1) = ..., x = ...)
        start = 0
        if len(chain) >= 2 and chain[0][0] != '<prev>' and is_label(chain[0][0], chain[0][1], True):
            lab = strip_words(chain[0][0]).strip()
            start = 1
            # a bare single-letter label with numeric rhs becomes a binding
            if re.fullmatch(r'[a-z]', lab):
                vals = [e for _, e in chain[1:] if e is not None and not e.free_symbols]
                if vals:
                    z = nvalue(vals[-1])
                    sym = LETTERS[lab]
                    if z is not None and abs(z.imag) < 1e-9:
                        rec['n_eq'] += 1
                        new = frozenset([round(z.real, 6)])
                        if sym in sols and sols[sym] and not (new & sols[sym]):
                            # stated solution differs from what earlier equations imply
                            if not any(abs(a - b) <= 1e-3 * max(1, abs(a)) for a in new for b in sols[sym]):
                                rec['eq_changed'] += 1; rec['reasons'].append(f'{lab}={z.real:g} contradicts earlier equation')
                        sols[sym] = new; binding[sym] = vals[-1]
        ch = chain[start:]
        for j in range(len(ch) - 1):
            (ra, a), (rb, b) = ch[j], ch[j + 1]
            if a is None or b is None:
                continue
            raw = ra + ' ' + rb
            rec['n_checks'] += 1
            ok = identity(a, b, raw, approx)
            if ok is None:
                continue
            if ok:
                rec['n_valid'] += 1; continue
            fs = a.free_symbols | b.free_symbols
            if not fs and runon_ok(a, rb, approx):
                rec['n_valid'] += 1; rec['runon'] = rec.get('runon', 0) + 1; continue
            if not fs:
                rec['n_invalid'] += 1; rec['reasons'].append(f'{ra.strip()} != {rb.strip()}'); continue
            # symbolic: identity under bindings?
            if binding and identity(a, b, raw, approx, binding):
                rec['n_valid'] += 1; continue
            # otherwise an equation: compare its solution set with the last one in that variable
            v, sol = solve1(a, b)
            rec['n_eq'] += 1
            if v is None or sol is None:
                continue
            if v in sols and sols[v] and sol is not None:
                same = any(abs(p - q) <= 1e-3 * max(1, abs(q)) for p in sol for q in sols[v])
                if not same:
                    rec['eq_changed'] += 1
                    rec['reasons'].append(f'{v}: {sorted(sols[v])} -> {sorted(sol)}')
            if sol:
                sols[v] = sol
            if len(sol) == 1 and (a == v or b == v):
                binding[v] = sp.Float(next(iter(sol)))
        # values written on the line (results of chains)
        for _, e in ch:
            if e is not None and not e.free_symbols:
                z = nvalue(e)
                if z is not None and abs(z.imag) < 1e-9:
                    rec['values'].append(round(z.real, 6))
        if ch and ch[-1][1] is not None:
            last = (ch[-1][1], bool(re.search(r'[+\-*/^]|sqrt|cbrt|log', strip_words(ch[-1][0]))))
    rec['invalid'] = bool(rec['n_invalid'] or rec['eq_changed'])
    return rec, sols, binding, last


# ----------------------------------------------------------------- final answer
def answer_targets(answer, question=''):
    """Parsed target values/expressions of the correct answer (list of sympy exprs)."""
    xvar = has_var_x(question) or bool(re.search(r'x', clean_latex(answer)))
    res = []
    for var in number_variants(clean_latex(answer).translate(DASHES)):
        for st in statements(var, xvar):
            parts = EQ.split(st)
            e = parse_side(parts[-1]) if parts else None
            if e is not None:
                res.append(e)
    return res


def final_answer_flag(transcript, answer, question='', k=2):
    """1 if the last k lines contain a value/expression equal to the correct answer, 0 if the
    answer is parseable but not matched, None if the answer cannot be parsed."""
    tg = answer_targets(answer, question)
    tg = [t for t in tg if t is not None]
    if not tg:
        return None
    xvar = has_var_x(question)
    approx = bool(re.search(r'π|\\pi|3[.,]14|22/7', str(question) + str(transcript)))
    lines = [l for l in str(transcript).splitlines() if l.strip()][-k:]
    cands = []
    for line in lines:
        for var in number_variants(line):
            for st in statements(var, xvar):
                for p in EQ.split(st):
                    e = parse_side(p)
                    if e is not None:
                        cands.append((p, e))
    # main target = the last parsed answer item (e.g. "P(x)=..., Harga = Rp875.000" -> both needed)
    hits = 0
    for t in tg:
        ok = False
        for p, e in cands:
            r = identity(e, t, p + ' ' + str(answer), approx)
            if r:
                ok = True; break
        hits += ok
    return hits / len(tg)
