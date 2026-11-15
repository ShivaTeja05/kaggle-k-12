"""Mal-rule (buggy-rule) explanation of invalid arithmetic steps + correct-path features.
For each numerically invalid pair  A = B  found by the checker, re-evaluate A under known buggy rules;
if a buggy rule reproduces B, emit that rule's taxonomy code.  Also:
  ANS-01 path feature: final answer wrong, but the last value written appears in other students'
  correct solutions of the same question (stopped at / reported an intermediate quantity)."""
import re, sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from symengine import parse_side, nvalue, close
from symutil import typ

NUM = r'(-?\d+(?:\.\d+)?)'


def val(s):
    e = parse_side(s)
    if e is None or e.free_symbols:
        return None
    z = nvalue(e)
    return None if z is None or abs(z.imag) > 1e-9 else z.real


def l2r(s):
    """Evaluate a flat + - * / expression strictly left to right (no precedence)."""
    t = s.replace(' ', '')
    if '(' in t or '^' in t or 'sqrt' in t:
        return None
    toks = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', t)
    if not toks or ''.join(toks) != t:
        return None
    if toks[0] == '-':
        toks = ['0'] + toks
    try:
        acc = float(toks[0])
        for op, x in zip(toks[1::2], toks[2::2]):
            x = float(x)
            acc = acc + x if op == '+' else acc - x if op == '-' else acc * x if op == '*' else acc / x
    except (ValueError, ZeroDivisionError, IndexError):
        return None
    if sum(o in '*/' for o in toks[1::2]) == 0 or sum(o in '+-' for o in toks[1::2]) == 0:
        return None            # needs mixed precedence to be informative
    return acc


RULES = [
    ('EXP-01', lambda s: re.sub(r'(\d+)\s*\^\s*\(?\s*(\d+)\s*\)?', r'(\1*\2)', s)),                      # a^n = a*n
    ('FRA-01', lambda s: re.sub(r'(\d+)\s*/\s*(\d+)\s*([+\-])\s*(\d+)\s*/\s*(\d+)', r'((\1\3\4)/(\2\3\5))', s)),  # tops+bottoms
    ('FRA-03', lambda s: re.sub(r'\(?(\d+)\s*/\s*(\d+)\)?\s*/\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)', r'((\1/\2)*(\3/\4))', s)),
    ('FRA-03', lambda s: re.sub(r'\(\(?(\d+)\s*/\s*(\d+)\)?\)\s*/\s*\(\(?(\d+)\s*/\s*(\d+)\)?\)', r'((\1/\2)*(\3/\4))', s)),
    ('EXP-03', lambda s: re.sub(r'\(([^()]+?)\s*([+\-])\s*([^()]+?)\)\s*\^\s*\(?2\)?', r'((\1)^2\2(\3)^2)', s)),
    ('EXP-07', lambda s: re.sub(r'sqrt\(([^()]+?)\s*([+\-])\s*([^()]+?)\)', r'(sqrt(\1)\2sqrt(\3))', s)),
    ('EXP-07', lambda s: re.sub(r'sqrt\((\d+)\)\s*([+\-])\s*sqrt\((\d+)\)', r'sqrt(\1\2\3)', s)),
    ('EXP-02', lambda s: re.sub(r'(\d+)\s*\^\s*\(?(\d+)\)?\s*\*\s*\1\s*\^\s*\(?(\d+)\)?', r'(\1^(\2*\3))', s)),   # a^m a^n = a^(mn)
    ('EXP-02', lambda s: re.sub(r'(\d+)\s*\^\s*\(?(\d+)\)?\s*\*\s*\1\s*\^\s*\(?(\d+)\)?', r'((\1*\1)^(\2+\3))', s)),
    ('EXP-05', lambda s: re.sub(r'(\d+)\s*\^\s*\(?0\)?', r'0', s)),
    ('RAT-02', lambda s: s.replace('/100)', ')')),
]


def explain(ra, rb):
    """Return list of codes whose buggy rule turns ra into the value of rb."""
    vb = val(rb)
    if vb is None:
        return []
    va = val(ra)
    out = []
    raw = ra + ' ' + rb
    if va is not None and abs(va) > 1e-9 and close(complex(-va), complex(vb), raw, False):
        out.append('SIGN')
    z = l2r(ra)
    if z is not None and close(complex(z), complex(vb), raw, False):
        out.append('OOO-01')
    for code, f in RULES:
        t = f(ra)
        if t != ra:
            z = val(t)
            if z is not None and close(complex(z), complex(vb), raw, False):
                out.append(code)
    if not out and va is not None:
        d = abs(va - vb)
        if d > 0 and (d <= max(10, 0.1 * abs(va)) or sorted(str(int(abs(vb)))) == sorted(str(int(abs(va))))):
            out.append('SLIP')
    return out


def row_tokens(rec, final_wrong=None, last_in_path=None):
    toks = collections.Counter()
    first = None
    for x in rec['lines']:
        for rz in x['reasons']:
            if typ(rz) != 'arith' or ' != ' not in rz:
                continue
            ra, rb = rz.split(' != ', 1)
            ex = explain(ra, rb)
            for c in ex:
                toks['mal_' + c] += 1
            if first is None:
                first = ex
    if first:
        for c in first:
            toks['first_mal_' + c] += 1
    if final_wrong and last_in_path:
        toks['path_ANS-01'] += 1
    return dict(toks)
