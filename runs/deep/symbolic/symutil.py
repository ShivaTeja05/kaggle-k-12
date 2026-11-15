"""Shared helpers (copied from eval_localise.py so importing has no side effects)."""
import re, json, difflib


def root_lines(m):
    for a in (json.loads(m) if m else []):
        if a.get('root'):
            nums = [int(x) for x in re.findall(r'L(\d+)', a['step'])]
            if len(nums) == 2:
                nums = list(range(nums[0], nums[1] + 1))
            return set(nums), a['span']
    return None, None


def body(l):
    return re.sub(r'^\s*L\d+\s*\|\s*', '', l)


def norm(s):
    return re.sub(r'\s+', '', str(s).replace('−', '-').replace('×', 'x').lower())


def map_span(span, transcript):
    """Map a gold span onto the line numbers of another transcript (best fuzzy line)."""
    lines = [l for l in str(transcript).splitlines() if l.strip()]
    sp_ = norm(span); best, bi = 0, None
    for i, l in enumerate(lines):
        b = norm(body(l))
        if not b or not sp_:
            continue
        m = difflib.SequenceMatcher(None, sp_, b, autojunk=False)
        lm = m.find_longest_match(0, len(sp_), 0, len(b)).size / len(sp_)
        sc = max(lm, m.ratio())
        if sc > best:
            best, bi = sc, i
    if bi is None:
        return None, best
    mm = re.match(r'\s*L(\d+)', lines[bi])
    return (int(mm.group(1)) if mm else bi + 1), best


# ---------------------------------------------------------------- reference-path values


def typ(reason):
    if reason.startswith('cont:'): return 'cont'
    if 'contradicts' in reason: return 'bind'
    if re.match(r'^\w+: \[', reason): return 'eq'
    return 'arith'


def first_of(L, types):
    return next((x['line'] for x in L if any(typ(z) in types for z in x['reasons'])), None)

