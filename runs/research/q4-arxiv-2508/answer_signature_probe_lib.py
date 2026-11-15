import re
def final_value(text):
    if not isinstance(text, str) or not text.strip(): return None
    lines = [l.split('|', 1)[-1] for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        s = line.replace('−', '-').replace('–', '-').replace(' ', '').replace('$', '')
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', s)
        parts = re.split(r'=|≈', s)
        cand = parts[-1].strip(' .,')
        m = re.findall(r'-?\d+(?:[.,]\d+)?(?:/-?\d+(?:[.,]\d+)?)?', cand)
        if m: return m[-1].replace(',', '.')
    return None

