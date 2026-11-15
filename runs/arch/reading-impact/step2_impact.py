"""Reading impact on per-row MAP: single models, reference blend, hybrid/oracle reading fixes."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/reading-impact')
import numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT + 'scripts')
from evaluate_claude import claude_scores

out = {}
R = pd.read_csv(ROOT + 'runs/arch/reading-impact/per_row_reading.csv', keep_default_na=False,
                dtype={'trace_id': str})
assert R.trace_id.tolist() == ids
num = lambda c: pd.to_numeric(R[c], errors='coerce')
P['claude'] = claude_scores(ROOT + 'runs/claude/dev_opus.jsonl', ids, classes)[0]
REF = [('32_tx', .35), ('8_tx_e3', .30), ('vl', .35)]
ref = rs(geo(REF)); out['ref_blend_map3'] = float(ref.mean())
S = {k: rs(P[k]) for k in P}
S['blend_ref'] = ref

# ---------- 1. gold vs tx, same model ----------
pairs = {'32B': ('32_tx', '32_gold'), '8B_s0': ('s0_tx', 's0_gold'), '8B_s25': ('s25_tx', 's25_gold'), '8B_s100': ('s100_tx', 's100_gold')}
g = {}
for n, (a, b) in pairs.items():
    d, lo, hi = boot(S[b], S[a])
    g[n] = dict(tx=float(S[a].mean()), gold=float(S[b].mean()), delta=d, ci=[lo, hi],
                rows_gold_better=int((S[b] > S[a]).sum()), rows_tx_better=int((S[a] > S[b]).sum()))
out['single_model_gold_minus_tx'] = g

# ---------- 2. ensemble-level gold ceilings (fixed weights, no tuning) ----------
ens = {}
def ens_entry(name, parts):
    s = rs(geo(parts)); d, lo, hi = boot(s, ref)
    ens[name] = dict(map3=float(s.mean()), delta_vs_ref=d, ci=[lo, hi]); return s
ens_entry('ref 32tx+e3tx+vl', REF)
ens_entry('32gold+e3tx+vl (only 32B reads gold)', [('32_gold', .35), ('8_tx_e3', .30), ('vl', .35)])
for s_ in ('s25', 's100'):
    a = ens_entry(f'32tx+{s_}tx+vl (proxy)', [('32_tx', .35), (f'{s_}_tx', .30), ('vl', .35)])
    b = ens_entry(f'32gold+{s_}gold+vl (proxy, all text reads gold)', [('32_gold', .35), (f'{s_}_gold', .30), ('vl', .35)])
    d, lo, hi = boot(b, a)
    ens[f'{s_} proxy gold-minus-tx'] = dict(delta=d, ci=[lo, hi])
# estimated e3-gold: shift e3 by the s25 gold/tx log-ratio (s25 tx single 0.666 ~ e3 0.667). ESTIMATE.
P['e3_goldproxy'] = normalize(P['8_tx_e3'] * P['s25_gold'] / P['s25_tx'])
ens_entry('32gold+e3goldproxy+vl (ESTIMATE)', [('32_gold', .35), ('e3_goldproxy', .30), ('vl', .35)])
out['ensemble'] = ens

# ---------- 3. reading quality vs loss ----------
isroot = R.root_step != ''
span_ok = (num('m_span_dist') == 0)
span_near = (num('m_span_dist') <= 0.15)
rl_ok = (num('m_rootline_cer_best') <= 0.10)
pcer = num('m_page_cer').clip(upper=1.0)
groups = {
  'NONE rows': ~isroot,
  'root span exact (machine)': isroot & span_ok,
  'root span misread (machine)': isroot & ~span_ok,
  'root span badly misread (dist>0.15)': isroot & ~span_near,
  'root line ok (CER<=0.10), rest of page CER>0.10': isroot & rl_ok & (num('m_rest_cer') > 0.10),
  'root line ok, rest of page ok': isroot & rl_ok & (num('m_rest_cer') <= 0.10),
  'root line bad (CER>0.10)': isroot & ~rl_ok,
  'page CER<=0.05': pcer <= 0.05, 'page CER 0.05-0.25': (pcer > 0.05) & (pcer <= 0.25), 'page CER>0.25': pcer > 0.25,
}
tab = []
for n, m in groups.items():
    m = m.fillna(False).values
    rec = dict(group=n, n=int(m.sum()))
    for k in ('32_tx', '32_gold', 's100_tx', 's100_gold', 's25_tx', 's25_gold', '8_tx_e3', 'vl', 'blend_ref', 'claude'):
        rec[k] = round(float(S[k][m].mean()), 3)
    rec['32 gold-tx'] = round(rec['32_gold'] - rec['32_tx'], 3)
    rec['s100 gold-tx'] = round(rec['s100_gold'] - rec['s100_tx'], 3)
    rec['s25 gold-tx'] = round(rec['s25_gold'] - rec['s25_tx'], 3)
    rec['ref_blend_points_lost'] = round(float((1 - ref[m]).sum()), 1)
    tab.append(rec)
tab = pd.DataFrame(tab); out['groups'] = tab.to_dict('records')
print(tab.to_string())

# misread root span -> wrong code?  top-1 and top-3 miss rates
def miss(k, m, top=1):
    t = topk(P[k] if k in P else geo(REF), classes)[:, :top]
    return float(np.mean([y[i] not in t[i] for i in np.flatnonzero(m)]))
P['blend_ref'] = geo(REF)
mr = {}
for n in ('root span exact (machine)', 'root span misread (machine)', 'root span badly misread (dist>0.15)'):
    m = groups[n].fillna(False).values
    mr[n] = {k: dict(top1_wrong=round(miss(k, m, 1), 3), top3_miss=round(miss(k, m, 3), 3))
             for k in ('32_tx', '32_gold', 's100_tx', 's100_gold', 'blend_ref', 'claude')}
out['miss_rates_by_root_span'] = mr

# correlations (Spearman) of CER with per-row loss
def sp(a, b):
    a, b = pd.Series(a), pd.Series(b); ok = a.notna() & b.notna()
    return round(float(a[ok].rank().corr(b[ok].rank())), 3)
cor = {}
for k in ('32_tx', '8_tx_e3', 's100_tx', 'blend_ref', '32_gold', 'claude'):
    cor[k] = {c: sp(R[c].pipe(pd.to_numeric, errors='coerce'), 1 - S[k]) for c in ('m_page_cer', 'm_rootline_cer_best', 'm_span_dist', 'gold_chars')}
cor['32 (gold-tx) gain'] = {c: sp(pd.to_numeric(R[c], errors='coerce'), S['32_gold'] - S['32_tx']) for c in ('m_page_cer', 'm_rootline_cer_best', 'm_span_dist', 'gold_chars')}
cor['s100 (gold-tx) gain'] = {c: sp(pd.to_numeric(R[c], errors='coerce'), S['s100_gold'] - S['s100_tx']) for c in ('m_page_cer', 'm_rootline_cer_best', 'm_span_dist', 'gold_chars')}
out['spearman_cer_vs_loss'] = cor

# ---------- 4. hybrid oracle reading fixes inside the ensemble ----------
# Replace tx-based text bundles with gold-based ones ONLY on rows in a subset (simulates fixing that reading error).
def hybrid(mask, e3_gold='e3_goldproxy'):
    m = mask.fillna(False).values[:, None]
    P['h32'] = np.where(m, P['32_gold'], P['32_tx'])
    P['h8'] = np.where(m, P[e3_gold], P['8_tx_e3'])
    return rs(geo([('h32', .35), ('h8', .30), ('vl', .35)]))
def hybrid32(mask):
    m = mask.fillna(False).values[:, None]
    P['h32'] = np.where(m, P['32_gold'], P['32_tx'])
    return rs(geo([('h32', .35), ('8_tx_e3', .30), ('vl', .35)]))
hy = {}
rng = np.random.default_rng(1)
fixes = {
  'fix root span only (rows with span misread)': isroot & ~span_ok,
  'fix root line only (root line CER>0.10)': isroot & ~rl_ok,
  'fix non-root page only (root line ok, rest CER>0.10)': isroot & rl_ok & (num('m_rest_cer') > 0.10),
  'fix NONE rows with page CER>0.10': ~isroot & (pcer > 0.10),
  'fix all rows with page CER>0.10': pcer > 0.10,
  'fix all rows (perfect reading)': pd.Series(True, index=R.index),
}
for n, m in fixes.items():
    s32 = hybrid32(m); sall = hybrid(m)
    d1 = boot(s32, ref); d2 = boot(sall, ref)
    hy[n] = dict(rows=int(m.fillna(False).sum()),
                 only_32B_gold=dict(map3=round(float(s32.mean()), 4), delta=round(d1[0], 4), ci=[round(d1[1], 4), round(d1[2], 4)]),
                 both_text_gold_ESTIMATE=dict(map3=round(float(sall.mean()), 4), delta=round(d2[0], 4), ci=[round(d2[1], 4), round(d2[2], 4)]))
    # random control of same size
    k = int(m.fillna(False).sum())
    if 0 < k < len(R):
        ds = []
        for _ in range(200):
            rm = pd.Series(False, index=R.index); rm.iloc[rng.choice(len(R), k, replace=False)] = True
            ds.append(hybrid32(rm).mean() - ref.mean())
        hy[n]['random_same_size_only_32B_delta_mean'] = round(float(np.mean(ds)), 4)
out['hybrid_reading_fixes'] = hy
print(json.dumps({k: out[k] for k in ('single_model_gold_minus_tx', 'ensemble', 'miss_rates_by_root_span', 'spearman_cer_vs_loss', 'hybrid_reading_fixes')}, indent=1))

# ---------- 5. Claude reading/localisation vs gold ----------
c = {}
c['machine_page_cer_median'] = float(num('m_page_cer').median()); c['claude_page_cer_median'] = float(num('c_page_cer').median())
c['machine_span_exact_rate'] = float((num('m_span_dist')[isroot] == 0).mean())
c['claude_span_exact_rate'] = float((num('c_span_dist')[isroot] == 0).mean())
c['machine_rootline_ok_rate'] = float((num('m_rootline_cer_best')[isroot] <= .1).mean())
c['claude_rootline_ok_rate'] = float((num('c_rootline_cer_best')[isroot] <= .1).mean())
cspan_ok = isroot & (num('c_span_dist') == 0)
c['claude_map_when_claude_span_exact'] = float(S['claude'][cspan_ok.values].mean()); c['n_c_ok'] = int(cspan_ok.sum())
c['claude_map_when_claude_span_misread'] = float(S['claude'][(isroot & ~cspan_ok).values].mean()); c['n_c_bad'] = int((isroot & ~cspan_ok).sum())
def first_line(s):
    import re; m = re.match(r'L(\d+)', str(s)); return int(m.group(1)) if m else None
def contains(rs_, pred):
    import re
    m = re.match(r'L(\d+)(?:\s*-\s*L?(\d+))?', rs_); p = first_line(pred)
    if not m or p is None: return False
    a = int(m.group(1)); b = int(m.group(2) or a); return a <= p <= b
loc = [contains(a, b) for a, b in zip(R.root_step[isroot], R.c_root_line_pred[isroot])]
c['claude_root_line_localisation_acc'] = float(np.mean(loc))
c['claude_map_when_root_localised'] = float(S['claude'][np.flatnonzero(isroot)[np.array(loc)]].mean())
c['claude_map_when_root_not_localised'] = float(S['claude'][np.flatnonzero(isroot)[~np.array(loc)]].mean())
c['n_not_localised'] = int((~np.array(loc)).sum())
out['claude_reading'] = c
print(json.dumps(c, indent=1))

# ---------- 6. error-erasure: tx model predicts NONE where gold model gets the error ----------
t32tx, t32g = topk(P['32_tx'], classes)[:, 0], topk(P['32_gold'], classes)[:, 0]
yy = np.array(y)
er = {}
for a, b, n in (('32_tx', '32_gold', '32B'), ('s100_tx', 's100_gold', '8B_s100')):
    ta, tb = topk(P[a], classes)[:, 0], topk(P[b], classes)[:, 0]
    er[n] = dict(gold_top1_right_tx_top1_NONE=int(((tb == yy) & (ta == 'NONE') & (yy != 'NONE')).sum()),
                 gold_top1_right_tx_top1_other_code=int(((tb == yy) & (ta != yy) & (ta != 'NONE')).sum()),
                 tx_top1_right_gold_wrong=int(((ta == yy) & (tb != yy)).sum()),
                 NONE_rows_false_alarm_tx=int(((yy == 'NONE') & (ta != 'NONE')).sum()),
                 NONE_rows_false_alarm_gold=int(((yy == 'NONE') & (tb != 'NONE')).sum()))
out['error_erasure'] = er
print(json.dumps(er, indent=1))
# ref blend top1 breakdown
tb = topk(P['blend_ref'], classes)[:, 0]
out['blend_ref_top1'] = dict(correct=int((tb == yy).sum()), pred_NONE_on_error=int(((tb == 'NONE') & (yy != 'NONE')).sum()),
                             pred_code_on_NONE=int(((tb != 'NONE') & (yy == 'NONE')).sum()),
                             wrong_code_on_error=int(((tb != 'NONE') & (yy != 'NONE') & (tb != yy)).sum()))
print(out['blend_ref_top1'])
json.dump(out, open(ROOT + 'runs/arch/reading-impact/results.json', 'w'), indent=1, default=float)
