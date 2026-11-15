"""Part 6: regime-mixture projection of public MAP@3 and of what closing each regime is worth on test (estimates)."""
from lcommon import *
D = get_all(); dev, fit, y, preds = D['dev'], D['fit'], D['y'], D['preds']
S = {k: score(y, v) for k, v in preds.items()}
fq = {qq: set(g.root_code) for qq, g in fit.groupby('question_id')}
q = dev.question_id.to_numpy()
reg = np.array(['none' if t == 'NONE' else ('novel' if (qq not in fq or t not in fq[qq]) else 'seen') for t, qq in zip(y, q)])
p3 = json.load(open(OUT / 'p3_question_regime.json')); p3b = json.load(open(OUT / 'p3b_calib.json'))
none_share = float((D['tr'].root_code == 'NONE').mean())
res = {'dev_regime_rows': Counter(reg)}
for k in ['blend', 'blend_sk', 'claude', 'scale_100']:
    res[k] = {r: round(float(S[k][reg == r].mean()), 3) for r in ['none', 'seen', 'novel']}
def proj(sc, pnew):
    return none_share * sc['none'] + (1 - none_share) * (pnew * sc['novel'] + (1 - pnew) * sc['seen'])
out = {}
for pn_name, pn in [('A', p3['A_test_predicted_new_incl_unseen(err rows)']), ('A_lo', p3b['A_test_bootstrap_ci95'][0]), ('A_hi', p3b['A_test_bootstrap_ci95'][1]),
                    ('C', p3['test_estimate_err_rows_new_code_incl_unseen_q']['C_claude_proxy(seen q)+unseen q']), ('dev', 0.53)]:
    out[pn_name] = {k: round(proj(res[k], pn), 3) for k in ['blend', 'blend_sk', 'claude']}
    # value of lifting novel rows to Claude's level, all else equal
    b = dict(res['blend_sk']); b2 = dict(b); b2['novel'] = res['claude']['novel']
    out[pn_name]['blend_sk_with_claude_level_on_novel'] = round(proj(b2, pn), 3)
    b3 = dict(b); b3['novel'] = b['novel'] + 0.10
    out[pn_name]['blend_sk_with_novel_+0.10'] = round(proj(b3, pn), 3)
res['projection_by_pnew'] = out
res['public_observed'] = {'blend': 0.7085, 'blend_sk': 0.71716}
res['note'] = 'dev regime scores come from fit-trained models; test uses full-train models. Projection assumes regime scores transfer.'
json.dump(res, open(OUT / 'p6_projection.json', 'w'), indent=1, default=int); print(json.dumps(res, indent=1, default=int))
