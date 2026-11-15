"""T2. Realistic-failure simulation of the R1D student, and calibration of Gate G1.
The d5 dose-response replaced rationales at RANDOM rows with a same-question rationale of a random code. A fine-tuned
student fails differently: (i) it is faithful mostly where it already knows the answer, and (ii) where it is wrong it
writes a fluent rationale for ITS OWN wrong decision. We model the student's decision with the existing Claude-distilled
8B (scale_100, machine transcripts, top-1 agrees with Claude on 68.8% of dev rows).
 - faithful rows: Claude's real dev rationale (as in d4)
 - unfaithful rows: a Claude rationale from an unlabelled photo whose first code == the student's top-1
   (same question if one exists, else any question), decoded OUT-OF-FOLD (3-fold over the 26k) so the decoder has not
   memorised the substitute text.
Faithful set = rows where student top-1 == Claude top-1, then extra rows added (random, or adversarially: seen-pair and
NONE rows first) until the target agreement is reached. 5 seeds per setting; nested cross-fit fusion as T1."""
import json
import numpy as np
from sklearn.model_selection import KFold
from common import *

full = cached_decoder('full', U_txt, U_y)
oof_f = OUT/'unl_oof_lr.npy'
if oof_f.exists(): U_oof = np.load(oof_f)
else:
    U_oof = np.zeros((len(U_ids), len(classes)))
    for k, (a, b) in enumerate(KFold(3, shuffle=True, random_state=0).split(U_ids)):
        d = Decoder([U_txt[i] for i in a], U_y[a]); U_oof[b] = d.lr_proba([U_txt[i] for i in b])
    np.save(oof_f, U_oof)
U_S = desc_sim(U_txt)
dev_lr = full.lr_proba(dev_txt); dev_S = desc_sim(dev_txt)
stud = aligned_bundle(ROOT/'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', ids, classes); s1 = stud.argmax(1)
agree = s1 == claude_first
by_qc = {}; by_c = {}; by_q = {}
for j, (q, c) in enumerate(zip(U_q, U_y)): by_qc.setdefault((q, c), []).append(j); by_c.setdefault(c, []).append(j); by_q.setdefault(q, []).append(j)
easy = np.isin(bins, ['code_seen_for_q', 'NONE'])
res = {'substitute_source_counts': {'same_q_same_code': int(sum(bool(by_qc.get((qid[i], s1[i]))) for i in np.flatnonzero(~agree))), 'other_q_same_code': int(sum((not by_qc.get((qid[i], s1[i]))) and bool(by_c.get(s1[i])) for i in np.flatnonzero(~agree))), 'fallback': int(sum((not by_qc.get((qid[i], s1[i]))) and (not by_c.get(s1[i])) for i in np.flatnonzero(~agree)))}, 'student_proxy': 'scale_100 8B tx', 'base_agreement': round(float(agree.mean()), 4), 'base_agreement_unseen106': round(float(agree[unseen].mean()), 4)}
for mode in ['random', 'adversarial_easy_first']:
    for target in ['base', 0.75, 0.80, 0.85, 0.90, 1.0]:
        outs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            faithful = agree.copy(); rest = np.flatnonzero(~agree)
            n_need = 0 if target == 'base' else max(0, int(round(target*len(ids))) - faithful.sum())
            if mode == 'random': order = rng.permutation(rest)
            else: order = np.concatenate([rng.permutation(rest[easy[rest]]), rng.permutation(rest[~easy[rest]])])
            faithful[order[:n_need]] = True
            plr = dev_lr.copy(); S = dev_S.copy()
            for i in np.flatnonzero(~faithful):
                pool = by_qc.get((qid[i], s1[i])) or by_c.get(s1[i]) or by_q.get(qid[i]) or list(range(len(U_ids)))
                j = pool[rng.integers(len(pool))]; plr[i] = U_oof[j]; S[i] = U_S[j]
            r, cf = fuse_crossfit(plr, S, seeds=1)
            outs.append((r, cf))
        cf = np.mean([o[1] for o in outs], 0)
        key = f'{mode}_{target}'
        res[key] = {'faithful_frac': round(float(faithful.mean()), 4) if target != 'base' else round(float(agree.mean()), 4),
                    'faithful_frac_unseen106_last_seed': round(float(faithful[unseen].mean()), 4),
                    'G1a_stage_top1_agree_claude': round(float(np.mean([o[0]['stage_top1_agree_claude'] for o in outs])), 4),
                    'stage_single': round(float(np.mean([o[0]['stage_single_dw20'] for o in outs])), 4),
                    'fused_crossfit': round(float(cf.mean()), 4), 'delta_vs_0.7182': round(float(cf.mean()-sb.mean()), 4), 'ci': ci_(cf, sb),
                    'G1b_unseen106_fused': round(float(cf[unseen].mean()), 4), 'bins': pb(cf)}
        print(key, json.dumps(res[key]), flush=True)
        if target == 'base' and mode == 'adversarial_easy_first': pass
json.dump(res, open(OUT/'t2_fidelity_sim.json', 'w'), indent=1)
