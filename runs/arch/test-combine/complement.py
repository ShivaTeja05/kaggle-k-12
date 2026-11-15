"""Supplement: (1) 'complementary agreement' = how often a proxy student picks Claude's top-1 on the rows where the
blend's top-1 differs from Claude's (the only rows a verifier can move); (2) increment over Stage 0 (Sinkhorn alone),
paired question bootstrap; (3) what complementary agreement an oracle at r has."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-combine')
import runpy
g = runpy.run_path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-combine/cpu_test.py', run_name='lib')
np = g['np']; P, C, c1, y, yi, up, qid, ref, n = (g[k] for k in ['P', 'C', 'c1', 'y', 'yi', 'up', 'qid', 'ref', 'n'])
sc, restrict, TOP10, fuse_pts, crossfit, boot_ci, aligned_bundle, ROOT, ids, classes, s0 = (g[k] for k in
    ['sc', 'restrict', 'TOP10', 'fuse_pts', 'crossfit', 'boot_ci', 'aligned_bundle', 'ROOT', 'ids', 'classes', 's0_out'])
dis = P.argmax(1) != c1; cl_right = c1 == yi
out = {'rows_blend_ne_claude': int(dis.sum()), 'of_which_claude_correct_top1': int((dis & cl_right).sum()),
       'of_which_blend_correct_top1': int((dis & (P.argmax(1) == yi)).sum())}
for name, path in g['proxies'].items():
    V = C if path is None else aligned_bundle(ROOT/path, ids, classes)
    V10 = restrict(V, TOP10[.5]); v1 = V10.argmax(1)
    o, _ = crossfit(fuse_pts(V))
    d, c = boot_ci(o, s0, qid)
    out[name] = dict(alone_in_top10=float(sc(V10).mean()), alone_in_top10_unseen_pair=float(sc(V10)[up].mean()),
                     comp_agree=float(np.mean(v1[dis] == c1[dis])),
                     comp_agree_where_claude_right=float(np.mean(v1[dis & cl_right] == c1[dis & cl_right])),
                     picks_truth_where_blend_wrong_claude_right=float(np.mean(v1[dis & cl_right] == yi[dis & cl_right])),
                     fused_vs_stage0=dict(delta=d, ci95=c))
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT/'runs/arch/test-combine/complement.json', 'w'), indent=1)
