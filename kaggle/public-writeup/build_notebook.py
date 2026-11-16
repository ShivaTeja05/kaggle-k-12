"""Build the public Kaggle notebook (method, runnable fusion code, a CPU baseline, negative results).

The notebook uses only train labels and test metadata, prints aggregates only (no student work, no photos), and needs
no GPU and no internet. Usage: python kaggle/public-writeup/build_notebook.py  ->  kaggle/public-writeup/notebook.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = 'https://github.com/ShivaTeja05/kaggle-k-12'
cells = []
md = lambda text: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')})
code = lambda text: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n')})

md(f"""
# K-12 misconception detection: method, runnable fusion code and negative results

**Public MAP@3 0.76383** (about rank 65 of 481 at the time of writing). This is not a top solution; it is a complete,
honest record of one: what moved the score, what did not, and the code for the parts that run on a CPU.

Full code (training, inference, launchers, 96 unit tests, every measured result): **{REPO}** (Apache 2.0).

What this notebook contains:

1. The system behind the score, and the public score after each change.
2. Runnable code: MAP@3, label-free Sinkhorn balancing, and the rule that fuses a reasoning model's top-3 answer into a
   classifier's probabilities.
3. A CPU baseline from question and topic priors, cross-validated here, which writes a valid `submission.csv`.
4. What did not work, with measured numbers.

Nothing here prints student work or photos; only label counts are shown.
""")

md("""
## 1. The system

```
photo + question + answer key
   |-- transcriber (Qwen3-VL-8B LoRA) --> text reasoner Qwen3-8B LoRA ----\\
   |                                  \\-> text reasoner Qwen3-32B QLoRA ---+--> geometric blend
   |-- vision classifier (Qwen3-VL-8B, distilled from frontier-model labels on unlabelled photos) --/
                                                                              |
                         pointwise yes/no ranker (Qwen3-14B) over (row, candidate code) pairs
                                                                              |
              top probability below 0.817?  -- yes -->  Qwen3.8-27B (FP8, vLLM), zero-shot with thinking
                                                                              |
                               fusion of its top-3 answer + Sinkhorn label balancing --> top-3 codes
```

All final inference is offline with open-weight models. A frontier API labelled training and unlabelled photos only,
never test photos, which the host allows for training data.

| Date (2026) | Public MAP@3 | What changed |
|---|---|---|
| 8 Nov | 0.58216 | Qwen3-VL-4B LoRA image classifier |
| 8 Nov | 0.59800 | Temperature-softened classifier mixed with a question/topic prior |
| 11 Nov | 0.62383 | Qwen3-VL-8B trained with teacher-transcription supervision |
| 11 Nov | 0.66183 | **Read, then diagnose**: a vision transcriber feeds a fine-tuned text reasoner |
| 11 Nov | 0.68283 | Vision model distilled from frontier-model labels on unlabelled photos |
| 11 Nov | 0.70850 | Three-model geometric ensemble with a Qwen3-32B QLoRA reasoner |
| 12 Nov | 0.71716 | Label-free Sinkhorn balancing of the predicted label mix |
| 13 Nov | 0.73183 | Fused with a Qwen3-14B pointwise yes/no ranker |
| 14 Nov | 0.76266 | Qwen3.8-27B with thinking on the least-confident test photos |
| 15 Nov | **0.76383** | Same, all 1,593 least-confident photos answered |

Three things carried most of the gain:

- **Read, then diagnose.** Train the text reasoner on *machine* transcripts, because the teachers' typed transcripts do
  not exist at test time.
- **An untrained reasoning model on the uncertain rows only.** Qwen3.8-27B zero-shot with thinking was the strongest
  single reader on question-code pairs never seen in training, where every trained model was weak.
- **Sinkhorn balancing.** Nudging the predicted label mix toward the training label mix, with no test labels.
""")

md("""
## 2. Core code: metric, Sinkhorn balancing, fusing a reasoning model's answer

These are the functions used for the submitted file (`scripts/fuse_candidate.py`, `scripts/fuse_vlm.py` and
`src/k12/core.py` in the repository), made self-contained.
""")

code('''
import glob, os
import numpy as np
import pandas as pd


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def topk(p, classes, k=3):
    return np.asarray(classes)[np.argsort(-normalize(p), axis=1, kind='stable')[:, :k]]


def row_scores(y, predictions):
    """MAP@3 per row with one true label: 1, 1/2 or 1/3 for rank 1, 2 or 3, else 0."""
    out = []
    for target, codes in zip(y, predictions):
        hits = [1.0 / (i + 1) for i, c in enumerate(list(codes)[:3]) if c == target]
        out.append(hits[0] if hits else 0.0)
    return np.asarray(out)


def marginals(codes, classes):
    """Smoothed label mix of the training set: the target of the Sinkhorn step."""
    index = {c: i for i, c in enumerate(classes)}
    pi = np.bincount(pd.Series(codes).map(index), minlength=len(classes)) + .5
    return pi / pi.sum()


def sinkhorn(P, target, lam, iters=50):
    """Label-free balancing: find per-class offsets b so the mean prediction matches `target`, apply a share `lam`."""
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))


SHARES = (.6, .25, .15)  # weight of a reader's first, second and third code


def rank_shares(answers, classes):
    """Rows x classes matrix from a reader's ranked codes; a row without an answer stays all zero."""
    index = {c: i for i, c in enumerate(classes)}
    S = np.zeros((len(answers), len(classes)))
    for row, codes in enumerate(answers):
        seen = set()
        for share, c in zip(SHARES, codes or ()):
            if c in index and c not in seen:
                S[row, index[c]] += share; seen.add(c)
    return S


def fuse_reader(P, S, threshold=.817, floor=.01, weight=1., target=None, lam=.5):
    """Multiply the classifier's probabilities by the reader's rank shares, only on rows where the classifier is unsure
    (top probability below `threshold`) and the reader answered; then balance the label mix."""
    sent = (P.max(1) < threshold) & (S.sum(1) > 0)
    Q = np.where(sent[:, None], normalize(S + floor) ** weight, 1.)
    F = normalize(P * Q)
    return (sinkhorn(F, target, lam) if target is not None else F), sent
''')

md("""
## 3. A CPU baseline: question and topic priors

About 96% of test rows are on a question that also appears in train, so "which codes did teachers give other students on
this question?" is a strong prior by itself. The model below backs off from question to topic to the global label mix.
It uses only columns that exist in `test.csv`.
""")

code('''
def find(name):
    root = os.environ.get('K12_DATA', '/kaggle/input')
    hits = sorted(glob.glob(os.path.join(root, '**', name), recursive=True), key=lambda p: ('competition_data' not in p, len(p)))
    if not hits:
        raise FileNotFoundError(f'{name} not found under {root}: attach the competition data to this notebook')
    return hits[0]


train = pd.read_csv(find('train.csv'), dtype=str, keep_default_na=False)
test = pd.read_csv(find('test.csv'), dtype=str, keep_default_na=False)
taxonomy = pd.read_csv(find('taxonomy_v1.csv'), dtype=str, keep_default_na=False)
classes = sorted(set(taxonomy['id']) | {'NONE'})
y = train.root_code.to_numpy()
print(f'train rows {len(train)}, test rows {len(test)}, classes {len(classes)}')
print(f"rows with no major error (NONE): {(y == 'NONE').mean():.1%}")
print(f'codes that occur once in train: {(train.root_code.value_counts() == 1).sum()}')
print(f'test rows on a question seen in train: {test.question_id.isin(set(train.question_id)).mean():.1%}')
''')

code('''
class HierarchicalPrior:
    """P(code | question) backed off to P(code | topic), backed off to the global label mix."""

    def __init__(self, classes, question_strength=2.0, topic_strength=8.0):
        self.classes, self.a, self.b = list(classes), question_strength, topic_strength

    def fit(self, df):
        index = {c: i for i, c in enumerate(self.classes)}; k = len(index)
        self.global_p = np.bincount(df.root_code.map(index), minlength=k) + .5
        self.global_p = self.global_p / self.global_p.sum()
        self.topic, self.question = {}, {}
        for col, store in (('topic_name', self.topic), ('question_id', self.question)):
            for key, g in df.groupby(col, sort=False):
                store[str(key)] = np.bincount(g.root_code.map(index), minlength=k).astype(float)
        return self

    def predict_proba(self, df):
        zero = np.zeros(len(self.classes)); out = []
        for r in df.itertuples(index=False):
            t = self.topic.get(str(r.topic_name), zero)
            pt = (t + self.b * self.global_p) / (t.sum() + self.b)
            q = self.question.get(str(r.question_id), zero)
            out.append((q + self.a * pt) / (q.sum() + self.a))
        return np.asarray(out)


def cross_validate(groups=None, folds=5, seed=2026):
    """Out-of-fold predictions. groups=None: rows shuffled; groups=question ids: whole questions held out."""
    rng = np.random.default_rng(seed)
    if groups is None:
        fold = rng.permutation(len(train)) % folds
    else:
        keys = pd.unique(groups); order = rng.permutation(len(keys))
        fold = pd.Series(groups).map({k: i % folds for i, k in zip(order, keys)}).to_numpy()
    P = np.zeros((len(train), len(classes)))
    for f in range(folds):
        model = HierarchicalPrior(classes).fit(train[fold != f])
        P[fold == f] = model.predict_proba(train[fold == f])
    return P


P_rows = cross_validate()
P_questions = cross_validate(train.question_id.to_numpy())
print(f'prior baseline, MAP@3, rows held out at random:   {row_scores(y, topk(P_rows, classes)).mean():.4f}')
print(f'prior baseline, MAP@3, whole questions held out:  {row_scores(y, topk(P_questions, classes)).mean():.4f}')
print('The gap between the two lines is the value of having seen the question before.')
''')

code('''
prior = HierarchicalPrior(classes).fit(train)
P_test = prior.predict_proba(test)
submission = pd.DataFrame({'trace_id': test.trace_id, 'root_code': [' '.join(row) for row in topk(P_test, classes)]})
submission.to_csv('submission.csv', index=False)
print(len(submission), 'rows written to submission.csv (the prior baseline, not the 0.764 system)')
''')

md("""
## 4. How the reasoning model's answer is fused (a demonstration on simulated answers)

In the real system the classifier is the trained ensemble and the reader is Qwen3.8-27B, which returns three ranked
codes per photo. Neither can run in this notebook, so the cell below shows the *mechanics* with stand-ins: the
classifier is the out-of-fold prior from section 3, and the reader is **simulated** (it names the true code first on a
fixed share of rows, a wrong code otherwise). The printed numbers illustrate the fusion rule; they are not results of
the real system. Those are in the tables of sections 1 and 5.
""")

code('''
def simulated_reader(y, classes, first=.55, second=.12, third=.06, seed=7):
    """A stand-in reader: the truth is ranked 1st / 2nd / 3rd with the given shares, the other slots are random codes."""
    rng = np.random.default_rng(seed); answers = []
    for target in y:
        wrong = [c for c in rng.choice(classes, 4, replace=False) if c != target][:3]
        u = rng.random()
        slot = 0 if u < first else 1 if u < first + second else 2 if u < first + second + third else None
        if slot is not None:
            wrong[slot] = target
        answers.append(wrong)
    return answers


S = rank_shares(simulated_reader(y, classes), classes)
target = marginals(y, classes)
base = row_scores(y, topk(P_rows, classes))
reader = row_scores(y, topk(S + 1e-4, classes))
for threshold in (.5, .817, 1.01):
    fused, sent = fuse_reader(P_rows, S, threshold=threshold, target=target)
    print(f'threshold {threshold:<5}: reader used on {sent.mean():5.1%} of rows | classifier {base.mean():.3f} | '
          f'simulated reader {reader.mean():.3f} | fused {row_scores(y, topk(fused, classes)).mean():.3f}')
balanced = sinkhorn(P_rows, target, .75)
print(f'Sinkhorn balancing alone on the classifier: {base.mean():.3f} -> {row_scores(y, topk(balanced, classes)).mean():.3f} '
      '(no help on a prior that already matches the label mix; on the trained ensemble it gave +0.009 public)')
''')

md("""
Why multiply rather than replace: the classifier knows the question's history and the reader knows the photo. On the
270 development rows that Qwen3.8-27B answered, the trained ensemble scored 0.775, Qwen3.8-27B alone 0.759, and the
product 0.804.
Running the reader only where the classifier's top probability is below 0.817 covered 46% of the test photos, so the
27B model ran on under half the rows.

Serving notes for Qwen3.8-27B (FP8) with vLLM on one 48 GB GPU, measured:

- 20 to 30 parallel requests. At 40 the KV cache fills and total throughput halves.
- The submitted file used medium reasoning effort. Maximum effort (`xhigh`) wrote about 9,100 tokens a photo, about 82
  photos an hour on an RTX 6000 Ada, for a gain on development of +0.010 that was not statistically confirmed.
- The answer is requested as one JSON object under a schema: transcription, root line, reasoning, three codes.
""")

md(f"""
## 5. What did not work

All figures are MAP@3 on one fixed 301-row development fold; "fused" is the change to the full system. Development ran
about 0.03 above the public leaderboard.

| Idea | Result | Why it failed |
|---|---|---|
| Fine-tuning Qwen3.8-27B (LoRA) with a classification layer on the 1,199 labelled rows | 0.658 alone, +0.000 fused | The layer reads the state before any reasoning happens |
| The same fine-tuned model writing its answer | 0.698 alone, -0.004 fused | Learned the familiar cases the ensemble already covers |
| The same fine-tuned model with thinking switched on | 0.599 | Training without thinking traces collapsed its reasoning |
| Qwen3.5-9B fine-tune on Kaggle T4s | 0.562 alone, -0.013 fused | Weak on unseen question-code pairs |
| A 14B judge reading the 27B model's description of the photo | 0.777 alone, -0.002 fused | Anchored on the first model's top code |
| Continuing the ranker's training on more pseudo-labelled rows | 0.696 alone vs 0.709 before | More of the same labels did not help |
| A 180B open model as a second reader (zero-shot, thinking) | 0.649 vs 0.638 for the 27B on the same rows, fused -0.001 to -0.012 | No stronger than the 27B on this task |
| Nearest-neighbour look-up of similar workings | -0.006 to -0.016 fused | Near zero on unseen pairs |
| Per-question error record as a multiplicative prior | -0.001 to -0.010 fused | The trained models already carry it |
| Re-ranking with the teachers' examples for the question in the prompt | +0.009 at best, interval through zero | Too small to trust |

Where the remaining loss sits on development (61 points lost of 301): right line of working but wrong code 34, wrong
line 11, an error called `NONE` 9, `NONE` called an error 7. Most of the loss is choosing between neighbouring codes,
not reading the handwriting.

Lessons that transfer:

- Write the pass mark before the run and report every gain with a paired bootstrap interval over questions. With 301
  development rows, a gain under about +0.02 is noise.
- Score by row type (no error, seen question-code pair, unseen pair). Every model trained on 1,200 labels was strong on
  seen pairs and weak on unseen ones; only free reasoning helped there.
- A thinking model should be fine-tuned with its thinking traces, or not at all.

Code, plans, the full experiment table and the per-run log: {REPO}
""")

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, c in enumerate(cells):
    c['id'] = f'cell-{i:02d}'
(HERE / 'notebook.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
print('cells', len(cells))
