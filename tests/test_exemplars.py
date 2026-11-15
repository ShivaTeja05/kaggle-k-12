"""Example index and cards (ARCHITECTURE_REFORM_2026.md 4.4 and the Exemplars row of section 6).

Synthetic tests always run. Tests on the competition data skip when the data or the shortlist file is absent.
Run: PYTHONPATH=src:scripts python -m unittest tests/test_exemplars.py
"""
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from k12.exemplars import ExemplarIndex, Tfidf, cosine, question_key, render_card, root_line, same_question

DATA = ROOT / 'data/bundle/x/competition_data'
SHORTLIST = ROOT / 'data/v10/shortlist_dev.json'
MARKERS = ('CORRECT', 'true code', 'gold', 'the answer is', 'right code')


def row(trace, question, code, text='why', step='L1', span='x', working='L1 | a = b\nL2 | c', topic='T',
        question_text=None):
    findings = [] if code == 'NONE' else [{'step': step, 'span': span, 'category': code, 'severity': 'major', 'root': True, 'text': text},
                                          {'step': 'L2', 'span': 'c', 'category': 'CAL-01', 'severity': 'minor', 'root': False, 'text': 'DOWNSTREAM'}]
    return {'trace_id': trace, 'question_id': question, 'question_text': question_text or f'question about {question}',
            'topic_name': topic, 'root_code': code, 'transcription_steps': working, 'misconceptions': json.dumps(findings)}


def frame(*rows):
    return pd.DataFrame(list(rows))


class Query:
    """A prompt row that has no label: reading root_code or misconceptions from it is an error."""

    def __init__(self, trace_id, question_id, question_text, topic_name='T'):
        self.trace_id, self.question_id, self.question_text, self.topic_name = trace_id, question_id, question_text, topic_name


class IndexContents(unittest.TestCase):
    def test_entries_hold_the_root_finding_only(self):
        index = ExemplarIndex(frame(row('a', 'q1', 'ALG-06', text='moved  the term', step='L1-L3', span=''),
                                    row('n', 'q2', 'NONE'), {**row('b', 'q3', 'CAL-01'), 'misconceptions': '[]'}))
        self.assertEqual([e['trace_id'] for e in index.entries], ['a'])
        e = index.entries[0]
        self.assertEqual((e['code'], e['step'], e['span'], e['explanation'], e['question_id'], e['topic']),
                         ('ALG-06', 'L1-L3', '', 'moved the term', 'q1', 'T'))
        self.assertNotIn('DOWNSTREAM', json.dumps(e))
        self.assertEqual([e['trace_id'] for e in index.nones], ['n'])
        self.assertEqual(index.skipped, ['b'])  # an error row with no root finding has nothing to show
        self.assertEqual(index.trace_ids(), {'a', 'n'})

    def test_tfidf_cosine(self):
        t = Tfidf(['sign flipped across equals', 'sign flipped across equals', 'area and perimeter'])
        X = t.vectors(['sign flipped across equals', 'area and perimeter', 'unknown words only'])
        self.assertAlmostEqual(float(cosine(X[:1], X[0])[0]), 1.)
        self.assertAlmostEqual(float(cosine(X[:1], X[1])[0]), 0.)
        self.assertEqual(float(cosine(X[2:], X[0])[0]), 0.)  # words outside the vocabulary give an empty vector, not NaN
        np.testing.assert_allclose(np.linalg.norm(X[:2], axis=1), 1.)


class Selection(unittest.TestCase):
    def setUp(self):
        self.rows = [row('r', 'q1', 'ALG-06', 'sign not flipped when moving a term'),
                     row('s', 'q1', 'ALG-06', 'sign not flipped when moving a term across'),
                     row('o1', 'q2', 'ALG-06', 'sign not flipped when moving a term'),
                     row('o2', 'q3', 'ALG-06', 'completely different wording here'),
                     row('only', 'q1', 'FRA-01', 'added tops and bottoms'),
                     row('lone', 'q9', 'EXP-01', 'power as product'),
                     row('n1', 'q1', 'NONE'), row('n2', 'q2', 'NONE', topic='T'), row('n3', 'q3', 'NONE', topic='U')]
        self.index = ExemplarIndex(frame(*self.rows))

    def test_a_row_is_never_its_own_example_in_any_mode(self):
        for mode in ('cross', 'same'):
            for r in self.rows:
                if r['root_code'] == 'NONE':
                    continue
                got = self.index.select(r['root_code'], r['trace_id'], r['question_id'], r['question_text'], mode)
                self.assertTrue(got is None or got['trace_id'] != r['trace_id'], (mode, r['trace_id']))
        # the only example of its code: nothing to show, in either mode
        self.assertIsNone(self.index.select('EXP-01', 'lone', 'q9', 'question about q9', 'cross'))
        self.assertIsNone(self.index.select('EXP-01', 'lone', 'q9', 'question about q9', 'same'))
        for r in self.rows:
            got = self.index.none_example(r['trace_id'], r['question_id'], r['topic_name'])
            self.assertTrue(got is None or got['trace_id'] != r['trace_id'])
        # the trace_id filter holds on its own, not only because a row sits on its own question
        lone = ExemplarIndex(frame(row('n1', 'q1', 'NONE')))
        self.assertIsNone(lone.none_example('n1', 'some other question id', 'T'))
        self.assertEqual(lone.none_example('dev', 'some other question id', 'T')['trace_id'], 'n1')

    def test_cross_mode_never_uses_the_rows_own_question(self):
        shown = 0
        for mu in (.5, 0, 5):  # mu = 0 is the required comparison of 4.4.2 and takes another branch of select
            for r in self.rows:
                for code in ('ALG-06', 'FRA-01', 'EXP-01'):
                    got = self.index.select(code, r['trace_id'], r['question_id'], r['question_text'], 'cross', mu)
                    if got is not None:
                        shown += 1
                        self.assertNotEqual(got['question_id'], r['question_id'])
                        self.assertFalse(same_question(got, r['question_id'], r['question_text']))
        self.assertGreater(shown, 20)
        # with nothing but own-question examples there is no card, at any mu
        for mu in (.5, 0):
            self.assertIsNone(self.index.select('FRA-01', 'dev', 'q1', 'question about q1', 'cross', mu))

    def test_same_wording_under_another_id_counts_as_the_same_question(self):
        index = ExemplarIndex(frame(row('a', 'q1', 'ALG-06', question_text='Solve  2x = 4'), row('b', 'q2', 'ALG-06', question_text='solve 2x = 4')))
        self.assertIsNone(index.select('ALG-06', 'new', 'q7', 'Solve 2x = 4', 'cross'))
        self.assertEqual(index.select('ALG-06', 'new', 'q7', 'Solve 2x = 4', 'same')['trace_id'], 'a')
        self.assertEqual(question_key('Solve  2x = 4'), question_key('solve 2x = 4'))

    def test_none_when_no_example_on_another_question(self):
        self.assertIsNone(self.index.select('FRA-01', 'dev', 'q1', 'question about q1', 'cross'))
        self.assertIsNone(self.index.select('STA-01', 'dev', 'q1', 'question about q1', 'cross'))  # code with no example at all
        self.assertIsNone(self.index.select('STA-01', 'dev', 'q1', 'question about q1', 'same'))

    def test_same_mode_prefers_the_question_and_falls_back_to_cross(self):
        self.assertEqual(self.index.select('FRA-01', 'dev', 'q1', 'question about q1', 'same')['trace_id'], 'only')
        got = self.index.select('ALG-06', 'r', 'q1', 'question about q1', 'same')
        self.assertEqual(got['trace_id'], 's')  # same question, and not the row itself
        self.assertEqual(self.index.select('ALG-06', 'dev', 'q5', 'question about q5', 'same'),
                         self.index.select('ALG-06', 'dev', 'q5', 'question about q5', 'cross'))
        self.assertIsNone(self.index.select('FRA-01', 'only', 'q1', 'question about q1', 'same'))
        with self.assertRaises(ValueError):
            self.index.select('ALG-06', 'dev', 'q5', 'x', 'nearest')

    def test_cross_mode_takes_the_most_typical_explanation(self):
        self.assertEqual(self.index.select('ALG-06', 'dev', 'q5', 'something else', 'cross', mu=0)['trace_id'], 'o1')

    def test_typical_means_typical_of_the_code_not_of_all_examples(self):
        # Most ALG-06 explanations talk about a sign; one talks about area, which is how every FRA-01 row is worded.
        # The mean must be taken over the code's own examples, so the odd one out is not chosen.
        rows = [row('a1', 'q2', 'ALG-06', 'sign not flipped when moving the term'),
                row('a2', 'q3', 'ALG-06', 'sign not flipped when moving the term across'),
                row('a3', 'q4', 'ALG-06', 'area of the rectangle')]
        rows += [row(f'f{n}', f'q{10 + n}', 'FRA-01', 'area of the rectangle') for n in range(8)]
        index = ExemplarIndex(frame(*rows))
        self.assertIn(index.select('ALG-06', 'dev', 'q1', 'unrelated', 'cross', mu=0)['trace_id'], ('a1', 'a2'))
        # and the row itself is left out of that mean: without a1, a2 is the typical one
        self.assertEqual(index.select('ALG-06', 'a1', 'q2', 'question about q2', 'cross', mu=0)['trace_id'], 'a2')

    def test_the_penalty_moves_the_choice_away_from_a_look_alike_question(self):
        index = ExemplarIndex(frame(
            row('near', 'q2', 'ALG-06', 'sign not flipped', question_text='a shop sells pens and books for rupiah'),
            row('near2', 'q3', 'ALG-06', 'sign not flipped', question_text='a shop sells pens and books for rupiah today'),
            row('far', 'q4', 'ALG-06', 'sign not flipped at all', question_text='the angle of a triangle')))
        ask = 'a shop sells pens and books for rupiah each'
        self.assertIn(index.select('ALG-06', 'dev', 'q1', ask, 'cross', mu=0)['trace_id'], ('near', 'near2'))
        self.assertEqual(index.select('ALG-06', 'dev', 'q1', ask, 'cross', mu=5)['trace_id'], 'far')

    def test_ties_go_to_the_shorter_working_then_the_trace_id(self):
        index = ExemplarIndex(frame(row('c', 'q2', 'ALG-06', 'same words', working='L1 | long long long\nL2 | more'),
                                    row('b', 'q3', 'ALG-06', 'same words', working='L1 | ab'),
                                    row('a', 'q4', 'ALG-06', 'same words', working='L1 | cd')))
        self.assertEqual(index.select('ALG-06', 'dev', 'q1', 'unrelated', 'cross', mu=0)['trace_id'], 'a')

    def test_correct_example_comes_from_another_question_same_topic_first(self):
        got = self.index.none_example('dev', 'q1', 'T')
        self.assertEqual(got['trace_id'], 'n2')  # n1 is on the row's question, n3 is another topic
        self.assertEqual(self.index.none_example('dev', 'q2', 'U')['trace_id'], 'n3')
        self.assertEqual(self.index.none_example('dev', 'q2', 'no such topic')['trace_id'], self.index.none_example('dev', 'q2', 'zzz')['trace_id'])
        self.assertIsNone(ExemplarIndex(frame(row('n', 'q1', 'NONE'), row('a', 'q2', 'ALG-06'))).none_example('dev', 'q1', 'T'))

    def test_output_is_deterministic_and_independent_of_row_order(self):
        again = ExemplarIndex(frame(*reversed(self.rows)))
        for mode in ('cross', 'same'):
            for r in self.rows:
                for code in ('ALG-06', 'FRA-01'):
                    args = (code, r['trace_id'], r['question_id'], r['question_text'], mode)
                    a, b = self.index.select(*args), again.select(*args)
                    self.assertEqual(a, b)
                    self.assertEqual(render_card(code, 'n', 'd', a), render_card(code, 'n', 'd', b))
                    self.assertEqual(self.index.select(*args), a)
        self.assertEqual(self.index.none_example('dev', 'q1', 'T'), again.none_example('dev', 'q1', 'T'))


class Cards(unittest.TestCase):
    def entry(self, **changes):
        base = {'trace_id': 'a', 'question_id': 'q2', 'question_text': 'Solve\n2x = 4', 'topic': 'T', 'code': 'ALG-06',
                'working': 'L1 | 2x = 4\nL2 | x = 4 - 2\nL3 | x = 2', 'step': 'L2', 'span': 'x = 4 - 2', 'explanation': 'Subtracted instead of dividing.'}
        return dict(base, **changes)

    def test_card_layout(self):
        card = render_card('ALG-06', 'Moving terms', 'Terms moved without flipping.', self.entry())
        self.assertEqual(card.splitlines(), [
            'ALG-06 | Moving terms | Terms moved without flipping.',
            'Example from another student (different question):',
            '  Question: Solve 2x = 4',
            '  Working:  L1 | 2x = 4   L2 | x = 4 - 2   L3 | x = 2',
            '  Root line: L2 | x = 4 - 2',
            '  Why this is ALG-06: Subtracted instead of dividing.'])
        self.assertIn('(same question)', render_card('ALG-06', 'n', 'd', self.entry(), same_question=True))

    def test_description_only_when_there_is_no_example(self):
        self.assertEqual(render_card('STA-01', 'Mean formula', 'Mean misapplied.', None), 'STA-01 | Mean formula | Mean misapplied.')

    def test_range_step_and_empty_span_render(self):
        self.assertEqual((root_line('L1-L3'), root_line('L12'), root_line(' L2 – L4 '), root_line('whole page')), ((1, 3), (12, 12), (2, 4), None))
        ranged = render_card('ALG-06', 'n', 'd', self.entry(step='L1-L3', span='2x'))
        self.assertIn('  Root line: L1-L3 | 2x\n', ranged)
        whole = render_card('ALG-06', 'n', 'd', self.entry(step='L2', span=''))
        self.assertIn('  Root line: L2 | x = 4 - 2\n', whole)  # the whole line when the teacher marked no span
        both = render_card('ALG-06', 'n', 'd', self.entry(step='L2-L3', span=''))
        self.assertIn('  Root line: L2-L3 | x = 4 - 2 ; x = 2\n', both)
        missing = render_card('ALG-06', 'n', 'd', self.entry(step='L9', span=''))
        self.assertIn('  Root line: L9\n', missing)  # no such line in the working: the label alone, no crash

    def test_correct_working_card(self):
        card = render_card('NONE', 'No major error', 'Correct.', self.entry(code='NONE', step='', span='', explanation=''))
        self.assertEqual(card.splitlines()[-2:], ['  Root line: none', '  Why this is NONE: no major error'])

    def test_card_text_cannot_mark_the_true_code(self):
        # Nothing about the row being diagnosed reaches the card: the renderer takes no label, and the request
        # builder builds cards from a row object that has no label at all.
        self.assertEqual(list(inspect.signature(render_card).parameters), ['code', 'name', 'description', 'entry', 'same_question', 'why_words'])
        self.assertEqual(list(inspect.signature(ExemplarIndex.select).parameters),
                         ['self', 'code', 'row_trace_id', 'question_id', 'question_text', 'mode', 'mu'])
        from build_vlm_requests_v2 import cards_for, row_block
        index = ExemplarIndex(frame(row('a', 'q2', 'ALG-06'), row('b', 'q3', 'FRA-01'), row('n', 'q4', 'NONE')))
        names = {'ALG-06': 'Moving terms', 'FRA-01': 'Adding tops', 'STA-01': 'Mean', 'NONE': 'No major error'}
        query = Query('dev', 'q1', 'question about q1')
        for mode in ('none', 'cross', 'same'):
            cards, sources = cards_for(query, ['ALG-06', 'FRA-01', 'NONE'], index, names, names, mode)
            block, _ = row_block(query, ['ALG-06', 'FRA-01', 'NONE'], index, names, names, mode)
            for text in list(cards.values()) + [block]:
                for marker in MARKERS:
                    self.assertNotIn(marker.lower(), text.lower())
            # every shortlisted code is rendered the same way, so the layout cannot single one out
            self.assertEqual(len({c.count('\n') for c in cards.values()}), 1)
            self.assertEqual(set(sources), {'ALG-06', 'FRA-01', 'NONE'})


    def test_builder_says_where_each_example_came_from(self):
        from build_vlm_requests_v2 import cards_for
        index = ExemplarIndex(frame(row('a', 'q1', 'ALG-06'), row('b', 'q2', 'ALG-06'), row('n', 'q4', 'NONE')))
        names = {'ALG-06': 'Moving terms', 'STA-01': 'Mean', 'NONE': 'No major error'}
        query, codes = Query('dev', 'q1', 'question about q1'), ['ALG-06', 'STA-01', 'NONE']
        cards, sources = cards_for(query, codes, index, names, names, 'same')
        self.assertEqual(sources, {'ALG-06': {'trace_id': 'a', 'same_question': True}, 'STA-01': None,
                                   'NONE': {'trace_id': 'n', 'same_question': False}})
        self.assertIn('Example from another student (same question):', cards['ALG-06'])
        self.assertIn('Example from another student (different question):', cards['NONE'])
        self.assertEqual(cards['STA-01'], 'STA-01 | Mean | Mean')  # no example anywhere: the description only
        cards, sources = cards_for(query, codes, index, names, names, 'cross')
        self.assertEqual(sources['ALG-06'], {'trace_id': 'b', 'same_question': False})
        self.assertIn('(different question)', cards['ALG-06'])
        self.assertNotIn('same question', ''.join(cards.values()))
        cards, sources = cards_for(query, codes, index, names, names, 'none')
        self.assertEqual(set(sources.values()), {None})
        self.assertFalse(any('\n' in card for card in cards.values()))


@unittest.skipUnless((DATA / 'train.csv').exists(), 'competition data is not on this machine')
class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from k12.core import make_folds, read_csv
        train = read_csv(DATA / 'train.csv')
        folds = make_folds(train)
        cls.dev, cls.fit = train.loc[folds == 0].reset_index(drop=True), train.loc[folds != 0].reset_index(drop=True)
        cls.taxonomy = read_csv(DATA / 'taxonomy_v1.csv')
        cls.index = ExemplarIndex(cls.fit)

    def test_no_development_or_test_row_in_an_index_built_from_fit_rows(self):
        from k12.core import read_csv
        ids = self.index.trace_ids()
        self.assertEqual(ids, set(self.fit.trace_id))
        self.assertFalse(ids & set(self.dev.trace_id))
        self.assertFalse(ids & set(read_csv(DATA / 'test.csv').trace_id))
        self.assertEqual((len(self.index.entries), len(self.index.skipped)), (int((self.fit.root_code != 'NONE').sum()), 0))

    def test_measured_facts_of_section_4_4(self):
        entries = self.index.entries
        self.assertEqual(len(entries), 794)
        self.assertEqual(sum(bool(e['span']) for e in entries), 792)
        self.assertEqual(sum(root_line(e['step'])[0] != root_line(e['step'])[1] for e in entries), 121)
        classes = set(self.taxonomy.id) - {'NONE'}
        self.assertEqual(len(classes - set(self.index.by_code)), 7)  # codes with no fit example
        self.assertEqual(sum(len({entries[j]['question_id'] for j in rows}) == 1 for rows in self.index.by_code.values()), 11)

    @unittest.skipUnless(SHORTLIST.exists(), 'run scripts/build_shortlist.py first')
    def test_every_shortlisted_code_with_an_example_elsewhere_gets_exactly_one_card(self):
        from build_vlm_requests_v2 import code_texts, row_block
        names, descriptions = code_texts(self.taxonomy)
        shortlist = json.loads(SHORTLIST.read_text())
        fit_key = self.fit.question_text.map(question_key)
        fit_ids, dev_ids, checked, with_card = set(self.fit.trace_id), set(self.dev.trace_id), 0, 0
        fit_question = dict(zip(self.fit.trace_id, self.fit.question_id))
        fit_text, same = dict(zip(self.fit.trace_id, fit_key)), 0
        fit_code = dict(zip(self.fit.trace_id, self.fit.root_code))
        for r in self.dev.itertuples(index=False):
            if not shortlist[r.trace_id]['sent']:
                continue
            query = Query(r.trace_id, r.question_id, r.question_text, r.topic_name)  # no label on the prompt row
            codes = shortlist[r.trace_id]['codes']
            elsewhere = (self.fit.question_id != r.question_id) & (fit_key != question_key(r.question_text))
            for mode in ('cross', 'same'):
                block, sources = row_block(query, codes, self.index, names, descriptions, mode)
                self.assertEqual(block.count('Example from another student'), sum(s is not None for s in sources.values()))
                for code in codes:
                    self.assertEqual(block.count(f'] {code} | '), 1)
                    source = sources[code]
                    if source is not None:
                        self.assertIn(source['trace_id'], fit_ids)
                        self.assertNotIn(source['trace_id'], dev_ids)
                        self.assertNotEqual(source['trace_id'], r.trace_id)
                    if code == 'NONE':  # the correct working is from another question in both modes
                        self.assertIsNotNone(source)
                        self.assertFalse(source['same_question'])
                        self.assertNotEqual(fit_question[source['trace_id']], r.question_id)
                        self.assertNotEqual(fit_text[source['trace_id']], question_key(r.question_text))
                        self.assertEqual(fit_code[source['trace_id']], 'NONE')
                        continue
                    if source is not None:
                        self.assertEqual(fit_code[source['trace_id']], code)  # the example really carries the card's code
                    expected = bool(((self.fit.root_code == code) & elsewhere).any())
                    if mode == 'cross':
                        self.assertEqual(source is not None, expected, (r.trace_id, code))
                        self.assertEqual(block.count(f'Why this is {code}:'), int(expected))
                        if source is not None:
                            self.assertFalse(source['same_question'])
                            self.assertNotEqual(fit_question[source['trace_id']], r.question_id)
                        checked += 1; with_card += expected
                    else:  # same-question mode shows an example whenever the code has any fit example ...
                        self.assertEqual(source is not None, bool((self.fit.root_code == code).any()))
                        here = bool(((self.fit.root_code == code) & ~elsewhere).any())
                        if source is not None:  # ... from the row's own question exactly when the fit rows have one there
                            self.assertEqual(source['same_question'], here, (r.trace_id, code))
                            self.assertEqual(fit_question[source['trace_id']] == r.question_id
                                             or fit_text[source['trace_id']] == question_key(r.question_text), here)
                        same += here
                if mode == 'same':
                    self.assertEqual(block.count('(same question)'), sum(bool(s and s['same_question']) for s in sources.values()))
                else:
                    self.assertNotIn('(same question)', block)
        self.assertGreater(checked, 1000)
        self.assertGreater(with_card, 0); self.assertLess(with_card, checked)  # both kinds occur, so the test is not vacuous
        self.assertGreater(same, 100)  # same-question examples do occur


if __name__ == '__main__':
    unittest.main()
