"""Prompt v2, its answer schema and the shortlist block (ARCHITECTURE_REFORM_2026.md 4.1.2, 4.4.4 and the Prompts
row of section 6), plus the check that the v1 request builder still writes today's bytes.

Synthetic tests always run. Tests on the competition data skip when the data is absent.
Run: PYTHONPATH=src:scripts python -m unittest tests/test_prompts.py
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from k12.prompts import (KEYS_V2, LINE_CHECK_KEYS, answer_schema_v2, key_sentence_v2, shortlist_block, shortlist_order,
                         system_prompt_v2)

DATA = ROOT / 'data/bundle/x/competition_data'
V1_REQUESTS = ROOT / 'data/v9/q38_dev_requests.jsonl'
V1_SHA256 = 'c1ae4272f09a9f31dd80b2665a3290dcecb20d4285045a55e03d49408fdcb36d'
SHORTLIST = ROOT / 'data/v10/shortlist_dev.json'
V2_REQUESTS = ROOT / 'data/v10'
Query = collections.namedtuple('Query', 'trace_id question_id question_text topic_name')  # a prompt row without a label
CLASSES = ['ALG-06', 'CAL-01', 'FRA-01', 'NONE', 'STA-01']
TAXONOMY = pd.DataFrame({'id': ['ALG-06', 'CAL-01'], 'name': ['Moving terms', 'Slip'], 'description': ['Terms moved.', 'A slip.'],
                         'nature': ['procedural', 'procedural'], 'grade_band': ['7', '4'], 'section': ['ALGEBRA', 'NUMBER']})
ANSWER = {'transcription': 'L1 | 2x = 4\nL2 | x = 4 - 2', 'line_checks': [{'line': 'L1', 'left': '2x', 'right': '4', 'equal': True, 'follows': True},
                                                                            {'line': 'L2', 'left': 'x', 'right': '4 - 2', 'equal': False, 'follows': False}],
          'root_line': 'L2', 'root_line_text': 'x = 4 - 2', 'reasoning': 'Subtracted instead of dividing.',
          'first_code': 'ALG-06', 'second_code': 'CAL-01', 'third_code': 'NONE'}


def quoted(sentence):
    return tuple(re.findall(r'"([a-z_]+)"', sentence))


class PromptV2(unittest.TestCase):
    def test_key_sentence_and_schema_agree(self):
        schema, sentences = answer_schema_v2(CLASSES), key_sentence_v2().split('. ')
        expected = ('transcription', 'line_checks', 'root_line', 'root_line_text', 'reasoning', 'first_code', 'second_code', 'third_code')
        self.assertEqual(KEYS_V2, expected)
        self.assertEqual(quoted(sentences[0]), expected)  # the keys the prompt names, in the order it names them
        self.assertEqual(tuple(schema['required']), expected)
        self.assertEqual(tuple(schema['properties']), expected)  # a constrained decoder writes the fields in this order
        checks = schema['properties']['line_checks']['items']
        named = quoted(sentences[1].split(';')[0])
        self.assertEqual(named, ('line_checks',) + LINE_CHECK_KEYS)
        self.assertEqual(tuple(checks['required']), ('line', 'left', 'right', 'equal', 'follows'))
        self.assertEqual(tuple(checks['properties']), tuple(checks['required']))
        keys = list(expected)
        self.assertLess(keys.index('transcription'), keys.index('line_checks'))
        self.assertLess(keys.index('line_checks'), keys.index('root_line'))  # 4.1.2: between transcription and root_line

    def test_schema_has_a_valid_json_schema_shape(self):
        schema = answer_schema_v2(CLASSES)
        self.assertEqual(json.loads(json.dumps(schema)), schema)
        self.assertEqual((schema['type'], schema['additionalProperties']), ('object', False))
        checks = schema['properties']['line_checks']
        self.assertEqual((checks['type'], checks['items']['type'], checks['items']['additionalProperties']), ('array', 'object', False))
        self.assertEqual({k: v['type'] for k, v in checks['items']['properties'].items()},
                         {'line': 'string', 'left': 'string', 'right': 'string', 'equal': 'boolean', 'follows': 'boolean'})
        for key in ('first_code', 'second_code', 'third_code'):
            self.assertEqual(schema['properties'][key], {'type': 'string', 'enum': CLASSES})
        for key in ('transcription', 'root_line_text', 'reasoning'):
            self.assertEqual(schema['properties'][key], {'type': 'string'})
        pattern = re.compile(schema['properties']['root_line']['pattern'])
        self.assertTrue(all(pattern.search(x) for x in ('L1', 'L12', 'none')))
        self.assertFalse(any(pattern.search(x) for x in ('L2 | x = 4 - 2', 'None', 'line 2', 'L', '', 'L1-L3')))
        self.assertEqual(answer_schema_v2(CLASSES, root_line_pattern=False)['properties']['root_line'], {'type': 'string'})

    def test_schema_accepts_a_v2_answer_and_rejects_a_v1_one(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest('jsonschema is not installed')
        schema = answer_schema_v2(CLASSES)
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        self.assertEqual(list(validator.iter_errors(ANSWER)), [])
        self.assertEqual(list(validator.iter_errors(dict(ANSWER, root_line='none', root_line_text='', line_checks=[]))), [])
        v1 = {k: v for k, v in ANSWER.items() if k not in ('line_checks', 'root_line_text')}
        bad = [v1, dict(ANSWER, root_line='L2 | x = 4 - 2'), dict(ANSWER, first_code='XYZ-99'), dict(ANSWER, extra=1),
               dict(ANSWER, line_checks=[{'line': 'L1', 'left': '2x', 'right': '4', 'equal': 'yes', 'follows': True}]),
               dict(ANSWER, line_checks=[{'line': 'L1', 'left': '2x', 'right': '4', 'equal': True}])]
        for answer in bad:
            self.assertTrue(list(validator.iter_errors(answer)))

    def test_v2_text_states_the_three_changes(self):
        text = system_prompt_v2(TAXONOMY)
        for needed in ('line_checks', 'root_line_text', '"L<n>"', '"none"', 'character by character', 'Second look',
                       'look at the photo again', 'ALG-06 | Moving terms | Terms moved. | procedural | grade band 7 | ALGEBRA'):
            self.assertIn(needed, text)
        self.assertLess(text.index('Second look'), text.index('Choose the code'))  # the second look comes before the answer
        self.assertTrue(text.endswith('CAL-01 | Slip | A slip. | procedural | grade band 4 | NUMBER\n'))

    def test_v2_keeps_the_v1_wording(self):
        try:
            from claude_label import system_prompt
        except ImportError as error:
            self.skipTest(f'scripts/claude_label.py cannot be imported here: {error}')
        v1, v2 = system_prompt(TAXONOMY), system_prompt_v2(TAXONOMY)
        head, procedure = v1.split('Procedure:\n')
        self.assertTrue(v2.startswith(head + 'Procedure:\n'))  # role and the definition of a root, word for word
        steps, taxonomy = procedure.split('\n\nTaxonomy')
        one, two, three = steps.split('\n')
        self.assertIn(one + '\n', v2)
        self.assertIn(two[3:].split('. ')[0], v2)  # "Check each line against the question, the answer key and the previous line"
        self.assertIn(two[3:].split('. ')[1], v2)  # "Find the first line where a major error enters, ..."
        self.assertIn(three[3:], v2)
        self.assertTrue(v2.endswith('\n\nTaxonomy' + taxonomy))


class ShortlistBlock(unittest.TestCase):
    codes = ['ALG-06', 'CAL-01', 'FRA-01', 'NONE', 'STA-01', 'EXP-01', 'INT-03', 'WOP-01']
    cards = {c: f'{c} | name of {c} | what {c} means' for c in codes}
    others = {'GEO-01': 'Shape properties', 'PRO-01': 'And added'}

    def listed(self, block):
        return re.findall(r'^\[(\d+)\] ([A-Za-z]+-?\d*) \| ', block, flags=re.M)

    def test_order_differs_across_rows_and_is_stable_for_a_row(self):
        orders = {t: tuple(c for _, c in self.listed(shortlist_block(t, self.codes, self.cards, self.others))) for t in map(str, range(200))}
        self.assertGreater(len(set(orders.values())), 150)
        for t, order in orders.items():
            self.assertEqual(sorted(order), sorted(self.codes))  # every code once
            self.assertEqual(order, tuple(shortlist_order(t, self.codes)))
            self.assertEqual(shortlist_block(t, self.codes, self.cards, self.others), shortlist_block(t, self.codes, self.cards, self.others))
            self.assertEqual(order, tuple(shortlist_order(t, list(reversed(self.codes)))))  # the input order (a ranking) cannot leak
        # a code sits in every position about equally often, so the position says nothing
        position = np.array([[order.index(c) for c in self.codes] for order in orders.values()])
        for column in position.T:
            self.assertEqual(set(column), set(range(len(self.codes))))
            self.assertLess(abs(column.mean() - (len(self.codes) - 1) / 2), .6)

    def test_block_carries_no_scores(self):
        block = shortlist_block('abc', self.codes, self.cards, self.others)
        self.assertEqual([int(n) for n, _ in self.listed(block)], list(range(1, len(self.codes) + 1)))
        self.assertIsNone(re.search(r'\d\.\d|\d\s*%|percent', block))
        for word in ('probab', 'score', 'confiden', 'likel', 'rank', 'weight', 'CORRECT'):
            self.assertNotIn(word.lower(), block.lower())
        self.assertEqual(list(inspect_parameters(shortlist_block)), ['row_trace_id', 'codes', 'cards', 'other_code_names'])

    def test_block_says_the_answer_may_be_outside_and_names_the_other_codes(self):
        block = shortlist_block('abc', self.codes, self.cards, self.others)
        self.assertIn('a code outside the list', block)
        self.assertIn('order carries no information', block)
        tail = block.split('Codes not on the list')[1]
        self.assertIn('GEO-01 (Shape properties); PRO-01 (And added)', tail)
        self.assertFalse(any(c in tail for c in self.codes))
        self.assertTrue(block.endswith('Now diagnose the following student.'))
        self.assertNotIn('with an example', block)  # descriptions only: no sentence about examples
        with_example = dict(self.cards, **{'ALG-06': self.cards['ALG-06'] + '\nExample from another student (different question):\n  Question: q'})
        self.assertIn('says nothing about the next student', shortlist_block('abc', self.codes, with_example, self.others))


def inspect_parameters(function):
    import inspect
    return inspect.signature(function).parameters


class ShortlistBuilder(unittest.TestCase):
    def test_shortlist_is_top_k_plus_zero_shot_plus_none(self):
        try:
            from build_shortlist import shortlists
        except ImportError as error:
            self.skipTest(f'scripts/build_shortlist.py cannot be imported here: {error}')
        classes = ['A', 'B', 'C', 'D', 'NONE']
        SUB = np.array([[.5, .3, .1, .06, .04], [.1, .2, .3, .35, .05], [.04, .06, .1, .3, .5]])
        S = np.array([[0, 0, .25, .6, .15], [.6, .25, .15, 0, 0], [0, 0, 0, 0, 0]])
        top, lists = shortlists(SUB, S, np.array([1, 0, 0]), classes, k=2)
        self.assertEqual([list(t) for t in top], [['A', 'B'], ['D', 'C'], ['NONE', 'D']])
        self.assertEqual(lists, [['A', 'B', 'C', 'D', 'NONE'], ['C', 'D', 'NONE'], ['D', 'NONE']])  # row 2 has no answer on file


@unittest.skipUnless((DATA / 'train.csv').exists(), 'competition data is not on this machine')
class RealData(unittest.TestCase):
    def test_v2_builder_refuses_test_rows(self):
        import build_vlm_requests_v2
        with mock.patch.object(sys, 'argv', ['build_vlm_requests_v2.py', '--split', 'test']):
            with self.assertRaises(SystemExit) as stop:
                build_vlm_requests_v2.main()
        self.assertIn('does not build test requests', str(stop.exception))

    def test_v2_system_prompt_names_every_code(self):
        from k12.core import label_space, read_csv
        taxonomy = read_csv(DATA / 'taxonomy_v1.csv')
        text = system_prompt_v2(taxonomy) + key_sentence_v2()
        self.assertTrue(all(f'\n{code} | ' in text for code in taxonomy.id))
        self.assertEqual(len(answer_schema_v2(label_space(taxonomy))['properties']['first_code']['enum']), 76)

    @unittest.skipUnless(V1_REQUESTS.exists(), 'the v1 dev request file is not on this machine')
    def test_v1_request_builder_still_writes_todays_bytes(self):
        # Runs scripts/build_vlm_requests.py's own main() on the dev fold (about 10 s: 301 photos are re-encoded).
        import build_vlm_requests
        self.assertEqual(hashlib.sha256(V1_REQUESTS.read_bytes()).hexdigest(), V1_SHA256)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'q38_dev_requests.jsonl'
            with mock.patch.object(sys, 'argv', ['build_vlm_requests.py', '--data', str(DATA), '--out', str(out)]), \
                    mock.patch('builtins.print'):
                build_vlm_requests.main()
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), V1_SHA256)


@unittest.skipUnless((DATA / 'train.csv').exists() and SHORTLIST.exists(), 'needs the competition data and scripts/build_shortlist.py output')
class ShortlistFile(unittest.TestCase):
    def test_file_has_no_rank_or_score_and_reproduces_section_4_4_5(self):
        try:
            from fuse_candidate import unseen_pairs
        except ImportError as error:
            self.skipTest(f'scripts/fuse_candidate.py cannot be imported here: {error}')
        from k12.core import make_folds, read_csv
        train = read_csv(DATA / 'train.csv')
        folds = make_folds(train)
        dev, fit = train.loc[folds == 0].reset_index(drop=True), train.loc[folds != 0].reset_index(drop=True)
        shortlist = json.loads(SHORTLIST.read_text())
        self.assertEqual(list(shortlist), dev.trace_id.tolist())
        for entry in shortlist.values():
            self.assertEqual(set(entry), {'sent', 'codes', 'has_zero_shot'})  # nothing a rank or a score could hide in
            self.assertEqual(entry['codes'], sorted(set(entry['codes'])))  # alphabetical, so the order is not a ranking
            self.assertIn('NONE', entry['codes'])
            self.assertTrue(8 <= len(entry['codes']) <= 12)  # top 8, at most 3 more from the zero-shot answer, NONE
            self.assertTrue(entry['has_zero_shot'] or len(entry['codes']) <= 9)
        y, qid = dev.root_code.to_numpy(), dev.question_id.to_numpy()
        new_code, new_question = unseen_pairs(fit, y, qid)
        hard = new_code | new_question
        sent = np.array([shortlist[t]['sent'] for t in dev.trace_id])
        have = np.array([shortlist[t]['has_zero_shot'] for t in dev.trace_id])
        hit = np.array([code in shortlist[t]['codes'] for t, code in zip(dev.trace_id, y)])
        self.assertEqual((len(dev), int(sent.sum()), int(have.sum()), int((sent & have).sum())), (301, 158, 270, 135))
        count = lambda rows: (int(rows.sum()), int(hit[rows].sum()))
        self.assertEqual(count(hard), (106, 94))  # 0.887
        self.assertEqual(count(hard & have), (83, 75))  # 0.904 on the answered hard rows
        self.assertEqual(count(hard & have & sent), (68, 60))  # 0.88 where the shortlist is used
        self.assertEqual(count(have & sent), (135, 127))


@unittest.skipUnless((DATA / 'train.csv').exists() and SHORTLIST.exists(), 'needs the competition data and scripts/build_shortlist.py output')
class Builder(unittest.TestCase):
    """scripts/build_vlm_requests_v2.py's own main(), end to end (about 3 s a run: 135 rows' photos are encoded)."""

    @classmethod
    def setUpClass(cls):
        import build_vlm_requests_v2
        from k12.core import TRAIN_ONLY, make_folds, read_csv
        cls.builder = build_vlm_requests_v2
        cls.train = read_csv(DATA / 'train.csv')
        cls.folds = make_folds(cls.train)
        cls.taxonomy = read_csv(DATA / 'taxonomy_v1.csv')
        cls.label_columns = sorted(TRAIN_ONLY & set(cls.train.columns))
        cls.folder = tempfile.TemporaryDirectory()
        cls.files = cls.build(Path(cls.folder.name) / 'plain')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def build(cls, out):
        argv = ['build_vlm_requests_v2.py', '--data', str(DATA), '--shortlist', str(SHORTLIST), '--out-dir', str(out)]
        with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
            cls.builder.main()
        return {arm: out / f'q38_dev_{arm}_requests.jsonl' for arm in cls.builder.ARMS}

    def requests(self, arm):
        lines = self.files[arm].read_text().splitlines()
        return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]

    def test_rows_are_the_sent_and_answered_development_rows_and_every_arm_has_the_same_rows(self):
        shortlist = json.loads(SHORTLIST.read_text())
        dev = self.train.loc[self.folds == 0]
        expected = [t for t in dev.trace_id if shortlist[t]['sent'] and shortlist[t]['has_zero_shot']]
        self.assertEqual(len(expected), 135)
        for arm, mode in self.builder.ARMS.items():
            header, requests = self.requests(arm)
            self.assertEqual([r['trace_id'] for r in requests], expected)  # unsent and unanswered rows are not written
            self.assertEqual((header['rows'], header['arm'], header['prompt'], header['cards']), (135, arm, 'v2', mode))
            self.assertEqual(header['schema'], answer_schema_v2(header['classes']))

    def test_request_text_is_what_a_row_without_a_label_and_an_index_of_fit_rows_give(self):
        # Rebuilds every shortlist block here from a label-free row and an index of the fit rows alone. A builder that
        # indexed development rows, read the row's label or marked a card would write different text.
        from k12.exemplars import ExemplarIndex
        index = ExemplarIndex(self.train.loc[self.folds != 0].reset_index(drop=True))
        names, descriptions = self.builder.code_texts(self.taxonomy)
        shortlist = json.loads(SHORTLIST.read_text())
        rows = {r.trace_id: r for r in self.train.loc[self.folds == 0].itertuples(index=False)}
        system = system_prompt_v2(self.taxonomy) + key_sentence_v2()
        v1 = {}
        if V1_REQUESTS.exists():  # the photos and the per-row text must be those of the v1 (A0) request
            v1 = {r['trace_id']: r['messages'][1]['content'] for r in map(json.loads, V1_REQUESTS.read_text().splitlines()[1:])}
        compared = 0
        for arm, mode in self.builder.ARMS.items():
            for request in self.requests(arm)[1]:
                r = rows[request['trace_id']]
                self.assertEqual([m['role'] for m in request['messages']], ['system', 'user'])
                self.assertEqual(request['messages'][0]['content'], system)
                content = request['messages'][1]['content']
                if mode is not None:
                    query = Query(r.trace_id, r.question_id, r.question_text, r.topic_name)
                    block, _ = self.builder.row_block(query, shortlist[r.trace_id]['codes'], index, names, descriptions, mode)
                    self.assertEqual(content[0], {'type': 'text', 'text': block})
                    content = content[1:]
                self.assertEqual(sum(part['type'] == 'image_url' for part in content), 2 if str(r.question_image_file).strip() else 1)
                self.assertEqual(content[-1]['text'], f'Grade: {r.grade}\nTopic: {r.topic_name}\nQuestion: {r.question_text}\n'
                                                      f'Answer key: {r.correct_answer}')
                if v1:
                    self.assertEqual(content, v1[r.trace_id])
                    compared += 1
        if v1:
            self.assertEqual(compared, 4 * 135)

    def test_no_request_changes_when_the_development_labels_change(self):
        # Give every development row another development row's label, transcript and annotations (the folds are held
        # fixed, because make_folds itself reads them). Not one byte of any arm may change.
        dev = np.flatnonzero(self.folds == 0)
        other, donor = self.train.copy(), np.random.default_rng(0).permutation(dev)
        for column in self.label_columns:
            other.iloc[dev, other.columns.get_loc(column)] = self.train[column].to_numpy()[donor]
        self.assertGreater(int((other.root_code != self.train.root_code).sum()), 150)
        self.assertTrue(other.loc[self.folds != 0].equals(self.train.loc[self.folds != 0]))
        real = self.builder.read_csv
        fake = lambda path: other.copy() if Path(path).name == 'train.csv' else real(path)
        with mock.patch.object(self.builder, 'read_csv', fake), mock.patch.object(self.builder, 'make_folds', lambda frame: self.folds):
            changed = self.build(Path(self.folder.name) / 'other-labels')
        for arm in self.builder.ARMS:
            self.assertEqual(hashlib.sha256(changed[arm].read_bytes()).hexdigest(),
                             hashlib.sha256(self.files[arm].read_bytes()).hexdigest(), arm)

    def test_request_files_on_disk_are_what_the_builder_writes_today(self):
        on_disk = {arm: V2_REQUESTS / f'q38_dev_{arm}_requests.jsonl' for arm in self.builder.ARMS}
        if not all(path.exists() for path in on_disk.values()):
            self.skipTest('data/v10 request files are not on this machine')
        for arm, path in on_disk.items():  # a stale file (built before a prompt or card change) fails here
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(self.files[arm].read_bytes()).hexdigest(), arm)


if __name__ == '__main__':
    unittest.main()
