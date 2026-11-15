"""Reasoner prompt v2: system prompt, key-list sentence, answer schema and the per-row shortlist block.

v1 lives in scripts/claude_label.py::system_prompt and scripts/build_vlm_requests.py and is not touched; the v1
wording is copied here. v2 changes three things (ARCHITECTURE_REFORM_2026.md 4.1.2): the line check becomes an
output field (line_checks), root_line becomes a label with the line copied into root_line_text, and the model
takes a second look at the photo at its root line before answering.
"""
import random
import zlib

KEYS_V2 = ('transcription', 'line_checks', 'root_line', 'root_line_text', 'reasoning', 'first_code', 'second_code', 'third_code')
LINE_CHECK_KEYS = ('line', 'left', 'right', 'equal', 'follows')
ROOT_LINE_PATTERN = '^(L[0-9]+|none)$'


def system_prompt_v2(taxonomy):
    rows = '\n'.join(f'{r.id} | {r.name} | {r.description} | {r.nature} | grade band {r.grade_band} | {r.section}'
                     for r in taxonomy.itertuples(index=False))
    return f"""You are an experienced mathematics teacher diagnosing Indonesian K-12 students' handwritten working.

For each photo, identify the ROOT misconception: the major error that the rest of the working follows from. It is not necessarily the first error written on the page. A root is always a major error, one that changes the reasoning or the result. Answer NONE when the working has no major error: it is fully correct, or carries only minor slips such as imprecise notation that do not change the result. About a third of workings are NONE, and false alarms on correct work are as harmful as missed errors. A correct final answer does not prove the reasoning is correct, and a wrong final answer can come from a minor slip. Respect crossed-out work: only the student's final, uncrossed reasoning counts. Do not invent steps that are not written.

Procedure:
1. Transcribe the student's working line by line as "L1 | ...", "L2 | ...", preserving the student's own symbols and mistakes exactly; never correct them. Mark unreadable parts as [unclear].
2. Check each line against the question, the answer key and the previous line, and write the check down in line_checks: for every line that contains "=", give the line label, the left side and the right side exactly as the student wrote them (for a chain a = b = c, the left side is a and the right side is everything after the first "="), whether the two sides really are equal (for a chain, whether every "=" holds), and whether the line follows from the previous line (for the first line, from the question).
3. Find the first line where a major error enters, and decide whether later errors are consequences of it. Give that line in root_line as its label only, "L<n>" (for example "L3"), or "none" when the working has no major error. Copy the line character by character from your transcription into root_line_text (leave it empty when root_line is "none").
4. Second look: if you chose a root line, then before you write your answer look at the photo again at that line and compare it with your transcription of that line, character by character. If your copy differs from what the student wrote, correct the transcription, then check that line and your diagnosis again.
5. Choose the code whose definition best matches that root error; use the descriptions, not just the names. Then give your second and third most likely codes, which may include NONE.

Taxonomy (code | name | description | nature | grade band | section). NONE means no major misconception.
{rows}
"""


def key_sentence_v2():
    """Appended to the system prompt, as the v1 sentence is in build_vlm_requests.py; lists the keys of the schema in order."""
    quoted = lambda keys: ', '.join(f'"{k}"' for k in keys)
    return (f'\nAnswer with one JSON object with exactly these keys, in this order: {quoted(KEYS_V2)}. "line_checks" is a list '
            f'with one object for every line that contains "=", each with exactly these keys: {quoted(LINE_CHECK_KEYS)}; '
            '"equal" and "follows" are true or false. "root_line" is "L<n>" or "none". Each code is a taxonomy code or '
            'NONE, and the three codes are different.\n')


def answer_schema_v2(classes, root_line_pattern=True):
    """JSON schema of a v2 answer; properties are in KEYS_V2 order because a constrained decoder writes them in that order.

    root_line_pattern=False drops the regular expression on root_line, for a server whose decoder rejects "pattern".
    """
    code = {'type': 'string', 'enum': list(classes)}
    text, flag = {'type': 'string'}, {'type': 'boolean'}
    check = {'type': 'object', 'additionalProperties': False, 'required': list(LINE_CHECK_KEYS),
             'properties': {'line': text, 'left': text, 'right': text, 'equal': flag, 'follows': flag}}
    fields = {'transcription': text, 'line_checks': {'type': 'array', 'items': check},
              'root_line': {'type': 'string', 'pattern': ROOT_LINE_PATTERN} if root_line_pattern else text,
              'root_line_text': text, 'reasoning': text,
              'first_code': code, 'second_code': code, 'third_code': code}
    return {'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


def shortlist_order(row_trace_id, codes):
    """The row's shortlist in an order that depends only on the trace_id and the set of codes, never on a score."""
    order = sorted(set(codes))
    random.Random(zlib.crc32(str(row_trace_id).encode())).shuffle(order)
    return order


def shortlist_block(row_trace_id, codes, cards, other_code_names):
    """Text shown before a student's photo: the shortlisted codes (cards[code] is the text for each) in an order
    shuffled per row, with no scores, then the names of every other code, which may still be the answer.

    other_code_names maps each remaining code to its name.
    """
    order = shortlist_order(row_trace_id, codes)
    examples = any('\n' in cards[c] for c in order)
    intro = ('Candidate codes for the next student, picked by an automatic first pass. They are listed in random order; the '
             'order carries no information. The list is a hint, not a ruling: the root code may be one of these or a code '
             'outside the list, so diagnose the working yourself first and then compare. NONE is always on the list.')
    if examples:
        intro += (' Where a code comes with an example, it is another student\'s working that expert annotators diagnosed '
                  'with that code. An example shows what the code looks like; it says nothing about the next student.')
    body = ('\n\n' if examples else '\n').join(f'[{n}] {cards[c]}' for n, c in enumerate(order, 1))
    rest = '; '.join(f'{c} ({other_code_names[c]})' for c in sorted(other_code_names))
    return (f'{intro}\n\n{body}\n\nCodes not on the list (defined in the taxonomy above; the answer may be one of them): '
            f'{rest}\n\nNow diagnose the following student.')
