"""Send prepared chat requests to a local OpenAI-compatible server (vLLM) and save the ranked codes.

Standalone on purpose (standard library only) so it runs on a pod without the project code.
Output lines match runs/claude/*.jsonl: {"trace_id", "answer": {... "first_code", "second_code", "third_code"}}.

Usage: python vlm_zero_shot_client.py --requests R.jsonl --out O.jsonl --mode nothink|think [--n 3]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

KEYS = ('first_code', 'second_code', 'third_code')


def parse(text, classes):
    """The answer object from the model's text; falls back to pulling the three codes out with a pattern."""
    text = text or ''
    if '</think>' in text:
        text = text.split('</think>')[-1]
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        try:
            answer = json.loads(text[start:end + 1])
            if isinstance(answer, dict) and answer.get('first_code') in classes:
                return answer, 'json'
        except json.JSONDecodeError:
            pass
    answer = {}
    for key in KEYS:
        found = re.search(rf'"?{key}"?\s*[:=]\s*"?([A-Z]{{2,4}}-\d{{2}}|NONE)', text)
        if found and found.group(1) in classes:
            answer[key] = found.group(1)
    return (answer, 'pattern') if answer.get('first_code') else ({}, 'failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8000/v1'); ap.add_argument('--model', default='q38')
    ap.add_argument('--mode', choices=['nothink', 'think'], required=True)
    ap.add_argument('--effort', default='medium', help='reasoning effort in think mode: low, medium or xhigh')
    ap.add_argument('--n', type=int, default=1, help='samples per photo')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--max-tokens', type=int)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--deadline', type=float, help='unix time after which no new request is started')
    ap.add_argument('--no-schema', action='store_true', help='do not constrain the answer to the JSON schema')
    a = ap.parse_args()
    lines = Path(a.requests).read_text().splitlines()
    header = json.loads(lines[0]); classes = set(header['classes'])
    todo = [json.loads(l) for l in lines[1:]]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['trace_id'] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in todo if r['trace_id'] not in done][:a.limit]

    def post(payload):
        request = urllib.request.Request(a.base_url.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
        try:
            with urllib.request.urlopen(request, timeout=1800) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:400]}') from None

    think = a.mode == 'think'
    base = dict(model=a.model, n=a.n, seed=a.seed,
                max_tokens=a.max_tokens or (14000 if think else 3000),
                temperature=1.0 if think else 0.7, top_p=0.95 if think else 0.8)
    template = {'enable_thinking': True, 'reasoning_effort': a.effort} if think else {'enable_thinking': False}
    schema = {'type': 'json_schema', 'json_schema': {'name': 'diagnosis', 'strict': True, 'schema': header['schema']}}
    state = {'schema': not a.no_schema}
    lock = threading.Lock(); t0 = time.time(); stats = {'rows': 0, 'out_tokens': 0, 'failed': 0, 'length': 0, 'skipped': 0}

    def call(request):
        if a.deadline and time.time() > a.deadline:
            return None
        last = None
        for attempt in range(3):
            kwargs = dict(base, messages=request['messages'], top_k=20, chat_template_kwargs=template)
            if state['schema']:
                kwargs['response_format'] = schema
            try:
                response = post(kwargs)
                break
            except Exception as exc:  # a server that rejects the schema is retried once without it, for every later row too
                last = repr(exc)[:300]
                if state['schema'] and any(w in last.lower() for w in ('response_format', 'schema', 'guided', 'structured', 'grammar')):
                    state['schema'] = False
                time.sleep(2)
        else:
            return {'trace_id': request['trace_id'], 'answer': {}, 'error': last}
        samples, how, finish, reasoning = [], [], [], 0
        for choice in response['choices']:
            message = choice['message']
            thought = message.get('reasoning_content') or message.get('reasoning') or ''
            text = message.get('content') or ''
            answer, method = parse(text, classes)
            if method == 'failed' and thought:  # some parsers file the whole reply under reasoning
                answer, method = parse(thought, classes)
            samples.append(answer); how.append(method); finish.append(choice.get('finish_reason')); reasoning += len(thought)
        usage = response.get('usage') or {}
        return {'trace_id': request['trace_id'], 'answer': samples[0], 'samples': samples, 'parsed_by': how, 'finish': finish,
                'reasoning_chars': reasoning, 'schema': state['schema'],
                'usage': {'prompt_tokens': usage.get('prompt_tokens', 0), 'completion_tokens': usage.get('completion_tokens', 0)},
                'raw_first': (response['choices'][0]['message'].get('content') or '')[:200] if how[0] != 'json' else None}

    with ThreadPoolExecutor(a.workers) as pool, out.open('a', buffering=1) as handle:
        for future in as_completed([pool.submit(call, r) for r in todo]):
            record = future.result()
            with lock:
                if record is None:
                    stats['skipped'] += 1; continue
                handle.write(json.dumps(record) + '\n')
                stats['rows'] += 1
                stats['out_tokens'] += record.get('usage', {}).get('completion_tokens', 0)
                stats['failed'] += not record['answer'].get('first_code')
                stats['length'] += 'length' in (record.get('finish') or [])
                if stats['rows'] % 20 == 0 or stats['rows'] == len(todo):
                    seconds = time.time() - t0
                    print(json.dumps({'mode': a.mode, **stats, 'of': len(todo), 'seconds': round(seconds, 1),
                                      'out_tok_per_s': round(stats['out_tokens'] / max(seconds, 1), 1), 'schema': state['schema']}), flush=True)
    print(json.dumps({'done': a.mode, **stats, 'seconds': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
