import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from k12.core import read_csv, label_space


def validate(data,submission):
    test=read_csv(Path(data)/'test.csv');tax=read_csv(Path(data)/'taxonomy_v1.csv');out=read_csv(submission)
    if out.columns.tolist()!=['trace_id','root_code']:raise ValueError('Expected trace_id,root_code header')
    if not out.trace_id.is_unique or set(out.trace_id)!=set(test.trace_id):raise ValueError('Missing, duplicate or unexpected IDs')
    allowed=set(label_space(tax))
    for r in out.itertuples(index=False):
        codes=str(r.root_code).split()
        if not 1<=len(codes)<=3 or len(set(codes))!=len(codes) or not set(codes)<=allowed:
            raise ValueError(f'Invalid ranked codes for trace {r.trace_id}')
    print(f'VALID: {len(out)} rows, exact test ID coverage, 1–3 distinct valid codes per row')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--data',default='data/raw');a=p.parse_args();validate(a.data,a.submission)
