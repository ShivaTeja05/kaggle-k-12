import json,sys,subprocess,time
from pathlib import Path
wheel_dirs=sorted({p.parent for p in Path('/kaggle/input').rglob('xgrammar-*.whl')})
assert len(wheel_dirs)==1
subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(wheel_dirs[0]),'--no-deps','xgrammar==0.2.7','apache-tvm-ffi==0.1.11'],check=True)
import xgrammar as xgr
from transformers import PreTrainedTokenizerFast
"""Compact decoding grammar, independently compilable without model weights."""
import json
import re


def ranking_regex(classes):
    code='(?:'+'|'.join(re.escape(c) for c in classes)+')'
    return code+' '+code+' '+code


def evidence_grammar(classes):
    # An unbounded single-line loop avoids expanding 240 separate character
    # positions during token-mask compilation. Prompt and token cap limit length.
    choices=' | '.join(json.dumps(c) for c in classes)
    return ('root ::= "EVIDENCE: " evidence "\\nFINAL: " code " " code " " code\n'
            'evidence ::= [^\\n]*\ncode ::= '+choices+'\n')

classes=['ALG-01', 'ALG-02', 'ALG-03', 'ALG-04', 'ALG-05', 'ALG-06', 'ANG-01', 'ANG-02', 'ANS-01', 'ANS-02', 'CAL-01', 'CHK-01', 'COR-01', 'Dec-01', 'Dec-02', 'EXP-01', 'EXP-02', 'EXP-03', 'EXP-04', 'EXP-05', 'EXP-06', 'EXP-07', 'FRA-01', 'FRA-02', 'FRA-03', 'FRA-04', 'FRA-05', 'FRA-06', 'FRM-01', 'FUN-01', 'GEO-01', 'INT-01', 'INT-02', 'INT-03', 'INT-04', 'INV-01', 'KPF-01', 'KPF-02', 'KPF-03', 'KPF-04', 'LIN-01', 'LOG-02', 'MEA-01', 'MEA-02', 'MEA-03', 'MEA-04', 'MEA-05', 'NONE', 'NOT-01', 'NOT-02', 'OOO-01', 'OOO-02', 'OPS-01', 'OPS-02', 'OTH-00', 'PRO-01', 'PRO-02', 'PRO-03', 'PRO-04', 'PV-01', 'QUA-02', 'RAT-01', 'RAT-02', 'REL-01', 'SEQ-01', 'SEQ-02', 'SET-01', 'SIM-01', 'STA-01', 'STA-03', 'STA-04', 'STA-05', 'UNT-01', 'VIS-01', 'WOP-01', 'WOP-02']
paths=[p for p in Path('/kaggle/input').rglob('tokenizer.json') if (p.parent/'config.json').exists()]
assert len(paths)==1,paths
config=json.loads((paths[0].parent/'config.json').read_text())
tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(paths[0]))
vocab_size=config.get('text_config',config)['vocab_size']
info=xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=vocab_size)
compiler=xgr.GrammarCompiler(info,max_threads=2)
report={'xgrammar':'0.2.7','model_source':'boristown/qwen3-5-27b-awq-4bit/transformers/default/1','vocab_size':vocab_size,'uses_gpu':False}
start=time.monotonic()
compiled=compiler.compile_grammar(evidence_grammar(classes))
report['evidence_grammar_seconds']=time.monotonic()-start
start=time.monotonic()
compiler.compile_regex(ranking_regex(classes))
report['code_regex_seconds']=time.monotonic()-start
report['passed']=True
Path('/kaggle/working/grammar_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
