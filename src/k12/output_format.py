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
