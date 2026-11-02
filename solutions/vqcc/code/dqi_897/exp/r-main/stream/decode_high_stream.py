"""Streamed high full-head targets plus monic-root fusion; unfrozen experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter import clone
from high_stream import convert_safe
from final_decoder import decode as base
core=clone(base.decode,dict(vars(base),convert_safe=convert_safe))
def decode(b,S,E):core(b,S,E)
