"""Complete direct route + full low heads + checked small truncated products."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter import clone
from truncated_converter import convert_safe
from final_decoder import decode as base
decode=clone(base.decode,dict(vars(base),convert_safe=convert_safe))
