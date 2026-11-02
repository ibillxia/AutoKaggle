from pathlib import Path
import sys
from types import FunctionType

here = Path(__file__).resolve().parent
root = here.parents[2]
sys.path[:0] = [str(here), str(root / 'playbook/kit/decoder'),
                str(root / 'playbook/kit/decoder/exp/convert'),
                str(root / 'playbook/kit/decoder/exp/arithmetic'),
                str(root / 'playbook/kit')]
from final_decoder import decode as ref
from high_stream import convert_safe
import replay_erase

env = dict(ref.decode.__globals__, convert_safe=convert_safe,
           extraction=replay_erase.extraction)
decode = FunctionType(ref.decode.__code__, env, 'decode',
                      ref.decode.__defaults__, ref.decode.__closure__)
