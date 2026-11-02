"""Direct candidate plus lead's checked k2/k3 truncated products:84T/pair."""
from pathlib import Path
import sys,importlib.util
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter import clone
import direct_converter as direct
spec=importlib.util.spec_from_file_location('bconvert_frozen_truncated_small',Path(__file__).with_name('truncated_small.py'))
small=importlib.util.module_from_spec(spec);spec.loader.exec_module(small)


def make_converter():
 original_factory=direct.mh.make_converter
 def factory(return_scope=False):
  cv,scope,short=original_factory(True);oldconv=scope['convolution_xor']
  def convolution(b,x,y,z,reverse=False):
   k=len(z)
   if k in (2,3) and len(x)==k and len(y)==2:
    small.accumulate(b,[q for w in x for q in w],[q for w in y for q in w],[q for w in z for q in w],k)
   else:oldconv(b,x,y,z,reverse)
  scope['convolution_xor']=convolution;short['old']=SimpleNamespace(**scope)
  return (cv,scope,short) if return_scope else cv
 return clone(direct.make_converter,dict(vars(direct),mh=SimpleNamespace(make_converter=factory)))()
convert_safe=make_converter()
