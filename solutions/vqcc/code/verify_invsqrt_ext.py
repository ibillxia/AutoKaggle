"""Run the official invsqrt verifier on an arbitrary .kmb, with kickmix importable.

env.py must run first: it puts the locally built kickmix on sys.path and registers the
DLL directory its dependencies live in.

Usage: python verify_invsqrt_ext.py PATH_TO_KMB
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

verifier = env.DATA / 'verifiers' / 'invsqrt' / 'verify_submission_invsqrt.py'
sys.argv = ['verify_submission_invsqrt', str(Path(sys.argv[1]).resolve())]
runpy.run_path(str(verifier), run_name='__main__')
