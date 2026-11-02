"""Run the official dqi verifier on an arbitrary .kmb, with kickmix importable.

Usage: python verify_dqi_ext.py PATH_TO_KMB
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

verifier = env.DATA / 'verifiers' / 'dqi' / 'verify_submission_dqi.py'
sys.argv = ['verify_submission_dqi', str(Path(sys.argv[1]).resolve())]
runpy.run_path(str(verifier), run_name='__main__')
