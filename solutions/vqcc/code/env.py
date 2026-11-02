"""Paths and imports shared by the local verifier runs.

kickmix's ``Circuit.to_file``/``from_file`` copy data through a C ``tmpfile()``, which on
this machine fails with EACCES because the user's %TEMP% is not writable for these
processes. Pointing TMP/TEMP at a project-local scratch directory before kickmix is
imported fixes it; ``verifiers`` also need the directory to exist.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # solutions/vqcc
DATA = ROOT / 'data'
BUILD = ROOT / 'build'
SCRATCH = ROOT / '_scratch'
SAMPLES = DATA / 'sample_submission'

SCRATCH.mkdir(exist_ok=True)
os.environ['TMP'] = str(SCRATCH)
os.environ['TEMP'] = str(SCRATCH)

for _p in (BUILD, DATA / 'kickmix', DATA / 'verifiers'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The locally built kickmix.pyd resolves libwinpthread-1.dll from build/, and Python 3.8+
# no longer searches a module's own directory for its dependencies.
if hasattr(os, 'add_dll_directory'):
    try:
        # Keep the returned cookie alive: if it is garbage collected the directory is
        # removed from the search path again before kickmix.pyd is ever imported.
        _dll_dir_cookie = os.add_dll_directory(str(BUILD))
    except OSError:
        pass


def import_kickmix():
    import kickmix  # noqa: E402
    return kickmix
