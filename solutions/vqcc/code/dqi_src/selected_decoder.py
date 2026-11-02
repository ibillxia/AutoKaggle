"""Notebook-facing wrapper for Kozykappa V3's readable DQI source.

Place upstream dqi_driver.py and its gen_dqi_v6/ directory beside this file.
The competition kickmix package must already be on sys.path. This wrapper
performs source generation only; use the unchanged official full verifier.
"""
from hashlib import sha256
from pathlib import Path

EXPECTED_SHA256 = '529b075009f7758abeda83a63407f0c1feeb51dd70631ae4335dfd28721b7e08'
EXPECTED_QUBITS = 1789
EXPECTED_TOFFOLIS = 445229


def build_selected(destination, work):
    import dqi_driver
    destination = Path(destination)
    destination.parent.mkdir(exist_ok=True, parents=True)
    report = dqi_driver.build(destination, Path(work))
    digest = sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    actual = digest.hexdigest()
    assert actual == EXPECTED_SHA256, (actual, EXPECTED_SHA256)
    assert report['num_qubits'] == EXPECTED_QUBITS, report
    assert report['toffoli_count'] == EXPECTED_TOFFOLIS, report
    return report | {'sha256': actual, 'upstream': 'Kozykappa V3 / run361677445'}
