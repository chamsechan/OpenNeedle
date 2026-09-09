"""Build current sources or verify two isolated libraries including window wrap.

Snapshot the pre-change sources with `build attention_before` before editing.
Timing uses experiment_decode.py bench so variants share the same requests.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import experiment_decode as experiment

ROOT = experiment.ROOT
OUT = experiment.OUT
FLAGS = ['-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-shared', '-pthread', '-fopenmp']


def build(name):
    dest = OUT / name
    if dest.exists():
        raise SystemExit(f'Refusing to overwrite experiment snapshot: {dest}')
    dest.mkdir(parents=True)
    for path in (ROOT / 'needle2/csrc').glob('*.cpp'):
        (dest / path.name).write_bytes(path.read_bytes())
    subprocess.run(['c++', *FLAGS, str(dest / 'cq.cpp'), '-o', str(dest / 'lib.so')], check=True)
    print(dest / 'lib.so')


def verify(before, after):
    import numpy as np
    for name in [before, after]:
        subprocess.run([sys.executable, str(ROOT / 'scripts/verify_mhc.py'), 'dump', name], check=True)
    a = np.load(OUT / f'{before}_mhc_validation.npz')
    b = np.load(OUT / f'{after}_mhc_validation.npz')
    assert a.files == b.files
    for key in a.files:
        # Compare raw float bytes as well as values, including signed zeros.
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
        assert a[key].tobytes() == b[key].tobytes(), key
    report = dict(bitwise_equal=True, arrays=len(a.files),
                  elements=sum(a[k].size for k in a.files),
                  coverage='Full-model FP32/SDOT x FP32/INT8 KV x 1/4 threads; 280 steps with pinned prefix, window wrap, prefix restore; tiny archives',
                  variants={})
    for name in [before, after]:
        report['variants'][name] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((OUT / name).iterdir()) if p.suffix in ['.cpp', '.so']}
    (ROOT / 'reports/attention_numerical_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2])
    elif sys.argv[1] == 'verify':
        verify(*sys.argv[2:4])
    else:
        raise SystemExit('Expected build NAME or verify BEFORE AFTER')
