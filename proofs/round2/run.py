"""Run from repo root: .venv/bin/python proofs/round2/run.py MODE [pytest args]."""
import ast
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laplace.adapters import lmstudio
import pytest

mode = sys.argv[1]
if mode == 'old':
    source = subprocess.check_output(['git', 'show', '6072abf:laplace/adapters/lmstudio.py'], text=True)
    exec(compile(source, '6072abf/lmstudio.py', 'exec'), lmstudio.__dict__)
elif mode == 'mutation':
    source = subprocess.check_output(['git', 'show', '6072abf^:laplace/adapters/lmstudio.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'ensure_loaded')
    code = ast.Module(body=[method], type_ignores=[])
    namespace = dict(lmstudio.__dict__)
    exec(compile(code, 'guard-reverted/lmstudio.py', 'exec'), namespace)
    lmstudio.LMStudioAdapter.ensure_loaded = namespace['ensure_loaded']
else:
    assert mode == 'new'
print(f'MODE={mode}; adapter={lmstudio.__file__}', flush=True)
raise SystemExit(pytest.main(sys.argv[2:]))
