"""
conftest.py

Ensures `detection/` and `ml/` are importable from `tests/`, regardless
of the current working directory pytest is invoked from.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DETECTION_DIR = os.path.join(_REPO_ROOT, "detection")
_ML_DIR = os.path.join(_REPO_ROOT, "ml")

for _path in (_REPO_ROOT, _DETECTION_DIR, _ML_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)