"""Pytest configuration anchor.

Its presence at the project root puts the root on ``sys.path`` so tests can
import the ``catalyst`` package and the shared ``tests.strategies`` module by
their absolute paths regardless of the working directory pytest is invoked
from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
