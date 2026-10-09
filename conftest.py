"""Root pytest configuration.

Markdown tutorials under ``docs/tutorials`` are executed as doctests through
sybil, but only when that directory is passed explicitly
(``uv run pytest docs/tutorials``); the default ``testpaths`` cover the Python
suites only. If sybil is not installed the markdown collector is simply absent,
so ``uv run pytest`` works without ``--noconftest``.
"""

try:
    from sybil import Sybil
    from sybil.parsers.markdown import PythonCodeBlockParser
except ImportError:  # pragma: no cover
    Sybil = None

if Sybil is not None:
    pytest_collect_file = Sybil(
        parsers=[PythonCodeBlockParser()],
        patterns=["*.md"],
    ).pytest()
