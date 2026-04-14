"""OpenAgent end-to-end test suite.

Each ``test_<category>.py`` module registers its tests into the shared
``TESTS`` list via the ``@test(category, name)`` decorator. The driver
(``scripts/test_openagent.py``) imports every module in this package so
registration happens as a side-effect, then runs them in registration
order.

Keep test modules focused: one category per file, short imports, no
cross-module state. Shared helpers live in ``_framework.py`` and
``_setup.py``.
"""
