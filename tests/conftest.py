"""Shared pytest configuration.

The sys.path manipulation this file used to perform is gone: quantflow is an
installed package, so tests import it the same way anything else does.

Tests are offline by design — no database, no network, no model training. The
integration and eval markers exist for suites that need those; see
pyproject.toml.
"""
