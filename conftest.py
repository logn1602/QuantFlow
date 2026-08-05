"""
conftest.py
-----------
Puts the project root on sys.path so tests can import top-level modules
(backtest, ensemble, xgboost_model, config) without installing the package.

Tests are offline by design: no database, no network, no model training.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
