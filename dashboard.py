"""
dashboard.py
------------
Streamlit Cloud entry point.

The dashboard itself lives at src/quantflow/dashboard/app.py. This file exists
only because Streamlit Community Cloud stores an app's main file path at deploy
time and offers no way to edit it afterward (App settings covers URL, Python
version, sharing, and secrets — nothing else). The deployed app at
quantflow-analytics.streamlit.app still points here, so this is the seam that
keeps that URL alive without deleting and redeploying the app.

runpy, not `import`: Streamlit re-executes its main script on every widget
interaction. An import would be a no-op after the first run because the module
would already be in sys.modules, so the page would never redraw. run_module
executes the module body fresh each time.

Local development does not go through this file — use `make dashboard`, which
runs the real module path directly.
"""

import runpy

runpy.run_module("quantflow.dashboard.app", run_name="__main__")
