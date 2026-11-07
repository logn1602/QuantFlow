"""Forecasting models. Training and prediction only — evaluation lives in
quantflow.evaluation so scoring cannot quietly depend on training internals."""

# Trading days every model forecasts. The ensemble combines the four base
# models date by date and refuses to run unless they all cover exactly the
# same dates, so the horizon is defined once.
FORECAST_DAYS = 7
