"""US equity market time.

The one place that knows which time zone the exchange keeps and when a
session's daily bar can be treated as final.
"""

import datetime as dt

EXCHANGE_TZ = "America/New_York"

# The regular session closes at 16:00 New York and Yahoo's daily bar picks up
# the closing-auction price shortly after. An hour of margin before treating
# today's daily bar as final: stored any earlier, it is a snapshot of a session
# still trading, and with ON CONFLICT DO NOTHING it would never be corrected.
DAILY_BAR_FINAL_AFTER = dt.time(17, 0)
