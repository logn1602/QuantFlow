# Trending-tier fixtures

One real answer from each Yahoo list the trending tier reads, captured on
2026-11-10 at 20:07 UTC (15:07 New York, during the session), read-only, from
the owner's laptop:

| File | What it is |
|---|---|
| `most_actives.json`, `day_gainers.json`, `day_losers.json` | What `yf.screen(name, count=25)` returned (yfinance 1.7.0): the body's `finance.result[0]` |
| `trending_tickers.json` | The whole body of `GET https://query1.finance.yahoo.com/v1/finance/trending/US?count=25&useQuotes=true&fields=...` |

Each quote is trimmed to the fields the sources read (`symbol`, `shortName`,
`quoteType`, `regularMarketPrice`, `regularMarketChangePercent`,
`regularMarketVolume`, `marketCap`, `firstTradeDateMilliseconds`), plus
`exchange` and, on the trending list, Yahoo's `trendingScore`. The structure
around the quotes is Yahoo's, unchanged apart from fields the sources never read.

What they hold that the tests rely on: an index (`^TNX`, which symbol validation
drops), a private company (`ANTH.PVT`, which the EQUITY filter drops), a first
trade date before 1970 (a negative timestamp), names with trailing spaces or cut
short by Yahoo, prices under $5, two core tickers (`NVDA`, `MSFT`), and `SPCX`
on `most_actives` with its first trade on 2026-06-12.
