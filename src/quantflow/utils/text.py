"""Text from third-party feeds, made safe to store.

Apart from features.sentiment, which uses it for headlines, so the trending
tier's rules can clean names without importing the sentiment stage: that
module turns every warning off as it is imported, for the whole process.
"""


def storable_text(value: str | None, limit: int) -> str:
    """Feed text made safe to bind: no NUL bytes, no code points UTF-8 cannot
    encode (lone surrogates become "?"), at most `limit` characters.

    Headlines and URLs come from third-party feeds. psycopg2 rejects both
    defects on the client side, before anything reaches the server, so in one
    batched INSERT a single malformed article would fail the whole write and
    stop the daily refresh with it.
    """
    cleaned = (value or "").replace("\x00", "")
    return cleaned.encode("utf-8", "replace").decode("utf-8")[:limit]
