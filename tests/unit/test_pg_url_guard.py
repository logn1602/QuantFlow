"""tests/unit/test_pg_url_guard.py

Covers the check that keeps the integration suite off the live database.

It used to look only at the URL's host, but libpq takes connection targets
from the query string too: ?host=, ?hostaddr= and ?service= all point the
connection somewhere other than the host the URL names. PGHOSTADDR and
PGSERVICE in the environment do the same.
"""

import pytest


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u@localhost:5432/scratch?host=db.example.supabase.co",
        "postgresql://u@localhost:5432/scratch?hostaddr=203.0.113.7",
        "postgresql://u@localhost:5432/scratch?service=live",
        "postgresql://u@db.example.supabase.co:6543/postgres",
        "mysql://u@localhost/scratch",
    ],
)
def test_the_local_only_guard_rejects_anything_that_can_leave_this_machine(
    is_local_pg_url, url
):
    assert not is_local_pg_url(url)


@pytest.mark.parametrize(
    "url",
    ["postgresql://u@127.0.0.1:5432/scratch", "postgresql://u@localhost/scratch"],
)
def test_the_local_only_guard_accepts_a_plain_local_url(
    is_local_pg_url, monkeypatch, url
):
    monkeypatch.delenv("PGHOSTADDR", raising=False)
    monkeypatch.delenv("PGSERVICE", raising=False)
    assert is_local_pg_url(url)


@pytest.mark.parametrize(
    "var, value", [("PGHOSTADDR", "203.0.113.7"), ("PGSERVICE", "live")]
)
def test_the_local_only_guard_rejects_a_libpq_redirect_from_the_environment(
    is_local_pg_url, monkeypatch, var, value
):
    """libpq fills every setting the URL leaves out from the environment, so a
    localhost URL still connects to PGHOSTADDR, or wherever PGSERVICE's entry
    in the service file points."""
    monkeypatch.setenv(var, value)
    assert not is_local_pg_url("postgresql://u@localhost:5432/scratch")
