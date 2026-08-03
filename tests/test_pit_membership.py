"""Point-in-time S&P 500 membership reconstruction — pure logic, no network."""

import pandas as pd

from screener.pit_membership import all_time_tickers, constituents_as_of


def _changes(rows):
    df = pd.DataFrame(rows, columns=["date", "added_ticker", "removed_ticker"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_no_changes_after_target_date_returns_current_list_unchanged():
    current = {"AAA", "BBB", "CCC"}
    changes = _changes([])
    members = constituents_as_of("2020-01-01", current, changes)
    assert members == current


def test_a_later_addition_is_undone_for_an_earlier_target_date():
    # NEWCO was added in 2022 -> as of 2020 it must NOT be a member yet,
    # even though it's in today's current-constituent list.
    current = {"AAA", "NEWCO"}
    changes = _changes([("2022-06-01", "NEWCO", "OLDCO")])
    members = constituents_as_of("2020-01-01", current, changes)
    assert "NEWCO" not in members
    assert "OLDCO" in members  # it was removed in 2022 -> still a member in 2020
    assert "AAA" in members


def test_target_date_on_the_effective_date_reflects_the_change():
    # A change effective exactly on the target date is already in force.
    current = {"AAA", "NEWCO"}
    changes = _changes([("2022-06-01", "NEWCO", "OLDCO")])
    members = constituents_as_of("2022-06-01", current, changes)
    assert "NEWCO" in members
    assert "OLDCO" not in members


def test_a_name_added_then_later_removed_round_trips_correctly():
    # TICKR was added in 2018, then removed in 2023. Today's list doesn't
    # contain it. It should be a member for 2019-2022, but not before 2018
    # or after 2023.
    current = {"AAA"}  # TICKR is not in today's list
    changes = _changes([
        ("2018-03-01", "TICKR", "GONE1"),
        ("2023-09-01", "BACK1", "TICKR"),
    ])
    assert "TICKR" not in constituents_as_of("2017-01-01", current, changes)
    assert "TICKR" in constituents_as_of("2020-01-01", current, changes)
    assert "TICKR" not in constituents_as_of("2024-01-01", current, changes)


def test_multiple_changes_on_different_dates_compound_correctly():
    current = {"A4"}
    changes = _changes([
        ("2021-01-01", "A2", "A1"),
        ("2022-01-01", "A3", "A2"),
        ("2023-01-01", "A4", "A3"),
    ])
    # Walking back through all three swaps from A4 (today) should land on A1.
    members = constituents_as_of("2020-06-01", current, changes)
    assert members == {"A1"}


def test_all_time_tickers_unions_current_and_changes_since_a_date():
    current = {"AAA", "BBB"}
    changes = _changes([
        ("2019-01-01", "OLD1", "OLD2"),   # before `since` -> excluded
        ("2021-01-01", "NEW1", "NEW2"),   # on/after `since` -> included
    ])
    out = all_time_tickers(current, changes, since="2020-01-01")
    assert out == {"AAA", "BBB", "NEW1", "NEW2"}
    assert "OLD1" not in out and "OLD2" not in out


def test_all_time_tickers_with_no_since_includes_everything():
    current = {"AAA"}
    changes = _changes([("2019-01-01", "OLD1", "OLD2")])
    out = all_time_tickers(current, changes, since=None)
    assert out == {"AAA", "OLD1", "OLD2"}
