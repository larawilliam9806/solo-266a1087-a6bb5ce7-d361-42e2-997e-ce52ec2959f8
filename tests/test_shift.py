"""Observation-batch time correction (纠时): rules, atomicity, idempotency."""

import pytest

from conftest import load, make_review, results_for
from app import shift

T = 180.0


def _obs_batch_id(conn, station="S1"):
    row = conn.execute(
        "SELECT id FROM batches WHERE station = ? AND kind = 'observation' "
        "ORDER BY id DESC LIMIT 1",
        (station,),
    ).fetchone()
    assert row is not None
    return row["id"]


def _hours(conn, station="S1"):
    rows = conn.execute(
        "SELECT hour_utc, level_cm, batch_id FROM observation "
        "WHERE station = ? ORDER BY hour_utc",
        (station,),
    ).fetchall()
    return rows


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "raw,expected",
    [("3", 3), ("+3", 3), ("-2", -2), (" 4 ", 4), ("007", 7)],
)
def test_parse_offset_ok(raw, expected):
    assert shift.parse_offset(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "0", "+0", "-0", "abc", "1.5", "1e3", "3h", "999999"])
def test_parse_offset_rejected(raw):
    with pytest.raises(shift.ShiftError):
        shift.parse_offset(raw)


def test_parse_offset_zero_says_non_zero():
    with pytest.raises(shift.ShiftError, match="不能为 0"):
        shift.parse_offset("0")


# ------------------------------------------------------------------ success


def test_shift_moves_whole_batch_keeping_levels_and_batch(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    receipt = shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=3)
    assert receipt.batch_id == bid
    assert receipt.offset_hours == 3
    assert receipt.rows_moved == 2
    assert not receipt.replayed
    rows = _hours(conn)
    assert [r["hour_utc"] for r in rows] == [
        "2026-05-01T04:00:00Z",
        "2026-05-01T05:00:00Z",
    ]
    assert [r["level_cm"] for r in rows] == [100, 110]
    assert all(r["batch_id"] == bid for r in rows)


def test_shift_negative_offset_moves_backwards(conn):
    load(
        conn,
        "S1",
        observations=[
            ("2026-05-01 01:00", 100),
            ("2026-05-01 02:00", 110),
            ("2026-05-01 03:00", 120),
        ],
    )
    bid = _obs_batch_id(conn)
    shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=-2)
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-04-30T23:00:00Z",
        "2026-05-01T00:00:00Z",
        "2026-05-01T01:00:00Z",
    ]


def test_landing_may_overlap_own_batch(conn):
    # Consecutive hours shifted by +1: each landing is another row of the
    # same batch, which the rules explicitly allow.
    load(
        conn,
        "S1",
        observations=[
            ("2026-05-01 01:00", 100),
            ("2026-05-01 02:00", 110),
            ("2026-05-01 03:00", 120),
        ],
    )
    bid = _obs_batch_id(conn)
    receipt = shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=1)
    assert receipt.rows_moved == 3
    rows = _hours(conn)
    assert [r["hour_utc"] for r in rows] == [
        "2026-05-01T02:00:00Z",
        "2026-05-01T03:00:00Z",
        "2026-05-01T04:00:00Z",
    ]
    assert [r["level_cm"] for r in rows] == [100, 110, 120]


def test_shift_leaves_forecasts_untouched(conn):
    load(
        conn,
        "S1",
        forecasts=[("2026-05-01 05:00", "2026-04-30 20:00", "2026-04-30 20:04", 182)],
        observations=[("2026-05-01 05:00", 186)],
    )
    bid = _obs_batch_id(conn)
    shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=2)
    row = conn.execute("SELECT target_utc, level_cm FROM forecast").fetchone()
    assert row["target_utc"] == "2026-05-01T05:00:00Z"
    assert row["level_cm"] == 182


# ------------------------------------------------------------------ refusals


def test_landing_occupied_by_other_batch_refused(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    load(conn, "S1", observations=[("2026-05-01 04:00", 130)])
    with pytest.raises(shift.ShiftError) as excinfo:
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=2)
    assert excinfo.value.conflicts == ["2026-05-01T04:00:00Z"]
    # whole batch untouched
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T01:00:00Z",
        "2026-05-01T02:00:00Z",
        "2026-05-01T04:00:00Z",
    ]
    assert conn.execute("SELECT COUNT(*) c FROM time_shifts").fetchone()["c"] == 0


def test_landing_occupied_by_unbatched_row_refused(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    conn.execute(
        "INSERT INTO observation (station, hour_utc, level_cm, batch_id, updated_utc) "
        "VALUES ('S1', '2026-05-01T03:00:00Z', 130, NULL, '2026-05-01T03:00:00Z')"
    )
    with pytest.raises(shift.ShiftError) as excinfo:
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=2)
    assert excinfo.value.conflicts == ["2026-05-01T03:00:00Z"]
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T01:00:00Z",
        "2026-05-01T03:00:00Z",
    ]


def test_multiple_conflicts_all_listed(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    load(conn, "S1", observations=[("2026-05-01 03:00", 120), ("2026-05-01 04:00", 130)])
    with pytest.raises(shift.ShiftError) as excinfo:
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=2)
    assert excinfo.value.conflicts == ["2026-05-01T03:00:00Z", "2026-05-01T04:00:00Z"]


def test_backfill_overwrite_marks_batch_incomplete(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    # Backfill covers one row of the first batch (it takes over the hour).
    load(conn, "S1", observations=[("2026-05-01 01:00", 999)])
    with pytest.raises(shift.ShiftError, match="不完整"):
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=3)
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T01:00:00Z",
        "2026-05-01T02:00:00Z",
    ]


def test_forecast_batch_rejected(conn):
    load(
        conn,
        "S1",
        forecasts=[("2026-05-01 05:00", "2026-04-30 20:00", "2026-04-30 20:04", 182)],
    )
    bid = conn.execute("SELECT id FROM batches").fetchone()["id"]
    with pytest.raises(shift.ShiftError, match="不是实测批次"):
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=1)


def test_unknown_batch_rejected(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    with pytest.raises(shift.ShiftError, match="不存在"):
        shift.correct_batch(conn, station="S1", batch_id=999, offset_hours=1)


def test_wrong_station_rejected(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    with pytest.raises(shift.ShiftError, match="不属于站"):
        shift.correct_batch(conn, station="OTHER", batch_id=bid, offset_hours=1)


# ------------------------------------------------------- idempotency / retry


def test_same_batch_same_offset_replays_receipt(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    first = shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=3)
    second = shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=3)
    assert second.replayed
    assert (second.offset_hours, second.rows_moved, second.created_utc) == (
        first.offset_hours,
        first.rows_moved,
        first.created_utc,
    )
    # Moved once, not twice.
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T04:00:00Z",
        "2026-05-01T05:00:00Z",
    ]
    assert conn.execute("SELECT COUNT(*) c FROM time_shifts").fetchone()["c"] == 1


def test_different_offset_after_success_refused(conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=3)
    with pytest.raises(shift.ShiftError, match=r"已按 \+3 小时纠时"):
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=-1)
    assert [r["hour_utc"] for r in _hours(conn)] == ["2026-05-01T04:00:00Z"]


# -------------------------------------------------------------- atomicity


def test_failed_save_leaves_observations_untouched(conn):
    load(
        conn,
        "S1",
        observations=[
            ("2026-05-01 01:00", 100),
            ("2026-05-01 02:00", 110),
            ("2026-05-01 03:00", 120),
        ],
    )
    bid = _obs_batch_id(conn)
    # Blow up on the third UPDATE (offset -2 lands 03:00 onto 01:00), after
    # two rows have already been moved inside the transaction.
    conn.execute(
        "CREATE TRIGGER boom BEFORE UPDATE ON observation "
        "WHEN NEW.hour_utc = '2026-05-01T01:00:00Z' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(Exception):
        shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=-2)
    rows = _hours(conn)
    assert [r["hour_utc"] for r in rows] == [
        "2026-05-01T01:00:00Z",
        "2026-05-01T02:00:00Z",
        "2026-05-01T03:00:00Z",
    ]
    assert [r["level_cm"] for r in rows] == [100, 110, 120]
    assert conn.execute("SELECT COUNT(*) c FROM time_shifts").fetchone()["c"] == 0


# ------------------------------------------------------- review interaction


def test_new_review_uses_shifted_observations_old_review_keeps_snapshot(conn):
    load(
        conn,
        "S1",
        forecasts=[("2026-05-01 05:00", "2026-04-30 20:00", "2026-04-30 20:04", 182)],
        observations=[("2026-05-01 04:00", 150), ("2026-05-01 05:00", 186)],
    )
    rid_old = make_review(conn, "S1", "2026-05-01 00:00", "2026-05-01 12:00", T)
    bid = _obs_batch_id(conn)
    shift.correct_batch(conn, station="S1", batch_id=bid, offset_hours=2)
    rid_new = make_review(conn, "S1", "2026-05-01 00:00", "2026-05-01 12:00", T)

    old_rows = results_for(conn, rid_old)
    assert list(old_rows) == ["2026-05-01T05:00:00Z"]  # frozen snapshot
    assert old_rows["2026-05-01T05:00:00Z"]["warning_valid"] == 1

    new_rows = results_for(conn, rid_new)
    # Corrected observations (04→06, 05→07) with the original forecasts:
    # the crossing moved to 07:00 and no forecast targets it anymore.
    assert list(new_rows) == ["2026-05-01T07:00:00Z"]
    assert new_rows["2026-05-01T07:00:00Z"]["warning_valid"] == 0


# ------------------------------------------------------------------ web


def _post_shift(client, station, batch_id, offset):
    return client.post(
        "/shifts",
        data={"station": station, "batch_id": str(batch_id), "offset_hours": offset},
        follow_redirects=True,
    )


def test_data_page_offers_shift_form(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    text = client.get("/data?station=S1").get_data(as_text=True)
    assert "实测批次纠时" in text
    assert "偏移小时" in text


def test_shift_via_web_receipt(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    text = _post_shift(client, "S1", bid, "2").get_data(as_text=True)
    assert "纠时回执" in text
    assert f"批次 #{bid}" in text
    assert "+2 小时" in text
    assert "移动 2 行" in text
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T03:00:00Z",
        "2026-05-01T04:00:00Z",
    ]


def test_shift_via_web_conflict_lists_hours(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    load(conn, "S1", observations=[("2026-05-01 03:00", 130)])
    text = _post_shift(client, "S1", bid, "2").get_data(as_text=True)
    assert "冲突整点 2026-05-01 03:00Z" in text
    assert "整批未移动" in text
    assert [r["hour_utc"] for r in _hours(conn)] == [
        "2026-05-01T01:00:00Z",
        "2026-05-01T03:00:00Z",
    ]


def test_shift_via_web_incomplete_batch(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100), ("2026-05-01 02:00", 110)])
    bid = _obs_batch_id(conn)
    load(conn, "S1", observations=[("2026-05-01 01:00", 999)])
    text = _post_shift(client, "S1", bid, "2").get_data(as_text=True)
    assert "不完整" in text


def test_shift_via_web_replay_shows_original_receipt(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    _post_shift(client, "S1", bid, "2")
    text = _post_shift(client, "S1", bid, "2").get_data(as_text=True)
    assert "重复提交" in text
    assert f"批次 #{bid}" in text
    assert "+2 小时" in text
    assert "移动 1 行" in text
    # Still only moved once.
    assert [r["hour_utc"] for r in _hours(conn)] == ["2026-05-01T03:00:00Z"]


def test_shift_via_web_different_offset_warns_already_shifted(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    _post_shift(client, "S1", bid, "2")
    text = _post_shift(client, "S1", bid, "5").get_data(as_text=True)
    assert "已按 +2 小时纠时" in text
    assert "不能再次纠时" in text
    assert [r["hour_utc"] for r in _hours(conn)] == ["2026-05-01T03:00:00Z"]


def test_shift_via_web_rejects_bad_offset(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    text = _post_shift(client, "S1", bid, "0").get_data(as_text=True)
    assert "不能为 0" in text
    text = _post_shift(client, "S1", bid, "1.5").get_data(as_text=True)
    assert "必须是整数" in text
    assert [r["hour_utc"] for r in _hours(conn)] == ["2026-05-01T01:00:00Z"]


def test_shift_via_web_history_listed(client, conn):
    load(conn, "S1", observations=[("2026-05-01 01:00", 100)])
    bid = _obs_batch_id(conn)
    _post_shift(client, "S1", bid, "-2")
    text = client.get("/data?station=S1").get_data(as_text=True)
    assert "纠时记录" in text
    assert "-2 小时" in text
