"""Observation-batch time correction (实测批次纠时).

A drifting tide-gauge clock stamps a whole batch of observations at the wrong
whole hour.  This module moves every row of one observation batch by a fixed
non-zero whole-hour offset, atomically, and records the correction so that:

* a retry with the same batch and offset replays the original receipt instead
  of shifting the batch a second time;
* submitting a different offset for an already-corrected batch is refused.

Only the batch's own rows and their landing hours are checked, against the
data committed before the write (one ``BEGIN IMMEDIATE`` transaction).  If
anything fails, the transaction rolls back and the observations stay as they
were.  Forecasts and saved review snapshots are never touched, so new reviews
see the corrected observations alongside the original forecasts, while old
reviews keep reading their frozen snapshot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from . import db, times

# Sanity bound on the offset, about one year in hours.  A drifting clock is
# hours to days off; this only stops typos from throwing rows into far dates.
MAX_SHIFT_HOURS = 24 * 367

_OFFSET_RE = re.compile(r"^[+-]?\d+$")


class ShiftError(ValueError):
    """A refused correction, carrying a human readable explanation.

    ``conflicts`` lists the occupied landing hours when the refusal is about
    colliding observations, so the UI can enumerate them.
    """

    def __init__(self, message: str, conflicts: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.conflicts = conflicts or []


@dataclass
class ShiftReceipt:
    """What a successful (or replayed) correction reports back."""

    batch_id: int
    station: str
    offset_hours: int
    rows_moved: int
    created_utc: str
    replayed: bool = False  # True when returned for a repeat submission


def parse_offset(raw: str | None) -> int:
    """Validate the user supplied offset: a non-zero whole number of hours."""
    text = (raw or "").strip()
    if not text:
        raise ShiftError("请填写偏移小时数（非零整数）")
    if not _OFFSET_RE.match(text):
        raise ShiftError(f"偏移小时数必须是整数：{raw!r}")
    offset = int(text, 10)
    if offset == 0:
        raise ShiftError("偏移不能为 0：纠时必须把整批移到别的整点")
    if abs(offset) > MAX_SHIFT_HOURS:
        raise ShiftError(f"偏移超出合理范围（±{MAX_SHIFT_HOURS} 小时）：{offset}")
    return offset


def correct_batch(conn, *, station: str, batch_id: int, offset_hours: int) -> ShiftReceipt:
    """Move one observation batch by ``offset_hours`` and record the shift.

    Raises :class:`ShiftError` on any refusal; nothing is written in that
    case.  On success the whole batch sits at the new whole hours with its
    levels and batch number unchanged.
    """
    now = times.fmt(times.utc_now())
    with db.transaction(conn):
        batch = conn.execute(
            "SELECT id, station, kind, rows_ok FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ShiftError(f"批次 #{batch_id} 不存在")
        if batch["station"] != station:
            raise ShiftError(f"批次 #{batch_id} 不属于站 {station}")
        if batch["kind"] != "observation":
            raise ShiftError(f"批次 #{batch_id} 不是实测批次，纠时只针对实测批次")

        prior = conn.execute(
            "SELECT batch_id, offset_hours, rows_moved, created_utc FROM time_shifts "
            "WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if prior is not None:
            if prior["offset_hours"] == offset_hours:
                # Same batch, same offset: replay the original receipt rather
                # than shifting a second time.
                return ShiftReceipt(
                    batch_id=batch_id,
                    station=station,
                    offset_hours=prior["offset_hours"],
                    rows_moved=prior["rows_moved"],
                    created_utc=prior["created_utc"],
                    replayed=True,
                )
            raise ShiftError(
                f"批次 #{batch_id} 已按 {prior['offset_hours']:+d} 小时纠时"
                f"（{times.display(prior['created_utc'])}），不能再次纠时"
            )

        rows = conn.execute(
            "SELECT id, hour_utc FROM observation "
            "WHERE station = ? AND batch_id = ? ORDER BY hour_utc",
            (station, batch_id),
        ).fetchall()
        if not rows:
            raise ShiftError(f"批次 #{batch_id} 没有可移动的实测行")
        if len(rows) != batch["rows_ok"]:
            # A later backfill overwrote some of this batch's rows (they now
            # carry the newer batch id), so the batch no longer sits where it
            # was written and must not be shifted as a whole.
            raise ShiftError(
                f"批次 #{batch_id} 不完整：入库 {batch['rows_ok']} 行，现存 {len(rows)} 行，"
                "补档已覆盖本批记录，拒绝纠时"
            )

        delta = timedelta(hours=offset_hours)
        moves = [
            (row["id"], row["hour_utc"], times.fmt(times.parse_stored(row["hour_utc"]) + delta))
            for row in rows
        ]

        # Landing hours may coincide with this batch's own original hours (the
        # whole batch moves together), but not with rows of other batches or
        # unbatched observations.
        targets = [new for _, _, new in moves]
        placeholders = ",".join("?" for _ in targets)
        conflicts = conn.execute(
            f"SELECT hour_utc FROM observation "
            f"WHERE station = ? AND hour_utc IN ({placeholders}) "
            f"AND (batch_id IS NULL OR batch_id != ?) "
            f"ORDER BY hour_utc",
            (station, *targets, batch_id),
        ).fetchall()
        if conflicts:
            hours = [row["hour_utc"] for row in conflicts]
            raise ShiftError(
                f"落点已有其他批次或未归批的实测，共 {len(hours)} 个冲突整点，整批未移动",
                conflicts=hours,
            )

        # Update in the direction of the shift (positive: latest hour first)
        # so rows of this batch never collide with each other mid-update.
        ordered = sorted(moves, key=lambda move: move[1], reverse=offset_hours > 0)
        for row_id, _old_hour, new_hour in ordered:
            conn.execute(
                "UPDATE observation SET hour_utc = ?, updated_utc = ? WHERE id = ?",
                (new_hour, now, row_id),
            )
        conn.execute(
            "INSERT INTO time_shifts (batch_id, station, offset_hours, rows_moved, created_utc) "
            "VALUES (?, ?, ?, ?, ?)",
            (batch_id, station, offset_hours, len(moves), now),
        )
    return ShiftReceipt(
        batch_id=batch_id,
        station=station,
        offset_hours=offset_hours,
        rows_moved=len(moves),
        created_utc=now,
    )
