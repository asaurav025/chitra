#!/usr/bin/env python3
"""
Report — and optionally delete — rows orphaned by foreign keys that were never
enforced.

Background: `PRAGMA foreign_keys` is per-connection and defaults to OFF in
SQLite, and neither connection helper used to set it. Every `ON DELETE CASCADE`
in the schema was therefore a no-op, so deleting a photo left its tags,
embeddings, clusters and faces behind. Enabling the pragma stops new orphans
appearing; it does not retroactively clean up the ones already there. This
script does that one-off cleanup.

Dry run by default — it only ever reports. Deletion requires an explicit
`--apply`. It is idempotent: a second run finds nothing.

Relations are read out of the schema itself (`PRAGMA foreign_key_list`) rather
than hardcoded, so a new table with a foreign key is covered automatically, and
each relation is handled according to the action its own schema declares:

    ON DELETE CASCADE     the orphan child row is deleted
    ON DELETE SET NULL    the dangling column is set to NULL, row kept
    anything else         reported only, never touched

Usage:
    python scripts/cleanup_orphans.py                    # dry run on photo.db
    python scripts/cleanup_orphans.py --db /tmp/copy.db  # dry run elsewhere
    python scripts/cleanup_orphans.py --apply            # actually delete
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import List, NamedTuple


DB_DEFAULT_PATH = os.environ.get("CHITRA_DB_PATH", "photo.db")

# Actions we know how to repair. Anything else is reported and left alone.
ACTION_DELETE = "CASCADE"
ACTION_SET_NULL = "SET NULL"


class Relation(NamedTuple):
    """One foreign key: child.column -> parent.column, with its delete action."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    on_delete: str

    def __str__(self) -> str:
        return (
            f"{self.child_table}.{self.child_column} -> "
            f"{self.parent_table}.{self.parent_column}"
        )


def discover_relations(conn: sqlite3.Connection) -> List[Relation]:
    """Read every foreign key declared in the schema."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]

    relations: List[Relation] = []
    for table in tables:
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            # (id, seq, parent_table, from_column, to_column, on_update, on_delete, match)
            parent_table, child_column, parent_column = row[2], row[3], row[4]
            if parent_column is None:
                # Implicit reference to the parent's primary key.
                pk = conn.execute(f'PRAGMA table_info("{parent_table}")').fetchall()
                parent_column = next((c[1] for c in pk if c[5]), "rowid")
            relations.append(
                Relation(table, child_column, parent_table, parent_column, row[6])
            )
    return relations


def count_orphans(conn: sqlite3.Connection, rel: Relation) -> int:
    """How many child rows point at a parent that no longer exists."""
    sql = (
        f'SELECT COUNT(*) FROM "{rel.child_table}" '
        f'WHERE "{rel.child_column}" IS NOT NULL '
        f'AND "{rel.child_column}" NOT IN '
        f'(SELECT "{rel.parent_column}" FROM "{rel.parent_table}")'
    )
    return conn.execute(sql).fetchone()[0]


def repair(conn: sqlite3.Connection, rel: Relation) -> int:
    """Delete or NULL the orphans for one relation. Returns rows affected."""
    where = (
        f'WHERE "{rel.child_column}" IS NOT NULL '
        f'AND "{rel.child_column}" NOT IN '
        f'(SELECT "{rel.parent_column}" FROM "{rel.parent_table}")'
    )
    if rel.on_delete == ACTION_SET_NULL:
        sql = f'UPDATE "{rel.child_table}" SET "{rel.child_column}" = NULL {where}'
    else:
        sql = f'DELETE FROM "{rel.child_table}" {where}'
    return conn.execute(sql).rowcount


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report or clean up rows orphaned by unenforced foreign keys.",
    )
    parser.add_argument(
        "--db",
        default=DB_DEFAULT_PATH,
        help=f"database to inspect (default: {DB_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete the orphans. Without this the script only reports.",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"error: no such database: {args.db}", file=sys.stderr)
        return 2

    mode = "APPLY (rows will be deleted)" if args.apply else "DRY RUN (no changes)"
    print(f"Database: {args.db}")
    print(f"Mode:     {mode}\n")

    conn = sqlite3.connect(args.db)
    try:
        # Enforce foreign keys for this connection so that deleting an orphan
        # parent cascades to its own children in the same pass.
        conn.execute("PRAGMA foreign_keys=ON")
        if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            print("error: could not enable PRAGMA foreign_keys", file=sys.stderr)
            return 2

        relations = discover_relations(conn)
        if not relations:
            print("No foreign keys declared in this schema; nothing to check.")
            return 0

        violations_before = len(conn.execute("PRAGMA foreign_key_check").fetchall())

        total = 0
        skipped = 0
        # Repeat to a fixed point: deleting an orphan face orphans its
        # face_thumbs row, which the next pass then picks up.
        for pass_number in range(1, 11):
            pass_total = 0
            for rel in relations:
                found = count_orphans(conn, rel)
                if not found:
                    continue

                if rel.on_delete in (ACTION_DELETE, ACTION_SET_NULL):
                    verb = "delete" if rel.on_delete == ACTION_DELETE else "null out"
                    if args.apply:
                        affected = repair(conn, rel)
                        print(f"  {rel}: {verb}d {affected} orphan row(s)")
                        pass_total += affected
                    else:
                        print(
                            f"  {rel}: {found} orphan row(s) would be {verb}d "
                            f"(ON DELETE {rel.on_delete})"
                        )
                        pass_total += found
                else:
                    print(
                        f"  {rel}: {found} orphan row(s) — ON DELETE "
                        f"{rel.on_delete or 'NO ACTION'}, left alone; "
                        f"needs a human decision"
                    )
                    skipped += found

            total += pass_total
            if not args.apply or pass_total == 0:
                break
            print(f"  -- pass {pass_number} affected {pass_total} row(s) --")
        else:
            print("warning: still finding orphans after 10 passes", file=sys.stderr)

        if args.apply:
            conn.commit()

        violations_after = len(conn.execute("PRAGMA foreign_key_check").fetchall())

        print()
        print(f"Foreign key violations before: {violations_before}")
        if args.apply:
            print(f"Foreign key violations after:  {violations_after}")
            print(f"Rows repaired:                 {total}")
        else:
            print(f"Rows that would be repaired:   {total}")
        if skipped:
            print(f"Rows left for a human:         {skipped}")

        if total == 0 and skipped == 0:
            print("\nNothing to do — no orphans found.")
        elif not args.apply:
            print("\nDry run: nothing was changed. Re-run with --apply to clean up.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
