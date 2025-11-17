from __future__ import annotations
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Input, Static
from textual.reactive import reactive
from textual.containers import Horizontal, Vertical
import sqlite3

class ChitraTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 3; }
    #content { }
    """

    db_path = reactive("photo.db")

    def __init__(self, db_path: str = "photo.db"):
        super().__init__()
        self.db_path = db_path

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            yield Input(placeholder="Search (CLIP not live here; use `search` cmd)", id="search")
        with Vertical(id="content"):
            self.table = DataTable(id="grid")
            yield self.table
        yield Footer()

    def on_mount(self):
        self.table.add_columns("ID", "Path", "Tags", "Cluster")
        self.refresh_rows()

    def refresh_rows(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT p.id, p.file_path,
                   coalesce((SELECT group_concat(tag, ', ') FROM tags t WHERE t.photo_id = p.id), ''),
                   coalesce((SELECT cluster_id FROM clusters c WHERE c.photo_id = p.id LIMIT 1), '')
            FROM photos p
            ORDER BY p.id DESC
            LIMIT 500
        """)
        rows = cur.fetchall()
        conn.close()
        self.table.clear()
        for r in rows:
            self.table.add_row(str(r[0]), r[1], r[2], str(r[3]))

def run_tui(db_path: str = "photo.db"):
    app = ChitraTUI(db_path=db_path)
    app.run()
