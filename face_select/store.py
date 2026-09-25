"""SQLite 캐시: 처리한 사진과 얼굴 임베딩을 저장해 중단 후 재개/기준값 재조정을 가능하게 한다."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    name TEXT,
    md5 TEXT,
    taken_time TEXT,
    status TEXT,          -- ok | error
    n_faces INTEGER,
    error TEXT,
    processed_at TEXT
);
CREATE TABLE IF NOT EXISTS faces (
    photo_id TEXT,
    idx INTEGER,
    det_score REAL,
    embedding BLOB,
    PRIMARY KEY (photo_id, idx)
);
CREATE TABLE IF NOT EXISTS exported (
    photo_id TEXT,
    folder_id TEXT,
    shortcut_id TEXT,
    PRIMARY KEY (photo_id, folder_id)
);
"""


class Store:
    def __init__(self, path: str = "cache.sqlite3"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def is_done(self, photo_id: str, md5: str | None) -> bool:
        row = self.conn.execute("SELECT md5, status FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return bool(row) and row[1] == "ok" and row[0] == md5

    def save_result(self, photo, faces) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo.id,))
            self.conn.execute(
                "INSERT OR REPLACE INTO photos VALUES (?, ?, ?, ?, 'ok', ?, NULL, ?)",
                (photo.id, photo.name, photo.md5, photo.taken_time, len(faces), now),
            )
            self.conn.executemany(
                "INSERT INTO faces VALUES (?, ?, ?, ?)",
                [(photo.id, i, f.det_score, f.embedding.astype(np.float32).tobytes()) for i, f in enumerate(faces)],
            )

    def save_error(self, photo, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO photos VALUES (?, ?, ?, ?, 'error', 0, ?, ?)",
                (photo.id, photo.name, photo.md5, photo.taken_time, error[:500], now),
            )

    def iter_photo_embeddings(self) -> Iterator[tuple[str, str, str | None, list[np.ndarray]]]:
        """(photo_id, name, taken_time, [embeddings]) — 얼굴이 1개 이상인 사진만."""
        cur = self.conn.execute(
            "SELECT p.id, p.name, p.taken_time, f.embedding FROM photos p "
            "JOIN faces f ON f.photo_id = p.id WHERE p.status = 'ok' ORDER BY p.id, f.idx"
        )
        current, name, taken, embs = None, None, None, []
        for pid, pname, ptaken, blob in cur:
            if pid != current:
                if current is not None:
                    yield current, name, taken, embs
                current, name, taken, embs = pid, pname, ptaken, []
            embs.append(np.frombuffer(blob, dtype=np.float32))
        if current is not None:
            yield current, name, taken, embs

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM photos GROUP BY status").fetchall()
        with_faces = self.conn.execute("SELECT COUNT(*) FROM photos WHERE n_faces > 0").fetchone()[0]
        return {**dict(rows), "with_faces": with_faces}

    def is_exported(self, photo_id: str, folder_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM exported WHERE photo_id = ? AND folder_id = ?", (photo_id, folder_id)
        ).fetchone()
        return row is not None

    def mark_exported(self, photo_id: str, folder_id: str, shortcut_id: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO exported VALUES (?, ?, ?)", (photo_id, folder_id, shortcut_id))
