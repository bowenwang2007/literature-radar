"""Durable personal reading state kept beside its literature database."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, timedelta
from uuid import uuid4


class ReadingDesk:
    def __init__(self, database):
        self.path = database.path.with_suffix(database.path.suffix + ".reading.sqlite")

    @contextmanager
    def writer(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS shelf (
                profile TEXT, paper INTEGER, identity TEXT, added TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(profile,paper))""")
            yield conn

    def get(self, key, default=None):
        if not self.path.exists():
            return default
        with closing(sqlite3.connect(self.path, timeout=15)) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='preferences' AND type='table'"
            ).fetchone():
                return default
            row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        if self.get(key) == value:
            return
        with self.writer() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences VALUES (?,?)", (key, json.dumps(value))
            )

    def watch(self, profile):
        return self.get("watch/" + profile, {})

    def save_watch(self, profile, keys, topics, automatic, today):
        old = self.watch(profile)
        if set(old.get("journals", [])) != set(keys):
            old = {"series": uuid4().hex, "start": (today - timedelta(days=29)).isoformat()}
        self.put("watch/" + profile, dict(old, journals=keys, topics=topics, automatic=automatic))

    def shelf(self, profile):
        if not self.path.exists():
            return {}
        with closing(sqlite3.connect(self.path, timeout=15)) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='shelf' AND type='table'"
            ).fetchone():
                return {}
            return dict(
                conn.execute(
                    "SELECT paper,identity FROM shelf WHERE profile=? "
                    "ORDER BY added DESC,paper DESC",
                    (profile,),
                )
            )

    def bookmark(self, profile, paper, enabled):
        with self.writer() as conn:
            key = (profile, int(paper["id"]))
            if enabled:
                conn.execute(
                    "INSERT OR REPLACE INTO shelf(profile,paper,identity) VALUES(?,?,?)",
                    (*key, identity(paper)),
                )
            else:
                conn.execute("DELETE FROM shelf WHERE profile=? AND paper=?", key)


def identity(paper):
    return str(paper.get("doi") or paper["title"]).strip().casefold()


def parse_topics(value):
    by_folded = {}
    for topic in re.split("[,，;；\n]", value):
        if text := topic.strip():
            by_folded.setdefault(text.casefold(), text)
    return list(by_folded.values())


def matching_topics(paper, topics):
    if not topics:
        return []
    haystack = " ".join(
        str(paper.get(k) or "") for k in ("title", "abstract", "chinese_title", "chinese_abstract")
    ).casefold()
    return [
        t
        for t in topics
        if (
            re.search(r"(?<![a-z0-9])" + re.escape(t.casefold()) + r"(?![a-z0-9])", haystack)
            if t.isascii()
            else t.casefold() in haystack
        )
    ]


def tracking_window(database, queue, profile, watch, today):
    """Only a fully successful run of this exact watch advances its frontier."""
    start = date.fromisoformat(watch["start"])
    if not queue.path.exists():
        return start, None, None
    with closing(queue.connect()) as conn:
        conn.execute("ATTACH DATABASE ? AS library", (str(database.path.resolve()),))
        conn.execute("BEGIN")
        selector = "j.profile=? AND json_extract(j.request,'$.series')=?"
        args = (profile, watch["series"])
        latest = conn.execute(
            f"SELECT j.id,j.phase,j.run_id,json_extract(j.request,'$.until') until "
            f"FROM jobs j WHERE {selector} ORDER BY j.id DESC LIMIT 1",
            args,
        ).fetchone()
        row = conn.execute(
            "SELECT sr.query_end_date FROM jobs j JOIN library.sync_runs sr ON sr.id=j.run_id "
            f"WHERE {selector} AND sr.profile_id=j.profile AND sr.started_at=j.started "
            "AND sr.status='success' ORDER BY j.id DESC LIMIT 1",
            args,
        ).fetchone()
    frontier = date.fromisoformat(row[0]) if row else None
    since = max(start, frontier - timedelta(days=6)) if frontier else start
    return min(since, today), frontier, latest


def update_watch(
    database, queue, profile, watch, journals, today, *, retry_partial=False, window=None
):
    since, frontier, latest = window or tracking_window(database, queue, profile, watch, today)
    retry_after = 0
    if latest:
        same_day = latest["until"] == today.isoformat()
        if latest["phase"] != "done" or (same_day and (frontier == today or not retry_partial)):
            return latest["id"]
        retry_after = latest["id"]
    return queue.submit(
        profile, journals, since, today, False, series=watch["series"], retry_after=retry_after
    )


def to_ris(papers, *, include_abstract=True):
    lines = []
    for paper in papers:
        lines.append("TY  - JOUR")

        def field(tag, value):
            if value:
                lines.append(tag + "  - " + " ".join(str(value).split()))

        field("TI", paper["title"])
        for author in paper.get("authors") or []:
            if isinstance(author, dict):
                author = ", ".join(
                    filter(None, (author.get("family"), author.get("given")))
                ) or author.get("name")
            field("AU", author)
        field("JO", paper.get("journal"))
        field("PY", str(paper.get("publication_date") or "")[:4])
        field("DO", paper.get("doi"))
        field("UR", paper.get("source_url"))
        field("VL", paper.get("volume"))
        field("IS", paper.get("issue"))
        if include_abstract:
            abstract = paper.get("abstract")
            if abstract and paper.get("abstract_source") == "publisher_excerpt":
                abstract = "[Abstract excerpt] " + str(abstract)
            field("AB", abstract)
        lines.extend(["ER  - ", ""])
    return "\r\n".join(lines)
