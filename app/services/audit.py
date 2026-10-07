import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()

def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()

def get_recent_audits(limit: int = 50) -> list[dict]:
    con = sqlite3.connect(settings.audit_db_path)
    con.row_factory = sqlite3.Row
    cursor = con.cursor()
    cursor.execute(
        "SELECT id, created_at, question, source_used, trace_json FROM query_audit ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "question": r["question"],
            "source_used": r["source_used"],
            "trace": json.loads(r["trace_json"]) if r["trace_json"] else [],
        })
    con.close()
    return results