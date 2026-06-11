"""
Биржа-цифровой — SQLite хранение истории отчётов.
"""
import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "birza_digital.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            exchange TEXT,
            timeframe TEXT,
            analysis_type TEXT,
            ai_provider TEXT DEFAULT 'claude',
            price REAL,
            n_bars INTEGER,
            report_text TEXT,
            report_json TEXT,
            ai_response TEXT,
            csv_filename TEXT,
            elapsed_sec REAL
        )
    """)
    conn.commit()
    conn.close()


def save_report(ticker, exchange, timeframe, analysis_type, ai_provider,
                price, n_bars, report_text, report_data, ai_response,
                csv_filename, elapsed_sec):
    conn = get_db()
    data_clean = _clean_for_json(report_data)
    from datetime import datetime
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO reports (created_at, ticker, exchange, timeframe, analysis_type,
                            ai_provider, price, n_bars, report_text,
                            report_json, ai_response, csv_filename, elapsed_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_local, ticker, exchange, timeframe, analysis_type, ai_provider,
          price, n_bars, report_text,
          json.dumps(data_clean, ensure_ascii=False, default=str),
          ai_response, csv_filename, elapsed_sec))
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return row_id


def get_report(report_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return row


def get_all_reports():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, created_at, ticker, exchange, timeframe, analysis_type, "
        "ai_provider, price, n_bars, csv_filename, elapsed_sec "
        "FROM reports ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def delete_report(report_id):
    conn = get_db()
    conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


def _clean_for_json(obj):
    """Рекурсивно очищает объект для JSON-сериализации."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        skip_keys = {"df", "close", "high", "low", "open_", "volume",
                     "atr", "returns", "clean_returns", "outlier_mask"}
        return {k: _clean_for_json(v) for k, v in obj.items()
                if not callable(v) and k not in skip_keys}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size > 100:
            return f"[array shape={obj.shape}]"
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif callable(obj):
        return "[function]"
    return obj
