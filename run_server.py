"""
Production entry для Биржа-цифровой.
Запускает Flask через waitress (production WSGI).
Используется Task Scheduler-ом для автозапуска.
"""
import os
import sys
import logging

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

# Логирование в файл
LOG_FILE = os.path.join(BASE, "server.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

try:
    from waitress import serve
    from app import app
    from db import init_db

    init_db()
    logging.info("Starting waitress on 0.0.0.0:5010")
    print("Биржа-цифровой: http://localhost:5010")
    serve(app, host="0.0.0.0", port=5010, threads=8)
except Exception as e:
    logging.exception("Server crashed: %s", e)
    raise
