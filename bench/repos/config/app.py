"""Application that reads config — some keys drifted."""
import os

# Bug 1: DATABASE_URL was renamed to DB_URL but code still uses old name
DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://localhost:5432/mydb")

# Bug 2: APP_PORT was renamed to PORT but some code still uses APP_PORT
APP_PORT = int(os.environ.get("APP_PORT", "8080"))

# Bug 3: RETRY_COUNT renamed to MAX_RETRIES
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "3"))

def connect_db():
    # FAIL: Should use DB_URL not DATABASE_URL
    import psycopg2
    return psycopg2.connect(DATABASE_URL)  # ERROR: old config key

def start_server():
    # FAIL: Should use PORT not APP_PORT
    return f"Starting on port {APP_PORT}"
