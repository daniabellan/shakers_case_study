import os

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class PostgresHistoryChat:
    def __init__(self):
        self.config = {
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "host": os.getenv("PGHOST", "localhost"),
            "port": os.getenv("PGPORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "user_query_history"),
        }

    def ensure_database(self):
        temp_conn = psycopg2.connect(
            dbname="postgres",  # conexión temporal para creación
            user=self.config["user"],
            password=self.config["password"],
            host=self.config["host"],
            port=self.config["port"],
        )
        temp_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = temp_conn.cursor()

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.config["database"],))
        exists = cur.fetchone()

        if not exists:
            cur.execute(f"CREATE DATABASE {self.config['database']}")
            print(f" Base de datos '{self.config['database']}' creada.")
        else:
            print(f" Base de datos '{self.config['database']}' ya existe.")

        cur.close()
        temp_conn.close()

    def ensure_tables(self):
        conn = psycopg2.connect(**self.config)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_query_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT,
                intent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        cur.close()
        conn.close()
        print(" Tabla 'user_query_history' lista.")

    def initialize(self):
        self.ensure_database()
        self.ensure_tables()

    def log_user_query(self, user_id: str, query: str, response: str, intent: str = None):
        conn = psycopg2.connect(**self.config)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_query_history (user_id, query, response, intent)
            VALUES (%s, %s, %s, %s)
        """,
            (user_id, query, response, intent),
        )
        conn.commit()
        cur.close()
        conn.close()
