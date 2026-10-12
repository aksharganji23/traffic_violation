import sqlite3
from datetime import datetime


class ViolationDatabase:

    def __init__(self, database="traffic.db"):

        self.connection = sqlite3.connect(
            database,
            check_same_thread=False
        )

        self.create_table()

    def create_table(self):

        query = """
        CREATE TABLE IF NOT EXISTS violations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            vehicle_number TEXT,

            violation_type TEXT,

            confidence REAL,

            timestamp TEXT,

            evidence_path TEXT

        )
        """

        self.connection.execute(query)

        self.connection.commit()

    def add_violation(
        self,
        vehicle_number,
        violation_type,
        confidence,
        evidence_path
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        query = """
        INSERT INTO violations
        (
            vehicle_number,
            violation_type,
            confidence,
            timestamp,
            evidence_path
        )
        VALUES (?, ?, ?, ?, ?)
        """

        self.connection.execute(
            query,
            (
                vehicle_number,
                violation_type,
                confidence,
                timestamp,
                evidence_path
            )
        )

        self.connection.commit()

    def get_violations(self):

        cursor = self.connection.execute(
            """
            SELECT *
            FROM violations
            ORDER BY id DESC
            """
        )

        return cursor.fetchall()