"""
db_utils.py

This module provides a LiteDatabase class for querying the arvo.db SQLite database.
The database contains metadata for the RCAbench benchmark, including vulnerability reports,
reproducers, patches, and other relevant information.

The arvo table schema:
- localId: INTEGER PRIMARY KEY
- project: TEXT NOT NULL
- reproduced: BOOLEAN NOT NULL
- reproducer_vul: TEXT
- reproducer_fix: TEXT
- patch_located: BOOLEAN
- patch_url: TEXT
- verified: BOOLEAN
- fuzz_target: TEXT
- fuzz_engine: TEXT
- sanitizer: TEXT
- crash_type: TEXT
- crash_output: TEXT
- severity: TEXT
- report: TEXT
- fix_commit: TEXT
- language: TEXT
- repo_addr: TEXT DEFAULT NULL
- submodule_bug: BOOLEAN DEFAULT 0

Usage:
    from db_utils import LiteDatabase

    db = LiteDatabase('path/to/arvo.db')
    records = db.get_all_records()
    record = db.get_record_by_id(1)
"""

import sqlite3
from rcabench.__init__ import DEFAULT_DATA_DIR

class LiteDatabase:
    """
    A class to handle SQLite database operations for the arvo.db.
    """
    
    def __init__(self, db_path: str=f"{DEFAULT_DATA_DIR}/arvo.db"):
        """
        Initializes the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_all_records(self):
        """
        Retrieves all records from the arvo table.

        Returns:
            list: A list of tuples, each representing a row in the table.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM arvo")
        rows = cursor.fetchall()
        return rows

    def get_record_by_id(self, local_id):
        """
        Retrieves a single record by its localId.

        Args:
            local_id (int): The localId of the record to retrieve.

        Returns:
            tuple or None: The record as a tuple if found, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM arvo WHERE localId = ?", (local_id,))
        row = cursor.fetchone()
        return row

    def get_records_by_project(self, project):
        """
        Retrieves all records for a specific project.

        Args:
            project (str): The name of the project.

        Returns:
            list: A list of tuples for the matching records.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM arvo WHERE project = ?", (project,))
        rows = cursor.fetchall()
        return rows

    def get_verified_records(self):
        """
        Retrieves all verified records.

        Returns:
            list: A list of tuples for verified records.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM arvo WHERE verified = 1")
        rows = cursor.fetchall()
        return rows

    def get_records_by_severity(self, severity):
        """
        Retrieves all records with a specific severity level.

        Args:
            severity (str): The severity level (e.g., 'HIGH', 'MEDIUM').

        Returns:
            list: A list of tuples for the matching records.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM arvo WHERE severity = ?", (severity,))
        rows = cursor.fetchall()
        return rows
    
    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
