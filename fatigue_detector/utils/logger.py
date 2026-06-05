"""SQLite event logger for fatigue detection events.

Logs state changes and periodic snapshots to an SQLite database for
post-session analysis and dashboard visualization.
"""

import sqlite3
import os
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional


class FatigueLogger:
    """Thread-safe SQLite logger for driver fatigue events.

    Creates a new connection for each operation to ensure thread safety.
    Logs two types of events:
    - STATE_CHANGE: recorded when the driver's state transitions
    - SNAPSHOT: periodic captures of all metrics (default every 30s)

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

        # Ensure the directory exists
        db_dir = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(db_dir, exist_ok=True)

        self._create_table()
        self.last_snapshot_time = time.time()
        self.last_state: Optional[str] = None

    @contextmanager
    def _get_connection(self):
        """Context manager for thread-safe database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_table(self) -> None:
        """Create the events table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    ear REAL,
                    mar REAL,
                    perclos REAL,
                    pitch REAL,
                    yaw REAL,
                    distracted INTEGER DEFAULT 0,
                    yawning INTEGER DEFAULT 0,
                    expression TEXT DEFAULT ''
                )
            ''')

    def log_event(self, event_type: str, state: str, ear: float,
                  mar: float, perclos: float, pitch: float, yaw: float,
                  distracted: bool, yawning: bool,
                  expression: str = '') -> None:
        """Log a single event to the database.

        Parameters
        ----------
        event_type : str
            Type of event ('STATE_CHANGE', 'SNAPSHOT', etc.)
        state : str
            Current driver state name.
        ear, mar, perclos, pitch, yaw : float
            Current metric values.
        distracted, yawning : bool
            Current flags.
        expression : str
            Detected facial expression (from CNN, if enabled).
        """
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT INTO events
                   (timestamp, event_type, state, ear, mar, perclos,
                    pitch, yaw, distracted, yawning, expression)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    datetime.now().isoformat(),
                    event_type, state,
                    ear, mar, perclos, pitch, yaw,
                    int(distracted), int(yawning), expression
                )
            )

    def log_state_change(self, state, ear: float, mar: float,
                         perclos: float, pitch: float, yaw: float,
                         distracted: bool, yawning: bool,
                         expression: str = '') -> None:
        """Log a state change event (only if state actually changed).

        Parameters
        ----------
        state : DriverState or str
            The new driver state.
        """
        state_name = state.name if hasattr(state, 'name') else str(state)

        if state_name != self.last_state:
            self.log_event(
                'STATE_CHANGE', state_name,
                ear, mar, perclos, pitch, yaw,
                distracted, yawning, expression
            )
            self.last_state = state_name

    def log_snapshot(self, state, ear: float, mar: float,
                     perclos: float, pitch: float, yaw: float,
                     distracted: bool, yawning: bool,
                     expression: str = '',
                     interval: float = 30.0) -> None:
        """Log a periodic snapshot (rate-limited by interval).

        Parameters
        ----------
        interval : float
            Minimum seconds between snapshots. Default 30.
        """
        now = time.time()
        if now - self.last_snapshot_time >= interval:
            state_name = state.name if hasattr(state, 'name') else str(state)
            self.log_event(
                'SNAPSHOT', state_name,
                ear, mar, perclos, pitch, yaw,
                distracted, yawning, expression
            )
            self.last_snapshot_time = now

    def get_recent_events(self, limit: int = 100) -> list:
        """Retrieve the most recent events from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM events ORDER BY id DESC LIMIT ?',
                (limit,)
            )
            return cursor.fetchall()

    def get_session_summary(self) -> tuple:
        """Get aggregate statistics for the current session."""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_events,
                    SUM(CASE WHEN state = 'DROWSY' THEN 1 ELSE 0 END) as drowsy_count,
                    SUM(CASE WHEN state = 'VERY_DROWSY' THEN 1 ELSE 0 END) as very_drowsy_count,
                    SUM(CASE WHEN state = 'ASLEEP' THEN 1 ELSE 0 END) as asleep_count,
                    SUM(distracted) as distraction_count,
                    SUM(yawning) as yawn_count,
                    AVG(ear) as avg_ear,
                    AVG(perclos) as avg_perclos,
                    MIN(timestamp) as session_start,
                    MAX(timestamp) as session_end
                FROM events
            ''')
            return cursor.fetchone()
