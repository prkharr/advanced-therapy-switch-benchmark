"""Optional Snowpark transport. Importing this module never creates a connection."""

from __future__ import annotations

import importlib.util
import re

INTEGRATION_STATUS = {
    "implementation": "IMPLEMENTED",
    "environment_verification": "NOT VERIFIED IN SENTINEL",
}


def capabilities():
    try:
        available = importlib.util.find_spec("snowflake.snowpark") is not None
    except (ModuleNotFoundError, ValueError):
        available = False
    return {**INTEGRATION_STATUS, "snowpark_installed": available}


def _identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}", value):
        raise ValueError("Use an unquoted one-, two-, or three-part table identifier")
    return value


class SnowflakeAdapter:
    def __init__(self, session=None, *, use_active_session=False, max_rows=1_000_000):
        if session is None:
            if not use_active_session:
                raise ValueError("Supply a session or explicitly request the active session")
            from snowflake.snowpark.context import get_active_session

            session = get_active_session()
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.session, self.max_rows = session, int(max_rows)

    def read_tables(self, table_names):
        result = {}
        for canonical, name in table_names.items():
            # Materialization is deliberately bounded. Larger extracts must be
            # prepared upstream; silently truncating modeling data is forbidden.
            frame = self.session.table(_identifier(str(name))).limit(self.max_rows + 1).to_pandas()
            if len(frame) > self.max_rows:
                raise ValueError(f"{canonical} exceeds max_rows; prepare a bounded input upstream")
            result[canonical] = frame
        return result
