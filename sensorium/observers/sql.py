"""SQL observer via in-memory transpilation to observer tables.

This observer does not use SQLite. It treats SQL as a compact query language
over observer-derived in-memory tables (`simulation`, `transitions`, `folds`,
`particles`, `modes`, `scalars`) and executes a supported SQL subset directly.
"""

from __future__ import annotations

import numpy as np
import sqlparse

from sensorium.observers.base import ObserverProtocol


class SQLObserver(ObserverProtocol):
    """The SQLObserver is an observer that executes SQL queries against the state.
    
    Uses sqlparse to build an abstract syntax tree (AST) which is then executed and
    transpiled to be compatible with the simulation state, and other mechanisms to
    do inference via observation. This includes the ability to automatically inject
    dark particles into the system, which makes the system respond to a query that
    requires input, and hiding them from the observation process.
    """

    def __init__(self, sql_query: str):
        self.sql_query = sql_query.strip()
        self.ast = sqlparse.parse(self.sql_query)

    def observe(self) -> dict:
        """Execute SQL-like query against observer tables."""
        for node in self.ast:
            if isinstance(node, sqlparse.sql.Identifier):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Function):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Keyword):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Literal):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Operator):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Parenthesis):
                print(node.get_name())
            elif isinstance(node, sqlparse.sql.Wildcard):
                print(node.get_name())