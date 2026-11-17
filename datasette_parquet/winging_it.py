import functools
import sqlite3
import threading

import duckdb

from .rewrite import rewrite, NO_OP_SQL


# A collection of classes to provide a facade that mimics the sqlite3 DB-API
# interface.
class Row:
    def __init__(self, columns, tpl):
        self.columns = columns
        self.tpl = tpl

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tpl[key]
        else:
            return self.tpl[self.columns[key]]

def fixup_params(sql, parameters):
    # Sometimes we skip queries that DuckDB can't handle, eg DATE(...) facet queries.
    # If the old query had parameters, sending them with the new query will
    # cause an assertion to fail. So return an empty list of parameters.
    if sql == NO_OP_SQL:
        return sql, []

    if isinstance(parameters, (tuple, list)):
        return sql, parameters

    if isinstance(parameters, dict):
        # from sqlglot 21.2 onwards (Feb 2024), :foo is replaced with $foo already
        # (previously we had to do this here)

        # On the custom SQL page, Datasette jams any query parameter it finds
        # into the parameters for the backend. DuckDB is strict on unexpected
        # parameters, so try to remove them
        parameters = {k: v for k, v in parameters.items() if '${}'.format(k) in sql}

    return sql, parameters

# PRAGMA is for Datasette's own introspection (PRAGMA table_info); those that
# change settings are typed SET, or refused by lock_configuration.
ALLOWED_STATEMENT_TYPES = {duckdb.StatementType.SELECT, duckdb.StatementType.PRAGMA}

def prepare(raw_conn, sql, parameters):
    """Translate Datasette's SQLite-flavoured SQL for DuckDB, and only let one read through.

    Datasette's validation only checks how the SQL starts, which doesn't hold
    for DuckDB: its execute() runs every ;-separated statement, and it nests
    /* */ comments, so '/* /* */ select 1 */ CREATE ...' reads as a SELECT to
    Datasette but is a CREATE to DuckDB. So check what DuckDB will run.
    """
    sql = rewrite(sql)
    statements = raw_conn.extract_statements(sql)
    if len(statements) > 1:
        raise sqlite3.ProgrammingError('You can only execute one statement at a time.')
    if statements and statements[0].type not in ALLOWED_STATEMENT_TYPES:
        raise sqlite3.ProgrammingError('Statement may only be a SELECT.')
    return fixup_params(sql, parameters)

def sqlite_interrupt(method):
    """Report an interrupted query the way sqlite3 does, so Datasette shows its time-limit error."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except duckdb.InterruptException as e:
            raise sqlite3.OperationalError('interrupted') from e
    return wrapper

class Watchdog:
    """Stands in for SQLite's progress handler, which DuckDB doesn't have.

    Datasette enforces its time limits (sql_time_limit_ms, ?_timelimit=) by
    installing a progress handler that returns true once the deadline passes.
    This thread polls that handler instead and interrupts the watched cursors.
    It keeps interrupting until stopped: DuckDB ignores an interrupt that
    arrives before the query starts (say, while sqlglot parses a long query
    under ?_timelimit=1).
    """
    POLL_SECONDS = 0.005

    def __init__(self, handler):
        self.handler = handler
        self.cursors = []
        self.stopped = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def watch(self, raw_cursor):
        self.cursors.append(raw_cursor)

    def run(self):
        while not self.stopped.wait(self.POLL_SECONDS):
            if self.handler():
                for raw_cursor in list(self.cursors):
                    raw_cursor.interrupt()

    def stop(self):
        self.stopped.set()

class ProxyCursor:
    def __init__(self, conn, existing_cursor=None):
        self.conn = conn

        if existing_cursor:
            self.cursor = existing_cursor
        else:
            self.cursor = self.conn.cursor()

    @sqlite_interrupt
    def execute(self, sql, parameters=None):
        sql, parameters = prepare(self.cursor, sql, parameters)
        return self.cursor.execute(sql, parameters)

    @sqlite_interrupt
    def fetchone(self):
        tpl = self.cursor.fetchone()

        if not tpl:
            return tpl

        columns = {}
        for i, x in enumerate(self.cursor.description):
            columns[x[0]] = i

        return Row(columns, tpl)

    @sqlite_interrupt
    def fetchmany(self, size=1):
        tpls = self.cursor.fetchmany(size)
        columns = {}
        for i, x in enumerate(self.cursor.description):
            columns[x[0]] = i

        return [Row(columns, tpl) for tpl in tpls]

    @sqlite_interrupt
    def fetchall(self):
        tpls = self.cursor.fetchall()
        columns = {}
        for i, x in enumerate(self.cursor.description):
            columns[x[0]] = i

        return [Row(columns, tpl) for tpl in tpls]

    def __iter__(self):
        return self

    @sqlite_interrupt
    def __next__(self):
        rv = self.cursor.fetchone()

        if rv == None:
            raise StopIteration

        columns = {}
        for i, x in enumerate(self.cursor.description):
            columns[x[0]] = i
        rv = Row(columns, rv)
        return rv

    def __getattr__(self, name):
        return getattr(self.cursor, name)

class ProxyConnection:
    def __init__(self, conn):
        self.conn = conn
        # Datasette shares one connection across its query threads, and each
        # thread installs its own time limit.
        self.local = threading.local()

    def __enter__(self):
        pass

    def __exit__(self, exc_type,exc_value, exc_traceback):
        pass

    def execute(self, sql, parameters=None):
        sql, parameters = prepare(self.conn, sql, parameters)
        rv = self.conn.execute(sql, parameters)

        return ProxyCursor(self.conn, rv)

    def fetchall(self):
        raise Exception('TODO: ProxyConnection.fetchall is not implemented')

    def set_progress_handler(self, handler, n):
        watchdog = getattr(self.local, 'watchdog', None)
        if watchdog:
            watchdog.stop()
        self.local.watchdog = Watchdog(handler) if handler else None

    def cursor(self):
        cursor = ProxyCursor(self.conn)
        # Datasette runs time-limited queries on a fresh cursor, which is a
        # DuckDB connection of its own, so interrupting it leaves the queries
        # of other threads alone.
        watchdog = getattr(self.local, 'watchdog', None)
        if watchdog:
            watchdog.watch(cursor.cursor)
        return cursor
