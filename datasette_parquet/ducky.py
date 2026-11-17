import asyncio
import duckdb
import logging
import sqlite3
from .debounce import debounce
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, LoggingEventHandler
from datasette.database import Database, Results
from .ddl import create_views
from .winging_it import ProxyConnection

logger = logging.getLogger(__name__)

def convert_duckdb_error(e):
    """Convert a duckdb exception into the closest sqlite3 equivalent.

    Datasette only knows how to handle sqlite3 exceptions: a caught
    sqlite3.DatabaseError (or subclass) is reported as HTTP 400 with the
    error message in the response body, while any other exception escapes
    as an unhandled HTTP 500. So every duckdb.Error must be converted.
    """
    msg = str(e)
    if isinstance(e, duckdb.BinderException) and 'not found in FROM clause' in msg:
        # DuckDB reports both a double-quoted string literal (WHERE x = "foo")
        # and a reference to a genuinely missing column with the same message;
        # the two cases can't be distinguished, so cover both honestly.
        msg += (
            '\nNote: in DuckDB, double quotes denote identifiers (e.g. column names), '
            'not string literals. If you meant a string value, use single quotes; '
            'if you meant a column, it does not exist in this query.'
        )
    if isinstance(e, duckdb.OperationalError):
        return sqlite3.OperationalError(msg)
    if isinstance(e, duckdb.ProgrammingError):
        return sqlite3.ProgrammingError(msg)
    if isinstance(e, duckdb.InternalError):
        return sqlite3.InternalError(msg)
    # Everything else (DataError, NotSupportedError, IntegrityError,
    # FatalException, InterruptException, ...) becomes a generic
    # sqlite3.DatabaseError so no duckdb error ever escapes as a 500. These
    # are the exception types least likely to be an ordinary client mistake
    # (a wedged connection, a cancelled query, a permission error) - they'll
    # now render as an unremarkable 400 to the client, so log them here or
    # they become invisible to us.
    logger.warning('Unrecognized duckdb error converted to a generic DatabaseError: %r', e)
    return sqlite3.DatabaseError(msg)

class SchemaEventHandler(FileSystemEventHandler):
    """React to files being added/removed from the watched directory."""

    def __init__(self, reload):
        super().__init__()

        self.reload = reload

    @debounce(1)
    def on_event(self):
        self.reload()

    def on_moved(self, event):
        super().on_moved(event)
        self.on_event()

    def on_created(self, event):
        super().on_created(event)
        self.on_event()

    def on_deleted(self, event):
        super().on_deleted(event)
        self.on_event()

    def on_modified(self, event):
        super().on_modified(event)
        self.on_event()

# Datasette lets anyone run arbitrary SELECTs, which is safe on SQLite because
# SQLite can't reach outside its database file. DuckDB can: read_text(),
# read_csv() & co. take local paths and URLs, ATTACH opens other databases
# (including MySQL/Postgres), COPY ... TO writes files, and known extensions
# are installed from the internet on demand. So every connection is locked
# down at the engine level, independently of Datasette's SQL validation.
#
# These three can only be set when the database is opened.
CONNECT_CONFIG = {
    'autoinstall_known_extensions': False,
    'autoload_known_extensions': False,
    'allow_community_extensions': False,
}


def _sql_string(value):
    return "'{}'".format(value.replace("'", "''"))


def lock_down(raw_conn, allowed_directory=None):
    """Cut the connection off from the filesystem and network, for good.

    Must run after the connection has opened its own database file and
    created any views, both of which need external access. Settings are
    per-database, so cursors (duplicate connections) inherit them.
    """
    if allowed_directory is not None:
        # Directory mode: the views read their files at query time.
        raw_conn.execute('SET allowed_directories = [{}]'.format(_sql_string(allowed_directory)))
    raw_conn.execute('SET enable_external_access = false')
    # Last, so that no query can SET any of the above back.
    raw_conn.execute('SET lock_configuration = true')


def create_file_connection(file):
    raw_conn = duckdb.connect(file, read_only=True, config=CONNECT_CONFIG)
    lock_down(raw_conn)
    return ProxyConnection(raw_conn)


def create_directory_connection(directory):
    raw_conn = duckdb.connect(config=CONNECT_CONFIG)

    for create_view_stmt in create_views(directory):
        raw_conn.execute(create_view_stmt)

    lock_down(raw_conn, allowed_directory=directory)
    return ProxyConnection(raw_conn)

class DuckDatabase(Database):
    def __init__(self, ds, directory=None, file=None, watch=None):
        super().__init__(ds)

        self.engine = 'duckdb'

        if directory:
            conn = create_directory_connection(directory)

            def reload():
                self.conn.conn.close()
                self.conn = create_directory_connection(directory)

            event_handler = SchemaEventHandler(reload)
            observer = Observer()
            observer.schedule(event_handler, directory, recursive=True)
            observer.start()
        elif file:
            conn = create_file_connection(file)
        else:
            raise Exception('must specify directory or file')

        self.conn = conn

    @property
    def size(self):
        # TODO: implement this? Not sure if it's useful.
        return 0

    async def execute_fn(self, fn):
        if self.ds.executor is None:
            raise Exception('non-threaded mode not supported')

        def in_thread():
            try:
                return fn(self.conn)
            except duckdb.Error as e:
                raise convert_duckdb_error(e)

        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )

    async def execute_write_fn(self, fn, block=True):
        if self.ds.executor is None:
            raise Exception('non-threaded mode not supported')

        def in_thread():
            try:
                return fn(self.conn)
            except duckdb.Error as e:
                raise convert_duckdb_error(e)

        # We lie, we'll always block.
        return await asyncio.get_event_loop().run_in_executor(
            self.ds.executor, in_thread
        )
