import asyncio
import duckdb
import logging
import re
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

# Why each of these matters: README.md, Security.
LOCKDOWN = [
    # Before external access goes off, which would otherwise allow the temp
    # directory on its own, and with it other queries' spill files.
    ('temp_directory', ''),
    ('enable_external_access', False),
    # Redundant with external access off; kept as defence in depth.
    ('autoinstall_known_extensions', False),
    ('autoload_known_extensions', False),
    ('allow_community_extensions', False),
    # Last, so that no query can SET any of the above back.
    ('lock_configuration', True),
]

# Where httpfs looks for the system's CA bundle (as DuckDB 2.0 lists them).
# With external access off it refuses to read one that isn't allowed, so
# HTTPS fails without them. They hold only public certificates.
CA_BUNDLES = [
    '/etc/ssl/certs/ca-certificates.crt',
    '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem',
    '/etc/pki/tls/certs/ca-bundle.crt',
    '/etc/ssl/ca-bundle.pem',
    '/etc/ssl/cert.pem',
]

# DuckDB matches allowed_directories as plain string prefixes, so a URL
# prefix has to end its host with a slash: 'https://example.com' would also
# allow 'https://example.com.evil.net/' and 'https://example.com@evil.net/'.
# And it has to stop there: DuckDB resolves '..' only after the check, so
# 'https://example.com/data/../' reads outside '/data/' all the same.
ORIGIN_RE = re.compile(r'https?://[^/?#@\\]+/')


def check_urls(urls):
    if isinstance(urls, str):
        raise ValueError('datasette-parquet: urls must be a list of origins')
    for url in urls:
        if not ORIGIN_RE.fullmatch(url):
            raise ValueError("datasette-parquet: urls must be origins like 'https://example.com/', got {!r}".format(url))
    return list(urls)


def enable_urls(raw_conn):
    """Let queries read over HTTP(S), from the origins that lock_down allows."""
    # Secrets are disabled along with external access anyway; without this,
    # the first HTTP read tries to open ~/.duckdb/stored_secrets and fails.
    raw_conn.execute('SET allow_persistent_secrets = false')
    try:
        raw_conn.execute('LOAD httpfs')
    except duckdb.IOException as e:
        raise RuntimeError(
            'datasette-parquet: urls needs DuckDB\'s httpfs extension installed '
            'ahead of time (INSTALL httpfs), since the lockdown stops DuckDB installing it'
        ) from e
    raw_conn.execute('SET allowed_paths = ?', [CA_BUNDLES])


def lock_down(raw_conn, allowed_directory=None, settings=None, urls=()):
    """Cut the connection off from the filesystem and network, for good.

    `settings` are the instance's own DuckDB settings (say, threads and
    memory_limit), applied first so the lockdown always wins and no query can
    change them either. Must run after the connection has opened its own
    database file and created any views, both of which need external access.
    They're SET GLOBAL because Datasette runs queries on cursors, which are
    sessions of their own: a session-scoped setting would reach none of them,
    so DuckDB refusing one at startup beats ignoring it.

    `urls` are the origins (scheme, host and port) queries may still read from.
    """
    allowed = check_urls(urls)
    if allowed:
        # First, so that settings can include httpfs's own (http_timeout).
        enable_urls(raw_conn)
    for name, value in (settings or {}).items():
        raw_conn.execute('SET GLOBAL {} = ?'.format(name), [value])
    if allowed_directory is not None:
        # Directory mode: the views read their files at query time.
        allowed.append(allowed_directory)
    if allowed:
        raw_conn.execute('SET allowed_directories = ?', [allowed])
    for name, value in LOCKDOWN:
        raw_conn.execute('SET {} = ?'.format(name), [value])


def create_file_connection(file, settings=None, urls=()):
    raw_conn = duckdb.connect(file, read_only=True)
    lock_down(raw_conn, settings=settings, urls=urls)
    return ProxyConnection(raw_conn)


def create_directory_connection(directory, settings=None, urls=()):
    raw_conn = duckdb.connect()

    for create_view_stmt in create_views(directory):
        raw_conn.execute(create_view_stmt)

    lock_down(raw_conn, allowed_directory=directory, settings=settings, urls=urls)
    return ProxyConnection(raw_conn)

class DuckDatabase(Database):
    def __init__(self, ds, directory=None, file=None, watch=None, settings=None, urls=()):
        super().__init__(ds)

        self.engine = 'duckdb'

        if directory:
            conn = create_directory_connection(directory, settings, urls)

            def reload():
                self.conn.conn.close()
                self.conn = create_directory_connection(directory, settings, urls)

            event_handler = SchemaEventHandler(reload)
            observer = Observer()
            observer.schedule(event_handler, directory, recursive=True)
            observer.start()
        elif file:
            conn = create_file_connection(file, settings, urls)
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
