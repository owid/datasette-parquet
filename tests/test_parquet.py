import sqlite3
from datasette.app import Datasette
from .create_db import create_dbs
import pytest
import duckdb
import sqlglot
from datasette_parquet import rewrite as rewrite_module
from datasette_parquet.winging_it import ProxyConnection

@pytest.fixture(scope="session")
def datasette():
    create_dbs('./fixtures')
    metadata = {
        'plugins': {
            'datasette-parquet': {
                'trove': {
                    'directory': './fixtures'
                },
                'duckdb': {
                    'file': './fixtures/fixtures.duckdb',
                    'settings': {'threads': 2},
                }

            }
        }
    }

    return Datasette(
        [],
        memory=True,
        metadata=metadata,
    )

@pytest.mark.asyncio
async def test_plugin_is_installed(datasette):
    response = await datasette.client.get("/-/plugins.json")
    assert response.status_code == 200
    installed_plugins = {p["name"] for p in response.json()}
    assert "datasette-parquet" in installed_plugins

@pytest.mark.asyncio
async def test_file_mode(datasette):
    response = await datasette.client.get('/duckdb')
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_directory_mode(datasette):
    response = await datasette.client.get('/trove')
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_json_works(datasette):
    response = await datasette.client.get("/trove/fixtures.json?_size=max&_labels=on&_shape=objects")
    assert response.status_code == 200
    assert response.json()['rows'] == [{'date': '2023-01-01', 'ts': '2023-01-02T03:04:05' }]

@pytest.mark.asyncio
async def test_extraneous_parameters(datasette):
    response = await datasette.client.get("/trove?sql=select+%2A+from+fixtures&_hide_sql=1")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_sql_json(datasette):
    response = await datasette.client.get("/trove.json?sql=select+%2A+from+fixtures&_hide_sql=1")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_duckdb_table(datasette):
    response = await datasette.client.get("/duckdb/fixtures")
    assert response.status_code == 200

def test_fetchone():
    raw_conn = duckdb.connect()
    conn = ProxyConnection(raw_conn)
    fetched = conn.execute('SELECT 1 AS col').fetchone()
    assert fetched['col'] == 1


def test_double_quote_usage_for_literal_raises_binder_error():
    raw_conn = duckdb.connect()
    conn = ProxyConnection(raw_conn)

    # try reading the parquet file in trove/userdata1.parquet
    explodey_string_with_double_quotes = 'SELECT * from "./trove/userdata1.parquet" WHERE first_name = "Amanda"'

    # DuckDB's own BinderException passes through untouched; it is converted
    # to sqlite3.ProgrammingError at the DuckDatabase.execute_fn boundary.
    with pytest.raises(duckdb.BinderException):
        conn.execute(explodey_string_with_double_quotes).fetchall()


@pytest.mark.asyncio
async def test_double_quoted_literal_is_400_with_hint(datasette):
    response = await datasette.client.get(
        '/duckdb.json', params={'sql': 'select * from fixtures where date = "2023-01-01"'}
    )
    assert response.status_code == 400
    body = response.json()
    assert body['ok'] is False
    # DuckDB's original message is preserved...
    assert 'not found in FROM clause' in body['error']
    # ...and the ambiguity between literal-vs-missing-column is explained.
    assert 'single quotes' in body['error']


@pytest.mark.asyncio
async def test_missing_column_is_400(datasette):
    response = await datasette.client.get(
        '/duckdb.json', params={'sql': 'select nosuchcol from fixtures'}
    )
    assert response.status_code == 400
    assert 'nosuchcol' in response.json()['error']


@pytest.mark.asyncio
async def test_missing_table_is_400(datasette):
    response = await datasette.client.get(
        '/duckdb.json', params={'sql': 'select * from nosuchtable'}
    )
    assert response.status_code == 400
    assert 'nosuchtable' in response.json()['error']


@pytest.mark.asyncio
async def test_unparseable_sql_is_400_with_duckdb_message(datasette):
    # sqlglot can't parse this; it must fall through to DuckDB, whose parser
    # error is returned as a 400 (not an unhandled 500).
    response = await datasette.client.get(
        '/duckdb.json', params={'sql': 'select from where'}
    )
    assert response.status_code == 400
    assert 'Parser Error' in response.json()['error']


def test_rewrite_falls_back_to_original_sql_when_sqlglot_cant_parse(monkeypatch):
    # Pins the fallback branch directly (independent of whether some future
    # sqlglot version becomes lenient enough to parse 'select from where' -
    # see test_unparseable_sql_is_400_with_duckdb_message above, which
    # exercises the same behavior end-to-end but would silently stop
    # covering the fallback if that ever happens).
    def raise_sqlglot_error(*args, **kwargs):
        raise sqlglot.errors.SqlglotError('boom')

    monkeypatch.setattr(rewrite_module.sqlglot, 'transpile', raise_sqlglot_error)
    assert rewrite_module.rewrite('select from where') == 'select from where'


# The engine-level lockdown: arbitrary SELECTs must not reach outside the
# served data. Each of these is a valid SELECT that Datasette's own SQL
# validation lets through.
ESCAPES = [
    "select * from read_text('/etc/hosts')",
    "select * from read_csv('tests/fixtures.sql')",
    "select * from read_text('./fixtures/../tests/fixtures.sql')",
    "select * from read_text('http://127.0.0.1:9/')",
]


@pytest.mark.asyncio
@pytest.mark.parametrize('db', ['duckdb', 'trove'])
@pytest.mark.parametrize('sql', ESCAPES)
async def test_external_access_is_blocked(datasette, db, sql):
    response = await datasette.client.get('/{}.json'.format(db), params={'sql': sql})
    assert response.status_code == 400
    assert 'disabled by configuration' in response.json()['error']


@pytest.mark.asyncio
async def test_instance_settings_are_applied(datasette):
    response = await datasette.client.get('/duckdb.json', params={'sql': "select current_setting('threads') as t", '_shape': 'array'})
    assert response.json() == [{'t': 2}]


@pytest.mark.asyncio
@pytest.mark.parametrize('db', ['duckdb', 'trove'])
async def test_lockdown_cannot_be_undone(datasette, db):
    await datasette.invoke_startup()
    raw_conn = datasette.get_database(db).conn.conn
    with pytest.raises(duckdb.InvalidInputException, match='configuration'):
        raw_conn.execute('SET enable_external_access = true')


@pytest.mark.asyncio
@pytest.mark.parametrize('sql', [
    # sqlglot transpiles this one...
    'select 1; select 2',
    # ...and passes this one through untouched (it contains from ').
    "select * from 'fixtures/fixtures.parquet'; select 2",
])
async def test_multiple_statements_are_rejected(datasette, sql):
    response = await datasette.client.get('/duckdb.json', params={'sql': sql})
    assert response.status_code == 400
    assert 'one statement at a time' in response.json()['error']


@pytest.mark.asyncio
@pytest.mark.parametrize('db', ['duckdb', 'trove'])
async def test_only_selects_get_through(datasette, db):
    # DuckDB nests /* */ comments, SQLite and Datasette's validation don't:
    # Datasette sees a SELECT, DuckDB a single CREATE.
    sql = '/* /* */ select 1 /* */ */ CREATE TABLE smuggled (a int)'
    response = await datasette.client.get('/{}.json'.format(db), params={'sql': sql})
    assert response.status_code == 400
    assert 'may only be a SELECT' in response.json()['error']


def test_only_introspection_runs_pragmas():
    # Datasette's validation already refuses user SQL mentioning PRAGMA; this
    # is the second line, should that ever change.
    conn = ProxyConnection(duckdb.connect())
    assert conn.execute('PRAGMA show_tables').fetchall() == []
    with pytest.raises(sqlite3.ProgrammingError, match='may only be a SELECT'):
        conn.execute('/* /* */ select 1 /* */ */ PRAGMA show_tables')


# Several seconds unless interrupted, so a broken time limit fails rather than hangs.
SLOW_SQL = 'select sum(a.range * b.range) as s from range(100000000) a, range(50) b'
# A long query takes a while to parse, so with ?_timelimit=1 the deadline
# passes before DuckDB has even started it.
PADDED_SLOW_SQL = 'with pad as (select {}) {}'.format(', '.join('{} as c{}'.format(i, i) for i in range(500)), SLOW_SQL)


@pytest.mark.asyncio
@pytest.mark.parametrize('sql, timelimit', [(SLOW_SQL, 200), (PADDED_SLOW_SQL, 1)])
async def test_time_limit_interrupts_the_query(datasette, sql, timelimit):
    response = await datasette.client.get('/duckdb.json', params={'sql': sql, '_timelimit': timelimit})
    assert response.status_code == 400
    assert 'took too long' in response.json()['error']

    # The interrupt hit only that query: the database keeps serving.
    response = await datasette.client.get('/duckdb.json', params={'sql': 'select 1 as one', '_shape': 'array'})
    assert response.status_code == 200
    assert response.json() == [{'one': 1}]
