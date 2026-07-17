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
                    'file': './fixtures/fixtures.duckdb'
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
