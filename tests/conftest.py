"""A throwaway `recipe_test` database, built from sql/schema.sql inside the recipe-paradedb container.

Tests that use it are skipped when the container isn't running.
"""

import subprocess

import psycopg
import pytest

from api import config

CONTAINER = "recipe-paradedb"
TEST_DB = "recipe_test"
TEST_URL = config.DATABASE_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture(scope="session")
def test_db_url():
    try:
        admin = psycopg.connect(config.DATABASE_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("recipe-paradedb is not running")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    subprocess.run(["docker", "exec", CONTAINER, "psql", "-q", "-U", "recipe", "-d", TEST_DB,
                    "-v", "ON_ERROR_STOP=1", "-f", "/docker-entrypoint-initdb.d/sql/schema.sql"],
                   check=True, capture_output=True)
    return TEST_URL


@pytest.fixture
def db(test_db_url):
    """A connection to the test DB with an empty registry and no chunk partitions."""
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        for (name,) in conn.execute("SELECT name FROM chunking_strategy").fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "chunk_{name}", "chunk_{name}_staging"')
        conn.execute("DELETE FROM chunking_strategy")
        yield conn


def add_strategy(conn, name, loaded=True):
    conn.execute("INSERT INTO chunking_strategy (name, method, parameters, loaded_at) "
                 "VALUES (%s, %s, '{}', CASE WHEN %s THEN now() END)", [name, name, loaded])
