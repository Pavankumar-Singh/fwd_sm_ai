import os
import pytest

try:
    import server.main as appmod
except Exception:
    pytest.skip("server.main not importable", allow_module_level=True)

HAS_ASYNCPG = getattr(appmod, 'HAS_ASYNCPG', False)


pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def database_url():
    return os.getenv('DATABASE_URL')


@pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
@pytest.mark.skipif(not os.getenv('DATABASE_URL'), reason="DATABASE_URL not configured")
async def test_fetch_credit_records(database_url):
    # This test will run only when asyncpg is installed and DATABASE_URL provided.
    res = await appmod.fetch_credit_records_tool({'limit': 1})
    assert isinstance(res, dict)
    assert res.get('ok') is True
    assert isinstance(res.get('rows'), list)
