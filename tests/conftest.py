import pytest

from qquant.config import DataConfig
from qquant.data import load_snapshot


@pytest.fixture(scope="session")
def snapshot(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    return load_snapshot(DataConfig(source="synthetic", cache_dir=str(cache)))
