import pytest


@pytest.fixture(autouse=True)
def committed_hosts_file(monkeypatch):
    """Tests read the committed hosts file, not a PLLM_CONFIG set in the shell."""
    monkeypatch.delenv("PLLM_CONFIG", raising=False)
