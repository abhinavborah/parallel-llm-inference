import pytest

from pllm import config

VALID = """
network = "test-lan"

[gateway]
host = "10.0.0.1"
port = 50050

[stage0]
host = "10.0.0.2"
port = 50051

[stage1]
host = "10.0.0.1"
port = 50052
"""


def write(tmp_path, text):
    path = tmp_path / "hosts.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_committed_file_is_valid_single_host():
    cfg = config.load(config.DEFAULT_PATH)
    assert cfg.network == "single-host"
    for role in config.ROLES:
        assert getattr(cfg, role).host == "127.0.0.1"


def test_load_reads_every_role(tmp_path):
    cfg = config.load(write(tmp_path, VALID))
    assert cfg.network == "test-lan"
    assert cfg.gateway.address == "10.0.0.1:50050"
    assert cfg.stage0.address == "10.0.0.2:50051"
    assert cfg.stage1.address == "10.0.0.1:50052"


def test_environment_variable_selects_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PLLM_CONFIG", str(write(tmp_path, VALID)))
    assert config.load().network == "test-lan"


def test_default_path_without_environment_variable(monkeypatch):
    monkeypatch.delenv("PLLM_CONFIG", raising=False)
    assert config.config_path() == config.DEFAULT_PATH


@pytest.mark.parametrize(
    "old, new, message",
    [
        ('[stage1]\nhost = "10.0.0.1"\nport = 50052\n', "", r"missing \[stage1\]"),
        ("port = 50051", "port = 70000", "port must be"),
        ("port = 50051", 'port = "50051"', "port must be"),
        ("port = 50051", "port = true", "port must be"),
        ('host = "10.0.0.2"', 'host = ""', "host must be"),
        ('network = "test-lan"', "", "network"),
        ("port = 50052", "port = 50050", "share 10.0.0.1:50050"),
    ],
)
def test_invalid_files_are_rejected(tmp_path, old, new, message):
    assert old in VALID
    with pytest.raises(ValueError, match=message):
        config.load(write(tmp_path, VALID.replace(old, new)))


def test_the_dashboard_listens_on_the_gateways_host_unless_the_file_says_otherwise(tmp_path):
    assert config.dashboard_endpoint(write(tmp_path, VALID)) == config.Endpoint(
        "10.0.0.1", config.DASHBOARD_PORT
    )
    table = VALID + '\n[dashboard]\nhost = "10.0.0.3"\nport = 8080\n'
    assert config.dashboard_endpoint(write(tmp_path, table)).address == "10.0.0.3:8080"


@pytest.mark.parametrize(
    "table",
    ['host = ""\nport = 8080', 'host = "10.0.0.3"\nport = 0', 'host = "10.0.0.3"\nport = true'],
)
def test_a_bad_dashboard_table_is_rejected(tmp_path, table):
    with pytest.raises(ValueError, match="dashboard"):
        config.dashboard_endpoint(write(tmp_path, VALID + "\n[dashboard]\n" + table + "\n"))


def test_the_committed_file_puts_the_dashboard_on_loopback():
    endpoint = config.dashboard_endpoint(config.DEFAULT_PATH)
    assert endpoint == config.Endpoint("127.0.0.1", config.DASHBOARD_PORT)
