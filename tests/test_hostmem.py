import json

import pytest

from pllm import hostmem, runrecord

GIB = hostmem.GIB


@pytest.mark.parametrize(
    "available, headroom, expected",
    [
        (int(3.07 * GIB), GIB // 2, int(2.5 * GIB) // hostmem.STEP * hostmem.STEP),
        (GIB, GIB // 2, 5 * hostmem.STEP),
        (GIB // 4, GIB // 2, 0),  # less than the headroom gives no budget, not a negative one
    ],
)
def test_budget_is_available_less_headroom_rounded_down(available, headroom, expected):
    assert hostmem.budget(available, headroom) == expected
    assert expected % hostmem.STEP == 0


def test_sample_reads_once_per_interval():
    values = iter([5, 3, 4])
    assert hostmem.sample(0.03, 0.01, read=lambda: next(values)) == [5, 3, 4]


def test_sample_reads_at_least_once():
    assert hostmem.sample(0.001, 1.0, read=lambda: 7) == [7]


def test_top_processes_are_largest_first():
    found = hostmem.top_processes(5)
    assert 0 < len(found) <= 5
    sizes = [p["rss_bytes"] for p in found]
    assert sizes == sorted(sizes, reverse=True)


def test_the_command_records_the_minimum_and_the_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path)
    readings = [3 * GIB, int(2.6 * GIB), 3 * GIB]
    monkeypatch.setattr(hostmem, "sample", lambda seconds, interval: readings)
    hostmem.main(["--seconds", "3", "--headroom-gib", "0.5"])

    (path,) = tmp_path.glob("*-host-memory.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["available_min_bytes"] == int(2.6 * GIB)
    assert rec["budget_bytes"] == hostmem.budget(int(2.6 * GIB), GIB // 2)
    assert rec["samples"] == 3
    assert rec["top_processes"]
    assert "worker budget" in capsys.readouterr().out


def test_bad_arguments_are_rejected():
    with pytest.raises(SystemExit):
        hostmem.main(["--seconds", "0"])
    with pytest.raises(SystemExit):
        hostmem.main(["--headroom-gib", "-1"])
