"""The beat gap recorder on metrics files written by hand (#124)."""

import json

import pytest

from pllm import beatgaps, metrics, runrecord
from pllm.proto import protocol_pb2

S = 1_000_000_000


def write(path, beats, requests=(), declared=(), timeout=2.0):
    """A metrics file: the run line, beats as (stage, epoch, seconds), requests as
    (arrived, last token) in seconds, and declarations as stages."""
    lines = [{"type": "run", "heartbeat_timeout_s": timeout, "network": "test"}]
    lines += [
        {
            "type": "beat",
            "t_ns": round(t * S),
            "stage": stage,
            "epoch": epoch,
            "busy_ns": 0,
            "steps": 0,
        }
        for stage, epoch, t in beats
    ]
    lines += [
        {
            "type": "request",
            "outcome": "max_tokens",
            "output_tokens": 4,
            "arrived_ns": round(a * S),
            "first_token_ns": round(a * S) + 1,
            "last_token_ns": round(b * S),
        }
        for a, b in requests
    ]
    lines += [
        {"type": "detected", "path": "heartbeat_timeout", "stage": stage, "declared": True}
        for stage in declared
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def run_of(path):
    return beatgaps.summary(beatgaps.report.load(path), beatgaps.declarations(path), 1.5)


def test_gaps_are_counted_per_stage_and_split_into_idle_and_load(tmp_path):
    # stage 0 beats every 0.5 s from 0 to 10 s, with one 1.5 s gap ending at 6 s (in load)
    times = [t / 2 for t in range(0, 21) if t / 2 not in (5.0, 5.5)]
    beats = [(0, 1, t) for t in times] + [(1, 2, t) for t in (0.0, 0.5, 1.0)]
    s = run_of(write(tmp_path / "a.metrics.jsonl", beats, requests=[(4.0, 7.0)]))
    zero = s["stages"][0]
    assert zero["all"]["gaps"] == len(times) - 1
    assert zero["load"]["gaps"] == 5  # ending at 4.0, 4.5, 6.0, 6.5 and 7.0
    assert zero["load"]["max_s"] == pytest.approx(1.5)
    assert zero["idle"]["max_s"] == pytest.approx(0.5)
    assert zero["all"]["median_s"] == pytest.approx(0.5)
    assert s["stages"][1]["all"] == {"gaps": 2, "median_s": 0.5, "p99_s": 0.5, "max_s": 0.5}
    assert s["max_gap_s"] == pytest.approx(1.5)
    assert s["passed"] and s["declared"] == 0
    assert s["suggested_timeout_s"] == 2.5  # 1.5 x 1.5 = 2.25, rounded up to 0.5 s


def test_a_restart_starts_a_new_series(tmp_path):
    beats = [(1, 1, 0.0), (1, 1, 0.5), (1, 3, 9.0), (1, 3, 9.5)]
    s = run_of(write(tmp_path / "a.metrics.jsonl", beats))
    assert s["stages"][1]["all"]["gaps"] == 2 and s["max_gap_s"] == pytest.approx(0.5)


def test_a_declaration_fails_the_run(tmp_path):
    s = run_of(write(tmp_path / "a.metrics.jsonl", [(0, 1, 0), (0, 1, 0.5)], declared=[0]))
    assert s["declared"] == 1 and not s["passed"]


def test_a_gap_at_the_timeout_fails_the_run(tmp_path):
    s = run_of(write(tmp_path / "a.metrics.jsonl", [(0, 1, 0), (0, 1, 2.0)], timeout=2.0))
    assert not s["passed"]


def test_a_file_with_no_beats_reports_no_gaps(tmp_path):
    s = run_of(write(tmp_path / "a.metrics.jsonl", []))
    assert s["stages"] == {} and s["max_gap_s"] is None and not s["passed"]
    assert "no beat gaps" in beatgaps.render(s)


def test_main_prints_the_four_numbers_and_writes_a_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    path = write(tmp_path / "a.metrics.jsonl", [(0, 1, t / 2) for t in range(5)])
    assert beatgaps.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| 0 | all | 4 | 500.0 | 500.0 | 500.0 |" in out
    assert "verdict: pass" in out and "suggested --heartbeat-timeout" in out
    (record,) = (tmp_path / "runs").glob("*.json")
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["script"] == "beatgaps" and saved["results"][0]["max_gap_s"] == 0.5


def test_main_warns_when_the_suggestion_reaches_keepalive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    path = write(tmp_path / "a.metrics.jsonl", [(0, 1, 0), (0, 1, 6.0)], timeout=7.0)
    assert beatgaps.main([str(path)]) == 0
    assert "warning: the suggestion is not below the keepalive timeout" in capsys.readouterr().out


@pytest.mark.parametrize("margin", ["1", "0.5", "nan", "x"])
def test_main_refuses_a_margin_that_adds_nothing(margin, tmp_path):
    path = write(tmp_path / "a.metrics.jsonl", [])
    with pytest.raises(SystemExit):
        beatgaps.main([str(path), "--margin", margin])


def test_a_file_written_by_the_gateway_recorder_is_read(tmp_path):
    """The real writer, not hand-made lines: beats, and a declaration from the detector."""
    times = iter([0, 0, round(0.5 * S), round(1.25 * S), round(2.0 * S), round(9 * S)])
    path = tmp_path / "g.metrics.jsonl"
    recorder = metrics.Recorder(path, {"heartbeat_timeout_s": 2.0}, clock=lambda: next(times))
    for _ in range(4):
        recorder.beat(0, protocol_pb2.Beat(stage=0, epoch=5, busy_ns=0, steps=0))
    recorder.event({"event": "detected", "path": "stream_break", "stage": 0, "declared": True})
    recorder.finish()
    s = run_of(path)
    assert s["stages"][0]["all"]["gaps"] == 3
    assert s["max_gap_s"] == pytest.approx(0.75)
    assert s["declared"] == 1 and not s["passed"]
