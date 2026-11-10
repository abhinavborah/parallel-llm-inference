"""The report: every plot and table from the metrics files (R7.3 to R7.6).

Most tests build a metrics file by hand with known times, so each number can be checked
exactly. The last ones run a real gateway and two real stage servers on loopback, with
their heartbeats, and build the report from the file the gateway wrote.
"""

import csv
import json
import random
import threading
import time

import numpy
import pytest
from test_fencing import nack, prefill, send
from test_gateway import WAIT, System, wait_for
from test_heartbeat import Beating
from test_metrics import lines, of_type
from test_model import tiny_reference

from pllm import report
from pllm.metrics import Recorder

MS = 1_000_000  # nanoseconds
S = 1_000_000_000


def request(rid, arrived, first, last, tokens, outcome="max_tokens"):
    return {
        "type": "request",
        "t_ns": (last or arrived) + 1,
        "outcome": outcome,
        "code": None,
        "rid": rid,
        "output_tokens": tokens,
        "arrived_ns": arrived,
        "first_token_ns": first,
        "last_token_ns": last,
    }


def beat(t, stage, busy, steps=0, epoch=1):
    return dict(type="beat", t_ns=t, stage=stage, epoch=epoch, busy_ns=busy, steps=steps)


def write(path, rows, end=True):
    rows = [{"type": "run", "clock": "perf_counter_ns", "start_ns": 0, "model": "tiny"}, *rows]
    if end:
        rows.append({"type": "run_end", "t_ns": 10**12, "stage0_stage1_complete": True})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


# -------------------------------------------------------------- definitions


def test_percentiles_match_numpys_default():
    rng = random.Random(0)
    for n in (1, 2, 3, 7, 100, 101):
        values = [rng.uniform(0, 50) for _ in range(n)]
        for p in (0, 1, 50, 95, 99, 100):
            assert report.percentile(values, p) == pytest.approx(numpy.percentile(values, p))
    with pytest.raises(ValueError):
        report.percentile([], 50)


def test_ttft_tpot_and_end_to_end_are_measured_from_arrival_at_the_gateway(tmp_path):
    run = report.load(
        write(
            tmp_path / "a.metrics.jsonl",
            [
                request("a", 0, 100 * MS, 400 * MS, 4),  # 100 ms, then 3 gaps of 100 ms
                request("b", 50 * MS, 250 * MS, 250 * MS, 1),  # one token: no TPOT
            ],
        )
    )
    lat = report.latencies(run)
    assert lat["ttft"] == pytest.approx([100, 200])
    assert lat["e2e"] == pytest.approx([400, 200])
    assert lat["tpot"] == pytest.approx([100])


def test_only_completed_requests_give_latencies_and_throughput(tmp_path):
    rows = [
        request("a", 100 * MS, 200 * MS, 1 * S, 10),
        request("b", 100 * MS, 300 * MS, 600 * MS, 5, outcome="end of sequence"),
        request("c", 0, 100 * MS, 3 * S, 30, outcome="failed"),
        request("d", 0, 100 * MS, 300 * MS, 3, outcome="cancelled by the client"),
        request(None, 0, None, None, 0, outcome="rejected"),  # the first to arrive
    ]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert (s["requests"], s["completed"], s["output_tokens"]) == (5, 2, 15)
    assert s["outcomes"] == {
        "max_tokens": 1,
        "end of sequence": 1,
        "failed": 1,
        "cancelled by the client": 1,
        "rejected": 1,
    }
    assert s["window_s"] == pytest.approx(1.0)  # first arrival to the last completed token
    assert s["throughput_tok_s"] == pytest.approx(15.0)
    assert s["latency"]["ttft"]["n"] == 2
    assert s["latency"]["ttft"]["p50"] == pytest.approx(150)  # 100 and 200 ms


def test_percentiles_are_reported_with_their_count(tmp_path):
    rows = [request(str(i), 0, (i + 1) * MS, (i + 1) * MS + 9 * MS, 10) for i in range(100)]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    ttft = s["latency"]["ttft"]
    assert ttft["n"] == 100
    assert (ttft["p50"], ttft["p95"], ttft["p99"]) == pytest.approx((50.5, 95.05, 99.01))
    assert s["latency"]["tpot"]["p99"] == pytest.approx(1.0)


def test_busy_time_is_read_off_the_beat_totals_inside_the_window(tmp_path):
    rows = [
        request("a", 1 * S, 1 * S + MS, 3 * S, 2),  # the window is 1 s to 3 s
        beat(0, 0, 0, 0),
        beat(2 * S, 0, 1 * S, 20),  # stage 0 busy half the time from 0 s to 2 s
        beat(4 * S, 0, 3 * S, 60),  # and all the time from 2 s to 4 s
        beat(0, 1, 0, 0),
        beat(4 * S, 1, 1 * S, 40),
    ]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert s["window_s"] == pytest.approx(2.0)
    stage0, stage1 = s["stages"][0], s["stages"][1]
    assert stage0["busy_s"] == pytest.approx(0.5 + 1.0)  # 1 s to 2 s at half, 2 s to 3 s full
    assert stage0["utilisation"] == pytest.approx(0.75)
    assert stage0["steps"] == 30  # 10 then 20
    assert stage1["busy_s"] == pytest.approx(0.5) and stage1["utilisation"] == pytest.approx(0.25)


def test_a_restarted_stage_counts_each_incarnation_from_its_own_zero(tmp_path):
    rows = [
        request("a", 0, MS, 4 * S, 2),
        beat(0, 1, 0, epoch=1),
        beat(2 * S, 1, 1 * S, epoch=1),  # the old process: busy 1 s, then it dies
        beat(2 * S + 1, 1, 0, epoch=2),  # the new one counts from 0 again
        beat(4 * S, 1, 2 * S, epoch=2),
    ]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert s["stages"][1]["busy_s"] == pytest.approx(3.0, abs=1e-6)  # never 1 - 0 + ...


def test_a_stage_with_no_beats_is_not_reported_as_idle(tmp_path):
    rows = [request("a", 0, MS, S, 2), beat(0, 0, 0), beat(S, 0, S // 2)]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert s["stages"][0]["utilisation"] == pytest.approx(0.5)
    assert s["stages"][1] == {
        "busy_s": None,
        "utilisation": None,
        "steps": None,
        "max_beat_gap_s": None,
    }


def test_the_longest_stretch_without_a_beat_is_reported_and_warned_of(tmp_path):
    rows = [
        request("a", 1 * S, 1 * S + MS, 9 * S, 2),  # the window is 1 s to 9 s
        beat(0, 0, 0),
        beat(2 * S, 0, S),
        beat(4 * S, 0, 2 * S),
        beat(6 * S, 0, 3 * S),  # then 5 s with no beat, past the end of the window:
        beat(11 * S, 0, 5 * S),  # the beat after the window bounds the last stretch
        beat(5 * S, 1, 0),  # stage 1's first beat comes 4 s into the window
        beat(7 * S, 1, S),
        beat(9 * S, 1, 2 * S),
    ]
    path = write(tmp_path / "a.metrics.jsonl", rows)
    s = report.summary(report.load(path))
    assert s["stages"][0]["max_beat_gap_s"] == pytest.approx(5.0)
    assert s["stages"][1]["max_beat_gap_s"] == pytest.approx(4.0)
    report.main([str(path), "--out", str(tmp_path / "out")])
    text = (tmp_path / "out" / "a" / "summary.md").read_text(encoding="utf-8")
    assert "Stage 0 sent no beat for 5.0 s" in text and "Stage 1 sent no beat for 4.0 s" in text
    stages = read_csv(tmp_path / "out" / "a" / "stages.csv")
    assert float(stages[0]["max_beat_gap_s"]) == pytest.approx(5.0)


def test_steady_beats_give_no_warning(tmp_path):
    rows = [request("a", 0, MS, 2 * S, 2)]
    rows += [beat(i * S // 2, stage, i * S // 4) for stage in (0, 1) for i in range(6)]
    path = write(tmp_path / "a.metrics.jsonl", rows)
    report.main([str(path), "--out", str(tmp_path / "out")])
    assert "sent no beat" not in (tmp_path / "out" / "a" / "summary.md").read_text("utf-8")


def test_a_run_cut_short_is_read_and_flagged(tmp_path):
    path = write(tmp_path / "a.metrics.jsonl", [request("a", 0, MS, S, 2)], end=False)
    with path.open("a", encoding="utf-8") as file:
        file.write('{"type": "hop", "t_ns": 12')  # the gateway died mid-line
    run = report.load(path)
    s = report.summary(run)
    assert (s["completed"], s["complete_run"], s["bad_lines"]) == (1, False, 1)


def test_a_line_missing_a_field_is_counted_not_fatal(tmp_path):
    broken = request("b", 0, MS, S, 2)
    del broken["arrived_ns"]
    rows = [request("a", 0, MS, S, 2), broken, {"type": "beat", "t_ns": 5}]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert (s["requests"], s["bad_lines"]) == (1, 2)


def test_a_run_with_no_completed_request_has_no_window(tmp_path):
    rows = [request(None, 0, None, None, 0, outcome="rejected")]
    s = report.summary(report.load(write(tmp_path / "a.metrics.jsonl", rows)))
    assert s["window_s"] is None and s["throughput_tok_s"] is None and s["stages"] == {}
    assert s["latency"]["ttft"] == {"n": 0}


# -------------------------------------------------------------- the command


def two_runs(folder):
    folder.mkdir()
    write(
        folder / "20261110T1-gateway.metrics.jsonl",
        [request("a", 0, 100 * MS, 400 * MS, 4), beat(0, 0, 0), beat(S, 0, S // 4)],
    )
    write(folder / "20261110T2-gateway.metrics.jsonl", [request("b", 0, 50 * MS, 150 * MS, 3)])
    (folder / "notes.txt").write_text("not a log", encoding="utf-8")


def test_one_command_builds_every_table_and_plot(tmp_path):
    two_runs(tmp_path / "logs")
    out = tmp_path / "report"
    assert report.main([str(tmp_path / "logs"), "--out", str(out)]) == 0
    runs = read_csv(out / "runs.csv")
    assert [r["run"] for r in runs] == ["20261110T1-gateway", "20261110T2-gateway"]
    assert float(runs[0]["ttft_p50_ms"]) == pytest.approx(100)
    assert float(runs[1]["throughput_tok_s"]) == pytest.approx(3 / 0.15)
    assert (out / "index.md").is_file()
    first = out / "20261110T1-gateway"
    for name in ("summary.md", "latency.csv", "stages.csv", "bytes.csv", "latency_cdf.png"):
        assert (first / name).stat().st_size > 0, name
    assert (first / "utilisation.png").read_bytes().startswith(b"\x89PNG")
    assert not (out / "20261110T2-gateway" / "utilisation.png").exists()  # no beats
    latency = read_csv(first / "latency.csv")
    assert [r["metric"] for r in latency] == ["TTFT", "TPOT", "End-to-end"]
    assert float(latency[1]["p99_ms"]) == pytest.approx(100)


def test_running_again_replaces_the_report(tmp_path):
    two_runs(tmp_path / "logs")
    out = tmp_path / "report"
    report.main([str(tmp_path / "logs"), "--out", str(out)])
    (tmp_path / "logs" / "20261110T2-gateway.metrics.jsonl").unlink()
    report.main([str(tmp_path / "logs"), "--out", str(out)])
    assert [r["run"] for r in read_csv(out / "runs.csv")] == ["20261110T1-gateway"]
    assert not (out / "20261110T2-gateway").exists()


def test_a_folder_it_did_not_make_is_never_removed(tmp_path, capsys):
    two_runs(tmp_path / "logs")
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "thesis.tex").write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "logs"), "--out", str(mine)])
    assert (mine / "thesis.tex").read_text(encoding="utf-8") == "keep me"
    with pytest.raises(SystemExit):  # the logs inside the output folder
        report.main([str(tmp_path / "logs"), "--out", str(tmp_path)])
    assert (tmp_path / "logs" / "20261110T1-gateway.metrics.jsonl").exists()


def test_a_log_kept_inside_an_old_report_is_never_removed(tmp_path):
    two_runs(tmp_path / "logs")
    out = tmp_path / "report"
    report.main([str(tmp_path / "logs"), "--out", str(out)])
    kept = out / "kept.metrics.jsonl"
    kept.write_bytes((tmp_path / "logs" / "20261110T1-gateway.metrics.jsonl").read_bytes())
    with pytest.raises(SystemExit):
        report.main([str(kept), "--out", str(out)])
    assert kept.exists() and (out / "runs.csv").exists()


def test_a_failed_build_leaves_the_old_report_in_place(tmp_path, monkeypatch):
    two_runs(tmp_path / "logs")
    out = tmp_path / "report"
    report.main([str(tmp_path / "logs"), "--out", str(out)])
    before = (out / "runs.csv").read_bytes()

    def broken(*args):
        raise RuntimeError("disk full")

    monkeypatch.setattr(report, "plot_run", broken)
    with pytest.raises(RuntimeError):
        report.main([str(tmp_path / "logs"), "--out", str(out)])
    assert (out / "runs.csv").read_bytes() == before
    monkeypatch.undo()
    assert report.main([str(tmp_path / "logs"), "--out", str(out)]) == 0  # and recovers
    assert not (tmp_path / "report.new").exists()


def test_an_output_path_that_is_a_file_or_a_stranger_beside_it_is_left_alone(tmp_path):
    two_runs(tmp_path / "logs")
    target = tmp_path / "out.csv"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "logs"), "--out", str(target)])
    assert target.read_text(encoding="utf-8") == "keep"
    stranger = tmp_path / "report.new"  # the folder the build would use first
    stranger.mkdir()
    (stranger / "draft.md").write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "logs"), "--out", str(tmp_path / "report")])
    assert (stranger / "draft.md").read_text(encoding="utf-8") == "keep"


def test_no_logs_or_a_missing_path_is_an_error(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert report.main([str(tmp_path / "empty"), "--out", str(tmp_path / "out")]) == 1
    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "nowhere")])


def test_two_logs_of_one_name_get_two_folders(tmp_path):
    for folder in ("x", "y"):
        (tmp_path / folder).mkdir()
        write(tmp_path / folder / "run.metrics.jsonl", [request("a", 0, MS, S, 2)])
    out = tmp_path / "report"
    report.main([str(tmp_path / "x"), str(tmp_path / "y"), "--out", str(out)])
    assert [r["run"] for r in read_csv(out / "runs.csv")] == ["run", "run-2"]


# -------------------------------------------------------------- whole system


@pytest.fixture
def system(tmp_path):
    made = []
    tiny_reference(tmp_path, layers=4)

    def make(**kwargs):
        run = Beating(tmp_path, 2, **kwargs)
        made.append(run)
        return run

    yield make
    for run in made:
        run.gateway.recorder.finish()
        run.close()


def test_a_stage_counts_its_busy_time_and_only_steps_it_passed_on(system):
    run = system(delay0=0.05)
    run.tokens(max_tokens=4)
    busy0, steps0 = run.servicer0.busy
    busy1, steps1 = run.servicer1.busy
    assert steps0 == steps1 == 4
    assert busy0 >= 4 * 0.05 * 1e9  # the step delay is inside the stage's step
    assert 0 < busy1 < busy0


def test_a_step_still_running_counts_in_the_busy_total(system):
    run = system(delay0=1.5)
    caller = threading.Thread(target=lambda: run.tokens(max_tokens=1))
    caller.start()
    wait_for(lambda: run.servicer0.busy[0] > 0, "stage 0 to start its step")
    first, steps = run.servicer0.busy
    time.sleep(0.5)
    later, steps_later = run.servicer0.busy
    assert steps == steps_later == 0  # the step has not finished
    assert later - first >= 0.4e9  # but its time is already counted
    caller.join(WAIT)
    assert run.servicer0.busy[1] == 1


def test_a_step_refused_with_a_nack_is_not_counted_as_a_step(tmp_path):
    tiny_reference(tmp_path, layers=4)
    run = System(tmp_path, 2)
    try:
        send(run.ports[0], prefill(run, "old", target_epoch=run.servicer0.own_epoch - 1))
        assert nack(run.servicer0).reason == "WRONG_EPOCH"
        send(run.ports[0], prefill(run, "new"))
        wait_for(lambda: "new" in run.servicer1.store, "the right epoch to be served")
        assert run.servicer0.busy[1] == 1
    finally:
        run.close()


def test_the_report_of_a_real_run_gives_its_requests_and_stage_busy_time(tmp_path, system):
    run = system(delay0=0.05)
    path = tmp_path / "live.metrics.jsonl"
    run.gateway.recorder = Recorder(path, {"model": "tiny"})
    for _ in range(3):
        run.tokens(max_tokens=4)
    busy0, steps0 = run.servicer0.busy
    run.gateway.close()
    run.gateway.recorder.finish()
    rows = lines(path)
    beats = of_type(rows, "beat")
    assert {b["stage"] for b in beats} == {0, 1}
    assert all(b["steps"] <= steps0 and b["busy_ns"] <= busy0 for b in beats)
    s = report.summary(report.load(path))
    assert (s["completed"], s["output_tokens"]) == (3, 12)
    assert s["latency"]["tpot"]["n"] == 3 and s["latency"]["ttft"]["p50"] > 0
    stage0 = s["stages"][0]
    # All of stage 0's work falls inside the window, so the report must match its own
    # total to within the longest beat gap at each end of the window.
    assert stage0["busy_s"] == pytest.approx(busy0 / 1e9, abs=2 * stage0["max_beat_gap_s"])
    assert 0 < stage0["utilisation"] <= 1.05
    assert report.main([str(path), "--out", str(tmp_path / "out")]) == 0
