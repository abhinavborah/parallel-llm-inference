"""X4 runner (R7.11): the fit, the aggregates, the argument checks, setting the delay through
the control file, and the sweep on a real chain on loopback."""

import threading

import pytest
from test_hop_delay import controllable, system  # noqa: F401 (system is a fixture)

from pllm import faults, metrics, x4
from pllm.faults import HopDelay, HopDelayControl


def test_an_exact_line_has_its_slope_and_a_zero_width_interval():
    xs = [0, 1, 5, 20, 50] * 2
    result = x4.fit(xs, [9 + 2 * x for x in xs])
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(9.0)
    assert result["ci95"] == pytest.approx([2.0, 2.0])


def test_the_interval_uses_t_for_the_degrees_of_freedom():
    xs, ys = [0, 10, 20, 30], [0.0, 11.0, 19.0, 31.0]
    result = x4.fit(xs, ys)
    low, high = result["ci95"]
    assert (high - low) / 2 == pytest.approx(4.303 * result["se"])  # df 2


def test_too_few_points_give_no_slope():
    assert x4.fit([0, 0, 0], [1, 2, 3])["slope"] is None
    assert x4.fit([0, 5], [1, 2])["slope"] is None


def cell(d, c, tpot, throughput=10.0):
    return {
        "delay_ms": d,
        "concurrency": c,
        "tpot_mean_ms": tpot,
        "tpot_p50_ms": tpot,
        "ttft_mean_ms": 2 * tpot,
        "ttft_p50_ms": 2 * tpot,
        "throughput_tok_s": throughput,
        "failed_requests": 0,
    }


def test_one_fit_per_concurrency_says_whether_it_holds_two():
    cells = [cell(d, 1, 9 + 2 * d + e) for d in (0, 5, 20) for e in (-0.5, 0.5)]
    cells += [cell(d, 4, 9 + d + e) for d in (0, 5, 20) for e in (-0.5, 0.5)]
    fits = {line["concurrency"]: line for line in x4.fit_lines(cells)}
    assert fits[1]["checked"] and fits[1]["holds_expected"] is True
    assert not fits[4]["checked"]  # reported only: queueing hides part of the delay
    assert fits[4]["holds_expected"] is False  # a slope of 1


def test_aggregates_group_by_delay_and_concurrency():
    cells = [cell(0, 1, 10.0, 5.0), cell(0, 1, 12.0, 7.0), cell(5, 1, 20.0)]
    rows = {(r["delay_ms"], r["concurrency"]): r for r in x4.aggregate(cells)}
    assert rows[(0, 1)]["repeats"] == 2
    assert rows[(0, 1)]["tpot_mean_ms"]["mean"] == 11.0
    assert rows[(0, 1)]["throughput_tok_s"]["min"] == 5.0
    assert rows[(5, 1)]["tpot_mean_ms"]["stdev"] is None  # one repeat


def test_each_repeat_starts_the_delays_one_further_along():
    delays = [0, 1, 5, 20, 50]
    assert x4.rotated(delays, 0) == delays
    assert x4.rotated(delays, 1) == [1, 5, 20, 50, 0]
    assert x4.rotated(delays, 5) == delays


def test_the_expectation_is_two_d_per_token():
    line = x4.expectation([0, 1, 5, 20, 50], [1, 4])
    assert line["added_per_token_ms"] == {"0": 0, "1": 2, "5": 10, "20": 40, "50": 100}
    assert line["delayed_hops"] == ["gateway_stage0", "stage0_stage1"]


def gateway_file(tmp_path, **info):
    path = tmp_path / "gw.metrics.jsonl"
    metrics.Recorder(path, {"model": "smollm2-135m", **info}).finish()
    return path


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--delays", "5"], "at least two"),
        (["--delays", "0", "-1"], "at least two"),
        (["--timeout", "5"], "--timeout must cover the delay"),
        (["--requests", "4", "--concurrency", "4"], "twice the largest"),
    ],
)
def test_bad_arguments_are_refused(tmp_path, capsys, extra, message):
    path = gateway_file(tmp_path, hop_delay_control=True)
    with pytest.raises(SystemExit):
        x4.main(["smollm2-135m", "--metrics", str(path), *extra])
    assert message in capsys.readouterr().err


def test_a_gateway_without_the_control_is_refused(tmp_path, capsys):
    path = gateway_file(tmp_path, hop_delay_control=False)
    with pytest.raises(SystemExit):
        x4.main(["smollm2-135m", "--metrics", str(path)])
    assert "--hop-delay-control" in capsys.readouterr().err


# -------------------------------------------------------------- the control file


def controls(path, applied):
    stop = threading.Event()
    for name in faults.CONTROLLED:
        HopDelayControl(name, lambda d, n=name: applied.__setitem__(n, d), path, 0.02).start(stop)
    return stop


def test_a_change_in_the_file_is_applied_and_acked_by_both(tmp_path):
    path, applied = tmp_path / "hop_delay_ms", {}
    stop = controls(path, applied)
    try:
        assert applied == {"gateway": HopDelay(), "stage1": HopDelay()}  # no file: 0
        x4.set_delay(path, 20, timeout=2)
        assert applied == {"gateway": HopDelay(0.02), "stage1": HopDelay(0.02)}
        assert [faults.read_delay_ms(p) for p in faults.ack_paths(path)] == [20, 20]
    finally:
        stop.set()


def test_a_delay_no_process_acks_times_out_naming_it(tmp_path):
    path = tmp_path / "hop_delay_ms"
    stop = threading.Event()
    HopDelayControl("gateway", lambda d: None, path, 0.02).start(stop)
    try:
        with pytest.raises(TimeoutError, match="stage1"):
            x4.set_delay(path, 5, timeout=0.3)
    finally:
        stop.set()


# -------------------------------------------------------------- the X4 sweep on a chain


def test_the_x4_sweep_states_the_expectation_first_and_finds_the_slope(system, tmp_path):  # noqa: F811
    run = controllable(system())
    run.gateway.recorder = metrics.Recorder(tmp_path / "gw.metrics.jsonl", {"model": "tiny"})
    path = tmp_path / "hop_delay_ms"

    def apply_gateway(hold):
        run.gateway.hop_delay = hold

    def apply_stage1(hold):
        run.servicer1.hop_delay = hold

    stop = threading.Event()
    HopDelayControl("gateway", apply_gateway, path, 0.02).start(stop)
    HopDelayControl("stage1", apply_stage1, path, 0.02).start(stop)

    class Args:
        delays, concurrency, repeats, requests, settle = [0, 10, 20], [1, 2], 1, 4, 0.1

    lines = []

    def call(job):
        return run.call(prompt=" ".join(map(str, job.ids)), max_tokens=6)

    try:
        cells = x4.sweep(
            call,
            tmp_path / "gw.metrics.jsonl",
            Args,
            [[5, 9, 100], [3, 42, 7]],
            lines.append,
            path,
        )
    finally:
        stop.set()
    assert lines[0]["type"] == "expectation"
    assert lines[0]["expected_slope"] == 2.0
    assert sorted({c["delay_ms"] for c in cells}) == [0, 10, 20]
    assert all(c["tpot_p50_ms"] is not None for c in cells)
    (fit1,) = [line for line in lines if line["type"] == "fit" and line["concurrency"] == 1]
    assert fit1["n"] == 3 and 1.2 < fit1["slope"] < 3.5
    assert lines[-1]["type"] == "summary" and lines[-1]["passed"]
    assert faults.read_delay_ms(path) == 0  # set back at the end
