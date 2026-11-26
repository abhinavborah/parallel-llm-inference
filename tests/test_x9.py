"""X9 (R7.16, R8.6, R8.7): the capacity-aware split, the memory counters, the Host B
monitor, the probe, the verdicts, the split override, and one sweep on a real chain on
loopback. Small random models only."""

import ctypes
import json
import os
import socket
import sys
import threading
from types import SimpleNamespace

import pytest
from test_gateway import System
from test_model import tiny_reference

from pllm import (
    capacity,
    config,
    launch,
    metrics,
    procmem,
    report,
    runrecord,
    x1b,
    x9,
    x9monitor,
    x9probe,
)
from pllm.capacity import HostCapacity, NoSplit, choose_split, sweep_points
from pllm.control import ExpectedModel
from pllm.heartbeat import Heartbeat
from pllm.model import ModelConfig
from pllm.models import PINNED
from pllm.supervisor import worker_command
from pllm.worker import main as worker_main
from pllm.worker import stage_layers

MIB = 1024 * 1024
MODEL = "smollm2-135m"
REVISION = PINNED[MODEL].revision
SMOLLM2_CONFIG = {
    "model_type": "llama",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


def host(**kwargs) -> HostCapacity:
    values = {
        "free_bytes": 10_000,
        "fixed_bytes": 100,
        "bytes_per_layer": 10,
        "fixed_seconds": 0.0,
        "seconds_per_layer": 1.0,
    }
    return HostCapacity(**{**values, **kwargs})


# -------------------------------------------------------------- the split function


def test_a_slower_host_b_gets_fewer_layers_than_an_even_split_by_speed():
    split = choose_split(30, host(seconds_per_layer=2.0), host())
    assert split.k == 10 and split.even_k == 15 and split.k < split.even_k
    assert split.host_b_step_seconds == split.host_a_step_seconds == 20
    assert split.unconstrained_k == 10 and not split.forced_below_even
    assert split.allowed == tuple(range(1, 15))


def test_the_fixed_time_of_a_stage_moves_the_balance():
    # Host A's norm and head cost 7 time units, so it takes fewer layers than 2 to 1 would
    split = choose_split(30, host(seconds_per_layer=2.0), host(fixed_seconds=7.0))
    assert split.k == 12  # max(24, 25) beats max(22, 26) and max(26, 24)


def test_host_b_memory_limits_k_and_a_need_equal_to_the_free_memory_fits():
    slow_a = host(seconds_per_layer=5.0)
    exact = choose_split(30, host(free_bytes=100 + 7 * 10), slow_a)
    assert exact.k == 7 and exact.host_b_bytes == 170
    assert choose_split(30, host(free_bytes=100 + 7 * 10 - 1), slow_a).k == 6


def test_host_a_memory_pushes_layers_to_host_b():
    # Host A holds at most 18 layers, so k is at least 12 although speed would choose 3
    split = choose_split(30, host(seconds_per_layer=9.0), host(free_bytes=100 + 18 * 10))
    assert split.k == 12 and split.allowed == (12, 13, 14)


@pytest.mark.parametrize("b_seconds", [1.0, 0.25])
def test_a_choice_that_only_the_even_rule_made_says_so(b_seconds):
    split = choose_split(30, host(seconds_per_layer=b_seconds), host())
    assert split.k == 14 and split.forced_below_even
    assert split.unconstrained_k >= 15


def test_on_a_tie_the_smaller_k_wins():
    # n = 5: k = 1 gives max(1, 2) and k = 2 gives max(2, 1.5)
    split = choose_split(5, host(), host(seconds_per_layer=0.5))
    assert split.k == 1


def test_the_smallest_model_that_can_have_a_split_below_even():
    assert choose_split(3, host(), host()).k == 1
    with pytest.raises(NoSplit, match="no split below an even one"):
        choose_split(2, host(), host())


@pytest.mark.parametrize(
    "host_b, host_a, reason",
    [
        (host(free_bytes=109), host(), "Host B cannot hold one layer"),
        (host(), host(free_bytes=100 + 15 * 10), "Host A cannot hold the 16 layers"),
        (host(free_bytes=0), host(free_bytes=0), "Host B cannot hold one layer"),
        # each fits something, but no k fits both
        (host(free_bytes=100 + 5 * 10), host(free_bytes=100 + 18 * 10), "both hosts at the same"),
    ],
)
def test_no_split_is_returned_when_the_constraints_cannot_be_met(host_b, host_a, reason):
    with pytest.raises(NoSplit, match=reason):
        choose_split(30, host_b, host_a)


@pytest.mark.parametrize("layers", [1, 0, -3, 2.5, True, None])
def test_a_bad_layer_count_is_refused(layers):
    with pytest.raises(ValueError, match="num_layers"):
        choose_split(layers, host(), host())


@pytest.mark.parametrize("name", list(vars(host())))
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "3"])
def test_a_missing_or_unusable_measurement_is_refused(name, value):
    with pytest.raises(ValueError, match=name):
        host(**{name: value})


@pytest.mark.parametrize(
    "bad",
    [
        {"free_bytes": -1},
        {"fixed_bytes": -1},
        {"bytes_per_layer": 0},
        {"seconds_per_layer": 0.0},
        {"seconds_per_layer": -1.0},
    ],
)
def test_a_measurement_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        host(**bad)


def test_a_fit_that_predicts_no_time_is_refused():
    with pytest.raises(ValueError, match="not positive"):
        choose_split(30, host(fixed_seconds=-1.0), host())


def test_the_even_split_is_half_the_layers():
    assert [capacity.even_split(n) for n in (3, 4, 30)] == [1, 2, 15]


def test_the_sweep_goes_from_even_to_the_chosen_split():
    assert sweep_points(15, 6) == [15, 12, 9, 6]
    assert sweep_points(15, 14) == [15, 14]
    assert sweep_points(15, 5, points=2) == [15, 5]
    for even, chosen, points in ((15, 15, 4), (15, 16, 4), (15, 0, 4), (15, 6, 1)):
        with pytest.raises(ValueError):
            sweep_points(even, chosen, points)


# -------------------------------------------------------------- memory counters


def test_the_process_list_is_read_entry_by_entry():
    size = ctypes.sizeof(procmem._ProcessEntry)
    data = b""
    for pid, hard, faults, peak, more in ((4, 7, 100, 4096, True), (88, 0, 5, 8192, False)):
        entry = procmem._ProcessEntry(
            NextEntryOffset=size + 16 if more else 0,
            UniqueProcessId=pid,
            HardFaultCount=hard,
            PageFaultCount=faults,
            PeakWorkingSetSize=peak,
        )
        data += bytes(entry) + (b"\0" * 16 if more else b"")
    assert procmem.parse_process_entries(data) == {4: (7, 100, 4096), 88: (0, 5, 8192)}
    assert procmem.parse_process_entries(b"") == {}


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="64-bit layout")
def test_the_fields_sit_where_the_64_bit_windows_layout_has_them():
    offsets = {
        "HardFaultCount": 16,
        "UniqueProcessId": 80,
        "PageFaultCount": 128,
        "PeakWorkingSetSize": 136,
    }
    for name, offset in offsets.items():
        assert getattr(procmem._ProcessEntry, name).offset == offset


def test_the_hard_fault_count_is_used_only_when_its_neighbours_match_psutil():
    before = SimpleNamespace(num_page_faults=100, peak_wset=5000)
    after = SimpleNamespace(num_page_faults=110, peak_wset=6000)
    assert procmem.check_layout((3, 105, 5000), before, after)
    assert not procmem.check_layout((3, 99, 5000), before, after)
    assert not procmem.check_layout((3, 105, 6001), before, after)
    assert not procmem.check_layout(None, before, after)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_this_windows_gives_a_hard_fault_count_for_this_process():
    faults = procmem.windows_hard_faults()
    assert faults is not None and os.getpid() in faults


def test_the_linux_files_are_parsed():
    stat = "12 (a ) odd (name) S 1 12 12 0 -1 4194304 900 0 37 0 5 1 0 0"
    assert procmem.parse_stat_majflt(stat) == 37
    assert procmem.parse_vm_hwm("Name:\tx\nVmHWM:\t  2048 kB\n") == 2048 * 1024
    assert procmem.parse_vm_hwm("Name:\tx\n") is None


class Proc:
    pid = 9

    def __init__(self, **info):
        self.info = SimpleNamespace(**info)

    def memory_info(self):
        return self.info


def test_each_system_reads_its_own_counters_and_leaves_the_rest_unknown():
    win = procmem.process_counters(Proc(rss=10, peak_wset=20, num_page_faults=30), {9: 4}, "win32")
    assert (win["peak_rss_bytes"], win["page_faults"], win["disk_faults"]) == (20, 30, 4)
    unread = procmem.process_counters(Proc(rss=10, peak_wset=20, num_page_faults=30), None, "win32")
    assert unread["disk_faults"] is None and unread["page_faults"] == 30
    mac = procmem.process_counters(Proc(rss=10, pfaults=30, pageins=2), None, "darwin")
    assert (mac["peak_rss_bytes"], mac["page_faults"], mac["disk_faults"]) == (None, 30, 2)
    other = procmem.process_counters(Proc(rss=10), None, "sunos5")
    assert other["peak_rss_bytes"] is other["page_faults"] is other["disk_faults"] is None
    assert procmem.disk_counter_name("sunos5") is None


def test_this_process_tree_can_be_read():
    snap = procmem.tree_snapshot(os.getpid())
    assert snap["rss_bytes"] > 0 and snap["processes"]
    assert snap["disk_faults"] is not None
    assert (snap["peak_rss_bytes"] is None) == (sys.platform == "darwin")
    assert procmem.rss_reader(os.getpid())() > 0


def snapshot_of(*rows):
    keys = ("peak", "faults", "disk")
    return {"processes": [{"pid": p, **dict(zip(keys, v, strict=True))} for p, *v in rows]}


def test_a_counter_growth_is_unknown_unless_the_same_processes_give_it():
    start = snapshot_of((1, 0, 5, 1), (2, 0, 10, 0))
    assert procmem.delta(start, snapshot_of((2, 0, 12, 0), (1, 0, 8, 1)), "faults") == 5
    assert procmem.delta(start, snapshot_of((1, 0, 8, 1)), "faults") is None
    assert procmem.delta(start, snapshot_of((1, 0, 8, 1), (3, 0, 12, 0)), "faults") is None
    assert procmem.delta(start, snapshot_of((1, 0, 4, 1), (2, 0, 12, 0)), "faults") is None
    assert procmem.delta(start, snapshot_of((1, 0, None, 1), (2, 0, 12, 0)), "faults") is None
    assert procmem.delta({"processes": []}, {"processes": []}, "faults") is None
    assert procmem.total([{"x": 1}, {"x": None}], "x") is None
    assert procmem.total([], "x") is None


# -------------------------------------------------------------- the Host B monitor


def fake_monitor(sizes=(100, 300, 200), status=None, **kwargs):
    left = list(sizes)

    def rss():
        return left.pop(0) if len(left) > 1 else left[0]

    return x9monitor.Monitor(
        status or (lambda: {"pid": 42, "restarts": 0}),
        interval=0.005,
        commit="c1",
        snapshot=lambda pid: {"processes": [{"pid": pid}], "peak_rss_bytes": 500},
        reader=lambda pid: rss,
        args=lambda pid: ["tiny", "--stage", "0", "--serve"],
        **kwargs,
    )


def test_the_worker_arguments_leave_out_the_interpreter_path(monkeypatch):
    command = ["C:/somewhere/python.exe", "-m", "pllm.worker", "tiny", "--stage", "0"]
    monkeypatch.setattr(
        x9monitor.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: command)
    )
    assert x9monitor.worker_args(1) == ["tiny", "--stage", "0"]
    command = ["python", "other.py"]
    assert x9monitor.worker_args(1) is None


def test_a_window_keeps_the_largest_sample():
    monitor = fake_monitor()
    assert monitor.start()["pid"] == 42
    threading.Event().wait(0.1)
    stop = monitor.stop()
    window = stop["window"]
    assert window["rss_max_bytes"] == 300 and window["samples"] >= 3
    assert window["read_errors"] == 0 and window["max_gap_s"] > 0
    assert stop["commit"] == "c1" and stop["peak_rss_bytes"] == 500
    with pytest.raises(x9monitor.NoWorker, match="no window"):
        monitor.stop()


@pytest.mark.parametrize("status", [lambda: {"pid": None}, lambda: (_ for _ in ()).throw(OSError)])
def test_with_no_worker_the_monitor_says_so(status):
    with pytest.raises(x9monitor.NoWorker):
        fake_monitor(status=status).snapshot()


def test_the_monitor_answers_over_http_and_the_runner_reads_it():
    server = x9monitor.serve(fake_monitor(), ("127.0.0.1", 0))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert x9.monitor_get(url, "/snapshot")["pid"] == 42
        with pytest.raises(RuntimeError, match="503.*no window"):
            x9.monitor_get(url, "/stop")
        x9.monitor_get(url, "/start")
        assert "window" in x9.monitor_get(url, "/stop")
        with pytest.raises(RuntimeError, match="404"):
            x9.monitor_get(url, "/nothing")
    finally:
        server.shutdown()
    server.server_close()
    with pytest.raises(RuntimeError, match="is it running"):
        x9.monitor_get(url, "/snapshot", timeout=2)


# -------------------------------------------------------------- the probe


def test_the_probe_counts_cover_each_hosts_side_of_the_sweep():
    assert x9.probe_counts(30, 0) == (5, 15)
    assert x9.probe_counts(30, 1) == (15, 25)
    assert x9.layer_range(30, 0, 5) == (0, 5)
    assert x9.layer_range(30, 1, 25) == (5, 30)


@pytest.mark.parametrize("layers", ["0:1", "2:4"])
def test_a_probe_process_loads_a_stage_and_times_its_steps(tmp_path, layers):
    tiny_reference(tmp_path, layers=4)
    command = [sys.executable, "-m", "pllm.x9probe", str(tmp_path), "--layers", layers]
    command += ["--threads", "1", "--prompt", "4", "--steps", "3"]
    row = x9.probe_row(x9.run_probe_process(command))
    assert row["layers"] == int(layers[2]) - int(layers[0])
    assert row["steps"] == 3 and row["step_seconds_median"] > 0
    assert row["peak_rss_bytes"] > 0 and row["load_peak_rss_bytes"] > 0
    assert row["peak_rss_source"] == (
        "the process itself" if sys.platform == "darwin" else "process tree"
    )


def test_a_probe_process_that_fails_is_an_error(tmp_path):
    command = [sys.executable, "-m", "pllm.x9probe", str(tmp_path / "none"), "--layers", "0:1"]
    command += ["--threads", "1", "--prompt", "4", "--steps", "1"]
    with pytest.raises(RuntimeError, match="no result"):
        x9.run_probe_process(command)


def test_the_probe_times_only_the_steps_after_the_warm_up(tmp_path):
    tiny_reference(tmp_path, layers=4)
    from pllm.worker import StageWorker, WorkerConfig

    calls = []
    worker = StageWorker(WorkerConfig(tmp_path, 0, 2), delay=lambda: calls.append(1))
    assert len(x9probe.time_steps(worker, prompt=3, steps=5, warmup=2)) == 5
    assert len(calls) == 1 + 2 + 5


def probe_rows(by_count: dict) -> list[dict]:
    return [
        {"layers": layers, "step_seconds_median": seconds, "peak_rss_bytes": peak}
        for layers, runs in by_count.items()
        for seconds, peak in runs
    ]


def test_the_fit_goes_through_the_median_time_and_the_largest_peak():
    fit = x9.fit_host(
        probe_rows(
            {
                5: [(0.012, 250), (0.011, 248), (0.5, 249)],
                15: [(0.032, 350), (0.031, 345), (0.030, 349)],
            }
        )
    )
    assert fit["layer_counts"] == [5, 15]
    assert fit["seconds_per_layer"] == pytest.approx((0.031 - 0.012) / 10)
    assert fit["fixed_seconds"] == pytest.approx(0.012 - 5 * 0.0019)
    assert fit["bytes_per_layer"] == 10 and fit["fixed_bytes"] == 200
    assert fit["per_count"][0]["step_seconds"]["n"] == 3


@pytest.mark.parametrize(
    "by_count, message",
    [
        ({5: [(0.01, 100)]}, "two layer counts"),
        ({5: [(0.03, 100)], 15: [(0.02, 200)]}, "median step time did not grow"),
        ({5: [(0.01, 200)], 15: [(0.02, 200)]}, "peak memory did not grow"),
        ({5: [(0.01, 10)], 15: [(0.02, 200)]}, "below 0"),
    ],
)
def test_measurements_that_cannot_give_a_line_are_refused(by_count, message):
    with pytest.raises(ValueError, match=message):
        x9.fit_host(probe_rows(by_count))


def test_the_probe_command_alternates_the_counts_and_writes_the_fit(tmp_path, monkeypatch):
    """The measurement processes are stubbed here, so the line is known; the real ones
    are run above."""
    monkeypatch.setattr(x9, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(x9, "PINNED", {MODEL: SimpleNamespace(revision=REVISION, files={})})
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / "config.json").write_text(json.dumps(SMOLLM2_CONFIG), encoding="utf-8")
    ranges = []

    def fake(command):
        start, end = map(int, command[command.index("--layers") + 1].split(":"))
        ranges.append((start, end))
        layers = end - start
        return {
            "layer_start": start,
            "layer_end": end,
            "step_seconds": [0.001 * layers + 0.002] * 3,
            "load_seconds": 0.1,
            "own_load_peak_rss_bytes": 1,
            "own_peak_rss_bytes": 1,
            "tree_peak_rss_bytes": (200 + 10 * layers) * MIB,
        }

    monkeypatch.setattr(x9, "run_probe_process", fake)
    argv = ["probe", MODEL, "--stage", "1", "--threads", "2", "--repeats", "2"]
    assert x9.main(argv) == 0
    assert ranges == [(15, 30), (5, 30), (5, 30), (15, 30)]  # 15 and 25 layers, in turn
    written = next((tmp_path / "runs").glob("*-x9-probe.json"))
    record = x9.load_record(written, "x9-probe")
    assert record["stage"] == 1 and record["threads"] == 2 and len(record["processes"]) == 4
    assert record["fit"]["bytes_per_layer"] == 10 * MIB
    assert record["fit"]["seconds_per_layer"] == pytest.approx(0.001)
    with pytest.raises(SystemExit):
        x9.main([*argv, "--layer-counts", "5", "30"])


# -------------------------------------------------------------- choose


def choice_records(**changes) -> dict:
    def fit(seconds_per_layer, counts):
        return {
            "layer_counts": counts,
            "fixed_seconds": 0.001,
            "seconds_per_layer": seconds_per_layer,
            "fixed_bytes": 200 * MIB,
            "bytes_per_layer": 10 * MIB,
        }

    records = {
        "hostmem_b": {"host": "winB", "budget_bytes": 900 * MIB},
        "hostmem_a": {"host": "macA", "budget_bytes": 9000 * MIB},
        "probe_b": {"stage": 0, "host": "winB", "fit": fit(0.004, [5, 15]), "threads": 4},
        "probe_a": {"stage": 1, "host": "macA", "fit": fit(0.001, [15, 25]), "threads": 4},
    }
    for key in ("probe_a", "probe_b"):
        records[key].update(model=MODEL, revision=REVISION, commit="c1")
    for name, value in changes.items():
        key, _, entry = name.partition("__")
        records[key][entry] = value
    return records


def test_good_inputs_pass_the_checks():
    x9.check_choice_inputs(choice_records(), MODEL)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"hostmem_b__budget_bytes": None}, "no measured budget_bytes"),
        ({"hostmem_a__budget_bytes": True}, "no measured budget_bytes"),
        ({"probe_b__model": "qwen3-0.6b"}, "pinned revision"),
        ({"probe_a__revision": "other"}, "pinned revision"),
        ({"probe_b__stage": 1}, "must be of stage 0"),
        ({"probe_a__fit": None}, "no fit"),
        ({"hostmem_b__host": "elsewhere"}, "not from the same host"),
        ({"probe_a__host": None}, "not from the same host"),
        ({"probe_a__commit": "c2"}, "one clean commit"),
        ({"probe_a__commit": None}, "one clean commit"),
    ],
)
def test_inputs_that_are_missing_or_do_not_belong_are_refused(change, message):
    with pytest.raises(x9.BadInput, match=message):
        x9.check_choice_inputs(choice_records(**change), MODEL)


def test_one_host_or_a_dirty_commit_is_allowed_only_in_a_trial():
    records = choice_records(hostmem_a__host="winB", probe_a__host="winB")
    with pytest.raises(x9.BadInput, match="from one host"):
        x9.check_choice_inputs(records, MODEL)
    x9.check_choice_inputs(records, MODEL, trial=True)
    dirty = choice_records(probe_a__commit="c1-dirty", probe_b__commit="c1-dirty")
    with pytest.raises(x9.BadInput, match="clean commit"):
        x9.check_choice_inputs(dirty, MODEL)


def test_the_cache_reserved_per_layer_follows_the_kv_store():
    smollm2 = ModelConfig.from_dict(SMOLLM2_CONFIG)
    assert x9.kv_bytes_per_layer(smollm2, 1) * 15 == 23_040  # pllm.kvstore
    fit = choice_records()["probe_b"]["fit"]
    got = x9.host_capacity(fit, 1000, kv_layer=7, requests=4)
    assert got.fixed_bytes == fit["fixed_bytes"] + 7
    assert got.bytes_per_layer == fit["bytes_per_layer"] + 28


def test_the_choice_uses_the_smaller_of_the_measured_and_stated_budget():
    smollm2 = ModelConfig.from_dict(SMOLLM2_CONFIG)
    out = x9.decide(choice_records(), smollm2, 716 * MIB, positions=160, requests=4)
    assert out["capacity_b"]["free_bytes"] == 716 * MIB  # measured 900
    assert out["result"]["k"] < 15 and out["sweep"][0] == 15
    assert out["sweep"][-1] == out["result"]["k"]
    assert out["no_split"] is None
    low = choice_records(hostmem_b__budget_bytes=205 * MIB)
    out = x9.decide(low, smollm2, 716 * MIB, positions=160, requests=4)
    assert out["capacity_b"]["free_bytes"] == 205 * MIB
    assert out["result"] is None and out["sweep"] is None
    assert "Host B cannot hold one layer" in out["no_split"]


def test_a_choice_outside_the_probed_counts_is_marked():
    smollm2 = ModelConfig.from_dict(SMOLLM2_CONFIG)
    slow = choice_records()
    slow["probe_b"]["fit"] = {**slow["probe_b"]["fit"], "seconds_per_layer": 0.05}
    out = x9.decide(slow, smollm2, 716 * MIB, positions=160, requests=4)
    assert out["result"]["k"] < 5
    assert out["extrapolated"] == {"b": True, "a": True}


@pytest.mark.parametrize("budget_mib, code", [(716.8, 0), (100.0, 1)])
def test_the_choose_command_writes_a_record(tmp_path, monkeypatch, capsys, budget_mib, code):
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / "config.json").write_text(json.dumps(SMOLLM2_CONFIG), encoding="utf-8")
    monkeypatch.setattr(x9, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    argv = ["choose", MODEL, "--budget-b-mib", str(budget_mib)]
    for key, rec in choice_records().items():
        script = "host-memory" if key.startswith("hostmem") else "x9-probe"
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({"script": script, "status": "ok", **rec}), encoding="utf-8")
        argv += [f"--{key.replace('_', '-')}", str(path)]
    assert x9.main(argv) == code
    written = next((tmp_path / "runs").glob("*-x9-choice.json"))
    record = x9.load_record(written, "x9-choice")
    assert (record["result"] is None) == bool(code)
    out = capsys.readouterr().out
    assert ("no split:" in out) == bool(code)


def test_a_record_that_failed_is_not_read(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"script": "x9", "status": "error"}), encoding="utf-8")
    for bad, script in ((path, "x9"), (tmp_path / "none.json", "x9")):
        with pytest.raises(x9.BadInput):
            x9.load_record(bad, script)


# -------------------------------------------------------------- Host B verdicts


def snap(disk=10, peak=400, faults=1000, pid=7, restarts=0, **extra):
    return {
        "host": "winB",
        "pid": pid,
        "restarts": restarts,
        "processes": [
            {"pid": pid, "disk_faults": disk, "page_faults": faults, "peak_rss_bytes": peak}
        ],
        "peak_rss_bytes": peak,
        "disk_faults": disk,
        "page_faults": faults,
        **extra,
    }


def window(loaded, disk=(10, 10), peak=400, sampled=300, **stop_extra):
    start = snap(disk=disk[0])
    stop = snap(disk=disk[1], peak=peak, faults=1200, **stop_extra)
    stop["window"] = {"rss_max_bytes": sampled, "samples": 9, "max_gap_s": 0.06}
    return x9.window_fields(loaded, start, stop)


def test_a_run_within_budget_with_no_disk_fault_in_its_cells():
    loaded = snap(disk=10)
    windows = [window(loaded), window(loaded, sampled=320)]
    out = x9.host_b_result(loaded, windows, snap(disk=12), True, 500)
    assert out["within_budget"] is True and out["no_disk_faults_in_windows"] is True
    assert out["load_peak_rss_bytes"] == out["lifetime_peak_rss_bytes"] == 400
    assert out["decode_raised_peak"] is False and out["decode_peak_rss_bytes"] is None
    assert out["decode_sampled_rss_max_bytes"] == 320
    assert out["load_disk_faults"] == 10 and out["window_disk_faults"] == 0
    assert out["outside_window_disk_faults"] == 2  # reported, not judged
    assert out["window_page_faults"] == 400  # soft faults do not count against paging


def test_a_decode_that_raises_the_lifetime_peak_gives_the_decode_peak():
    loaded = snap()
    out = x9.host_b_result(loaded, [window(loaded, peak=450)], snap(peak=450), True, 500)
    assert out["decode_raised_peak"] is True and out["decode_peak_rss_bytes"] == 450
    assert out["load_peak_rss_bytes"] == 400


def test_one_disk_fault_in_a_cell_is_not_excused():
    loaded = snap()
    out = x9.host_b_result(loaded, [window(loaded, disk=(10, 11))], snap(disk=11), True, 500)
    assert out["window_disk_faults"] == 1 and out["no_disk_faults_in_windows"] is False


def test_a_peak_over_the_budget_fails_even_when_other_numbers_are_unknown():
    loaded = snap(peak=600)
    out = x9.host_b_result(loaded, [window(loaded, peak=600)], snap(peak=600), False, 500)
    assert out["within_budget"] is False and out["load_peak_rss_bytes"] is None


def test_unknown_numbers_never_pass():
    # no peak, as on macOS
    loaded = snap(peak=None)
    out = x9.host_b_result(loaded, [window(loaded, peak=None)], snap(peak=None), True, 500)
    assert out["within_budget"] is None
    # no disk fault counter
    loaded = snap(disk=None)
    out = x9.host_b_result(loaded, [window(loaded, disk=(None, None))], snap(disk=None), True, 500)
    assert out["window_disk_faults"] is None and out["no_disk_faults_in_windows"] is None
    # a load peak that is not separable
    loaded = snap()
    out = x9.host_b_result(loaded, [window(loaded)], snap(), False, 500)
    assert out["within_budget"] is None and out["load_disk_faults"] is None
    # no cell at all
    out = x9.host_b_result(loaded, [], snap(), True, 500)
    assert out["no_disk_faults_in_windows"] is None
    assert out["decode_sampled_rss_max_bytes"] is None


@pytest.mark.parametrize("change", [{"pid": 8}, {"restarts": 1}])
def test_a_restarted_worker_gives_no_verdict(change):
    loaded = snap()
    out = x9.host_b_result(loaded, [window(loaded)], snap(**change), True, 500)
    assert out["within_budget"] is None and out["no_disk_faults_in_windows"] is None
    assert out["lifetime_peak_rss_bytes"] is None
    inside = window(loaded, **change)
    assert not inside["same_worker"] and inside["disk_faults"] is None


def load_run(steps=(0,), requests=()):
    beats = [
        {"stage": 0, "epoch": 1, "t_ns": i, "steps": s, "busy_ns": s} for i, s in enumerate(steps)
    ]
    return report.Run(path=None, beats=beats, requests=list(requests))


def test_the_load_peak_counts_only_before_any_request():
    assert x9.load_is_clean(load_run((0, 0)))
    assert x9.load_is_clean(load_run((0,), [{"rid": None}]))  # refused before it had an id
    assert not x9.load_is_clean(load_run((0, 3)))
    assert not x9.load_is_clean(load_run((0,), [{"rid": "a"}]))
    assert not x9.load_is_clean(load_run(()))


@pytest.mark.parametrize(
    "args, ok",
    [
        (["tiny", "--stage", "0", "--serve"], True),
        (["tiny", "--stage", "0", "--serve", "--split-point", "9"], True),
        (["tiny", "--stage", "0", "--serve", "--split-point", "10"], False),
        (["tiny", "--stage", "1", "--serve"], False),
        (["other", "--stage", "0", "--serve"], False),
        (["tiny", "--stage", "0"], False),
        (None, False),
    ],
)
def test_the_monitor_must_be_reading_this_runs_stage_0_worker(args, ok):
    assert x9.worker_matches({"worker_args": args}, "tiny", 9) is ok


def ready_snapshot(**changes):
    args = ["tiny", "--stage", "0", "--serve", "--split-point", "9"]
    return {**snap(), "commit": "c1", "worker_args": args, **changes}


def ready_run(commit="c1", steps=(0,)):
    run = load_run(steps)
    run.info = {"commit": commit}
    return run


def test_a_run_that_can_give_a_valid_record_passes_the_preflight():
    assert x9.preflight(ready_snapshot(), ready_run(), "c1", "tiny", 9) == []


@pytest.mark.parametrize(
    "snapshot, run, commit, message",
    [
        ({}, {"steps": (4,)}, "c1", "already served a request"),
        ({"worker_args": ["tiny", "--stage", "1"]}, {}, "c1", "not reading a stage 0 worker"),
        ({"peak_rss_bytes": None}, {}, "c1", "peak_rss_bytes"),
        ({"disk_faults": None}, {}, "c1", "disk_faults"),
        ({"commit": "c2"}, {}, "c1", "one clean commit"),
        ({}, {"commit": "c2"}, "c1", "one clean commit"),
        ({}, {}, None, "one clean commit"),
        ({"commit": "c1-dirty"}, {"commit": "c1-dirty"}, "c1-dirty", "one clean commit"),
        ({"host": socket.gethostname()}, {}, "c1", "on this laptop"),
    ],
)
def test_the_preflight_names_what_would_spoil_the_record(snapshot, run, commit, message):
    problems = x9.preflight(ready_snapshot(**snapshot), ready_run(**run), commit, "tiny", 9)
    assert len(problems) == 1 and message in problems[0]


def test_only_trial_exempt_checks_may_fail_in_a_trial():
    good = {name: True for name in x9.CHECKS}
    assert x9.is_valid(good, False)
    assert not x9.is_valid({**good, "two_hosts": False}, False)
    assert x9.is_valid({**good, "two_hosts": False}, True)
    assert not x9.is_valid({**good, "load_separable": False}, True)
    missing = {name: True for name in x9.CHECKS[1:]}
    assert not x9.is_valid(missing, False)
    assert not x9.is_valid({**good, "requests_ok": None}, False)


# -------------------------------------------------------------- summary


def run_record(k, started=None, valid=True, within=True, paging=True, **extra):
    cells = [
        {
            "concurrency": c,
            "throughput_tok_s": 10.0 + r,
            "stage0_busy_s": 1.0,
            "stage1_busy_s": 0.5,
            "stage0_busy_ms_per_step": 20.0,
            "stage1_busy_ms_per_step": 10.0,
        }
        for c in (1, 4)
        for r in range(5)
    ]
    return {
        "script": "x9",
        "status": "ok",
        "model": MODEL,
        "revision": REVISION,
        "commit": "c1",
        "network": "lan",
        "concurrency": [1, 4],
        "repeats": 5,
        "requests": 16,
        "new_tokens": 32,
        "length": 128,
        "budget_bytes": 716 * MIB,
        "host": "macA",
        "host_b_host": "winB",
        "host_b_threads": 4,
        "split": k,
        "started": started or f"2026-11-27T10:{k:02d}:00",
        "trial": False,
        "valid": valid,
        "cells": cells,
        "host_b": {
            "load_peak_rss_bytes": 400 * MIB,
            "lifetime_peak_rss_bytes": 410 * MIB,
            "within_budget": within,
            "no_disk_faults_in_windows": paging,
        },
        **extra,
    }


def choice_record(k=6, **extra):
    return {
        "script": "x9-choice",
        "status": "ok",
        "model": MODEL,
        "revision": REVISION,
        "host_a": "macA",
        "host_b": "winB",
        "threads_b": 4,
        "budget_b_bytes": 716 * MIB,
        "trial": False,
        "result": {"k": k, "even_k": 15, "forced_below_even": False},
        "sweep": sweep_points(15, k),
        **extra,
    }


def sweep_records(**chosen):
    return [run_record(k) for k in (15, 12, 9)] + [run_record(6, **chosen)]


def test_a_complete_sweep_with_a_good_chosen_split_passes():
    found = x9.verdict(choice_record(), sweep_records())
    assert found["passed"] and found["missing_splits"] == []
    assert found["criterion_3_chosen_within_budget_no_disk_faults"] is True


def test_the_even_split_over_budget_is_a_result_not_a_failure_of_the_chosen_one():
    records = [run_record(15, within=False), run_record(12), run_record(9), run_record(6)]
    assert x9.verdict(choice_record(), records)["passed"]


@pytest.mark.parametrize(
    "chosen, third",
    [
        ({"within": False}, False),
        ({"paging": False}, False),
        ({"within": None}, None),
        ({"paging": None}, None),
        ({"valid": False}, None),
        ({"within": False, "valid": False}, False),
    ],
)
def test_the_chosen_split_must_be_shown_to_be_within_budget_and_not_paging(chosen, third):
    found = x9.verdict(choice_record(), sweep_records(**chosen))
    assert found["criterion_3_chosen_within_budget_no_disk_faults"] is third
    assert not found["passed"]


def test_every_deployment_of_the_chosen_split_counts():
    again = run_record(6, started="2026-11-27T12:00:00", paging=False)
    found = x9.verdict(choice_record(), [*sweep_records(), again])
    assert found["criterion_3_chosen_within_budget_no_disk_faults"] is False


@pytest.mark.parametrize("drop", [15, 12, 6])
def test_a_missing_split_fails_the_sweep(drop):
    records = [r for r in sweep_records() if r["split"] != drop]
    found = x9.verdict(choice_record(), records)
    assert found["missing_splits"] == [drop] and not found["passed"]
    if drop == 6:
        assert found["criterion_3_chosen_within_budget_no_disk_faults"] is None


def test_a_trial_or_no_split_never_passes():
    assert not x9.verdict(choice_record(trial=True), sweep_records())["passed"]
    assert not x9.verdict(choice_record(), sweep_records(trial=True))["passed"]
    assert not x9.verdict(choice_record(result=None), sweep_records())["passed"]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"commit": "c2"}, "differ in commit"),
        ({"budget_bytes": 1}, "differ in budget_bytes"),
        ({"host_b_threads": 2}, "differ in host_b_threads"),
        ({"requests": 8}, "differ in requests"),
        ({"network": None}, "lacks network"),
        ({"started": "2026-11-27T10:15:00"}, "given twice"),
    ],
)
def test_runs_that_cannot_be_compared_are_refused(change, message):
    with pytest.raises(x9.BadInput, match=message):
        x9.check_comparable([run_record(15), run_record(6, **change)])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model": "qwen3-0.6b"}, "another model"),
        ({"host_b": "x"}, "another Host B"),
        ({"host_a": "x"}, "another Host A"),
        ({"threads_b": 2}, "probed with 2 threads"),
        ({"budget_b_bytes": 1}, "different Host B budgets"),
    ],
)
def test_a_choice_made_for_other_runs_is_refused(change, message):
    with pytest.raises(x9.BadInput, match=message):
        x9.check_choice(choice_record(**change), sweep_records())


def test_the_summary_command_writes_the_tables_and_shows_unknown_as_unknown(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    choice = tmp_path / "choice.json"
    choice.write_text(json.dumps(choice_record()), encoding="utf-8")
    paths = []
    for rec in sweep_records(paging=None):
        paths.append(tmp_path / f"{rec['split']}.json")
        paths[-1].write_text(json.dumps(rec), encoding="utf-8")
    argv = ["summary", "--choice", str(choice), *map(str, paths)]
    assert x9.main(argv) == 1
    text = next((tmp_path / "runs").glob("*-x9-summary.md")).read_text(encoding="utf-8")
    assert "| 15 | 1 | 12.00 sd 1.58 (min 10.00, max 14.00, n 5)" in text
    assert "unknown" in text and "- passed: false" in text
    paths[-1].write_text(json.dumps(run_record(6)), encoding="utf-8")
    assert x9.main(argv) == 0
    assert "- passed: true" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "--rtt-ms is required"),
        (["--rtt-ms", "-1"], "--rtt-ms must be"),
        (["--rtt-ms", "3", "--requests", "4"], "twice the largest"),
        (["--rtt-ms", "3", "--concurrency", "1", "1"], "different positive"),
        (["--rtt-ms", "3", "--repeats", "0"], "--repeats must be positive"),
    ],
)
def test_bad_run_arguments_are_refused(tmp_path, capsys, extra, message):
    path = tmp_path / "gw.metrics.jsonl"
    metrics.Recorder(path, {"model": MODEL, "split": 15}).finish()
    argv = ["run", MODEL, "--metrics", str(path), "--budget-mib", "716.8", *extra]
    with pytest.raises(SystemExit):
        x9.main(argv)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "info, message",
    [({"model": "qwen3-0.6b", "split": 14}, "the gateway serves"), ({"model": MODEL}, "no split")],
)
def test_a_gateway_that_does_not_fit_the_run_is_refused(tmp_path, capsys, info, message):
    path = tmp_path / "gw.metrics.jsonl"
    metrics.Recorder(path, info).finish()
    argv = ["run", MODEL, "--metrics", str(path), "--budget-mib", "716.8", "--rtt-ms", "3"]
    with pytest.raises(SystemExit):
        x9.main(argv)
    assert message in capsys.readouterr().err


# -------------------------------------------------------------- one k for both laptops


def test_the_launcher_gives_one_split_point_to_the_gateway_and_both_stages():
    roles = ["gateway", "stage0", "stage1"]
    for command in launch.commands(MODEL, roles, 3, split_point=9).values():
        assert command[command.index("--split-point") + 1] == "9"
    for command in launch.commands(MODEL, roles, 3).values():
        assert "--split-point" not in command


def test_the_supervisor_passes_the_split_point_to_its_worker():
    command = worker_command(MODEL, 0, 3, split_point=9)
    assert command[-2:] == ["--split-point", "9"]
    assert "--split-point" not in worker_command(MODEL, 0, 3)


def test_a_worker_and_the_gateway_given_one_k_agree_on_the_layer_ranges(tmp_path):
    tiny_reference(tmp_path, layers=4)
    expected = ExpectedModel(b"", 4, 1)
    for stage in (0, 1):
        assert stage_layers(tmp_path, "tiny", stage, 1) == expected.layer_range(stage)
    for bad in (0, 4):
        with pytest.raises(ValueError, match="outside 1 to 3"):
            stage_layers(tmp_path, "tiny", 0, bad)


def test_the_override_leaves_the_committed_split_point_alone():
    assert config.split_point(MODEL, config.DEFAULT_PATH) == 15


def test_a_split_point_with_a_layer_range_is_refused(capsys):
    with pytest.raises(SystemExit):
        worker_main([MODEL, "--layers", "0:15", "--split-point", "9"])
    assert "--split-point needs --stage" in capsys.readouterr().err


# -------------------------------------------------------------- the sweep on a chain


def test_a_sweep_on_a_chain_reports_speed_busy_time_and_the_workers_counters(tmp_path):
    """A real gateway and two stages on loopback, with their heartbeats; the monitor reads
    this test process in place of a Host B worker."""
    tiny_reference(tmp_path, layers=4)
    run = System(tmp_path, 2, delay0=0.02, delay1=0.02)
    path = tmp_path / "gw.metrics.jsonl"
    run.gateway.recorder = metrics.Recorder(path, {"model": "tiny", "commit": "c1", "split": 2})
    stop = threading.Event()
    hearts = [
        threading.Thread(
            target=Heartbeat(run.address, servicer, interval=0.05).run, args=(stop,), daemon=True
        )
        for servicer in (run.servicer0, run.servicer1)
    ]
    monitor = x9monitor.Monitor(
        lambda: {"pid": os.getpid(), "restarts": 0},
        interval=0.01,
        commit="c1",
        args=lambda pid: ["tiny", "--stage", "0", "--serve"],
    )
    actions = {"/snapshot": monitor.snapshot, "/start": monitor.start, "/stop": monitor.stop}

    def call(job):
        return run.call(prompt=" ".join(map(str, job.ids)), max_tokens=6)

    args = SimpleNamespace(repeats=2, concurrency=[1, 2], requests=4, settle=0.3)
    lines = []
    try:
        for heart in hearts:
            heart.start()
        assert x1b.wait_until_ready(path, 10, poll=0.02, grace=0.2)
        prompt_ids = [[5, 9, 100], [3, 42, 7]]
        cells, loaded, clean, final = x9.sweep(
            call, path, lambda p: actions[p](), args, 2, prompt_ids, lines.append
        )
        whole = report.load(path)
        again = x9.load_is_clean(whole)
    finally:
        stop.set()
        run.close()
        run.gateway.recorder.finish()
    assert clean and not again  # after the sweep the load peak is no longer separable
    assert [(c["repeat"], c["concurrency"]) for c in cells] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert lines == cells
    for line in cells:
        assert line["split"] == 2 and line["completed"] == 4 and line["failed_requests"] == 0
        assert line["throughput_tok_s"] > 0
        for stage in (0, 1):
            assert line[f"stage{stage}_busy_s"] > 0
            assert line[f"stage{stage}_busy_ms_per_step"] > 0
        assert line["host_b"]["same_worker"]
        assert line["host_b"]["sampled_rss_max_bytes"] > 0
        assert line["host_b"]["disk_faults"] is not None
    host_b = x9.host_b_result(loaded, [c["host_b"] for c in cells], final, clean, 1 << 50)
    # macOS reports no peak for another process, so there the budget stays unknown
    assert host_b["within_budget"] is (None if sys.platform == "darwin" else True)
    rec = {"model": "tiny", "commit": "c1", "host": "gateway-laptop", "repeats": 2}
    checks = x9.run_checks(cells, loaded, clean, final, host_b, whole, rec, 2)
    assert set(checks) == set(x9.CHECKS)
    for name in ("requests_ok", "stage_busy_reported", "worker_unchanged", "worker_matches"):
        assert checks[name], name
    assert checks["load_separable"] and checks["same_commit"] and checks["two_hosts"]
    assert not checks["enough_repeats"] and not checks["rtt_recorded"]
    assert checks["counters_available"] is (sys.platform != "darwin")
    rows = x9.aggregate(cells)
    assert [(r["split"], r["concurrency"], r["repeats"]) for r in rows] == [(2, 1, 2), (2, 2, 2)]
    assert rows[0]["throughput_tok_s"]["n"] == 2


def test_the_run_command_writes_a_record_the_summary_reads(tmp_path, monkeypatch, capsys):
    """x9 run as typed, as a trial on one machine: a real gateway and stages on loopback,
    the monitor over HTTP reading this test process, and then x9 summary on its record."""
    tiny_reference(tmp_path, layers=4)
    run = System(tmp_path, 2, delay0=0.02, delay1=0.02)
    path = tmp_path / "gw.metrics.jsonl"
    info = {"model": MODEL, "commit": runrecord.commit_hash(), "split": 2}
    run.gateway.recorder = metrics.Recorder(path, info)

    def endpoint(address):
        name, port = address.rsplit(":", 1)
        return config.Endpoint(name, int(port))

    stages = [endpoint(f"127.0.0.1:{run.ports[stage]}") for stage in (0, 1)]
    hosts = config.HostConfig("loopback", endpoint(run.address), *stages)
    monkeypatch.setattr(config, "load", lambda path=None: hosts)
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(x9.prompts, "prompts", lambda model, length: [[5, 9, 100], [3, 42, 7]])
    words = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids)))
    monkeypatch.setattr(x9.prompts, "load_tokenizer", lambda model: words)
    monitor = x9monitor.Monitor(
        lambda: {"pid": os.getpid(), "restarts": 0},
        interval=0.01,
        commit=info["commit"],
        args=lambda pid: [MODEL, "--stage", "0", "--serve", "--threads", "1"],
    )
    server = x9monitor.serve(monitor, ("127.0.0.1", 0))
    stop = threading.Event()
    hearts = [
        threading.Thread(
            target=Heartbeat(run.address, servicer, interval=0.05).run, args=(stop,), daemon=True
        )
        for servicer in (run.servicer0, run.servicer1)
    ]
    argv = ["run", MODEL, "--metrics", str(path), "--budget-mib", "1000000", "--trial"]
    argv += ["--monitor", f"http://127.0.0.1:{server.server_address[1]}", "--repeats", "1"]
    argv += ["--concurrency", "1", "2", "--requests", "4", "--new-tokens", "6"]
    argv += ["--settle", "0.3", "--wait", "10"]
    try:
        for heart in hearts:
            heart.start()
        # without --trial, one machine is refused before any request is sent
        assert x1b.wait_until_ready(path, 10, poll=0.02, grace=0.2)
        strict = [a for a in argv if a != "--trial"] + ["--rtt-ms", "1", "--repeats", "5"]
        with pytest.raises(SystemExit):
            x9.main(strict)
        assert "on this laptop" in capsys.readouterr().err
        assert not report.load(path).requests
        code = x9.main(argv)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        run.close()
        run.gateway.recorder.finish()
    mac = sys.platform == "darwin"  # no peak RSS for another process there
    assert code == int(mac)
    written = next((tmp_path / "runs").glob("*-x9.json"))
    record = x9.load_record(written, "x9")
    assert record["split"] == 2 and record["trial"] and record["valid"] is not mac
    assert record["host_b_threads"] == 1 and len(record["cells"]) == 2
    assert record["checks"]["load_separable"] and record["checks"]["requests_ok"]
    assert not record["checks"]["two_hosts"]
    assert record["host_b"]["within_budget"] is (None if mac else True)
    lines = (tmp_path / "runs" / record["results_file"]).read_text(encoding="utf-8")
    kinds = [json.loads(line).get("type") for line in lines.splitlines()]
    assert kinds == ["cell", "cell", "aggregate", "aggregate", "summary"]
    choice = tmp_path / "choice.json"
    chosen = choice_record(k=1, trial=True, budget_b_bytes=record["budget_bytes"])
    chosen["result"]["even_k"] = 2
    chosen["sweep"] = [2, 1]
    choice.write_text(json.dumps(chosen), encoding="utf-8")
    capsys.readouterr()
    assert x9.main(["summary", "--choice", str(choice), str(written)]) == 1
    text = capsys.readouterr().out
    assert "| 2 | 1 |" in text and "- passed: false" in text
    assert "- missing_splits: " + ("[2, 1]" if mac else "[1]") in text
