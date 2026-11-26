"""X9 (R7.16, R8.6, R8.7): the capacity-aware split, the memory counters, the Host B
monitor, the probe, the verdicts, the split override, and one sweep on a real chain on
loopback. Small random models only."""

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest
from test_model import tiny_reference

from pllm import (
    capacity,
    config,
    launch,
    procmem,
)
from pllm.capacity import HostCapacity, NoSplit, choose_split, sweep_points
from pllm.control import ExpectedModel
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
