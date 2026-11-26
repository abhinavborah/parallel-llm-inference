"""The X2 runner: round-robin serving, cell extraction, ordering checks and the spread."""

import json
import threading
import time

import grpc
import pytest
import torch
from test_model import tiny_reference

from pllm import metrics, report, x2
from pllm.model import Model
from pllm.reference import decode

VOCAB = 16


class Clock:
    """Goes up by 1 ms on every read, so every stamp is distinct and ordered."""

    def __init__(self):
        self.t = 0

    def __call__(self):
        self.t += 1_000_000
        return self.t


class FakeModel:
    """Next token is (start_pos + number of inputs) % VOCAB; records which cache each
    forward used, so the serving order can be read back."""

    def __init__(self):
        self.calls = []

    def new_cache(self):
        return object()

    def forward(self, tokens, start_pos, cache):
        self.calls.append((cache, start_pos, len(tokens)))
        logits = torch.zeros(len(tokens), VOCAB)
        logits[-1, (start_pos + len(tokens)) % VOCAB] = 1.0
        return logits


def single(tmp_path, cap=x2.MAX_RUNNING, eos=(), model=None):
    recorder = metrics.Recorder(tmp_path / "s.metrics.jsonl", {"model": "fake"}, clock=Clock())
    return x2.SingleProcess(model or FakeModel(), recorder, list(eos), cap=cap), recorder


def cell_of(count, lengths=(3, 4, 5)):
    return x2.jobs([[1] * n for n in lengths], count)


# -------------------------------------------------------------- workload


def test_jobs_cycle_the_prompts_in_order():
    cell = x2.jobs([[1], [2, 2], [3, 3, 3]], 5)
    assert [job.prompt for job in cell] == [0, 1, 2, 0, 1]
    assert [job.ids for job in cell] == [[1], [2, 2], [3, 3, 3], [1], [2, 2]]
    assert x2.jobs([[1], [2, 2]], 3)[0] is not x2.jobs([[1], [2, 2]], 3)[0]  # fresh per cell


# -------------------------------------------------------------- single process


def test_each_running_request_takes_one_step_per_turn(tmp_path):
    server, _ = single(tmp_path)
    cell = cell_of(3)
    server.serve(cell, 3, new_tokens=3, prefix="c")
    caches = [call[0] for call in server.model.calls]
    first_use = list(dict.fromkeys(caches))  # one cache per request, in first-use order
    assert len(first_use) == 3
    rounds = [first_use.index(c) for c in caches]
    assert rounds == [0, 1, 2] * 3  # prefill round, then two decode rounds
    assert [call[2] for call in server.model.calls[:3]] == [3, 4, 5]  # prefill: whole prompt
    assert all(call[2] == 1 for call in server.model.calls[3:])
    for job in cell:
        assert len(job.tokens) == 3 and job.steps == [0, 1, 2]
        assert job.cache is None  # freed


def test_decode_positions_follow_the_prompt(tmp_path):
    server, _ = single(tmp_path)
    cell = cell_of(1, lengths=(4,))
    server.serve(cell, 1, new_tokens=3, prefix="c")
    assert [call[1] for call in server.model.calls] == [0, 4, 5]
    assert cell[0].tokens == [4, 5, 6]  # (start + inputs) % VOCAB


def test_closed_loop_replaces_a_finished_request_at_once(tmp_path):
    server, _ = single(tmp_path)
    cell = cell_of(4, lengths=(1,))
    server.serve(cell, 2, new_tokens=2, prefix="c")
    assert [job.rid for job in cell] == ["c-0", "c-1", "c-2", "c-3"]
    # two outstanding at first; the third arrives only after the first ended
    assert cell[0].arrived_ns < cell[1].arrived_ns < cell[0].last_token_ns
    assert cell[2].arrived_ns > cell[0].last_token_ns
    assert cell[3].arrived_ns > cell[1].last_token_ns


def test_the_running_cap_queues_the_rest_in_arrival_order(tmp_path):
    server, _ = single(tmp_path, cap=2)
    cell = cell_of(4, lengths=(2,))
    server.serve(cell, 4, new_tokens=2, prefix="c")
    first_use = list(dict.fromkeys(call[0] for call in server.model.calls))
    assert len(first_use) == 4
    # all four arrive at the start, but the last two get their first step only after a
    # running one ended, so their wait counts in TTFT
    assert cell[3].arrived_ns < cell[0].last_token_ns < cell[2].first_token_ns
    assert cell[2].first_token_ns < cell[3].first_token_ns
    assert max(len(job.tokens) for job in cell) == 2


def test_end_of_sequence_ends_a_request_early(tmp_path):
    server, recorder = single(tmp_path, eos=[5])
    cell = cell_of(1, lengths=(4,))
    server.serve(cell, 1, new_tokens=10, prefix="c")
    assert cell[0].tokens == [4, 5]
    recorder.finish()
    run = report.load(tmp_path / "s.metrics.jsonl")
    assert [r["outcome"] for r in run.requests] == ["end of sequence"]


def test_single_mode_file_reads_through_report(tmp_path):
    server, recorder = single(tmp_path)
    server.serve(cell_of(2), 1, new_tokens=1, prefix="warmup")
    cell = cell_of(6)
    server.serve(cell, 3, new_tokens=4, prefix="x2")
    recorder.finish()
    run, missing = x2.cell_run(tmp_path / "s.metrics.jsonl", {job.rid for job in cell})
    assert missing == 0 and len(run.requests) == 6  # the warm-up is left out
    line = x2.cell_line("single", 3, 0, cell, run, missing)
    assert line["completed"] == 6 and line["output_tokens"] == 24
    assert line["ttft_n"] == line["tpot_n"] == 6
    assert 0 < line["stage0_utilisation"] <= 1
    assert line["stage1_utilisation"] is None
    assert line["hop_order_violations"] is None and line["stream_step_violations"] is None
    assert set(line["outputs"]) == {"0", "1", "2"}
    json.dumps(line, allow_nan=False)


def test_single_mode_matches_the_reference_on_a_real_model(tmp_path):
    tiny_reference(tmp_path, layers=2)
    model = Model.load(tmp_path)
    ids = [[5, 9, 100, 3], [11], [60, 61, 62]]
    server, _ = single(tmp_path, model=model)
    cell = x2.jobs(ids, 4)
    server.serve(cell, 2, new_tokens=5, prefix="c")
    for job in cell:
        assert job.tokens == decode(model, job.ids, 5)[0]


def test_peak_rss_is_a_plausible_byte_count():
    assert 10 * 2**20 < x2.rss() <= x2.peak_rss() < 64 * 2**30


def test_serving_samples_the_rss_at_beats(tmp_path):
    server, _ = single(tmp_path)
    assert server.rss_serving_max == 0
    server.serve(cell_of(2), 1, new_tokens=2, prefix="c")
    assert 10 * 2**20 < server.rss_serving_max <= x2.peak_rss()


# -------------------------------------------------------------- pipeline client


class Failed(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.RESOURCE_EXHAUSTED

    def details(self):
        return "queue full"


def test_pipeline_keeps_concurrency_requests_outstanding(tmp_path):
    lock = threading.Lock()
    now = {"in": 0, "max": 0}

    def call(job):
        with lock:
            now["in"] += 1
            now["max"] = max(now["max"], now["in"])
        try:
            time.sleep(0.02)
            if job.prompt == 1 and job.ids == [9]:
                raise Failed()
            for step in range(3):
                yield type("Token", (), {"rid": f"r{id(job)}", "token": step, "step": step})
        finally:
            with lock:
                now["in"] -= 1

    cell = x2.jobs([[1], [9], [3]], 9)
    x2.serve_pipeline(call, cell, 3)
    assert now["max"] == 3
    failed = [job for job in cell if job.error]
    assert len(failed) == 3 and failed[0].error == "RESOURCE_EXHAUSTED: queue full"
    assert all(job.tokens == [0, 1, 2] for job in cell if not job.error)


def hop(rid, step, hop="stage1_gateway", attempt=0):
    return {"type": "hop", "hop": hop, "rid": rid, "attempt": attempt, "step": step}


def test_hop_order_counts_a_step_that_does_not_rise(tmp_path):
    path = tmp_path / "g.metrics.jsonl"
    rows = [
        hop("a", 0, "gateway_stage0"),
        hop("a", 0),
        hop("b", 0),
        hop("a", 1),
        hop("a", 1),  # repeated: a violation
        hop("a", 0, attempt=1),  # a new attempt starts afresh
        hop("z", 0),
        hop("z", 0),  # another cell's request: not counted
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"type": "ho', encoding="utf-8")
    hops = x2.read_hops(path, {"a", "b"})
    assert len(hops) == 6
    assert x2.hop_violations(hops) == 1
    assert x2.hop_violations([hop("a", 0, "gateway_stage0"), hop("a", 0)]) == 0  # per hop


def test_stream_steps_must_run_from_zero_without_gaps():
    ok, gap, repeat = x2.Job(0, [1]), x2.Job(0, [1]), x2.Job(0, [1])
    ok.steps, gap.steps, repeat.steps = [0, 1, 2], [0, 2], [0, 1, 1]
    assert x2.stream_violations([ok, gap, repeat]) == 2


def test_cell_run_counts_requests_with_no_line(tmp_path):
    path = tmp_path / "g.metrics.jsonl"
    rows = [
        {"type": "run", "model": "m"},
        {
            "type": "request",
            "rid": "a",
            "outcome": "max_tokens",
            "output_tokens": 2,
            "arrived_ns": 0,
            "first_token_ns": 10,
            "last_token_ns": 30,
        },
        {
            "type": "request",
            "rid": "other",
            "outcome": "max_tokens",
            "output_tokens": 2,
            "arrived_ns": 0,
            "first_token_ns": 10,
            "last_token_ns": 30,
        },
        {"type": "beat", "t_ns": 5, "stage": 0, "epoch": 1, "busy_ns": 0, "steps": 0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    run, missing = x2.cell_run(path, {"a", "b"})
    assert [r["rid"] for r in run.requests] == ["a"] and missing == 1
    assert len(run.beats) == 1


# -------------------------------------------------------------- spread


def test_spread_over_repeats():
    s = x2.spread([2.0, None, 4.0, 6.0])
    assert s == {"n": 3, "mean": 4.0, "stdev": 2.0, "min": 2.0, "max": 6.0}
    assert x2.spread([1.0])["stdev"] is None
    assert x2.spread([None])["n"] == 0


def line(mode, c, throughput, violations=0, failed=0):
    out = {name: None for name in x2.METRICS}
    out.update(
        mode=mode,
        concurrency=c,
        throughput_tok_s=throughput,
        failed_requests=failed,
        missing_in_metrics=0,
        hop_order_violations=violations if mode == "pipeline" else None,
        stream_step_violations=0 if mode == "pipeline" else None,
    )
    return out


def test_aggregate_groups_by_mode_and_concurrency():
    cells = [line("pipeline", 2, 10.0), line("pipeline", 2, 20.0, violations=1)]
    cells += [line("pipeline", 1, 5.0), line("single", 1, 8.0)]
    rows = x2.aggregate(cells)
    assert [(r["mode"], r["concurrency"], r["repeats"]) for r in rows] == [
        ("pipeline", 1, 1),
        ("pipeline", 2, 2),
        ("single", 1, 1),
    ]
    assert rows[1]["throughput_tok_s"]["mean"] == 15.0
    assert rows[1]["hop_order_violations"] == 1
    assert rows[2]["hop_order_violations"] is None
    summary = x2.summary_line(cells)
    assert summary["ordering_violations"] == 1 and not summary["passed"]
    assert x2.summary_line([line("single", 1, 8.0)])["passed"]
    assert not x2.summary_line([line("pipeline", 1, 8.0, failed=1)])["passed"]


@pytest.mark.parametrize(
    "argv",
    [
        ["single", "smollm2-135m", "--requests", "3", "--concurrency", "2"],
        ["single", "smollm2-135m", "--repeats", "0"],
        ["pipeline", "smollm2-135m"],
    ],
)
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit) as stop:
        x2.main(argv)
    assert stop.value.code == 2
