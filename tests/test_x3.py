"""X3: bytes per hop from a real gateway run on loopback, and runs that cannot show them fail.

The tiny model has hidden size 64, so an activation position is 256 bytes.
"""

import json
import shutil
import threading

import pytest
from test_gateway import HIDDEN, LAYERS, System
from test_model import tiny_reference

from pllm import metrics, report, runrecord, x1b, x2, x3
from pllm.proto import protocol_pb2

IDS = [[5, 9, 100, 3, 42, 7, 11, 12], [60, 61, 62, 63, 64, 65, 66, 67]]
NEW = 4
MODEL = "smollm2-135m"  # the name main checks; the weights are the tiny model's


@pytest.fixture
def served(tmp_path):
    """A complete metrics file of two 8-token prompts, and what the client received."""
    model_dir = tmp_path / "models" / MODEL
    tiny_reference(model_dir, layers=LAYERS)
    path = tmp_path / "gateway.metrics.jsonl"
    system = System(model_dir, 2)
    system.gateway.recorder = metrics.Recorder(path, {"model": MODEL, "split": 2})

    def call(ids):
        return system.call(prompt=" ".join(map(str, ids)), max_tokens=NEW)

    try:
        got = [x1b.serve(call, ids, NEW, frozenset()) for ids in IDS]
        system.gateway.close()  # ends the stream to Stage 0, so its byte totals arrive
        system.gateway.recorder.finish()
    finally:
        system.close()
    return path, got


def analyse(path, got, failed=0):
    run = report.load(path)
    rids = {g["rid"] for g in got}
    table = x3.rows(run, x2.read_hops(path, rids))
    return table, x3.problems(run, rids, failed, table, HIDDEN)


def lines(path):
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]


def request_line(rid):
    return {
        "type": "request",
        "t_ns": 1,
        "rid": rid,
        "outcome": "max_tokens",
        "output_tokens": 1,
        "arrived_ns": 0,
        "first_token_ns": 1,
        "last_token_ns": 1,
    }


def rewrite(path, change):
    """The file with change(line) applied to every line; None drops the line."""
    kept = [change(line) for line in lines(path)]
    path.write_text("".join(json.dumps(line) + "\n" for line in kept if line), encoding="utf-8")


def test_activation_bytes_are_hidden_size_x_4_per_position_within_the_overhead(served):
    path, got = served
    assert all(g["error"] is None and len(g["tokens"]) == NEW for g in got)
    table, found = analyse(path, got)
    assert found == []
    forwarded = {r["phase"]: r for r in table if r["hop"] == metrics.STAGE0_STAGE1}
    assert forwarded["prefill"]["positions"] == 2 * 8
    assert forwarded["prefill"]["payload_bytes"] == 2 * 8 * HIDDEN * 4
    assert forwarded["decode"]["messages"] == forwarded["decode"]["positions"] == 2 * (NEW - 1)
    assert forwarded["decode"]["payload_bytes"] == 2 * (NEW - 1) * HIDDEN * 4
    for r in forwarded.values():
        assert 0 < r["overhead_mean_bytes"] <= x3.OVERHEAD_BOUND
        assert r["overhead_max_bytes"] is None and r["source"] == "stage 0 totals"
    sent = [r for r in table if r["hop"] == metrics.GATEWAY_STAGE0]
    assert {r["phase"] for r in sent} == {"prefill", "decode"}
    assert all(r["payload_bytes"] == 0 and r["overhead_max_bytes"] > 0 for r in sent)
    results = [r for r in table if r["hop"] == metrics.STAGE1_GATEWAY]
    assert sum(r["messages"] for r in results) == 2 * NEW


def test_the_summary_gives_the_mean_and_largest_overhead_per_hop(served):
    path, got = served
    table, found = analyse(path, got)
    s = x3.summary_line(table, found, HIDDEN)
    assert s["passed"] and s["expected_payload_bytes_per_position"] == HIDDEN * 4
    assert len(s["overhead"]) == len(table)
    assert all(o["mean_bytes"] is not None for o in s["overhead"])


# -------------------------------------------------------------- runs that fail


def test_a_file_without_run_end_fails(served):
    path, got = served
    rewrite(path, lambda line: None if line["type"] == "run_end" else line)
    assert x3.wait_for_end(path, 0.2, poll=0.05).end is None
    assert "no run_end" in analyse(path, got)[1][0]


def test_incomplete_stage_0_totals_fail(served):
    path, got = served
    rewrite(
        path,
        lambda line: (
            {**line, "stage0_stage1_complete": False} if line["type"] == "run_end" else line
        ),
    )
    assert any("incomplete" in p for p in analyse(path, got)[1])


def test_a_request_from_another_client_fails(served):
    path, got = served
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(request_line("other")) + "\n")
    assert any("not this run's" in p for p in analyse(path, got)[1])


def test_a_failed_request_fails(served):
    path, got = served
    assert any("failed at the client" in p for p in analyse(path, got, failed=1)[1])


def test_a_payload_that_is_not_hidden_size_x_4_per_position_fails(served):
    path, got = served

    def short(line):
        if line["type"] == "bytes" and line["hop"] == metrics.STAGE0_STAGE1:
            return {**line, "payload_bytes": line["payload_bytes"] - 4}
        return line

    rewrite(path, short)
    found = analyse(path, got)[1]
    assert sum("payload" in p for p in found) == 2  # prefill and decode


def test_overhead_above_the_bound_fails(served):
    path, got = served

    def padded(line):
        if line["type"] == "bytes" and line["hop"] == metrics.STAGE0_STAGE1:
            return {**line, "bytes": line["bytes"] + line["messages"] * x3.OVERHEAD_BOUND}
        return line

    rewrite(path, padded)
    assert any("overhead per message" in p for p in analyse(path, got)[1])


def test_a_lost_forwarded_step_fails(served):
    path, got = served

    def lost(line):
        if line["type"] == "bytes" and line["hop"] == metrics.STAGE0_STAGE1:
            if line["phase"] == "decode":
                return {**line, "positions": line["positions"] - 1}
        return line

    rewrite(path, lost)
    assert any("not as sent" in p for p in analyse(path, got)[1])


def test_the_wait_ends_when_the_gateway_stops(served, tmp_path):
    path, _ = served
    end = lines(path)[-1]
    rewrite(path, lambda line: None if line["type"] == "run_end" else line)

    def stop():
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(end) + "\n")

    timer = threading.Timer(0.2, stop)
    timer.start()
    assert x3.wait_for_end(path, 10, poll=0.05).end == end
    timer.join()


# -------------------------------------------------------------- main


def fresh(path, model=MODEL, extra=(), stages=(0, 1)):
    head = {"type": "run", "model": model, "split": 2, "clock": "perf_counter_ns", "start_ns": 0}
    beats = [
        {"type": "beat", "t_ns": 1, "stage": stage, "epoch": 1, "busy_ns": 0, "steps": 0}
        for stage in stages
    ]
    lines = (head, *beats, *extra)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    return path


def test_main_sends_nothing_until_both_stages_beat(tmp_path, monkeypatch):
    tiny_reference(tmp_path / "models" / MODEL, layers=LAYERS)
    monkeypatch.setattr(x3, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(x3.prompts, "prompts", lambda model, length: IDS)
    monkeypatch.setattr(x3.prompts, "load_tokenizer", lambda model: Words())
    monkeypatch.setattr(x3.prompts, "round_trips", lambda tokenizer, ids: True)
    monkeypatch.setattr(x3.client, "generate", lambda *a: pytest.fail("a request was sent"))
    path = fresh(tmp_path / "g.metrics.jsonl", stages=(0,))
    with pytest.raises(SystemExit):
        x3.main([MODEL, "--length", "16", "--metrics", str(path), "--wait", "0.2"])


def test_main_refuses_a_file_that_is_not_fresh_or_for_another_model(tmp_path, monkeypatch):
    monkeypatch.setattr(x3.client, "generate", lambda *a: pytest.fail("a request was sent"))
    for path in (
        fresh(tmp_path / "a.jsonl", model="qwen3-0.6b"),
        fresh(tmp_path / "b.jsonl", extra=[request_line("r")]),
        fresh(tmp_path / "c.jsonl", extra=[{"type": "run_end", "t_ns": 1}]),
        fresh(tmp_path / "d.jsonl", extra=[{"type": "request", "rid": "cut"}]),
        tmp_path / "missing.jsonl",
    ):
        with pytest.raises(SystemExit):
            x3.main([MODEL, "--length", "16", "--metrics", str(path), "--wait", "1"])


def test_an_unreadable_line_fails(served):
    path, got = served
    with path.open("a", encoding="utf-8") as file:
        file.write('{"type": "request", "rid"\n')
    assert any("unreadable" in p for p in analyse(path, got)[1])


class Words:
    def decode(self, ids):
        return " ".join(map(str, ids))


def test_main_writes_the_lines_and_a_record_with_its_command(served, tmp_path, monkeypatch, capsys):
    done, got = served
    path = fresh(tmp_path / "run.metrics.jsonl")
    streams = iter(got)

    def generate(address, text, max_tokens, timeout):
        g = next(streams)
        for step, token in enumerate(g["tokens"]):
            yield protocol_pb2.Token(rid=g["rid"], step=step, token=token)

    def stopped(metrics_path, wait):
        shutil.copyfile(done, metrics_path)  # the gateway's lines up to its stop
        return report.load(metrics_path)

    monkeypatch.setattr(x3, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(x3.prompts, "prompts", lambda model, length: IDS)
    monkeypatch.setattr(x3.prompts, "load_tokenizer", lambda model: Words())
    monkeypatch.setattr(x3.prompts, "round_trips", lambda tokenizer, ids: True)
    monkeypatch.setattr(x3, "eos_ids", lambda model_dir: [])
    monkeypatch.setattr(x3.client, "generate", generate)
    monkeypatch.setattr(x3, "wait_for_end", stopped)
    monkeypatch.setattr(x3.sys, "argv", ["x3", "the", "command"])
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    argv = [MODEL, "--length", "16", "--metrics", str(path), "--count", "2", "--new-tokens"]
    assert x3.main([*argv, str(NEW)]) == 0
    (record_file,) = (tmp_path / "runs").glob("*-x3.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    assert rec["status"] == "ok" and rec["argv"] == ["x3", "the", "command"]
    assert rec["hidden_size"] == HIDDEN and rec["length"] == 16 and rec["split"] == 2
    assert [r["output_tokens"] for r in rec["requests"]] == [NEW, NEW]
    assert rec["summary"]["passed"] and rec["summary"]["problems"] == []
    captured = capsys.readouterr()
    assert x3.STOP_NOW in captured.err
    assert (tmp_path / "runs" / rec["results_file"]).read_text(encoding="utf-8") == captured.out
