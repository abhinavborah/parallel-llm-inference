"""X1b: agreement with each host's reference, and references that cannot be trusted are refused.

The pipeline tests run a real gateway and two real stages on loopback with the tiny model.
"""

import json
import math
import threading

import grpc
import pytest
import test_x1a
from test_gateway import LAYERS, System
from test_model import tiny_reference

from pllm import prompts, runrecord, x1a, x1b
from pllm.model import Model
from pllm.proto import protocol_pb2
from pllm.reference import decode

NEW = 6
IDS = [[5, 9, 100, 3, 42, 7], [60, 61, 62, 63], [1, 2, 3, 4, 5, 6, 7, 8]]
IDENTITY = {"model": "smollm2-135m", "revision": "r1", "length": 8, "new_tokens": NEW}


def entries_for(model_dir):
    model = Model.load(model_dir)
    traces = [decode(model, ids, NEW) for ids in IDS]
    return x1a.reference_entries(traces, IDS, list(range(len(IDS))))


def record(entries, host="host-a", **changes):
    rec = {"script": "x1a", "status": "ok", **IDENTITY, "threads": 2, "host": host}
    identity = {**IDENTITY, "threads": 2, "prompt_set_sha256": prompts.PROMPTS_SHA256}
    rec["reference"] = entries
    rec["reference_sha256"] = x1a.checksum({"identity": identity, "entries": entries})
    rec["summary"] = {"matched": 5, "comparisons": 5, "non_finite": False}
    rec.update(changes)
    return rec


def write(path, rec):
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


@pytest.fixture
def pipeline(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    system = System(tmp_path, 2)
    yield tmp_path, lambda ids: system.call(prompt=" ".join(map(str, ids)), max_tokens=NEW)
    system.close()


def lines_of(call, references, eos=()):
    lines = list(x1b.run(call, references, IDS, NEW, eos))
    assert lines[-1]["type"] == "served"
    return lines[:-1]


def by_type(lines, kind):
    return [line for line in lines if line["type"] == kind]


# -------------------------------------------------------------- agreement through the pipeline


def test_the_pipeline_agrees_with_each_reference_on_one_host(pipeline):
    model_dir, call = pipeline
    entries = entries_for(model_dir)
    lines = lines_of(call, {"a.json": record(entries), "b.json": record(entries, "host-b")})
    results = by_type(lines, "result")
    assert len(results) == 2 * len(IDS)
    assert all(r["first_divergence"] is None and r["agreed_tokens"] == NEW for r in results)
    assert all(r["max_logit_diff"] == 0.0 and not r["non_finite"] for r in results)
    for s in by_type(lines, "summary"):
        assert s["sequence_agreement_percent"] == 100.0 and s["token_agreement_percent"] == 100.0
        assert s["failed_requests"] == 0 and s["first_divergence_min"] is None
    assert [s["reference_host"] for s in by_type(lines, "summary")] == ["host-a", "host-b"]


def test_a_changed_token_is_the_first_divergence(pipeline):
    model_dir, call = pipeline
    entries = entries_for(model_dir)
    entries[1]["tokens"][3] = (entries[1]["tokens"][3] + 1) % 128
    lines = lines_of(call, {"a.json": record(entries)})
    second = by_type(lines, "result")[1]
    assert second["first_divergence"] == 3 and second["agreed_tokens"] == 3
    (s,) = by_type(lines, "summary")
    assert s["prompts_agreeing"] == 2 and s["first_divergence_min"] == 3
    assert s["token_agreement_percent"] == pytest.approx(100.0 * (3 * NEW - 3) / (3 * NEW))


def test_a_changed_top_logit_shows_in_the_logit_difference(pipeline):
    model_dir, call = pipeline
    entries = entries_for(model_dir)
    entries[0]["top_logits"][2] += 0.5
    (s,) = by_type(lines_of(call, {"a.json": record(entries)}), "summary")
    assert s["max_logit_diff"] == pytest.approx(0.5, abs=1e-5)
    assert s["sequence_agreement_percent"] == 100.0  # logits differ, tokens agree


class Failed(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "stage 0 is down"


def test_a_failed_request_is_counted_and_not_compared(pipeline):
    model_dir, call = pipeline
    entries = entries_for(model_dir)

    def flaky(ids):
        if ids == IDS[1]:
            raise Failed()
        return call(ids)

    lines = lines_of(flaky, {"a.json": record(entries)})
    assert by_type(lines, "result")[1]["error"] == "UNAVAILABLE: stage 0 is down"
    (s,) = by_type(lines, "summary")
    assert s["failed_requests"] == 1 and s["prompts"] == 3 and s["prompts_agreeing"] == 2


# -------------------------------------------------------------- comparison rules


def test_steps_after_the_first_divergence_are_not_compared():
    got = x1b.compare([1, 2, 3], [1.0, 2.0, 3.0], [1, 9, 3], [1.0, 2.5, 30.0])
    assert got["first_divergence"] == 1 and got["agreed_tokens"] == 1
    assert got["max_logit_diff"] == 0.5  # the divergence step counts, the one after it not


def test_a_stop_at_end_of_sequence_agrees_on_the_tokens_streamed():
    got = x1b.compare([1, 2, 7, 4], [1.0] * 4, [1, 2, 7], [1.0] * 3)
    assert got["first_divergence"] is None and got["compared_tokens"] == 3


def test_a_nan_logit_is_not_hidden():
    got = x1b.compare([1], [1.0], [1], [math.nan])
    assert got["non_finite"] and got["max_logit_diff"] is None


def tokens(*ids):
    stream = [protocol_pb2.Token(rid="r", step=i, token=t) for i, t in enumerate(ids)]
    return lambda _ids: iter(stream)


def test_a_short_stream_must_end_on_an_end_of_sequence_token():
    assert x1b.serve(tokens(1, 7), [1], 4, {7})["error"] is None
    assert "without an end of sequence" in x1b.serve(tokens(1, 2), [1], 4, {7})["error"]
    assert "more than" in x1b.serve(tokens(1, 2, 3), [1], 2, {7})["error"]


# -------------------------------------------------------------- references are checked


def plain_entries():
    return [
        {
            "index": i,
            "prompt_tokens": len(ids),
            "prompt_sha256": x1b.prompt_sha256(ids),
            "tokens": [1] * NEW,
            "top_logits": [0.5] * NEW,
        }
        for i, ids in enumerate(IDS)
    ]


def test_a_valid_record_loads(tmp_path):
    rec = x1b.load_reference(write(tmp_path / "a.json", record(plain_entries())))
    assert rec["host"] == "host-a"
    x1b.check_references([rec, rec], IDS)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"status": "error"}, "status ok"),
        ({"script": "x2"}, "status ok"),
        ({"threads": 4}, "checksum"),
        ({"length": 16}, "checksum"),
        ({"reference_sha256": "0" * 64}, "checksum"),
    ],
)
def test_a_record_that_cannot_be_trusted_is_refused(tmp_path, changes, match):
    with pytest.raises(x1b.BadReference, match=match):
        x1b.load_reference(write(tmp_path / "a.json", record(plain_entries(), **changes)))


def test_an_edited_reference_token_is_refused(tmp_path):
    rec = record(plain_entries())
    rec["reference"][0]["tokens"][0] = 99
    with pytest.raises(x1b.BadReference, match="checksum"):
        x1b.load_reference(write(tmp_path / "a.json", rec))


def test_a_file_that_is_not_a_record_is_refused(tmp_path):
    (tmp_path / "a.json").write_text("{cut", encoding="utf-8")
    with pytest.raises(x1b.BadReference, match="readable"):
        x1b.load_reference(tmp_path / "a.json")


def test_references_with_other_settings_or_prompts_are_refused():
    a = record(plain_entries())
    with pytest.raises(x1b.BadReference, match="new_tokens"):
        x1b.check_references([a, {**a, "new_tokens": NEW + 1}], IDS)
    with pytest.raises(x1b.BadReference, match="different prompts"):
        x1b.check_references([a, {**a, "reference": a["reference"][:2]}], IDS)
    with pytest.raises(x1b.BadReference, match="prompt 1 is not"):
        x1b.check_references([a], [IDS[0], [9, 9], IDS[2]])


# -------------------------------------------------------------- main


def beat(stage):
    return {"type": "beat", "t_ns": 1, "stage": stage, "epoch": 1, "busy_ns": 0, "steps": 0}


def metrics_file(path, model="smollm2-135m", stages=(0, 1)):
    head = {"type": "run", "model": model, "split": 15}
    lines = [head, *(beat(stage) for stage in stages)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    return path


def test_the_wait_ends_once_both_stages_beat(tmp_path):
    path = metrics_file(tmp_path / "g.metrics.jsonl", stages=(0,))
    assert not x1b.wait_until_ready(path, 0.2, poll=0.05, grace=0)

    def second():
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(beat(1)) + "\n")

    timer = threading.Timer(0.2, second)
    timer.start()
    assert x1b.wait_until_ready(path, 10, poll=0.05, grace=0)
    timer.join()


def test_main_sends_nothing_until_both_stages_beat(tmp_path, monkeypatch):
    monkeypatch.setattr(x1b.client, "generate", lambda *a: pytest.fail("a request was sent"))
    monkeypatch.setattr(x1b.prompts, "prompts", lambda model, length: IDS)
    monkeypatch.setattr(x1b.prompts, "load_tokenizer", lambda model: Words())
    monkeypatch.setattr(x1b.prompts, "round_trips", lambda tokenizer, ids: True)
    ref = write(tmp_path / "a.json", record(plain_entries()))
    metrics = metrics_file(tmp_path / "g.metrics.jsonl", stages=(1,))
    argv = ["--metrics", str(metrics), "--reference", str(ref), "--wait", "0.2"]
    with pytest.raises(SystemExit):
        x1b.main(["smollm2-135m", *argv])


def test_a_record_written_by_x1a_loads_as_a_reference(tmp_path, monkeypatch, capsys):
    code, rec, _ = test_x1a.run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    loaded = x1b.load_reference(rec["record_file"])
    x1b.check_references([loaded, loaded], test_x1a.prompts_for_tests())
    assert loaded["new_tokens"] == 4 and len(loaded["reference"]) == 3


def test_main_refuses_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(x1b.client, "generate", lambda *a: pytest.fail("a request was sent"))
    good = write(tmp_path / "a.json", record(plain_entries()))
    bad = record(plain_entries())
    bad["reference"][0]["top_logits"][0] = 1.0
    bad = write(tmp_path / "b.json", bad)
    metrics = metrics_file(tmp_path / "g.metrics.jsonl")
    other = metrics_file(tmp_path / "q.metrics.jsonl", "qwen3-0.6b")
    for argv in (
        ["--metrics", str(other), "--reference", str(good)],
        ["--metrics", str(metrics), "--reference", str(good), str(bad)],
        ["--metrics", str(tmp_path / "none"), "--reference", str(good)],
    ):
        with pytest.raises(SystemExit):
            x1b.main(["smollm2-135m", *argv])


class Words:
    def decode(self, ids):
        return " ".join(map(str, ids))


def test_main_writes_a_record_with_its_command(tmp_path, monkeypatch, capsys):
    entries = plain_entries()
    monkeypatch.setattr(x1b.prompts, "prompts", lambda model, length: IDS)
    monkeypatch.setattr(x1b.prompts, "load_tokenizer", lambda model: Words())
    monkeypatch.setattr(x1b.prompts, "round_trips", lambda tokenizer, ids: True)
    monkeypatch.setattr(x1b, "eos_ids", lambda model_dir: [])
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")

    def generate(address, text, max_tokens, timeout):
        assert max_tokens == NEW
        for i in range(NEW):
            yield protocol_pb2.Token(rid=text, step=i, token=1, top_logit=0.5)

    monkeypatch.setattr(x1b.client, "generate", generate)
    monkeypatch.setattr(x1b.sys, "argv", ["x1b", "the", "command"])
    ref = write(tmp_path / "a.json", record(entries))
    metrics = metrics_file(tmp_path / "g.metrics.jsonl")
    assert x1b.main(["smollm2-135m", "--metrics", str(metrics), "--reference", str(ref)]) == 0
    (record_file,) = (tmp_path / "runs").glob("*-x1b.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    assert rec["status"] == "ok" and rec["argv"] == ["x1b", "the", "command"]
    assert rec["split"] == 15 and rec["metrics_file"] == metrics.name
    assert rec["references"][0]["x1a_passed"] is True
    assert [s["token_agreement_percent"] for s in rec["summary"]] == [100.0]
    assert [r["tokens"] for r in rec["served"]] == [[1] * NEW] * len(IDS)
    lines = (tmp_path / "runs" / rec["results_file"]).read_text(encoding="utf-8")
    assert lines == capsys.readouterr().out and len(lines.splitlines()) == len(IDS) + 1
