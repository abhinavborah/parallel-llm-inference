"""The X1a harness: exact match reported per prompt and split, and a failure is seen."""

import json

import pytest
import torch
from test_model import tiny_reference

from pllm import runrecord, x1a
from pllm.model import Model
from pllm.split import InProcessSplit

LAYERS = 5
THREADS = 2


@pytest.fixture(autouse=True)
def fixed_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    yield
    torch.set_num_threads(before)


def prompts_for_tests():
    generator = torch.Generator().manual_seed(7)
    return [torch.randint(0, 128, (n,), generator=generator).tolist() for n in (6, 9, 12)]


IDENTITY = {"model": "tiny", "length": 9, "threads": THREADS}


def sweep(tmp_path, load_split=None, points=None, threads=THREADS):
    load_split = load_split or (lambda k: InProcessSplit.load(tmp_path, k))
    return list(
        x1a.run_sweep(
            lambda: Model.load(tmp_path),
            load_split,
            prompts_for_tests(),
            points or [1, 2, 3, 4],
            new_tokens=6,
            threads=threads,
            identity=IDENTITY,
        )
    )


@pytest.mark.parametrize("tie", [True, False])
def test_every_prompt_and_split_matches_exactly(tmp_path, tie):
    tiny_reference(tmp_path, tie, layers=LAYERS)
    lines = sweep(tmp_path)
    results, summary = lines[:-1], lines[-1]
    assert len(results) == 3 * 4
    assert {(r["index"], r["split"]) for r in results} == {
        (i, k) for i in range(3) for k in (1, 2, 3, 4)
    }
    assert all(r["tokens_match"] and r["max_logit_diff"] == 0.0 for r in results)
    assert summary["match_percent"] == 100.0 and summary["max_logit_diff"] == 0.0
    assert summary["comparisons"] == summary["matched"] == 12
    assert summary["threads"] == THREADS and summary["split_points"] == [1, 2, 3, 4]
    for line in lines:  # JSON lines: every line is serialisable on its own
        json.loads(json.dumps({k: v for k, v in line.items() if k != "reference"}))


def test_a_perturbed_stage_is_reported_as_a_mismatch(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)

    def broken(k):
        split = InProcessSplit.load(tmp_path, k)
        split.stage1.layers[0].wo = split.stage1.layers[0].wo * 1.01
        return split

    lines = sweep(tmp_path, broken, points=[2])
    results, summary = lines[:-1], lines[-1]
    assert all(r["max_logit_diff"] > 0 for r in results)
    assert summary["max_logit_diff"] > 0


def test_a_different_token_is_reported_as_not_matching(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)

    def other_head(k):
        split = InProcessSplit.load(tmp_path, k)
        split.stage1.lm_head = torch.randn_like(split.stage1.lm_head) * 5
        return split

    lines = sweep(tmp_path, other_head, points=[2])
    assert lines[-1]["matched"] < lines[-1]["comparisons"]
    assert lines[-1]["match_percent"] < 100.0


def test_a_changed_thread_count_stops_the_run(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    with pytest.raises(RuntimeError, match="thread count"):
        sweep(tmp_path, threads=THREADS + 1)


def test_a_thread_count_changed_while_the_split_runs_stops_the_run(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)

    def changes_threads(k):
        torch.set_num_threads(THREADS + 1)
        return InProcessSplit.load(tmp_path, k)

    with pytest.raises(RuntimeError, match="thread count"):
        sweep(tmp_path, changes_threads, points=[2])


def test_the_reference_is_stored_with_a_checksum_that_detects_a_change(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    summary = sweep(tmp_path, points=[2])[-1]
    entries = summary["reference"]
    assert [e["index"] for e in entries] == [0, 1, 2]
    assert all(len(e["tokens"]) == len(e["top_logits"]) == 6 for e in entries)
    assert all(len(e["prompt_sha256"]) == 64 for e in entries)
    stored = summary["reference_sha256"]
    assert x1a.checksum({"identity": IDENTITY, "entries": entries}) == stored
    # a changed token, a changed identity, or a different prompt each change it
    entries[1]["tokens"][0] += 1
    assert x1a.checksum({"identity": IDENTITY, "entries": entries}) != stored
    entries[1]["tokens"][0] -= 1
    assert x1a.checksum({"identity": IDENTITY | {"threads": 9}, "entries": entries}) != stored
    entries[2]["prompt_sha256"] = "0" * 64
    assert x1a.checksum({"identity": IDENTITY, "entries": entries}) != stored
    assert summary["interop_threads"] == torch.get_num_interop_threads()
    # and it is the same on a second run of the same code
    assert sweep(tmp_path, points=[2])[-1]["reference_sha256"] == summary["reference_sha256"]


def test_a_nan_difference_is_not_hidden(tmp_path):
    row = torch.zeros(4)
    nan_row = torch.tensor([0.0, float("nan"), 0.0, 0.0])
    same, diff = x1a.compare(([0], [row]), ([0], [nan_row]))
    assert same and diff == float("inf")
    # and it is written as strict JSON, not Infinity
    fields = x1a.diff_fields(diff)
    assert fields == {"max_logit_diff": None, "non_finite": True}
    json.loads(json.dumps(fields, allow_nan=False))
    assert x1a.diff_fields(0.0) == {"max_logit_diff": 0.0, "non_finite": False}


def run_main(tmp_path, monkeypatch, capsys):
    """main on a tiny model saved under a models folder, with runs/ not yet created."""
    tiny_reference(tmp_path / "models" / "smollm2-135m", layers=LAYERS)
    monkeypatch.setattr(x1a, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(x1a, "verify", lambda path, expected: None)
    monkeypatch.setattr(x1a.prompts, "prompts", lambda model, length: prompts_for_tests())
    monkeypatch.setattr(runrecord, "RUNS_DIR", tmp_path / "runs")
    argv = ["smollm2-135m", "--threads", str(THREADS), "--new-tokens", "4"]
    code = x1a.main(argv)
    (record_file,) = (tmp_path / "runs").glob("*-x1a.json")
    rec = json.loads(record_file.read_text(encoding="utf-8"))
    rec["record_file"] = record_file
    return code, rec, capsys.readouterr().out


def test_the_lines_are_written_next_to_the_record_on_a_fresh_checkout(
    tmp_path, monkeypatch, capsys
):
    code, rec, out = run_main(tmp_path, monkeypatch, capsys)
    assert code == 0 and rec["status"] == "ok"
    lines_file = tmp_path / "runs" / rec["results_file"]
    assert lines_file == rec["record_file"].with_suffix(".jsonl")
    text = lines_file.read_text(encoding="utf-8")
    assert text == out
    lines = [json.loads(line) for line in text.splitlines()]
    assert len(lines) == 3 * len(x1a.split_points(LAYERS)) + 1
    assert all(line["tokens_match"] for line in lines[:-1])
    assert lines[-1]["type"] == "summary" and lines[-1]["matched"] == len(lines) - 1


def test_a_non_finite_difference_fails_the_run(tmp_path, monkeypatch, capsys):
    real = x1a.compare
    monkeypatch.setattr(x1a, "compare", lambda want, got: (real(want, got)[0], float("inf")))
    code, rec, out = run_main(tmp_path, monkeypatch, capsys)
    assert code == 1
    assert rec["summary"]["non_finite"] and rec["summary"]["max_logit_diff"] is None
    assert "Infinity" not in out


def test_stored_logit_rows_do_not_keep_the_prefill_output_alive(tmp_path):
    _, model = tiny_reference(tmp_path, layers=LAYERS)
    from pllm.reference import decode

    _, rows = decode(model, list(range(1, 41)), 4)
    assert all(row.untyped_storage().nbytes() == row.nelement() * 4 for row in rows)


def test_split_points_are_the_five_tested_ones_without_repeats():
    assert x1a.split_points(30) == [1, 7, 15, 22, 29]
    assert x1a.split_points(28) == [1, 7, 14, 21, 27]
    assert x1a.split_points(5) == [1, 2, 3, 4]
    assert x1a.split_points(2) == [1]


def test_steps_after_the_first_differing_token_are_not_compared():
    row = torch.zeros(4)
    other = torch.tensor([0.0, 0.0, 0.0, 9.0])
    same, diff = x1a.compare(([0, 1], [row, row]), ([0, 2], [row, other]))
    assert not same and diff == 9.0
    same, diff = x1a.compare(([1, 1], [row, row]), ([2, 2], [row + 1, other]))
    assert not same and diff == 1.0  # only the first step, whose prefix is shared
