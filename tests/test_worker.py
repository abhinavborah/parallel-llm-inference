"""The stage worker: its layers only, a mapped embedding, one delay per step (R3.1, R3.2)."""

import ast
import logging
import threading
import time

import psutil
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_model import tiny_reference

from pllm.config import REPO_ROOT
from pllm.model import Model
from pllm.models import MODELS_DIR, PINNED, verify
from pllm.split import EMBED, MappedEmbedding, Stage
from pllm.worker import StageWorker, WorkerConfig, parse_layers

LAYERS = 5


def make_worker(tmp_path, start, end, delay=None, step_delay=0.0):
    return StageWorker(WorkerConfig(tmp_path, start, end, step_delay), delay)


@pytest.mark.parametrize("tie", [True, False])
def test_a_mapped_embedding_gives_the_same_rows_as_the_full_one(tmp_path, tie):
    _, model = tiny_reference(tmp_path, tie, layers=LAYERS)
    mapped = MappedEmbedding(tmp_path / "model.safetensors")
    ids = torch.tensor([5, 9, 5, 127, 0, 9])  # repeats and both ends of the table
    assert torch.equal(mapped[ids], model.embed[ids])


def test_a_mapped_embedding_rejects_ids_outside_the_table_and_accepts_none(tmp_path):
    _, model = tiny_reference(tmp_path, layers=LAYERS)
    mapped = MappedEmbedding(tmp_path / "model.safetensors")
    for bad in (-1, 128, 5000):
        with pytest.raises(IndexError, match="outside the embedding of 128 rows"):
            mapped[torch.tensor([3, bad])]
    empty = mapped[torch.tensor([], dtype=torch.int64)]
    assert empty.shape == (0, model.embed.shape[1]) and empty.dtype == torch.float32


def test_the_first_stage_keeps_a_lookup_and_reads_no_embedding_matrix(tmp_path, monkeypatch):
    tiny_reference(tmp_path, layers=LAYERS)
    seen = []
    real = Stage.__init__

    def spy(self, config, weights, *args, **kwargs):
        seen.extend(weights)
        real(self, config, weights, *args, **kwargs)

    monkeypatch.setattr(Stage, "__init__", spy)
    first = Stage.load(tmp_path, 0, 2, mapped_embedding=True)
    assert isinstance(first.embed, MappedEmbedding)
    assert EMBED not in seen

    # A stage that is also last still reads the matrix, as the tied head.
    seen.clear()
    whole = Stage.load(tmp_path, 0, LAYERS, mapped_embedding=True)
    assert isinstance(whole.embed, MappedEmbedding)
    assert EMBED in seen and whole.lm_head.shape[0] == 128


@pytest.mark.parametrize("tie", [True, False])
def test_workers_in_a_chain_equal_the_single_process_model(tmp_path, tie):
    _, model = tiny_reference(tmp_path, tie, layers=LAYERS)
    tokens = torch.randint(0, 128, (12,), generator=torch.Generator().manual_seed(4))
    cache = model.new_cache()
    expected = model.forward(tokens, 0, cache)
    expected_next = model.forward(tokens[:1], 12, cache)
    for k in range(1, LAYERS):
        first, second = make_worker(tmp_path, 0, k), make_worker(tmp_path, k, LAYERS)
        caches = first.new_cache(), second.new_cache()
        out = second.step(first.step(tokens, 0, caches[0]), 0, caches[1])
        assert torch.equal(out, expected), f"k={k}"
        out = second.step(first.step(tokens[:1], 12, caches[0]), 12, caches[1])
        assert torch.equal(out, expected_next), f"k={k}"


def test_a_worker_holds_and_caches_only_its_layers(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    middle = make_worker(tmp_path, 1, 4)
    assert len(middle.stage.layers) == len(middle.new_cache().keys) == 3
    assert middle.stage.embed is None and middle.stage.lm_head is None


def test_a_worker_logs_its_layers_load_time_and_rss(tmp_path, caplog):
    tiny_reference(tmp_path, layers=LAYERS)
    with caplog.at_level(logging.INFO, logger="pllm.worker"):
        made = make_worker(tmp_path, 0, 2)
    (record,) = caplog.records
    message = record.getMessage()
    assert "layers 0 to 1" in message and "rss" in message and "MiB" in message
    assert made.load_seconds > 0 and made.rss_after_load_mib > 0


def test_the_delay_is_called_once_per_step_before_the_stage_runs(tmp_path):
    tiny_reference(tmp_path, layers=LAYERS)
    events = []
    made = make_worker(tmp_path, 0, 2, delay=lambda: events.append("delay"))
    real = made.stage.forward

    def traced(*args):
        events.append("compute")
        return real(*args)

    made.stage.forward = traced
    cache = made.new_cache()
    made.step(torch.tensor([1, 2, 3]), 0, cache)
    made.step(torch.tensor([4]), 3, cache)
    assert events == ["delay", "compute", "delay", "compute"]


def test_the_default_delay_sleeps_for_nothing_and_a_set_delay_sleeps(tmp_path, monkeypatch):
    tiny_reference(tmp_path, layers=LAYERS)
    slept = []
    monkeypatch.setattr("pllm.faults.time.sleep", slept.append)
    plain = make_worker(tmp_path, 0, 2)
    plain.step(torch.tensor([1]), 0, plain.new_cache())
    assert slept == []
    timed = make_worker(tmp_path, 0, 2, step_delay=0.25)
    timed.step(torch.tensor([1]), 0, timed.new_cache())
    assert slept == [0.25]


def test_layer_ranges_are_parsed_and_checked(tmp_path):
    assert parse_layers("0:15") == (0, 15)
    for bad in ("", "5", "a:b", "1:", ":4", "-1:3"):
        with pytest.raises(ValueError, match="START:END"):
            parse_layers(bad)
    tiny_reference(tmp_path, layers=LAYERS)
    with pytest.raises(ValueError, match="outside"):
        make_worker(tmp_path, 3, LAYERS + 1)


def test_worker_code_has_no_operating_system_branches():
    tree = ast.parse((REPO_ROOT / "src" / "pllm" / "worker.py").read_text(encoding="utf-8"))
    attributes = {
        f"{n.value.id}.{n.attr}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    imported = {
        alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names
    }
    assert not attributes & {"sys.platform", "os.name", "platform.system", "os.getuid"}
    assert not imported & {"posix", "winreg", "msvcrt", "fcntl", "platform"}


def peak_growth_mib(action):
    """Run action and return how far the process RSS rose at its highest, in MiB.

    A sampler thread watches during the call, so a table that is read and freed again
    inside it still counts.
    """
    process = psutil.Process()
    before = process.memory_info().rss
    peak, done = before, threading.Event()

    def watch():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, process.memory_info().rss)
            time.sleep(0.001)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        action()
    finally:
        done.set()
        watcher.join()
    return (peak - before) / 2**20


def test_embedding_rows_do_not_bring_the_table_into_memory(tmp_path):
    """A table of about 200 MiB: a lookup of a few rows must raise RSS by a small fraction."""
    vocab, width = 50_000, 1024
    path = tmp_path / "model.safetensors"
    save_file({EMBED: torch.zeros(vocab, width)}, path)
    table_mib = vocab * width * 4 / 2**20

    def read_everything():
        with safe_open(path, framework="pt") as f:
            f.get_tensor(EMBED).clone()

    # The control: this measurement does see a table being read in full.
    assert peak_growth_mib(read_everything) > table_mib / 2

    ids = torch.tensor([7, 38_777, 7, 49_999])
    rows = []
    grown = peak_growth_mib(lambda: rows.append(MappedEmbedding(path)[ids]))
    assert rows[0].shape == (4, width)
    assert grown < table_mib / 10, f"rss rose {grown:.0f} MiB for a {table_mib:.0f} MiB table"


SMOLLM2_DIR = MODELS_DIR / "smollm2-135m"


@pytest.mark.skipif(
    not (SMOLLM2_DIR / "model.safetensors").exists(), reason="model weights not downloaded"
)
def test_smollm2_workers_equal_the_single_process_model():
    for filename, expected in PINNED["smollm2-135m"].files.items():
        verify(SMOLLM2_DIR / filename, expected)
    model = Model.load(SMOLLM2_DIR)
    n = model.config.num_layers
    tokens = torch.randint(
        0, model.config.vocab_size, (20,), generator=torch.Generator().manual_seed(2)
    )
    cache = model.new_cache()
    expected = [model.forward(tokens[:14], 0, cache), model.forward(tokens[14:15], 14, cache)]
    del model
    for k in (1, n // 2, n - 1):
        first = StageWorker(WorkerConfig(SMOLLM2_DIR, 0, k))
        second = StageWorker(WorkerConfig(SMOLLM2_DIR, k, n))
        caches = first.new_cache(), second.new_cache()
        steps = [(tokens[:14], 0), (tokens[14:15], 14)]
        for (ids, pos), want in zip(steps, expected, strict=True):
            out = second.step(first.step(ids, pos, caches[0]), pos, caches[1])
            assert torch.equal(out, want), f"k={k} pos={pos}"
