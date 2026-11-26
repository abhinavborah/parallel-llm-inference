"""X9 (R7.16, R8.6, R8.7): the capacity-aware split, the memory counters, the Host B
monitor, the probe, the verdicts, the split override, and one sweep on a real chain on
loopback. Small random models only."""

import pytest
from test_model import tiny_reference

from pllm import (
    config,
    launch,
)
from pllm.capacity import HostCapacity
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
