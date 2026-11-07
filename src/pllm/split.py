"""The model cut into two stages that run in one process.

Stage 0 holds the embedding and layers 0 to k-1. Stage 1 holds layers k to n-1, the
final norm and the LM head. Each stage runs the same layer code as the single-process
model (LayerStack), and the hidden states pass between them unchanged, so for the same
inputs on one host the logits are bitwise equal to the single-process ones.

Equality needs the same thread count as the run it is compared with (R8.9).

With tied embeddings the LM head is the embedding matrix. Loaded from a file, stage 1
reads its own copy of that matrix, as a stage running in another process must.

Stage 0 reads the whole embedding into memory by default. With mapped_embedding it keeps
only a MappedEmbedding, which reads the rows of the tokens it is given from the file, so
Host B stays within its budget (R3.2). Rows are widened to fp32 exactly as the full
upcast would, so the two paths give the same bits.

    stages = InProcessSplit.load(model_dir, k)

The split has the same new_cache and forward calls as Model, so the code that drives a
single-process model can drive it too.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from pllm.model import KVCache, LayerStack, ModelConfig, layer_from_weights, rms_norm

EMBED = "model.embed_tokens.weight"
NORM = "model.norm.weight"
LM_HEAD = "lm_head.weight"


class MappedEmbedding:
    """Embedding rows read from the checkpoint file on demand.

    The safetensors header is parsed once. Each lookup opens the file, reads only the
    bytes of the requested rows and closes it again, so no handle stays open and the
    rest of the table never enters memory.
    """

    DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            size = int.from_bytes(f.read(8), "little")
            entry = json.loads(f.read(size))[EMBED]
        if entry["dtype"] not in self.DTYPES:
            raise ValueError(f"{self.path}: embedding dtype {entry['dtype']} is not supported")
        self.dtype = self.DTYPES[entry["dtype"]]
        if len(entry["shape"]) != 2:
            raise ValueError(f"{self.path}: embedding shape {entry['shape']} is not 2-D")
        self.vocab, self.width = entry["shape"]
        self.row_bytes = self.width * self.dtype.itemsize
        begin, end = entry["data_offsets"]
        if end - begin != self.vocab * self.row_bytes:
            raise ValueError(f"{self.path}: embedding data size does not match its shape")
        self.offset = 8 + size + begin

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        unique, inverse = torch.unique(ids, return_inverse=True)
        if not len(unique):
            return torch.empty(*ids.shape, self.width)
        if not 0 <= unique[0] <= unique[-1] < self.vocab:
            bad = unique[0] if unique[0] < 0 else unique[-1]
            raise IndexError(f"token id {bad.item()} outside the embedding of {self.vocab} rows")
        data = bytearray()
        with open(self.path, "rb") as f:
            for i in unique.tolist():
                f.seek(self.offset + i * self.row_bytes)
                data += f.read(self.row_bytes)
        if len(data) != len(unique) * self.row_bytes:
            raise ValueError(f"{self.path}: file ends inside the embedding table")
        # safetensors is little-endian, as are both hosts; frombuffer uses native order.
        rows = torch.frombuffer(data, dtype=self.dtype).view(len(unique), self.width)
        return rows.to(torch.float32)[inverse]


def check_layer_range(config: ModelConfig, layer_start: int, layer_end: int) -> None:
    if not 0 <= layer_start < layer_end <= config.num_layers:
        raise ValueError(f"layers {layer_start} to {layer_end} outside 0 to {config.num_layers}")


class Stage(LayerStack):
    """Layers layer_start to layer_end - 1, with the embedding if first and the norm and
    LM head if last."""

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        layer_start: int,
        layer_end: int,
        embed: "torch.Tensor | MappedEmbedding | None" = None,
    ):
        check_layer_range(config, layer_start, layer_end)
        super().__init__(
            config,
            [layer_from_weights(weights, i, config.qk_norm) for i in range(layer_start, layer_end)],
        )
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.first = layer_start == 0
        self.last = layer_end == config.num_layers
        self.embed = embed if embed is not None else (weights[EMBED] if self.first else None)
        self.norm = weights[NORM] if self.last else None
        self.lm_head = None
        if self.last:
            self.lm_head = weights[EMBED if config.tie_word_embeddings else LM_HEAD]

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        layer_start: int,
        layer_end: int,
        mapped_embedding: bool = False,
    ) -> "Stage":
        """Read only this stage's tensors from model.safetensors, upcast to fp32.

        With mapped_embedding the first stage reads embedding rows on demand instead of
        holding the matrix. A stage that is also last still reads it, as the tied head.
        """
        model_dir = Path(model_dir)
        config = ModelConfig.from_file(model_dir / "config.json")
        check_layer_range(config, layer_start, layer_end)
        first, last = layer_start == 0, layer_end == config.num_layers
        wanted = {EMBED} if first and not mapped_embedding else set()
        if last:
            wanted |= {NORM, EMBED if config.tie_word_embeddings else LM_HEAD}
        weights = {}
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            for name in f.keys():
                parts = name.split(".")
                in_range = (
                    parts[:2] == ["model", "layers"] and layer_start <= int(parts[2]) < layer_end
                )
                if name in wanted or in_range:
                    weights[name] = f.get_tensor(name).to(torch.float32)
        embed = (
            MappedEmbedding(model_dir / "model.safetensors") if first and mapped_embedding else None
        )
        return cls(config, weights, layer_start, layer_end, embed)

    @torch.inference_mode()
    def forward(self, inputs: torch.Tensor, start_pos: int, cache: KVCache) -> torch.Tensor:
        """Token ids (first stage) or hidden states shaped (tokens, hidden) in; hidden
        states out, or logits shaped (tokens, vocab) from the last stage."""
        if self.first:
            if inputs.dtype not in (torch.int32, torch.int64) or inputs.ndim != 1:
                raise ValueError(
                    f"expected integer token ids shaped (tokens,), "
                    f"got {inputs.dtype} {tuple(inputs.shape)}"
                )
            x = self.embed[inputs]
        else:
            hidden = self.config.hidden_size
            if inputs.dtype != torch.float32 or inputs.ndim != 2 or inputs.shape[1] != hidden:
                raise ValueError(
                    f"expected float32 hidden states shaped (tokens, {hidden}), "
                    f"got {inputs.dtype} {tuple(inputs.shape)}"
                )
            x = inputs
        x = self.run_layers(x, start_pos, cache)
        if self.last:
            return F.linear(rms_norm(x, self.norm, self.config), self.lm_head)
        return x


class InProcessSplit:
    """Stage 0 then stage 1 in one process, with the interface of Model."""

    def __init__(self, stage0: Stage, stage1: Stage):
        if not (stage0.first and stage1.last and stage0.layer_end == stage1.layer_start):
            raise ValueError("stage 0 must end where stage 1 starts, covering the whole model")
        self.stage0, self.stage1 = stage0, stage1
        self.config = stage0.config

    @classmethod
    def load(cls, model_dir: str | Path, k: int) -> "InProcessSplit":
        """Split after layer k - 1: stage 0 has layers 0 to k-1, stage 1 the rest."""
        n = ModelConfig.from_file(Path(model_dir) / "config.json").num_layers
        if not 1 <= k <= n - 1:
            raise ValueError(f"split point {k} outside 1 to {n - 1}")
        return cls(Stage.load(model_dir, 0, k), Stage.load(model_dir, k, n))

    def new_cache(self) -> tuple[KVCache, KVCache]:
        return self.stage0.new_cache(), self.stage1.new_cache()

    def forward(
        self, tokens: torch.Tensor, start_pos: int, cache: tuple[KVCache, KVCache]
    ) -> torch.Tensor:
        hidden = self.stage0.forward(tokens, start_pos, cache[0])
        return self.stage1.forward(hidden, start_pos, cache[1])
