"""The fixed evaluation prompt set, used by every experiment.

50 passages from one public-domain book, each cut to 16, 128 and 512 tokens, so the
three lengths are nested prefixes of the same passage. Token ids are stored once per
pinned tokenizer, keyed by the tokenizer.json sha256, and are canonical; the text is
only for reading. Every cut is chosen so that encoding the decoded ids gives the same
ids back. The file is checked against PROMPTS_SHA256 on load.

Rebuild (deterministic for the recorded book file; prints the new checksum):
uv run python -m pllm.prompts path/to/pg1661.txt
"""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pllm import runrecord
from pllm.config import REPO_ROOT
from pllm.models import MODELS_DIR, PINNED, sha256_of, verify

PROMPTS_PATH = REPO_ROOT / "data" / "prompt_set.json"
PROMPTS_SHA256 = "cbf6eff14a0e351b7c82e008e4ae9dd241f8f6b385360044f5cffa0469e0d3af"
LENGTHS = (16, 128, 512)
PASSAGES = 50
BOOK = {
    "title": "The Adventures of Sherlock Holmes, by Arthur Conan Doyle",
    "project_gutenberg_ebook": 1661,
    "url": "https://www.gutenberg.org/ebooks/1661",
    "text_url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "accessed": "2026-11-03",
    "sha256": "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
}
# first paragraph of the first story; title page, contents and headings before it are dropped
FIRST_PARAGRAPH = "To Sherlock Holmes she is always"
TEXT_RULE = (
    "Project Gutenberg header and footer removed; front matter before the first story "
    "paragraph removed; paragraphs unwrapped to single lines and joined by blank lines; "
    "em dashes replaced by hyphens."
)


def load(path: str | Path = PROMPTS_PATH, expected: str = PROMPTS_SHA256) -> dict:
    """Read the prompt set after checking its sha256; ChecksumError on a mismatch."""
    verify(path, expected)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompts(model: str, length: int, data: dict | None = None) -> list[list[int]]:
    """The 50 prompts of one length as token ids for a pinned model's tokenizer."""
    if length not in LENGTHS:
        raise ValueError(f"length {length} is not one of {LENGTHS}")
    data = data if data is not None else load()
    key = PINNED[model].files["tokenizer.json"]
    if key not in data["tokenizers"]:
        raise ValueError(f"prompt set has no ids for the {model} tokenizer; rebuild it")
    return [passage["ids"][key][:length] for passage in data["passages"]]


def load_tokenizer(model: str) -> Tokenizer:
    """The pinned tokenizer of a downloaded model, checked against its sha256."""
    path = MODELS_DIR / model / "tokenizer.json"
    verify(path, PINNED[model].files["tokenizer.json"])
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    # Raw text: no chat template and no special tokens.
    return tokenizer.encode(text, add_special_tokens=False).ids


def round_trips(tokenizer: Tokenizer, ids: list[int]) -> bool:
    return encode(tokenizer, tokenizer.decode(ids)) == ids


def paragraphs(raw: str) -> list[str]:
    """Book body as one string per paragraph, following TEXT_RULE."""
    body = raw.split("*** START OF", 1)[1].split("\n", 1)[1]
    body = body.split("*** END OF", 1)[0].replace(chr(0x2014), "-")
    found = [p for p in (" ".join(p.split()) for p in re.split(r"\n\s*\n", body)) if p]
    first = next(i for i, p in enumerate(found) if p.startswith(FIRST_PARAGRAPH))
    return found[first:]


def build(raw: str, tokenizers: dict[str, Tokenizer]) -> dict:
    """Pick PASSAGES passages spread evenly through the book, one per slot of paragraphs.

    Within its slot a passage starts at the first paragraph whose cuts all round-trip for
    every tokenizer, and it never overlaps the passage before it.
    """
    paras = paragraphs(raw)
    slot = len(paras) // PASSAGES
    longest = max(LENGTHS)
    passages = []
    end = 0
    for k in range(PASSAGES):
        for start in range(max(k * slot, end), (k + 1) * slot):
            stop = start + 1
            while True:
                text = "\n\n".join(paras[start:stop])
                ids = {key: encode(tok, text) for key, tok in tokenizers.items()}
                if all(len(i) > longest for i in ids.values()) or stop == len(paras):
                    break
                stop += 1
            ids = {key: i[:longest] for key, i in ids.items()}
            if all(
                len(i) == longest and round_trips(tokenizers[key], i[:n])
                for key, i in ids.items()
                for n in LENGTHS
            ):
                break
        else:
            raise ValueError(f"no passage in slot {k} round-trips for every tokenizer")
        decoded = [tokenizers[key].decode(i) for key, i in ids.items()]
        shown = text[: max(len(d) for d in decoded)]
        assert all(shown.startswith(d) for d in decoded)
        passages.append({"text": shown, "ids": ids})
        end = stop

    names = {spec.files["tokenizer.json"]: name for name, spec in PINNED.items()}
    return {
        "description": "Fixed evaluation prompt set: 50 passages, each cut to 16, 128 and "
        "512 tokens. Token ids are canonical; text is for reading.",
        "source": BOOK,
        "text_rule": TEXT_RULE,
        "add_special_tokens": False,
        "chat_template": None,
        "lengths": list(LENGTHS),
        "tokenizers": {
            key: {
                "model": names[key],
                "repo": PINNED[names[key]].repo,
                "revision": PINNED[names[key]].revision,
            }
            for key in tokenizers
        },
        "passages": passages,
    }


def dump(data: dict) -> str:
    """JSON with one passage per line, so a diff stays readable."""
    head = {k: v for k, v in data.items() if k != "passages"}
    lines = ",\n".join(json.dumps(p, ensure_ascii=False) for p in data["passages"])
    return (
        json.dumps(head, indent=2, ensure_ascii=False)[:-2]
        + ',\n"passages": [\n'
        + lines
        + "\n]}\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild the prompt set from the book file.")
    parser.add_argument("book", type=Path, help=f"the text file from {BOOK['text_url']}")
    args = parser.parse_args(argv)
    with runrecord.record("prompt_set") as rec:
        rec.update(book_sha256=BOOK["sha256"])
        verify(args.book, BOOK["sha256"])
        tokenizers = {}
        for name, spec in PINNED.items():
            path = hf_hub_download(
                spec.repo, "tokenizer.json", revision=spec.revision, local_dir=MODELS_DIR / name
            )
            verify(path, spec.files["tokenizer.json"])
            tokenizers[spec.files["tokenizer.json"]] = Tokenizer.from_file(path)
        rec.update(tokenizer_sha256=sorted(tokenizers))
        data = build(args.book.read_text(encoding="utf-8-sig"), tokenizers)
        PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Binary write keeps LF line endings on Windows, so the checksum is the same everywhere.
        PROMPTS_PATH.write_bytes(dump(data).encode("utf-8"))
        rec.update(prompts_sha256=sha256_of(PROMPTS_PATH))
    print(f"wrote {PROMPTS_PATH}, sha256 {rec['prompts_sha256']}")


if __name__ == "__main__":
    main()
