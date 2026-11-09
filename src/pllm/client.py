"""Send one prompt to the gateway and print the tokens as they arrive (R4.1).

Usage:
    uv run python -m pllm.client "The capital of France is" --max-tokens 16 --timeout 30

The gateway address comes from the hosts file (PLLM_CONFIG or config/hosts.toml).
"""

import argparse
import sys

import grpc

from pllm import config
from pllm.proto import protocol_pb2, protocol_pb2_grpc


def generate(address: str, prompt: str, max_tokens: int, timeout: float):
    """Yield each Token the gateway streams back for prompt."""
    with grpc.insecure_channel(address) as channel:
        stub = protocol_pb2_grpc.FrontendStub(channel)
        request = protocol_pb2.GenerateReq(
            prompt=prompt, max_tokens=max_tokens, timeout_ms=int(timeout * 1000)
        )
        # The gateway enforces the deadline; the client waits a little longer for its answer.
        yield from stub.Generate(request, timeout=timeout + 5)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to the gateway.")
    parser.add_argument("prompt")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    try:
        address = config.load().gateway.address
    except (ValueError, OSError) as err:
        parser.error(str(err))
    count, rid = 0, None
    try:
        for token in generate(address, args.prompt, args.max_tokens, args.timeout):
            rid = token.rid
            count += 1
            print(token.text, end="", flush=True)
    except grpc.RpcError as err:
        print(f"\nerror: {err.code().name}: {err.details()}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"\n[{count} tokens, request {rid}]", file=sys.stderr)


if __name__ == "__main__":
    main()
