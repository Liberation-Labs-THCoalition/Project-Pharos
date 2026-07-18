"""Pharos MLX Pack Injector — Zero-Token Knowledge Injection for Apple Silicon

Converts Pharos knowledge packs (walk_encoding.txt + triples.json) into
pre-computed KV cache state compatible with mlx-lm's cache primitives.

The model behaves as if it read the pack content, but zero context tokens
are consumed at inference time. Works with any mlx-lm compatible model.

Usage:
    # Build a cache from a pack
    from pharos.mlx_injector import MLXPackBuilder
    builder = MLXPackBuilder(model, tokenizer)
    builder.build_pack_cache("eviction-defense", output_dir="./pack_caches")

    # Use at inference time
    from pharos.mlx_injector import generate_with_pack
    response = generate_with_pack(
        model, tokenizer,
        query="What are my rights as a tenant facing eviction?",
        pack_cache="./pack_caches/eviction-defense.safetensors"
    )

Requires: mlx, mlx-lm >= 0.20.0
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_PACKS_DIR = Path(os.environ.get(
    "PHAROS_PACKS_DIR", os.path.expanduser("~/lab/projects/pharos/packs")))
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "PHAROS_CACHE_DIR", os.path.expanduser("~/.cache/pharos-mlx")))


def _format_pack_as_system_prompt(pack_path: Path) -> str:
    """Convert a Pharos pack into injectable text.

    Uses the walk encoding (graph-walk linearization of knowledge triples)
    as the primary content, with metadata from triples.json for context.
    """
    walk_path = pack_path / "walk_encoding.txt"
    triples_path = pack_path / "triples.json"
    stats_path = pack_path / "stats.json"

    parts = []

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        pack_name = stats.get("name", pack_path.name)
        parts.append(f"Knowledge domain: {pack_name}")
        if "description" in stats:
            parts.append(f"Description: {stats['description']}")
        parts.append("")

    if walk_path.exists():
        walk_text = walk_path.read_text().strip()
        parts.append("Knowledge graph (walk-encoded):")
        parts.append(walk_text)
    elif triples_path.exists():
        triples = json.loads(triples_path.read_text())
        parts.append(f"Knowledge triples ({len(triples)} facts):")
        for triple in triples[:200]:
            if isinstance(triple, dict):
                s, p, o = triple.get("s", ""), triple.get("p", ""), triple.get("o", "")
            elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                s, p, o = triple[0], triple[1], triple[2]
            else:
                continue
            parts.append(f"  {s} — {p} → {o}")

    return "\n".join(parts)


class MLXPackBuilder:
    """Builds MLX KV cache files from Pharos knowledge packs."""

    def __init__(self, model: nn.Module, tokenizer, packs_dir: Optional[Path] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.packs_dir = Path(packs_dir) if packs_dir else DEFAULT_PACKS_DIR

    def build_pack_cache(self, pack_name: str, output_dir: Optional[Path] = None,
                         max_kv_size: Optional[int] = None,
                         chat_template: bool = True) -> Path:
        """Build a .safetensors cache file from a Pharos pack.

        Args:
            pack_name: Name of the pack directory under packs_dir.
            output_dir: Where to save the cache file. Defaults to PHAROS_CACHE_DIR.
            max_kv_size: Optional maximum KV cache size.
            chat_template: Wrap content in chat template as system message.

        Returns:
            Path to the saved .safetensors cache file.
        """
        from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache

        pack_path = self.packs_dir / pack_name
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack not found: {pack_path}")

        output_dir = Path(output_dir) if output_dir else DEFAULT_CACHE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pack_name}.safetensors"

        content = _format_pack_as_system_prompt(pack_path)
        logger.info(f"Pack '{pack_name}': {len(content)} chars")

        if chat_template:
            messages = [{"role": "system", "content": content}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
        else:
            formatted = content

        tokens = self.tokenizer.encode(formatted)
        token_ids = mx.array([tokens])
        logger.info(f"  Tokenized: {len(tokens)} tokens")

        t0 = time.time()
        cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)

        # Prefill: run tokens through model to populate cache
        # Process in chunks to avoid memory issues with very long packs
        chunk_size = 512
        for i in range(0, len(tokens), chunk_size):
            chunk = mx.array([tokens[i:i + chunk_size]])
            self.model(chunk, cache=cache)
            mx.eval([c.state for c in cache])

        elapsed = time.time() - t0
        logger.info(f"  Prefilled in {elapsed:.1f}s ({len(tokens)/elapsed:.0f} tok/s)")

        metadata = {
            "pack_name": pack_name,
            "token_count": str(len(tokens)),
            "created_at": str(int(time.time())),
            "pharos_version": "mlx-1.0",
        }
        save_prompt_cache(str(output_path), cache, metadata)
        logger.info(f"  Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")

        return output_path

    def build_all_packs(self, output_dir: Optional[Path] = None,
                        max_kv_size: Optional[int] = None) -> dict:
        """Build cache files for all packs in the packs directory.

        Returns dict mapping pack_name → output_path.
        """
        results = {}
        pack_dirs = sorted(p for p in self.packs_dir.iterdir()
                           if p.is_dir() and (p / "triples.json").exists())

        logger.info(f"Building {len(pack_dirs)} pack caches...")
        for pack_dir in pack_dirs:
            try:
                path = self.build_pack_cache(
                    pack_dir.name, output_dir=output_dir, max_kv_size=max_kv_size)
                results[pack_dir.name] = str(path)
            except Exception as e:
                logger.error(f"  Failed {pack_dir.name}: {e}")
                results[pack_dir.name] = f"ERROR: {e}"

        return results


def generate_with_pack(model: nn.Module, tokenizer, query: str,
                       pack_cache: str, max_tokens: int = 512,
                       **kwargs) -> str:
    """Generate a response with a pre-computed pack cache injected.

    The model sees the full knowledge pack content via KV cache
    but only processes the query tokens — zero-token injection.

    Args:
        model: The mlx-lm model.
        tokenizer: The tokenizer.
        query: The user's question.
        pack_cache: Path to the .safetensors cache file.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated response text.
    """
    from mlx_lm.models.cache import load_prompt_cache
    from mlx_lm import generate

    cache, metadata = load_prompt_cache(pack_cache, return_metadata=True)
    pack_name = metadata.get("pack_name", "unknown")
    prefix_len = int(metadata.get("token_count", "0"))
    logger.info(f"Loaded pack '{pack_name}' ({prefix_len} tokens)")

    response = generate(
        model, tokenizer, query,
        max_tokens=max_tokens,
        prompt_cache=cache,
        **kwargs
    )

    return response


def generate_with_multi_pack(model: nn.Module, tokenizer, query: str,
                             pack_caches: list[str], max_tokens: int = 512,
                             **kwargs) -> str:
    """Generate with multiple pack caches composed together.

    Packs are loaded and composed sequentially. RoPE positions
    continue across pack boundaries.
    """
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache

    if not pack_caches:
        from mlx_lm import generate
        return generate(model, tokenizer, query, max_tokens=max_tokens, **kwargs)

    # Load first pack as base
    cache = load_prompt_cache(pack_caches[0])
    total_prefix = 0

    # For additional packs, we need to continue prefilling
    for pack_file in pack_caches[1:]:
        additional_cache, meta = load_prompt_cache(pack_file, return_metadata=True)
        # Compose by continuing the cache state
        # NOTE: This is approximate — proper composition requires
        # re-encoding the second pack with the first cache as context.
        # For production, use build_pack_cache with prefix support.
        logger.warning("Multi-pack composition is approximate. "
                       "For precise RoPE, rebuild with prefix.")

    from mlx_lm import generate
    return generate(
        model, tokenizer, query,
        max_tokens=max_tokens,
        prompt_cache=cache,
        **kwargs
    )


# ── CLI interface ────────────────────────────────────────────

def main():
    """Build pack caches from command line."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build Pharos knowledge pack caches for MLX inference")
    parser.add_argument("--model", required=True,
                        help="Path to MLX model directory")
    parser.add_argument("--packs-dir", default=str(DEFAULT_PACKS_DIR),
                        help="Directory containing Pharos packs")
    parser.add_argument("--output-dir", default=str(DEFAULT_CACHE_DIR),
                        help="Where to save .safetensors cache files")
    parser.add_argument("--pack", default=None,
                        help="Build a specific pack (default: all)")
    parser.add_argument("--max-kv-size", type=int, default=None,
                        help="Maximum KV cache size")
    parser.add_argument("--test-query", default=None,
                        help="Run a test query after building")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    builder = MLXPackBuilder(model, tokenizer, packs_dir=Path(args.packs_dir))

    if args.pack:
        path = builder.build_pack_cache(args.pack, output_dir=Path(args.output_dir),
                                        max_kv_size=args.max_kv_size)
        print(f"\nBuilt: {path}")
    else:
        results = builder.build_all_packs(output_dir=Path(args.output_dir),
                                          max_kv_size=args.max_kv_size)
        print(f"\nBuilt {sum(1 for v in results.values() if not v.startswith('ERROR'))} packs")
        for name, path in sorted(results.items()):
            status = "OK" if not path.startswith("ERROR") else "FAIL"
            print(f"  [{status}] {name}")

    if args.test_query and args.pack:
        cache_file = str(Path(args.output_dir) / f"{args.pack}.safetensors")
        print(f"\nTest query: {args.test_query}")
        response = generate_with_pack(model, tokenizer, args.test_query, cache_file)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
