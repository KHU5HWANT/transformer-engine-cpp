#!/usr/bin/env python3
"""
scripts/preprocess.py — Phase 3.1: Text-to-Binary Preprocessor
================================================================

Converts raw text datasets into compact binary .bin files that the C++
DataLoader (Step 3.2) can memory-map and stream at full disk bandwidth.

Usage
─────
  python scripts/preprocess.py --dataset math    --output data/math_train.bin
  python scripts/preprocess.py --dataset stories --output data/stories_train.bin
  python scripts/preprocess.py --dataset wiki    --output data/wiki_train.bin

Binary format  (strict, must match the C++ DataLoader header parser)
─────────────
  Offset  Bytes  C type    Value / Description
  ──────  ─────  ──────    ───────────────────
       0      4  uint32    Magic: 0xDEADBEEF  (sanity check on load)
       4      4  uint32    vocab_size (always 256 for char-level)
       8      8  uint64    num_tokens  (number of uint16 IDs in the payload)
      16  2×N    uint16[]  Token-ID payload  (little-endian, one ID per char)

All multi-byte integers are little-endian (x86 native).
Total file size: 16 + 2 × num_tokens  bytes.

Tokeniser
──────────
Character-level, vocab_size = 256.
  token_id = ord(character)   for characters with ord < 256
  token_id = ord('?')  = 63   for any character with ord ≥ 256 (rare, safe fallback)

This gives a 256-way bijection for all printable ASCII and Latin-1 characters.
No BPE or WordPiece is needed — character-level is the simplest viable baseline
for curriculum learning on math and short-story data.

Caching
────────
If the output .bin file already exists, the script prints a notice and exits
immediately.  Delete the file to force regeneration.

Datasets
─────────
  math    — Synthetic addition equations: "{a}+{b}={c}\\n"
             No internet access required.  Equations are ordered by difficulty
             (1-digit → 2-digit → 3-digit) to support curriculum scheduling.
  stories — roneneldan/TinyStories via HuggingFace Hub (pip install datasets)
  wiki    — Simple English Wikipedia (20220301.simple) via HuggingFace Hub
"""

from __future__ import annotations

import argparse
import array
import logging
import os
import random
import struct
import sys
from typing import Iterator


# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC_NUMBER : int = 0xDEAD_BEEF
VOCAB_SIZE   : int = 256
HEADER_FMT   : str = '<IIQ'               # little-endian uint32, uint32, uint64
HEADER_BYTES : int = struct.calcsize(HEADER_FMT)   # == 16

# Characters with ord >= VOCAB_SIZE are replaced with FALLBACK_TOKEN.
FALLBACK_TOKEN : int = ord('?')           # 63 — visible, non-NUL, non-control

MIN_MATH_EQUATIONS : int = 500_000       # enforced lower bound for math mode

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ═════════════════════════════════════════════════════════════════════════════
# Character-level tokeniser
# ═════════════════════════════════════════════════════════════════════════════

def tokenize_text(text: str) -> array.array:
    """
    Convert a string to a flat array of uint16 character-ordinal IDs.

    Design choices
    ─────────────
    • Returns array.array('H') (unsigned short, 2 bytes) so the result can be
      written to disk with a single .tofile() call — equivalent to fwrite() in C.
    • Characters above Latin-1 (ord ≥ 256) are mapped to FALLBACK_TOKEN ('?').
      This is rare in math/stories/wiki text but prevents buffer-overrun bugs in
      the C++ DataLoader which assumes token_id < 256 = vocab_size.
    • We do NOT skip out-of-range characters because that would silently shift
      token positions, breaking any position-dependent curriculum logic.

    Parameters
    ──────────
    text : str
        Raw UTF-8 string.

    Returns
    ───────
    array.array('H') of length len(text), values in [0, 255].
    """
    buf = array.array('H')
    for ch in text:
        code = ord(ch)
        buf.append(code if code < VOCAB_SIZE else FALLBACK_TOKEN)
    return buf


# ═════════════════════════════════════════════════════════════════════════════
# Binary serialisation
# ═════════════════════════════════════════════════════════════════════════════

def write_bin(output_path: str, tokens: array.array) -> None:
    """
    Write the binary .bin file consumed by the C++ DataLoader.

    Header layout (16 bytes, little-endian):
      [0:4]  uint32  magic     = 0xDEADBEEF
      [4:8]  uint32  vocab     = 256
      [8:16] uint64  n_tokens  = len(tokens)

    Payload:
      uint16[n_tokens]  — little-endian token IDs

    On big-endian machines (rare but possible) the array is byte-swapped in
    place before writing and swapped back afterwards.  The struct header is
    always written with explicit '<' (little-endian) format.

    Parameters
    ──────────
    output_path : str
        Destination file.  Parent directories are created automatically.
    tokens : array.array('H')
        Token ID stream.  All values must be in [0, 65535]; in practice [0, 255].
    """
    n_tokens = len(tokens)
    log.info("Serialising %d tokens → %s", n_tokens, output_path)

    # Ensure output directory exists (handles both relative and absolute paths)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # On big-endian platforms, swap bytes so the payload matches the
    # little-endian expectation of the C++ DataLoader.
    swapped = False
    if sys.byteorder == "big":
        tokens.byteswap()
        swapped = True

    try:
        with open(output_path, "wb") as fout:
            # ── Header (16 bytes) ─────────────────────────────────────────────
            fout.write(struct.pack(HEADER_FMT, MAGIC_NUMBER, VOCAB_SIZE, n_tokens))

            # ── Payload (2 × n_tokens bytes) ──────────────────────────────────
            # array.tofile() calls fwrite() internally — no extra copies.
            tokens.tofile(fout)
    finally:
        # Restore the array to native order regardless of write success.
        if swapped:
            tokens.byteswap()

    file_bytes = os.path.getsize(output_path)
    log.info(
        "Done.  %s  |  %d tokens  |  %.2f MB",
        output_path, n_tokens, file_bytes / (1 << 20),
    )

    # Sanity-check: expected size = header + 2 bytes × n_tokens
    expected = HEADER_BYTES + 2 * n_tokens
    if file_bytes != expected:
        log.error(
            "Size mismatch! Expected %d bytes, got %d bytes. "
            "File may be corrupt.", expected, file_bytes,
        )
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset generators
# ═════════════════════════════════════════════════════════════════════════════

def generate_math(n_equations: int) -> Iterator[str]:
    """
    Generate synthetic integer addition and subtraction equations for curriculum learning.

    Equation format:  "{a}+{b}={a+b}\\n" or "{a}-{b}={a-b}\\n"
    Examples:         "3+7=10\\n",  "91-42=49\\n",  "307+648=955\\n"

    Difficulty tiers (ordered easiest → hardest for curriculum scheduling)
    ────────────────────────────────────────────────────────────────────────
      Tier 1 (1-digit,  20%):  a, b ∈ [0, 9]
      Tier 2 (2-digit,  40%):  a, b ∈ [10, 99]
      Tier 3 (3-digit,  40%):  a, b ∈ [100, 999]

    Ordering the tiers sequentially (not randomly shuffled) means the C++
    curriculum scheduler (Step 3.3) can serve easy examples first by simply
    advancing through the file — no random-access index is needed.

    Parameters
    ──────────
    n_equations : int
        Total number of equations to generate (≥ MIN_MATH_EQUATIONS).

    Yields
    ──────
    str — one equation string per call, e.g. "123+456=579\\n".
    """
    tier1_n = int(n_equations * 0.20)
    tier2_n = int(n_equations * 0.40)
    tier3_n = n_equations - tier1_n - tier2_n    # absorbs rounding remainder

    tiers = [
        (0,   9,   tier1_n, "1-digit"),
        (10,  99,  tier2_n, "2-digit"),
        (100, 999, tier3_n, "3-digit"),
    ]

    for lo, hi, count, label in tiers:
        log.info("  Generating %d %s equations (a,b ∈ [%d,%d])…", count, label, lo, hi)
        for _ in range(count):
            a = random.randint(lo, hi)
            b = random.randint(lo, hi)
            if random.random() < 0.5:
                yield f"{a}+{b}={a + b}\n"
            else:
                # Prevent negative answers for simplicity
                if a < b:
                    a, b = b, a
                yield f"{a}-{b}={a - b}\n"


def generate_stories() -> Iterator[str]:
    """
    Stream plain text from roneneldan/TinyStories (HuggingFace Hub).

    Each example is yielded as:
        <story text>\\n\\n

    The double newline acts as a document-boundary token that the DataLoader
    can use to avoid cross-story context contamination during training.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset   # type: ignore[import]
    except ImportError:
        log.error(
            "Package 'datasets' not found.  Install it with:\n"
            "    pip install datasets",
        )
        sys.exit(1)

    log.info("Loading roneneldan/TinyStories (HuggingFace Hub)…")
    log.info("This may take a few minutes on first run; cached on subsequent runs.")

    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        trust_remote_code=False,
    )
    log.info("Loaded %d stories.", len(ds))

    for i, example in enumerate(ds):
        text: str = example.get("text", "")
        if text:
            yield text.rstrip() + "\n\n"   # normalise trailing whitespace
        if (i + 1) % 100_000 == 0:
            log.info("  %d / %d stories streamed…", i + 1, len(ds))


def generate_wiki() -> Iterator[str]:
    """
    Stream article text from Simple English Wikipedia (HuggingFace Hub).

    Uses the '20220301.simple' configuration:
      ≈ 200,000 articles,  ≈ 130 MB of text.

    Each article is yielded as:
        = <title> =\\n<article text>\\n\\n

    Including the title (formatted as a level-1 wiki heading) helps the model
    learn document structure.  The double newline is the document separator.

    For full English Wikipedia, change the config to '20220301.en'
    (~6 million articles, ~20 GB of text).

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset   # type: ignore[import]
    except ImportError:
        log.error(
            "Package 'datasets' not found.  Install it with:\n"
            "    pip install datasets",
        )
        sys.exit(1)

    log.info("Loading wikimedia/wikipedia 20231101.simple (HuggingFace Hub, streaming)…")
    log.info("This may take a few minutes on first run; cached on subsequent runs.")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.simple",
        split="train",
        streaming=True,
    )
    log.info("Dataset iterator ready (streaming — total article count unknown).")  # len() unavailable on IterableDataset

    for i, example in enumerate(ds):
        title: str = example.get("title", "").strip()
        text:  str = example.get("text",  "").strip()
        if text:
            yield f"= {title} =\n{text}\n\n"
        if (i + 1) % 10_000 == 0:
            log.info("  %d articles streamed so far…", i + 1)


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocess.py",
        description=(
            "Phase 3.1 Preprocessor — converts text datasets into binary .bin files "
            "for the C++ DataLoader.\n\n"
            "Binary format (16-byte header + uint16 payload):\n"
            "  uint32  magic      = 0xDEADBEEF\n"
            "  uint32  vocab_size = 256\n"
            "  uint64  num_tokens\n"
            "  uint16  token_ids[num_tokens]\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        choices=["math", "stories", "wiki"],
        required=True,
        metavar="DATASET",
        help=(
            "Dataset to process. One of:\n"
            "  math    — Synthetic addition equations (offline, no download).\n"
            "  stories — roneneldan/TinyStories (requires 'datasets' package).\n"
            "  wiki    — Simple English Wikipedia (requires 'datasets' package)."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the output .bin file (e.g. data/math_train.bin).",
    )

    parser.add_argument(
        "--n_math_equations",
        type=int,
        default=MIN_MATH_EQUATIONS,
        metavar="N",
        help=(
            f"Number of addition equations (math mode only). "
            f"Default and minimum: {MIN_MATH_EQUATIONS:,}."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible math generation. Default: 42.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file even if it already exists.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    # ── Cache check ───────────────────────────────────────────────────────────
    if os.path.exists(args.output) and not args.force:
        size_mb = os.path.getsize(args.output) / (1 << 20)
        log.info(
            "Cache hit: %s already exists (%.2f MB). "
            "Use --force to regenerate.",
            args.output, size_mb,
        )
        return

    # ── Argument validation ────────────────────────────────────────────────────
    if args.dataset == "math" and args.n_math_equations < MIN_MATH_EQUATIONS:
        log.warning(
            "--n_math_equations=%d is below the recommended minimum of %d. "
            "The model may underfit on the math curriculum.",
            args.n_math_equations, MIN_MATH_EQUATIONS,
        )

    # ── RNG seed ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    log.info("Random seed set to %d.", args.seed)

    # ── Choose generator ──────────────────────────────────────────────────────
    if args.dataset == "math":
        log.info("Mode: math  |  %d equations", args.n_math_equations)
        text_stream = generate_math(args.n_math_equations)

    elif args.dataset == "stories":
        log.info("Mode: stories  |  roneneldan/TinyStories")
        text_stream = generate_stories()

    else:  # wiki
        log.info("Mode: wiki  |  Wikipedia 20220301.simple")
        text_stream = generate_wiki()

    # ── Tokenise stream ────────────────────────────────────────────────────────
    log.info("Tokenising (char-level, vocab_size=%d)…", VOCAB_SIZE)

    all_tokens  = array.array('H')
    n_docs      = 0
    n_chars     = 0
    LOG_INTERVAL = 50_000         # report progress every 50k chars

    for text in text_stream:
        chunk = tokenize_text(text)
        all_tokens.extend(chunk)
        n_docs  += 1
        n_chars += len(text)

        # Progress report at regular character-count intervals
        if n_chars % LOG_INTERVAL < len(text):
            log.info(
                "  %10d tokens  |  %10d chars  |  %d docs",
                len(all_tokens), n_chars, n_docs,
            )

    log.info(
        "Tokenisation complete:  %d tokens  |  %d chars  |  %d docs",
        len(all_tokens), n_chars, n_docs,
    )

    if len(all_tokens) == 0:
        log.error("No tokens were produced. Aborting without writing a file.")
        sys.exit(1)

    # ── Write binary file ──────────────────────────────────────────────────────
    write_bin(args.output, all_tokens)


if __name__ == "__main__":
    main()
