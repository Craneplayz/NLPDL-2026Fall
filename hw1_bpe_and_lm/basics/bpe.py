from collections import Counter
from itertools import repeat

import regex as re

from concurrent.futures import ProcessPoolExecutor
from basics.pretokenization_example import find_chunk_boundaries


def pretokenize(input_path: str, start: int, end:int, special_tokens:list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    counts = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    delimiter = "|".join(escaped)
    pieces = re.split(delimiter, chunk) if delimiter else [chunk]

    for piece in pieces:
        for match in re.finditer(PAT, piece):
            word= match.group().encode("UTF-8")
            byte_tuple = tuple(bytes([b]) for b in word)
            counts[byte_tuple] += 1

    return counts


def bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    num_processes = 4

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    counts= Counter()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(
            executor.map(
                pretokenize,
                repeat(input_path),
                boundaries[:-1],
                boundaries[1:],
                repeat(special_tokens),
            )
        )

        for result in results:
            counts.update(result)



    vocabulary = [""] * vocab_size
    vocabulary[-1] = "<|endoftext|>"

    return counts
