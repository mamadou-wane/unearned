"""Byte-level BPE tokenizer: trainer, vocabulary, encoder, and decoder."""

import re
from collections import Counter
from collections.abc import Iterable

NUM_BYTE_TOKENS = 256

_PRETOKEN = re.compile(
    rb"[ \t\n\r\f\v]*"  # leading whitespace stays with the run after it
    rb"(?:[0-9A-Za-z_\x80-\xff]+"  # one run of identifier bytes
    rb"|[^ \t\n\r\f\v0-9A-Za-z_\x80-\xff]+)"  # or one run of other bytes
    rb"|[ \t\n\r\f\v]+"  # whitespace with nothing after it
)


def pretokenize(document: bytes) -> list[bytes]:
    """Split bytes into deterministic pre-tokens without losing data."""
    return _PRETOKEN.findall(document)


def _prepare_special_tokens(
    special_tokens: Iterable[bytes],
) -> tuple[tuple[bytes, ...], re.Pattern[bytes] | None]:
    """Validate spellings and build a literal, leftmost-longest matcher."""
    if isinstance(special_tokens, (bytes, bytearray, str)):
        raise TypeError("special_tokens must be an iterable of bytes objects")
    tokens = tuple(special_tokens)
    if any(not isinstance(token, bytes) for token in tokens):
        raise TypeError("each special token must be bytes")
    if any(not token for token in tokens):
        raise ValueError("special tokens must not be empty")
    if len(set(tokens)) != len(tokens):
        raise ValueError("special tokens must be distinct")
    if not tokens:
        return tokens, None
    # Match longer spellings first, without changing registration order/ids.
    alternatives = (re.escape(token) for token in sorted(tokens, key=len, reverse=True))
    return tokens, re.compile(b"(" + b"|".join(alternatives) + b")")


def _merge_pair(tokens: Iterable[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Apply non-overlapping left-to-right replacements."""
    tokens = list(tokens)
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


class BPETokenizer:
    """Byte-level encoder and decoder with ordered merges and registered specials.

    Token ids 0 to 255 represent byte values; merge k creates id 256 + k.
    Special ids follow learned merges, in registration order. Registered
    spellings match literal bytes everywhere, including inside source literals:
    leftmost first, then longest at the same position. Specials are atomic
    tokens and merge boundaries. No specials are inserted automatically;
    decoding reproduces their bytes.
    """

    def __init__(
        self, merges: Iterable[tuple[int, int]], special_tokens: Iterable[bytes] = ()
    ):
        self.special_tokens, self._special_pattern = _prepare_special_tokens(special_tokens)
        vocab = [bytes([value]) for value in range(NUM_BYTE_TOKENS)]
        rank: dict[tuple[int, int], int] = {}
        for left, right in merges:
            if not (0 <= left < len(vocab) and 0 <= right < len(vocab)):
                raise ValueError(
                    f"merge {len(rank)} is {(left, right)}, but only token ids "
                    f"0 to {len(vocab) - 1} exist before it"
                )
            if (left, right) in rank:
                raise ValueError(f"the pair {(left, right)} is merged twice")
            rank[(left, right)] = len(rank)
            vocab.append(vocab[left] + vocab[right])
        self.merges: tuple[tuple[int, int], ...] = tuple(rank)
        self._special_ids = {
            token: len(vocab) + offset for offset, token in enumerate(self.special_tokens)
        }
        vocab.extend(self.special_tokens)
        self.vocab: tuple[bytes, ...] = tuple(vocab)
        self._rank = rank

    def encode(self, document: bytes) -> list[int]:
        """Recognize configured specials, then apply BPE to ordinary segments."""
        ids: list[int] = []
        parts = self._special_pattern.split(document) if self._special_pattern else [document]
        for part in parts:
            if part in self._special_ids:
                ids.append(self._special_ids[part])
            else:
                for pretoken in pretokenize(part):
                    ids.extend(self._encode_pretoken(pretoken))
        return ids

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        # Apply the earliest-learned merge present until none applies.
        tokens = list(pretoken)
        while len(tokens) > 1:
            earliest = min(
                zip(tokens, tokens[1:]),
                key=lambda pair: self._rank.get(pair, len(self._rank)),
            )
            if earliest not in self._rank:
                break
            new_id = NUM_BYTE_TOKENS + self._rank[earliest]
            tokens = _merge_pair(tokens, earliest, new_id)
        return tokens

    def decode(self, ids: Iterable[int]) -> bytes:
        """Decode token ids back to bytes. Raises ValueError on an unknown id."""
        pieces = []
        for token in ids:
            if not 0 <= token < len(self.vocab):
                raise ValueError(
                    f"unknown token id {token}; valid ids are 0 to {len(self.vocab) - 1}"
                )
            pieces.append(self.vocab[token])
        return b"".join(pieces)


def train_bpe(
    documents: Iterable[bytes], num_merges: int, special_tokens: Iterable[bytes] = ()
) -> BPETokenizer:
    """Learn up to `num_merges` merges from an iterable of bytes documents.

    Pairs never cross document, pre-token, or special-token boundaries.
    Registered specials do not contribute to pair counts; their ids follow
    the merges actually learned. Matching follows BPETokenizer's contract.
    Training stops when no adjacent pair remains.
    """
    if isinstance(documents, (bytes, bytearray, str)):
        raise TypeError("documents must be an iterable of bytes objects, one per document")
    if num_merges < 0:
        raise ValueError(f"num_merges must be 0 or more, got {num_merges}")
    special_tokens, special_pattern = _prepare_special_tokens(special_tokens)

    # Count ordinary pre-tokens only; special-token spans are merge boundaries.
    words: dict[tuple[int, ...], int] = Counter(
        tuple(pretoken)
        for document in documents
        for part in (special_pattern.split(document)[::2] if special_pattern else [document])
        for pretoken in pretokenize(part)
    )
    vocab = [bytes([value]) for value in range(NUM_BYTE_TOKENS)]
    merges: list[tuple[int, int]] = []

    for _ in range(num_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, occurrences in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += occurrences
        if not pair_counts:
            break

        # Highest count wins; ties go to the greatest (left bytes, right bytes).
        best = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        new_id = len(vocab)
        merges.append(best)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        # Two different words spell different bytes, so their counts never collide.
        words = {
            tuple(_merge_pair(word, best, new_id)): occurrences
            for word, occurrences in words.items()
        }

    return BPETokenizer(merges, special_tokens=special_tokens)
