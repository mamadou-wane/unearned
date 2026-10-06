"""Tests for the byte-level BPE tokenizer.

Every expected merge list and token sequence written out in this file was
worked out by hand, and the working is in the comments. The `reference_*`
helpers restate pre-tokenization, training, and encoding in a second, slower
form, so the tests on the C++ fixtures and on random corpora do not compare the
implementation with itself.
"""

import itertools
import random
import string
from collections import Counter
from pathlib import Path

import pytest

from unearned.tokenizer import BPETokenizer, pretokenize, train_bpe

A, B, C, D, F, Y, Z = b"abcdfyz"
SPACE, COMMA = b" ,"
BYTE_VOCAB = tuple(bytes([value]) for value in range(256))

# Synthetic C++ sources, read as bytes. They are never compiled or run.
FIXTURE_DOCUMENTS = [
    path.read_bytes()
    for path in sorted((Path(__file__).parent / "fixtures").glob("*.cpp"))
]

WHITESPACE = b" \t\n\r\f\v"
ASCII_IDENTIFIER = (string.ascii_letters + string.digits + "_").encode()


def byte_class(value: int) -> str:
    if value in WHITESPACE:
        return "whitespace"
    if value >= 0x80 or value in ASCII_IDENTIFIER:
        return "identifier"
    return "other"


def reference_pretokenize(document: bytes) -> list[bytes]:
    """The pre-token rule as a boundary test between neighbouring bytes.

    A pre-token ends after a byte that is not whitespace when the next byte is
    in a different class. Whitespace never ends a pre-token.
    """
    chunks = []
    start = 0
    for i in range(1, len(document)):
        before, after = byte_class(document[i - 1]), byte_class(document[i])
        if before != "whitespace" and after != before:
            chunks.append(document[start:i])
            start = i
    if document:
        chunks.append(document[start:])
    return chunks


def reference_merge(tokens: list[int], pair, new_id: int) -> list[int]:
    merged = []
    i = 0
    while i < len(tokens):
        if tuple(tokens[i : i + 2]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def reference_train(documents, num_merges: int):
    """Train without grouping equal pre-tokens.

    Every pre-token occurrence is its own list, and every pair is recounted
    from those lists before each merge.
    """
    occurrences = [
        list(chunk) for document in documents for chunk in reference_pretokenize(document)
    ]
    token_bytes = {value: bytes([value]) for value in range(256)}
    merges = []
    for _ in range(num_merges):
        counts = Counter(
            pair for tokens in occurrences for pair in zip(tokens, tokens[1:])
        )
        if not counts:
            break
        best = max(
            counts,
            key=lambda pair: (counts[pair], token_bytes[pair[0]], token_bytes[pair[1]]),
        )
        new_id = 256 + len(merges)
        token_bytes[new_id] = token_bytes[best[0]] + token_bytes[best[1]]
        merges.append(best)
        occurrences = [reference_merge(tokens, best, new_id) for tokens in occurrences]
    return tuple(merges)


def reference_encode(merges, document: bytes) -> list[int]:
    """Replay every merge, in the order learned, over each pre-token."""
    ids = []
    for chunk in reference_pretokenize(document):
        tokens = list(chunk)
        for rank, pair in enumerate(merges):
            tokens = reference_merge(tokens, pair, 256 + rank)
        ids.extend(tokens)
    return ids


@pytest.fixture(scope="module")
def cpp_tokenizer() -> BPETokenizer:
    assert len(FIXTURE_DOCUMENTS) == 3
    return train_bpe(FIXTURE_DOCUMENTS, 200)


# Pre-token boundaries


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (b"", []),
        (b"int x = 0;\n", [b"int", b" x", b" =", b" 0", b";", b"\n"]),
        (
            b"int x = 0;\n    return x;\n}",
            [b"int", b" x", b" =", b" 0", b";", b"\n    return", b" x", b";", b"\n}"],
        ),
        (b"a->b", [b"a", b"->", b"b"]),
        (
            b"std::vector<uint32_t> v_1;",
            [b"std", b"::", b"vector", b"<", b"uint32_t", b">", b" v_1", b";"],
        ),
        (b"x+=y;//z", [b"x", b"+=", b"y", b";//", b"z"]),
        # Whitespace only, leading whitespace, and trailing whitespace.
        (b"\t\r\n ", [b"\t\r\n "]),
        (b"  a", [b"  a"]),
        (b"a \n", [b"a", b" \n"]),
        (b"\x0b\x0c;", [b"\x0b\x0c;"]),
        # 0xc3 0xa9 is one UTF-8 character and 0xff is not valid UTF-8. Both
        # are identifier bytes. NUL and 0x01 are "other".
        (b"caf\xc3\xa9 \xff\x00\x01", [b"caf\xc3\xa9", b" \xff", b"\x00\x01"]),
        (b"a\x00b", [b"a", b"\x00", b"b"]),
        # 0x1c counts as whitespace for str.isspace, but not here.
        (b"a\x1cb", [b"a", b"\x1c", b"b"]),
    ],
)
def test_pretokenize_examples(document, expected):
    assert pretokenize(document) == expected


def test_pretokenize_keeps_every_byte_once_on_every_short_document():
    # All documents of one or two bytes put each byte value next to every
    # other. Longer documents over two bytes per class, plus NUL and 0xff,
    # cover every sequence of class changes up to length five.
    every_byte = [bytes([value]) for value in range(256)]
    two_per_class = [b" ", b"\n", b"a", b"_", b"\x80", b"\xff", b";", b"\x00"]
    documents = itertools.chain(
        (b"".join(p) for n in (1, 2) for p in itertools.product(every_byte, repeat=n)),
        (
            b"".join(p)
            for n in (3, 4, 5)
            for p in itertools.product(two_per_class, repeat=n)
        ),
    )
    for document in documents:
        chunks = pretokenize(document)
        assert b"".join(chunks) == document
        assert chunks == reference_pretokenize(document)


# Training


def test_training_follows_the_hand_worked_example():
    # One pre-token: a a a b d a a a b a c
    #
    # Step 1. (a,a)=4, because each "aaa" holds the pair twice. (a,b)=2.
    # (b,d), (d,a), (b,a), (a,c) are 1 each. Merge (a,a) as 256:
    #     256 a b d 256 a b a c
    # Step 2. (256,a)=2 and (a,b)=2 tie; the rest are 1. b"aa" > b"a", so
    # (256,a) wins. Merge it as 257:
    #     257 b d 257 b a c
    # Step 3. (257,b)=2; the rest are 1. Merge it as 258:
    #     258 d 258 a c
    tokenizer = train_bpe([b"aaabdaaabac"], 3)

    assert tokenizer.merges == ((A, A), (256, A), (257, B))
    assert tokenizer.vocab[:256] == BYTE_VOCAB
    assert tokenizer.vocab[256:] == (b"aa", b"aaa", b"aaab")
    assert tokenizer.encode(b"aaabdaaabac") == [258, D, 258, A, C]


def test_frequency_tie_goes_to_the_greatest_pair():
    # a b c d: (a,b), (b,c), (c,d) are 1 each, and (c,d) compares greatest.
    #     a b 256            256 = "cd"
    # (a,b) and (b,256) tie. b"b" > b"a", so (b,256) wins.
    #     a 257              257 = "bcd"
    # (a,257) is the only pair.
    #     258                258 = "abcd"
    tokenizer = train_bpe([b"abcd"], 3)
    assert tokenizer.merges == ((C, D), (B, 256), (A, 257))
    assert tokenizer.vocab[256:] == (b"cd", b"bcd", b"abcd")

    # Equal left tokens: the right token decides, in either document order.
    assert train_bpe([b"ab", b"ac"], 1).merges == ((A, C),)
    assert train_bpe([b"ac", b"ab"], 1).merges == ((A, C),)


def test_frequency_tie_compares_token_bytes_not_token_ids():
    # Step 1. (a,a)=3: twice from "aaz" and once from "aa". (a,z)=2, (b,z)=2.
    # Merge (a,a) as 256. The words are now 256 z (twice), b z (twice), 256.
    # Step 2. (256,z)=2 and (b,z)=2 tie. Token 256 has the larger id, but its
    # bytes b"aa" sort before b"b", so (b,z) wins.
    tokenizer = train_bpe([b"aaz", b"aaz", b"bz", b"bz", b"aa"], 2)
    assert tokenizer.merges == ((A, A), (B, Z))

    # The same on the right token. (c,d)=3 merges as 256, leaving a 256 and
    # a z. (a,256) and (a,z) tie with equal left tokens. b"z" > b"cd", so
    # (a,z) wins although 256 is the larger id.
    tokenizer = train_bpe([b"cd", b"cd", b"acd", b"az"], 2)
    assert tokenizer.merges == ((C, D), (A, Z))


def test_frequency_tie_reads_a_merged_token_as_left_bytes_then_right_bytes():
    # (a,b)=3 merges as 256 = b"ab". Then (256,z)=1 and (b,z)=1 tie, and
    # b"b" > b"ab", so (b,z) wins. If token 256 were read as b"ba", (256,z)
    # would win.
    tokenizer = train_bpe([b"ab", b"ab", b"abz", b"bz"], 2)
    assert tokenizer.merges == ((A, B), (B, Z))


def test_a_repeated_pretoken_counts_once_per_occurrence():
    # Step 1. (a,b)=3, (c,d)=2, (z,y)=1. Merge (a,b).
    # Step 2. (c,d)=2 beats (z,y)=1.
    # If each distinct pre-token counted once, at either step, the pairs would
    # tie at 1 and (z,y) would win on bytes.
    tokenizer = train_bpe([b"ab"] * 3 + [b"cd"] * 2 + [b"zy"], 2)
    assert tokenizer.merges == ((A, B), (C, D))


def test_overlapping_pairs_are_counted_and_merged_left_to_right():
    # Step 1. "aaa" holds (a,a) twice, so (a,a)=2 beats (z,y)=1. Counting only
    # non-overlapping occurrences would tie them at 1, and (z,y) would win.
    # Merging left to right turns a a a into 256 a, not a 256.
    # Step 2. (256,a)=1 and (z,y)=1 tie. b"z" > b"aa", so (z,y) wins as 257.
    # Step 3. (256,a) is the only pair left. Merge it as 258.
    tokenizer = train_bpe([b"aaa", b"zy"], 3)
    assert tokenizer.merges == ((A, A), (Z, Y), (256, A))
    assert tokenizer.vocab[256:] == (b"aa", b"zy", b"aaa")

    # a a a a   -> 256 256, and no merge joins (256,256).
    # a a a a a -> 256 256 a -> 256 258.
    assert tokenizer.encode(b"aaa") == [258]
    assert tokenizer.encode(b"aaaa") == [256, 256]
    assert tokenizer.encode(b"aaaaa") == [256, 258]


def test_pairs_are_not_counted_across_documents():
    # Ten one-byte documents hold no pair at all. Joined, they would spell
    # "ababababab" and (a,b) would merge.
    assert train_bpe([b"a", b"b"] * 5, 3).merges == ()
    assert train_bpe([b"ab"] * 5, 3).merges == ((A, B),)


def test_pairs_are_not_counted_across_pretoken_boundaries():
    # "ab,ab,ab" is the pre-tokens ab , ab , ab. Only (a,b) is ever adjacent.
    # After it merges, every pre-token is one token and training stops.
    assert train_bpe([b"ab,ab,ab"], 5).merges == ((A, B),)

    # "x y y y" is the pre-tokens "x", " y", " y", " y". The space belongs to
    # the y after it, so (space,y) is a pair and (x,space) is not.
    assert train_bpe([b"x y y y"], 5).merges == ((SPACE, Y),)


def test_training_keeps_non_ascii_bytes_in_the_identifier_pretoken():
    # b"caf\xc3\xa9" is one pre-token: c a f C3 A9, four pairs at 1 each.
    # (C3,A9) has the greatest left byte and merges as 256:   c a f 256
    # (f,256) beats (c,a) and (a,f) on its left byte, as 257:  c a 257
    # (c,a) beats (a,257) on its left byte, as 258:            258 257
    # (258,257) is the only pair left, as 259.
    # If training split before 0xC3, the pair (f,256) could not exist.
    tokenizer = train_bpe([b"caf\xc3\xa9"], 4)
    assert tokenizer.merges == ((0xC3, 0xA9), (F, 256), (C, A), (258, 257))
    assert tokenizer.encode(b"caf\xc3\xa9") == [259]


def test_training_stops_when_no_pair_is_left():
    # "abcd" allows three merges (worked out above), whatever the budget.
    assert len(train_bpe([b"abcd"], 10).merges) == 3


def test_zero_merges_gives_the_byte_tokenizer():
    tokenizer = train_bpe([b"abab"], 0)
    assert tokenizer.merges == ()
    assert tokenizer.vocab == BYTE_VOCAB
    assert tokenizer.encode(b"abab") == [A, B, A, B]
    assert tokenizer.encode(bytes(range(256))) == list(range(256))


@pytest.mark.parametrize("documents", [[], [b""], [b"", b""], [b"a"], [b"a", b""]])
def test_empty_and_single_byte_input_learn_nothing(documents):
    tokenizer = train_bpe(documents, 5)
    assert tokenizer.merges == ()
    assert tokenizer.vocab == BYTE_VOCAB
    assert tokenizer.encode(b"") == []
    assert tokenizer.decode([]) == b""
    assert tokenizer.encode(b"a") == [A]
    assert tokenizer.decode([A]) == b"a"


def test_training_rejects_bad_arguments():
    with pytest.raises(ValueError):
        train_bpe([b"ab"], -1)
    # One bytes or str object is not a collection of documents. The empty ones
    # matter: they would otherwise pass as a corpus with no documents.
    for not_a_corpus in (b"abab", "abab", b"", ""):
        with pytest.raises(TypeError):
            train_bpe(not_a_corpus, 1)


def test_training_is_repeatable():
    first = train_bpe(FIXTURE_DOCUMENTS, 200)
    second = train_bpe(FIXTURE_DOCUMENTS, 200)
    assert first.merges == second.merges
    assert first.vocab == second.vocab

    # Corpus order and iterator type must not affect merge selection.
    assert train_bpe(reversed(FIXTURE_DOCUMENTS), 200).merges == first.merges
    assert train_bpe(iter(FIXTURE_DOCUMENTS), 200).merges == first.merges


def test_training_matches_the_reference_on_random_small_corpora():
    # Two to four distinct bytes per corpus, so ties and overlapping pairs
    # occur at almost every step. Budgets run past the point where pairs run out.
    rng = random.Random(336)
    alphabets = [b"ab", b"abc", b"ab ", b"a;b ", b"a\x80 ;"]
    for _ in range(400):
        alphabet = rng.choice(alphabets)
        documents = [
            bytes(rng.choice(alphabet) for _ in range(rng.randrange(13)))
            for _ in range(rng.randrange(1, 7))
        ]
        budget = rng.randrange(12)
        assert train_bpe(documents, budget).merges == reference_train(documents, budget)


# Encoding


def test_encode_uses_learned_merge_order_not_frequency():
    # Training: (b,c)=3 merges first as 256, then (a,b)=2 as 257.
    tokenizer = train_bpe([b"bc"] * 3 + [b"ab"] * 2, 2)
    assert tokenizer.merges == ((B, C), (A, B))

    # In "abcab" the pair (a,b) occurs twice and (b,c) once. Counting would
    # merge (a,b) first and give 257 c 257. The learned order merges (b,c)
    # first:  a b c a b -> a 256 a b -> a 256 257.
    assert tokenizer.encode(b"abcab") == [A, 256, 257]


def test_encode_with_the_hand_worked_tokenizer():
    # Merges, in order: (a,a) as 256, (256,a) as 257, (257,b) as 258.
    tokenizer = BPETokenizer([(A, A), (256, A), (257, B)])
    assert tokenizer.encode(b"abab") == [A, B, A, B]
    assert tokenizer.encode(b"aab") == [256, B]
    assert tokenizer.encode(b"aaab") == [258]
    # a a a a b -> 256 256 b. The pairs (256,256) and (256,b) are not merges.
    assert tokenizer.encode(b"aaaab") == [256, 256, B]


def test_encode_merges_left_to_right_without_overlap():
    tokenizer = BPETokenizer([(A, A)])
    assert tokenizer.encode(b"aaa") == [256, A]
    assert tokenizer.encode(b"aaaaa") == [256, 256, A]


def test_encode_does_not_merge_across_pretoken_boundaries():
    # "a" and "," are different pre-tokens, so the merge (a,",") cannot apply.
    assert BPETokenizer([(A, COMMA)]).encode(b"a,") == [A, COMMA]
    assert BPETokenizer([(A, B)]).encode(b"ab,ab") == [256, COMMA, 256]

    # "a b" is the pre-tokens "a" and " b".
    assert BPETokenizer([(SPACE, B)]).encode(b"a b") == [A, 256]
    assert BPETokenizer([(A, SPACE)]).encode(b"a b") == [A, SPACE, B]
    # Trailing whitespace is its own pre-token.
    assert BPETokenizer([(A, SPACE)]).encode(b"a ") == [A, SPACE]


def test_tokenizer_rejects_malformed_merges():
    with pytest.raises(ValueError):
        BPETokenizer([(256, A)])  # token 256 does not exist before merge 0
    with pytest.raises(ValueError):
        BPETokenizer([(-1, A)])
    with pytest.raises(ValueError):
        BPETokenizer([(A, A), (A, A)])


def test_decode_rejects_unknown_ids():
    tokenizer = BPETokenizer([(A, A)])
    assert tokenizer.decode([256, B]) == b"aab"
    with pytest.raises(ValueError):
        tokenizer.decode([257])
    with pytest.raises(ValueError):
        tokenizer.decode([-1])


# The C++ fixtures


def test_fixture_training_uses_the_whole_budget(cpp_tokenizer):
    assert len(cpp_tokenizer.merges) == 200
    assert len(cpp_tokenizer.vocab) == 456


def test_fixture_merges_match_the_reference_trainer(cpp_tokenizer):
    assert cpp_tokenizer.merges == reference_train(FIXTURE_DOCUMENTS, 200)


def test_fixture_encoding_matches_the_reference_and_compresses(cpp_tokenizer):
    for document in FIXTURE_DOCUMENTS:
        ids = cpp_tokenizer.encode(document)
        assert ids == reference_encode(cpp_tokenizer.merges, document)
        assert len(ids) < len(document)


def test_learned_tokens_are_distinct_and_stay_inside_one_pretoken(cpp_tokenizer):
    assert len(set(cpp_tokenizer.vocab)) == len(cpp_tokenizer.vocab)
    for token in cpp_tokenizer.vocab[256:]:
        assert len(reference_pretokenize(token)) == 1


# Round trips


@pytest.mark.parametrize(
    "document",
    [
        b"",
        b"a",
        b" ",
        b"\t\tint  x ;  \r\n\r\n   ",
        "héllo, wörld, 日本語, ∀x ∈ ℝ".encode(),
        b"\x00",
        b"a\x00\x00b\x00",
        b"\xff\xfe\x80abc\xc3",  # stray continuation byte, truncated sequence
        b"\xc3\x28\xa0\xa1\xe2\x28\xa1\xf0\x28\x8c\xbc",  # invalid UTF-8
        bytes(range(256)),
        bytes(reversed(range(256))) * 2,
        *FIXTURE_DOCUMENTS,
    ],
)
def test_round_trip_returns_the_original_bytes(cpp_tokenizer, document):
    ids = cpp_tokenizer.encode(document)
    assert all(0 <= token < len(cpp_tokenizer.vocab) for token in ids)
    assert cpp_tokenizer.decode(ids) == document


def test_round_trip_on_random_documents(cpp_tokenizer):
    rng = random.Random(20261005)
    # Bytes that occur in the fixtures, so that merges apply, plus NUL and two
    # bytes that never occur in valid UTF-8.
    likely = b"abst_:;(){}<> \n\t" + b"\x00\x80\xff"
    for _ in range(300):
        length = rng.randrange(200)
        for document in (
            rng.randbytes(length),
            bytes(rng.choice(likely) for _ in range(length)),
        ):
            assert cpp_tokenizer.decode(cpp_tokenizer.encode(document)) == document


# Special tokens


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (b"", []),
        (b"<eos>", [256]),
        (b"<eos><eos>", [256, 256]),
        (b"a<eos>b<eos>", [A, 256, B, 256]),
        (b"<eos>\n\n", [256, 10, 10]),
        (b"<eos>\n\nb", [256, 10, 10, B]),
        (b"<eos", list(b"<eos")),
        (b"<other>", list(b"<other>")),
    ],
)
def test_special_tokens_encode_atomically_and_round_trip(document, expected):
    tokenizer = BPETokenizer([], special_tokens=[b"<eos>"])
    assert tokenizer.encode(document) == expected
    assert tokenizer.decode(expected) == document


def test_special_ids_follow_merges_in_registration_order():
    tokenizer = BPETokenizer([(A, B)], special_tokens=[b"<start>", b"<end>"])
    assert tokenizer.vocab[256:] == (b"ab", b"<start>", b"<end>")
    assert tokenizer.encode(b"ab<end><start>ab") == [256, 258, 257, 256]
    assert tokenizer.decode([256, 258, 257, 256]) == b"ab<end><start>ab"
    with pytest.raises(ValueError):
        tokenizer.decode([259])


def test_special_tokens_take_priority_over_existing_merges():
    # Both learned pairs would apply to "abc" without the special "b".
    tokenizer = BPETokenizer([(A, B), (256, C)], special_tokens=[b"b"])
    assert tokenizer.encode(b"abc") == [A, 258, C]
    assert tokenizer.decode([A, 258, C]) == b"abc"


def test_special_token_overlap_uses_leftmost_then_longest_match():
    tokenizer = BPETokenizer([], special_tokens=[b"ab", b"aba", b"bc"])
    # Match aba rather than ab at offset 0. Later, ab wins over bc because
    # its start is earlier. Registration order controls ids, not match order.
    assert tokenizer.encode(b"abaabc") == [257, 256, C]
    assert tokenizer.decode([257, 256, C]) == b"abaabc"


@pytest.mark.parametrize("special", [b".*+?()[]{}\\|^$", b"\x00\xff", "日本語".encode()])
def test_special_spellings_are_literal_bytes(special):
    tokenizer = train_bpe([special] * 10, 10, special_tokens=[special])
    assert tokenizer.merges == ()
    assert tokenizer.encode(b"a" + special + b"b") == [A, 256, B]
    document = bytes(range(256)) + special + b"\xff\x00"
    assert tokenizer.decode(tokenizer.encode(document)) == document


def test_training_excludes_special_contents_and_keeps_the_boundary():
    # The marker's many (z,z) pairs cannot enter the counts. Each ordinary
    # segment is "ab", so only (a,b) can merge. Removing the marker and
    # joining its neighbours would wrongly allow an (ab,ab) merge as well.
    tokenizer = train_bpe([b"abzzzzab"] * 3, 10, special_tokens=[b"zzzz"])
    assert tokenizer.merges == ((A, B),)
    assert tokenizer.vocab[256:] == (b"ab", b"zzzz")
    assert tokenizer.encode(b"abzzzzab") == [256, 257, 256]


def test_training_uses_the_same_longest_special_match_as_encoding():
    tokenizer = train_bpe([b"abXYzzab"], 10, special_tokens=[b"XY", b"XYzz"])
    assert tokenizer.merges == ((A, B),)
    assert tokenizer.encode(b"abXYzzab") == [256, 258, 256]


@pytest.mark.parametrize("documents", [[], [b""], [b"<eos><eos>"]])
def test_special_ids_exist_even_when_training_learns_no_merges(documents):
    tokenizer = train_bpe(documents, 10, special_tokens=[b"<eos>"])
    assert tokenizer.merges == ()
    assert tokenizer.encode(b"<eos>") == [256]
    assert tokenizer.decode([256]) == b"<eos>"


def test_unconfigured_markers_remain_ordinary_bytes():
    tokenizer = train_bpe([b"<eos>"], 0)
    assert tokenizer.encode(b"<eos>") == list(b"<eos>")
    assert tokenizer.vocab == BYTE_VOCAB


def test_special_training_is_repeatable_with_one_shot_iterables():
    special_tokens = [b"<start>", b"<end>"]
    documents = [b"<start>" + document + b"<end>" for document in FIXTURE_DOCUMENTS]
    first = train_bpe(documents, 200, special_tokens=special_tokens)
    second = train_bpe(reversed(documents), 200, special_tokens=iter(special_tokens))
    assert first.merges == second.merges == reference_train(FIXTURE_DOCUMENTS, 200)
    assert first.vocab == second.vocab
    for document in documents:
        ids = first.encode(document)
        assert ids == second.encode(document)
        assert ids[0] == 456
        assert ids[-1] == 457
        assert first.decode(ids) == document


@pytest.mark.parametrize(
    ("special_tokens", "error"),
    [
        ([b""], ValueError),
        ([b"<eos>", b"<eos>"], ValueError),
        (b"<eos>", TypeError),
        (b"", TypeError),
        ("<eos>", TypeError),
        (bytearray(b"<eos>"), TypeError),
        (["<eos>"], TypeError),
        ([bytearray(b"<eos>")], TypeError),
        ([256], TypeError),
    ],
)
def test_invalid_special_definitions_are_rejected(special_tokens, error):
    with pytest.raises(error):
        BPETokenizer([], special_tokens=special_tokens)
    with pytest.raises(error):
        train_bpe([], 0, special_tokens=special_tokens)
