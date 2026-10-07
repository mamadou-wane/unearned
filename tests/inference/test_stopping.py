"""Lexical stops are independent of token boundaries and caller-owned state."""

from dataclasses import FrozenInstanceError

import pytest

from unearned.stopping import FunctionStop


COMPLETE = [
    b'int f(){ return 1; }',
    b'int f(){ if (true) { return 1; } else { return 0; } }',
    rb'''int f(){ const char* s = "\"}{"; return 0; }''',
    rb'''int f(){ char q = '\''; char slash = '\\'; return '}'; }''',
    b'int f(){ /* } { */ return 0; }',
    b'int f(){ // } {\n return 0; }',
    b'int f(){ /* many *** } ***/ return 0; }',
    b'int f(){ /**/ auto s = "# }"; /* # } */ return 0; }',
    b'int f(){ auto s = R"( } { ")"; return 0; }',
    b'int f(){ auto s = R"tag( } { )other" )tag"; return 0; }',
    b'int f(){ auto s = u8R"cpp( } \\ \" )cpp"; return 0; }',
    b'int f(){ auto s = uR"(})"; return 0; }',
    b'int f(){ auto s = UR"(})"; return 0; }',
    b'int f(){ auto s = LR"(})"; return 0; }',
    b'int f(){ auto s = R"tag(# } {)tag"; return 0; }',
    b'int f(){ auto s = R"abcdefghijklmnop(})abcdefghijklmnop"; }',
    b"int f(){ return 1'000 + 0xDE'AD + 0b10'01; }",
    b"int f(){ auto x = 1'000.25e+2; return '}'; }",
    b'int f(){ auto s = u8"}"; auto c = L\'}\'; return 0; }',
    b'int f(){ // } \\\n still }\n return 0; }',
    b'int f(){ // } \\\r\n still }\r\n return 0; }',
    b'int f(){ /\\\n* } *\\\n/ return 0; }',
    b'int f(){ /\\\n/ }\n return 0; }',
    b'int f(){ const char* s="brace \\\n }"; return 0; }',
    b'int f(){ auto s=u\\\n8R"tag(})tag"; return 0; }',
    b'int f(){ auto s=R"tag( } )ta\\\ng" } )tag"; return 0; }',
]


@pytest.mark.parametrize('source', COMPLETE)
def test_function_end_is_independent_of_every_two_piece_split(source):
    for split in range(len(source) + 1):
        initial = FunctionStop()
        first = initial.feed(source[:split])
        snapshot = (first.source, first.end)
        complete = first.feed(source[split:])
        assert initial.source == b'' and initial.end is None
        assert (first.source, first.end) == snapshot
        assert complete.source == source
        assert complete.end == len(source)


@pytest.mark.parametrize('source', COMPLETE)
def test_function_end_is_independent_of_single_byte_pieces(source):
    state = FunctionStop()
    for offset, byte in enumerate(source):
        state = state.feed(bytes((byte,)))
        assert state.end == (len(source) if offset == len(source) - 1 else None)
    assert state.source == source


@pytest.mark.parametrize('source', [
    b'', b'int f()', b'int f(){', b'} int f(){ return 0; }',
    b'int f(){ "unfinished }', b"int f(){ '}",
    b'int f(){ /* unfinished }', b'int f(){ // unfinished }',
    b'int f(){ R"unfinished', b'int f(){ R"tag( } )wrong"; }',
    b'int f(){ R"abcdefghijklmnopq(})abcdefghijklmnopq"; }',
    b'int f(){ R"bad delimiter(})bad delimiter"; }',
    b'int f(){ R"bad\\delimiter(})bad\\delimiter"; }',
    b'int f(){ "invalid\n }"; }', b"int f(){ 'invalid\n }'; }",
    b'int f(){\n#if 0\n}\n#endif\nreturn 0; }',
])
def test_incomplete_or_unsupported_source_does_not_claim_a_boundary(source):
    assert FunctionStop(source).end is None
    for split in range(len(source) + 1):
        assert FunctionStop().feed(source[:split]).feed(source[split:]).end is None


@pytest.mark.parametrize('prefix,suffix', [
    (b'int f(){ /* in a comment', b' } */ return 0; }'),
    (b'int f(){ const char* s="unfinished', b' }"; return 0; }'),
    (b'int f(){ auto s=R"del', b'im( } )delim"; return 0; }'),
    (b'int f(){ // continued \\', b'\n }\nreturn 0; }'),
])
def test_source_prefix_initializes_actual_lexical_state(prefix, suffix):
    state = FunctionStop().feed(prefix)
    assert state.end is None
    complete = state.feed(suffix)
    assert complete.end == len(prefix + suffix)
    assert state.source == prefix and state.end is None


def test_completion_retains_whole_piece_and_reports_first_boundary():
    prefix = b'int f(){ '
    final = b'return 0; } trailing bytes { another body }'
    state = FunctionStop(prefix)
    complete = state.feed(final)
    assert complete.source == prefix + final
    assert complete.end == len(prefix + b'return 0; }')
    assert complete.source[:complete.end] == b'int f(){ return 0; }'
    assert complete.feed(b'more bytes') is complete
    assert state.source == prefix and state.end is None


def test_source_is_immutable_and_end_is_derived():
    state = FunctionStop(b'int f(){ return 0; }')
    with pytest.raises(FrozenInstanceError):
        state.source = b'changed'
    with pytest.raises(FrozenInstanceError):
        state.end = 0


@pytest.mark.parametrize('value', ['text', bytearray(b'text'), [123], None])
def test_source_and_pieces_must_be_bytes(value):
    with pytest.raises(TypeError):
        FunctionStop(value)
    with pytest.raises(TypeError):
        FunctionStop().feed(value)
    with pytest.raises(TypeError):
        FunctionStop(b'int f(){}').feed(value)
