"""A byte-level lexical boundary detector for one generated C++ function."""

from dataclasses import dataclass, field


def _next(source: bytes, position: int) -> tuple[int | None, int]:
    # Splicing precedes comment/string recognition, but not raw-string contents.
    while position < len(source):
        if source.startswith(b'\\\r\n', position):
            position += 3
        elif source.startswith(b'\\\n', position):
            position += 2
        else:
            return source[position], position + 1
    return None, position


def _identifier_start(char: int | None) -> bool:
    return char is not None and (65 <= char <= 90 or 97 <= char <= 122
                                 or char == 95 or char >= 128)


def _digit(char: int | None) -> bool:
    return char is not None and 48 <= char <= 57


def _identifier_part(char: int | None) -> bool:
    return _identifier_start(char) or _digit(char)


def _number_end(source: bytes, position: int, previous: int) -> int:
    # An apostrophe inside a preprocessing number is not a character literal.
    while True:
        char, following = _next(source, position)
        if _identifier_part(char) or char == 46:
            previous, position = char, following
        elif char in (43, 45) and previous in (69, 101, 80, 112):
            previous, position = char, following
        elif char == 39:
            after, end = _next(source, following)
            if not _identifier_part(after):
                return position
            previous, position = after, end
        else:
            return position


def _quoted_end(source: bytes, position: int, quote: int) -> int | None:
    while True:
        char, position = _next(source, position)
        if char is None or char in (10, 13):
            return None
        if char == quote:
            return position
        if char == 92:
            char, position = _next(source, position)
            if char is None or char in (10, 13):
                return None


def _raw_end(source: bytes, position: int) -> int | None:
    delimiter = bytearray()
    while position < len(source):
        char = source[position]
        position += 1
        if char == 40:
            terminator = b')' + bytes(delimiter) + b'"'
            closing = source.find(terminator, position)
            return None if closing < 0 else closing + len(terminator)
        if len(delimiter) == 16 or not 33 <= char <= 126 or char in (41, 92):
            return None
        delimiter.append(char)
    return None


def _function_end(source: bytes) -> int | None:
    depth = 0
    position = 0
    while True:
        char, position = _next(source, position)
        if char is None:
            return None
        if char == 35:
            # Directives require preprocessing; do not claim a lexical close.
            return None
        if char == 47:
            after, following = _next(source, position)
            if after == 47:
                position = following
                while True:
                    char, position = _next(source, position)
                    if char is None:
                        return None
                    if char in (10, 13):
                        break
                continue
            if after == 42:
                position = following
                while True:
                    char, position = _next(source, position)
                    if char is None:
                        return None
                    if char == 42:
                        after, following = _next(source, position)
                        if after == 47:
                            position = following
                            break
                continue
        if _identifier_start(char):
            word = bytearray((char,))
            while True:
                after, following = _next(source, position)
                if not _identifier_part(after):
                    break
                word.append(after)
                position = following
            if word in (b'R', b'u8R', b'uR', b'UR', b'LR') and after == 34:
                position = _raw_end(source, following)
                if position is None:
                    return None
            continue
        after, _ = _next(source, position)
        if _digit(char) or char == 46 and _digit(after):
            position = _number_end(source, position, char)
            continue
        if char in (34, 39):
            position = _quoted_end(source, position, char)
            if position is None:
                return None
            continue
        if char == 123:
            depth += 1
        elif char == 125:
            if depth == 0:
                return None
            depth -= 1
            if depth == 0:
                return position


@dataclass(frozen=True)
class FunctionStop:
    """Accumulated source and the exclusive byte offset of its first body close.

    The first ordinary opening brace designates the body. Supply only that
    function's source prefix, excluding instruction text and earlier braced
    declarations/initializers. Quotes, character literals, comments, raw literals
    and numeric separators hide their internal braces. LF/CRLF line splices are
    handled outside raw bodies; raw delimiters must be literal ASCII characters.

    This is not a C++ grammar or preprocessor: macros and alternative brace
    spellings are unsupported, and a normal-source # prevents a completion.
    Unclosed quotes/comments/raw literals and invalid raw delimiters produce
    no boundary. A detected boundary is not evidence of compilable code.

    The completing piece is retained whole, including bytes after end. Once
    complete, feed returns this state without consuming further pieces. No
    mutable parser state or UTF-8 decoding is needed across token boundaries.
    """

    source: bytes = b''
    end: int | None = field(init=False, default=None)

    def __post_init__(self):
        if not isinstance(self.source, bytes):
            raise TypeError('source must be bytes')
        object.__setattr__(self, 'end', _function_end(self.source))

    def feed(self, piece: bytes) -> 'FunctionStop':
        if not isinstance(piece, bytes):
            raise TypeError('piece must be bytes')
        if self.end is not None:
            return self
        return FunctionStop(self.source + piece)
