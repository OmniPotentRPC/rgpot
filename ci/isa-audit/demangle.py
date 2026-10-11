"""Demangle C++ (Itanium) and Rust (legacy and v0) symbol names.

Rust v0 follows the grammar in RFC 2603 and the printing rules of
rustc-demangle (crate disambiguators omitted, the ``{:#}`` form).
C++ and Rust legacy names go through the ``cxxfilt`` package, which
calls the system ``c++filt``. A legacy Rust hash suffix
(``::h`` plus 16 hex digits) is removed so an allowlist can name the
item.
"""

from __future__ import annotations

import re

try:
    import cxxfilt
except ImportError:  # pragma: no cover - the requirements file pins cxxfilt
    cxxfilt = None  # type: ignore[assignment]

_MAX_DEPTH = 500
_LEGACY_HASH = re.compile(r"::h[0-9a-f]{16}$")
_BASIC = {
    ord("b"): "bool",
    ord("c"): "char",
    ord("e"): "str",
    ord("u"): "()",
    ord("a"): "i8",
    ord("s"): "i16",
    ord("l"): "i32",
    ord("x"): "i64",
    ord("n"): "i128",
    ord("i"): "isize",
    ord("h"): "u8",
    ord("t"): "u16",
    ord("m"): "u32",
    ord("y"): "u64",
    ord("o"): "u128",
    ord("j"): "usize",
    ord("f"): "f32",
    ord("d"): "f64",
    ord("z"): "!",
    ord("p"): "_",
    ord("v"): "...",
}


class DemangleError(Exception):
    """The mangled string is not a symbol this module can decode."""


class _Parser:
    def __init__(self, sym: str, pos: int = 0, depth: int = 0) -> None:
        self.sym = sym
        self.pos = pos
        self.depth = depth
        self.bound_lifetimes = 0

    def _bump(self) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise DemangleError("recursion limit")

    def _peek(self) -> int | None:
        if self.pos >= len(self.sym):
            return None
        return self.sym[self.pos]

    def _next(self) -> int:
        byte = self._peek()
        if byte is None:
            raise DemangleError("truncated")
        self.pos += 1
        return ord(byte)

    def _eat(self, want: int) -> bool:
        if self._peek() is not None and ord(self._peek()) == want:  # type: ignore[arg-type]
            self.pos += 1
            return True
        return False

    def _digit10(self) -> int:
        byte = self._peek()
        if byte is None or not ("0" <= byte <= "9"):
            raise DemangleError("expected decimal digit")
        self.pos += 1
        return ord(byte) - ord("0")

    def _digit62(self) -> int:
        byte = self._peek()
        if byte is None:
            raise DemangleError("expected base-62 digit")
        self.pos += 1
        if "0" <= byte <= "9":
            return ord(byte) - ord("0")
        if "a" <= byte <= "z":
            return 10 + ord(byte) - ord("a")
        if "A" <= byte <= "Z":
            return 36 + ord(byte) - ord("A")
        raise DemangleError("expected base-62 digit")

    def _integer62(self) -> int:
        if self._eat(ord("_")):
            return 0
        value = 0
        while not self._eat(ord("_")):
            value = value * 62 + self._digit62()
        return value + 1

    def _opt_integer62(self, tag: int) -> int:
        if not self._eat(tag):
            return 0
        return self._integer62() + 1

    def _disambiguator(self) -> int:
        return self._opt_integer62(ord("s"))

    def _hex_nibbles(self) -> str:
        start = self.pos
        while True:
            byte = self._next()
            if chr(byte) in "0123456789abcdef":
                continue
            if byte == ord("_"):
                break
            raise DemangleError("bad hex nibble")
        return self.sym[start : self.pos - 1]

    def _hex_uint(self) -> int:
        text = self._hex_nibbles()
        if not text:
            raise DemangleError("empty hex")
        return int(text, 16)

    def _ident(self) -> str:
        puny = self._eat(ord("u"))
        length = self._digit10()
        if length != 0:
            while True:
                byte = self._peek()
                if byte is None or not ("0" <= byte <= "9"):
                    break
                length = length * 10 + self._digit10()
        self._eat(ord("_"))
        end = self.pos + length
        if end > len(self.sym):
            raise DemangleError("identifier overruns the symbol")
        raw = self.sym[self.pos : end]
        self.pos = end
        if not puny:
            return raw
        split = raw.rfind("_")
        if split < 0:
            ascii_part, payload = "", raw
        else:
            ascii_part, payload = raw[:split], raw[split + 1 :]
        decoded = _punycode_decode(ascii_part, payload)
        if decoded is None:
            shown = (ascii_part + "-" + payload) if ascii_part else payload
            return "punycode{" + shown + "}"
        return decoded

    def _namespace(self) -> str | None:
        byte = self._next()
        if ord("A") <= byte <= ord("Z"):
            return chr(byte)
        if ord("a") <= byte <= ord("z"):
            return None
        raise DemangleError("bad namespace")

    def _backref(self, method: str, *args: object) -> str:
        start = self.pos - 1
        index = self._integer62()
        if index >= start:
            raise DemangleError("bad backref")
        child = _Parser(self.sym, index, self.depth)
        child.bound_lifetimes = self.bound_lifetimes
        return getattr(child, method)(*args)

    def _lifetime(self, index: int) -> str:
        if index == 0:
            return "'_"
        depth = self.bound_lifetimes - index
        if depth < 0:
            raise DemangleError("bad lifetime")
        if depth < 26:
            return "'" + chr(ord("a") + depth)
        return "'_" + str(depth)

    def _in_binder(self, method: str) -> str:
        count = self._opt_integer62(ord("G"))
        prefix = ""
        if count:
            parts = []
            for _ in range(count):
                self.bound_lifetimes += 1
                parts.append(self._lifetime(1))
            prefix = "for<" + ", ".join(parts) + "> "
        try:
            body = getattr(self, method)()
        finally:
            self.bound_lifetimes -= count
        return prefix + body

    def _sep_list(self, method: str, sep: str) -> list[str]:
        items: list[str] = []
        while self._peek() is not None and not self._eat(ord("E")):
            items.append(getattr(self, method)())
        return items

    def path(self, in_value: bool) -> str:
        self._bump()
        tag = self._next()
        if tag == ord("C"):
            self._disambiguator()
            return self._ident()
        if tag == ord("N"):
            namespace = self._namespace()
            parent = self.path(in_value)
            dis = self._disambiguator()
            name = self._ident()
            if namespace is not None:
                label = {"C": "closure", "S": "shim"}.get(namespace, namespace)
                if name:
                    label = label + ":" + name
                return parent + "::{" + label + "#" + str(dis) + "}"
            if name:
                return parent + "::" + name
            return parent
        if tag in (ord("M"), ord("X"), ord("Y")):
            if tag != ord("Y"):
                self._disambiguator()
                self.path(False)
            text = "<" + self.type()
            if tag != ord("M"):
                text += " as " + self.path(False)
            return text + ">"
        if tag == ord("I"):
            text = self.path(in_value)
            if in_value:
                text += "::"
            args = self._sep_list("generic_arg", ", ")
            return text + "<" + ", ".join(args) + ">"
        if tag == ord("B"):
            return self._backref("path", in_value)
        raise DemangleError("bad path tag")

    def generic_arg(self) -> str:
        if self._eat(ord("L")):
            return self._lifetime(self._integer62())
        if self._eat(ord("K")):
            return self.const(False)
        return self.type()

    def type(self) -> str:
        splat = ""
        if self._eat(ord("w")):
            splat = "#[splat] "
        tag = self._next()
        basic = _BASIC.get(tag)
        if basic is not None:
            return splat + basic
        self._bump()
        if tag in (ord("R"), ord("Q")):
            text = "&"
            if self._eat(ord("L")):
                life = self._integer62()
                if life != 0:
                    text += self._lifetime(life) + " "
            if tag != ord("R"):
                text += "mut "
            return splat + text + self.type()
        if tag in (ord("P"), ord("O")):
            text = "*mut " if tag == ord("O") else "*const "
            return splat + text + self.type()
        if tag in (ord("A"), ord("S")):
            text = "[" + self.type()
            if tag == ord("A"):
                text += "; " + self.const(True)
            return splat + text + "]"
        if tag == ord("T"):
            items = self._sep_list("type", ", ")
            body = ", ".join(items)
            if len(items) == 1:
                body += ","
            return splat + "(" + body + ")"
        if tag == ord("F"):
            return splat + self._fn_sig()
        if tag == ord("D"):
            text = "dyn " + self._in_binder("_dyn_binder")
            if not self._eat(ord("L")):
                raise DemangleError("dyn missing lifetime")
            life = self._integer62()
            if life != 0:
                text += " + " + self._lifetime(life)
            return splat + text
        if tag == ord("B"):
            return splat + self._backref("type")
        if tag == ord("W"):
            return splat + self.type() + " is " + self.pat()
        self.pos -= 1
        return splat + self.path(False)

    def _fn_sig(self) -> str:
        count = self._opt_integer62(ord("G"))
        prefix = ""
        if count:
            parts = []
            for _ in range(count):
                self.bound_lifetimes += 1
                parts.append(self._lifetime(1))
            prefix = "for<" + ", ".join(parts) + "> "
        try:
            return prefix + self._fn_sig_body()
        finally:
            self.bound_lifetimes -= count

    def _fn_sig_body(self) -> str:
        unsafe = self._eat(ord("U"))
        abi = None
        if self._eat(ord("K")):
            if self._eat(ord("C")):
                abi = "C"
            else:
                abi = self._ident().replace("_", "-")
        text = ""
        if unsafe:
            text += "unsafe "
        if abi is not None:
            text += 'extern "' + abi + '" '
        args = self._sep_list("type", ", ")
        text += "fn(" + ", ".join(args) + ")"
        if self._eat(ord("u")):
            return text
        return text + " -> " + self.type()

    def _dyn_binder(self) -> str:
        return " + ".join(self._sep_list("dyn_trait", " + "))

    def dyn_trait(self) -> str:
        opened = False
        if self._eat(ord("B")):
            return self._backref("dyn_trait")
        if self._eat(ord("I")):
            text = self.path(False) + "<"
            args = self._sep_list("generic_arg", ", ")
            text += ", ".join(args)
            opened = True
        else:
            text = self.path(False)
        while self._eat(ord("p")):
            if not opened:
                text += "<"
                opened = True
            else:
                text += ", "
            text += self._ident() + " = "
            if self._eat(ord("K")):
                text += self.const(False)
            else:
                text += self.type()
        if opened:
            text += ">"
        return text

    def pat(self) -> str:
        tag = self._next()
        if tag == ord("R"):
            return self.const(False) + "..=" + self.const(False)
        if tag == ord("O"):
            self._bump()
            items = [self.pat()]
            while not self._eat(ord("E")):
                if self._peek() is None:
                    raise DemangleError("truncated pattern")
                items.append(self.pat())
            return " | ".join(items)
        if tag == ord("N"):
            return "!null"
        raise DemangleError("bad pattern")

    def const(self, in_value: bool) -> str:
        tag = self._next()
        self._bump()
        opened = False

        def brace(body: str) -> str:
            nonlocal opened
            if in_value:
                return body
            opened = True
            return "{" + body + "}"

        if tag == ord("p"):
            text = "_"
        elif tag in (ord("h"), ord("t"), ord("m"), ord("y"), ord("o"), ord("j")):
            text = str(self._hex_uint())
        elif tag in (ord("a"), ord("s"), ord("l"), ord("x"), ord("n"), ord("i")):
            sign = "-" if self._eat(ord("n")) else ""
            text = sign + str(self._hex_uint())
        elif tag == ord("b"):
            value = self._hex_uint()
            if value not in (0, 1):
                raise DemangleError("bad bool const")
            text = "true" if value else "false"
        elif tag == ord("c"):
            text = "'" + chr(self._hex_uint()) + "'"
        elif tag == ord("e"):
            text = brace("*" + self._str_lit())
        elif tag in (ord("R"), ord("Q")):
            if tag == ord("R") and self._eat(ord("e")):
                text = self._str_lit()
            else:
                body = "&"
                if tag != ord("R"):
                    body += "mut "
                text = brace(body + self.const(True))
        elif tag == ord("A"):
            items = self._sep_list("const_in_value", ", ")
            text = brace("[" + ", ".join(items) + "]")
        elif tag == ord("T"):
            items = self._sep_list("const_in_value", ", ")
            body = ", ".join(items)
            if len(items) == 1:
                body += ","
            text = brace("(" + body + ")")
        elif tag == ord("V"):
            text = brace(self._const_value())
        elif tag == ord("B"):
            text = self._backref("const", in_value)
        else:
            raise DemangleError("bad const")
        if opened and not (text.startswith("{") and text.endswith("}")):
            pass
        return text

    def const_in_value(self) -> str:
        return self.const(True)

    def _const_value(self) -> str:
        text = self.path(True)
        kind = self._next()
        if kind == ord("U"):
            return text
        if kind == ord("T"):
            items = self._sep_list("const_in_value", ", ")
            return text + "(" + ", ".join(items) + ")"
        if kind == ord("S"):
            fields = []
            while self._peek() is not None and not self._eat(ord("E")):
                self._disambiguator()
                name = self._ident()
                fields.append(name + ": " + self.const(True))
            return text + " { " + ", ".join(fields) + " }"
        raise DemangleError("bad const value")

    def _str_lit(self) -> str:
        raw = bytes.fromhex(self._hex_nibbles())
        return '"' + raw.decode("utf-8") + '"'


def _punycode_decode(ascii_part: str, payload: str) -> str | None:
    """Decode Rust's ``_``-separated punycode identifier."""
    try:
        joined = (ascii_part + "-" + payload) if payload else ascii_part
        return joined.encode("ascii").decode("punycode")
    except (UnicodeError, ValueError):
        return None


def _rust_v0_body(body: str) -> str | None:
    if not body or not ("A" <= body[0] <= "Z"):
        return None
    if any(ord(ch) > 127 for ch in body):
        return None
    parser = _Parser(body)
    try:
        text = parser.path(True)
        if parser._peek() is not None and "A" <= parser._peek() <= "Z":
            parser.path(True)
    except DemangleError:
        return None
    rest = parser.sym[parser.pos :]
    if rest and rest[0] not in ".$":
        return None
    return text


def demangle_rust_v0(name: str) -> str | None:
    """Return the v0 demangling, or None when ``name`` is not v0."""
    if name.startswith("__R") and len(name) > 3:
        body = name[3:]
    elif name.startswith("_R") and len(name) > 2:
        body = name[2:]
    elif name.startswith("R") and len(name) > 1:
        body = name[1:]
    else:
        return None
    return _rust_v0_body(body)


def demangle_rust_legacy(name: str) -> str | None:
    """Decode a legacy ``_ZN...E`` Rust symbol, dropping the hash."""
    text = name
    if text.startswith("__ZN"):
        text = text[1:]
    if not text.startswith("_ZN"):
        return None
    index = 3
    parts: list[str] = []
    while index < len(text) and text[index] != "E":
        if not text[index].isdigit():
            return None
        end = index
        while end < len(text) and text[end].isdigit():
            end += 1
        length = int(text[index:end])
        index = end
        if index + length > len(text):
            return None
        parts.append(text[index : index + length])
        index += length
    if not parts or index >= len(text) or text[index] != "E":
        return None
    if re.fullmatch(r"h[0-9a-f]{16}", parts[-1]):
        parts = parts[:-1]
    if not parts:
        return None
    return "::".join(parts)


def demangle_cxx(name: str) -> str | None:
    """Demangle an Itanium C++ name. Returns None if nothing changed."""
    if cxxfilt is None:
        return None
    try:
        text = cxxfilt.demangle(name)
    except Exception:
        return None
    if not isinstance(text, str) or text == name:
        return None
    stripped = _LEGACY_HASH.sub("", text)
    return stripped


def demangle(name: str) -> str:
    """Demangle ``name``. An unrecognised name is returned unchanged."""
    if not name:
        return name
    rust = demangle_rust_v0(name)
    if rust:
        return rust
    legacy = demangle_rust_legacy(name)
    if legacy:
        return legacy
    cxx = demangle_cxx(name)
    if cxx:
        return cxx
    return name
