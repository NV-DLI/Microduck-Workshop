"""Small, explicit scanners for JavaScript, JSON, CSS and SVG references.

These scanners are intentionally limited. The JavaScript lexer recognises
comments, string/template/regular-expression literals and enough punctuation to
find:

* static ``import ... from "x"``, ``import "x"`` and ``export ... from "x"``;
* ``import("x")`` with a single literal argument;
* ``new URL("x", import.meta.url)``;
* any other plain string literal (candidate resource path).

They do not evaluate JavaScript: computed paths such as ``"./a/" + name`` are
invisible and must be declared in the export manifest.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

ASSET_EXTENSIONS = frozenset(
    "js mjs cjs json geojson gltf glb bin wasm css html htm svg png jpg jpeg gif webp avif "
    "ico bmp mp4 webm ogv ogg mov m4v mp3 wav m4a flac vtt srt woff woff2 ttf otf eot pdf zip "
    "txt csv tsv md py ipynb yaml yml xml usd usda usdz hdr exr ktx2 basis".split()
)

_REGEX_PRECEDERS = frozenset(
    "return typeof instanceof in of new delete void throw case do else yield await".split()
)

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<lc>//[^\n\r]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<str>"(?:[^"\\\n\r]|\\(?:\r\n|.))*"|'(?:[^'\\\n\r]|\\(?:\r\n|.))*')
  | (?P<num>0[xXoObB][0-9a-fA-F_]+n?|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?n?)
  | (?P<name>[A-Za-z_$\u0080-￿][\w$\u0080-￿]*)
  | (?P<punct>\?\.|\.\.\.|=>|[{}()\[\];,<>+\-*%&|^!~?:=.@#\\])
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass
class Token:
    type: str  # name, num, punct, str, tmpl, regex
    value: str  # decoded value for str/tmpl; raw text otherwise
    start: int
    end: int
    has_expr: bool = False


class LexError(ValueError):
    pass


def _decode(body: str) -> str:
    if "\\" not in body:
        return body
    out: list[str] = []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(body):
            break
        ch = body[i]
        if ch in simple:
            out.append(simple[ch])
            i += 1
        elif ch == "x" and i + 2 < len(body) + 1:
            try:
                out.append(chr(int(body[i + 1 : i + 3], 16)))
                i += 3
            except ValueError:
                out.append("x")
                i += 1
        elif ch == "u":
            if body[i + 1 : i + 2] == "{":
                end = body.find("}", i)
                try:
                    out.append(chr(int(body[i + 2 : end], 16)))
                    i = end + 1
                except ValueError:
                    out.append("u")
                    i += 1
            else:
                try:
                    out.append(chr(int(body[i + 1 : i + 5], 16)))
                    i += 5
                except ValueError:
                    out.append("u")
                    i += 1
        elif ch in "\r\n":
            i += 2 if body[i : i + 2] == "\r\n" else 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def tokenize(text: str) -> list[Token]:
    """Return significant tokens (comments and whitespace dropped)."""
    tokens: list[Token] = []
    stack: list[tuple[str, Token | None]] = []  # ("{", None) or ("${", template token)
    i = 0
    n = len(text)

    def regex_allowed() -> bool:
        if not tokens:
            return True
        last = tokens[-1]
        if last.type in ("num", "str", "tmpl", "regex"):
            return False
        if last.type == "name":
            return last.value in _REGEX_PRECEDERS
        return last.value not in (")", "]", "}")

    def scan_template(pos: int) -> tuple[int, bool]:
        """Scan template characters; return (index after ` or ${, closed)."""
        j = pos
        while j < n:
            c = text[j]
            if c == "\\":
                j += 2
                continue
            if c == "`":
                return j + 1, True
            if c == "$" and text[j + 1 : j + 2] == "{":
                return j + 2, False
            j += 1
        raise LexError(f"unterminated template literal at {pos}")

    def continue_template(tmpl: Token, pos: int) -> int:
        j, closed = scan_template(pos)
        tmpl.end = j
        if closed:
            if not tmpl.has_expr:
                tmpl.value = _decode(text[tmpl.start + 1 : j - 1])
        else:
            tmpl.has_expr = True
            stack.append(("${", tmpl))
        return j

    while i < n:
        c = text[i]
        if c == "`":
            tmpl = Token("tmpl", "", i, i)
            tokens.append(tmpl)
            i = continue_template(tmpl, i + 1)
            continue
        if c == "}" and stack and stack[-1][0] == "${":
            _, tmpl = stack.pop()
            i = continue_template(tmpl, i + 1)
            continue
        if c == "/":
            if text[i + 1 : i + 2] in ("/", "*"):
                m = _TOKEN_RE.match(text, i)
                if not m or m.lastgroup not in ("lc", "bc"):
                    raise LexError(f"unterminated comment at {i}")
                i = m.end()
                continue
            if regex_allowed():
                j = i + 1
                in_class = False
                while j < n:
                    d = text[j]
                    if d == "\\":
                        j += 2
                        continue
                    if d in "\n\r":
                        raise LexError(f"unterminated regex at {i}")
                    if d == "[":
                        in_class = True
                    elif d == "]":
                        in_class = False
                    elif d == "/" and not in_class:
                        break
                    j += 1
                j += 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                tokens.append(Token("regex", text[i:j], i, j))
                i = j
                continue
            tokens.append(Token("punct", "/", i, i + 1))
            i += 1
            continue
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise LexError(f"unexpected character {c!r} at {i}")
        kind = m.lastgroup
        if kind in ("ws", "lc", "bc"):
            i = m.end()
            continue
        raw = m.group(0)
        if kind == "str":
            tokens.append(Token("str", _decode(raw[1:-1]), i, m.end()))
        else:
            if raw == "{" and kind == "punct":
                stack.append(("{", None))
            elif raw == "}" and kind == "punct" and stack:
                stack.pop()
            tokens.append(Token(kind, raw, i, m.end()))
        i = m.end()
    return tokens


def lex(text: str) -> list[Token]:
    return tokenize(text)


@dataclass
class JsRef:
    start: int
    end: int
    value: str
    kind: str  # import | module-url | literal | object-key
    quote: str


# Calls whose first argument is a URL, and properties that take one, for dynamic-site reports.
RESOURCE_CALLS = frozenset(
    "import fetch URL Worker SharedWorker importScripts load loadAsync Audio EventSource "
    "open sendBeacon setAttribute".split()
)
RESOURCE_PROPERTIES = frozenset("src href poster data srcset".split())


def _is(tok: Token | None, type_: str, value: str | None = None) -> bool:
    return tok is not None and tok.type == type_ and (value is None or tok.value == value)


def js_references(text: str) -> tuple[list[JsRef], list[str]]:
    """Classify string literals of a JavaScript (or JSON) source."""
    toks = lex(text)
    refs: list[JsRef] = []
    dynamic: list[str] = []

    def at(k: int) -> Token | None:
        return toks[k] if 0 <= k < len(toks) else None

    def resource_position(k: int) -> bool:
        """True when token k is the URL argument of a loader call or a URL property assignment."""
        prev, prev2 = at(k - 1), at(k - 2)
        if _is(prev, "punct", "(") and prev2 is not None and prev2.type == "name":
            return prev2.value in RESOURCE_CALLS
        if _is(prev, "punct", "=") and prev2 is not None and prev2.type == "name":
            return prev2.value in RESOURCE_PROPERTIES
        return False

    for k, tok in enumerate(toks):
        if tok.type == "tmpl" and tok.has_expr:
            # Report only computed URLs in loader positions, e.g. fetch(`${base}/x.json`).
            # Display strings such as `three.js r${REVISION}` are not resources.
            if resource_position(k):
                dynamic.append(text[tok.start : tok.end][:160])
            continue
        if tok.type not in ("str", "tmpl"):
            continue
        if resource_position(k) and _is(at(k + 1), "punct", "+"):
            # "./models/" + name: the exporter cannot know which files this selects.
            dynamic.append(text[tok.start : at(k + 1).end][:160] + " ...")
        prev, prev2 = at(k - 1), at(k - 2)
        nxt = at(k + 1)
        quote = text[tok.start]
        kind = "literal"
        if _is(prev, "name", "from"):
            kind = "import"
        elif _is(prev, "name", "import") and not _is(prev2, "punct", "."):
            kind = "import"
        elif (
            _is(prev, "punct", "(")
            and _is(prev2, "name", "import")
            and not _is(at(k - 3), "punct", ".")
            and nxt is not None
            and nxt.type == "punct"
            and nxt.value in (")", ",")
        ):
            kind = "import"
        elif (
            _is(prev, "punct", "(")
            and _is(prev2, "name", "URL")
            and _is(at(k - 3), "name", "new")
            and _is(nxt, "punct", ",")
            and [t.value for t in toks[k + 2 : k + 7]] == ["import", ".", "meta", ".", "url"]
        ):
            kind = "module-url"
        elif _is(nxt, "punct", ":") and prev is not None and prev.value in ("{", ","):
            kind = "object-key"
        refs.append(JsRef(tok.start, tok.end, tok.value, kind, quote))
    return refs, dynamic


def split_url(value: str) -> tuple[str, str]:
    """Split a URL-ish value into (path, suffix) where suffix holds ?query#fragment."""
    cut = len(value)
    for ch in "?#":
        pos = value.find(ch)
        if pos != -1:
            cut = min(cut, pos)
    return value[:cut], value[cut:]


def is_external(value: str) -> bool:
    v = value.strip()
    if not v or v.startswith("#") or v.startswith("//"):
        return True
    return bool(urlsplit(v).scheme) and not re.match(r"^[A-Za-z]:[\\/]", v)


def looks_like_path(value: str) -> bool:
    """Candidate local resource: no whitespace, has a file extension."""
    if not value or len(value) > 1024 or is_external(value):
        return False
    if re.search(r"[\s<>{}|^`\"'\\]", value):
        return False
    path, _ = split_url(value)
    ext = posixpath.splitext(posixpath.basename(path))[1][1:]
    return bool(ext) and ext.isalnum() and len(ext) <= 8


def has_asset_extension(value: str) -> bool:
    path, _ = split_url(value)
    return posixpath.splitext(path)[1][1:].lower() in ASSET_EXTENSIONS


# --------------------------------------------------------------------------- CSS

_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_URL_RE = re.compile(
    r"""url\(\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^)\s'"]*))\s*\)""", re.IGNORECASE
)
_CSS_IMPORT_RE = re.compile(r"""@import\s+(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)')""", re.IGNORECASE)
_FONT_FACE_RE = re.compile(r"@font-face\s*\{[^{}]*\}", re.IGNORECASE)
_SRC_DECL_RE = re.compile(r"(\bsrc\s*:\s*)([^;}]*)", re.IGNORECASE)


@dataclass
class CssRef:
    start: int  # span of the whole url(...) or the quoted @import string
    end: int
    value: str
    kind: str  # url | import


def _split_top_level(value: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def prune_font_formats(css: str) -> str:
    """Inside @font-face, keep only woff2 sources when a woff2 source exists."""

    def fix_block(block: re.Match[str]) -> str:
        def fix_src(decl: re.Match[str]) -> str:
            entries = _split_top_level(decl.group(2))
            woff2 = [
                e for e in entries
                if re.search(r"format\(\s*['\"]?woff2", e, re.I)
                or re.search(r"\.woff2(?:[?#][^'\")]*)?['\"]?\s*\)", e, re.I)
            ]
            locals_ = [e for e in entries if re.match(r"\s*local\(", e, re.I)]
            if not woff2 or len(woff2) == len(entries):
                return decl.group(0)
            return decl.group(1) + ",".join(locals_ + woff2).strip()

        return _SRC_DECL_RE.sub(fix_src, block.group(0))

    return _FONT_FACE_RE.sub(fix_block, css)


def css_references(css: str) -> list[CssRef]:
    comments = [(m.start(), m.end()) for m in _CSS_COMMENT_RE.finditer(css)]

    def in_comment(pos: int) -> bool:
        return any(a <= pos < b for a, b in comments)

    refs: list[CssRef] = []
    for m in _CSS_URL_RE.finditer(css):
        if in_comment(m.start()):
            continue
        value = next(v for v in (m.group("dq"), m.group("sq"), m.group("bare")) if v is not None)
        refs.append(CssRef(m.start(), m.end(), value.strip(), "url"))
    for m in _CSS_IMPORT_RE.finditer(css):
        if in_comment(m.start()):
            continue
        value = m.group("dq") if m.group("dq") is not None else m.group("sq")
        refs.append(CssRef(m.start(1) - 1 if m.group("dq") is not None else m.start(2) - 1,
                           m.end(1) + 1 if m.group("dq") is not None else m.end(2) + 1,
                           value.strip(), "import"))
    refs.sort(key=lambda r: r.start)
    return refs


_SOURCEMAP_RE = re.compile(r"(?m)^[ \t]*(?://[#@]\s*sourceMappingURL=\S*|/\*[#@]\s*sourceMappingURL=[^*]*\*/)[ \t]*$")


def strip_source_maps(text: str) -> str:
    return _SOURCEMAP_RE.sub("", text)


# --------------------------------------------------------------------------- SVG

_SVG_HREF_RE = re.compile(r"""\b(?:xlink:)?href\s*=\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)')""")


def svg_references(svg: str) -> list[CssRef]:
    refs = []
    for m in _SVG_HREF_RE.finditer(svg):
        group = "dq" if m.group("dq") is not None else "sq"
        refs.append(CssRef(m.start(group), m.end(group), m.group(group), "href"))
    refs.extend(css_references(svg))
    refs.sort(key=lambda r: r.start)
    return refs
