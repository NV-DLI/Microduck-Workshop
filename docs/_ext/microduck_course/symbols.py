"""Explained math symbols.

Authors define a page's symbols once with the ``{symbols}`` directive (YAML:
``key: {tex: ..., text: ...}``) and mark every occurrence in TeX with
``\\sym{key}`` or, when the occurrence is written differently,
``\\symx{key}{custom TeX}``. At build time each mark becomes
``\\htmlData{sym=key}{TeX}``; the browser shows the explanation on hover,
focus or tap. Unknown keys and keys that are defined but never used are build
errors, so incomplete annotation is caught when the site is built.
"""

from __future__ import annotations

import json
import re
from html import escape as e

KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_MARK = re.compile(r"\\sym(x?)\{([^{}]*)\}")


class SymbolError(ValueError):
    pass


def _braced(text: str, start: int) -> tuple[str, int]:
    """Return the balanced ``{...}`` group starting at ``start`` and the index after it."""
    if start >= len(text) or text[start] != "{":
        raise SymbolError("\\symx needs {key}{TeX}")
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise SymbolError("unbalanced braces after \\symx")


def resolve(tex: str, table: dict, used: list) -> str:
    """Replace \\sym/\\symx marks with \\htmlData; record keys in order of use."""
    out = []
    pos = 0
    while True:
        m = _MARK.search(tex, pos)
        if not m:
            out.append(tex[pos:])
            return "".join(out)
        out.append(tex[pos:m.start()])
        key = m.group(2).strip()
        if key not in table:
            raise SymbolError(f"unknown symbol key {key!r}; define it in this page's {{symbols}} block")
        if m.group(1):
            body, pos = _braced(tex, m.end())
        else:
            body, pos = table[key]["tex"], m.end()
        used.append(key)
        out.append(f"\\htmlData{{sym={key}}}{{{resolve(body, table, used)}}}")


def parse_table(data) -> dict:
    if not isinstance(data, dict) or not data:
        raise SymbolError("{symbols} expects a YAML mapping of key: {tex, text}")
    table = {}
    for key, value in data.items():
        if not KEY.match(str(key)):
            raise SymbolError(f"invalid symbol key {key!r}")
        if not isinstance(value, dict) or not value.get("tex") or not value.get("text"):
            raise SymbolError(f"symbol {key!r} needs tex and text")
        table[str(key)] = {"tex": str(value["tex"]), "text": str(value["text"])}
    return table


def json_script(table: dict) -> str:
    payload = json.dumps(table, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" class="course-symbols">{payload}</script>'


def text_html(text: str) -> str:
    """Explanation text; `$TeX$` segments render as inline math (no \\sym marks)."""
    parts = re.split(r"\$([^$]+)\$", text)
    return "".join(e(p) if i % 2 == 0 else "\\(" + e(p) + "\\)" for i, p in enumerate(parts))


def legend(keys: list, table: dict, lang: str) -> str:
    """Collapsed list of the symbols in one display formula (mobile and screen readers)."""
    seen = list(dict.fromkeys(keys))
    if not seen:
        return ""
    title = "变量含义" if lang == "zh" else "Variable meanings"
    items = "".join(
        f'<div><dt>\\({e(table[k]["tex"])}\\)</dt><dd>{text_html(table[k]["text"])}</dd></div>' for k in seen
    )
    return f'<details class="formula-symbols"><summary>{title}</summary><dl>{items}</dl></details>'
