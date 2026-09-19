"""AMS-eligible word count: body + acknowledgments; exclude title/abstract/
significance, data availability, references, captions, tables, figures."""
from __future__ import annotations

import re
from pathlib import Path

TEX = Path(r"e:\Research Paper\Research-1\NEW WAY\Paper Writing\WAF\waf_manuscript.tex")
tex = TEX.read_text(encoding="utf-8")
start = tex.find(r"\section{Introduction}")
data = tex.find(r"\datastatement")
body = tex[start:data]
body = body.replace(r"\bob{}", "Bay of Bengal")
body = body.replace(r"\sece{}", "SECE")
body = body.replace(r"\era{}", "ERA5")
body = body.replace(r"\ibtracs{}", "IBTrACS")


def strip_env(text: str, env: str) -> str:
    begin = "\\begin{" + env + "}"
    end = "\\end{" + env + "}"
    out = []
    pos = 0
    while True:
        i = text.find(begin, pos)
        if i < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:i])
        j = text.find(end, i)
        if j < 0:
            break
        pos = j + len(end)
    return "".join(out)


for env in ("figure", "table"):
    body = strip_env(body, env)

body = re.sub(r"%.*", " ", body)
body = re.sub(r"\\label\{[^}]*\}", " ", body)
body = re.sub(r"\\cite[pt]?\{[^}]*\}", " ", body)
body = re.sub(r"\\ref\{[^}]*\}", " ", body)
body = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r" \2 ", body)
body = re.sub(r"\\(textbf|emph|textit|texttt)\{([^}]*)\}", r" \2 ", body)
body = re.sub(r"\\(sece|bob|era|ibtracs)\{\}", r" \1 ", body)
body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", body)
body = re.sub(r"[{}\\$~^_&]", " ", body)
body = re.sub(r"[0-9]*pt", " ", body)
body = re.sub(r"\s+", " ", body)
words = [w for w in body.split() if re.search(r"[A-Za-z]", w)]
print("AMS-eligible words (body + acknowledgments):", len(words))
print("head:", " ".join(words[:20]))
print("tail:", " ".join(words[-20:]))
