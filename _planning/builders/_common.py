"""Helpers partilhados pelos builders dos notebooks 01 e 02 (nbformat)."""
import os
import nbformat as nbf

COLAB = "https://colab.research.google.com/github/the-ai-learning-hub/02-machine-learning-fundamentals/blob/main"
SLIDES = "https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/"

SETUP = '''# ⚙️ Preparar o ambiente: funciona localmente (VS Code/Jupyter) e no Google Colab
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    ROOT = Path("/content/02-machine-learning-fundamentals")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/the-ai-learning-hub/02-machine-learning-fundamentals.git",
                        str(ROOT)], check=True)
else:
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir() and (p / "data").is_dir())

try:  # o Colab já traz tudo; localmente usa `uv sync` (ver README)
    import sklearn, seaborn  # noqa: F401
except ImportError:  # plano B: instala as dependências declaradas no pyproject.toml
    import tomllib
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *deps], check=True)

sys.path.insert(0, str(ROOT))
print("Raiz do projeto:", ROOT.name, "(Colab)" if IN_COLAB else "(local)")'''


class NB:
    def __init__(self):
        self.cells, self.solutions = [], []

    def md(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def discussion(self, title, question, answer):
        self.md(f"""### 💬 Discussão — {title}
{question}

<details>
<summary>🧑‍🏫 Resposta-guia para o instrutor</summary>

{answer}
</details>""")

    def exercise(self, n, title, stars, intro, starter, solution, solution_text):
        self.md(f"### ✏️ Exercício {n} — {title}\n**Dificuldade:** {'⭐' * stars}\n\n{intro}")
        self.code(starter)
        self.md(f"""<details>
<summary>💡 Ver solução</summary>

```python
{solution.strip()}
```

{solution_text}
</details>""")
        self.solutions.append(solution)

    def header(self, title, subtitle, folder, nb_file, slide_id, objectives, duration, prereq, toc):
        self.md(f'''<div align="center" style="padding:40px 24px;background:linear-gradient(135deg,#0e1a2e 0%,#13223b 100%);border-radius:14px;border-bottom:5px solid #ffd43b;">
  <p style="margin:0;color:#6aaef0;font-family:monospace;font-weight:700;font-size:14px;letter-spacing:.16em;text-transform:uppercase;">The AI Learning Hub · Lição 02 · Machine Learning: fundamentos</p>
  <h1 style="margin:14px 0 6px;color:#f1f5fb;font-size:42px;">{title}</h1>
  <p style="margin:0;color:#b4c2d8;font-size:18px;">{subtitle}</p>
</div>''')
        self.md(f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/{folder}/notebooks/{nb_file}) "
                f"[![Slides](https://img.shields.io/badge/slides-Lição_02-ffd43b)]({SLIDES}#{slide_id})")
        obj = "\n".join(f"- {o}" for o in objectives)
        self.md(f"""## 🎯 Objetivos de aprendizagem
Ao terminares este notebook, vais conseguir:

{obj}

**⏱️ Duração estimada:** {duration} · **📋 Pré-requisitos:** {prereq} · **▶️ Como usar:** executa cada célula com `Shift`+`Enter`, por ordem.""")
        rows = "\n".join(f"{i}. [{t}](#sec-{i})" for i, t in enumerate(toc, 1))
        self.md(f"## 📑 Índice\n\n{rows}")
        self.md('<a id="sec-0"></a>\n## ⚙️ Preparar o ambiente')
        self.code(SETUP)

    def section(self, n, title):
        self.md(f'<a id="sec-{n}"></a>\n# {n}. {title}')

    def footer(self, prev, nxt):
        self.md(f"""---
<div align="center">

⬅️ {prev} &nbsp;·&nbsp; 🏠 [Lição 02](../../README.md) &nbsp;·&nbsp; {nxt} ➡️

**The AI Learning Hub** · Lição 02 — Machine Learning: fundamentos<br>
Diogo Freitas & João Francisco Botas · Licença MIT

</div>""")

    def save(self, path):
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.validate(nb)
        nbf.write(nb, path)
        if os.environ.get("VERIFY"):  # notebook de teste com as soluções como código
            v = nbf.v4.new_notebook()
            v.cells = list(self.cells) + [nbf.v4.new_code_cell(s) for s in self.solutions]
            v.metadata = nb.metadata
            nbf.write(v, os.environ["VERIFY"])
