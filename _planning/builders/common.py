"""Helpers partilhados pelos builders dos notebooks 03, 04 e 05 (modelo §7 do PLAN)."""
import nbformat as nbf

REPO = "the-ai-learning-hub/02-machine-learning-fundamentals"
PREV = {
    "03": ("02 · CRISP-DM: Titanic", "../../01_supervised_learning/notebooks/02_CRISP-DM_Titanic_Survival.ipynb"),
    "04": ("03 · Aprendizagem não supervisionada", "../../02_unsupervised_learning/notebooks/03_Unsupervised_Learning.ipynb"),
    "05": ("04 · Redes neuronais do zero", "../../03_neural_networks/notebooks/04_Neural_Networks_From_Scratch.ipynb"),
}
NEXT = {
    "03": ("04 · Redes neuronais do zero", "../../03_neural_networks/notebooks/04_Neural_Networks_From_Scratch.ipynb"),
    "04": ("05 · Projeto de Reinforcement Learning", "../../04_reinforcement_learning/notebooks/05_Reinforcement_Learning_Project.ipynb"),
    "05": None,
}

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
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    def head(self, num, title, subtitle, nbfile, slide_id, folder, objectives, duration, prereq, toc):
        self.md(f'''<div align="center" style="padding:40px 24px;background:linear-gradient(135deg,#0e1a2e 0%,#13223b 100%);border-radius:14px;border-bottom:5px solid #ffd43b;">
  <p style="margin:0;color:#6aaef0;font-family:monospace;font-weight:700;font-size:14px;letter-spacing:.16em;text-transform:uppercase;">The AI Learning Hub · Lição 02 · Machine Learning: fundamentos</p>
  <h1 style="margin:14px 0 6px;color:#f1f5fb;font-size:42px;">{num} · {title}</h1>
  <p style="margin:0;color:#b4c2d8;font-size:18px;">{subtitle}</p>
</div>''')
        colab = f"https://colab.research.google.com/github/{REPO}/blob/main/{folder}/notebooks/{nbfile}"
        self.md(f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) "
                f"[![Slides](https://img.shields.io/badge/slides-Lição_02-ffd43b)](https://the-ai-learning-hub.github.io/02-machine-learning-fundamentals/#{slide_id})")
        obj = "\n".join(f"- {o}" for o in objectives)
        self.md(f"## 🎯 Objetivos de aprendizagem\n\nNo fim deste notebook és capaz de:\n\n{obj}\n\n"
                f"**⏱️ Duração estimada:** {duration} · **📋 Pré-requisitos:** {prereq} · "
                f"**▶️ Como usar:** executa cada célula com `Shift`+`Enter`, por ordem.")
        t = "\n".join(f"{i}. [{s}](#sec-{i})" for i, s in enumerate(toc, 1))
        self.md(f"## 📑 Índice\n\n{t}")
        self.md('<a id="sec-0"></a>\n\n## ⚙️ Preparar o ambiente')
        self.code(SETUP)
        self.md("> 💡 **Dica:** no Colab não tens de instalar nada. Localmente, corre `uv sync` na raiz do projeto e escolhe o kernel `.venv` no Jupyter/VS Code.\n>\n> 📖 **Primeira vez?** Se algum termo (dataset, *feature*, alvo, modelo, treinar, prever, erro, *overfitting*) for novo, lê o [Vocabulário do notebook 01](../../01_supervised_learning/notebooks/01_CRISP-DM_House_Prices.ipynb#vocab). Cada notebook volta a explicar os termos novos na primeira vez que aparecem.")

    def sec(self, n, title):
        self.md(f'<a id="sec-{n}"></a>\n\n# {n}. {title}')

    def foot(self, key):
        prev = PREV[key]
        nxt = NEXT[key]
        left = f"⬅️ [Anterior: {prev[0]}]({prev[1]})"
        right = f" &nbsp;·&nbsp; [Seguinte: {nxt[0]}]({nxt[1]}) ➡️" if nxt else ""
        self.md(f'''---
<div align="center">

{left} &nbsp;·&nbsp; 🏠 [Lição 02](../../README.md){right}

**The AI Learning Hub** · Lição 02 — Machine Learning: fundamentos<br>
Diogo Freitas & João Francisco Botas · Licença MIT

</div>''')

    def save(self, path):
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.validate(nb)
        nbf.write(nb, path)
        print("escrito", path, len(self.cells), "células")


def disc(title, question, answer):
    return f"""### 💬 Discussão — {title}

{question}

<details>
<summary>🧑‍🏫 Resposta-guia para o instrutor</summary>

{answer}
</details>"""


def exercise(n, title, stars, statement, starter, solution_md, solution_code=None):
    sol = solution_md + (f"\n\n```python\n{solution_code}\n```" if solution_code else "")
    md = f"""### ✏️ Exercício {n} — {title}

**Dificuldade:** {'⭐' * stars}

{statement}"""
    det = f"""<details>
<summary>💡 Ver solução</summary>

{sol}
</details>"""
    return md, starter, det


def add_exercise(nb, n, title, stars, statement, starter, solution_md, solution_code=None):
    md, st, det = exercise(n, title, stars, statement, starter, solution_md, solution_code)
    nb.md(md)
    nb.code(st)
    nb.md(det)
