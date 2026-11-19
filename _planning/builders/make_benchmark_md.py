"""Lê o BENCHMARK_JSON do NB04 executado e escreve _planning/benchmark.md."""
import json
import subprocess
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[2]
nb = nbformat.read(root / "03_neural_networks/notebooks/04_Neural_Networks_From_Scratch.ipynb", 4)
data = None
for cell in nb.cells:
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        if "BENCHMARK_JSON=" in text:
            data = json.loads(text.split("BENCHMARK_JSON=", 1)[1].strip().splitlines()[0])
assert data, "BENCHMARK_JSON não encontrado"


# Valores lidos com Get-CimInstance / nvidia-smi nesta máquina (portátil de teste)
cpu_model = "AMD Ryzen 7 7735HS with Radeon Graphics"
cores = "8 núcleos / 16 threads"
ram = "31 GB"
gpu_mem = "8188 MiB de VRAM, driver 621.14"
hw = data["hardware"]
rows = data["rows"]
has_gpu = "cuda_ms" in rows[0]


def gf(n, ms):
    return 2 * n**3 / (ms / 1000) / 1e9


lines = ["# Benchmark CPU vs GPU (multiplicação de matrizes float32, PyTorch)", "",
         "Gerado a partir do notebook `04_Neural_Networks_From_Scratch.ipynb` (parte 3) via `src.benchmark` "
         "(aquecimento, sincronização CUDA, mediana de 10 repetições para n ≤ 1024 e 5 para n ≥ 2048).", "",
         "## Hardware e versões", "",
         f"- **CPU:** {cpu_model} ({cores}); PyTorch usa as threads indicadas em `device_name('cpu')`: {hw['cpu']}",
         f"- **GPU:** {hw['gpu']} ({gpu_mem})", f"- **RAM:** {ram}",
         f"- **Software:** Python {hw['python']} · PyTorch {hw['torch']} · NumPy {hw['numpy']} · {hw['os']}", "",
         "## Tempo por multiplicação n×n (ms, mediana) e débito (GFLOP/s)", ""]
if has_gpu:
    lines += ["| n | CPU (ms) | GPU (ms) | CPU (GFLOP/s) | GPU (GFLOP/s) | speed-up GPU/CPU |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['cpu_ms']:.3f} | {r['cuda_ms']:.3f} | {gf(r['n'], r['cpu_ms']):.0f} | {gf(r['n'], r['cuda_ms']):.0f} | {r['cpu_ms'] / r['cuda_ms']:.1f}× |")
else:
    lines += ["| n | CPU (ms) | CPU (GFLOP/s) |", "|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['cpu_ms']:.3f} | {gf(r['n'], r['cpu_ms']):.0f} |")
if data["transfer_ms"]:
    lines += ["", "## Transferência CPU → GPU → CPU (matriz n×n float32, ms)", "", "| n | transferência (ms) | tamanho (MB) | multiplicação na GPU (ms) |", "|---:|---:|---:|---:|"]
    for r in rows:
        n = r["n"]
        lines.append(f"| {n} | {data['transfer_ms'][str(n)]:.3f} | {n * n * 4 / 1e6:.2f} | {r['cuda_ms']:.3f} |")
lines += ["", "Nota: as medições variam com a temperatura, o modo de energia e a carga da máquina; usar como ordem de grandeza.", "",
          "## JSON para o deck", "", "```json", json.dumps(data, ensure_ascii=False, indent=2), "```", ""]
(root / "_planning" / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
print("escrito _planning/benchmark.md")
