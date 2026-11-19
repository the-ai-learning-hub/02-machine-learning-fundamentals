# Benchmark CPU vs GPU (multiplicação de matrizes float32, PyTorch)

Gerado a partir do notebook `04_Neural_Networks_From_Scratch.ipynb` (parte 3) via `src.benchmark` (aquecimento, sincronização CUDA, mediana de 10 repetições para n ≤ 1024 e 5 para n ≥ 2048).

## Hardware e versões

- **CPU:** AMD Ryzen 7 7735HS with Radeon Graphics (8 núcleos / 16 threads); PyTorch usa as threads indicadas em `device_name('cpu')`: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD · 8 threads
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB de VRAM, driver 621.14)
- **RAM:** 31 GB
- **Software:** Python 3.12.15 · PyTorch 2.11.0+cu128 · NumPy 2.5.5 · Windows-11-10.0.26300-SP0

## Tempo por multiplicação n×n (ms, mediana) e débito (GFLOP/s)

| n | CPU (ms) | GPU (ms) | CPU (GFLOP/s) | GPU (GFLOP/s) | speed-up GPU/CPU |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.009 | 0.053 | 58 | 10 | 0.2× |
| 256 | 0.086 | 0.046 | 390 | 729 | 1.9× |
| 1024 | 5.441 | 0.338 | 395 | 6354 | 16.1× |
| 2048 | 38.038 | 2.170 | 452 | 7917 | 17.5× |
| 4096 | 271.875 | 19.238 | 506 | 7144 | 14.1× |

## Transferência CPU → GPU → CPU (matriz n×n float32, ms)

| n | transferência (ms) | tamanho (MB) | multiplicação na GPU (ms) |
|---:|---:|---:|---:|
| 64 | 0.066 | 0.02 | 0.053 |
| 256 | 0.125 | 0.26 | 0.046 |
| 1024 | 0.833 | 4.19 | 0.338 |
| 2048 | 2.983 | 16.78 | 2.170 |
| 4096 | 11.059 | 67.11 | 19.238 |

Nota: as medições variam com a temperatura, o modo de energia e a carga da máquina; usar como ordem de grandeza.

## JSON para o deck

```json
{
  "hardware": {
    "cpu": "AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD · 8 threads",
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "torch": "2.11.0+cu128",
    "numpy": "2.5.5",
    "python": "3.12.15",
    "os": "Windows-11-10.0.26300-SP0"
  },
  "rows": [
    {
      "n": 64,
      "cpu_ms": 0.009,
      "cuda_ms": 0.053
    },
    {
      "n": 256,
      "cpu_ms": 0.086,
      "cuda_ms": 0.046
    },
    {
      "n": 1024,
      "cpu_ms": 5.441,
      "cuda_ms": 0.338
    },
    {
      "n": 2048,
      "cpu_ms": 38.038,
      "cuda_ms": 2.17
    },
    {
      "n": 4096,
      "cpu_ms": 271.875,
      "cuda_ms": 19.238
    }
  ],
  "transfer_ms": {
    "64": 0.066,
    "256": 0.125,
    "1024": 0.833,
    "2048": 2.983,
    "4096": 11.059
  }
}
```
