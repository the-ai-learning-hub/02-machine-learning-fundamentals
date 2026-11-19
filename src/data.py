"""Load the two real datasets used in the lesson, straight from their official sources.

* **Windsor house prices** (Anglin & Gençay, 1996): 546 houses sold in Windsor, Canada, in 1987.
  Official source: *Journal of Applied Econometrics* Data Archive (``ag-data.zip``).
* **Titanic** (``titanic3``, Harrell & Cason, Vanderbilt Biostatistics): 1309 passengers.
  Official source: Vanderbilt Department of Biostatistics (``hbiostat.org``).

Nothing is written to disk: each call downloads the original file into memory and parses it.
Both files are small (4 KB and 115 KB). See ``data/DATA_DICTIONARY.md``.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from functools import lru_cache

import pandas as pd

#: Official JAE archive page: http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/
#: (the server only works over plain http; its https certificate is invalid).
HOUSING_URL = "http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/ag-data.zip"
#: Same data as distributed in the R package AER (``HousePrices``), used only if the archive is down.
HOUSING_MIRROR = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv"

TITANIC_URL = "https://hbiostat.org/data/repo/titanic3.csv"

#: Column names (as in the R package AER) for the 12 unnamed columns of the JAE file, in order.
#: JAE readme order: sell lot bdms fb sty drv rec ffin ghw ca gar reg.
HOUSING_COLUMNS = ["price", "lotsize", "bedrooms", "bathrooms", "stories", "driveway",
                   "recreation", "fullbase", "gasheat", "aircon", "garage", "prefer"]
HOUSING_BINARY = ["driveway", "recreation", "fullbase", "gasheat", "aircon", "prefer"]
HOUSING_NUMERIC = ["lotsize", "bedrooms", "bathrooms", "stories", "garage"]


@lru_cache(maxsize=None)
def _download(url: str) -> bytes:
    """Download ``url`` into memory (cached for the rest of the Python session)."""
    print(f"A descarregar da fonte oficial: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ai-learning-hub-lesson"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _parse_jae(raw: bytes) -> pd.DataFrame:
    """Read the whitespace-separated ``ag-data.fil`` inside the JAE zip (in memory)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".fil"))
        text = archive.read(name).decode("ascii")
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=HOUSING_COLUMNS).astype(int)
    for col in HOUSING_BINARY:  # 1/0 → "yes"/"no", as in the R package AER
        df[col] = df[col].map({1: "yes", 0: "no"})
    return df


def load_housing() -> pd.DataFrame:
    """Windsor house prices (546 rows × 12 columns). ``price`` is in Canadian dollars of 1987 (CAD).

    Binary columns are ``"yes"``/``"no"`` text so the notebooks can show the encoding step.
    """
    try:
        return _parse_jae(_download(HOUSING_URL))
    except (OSError, zipfile.BadZipFile, StopIteration):
        print("O arquivo do JAE não respondeu; a usar a cópia do mesmo dataset no pacote R AER.")
        df = pd.read_csv(io.BytesIO(_download(HOUSING_MIRROR)))
        return df.drop(columns=["rownames"], errors="ignore")[HOUSING_COLUMNS]


def load_titanic() -> pd.DataFrame:
    """Titanic passengers (``titanic3``, 1309 rows, 14 columns).

    Downloads from Vanderbilt (hbiostat.org); if that fails, from OpenML (data id 40945).
    Columns: pclass, survived, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked,
    boat, body, home.dest. **boat** and **body** are only known *after* the sinking (leakage!).
    """
    try:
        return pd.read_csv(io.BytesIO(_download(TITANIC_URL)))
    except OSError:
        from sklearn.datasets import fetch_openml

        print("A Vanderbilt não respondeu; a usar o OpenML (data id 40945) …")
        df = fetch_openml(data_id=40945, as_frame=True).frame
        df["survived"] = df["survived"].astype(int)
        return df
