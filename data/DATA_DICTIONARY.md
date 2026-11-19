# 📚 Dicionário de dados · Lição 02

Esta lição usa **dois datasets reais** e vários **datasets sintéticos ou embutidos** no scikit-learn.

| Dataset | Notebooks | Linhas × colunas | Está no repo? | Como se carrega |
|---|---|---|---|---|
| Preços de casas em Windsor (1987) | 01, 03 (opcional) | 546 × 12 | ❌ descarregado da fonte oficial | `src.data.load_housing()` |
| Titanic (`titanic3`) | 02 | 1309 × 14 | ❌ descarregado da fonte oficial | `src.data.load_titanic()` |
| `make_blobs`, `make_moons` | 03, 04 | sintéticos | — (gerados no código) | `sklearn.datasets` |
| `load_digits` (dígitos 8×8) | 03, 04 | 1797 × 64 | — (vem com o scikit-learn) | `sklearn.datasets.load_digits()` |
| Gridworld | 05 | 5 × 6 células | — (definido no código) | `src.rl_utils.GridWorld()` |

---

## 1. Preços de casas em Windsor, Canadá

- **O que é:** preços de venda de 546 casas na cidade de Windsor (Ontário, Canadá), vendidas entre julho e setembro de **1987**.
- **Fonte original:** Anglin, P. M. & Gençay, R. (1996). *Semiparametric estimation of a hedonic price function*. *Journal of Applied Econometrics*, 11(6), 633–648. Dados publicados no *JAE Data Archive*.
- **Fonte oficial (de onde descarregamos):** [*Journal of Applied Econometrics* Data Archive](http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/) → ficheiro `ag-data.zip` (texto, 12 colunas sem nome; ordem descrita no `readme.ag.txt`: `sell lot bdms fb sty drv rec ffin ghw ca gar reg`). O servidor só funciona por `http://` (o certificado `https` está mal configurado). Os autores recolheram os dados do *Multiple Listing Service* de Windsor.
- **Como se carrega:** `load_housing()` descarrega o zip para a memória (nada é gravado no disco), dá às colunas os nomes usados no pacote R AER (`price`, `lotsize`, …) e converte 1/0 em `yes`/`no`. Se o arquivo estiver em baixo, usa a cópia do mesmo dataset no pacote R **AER** ([Rdatasets](https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html)).
- **Licença:** os dados não são redistribuídos neste repositório; são descarregados da fonte oficial. Cita Anglin & Gençay (1996) se os usares.

| Coluna | Tipo | Descrição | Valores |
|---|---|---|---|
| `price` | numérica (alvo) | Preço de venda, **dólares canadianos de 1987 (CAD)** | 25 000 – 190 000 (mediana 62 000) |
| `lotsize` | numérica | Área do **lote** em pés quadrados (1 pé² ≈ 0,093 m²) | 1 650 – 16 200 |
| `bedrooms` | discreta | N.º de quartos | 1 – 6 |
| `bathrooms` | discreta | N.º de casas de banho completas | 1 – 4 |
| `stories` | discreta | N.º de pisos (sem a cave) | 1 – 4 |
| `driveway` | binária | Tem entrada para carro? | yes / no |
| `recreation` | binária | Tem sala de lazer? | yes / no |
| `fullbase` | binária | Tem cave completa e acabada? | yes / no |
| `gasheat` | binária | Usa gás para aquecer a água? | yes / no |
| `aircon` | binária | Tem ar condicionado central? | yes / no |
| `garage` | discreta | N.º de lugares de garagem | 0 – 3 |
| `prefer` | binária | Fica no bairro preferido da cidade? | yes / no |

Sem valores em falta. Há 1 linha repetida (todas as colunas iguais); mantemo-la, porque duas casas podem ter exatamente as mesmas características. O preço tem assimetria positiva (1,21), por isso os modelos lineares usam `log(price)`.

> ⚠️ **Porque não usamos o `Housing.csv` do Kaggle?** É o mesmo dataset com os preços multiplicados por 70, colunas renomeadas, uma linha a menos e uma coluna `furnishingstatus` que **não existe** nos dados originais. Como não tem licença nem proveniência documentadas, preferimos a fonte original.

## 2. Titanic (`titanic3`)

- **O que é:** 1309 passageiros do RMS Titanic (abril de 1912), com a indicação de quem sobreviveu (38,2 %).
- **Fonte:** Frank E. Harrell Jr. & Thomas Cason, Vanderbilt University, Department of Biostatistics — [hbiostat.org/data](https://hbiostat.org/data/) (`titanic3.csv`), compilado a partir da *Encyclopedia Titanica*. Também em [OpenML (id 40945)](https://www.openml.org/d/40945).
- **Licença:** a página de origem disponibiliza o ficheiro para uso público, mas **não indica uma licença formal**. Por isso o ficheiro **não é incluído** no repositório: `load_titanic()` descarrega-o de `hbiostat.org` (ou, em alternativa, do OpenML) diretamente para a memória (nada é gravado no disco). Cita os autores se o usares.

| Coluna | Tipo | Descrição | Em falta |
|---|---|---|---|
| `pclass` | ordinal | Classe do bilhete (1 = 1.ª, 2 = 2.ª, 3 = 3.ª) | 0 |
| `survived` | binária (alvo) | 1 = sobreviveu, 0 = morreu | 0 |
| `name` | texto | Nome, com título (`Mr.`, `Mrs.`, `Miss.`, `Master.`…) | 0 |
| `sex` | categórica | `male` / `female` | 0 |
| `age` | numérica | Idade em anos (frações < 1 ano) | 263 |
| `sibsp` | discreta | N.º de irmãos/cônjuges a bordo | 0 |
| `parch` | discreta | N.º de pais/filhos a bordo | 0 |
| `ticket` | texto | Número do bilhete | 0 |
| `fare` | numérica | Preço do bilhete (libras de 1912) | 1 |
| `cabin` | texto | Cabine (a letra é o deck) | 1014 |
| `embarked` | categórica | Porto: C = Cherbourg, Q = Queenstown, S = Southampton | 2 |
| `boat` | texto | **Bote salva-vidas** ⚠️ *leakage* — só se sabe depois do naufrágio | 823 |
| `body` | numérica | **N.º de identificação do corpo** ⚠️ *leakage* | 1188 |
| `home.dest` | texto | Origem / destino | 564 |

Features criadas no notebook 02 (`src.preprocessing.add_titanic_features`): `title`, `family_size = sibsp + parch + 1`, `is_alone`, `deck` (letra da cabine ou `U` = desconhecido).

## 3. Datasets sintéticos e embutidos

- `make_blobs`, `make_moons` (scikit-learn): gerados no código com `random_state` fixo — ideais para ver o que cada algoritmo faz, porque sabemos a resposta certa.
- `load_digits` (scikit-learn, BSD-3): 1797 imagens 8×8 de dígitos manuscritos (cópia do *UCI Optical Recognition of Handwritten Digits*), usada para PCA e para a rede em PyTorch.
- Gridworld (notebook 05 e slides): mundo 5×6 definido em `src/rl_utils.py`.
