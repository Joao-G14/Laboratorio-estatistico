"""
preparacao.py — carga e tratamento do dataset.

Aqui o Pandas faz o que lhe cabe no projeto: ler o CSV, converter tipos,
derivar uma coluna e tratar ausentes. Nenhuma medida estatística é
calculada neste arquivo — isso é competência exclusiva de `minhastats.py`.

FONTE DOS DADOS
---------------
Registros de corridas de táxi da cidade de Nova York, coletados pela
New York City Taxi & Limousine Commission (TLC), órgão público que regula
o serviço e publica os "TLC Trip Record Data" mês a mês.

  * Fonte original (órgão):
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  * Arquivo efetivamente baixado (amostra pública do repositório
    seaborn-data, em CSV já consolidado):
    https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv

A amostra cobre corridas de março de 2019 (com algumas horas do dia 28/02),
nos táxis amarelos (Manhattan e aeroportos) e verdes (demais distritos).

DECISÕES DE TRATAMENTO (todas documentadas no RELATORIO.md)
-----------------------------------------------------------
1. `pickup`/`dropoff` viram datetime e originam `duracao_min`, a única
   coluna derivada do projeto.
2. Corridas com duração menor ou igual a zero são registros fisicamente
   impossíveis (erro de taxímetro) e são REMOVIDAS — são 6 linhas.
3. Ausentes nas colunas categóricas (até 0,7% das linhas) viram a categoria
   explícita "Não informado", em vez de descartar a linha inteira: as
   variáveis numéricas dessas corridas estão completas e são válidas.
4. `distance == 0` (45 linhas) e `passengers == 0` (96 linhas) são mantidas.
   Não são impossíveis — são prováveis falhas de digitação/registro — e a
   decisão de mantê-las está reportada nas limitações da análise.
"""

import os

import pandas as pd

CAMINHO_PADRAO = os.path.join("dados", "dataset.csv")

FONTE_ORIGINAL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
FONTE_ARQUIVO = ("https://raw.githubusercontent.com/mwaskom/seaborn-data/"
                 "master/taxis.csv")

ROTULO_AUSENTE = "Não informado"

# Dicionário de variáveis: nome no CSV -> (rótulo exibido, tipo, significado)
VARIAVEIS = {
    "distance": ("Distância (milhas)", "numérica contínua",
                 "Distância percorrida na corrida, medida pelo taxímetro."),
    "fare": ("Tarifa (US$)", "numérica contínua",
             "Valor da corrida pelo taxímetro, antes de gorjeta, pedágios "
             "e taxas."),
    "tip": ("Gorjeta (US$)", "numérica contínua",
            "Gorjeta registrada. Atenção: o taxímetro só registra gorjeta "
            "paga eletronicamente."),
    "tolls": ("Pedágios (US$)", "numérica contínua",
              "Total de pedágios repassados ao passageiro (pontes e túneis)."),
    "total": ("Total pago (US$)", "numérica contínua",
              "Soma de tarifa, gorjeta, pedágios e taxas obrigatórias."),
    "passengers": ("Passageiros", "numérica discreta",
                   "Número de passageiros informado pelo motorista."),
    "duracao_min": ("Duração (min)", "numérica contínua (derivada)",
                    "Minutos entre embarque e desembarque, calculada por "
                    "nós a partir de pickup e dropoff."),
    "color": ("Tipo de táxi", "categórica nominal",
              "yellow (licenciado para toda a cidade) ou green (fora do "
              "centro de Manhattan)."),
    "payment": ("Forma de pagamento", "categórica nominal",
                "credit card ou cash."),
    "pickup_borough": ("Distrito de embarque", "categórica nominal",
                       "Manhattan, Queens, Brooklyn, Bronx ou Staten Island."),
    "dropoff_borough": ("Distrito de desembarque", "categórica nominal",
                        "Distrito onde a corrida terminou."),
    "pickup_zone": ("Zona de embarque", "categórica nominal",
                    "Zona tarifária de origem (194 níveis)."),
    "dropoff_zone": ("Zona de desembarque", "categórica nominal",
                     "Zona tarifária de destino (203 níveis)."),
}

# Ordem em que as variáveis aparecem nos seletores da aplicação.
COLUNAS_NUMERICAS = ["distance", "fare", "tip", "tolls", "total",
                     "passengers", "duracao_min"]
COLUNAS_CATEGORICAS = ["color", "payment", "pickup_borough",
                       "dropoff_borough", "pickup_zone", "dropoff_zone"]

AEROPORTOS = ["JFK Airport", "LaGuardia Airport", "Newark Airport"]


def rotulo(coluna):
    """Nome amigável de uma coluna, para títulos de gráficos e tabelas."""
    return VARIAVEIS.get(coluna, (coluna,))[0]


def carregar(caminho=CAMINHO_PADRAO):
    """Lê o CSV e aplica as quatro decisões de tratamento documentadas acima.

    Devolve (df_tratado, relatorio_do_tratamento).
    """
    bruto = pd.read_csv(caminho, parse_dates=["pickup", "dropoff"])
    linhas_originais = len(bruto)

    df = bruto.copy()
    # 1. Coluna derivada: duração em minutos.
    df["duracao_min"] = (
        (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60.0
    )

    # 2. Remoção das corridas com duração impossível.
    impossiveis = int((df["duracao_min"] <= 0).sum())
    df = df[df["duracao_min"] > 0].copy()

    # 3. Categóricas ausentes viram categoria explícita.
    ausentes_por_coluna = {}
    for coluna in COLUNAS_CATEGORICAS:
        faltando = int(df[coluna].isna().sum())
        if faltando:
            ausentes_por_coluna[coluna] = faltando
        df[coluna] = df[coluna].fillna(ROTULO_AUSENTE)

    df = df.reset_index(drop=True)

    relatorio = {
        "linhas_originais": linhas_originais,
        "linhas_finais": len(df),
        "removidas_duracao_invalida": impossiveis,
        "ausentes_preenchidos": ausentes_por_coluna,
        "distancia_zero": int((df["distance"] == 0).sum()),
        "passageiros_zero": int((df["passengers"] == 0).sum()),
        "periodo_inicio": df["pickup"].min(),
        "periodo_fim": df["pickup"].max(),
    }
    return df, relatorio


def valores(df, coluna):
    """Coluna numérica como LISTA de floats — a fronteira da regra de ouro.

    Depois desta linha nenhum objeto do Pandas participa das contas: o que
    segue para `minhastats` é uma lista Python pura.
    """
    return df[coluna].dropna().astype(float).tolist()


def categorias(df, coluna):
    """Coluna categórica como lista de strings."""
    return df[coluna].astype(str).tolist()


def toca_aeroporto(df):
    """Máscara booleana: a corrida começou ou terminou em um aeroporto."""
    return (df["pickup_zone"].isin(AEROPORTOS)
            | df["dropoff_zone"].isin(AEROPORTOS))
