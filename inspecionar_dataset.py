"""
inspecionar_dataset.py — a inspeção que valida o dataset antes do trabalho.

Roda o checklist da Etapa 1 do guia sobre o arquivo em `dados/dataset.csv`
e diz, item a item, se o conjunto atende aos critérios da atividade
(≥ 1.000 registros, ≥ 4 variáveis numéricas, ≥ 2 categóricas, poucos
ausentes, variação real e categóricas com cardinalidade utilizável).

Uso:  python inspecionar_dataset.py [caminho/para/arquivo.csv]
"""

import sys

import pandas as pd

import preparacao

LIMITE_AUSENTES = 0.20          # acima de 20% de nulos a coluna é suspeita
LIMITE_CARDINALIDADE = 50       # categórica com centenas de níveis é ID


def marcar(condicao):
    return "OK   " if condicao else "FALHA"


def main(caminho=preparacao.CAMINHO_PADRAO):
    bruto = pd.read_csv(caminho)
    print(f"Arquivo: {caminho}")
    print(f"Dimensões: {bruto.shape[0]} linhas × {bruto.shape[1]} colunas\n")

    numericas = bruto.select_dtypes("number").columns.tolist()
    categoricas = [c for c in bruto.columns
                   if c not in numericas and not c.startswith(("pickup",
                                                               "dropoff"))
                   or c in ("pickup_zone", "dropoff_zone", "pickup_borough",
                            "dropoff_borough")]

    print("== Critérios da atividade ==")
    print(f"[{marcar(len(bruto) >= 1000)}] registros ≥ 1.000        "
          f"-> {len(bruto)}")
    print(f"[{marcar(len(numericas) >= 4)}] variáveis numéricas ≥ 4  "
          f"-> {len(numericas)}: {numericas}")
    print(f"[{marcar(len(categoricas) >= 2)}] variáveis categóricas ≥ 2 "
          f"-> {len(categoricas)}: {categoricas}")

    print("\n== Tipos ==")
    print(bruto.dtypes.to_string())

    print("\n== Fração de valores ausentes por coluna ==")
    ausentes = bruto.isna().mean().sort_values(ascending=False)
    for coluna, fracao in ausentes.items():
        alerta = "  <-- acima do limite" if fracao > LIMITE_AUSENTES else ""
        print(f"  {coluna:<18} {fracao:6.2%}{alerta}")

    print("\n== Há variação real? (colunas constantes são inúteis) ==")
    for coluna in numericas:
        unicos = bruto[coluna].nunique()
        print(f"  [{marcar(unicos > 1)}] {coluna:<16} {unicos} valores "
              f"distintos, amplitude "
              f"{bruto[coluna].max() - bruto[coluna].min():.2f}")

    print("\n== Cardinalidade das categóricas ==")
    for coluna in categoricas:
        unicos = bruto[coluna].nunique()
        nota = ("  (alta demais para gráfico de barras direto)"
                if unicos > LIMITE_CARDINALIDADE else "")
        print(f"  {coluna:<18} {unicos} níveis{nota}")

    print("\n== Resumo numérico ==")
    print(bruto.describe().round(3).to_string())

    print("\n== Depois do tratamento de preparacao.carregar() ==")
    tratado, relatorio = preparacao.carregar(caminho)
    for chave, valor in relatorio.items():
        print(f"  {chave:<28} {valor}")
    print(f"  colunas finais              {list(tratado.columns)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else preparacao.CAMINHO_PADRAO)
