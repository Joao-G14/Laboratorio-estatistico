"""
gerar_tabela_validacao.py — produz a tabela de validação do relatório.

Roda cada função de `minhastats.py` e a sua referência em NumPy/SciPy
sobre uma variável real do dataset, e imprime a diferença absoluta e a
tolerância declarada, em formato Markdown, pronto para o RELATORIO.md.

Este é o único script do projeto — junto com os testes — em que NumPy e
SciPy calculam estatística.

Uso:  python gerar_tabela_validacao.py
"""

import json
import os

import numpy as np
from scipy import stats

import minhastats as ms
import preparacao

TOLERANCIA_PADRAO = "1e-9"
TOLERANCIA_PERCENTIL = "1e-6"
TOLERANCIA_DISTRIBUICAO = "1e-10"


def main():
    df, _ = preparacao.carregar()
    total = preparacao.valores(df, "total")
    distancia = preparacao.valores(df, "distance")
    tarifa = preparacao.valores(df, "fare")
    passageiros = preparacao.valores(df, "passengers")

    reta = stats.linregress(distancia, tarifa)
    nossa_reta = ms.regressao_linear(distancia, tarifa)
    media_total = ms.media(total)
    desvio_total = ms.desvio_padrao(total)

    linhas = [
        ("media(total)", ms.media(total), np.mean(total), TOLERANCIA_PADRAO),
        ("mediana(total)", ms.mediana(total), np.median(total),
         TOLERANCIA_PADRAO),
        ("amplitude(total)", ms.amplitude(total), np.ptp(total),
         TOLERANCIA_PADRAO),
        ("variancia(total, amostral=True)", ms.variancia(total, True),
         np.var(total, ddof=1), TOLERANCIA_PADRAO),
        ("variancia(total, amostral=False)", ms.variancia(total, False),
         np.var(total, ddof=0), TOLERANCIA_PADRAO),
        ("desvio_padrao(total, amostral=True)", ms.desvio_padrao(total, True),
         np.std(total, ddof=1), TOLERANCIA_PADRAO),
        ("desvio_padrao(total, amostral=False)",
         ms.desvio_padrao(total, False), np.std(total, ddof=0),
         TOLERANCIA_PADRAO),
        ("coeficiente_variacao(total)", ms.coeficiente_variacao(total),
         100 * np.std(total, ddof=1) / np.mean(total), TOLERANCIA_PADRAO),
        ("percentil(total, 25)", ms.percentil(total, 25),
         np.percentile(total, 25), TOLERANCIA_PERCENTIL),
        ("percentil(total, 50)", ms.percentil(total, 50),
         np.percentile(total, 50), TOLERANCIA_PERCENTIL),
        ("percentil(total, 75)", ms.percentil(total, 75),
         np.percentile(total, 75), TOLERANCIA_PERCENTIL),
        ("percentil(total, 90)", ms.percentil(total, 90),
         np.percentile(total, 90), TOLERANCIA_PERCENTIL),
        ("assimetria(total)", ms.assimetria(total), stats.skew(total),
         TOLERANCIA_PADRAO),
        ("covariancia(distancia, tarifa)", ms.covariancia(distancia, tarifa),
         np.cov(distancia, tarifa, ddof=1)[0][1], TOLERANCIA_PADRAO),
        ("correlacao(distancia, tarifa)", ms.correlacao(distancia, tarifa),
         stats.pearsonr(distancia, tarifa).statistic, TOLERANCIA_PADRAO),
        ("correlacao(passageiros, tarifa)",
         ms.correlacao(passageiros, tarifa),
         stats.pearsonr(passageiros, tarifa).statistic, TOLERANCIA_PADRAO),
        ("regressao_linear -> b1", nossa_reta[1], reta.slope,
         TOLERANCIA_PADRAO),
        ("regressao_linear -> b0", nossa_reta[0], reta.intercept,
         TOLERANCIA_PADRAO),
        ("regressao_linear -> R²", nossa_reta[2], reta.rvalue ** 2,
         TOLERANCIA_PADRAO),
        ("erro_padrao_estimativa(dist, tarifa)",
         ms.erro_padrao_estimativa(distancia, tarifa),
         np.sqrt(np.sum((np.array(tarifa)
                         - (nossa_reta[0] + nossa_reta[1]
                            * np.array(distancia))) ** 2)
                 / (len(distancia) - 2)), TOLERANCIA_PADRAO),
        ("densidade_normal(20; media; desvio)",
         ms.densidade_normal(20.0, media_total, desvio_total),
         stats.norm.pdf(20.0, loc=media_total, scale=desvio_total),
         TOLERANCIA_DISTRIBUICAO),
        ("densidade_exponencial(5; 1/media)",
         ms.densidade_exponencial(5.0, 1 / ms.media(distancia)),
         stats.expon.pdf(5.0, scale=ms.media(distancia)),
         TOLERANCIA_DISTRIBUICAO),
        ("probabilidade_poisson(1; lambda)",
         ms.probabilidade_poisson(1, ms.media(passageiros)),
         stats.poisson.pmf(1, ms.media(passageiros)),
         TOLERANCIA_DISTRIBUICAO),
        ("probabilidade_binomial(3; 10; 0,3)",
         ms.probabilidade_binomial(3, 10, 0.3), stats.binom.pmf(3, 10, 0.3),
         TOLERANCIA_DISTRIBUICAO),
    ]

    print("| Função (minhastats.py) | Nosso valor | Referência "
          "(NumPy/SciPy) | Diferença absoluta | Tolerância |")
    print("|---|---|---|---|---|")
    registro = []
    maior_diferenca = 0.0
    for nome, nosso, deles, tolerancia in linhas:
        deles = float(deles)
        diferenca = abs(nosso - deles)
        maior_diferenca = max(maior_diferenca, diferenca)
        print(f"| `{nome}` | {nosso:.10f} | {deles:.10f} | "
              f"{diferenca:.2e} | {tolerancia} |")
        registro.append({"funcao": nome, "nosso": nosso, "referencia": deles,
                         "diferenca": diferenca, "tolerancia": tolerancia})

    print(f"\nMaior diferença absoluta observada: {maior_diferenca:.2e}")
    caminho = os.path.join("assets", "validacao.json")
    os.makedirs("assets", exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump({"linhas": registro, "maior_diferenca": maior_diferenca},
                  arquivo, ensure_ascii=False, indent=2)
    print(f"Tabela salva em {caminho}")


if __name__ == "__main__":
    main()
