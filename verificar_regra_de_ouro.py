"""
verificar_regra_de_ouro.py — auditoria estática do projeto.

A regra de ouro da atividade diz que toda medida estatística exibida ao
usuário precisa vir das funções escritas pela equipe. Este script confere
isso lendo o código-fonte, sem executar nada:

  1. `minhastats.py` não pode importar NumPy, SciPy, `statistics` ou Pandas;
  2. `app.py` não pode CHAMAR funções estatísticas dessas bibliotecas fora
     do painel de validação do Módulo 1, que é declaradamente uma exceção;
  3. `graficos.py` não pode calcular estatística nenhuma — ele só desenha.

Uso:  python verificar_regra_de_ouro.py
Sai com código 0 se tudo está em ordem, 1 se encontrou violação.
"""

import re
import sys

PROIBIDOS_NO_NUCLEO = [
    "import numpy", "from numpy", "import scipy", "from scipy",
    "import statistics", "from statistics", "import pandas", "from pandas",
]

# Chamadas estatísticas de biblioteca que não podem produzir número de tela.
CHAMADAS_PROIBIDAS = re.compile(
    r"\b(np|numpy)\.(mean|median|std|var|percentile|corrcoef|cov|ptp|polyfit)\b"
    r"|\bstats\.(pearsonr|linregress|skew|mode)\b"
    r"|\bstatistics\.(mean|median|mode|stdev|variance)\b"
    r"|\.(mean|median|std|var|quantile|corr|cov|describe|mode)\s*\("
)

# Sentinelas que delimitam, dentro de app.py, o único trecho autorizado a
# chamar NumPy/SciPy: o painel de validação do Módulo 1.
MARCA_INICIO_EXCECAO = "# regra-de-ouro: inicio-excecao-validacao"
MARCA_FIM_EXCECAO = "# regra-de-ouro: fim-excecao-validacao"


def ler(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        return arquivo.read()


def sem_comentarios(linha):
    """Descarta o que vem depois de '#' para não acusar um comentário."""
    return linha.split("#", 1)[0]


def sem_textos(linha):
    """Esvazia strings literais.

    Sem isso o verificador acusaria a própria documentação: a linha
    `"Mesma convenção do numpy.percentile."` menciona a função, mas não a
    chama.
    """
    linha = re.sub(r'"[^"]*"', '""', linha)
    linha = re.sub(r"'[^']*'", "''", linha)
    return linha


def codigo_efetivo(linha):
    """O que sobra da linha depois de tirar comentários e strings."""
    return sem_textos(sem_comentarios(linha))


def verificar_nucleo():
    codigo = ler("minhastats.py")
    falhas = [proibido for proibido in PROIBIDOS_NO_NUCLEO
              if proibido in codigo]
    if falhas:
        print("  FALHA: minhastats.py importa " + ", ".join(falhas))
        return False
    print("  OK: minhastats.py usa apenas Python puro e o módulo math.")
    return True


def verificar_app():
    linhas = ler("app.py").splitlines()
    dentro_da_excecao = False
    falhas = []
    for numero, linha in enumerate(linhas, start=1):
        if MARCA_INICIO_EXCECAO in linha:
            dentro_da_excecao = True
        elif dentro_da_excecao and MARCA_FIM_EXCECAO in linha:
            dentro_da_excecao = False
            continue
        if dentro_da_excecao:
            continue
        # `.head(`, `.columns`, `.tolist()` são manipulação, não estatística.
        if CHAMADAS_PROIBIDAS.search(codigo_efetivo(linha)):
            falhas.append((numero, linha.strip()))
    if falhas:
        print("  FALHA: app.py calcula estatística com biblioteca externa "
              "fora do painel de validação:")
        for numero, texto in falhas:
            print(f"    linha {numero}: {texto}")
        return False
    print("  OK: app.py só exibe números vindos de minhastats.py "
          "(exceto o painel de validação do Módulo 1, que é declarado).")
    return True


def verificar_graficos():
    linhas = ler("graficos.py").splitlines()
    falhas = [(numero, linha.strip())
              for numero, linha in enumerate(linhas, start=1)
              if CHAMADAS_PROIBIDAS.search(codigo_efetivo(linha))]
    if falhas:
        print("  FALHA: graficos.py está calculando, e não só desenhando:")
        for numero, texto in falhas:
            print(f"    linha {numero}: {texto}")
        return False
    print("  OK: graficos.py apenas desenha; recebe os números prontos.")
    return True


def main():
    print("Auditoria da regra de ouro\n" + "-" * 60)
    resultados = [verificar_nucleo(), verificar_app(), verificar_graficos()]
    print("-" * 60)
    if all(resultados):
        print("Tudo certo: as medidas exibidas vêm do núcleo próprio.")
        return 0
    print("Há violações da regra de ouro. Corrija antes de entregar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
