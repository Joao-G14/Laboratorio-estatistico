"""
identificacao.py — dados de identificação da entrega.

Tudo o que muda de uma entrega para outra (nome, matrícula, links) fica
aqui, em um lugar só. O gerador do PDF lê deste arquivo; assim, quando o
vídeo ficar pronto, basta preencher LINK_VIDEO e rodar
`python gerar_pdf.py` de novo.

>>> Tudo preenchido. Se algum link mudar, troque aqui e rode
>>> `python gerar_pdf.py` de novo. <<<
"""

import unicodedata

# --- Identificação ---------------------------------------------------------

NOME_GRUPO = "João Gabriel Amaral de Sales"
INTEGRANTES = [
    # (nome completo, matrícula)
    ("João Gabriel Amaral de Sales", "72650411"),
]

DISCIPLINA = "Matemática e Estatística para Computação"
PROFESSOR = "Prof. Romes Heriberto"
ATIVIDADE = "Sistematização — Laboratório Estatístico Interativo"

# --- Links (todos devem abrir em janela anônima) ---------------------------

LINK_REPOSITORIO = "https://github.com/Joao-G14/Laboratorio-estatistico"

# Fonte ORIGINAL do dataset (o órgão que publica), não o CSV do repositório.
LINK_DATASET_ORIGINAL = (
    "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
)
LINK_DATASET_ARQUIVO = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
)

# Vídeo de 3 a 5 minutos (YouTube não listado ou Drive com acesso liberado).
LINK_VIDEO = "https://youtu.be/L7YlqPDYn0U"


def pendencias():
    """Lista o que ainda está por preencher — usado pelo gerador do PDF."""
    faltando = []
    for nome, matricula in INTEGRANTES:
        if "PREENCHER" in nome:
            faltando.append("NOME_COMPLETO")
        if "PREENCHER" in matricula:
            faltando.append("MATRICULA")
    if "PREENCHER" in LINK_VIDEO:
        faltando.append("LINK_VIDEO")
    return faltando


def nome_do_arquivo_pdf():
    """SISTEMATIZACAO_MEC_NomeDoGrupo.pdf, no padrão pedido no enunciado.

    Acentos e cedilhas são removidos do nome do arquivo: o PDF vai ser
    anexado em um ambiente virtual e baixado por outra pessoa, e nome de
    arquivo com acento ainda quebra em alguns navegadores e servidores.
    """
    nome = INTEGRANTES[0][0]
    if "PREENCHER" in nome:
        return "SISTEMATIZACAO_MEC_NomeDoGrupo.pdf"
    sem_acento = unicodedata.normalize("NFKD", nome)
    sem_acento = sem_acento.encode("ascii", "ignore").decode("ascii")
    limpo = "".join(parte.capitalize() for parte in sem_acento.split())
    return f"SISTEMATIZACAO_MEC_{limpo}.pdf"
