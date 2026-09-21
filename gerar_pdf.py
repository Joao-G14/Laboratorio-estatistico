"""
gerar_pdf.py — monta o PDF de entrega.

O PDF é o relatório completo (dados, fórmulas, validação, gráficos,
descobertas e limitações) mais os links exigidos no enunciado: fonte
original do dataset, repositório e vídeo em destaque na última página.

Os NÚMEROS das tabelas não são digitados aqui: são lidos de
`assets/resultados.json` e `assets/validacao.json`, produzidos por
`gerar_figuras.py` e `gerar_tabela_validacao.py`. Assim o PDF não tem
como divergir do que a aplicação calcula.

Pré-requisitos (nesta ordem):
    python gerar_figuras.py
    python gerar_tabela_validacao.py
    python gerar_prints.py        # opcional, requer playwright

Uso:  python gerar_pdf.py
"""

import json
import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

import identificacao

PASTA = "assets"

# Paleta — a mesma de graficos.py, para o PDF e os gráficos falarem igual.
AZUL = colors.HexColor("#2a78d6")
AZUL_ESCURO = colors.HexColor("#184f95")
LARANJA = colors.HexColor("#eb6834")
TINTA = colors.HexColor("#0b0b0b")
TINTA_2 = colors.HexColor("#52514e")
TINTA_FRACA = colors.HexColor("#898781")
LINHA = colors.HexColor("#e1e0d9")
FUNDO_SUAVE = colors.HexColor("#f3f6fa")
ALERTA = colors.HexColor("#d03b3b")

LARGURA_UTIL = A4[0] - 4 * cm


# ---------------------------------------------------------------------------
# Fontes: DejaVu Sans vem junto com o matplotlib e tem Unicode completo
# (σ, λ, g₁, ⌈ ⌉), ao contrário da Helvetica embutida do reportlab.
# ---------------------------------------------------------------------------


def registrar_fontes():
    base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    variantes = {
        "DejaVu": "DejaVuSans.ttf",
        "DejaVu-Bold": "DejaVuSans-Bold.ttf",
        "DejaVu-Italic": "DejaVuSans-Oblique.ttf",
        "DejaVu-Mono": "DejaVuSansMono.ttf",
    }
    for nome, arquivo in variantes.items():
        pdfmetrics.registerFont(TTFont(nome, os.path.join(base, arquivo)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic")


def criar_estilos():
    padrao = getSampleStyleSheet()
    estilos = {}
    estilos["corpo"] = ParagraphStyle(
        "corpo", parent=padrao["Normal"], fontName="DejaVu", fontSize=9.5,
        leading=14.5, alignment=TA_JUSTIFY, textColor=TINTA,
        spaceAfter=7)
    estilos["h1"] = ParagraphStyle(
        "h1", parent=estilos["corpo"], fontName="DejaVu-Bold", fontSize=17,
        leading=21, alignment=0, textColor=AZUL_ESCURO, spaceBefore=6,
        spaceAfter=10)
    estilos["h2"] = ParagraphStyle(
        "h2", parent=estilos["corpo"], fontName="DejaVu-Bold", fontSize=12.5,
        leading=16, alignment=0, textColor=AZUL, spaceBefore=13,
        spaceAfter=6)
    estilos["h3"] = ParagraphStyle(
        "h3", parent=estilos["corpo"], fontName="DejaVu-Bold", fontSize=10.5,
        leading=14, alignment=0, textColor=TINTA, spaceBefore=9,
        spaceAfter=4)
    estilos["legenda"] = ParagraphStyle(
        "legenda", parent=estilos["corpo"], fontSize=8, leading=11,
        alignment=TA_CENTER, textColor=TINTA_FRACA, spaceBefore=3,
        spaceAfter=10)
    estilos["formula"] = ParagraphStyle(
        "formula", parent=estilos["corpo"], fontName="DejaVu",
        fontSize=10.5, leading=17, alignment=TA_CENTER, textColor=AZUL_ESCURO,
        spaceBefore=6, spaceAfter=8)
    estilos["destaque"] = ParagraphStyle(
        "destaque", parent=estilos["corpo"], fontSize=9.5, leading=14,
        leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=6,
        borderPadding=8, backColor=FUNDO_SUAVE, borderColor=AZUL,
        borderWidth=0)
    estilos["alerta"] = ParagraphStyle(
        "alerta", parent=estilos["destaque"],
        backColor=colors.HexColor("#fdf0ee"), borderColor=ALERTA)
    estilos["capa_titulo"] = ParagraphStyle(
        "capa_titulo", parent=estilos["corpo"], fontName="DejaVu-Bold",
        fontSize=30, leading=35, alignment=0, textColor=colors.white,
        spaceAfter=4)
    estilos["capa_sub"] = ParagraphStyle(
        "capa_sub", parent=estilos["corpo"], fontSize=12.5, leading=18,
        alignment=0, textColor=colors.HexColor("#cde2fb"))
    estilos["tabela"] = ParagraphStyle(
        "tabela", parent=estilos["corpo"], fontSize=8, leading=10.5,
        alignment=0, spaceAfter=0)
    estilos["tabela_cabecalho"] = ParagraphStyle(
        "tabela_cabecalho", parent=estilos["tabela"], fontName="DejaVu-Bold",
        textColor=colors.white)
    estilos["video"] = ParagraphStyle(
        "video", parent=estilos["corpo"], fontName="DejaVu-Bold", fontSize=13,
        leading=19, alignment=TA_CENTER, textColor=AZUL_ESCURO)
    return estilos


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def num(valor, casas=2):
    """Formato brasileiro: 1234.5 -> '1.234,50'."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def cientifico(valor):
    """3.7e-13 -> '3,70e-13'."""
    return f"{valor:.2e}".replace(".", ",")


def figura(nome, largura=LARGURA_UTIL, legenda=None, estilos=None):
    """Insere uma imagem de assets/, preservando a proporção."""
    caminho = os.path.join(PASTA, nome)
    if not os.path.exists(caminho):
        return []
    from reportlab.lib.utils import ImageReader
    largura_px, altura_px = ImageReader(caminho).getSize()
    altura = largura * altura_px / largura_px
    # Nenhuma figura ocupa mais que 2/3 da página útil.
    maximo = A4[1] - 9 * cm
    if altura > maximo:
        largura = largura * maximo / altura
        altura = maximo
    itens = [Image(caminho, width=largura, height=altura)]
    if legenda:
        itens.append(Paragraph(legenda, estilos["legenda"]))
    return itens


def tabela(dados, estilos, larguras=None, alinhar_direita=None):
    """Monta uma tabela com o cabeçalho azul e zebra clara."""
    corpo = []
    for i, linha in enumerate(dados):
        estilo = estilos["tabela_cabecalho"] if i == 0 else estilos["tabela"]
        corpo.append([Paragraph(str(celula), estilo) for celula in linha])

    t = Table(corpo, colWidths=larguras, repeatRows=1, hAlign="LEFT")
    comandos = [
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINHA),
        ("BOX", (0, 0), (-1, -1), 0.4, LINHA),
    ]
    for i in range(1, len(dados)):
        if i % 2 == 0:
            comandos.append(("BACKGROUND", (0, i), (-1, i), FUNDO_SUAVE))
    if alinhar_direita:
        for coluna in alinhar_direita:
            comandos.append(("ALIGN", (coluna, 1), (coluna, -1), "RIGHT"))
    t.setStyle(TableStyle(comandos))
    return t


def caixa(texto, estilos, tipo="destaque"):
    """Parágrafo em caixa colorida, com a barra lateral do tipo."""
    cor = ALERTA if tipo == "alerta" else AZUL
    interno = Table([[Paragraph(texto, estilos[tipo])]],
                    colWidths=[LARGURA_UTIL], hAlign="LEFT")
    interno.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 3, cor),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return interno


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------


def fundo_capa(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(AZUL_ESCURO)
    canvas.rect(0, A4[1] - 12.5 * cm, A4[0], 12.5 * cm, stroke=0, fill=1)
    canvas.setFillColor(LARANJA)
    canvas.rect(0, A4[1] - 12.9 * cm, A4[0], 0.4 * cm, stroke=0, fill=1)
    canvas.restoreState()


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINHA)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(TINTA_FRACA)
    canvas.drawString(2 * cm, 1.15 * cm,
                      "Laboratório Estatístico Interativo — Sistematização")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"{doc.page}")
    canvas.restoreState()


def montar_capa(estilos):
    integrantes = "<br/>".join(
        f"{nome} &nbsp;—&nbsp; matrícula {matricula}"
        for nome, matricula in identificacao.INTEGRANTES)

    itens = [
        Spacer(1, 1.4 * cm),
        Paragraph("SISTEMATIZAÇÃO", ParagraphStyle(
            "eyebrow", parent=estilos["capa_sub"], fontName="DejaVu-Bold",
            fontSize=10, textColor=colors.HexColor("#ffb24a"))),
        Spacer(1, 0.3 * cm),
        Paragraph("Laboratório<br/>Estatístico Interativo",
                  estilos["capa_titulo"]),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "Estatística implementada do zero sobre 6.427 corridas de táxi "
            "de Nova York", estilos["capa_sub"]),
        Spacer(1, 4.3 * cm),
        Paragraph(f"<b>{identificacao.DISCIPLINA}</b>", estilos["corpo"]),
        Paragraph(identificacao.PROFESSOR, estilos["corpo"]),
        Spacer(1, 0.5 * cm),
        Paragraph(f"<b>Grupo:</b> {identificacao.NOME_GRUPO} "
                  f"(entrega individual)", estilos["corpo"]),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Integrante(s)</b>", estilos["h3"]),
        Paragraph(integrantes, estilos["corpo"]),
        Spacer(1, 0.7 * cm),
    ]

    links = [
        ["Item", "Link"],
        ["Fonte original do dataset (NYC TLC)",
         f'<link href="{identificacao.LINK_DATASET_ORIGINAL}">'
         f'{identificacao.LINK_DATASET_ORIGINAL}</link>'],
        ["Arquivo CSV utilizado",
         f'<link href="{identificacao.LINK_DATASET_ARQUIVO}">'
         f'{identificacao.LINK_DATASET_ARQUIVO}</link>'],
        ["Repositório público (código completo)",
         f'<link href="{identificacao.LINK_REPOSITORIO}">'
         f'{identificacao.LINK_REPOSITORIO}</link>'],
        ["Vídeo de demonstração (3–5 min)",
         f'<link href="{identificacao.LINK_VIDEO}">'
         f'{identificacao.LINK_VIDEO}</link>'
         if "PREENCHER" not in identificacao.LINK_VIDEO
         else identificacao.LINK_VIDEO],
    ]
    itens.append(tabela(links, estilos, larguras=[6 * cm, LARGURA_UTIL - 6 * cm]))
    # A faixa azul da capa é desenhada pelo template "capa"; da página 2 em
    # diante vale o template "corpo", que desenha o rodapé com o número da
    # página. Sem este NextPageTemplate o fundo da capa se repetiria em
    # todas as folhas.
    itens.append(NextPageTemplate("corpo"))
    itens.append(PageBreak())
    return itens


# ---------------------------------------------------------------------------
# Corpo do relatório
# ---------------------------------------------------------------------------


def secao_resumo(estilos, resultados):
    d1 = resultados["descoberta1"]
    d3 = resultados["descoberta3"]
    reg = resultados["regressao"]
    itens = [
        Paragraph("1. Resumo executivo", estilos["h1"]),
        Paragraph(
            "Construímos uma biblioteca estatística em Python puro "
            "(<font face='DejaVu-Mono' size=9>minhastats.py</font>) — sem "
            "NumPy, SciPy ou <font face='DejaVu-Mono' size=9>statistics</font> "
            "— e uma aplicação Streamlit de sete módulos que a usa para "
            "analisar <b>6.427 corridas de táxi da cidade de Nova York</b>, "
            "registradas pela NYC Taxi &amp; Limousine Commission entre "
            "28/02/2019 e 31/03/2019.", estilos["corpo"]),
        Paragraph(
            "As 24 medidas do núcleo foram validadas contra NumPy/SciPy com "
            "diferença máxima de <b>4,09 × 10⁻¹²</b> — resíduo de ponto "
            "flutuante, não erro de fórmula. A suíte tem <b>104 testes "
            "automatizados</b> (94 do núcleo, 10 da interface), todos "
            "passando.", estilos["corpo"]),
        Paragraph("Os sete módulos", estilos["h2"]),
    ]
    modulos = [
        ["Módulo", "O que a aplicação entrega"],
        ["0 — Dados reais",
         "Fonte, verificação automática dos critérios, dicionário das 13 "
         "variáveis, decisões de tratamento e prévia."],
        ["1 — Núcleo estatístico",
         "Fórmulas implementadas e tabela de validação <b>ao vivo</b> contra "
         "NumPy/SciPy, com diferença e tolerância."],
        ["2 — Descritiva interativa",
         "Tendência central, dispersão e posição; tabela de frequências de "
         "Sturges; histograma; boxplot com outliers do IQR; barras/pizza; "
         "interpretação textual automática."],
        ["3 — Simulação",
         "Lei dos Grandes Números (moeda/dado) e Teorema Central do Limite "
         "sobre os nossos dados, com controles de repetições e tamanho de "
         "amostra."],
        ["4 — Distribuições teóricas",
         "Normal e Exponencial/Uniforme sobre o histograma em densidade; "
         "Poisson para a contagem de passageiros; erro de ajuste medido."],
        ["5 — Correlação e regressão",
         "Dispersão, r de Pearson, mínimos quadrados, R², resíduos e "
         "predição interativa com alerta de extrapolação."],
        ["6 — Descobertas",
         "As três descobertas com evidência e limite honesto, mais as "
         "limitações gerais da análise."],
    ]
    itens.append(tabela(modulos, estilos,
                        larguras=[4 * cm, LARGURA_UTIL - 4 * cm]))

    itens += [
        Paragraph("As três descobertas", estilos["h2"]),
        caixa(
            "<b>1. Quem paga em dinheiro nunca dá gorjeta — e o defeito é do "
            "instrumento, não do passageiro.</b> "
            f"{num(d1['cash']['pct_zero'], 1)}% das "
            f"{num(d1['cash']['n'], 0)} corridas em espécie registram "
            "gorjeta exatamente zero, porque o taxímetro só captura gorjeta "
            "eletrônica. O zero significa <i>“não medido”</i>, não "
            "<i>“não pago”</i>.", estilos),
        Spacer(1, 0.25 * cm),
        caixa(
            "<b>2. O número de passageiros não explica nada do preço:</b> "
            f"r = {num(resultados['descoberta2']['correlacoes']['fare'], 4)} "
            "com a tarifa e R² = "
            f"{num(resultados['descoberta2']['r2_passageiros_tarifa'], 6)}. "
            "A tarifa em Nova York é do veículo, não da cabeça — os "
            "passageiros extras viajam de graça.", estilos),
        Spacer(1, 0.25 * cm),
        caixa(
            "<b>3. Os outliers de preço têm endereço:</b> "
            f"{num(d3['pct_aero_outliers'], 1)}% das "
            f"{num(d3['n_outliers'], 0)} corridas marcadas como outliers "
            "pela regra do IQR começam ou terminam em um aeroporto, contra "
            f"{num(d3['pct_aero_geral'], 1)}% no dataset inteiro — "
            f"concentração {num(d3['razao'], 1)} vezes maior.", estilos),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "A relação central do dataset, usada como fio condutor: a "
            f"distância explica {num(100 * reg['r2'], 2)}% da variação da "
            f"tarifa (r = {num(reg['r'], 4)}), pela reta "
            f"ŷ = {num(reg['b1'], 4)}·x + {num(reg['b0'], 4)}.",
            estilos["corpo"]),
        PageBreak(),
    ]
    return itens


def secao_dataset(estilos, resultados):
    tratamento = resultados["tratamento"]
    itens = [
        Paragraph("2. O conjunto de dados", estilos["h1"]),
        Paragraph("2.1 De onde vêm os dados", estilos["h2"]),
        Paragraph(
            "Os registros são das corridas de táxi da cidade de Nova York, "
            "coletados e publicados como dado aberto pela <b>New York City "
            "Taxi &amp; Limousine Commission (TLC)</b>, a autarquia "
            "municipal que regula o serviço. A TLC publica os <i>TLC Trip "
            "Record Data</i> mês a mês: um registro por corrida, gerado pelo "
            "próprio taxímetro. Essa é a base primária.", estilos["corpo"]),
        Paragraph(
            "O arquivo que baixamos é uma <b>amostra consolidada em CSV</b>, "
            "distribuída no repositório público <font face='DejaVu-Mono' "
            "size=9>seaborn-data</font>, escolhida por ser diretamente "
            "reprodutível: uma URL única, sem cadastro, sem chave de API e "
            "sem etapa de conversão de Parquet. A cópia exata usada pela "
            "aplicação está versionada em <font face='DejaVu-Mono' size=9>"
            "dados/dataset.csv</font>, de modo que qualquer pessoa que clone "
            "o repositório reproduz os mesmos números.", estilos["corpo"]),
    ]
    fonte = [
        ["Item", "Valor"],
        ["Fonte original (órgão)",
         f'<link href="{identificacao.LINK_DATASET_ORIGINAL}">'
         f'{identificacao.LINK_DATASET_ORIGINAL}</link>'],
        ["Arquivo CSV baixado",
         f'<link href="{identificacao.LINK_DATASET_ARQUIVO}">'
         f'{identificacao.LINK_DATASET_ARQUIVO}</link>'],
        ["Período coberto", "28/02/2019 a 31/03/2019"],
        ["Registros", f"{num(tratamento['linhas_originais'], 0)} no arquivo "
                      f"→ {num(tratamento['linhas_finais'], 0)} após o "
                      "tratamento"],
        ["Variáveis", "6 numéricas + 1 derivada · 6 categóricas"],
    ]
    itens.append(tabela(fonte, estilos,
                        larguras=[5 * cm, LARGURA_UTIL - 5 * cm]))

    itens += [
        Paragraph("2.2 Por que escolhemos este dataset", estilos["h2"]),
        Paragraph(
            "Além de atender com folga aos critérios da atividade, o táxi é "
            "um serviço que qualquer pessoa entende — o que permitiu "
            "formular perguntas de verdade antes de olhar os números, em vez "
            "de apenas descrever colunas. E a estrutura estatística é rica: "
            "variáveis fortemente assimétricas (material ideal para o "
            "Teorema Central do Limite), uma relação linear quase de "
            "livro-texto (distância × tarifa, r = 0,92) e armadilhas reais "
            "de medição — que acabaram virando a descoberta mais "
            "interessante do trabalho.", estilos["corpo"]),
        Paragraph("2.3 As variáveis", estilos["h2"]),
    ]
    variaveis = [["Coluna", "Tipo", "Significado"]]
    for coluna, (rotulo, tipo, significado) in [
        ("distance", ("Distância (milhas)", "numérica contínua",
                      "Distância percorrida, medida pelo taxímetro.")),
        ("fare", ("Tarifa (US$)", "numérica contínua",
                  "Valor do taxímetro, antes de gorjeta e pedágios.")),
        ("tip", ("Gorjeta (US$)", "numérica contínua",
                 "Gorjeta registrada (ver descoberta 1).")),
        ("tolls", ("Pedágios (US$)", "numérica contínua",
                   "Pedágios de pontes e túneis repassados.")),
        ("total", ("Total pago (US$)", "numérica contínua",
                   "Tarifa + gorjeta + pedágios + taxas.")),
        ("passengers", ("Passageiros", "numérica discreta",
                        "Número informado pelo motorista.")),
        ("duracao_min", ("Duração (min)", "numérica contínua (derivada)",
                         "Minutos entre embarque e desembarque.")),
        ("color", ("Tipo de táxi", "categórica nominal",
                   "yellow (toda a cidade) ou green (fora do centro).")),
        ("payment", ("Forma de pagamento", "categórica nominal",
                     "credit card ou cash.")),
        ("pickup_borough", ("Distrito de embarque", "categórica nominal",
                            "Manhattan, Queens, Brooklyn, Bronx.")),
        ("dropoff_borough", ("Distrito de desembarque", "categórica nominal",
                             "Inclui Staten Island.")),
        ("pickup_zone", ("Zona de embarque", "categórica nominal",
                         "Zona tarifária de origem (194 níveis).")),
        ("dropoff_zone", ("Zona de desembarque", "categórica nominal",
                          "Zona tarifária de destino (203 níveis).")),
    ]:
        variaveis.append([f"<font face='DejaVu-Mono' size=7.5>{coluna}</font>",
                          tipo, significado])
    itens.append(tabela(variaveis, estilos,
                        larguras=[3.1 * cm, 3.6 * cm,
                                  LARGURA_UTIL - 6.7 * cm]))

    ausentes = tratamento["ausentes_preenchidos"]
    itens += [
        Paragraph("2.4 Decisões de tratamento", estilos["h2"]),
        Paragraph(
            "<b>1. Conversão de data/hora e coluna derivada.</b> "
            "<font face='DejaVu-Mono' size=9>pickup</font> e "
            "<font face='DejaVu-Mono' size=9>dropoff</font> viraram "
            "<i>datetime</i>, e deles derivamos "
            "<font face='DejaVu-Mono' size=9>duracao_min</font>. É a única "
            "coluna que criamos.", estilos["corpo"]),
        Paragraph(
            f"<b>2. Remoção de {tratamento['removidas_duracao_invalida']} "
            "corridas com duração ≤ 0.</b> Uma corrida que termina antes de "
            "começar é fisicamente impossível: é erro de taxímetro, não "
            f"dado. Restaram {num(tratamento['linhas_finais'], 0)} registros "
            f"dos {num(tratamento['linhas_originais'], 0)} originais "
            "(0,09% descartado).", estilos["corpo"]),
        Paragraph(
            "<b>3. Ausentes categóricos viraram categoria.</b> As colunas "
            f"payment ({ausentes.get('payment', 0)} linhas), pickup_zone e "
            f"pickup_borough ({ausentes.get('pickup_zone', 0)}), "
            "dropoff_zone e dropoff_borough "
            f"({ausentes.get('dropoff_zone', 0)}) tinham ausentes — no "
            "máximo 0,7% das linhas. Em vez de descartar a corrida inteira, "
            "preenchemos com a categoria explícita <b>“Não informado”</b>. "
            "As colunas numéricas dessas corridas estão completas e são "
            "válidas; jogá-las fora perderia informação boa por causa de um "
            "campo ruim. E, como a categoria é explícita, ela aparece nas "
            "tabelas de frequência em vez de sumir.", estilos["corpo"]),
        Paragraph(
            f"<b>4. Mantidas com ressalva:</b> {tratamento['distancia_zero']} "
            f"corridas com distância zero e {tratamento['passageiros_zero']} "
            "com zero passageiros. Não são impossíveis — provavelmente o "
            "motorista não digitou, ou a corrida foi curta demais para o "
            "odômetro registrar. Como não temos como distinguir “erro” de "
            "“valor real raro”, mantivemos e registramos nas limitações "
            "(seção 7).", estilos["corpo"]),
        PageBreak(),
    ]
    return itens


def secao_nucleo(estilos, validacao):
    itens = [
        Paragraph("3. O núcleo estatístico e sua validação", estilos["h1"]),
        Paragraph(
            "Todas as funções estão em <font face='DejaVu-Mono' size=9>"
            "minhastats.py</font>, escritas com Python puro e o módulo "
            "<font face='DejaVu-Mono' size=9>math</font> (apenas sqrt, exp, "
            "log, log10, pi, lgamma e ceil — funções matemáticas "
            "elementares, não estatísticas).", estilos["corpo"]),
        Paragraph("3.1 As fórmulas implementadas", estilos["h2"]),
        Paragraph("Tendência central e dispersão", estilos["h3"]),
        Paragraph(
            "x̄ = (1/n) · Σ xᵢ &nbsp;&nbsp;&nbsp;&nbsp; "
            "s² = Σ(xᵢ − x̄)² / (n − 1) &nbsp;&nbsp;&nbsp;&nbsp; "
            "σ² = Σ(xᵢ − x̄)² / n", estilos["formula"]),
        Paragraph(
            "s = √s² &nbsp;&nbsp;&nbsp;&nbsp; A = x_máx − x_mín "
            "&nbsp;&nbsp;&nbsp;&nbsp; CV = (s / x̄) × 100",
            estilos["formula"]),
        Paragraph(
            "A correção de Bessel (n − 1) existe porque x̄ foi estimada dos "
            "próprios dados: os desvios em torno dela já são, por "
            "construção, os menores possíveis. Dividir por n subestimaria a "
            "dispersão da população. No nosso dataset a diferença é pequena "
            "mas real — variância do total: 190,4867 (amostral) contra "
            "190,4571 (populacional).", estilos["corpo"]),
        Paragraph("Posição e detecção de outliers", estilos["h3"]),
        Paragraph(
            "posição = p · (n − 1) / 100 &nbsp;&nbsp;&nbsp;&nbsp; "
            "P_p = x[k] + f · (x[k+1] − x[k])", estilos["formula"]),
        Paragraph(
            "IQR = Q₃ − Q₁ &nbsp;&nbsp;&nbsp;&nbsp; "
            "outlier se x &lt; Q₁ − 1,5·IQR ou x &gt; Q₃ + 1,5·IQR",
            estilos["formula"]),
        Paragraph(
            "k é a parte inteira da posição e f a fracionária. Existem pelo "
            "menos nove convenções de percentil; adotamos a interpolação "
            "linear, a mesma do método padrão do "
            "<font face='DejaVu-Mono' size=9>numpy.percentile</font>, para "
            "que a validação seja direta e verificável.", estilos["corpo"]),
        Paragraph("Associação, forma e regressão", estilos["h3"]),
        Paragraph(
            "cov(x,y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1) &nbsp;&nbsp;&nbsp;&nbsp; "
            "r = cov(x,y) / (s_x · s_y)", estilos["formula"]),
        Paragraph(
            "g₁ = [ (1/n) · Σ(xᵢ − x̄)³ ] / σ³ &nbsp;&nbsp;&nbsp;&nbsp; "
            "Sk = 3·(x̄ − mediana) / s &nbsp;&nbsp;&nbsp;&nbsp; "
            "k = ⌈1 + 3,322 · log₁₀(n)⌉", estilos["formula"]),
        Paragraph(
            "b₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)² = cov(x,y) / var(x)"
            "&nbsp;&nbsp;&nbsp;&nbsp; b₀ = ȳ − b₁·x̄", estilos["formula"]),
        Paragraph(
            "R² = 1 − SQ_res / SQ_tot = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²"
            "&nbsp;&nbsp;&nbsp;&nbsp; s_e = √[ Σ(yᵢ − ŷᵢ)² / (n − 2) ]",
            estilos["formula"]),
        Paragraph("Distribuições teóricas", estilos["h3"]),
        Paragraph(
            "f_Normal(x) = e^(−½·((x−μ)/σ)²) / (σ√(2π)) "
            "&nbsp;&nbsp;&nbsp;&nbsp; f_Exp(x) = λ·e^(−λx), x ≥ 0",
            estilos["formula"]),
        Paragraph(
            "P_Poisson(X=k) = e^(−λ)·λᵏ / k! &nbsp;&nbsp;&nbsp;&nbsp; "
            "P_Bin(X=k) = C(n,k)·pᵏ·(1−p)^(n−k) &nbsp;&nbsp;&nbsp;&nbsp; "
            "f_Unif(x) = 1/(b−a)", estilos["formula"]),
        Paragraph(
            "Poisson e Binomial são calculadas em <b>escala logarítmica</b>, "
            "com <font face='DejaVu-Mono' size=9>math.lgamma</font> no lugar "
            "do fatorial: log P = −λ + k·log λ − log Γ(k+1). Com o fatorial "
            "direto, k grande estoura o float; em log, não.",
            estilos["corpo"]),
        PageBreak(),
        Paragraph("3.2 Validação contra NumPy/SciPy", estilos["h2"]),
        Paragraph(
            "Cada função foi comparada com uma referência independente, "
            "sobre as variáveis reais do dataset. Esta tabela é a saída de "
            "<font face='DejaVu-Mono' size=9>python "
            "gerar_tabela_validacao.py</font>.", estilos["corpo"]),
    ]

    linhas = [["Função (minhastats.py)", "Nosso valor", "Referência",
               "Diferença", "Tolerância"]]
    for linha in validacao["linhas"]:
        linhas.append([
            f"<font face='DejaVu-Mono' size=7>{linha['funcao']}</font>",
            num(linha["nosso"], 8), num(linha["referencia"], 8),
            cientifico(linha["diferenca"]), linha["tolerancia"]])
    itens.append(tabela(linhas, estilos,
                        larguras=[6.1 * cm, 3.1 * cm, 3.1 * cm, 2.1 * cm,
                                  LARGURA_UTIL - 14.4 * cm],
                        alinhar_direita=[1, 2, 3]))

    itens += [
        Spacer(1, 0.3 * cm),
        caixa(
            "<b>Maior diferença absoluta observada: "
            f"{cientifico(validacao['maior_diferenca'])}.</b> Somas de ponto "
            "flutuante feitas em ordens diferentes divergem nas últimas "
            "casas decimais — isso é esperado, não é bug.", estilos),
        Paragraph("Sobre as tolerâncias", estilos["h3"]),
        Paragraph(
            "<b>1e-9</b> para as medidas construídas por somatório: folgado "
            "o bastante para o ruído de arredondamento e apertado o bastante "
            "para pegar erro real de fórmula — trocar n−1 por n produz erro "
            "da ordem de 1/n ≈ 1,6 × 10⁻⁴, cem milhões de vezes maior que a "
            "tolerância. <b>1e-6</b> para percentis, que envolvem "
            "multiplicação e divisão de índices, com folga extra por haver "
            "várias convenções possíveis. <b>1e-10</b> para as densidades "
            "teóricas, que são avaliações diretas de exp/log e saem "
            "praticamente exatas.", estilos["corpo"]),
        Paragraph("3.3 A suíte de testes", estilos["h2"]),
        Paragraph(
            "<font face='DejaVu-Mono' size=9>pytest -v</font> roda <b>104 "
            "testes</b>: 94 do núcleo (cada função contra sua referência, "
            "mais os casos de borda — lista vazia, n = 1 na variância "
            "amostral, variável constante na correlação, vetores de "
            "tamanhos diferentes, p fora de [0, 100] — e as identidades de "
            "consistência: R² = r² na regressão simples, soma das "
            "probabilidades da Binomial igual a 1, área do histograma em "
            "densidade igual a 1) e 10 da interface, em que o "
            "<font face='DejaVu-Mono' size=9>AppTest</font> do Streamlit "
            "sobe a aplicação em memória, navega pelos sete módulos e "
            "verifica que nenhum levanta exceção.", estilos["corpo"]),
        Paragraph(
            "Um desses testes merece destaque: ele <b>lê o código-fonte</b> "
            "de minhastats.py e falha se encontrar "
            "<font face='DejaVu-Mono' size=9>import numpy</font>, "
            "<font face='DejaVu-Mono' size=9>import scipy</font>, "
            "<font face='DejaVu-Mono' size=9>import statistics</font> ou "
            "<font face='DejaVu-Mono' size=9>import pandas</font>. A regra "
            "de ouro deixou de ser uma promessa e virou um teste. Um "
            "auditor adicional, "
            "<font face='DejaVu-Mono' size=9>verificar_regra_de_ouro.py</font>, "
            "varre app.py e graficos.py procurando chamadas como "
            "<font face='DejaVu-Mono' size=9>np.mean</font> ou "
            "<font face='DejaVu-Mono' size=9>.describe()</font> fora do "
            "painel de validação do Módulo 1 — a única exceção, delimitada "
            "no código por marcadores explícitos.", estilos["corpo"]),
        caixa(
            "<b>Um bug real que os testes pegaram.</b> A primeira versão de "
            "<font face='DejaVu-Mono' size=9>tabela_frequencias_continua</font> "
            "calculava o limite superior da última classe como "
            "<i>mínimo + k × largura</i>. Por acúmulo de erro de ponto "
            "flutuante esse valor ficava uma fração abaixo do máximo "
            "observado, e <b>o maior valor do conjunto caía fora da "
            "tabela</b> — 499 elementos contados em vez de 500. O teste que "
            "exigia “soma das frequências = n” quebrou, e a correção foi "
            "fixar a última fronteira no próprio máximo. Sem o teste, o "
            "histograma estaria errado em um elemento e ninguém "
            "perceberia.", estilos, "alerta"),
        PageBreak(),
    ]
    return itens


def secao_descritiva(estilos, resultados):
    descritivas = resultados["descritivas"]
    itens = [
        Paragraph("4. Módulo 2 — Estatística descritiva", estilos["h1"]),
        Paragraph(
            "O usuário escolhe qualquer variável e recebe as três famílias "
            "de medidas, um percentil personalizado por slider, a tabela de "
            "frequências em classes de Sturges, histograma, boxplot com os "
            "outliers do IQR destacados e a interpretação textual "
            "automática.", estilos["corpo"]),
    ]
    itens += figura("print_modulo2_descritiva.png", LARGURA_UTIL,
                    "A aplicação em funcionamento: Módulo 2, variável "
                    "Distância.", estilos)

    cabecalho = ["Variável", "Média", "Mediana", "Desvio", "CV", "Q₁", "Q₃",
                 "g₁", "Outliers (IQR)"]
    rotulos = {"distance": "Distância (mi)", "fare": "Tarifa (US$)",
               "tip": "Gorjeta (US$)", "tolls": "Pedágios (US$)",
               "total": "Total (US$)", "passengers": "Passageiros",
               "duracao_min": "Duração (min)"}
    linhas = [cabecalho]
    for coluna, rotulo in rotulos.items():
        r = descritivas[coluna]
        linhas.append([
            rotulo, num(r["media"], 3), num(r["mediana"], 3),
            num(r["desvio_amostral"], 3), f"{num(r['cv'], 1)}%",
            num(r["q1"], 3), num(r["q3"], 3), num(r["assimetria"], 3),
            f"{num(r['n_outliers'], 0)} ({num(r['pct_outliers'], 2)}%)"])
    itens.append(tabela(linhas, estilos,
                        larguras=[2.7 * cm, 1.45 * cm, 1.8 * cm, 1.55 * cm,
                                  1.35 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm,
                                  LARGURA_UTIL - 12.75 * cm],
                        alinhar_direita=[1, 2, 3, 4, 5, 6, 7, 8]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>Duas coisas a notar nesta tabela.</b> Primeiro, <b>todas as "
            "sete variáveis têm g₁ &gt; 0</b> — não há uma única variável "
            "simétrica no dataset, o que faz sentido: preço, tempo e "
            "distância são grandezas com piso em zero e sem teto. Segundo, "
            "<b>o CV dos pedágios é "
            f"{num(descritivas['tolls']['cv'], 1)}%</b>, absurdamente alto, "
            "e o motivo é que a variável é quase toda zero: 75% das corridas "
            "não pagam pedágio nenhum, então o IQR é <b>zero</b> e a regra "
            "de Tukey degenera — ela passa a marcar como “outlier” qualquer "
            "corrida que simplesmente tenha pago pedágio. A aplicação "
            "detecta esse caso e exibe um aviso em vez de apresentar o "
            "número como se fosse anomalia. É um limite do método, não um "
            "achado sobre os dados.", estilos["corpo"]),
        PageBreak(),
    ]

    r = descritivas["total"]
    itens += figura("fig02_histograma_total.png", LARGURA_UTIL,
                    "Figura 1 — Histograma do valor total, com classes de "
                    "Sturges e as posições de média e mediana.", estilos)
    itens += [
        Paragraph(
            f"<b>Leitura.</b> O valor total pago tem média US$ "
            f"{num(r['media'])} e mediana US$ {num(r['mediana'])}, e o "
            f"coeficiente de assimetria vale g₁ = {num(r['assimetria'], 2)} "
            "— cauda longa à direita, exatamente o que o histograma mostra. "
            "A média está sendo puxada por uma minoria de corridas caras, "
            "então <b>para descrever a corrida típica a mediana é a medida "
            "honesta</b>.", estilos["corpo"]),
        caixa(
            "<b>A regra da interpretação automática — e por que trocamos a "
            "primeira.</b> Nossa primeira versão usava a regra intuitiva: "
            "“se média − mediana passar de meio desvio padrão, é "
            "assimétrica”. Ao rodá-la sobre o dataset descobrimos que ela "
            "<b>classificava as sete variáveis como simétricas</b> — "
            "inclusive os pedágios, com g₁ = 5,07. O motivo é uma armadilha "
            "bonita: em distribuição de cauda pesada, a própria cauda "
            "<b>infla o desvio padrão</b>, de modo que “meio desvio” vira um "
            "limiar enorme e nada o ultrapassa. A régua cresce junto com o "
            "que ela deveria medir. Trocamos o critério por |g₁| (&lt; 0,5 "
            "simétrica; &lt; 1 moderada; ≥ 1 forte), que é invariante de "
            "escala. O texto continua <i>reportando</i> média × mediana, que "
            "é a leitura intuitiva, mas quem <i>decide</i> é g₁.",
            estilos, "alerta"),
    ]
    itens += figura("fig02_boxplot_total.png", LARGURA_UTIL,
                    "Figura 2 — Boxplot do valor total com as cercas de "
                    "Tukey e os outliers destacados.", estilos)
    itens += [
        Paragraph(
            f"<b>Leitura.</b> Com Q₁ = {num(r['q1'])} e Q₃ = {num(r['q3'])}, "
            f"o IQR é {num(r['iqr'])} e a cerca superior cai em US$ "
            f"{num(r['limite_superior'])}. Acima dela há "
            f"<b>{num(r['n_outliers'], 0)} corridas "
            f"({num(r['pct_outliers'], 2)}%)</b>. São elas que a descoberta "
            "3 identifica.", estilos["corpo"]),
        PageBreak(),
    ]
    itens += figura("fig02_barras_distrito.png", LARGURA_UTIL,
                    "Figura 3 — Distribuição das corridas por distrito de "
                    "embarque (variável categórica).", estilos)
    itens += [
        Paragraph(
            "Para variáveis categóricas a tela mostra tabela de frequências "
            "com acumuladas e barras — ou pizza, quando há até cinco "
            "categorias; acima disso a pizza é omitida com justificativa em "
            "tela. Manhattan concentra <b>81,97%</b> dos embarques, Queens "
            "10,22%, Brooklyn 5,93% e Bronx 1,54%.", estilos["corpo"]),
    ]
    return itens


def secao_simulacao(estilos, resultados):
    lgn = resultados["lgn"]
    itens = [
        PageBreak(),
        Paragraph("5. Módulo 3 — Simulação de Monte Carlo", estilos["h1"]),
        Paragraph("5.1 Lei dos Grandes Números", estilos["h2"]),
    ]
    itens += figura("fig03_lgn.png", LARGURA_UTIL,
                    "Figura 4 — Convergência da frequência relativa para a "
                    "probabilidade teórica, em escala logarítmica.", estilos)
    itens.append(tabela([
        ["Após n lançamentos", "10", "100", "1.000",
         f"{num(lgn['n'], 0)}", "Teórica"],
        ["Frequência relativa de cara", num(lgn["f10"], 4),
         num(lgn["f100"], 4), num(lgn["f1000"], 4), num(lgn["final"], 4),
         "0,5000"],
    ], estilos, larguras=[5.4 * cm] + [(LARGURA_UTIL - 5.4 * cm) / 5] * 5,
        alinhar_direita=[1, 2, 3, 4, 5]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Erro final de {num(abs(lgn['final'] - 0.5), 4)}. O eixo x está "
            "em escala logarítmica de propósito: a convergência acontece por "
            "ordens de grandeza, e em escala linear os primeiros 50 "
            "lançamentos — justamente onde está a instabilidade interessante "
            "— ficariam espremidos contra a origem. <b>A leitura importante "
            "é que a convergência não é uma descida suave do erro: é uma "
            "oscilação que vai perdendo amplitude.</b> Mudando a semente, "
            "cada simulação zigue-zagueia diferente no começo — e todas "
            "terminam no mesmo lugar.", estilos["corpo"]),
        Paragraph("5.2 Teorema Central do Limite", estilos["h2"]),
    ]
    itens += figura("fig03_tcl.png", LARGURA_UTIL,
                    "Figura 5 — Distribuição das médias amostrais do valor "
                    "total, para três tamanhos de amostra, com a Normal "
                    "ajustada sobreposta.", estilos)
    linhas = [["n (tamanho da amostra)", "Média das médias",
               "Desvio observado", "σ/√n previsto", "g₁ das médias"]]
    for painel in resultados["tcl"]:
        linhas.append([
            str(painel["n"]), num(painel["media_das_medias"], 3),
            num(painel["desvio_observado"], 3),
            num(painel["sigma_sobre_raiz_n"], 3),
            num(painel["assimetria"], 3)])
    itens.append(tabela(linhas, estilos,
                        larguras=[4.4 * cm] + [(LARGURA_UTIL - 4.4 * cm) / 4] * 4,
                        alinhar_direita=[1, 2, 3, 4]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Sorteamos 2.000 amostras do valor total pago — uma população "
            "fortemente assimétrica (g₁ = 3,096; μ = 18,52; σ = 13,80) — e "
            "histogramamos as médias, cada uma calculada pela nossa "
            "<font face='DejaVu-Mono' size=9>ms.media</font>.",
            estilos["corpo"]),
        Paragraph(
            "<b>Três leituras.</b> (1) A média das médias fica colada em "
            "μ = 18,519 para qualquer n — a média amostral é um estimador "
            "<b>não viesado</b>. (2) O desvio observado acompanha σ/√n com "
            "erro abaixo de 5%: quadruplicar a amostra corta o erro pela "
            "metade. (3) A assimetria das médias <b>cai de 2,55 para "
            "0,57</b> conforme n vai de 2 a 30 — é literalmente a "
            "distribuição virando um sino, partindo de dados que são tudo "
            "menos normais. É por isso que a Normal aparece em todo lugar: "
            "não porque os fenômenos sejam normais, mas porque <b>médias</b> "
            "de qualquer coisa tendem a ser.", estilos["corpo"]),
        PageBreak(),
    ]
    return itens


def secao_distribuicoes(estilos, resultados):
    ajuste = resultados["ajuste_distancia"]
    poisson = resultados["poisson"]
    itens = [
        Paragraph("6. Módulo 4 — Distribuições teóricas", estilos["h1"]),
        Paragraph(
            "Os parâmetros são estimados <b>a partir dos próprios dados</b>, "
            "com as nossas funções: μ̂ = x̄ = 3,027 e σ̂ = s = 3,829 para a "
            "Normal; λ̂ = 1/x̄ = 0,3303 para a Exponencial. O histograma está "
            "em <b>densidade</b> (área = 1) — sem isso as escalas não "
            "bateriam e as curvas sumiriam rente ao eixo.",
            estilos["corpo"]),
    ]
    itens += figura("fig04_distancia_ajuste.png", LARGURA_UTIL,
                    "Figura 6 — Distância das corridas com as duas curvas "
                    "teóricas estimadas dos dados.", estilos)

    linhas = [["Distribuição", "Erro médio absoluto de densidade",
               "Maior erro em uma classe"]]
    for nome, valores in ajuste.items():
        linhas.append([nome, num(valores["erro_medio"], 6),
                       num(valores["erro_max"], 6)])
    itens.append(tabela(linhas, estilos,
                        larguras=[7 * cm, 5.5 * cm, LARGURA_UTIL - 12.5 * cm],
                        alinhar_direita=[1, 2]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>A Normal ajusta mal, e o motivo é identificável.</b> A "
            "distância tem g₁ = 3,007, longe do zero que a Normal "
            "pressupõe. Por ser simétrica, a curva normal coloca massa de "
            "probabilidade <b>à esquerda de zero</b> — região onde a "
            "variável nem pode existir, já que não há corrida de distância "
            "negativa — e ao mesmo tempo morre cedo demais na cauda direita, "
            "onde estão as corridas longas de aeroporto. A Exponencial, que "
            "já nasce assimétrica e ancorada em zero, erra <b>menos da "
            "metade</b>: descreve bem o decaimento geral, embora subestime o "
            "pico das corridas curtíssimas. Faz sentido teórico — a "
            "Exponencial é a distribuição natural de “distância até o "
            "próximo evento”, e uma corrida urbana é aproximadamente isso.",
            estilos["corpo"]),
        PageBreak(),
    ]
    itens += figura("fig04_poisson_passageiros.png", LARGURA_UTIL,
                    "Figura 7 — Número de passageiros: proporção observada "
                    "contra a Poisson com λ estimado dos dados.", estilos)

    chaves = sorted(poisson["observado"], key=int)
    linhas = [["k passageiros"] + chaves,
              ["Observado"] + [f"{num(100 * poisson['observado'][k], 2)}%"
                               for k in chaves],
              [f"Poisson(λ = {num(poisson['lambda'], 3)})"]
              + [f"{num(100 * poisson['teorico'][k], 2)}%" for k in chaves]]
    itens.append(tabela(linhas, estilos,
                        larguras=[4.2 * cm]
                        + [(LARGURA_UTIL - 4.2 * cm) / len(chaves)] * len(chaves),
                        alinhar_direita=list(range(1, len(chaves) + 1))))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Para a contagem de passageiros a candidata natural é a "
            "<b>Poisson</b>, com λ̂ = x̄. O ajuste é <b>ruim, e isso é o "
            "achado</b>: a Poisson prevê 21% de corridas com zero "
            "passageiros (observamos 1,5%) e 33% com um passageiro "
            "(observamos 73%). <b>Nenhum valor de λ consertaria isso</b> — o "
            "problema é conceitual. A Poisson descreve eventos raros e "
            "independentes ao longo de um intervalo; o número de passageiros "
            "de um táxi não é isso. É o tamanho de um grupo social, com um "
            "pico enorme em 1 (quem anda sozinho), um teto físico em 6 "
            "lugares e um segundo pico em 5 (grupos que ocupam o carro "
            "inteiro). Um modelo que não corresponde ao fenômeno não se "
            "salva com estimativa melhor de parâmetro.", estilos["corpo"]),
        PageBreak(),
    ]
    return itens


def secao_regressao(estilos, resultados):
    reg = resultados["regressao"]
    itens = [
        Paragraph("7. Módulo 5 — Correlação e regressão linear",
                  estilos["h1"]),
    ]
    itens += figura("fig05_regressao.png", LARGURA_UTIL,
                    "Figura 8 — Diagrama de dispersão de tarifa contra "
                    "distância, com a reta de mínimos quadrados e um ponto "
                    "de predição.", estilos)
    itens.append(tabela([
        ["Medida", "Valor"],
        ["Covariância", num(reg["cov"], 4)],
        ["r de Pearson", f"{num(reg['r'], 4)} (associação quase perfeita e "
                         "positiva)"],
        ["Reta de mínimos quadrados",
         f"ŷ = {num(reg['b1'], 4)}·x + {num(reg['b0'], 4)}"],
        ["R²", num(reg["r2"], 4)],
        ["Erro padrão da estimativa", num(reg["se"], 4)],
        ["Faixa observada de x", "[0,00 ; 36,70] milhas"],
    ], estilos, larguras=[6 * cm, LARGURA_UTIL - 6 * cm]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>Interpretação dos coeficientes.</b> Cada milha a mais está "
            f"<b>associada</b>, em média, a <b>US$ {num(reg['b1'])} a "
            f"mais</b> de tarifa. O intercepto b₀ = {num(reg['b0'])} é o "
            "valor previsto para uma corrida de distância zero — e aqui ele "
            "tem significado concreto, porque Nova York cobra uma "
            "<b>bandeirada</b> mais taxas fixas antes de o carro andar; "
            f"US$ {num(reg['b0'])} é uma estimativa plausível desse piso. O "
            f"R² = {num(reg['r2'], 4)} diz que a distância explica "
            f"<b>{num(100 * reg['r2'], 2)}%</b> da variação da tarifa; os "
            f"{num(100 * (1 - reg['r2']), 2)}% restantes vêm do que este "
            "modelo não vê — tempo parado no trânsito, sobretaxas de "
            "horário, tarifas fixas.", estilos["corpo"]),
        Paragraph(
            "Exemplo de predição: para <b>5 milhas</b> a reta prevê "
            f"<b>US$ {num(reg['previsao_5_milhas'])}</b>, com margem típica "
            f"de ± US$ {num(reg['se'])}. A aplicação avisa em vermelho se o "
            "usuário digitar um valor fora da faixa observada: extrapolar é "
            "chute vestido de matemática, porque a reta só foi validada "
            "dentro do intervalo dos dados.", estilos["corpo"]),
        PageBreak(),
    ]
    itens += figura("fig05_residuos.png", LARGURA_UTIL,
                    "Figura 9 — Resíduos da regressão. O funil que abre com "
                    "a distância indica variância crescente.", estilos)
    itens += [
        Paragraph(
            "<b>O que os resíduos denunciam.</b> Duas coisas. Primeira, o "
            "funil que abre com a distância — corridas longas erram mais, em "
            "valor absoluto, que corridas curtas (heterocedasticidade). "
            "Segunda, e mais interessante: há uma <b>faixa horizontal de "
            "tarifas idênticas em US$ 52,00</b> no diagrama de dispersão. "
            f"São <b>{num(reg['tarifa_fixa_n'], 0)} corridas</b>, e "
            f"<b>{num(reg['tarifa_fixa_pct_aero'], 1)}% delas tocam um "
            "aeroporto</b>. Não é erro nem coincidência: é a <b>tarifa fixa "
            "JFK ↔ Manhattan</b>, uma regra tarifária da cidade que ignora o "
            "taxímetro. Naquela faixa o preço simplesmente não cresce com a "
            "distância, e nenhum ajuste de b₀ ou b₁ resolve — o que faltaria "
            "é uma variável indicadora de “corrida com tarifa fixa”, isto é, "
            "outro modelo. É o tipo de coisa que R² nenhum revela e que só "
            "aparece quando se olha o gráfico.", estilos["corpo"]),
        caixa(
            "<b>⚠ Correlação não implica causalidade.</b> Um exemplo limpo "
            "dentro deste mesmo dataset: <b>pedágios e gorjeta</b> têm "
            f"correlação positiva (r = {num(reg['r_tolls_tip'], 3)}). "
            "Ninguém dá gorjeta porque pagou pedágio. As duas sobem juntas "
            "porque acompanham um terceiro fator — corridas longas, de "
            "aeroporto, que atravessam pontes e túneis e rendem contas "
            "altas. A variável escondida é a distância; pedágio e gorjeta "
            "apenas compartilham essa causa comum. Pela mesma lógica, mesmo "
            "o r = 0,92 entre distância e tarifa é associação: o que "
            "<i>causa</i> o preço é a regra tarifária da cidade, que por "
            "acaso usa a distância como insumo.", estilos, "alerta"),
        PageBreak(),
    ]
    return itens


def secao_descobertas(estilos, resultados):
    d1 = resultados["descoberta1"]
    d2 = resultados["descoberta2"]
    d3 = resultados["descoberta3"]
    itens = [
        Paragraph("8. As três descobertas", estilos["h1"]),
        Paragraph(
            "8.1 Quem paga em dinheiro nunca dá gorjeta, e o defeito é do "
            "instrumento", estilos["h2"]),
        caixa(
            f"<b>Afirmação.</b> Todas as {num(d1['cash']['n'], 0)} corridas "
            "pagas em dinheiro registram gorjeta exatamente zero. Isso não "
            "mede generosidade: mede o que o taxímetro consegue enxergar.",
            estilos),
    ]
    itens += figura("fig06_gorjeta_pagamento.png", LARGURA_UTIL * 0.85,
                    "Figura 10 — Gorjeta média por forma de pagamento.",
                    estilos)
    linhas = [["Forma de pagamento", "Corridas", "Gorjeta média",
               "Gorjeta mediana", "% com gorjeta zero"]]
    for nome in ("credit card", "cash", "Não informado"):
        if nome in d1:
            g = d1[nome]
            linhas.append([nome, num(g["n"], 0), f"US$ {num(g['media'])}",
                           f"US$ {num(g['mediana'])}",
                           f"{num(g['pct_zero'], 1)}%"])
    itens.append(tabela(linhas, estilos,
                        larguras=[4 * cm] + [(LARGURA_UTIL - 4 * cm) / 4] * 4,
                        alinhar_direita=[1, 2, 3, 4]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>Por que isso importa.</b> A leitura ingênua seria “quem paga "
            "em dinheiro é mão-fechada”. Ela está errada, e o dado que a "
            f"desmonta é o <b>{num(d1['cash']['pct_zero'], 1)}%</b>: não é "
            "uma tendência forte, é a totalidade — e comportamento humano "
            "não produz unanimidade. O taxímetro de Nova York só registra a "
            "gorjeta quando ela passa pela maquininha; gorjeta em espécie "
            "vai direto para o motorista e nunca entra no sistema. <b>O zero "
            "aqui significa “não medido”, não “não pago”.</b>",
            estilos["corpo"]),
        Paragraph(
            "<b>Limite honesto.</b> A consequência é que a gorjeta média de "
            "US$ 1,98 calculada sobre o dataset inteiro está "
            "<b>subestimada por construção</b>: ela divide o total "
            "arrecadado em cartão por <i>todas</i> as corridas, inclusive as "
            f"{num(d1['cash']['n'], 0)} em que a gorjeta existiu mas não foi "
            "vista. Qualquer conclusão sobre gorjeta neste dataset só vale "
            "<b>dentro do subconjunto pago com cartão</b> — e mesmo aí o "
            "recorte não é aleatório, porque quem escolhe pagar com cartão "
            "pode ser sistematicamente diferente de quem paga em espécie. É "
            "um erro que só aparece olhando a distribuição; a média sozinha "
            "o esconde.", estilos["corpo"]),
        PageBreak(),
        Paragraph("8.2 O número de passageiros não explica nada do preço",
                  estilos["h2"]),
        caixa(
            "<b>Afirmação.</b> A correlação entre número de passageiros e "
            f"tarifa é r = {num(d2['correlacoes']['fare'], 4)}, e a "
            f"regressão devolve R² = "
            f"{num(d2['r2_passageiros_tarifa'], 6)}. O número de pessoas no "
            f"carro explica <b>{num(100 * d2['r2_passageiros_tarifa'], 4)}%"
            "</b> da variação do preço.", estilos),
    ]
    itens += figura("fig06_matriz_correlacao.png", LARGURA_UTIL * 0.72,
                    "Figura 11 — Matriz de correlação. A linha de "
                    "Passageiros é a única pálida do mapa inteiro.", estilos)
    rotulos = {"distance": "Distância", "fare": "Tarifa", "tip": "Gorjeta",
               "tolls": "Pedágios", "total": "Total pago",
               "duracao_min": "Duração"}
    linhas = [["Variável", "r com nº de passageiros", "r²"]]
    for coluna, rotulo in rotulos.items():
        r = d2["correlacoes"][coluna]
        linhas.append([rotulo, f"{'+' if r >= 0 else '−'}{num(abs(r), 4)}",
                       num(r * r, 6)])
    itens.append(tabela(linhas, estilos,
                        larguras=[5 * cm, 5 * cm, LARGURA_UTIL - 10 * cm],
                        alinhar_direita=[1, 2]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Para efeito de comparação, na mesma matriz: distância × tarifa "
            "= 0,92; tarifa × total = 0,97; distância × duração = 0,82.",
            estilos["corpo"]),
        Paragraph(
            "<b>Por que isso é uma descoberta.</b> Porque a intuição de "
            "quase todo mundo prevê o contrário. Em ônibus, avião, trem e "
            "aplicativo com categoria por capacidade, mais gente custa mais "
            "caro. Em táxi de Nova York, não: a tarifa é do <b>veículo</b>, "
            "cobrada por distância percorrida e por tempo parado, e os "
            "passageiros adicionais viajam de graça. O dado não está com "
            "defeito — a intuição é que estava. Uma correlação "
            "<b>ausente</b> onde todos jurariam que existe costuma ser mais "
            "informativa que uma correlação forte esperada.",
            estilos["corpo"]),
        Paragraph(
            "<b>Limite honesto.</b> O r de Pearson mede associação "
            "<b>linear</b>. Um r ≈ 0 não prova ausência total de relação: "
            "poderia existir um efeito não linear — por exemplo, só grupos "
            "de 5 ou 6 pessoas mudarem algo — que Pearson não captura. O que "
            "torna essa hipótese pouco plausível é a <b>consistência</b>: a "
            "ausência se repete contra as seis variáveis numéricas, com "
            "sinais alternando entre positivo e negativo, que é exatamente o "
            "padrão de ruído em torno de zero. Ainda assim, provar ausência "
            "de relação exigiria um teste que não fizemos.",
            estilos["corpo"]),
        PageBreak(),
        Paragraph("8.3 Os outliers de preço têm endereço: são as corridas de "
                  "aeroporto", estilos["h2"]),
        caixa(
            f"<b>Afirmação.</b> {num(d3['pct_aero_outliers'], 1)}% das "
            "corridas marcadas como outliers de preço pela regra do IQR "
            "começam ou terminam em um aeroporto, contra "
            f"{num(d3['pct_aero_geral'], 1)}% no dataset inteiro — "
            f"concentração <b>{num(d3['razao'], 1)} vezes maior</b>. Os "
            "valores fora da curva não são ruído: são um segmento de "
            "mercado.", estilos),
    ]
    itens += figura("fig06_outliers_aeroporto.png", LARGURA_UTIL * 0.85,
                    "Figura 12 — Proporção de corridas com ponta em "
                    "aeroporto, dentro e fora do grupo de outliers.",
                    estilos)
    itens.append(tabela([
        ["Grupo", "n", "Total mediano", "% com pedágio",
         "% tocando aeroporto"],
        ["Outliers (IQR)", num(d3["n_outliers"], 0),
         f"US$ {num(d3['mediana_outliers'])}",
         f"{num(d3['pct_pedagio_outliers'], 1)}%",
         f"{num(d3['pct_aero_outliers'], 1)}%"],
        ["Demais corridas", num(6427 - d3["n_outliers"], 0),
         f"US$ {num(d3['mediana_demais'])}",
         f"{num(d3['pct_pedagio_demais'], 1)}%",
         f"{num(d3['pct_aero_geral'], 1)}%"],
    ], estilos, larguras=[4 * cm] + [(LARGURA_UTIL - 4 * cm) / 4] * 4,
        alinhar_direita=[1, 2, 3, 4]))
    itens += [
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>Evidência.</b> A regra de Tukey sobre o valor total (cerca "
            f"superior em US$ {num(d3['limite_superior'])}) marca "
            f"{num(d3['n_outliers'], 0)} corridas "
            f"({num(d3['pct_outliers'], 2)}%) como outliers. O pedágio "
            "confirma por um caminho independente: ele aparece em "
            f"{num(d3['pct_pedagio_outliers'], 1)}% dos outliers e em apenas "
            f"{num(d3['pct_pedagio_demais'], 1)}% das demais corridas — são "
            "viagens que atravessam pontes e túneis para sair da ilha de "
            "Manhattan. E a confirmação mais visual está na Figura 8: a "
            "faixa horizontal de 131 corridas com tarifa idêntica de "
            "US$ 52,00 é a tarifa fixa JFK ↔ Manhattan, uma regra tarifária "
            "<b>visível a olho nu no diagrama de dispersão</b>.",
            estilos["corpo"]),
        Paragraph(
            f"<b>Por que importa.</b> Esses {num(d3['n_outliers'], 0)} "
            "pontos não são erro de medição, e descartá-los como “ruído” — o "
            "reflexo automático de quem vê a palavra <i>outlier</i> — "
            "apagaria um segmento real e economicamente relevante do "
            "serviço. A conclusão prática é mais forte do que “há "
            "outliers”: <b>não existe uma distribuição de preço de táxi em "
            "Nova York, existem duas populações misturadas.</b> O trajeto "
            f"urbano curto, com mediana de US$ {num(d3['mediana_demais'])}, "
            "e a corrida de aeroporto, com mediana de US$ "
            f"{num(d3['mediana_outliers'])} — quase quatro vezes mais. Uma "
            "média calculada sobre a mistura não descreve nem uma nem "
            "outra.", estilos["corpo"]),
        Paragraph(
            "<b>Limite honesto.</b> Três ressalvas. (1) “Aeroporto” aqui é a "
            "zona de embarque ou desembarque <b>registrada</b>; corridas "
            "de/para Newark aparecem pouco por ser um aeroporto fora da "
            "cidade. (2) “Outlier pela regra do IQR” é uma <b>convenção</b> "
            "— o fator 1,5 —, não uma verdade da natureza; com 3,0·IQR o "
            "recorte seria outro. (3) Os "
            f"{num(100 - d3['pct_aero_outliers'], 1)}% de outliers que "
            "<b>não</b> tocam aeroporto continuam sem explicação nesta "
            "análise; provavelmente são corridas longas entre distritos, mas "
            "não testamos.", estilos["corpo"]),
        PageBreak(),
    ]
    return itens


def secao_limitacoes(estilos, resultados):
    tratamento = resultados["tratamento"]
    limitacoes = [
        ("Não vale para outros períodos.",
         "A amostra cobre 28/02/2019 a 31/03/2019: um mês de fim de "
         "inverno, antes da pandemia. Sazonalidade, feriados, clima e a "
         "mudança de comportamento urbano pós-2020 estão fora do alcance "
         "destes dados. Nada aqui descreve o táxi de Nova York “em geral” — "
         "descreve março de 2019."),
        ("Não é a população de corridas.",
         f"São {num(tratamento['linhas_finais'], 0)} corridas de um universo "
         "de milhões por mês. É uma amostra pública e <b>não sabemos qual "
         "processo a gerou</b> — se foi sorteio aleatório simples, se houve "
         "filtro por região ou por tipo de corrida. Sem isso, nenhuma "
         "extrapolação formal para a cidade inteira é defensável, e nenhum "
         "intervalo de confiança que calculássemos teria a cobertura que "
         "anuncia."),
        ("Nada aqui é causal.",
         "Todo o trabalho é de associação. A regressão prevê bem a tarifa a "
         "partir da distância, mas não prova que a distância <i>causa</i> o "
         "preço — o que causa o preço é a regra tarifária municipal. O "
         "exemplo de pedágio × gorjeta (r = 0,414) mostra como duas "
         "variáveis sobem juntas sem qualquer relação direta."),
        ("O zero da gorjeta é ambíguo.",
         "Conforme a descoberta 1, “gorjeta = 0” pode significar “não pagou” "
         "ou “pagou em espécie e o sistema não viu”. Isso contamina toda e "
         "qualquer conclusão sobre generosidade do passageiro, e subestima "
         "por construção as médias de gorjeta e de valor total."),
        ("Mantivemos dados provavelmente defeituosos.",
         f"As {tratamento['distancia_zero']} corridas com distância zero e "
         f"as {tratamento['passageiros_zero']} com zero passageiros foram "
         "preservadas por não serem fisicamente impossíveis, mas quase "
         "certamente são falhas de registro. Elas puxam levemente para baixo "
         "as medidas dessas duas variáveis — o efeito é pequeno (0,7% e 1,5% "
         "das linhas), mas existe."),
        ("Só testamos relações lineares.",
         "O r de Pearson e a reta de mínimos quadrados não enxergam curvas, "
         "patamares ou efeitos de limiar. A própria tarifa fixa de US$ 52,00 "
         "é a prova: é uma estrutura real nos dados que o modelo linear não "
         "consegue representar."),
        ("A regra do IQR tem pressupostos.",
         "Ela assume dispersão no miolo da distribuição. Em variáveis "
         "infladas de zeros, como os pedágios (75% das corridas pagam zero, "
         "IQR = 0), a regra degenera e classifica como “outlier” qualquer "
         "valor não nulo. A aplicação sinaliza esse caso, mas ele mostra que "
         "“outlier” é um resultado de método, não uma propriedade "
         "intrínseca do dado."),
        ("A validação é contra NumPy/SciPy, não contra a verdade.",
         "Se o NumPy e a nossa implementação errassem a mesma fórmula da "
         "mesma forma, os testes passariam. Isso é mitigado pelos testes de "
         "propriedade — identidades como R² = r², somas de probabilidade "
         "iguais a 1, área do histograma igual a 1 — e pelos casos de borda, "
         "mas não é eliminado."),
    ]
    itens = [
        Paragraph("9. Limitações — o que esta análise <i>não</i> permite "
                  "concluir", estilos["h1"]),
        Paragraph(
            "A parte mais fácil de um trabalho estatístico é afirmar; a mais "
            "honesta é delimitar. Oito coisas que os nossos números não "
            "autorizam:", estilos["corpo"]),
    ]
    for i, (titulo, texto) in enumerate(limitacoes, start=1):
        itens.append(Paragraph(f"<b>{i}. {titulo}</b> {texto}",
                               estilos["corpo"]))
    itens.append(PageBreak())
    return itens


def secao_reprodutibilidade_e_video(estilos):
    # &nbsp; porque o HTML do reportlab colapsa espaços repetidos e os
    # comentários alinhados perderiam a coluna.
    def alinhar(comando, comentario, coluna=36):
        espacos = "&nbsp;" * max(2, coluna - len(comando))
        return f"{comando}{espacos}# {comentario}"

    comandos = "<br/>".join([
        "git clone https://github.com/Joao-G14/Laboratorio-estatistico.git",
        "cd Laboratorio-estatistico",
        "python -m venv .venv &amp;&amp; .venv\\Scripts\\activate",
        "pip install -r requirements.txt",
        "",
        alinhar("pytest -v", "104 testes"),
        alinhar("python verificar_regra_de_ouro.py", "auditoria da regra de ouro"),
        alinhar("python gerar_figuras.py", "refaz as figuras deste relatório"),
        alinhar("streamlit run app.py", "a aplicação"),
    ])
    itens = [
        Paragraph("10. Reprodutibilidade", estilos["h1"]),
        Paragraph(
            "O dataset está versionado no repositório e todas as simulações "
            "usam <b>semente fixa</b> (42 nas figuras deste relatório), "
            "então os números acima são reproduzíveis exatamente. As figuras "
            "deste PDF e os gráficos da aplicação saem do <b>mesmo "
            "módulo</b> <font face='DejaVu-Mono' size=9>graficos.py</font>, "
            "alimentado pelo <b>mesmo</b> "
            "<font face='DejaVu-Mono' size=9>minhastats.py</font>: não "
            "existem duas implementações que possam divergir. E as tabelas "
            "numéricas deste documento são lidas dos arquivos JSON gerados "
            "pelos scripts — nenhum número foi digitado à mão.",
            estilos["corpo"]),
        Paragraph(
            f"<font face='DejaVu-Mono' size=8.5>{comandos}</font>",
            ParagraphStyle("cmd", parent=estilos["corpo"],
                           backColor=colors.HexColor("#0e1621"),
                           textColor=colors.HexColor("#e7edf4"),
                           borderPadding=10, leading=13, alignment=0,
                           spaceBefore=6, spaceAfter=14)),
        Spacer(1, 1.2 * cm),
        Paragraph("Links da entrega", estilos["h1"]),
    ]

    video_preenchido = "PREENCHER" not in identificacao.LINK_VIDEO
    video_html = (f'<link href="{identificacao.LINK_VIDEO}">'
                  f'{identificacao.LINK_VIDEO}</link>'
                  if video_preenchido else identificacao.LINK_VIDEO)

    # ▶ em vez de um emoji: a DejaVu Sans não tem emojis, e o caractere
    # ausente sairia como um retângulo vazio no PDF.
    destaque = Table([[Paragraph(
        "▶ &nbsp; <b>VÍDEO DE DEMONSTRAÇÃO (3–5 min)</b><br/><br/>"
        f"<font size=11>{video_html}</font>", estilos["video"])]],
        colWidths=[LARGURA_UTIL], hAlign="LEFT")
    destaque.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 2, AZUL),
        ("BACKGROUND", (0, 0), (-1, -1), FUNDO_SUAVE),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    itens += [destaque, Spacer(1, 0.7 * cm)]

    itens.append(tabela([
        ["Item", "Link"],
        ["Fonte original do dataset (NYC TLC)",
         f'<link href="{identificacao.LINK_DATASET_ORIGINAL}">'
         f'{identificacao.LINK_DATASET_ORIGINAL}</link>'],
        ["Arquivo CSV utilizado",
         f'<link href="{identificacao.LINK_DATASET_ARQUIVO}">'
         f'{identificacao.LINK_DATASET_ARQUIVO}</link>'],
        ["Repositório público (código, testes e relatório)",
         f'<link href="{identificacao.LINK_REPOSITORIO}">'
         f'{identificacao.LINK_REPOSITORIO}</link>'],
    ], estilos, larguras=[6.5 * cm, LARGURA_UTIL - 6.5 * cm]))

    integrantes = "<br/>".join(
        f"{nome} — matrícula {matricula}"
        for nome, matricula in identificacao.INTEGRANTES)
    itens += [
        Spacer(1, 0.7 * cm),
        Paragraph(f"<b>Grupo:</b> {identificacao.NOME_GRUPO} "
                  f"(entrega individual)", estilos["corpo"]),
        Paragraph("<b>Integrante(s)</b>", estilos["h3"]),
        Paragraph(integrantes, estilos["corpo"]),
    ]
    return itens


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------


def main():
    registrar_fontes()
    estilos = criar_estilos()

    with open(os.path.join(PASTA, "resultados.json"), encoding="utf-8") as f:
        resultados = json.load(f)
    with open(os.path.join(PASTA, "validacao.json"), encoding="utf-8") as f:
        validacao = json.load(f)

    faltando = identificacao.pendencias()
    if faltando:
        print("AVISO: ainda faltam preencher em identificacao.py -> "
              + ", ".join(sorted(set(faltando))))

    saida = identificacao.nome_do_arquivo_pdf()
    doc = BaseDocTemplate(
        saida, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2.2 * cm,
        title="Sistematização — Laboratório Estatístico Interativo",
        author=identificacao.INTEGRANTES[0][0],
        subject="Matemática e Estatística para Computação",
    )
    quadro = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                   id="normal")
    doc.addPageTemplates([
        PageTemplate(id="capa", frames=[quadro], onPage=fundo_capa),
        PageTemplate(id="corpo", frames=[quadro], onPage=rodape),
    ])

    historia = montar_capa(estilos)
    historia += secao_resumo(estilos, resultados)
    historia += secao_dataset(estilos, resultados)
    historia += secao_nucleo(estilos, validacao)
    historia += secao_descritiva(estilos, resultados)
    historia += secao_simulacao(estilos, resultados)
    historia += secao_distribuicoes(estilos, resultados)
    historia += secao_regressao(estilos, resultados)
    historia += secao_descobertas(estilos, resultados)
    historia += secao_limitacoes(estilos, resultados)
    historia += secao_reprodutibilidade_e_video(estilos)

    doc.build(historia)
    print(f"PDF gerado: {saida}")


if __name__ == "__main__":
    main()
