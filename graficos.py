"""
graficos.py — camada de visualização do laboratório.

Separado de `minhastats.py` (que só calcula) e de `app.py` (que só monta a
interface), para que a aplicação Streamlit e o gerador de figuras do
relatório desenhem exatamente os mesmos gráficos, com a mesma identidade
visual, sem duplicar código.

Nenhum número é calculado aqui: todas as medidas desenhadas chegam prontas,
vindas de `minhastats.py`. Este módulo apenas posiciona tinta.

Decisões visuais (aplicadas em todos os gráficos):
  * paleta categórica de hues fixos, atribuídos por ordem e nunca reciclados;
  * eixos e grade discretos (hairline cinza) para que o dado fique em primeiro
    plano; molduras superior e direita removidas;
  * barras separadas por um fio da cor do fundo, para leitura de contorno;
  * legenda sempre presente quando há duas ou mais séries;
  * um único eixo y por gráfico — nunca eixo duplo.
"""

import matplotlib

matplotlib.use("Agg")                   # backend sem janela (Streamlit e PDF)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_FRACA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
ALERTA = "#d03b3b"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_2,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "xtick.color": TINTA_FRACA,
    "ytick.color": TINTA_FRACA,
    "legend.frameon": False,
    "figure.dpi": 130,
})


def _num(valor, casas=2):
    """Formata no padrão brasileiro: 1234.5 -> '1.234,50'.

    O símbolo § é só um separador temporário para trocar ponto e vírgula
    de lugar sem que uma substituição desfaça a outra.
    """
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def _limpar(ax, grade="y"):
    """Tira as molduras supérfluas e deixa a grade só no eixo que ajuda a ler."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(EIXO)
    ax.spines["bottom"].set_color(EIXO)
    ax.grid(axis=grade, alpha=0.9)
    ax.set_axisbelow(True)


def _formatar_milhares(ax, eixo="y"):
    """1234 -> 1.234, no padrão brasileiro."""
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", "."))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Módulo 2 — descritiva
# ---------------------------------------------------------------------------


def histograma(dados, classes, titulo, rotulo_x, media=None, mediana=None):
    """Histograma desenhado a partir da tabela de classes de minhastats.

    As alturas das barras vêm da tabela de frequências (regra de Sturges)
    calculada pelo núcleo — o matplotlib não conta nada aqui.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    esquerdas = [c["inferior"] for c in classes]
    larguras = [c["superior"] - c["inferior"] for c in classes]
    alturas = [c["fi"] for c in classes]
    ax.bar(esquerdas, alturas, width=larguras, align="edge",
           color=SERIE[0], edgecolor=SUPERFICIE, linewidth=1.2)

    if media is not None:
        ax.axvline(media, color=SERIE[1], linewidth=2,
                   label=f"média = {_num(media)}")
    if mediana is not None:
        ax.axvline(mediana, color=SERIE[2], linewidth=2, linestyle=(0, (5, 3)),
                   label=f"mediana = {_num(mediana)}")
    if media is not None or mediana is not None:
        ax.legend(loc="upper right", fontsize=8.5, labelcolor=TINTA_2)

    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("frequência absoluta")
    _formatar_milhares(ax)
    _limpar(ax)
    fig.tight_layout()
    return fig


def boxplot(dados, outliers, limite_inferior, limite_superior,
            titulo, rotulo_x):
    """Boxplot com as cercas de Tukey e os outliers destacados."""
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.boxplot(
        dados, orientation="horizontal", widths=0.45, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#cde2fb", edgecolor=SERIE[0], linewidth=1.4),
        medianprops=dict(color=SERIE[1], linewidth=2),
        whiskerprops=dict(color=SERIE[0], linewidth=1.4),
        capprops=dict(color=SERIE[0], linewidth=1.4),
    )
    if outliers:
        ax.plot(outliers, [1] * len(outliers), "o", markersize=4.5,
                color=ALERTA, alpha=0.55, markeredgewidth=0,
                label=f"{_num(len(outliers), 0)} outliers (regra do IQR)")
    ax.axvline(limite_superior, color=TINTA_FRACA, linestyle=(0, (2, 3)),
               linewidth=1.3,
               label=f"Q3 + 1,5·IQR = {_num(limite_superior)}")
    if limite_inferior > min(dados):
        ax.axvline(limite_inferior, color=TINTA_FRACA, linestyle=(0, (2, 3)),
                   linewidth=1.3)
    ax.legend(loc="upper right", fontsize=8.5, labelcolor=TINTA_2)
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_yticks([])
    _limpar(ax, grade="x")
    fig.tight_layout()
    return fig


def barras_categoricas(tabela, titulo, rotulo_valor="corridas", maximo=12):
    """Barras horizontais para uma variável categórica.

    Acima de `maximo` categorias as demais são somadas em "Outras": hues
    categóricos não são recicláveis, e 200 barras não comunicam nada.
    """
    linhas = list(tabela)
    if len(linhas) > maximo:
        resto = sum(linha["fi"] for linha in linhas[maximo:])
        total = sum(linha["fi"] for linha in linhas)
        linhas = linhas[:maximo] + [{"categoria": "Outras", "fi": resto,
                                     "fri": resto / total}]
    categorias = [str(linha["categoria"]) for linha in linhas][::-1]
    valores = [linha["fi"] for linha in linhas][::-1]

    altura = max(2.4, 0.34 * len(categorias) + 1.0)
    fig, ax = plt.subplots(figsize=(6.6, altura))
    cores = [TINTA_FRACA if c == "Outras" else SERIE[0] for c in categorias]
    ax.barh(categorias, valores, height=0.68, color=cores)
    limite = max(valores)
    for y, v in enumerate(valores):
        ax.text(v + limite * 0.015, y, _num(v, 0),
                va="center", fontsize=8.5, color=TINTA_2)
    ax.set_xlim(0, limite * 1.16)
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_valor)
    _limpar(ax, grade="x")
    fig.tight_layout()
    return fig


def pizza_categoricas(tabela, titulo):
    """Pizza — só faz sentido com poucas categorias (usamos até 5)."""
    linhas = list(tabela)[:5]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.pie(
        [linha["fi"] for linha in linhas],
        labels=[str(linha["categoria"]) for linha in linhas],
        autopct="%1.1f%%", startangle=90, counterclock=False,
        colors=SERIE[:len(linhas)],
        wedgeprops=dict(edgecolor=SUPERFICIE, linewidth=2),
        textprops=dict(color=TINTA_2, fontsize=9),
    )
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Módulo 3 — simulação
# ---------------------------------------------------------------------------


def grafico_lgn(frequencias, probabilidade_teorica, titulo, rotulo_evento):
    """Lei dos Grandes Números: frequência relativa acumulada × nº de ensaios.

    Eixo x em escala log porque a convergência acontece por ordens de
    grandeza: em escala linear os primeiros 50 ensaios — justamente onde está
    a instabilidade interessante — ficariam espremidos contra a origem.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(range(1, len(frequencias) + 1), frequencias,
            color=SERIE[0], linewidth=1.4, label="frequência relativa simulada")
    ax.axhline(probabilidade_teorica, color=ALERTA, linestyle=(0, (5, 3)),
               linewidth=1.8,
               label=f"probabilidade teórica = "
                     f"{_num(probabilidade_teorica, 4)}")
    ax.set_xscale("log")
    ax.set_title(titulo)
    ax.set_xlabel("nº de repetições (escala log)")
    ax.set_ylabel(f"freq. relativa de {rotulo_evento}")
    ax.legend(loc="upper right", fontsize=8.5, labelcolor=TINTA_2)
    _limpar(ax)
    fig.tight_layout()
    return fig


def grafico_tcl(paineis, titulo_geral):
    """Teorema Central do Limite: um painel por tamanho de amostra.

    `paineis` é uma lista de dicionários com as chaves:
        n, classes (tabela de frequências das médias, já em densidade),
        curva_x, curva_y (Normal teórica), media, desvio.
    """
    fig, eixos = plt.subplots(1, len(paineis), figsize=(9.6, 3.0))
    if len(paineis) == 1:
        eixos = [eixos]
    for ax, painel in zip(eixos, paineis):
        classes = painel["classes"]
        esquerdas = [c["inferior"] for c in classes]
        larguras = [c["superior"] - c["inferior"] for c in classes]
        alturas = [c["densidade"] for c in classes]
        ax.bar(esquerdas, alturas, width=larguras, align="edge",
               color=SERIE[0], edgecolor=SUPERFICIE, linewidth=0.8)
        ax.plot(painel["curva_x"], painel["curva_y"], color=SERIE[1],
                linewidth=2)
        ax.set_title(f"amostras de n = {painel['n']}", fontsize=9.5)
        ax.set_yticks([])
        ax.set_xlabel("média amostral")
        _limpar(ax, grade="x")
    fig.suptitle(titulo_geral, fontsize=10.5, color=TINTA, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Módulo 4 — distribuições teóricas
# ---------------------------------------------------------------------------


def histograma_com_curvas(classes, curvas, titulo, rotulo_x):
    """Histograma em DENSIDADE com curvas teóricas sobrepostas.

    Densidade (e não frequência) é obrigatório: só assim a área do
    histograma soma 1 e fica na mesma escala das densidades teóricas —
    caso contrário a curva "some" rente ao eixo.

    `curvas` é uma lista de (rótulo, xs, ys).
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    esquerdas = [c["inferior"] for c in classes]
    larguras = [c["superior"] - c["inferior"] for c in classes]
    alturas = [c["densidade"] for c in classes]
    ax.bar(esquerdas, alturas, width=larguras, align="edge",
           color="#b7d3f6", edgecolor=SUPERFICIE, linewidth=1.0,
           label="dados observados")
    for i, (rotulo, xs, ys) in enumerate(curvas):
        ax.plot(xs, ys, color=SERIE[i + 1], linewidth=2, label=rotulo)
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("densidade")
    ax.legend(loc="upper right", fontsize=8.5, labelcolor=TINTA_2)
    _limpar(ax)
    fig.tight_layout()
    return fig


def barras_observado_teorico(categorias, observado, teorico, rotulo_teorico,
                             titulo, rotulo_x):
    """Comparação observado × teórico para uma variável DISCRETA.

    Duas séries lado a lado, com legenda — nunca um eixo duplo.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    posicoes = list(range(len(categorias)))
    largura = 0.40
    ax.bar([p - largura / 2 - 0.01 for p in posicoes], observado, largura,
           color=SERIE[0], label="observado (dados)")
    ax.bar([p + largura / 2 + 0.01 for p in posicoes], teorico, largura,
           color=SERIE[1], label=rotulo_teorico)
    ax.set_xticks(posicoes)
    ax.set_xticklabels([str(c) for c in categorias])
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("proporção")
    ax.legend(fontsize=8.5, labelcolor=TINTA_2)
    _limpar(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Módulo 5 — regressão
# ---------------------------------------------------------------------------


def dispersao_com_reta(x, y, b0, b1, r, r2, rotulo_x, rotulo_y,
                       ponto_previsto=None):
    """Diagrama de dispersão com a reta de mínimos quadrados por cima."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, y, "o", markersize=3.2, color=SERIE[0], alpha=0.28,
            markeredgewidth=0, label=f"{_num(len(x), 0)} corridas")

    x_min, x_max = min(x), max(x)
    reta_x = [x_min, x_max]
    reta_y = [b0 + b1 * x_min, b0 + b1 * x_max]
    sinal = "+" if b0 >= 0 else "−"
    equacao = f"ŷ = {_num(b1, 4)}·x {sinal} {_num(abs(b0), 4)}"
    ax.plot(reta_x, reta_y, color=SERIE[1], linewidth=2.4,
            label=f"{equacao}   (R² = {_num(r2, 3)})")

    if ponto_previsto is not None:
        px, py = ponto_previsto
        # Slot 3 (e não o amarelo): azul, laranja e água formam um trio que
        # se separa mesmo para quem tem daltonismo; amarelo ao lado de
        # laranja, não.
        ax.plot([px], [py], "o", markersize=9, color=SERIE[2],
                markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=5,
                label=f"predição: x = {_num(px)} → ŷ = {_num(py)}")

    ax.set_title(f"{rotulo_y} × {rotulo_x}   |   r de Pearson = {_num(r, 4)}")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel(rotulo_y)
    ax.legend(loc="upper left", fontsize=8.5, labelcolor=TINTA_2)
    _limpar(ax, grade="both")
    fig.tight_layout()
    return fig


def grafico_residuos(x, residuos, rotulo_x):
    """Resíduos × x — o diagnóstico que revela se a reta era adequada.

    Uma nuvem sem padrão em torno do zero é bom sinal; um funil que abre
    indica variância crescente (heterocedasticidade).
    """
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(x, residuos, "o", markersize=2.8, color=SERIE[2], alpha=0.25,
            markeredgewidth=0)
    ax.axhline(0, color=ALERTA, linewidth=1.6)
    ax.set_title("Resíduos da regressão (y observado − ŷ previsto)")
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("resíduo")
    _limpar(ax, grade="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Módulo 6 — descobertas
# ---------------------------------------------------------------------------


def barras_comparativas(categorias, valores, titulo, rotulo_valor,
                        destaque=None, casas=2):
    """Barras de uma medida por grupo — o formato das descobertas por contraste."""
    fig, ax = plt.subplots(figsize=(6.6, max(2.4, 0.5 * len(categorias) + 1.4)))
    cores = [SERIE[1] if (destaque and c == destaque) else SERIE[0]
             for c in categorias]
    ax.barh(list(categorias)[::-1], list(valores)[::-1], height=0.6,
            color=cores[::-1])
    limite = max(valores) if max(valores) > 0 else 1
    for y, v in enumerate(list(valores)[::-1]):
        ax.text(v + limite * 0.02, y, _num(v, casas),
                va="center", fontsize=9, color=TINTA_2)
    ax.set_xlim(0, limite * 1.22)
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_valor)
    _limpar(ax, grade="x")
    fig.tight_layout()
    return fig


def matriz_correlacao(rotulos, matriz, titulo):
    """Mapa de calor divergente da matriz de correlação.

    Escala divergente azul↔vermelho com cinza no meio, porque o zero aqui
    é um ponto de virada de sentido (associação negativa × positiva), não
    apenas "pouco".
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    imagem = ax.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rotulos)))
    ax.set_yticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(rotulos, fontsize=8.5)
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            valor = matriz[i][j]
            if abs(valor) < 0.005:       # evita imprimir "-0,00"
                valor = 0.0
            ax.text(j, i, f"{valor:.2f}".replace(".", ","),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if abs(valor) > 0.55 else TINTA)
    ax.set_title(titulo)
    ax.grid(False)
    barra = fig.colorbar(imagem, ax=ax, shrink=0.82)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8, colors=TINTA_FRACA)
    fig.tight_layout()
    return fig
