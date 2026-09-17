"""
gerar_prints.py — capturas de tela da aplicação em funcionamento.

Sobe o Streamlit em uma porta livre, abre cada módulo em um navegador sem
janela e salva um PNG por módulo em `assets/print_*.png`. São essas
imagens que ilustram o README e o relatório.

Dependência extra, fora do requirements.txt porque só serve para gerar a
documentação:

    pip install playwright
    python -m playwright install chromium

Uso:  python gerar_prints.py
"""

import os
import socket
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

PASTA = "assets"
ESPERA_APP = 25          # segundos para o servidor subir
ESPERA_MODULO = 6        # segundos para o módulo terminar de rodar

PAGINAS = [
    ("Módulo 0", "print_modulo0_dados.png"),
    ("Módulo 1", "print_modulo1_nucleo.png"),
    ("Módulo 2", "print_modulo2_descritiva.png"),
    ("Módulo 3", "print_modulo3_simulacao.png"),
    ("Módulo 4", "print_modulo4_distribuicoes.png"),
    ("Módulo 5", "print_modulo5_regressao.png"),
    ("Módulo 6", "print_modulo6_descobertas.png"),
]


def porta_livre():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    os.makedirs(PASTA, exist_ok=True)
    porta = porta_livre()
    servidor = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(porta),
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        print(f"Servidor subindo na porta {porta}...")
        time.sleep(ESPERA_APP)
        with sync_playwright() as p:
            navegador = p.chromium.launch()
            pagina = navegador.new_page(
                viewport={"width": 1500, "height": 1000},
                device_scale_factor=2,          # captura nítida para o PDF
            )
            pagina.goto(f"http://localhost:{porta}", wait_until="networkidle",
                        timeout=90_000)
            time.sleep(ESPERA_MODULO)

            for prefixo, arquivo in PAGINAS:
                alvo = pagina.locator(f'label:has-text("{prefixo}")').first
                alvo.click()
                time.sleep(ESPERA_MODULO)
                pagina.wait_for_load_state("networkidle")
                caminho = os.path.join(PASTA, arquivo)
                pagina.screenshot(path=caminho, full_page=False)
                print(f"  print: {caminho}")

            navegador.close()
    finally:
        servidor.terminate()
        servidor.wait(timeout=20)
        print("Servidor encerrado.")


if __name__ == "__main__":
    main()
