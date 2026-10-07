"""
Gera o `manual-impresso.pdf` a partir do `manual-impresso.html`.

    pip install playwright
    python -m playwright install chromium
    python preparar_pdf.py

Antes de imprimir, confere o que o olho nao pega numa leitura rapida: se o
conteudo de alguma pagina passou da altura da folha. Como a paginacao do
manual e explicita, uma <section class="pagina"> que transborda nao vira
duas paginas no PDF, ela vira uma pagina com o fim do texto cortado, sem
aviso nenhum. A checagem compara scrollHeight com clientHeight de cada
folha e recusa gerar o arquivo se alguma estourar.
"""

import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(AQUI, "manual-impresso.html")
SAIDA = os.path.join(AQUI, "manual-impresso.pdf")

MEDIDOR = """
() => Array.from(document.querySelectorAll('.pagina')).map((folha, i) => {
  const corpo = folha.querySelector('.corpo');
  return {
    n: i + 1,
    titulo: (folha.querySelector('.titulo-doc') || {}).innerText || '',
    sobra: folha.scrollHeight - folha.clientHeight,
    sobraCorpo: corpo ? corpo.scrollHeight - corpo.clientHeight : 0,
  };
})
"""


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright nao instalado. Rode:\n"
            "    pip install playwright\n"
            "    python -m playwright install chromium"
        )

    if not os.path.exists(ENTRADA):
        raise SystemExit("nao achei %s" % ENTRADA)

    from pathlib import Path

    with sync_playwright() as p:
        navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(Path(ENTRADA).resolve().as_uri())
        pagina.emulate_media(media="print")
        pagina.wait_for_load_state("networkidle")

        folhas = pagina.evaluate(MEDIDOR)
        estouradas = [f for f in folhas
                      if f["sobra"] > 1 or f["sobraCorpo"] > 1]

        if estouradas:
            print("conteudo passou da folha:", file=sys.stderr)
            for f in estouradas:
                print("  pagina %d (%s): %d px a mais"
                      % (f["n"], f["titulo"].replace("\n", " / ").strip(),
                         max(f["sobra"], f["sobraCorpo"])), file=sys.stderr)
            navegador.close()
            raise SystemExit("PDF nao gerado. Corte texto ou reduza figura.")

        pagina.pdf(path=SAIDA, print_background=True,
                   prefer_css_page_size=True)
        navegador.close()

    print("%d folhas, nenhuma estourada." % len(folhas))
    print(os.path.relpath(SAIDA, AQUI))


if __name__ == "__main__":
    main()
