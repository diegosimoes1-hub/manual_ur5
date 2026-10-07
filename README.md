# manual_ur5

Manual de operação do **UR5 CB2 com PolyScope 1.8**, escrito para quem nunca
operou um UR5: o robô e seus números, a tela do
[`cobot_testing_ur5_app`](https://github.com/ricardobertolin/cobot_testing_ur5_app)
passo a passo, e a operação pelo teach pendant sem software nenhum. Ligar,
inicializar as seis juntas, a aba `Move`, o freedrive e o desligamento, nas
telas do CB2, que é onde a maioria dos tutoriais de internet erra por serem
de CB3.

- [`manual.html`](manual.html) para ler na tela, com links.
- [`manual-impresso.pdf`](manual-impresso.pdf), 17 páginas em tamanho Carta,
  preto e branco, no formato de folha de fabricante: ficha técnica, nota de
  aplicação e guias de operação, cada um com código no rodapé e numeração
  própria. É para sair no papel e ficar na célula, onde não há navegador.

## A paginação é explícita

Uma `<section class="pagina">` por folha. Deixar o navegador quebrar sozinho
põe corte no meio de procedimento, e quem está com o pendant na mão não vira
página para achar o passo 7. O custo disso é que uma página que transborda
não vira duas, vira uma com o fim cortado e sem aviso: por isso o
`preparar_pdf.py` compara `scrollHeight` com `clientHeight` de cada folha e
**se recusa a gerar o arquivo** se alguma estourar.

## As figuras vêm do software, não do desenho

As vistas do robô saem do mesmo cache de malhas que o twin do aplicativo
consome e da mesma cadeia cinemática do `modelo_ur5.py`. As cotas da figura
são lidas do código, e não digitadas no texto: cota de manual que discorda
do software é pior que cota nenhuma.

Por isso o `preparar_manual.py` depende do repositório do aplicativo, que
não está aqui. Copiar o modelo para cá seria exatamente o jeito de as duas
coisas divergirem com o tempo. Deixe os dois checkouts lado a lado:

```
git clone https://github.com/ricardobertolin/cobot_testing_ur5_app
pip install matplotlib playwright && python -m playwright install chromium
python preparar_manual.py     # as figuras, em figuras/
python preparar_pdf.py        # o PDF, conferindo o transbordo
```

Se o aplicativo estiver em outro lugar, aponte `APP_UR5` para o checkout
dele. As figuras saem em `figuras/` deste repositório nos dois casos.

As figuras e o PDF vão versionados justamente para que ler o manual não
dependa de rodar nada: `matplotlib` e `playwright` só entram para refazer.

## Os arquivos

| | |
|---|---|
| `manual.html` | O manual para ler na tela, com links. |
| `manual-impresso.html` | A edição impressa, paginada à mão em folhas de tamanho Carta. |
| `manual-impresso.pdf` | O resultado, pronto para imprimir. |
| `figuras/` | As sete figuras do manual. |
| `preparar_manual.py` | Gera as figuras. Traz um rasterizador com z-buffer em numpy, umas quarenta linhas, porque o matplotlib 3D ordena polígono por centroide e erra justamente onde o punho passa por cima do antebraço. |
| `preparar_pdf.py` | Fecha o PDF. Antes disso confere se alguma folha transbordou, e recusa gerar se sim. |

## Marcas

Universal Robots, UR5 e PolyScope são marcas da Universal Robots A/S.
Documento independente, não produzido nem endossado por ela.
