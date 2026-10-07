"""
Gera as figuras do manual impresso, em `figuras/`.

As vistas do robo saem do MESMO cache de malhas que o twin do navegador
consome (`malhas/ur5_elo*.npz`) e da MESMA cadeia cinematica do
`modelo_ur5.py`. Isso nao e capricho: uma figura de manual desenhada a
parte envelhece sozinha, e no dia em que o CAD mudar ninguem lembra de
refazer o desenho. Aqui basta rodar de novo.

Os dois vivem no repositorio do aplicativo, nao neste. Copiar o modelo
para ca seria exatamente o jeito de a cota do manual passar a discordar do
software. Deixe os dois checkouts lado a lado,

    git clone https://github.com/ricardobertolin/cobot_testing_ur5_app
    python preparar_manual.py

ou aponte APP_UR5 para onde ele esta:

    APP_UR5=/caminho/do/cobot_testing_ur5_app python preparar_manual.py

As figuras saem em `figuras/` deste repositorio nos dois casos.

Nao usa vedo nem VTK. O rasterizador abaixo e umas quarenta linhas de
numpy, e existe por dois motivos. Primeiro, o manual e impresso: o que se
quer e uma vista limpa em escala de cinza com contorno preto, e nao um
render bonito de tela. Segundo, o matplotlib 3D ordena poligonos por
centroide, o que erra justamente onde este robo se sobrepoe, que e o punho
contra o antebraco. Z-buffer por pixel nao erra.

Dependencias: numpy e matplotlib.
"""

import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "figuras")

# O modelo e o cache de malhas moram no repositorio do aplicativo. O
# `modelo_ur5.py` acha a pasta `malhas/` pelo proprio __file__, entao basta
# colocar o checkout dele no sys.path: as malhas vem de la, as figuras saem
# em SAIDA aqui.
APP = os.environ.get("APP_UR5") or os.path.join(os.path.dirname(AQUI),
                                                "cobot_testing_ur5_app")
if not os.path.isfile(os.path.join(APP, "modelo_ur5.py")):
    raise SystemExit(
        "nao achei o modelo_ur5.py em %s\n"
        "Clone o aplicativo ao lado deste repositorio:\n"
        "    git clone https://github.com/ricardobertolin/"
        "cobot_testing_ur5_app\n"
        "ou aponte APP_UR5 para o checkout dele." % APP
    )
sys.path.insert(0, APP)

import modelo_ur5 as mod

# Impressao em 300 dpi. Uma figura de 8 cm de largura pede uns 950 px, e o
# supersampling de 2x entrega borda limpa sem precisar de antialias real.
SS = 2

NOMES_JUNTAS = [
    ("J1", "Base"),
    ("J2", "Ombro"),
    ("J3", "Cotovelo"),
    ("J4", "Punho 1"),
    ("J5", "Punho 2"),
    ("J6", "Punho 3"),
]


# ============================================================
# RASTERIZADOR
# ============================================================

def _normalizar(v):
    return v / np.linalg.norm(v)


def camera(olho, alvo, cima=(0.0, 0.0, 1.0)):
    """Matriz mundo->camera. A camera olha na direcao -Z, como em OpenGL."""
    olho = np.asarray(olho, dtype=float)
    alvo = np.asarray(alvo, dtype=float)
    frente = _normalizar(alvo - olho)
    lado = _normalizar(np.cross(frente, np.asarray(cima, dtype=float)))
    acima = np.cross(lado, frente)
    return np.array([lado, acima, -frente]), olho


def renderizar(corpos, olho, alvo, largura, altura, fov=28.0,
               orto=None, cima=(0.0, 0.0, 1.0), margem=0.035):
    """
    Desenha uma lista de (vertices Nx3, faces Mx3, tom 0..1) e devolve
    (imagem, projetar).

    `imagem` e um array (altura, largura) de luminancia, 1.0 = papel branco.
    `projetar` leva um ponto 3D do mundo para pixel da imagem final, para
    que as chamadas de cota e de rotulo caiam exatamente sobre a peca.

    Com `orto` preenchido (meia-altura da vista, em metros) a projecao e
    ortografica, que e o que vale para desenho cotado: em perspectiva duas
    cotas iguais saem com tamanhos diferentes no papel.

    `margem` recorta o branco em volta e deixa essa fracao de folga, para
    nao ter que acertar enquadramento na tentativa e erro. Passando None o
    recorte nao acontece, que e o que vale quando duas figuras precisam
    sair na mesma escala e com a base alinhada.
    """
    L, A = largura * SS, altura * SS
    R, centro = camera(olho, alvo, cima)

    profundidade = np.full((A, L), np.inf)
    tom = np.ones((A, L), dtype=float)

    if orto is None:
        escala = 1.0 / math.tan(math.radians(fov) * 0.5)
    aspecto = L / A

    def _para_camera(p):
        return (np.atleast_2d(p) - centro) @ R.T

    def _para_tela(pc):
        if orto is None:
            z = np.maximum(-pc[:, 2], 1e-6)
            x = pc[:, 0] * escala / (z * aspecto)
            y = pc[:, 1] * escala / z
        else:
            z = -pc[:, 2]
            x = pc[:, 0] / (orto * aspecto)
            y = pc[:, 1] / orto
        return np.column_stack([(x * 0.5 + 0.5) * L, (0.5 - y * 0.5) * A, z])

    # Luz fixa na camera, vinda de cima e da esquerda. Luz presa ao mundo
    # deixaria um lado do robo preto dependendo do angulo escolhido.
    luz = _normalizar(np.array([-0.45, 0.6, 0.65])) @ R

    for vertices, faces, base_tom in corpos:
        pc = _para_camera(vertices)
        tela = _para_tela(pc)

        tri = tela[faces]                      # (M, 3, 3)
        mundo = vertices[faces]

        normal = np.cross(mundo[:, 1] - mundo[:, 0], mundo[:, 2] - mundo[:, 0])
        comprimento = np.linalg.norm(normal, axis=1)
        vivo = comprimento > 1e-12
        normal[vivo] /= comprimento[vivo, None]

        # Area com sinal na tela. Serve para descartar triangulo degenerado
        # e para achar as coordenadas baricentricas, e NAO para decidir que
        # face esta virada para tras.
        #
        # A versao anterior cortava por `area < 0`, que e o backface
        # culling de sempre. Ele exige orientacao coerente das faces, e as
        # malhas simplificadas do CAD nao tem: medindo a normal contra o
        # centroide, a fracao de triangulos virados para fora fica em torno
        # de 0,5 em toda peca do FANUC, ou seja, o sentido e quase
        # aleatorio. O resultado no papel era peca com furo e parede fina
        # transparente, porque a face da frente era descartada e a de tras
        # tambem.
        #
        # Desenhar os dois lados e deixar o z-buffer resolver custa o dobro
        # de triangulos e nao depende de orientacao nenhuma. O sombreado ja
        # usa |n . luz|, entao face invertida acende igual.
        area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) -
                (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        visivel = (vivo & (np.abs(area) > 1e-9) &
                   np.all(tri[:, :, 2] > 1e-6, axis=1))

        lambert = np.abs(normal @ luz)
        cinza = base_tom * (0.30 + 0.70 * lambert)

        for indice in np.nonzero(visivel)[0]:
            t = tri[indice]
            x0 = max(int(np.floor(t[:, 0].min())), 0)
            x1 = min(int(np.ceil(t[:, 0].max())) + 1, L)
            y0 = max(int(np.floor(t[:, 1].min())), 0)
            y1 = min(int(np.ceil(t[:, 1].max())) + 1, A)
            if x1 <= x0 or y1 <= y0:
                continue

            xs = np.arange(x0, x1) + 0.5
            ys = np.arange(y0, y1) + 0.5
            gx, gy = np.meshgrid(xs, ys)

            d = area[indice]
            w0 = ((t[1, 0] - gx) * (t[2, 1] - gy) -
                  (t[2, 0] - gx) * (t[1, 1] - gy)) / d
            w1 = ((t[2, 0] - gx) * (t[0, 1] - gy) -
                  (t[0, 0] - gx) * (t[2, 1] - gy)) / d
            w2 = 1.0 - w0 - w1

            dentro = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not dentro.any():
                continue

            # Interpola 1/z, que e o que varia linearmente na tela. Com z
            # cru o punho passa por dentro do antebraco em vista inclinada.
            inv = w0 / t[0, 2] + w1 / t[1, 2] + w2 / t[2, 2]
            z = np.where(inv > 1e-12, 1.0 / np.maximum(inv, 1e-12), np.inf)

            recorte = (slice(y0, y1), slice(x0, x1))
            melhor = dentro & (z < profundidade[recorte])
            if not melhor.any():
                continue
            profundidade[recorte] = np.where(melhor, z, profundidade[recorte])
            tom[recorte] = np.where(melhor, cinza[indice], tom[recorte])

    grande = _contornar(tom, profundidade)
    x0, y0 = 0, 0
    if margem is not None:
        x0, y0, grande = _recortar(grande, margem)

    alto, largo = grande.shape[0] // SS, grande.shape[1] // SS
    imagem = grande[:alto * SS, :largo * SS].reshape(
        alto, SS, largo, SS).mean(axis=(1, 3))

    def projetar(p):
        tela = _para_tela(_para_camera(p))[:, :2]
        return np.column_stack([(tela[:, 0] - x0) / SS, (tela[:, 1] - y0) / SS])

    return imagem, projetar


def _recortar(imagem, margem):
    """Corta o branco em volta, deixando `margem` do maior lado de folga."""
    tinta = imagem < 0.995
    if not tinta.any():
        return 0, 0, imagem
    linhas = np.nonzero(tinta.any(axis=1))[0]
    colunas = np.nonzero(tinta.any(axis=0))[0]
    folga = int(round(margem * max(linhas[-1] - linhas[0],
                                   colunas[-1] - colunas[0]))) + SS

    # Alinhado ao supersampling, senao a media de 2x2 do downsample pega
    # meio pixel de cada lado e a borda sai borrada.
    y0 = max(int(linhas[0]) - folga, 0) // SS * SS
    x0 = max(int(colunas[0]) - folga, 0) // SS * SS
    y1 = min(int(linhas[-1]) + folga + 1, imagem.shape[0])
    x1 = min(int(colunas[-1]) + folga + 1, imagem.shape[1])
    return x0, y0, imagem[y0:y1, x0:x1]


def _contornar(tom, profundidade, salto=0.004):
    """
    Preto na silhueta e nas quebras de profundidade.

    Sem isso a vista fica um borrao cinza no papel: impressora a laser come
    justamente as diferencas suaves de tom que separariam o punho do
    antebraco. A quebra de profundidade e o contorno que um desenho tecnico
    teria.
    """
    solido = np.isfinite(profundidade)
    z = np.where(solido, profundidade, 0.0)

    borda = np.zeros_like(solido)
    for eixo in (0, 1):
        d = np.abs(np.diff(z, axis=eixo)) > salto
        vizinhos = np.roll(solido, 1, axis=eixo) & solido
        d = d & np.delete(vizinhos, 0, axis=eixo)
        pad = [(0, 0), (0, 0)]
        pad[eixo] = (1, 0)
        borda |= np.pad(d, pad)
        pad[eixo] = (0, 1)
        borda |= np.pad(d, pad)

        s = solido.astype(np.int8)
        troca = np.diff(s, axis=eixo) != 0
        pad[eixo] = (1, 0)
        borda |= np.pad(troca, pad)
        pad[eixo] = (0, 1)
        borda |= np.pad(troca, pad)

    # Engrossa um pixel. A imagem ainda vai encolher SS vezes, e uma linha
    # de um pixel so vira cinza claro nessa media e some na impressao.
    grossa = borda.copy()
    for eixo in (0, 1):
        grossa |= np.roll(borda, 1, axis=eixo) | np.roll(borda, -1, axis=eixo)

    return np.where(grossa, 0.0, tom)


def corpos_na_pose(graus, tons=None):
    """Malhas do cache ja transformadas para a pose dada, em graus."""
    q = np.radians(np.asarray(graus, dtype=float))
    corpos_t = mod.transformadas(q)
    malhas = mod.carregar_malhas()

    if tons is None:
        # Base escura, braco claro, punho 3 escuro. Mesma leitura do twin,
        # traduzida para cinza.
        tons = [0.42, 0.80, 0.86, 0.82, 0.76, 0.76, 0.46]

    saida = []
    for indice, (_, vertices, faces, _) in enumerate(malhas):
        R, t = corpos_t[indice]
        saida.append((vertices @ R.T + t, faces, tons[indice]))
    return saida


def juntas_na_pose(graus):
    """Ponto e eixo de cada junta, no mundo, para a pose dada."""
    q = np.radians(np.asarray(graus, dtype=float))
    corpos_t = mod.transformadas(q)
    pontos, eixos = [], []
    for i in range(6):
        R, t = corpos_t[i]
        pontos.append(R @ mod.PONTOS[i] + t)
        eixos.append(R @ mod.EIXOS[i])
    return np.array(pontos), np.array(eixos)


# ============================================================
# FOLHA
# ============================================================

def folha_da_imagem(imagem, largura_cm):
    """Folha com a proporção exata da imagem já recortada."""
    return folha(largura_cm, largura_cm * imagem.shape[0] / imagem.shape[1])


def folha(largura_cm, altura_cm):
    fig = plt.figure(figsize=(largura_cm / 2.54, altura_cm / 2.54), dpi=300)
    eixo = fig.add_axes([0, 0, 1, 1])
    eixo.set_axis_off()
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1)
    return fig, eixo


def gravar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, nome)
    fig.savefig(caminho, dpi=300, facecolor="white")
    plt.close(fig)
    print("   ", os.path.relpath(caminho, AQUI))


def _desenhar(eixo, imagem):
    eixo.imshow(imagem, cmap="gray", vmin=0.0, vmax=1.0,
                extent=[0, imagem.shape[1], imagem.shape[0], 0],
                interpolation="antialiased")
    eixo.set_xlim(0, imagem.shape[1])
    eixo.set_ylim(imagem.shape[0], 0)
    eixo.set_axis_off()


def _chamada(eixo, xy_peca, xy_texto, texto, tamanho=7.5, ha=None):
    """
    Linha de chamada com bolinha na peca, como desenho de catalogo.

    Sem `ha` o lado sai do proprio desenho: rotulo a direita da peca cresce
    para a direita. No desenho do pendant isso joga texto para fora da
    folha, e ai o lado vai na mao.
    """
    eixo.annotate(
        texto, xy=xy_peca, xytext=xy_texto, fontsize=tamanho,
        family="DejaVu Sans", color="black", va="center",
        ha=ha or ("left" if xy_texto[0] > xy_peca[0] else "right"),
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.6,
                        shrinkA=0, shrinkB=2,
                        connectionstyle="arc3,rad=0"),
    )
    eixo.plot([xy_peca[0]], [xy_peca[1]], marker="o", markersize=2.2,
              color="black", zorder=5)


# ============================================================
# AS FIGURAS
# ============================================================

def fig_vista():
    """Vista geral, pose de trabalho. E a figura de capa."""
    corpos = corpos_na_pose([0, -60, 90, -120, -90, 0])
    imagem, _ = renderizar(corpos, olho=(1.55, -1.75, 1.35),
                           alvo=(0.0, -0.12, 0.52),
                           largura=1400, altura=1400, fov=30.0, margem=0.03)
    fig, eixo = folha_da_imagem(imagem, 12.0)
    _desenhar(eixo, imagem)
    gravar(fig, "fig-01-vista.png")


def fig_juntas():
    """
    A mesma vista, com as seis juntas chamadas.

    As chamadas sao colocadas sozinhas: cada junta sai para a margem mais
    proxima e, dentro de cada margem, a ordem vertical dos rotulos repete a
    ordem vertical das juntas. E o que impede linha de chamada cruzada, que
    e o defeito classico de figura desenhada na mao.
    """
    pose = [0, -70, 100, -120, -90, 0]
    imagem, projetar = renderizar(corpos_na_pose(pose), olho=(1.35, -1.95, 1.15),
                                  alvo=(0.0, -0.14, 0.52),
                                  largura=1500, altura=1500, fov=30.0,
                                  margem=0.02)
    pontos, _ = juntas_na_pose(pose)
    tela = projetar(pontos)

    A, L = imagem.shape[0], imagem.shape[1]
    corredor = 0.42 * L                       # espaco lateral para os rotulos

    fig, eixo = folha(14.0, 14.0 * A / (L + 2 * corredor))
    _desenhar(eixo, imagem)
    eixo.set_xlim(-corredor, L + corredor)
    eixo.set_ylim(A * 1.07, -A * 0.05)

    esquerda = [i for i in range(6) if tela[i, 0] < 0.5 * L]
    direita = [i for i in range(6) if i not in esquerda]

    # O texto encosta na borda do desenho e cresce para fora: e a linha de
    # chamada que atravessa o corredor, nao o rotulo.
    for indices, x_texto in ((esquerda, -corredor * 0.08),
                             (direita, L + corredor * 0.08)):
        if not indices:
            continue
        indices = sorted(indices, key=lambda i: tela[i, 1])
        passo = A * 0.82 / max(len(indices), 2)
        topo = A * 0.5 - passo * (len(indices) - 1) / 2
        for ordem, i in enumerate(indices):
            sigla, nome = NOMES_JUNTAS[i]
            _chamada(eixo, (tela[i, 0], tela[i, 1]),
                     (x_texto, topo + ordem * passo),
                     "%s  %s" % (sigla, nome), tamanho=9)

    eixo.text(0.5 * L, A * 1.045, "flange da ferramenta na ponta de J6",
              fontsize=7.5, style="italic", ha="center", va="bottom",
              color="0.25")
    gravar(fig, "fig-02-juntas.png")


def fig_cotas():
    """
    Vista lateral ortografica com as cotas da cadeia.

    Os numeros sao os mesmos d1, a2, a3, d4, d5 e d6 do `modelo_ur5.py`,
    lidos de `PONTOS`, e nao digitados aqui: cota de manual que discorda do
    codigo e pior que cota nenhuma.
    """
    pose = [0, -90, 0, -90, 0, 0]              # a pose Vertical, braco esticado
    centro = (0.0, -0.09, 0.50)
    imagem, projetar = renderizar(
        corpos_na_pose(pose), olho=(2.6, centro[1], centro[2]), alvo=centro,
        largura=900, altura=1500, orto=0.575, margem=0.01)

    A, L = imagem.shape[0], imagem.shape[1]
    esq, dir_ = 0.55 * L, 0.78 * L             # corredores de cota

    fig, eixo = folha(9.5, 9.5 * (A * 1.06) / (L + esq + dir_))
    _desenhar(eixo, imagem)
    eixo.set_xlim(-esq, L + dir_)
    eixo.set_ylim(A * 1.045, -A * 0.015)

    def y_de(z):
        return projetar(np.array([[0.0, centro[1], z]]))[0, 1]

    # PONTOS[4] e PONTOS[3] tem o MESMO z: na pose do CAD o d5 e um
    # deslocamento lateral, nao vertical. Quem fecha a altura e PONTOS[5],
    # o eixo do punho 3, que e onde o flange nasce.
    z = [0.0, mod.PONTOS[1][2], mod.PONTOS[2][2], mod.PONTOS[3][2],
         mod.PONTOS[5][2]]
    cotas = [
        (z[0], z[1], "d1 = %.3f" % ((z[1] - z[0]) * 1000)),
        (z[1], z[2], "a2 = %.1f" % ((z[2] - z[1]) * 1000)),
        (z[2], z[3], "a3 = %.2f" % ((z[3] - z[2]) * 1000)),
        (z[3], z[4], "d5 = %.2f" % ((z[4] - z[3]) * 1000)),
    ]

    x_cota = L + dir_ * 0.52
    for z_a, z_b, texto in cotas:
        ya, yb = y_de(z_a), y_de(z_b)
        eixo.annotate("", xy=(x_cota, ya), xytext=(x_cota, yb),
                      arrowprops=dict(arrowstyle="<|-|>", color="black",
                                      linewidth=0.7, mutation_scale=6,
                                      shrinkA=0, shrinkB=0))
        for y in (ya, yb):
            eixo.plot([0.42 * L, x_cota + dir_ * 0.10], [y, y], color="black",
                      linewidth=0.45, linestyle=(0, (4, 2)))
        # Cota curta nao cabe deitada dentro do proprio intervalo.
        if abs(ya - yb) > 0.07 * A:
            eixo.text(x_cota - dir_ * 0.09, (ya + yb) / 2, texto, fontsize=7,
                      rotation=90, ha="center", va="center",
                      family="DejaVu Sans")
        else:
            eixo.text(x_cota + dir_ * 0.16, (ya + yb) / 2, texto, fontsize=7,
                      ha="left", va="center", family="DejaVu Sans")

    # Cota total, do plano de fixacao ate o eixo do punho 3.
    y0, y4 = y_de(z[0]), y_de(z[4])
    x_total = -esq * 0.55
    eixo.annotate("", xy=(x_total, y0), xytext=(x_total, y4),
                  arrowprops=dict(arrowstyle="<|-|>", color="black",
                                  linewidth=0.8, mutation_scale=7))
    eixo.text(x_total - esq * 0.10, (y0 + y4) / 2, "%.3f" % (z[4] * 1000),
              fontsize=8.5, rotation=90, ha="center", va="center",
              family="DejaVu Sans", weight="bold")
    for y in (y0, y4):
        eixo.plot([x_total - esq * 0.02, 0.42 * L], [y, y], color="black",
                  linewidth=0.45, linestyle=(0, (4, 2)))

    for y, rotulo in ((y0, "plano de fixação"), (y4, "eixo do punho 3")):
        eixo.text(-esq * 0.06, y - 0.008 * A, rotulo, fontsize=6.8,
                  style="italic", color="0.25", va="bottom", ha="right")
    eixo.text((L + dir_ - esq) / 2, A * 1.035,
              "cotas em mm  ·  pose Vertical  ·  d4 e d6 saem de lado",
              fontsize=6.4, ha="center", va="bottom", color="0.25")
    gravar(fig, "fig-03-cotas.png")


def fig_envelope():
    """Envelope de trabalho, em corte. Desenho esquematico, sem CAD."""
    fig, eixo = folha(12.5, 12.5 * 2.45 / 3.05)
    eixo.set_xlim(-1.55, 1.50)
    eixo.set_ylim(-0.95, 1.50)
    eixo.set_aspect("equal")
    eixo.set_axis_off()

    z_ombro = mod.PONTOS[1][2]
    alcance = mod.ALCANCE
    raio_morto = 0.180

    eixo.add_patch(Circle((0, z_ombro), alcance, facecolor="0.90",
                          edgecolor="black", linewidth=1.0, zorder=0))

    # Coluna morta, em volta do eixo da base, acima e abaixo.
    eixo.add_patch(Rectangle((-raio_morto, -0.92), 2 * raio_morto, 2.35,
                             facecolor="white", edgecolor="black",
                             linewidth=0.7, linestyle=(0, (5, 3)), zorder=1))

    # Plano de fixacao e pedestal.
    eixo.plot([-1.00, 1.00], [0, 0], color="black", linewidth=1.1, zorder=3)
    for x in np.arange(-0.98, 1.01, 0.062):
        eixo.plot([x, x - 0.040], [0, -0.040], color="black", linewidth=0.5,
                  zorder=3)
    eixo.add_patch(Rectangle((-0.0745, 0), 0.149, z_ombro, facecolor="0.55",
                             edgecolor="black", linewidth=0.8, zorder=4))

    eixo.annotate("", xy=(0, z_ombro),
                  xytext=(alcance * 0.707, z_ombro + alcance * 0.707),
                  arrowprops=dict(arrowstyle="<|-", color="black",
                                  linewidth=0.8, mutation_scale=8), zorder=5)
    eixo.text(alcance * 0.40, z_ombro + alcance * 0.46, "R850",
              fontsize=10.5, weight="bold", family="DejaVu Sans",
              ha="left", va="bottom", zorder=6)

    eixo.annotate("", xy=(-raio_morto, 1.13), xytext=(raio_morto, 1.13),
                  arrowprops=dict(arrowstyle="<|-|>", color="black",
                                  linewidth=0.7, mutation_scale=7), zorder=5)
    eixo.text(0, 1.155, "Ø360", fontsize=7.5, ha="center", va="bottom",
              family="DejaVu Sans", zorder=6)
    eixo.annotate("coluna a evitar", xy=(raio_morto * 0.4, -0.62),
                  xytext=(0.52, -0.72), fontsize=7.5, style="italic",
                  va="center", ha="left", zorder=6,
                  arrowprops=dict(arrowstyle="-", color="black",
                                  linewidth=0.6, shrinkB=1))

    eixo.text(-1.52, 1.47, "ENVELOPE DE TRABALHO", fontsize=9,
              weight="bold", family="DejaVu Sans", va="top")
    eixo.text(-1.52, 1.35,
              "Alcance de 850 mm medido do eixo\n"
              "da base. Dentro da coluna central a\n"
              "ponta anda devagar enquanto as\n"
              "juntas giram rápido: evite passar\n"
              "por cima da própria base.",
              fontsize=7, va="top", color="0.15", linespacing=1.5)
    eixo.text(-1.52, -0.60,
              "A mesa e o piso cortam a parte\n"
              "de baixo da esfera. O que sobra\n"
              "é o espaço realmente utilizável.",
              fontsize=6.6, va="top", color="0.3", linespacing=1.5,
              style="italic")
    eixo.text(1.48, -0.92, "corte vertical  ·  cotas em mm", fontsize=6.5,
              ha="right", va="bottom", color="0.25")
    gravar(fig, "fig-04-envelope.png")


def fig_poses():
    """As tres poses guardadas, lado a lado."""
    poses = [
        ("Zero", [0, 0, 0, 0, 0, 0]),
        ("Vertical", [0, -90, 0, -90, 0, 0]),
        ("Trabalho", [0, -60, 90, -120, -90, 0]),
    ]
    # Sem recorte automatico e com a mesma camera nas tres: elas precisam
    # sair na mesma escala e com a base na mesma altura, senao a
    # comparacao entre as poses nao quer dizer nada.
    imagens = [
        renderizar(corpos_na_pose(pose), olho=(2.3, -0.55, 0.62),
                   alvo=(0.0, -0.10, 0.55), largura=640, altura=900,
                   orto=0.63, margem=None)[0]
        for _, pose in poses
    ]
    # Faixa vertical comum (a escala e a altura tem que bater entre as
    # tres), mas janela horizontal centrada em cada pose: a Zero e baixa e
    # larga, a Vertical e alta e estreita, e uma janela unica deixaria dois
    # tercos de papel em branco.
    tintas = [im < 0.995 for im in imagens]
    juntas = np.logical_or.reduce(tintas)
    linhas = np.nonzero(juntas.any(axis=1))[0]
    y0, y1 = max(linhas[0] - 10, 0), linhas[-1] + 11

    faixas = [np.nonzero(t.any(axis=0))[0] for t in tintas]
    meia = max(f[-1] - f[0] for f in faixas) // 2 + 10
    recortes = []
    for imagem, faixa in zip(imagens, faixas):
        centro = (faixa[0] + faixa[-1]) // 2
        x0 = min(max(centro - meia, 0), imagem.shape[1] - 2 * meia)
        recortes.append(imagem[y0:y1, x0:x0 + 2 * meia])

    fig = plt.figure(figsize=(15.5 / 2.54, 9.4 / 2.54), dpi=300)

    for coluna, ((nome, pose), imagem) in enumerate(zip(poses, recortes)):
        eixo = fig.add_axes([coluna / 3.0 + 0.015, 0.22, 1 / 3.0 - 0.03, 0.76])
        _desenhar(eixo, imagem)
        eixo.text(0.5, -0.035, nome, transform=eixo.transAxes, fontsize=9.5,
                  weight="bold", family="DejaVu Sans", ha="center", va="top")
        eixo.text(0.5, -0.105, "[%s]" % ", ".join(str(v) for v in pose),
                  transform=eixo.transAxes, fontsize=6.6, ha="center",
                  va="top", family="DejaVu Sans Mono", color="0.3")
    fig.text(0.5, 0.045, "as três na mesma escala   ·   ângulos em graus, na "
             "ordem J1 a J6", fontsize=6.6, ha="center", va="top", color="0.25")
    gravar(fig, "fig-05-poses.png")


def fig_pendant():
    """
    O teach pendant do CB2, desenhado a partir da foto do aparelho do
    laboratorio.

    A versao anterior era um retrato com a emergencia em cima e o botao de
    energia embaixo, que e o formato do pendant do CB3. O CB2 e
    PAISAGEM: a tela ocupa a esquerda inteira e os dois botoes ficam
    empilhados numa faixa a direita, energia em cima e emergencia logo
    abaixo. Quem procurasse pelo desenho antigo procuraria no lugar errado.
    """
    fig, eixo = folha(12.5, 12.5 * 128.0 / 152.0)
    eixo.set_xlim(0, 152)
    eixo.set_ylim(0, 128)
    eixo.set_aspect("equal")
    eixo.set_axis_off()

    eixo.text(2, 125, "TEACH PENDANT  ·  UR5 CB2", fontsize=8.5,
              weight="bold", family="DejaVu Sans", va="top")
    eixo.text(2, 118, "esquemático, a partir do aparelho do laboratório",
              fontsize=6, va="top", color="0.4", style="italic")

    # ---------- frente ----------
    eixo.add_patch(FancyBboxPatch((6, 46), 90, 62,
                                  boxstyle="round,pad=0,rounding_size=5",
                                  facecolor="0.90", edgecolor="black",
                                  linewidth=1.3))
    # A tela e quase toda a frente, deitada.
    eixo.add_patch(Rectangle((12, 54), 66, 48, facecolor="0.42",
                             edgecolor="black", linewidth=0.9))
    eixo.text(45, 80, "tela de toque", fontsize=7, ha="center", va="center",
              color="white")
    eixo.text(45, 74, "PolyScope", fontsize=5.6, ha="center", va="center",
              color="0.85", style="italic")
    eixo.text(13, 50, "UNIVERSAL ROBOTS", fontsize=4.6, va="center",
              color="0.35", family="DejaVu Sans", weight="bold")

    # Faixa da direita: energia em cima, emergencia embaixo.
    eixo.add_patch(Circle((87, 96), 4.2, facecolor="0.80", edgecolor="black",
                          linewidth=1.0))
    eixo.plot([87, 87], [94.4, 97.6], color="black", linewidth=1.0)
    eixo.add_patch(Circle((87, 95.8), 2.4, facecolor="none",
                          edgecolor="black", linewidth=0.8))

    eixo.add_patch(Circle((87, 73), 7.4, facecolor="0.72", edgecolor="black",
                          linewidth=0.9))                       # colar
    eixo.add_patch(Circle((87, 73), 5.6, facecolor="0.30", edgecolor="black",
                          linewidth=1.0))                       # cogumelo
    eixo.text(87, 73, "E-STOP", fontsize=3.4, ha="center", va="center",
              color="white", weight="bold")

    # ---------- verso ----------
    eixo.add_patch(FancyBboxPatch((6, 8), 90, 30,
                                  boxstyle="round,pad=0,rounding_size=5",
                                  facecolor="white", edgecolor="0.45",
                                  linewidth=0.9, linestyle=(0, (4, 3))))
    eixo.add_patch(FancyBboxPatch((44, 16), 16, 14,
                                  boxstyle="round,pad=0,rounding_size=2",
                                  facecolor="0.80", edgecolor="black",
                                  linewidth=1.0))
    eixo.text(51, 40, "verso", fontsize=6, ha="center", va="bottom",
              style="italic", color="0.4")

    # ---------- chamadas ----------
    _chamada(eixo, (91.2, 96), (104, 104),
             "2  liga / desliga", tamanho=7.5, ha="left")
    _chamada(eixo, (94.4, 73), (104, 76),
             "1  parada de emergência", tamanho=7.5, ha="left")
    _chamada(eixo, (60, 23), (104, 23),
             "3  Freedrive\n     (no verso)", tamanho=7.5, ha="left")
    gravar(fig, "fig-06-pendant.png")


def fig_move():
    """Mapa da tela Move do PolyScope, esquematico."""
    fig, eixo = folha(13.5, 8.6)
    eixo.set_xlim(0, 135)
    eixo.set_ylim(0, 86)
    eixo.set_aspect("equal")
    eixo.set_axis_off()

    eixo.add_patch(Rectangle((4, 6), 127, 74, facecolor="white",
                             edgecolor="black", linewidth=1.1))

    # Abas.
    abas = ["Program", "Installation", "Move", "I/O", "Log"]
    x = 6
    for nome in abas:
        largura = 5.2 + 2.1 * len(nome)
        ativa = nome == "Move"
        eixo.add_patch(Rectangle((x, 70), largura, 8,
                                 facecolor="black" if ativa else "0.90",
                                 edgecolor="black", linewidth=0.8))
        eixo.text(x + largura / 2, 74, nome, fontsize=6.5, ha="center",
                  va="center", family="DejaVu Sans",
                  color="white" if ativa else "black",
                  weight="bold" if ativa else "normal")
        x += largura + 1.6

    def painel(x0, y0, w, h, titulo, linhas, monoespacado=False):
        """Caixa com barra de titulo. As linhas se distribuem no corpo."""
        eixo.add_patch(Rectangle((x0, y0), w, h, facecolor="0.965",
                                 edgecolor="black", linewidth=0.8))
        eixo.add_patch(Rectangle((x0, y0 + h - 6), w, 6, facecolor="0.82",
                                 edgecolor="black", linewidth=0.8))
        eixo.text(x0 + 2.5, y0 + h - 3, titulo, fontsize=6.4, va="center",
                  family="DejaVu Sans", weight="bold")
        if not linhas:
            return
        corpo = h - 6
        passo = corpo / (len(linhas) + 0.5)
        for i, linha in enumerate(linhas):
            eixo.text(x0 + 2.5, y0 + corpo - passo * (i + 0.75), linha,
                      fontsize=5.9, va="center", color="0.15",
                      family="DejaVu Sans Mono" if monoespacado
                      else "DejaVu Sans")

    painel(7, 22, 52, 44, "Robot   (vista 3D)", [])
    eixo.text(33, 42, "o robô desenhado", fontsize=6.5, ha="center",
              color="0.5", family="DejaVu Sans")
    eixo.text(33, 38, "TCP = bolinha azul", fontsize=5.8, ha="center",
              color="0.55", style="italic")

    painel(62, 45, 66, 21, "Feature:  Base / Tool / View",
           ["X    -109.16 mm      RX   0.0000",
            "Y     -19.45 mm      RY  -3.1416",
            "Z     902.41 mm      RZ   0.0000"], monoespacado=True)

    painel(62, 22, 66, 21, "Move Tool  /  Move Joints",
           ["setas retas   →  a ponta anda em linha reta",
            "setas curvas  →  gira em volta do TCP",
            "barras        →  cada junta, de -360° a +360°",
            "solte a seta  →  para na hora"])

    eixo.add_patch(Rectangle((7, 9), 121, 10, facecolor="0.90",
                             edgecolor="black", linewidth=0.8))
    eixo.text(10, 14, "Freedrive", fontsize=6.4, va="center",
              family="DejaVu Sans", weight="bold")
    eixo.text(32, 14, "(ou o botão no verso do pendant)", fontsize=5.9,
              va="center", color="0.3", style="italic")
    eixo.text(84, 14, "velocidade", fontsize=5.9, va="center", ha="right",
              color="0.3")
    eixo.add_patch(Rectangle((86, 11.5), 38, 5, facecolor="white",
                             edgecolor="black", linewidth=0.7))
    eixo.add_patch(Rectangle((86, 11.5), 11, 5, facecolor="0.35",
                             edgecolor="black", linewidth=0.7))

    eixo.text(4, 84, "ABA MOVE  ·  POLYSCOPE 1.8  ·  ESQUEMÁTICO",
              fontsize=7.5, weight="bold", family="DejaVu Sans", va="top")
    gravar(fig, "fig-07-move.png")


def main():
    if not mod.cache_existe():
        raise SystemExit(
            "cache de malhas ausente em malhas/. "
            "Rode: python modelo_ur5.py --preparar"
        )
    print("gerando figuras do manual:")
    fig_vista()
    fig_juntas()
    fig_cotas()
    fig_envelope()
    fig_poses()
    fig_pendant()
    fig_move()
    print("pronto.")


if __name__ == "__main__":
    main()
