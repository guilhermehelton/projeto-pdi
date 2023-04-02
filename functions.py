import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def showImage(imagem):
    plt.imshow(imagem, cmap='gray')
    plt.show()


def returnLimiar(imagem):
    # carrega a imagem
    img = cv.imread(imagem, 0)

    # define as coordenadas das regiões que serão analisadas
    regioes = [
        [(264, 728), (408, 872)],  # região 1
        [(408, 872), (552, 1016)],  # região 2
        [(552, 728), (696, 872)],  # região 3
        [(696, 584), (840, 728)],  # região 4
        [(552, 1016), (696, 1163)],  # região 5
        [(408, 1163), (552, 1307)]  # região 6
    ]

    valores_maximos = []

    # obtém os valores máximos de intensidade de cada região
    for i, regiao in enumerate(regioes):
        x1, y1 = regiao[0]
        x2, y2 = regiao[1]
        roi = img[y1:y2, x1:x2]
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(roi)
        valores_maximos.append(min_val)
    valores_maximos.remove(max(valores_maximos))
    valores_maximos.remove(min(valores_maximos))

    limiar = sum(valores_maximos) / 4

    return math.floor(limiar)


def getImagemLimiarizada(imagem):
    limiar = returnLimiar(imagem)
    img = cv.imread(imagem, 0)
    _, resultImage = cv.threshold(img, limiar, 255, cv.THRESH_BINARY)
    return resultImage


def fillImageWithRectangle(imagemLimiarizada):
    startPoint = (0, 0)
    endPoint = (2048, 2048)
    image = cv.rectangle(imagemLimiarizada, startPoint, endPoint, 0, 100)
    return image


def fillImageDownwards(image):
    for y in range(51, 1998):
        for x in range(51, 1998):
            if image[x][y] == 0:
                break
            if image[x][y] == 255:
                image[x][y] = 0
    return image


def fillImageUpwards(image):
    for y in range(51, 1998):
        for x in reversed(range(51, 1998)):
            if image[x][y] == 0:
                break
            if image[x][y] == 255:
                image[x][y] = 0
    return image


def getImagemPreenchida(imagemLimiarizada):
    imagemPreenchida = fillImageWithRectangle(imagemLimiarizada)
    imagemPreenchida = fillImageDownwards(imagemPreenchida)
    imagemPreenchida = fillImageUpwards(imagemPreenchida)
    return imagemPreenchida


def criaRetangulo(largura, altura):
    retangulo = np.ones((largura, altura), np.uint8)

    return retangulo


def criaDisco(raio):
    disco = np.ones((2*raio, 2*raio), np.uint8)

    for linha in range(raio):
        quantidadeDeZeros = raio - (raio - linha)
        for coluna in range(len(disco)):
            if (coluna+1 <= quantidadeDeZeros or coluna+1 >= (len(disco) - quantidadeDeZeros)):
                disco[(raio - 1) - linha][coluna] = 0

    for linha in range(raio+1):
        quantidadeDeZeros = raio - (raio - linha)
        for coluna in range(len(disco)):
            if (coluna+1 <= quantidadeDeZeros or coluna+1 >= (len(disco) - quantidadeDeZeros)):
                disco[(raio - 1) + linha][coluna] = 0

    return disco


def getImagemSemRuido(imagemLimiarizadaInicial, imagemPreenchida):
    ruido = imagemLimiarizadaInicial - imagemPreenchida
    retangulo1 = criaRetangulo(30, 10)
    retangulo2 = criaRetangulo(10, 30)
    disco = criaDisco(35)
    # operações morfologícas de dilatação
    ruido_dilatado = cv.dilate(ruido, retangulo1)
    ruido_dilatado = cv.dilate(ruido_dilatado, retangulo2)
    ruido_dilatado = cv.dilate(ruido_dilatado, disco)
    imagemSemRuido = imagemLimiarizadaInicial - ruido_dilatado

    return imagemSemRuido
