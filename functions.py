import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import (
    dilation, disk, rectangle, closing, square, octagon)


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

def getImagemLimiarizadaOtsu(imagem):
    img = cv.imread(imagem, 0)
    _, resultImage = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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


def calculaMedidasRetangulo(alturaImagem, larguraImagem):
    return int(alturaImagem * 0.0146484375), int(larguraImagem * 0.0048828125)


def calculaRaioDisco(alturaImagem, larguraImagem):
    return max([int(alturaImagem * 0.029296875), int(larguraImagem * 0.029296875)])


def corrigirRuidoImagem(imagemLimiarizadaInicial, imagemPreenchida):
    mascaraRuido = imagemLimiarizadaInicial - imagemPreenchida
    # pega o tamanho da imagem
    altura, largura = mascaraRuido.shape

    # pega as medidas dos elementos estruturantes
    alturaRetangulo, larguraRetangulo = calculaMedidasRetangulo(
        altura, largura)
    raioDisco = calculaRaioDisco(altura, largura)

    # criando elementos estruturantes
    retangulo1 = rectangle(alturaRetangulo, larguraRetangulo)
    retangulo2 = rectangle(larguraRetangulo, alturaRetangulo)
    disco = disk(raioDisco)

    # operações morfologícas de dilatação na mascara
    ruido_dilatado = dilation(
        mascaraRuido, [(disco, 1), (retangulo1, 1), (retangulo2, 1)])

    imagemSemRuido = imagemLimiarizadaInicial - ruido_dilatado

    return imagemSemRuido


def corrigirBuracosImagem(imagemSemRuido):
    # pega as medidas dos elementos estruturantes
    ladoQuadrado = 35
    raioDisco = 35

    # criando elementos estruturantes
    quadrado = square(ladoQuadrado)
    disco = disk(raioDisco)

    # aplicando operação morfológica fechamento
    imagemSemBuracos = closing(imagemSemRuido, [(quadrado, 1), (disco, 1)])

    return imagemSemBuracos


def dividirImagem(imagem, porcentagem):
    quantidadeLinhasDivisao = int(len(imagem) * porcentagem)

    imagemDividida = np.split(imagem, [quantidadeLinhasDivisao])
    return imagemDividida


def corrigirContornosImagem(imagemSemBuracos):
    [parteDeCima, parteDeBaixo] = dividirImagem(imagemSemBuracos, 0.8)

    # criando elementos estruturantes para dilatação
    retangulo = rectangle(40, 2)
    octogono = octagon(17, 17)

    # operações morfologícas de dilatação na parte de cima
    parteDeCima = dilation(
        parteDeCima, [(retangulo, 1), (octogono, 1)])

    # criando elemento estruturante para fechamento
    disco = disk(25)

    # operação morfologíca de fechamento na parte de cima
    parteDeCima = closing(parteDeCima, disco)

    # juntar as imagens
    imagemCorrigida = np.concatenate((parteDeCima, parteDeBaixo))

    return imagemCorrigida

def getEstatisticas(imagem, gt):
    openedGT = cv.imread(gt, 0)
    truePositive = falsePositive = trueNegative = falseNegative = 0
    for x, xValue in enumerate(imagem):
        for y, yValue in enumerate(xValue):
            gtPixel = openedGT[x][y]
            if gtPixel == 0:
                if yValue <= 50:
                    trueNegative += 1
                else:
                    falsePositive += 1
            else:
                if yValue <= 50:
                    falseNegative += 1
                else:
                    truePositive += 1
    imageSize = 2048 ** 2
    truePositive /= imageSize
    falseNegative /= imageSize
    falsePositive /= imageSize
    trueNegative /= imageSize
    return [truePositive, falseNegative, falsePositive, trueNegative]