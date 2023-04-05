import math
import cv2 as cv
import matplotlib.pyplot as plt


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
