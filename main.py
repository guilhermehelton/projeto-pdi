import math
import matplotlib.pyplot as plt
import cv2

#T1
"""
regioes = [
        [(275, 660), (408, 793)],  # região 1
        [(408, 793), (541, 926)],  # região 2
        [(541, 660), (674, 793)],  # região 3
        [(674, 527), (807, 660)],  # região 4
        [(541, 926), (674, 1059)],  # região 5
        [(408, 1059), (541, 1192)]  # região 6
    ]
"""
#T2
"""
regioes = [
        [(275, 660), (420, 805)],  # região 1
        [(420, 805), (565, 950)],  # região 2
        [(565, 660), (710, 805)],  # região 3
        [(710, 527), (855, 660)],  # região 4
        [(565, 950), (710, 1095)],  # região 5
        [(420, 1095), (565, 1240)]  # região 6
    ]
"""
#T3
"""
regioes = [
        [(264, 728),(408, 872)],  # região 1
        [(408, 872),(552, 1016)],  # região 2
        [(552, 728),(696, 872)],  # região 3
        [(696, 584),(840, 728)],  # região 4
        [(552, 1016),(696, 1163)],  # região 5
        [(408, 1163),(552, 1307)]  # região 6
    ]
"""

def pintarImage(imagem):
    # carrega a imagem
    img = cv2.imread(imagem, 0)
    #img = invert(img)
    # define as coordenadas das regiões que serão analisadas
    regioes = [
        [(264, 728),(408, 872)],  # região 1
        [(408, 872),(552, 1016)],  # região 2
        [(552, 728),(696, 872)],  # região 3
        [(696, 584),(840, 728)],  # região 4
        [(552, 1016),(696, 1163)],  # região 5
        [(408, 1163),(552, 1307)]  # região 6
    ]

    valores_maximos = []
    # obtém os valores máximos de intensidade de cada região
    for regiao in regioes:
        x1, y1 = regiao[0]
        x2, y2 = regiao[1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    plt.imshow(img, cmap='gray')
    plt.show()

def returnLimiar(imagem):
    # carrega a imagem
    img = cv2.imread(imagem, 0)

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
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        valores_maximos.append(min_val);
    valores_maximos.remove(max(valores_maximos))
    valores_maximos.remove(min(valores_maximos))

    limiar = sum(valores_maximos) / 4

    return math.floor(limiar)

def showImagemBinarizada(imagem, limiar):
    img = cv2.imread(imagem, 0)

    _, imgAdapMean = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)

    plt.imshow(img, cmap='gray')
    plt.imshow(imgAdapMean, cmap='gray')
    plt.show()

if __name__ == '__main__':
    imagem = './images/JPCNN001.bmp'
    limiar = returnLimiar(imagem)
    showImagemBinarizada(imagem, limiar)