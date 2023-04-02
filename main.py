import functions

if __name__ == '__main__':
    imagem = './images/JPCNN001.bmp'
    imagemLimiarizada = functions.getImagemLimiarizada(imagem)
    imagemLimiarizadaInicial = imagemLimiarizada.copy()
    # functions.showImage(imagemLimiarizada)
    imagemPreenchida = functions.getImagemPreenchida(imagemLimiarizada)
    # functions.showImage(imagemPreenchida)
    imagemSemRuido = functions.getImagemSemRuido(
        imagemLimiarizadaInicial, imagemPreenchida)
    functions.showImage(imagemSemRuido)
