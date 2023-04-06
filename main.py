import functions

if __name__ == '__main__':
    imagem = './images/JPCNN001.bmp'
    imagemLimiarizada = functions.getImagemLimiarizada(imagem)
    imagemLimiarizadaInicial = imagemLimiarizada.copy()
    # functions.showImage(imagemLimiarizada)
    imagemPreenchida = functions.getImagemPreenchida(imagemLimiarizada)
    # functions.showImage(imagemPreenchida)
    imagemSemRuido = functions.corrigirRuidoImagem(
        imagemLimiarizadaInicial, imagemPreenchida)
    # functions.showImage(imagemSemRuido)
    imagemSemBuracos = functions.corrigirBuracosImagem(
        imagemSemRuido)
    # functions.showImage(imagemSemBuracos)
    imagemSegmentada = functions.corrigirContornosImagem(
        imagemSemBuracos)
    functions.showImage(imagemSegmentada)
