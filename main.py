import functions

if __name__ == '__main__':
    imagem = './images/JPCNN001.bmp'
    imagemLimiarizada = functions.getImagemLimiarizada(imagem)
    functions.showImage(imagemLimiarizada)