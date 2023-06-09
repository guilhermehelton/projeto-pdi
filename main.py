import functions
import images

def getImagemSegmentada(image, isOtsu):
    imagem = f'./images/{image}.bmp'
    if isOtsu:
        imagemLimiarizada = functions.getImagemLimiarizadaOtsu(imagem)
    else:
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
    imagemSegmentada = functions.corrigirContornosImagem(imagemSemBuracos)
    # functions.showImage(imagemSegmentada)
    return imagemSegmentada

def getEstatisticas(isOtsu):
    values = []
    for image in images.Image:
        imagemSegmentada = getImagemSegmentada(image.value, isOtsu)
        gt = f'./GTs/{image.value}GT.bmp'
        valores = functions.getEstatisticas(imagemSegmentada, gt)
        values.append([image.value, valores])
    return values

def formattedValue(value):
  return "%.3f" % value

def printEstatisticas(values):
    text = "Nome da imagem\tTP\tFN\tFP\tTN\n"
    for value in values:
        text += f"{value[0]}\t{formattedValue(value[1][0])}\t{formattedValue(value[1][1])}\t{formattedValue(value[1][2])}\t{formattedValue(value[1][3])}\n"
    print(text)

if __name__ == '__main__':
    imagem = images.Image.NN001
    imagemSegmentada = getImagemSegmentada(imagem.value, False)
    functions.showImage(imagemSegmentada)
    # values = getEstatisticas()
    # printEstatisticas(values)
    
