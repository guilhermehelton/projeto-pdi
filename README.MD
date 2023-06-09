# Projeto final da disciplina de PDI

Replicação e modificação de um artigo cientifíco na área de PDI.

## Referência do artigo

ARAÚJO, Augusto R. V. F. de; MENEZES, Luiza C. de; CONCI, Aura. Segmentação das áreas pulmonares em radiografias torácicas digitais. In: SIMPÓSIO BRASILEIRO DE COMPUTAÇÃO APLICADA À SAÚDE (SBCAS), 21. , 2021, Evento Online. Anais [...]. Porto Alegre: Sociedade Brasileira de Computação, 2021 . p. 358-368. ISSN 2763-8952. DOI: [https://doi.org/10.5753/sbcas.2021.16079](https://doi.org/10.5753/sbcas.2021.16079/).

### 📋 Pré-requisitos

O projeto foi implementado em Python v3 e faz uso das bibliotecas Scikit Image e OpenCV.

```
pip install opencv-python
```

```
pip install opencv-contrib-python

```

```
python3 -m pip install -U pip

```

```
python -m pip install -U scikit-image

```

### Resultados
| Nome da imagem  | TP | FN | FP | TN |
| ----- | ---- | ---- | ---- | ---- |
| JPCNN001	| 0,335 | 0,042 | 0,027 | 0,596 |
| JPCNN003	| 0,040 | 0,192 | 0,035 | 0,732 |
| JPCNN005	| 0,333 | 0,050 | 0,034 | 0,583 |
| JPCNN006	| 0,296 | 0,025 | 0,033 | 0,645 |
| JPCLN006	| 0,258 | 0,030 | 0,030 | 0,681 |
| JPCNN008	| 0,284 | 0,062 | 0,018 | 0,636 |
| JPCLN008	| 0,349 | 0,045 | 0,016 | 0,591 |
| JPCNN009	| 0,292 | 0,026 | 0,034 | 0,648 |
| JPCLN009	| 0,104 | 0,194 | 0,011 | 0,691 |
| JPCNN010	| 0,318 | 0,041 | 0,023 | 0,618 |
| **Média**	| **0,261** | **0,071** | **0,026** | **0,642** |

### Estatísticas
| Medida | Fórmula | Valor |
| ------ | ------- | ----- |
| **Acurácia**          |	(TP + TN) / (TP + FN + FP + TN) | 0,903 |
| **Sensibilidade**     |	TP/(TP+FN)	                    | 0,786 |
| **Especificidade**    |	TN/(TN+FP)	                    | 0,961 |
| **F1 Score**          |	2TP/(2TP + FP + FN)	            | 0,843 |
| **Índice de Jaccard** |	TP/(TP+FP+FN)                   | 0,729 |


### Resultados usando limiarização com algoritmo de Otsu
| Nome da imagem  | TP | FN | FP | TN |
| ----- | ---- | ---- | ---- | ---- |
| JPCNN001 | 0,346 | 0,031 | 0,039 | 0,584 |
| JPCNN003 | 0,157 | 0,075 | 0,010 | 0,758 |
| JPCNN005 | 0,336 | 0,048 | 0,036 | 0,581 |
| JPCNN006 | 0,287 | 0,034 | 0,027 | 0,652 |
| JPCLN006 | 0,167 | 0,121 | 0,005 | 0,706 |
| JPCNN008 | 0,289 | 0,057 | 0,020 | 0,635 |
| JPCLN008 | 0,002 | 0,391 | 0     | 0,607 |
| JPCNN009 | 0,055 | 0,264 | 0     | 0,681 |
| JPCLN009 | 0,196 | 0,102 | 0,031 | 0,671 |
| JPCNN010 | 0,314 | 0,045 | 0,021 | 0,620 |
| **Média** | **0,215** | **0,117** | **0,018** | **0,650** |

### Estatísticas
| Medida | Fórmula | Valor |
| ------ | ------- | ----- |
| **Acurácia**          |	(TP + TN) / (TP + FN + FP + TN) | 0,721 |
| **Sensibilidade**     |	TP/(TP+FN)	                    | 0,648 |
| **Especificidade**    |	TN/(TN+FP)	                    | 0,973 |
| **F1 Score**          |	2TP/(2TP + FP + FN)	            | 0,761 |
| **Índice de Jaccard** |	TP/(TP+FP+FN)                   | 0,614 |

**Legenda**:
**TP** = positivos verdadeiros
**FN** = falsos negativos
**FP** = falsos positivos
**TN** = negativos verdadeiros


## ✒️ Autores

- **Francisco Hiago Teixeira Chagas** - [hiagochagas](https://github.com/hiagochagas)
- **Guilherme Helton Silva Florindo** - [guilhermehelton](https://github.com/guilhermehelton/)
- **Joao Gabriel Soares Moura** -
- **Pedro Emanuel dos Santos Rodrigues** - [pedrowemanuel](https://github.com/pedrowemanuel)
