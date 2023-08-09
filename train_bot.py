# Biblioteca de pré-processamento de dados de texto
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

nltk.download('punkt')

palavras = []
classes = []
lista_palavras_tags = []
palavras_ignoradas = ['?', '!', ',', '.', "'s", "'m"]
arquivo_dados_treinamento = open('intents.json').read()
intencoes = json.loads(arquivo_dados_treinamento)

# Função para adicionar palavras raiz
def obter_palavras_raiz(palavras, palavras_ignoradas):
    palavras_raiz = []
    for palavra in palavras:
        if palavra not in palavras_ignoradas:
            w = stemmer.stem(palavra.lower())
            palavras_raiz.append(w)
    return palavras_raiz

for intencao in intencoes['intents']:
    # Adicionar todas as palavras dos padrões à lista
    for padrao in intencao['patterns']:
        palavras_padrao = nltk.word_tokenize(padrao)
        palavras.extend(palavras_padrao)
        lista_palavras_tags.append((palavras_padrao, intencao['tag']))
    # Adicionar todas as tags à lista de classes
    if intencao['tag'] not in classes:
        classes.append(intencao['tag'])
        palavras_raiz = obter_palavras_raiz(palavras, palavras_ignoradas)

print(palavras_raiz)
print(lista_palavras_tags[0])
print(classes)

# Criar o corpus de palavras para o chatbot
def criar_corpus_chatbot(palavras_raiz, classes):
    palavras_raiz = sorted(list(set(palavras_raiz)))
    classes = sorted(list(set(classes)))

    pickle.dump(palavras_raiz, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    return palavras_raiz, classes

palavras_raiz, classes = criar_corpus_chatbot(palavras_raiz, classes)

print(palavras_raiz)
print(classes)

dados_treinamento = []
numero_de_tags = len(classes)
etiquetas = [0] * numero_de_tags

# Criar um saco de palavras e codificação de etiquetas
for palavras_tags in lista_palavras_tags:
    saco_de_palavras = []
    palavras_padrao = palavras_tags[0]

    for palavra in palavras_padrao:
        indice = palavras_padrao.index(palavra)
        palavra = stemmer.stem(palavra.lower())
        palavras_padrao[indice] = palavra

    for palavra in palavras_raiz:
        if palavra in palavras_padrao:
            saco_de_palavras.append(1)
        else:
            saco_de_palavras.append(0)
    print(saco_de_palavras)

    codificacao_etiquetas = list(etiquetas)  # Inicialmente, as etiquetas serão todas zeros
    tag = palavras_tags[1]  # Salvar a tag
    indice_tag = classes.index(tag)  # Ir para o índice da tag
    codificacao_etiquetas[indice_tag] = 1  # Anexar 1 àquele índice

    dados_treinamento.append([saco_de_palavras, codificacao_etiquetas])

print(dados_treinamento[0])

# Criar dados de treinamento
def preprocessar_dados_treinamento(dados_treinamento):
    dados_treinamento = np.array(dados_treinamento, dtype=object)

    treino_x = list(dados_treinamento[:, 0])
    treino_y = list(dados_treinamento[:, 1])

    print(treino_x[0])
    print(treino_y[0])

    return treino_x, treino_y

treino_x, treino_y = preprocessar_dados_treinamento(dados_treinamento)
