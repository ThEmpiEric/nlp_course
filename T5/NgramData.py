from typing import Tuple 
import numpy as np  
from nltk import ngrams # extraer los N-gramas
from nltk.probability import FreqDist



class NgramData():
    """
    NgramData procesa y transformar corpus de texto en conjuntos de datos de n-gramas.
    - Tokenizar el texto y limpiar tokens no deseados.
    - Construir un vocabulario limitado a las palabras más frecuentes.
    - Mapear palabras a identificadores numéricos y viceversa.
    - Generar representaciones numéricas de secuencias de palabras (n-gramas) para entrenamiento.

    Parámetros:
    - N: El tamaño del n-grama 
    - v_max: El tamaño máximo del vocabulario.
    - tokenizer: Una función para tokenizar el texto. 
    - pretrain_embedding: Embeddings preentrenados opcionales para inicializar las representaciones de palabras.
    """
    def __init__(self, N: int, 
                 v_max: int = 5000, 
                 tokenizer = None, 
                 pretrain_embedding = None) -> None:
        self.N = N 
        self.v_max = v_max
        self.tokenizer = tokenizer if tokenizer else self.defaul_tokenizer
        self.punct = set(['@usuario', '#', 'http', 'https', 'www','!', '?', '.', ',',
                           ':', ';', '-', '_', '(', ')', '[', ']', '{', '}', 
                           '"', "'",'\n', '\t', '\r', '>','<', '”'])
        self.UNK = '<unk>'
        self.SOS = '<s>'
        self.EOS = '</s>'
        self.pretrain_embedding = pretrain_embedding

    def get_vocab_size(self) -> int:
        return len(self.vocab) 
    
    def defaul_tokenizer(self, doc: str) -> list: 
        return doc.split(' ')

    def remove_tokens(self, word: str) -> bool:
        return True if word in self.punct or word.isnumeric() else False

    def sorted_freqdict(self, fdic: FreqDist) -> list:
        aux = [key for key in fdic]  # (claves)
        aux.sort(key=lambda x: fdic[x], reverse=True)
        return aux
    
    def get_vocab(self, corpus: list) -> set:
        freq_dist = FreqDist([word.lower() for sentence in corpus for word in self.tokenizer(sentence) 
                              if not self.remove_tokens(word)])
        sorted_words = self.sorted_freqdict(freq_dist)[:self.v_max-3]
        return set(sorted_words) 

    def fit(self, corpus: list) -> None: 
        self.vocab = self.get_vocab(corpus)
        # Agregar los tokens especiales 
        self.vocab.add(self.UNK)
        self.vocab.add(self.SOS)
        self.vocab.add(self.EOS)

        # Diccionario de mapeo word2id and id2word
        self.word2id = {}
        self.id2word = {}

        # Embeddings 
        if self.pretrain_embedding is not None: 
            self.embedding_matrix = np.empty([len(self.vocab), self.pretrain_embedding.vector_size])
         
        # Llenar los diccionarios necesarios para el modelo
        id = 0 
        for doc in corpus: 
            for Word in self.tokenizer(doc): 
                word = Word.lower()
                if word in self.vocab and not word in self.word2id: 
                    self.word2id[word] = id
                    self.id2word[id] = word

                    # Construir la sub-matriz de las palabras que hay en nuestro vocab (si hay embeddings) 
                    if self.pretrain_embedding is not None:
                        if word in self.pretrain_embedding: 
                            self.embedding_matrix[id] = self.pretrain_embedding[word]
                        else: 
                            self.embedding_matrix[id] = np.random.rand(self.pretrain_embedding.vector_size) 
                    id += 1
                
        # Agregamos tokens especiales 
        self.word2id.update(
            {
                self.UNK: id, 
                self.SOS: id+1,
                self.EOS: id+2
            }
        )
        self.id2word.update(
            {
                id: self.UNK, 
                id+1: self.SOS,
                id+2: self.EOS
            }
        )


    def transform(self, corpus: list) -> Tuple[np.ndarray, np.ndarray]:
        X_ngrams = [] 
        y = [] #next to add 

        for doc in corpus:
            sentence = self.get_ngrams(doc)
            for word_window in sentence: 
                words_windows_ids = [self.word2id[w] for w in word_window]
                X_ngrams.append(list(words_windows_ids[:-1]))
                y.append(words_windows_ids[-1])
        
        return np.array(X_ngrams), np.array(y)

    def get_ngrams(self,doc: str) -> list: 
        doc_tokens = self.tokenizer(doc)
        doc_tokens = self.remplace_unk(doc_tokens)
        doc_tokens = [word.lower() for word in doc_tokens]
        doc_tokens = [self.SOS]*(self.N-1) + doc_tokens + [self.EOS]
        return list(ngrams(doc_tokens, self.N))
                
    def remplace_unk(self,doc_tokens: list) -> list:
        for i, token in enumerate(doc_tokens): 
            if token.lower() not in self.vocab: 
                doc_tokens[i] = self.UNK
        return doc_tokens
