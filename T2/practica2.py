import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import TweetTokenizer

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

def get_text_from_file(path_file: str, target_file: str) -> list:
    file_txt = []
    target_txt = []

    with open(path_file, 'r', encoding="utf8") as f_corpus, open(target_file, 'r', encoding="utf8") as f_target:
        for tweet in f_corpus:
            file_txt += [tweet]
        for target in f_target:
            target_txt += [target]
    
    file_txt = list(map(str.strip, file_txt))
    target_txt = list(map(int, target_txt))
    
    return file_txt, target_txt

def order_dic_frec(fdic):
    aux = [(fdic[key], key) for key in fdic]
    aux.sort()
    aux.reverse()
    return aux

def BoW(text, vocabulario, dict_index):
    BOW = np.zeros((len(text), len(vocabulario)), dtype=int)
    
    cont_doc = 0
    
    for doc in text:
        fdist_doc = nltk.FreqDist(tk.tokenize(doc))
        
        for word in fdist_doc:
            if word in dict_index:
                BOW[cont_doc, dict_index[word]] = 1
        cont_doc += 1
    
    return BOW

def main():
    # Cargar datos
    file_txt_train, target_txt_train = get_text_from_file("./data_mex/mex20_train.txt", "./data_mex/mex20_train_labels.txt")
    file_txt_val, target_txt_val = get_text_from_file("./data_mex/mex20_val (1).txt", "./data_mex/mex20_val_labels (1).txt")
    
    # Tokenizer y frecuencia de palabras
    global tk, fdist, V, dict_index
    tk = TweetTokenizer()
    corpus = []
    for file in file_txt_train:
        corpus += tk.tokenize(file)
    
    fdist = nltk.FreqDist(corpus)
    V = order_dic_frec(fdist)
    V = V[:5000]
    
    # Indices (caracter top, 0 para el mayor frec)
    dict_index = dict()
    count = 0
    for weight, word in V:
        dict_index[word] = count
        count += 1
    
    # Bolsa de Términos
    BOW_tr = BoW(file_txt_train, V, dict_index)
    BOW_val = BoW(file_txt_val, V, dict_index)
    
    # Clasificación con SVM
    parametros = {'C': [0.5, .12, .25, .5, 1, 2, 4]}
    clasificador = svm.LinearSVC(class_weight='balanced')
    grid = GridSearchCV(estimator=clasificador, param_grid=parametros, n_jobs=8, scoring="f1_macro", cv=5)
    grid.fit(BOW_tr, target_txt_train)
    y_pred = grid.predict(BOW_val)
    
    # Métricas
    p, r, f, _ = precision_recall_fscore_support(target_txt_val, y_pred, average='macro', pos_label=1)
    print(confusion_matrix(target_txt_val, y_pred))
    print(metrics.classification_report(target_txt_val, y_pred))

if __name__ == '__main__':
    main()