import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt


def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.

    Argumento(s):
    history -- Objeto retornado pela função fit do keras.

    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}


def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.

    Argumento(s):
    history -- Objeto retornado pela função fit do keras.

    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()


def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics


def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
        
def datasets_split():
    base_datapath = "data/"

    train = pd.read_csv(base_datapath + "train", sep="\t")
    validation = pd.read_csv(base_datapath + "validation", sep="\t")
    test = pd.read_csv(base_datapath + "testDefault", sep="\t")
    
    return (train.iloc[:, 2:-2].values,
    train.iloc[:, -1].values,
    validation.iloc[:, 2:-2].values,
    validation.iloc[:, -1].values,
    test.iloc[:, 2:-2].values,
    test.iloc[:, -1].values) 

def ensembles_datasets_split():
    datasets = []
    for i in range(1, 6):
        base_datapath = "data/datasets-projeto/dt"+str(i)

        train = pd.read_csv(base_datapath + "train", sep="\t")
        validation = pd.read_csv(base_datapath + "validation", sep="\t")
        
        datasets.append((train.iloc[:, 2:-2].values,
                         train.iloc[:, -1].values,
                         validation.iloc[:, 2:-2].values,
                         validation.iloc[:, -1].values))
    return datasets
        
        

def results_summary(history, Y_test, y_pred_class, y_pred_scores):
    ## Matriz de confusão
    print('Matriz de confusão no conjunto de teste:')
    print(confusion_matrix(Y_test, y_pred_class))

    ## Resumo dos resultados
    losses = extract_final_losses(history)
    print()
    print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
    print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
    print('\nPerformance no conjunto de teste:')
    accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(Y_test, y_pred_class, y_pred_scores)
    print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
