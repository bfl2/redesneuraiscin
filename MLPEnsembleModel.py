import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import utils as utils
from sklearn.ensemble import  VotingClassifier


#Os datasets para o ensemble:
datasets = utils.ensembles_datasets_split()
test = pd.read_csv("data/testDefault", sep="\t")
X_test, Y_test = test.iloc[:, 2:-2].values, test.iloc[:, -1].values
X_train1, Y_train1, X_val1, Y_val1 = datasets[0]
X_train2, Y_train2, X_val2, Y_val2 = datasets[1]
X_train3, Y_train3, X_val3, Y_val3 = datasets[2]
X_train4, Y_train4, X_val4, Y_val4 = datasets[3]
X_train5, Y_train5, X_val5, Y_val5 = datasets[4]

input_dim = X_train1.shape[1]

#Treinando ensemble mlps de modelos heterogêneos
mlp1 = models.Sequential()
mlp1.add(layers.Dense(100, activation='relu', input_dim=input_dim,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp1.add(layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
mlp1.add(layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
mlp1.add(layers.Dense(1, activation='sigmoid'))
mlp1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
mlp1.fit(X_train1, Y_train1, batch_size=64, epochs=100000,
                         callbacks=[callbacks.EarlyStopping(patience=10)], validation_data=(X_val1, Y_val1))

mlp2 = models.Sequential()
mlp2.add(layers.Dense(128, activation='tanh', input_dim=input_dim,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp2.add(layers.Dense(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
mlp2.add(layers.Dense(32, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
mlp2.add(layers.Dense(1, activation='sigmoid'))
mlp2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
mlp2.fit(X_train2, Y_train2, batch_size=64, epochs=100000,
                        callbacks=[callbacks.EarlyStopping(patience=10)], validation_data=(X_val2, Y_val2))

mlp3 = models.Sequential()
mlp3.add(layers.Dense(100, activation='tanh', input_dim=input_dim,
                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp3.add(layers.Dense(1, activation='sigmoid'))
mlp3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
mlp3.fit(X_train3, Y_train3, batch_size=64, epochs=100000,
                         callbacks=[callbacks.EarlyStopping(patience=10)], validation_data=(X_val3, Y_val3))

#Treinando ensemble mlps de modelos heterogêneos
mlp4 = models.Sequential()
mlp4.add(layers.Dense(128, activation='relu', input_dim=input_dim,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp4.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp4.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp4.add(layers.Dense(1, activation='sigmoid'))
mlp4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp4.fit(X_train4, Y_train4, batch_size=64, epochs=100000,
                         callbacks=[callbacks.EarlyStopping(patience=10)], validation_data=(X_val4, Y_val4))

mlp5 = models.Sequential()
mlp5.add(layers.Dense(128, activation='tanh', input_dim=input_dim,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp5.add(layers.Dense(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp5.add(layers.Dense(32, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
mlp5.add(layers.Dense(1, activation='sigmoid'))
mlp5.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
mlp5.fit(X_train5, Y_train5, batch_size=64, epochs=100000,
                         callbacks=[callbacks.EarlyStopping(patience=10)], validation_data=(X_val5, Y_val5))

#Criando ensembles com 3 e 5 mlps, respectivamente
ensemble1 = [mlp1, mlp2, mlp3]
ensemble2 = [mlp1, mlp2, mlp3, mlp4, mlp5]

#Usaremos a moda para pegar a classe com mais votos dos classificadores
from scipy import stats
from mlxtend.classifier import EnsembleVoteClassifier


eclf1 = EnsembleVoteClassifier(ensemble1, weights=[1,1,1,1,1],
                                refit=False, voting='soft')
eclf1.fit(X_train1, Y_train1)

eclf1_pred_class = eclf1.predict(X_test)
eclf1_pred_scores = eclf1.predict_proba(X_test)
accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(Y_test, eclf1_pred_class, eclf1_pred_scores)
utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)