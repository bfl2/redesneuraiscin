#importando as bibliotecas necessárias para o experimento:
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import utils
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from scipy import stats
from sklearn.ensemble import  VotingClassifier


base_datapath = "data/"

train = pd.read_csv(base_datapath + "train", sep="\t")
validation = pd.read_csv(base_datapath + "validation", sep="\t")
test = pd.read_csv(base_datapath + "testDefault", sep="\t")

train.shape, validation.shape, test.shape

X_train, Y_train, X_val, Y_val, X_test, Y_test = (train.iloc[:, 2:-2].values,
    train.iloc[:, -1].values,
    validation.iloc[:, 2:-2].values,
    validation.iloc[:, -1].values,
    test.iloc[:, 2:-2].values,
    test.iloc[:, -1].values)

X_train_val = np.append(X_train, X_val, axis=0)
Y_train_val = np.append(Y_train, Y_val, axis=0)



### Random Forest

rf_clf = RandomForestClassifier(n_estimators=100)  # Modifique aqui os hyperparâmetros
rf_clf.fit(X_train, Y_train)
rf_pred_class = rf_clf.predict(X_val)
rf_pred_scores = rf_clf.predict_proba(X_val)
accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(Y_val, rf_pred_class, rf_pred_scores)
utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

rf_pred_class = rf_clf.predict(X_test)
rf_pred_scores = rf_clf.predict_proba(X_test)
accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(Y_test, rf_pred_class, rf_pred_scores)
utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

###MLP

mlp = models.Sequential()

input_dim = X_train.shape[1]
mlp.add(layers.Dense(16, activation='tanh', input_dim=input_dim))
mlp.add(layers.Dense(1, activation='sigmoid'))
mlp.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = mlp.fit(X_train, Y_train, batch_size=64, epochs=100000,
                         callbacks=[callbacks.EarlyStopping(patience=3)], validation_data=(X_val, Y_val))

utils.plot_training_error_curves(history)

y_pred_scores = mlp.predict(X_test)
y_pred_class = mlp.predict_classes(X_test, verbose=0)
y_pred_scores_0 = 1 - y_pred_scores
y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)


###GRADIENT BOOSTING


xgb_clf9 = XGBClassifier(n_estimators=300, max_depth = 13, max_features = 90, n_jobs=-1)
xgb_clf9.fit(X_train_val, Y_train_val)

xgb_clf9_pred_class = xgb_clf9.predict(X_test)
xgb_clf9_pred_scores = xgb_clf9.predict_proba(X_test)
accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(
    Y_test, xgb_clf9_pred_class, xgb_clf9_pred_scores)
utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


### ENSEMBLE

ensemble = [rf_clf, mlp, xgb_clf9]

### random forest has no predict classes method error
y_pred3 = [rf_pred_class, y_pred_class, xgb_clf9_pred_class]
y_pred3 = np.array(y_pred3)
y_pred3 = stats.mode(y_pred3, axis=0).mode[0]

print(utils.accuracy_score(Y_test, y_pred3),
utils.recall_score(Y_test, y_pred3),
utils.precision_score(Y_test, y_pred3),
utils.f1_score(Y_test, y_pred3))


ens_clf = VotingClassifier(ensemble,
                           voting='soft')
ens_clf.fit(X_train, Y_train)
ens_pred_class = ens_clf.predict(X_test)
ens_pred_scores = ens_clf.predict_proba(X_test)
accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(y_pred3, ens_pred_class, ens_pred_scores[0])
print('\n\nEnsemble')
utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
