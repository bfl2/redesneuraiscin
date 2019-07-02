#importando as bibliotecas necessárias para o experimento:
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import utils as utils

test = pd.read_csv("data/testDefault", sep="\t")
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.datasets_split()
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

feat_importances = pd.DataFrame(rf_clf.feature_importances_, index=test.iloc[:, 2:-2].columns,
                                columns=['Importance']).sort_values(by='Importance',
                                                                    ascending=False)
feat_importances.head(100).plot.bar(color='gold')
print (feat_importances.head(100))