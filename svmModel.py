
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC
import utils
import pandas as pd

def SVM():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.partitionateDataset()

    svc_clf = SVC(probability=True)  # Modifique aqui os hyperparâmetros
    svc_clf.fit(X_train, Y_train)
    svc_pred_class = svc_clf.predict(X_val)
    svc_pred_scores = svc_clf.predict_proba(X_val)
    accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(Y_val, svc_pred_class, svc_pred_scores)
    print('Performance no conjunto de validação:')
    utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

    svc_pred_class = svc_clf.predict(X_test)
    svc_pred_scores = svc_clf.predict_proba(X_test)
    accuracy, recall, precision, f1, auroc, aupr = utils.compute_performance_metrics(Y_test, svc_pred_class, svc_pred_scores)
    print('\n\nSupport Vector Machine')
    utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

    return

SVM()