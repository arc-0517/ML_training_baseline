from sklearn.metrics import accuracy_score, f1_score

def return_result(model, X, y_true):
    y_pred = model.predict(X)
    result = {'acc': accuracy_score(y_true, y_pred),
              'f1': f1_score(y_true, y_pred, average='macro')}
    return result