from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class MLClassifier:
    def __init__(self, model):
        self.model = model

    def train(self, train_x, train_y):
        model.fit(train_x, train_y)
        return model
    
    def eval(self, test_x, test_y):
        pred_y = model.predict(test_x)
        loss = 1 - accuracy_score(pred_y, test_y)
        return loss