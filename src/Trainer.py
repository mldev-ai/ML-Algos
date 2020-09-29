from sklearn.metrics import accuracy_score, f1_score

class MLClassifier:
    def __init__(self, train_x, train_y, test_x, test_y, model):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
    
    def train(self):
        model_trained = self.model.fit(self.train_x, self.train_y)
        return model_trained
    
    def eval(self, trained_model):
        pred_y = trained_model.predict(self.test_x)
        loss = 1 - accuracy_score(pred_y, self.test_y)
        return loss, pred_y