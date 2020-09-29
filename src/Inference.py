from sklearn.metrics import accuracy_score, f1_score

class PredictClf:
    def eval(self, trained_model):
        pred_y = trained_model.predict(self.test_x)
        loss = 1 - accuracy_score(pred_y, self.test_y)
        return loss, pred_y