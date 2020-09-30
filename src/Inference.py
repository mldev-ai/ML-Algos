class PredictClf:
    """
    Predicting on test set from trained model
    """
    def __init__(self, test_x, trained_model):
        super().__init__()
        self.trained_model = trained_model
        self.test_x = test_x

    def predict_clf(self):
        return self.trained_model.predict(self.test_x)