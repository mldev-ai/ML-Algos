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