class TrainClf:
    def __init__(self, train_x, train_y, model):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.model = model
    
    def train_clf(self):
        model_trained = self.model.fit(self.train_x, self.train_y)
        return model_trained