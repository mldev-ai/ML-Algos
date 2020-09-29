from sklearn.ensemble import RandomForestClassifier

class ModelBase:
    def __init__(self, n_estimators, criterion):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion

    def forward(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion)