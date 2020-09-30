from sklearn.ensemble import RandomForestClassifier

class MLModel:
    """
    Define the model
    """
    def __init__(self, n_estimators=100, criterion="gini"):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion

    def get_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion)