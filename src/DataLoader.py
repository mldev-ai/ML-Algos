from sklearn.model_selection import train_test_split

class Dataset:

    """
    Class to split dataset into train and test sets
    """

    def __init__(self, features, labels, test_size=0.33, random_state=42):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
        self.features = features
        self.labels = labels
    
    def split_data(self):
        return train_test_split(self.features, self.labels, test_size=self.test_size, random_state=self.random_state)