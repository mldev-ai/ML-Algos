from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, test_size, random_state):
        super().__init__()
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, features, labels):
        return train_test_split(features, labels, test_size=self.test_size, random_state=self.random_state)