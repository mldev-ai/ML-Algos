from sklearn.feature_extraction.text import TfidfVectorizer

class Extractor:

    def __init__(self, train_docs, ngram_range=(1, 1)):
        super().__init__()
        self.ngram_range = ngram_range
        self.train_docs = train_docs
        self.vectorizer = TfidfVectorizer()

    def fit_extractor(self):
        self.vectorizer.fit(self.train_docs)

    def vectorize_data(self, docs):
        return (self.vectorizer.transform(docs)).toarray()