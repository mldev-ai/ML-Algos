import numpy as np
import pandas as pd
from src import *
from sklearn.metrics import accuracy_score, f1_score

"""
Main file to run everything
"""

def main():
    df = pd.read_csv("./data/text_classif_data.tsv", delimiter='\t', names=["text", "label"])
    print("Data Loaded")

    dt = Dataset(features=df["text"], labels=df["label"], test_size=0.33)
    train_x, test_x, train_y, test_y = dt.split_data()
    print("Data Split Done")

    tfidf = Extractor(train_x)
    tfidf.fit_extractor()
    train_vecs = tfidf.vectorize_data(train_x)
    test_vecs = tfidf.vectorize_data(test_x)
    print("Feature Extraction Done")

    print(f"df.shape: {df.shape}\ttrain_x.shape: {train_x.shape}\ttest_x.shape: {test_x.shape}\ttrain_y.shape: {train_y.shape}\ttest_y.shape: {test_y.shape}")
    print(f"train_vecs.shape: {train_vecs.shape}\ttest_vecs.shape: {test_vecs.shape}")

    mobj = MLModel(n_estimators=50)
    clf = mobj.get_model()
    print("Model Loaded")

    trainer = TrainClf(train_vecs, train_y, clf)
    trained_clf = trainer.train_clf()
    print("Model Trained")

    infer = PredictClf(test_vecs, trained_clf)
    pred_y = infer.predict_clf()
    print("Prediction Done")

    print(f"Accuracy: {accuracy_score(pred_y, test_y)}")
    print(f"MacroF1: {f1_score(pred_y, test_y, average='macro')}")

if __name__ == "__main__":
    main()