# ML-Text-Classification
Machine Learning Algorithm Toolbox
- Different shallow and deep learning algorithms for text classification

## Proposed Directory Structure

```sh
src
|--- data
|    |--- train.csv/tsv/txt
|    |--- test.csv/tsv/txt
|--- DataLoader.py
|--- Models.py
|--- Trainer.py
|--- Inference.py
|--- FeatureExtractor.py
main.py
```

## Expalination

Central to any ML systen are three key things: 
1. data, on which model will be trained; 
2. features, the representation of data that will be the input to the model; and 
3. algorithm (or model itself), which is going to be trained

A simple pipeline of any ML project can be defined as:
1. Prepare your data - split them into train and test sets. We'll do this using `DataLoader.py`
2. Represent your data - extract features or embed your data, can also be considered the pre-processing step. We'll do this usinf `FeatureExtractor.py`
3. Train the model. Will be done in `Trainer.py`
4. Predict using the model. Will be done using `Inference.py`

`main.py` will be a high-level wrapper to call different classes at a single place.