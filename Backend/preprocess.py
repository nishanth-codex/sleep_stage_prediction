# This module contains helper functions to take X and y and gives X_train, y_train, X_test, y_test ready as input
import numpy as np
from sklearn.model_selection import train_test_split 
import torch

label_map = {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 4,
                "Sleep stage R": 5
            }

def convert_label(y):
    y = [label_map[val] for val in y]
    return np.array(y)

def split_and_standardize(X,y,test_size=0.2,random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(
                                                    X,y,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=y
                                                )
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    mean = X_train.mean() # Standardize using training data only
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train,X_test,y_train,y_test


def get_model_config(X_train, config = {}):
    X_train = np.array(X_train)
    mean = X_train.mean()
    std = X_train.std()
    config.update(
        {
            "mean":float(mean),
            "std":float(std),
            "label_map":label_map,
        }
    )
    return config

def preprocess_pipeline(X,y,model_config = {}):
    # Preprocessing 
    y = convert_label(y)
    X_train,X_test,y_train,y_test = split_and_standardize(X,y)
    model_config = get_model_config(X_train, model_config)
    return X_train,X_test,y_train,y_test,model_config



