from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import requests
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np

import secrets
import hashlib
import hmac
import time
import base64
import io
import os
import sqlite3
import shutil
from preprocess import preprocess_pipeline
from neuralnetwork import SimpleModel, train_model, test_model
from utils import save_all
from pathlib import Path


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def BuildModel(data_file_path):

    # MODEL SPEC 
    model_config = {
        "input_dimension":6000,
        "hidden_layers":[128, 64],
        "output_dimension":6
    }

    # Load and get Data ready for Training
    data = np.load(data_file_path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    X_train,X_test,y_train,y_test,model_config = preprocess_pipeline(X,y,model_config)

    # Dimensions from the config
    d_in = model_config["d_in"]
    d_h = model_config["d_h"]
    d_out = model_config["d_out"]

    # Creating the Model
    model = SimpleModel(d_in, d_h, d_out)

    # Training and testing
    model.train_model(X_train, y_train, 200)
    model.test_model(X_test, y_test)

    save_all(model, model_config) # Filenames can be changed by accessing parameters model_file_path, model_config_file_path respectively


@app.get("/model_info")
def model_info():
    info = {}
    for name in ("sleep_model.pth", "model_config.json"):
        p = Path(name)
        info[name] = {"size": p.stat().st_size, "modified": p.stat().st_mtime} if p.exists() else None
    return info