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
from neuralnetwork import  SimpleModel, train_model, test_model
from utils import save_all
from pathlib import Path

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from Encryption.encryption_functions import encrypt_payload, decrypt_payload
from Communication.URLS import CLOUD_URL, OWNER_URL, USER_URL



# Key files:
USER_PUBLIC_KEY_FILE = "All_Keys/User/public.json"
CLOUD_PUBLIC_KEY_FILE = "All_Keys/Cloud/cloud_public.txt"


OWNER_PUBLIC_KEY_FILE = "All_Keys/Owner/owner_public.txt"
OWNER_PRIVATE_KEY_FILE = "All_Keys/Owner/owner_private.json"




PROTOCOL_VERSION = 1
KEY_SIZE = 32
NONCE_SIZE = 12
REPLAY_WINDOW_SECONDS = 60
MASK32 = 0xffffffff
OAEP_LABEL = b"secure-data-communicator/v1"


owner_data = {
    "owner_id":"alice",
    "api_key": "owner_API_K_12_@"
    }

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/deploy")
def DeployModel():
    model_path = "sleep_model.pth"
    config_path = "model_config.json"
    #cloud_pk_filepath = "cloud_public.txt" 
    
    if not os.path.exists(CLOUD_PUBLIC_KEY_FILE):
        raise HTTPException(
            status_code=404,
            detail=f"Deployment halted: Public key file '{CLOUD_PUBLIC_KEY_FILE}' was not found. Please obtain it from the Cloud Server."
        )
        
    try:
        with open(CLOUD_PUBLIC_KEY_FILE, "r", encoding="utf-8") as f:
            CLOUD_PUBLIC_KEY_STR = f.read().strip() # .strip() removes accidental spaces or line breaks
            
        if not CLOUD_PUBLIC_KEY_STR:
            raise ValueError("The public key file is empty.")
    
    except Exception as read_err:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read the public key configuration file: {read_err}"
        )
    try:
        parts = CLOUD_PUBLIC_KEY_STR.strip().split(":")
        cloud_pub_key = (int(parts[0]), int(parts[1]))
    except Exception:
        raise HTTPException(status_code=500,detail="Internal Error: Cloud Server public key configuration string format is invalid."
    )
    with open(model_path, "rb") as model_file, open(config_path, "rb") as config_file:
        model_bytes = model_file.read()
        config_bytes = config_file.read()
    try: 
        encrypted_model_bytes = encrypt_payload(model_bytes, cloud_pub_key)

        model_base64_str = base64.b64encode(encrypted_model_bytes).decode('utf-8')

        model_json_data = {
            "sender": "owner",
            "version": PROTOCOL_VERSION,
            "file_type": "model_weights",
            "timestamp_created": int(time.time()),
            "payload_base64": model_base64_str
        }
        
        model_output_path = "payload_model_transit.json"
        with open(model_output_path, "w", encoding="utf-8") as f:
            json.dump(model_json_data, f, indent=4)
    
    except Exception as err:
        raise HTTPException(status_code=500,detail=f"Cryptographic tracking engine execution fault: {err}"
        )
    
    try: 
        encrypted_config_bytes = encrypt_payload(config_bytes, cloud_pub_key)

        config_base64_str = base64.b64encode(encrypted_config_bytes).decode('utf-8')

        config_json_data = {
            "sender": "owner",
            "version": PROTOCOL_VERSION,
            "file_type": "Config",
            "timestamp_created": int(time.time()),
            "payload_base64": config_base64_str
        }
        
        config_output_path = "payload_config_transit.json"
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(config_json_data, f, indent=4)

    except Exception as err:
        raise HTTPException(status_code=500,detail=f"Cryptographic tracking engine execution fault: {err}"
        )
    with open("payload_model_transit.json", "r", encoding="utf-8") as model_packet, \
         open("payload_config_transit.json", "r", encoding="utf-8") as config_packet:
        
        upload_files = {
            "model_packet": ("payload_model_transit.json", model_packet, "application/json"),
            "config_packet": ("payload_config_transit.json", config_packet, "application/json")
            }
        
        response = requests.post(
            f"{CLOUD_URL}/deploy",
             files=upload_files,
             headers=owner_data,
             timeout=120
            )
        
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code,detail=f"Cloud Server rejected secure deployment envelope: {response.text}")
    return response.json()


@app.post("/deploy_")
def DeployModelWorking():
    try:
        with open("sleep_model.pth", "rb") as model, \
             open("model_config.json", "rb") as config:
            response = requests.post(
                f"{CLOUD_URL}/deploy",
                files={
                    "model_file": ("sleep_model.pth", model, "application/octet-stream"),
                    "config_file": ("model_config.json", config, "application/json"),
                },
                headers=owner_data,
                timeout=60,
            )
    except FileNotFoundError:
        print("Model not found")
        raise HTTPException(404, "No trained model found. Build a model first.")
    except requests.RequestException:
        raise HTTPException(503, "Cloud server unreachable")
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.json().get("detail", "Deploy failed"))
    return response.json()

def BuildModel(data_file_path):

    # MODEL SPEC 
    model_config = {
        "d_in":6000,
        "d_h":64,
        "d_out":6
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
    model = train_model(model, X_train, y_train, 200)
    test_model(model, X_test, y_test)

    save_all(model, model_config) # Filenames can be changed by accessing parameters model_file_path, model_config_file_path respectively

build_state = {"state": "idle", "detail": ""}

def _run_build(path):
    try:
        BuildModel(path)
        build_state.update(state="done", detail="Model trained. sleep_model.pth and model_config.json saved.")
    except Exception as e:
        build_state.update(state="error", detail=str(e))

@app.post("/build_model")
def build_model(background: BackgroundTasks, dataset: UploadFile = File(...)):
    if build_state["state"] == "running":
        raise HTTPException(409, "A build is already running")
    if not dataset.filename.endswith(".npz"):
        raise HTTPException(400, "Dataset must be a .npz file with arrays X and y")
    with open("uploaded_dataset.npz", "wb") as f:
        shutil.copyfileobj(dataset.file, f)
    build_state.update(state="running", detail="Training…")
    background.add_task(_run_build, "uploaded_dataset.npz")
    return {"detail": "Build started"}

@app.get("/build_status")
def build_status():
    return build_state


    
@app.post("/remove_user")
def RemoveUser(UID):

    if not os.path.exists(CLOUD_PUBLIC_KEY_FILE):
            raise HTTPException(
                status_code=404,
                detail=f"Deployment halted: Public key file '{CLOUD_PUBLIC_KEY_FILE}' was not found. Please obtain it from the Cloud Server."
            )
            
    try:
            with open(CLOUD_PUBLIC_KEY_FILE, "r", encoding="utf-8") as f:
                CLOUD_PUBLIC_KEY_STR = f.read().strip() # .strip() removes accidental spaces or line breaks
                
            if not CLOUD_PUBLIC_KEY_STR:
                raise ValueError("The public key file is empty.")
        
    except Exception as read_err:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read the public key configuration file: {read_err}"
            )
    try:
            parts = CLOUD_PUBLIC_KEY_STR.strip().split(":")
            cloud_pub_key = (int(parts[0]), int(parts[1]))
    except Exception:
            raise HTTPException(status_code=500,detail="Internal Error: Cloud Server public key configuration string format is invalid.")
    
    try:         
            plaintext_token_payload={"UID": UID}
            serialized_token = json.dumps(plaintext_token_payload).encode("utf-8")
            encrypted_token_bytes = encrypt_payload(serialized_token, cloud_pub_key)
            encrypted_token_base64 = base64.b64encode(encrypted_token_bytes).decode('utf-8')
             
             # 3. Construct the outer transport return wrapper envelope (leaves status visible for networking)
            return_payload = {
                "sender": "owner",
                "version": PROTOCOL_VERSION,
                "encrypted_result": encrypted_token_base64, # <--- Your sealed authorization credentials pack
                "timestamp_processed": int(time.time())
                    }

    except Exception as crypto_err:
            raise RuntimeError(f"Cryptographic tracking module login payload fault: {crypto_err}")

    
    response = requests.post(
        f"{CLOUD_URL}/remove_user",
        json = return_payload,
        headers=owner_data,
        timeout=10
    )

    status_code = response.status_code
        
    if status_code != 200:
            print(f"[✘] Cloud rejected administrative pending request query. Code: {status_code}")
            return response.json(), status_code
    server_envelope = response.json() 

        
    with open(OWNER_PRIVATE_KEY_FILE, "r", encoding="utf-8") as f:
            owner_key_data = json.load(f)
            
        # Reconstruct the owner's secret math parameters (d and n exponents)
    owner_private_key = (int(owner_key_data["d"]), int(owner_key_data["n"]))

    try:
            # 4. Isolate the sealed base64 encrypted pending database string
            base64_ciphertext = server_envelope["encrypted_result"]
            raw_ciphertext = base64.b64decode(base64_ciphertext)
            
            # 5. Decrypt using your zero-third-party hybrid verification engine
            decrypted_bytes = decrypt_payload(raw_ciphertext, owner_private_key)
            plaintext_pending_requests = json.loads(decrypted_bytes.decode("utf-8"))
            
            
            # Property 7: Explicit volatile memory sanitation ahead of exit paths
            del  raw_ciphertext, decrypted_bytes
            
            # Return the clean plaintext data dictionary along with the successful 200 status code
            return plaintext_pending_requests, status_code

    except Exception as crypto_err:
            raise RuntimeError(f"Pipeline failure unboxing secure pending requests data: {crypto_err}")

      


@app.get("/pending_user_requests")
def pending_user_requests(mode :str = "owner"):
    response = requests.get(
        f"{CLOUD_URL}/pending_user_requests",
        params={"mode": mode},
        headers=owner_data
    )
    #response.raise_for_status()
    return response.json(), response.status_code



@app.post("/owner_decision")
def owner_decision(decisions: dict):
    print("ABC")
    if not os.path.exists(CLOUD_PUBLIC_KEY_FILE):
                raise HTTPException(
                    status_code=404,
                    detail=f"Deployment halted: Public key file '{CLOUD_PUBLIC_KEY_FILE}' was not found. Please obtain it from the Cloud Server."
                )
    print("CLOUD PUBLIC KEY FILE CHECK")
                
    try:
                with open(CLOUD_PUBLIC_KEY_FILE, "r", encoding="utf-8") as f:
                    CLOUD_PUBLIC_KEY_STR = f.read().strip() # .strip() removes accidental spaces or line breaks
                    
                if not CLOUD_PUBLIC_KEY_STR:
                    raise ValueError("The public key file is empty.")
            
    except Exception as read_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read the public key configuration file: {read_err}"
                )
    print("Got the cloud public key")
    try:
                parts = CLOUD_PUBLIC_KEY_STR.strip().split(":")
                cloud_pub_key = (int(parts[0]), int(parts[1]))
    except Exception:
                raise HTTPException(status_code=500,detail="Internal Error: Cloud Server public key configuration string format is invalid.")
        
    try:         
                plaintext_token_payload=decisions
                serialized_token = json.dumps(plaintext_token_payload).encode("utf-8")
                encrypted_token_bytes = encrypt_payload(serialized_token, cloud_pub_key)
                encrypted_token_base64 = base64.b64encode(encrypted_token_bytes).decode('utf-8')
                 
                 # 3. Construct the outer transport return wrapper envelope (leaves status visible for networking)
                return_payload = {
                    "sender": "owner",
                    "version": PROTOCOL_VERSION,
                    "encrypted_result": encrypted_token_base64, # <--- Your sealed authorization credentials pack
                    "timestamp_processed": int(time.time())
                        }
    except Exception as crypto_err:
                raise RuntimeError(f"Cryptographic tracking module login payload fault: {crypto_err}")
    print("Encryption done!")
    
    response = requests.post(f"{CLOUD_URL}/owner_decision", json=return_payload, headers=owner_data, timeout=10)

    status_code = response.status_code
            
    if status_code != 200:
            print(f"[✘] Cloud rejected administrative pending request query. Code: {status_code}")
            return response.json(), status_code
    server_envelope = response.json() 
    
            
    with open(OWNER_PRIVATE_KEY_FILE, "r", encoding="utf-8") as f:
                owner_key_data = json.load(f)
                
            # Reconstruct the owner's secret math parameters (d and n exponents)
                owner_private_key = (int(owner_key_data["d"]), int(owner_key_data["n"]))
    
    try:
                # 4. Isolate the sealed base64 encrypted pending database string
                base64_ciphertext = server_envelope["encrypted_result"]
                raw_ciphertext = base64.b64decode(base64_ciphertext)
                
                # 5. Decrypt using your zero-third-party hybrid verification engine
                decrypted_bytes = decrypt_payload(raw_ciphertext, owner_private_key)
                plaintext_pending_requests = json.loads(decrypted_bytes.decode("utf-8"))
                
                
                # Property 7: Explicit volatile memory sanitation ahead of exit paths
                del  raw_ciphertext, decrypted_bytes
                
                # Return the clean plaintext data dictionary along with the successful 200 status code
                return plaintext_pending_requests, status_code
    
    except Exception as crypto_err:
                raise RuntimeError(f"Pipeline failure unboxing secure pending requests data: {crypto_err}")


@app.post("/user_data")
def get_cloud_user_data():
    response = requests.get(
            f"{CLOUD_URL}/user_data",
            headers=owner_data
        )
    return response.json()


@app.get("/model_info")
def model_info():
    info = {}
    for name in ("sleep_model.pth", "model_config.json"):
        p = Path(name)
        info[name] = {"size": p.stat().st_size, "modified": p.stat().st_mtime} if p.exists() else None
    return info

@app.get("/users")
def list_users():
    r = requests.get(f"{CLOUD_URL}/user_data", headers=owner_data, timeout=10)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.json().get("detail", "Could not load users"))
    return {
        uid: {k: v for k, v in u.items() if k != "password"}
        for uid, u in r.json().items() if isinstance(u, dict)
    }

@app.get("/model_info")
def model_info():
    info = {}
    for name in ("sleep_model.pth", "model_config.json"):
        p = Path(name)
        info[name] = {"size": p.stat().st_size, "modified": p.stat().st_mtime} if p.exists() else None
    return info