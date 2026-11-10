import torch
import json

def save_all(
            model,
            model_config, 
            model_file_path = "sleep_model.pth", 
            model_config_file_path = "model_config.json"
            ):
    torch.save(model.state_dict(), model_file_path)
    with open(model_config_file_path,"w") as file:
        json.dump(model_config, file)

    print("Model and Model configuration has saved succesfully")
