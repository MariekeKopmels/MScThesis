import os
import wandb
import torch

""" Stores the model to the disk. In case of storing a best model, the previous best will be overwritten
"""
def save_model(config, model, epoch, best=False):
    print("Saving model")
    
    # Create folder that stores this run's models
    os.chdir(config.model_path)
    os.makedirs(config.run_name, exist_ok=True)
    
    # Move the model to CPU, since my laptop can't open cuda and the server can't open mps
    wandb.unwatch()
    model = model.to("cpu")
    
    # Store the model as this epoch's model
    path = f"{config.model_path}/{config.run_name}/epoch_{epoch}.pt"
    torch.save(model, path)
    
    # Also overwrite the previous best model if it is a new best
    if best:
        path = f"{config.model_path}/{config.run_name}/best.pt"
        torch.save(model, path)
    
    model = model.to(config.device)
    wandb.watch(model, log="all", log_freq=1)
    
    
""" Returns a previously saved model in evaluation mode. Pretrained models are not stored in a run folder.
    If the model needed is from the current run, model_from_current_run=True and the correct folder is accessed.
"""
def load_model(config, model_name="best.pt", model_from_current_run=False):
    if model_from_current_run:
        file_name = f"{config.model_path}/{config.run_name}/{model_name}"
    else:
        file_name = f"{config.model_path}/{model_name}"
        
    model = torch.load(file_name).to(config.device)
    model.eval()
    
    return model
