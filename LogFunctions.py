import cv2
import wandb
import warnings
import DataFunctions
import torch
import os
import datetime

def init_device(config):
    # TODO: voor cuda maken, mac eruit slopen
    if config.machine == "TS2":
        # device = torch.cuda.get_device_name(torch.cuda.current_device())
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif config.device == "mac":
        torch.set_default_tensor_type('torch.FloatTensor')
    else: 
        warnings.warn(f"Device type not found, can only deal with cpu or CUDA and is {config.device}")

""" Prints and logs intermediate results to WandB.
"""
def log_metrics(config, mean_loss, tn, fn, fp, tp, stage):
    if config.log:
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
        print(f"{stage} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
        
        if stage != "batch":
            wandb.log({f"{stage}_accuracy": accuracy, f"{stage}_fn_rate": fn_rate, f"{stage}_fp_rate": fp_rate, f"{stage}_sensitivity": sensitivity, f"{stage}_f1_score": f1_score, f"{stage}_IoU": IoU})
    
    return IoU
    
""" Stores the model to the disk.
"""
def save_model(config, model, epoch, final=False):
    print("Saving model")
    if config.log:
        os.chdir(config.model_path)
        date_time = datetime.datetime.now()
        stamp = f"{date_time.day}/{date_time.month}/{date_time.year}-{date_time.hour}:{date_time.minute}"
        folder = f"{stamp}_Pretrained:{config.pretrained}_Dataset:{config.dataset}"
        os.makedirs(folder, exist_ok=True)
        path = config.model_path + f"/{folder}/epoch_{epoch}.pt"
        if final:
            path = config.model_path + f"/{folder}/final.pt"
        # TODO: Checken of dit idd de manier is om het model op te slaan en niet met model.state_dict()
        # Oddity doet het anders(met statedict): https://github.com/oddity-ai/oddity-ml/blob/master/backend/pytorch/utils/persistence.py   
        wandb.unwatch()
        
        # Store the model on CPU, since my laptop can't open cuda and the server can't open mps
        model = model.to("cpu")
        torch.save(model, path)
        model = model.to(config.device)
        wandb.watch(model, log="all", log_freq=1)
        

""" Logs test examples of input image, ground truth and model output.
"""
def log_example(config, example, target, output, stage="UNKNOWN"):
    if config.log:
        bw_output = (output >= 0.5).float()
        grinch = DataFunctions.make_grinch(config, example, bw_output)
        
        # Change cannels from YCrCb or BGR to RGB
        if config.colour_space == "YCrCb": 
            # print("Not converted YCrCb to RGB")
            example = cv2.cvtColor(example, cv2.COLOR_YCR_CB2BGR)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_YCR_CB2BGR)
        elif config.colour_space == "HSV":
            # print("Converted HSV to RGB")
            example = cv2.cvtColor(example, cv2.COLOR_HSV2BGR)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_HSV2BGR)
        elif config.colour_space == "BGR":
            # print("Converted BGR to RGB")
            example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_BGR2RGB)
        else:
            warnings.warn("No colour space found!")
            print(f"Current config.colour_space = {config.colour_space}")
            
        wandb.log({f"Input {stage} image":[wandb.Image(example)]})
        wandb.log({f"Target {stage} output": [wandb.Image(target)]})
        wandb.log({f"Model {stage} greyscale output": [wandb.Image(output)]})
        wandb.log({f"Model {stage} bw output": [wandb.Image(bw_output)]})
        wandb.log({f"Model {stage} grinch output": [wandb.Image(grinch)]})
        
