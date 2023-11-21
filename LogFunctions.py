import cv2
import wandb
import warnings
import DataFunctions
import torch

""" Prints and logs intermediate results to WandB.
"""
def log_metrics(mean_loss, tn, fn, fp, tp, stage):
    accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
    print(f"{stage} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
    
    if stage != "batch":
        wandb.log({f"{stage}_accuracy": accuracy, f"{stage}_fn_rate": fn_rate, f"{stage}_fp_rate": fp_rate, f"{stage}_sensitivity": sensitivity, f"{stage}_f1_score": f1_score, f"{stage}_IoU": IoU})
    
""" Stores the model to the disk.
"""
def save_model(config, model, epoch, final=False):
    path = config.model_path + f"epoch_{epoch}.pt"
    if final:
        path = path + f"final.pt"
    # TODO: Checken of dit idd de manier is om het model op te slaan en niet met model.state_dict()
    # torch.save(model, path)

""" Logs test examples of input image, ground truth and model output.
"""
def log_example(config, example, target, output, stage="UNKNOWN"):
    # Change cannels from YCrCb or BGR to RGB
    if config.colour_space == "YCrCb": 
        # print("Converted YCrCb to RGB")
        example = cv2.cvtColor(example, cv2.COLOR_YCR_CB2RGB)
    elif config.colour_space == "RGB":
        # print("Converted BGR to RGB")
        example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
    else:
        warnings.warn("No colour space found!")
        print(f"Current config.colour_space = {config.colour_space}")
      
    wandb.log({f"Input {stage} image":[wandb.Image(example)]})
    wandb.log({f"Target {stage} output": [wandb.Image(target)]})
    wandb.log({f"Model {stage} greyscale output": [wandb.Image(output)]})
    bw_output = (output >= 0.5).float()
    wandb.log({f"Model {stage} bw output": [wandb.Image(bw_output)]})
    
    grinch = DataFunctions.make_grinch(example, bw_output)
    wandb.log({f"Model {stage} grinch output": [wandb.Image(grinch)]})
    
# def store_grinch()