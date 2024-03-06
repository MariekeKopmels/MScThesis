import cv2
import wandb
import warnings
import Data.DataFunctions as DataFunctions
import numpy as np

""" Prints intermediate results to WandB, also logs them to WandB if not in batch stage.
"""
def log_metrics(config, mean_loss, tn, fn, fp, tp, stage):
    accuracy, fn_rate, fp_rate, sensitivity, f1_score, f2_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
    print(f"{stage} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, f2_score: {f2_score:.3f}, IoU: {IoU:.3f}") 
    
    if stage != "batch":
        wandb.log({f"{stage}_accuracy": accuracy, f"{stage}_fn_rate": fn_rate, f"{stage}_fp_rate": fp_rate, f"{stage}_sensitivity": sensitivity, f"{stage}_f1_score": f1_score, f"{stage}_f2_score": f2_score, f"{stage}_IoU": IoU, f"{stage}_mean_loss": mean_loss})
    
    return IoU

""" Logs intermediate results from the violence model to WandB.
"""    
def log_violence_metrics(config, mean_loss, accuracy, fn_rate, fp_rate, f1_score, f2_score, stage):
    
    print(f"{stage}_mean_loss = {mean_loss}, {stage}_fn_rate = {fn_rate}, {stage}_fp_rate = {fp_rate}, {stage}_f1 = {f1_score}, {stage}_f2 = {f2_score}")
    wandb.log({f"{stage}_accuracy": accuracy, f"{stage}_fn_rate": fn_rate, f"{stage}_fp_rate": fp_rate, f"{stage}_mean_loss": mean_loss, f"{stage}_f1": f1_score, f"{stage}_f2_score": f2_score})
    
""" Log intermediate results from the skin tone model to WandB.
"""
def log_skin_tone_metrics(config, mean_loss, mae, mse, stage):
    print(f"{stage}_mean_loss = {mean_loss}, {stage}_mae = {mae}, {stage}_mse = {mse}")
    wandb.log({f"{stage}_mean_loss": mean_loss, f"{stage}_mae": mae, f"{stage}_mse": mse})
 
""" Logs test examples of input image, ground truth and model output to WandB.
"""
def log_example(config, example, target, output, stage="UNKNOWN"):
    bw_output = (output >= 0.5).float()
    grinch = DataFunctions.make_grinch(config, example, bw_output)
    
    # Change channels from YCrCb, HSV or BGR to RGB
    if config.colour_space == "YCrCb": 
        example = cv2.cvtColor(example, cv2.COLOR_YCR_CB2RGB)
        grinch = cv2.cvtColor(grinch, cv2.COLOR_YCR_CB2RGB)
    elif config.colour_space == "HSV":
        example = cv2.cvtColor(example, cv2.COLOR_HSV2RGB_FULL)
        grinch = cv2.cvtColor(grinch, cv2.COLOR_HSV2RGB_FULL)
    elif config.colour_space == "BGR":
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
    
    return

""" Logs test examples of input video, ground truth and model output to WandB.
"""
def log_video_example(config, example, target, output, stage="UNKNOWN"):
    frames = example.numpy().astype(np.uint8)
    output = output.item()
    if config.architecture == "I3D_Violence":
        output = 1 if output>0.5 else 0
    elif config.architecture == "ResNet_SkinTone":
        output = round(output)
    wandb.log({f"{stage}, target={target.item()}, output={output}": wandb.Video(frames, fps=32)})