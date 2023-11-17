import cv2
import wandb
import warnings
import DataFunctions

""" Prints and logs intermediate results to WandB.
"""
def log_metrics(mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU, type):
    
    wandb.log({f"{type}accuracy": accuracy, f"{type}fn_rate": fn_rate, f"{type}fp_rate": fp_rate, f"{type}sensitivity": sensitivity, f"{type}f1_score": f1_score, f"{type}IoU": IoU})
    print(f"{type} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
    # Save the model
    # TODO: Save the model, fix: raises errors
    # torch.onnx.export(model, "model.onnx")
    # wandb.save("model.onnx")
    
""" Prints and logs intermediate results to WandB.
"""
def new_log_metrics(mean_loss, tn, fn, fp, tp, type):
    accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
    print(f"{type} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
    
    if type != "batch":
        wandb.log({f"{type}_accuracy": accuracy, f"{type}_fn_rate": fn_rate, f"{type}_fp_rate": fp_rate, f"{type}_sensitivity": sensitivity, f"{type}_f1_score": f1_score, f"{type}_IoU": IoU})
    

""" Logs test examples of input image, ground truth and model output.
"""
def log_example(config, example, target, output, type="validation"):
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
      
    wandb.log({f"Input {type} image":[wandb.Image(example)]})
    wandb.log({f"Target {type} output": [wandb.Image(target)]})
    wandb.log({f"Model {type} greyscale output": [wandb.Image(output)]})
    bw_image = (output >= 0.5).float()
    wandb.log({f"Model {type} bw output": [wandb.Image(bw_image)]})