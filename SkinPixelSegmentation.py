
from types import SimpleNamespace
import wandb
import MyModels
import DataFunctions
import LossFunctions
import torch
import torch.nn as nn
from torch import optim
import warnings
import argparse
import torch

# Dataset: 50859 skin samples and 194198 non-skin samples

# Options for loss function
loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    "CE": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
    "WBCE": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
    "L1": nn.L1Loss(),
}

# train+test mag maximaal 50859*2 (=101718) zijn (door balancen)
# TODO: om dit te fixen kijken naar oversamplen ipv undersamplen.
default_config = SimpleNamespace(
    num_epochs = 15,
    batch_size = 32, 
    train_size = 10000, #Should be 101718*0.9 but that's not an integer
    test_size = 1000, #Should be 101718*0.1 but that's not an integer
    lr = 0.0001, 
    momentum = 0.999, 
    colour_space = "RGB",
    loss_function = "L1", 
    optimizer = "Adam",
    # TODO: To mps
    device = torch.device("cpu"),
    dataset = "Skin_NonSkin", 
    architecture = "SkinClassifier"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--train_size', type=int, default=default_config.train_size, help='trains size')
    argparser.add_argument('--test_size', type=int, default=default_config.test_size, help='test size')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--momentum', type=float, default=default_config.momentum, help='momentum')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--loss_function', type=str, default=default_config.loss_function, help='loss function')
    argparser.add_argument('--optimizer', type=str, default=default_config.optimizer, help='optimizer')
    argparser.add_argument('--device', type=torch.device, default=default_config.device, help='device')
    argparser.add_argument('--dataset', type=str, default=default_config.dataset, help='dataset')
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help='architecture')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

""" Returns the optimizer based on the configurations
"""
def get_optimizer(config, model):
    if config.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    elif config.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "RMSprop":
        return optim.RMSprop(model.parameters(),lr=config.lr)
    else:
        warnings.warn("No matching optimizer found! Used default SGD")
        print(f"Current config.optimizer = {config.optimizer}")
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    

""" Trains the passed model. 
"""
def train(config, model, train_loader, loss_function, optimizer):
    wandb.watch(model, log="all", log_freq=1)
    model.train()
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        epoch_loss = 0.0
        batch = 0
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = 0, 0, 0, 0
        for pixels, labels in train_loader:  
            batch += 1
            # print(f"-------------------------Starting new Batch -------------------------")
            batch_loss, batch_tn, batch_fn, batch_fp, batch_tp = train_batch(config, pixels, labels, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            epoch_tn += batch_tn
            epoch_fn += batch_fn
            epoch_fp += batch_fp
            epoch_tp += batch_tp
        mean_loss = epoch_loss/config.train_size
        accuracy, fn_rate, fp_rate, sensitivity, f1_score = DataFunctions.metrics(batch_tn, batch_fn, batch_fp, batch_tp, pixels=True)
        train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, epoch)
        
""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, pixels, labels, model, optimizer, loss_function):
    pixels, labels = pixels.to(config.device), labels.to(config.device)
    pixels = pixels.float()
    labels = labels.float()
    
    outputs = model(pixels)
        
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    
    tn, fn, fp, tp = DataFunctions.pixel_confusion_matrix(config, outputs, labels)
    
    return loss, tn, fn, fp, tp
    
""" Prints and logs the intermediate training results to WandB.
"""
# TODO: also return IoU during training
def train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, epoch):
    wandb.log({"epoch": epoch, "mean_loss": mean_loss, "accuracy": accuracy, "fn_rate": fn_rate, "fp_rate": fp_rate, "sensitivity": sensitivity, "f1_score": f1_score})
    print(f"Mean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}") 
    
        
def test(config, model, test_loader, loss_function):
    print(f"-------------------------Start testing-------------------------")
    model.eval()
     
    with torch.no_grad():   
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        for pixels, labels in test_loader:  
            pixels, labels = pixels.to(config.device), labels.to(config.device)
            pixels = pixels.float()
            labels = labels.float()
            
            outputs = model(pixels)
                    
            batch_loss = loss_function(outputs, labels)   
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.pixel_confusion_matrix(config, outputs, labels, test=True)
            
            total_loss += batch_loss.item()
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp

        mean_test_loss = total_loss/config.test_size
        accuracy, fn_rate, fp_rate, sensitivity, f1_score = DataFunctions.metrics(total_tn, total_fn, total_fp, total_tp, pixels=True)
        test_log(config, mean_test_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score)

""" Prints and logs the test results to WandB.
"""
def test_log(config, mean_test_loss, test_accuracy, test_fn_rate, test_fp_rate, test_sensitivity, test_f1_score):
    wandb.log({"mean_test_loss": mean_test_loss, "test_accuracy": test_accuracy, "test_fn_rate": test_fn_rate, "test_fp_rate": test_fp_rate, "test_sensitivity": test_sensitivity, "test_f1_score": test_f1_score})
    print(f"Mean loss: {mean_test_loss:.6f}, accuracy: {test_accuracy:.3f}, fn_rate: {test_fn_rate:.3f}, fp_rate:: {test_fp_rate:.3f}, sensitivity: {test_sensitivity:.3f}, f1-score: {test_f1_score:.3f}")
    
def model_pipeline(hyperparameters):
    # Start WandB
    with wandb.init(project="pixel_skin_segmentation", config=hyperparameters): #mode="disabled", 
        # set hyperparameters
        config = wandb.config
        
        run_name = f"Train_size:{config.train_size}_Colourspace:{config.colour_space}_Num_epochs:{config.num_epochs}"
        wandb.run.name = run_name
        
        model = MyModels.SkinClassifier()
        train_loader, test_loader = DataFunctions.load_pixel_data(config, [], [], ("YCrCb" in config.colour_space))
        loss_function = loss_dictionary[config.loss_function]
        optimizer = get_optimizer(config, model)
        
        train(config, model, train_loader, loss_function, optimizer)
        test(config, model, test_loader, loss_function)

if __name__ == '__main__':
    parse_args()
    model = model_pipeline(default_config)