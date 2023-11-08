#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace
import LossFunctions
import DataFunctions
import UNet
import torch
import wandb
import torch.nn as nn
from torch import optim
from statistics import mean
from torchvision.transforms import ToPILImage


NO_PIXELS = 224 

# Options for loss function
loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    "CE": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
    "L1": nn.L1Loss(),
} 

# Default parameters
default_config = SimpleNamespace(
    num_epochs = 10,
    batch_size = 32, 
    train_size = 128, 
    test_size = 32,
    lr = 0.001, 
    momentum = 0.99, 
    loss_function = "IoU", 
    device = torch.device("mps"),
    dataset = "VisuAAL", 
    architecture = "UNet"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--momentum', type=float, default=default_config.momentum, help='momentum')
    argparser.add_argument('--train_size', type=int, default=default_config.train_size, help='trains size')
    argparser.add_argument('--test_size', type=int, default=default_config.test_size, help='test size')
    argparser.add_argument('--loss_function', type=str, default=default_config.loss_function, help='loss function')
    argparser.add_argument('--dataset', type=str, default=default_config.dataset, help='dataset')
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help='architecture')
    argparser.add_argument('--device', type=torch.device, default=default_config.device, help='device')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

""" Returns dataloaders, model, loss function and optimizer.
"""
def make(config):
    # Fetch data
    train_loader, test_loader = DataFunctions.load_data(config, [], [])
    
    # Make the model
    model = UNet.UNET().to(config.device)

    # Define loss function and optimizer
    loss_function = loss_dictionary[config.loss_function]
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    
    return model, train_loader, test_loader, loss_function, optimizer

""" Trains the passed model. 
"""
def train(config, model, train_loader, loss_function, optimizer):
    # Tell WandB to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=1)
    
    # Store the losses of epochs
    model.train()
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        epoch_loss = 0.0
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = 0, 0, 0, 0
        for images, targets in train_loader:  
            batch_loss, batch_tn, batch_fn, batch_fp, batch_tp = train_batch(config, images, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            epoch_tn += batch_tn
            epoch_fn += batch_fn
            epoch_fp += batch_fp
            epoch_tp += batch_tp
        mean_loss = epoch_loss/config.train_size
        accuracy, fn_rate, fp_rate, sensitivity = DataFunctions.metrics(epoch_tn, epoch_fn, epoch_fp, epoch_tp)
        train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, epoch)
        #TODO: Add validation per training epoch

""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, images, targets, model, optimizer, loss_function):
    
    images, targets = images.to(config.device), targets.to(config.device)
    images = images.float()
    targets = targets.float()
    outputs = model(images)
    
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    
    tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
    return loss, tn, fn, fp, tp

""" Prints and logs the intermediate training results to WandB.
"""
def train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, epoch):
    wandb.log({"epoch": epoch, "mean_loss": mean_loss, "accuracy": accuracy, "fn_rate": fn_rate, "fp_rate": fp_rate, "sensitivity": sensitivity})
    print(f"Mean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}") 
    
""" Tests the performance of a model on the test set. Prints and logs the results to WandB.
"""
def test(config, model, test_loader, loss_function):
    print(f"-------------------------Start testing-------------------------")
    
    model.eval()
    
    # Run the model on test examples
    with torch.no_grad():
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        for images, targets in test_loader:
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            
            # TODO: Logging of input image is a hassle, needs fixing
            DataFunctions.save_image("TestImage.jpg", images[0])
            # image = images[0].to("cpu")
            # print("Dims recieved for printing image: ", image.shape)
            # image = image.permute(1,2,0)
            # print("Dims recieved for printing image: ", image.shape)
            # image = image.numpy()
            # pil_image = ToPILImage()(image)
            wandb.log({"Input image": [wandb.Image("TestImage.jpg")]})
            # wandb.log({"Input image": [wandb.Image(pil_image)]})
            wandb.log({"Target output": [wandb.Image(targets[0])]})
            wandb.log({"Model greyscale output": [wandb.Image(outputs[0])]})
            print("Dims recieved: ", outputs[0].shape)
            bw_image = (outputs[0] >= 0.5).float()
            wandb.log({"Model bw output": [wandb.Image(bw_image)]})
            
            batch_loss = loss_function(outputs, targets)
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.confusion_matrix(outputs, targets)
            
            total_loss += batch_loss.item()
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp

        mean_test_loss = total_loss/config.test_size
        tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
        accuracy, fn_rate, fp_rate, sensitivity = DataFunctions.metrics(tn, fn, fp, tp)
        test_log(config, mean_test_loss, accuracy, fn_rate, fp_rate, sensitivity)

""" Prints and logs the test results to WandB.
"""
def test_log(config, mean_test_loss, test_accuracy, test_fn_rate, test_fp_rate, test_sensitivity):
    wandb.log({"mean_test_loss": mean_test_loss, "test_accuracy": test_accuracy, "test_fn_rate": test_fn_rate, "test_fp_rate": test_fp_rate, "test_sensitivity": test_sensitivity})
    print(f"Mean loss: {mean_test_loss:.6f}, accuracy: {test_accuracy:.3f}, fn_rate: {test_fn_rate:.3f}, fp_rate:: {test_fp_rate:.3f}, sensitivity: {test_sensitivity:.3f}") 
    
    # Save the model
    # TODO: Save the model, fix: raises errors
    # torch.onnx.export(model, "model.onnx")
    # wandb.save("model.onnx")

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # start wandb
    with wandb.init(mode="run", project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # set hyperparameters
        config = wandb.config

        # create model, data loaders, loss function and optimizer
        model, train_loader, test_loader, loss_function, optimizer = make(config)
        # print(f"Model: {model}")

        # train the model
        train(config, model, train_loader, loss_function, optimizer)

        # test the models performance
        test(config, model, test_loader, loss_function)

    return model

if __name__ == '__main__':
    parse_args()
    model = model_pipeline(default_config)