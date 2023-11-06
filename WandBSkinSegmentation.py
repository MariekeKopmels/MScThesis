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


NO_PIXELS = 224 

loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    "CE": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
    "L1": nn.L1Loss(),
} 

default_config = SimpleNamespace(
    num_epochs = 1,
    batch_size = 32, 
    train_size = 128, 
    test_size = 32,
    lr = 0.001, 
    momentum = 0.99, 
    loss_function = "IoU", 
    dataset = "VisuAAL", 
    architecture = "UNet", 
    device = torch.device("mps")
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

def make(config):
    # Fetch data
    train_loader, test_loader = DataFunctions.load_data(config, [], [])
    
    # Make the model
    model = UNet.UNET().to(config.device)

    # Define loss function and optimizer
    loss_function = loss_dictionary[config.loss_function]
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    
    return model, train_loader, test_loader, loss_function, optimizer

def train(config, model, train_loader, loss_function, optimizer):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_function, log="all", log_freq=1)
    
    # Store the losses
    train_losses = []
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_train_loss = 0.0
        # epoch_test_loss = 0.0
        # batch = 0
        for images, targets in train_loader:  
            loss, accuracy, fn_rate, fp_rate, sensitivity = train_batch(config, images, targets, model, optimizer, loss_function)
            epoch_train_loss += loss.item()
        train_epoch_log(config, epoch_train_loss, accuracy, fn_rate, fp_rate, sensitivity, epoch)
            
    train_losses.append(epoch_train_loss/config.train_size)
    print(f"mean train loss: {epoch_train_loss/config.train_size:.6f}")         
    
def train_batch(config, images, targets, model, optimizer, loss_function):
    images, targets = images.to(config.device), targets.to(config.device)
    images = images.float()
    targets = targets.float()
    outputs = model(images)
    
    # print(f"example output: {outputs[0].flatten()}")
    # print(f"example target: {targets[0].flatten()}")
    
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    
    tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
    accuracy, fn_rate, fp_rate, sensitivity = DataFunctions.metrics(tn, fn, fp, tp)
    return loss, accuracy, fn_rate, fp_rate, sensitivity

def train_epoch_log(config, epoch_train_loss, accuracy, fn_rate, fp_rate, sensitivity, epoch):
    wandb.log({"epoch": epoch, "mean_loss": epoch_train_loss, "accuracy": accuracy, 
               "fn_rate": fn_rate, "fp_rate": fp_rate, "sensitivity": sensitivity})
    print(f"mean train loss: {epoch_train_loss/config.train_size:.6f}, accuracy: {accuracy:.3f}, fn: {fn_rate:.3f}, fp:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}") 
    # print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    
def test(config, model, test_loader, loss_function):
    model.eval()
    test_losses = []
    
    # Run the model on some test examples
    with torch.no_grad():
        test_loss = 0.0
        for images, targets in test_loader:
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            
            loss = loss_function(outputs, targets)
            print(f"Loss: {loss}")
            print(f"Loss.item(): {loss.item()}")
            print(f"Test_loss: {test_loss}")
            print(f"Test_losses: {test_losses}")
            test_loss += loss.item()
            
        # TODO: Currently test_losses is ignored
        test_losses.append(test_loss/config.test_size)
        wandb.log({"mean test loss": test_loss/config.test_size})

    # Save the model in the exchangeable ONNX format
    # TODO: resolve error that rises here
    # torch.onnx.export(model, images, "model.onnx")
    # wandb.save("model.onnx")

def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(mode="disabled", project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, loss_function, optimizer = make(config)
        # print(f"Model: {model}")

        # and use them to train the model
        train(config, model, train_loader, loss_function, optimizer)

    # and test its final performance
    # TODO: test function afmaken
    test(config, model, test_loader, loss_function)

    return model

if __name__ == '__main__':
    parse_args()
    model = model_pipeline(default_config)