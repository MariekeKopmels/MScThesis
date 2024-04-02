
import argparse
from types import SimpleNamespace


def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--config', type=str, default='Config/OTS5.config', help='path to the configuration file.')
    argparser.add_argument('--WBCEweight', type=float, default=None, help='Weight for weighted binary cross-entropy loss.')
    argparser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
    argparser.add_argument('--colour_space', type=str, default=None, help='Color space for input data.')
    argparser.add_argument('--lr', type=float, default=None, help='Learning rate for training.')
    argparser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs for training.')
    argparser.add_argument('--dataset_size', type=int, default=None, help='Size of the dataset that is used.')
    argparser.add_argument('--sampletype', type=str, default=None, help='Type of samples that are used. Either "samples" or "grinchsamples".')
    argparser.add_argument('--pretrained', type=str, default=None, help='Load a pretrained (ImageNet) or new model.')
    args = argparser.parse_args()
    return args

""" Returns true is the value passed can be parsed to a float.
"""
def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
""" Returns true if the value passed can be parsed to a boolean.
"""
def isBool(value):
    return value.lower() == "true" or value.lower() == "false"

""" Loads the arguments from the configuration file.
"""
def load_config(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        for line in config_file:
            key, value = line.strip().split('=')
            if value.isdigit():
                config[key.strip()] = int(value)
            elif isFloat(value):
                config[key.strip()] = float(value)
            elif isBool(value):
                config[key.strip()] = (value.lower() == "true")
            else:
                
                config[key.strip()] = value
    return SimpleNamespace(**config)

