
import argparse
from types import SimpleNamespace


def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--config', type=str, default='Config/mac.config', help='path to the configuration file.')
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

