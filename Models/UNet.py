# U-Net implementation from Nikhil Tomar. 
# Retrieved from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py

import torch
import torch.nn as nn

# Convolutional block of U-Net
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
     
# Encoder block of U-Net
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        
        return x, p

# Decoder block of U-Net
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x

""" Definition of a U-Net model, with 4 encoder and decoder blocks. 
        Takes as input torch tensors of shape (batch_size, channels, height, width)
        Outputs are the raw and sigmoid outputs of the model in the shape of 
        torch tensors of shape (batch_size, height, width) 
        (One channels as it represents only 1 classification)
        The colour space that the model is trained for is stored in self.colour_space
        
        The raw output contains unprocessed float outputs whereas the regular outputs
        are in the range [0,1].
"""
class UNET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.dims = config.dims
        self.colour_space = config.colour_space
        """ Encoder """
        self.e1 = encoder_block(config.num_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        """ Sigmoid activation layer """
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):      
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        x = self.classifier(d4)
        """ Reformatting """
        # Squeeze output to get the desired shape (batch_size, height, width)
        raw_outputs = torch.squeeze(x)
        
        """ Sigmoid """
        # Put output through activation function to get outputs in range [0,1]
        outputs = self.sigmoid(raw_outputs)
        
        return raw_outputs, outputs

    