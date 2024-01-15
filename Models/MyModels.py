import torch
import torch.nn as nn
from Models.i3d import InceptionI3d

NO_PIXELS = 224

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
        Output is a torch tensor of shape (batch_size, height, width) as 
        it represents only 1 channel with values between 0 (background) and 1 (skin).
"""
# TODO: cleanup: verdelen in nn.Sequential blokken encoeder, bottleneck en decoder
class UNET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.dims = config.dims
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
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    # TODO: testen of len(batch_size) works with changing batch_sizes
    def forward(self, inputs):      
        batch_size = len(inputs)
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
        x = self.outputs(d4)
        """ Reformatting """
        # Reshape output to the desired shape (batch_size, height, width)
        outputs = x.reshape(batch_size, self.dims, self.dims)
        
        return outputs
    
""" Network used for pixel skin classifier. Very basic feed forward neural network.
"""
class SkinClassifier(nn.Module):
    def __init__(self):
        super(SkinClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Input: 3 features (R, G, B), Output: 32 hidden units
        self.fc2 = nn.Linear(32, 1)  # Output: 1 unit for binary classification (no activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)  
        x = self.sigmoid(x)
        return x

    
class I3DMultiTaskModel(InceptionI3d):
    def __init__(self, config):
        super(I3DMultiTaskModel, self).__init__()
        
        # Define the layers for the violence and skin colour heads
        self.violence_layers = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, config.num_violence_classes)
        )
        self.skincolour_layers = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, config.num_skincolour_classes)
        )


    def forward(self, x):
        # Reformat the input. 
        # Original input [batch_size, num_frames, channels, 224, 224]
        # Permuted input [batch_size, channels, num_frames, 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pass the input through the base I3D and extract features.
        # Squeeze to get features in shape [batch_size, num_features]
        # Where num_features=1024, as determined by the I3D model definition.
        base_features = super(I3DMultiTaskModel, self).extract_features(x)
        base_features = torch.squeeze(base_features)

        # Pass the features through violence and skin colour prediction heads.
        # Output of shape [batch_size] for violence and [batch_size, num_skincolour_classes] for skin colour        
        violence_output = self.violence_layers(base_features)
        violence_output = torch.squeeze(violence_output)
        skincolour_output = self.skincolour_layers(base_features)
        
        # print(f"{base_features.shape = }")
        # print(f"{violence_output.shape = }")
        # print(f"{skincolour_output.shape = }")

        return violence_output, skincolour_output