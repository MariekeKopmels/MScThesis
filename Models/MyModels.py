
import torch
import torch.nn as nn
from Models.I3D import InceptionI3d
import torchvision.models as models

NO_PIXELS = 224
    
""" (Pretrained) I3D model used for violence detection.
"""
class I3DViolenceModel(InceptionI3d):
    def __init__(self, config):
        super().__init__()
        
        # Create the model
        # Load the weights and customize the final layer if config.pretrained is True
        if config.pretrained: 
            self.model =  InceptionI3d()
            self.model.load_state_dict(torch.load(config.weights_path))
            self.model.replace_logits(config.num_violence_classes)
        else: 
            self.model = InceptionI3d(num_classes=config.num_violence_classes)
        
        # Sigmoid activation layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Shape of x: [batch_size, num_frames(16), num_channels, 224, 224]
        # Reformat the input to the shape [batch_size, num_channels, num_frames(16), 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        raw_outputs = torch.squeeze(self.model(x), dim=-1)
        outputs = self.sigmoid(raw_outputs)
        return raw_outputs, outputs
    

""" (Pretrained) I3D model used for skin tone prediction.
"""
class I3DSkintoneModel(InceptionI3d):
    def __init__(self, config):
        super().__init__()
        
        # Create the model
        # Load the weights and customize the final layer if config.pretrained is True
        if config.pretrained: 
            self.model = InceptionI3d()
            self.model.load_state_dict(torch.load(config.weights_path))
            self.model.replace_logits(1)
        else: 
            self.model = InceptionI3d(num_classes=1)
            
        # Apply scaling to ensure output in the range [1.0, 5.0]
        self.scale = nn.Parameter(torch.tensor(float(config.num_skintone_classes - 1)))  # Initialize scale parameter
    
    def forward(self, x):
        # Shape of x: [batch_size, num_frames(16), num_channels, 224, 224]
        # Reformat the input to the shape [batch_size, num_channels, num_frames(16), 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        # Pass the input through the I3D model
        x = self.model(x)
        # Apply scaling
        x = torch.sigmoid(x) * self.scale + 1.0
        x = x.squeeze(dim=1)
        return x
    

""" (Pretrained) ResNet 3D model used for skin tone prediction.
"""
class ResNetSkinToneModel(nn.Module):
    def __init__(self, config):
        super(ResNetSkinToneModel, self).__init__()
        # Load pre-trained ResNet-18 model
        self.resnet = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        # Replace the final fully connected layer with a single neuron
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        # Apply scaling to ensure output in the range [1.0, 5.0]
        self.scale = nn.Parameter(torch.tensor(float(config.num_skintone_classes - 1)))  # Initialize scale parameter
    
    def forward(self, x):
        # Shape of x: [batch_size, num_frames(16), num_channels, 224, 224]
        # Reformat the input to the shape [batch_size, num_channels, num_frames(16), 224, 224]
        x = x.permute(0, 2, 1, 3, 4)  # Swap num_frames and channels
        # Pass input through ResNet-18 model
        x = self.resnet(x)
        # Apply scaling
        x = torch.sigmoid(x) * self.scale + 1.0
        return x
