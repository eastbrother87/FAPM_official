import torch
from torch import nn
#from torchsummary import summary
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import time

class STPM(nn.Module):
    def __init__(self):
        super(STPM, self).__init__()
        self.init_features()
        
        def hook_t(module, input, output):
            self.features.append(output)
            
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)   
        self.resnet.layer2.register_forward_hook(hook_t)
        self.resnet.layer3.register_forward_hook(hook_t)
            
    def init_features(self):
        self.features=[]
          
    def forward(self, x):
        self.init_features()
        _ = self.resnet(x)
        
        return self.features
    