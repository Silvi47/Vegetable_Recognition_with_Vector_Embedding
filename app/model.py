import torch
import torchvision
import torch.nn as nn
from typing import Dict, Union
from quaterion_models.encoders import Encoder
from encoder import VegEncoder

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_encoder = torchvision.models.resnet152(weights=None)
        pre_trained_encoder.fc = nn.Identity()
        return VegEncoder(pre_trained_encoder)

def setup_model(checkpoint_path: str = 'vegetable_recog\encoders\default'):
    model = MyModel()
    model = model.configure_encoders()
    return model.load(checkpoint_path)

def Feature_Extractor(image: torch.Tensor, model: nn.Module):
    return model(image)

if __name__ == '__main__':
    model = setup_model()
    input_ = torch.randn(1, 3, 336, 336)
    output = model(input_)
    print(output.shape)