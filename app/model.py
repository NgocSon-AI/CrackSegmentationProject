# app/model.py
import torch
from torchvision import transforms
from models.model import TransMUNet  # Hoặc model bạn dùng

def load_model(weights_path='checkpoints/weights_deepcrack.model'):
    model = TransMUNet()
    checkpoints = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoints['model_weights'], strict=True)
    model.eval()
    return model
