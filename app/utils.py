# app/utils.py
import torch
from PIL import Image
import numpy as np
import io, base64
from torchvision import transforms

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor, image


# utils.py
def tensor_to_base64(tensor):
    array = tensor.squeeze().cpu().numpy()
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array).convert("L")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')



def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
