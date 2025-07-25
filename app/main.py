# app/main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import load_model
from app.utils import transform_image, tensor_to_base64, pil_to_base64
import torch
app = FastAPI()
model = load_model()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    tensor_img, pil_img = transform_image(contents)

    with torch.no_grad():
        output = model(tensor_img)
        if isinstance(output, (list, tuple)):
            output = output[0]

        output = torch.sigmoid(output)
        mask = output[0]  # (1, H, W)
        binary_mask = (mask > 0.5).float()



    mask_b64 = tensor_to_base64(binary_mask)
    orig_b64 = pil_to_base64(pil_img)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "orig_img": orig_b64,
        "mask_img": mask_b64
    })
