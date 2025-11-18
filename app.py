# app.py
import torch
import timm
import uvicorn
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ==== CONFIG ====
MODEL_PATH = "hrnet_kulitan_best.pt"
IMG_SIZE = 224

# Kulitan class names
CLASS_LABELS = [
    'a', 'b', 'bang', 'be', 'bi', 'bi-i', 'bo', 'bu', 'bu-u', 'da', 'dang', 'de', 'di', 'di-i', 'do', 'du', 'du-u',
    'e', 'ga', 'gang', 'ge', 'gi', 'gi-i', 'go', 'gu', 'gu-u', 'i', 'ka', 'kank', 'ke', 'ki', 'ki-i', 'ko', 'ku', 
    'ku-u', 'la', 'lang', 'le', 'li', 'li-i', 'lo', 'lu', 'lu-u', 'ma', 'mang', 'me', 'mi', 'mi-i', 'mo', 'mu', 
    'mu-u', 'na', 'nang', 'ne', 'nga', 'ngang', 'nge', 'ngi', 'ngi-i', 'ngo', 'ngu', 'ngu-u', 'ni', 'ni-i', 
    'no', 'nu', 'nu-u', 'o', 'pa', 'pang', 'pe', 'pi', 'pi-i', 'po', 'pu', 'pu-u', 'sa', 'sang', 'se', 'si', 
    'si-i', 'so', 'su', 'su-u', 'ta', 'tang', 'te', 'ti', 'ti-i', 'to', 'tu', 'tu-u', 'u', 'unknown'
]
NUM_CLASSES = len(CLASS_LABELS)

# ==== CHECK MODEL ====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file '{MODEL_PATH}' was not found.\n"
        "Since you're using Git LFS, make sure:\n"
        "1. The file is committed via Git LFS\n"
        "2. Render has 'Git LFS Support' enabled by default\n"
        "3. You triggered a fresh deploy in Render\n"
    )
else:
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✓ Found model locally: {MODEL_PATH} ({size_mb:.1f} MB)")

# ==== LOAD MODEL ====
print("Loading model into memory...")
model = timm.create_model("hrnet_w32", pretrained=False, num_classes=NUM_CLASSES)

ckpt = torch.load(MODEL_PATH, map_location="cpu")
if "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)

model.eval()
print("✓ Model loaded successfully!")

# ==== FASTAPI ====
app = FastAPI(title="Kulitan OCR API", version="1.0.0")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.tensor(image).unsqueeze(0)
    return tensor

@app.get("/")
async def root():
    return {
        "status": "Kulitan OCR API is running",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

        top_idx = int(np.argmax(probs))
        result = {
            "predicted_class": CLASS_LABELS[top_idx],
            "confidence": float(probs[top_idx])
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run locally: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
