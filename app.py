# app.py
import torch
import timm
import uvicorn
import numpy as np
import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ==== CONFIG ====
MODEL_PATH = "hrnet_kulitan_best.pt"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/1dw4e6r5fz2oze3i5p0dp/hrnet_kulitan_best.pt?rlkey=8rpg3jh284qc83p9hhobk84gx&st=47v7514b&dl=1"
IMG_SIZE = 224

# Kulitan class names
CLASS_LABELS = [
    'a', 'b', 'bang', 'be', 'bi', 'bi-i', 'bo', 'bu', 'bu-u', 'da', 'dang', 'de', 'di', 'di-i', 'do', 'du', 'du-u',
    'e', 'ga', 'gang', 'ge', 'gi', 'gi-i', 'go', 'gu', 'gu-u', 'i', 'ka', 'kank', 'ke', 'ki', 'ki-i', 'ko', 'ku', 'ku-u',
    'la', 'lang', 'le', 'li', 'li-i', 'lo', 'lu', 'lu-u', 'ma', 'mang', 'me', 'mi', 'mi-i', 'mo', 'mu', 'mu-u',
    'na', 'nang', 'ne', 'nga', 'ngang', 'nge', 'ngi', 'ngi-i', 'ngo', 'ngu', 'ngu-u', 'ni', 'ni-i', 'no', 'nu', 'nu-u',
    'o', 'pa', 'pang', 'pe', 'pi', 'pi-i', 'po', 'pu', 'pu-u', 'sa', 'sang', 'se', 'si', 'si-i', 'so', 'su', 'su-u',
    'ta', 'tang', 'te', 'ti', 'ti-i', 'to', 'tu', 'tu-u', 'u', 'unknown'
]
NUM_CLASSES = len(CLASS_LABELS)

# ==== DOWNLOAD MODEL FROM DROPBOX ====
def download_model():
    """Download model from Dropbox if not exists locally"""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Downloading from Dropbox...")
        print(f"File size: ~151MB - this may take a few minutes...")
        try:
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\rDownloading: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
            
            urllib.request.urlretrieve(DROPBOX_URL, MODEL_PATH, show_progress)
            print(f"\n✓ Model downloaded successfully to {MODEL_PATH}")
            
            # Verify file size
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✓ Model file size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"\n❌ ERROR downloading model: {e}")
            print(f"\nPlease ensure:")
            print(f"1. The Dropbox link ends with &dl=1")
            print(f"2. The file is publicly accessible")
            print(f"3. Server has internet connection")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)  # Remove partial download
            raise
    else:
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ Model already exists at {MODEL_PATH} ({file_size:.1f} MB)")

# Download model on startup
download_model()

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

# ==== FASTAPI APP ====
app = FastAPI(title="Kulitan OCR API", version="1.0.0")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    # Resize and normalize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    tensor = torch.tensor(image).unsqueeze(0)  # add batch dim
    return tensor

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "Kulitan OCR API is running",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict Kulitan character from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with predicted_class and confidence score
    """
    try:
        # Load image
        image = Image.open(file.file).convert("RGB")
        tensor = preprocess_image(image)

        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

        # Get top prediction
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