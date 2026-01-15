from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import base64
import io
from torchvision.utils import save_image
from model import Autoencoder

app = FastAPI()

# Enable CORS
# Enable CORS
# In production, set ALLOWED_ORIGINS env var to your frontend domain (e.g., "https://yourname.github.io")
# Multiple origins can be comma-separated
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import sys

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model_path = "pure_model_weights.pth" 

print(f"DEBUG: Current Working Directory: {os.getcwd()}")
print(f"DEBUG: Backend Directory content: {os.listdir('.')}")
if os.path.exists('../'):
    print(f"DEBUG: Parent Directory content: {os.listdir('../')}")

try:
    # Check current dir
    if os.path.exists(model_path):
        print(f"DEBUG: Found model in current directory: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    # Check parent dir (if running from backend/)
    elif os.path.exists(os.path.join("..", model_path)):
        model_path_parent = os.path.join("..", model_path)
        print(f"DEBUG: Found model in parent directory: {model_path_parent}")
        model.load_state_dict(torch.load(model_path_parent, map_location=device), strict=False)
    else:
        raise FileNotFoundError(f"Could not find {model_path} in . or ..")

    model.to(device)
    model.eval()
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Print stack trace
    import traceback
    traceback.print_exc()
    model_loaded = False

class LatentVector(BaseModel):
    x: float
    y: float

@app.get("/")
def read_root():
    return {
        "status": "Backend Running",
        "model_loaded": model_loaded,
        "device": str(device)
    }

@app.post("/predict")
def predict(vector: LatentVector):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with torch.no_grad():
        # Create latent vector
        z = torch.tensor([[vector.x, vector.y]], device=device)
        
        # Decode
        # Accessing the decoder directly from the Autoencoder wrapper
        img_tensor = model.decoder(z)
        
        # Post-process: [-1, 1] -> [0, 1]
        img_tensor = (img_tensor * 0.5) + 0.5
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PNG image
        buffer = io.BytesIO()
        save_image(img_tensor.squeeze(0), buffer, format="PNG")
        buffer.seek(0)
        
        # Base64 encode
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {"image": f"data:image/png;base64,{img_str}"}
