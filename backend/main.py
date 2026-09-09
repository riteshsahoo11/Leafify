import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import cv2
import numpy as np
import base64

# --- CONFIGURATION ---
MODEL_PATH = "fast_plant_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURE ---
# The trained model is based on MobileNetV2 architecture.
def get_model(num_classes):
    # Initialize MobileNetV2
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    return model

# --- GRAD-CAM IMPLEMENTATION ---
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Automatically find the last convolutional layer if not specified
        if target_layer is None:
            target_layer = self.find_last_conv_layer(self.model)
            
        # Hook for capturing gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def find_last_conv_layer(self, model):
        """Recursively find the last Conv2d layer in the model."""
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Forward Pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward Pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # 3. Generate Heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and Normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().detach().numpy()

def apply_heatmap(image_pil, heatmap_np):
    """
    Overlays the heatmap on the original PIL image and returns a base64 string.
    """
    # Convert PIL to OpenCV format (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap_np, (img_cv.shape[1], img_cv.shape[0]))
    
    # Convert heatmap to uint8 (0-255)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend images (0.6 original + 0.4 heatmap)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    
    # Convert back to RGB for PIL/Encoding
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


# --- APP SETUP ---
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLASS NAMES & TREATMENT ---
# Note: Ensure this list matches your training data EXACTLY.
CLASS_NAMES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
])

TREATMENT_RECOMMENDATIONS = {
    "Apple___Apple_scab": ["Apply fungicides like captan or sulfur.", "Prune infected branches to improve air circulation.", "Clean up fallen leaves to prevent overwintering spores."],
    "Apple___Black_rot": ["Remove and destroy mummified fruit.", "Prune out dead wood and cankers.", "Use appropriate fungicides during the growing season."],
    "Apple___Cedar_apple_rust": ["Remove nearby juniper hosts if possible.", "Apply fungicides such as myclobutanil.", "Plant resistant apple varieties."],
    "Apple___healthy": ["Continue regular care (watering, fertilizing).", "Monitor for pests.", "Maintain good air circulation."],
    "Tomato___Early_blight": ["Remove infected leaves immediately.", "Apply copper-based fungicides.", "Mulch soil to prevent spores from splashing onto leaves."],
    "Tomato___Late_blight": ["Destroy all infected plants immediately (do not compost).", "Apply fungicides like chlorothalonil preventatively.", "Ensure good airflow and avoid overhead watering."],
    "Tomato___healthy": ["Maintain consistent watering.", "Stake plants to keep them off the ground.", "Monitor for early signs of disease."]
}

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- GLOBAL MODEL VAR ---
model = None
grad_cam = None

@app.on_event("startup")
def load_model():
    global model, grad_cam
    try:
        # Initialize MobileNetV2 architecture
        model = get_model(len(CLASS_NAMES))
        
        # Load weights
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(model)
        
        print(f"Model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("CRITICAL: If this fails, ensure the 'ResNet9' class matches your 'CustomCNN' from training.")
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "mobilenetv2"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. Transform
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad = True # Enable gradients for Grad-CAM

        # 3. Inference (Standard)
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        
        predicted_index = pred_idx.item()
        predicted_class = CLASS_NAMES[predicted_index]
        conf_score = float(confidence.item()) * 100

        # 4. Generate Heatmap
        # We need to run this separately or carefully to ensure hooks capture data
        heatmap_grid = grad_cam.generate_heatmap(input_tensor, predicted_index)
        
        # 5. Overlay Heatmap on Original Image
        heatmap_base64 = apply_heatmap(image, heatmap_grid)

        # 6. Get Treatment
        treatment = TREATMENT_RECOMMENDATIONS.get(predicted_class, 
            TREATMENT_RECOMMENDATIONS.get(predicted_class.replace("_", " "), ["No specific advice available."]))

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{conf_score:.2f}%",
            "treatment": treatment,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)