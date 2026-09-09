import torch
import sys
sys.path.append('backend')
from backend.main import ResNet9, CLASS_NAMES

DEVICE = torch.device('cpu')
model = ResNet9(3, len(CLASS_NAMES)) 
state_dict = torch.load('backend/fast_plant_model.pth', map_location=DEVICE)
model.load_state_dict(state_dict)
print("SUCCESS!")
