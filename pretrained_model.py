from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Load the base model, excluding the final classification layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3) # Standard input size for MobileNetV2
)

# Freeze the base layers (the heart of Transfer Learning)
base_model.trainable = False
