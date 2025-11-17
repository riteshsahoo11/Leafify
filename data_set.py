import tensorflow as tf

# Keras automatically infers the class labels from the folder names!
train_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/PlantVillage/train',
    image_size=(224, 224), # Match your MobileNetV2 input size
    batch_size=32,
    seed=123
)
