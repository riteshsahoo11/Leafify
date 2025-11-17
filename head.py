from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Add layers for your specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_disease_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
