model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# This trains only the new classification head
model.fit(train_data, epochs=10, validation_data=val_data)
