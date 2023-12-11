# Create a sequential model
model = tf.keras.Sequential([
    # Convolutional layer with 64 filters, each with a 3x3 kernel, and input shape of (128, 128, 3)
    Conv2D(64, (3, 3), input_shape=(128, 128, 3)),

    # Batch normalization to normalize the activations of the convolutional layer
    BatchNormalization(),

    # Activation function (Rectified Linear Unit - ReLU) to introduce non-linearity
    Activation('relu'),

    # Max pooling layer with a 2x2 pool size for spatial down-sampling
    MaxPooling2D(2, 2),

    # You can repeat the above pattern to add more convolutional layers for deeper representation learning

    # Flatten layer to convert the 2D feature maps to a 1D vector
    Flatten(),

    # Fully connected dense layer with 256 neurons
    Dense(256),

    # Batch normalization for the dense layer
    BatchNormalization(),

    # Activation function (ReLU) for the dense layer
    Activation('relu'),

    # You can repeat the above pattern to add more hidden layers for deeper learning

    # Output layer with a single neuron and sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])


# Compile the model using Adam optimizer with a learning rate of your choice
model.compile(optimizer=Adam(learning_rate=0.01),
              # Binary cross-entropy loss is suitable for binary classification tasks
              loss='binary_crossentropy',
              # Metrics to monitor during training, accuracy is a common metric for classification
              metrics=['accuracy'])

# Train the model using the training data in batches of size 256, for 32 epochs
history = model.fit(train_data.batch(128),
                    epochs=8,
                    # Use the validation set for monitoring performance during training
                    validation_data=validation_data.batch(64))