# Function to plot training and validation history
def plot_training_history(history):
    # Set up the figure with two subplots (Loss and Accuracy)
    plt.figure(figsize=(8, 4))

    # Plot Training Loss and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training Accuracy and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout for better visualization
    plt.tight_layout()

    # Show the plot
    plt.show()
    
plot_training_history(history)

# Evaluate the model on the test data in batches of size 128
test_loss, test_accuracy = model.evaluate(test_data.batch(128))

# Print the test accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Number of images to display
num_images = 9

# Lists to store images and labels for visualization
images = []
labels = []

# Randomly shuffle the test data and take a sample of num_images
for example in test_data.shuffle(buffer_size=100).take(num_images):
    images.append(example[0].numpy())
    labels.append(example[1].numpy())

# Convert the list of images to a NumPy array
sample_images = np.array(images)

# Make predictions using the trained model on the sample images
predictions = model.predict(sample_images)

# Set up the plotting area
plt.figure(figsize=(10, 10))

# Plot each image along with true and predicted labels
for i in range(num_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])

    # Convert numerical labels to string labels (Dog or Cat)
    true_label = "Dog" if labels[i] == 1 else "Cat"
    pred_label = "Dog" if predictions[i] > 0.5 else "Cat"

    # Display true and predicted labels, with color highlighting correctness
    plt.title(f'True: {true_label}\nPred: {pred_label}', color='green' if true_label == pred_label else 'red')
    plt.axis('off')

# Show the plot
plt.show()