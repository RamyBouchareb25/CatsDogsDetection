# Importing necessary libraries for numerical operations
import numpy as np

# Importing TensorFlow for machine learning tasks
import tensorflow as tf

# Importing TensorFlow Datasets for convenient access to common datasets
import tensorflow_datasets as tfds

# Importing specific layers and optimizers from Keras for building neural networks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

# Importing Matplotlib for creating visualizations
import matplotlib.pyplot as plt




# Function for preprocessing images before feeding them into the neural network
def preprocess_image(image, label):
    # Resize the input image to the desired dimensions (128x128)
    image = tf.image.resize(image, (128, 128))

    # Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Return the preprocessed image and its corresponding label
    return image, label


# Define the name of the dataset
dataset_name = 'cats_vs_dogs'

# Load the dataset using TensorFlow Datasets, splitting it into training and testing sets
# as_supervised=True ensures that the dataset is loaded as (image, label) pairs
# with_info=True returns additional information about the dataset, such as metadata
(train_data, test_data), info = tfds.load(name=dataset_name, split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True)

# Apply the preprocess_image function to each image in the training and testing sets
train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

# Create a validation set by taking the first 1000 preprocessed images from the training set
validation_data = train_data.take(1000)

# Update the training set by skipping the first 1000 preprocessed images (used for validation)
train_data = train_data.skip(1000)



# Mapping of numerical labels to corresponding animal names
label2animal = {
    1: "Dog",  # Label 1 corresponds to a Dog
    0: "Cat"   # Label 0 corresponds to a Cat
}

# Function to plot images along with their corresponding labels
def plot_images(data, num_images=9):
    # Lists to store images and labels for plotting
    images = []
    labels = []

    # Iterate through the dataset and extract images and labels
    for example in data.take(num_images):
        image, label = example[0], example[1]
        images.append(image.numpy())
        labels.append(label)

    # Set up the plotting area
    plt.figure(figsize=(8, 8))

    # Plot each image along with its corresponding label
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        # Display the animal label based on the numerical label
        plt.title(f'Animal: {label2animal[labels[i].numpy()]}')
        plt.axis('off')

    # Show the plot
    plt.show()
    
    
    
plot_images(train_data)