# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2  # Import L2 regularizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np
import random


# %%
# Function to preprocess spectrograms
def preprocess_spectrogram(spectrogram, target_size):
    # Normalize
    spectrogram = spectrogram / np.max(spectrogram)
    # Resize to match MobileNet input
    spectrogram = tf.image.resize(spectrogram, target_size)
    # Convert to 3 channels (stack the same data for all channels)
    spectrogram = tf.image.grayscale_to_rgb(spectrogram)
    return spectrogram.numpy()

# Data augmentation for spectrograms
def augment_spectrogram(spectrogram):
    # Time shifting
    shift = random.randint(-10, 10)
    spectrogram = np.roll(spectrogram, shift, axis=1)

    # Frequency masking
    freq_mask = random.randint(0, 10)
    spectrogram[:, freq_mask:freq_mask + 10, :] = 0

    # Time masking
    time_mask = random.randint(0, 10)
    spectrogram[time_mask:time_mask + 10, :, :] = 0

    return spectrogram


# %%
initial_learning_rate = 1e-4  # Start with a higher learning rate
decay_steps = 1000            # Number of steps before applying decay
decay_rate = 0.9              # Rate of decay
learning_rate_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True  # Decay happens in discrete intervals
)

# %%
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

def create_regression_model(input_shape):
    # Define Exponential Decay for the learning rate
    initial_learning_rate = 1e-4
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Load MobileNetV2 with pre-trained weights from ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze the first 100 layers
        layer.trainable = False

    # Add custom layers with L2 regularization
    model = Sequential([
        base_model,
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.009)),  # L2 regularization
        BatchNormalization(),
        ReLU(),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.009)),  # L2 regularization
        Dropout(0.5),
        Dense(1, kernel_regularizer=l2(0.009))  # Output layer for regression
    ])

    # Compile the model with the learning rate schedule
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
        loss='mean_squared_error',  # Loss function for regression
        metrics=['mean_absolute_error']  # Metrics for regression
    )

    return model


# %%
# Input shape for spectrograms
input_shape = (128, 128, 3)

# Create the model for regression
model = create_regression_model(input_shape)

# Compile the model
model.compile(
    #optimizer=tf.keras.optimizers.Adam(learning_rate=ExponentialDecay(
     #   initial_learning_rate=1e-4,
      #  decay_steps=1000,
       # decay_rate=0.9,
        #staircase=True
    #)),
    loss='mean_squared_error',  # Loss function for regression
    metrics=['mean_absolute_error']  # Metrics for regression
)

# Display model summary
model.summary()


# %%
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(image_folder, txt_folder, target_size):
    images = []
    labels = []

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            # Read and parse the label as a list of floats
            with open(os.path.join(txt_folder, txt_file), 'r') as file:
                label = list(map(float, file.read().strip().split()))  # Convert space-separated floats to a list

            # Load the corresponding image
            image_name = os.path.splitext(txt_file)[0] + '.jpg'  # Assuming spectrograms are in .jpg format
            image_path = os.path.join(image_folder, image_name)
            if os.path.exists(image_path):
                image = load_img(image_path, target_size=target_size)
                image = img_to_array(image)
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


# %%

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError

def load_data(image_folder, txt_folder, target_size):
    images = []
    labels = []
    processed_files = 0  # Counter for processed files

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            # Read and parse the label as a list of floats
            with open(os.path.join(txt_folder, txt_file), 'r') as file:
                label = list(map(float, file.read().strip().split()))  # Convert space-separated floats to a list

            # Load the corresponding image
            image_name = os.path.splitext(txt_file)[0] + '.jpg'  # Assuming spectrograms are in .jpg format
            image_path = os.path.join(image_folder, image_name)

            if os.path.exists(image_path):
                try:
                    image = load_img(image_path, target_size=target_size)
                    image = img_to_array(image)
                    images.append(image)
                    labels.append(label)
                    processed_files += 1
                except UnidentifiedImageError:
                    print(f"Warning: {image_path} could not be identified as an image and was skipped.")
                except Exception as e:
                    print(f"Warning: An error occurred with {image_path} - {e}")
            else:
                print(f"Warning: Image {image_path} does not exist.")
    
    print(f"Total processed files: {processed_files}")
    return np.array(images), np.array(labels)


# %%
# Paths to dataset
image_folder = '/Users/goutham/Updated/reg_training_images'  # Replace with your spectrogram images path
txt_folder = '/Users/goutham/Updated/reg_training_labels'  # Replace with your labels path

# Load data
X, y = load_data(image_folder, txt_folder, (128, 128))

# Train-test split for regression
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Removed stratify=y

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255  # Normalize image pixel values to [0, 1]
)
val_datagen = ImageDataGenerator(
    rescale=1./255  # Normalize image pixel values to [0, 1]
)

# Generators for training and validation
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
val_generator = val_datagen.flow(X_val, y_val, batch_size=64)


# %%
X_train.shape

# %%
y_train.shape

# %%
# Generators for training and validation
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
val_generator = val_datagen.flow(X_val, y_val, batch_size=64)

# Callbacks for training
callbacks = [
    ModelCheckpoint('best_model_mobilenetv2_l2_regression.keras', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),  # Increased patience slightly for regression
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)  # Increased patience for LR reduction
]


# %%
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=callbacks
)

import os

# Define the directory to save the model
save_dir = '/Users/goutham/Updated'  # You can replace this with your desired directory path

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Full path to save the model
save_path = os.path.join(save_dir, 'mobilenetv2_l2_REGRESSION.keras')

# Save the trained model
model.save(save_path)

print(f"Model saved to: {save_path}")


# %%
model = tf.keras.models.load_model('/Users/goutham/Updated/mobilenetv2_l2_REGRESSION.keras')



# Show the model architecture
model.summary()

# %%
# Load test data
test_image_folder = '/Users/goutham/17k_test_images'  # Replace with your test spectrogram images path
test_txt_folder = '/Users/goutham/17k_test_labels'  # Replace with your test labels path
X_test, y_test = load_data(test_image_folder, test_txt_folder, (128, 128))


# %%
# Evaluate the model on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


# %%
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')


# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2

# Function to preprocess a single image
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or invalid format: {image_path}")
    image = cv2.resize(image, target_size)
    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    return image / 255.0  # Normalize

# Function to load test data from a folder
def load_test_data(image_folder, label_folder, target_size):
    image_paths = []
    labels = []

    for txt_file in os.listdir(label_folder):
        if txt_file.endswith('.txt'):
            with open(os.path.join(label_folder, txt_file), 'r') as file:
                label = int(file.read().strip())
                image_name = os.path.splitext(txt_file)[0] + '.jpg'  # Assuming images are in .jpg format
                image_path = os.path.join(image_folder, image_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(label)
                #else:
                    #print(f"Warning: No matching image for label {txt_file}")

    X = [preprocess_image(img_path, target_size) for img_path in image_paths]
    return np.array(X), np.array(labels)

# Function to evaluate a model on a dataset
def evaluate_model(model, X, y, folder_name):
    print(f"\nEvaluating for folder: {folder_name}")
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Handle ROC AUC calculation
    if len(np.unique(y)) > 1:  # Check if both classes are present
        roc_auc = roc_auc_score(y, y_pred_prob)
    else:
        print("N/A (Only one class in labels)")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc}")

    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix - {folder_name}")
    plt.show()



# %%
# Main script
if __name__ == "_main_":
    # Paths
    master_image_folder = "/Users/goutham/17k_test_images"  # Replace with your master image folder path
    master_label_folder = "/Users/goutham/17k_test_images"  # Replace with your master label folder path

    model_path = "C:/Users/Himanshu/Desktop/FOC_project/saved_model/mobilenetv2_l2_val_loss.keras"  # Replace with your model path
    target_size = (128, 128)  # Image target size

    # Load the model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Iterate through folders in the master folder
    for folder_name in os.listdir(master_image_folder):
        image_folder = os.path.join(master_image_folder, folder_name)
        label_folder = os.path.join(master_label_folder, folder_name)

        # Check if both image and label folders exist
        if os.path.isdir(image_folder) and os.path.isdir(label_folder):
            # Load data
            X_test, y_test = load_test_data(image_folder, label_folder, target_size)

            # Ensure data is not empty
            if len(X_test) > 0 and len(y_test) > 0:
                # Evaluate the model for the current folder
                evaluate_model(model, X_test, y_test, folder_name)
            else:
                print(f"Warning: No valid data found in folder {folder_name}")

# %%



