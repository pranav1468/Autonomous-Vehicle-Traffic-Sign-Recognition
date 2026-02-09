"""Data processing and preprocessing utilities"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(image, target_size=(48, 48)):
    """
    Preprocess a single image for model prediction

    Args:
        image: Input image (numpy array or file path)
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image ready for model input
    """
    # If image is a file path, read it
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def create_data_generator():
    """Create data augmentation generator for training"""
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
    )
    return datagen


def load_and_preprocess_dataset(data_dir, num_classes=43):
    """
    Load and preprocess the GTSRB dataset

    Args:
        data_dir: Directory containing the dataset
        num_classes: Number of traffic sign classes

    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    import os
    from tensorflow.keras.utils import to_categorical

    images = []
    labels = []

    for class_id in range(num_classes):
        class_path = os.path.join(data_dir, str(class_id))
        if not os.path.exists(class_path):
            continue

        file_names = sorted(os.listdir(class_path))

        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            img = cv2.imread(file_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(class_id)

    # Convert to numpy arrays
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    labels_categorical = to_categorical(labels, num_classes)

    return images, labels_categorical, labels
