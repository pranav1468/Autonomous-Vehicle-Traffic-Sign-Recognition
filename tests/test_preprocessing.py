"""Test data preprocessing functionality"""

import pytest
import numpy as np
from src.data_loader import preprocess_image, create_data_generator


def test_preprocess_image_shape():
    """Test that preprocessing returns correct shape"""
    # Create dummy image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Preprocess
    processed = preprocess_image(test_image)

    # Should be (1, 48, 48, 3) after preprocessing
    assert processed.shape == (1, 48, 48, 3)


def test_preprocess_image_normalization():
    """Test that image is normalized to [0, 1]"""
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    processed = preprocess_image(test_image)

    # Values should be between 0 and 1
    assert processed.min() >= 0
    assert processed.max() <= 1
    assert processed.dtype == np.float32


def test_preprocess_different_sizes():
    """Test preprocessing handles different input sizes"""
    sizes = [(50, 50, 3), (200, 200, 3), (30, 60, 3)]

    for size in sizes:
        test_image = np.random.randint(0, 255, size, dtype=np.uint8)
        processed = preprocess_image(test_image)

        # All should be resized to (1, 48, 48, 3)
        assert processed.shape == (1, 48, 48, 3)


def test_data_generator():
    """Test that data augmentation generator is created"""
    datagen = create_data_generator()

    assert datagen is not None
    assert datagen.rotation_range == 15
    assert datagen.width_shift_range == 0.1
