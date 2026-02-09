"""Test model functionality"""

import pytest
import numpy as np
from src.model import TrafficSignModel


def test_model_build():
    """Test that model can be built"""
    model_wrapper = TrafficSignModel()
    model = model_wrapper.build_model()

    assert model is not None
    assert model_wrapper.model is not None


def test_model_input_shape():
    """Test model expects correct input shape"""
    model_wrapper = TrafficSignModel()
    model_wrapper.build_model()

    # Model should accept (batch, 48, 48, 3)
    test_input = np.random.random((1, 48, 48, 3))
    output = model_wrapper.model.predict(test_input, verbose=0)

    assert output.shape == (1, 43)


def test_model_output_shape():
    """Test model outputs correct number of classes"""
    model_wrapper = TrafficSignModel()
    model_wrapper.build_model()

    test_input = np.random.random((1, 48, 48, 3))
    output = model_wrapper.model.predict(test_input, verbose=0)

    # Should output 43 class probabilities
    assert output.shape[1] == 43


def test_class_names():
    """Test that all class names are defined"""
    model_wrapper = TrafficSignModel()

    # Should have 43 classes
    assert len(model_wrapper.CLASS_NAMES) == 43

    # Test a few specific classes
    assert "Speed limit (20km/h)" in model_wrapper.CLASS_NAMES.values()
    assert "Stop" in model_wrapper.CLASS_NAMES.values()


def test_get_class_name():
    """Test class name retrieval"""
    model_wrapper = TrafficSignModel()

    # Test known class
    assert model_wrapper.get_class_name(14) == "Stop"
    assert model_wrapper.get_class_name(0) == "Speed limit (20km/h)"
