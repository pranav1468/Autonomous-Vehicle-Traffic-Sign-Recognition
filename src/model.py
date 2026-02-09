"""Traffic Sign Recognition Model Definition"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)


class TrafficSignModel:
    """CNN Model for Traffic Sign Classification"""

    # 43 German Traffic Sign classes
    CLASS_NAMES = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5 metric tons",
        11: "Right-of-way at the next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5 metric tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve to the left",
        20: "Dangerous curve to the right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing by vehicles over 3.5 metric tons",
    }

    def __init__(self):
        self.model = None
        self.input_shape = (48, 48, 3)
        self.num_classes = 43

    def build_model(self):
        """Build the CNN architecture matching the notebook"""
        model = Sequential(
            [
                # First Conv Block
                Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=self.input_shape),
                BatchNormalization(),
                Conv2D(32, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.2),
                # Second Conv Block
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.3),
                # Third Conv Block
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.4),
                # Dense Layers
                Flatten(),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
        return model

    def load_weights(self, weights_path):
        """Load pre-trained weights"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(weights_path)

    def load_model(self, model_path):
        """Load complete model from file"""
        self.model = tf.keras.models.load_model(model_path)

    def get_class_name(self, class_id):
        """Get human-readable class name from class ID"""
        return self.CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
