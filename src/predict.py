"""Prediction utilities for traffic sign recognition"""

import numpy as np
from src.model import TrafficSignModel
from src.data_loader import preprocess_image
import os


class TrafficSignPredictor:
    """Predictor class for traffic sign images"""

    def __init__(self, model_path=None):
        """
        Initialize the predictor

        Args:
            model_path: Path to the trained model file
        """
        self.model_wrapper = TrafficSignModel()

        if model_path is None:
            # Default to models directory
            current_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(current_dir, "models", "traffic_sign_classifier.h5")

        if os.path.exists(model_path):
            self.model_wrapper.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def predict(self, image, top_k=5):
        """
        Predict traffic sign class for an image

        Args:
            image: Input image (file path or numpy array)
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = self.model_wrapper.model.predict(processed_image, verbose=0)
        prediction_probs = predictions[0]

        # Get top-k predictions
        top_indices = np.argsort(prediction_probs)[-top_k:][::-1]

        # Get the top prediction
        top_class = top_indices[0]
        top_confidence = float(prediction_probs[top_class])

        # Get class name
        class_name = self.model_wrapper.get_class_name(top_class)

        # Prepare top-k results
        top_predictions = []
        for idx in top_indices:
            top_predictions.append(
                {
                    "class_id": int(idx),
                    "class_name": self.model_wrapper.get_class_name(idx),
                    "confidence": float(prediction_probs[idx]),
                }
            )

        return {
            "class_id": int(top_class),
            "class_name": class_name,
            "confidence": top_confidence,
            "top_predictions": top_predictions,
            "all_probabilities": prediction_probs.tolist(),
        }

    def predict_batch(self, images):
        """
        Predict traffic signs for multiple images

        Args:
            images: List of images (file paths or numpy arrays)

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


if __name__ == "__main__":
    # Example usage
    predictor = TrafficSignPredictor()
    print("Traffic Sign Predictor initialized successfully!")
    print(f"Model loaded with {predictor.model_wrapper.num_classes} classes")
