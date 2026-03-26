---
title: Traffic Sign Recognition AI
emoji: "🚦"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Autonomous Vehicle Traffic Sign Recognition Pipeline

This repository contains a complete pipeline for classifying traffic sign images using a trained Convolutional Neural Network and a Streamlit-based inference application.

## Architecture

1. **Traffic Sign Classification Model (CNN)**:
   - Defines a CNN with 3 convolutional blocks, batch normalization, dropout, and a 43-class softmax output.
   - Loads the trained `traffic_sign_classifier.h5` model for inference.
   - Maps prediction indices to human-readable traffic sign labels.

2. **Inference and Web Application**:
   - Preprocesses uploaded images by resizing them to `48x48`, converting them to RGB, and normalizing pixel values.
   - Runs inference through `src/predict.py`.
   - Displays the predicted class, confidence score, and top 5 alternatives in the Streamlit interface.

## Project Structure

```text
Traffic_project/
|-- src/
|   |-- data_loader.py             # Image preprocessing and dataset helpers
|   |-- model.py                   # CNN architecture and class labels
|   `-- predict.py                 # Prediction wrapper
|-- models/
|   `-- traffic_sign_classifier.h5 # Trained TensorFlow model
|-- tests/
|   |-- test_model.py              # Model shape and class mapping tests
|   `-- test_preprocessing.py      # Image preprocessing tests
|-- .streamlit/
|   `-- config.toml                # Streamlit configuration
|-- .github/
|   `-- workflows/
|       `-- test.yml               # CI checks for linting, tests, and Docker build
|-- app.py                         # Streamlit web application
|-- Dockerfile                     # Container setup
|-- DEPLOYMENT.md                  # Hugging Face Spaces deployment notes
|-- requirements.txt               # Python dependencies
|-- traffic-sign-recognition.ipynb # Training and experimentation notebook
`-- README.md                      # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Web Application

Launches the Streamlit app for uploading and classifying traffic sign images.

```bash
streamlit run app.py
```

Details:
- Model file: `models/traffic_sign_classifier.h5`
- Supported upload formats: `png`, `jpg`, `jpeg`, `webp`
- Configured application port: `7860`

### 2. Run the Docker Deployment

Builds and runs the same application inside a container.

```bash
docker build -t traffic-sign-recognition .
docker run -p 7860:7860 traffic-sign-recognition
```

Details:
- Exposes port `7860`
- Mirrors the deployment configuration used for Hugging Face Spaces

## Verification

To verify the project logic, run the included test suite (requires dependencies):

```bash
pytest tests -v
```

Optional quality checks:

```bash
flake8 src/ app.py --count --select=E9,F63,F7,F82 --show-source --statistics
black --check src/ tests/
```

These checks validate image preprocessing, model input and output expectations, and class-name mapping.
