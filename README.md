---
title: Traffic Sign Recognition AI
emoji: "🚦"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Autonomous Vehicle Traffic Sign Recognition Pipeline

This repository contains a complete pipeline for classifying traffic sign images using a Convolutional Neural Network trained on the German Traffic Sign Recognition Benchmark (GTSRB) and deployed through a Streamlit web application.

Live demo: https://huggingface.co/spaces/pranav9752/traffic-sign-recognition

## Architecture

1. **Traffic Sign Classification Model**:
   - Defines a CNN with 3 convolutional blocks, batch normalization, dropout, and a 43-class softmax output.
   - Loads the trained `traffic_sign_classifier.h5` model for inference.
   - Maps prediction indices to human-readable traffic sign labels.

2. **Prediction and Web App Layer**:
   - Preprocesses uploaded images by resizing them to `48x48`, converting to RGB, and normalizing pixel values.
   - Runs inference through `src/predict.py`.
   - Displays the predicted class, confidence score, and top 5 alternatives in the Streamlit interface.

## Project Structure

```text
Traffic_project/
|-- .github/
|   `-- workflows/
|       `-- test.yml              # CI checks for linting, tests, and Docker build
|-- .streamlit/
|   `-- config.toml               # Streamlit configuration
|-- models/
|   `-- traffic_sign_classifier.h5 # Trained TensorFlow model
|-- src/
|   |-- data_loader.py            # Image preprocessing and dataset helpers
|   |-- model.py                  # CNN architecture and class labels
|   `-- predict.py                # Prediction wrapper
|-- tests/
|   |-- test_model.py             # Model shape and class mapping tests
|   `-- test_preprocessing.py     # Image preprocessing tests
|-- app.py                        # Streamlit web application
|-- DEPLOYMENT.md                 # Deployment notes for Hugging Face Spaces
|-- Dockerfile                    # Container setup
|-- requirements.txt              # Python dependencies
|-- traffic-sign-recognition.ipynb # Training and experimentation notebook
`-- README.md                     # This file
```

## Setup

1. Install Python `3.10` and create a virtual environment.
2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Web Application

Starts the Streamlit app for uploading and classifying traffic sign images.

```bash
streamlit run app.py
```

Notes:
- The app uses the model stored at `models/traffic_sign_classifier.h5`.
- Supported upload formats: `png`, `jpg`, `jpeg`, `webp`.
- The app is configured to run on port `7860`.

### 2. Run with Docker

Builds and serves the same application in a container, matching the deployment setup used for Hugging Face Spaces.

```bash
docker build -t traffic-sign-recognition .
docker run -p 7860:7860 traffic-sign-recognition
```

Notes:
- The container exposes port `7860`.
- The Docker image installs the same dependencies listed in `requirements.txt`.

## Verification

To verify the project logic, run the included test suite and quality checks after installing dependencies:

```bash
pytest tests -v
flake8 src/ app.py --count --select=E9,F63,F7,F82 --show-source --statistics
black --check src/ tests/
```

These checks validate image preprocessing, model input and output expectations, class-name mapping, and basic code quality. The notebook and deployment files remain separate from the test suite.
