# Traffic Sign Recognition Pipeline

This repository contains a complete pipeline for classifying traffic sign images using Deep Learning and a Streamlit-based inference application.

## Architecture

1.  **Traffic Sign Classification (CNN)**:
    - Defines a Convolutional Neural Network with 3 convolutional blocks, batch normalization, and dropout.
    - Loads the trained `traffic_sign_classifier.h5` model for inference.
    - Predicts one of 43 traffic sign classes from the GTSRB label set.
    
2.  **Inference Application (Streamlit)**:
    - Preprocesses uploaded images by resizing them to `48x48`, converting them to RGB, and normalizing pixel values.
    - Uses `src/predict.py` to run inference on the trained model.
    - Displays the top prediction, confidence score, and top 5 alternatives in the web interface.

## Project Structure

```
Project/
├── src/
│   ├── data_loader.py     # Image preprocessing and dataset helpers
│   ├── model.py           # CNN architecture and class label mapping
│   ├── predict.py         # Prediction wrapper for inference
│   └── __init__.py
├── models/
│   └── traffic_sign_classifier.h5 # Trained TensorFlow model
├── tests/
│   ├── test_model.py      # Model shape and class mapping tests
│   ├── test_preprocessing.py # Image preprocessing tests
│   └── __init__.py
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── .github/
│   └── workflows/
│       └── test.yml       # CI pipeline
├── app.py                 # Streamlit application
├── Dockerfile             # Docker deployment setup
├── DEPLOYMENT.md          # Deployment guide
├── requirements.txt       # Dependencies
├── traffic-sign-recognition.ipynb # Training notebook
└── README.md              # This file
```

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Web Application
Launches the Streamlit interface for uploading and classifying traffic sign images.
```bash
streamlit run app.py --server.port 7860
```

Arguments:
- `app.py`: Streamlit entry point for the web application.
- `--server.port`: Port used for the local app run (default in this project: `7860`).

### 2. Docker Deployment
Builds and runs the traffic sign recognition app in a container.
```bash
docker build -t traffic-sign-recognition .
docker run -p 7860:7860 traffic-sign-recognition
```

Arguments:
- `-t`: Assigns the Docker image name `traffic-sign-recognition`.
- `-p`: Maps local port `7860` to container port `7860`.

## Verification
To verify the pipeline logic, run the included test suite (requires dependencies):
```bash
pytest tests -v
```
This test suite validates image preprocessing, model input and output shapes, and traffic sign class-name mapping.
