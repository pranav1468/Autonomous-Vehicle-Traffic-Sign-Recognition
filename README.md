---
title: Traffic Sign Recognition AI
emoji: "🚦"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Autonomous Vehicle Traffic Sign Recognition

Traffic sign recognition demo built with TensorFlow and Streamlit. The app classifies an uploaded road sign image into one of 43 German Traffic Sign Recognition Benchmark (GTSRB) classes and shows the top prediction plus the top 5 confidence scores.

[Live Demo](https://huggingface.co/spaces/pranav9752/traffic-sign-recognition) | [GitHub Repository](https://github.com/pranav1468/Autonomous-Vehicle-Traffic-Sign-Recognition)

## Why This Project Matters

Reliable traffic sign classification is a useful building block for autonomous driving and driver-assistance systems. This repository packages the trained model, a simple user-facing web app, tests, CI checks, and Docker deployment so the project is easy to review, run, and extend.

## Features

- Classifies traffic sign images across 43 output classes.
- Accepts `PNG`, `JPG`, `JPEG`, and `WebP` uploads.
- Displays the detected class, confidence score, and top 5 alternatives.
- Uses a Streamlit interface with a Plotly confidence chart.
- Includes a saved `.h5` model file for immediate inference.
- Ships with unit tests, GitHub Actions checks, and a Docker setup for deployment.

## Model and Dataset

- Dataset: German Traffic Sign Recognition Benchmark (GTSRB).
- Training notebook: `traffic-sign-recognition.ipynb`.
- Notebook data load: 39,209 training images.
- Input pipeline: resize to `48x48`, convert to RGB, normalize to `[0, 1]`.
- Architecture: CNN with 3 convolutional blocks, batch normalization, dropout, and a 43-class softmax output layer.
- Inference path: `app.py` -> `src/predict.py` -> `src/model.py`.

The training notebook logs validation accuracy reaching `99.76%` during training. Treat that as a notebook result, not a substitute for production-grade evaluation in safety-critical settings.

## Quick Start

Use Python `3.10` for the smoothest setup. Both the Docker image and GitHub Actions workflow are pinned to Python 3.10.

### 1. Clone the repository

```bash
git clone https://github.com/pranav1468/Autonomous-Vehicle-Traffic-Sign-Recognition.git
cd Autonomous-Vehicle-Traffic-Sign-Recognition
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
streamlit run app.py
```

The Streamlit config in this repo uses port `7860`, so the app should be available at `http://localhost:7860`.

## Testing and Quality Checks

Run the same kinds of checks used by the CI workflow:

```bash
pytest tests -v
flake8 src/ app.py --count --select=E9,F63,F7,F82 --show-source --statistics
black --check src/ tests/
docker build -t traffic-sign-recognition .
```

Current automated tests focus on preprocessing behavior and model input/output shape expectations.

## Project Structure

```text
.
|-- app.py
|-- DEPLOYMENT.md
|-- Dockerfile
|-- models/
|   `-- traffic_sign_classifier.h5
|-- src/
|   |-- data_loader.py
|   |-- model.py
|   `-- predict.py
|-- tests/
|   |-- test_model.py
|   `-- test_preprocessing.py
`-- traffic-sign-recognition.ipynb
```

## Deployment

This repository is configured for Docker-based deployment on Hugging Face Spaces.

- Live Space: [pranav9752/traffic-sign-recognition](https://huggingface.co/spaces/pranav9752/traffic-sign-recognition)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)

To test the container locally:

```bash
docker build -t traffic-sign-recognition .
docker run -p 7860:7860 traffic-sign-recognition
```

## Troubleshooting

- If the app or tests fail with `ModuleNotFoundError: No module named 'tensorflow'`, install dependencies inside a fresh Python 3.10 virtual environment.
- On Windows, TensorFlow may fail to install on newer Python versions such as Python 3.12. If that happens, switch to Python 3.10 to match CI and Docker, or try `pip install tensorflow-cpu`.
- If inference fails, make sure `models/traffic_sign_classifier.h5` exists in the `models/` directory.
- If port `7860` is already in use, run `streamlit run app.py --server.port 8501`.

## Notes

- This project is a computer vision demo, not a safety-certified driving system.
- The repository currently does not include a `LICENSE` file, so no license is declared here.
