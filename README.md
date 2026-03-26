# Autonomous Vehicle Traffic Sign Recognition — Pipeline

This repository builds and serves a **deep learning pipeline** for traffic sign classification using the GTSRB dataset and road-scene images.
It covers model training, image preprocessing, inference, testing, and deployment through a Streamlit web application.

### What's included

#### Root files

* **app.py** — Streamlit web app for uploading and classifying traffic sign images
* **traffic-sign-recognition.ipynb** — notebook for CNN training and evaluation
* **Dockerfile** — container setup for deployment
* **DEPLOYMENT.md** — deployment instructions
* **requirements.txt** — dependencies list
* **README.md** — project overview and workflow

---

#### src/ — main project modules

| File / Folder | Description |
|---------------|-------------|
| **data_loader.py** | Image preprocessing, resizing, normalization, and dataset helpers |
| **model.py** | CNN architecture and 43-class traffic sign label mapping |
| **predict.py** | Inference wrapper that loads the trained model and returns top predictions |

---

#### tests/ — validation scripts

Contains test suites to verify project components:

* **test_model.py** — validates model architecture, output shape, and class mapping
* **test_preprocessing.py** — validates image resizing, normalization, and augmentation setup

---

#### outputs (runtime)

Created when you train, test, or run the app locally.
Stores or uses:

* Trained model file (`models/traffic_sign_classifier.h5`)
* Prediction results shown in the Streamlit interface
* Docker image artifacts from local builds
* Test results from `pytest`

---

### Quick setup

**PowerShell**

```powershell
# Create virtual environment
python -m venv venv; .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

### Run the full pipeline

#### Pipeline 1: Model Training (CNN)

```powershell
# Step 1: Open the training notebook
jupyter notebook traffic-sign-recognition.ipynb

# Step 2: Run the notebook cells to train and save the model
```

**Outputs:**

* Trained model → `traffic_sign_classifier.h5`
* App-ready model → `models/traffic_sign_classifier.h5`

---

#### Pipeline 2: Web Inference (Streamlit App)

```powershell
# Step 1: Launch the web application
streamlit run app.py --server.port 7860
```

**Outputs:**

* Web app → `http://localhost:7860`
* Predictions → displayed in the browser interface

---

### Run tests

```powershell
python -m pytest tests/test_model.py
python -m pytest tests/test_preprocessing.py
```

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| **TensorFlow install error on Windows** | Try `pip install tensorflow-cpu` |
| **Model file not found** | Check that `models/traffic_sign_classifier.h5` exists |
| **Port already in use** | Run `streamlit run app.py --server.port 8502` |
| **Module import errors** | Run commands from the project root folder |

---

### Notes

* The CNN predicts **43 traffic sign classes** from the GTSRB label set.
* **Image preprocessing** resizes inputs to `48x48` and normalizes pixel values to `[0, 1]`.
* **Inference** uses the trained Keras model and returns the top 5 predictions in the Streamlit app.
* The project includes Docker and GitHub Actions support for deployment and verification.
