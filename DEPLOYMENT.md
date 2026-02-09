# DEPLOYMENT GUIDE: Traffic Sign Recognition on Hugging Face Spaces

## Quick Summary

This project is ready for deployment! Here's what you need to do:

### 1. Install Dependencies Locally (Important for Windows)

The model was trained with TensorFlow, but specific versions aren't available on Windows. Use:

```bash
cd c:\\Users\\prana\\OneDrive\\Desktop\\Traffic_project
.\\venv\\Scripts\\activate
pip install tensorflow streamlit numpy pandas opencv-python scikit-learn matplotlib seaborn pillow plotly pytest flake8 black
```

### 2. Test Locally (Optional but Recommended)

```bash
streamlit run app.py
```

Visit `http://localhost:8501` and test with a traffic sign image.

### 3. Deploy to Hugging Face Spaces

#### Step A: Create GitHub Repository

```bash
cd c:\\Users\\prana\\OneDrive\\Desktop\\Traffic_project
git init
git add .
git commit -m "Traffic sign recognition with Streamlit and Docker"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/traffic-sign-recognition.git
git branch -M main
git push -u origin main
```

#### Step B: Create Hugging Face Space

1. Go to: https://huggingface.co/new-space
2. **Space name**: `traffic-sign-recognition`
3. **SDK**: Choose **"Docker"**
4. Click "Create Space"

#### Step C: Link GitHub to Hugging Face

1. In your new Space, go to **Settings** → **Repository**
2. Click **"Link to GitHub"**
3. Select your `traffic-sign-recognition` repository
4. Enable **"Auto-deploy on push"**

OR manually push:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/traffic-sign-recognition
git push space main
```

#### Step D: Wait for Build

- Hugging Face will automatically build the Docker container
- Build takes ~5-10 minutes
- Monitor progress in the Space UI
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/traffic-sign-recognition`

## Files Summary

✅ **All files created and ready**:
- `app.py` - Streamlit web application
- `Dockerfile` - Hugging Face Spaces configuration
- `src/` - Modular Python code (model, data_loader, predict)
- `tests/` - Unit tests
- `models/traffic_sign_classifier.h5` - Trained model (16 MB, 99.7% accuracy)
- `.github/workflows/test.yml` - CI/CD pipeline
- `requirements.txt` - Python dependencies
- `README.md` - Documentation

## Testing Checklist

Before deploying:
- [ ] Model file exists: `models/traffic_sign_classifier.h5` (16 MB)
- [ ] Can import tensorflow locally
- [ ] Streamlit app runs locally
- [ ] Upload image works
- [ ] Prediction displays correctly

## Troubleshooting

**TensorFlow not installing on Windows?**
```bash
pip install tensorflow-cpu
```

**Port 8501 already in use?**
```bash
streamlit run app.py --server.port=8502
```

**Model not found error?**
Check: `models/traffic_sign_classifier.h5` exists and is 16 MB

## Architecture

```
User uploads image → Streamlit app.py 
  → src/predict.py (preprocesses image)
  → src/model.py (loads model, makes prediction)
  → Results displayed with Plotly chart
```

## Next Steps After Deployment

1. Update README.md with your actual Hugging Face Spaces URL
2. Add sample traffic sign images to repo
3. Test deployed app with various images
4. Share the link!

---

**You're all set! 🚀**
