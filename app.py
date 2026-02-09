"""
Traffic Sign Recognition Web Application
Clean, minimal Streamlit app for traffic sign classification
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from src.predict import TrafficSignPredictor

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean minimal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Simple header */
    .app-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .app-desc {
        font-size: 1rem;
        color: #64748b;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Fixed size result box */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .result-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    
    .result-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-confidence {
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .result-confidence.high { color: #4ade80; }
    .result-confidence.medium { color: #fbbf24; }
    .result-confidence.low { color: #f87171; }
    
    /* Placeholder box */
    .placeholder-box {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .placeholder-text {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Image display */
    .image-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Top predictions */
    .pred-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .pred-rank {
        width: 28px;
        height: 28px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 1rem;
    }
    
    .pred-name {
        flex: 1;
        font-size: 0.9rem;
        color: #334155;
    }
    
    .pred-score {
        font-weight: 600;
        color: #667eea;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    
    [data-testid="stFileUploader"] > div {
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Cache model loading
@st.cache_resource
def load_model():
    return TrafficSignPredictor()


def create_bar_chart(predictions):
    """Simple horizontal bar chart"""
    df = pd.DataFrame(predictions)
    df['pct'] = df['confidence'] * 100
    df = df.sort_values('pct', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['pct'],
        y=df['class_name'],
        orientation='h',
        marker=dict(color='#667eea', cornerradius=4),
        text=df['pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', range=[0, 105], title=''),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        font=dict(family='Inter')
    )
    
    return fig


def main():
    # Simple header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🚦 Traffic Sign Recognition</div>
        <div class="app-desc">Upload a traffic sign image to identify it</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        predictor = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    # Two column layout
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    with col_left:
        st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)
        
        # Simple file uploader - no decorative zone
        uploaded_file = st.file_uploader(
            "Drop image here or click to browse",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            # Show image info
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Format: {image.format or 'Unknown'}")
    
    with col_right:
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Make prediction immediately (no artificial delay)
            try:
                image_np = np.array(image.convert('RGB'))
                result = predictor.predict(image_np, top_k=5)
                
                # Determine confidence level
                conf = result['confidence']
                conf_class = "high" if conf > 0.7 else "medium" if conf > 0.4 else "low"
                
                # Fixed size result box
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-label">Detected Sign</div>
                    <div class="result-name">{result['class_name']}</div>
                    <div class="result-confidence {conf_class}">{conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Top predictions chart
                st.markdown("**Top 5 Predictions**")
                st.plotly_chart(create_bar_chart(result['top_predictions']), use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            # Placeholder when no image
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">📷</div>
                <div class="placeholder-text">Upload an image to see predictions</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Simple footer
    st.markdown("---")
    st.caption("Built with TensorFlow & Streamlit | Trained on GTSRB Dataset")


if __name__ == "__main__":
    main()
