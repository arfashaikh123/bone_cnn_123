# =====================================
# 🦴 Bone Fracture Detection - Streamlit App
# Local prediction using saved models + CNN
# =====================================

import streamlit as st
from PIL import Image
import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# TensorFlow for CNN model
import tensorflow as tf

# =====================================
# ⚙️ Configuration
# =====================================
st.set_page_config(page_title="FractureDetect", layout="wide", page_icon="🦴")

# CNN Configuration
CNN_IMG_SIZE = (180, 180)

# =====================================
# 📁 Load Saved Models & Components
# =====================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
IMG_SIZE = (64, 64)

@st.cache_resource
def load_models():
    """Load all models and components once"""
    models = {}
    
    # Load PCA and Label Encoder
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_transformer.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    # Load ML Models
    model_files = {
        "svm": "svm_model.pkl",
        "rf": "random_forest_model.pkl",
        "dt": "decision_tree_model.pkl",
        "knn": "knn_model.pkl",
        "lr": "logistic_model.pkl",
        "ensemble": "svm_knn_model.pkl",
    }
    
    for key, filename in model_files.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            models[key] = joblib.load(model_path)
            print(f"✅ Loaded {key}")
        else:
            print(f"⚠️ Model not found: {filename}")
    
    # Load CNN Model
    cnn_model = None
    cnn_path = os.path.join(MODEL_DIR, "fracture_classification_model.keras")
    if os.path.exists(cnn_path):
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            models["cnn"] = cnn_model
            print("✅ Loaded CNN model")
        except Exception as e:
            print(f"⚠️ Failed to load CNN model: {e}")
    
    return models, pca, encoder

# Load models at startup
try:
    models, pca, encoder = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# =====================================
# 🔍 Image Preprocessing
# =====================================
def preprocess_image(image_bytes):
    """Preprocess image for prediction"""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image format or empty file.")
    
    img = cv2.equalizeHist(cv2.resize(img, IMG_SIZE))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features.reshape(1, -1), img

# =====================================
# 🧠 Prediction Function
# =====================================
def predict_fracture(image_bytes, model_key):
    """Make prediction using selected model"""
    if model_key not in models:
        raise ValueError(f"Invalid model key '{model_key}'. Available: {list(models.keys())}")
    
    # CNN Model - Different preprocessing
    if model_key == "cnn":
        return predict_with_cnn(image_bytes)
    
    # ML Models - HOG + PCA preprocessing
    model = models[model_key]
    features, processed_img = preprocess_image(image_bytes)
    features_pca = pca.transform(features)
    
    # Get prediction
    pred_class = model.predict(features_pca)[0]
    
    # Get probability if available
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(features_pca)[0]
            confidence = float(np.max(prob) * 100)
        except Exception:
            confidence = 100.0
    else:
        confidence = 100.0
    
    # Get label
    label_name = encoder.inverse_transform([pred_class])[0]
    binary_label = "🔴 Fractured" if "fracture" in label_name.lower() else "🟢 No Fracture"
    
    return {
        "model_used": model_key.upper(),
        "predicted_class": label_name,
        "binary_label": binary_label,
        "confidence": round(confidence, 2),
        "processed_img": processed_img
    }


def predict_with_cnn(image_bytes):
    """Make prediction using CNN model"""
    # Load image from bytes
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Invalid image format or empty file.")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to CNN input size (180x180)
    img_resized = cv2.resize(img_rgb, CNN_IMG_SIZE)
    
    # Normalize and add batch dimension
    img_arr = img_resized / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Predict
    cnn_model = models["cnn"]
    pred = cnn_model.predict(img_arr, verbose=0)[0][0]
    
    # CNN output: higher value = not fractured, lower = fractured
    # pred > 0.5 means NOT fractured
    if pred > 0.5:
        binary_label = "🟢 No Fracture"
        confidence = pred * 100
        predicted_class = "not_fractured"
    else:
        binary_label = "🔴 Fractured"
        confidence = (1 - pred) * 100
        predicted_class = "fractured"
    
    # Create grayscale preview
    processed_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    return {
        "model_used": "CNN (Deep Learning)",
        "predicted_class": predicted_class,
        "binary_label": binary_label,
        "confidence": round(confidence, 2),
        "processed_img": processed_img,
        "raw_output": round(pred, 6)
    }

# =====================================
# 🎨 UI Components
# =====================================
def main_menu():
    st.markdown("""
    <style>
    .main-header {font-size:2.5rem; font-weight:700; margin-bottom:0.5em; color:#0099cc;}
    .stButton>button {background-color:#0099cc; color:white; border-radius:10px; padding:10px 25px;}
    .stButton>button:hover {background-color:#005577;}
    .prediction-box {padding:20px; border-radius:10px; margin:10px 0;}
    .success-box {background-color:#d4edda; border:1px solid #28a745;}
    .danger-box {background-color:#f8d7da; border:1px solid #dc3545;}
    </style>
    """, unsafe_allow_html=True)
    
    menu = ["🏠 Home", "🔬 Try Now", "ℹ️ About"]
    selected = st.sidebar.radio("Navigation", menu)
    st.markdown('<div class="main-header">🦴 FractureDetect</div>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Bone Fracture Detection*")
    st.markdown("---")
    return selected

# =====================================
# 🏠 Home Page
# =====================================
def home_page():
    st.subheader("📊 AI Models Overview")
    
    models_info = [
        {"name": "🧠 CNN (Deep Learning)", "accuracy": 0.95, "key": "cnn",
         "desc": "Convolutional Neural Network with 3 Conv layers. Processes 180x180 RGB images directly without manual feature extraction.",
         "use": "Best for complex patterns and high accuracy. Recommended choice."},
        {"name": "Support Vector Machine (SVM)", "accuracy": 0.77, "key": "svm",
         "desc": "SVM draws optimal decision boundaries to classify fracture and non-fracture X-ray images.",
         "use": "Ideal for smaller datasets and margin-based classification."},
        {"name": "Decision Tree", "accuracy": 0.75, "key": "dt",
         "desc": "A simple tree-based model that splits image features into logical branches.",
         "use": "Useful for interpretability and fast decisions."},
        {"name": "Random Forest", "accuracy": 0.97, "key": "rf",
         "desc": "An ensemble of decision trees that improves accuracy and reduces overfitting.",
         "use": "Best for robust performance and generalization."},
        {"name": "K-Nearest Neighbors (KNN)", "accuracy": 0.99, "key": "knn",
         "desc": "KNN classifies based on similarity to nearby samples.",
         "use": "Good for clear feature separations and smaller datasets."},
        {"name": "Logistic Regression", "accuracy": 0.75, "key": "lr",
         "desc": "A simple, interpretable linear model that predicts fracture probability.",
         "use": "Useful for baseline comparison and binary classification."},
        {"name": "SVM + KNN Ensemble", "accuracy": 0.99, "key": "ensemble",
         "desc": "Combines SVM's precision with KNN's neighborhood reasoning.",
         "use": "Best for maximizing accuracy."},
    ]
    
    col1, col2 = st.columns(2)
    for i, m in enumerate(models_info):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"### {m['name']}")
                
                # Accuracy bar
                accuracy_color = "#28a745" if m['accuracy'] >= 0.9 else "#ffc107" if m['accuracy'] >= 0.8 else "#dc3545"
                st.progress(m['accuracy'])
                st.markdown(f"**Accuracy: {m['accuracy']*100:.0f}%**")
                
                st.write(m['desc'])
                st.caption(f"💡 {m['use']}")
                
                # Show if model is loaded
                if models_loaded and m['key'] in models:
                    st.success("✅ Model loaded")
                else:
                    st.warning("⚠️ Model not available")
                st.markdown("---")

# =====================================
# 🔬 Try Now Page
# =====================================
def try_now_page():
    st.subheader("🔬 Try Fracture Detection")
    st.write("Upload an X-ray image and select a model to predict fracture status.")
    
    if not models_loaded:
        st.error(f"❌ Models could not be loaded: {load_error}")
        st.info("Make sure the 'saved_models' folder exists with all model files.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_options = {
            "🧠 CNN Deep Learning (95% acc)": "cnn",
            "Random Forest (97% acc)": "rf",
            "SVM (77% acc)": "svm",
            "KNN (99% acc)": "knn",
            "Decision Tree (75% acc)": "dt",
            "Logistic Regression (75% acc)": "lr",
            "SVM + KNN Ensemble (99% acc)": "ensemble"
        }
        
        # Filter to only available models
        available_options = {k: v for k, v in model_options.items() if v in models}
        
        if not available_options:
            st.error("No models available!")
            return
        
        model_choice = st.selectbox("🤖 Select Model", list(available_options.keys()))
        model_key = available_options[model_choice]
        
        uploaded = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])
        
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
    
    with col2:
        if uploaded:
            st.image(uploaded, caption="Uploaded X-ray Image", use_container_width=True)
    
    # Prediction
    if predict_btn:
        if not uploaded:
            st.error("⚠️ Please upload an image first!")
        else:
            with st.spinner("🔄 Analyzing image..."):
                try:
                    image_bytes = uploaded.getvalue()
                    result = predict_fracture(image_bytes, model_key)
                    
                    st.markdown("---")
                    st.subheader("📋 Results")
                    
                    # Result display
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        if "Fractured" in result['binary_label']:
                            st.error(f"### {result['binary_label']}")
                        else:
                            st.success(f"### {result['binary_label']}")
                        
                        st.metric("Confidence", f"{result['confidence']:.1f}%")
                        st.write(f"**Model Used:** {result['model_used']}")
                        st.write(f"**Raw Prediction:** {result['predicted_class']}")
                        
                        # Show raw output for CNN
                        if 'raw_output' in result:
                            st.write(f"**CNN Raw Output:** {result['raw_output']}")
                    
                    with col_r2:
                        img_size = "180x180 RGB" if model_key == "cnn" else "64x64 grayscale"
                        st.write(f"**Processed Image ({img_size}):**")
                        st.image(result['processed_img'], caption="Preprocessed", width=200)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Make sure you uploaded a valid X-ray image.")

# =====================================
# ℹ️ About Page
# =====================================
def about_page():
    st.subheader("ℹ️ About FractureDetect")
    
    st.markdown("""
    **FractureDetect** is an AI-powered tool that assists doctors, students, and radiologists 
    in identifying bone fractures from X-ray images. It combines deep learning (CNN) and classic 
    machine learning algorithms with image feature extraction (HOG + PCA) to deliver fast and interpretable results.
    
    ### 📊 Dataset
    - X-ray scans of six bone regions (Elbow, Finger, Forearm, Humerus, Shoulder, Wrist)
    - Labeled as 'Fractured' or 'Normal'
    - Training: ~8,863 images | Testing: 600 images
    
    ### 🧠 CNN Model (Deep Learning)
    - **Architecture:** 3 Convolutional layers (32, 64, 128 filters) + Dense layer
    - **Input:** 180x180 RGB images
    - **Augmentation:** Shear, Zoom, Horizontal Flip
    - **Optimizer:** Adam (learning rate: 1e-4)
    - **Test Accuracy:** ~95%
    
    ### 🔧 ML Models (Classical)
    - **Feature Extraction:** Histogram of Oriented Gradients (HOG)
    - **Dimensionality Reduction:** PCA (Principal Component Analysis)
    - **Input:** 64x64 grayscale images
    - **Models:** SVM, Random Forest, KNN, Decision Tree, Logistic Regression, Ensemble
    
    ### ⚠️ Limitations & Disclaimer
    - This tool is for **educational and research purposes only**
    - **Not a certified medical diagnostic tool**
    - Predictions should be verified by qualified medical professionals
    - Accuracy may vary with image quality and fracture type
    
    ### 🚀 Future Improvements
    - Deep learning models (CNN, Vision Transformers)
    - Data augmentation for class imbalance
    - Grad-CAM/saliency maps for explainability
    - Support for more bone types and fracture classifications
    """)
    
    st.markdown("---")
    st.info("🦴 FractureDetect - Making X-ray analysis accessible and educational")

# =====================================
# 🚀 Main App
# =====================================
selected = main_menu()

if "Home" in selected:
    home_page()
elif "Try Now" in selected:
    try_now_page()
else:
    about_page()
